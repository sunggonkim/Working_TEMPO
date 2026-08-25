#!/usr/bin/env python3
"""Strict open-loop/closed-loop streaming client for the TEMPO-PD router."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, BinaryIO, Callable, Sequence
from urllib import error, request

from eval.sota_4node import run_vllm_stream_metrics as base


SCHEMA = "tempo-pd-stream-metrics-raw-1"
ROUTER_SCHEMA = "tempo-live-pd-router-1"
# Optional in-process synchronization seam.  Canonical callers leave this
# unset; the completion-backed C4 wrapper installs it only long enough to
# publish the exact client ``perf_counter_ns`` workload epoch to its parent.
RUN_START_OBSERVER: Callable[[int], None] | None = None
_GLOBAL_REJECTION_KINDS = frozenset({
    "global_admission_queue_timeout",
    "global_telemetry_refresh_timeout",
    "global_telemetry_refresh_failed",
    "global_telemetry_validation_failed",
})
_SERVICE_LANE_FAILURE_KINDS = frozenset({
    "endpoint_bounded_global_route_timeout",
    "endpoint_bounded_queue_lease_timeout",
    "endpoint_service_lane_preflight_unavailable",
    "endpoint_service_lane_reservation_unavailable",
})


def _classify_http_error(exc: error.HTTPError) -> str:
    """Classify known global terminal responses without trusting status alone."""

    if not isinstance(exc, error.HTTPError):
        raise TypeError("exc must be urllib.error.HTTPError")
    fragments = [str(exc), str(getattr(exc, "reason", ""))]
    try:
        body = exc.read(16 * 1024)
    except (OSError, ValueError):
        body = b""
    if isinstance(body, bytes):
        fragments.append(body.decode("utf-8", errors="replace"))
    text = " ".join(fragments)
    # The frontend emits both the human-readable legacy detail and the
    # structured TEMPO-GO receipt.  Keep both spellings equivalent so the
    # client cannot turn an explicitly receipted business rejection into an
    # unclassified HTTP failure.
    if (
        "global telemetry refresh timed out" in text
        or "global_telemetry_refresh_timeout" in text
    ):
        return "global_telemetry_refresh_timeout"
    if (
        "global telemetry refresh failed" in text
        or "global_telemetry_refresh_failed" in text
    ):
        return "global_telemetry_refresh_failed"
    if (
        "global telemetry validation failed" in text
        or "global_telemetry_validation_failed" in text
    ):
        return "global_telemetry_validation_failed"
    if (
        "global admission queue timed out" in text
        or "global_admission_queue_timeout" in text
        or "tempo_go_global_reject" in text
    ):
        return "global_admission_queue_timeout"
    if "endpoint_bounded_queue_lease_timeout" in text:
        return "endpoint_bounded_queue_lease_timeout"
    if "endpoint_bounded_global_route_timeout" in text:
        return "endpoint_bounded_global_route_timeout"
    if (
        "tempo_go_service_lane_reservation_timeout" in text
        or "tempo_go_service_lane_reservation_unavailable" in text
        or "endpoint service-lane reservation unavailable" in text
    ):
        return "endpoint_service_lane_reservation_unavailable"
    if (
        "tempo_go_service_lane_preflight_failed" in text
        or "endpoint_service_lane_preflight_unavailable" in text
    ):
        return "endpoint_service_lane_preflight_unavailable"
    return "http_503" if int(exc.code) == 503 else "http_error"


def _is_terminal_global_reject(row: dict[str, Any]) -> bool:
    return (
        row.get("phase") == "rejected"
        and row.get("tempo_go_rejected") is True
        and row.get("global_decision_kind") == "reject"
        and row.get("error") is None
    )


def _is_terminal_service_lane_failure(row: dict[str, Any]) -> bool:
    """Recognize an explicit endpoint reservation failure receipt."""

    failure = row.get("frontend_tempo_go_reservation_failure")
    failure_kind = row.get("frontend_tempo_go_failure_kind")
    return (
        row.get("phase") == "failed"
        and row.get("error") in {None, failure_kind}
        and row.get("frontend_tempo_go_failure_scope") == "service_lane"
        and isinstance(failure, dict)
        and failure.get("schema") == "tempo-go-service-lane-reservation-v1"
        and failure.get("failure_kind") == failure_kind
    )


def _apply_decision_receipts(
    records: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> bool:
    """Close request terminal states against the router's decision ledger."""

    by_id = {row.get("request_id"): row for row in decision_rows}
    request_ids = {row.get("request_id") for row in records}
    decision_ids = [row.get("request_id") for row in decision_rows]
    exact = (
        len(decision_ids) == len(request_ids)
        and set(decision_ids) == request_ids
        and len(decision_ids) == len(set(decision_ids))
        and all(
            row.get("phase") == "complete" and row.get("error") is None
            or _is_terminal_global_reject(row)
            or _is_terminal_service_lane_failure(row)
            for row in decision_rows
        )
    )
    for record in records:
        decision = by_id.get(record.get("request_id"))
        if record.get("terminal_reject_candidate"):
            if isinstance(decision, dict) and _is_terminal_global_reject(decision):
                record["terminal_kind"] = "global_reject"
                record["terminal_reason"] = decision.get(
                    "global_decision_reason")
                record["contract_violations"] = []
                record["error"] = None
                record["valid"] = True
                continue
            record["contract_violations"] = [
                "unreceipted_terminal_reject",
            ]
            record["valid"] = False
            continue
        if record.get("terminal_service_lane_failure_candidate"):
            if isinstance(decision, dict) and _is_terminal_service_lane_failure(
                    decision):
                record["terminal_kind"] = "service_lane_failure"
                record["terminal_reason"] = decision.get(
                    "frontend_tempo_go_failure_kind")
                record["contract_violations"] = []
                record["error"] = None
                record["valid"] = True
                continue
            record["contract_violations"] = [
                "unreceipted_terminal_service_lane_failure",
            ]
            record["valid"] = False
            continue
        record["valid"] = (
            record.get("valid") is True
            and isinstance(decision, dict)
            and decision.get("phase") == "complete"
            and decision.get("error") is None
        )
    return exact


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise base.ContractError(message)


def _router_headers(response: Any, request_id: str) -> dict[str, Any]:
    headers = response.headers
    result = {
        "schema": headers.get("X-Tempo-PD-Schema"),
        "request_id": headers.get("X-Tempo-PD-Request-Id"),
        "mode": headers.get("X-Tempo-PD-Mode"),
        "route": headers.get("X-Tempo-PD-Route"),
        "reason": headers.get("X-Tempo-PD-Reason"),
        "workload_fingerprint": headers.get("X-Tempo-PD-Workload"),
        "profile_id": headers.get("X-Tempo-PD-Profile"),
        "manifest_id": headers.get("X-Tempo-PD-Manifest"),
    }
    _require(result["schema"] == ROUTER_SCHEMA, "router schema header mismatch")
    _require(result["request_id"] == request_id, "router request ID mismatch")
    _require(result["route"] in {
        "decoder_local_recompute_or_cache", "remote_prefill_live_kv",
    }, "router route header mismatch")
    for name in ("mode", "reason", "workload_fingerprint"):
        _require(isinstance(result[name], str) and result[name],
                 f"router {name} header is missing")
    return result


def _stream_record(
    stream: BinaryIO,
    *,
    dispatch_ns: int,
    run_start_ns: int,
    expected_tokens: int,
    route: str,
    clock_ns: Callable[[], int],
) -> dict[str, Any]:
    arrivals: list[int] = []
    token_values: list[str] = []
    token_proofs: list[str] = []
    text_fragments: list[str] = []
    finish_reasons: list[str] = []
    response_ids: set[str] = set()
    response_models: set[str] = set()
    violations: list[str] = []
    usage: dict[str, int] | None = None
    done_seen = False
    remote = route == "remote_prefill_live_kv"

    for event_ns, payload in base.iter_sse_data(stream, clock_ns=clock_ns):
        if payload == "[DONE]":
            done_seen = True
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise base.ContractError(f"malformed JSON SSE event: {exc}") from exc
        _require(isinstance(event, dict), "SSE event must be an object")
        if isinstance(event.get("id"), str):
            response_ids.add(event["id"])
        if isinstance(event.get("model"), str):
            response_models.add(event["model"])
        raw_usage = event.get("usage")
        if raw_usage is not None:
            names = ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(raw_usage, dict) and all(
                type(raw_usage.get(name)) is int and raw_usage[name] >= 0
                for name in names
            ):
                usage = {name: int(raw_usage[name]) for name in names}
            else:
                violations.append("usage_counts_invalid")
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            violations.append("choices_not_list")
            continue
        if not choices:
            continue
        if len(choices) != 1 or not isinstance(choices[0], dict):
            violations.append("not_exactly_one_choice")
            continue
        choice = choices[0]
        text_value = choice.get("text", "")
        if not isinstance(text_value, str):
            violations.append("choice_text_not_string")
            text_value = ""
        text_fragments.append(text_value)
        reason = choice.get("finish_reason")
        if reason is not None:
            if isinstance(reason, str):
                finish_reasons.append(reason)
            else:
                violations.append("finish_reason_not_string")
            continue
        logprobs = choice.get("logprobs")
        tokens = logprobs.get("tokens") if isinstance(logprobs, dict) else None
        if isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], str):
            token_values.append(tokens[0])
            token_proofs.append("vllm_logprobs_exactly_one")
            arrivals.append(event_ns - run_start_ns)
        elif remote and not arrivals and text_value and tokens is None:
            # The pinned official LMCache proxy creates exactly one head chunk
            # from a prefill request whose max_tokens is forced to one.
            token_values.append(text_value)
            token_proofs.append("official_lmcache_proxy_single_prefill_token")
            arrivals.append(event_ns - run_start_ns)
        else:
            violations.append("token_identity_or_cardinality_unproven")

    stream_end_ns = clock_ns()
    if not done_seen:
        violations.append("done_event_missing")
    if finish_reasons != ["length"]:
        violations.append("finish_reason_not_exactly_length")
    if usage is None:
        violations.append("final_usage_missing")
    elif remote:
        if usage["completion_tokens"] not in {expected_tokens - 1, expected_tokens}:
            violations.append("proxy_usage_completion_tokens_mismatch")
    elif usage["completion_tokens"] != expected_tokens:
        violations.append("usage_completion_tokens_mismatch")
    if len(arrivals) != expected_tokens:
        violations.append("requested_completion_tokens_mismatch")
    if remote:
        if not (1 <= len(response_ids) <= 2):
            violations.append("proxy_response_id_count_invalid")
        if token_proofs.count("official_lmcache_proxy_single_prefill_token") != 1:
            violations.append("proxy_first_token_proof_missing")
    elif len(response_ids) != 1:
        violations.append("response_id_not_stable")
    if len(response_models) != 1:
        violations.append("response_model_not_stable")
    dispatch_offset_ns = dispatch_ns - run_start_ns
    if any(value < dispatch_offset_ns for value in arrivals):
        violations.append("token_arrival_precedes_dispatch")
    if any(right < left for left, right in zip(arrivals, arrivals[1:])):
        violations.append("token_arrivals_not_monotonic")
    output_text = "".join(text_fragments)
    return {
        "http_status": 200,
        "dispatch_offset_ns": dispatch_offset_ns,
        "token_arrival_offsets_ns": arrivals,
        "stream_end_offset_ns": stream_end_ns - run_start_ns,
        "output_token_values": token_values,
        "output_token_proofs": token_proofs,
        "output_text": output_text,
        "output_text_sha256": base._sha256_bytes(output_text.encode("utf-8")),
        "finish_reason": finish_reasons[0] if len(finish_reasons) == 1 else None,
        "usage": usage,
        "done_seen": done_seen,
        "response_ids": sorted(response_ids),
        "response_models": sorted(response_models),
        "contract_violations": sorted(set(violations)),
        "error": None,
    }


def execute_request(
    item: base.WorkItem,
    *,
    endpoint: str,
    served_model_name: str,
    run_start_ns: int,
    timeout_s: float,
    seed: int,
    api_key: str | None,
    opener: Callable[..., Any] = request.urlopen,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    target_ns = run_start_ns + item.arrival_offset_ns
    delay_ns = target_ns - clock_ns()
    if delay_ns > 0:
        sleeper(delay_ns / 1_000_000_000.0)
    dispatch_ns = clock_ns()
    body = {
        "model": served_model_name,
        "prompt": item.prompt,
        "max_tokens": item.max_tokens,
        "temperature": 0.0,
        "seed": seed + item.index,
        "n": 1,
        "echo": False,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": 1,
    }
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "X-Tempo-Request-Id": item.request_id,
    }
    tenant_id = _business_tenant_id(item.request_id)
    if tenant_id is not None:
        headers["X-Tempo-Tenant-Id"] = tenant_id
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = request.Request(
        endpoint,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    common = {
        "request_index": item.index,
        "request_id": item.request_id,
        "ingress_lane": _ingress_lane(item),
        "prompt_sha256": base._sha256_bytes(item.prompt.encode("utf-8")),
        "prompt_utf8_bytes": len(item.prompt.encode("utf-8")),
        "requested_max_tokens": item.max_tokens,
        "scheduled_dispatch_offset_ns": item.arrival_offset_ns,
        "tempo_business_tenant_id": tenant_id,
    }
    try:
        with opener(http_request, timeout=timeout_s) as response:
            status = int(response.getcode())
            _require(status == 200, f"HTTP status {status}")
            router = _router_headers(response, item.request_id)
            streamed = _stream_record(
                response,
                dispatch_ns=dispatch_ns,
                run_start_ns=run_start_ns,
                expected_tokens=item.max_tokens,
                route=router["route"],
                clock_ns=clock_ns,
            )
            record = {**common, "router": router, **streamed}
    except (base.ContractError, error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
        end_ns = clock_ns()
        terminal_error_kind = (
            _classify_http_error(exc)
            if isinstance(exc, error.HTTPError)
            else "request_or_stream_error"
        )
        global_reject_candidate = terminal_error_kind in _GLOBAL_REJECTION_KINDS
        service_lane_failure_candidate = (
            terminal_error_kind in _SERVICE_LANE_FAILURE_KINDS)
        record = {
            **common,
            "router": None,
            "http_status": int(exc.code) if isinstance(exc, error.HTTPError) else None,
            "dispatch_offset_ns": dispatch_ns - run_start_ns,
            "token_arrival_offsets_ns": [],
            "stream_end_offset_ns": end_ns - run_start_ns,
            "output_token_values": [],
            "output_token_proofs": [],
            "output_text": "",
            "output_text_sha256": base._sha256_bytes(b""),
            "finish_reason": None,
            "usage": None,
            "done_seen": False,
            "response_ids": [],
            "response_models": [],
            "contract_violations": (
                [] if global_reject_candidate or service_lane_failure_candidate
                else [terminal_error_kind]),
            "error": None if global_reject_candidate or service_lane_failure_candidate
            else f"{type(exc).__name__}: {exc}",
            "terminal_reject_candidate": global_reject_candidate,
            "terminal_service_lane_failure_candidate": (
                service_lane_failure_candidate),
            "terminal_error_kind": terminal_error_kind,
            "transport_error": f"{type(exc).__name__}: {exc}",
        }
    record["valid"] = not record["contract_violations"] and record["error"] is None
    return record


def _ingress_lane(item: base.WorkItem) -> str:
    """Classify the already-frozen request identity for ingress scheduling.

    The C7 manifest embeds the business tenant in the request ID.  This is
    only a client-side ingress lane; it is not sent as a controller phase,
    route, or future-arrival hint.  Keeping the classification here makes the
    shared-pool artifact and the reserved-interactive artifact differ only in
    client admission, with identical request bodies and arrival timestamps.
    """

    request_id = item.request_id
    if any(marker in request_id for marker in (
        "-interactive-", "-latency-", "-foreground-",
    )):
        return "interactive"
    return "background"


def _business_tenant_id(request_id: str) -> str | None:
    """Return the frozen business class encoded in a request ID.

    This is an ingress identity only: it carries no phase, route, or future
    arrival information.  C7's managed-background arm maps both local and
    remote background traffic to the profile's single ``background`` budget.
    """

    for tenant in ("latency", "interactive", "batch", "background"):
        if f"-{tenant}-" in request_id:
            return tenant
    return None


def run_workload(
    items: Sequence[base.WorkItem],
    *,
    endpoint: str,
    served_model_name: str,
    timeout_s: float,
    max_workers: int,
    ingress_policy: str = "shared_pool",
    interactive_reserved_workers: int = 0,
    seed: int,
    api_key: str | None,
) -> tuple[int, int, list[dict[str, Any]]]:
    _require(type(max_workers) is int and max_workers > 0, "max_workers must be positive")
    _require(ingress_policy in {"shared_pool", "interactive_reserved"},
             "unsupported ingress policy")
    _require(
        type(interactive_reserved_workers) is int
        and interactive_reserved_workers >= 0,
        "interactive_reserved_workers must be a non-negative int",
    )
    if ingress_policy == "shared_pool":
        _require(interactive_reserved_workers == 0,
                 "shared_pool cannot reserve interactive workers")
    else:
        _require(
            0 < interactive_reserved_workers < max_workers,
            "interactive_reserved_workers must leave background workers",
        )
    start_ns = time.perf_counter_ns()
    observer = RUN_START_OBSERVER
    if observer is not None:
        observer(start_ns)
    records: list[dict[str, Any]] = []
    def submit(pool, selected):
        return [pool.submit(
            execute_request,
            item,
            endpoint=endpoint,
            served_model_name=served_model_name,
            run_start_ns=start_ns,
            timeout_s=timeout_s,
            seed=seed,
            api_key=api_key,
        ) for item in selected]

    if ingress_policy == "shared_pool":
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = submit(pool, items)
            for future in as_completed(futures):
                records.append(future.result())
    else:
        interactive = [item for item in items if _ingress_lane(item) == "interactive"]
        background = [item for item in items if _ingress_lane(item) == "background"]
        background_workers = max_workers - interactive_reserved_workers
        # Separate pools are deliberate: submitting background futures first
        # must never consume the business-reserved interactive lane.  Both
        # pools share the same open-loop clock and the same frozen workload.
        with (
            ThreadPoolExecutor(max_workers=interactive_reserved_workers)
            as interactive_pool,
            ThreadPoolExecutor(max_workers=background_workers)
            as background_pool,
        ):
            futures = submit(interactive_pool, interactive)
            futures.extend(submit(background_pool, background))
            for future in as_completed(futures):
                records.append(future.result())
    end_ns = time.perf_counter_ns()
    records.sort(key=lambda value: value["request_index"])
    return start_ns, end_ns, records


def _fetch_decisions(base_url: str, timeout_s: float) -> dict[str, Any]:
    with request.urlopen(base_url.rstrip("/") + "/tempo/decisions", timeout=timeout_s) as response:
        _require(int(response.getcode()) == 200, "decision endpoint failed")
        value = json.loads(response.read())
    _require(isinstance(value, dict) and value.get("schema") == ROUTER_SCHEMA,
             "decision artifact schema mismatch")
    _require(isinstance(value.get("decisions"), list), "decisions must be a list")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixed_local", "lmcache_always_remote", "tempo_auto"),
                        required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--default-max-tokens", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--ingress-policy",
        choices=("shared_pool", "interactive_reserved"),
        default="shared_pool",
    )
    parser.add_argument("--interactive-reserved-workers", type=int, default=0)
    parser.add_argument("--request-rate", type=float)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--api-key-env")
    args = parser.parse_args(argv)
    try:
        endpoint = base.completion_endpoint(args.base_url)
        _require(args.model.is_absolute() and (args.model / "config.json").is_file(),
                 "model must be an absolute local directory")
        items, workload_sha256 = base.load_workload(
            args.workload,
            default_max_tokens=args.default_max_tokens,
            request_rate=args.request_rate,
        )
        _require(math.isfinite(args.timeout_s) and args.timeout_s > 0,
                 "timeout_s must be positive")
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        if args.api_key_env:
            _require(api_key is not None, "API key environment variable is unset")
    except (base.ContractError, OSError) as exc:
        parser.error(str(exc))

    started_at = datetime.now(timezone.utc).isoformat()
    start_ns, end_ns, records = run_workload(
        items,
        endpoint=endpoint,
        served_model_name=args.served_model_name,
        timeout_s=args.timeout_s,
        max_workers=args.max_workers,
        ingress_policy=args.ingress_policy,
        interactive_reserved_workers=args.interactive_reserved_workers,
        seed=args.seed,
        api_key=api_key,
    )
    decisions = _fetch_decisions(args.base_url, args.timeout_s)
    request_ids = {item.request_id for item in items}
    decision_rows = [
        row for row in decisions["decisions"]
        if row.get("request_id") in request_ids
    ]
    decisions_exact = _apply_decision_receipts(records, decision_rows)
    terminal_contract_valid = (
        all(row["valid"] for row in records) and decisions_exact)
    global_rejected_count = sum(
        row.get("terminal_kind") == "global_reject" for row in records)
    terminal_error_counts: dict[str, int] = {}
    for row in records:
        kind = row.get("terminal_error_kind")
        if isinstance(kind, str):
            terminal_error_counts[kind] = terminal_error_counts.get(kind, 0) + 1
    performance_claim_allowed = (
        terminal_contract_valid and global_rejected_count == 0)
    artifact = {
        "schema": SCHEMA,
        "evidence": "actual_vllm_pd_router_client_stream",
        "run": {
            "run_id": args.run_id,
            "mode": args.mode,
            "endpoint": endpoint,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "client_window_ns": end_ns - start_ns,
        },
        "model": {
            "local_path": str(args.model),
            "served_model_name": args.served_model_name,
            "config_sha256": base._sha256_bytes((args.model / "config.json").read_bytes()),
        },
        "workload": {
            "schema": base.WORKLOAD_SCHEMA,
            "explicit_path": str(args.workload),
            "sha256": workload_sha256,
            "request_count": len(items),
            "max_workers": args.max_workers,
            "ingress_policy": args.ingress_policy,
            "interactive_reserved_workers": args.interactive_reserved_workers,
            "request_rate_per_s": args.request_rate,
            "seed": args.seed,
        },
        "requests": records,
        "router_decisions": decision_rows,
        "router_decision_endpoint": {
            key: value for key, value in decisions.items()
            if key != "decisions"
        },
        "validation": {
            "all_streams_valid": all(row["valid"] for row in records),
            "completed_count": sum(
                row.get("terminal_kind") != "global_reject" for row in records),
            "global_rejected_count": global_rejected_count,
            "terminal_error_counts": dict(sorted(terminal_error_counts.items())),
            "router_decisions_exact": decisions_exact,
            "terminal_contract_valid": terminal_contract_valid,
            "performance_claim_allowed": performance_claim_allowed,
        },
        "metric_contract": {
            "clock": "client time.perf_counter_ns",
            "remote_first_token": "official LMCache proxy max_tokens=1 head event",
            "subsequent_tokens": "one vLLM logprob token per SSE event",
            "retry_count": 0,
        },
    }
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    # A fully receipted overload reject is a valid terminal contract and must
    # still produce a native raw artifact. It is not, by itself, a performance
    # claim; that remains false whenever any request was globally rejected.
    return 0 if terminal_contract_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
