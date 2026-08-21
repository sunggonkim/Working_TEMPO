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
        "prompt_sha256": base._sha256_bytes(item.prompt.encode("utf-8")),
        "prompt_utf8_bytes": len(item.prompt.encode("utf-8")),
        "requested_max_tokens": item.max_tokens,
        "scheduled_dispatch_offset_ns": item.arrival_offset_ns,
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
            "contract_violations": ["request_or_stream_error"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    record["valid"] = not record["contract_violations"] and record["error"] is None
    return record


def run_workload(
    items: Sequence[base.WorkItem],
    *,
    endpoint: str,
    served_model_name: str,
    timeout_s: float,
    max_workers: int,
    seed: int,
    api_key: str | None,
) -> tuple[int, int, list[dict[str, Any]]]:
    _require(type(max_workers) is int and max_workers > 0, "max_workers must be positive")
    start_ns = time.perf_counter_ns()
    observer = RUN_START_OBSERVER
    if observer is not None:
        observer(start_ns)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(
            execute_request,
            item,
            endpoint=endpoint,
            served_model_name=served_model_name,
            run_start_ns=start_ns,
            timeout_s=timeout_s,
            seed=seed,
            api_key=api_key,
        ) for item in items]
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
        seed=args.seed,
        api_key=api_key,
    )
    decisions = _fetch_decisions(args.base_url, args.timeout_s)
    request_ids = {item.request_id for item in items}
    decision_rows = [row for row in decisions["decisions"] if row.get("request_id") in request_ids]
    decision_ids = [row.get("request_id") for row in decision_rows]
    decisions_exact = (
        len(decision_ids) == len(request_ids)
        and set(decision_ids) == request_ids
        and len(decision_ids) == len(set(decision_ids))
        and all(row.get("phase") == "complete" and row.get("error") is None
                for row in decision_rows)
    )
    valid = all(row["valid"] for row in records) and decisions_exact
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
            "router_decisions_exact": decisions_exact,
            "performance_claim_allowed": valid,
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
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
