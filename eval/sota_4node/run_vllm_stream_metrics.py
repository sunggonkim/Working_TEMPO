#!/usr/bin/env python3
"""Measure client-observed token arrivals from vLLM's completion SSE API.

This client deliberately uses only the OpenAI-compatible public endpoint.  A
token timestamp is accepted only when one SSE event identifies exactly one
output token through ``choices[0].logprobs.tokens``.  This avoids pretending
that a text fragment is necessarily one model token.

No server is started, discovered, retried, or stopped by this module.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, BinaryIO, Callable, Iterator, Sequence
from urllib import error, parse, request


SCHEMA = "tempo-vllm-stream-metrics-raw-1"
WORKLOAD_SCHEMA = "tempo-vllm-stream-workload-jsonl-1"
_WORKLOAD_KEYS = frozenset({"request_id", "prompt", "max_tokens", "arrival_offset_ms"})


class ContractError(ValueError):
    """The workload or streaming response cannot support exact metrics."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(encoded)


def _nonnegative_number(value: Any, field: str) -> float:
    _require(type(value) in (int, float), f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result) and result >= 0.0, f"{field} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class WorkItem:
    index: int
    request_id: str
    prompt: str
    max_tokens: int
    arrival_offset_ns: int


def load_workload(
    path: Path,
    *,
    default_max_tokens: int,
    request_rate: float | None,
) -> tuple[list[WorkItem], str]:
    """Load one explicit JSONL file without discovering neighboring files."""

    _require(path.is_file(), f"workload is not a file: {path}")
    _require(type(default_max_tokens) is int and default_max_tokens >= 2,
             "default_max_tokens must be at least 2 so TPOT is defined")
    if request_rate is not None:
        _require(math.isfinite(request_rate) and request_rate > 0.0,
                 "request_rate must be finite and positive")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError(f"workload is not UTF-8: {path}") from exc

    parsed: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON at workload line {line_number}: {exc}") from exc
        _require(isinstance(value, dict), f"workload line {line_number} must be an object")
        unknown = set(value) - _WORKLOAD_KEYS
        _require(not unknown, f"workload line {line_number} has unknown fields: {sorted(unknown)}")
        request_id = value.get("request_id")
        prompt = value.get("prompt")
        _require(isinstance(request_id, str) and request_id.strip(),
                 f"workload line {line_number} request_id must be nonempty")
        _require(isinstance(prompt, str) and prompt, f"workload line {line_number} prompt must be nonempty")
        max_tokens = value.get("max_tokens", default_max_tokens)
        _require(type(max_tokens) is int and max_tokens >= 2,
                 f"workload line {line_number} max_tokens must be at least 2")
        if "arrival_offset_ms" in value:
            _nonnegative_number(value["arrival_offset_ms"],
                                f"workload line {line_number} arrival_offset_ms")
        parsed.append(value)

    _require(bool(parsed), "workload must contain at least one request")
    identifiers = [str(value["request_id"]) for value in parsed]
    _require(len(identifiers) == len(set(identifiers)), "request_id values must be unique")
    explicit_arrivals = ["arrival_offset_ms" in value for value in parsed]
    _require(not (request_rate is not None and any(explicit_arrivals)),
             "request_rate and explicit arrival_offset_ms are mutually exclusive")
    _require(not any(explicit_arrivals) or all(explicit_arrivals),
             "arrival_offset_ms must be present on every request or none")

    items = []
    for index, value in enumerate(parsed):
        if all(explicit_arrivals):
            arrival_ms = float(value["arrival_offset_ms"])
        elif request_rate is not None:
            arrival_ms = index * 1000.0 / request_rate
        else:
            arrival_ms = 0.0
        items.append(WorkItem(
            index=index,
            request_id=str(value["request_id"]),
            prompt=str(value["prompt"]),
            max_tokens=int(value.get("max_tokens", default_max_tokens)),
            arrival_offset_ns=round(arrival_ms * 1_000_000.0),
        ))
    return items, _sha256_bytes(raw)


def completion_endpoint(base_url: str) -> str:
    """Build the one supported endpoint and reject ambiguous URLs."""

    parts = parse.urlsplit(base_url)
    _require(parts.scheme in {"http", "https"}, "base_url scheme must be http or https")
    _require(bool(parts.hostname), "base_url must have a host")
    _require(parts.username is None and parts.password is None, "base_url must not contain credentials")
    _require(not parts.query and not parts.fragment, "base_url must not contain query or fragment")
    path = parts.path.rstrip("/")
    _require(path in {"", "/v1"}, "base_url path must be empty or /v1")
    prefix = base_url.rstrip("/")
    return prefix + ("/completions" if path == "/v1" else "/v1/completions")


def iter_sse_data(
    stream: BinaryIO,
    *,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> Iterator[tuple[int, str]]:
    """Yield the receipt time and data payload of each complete SSE event."""

    data_lines: list[str] = []
    for raw_line in stream:
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
        except (AttributeError, UnicodeDecodeError) as exc:
            raise ContractError("SSE stream contains a non-UTF-8 line") from exc
        if not line:
            if data_lines:
                yield clock_ns(), "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if line == "data":
            data_lines.append("")
        elif line.startswith("data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        yield clock_ns(), "\n".join(data_lines)


def _stream_record(
    stream: BinaryIO,
    *,
    dispatch_ns: int,
    run_start_ns: int,
    expected_tokens: int,
    clock_ns: Callable[[], int],
) -> dict[str, Any]:
    arrivals: list[int] = []
    output_tokens: list[str] = []
    text_fragments: list[str] = []
    finish_reasons: list[str] = []
    response_ids: set[str] = set()
    response_models: set[str] = set()
    violations: list[str] = []
    usage: dict[str, int] | None = None
    done_seen = False

    for event_ns, payload in iter_sse_data(stream, clock_ns=clock_ns):
        if payload == "[DONE]":
            done_seen = True
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractError(f"malformed JSON SSE event: {exc}") from exc
        _require(isinstance(event, dict), "each SSE data event must be a JSON object")
        if isinstance(event.get("id"), str):
            response_ids.add(event["id"])
        if isinstance(event.get("model"), str):
            response_models.add(event["model"])

        raw_usage = event.get("usage")
        if raw_usage is not None:
            if not isinstance(raw_usage, dict):
                violations.append("usage_not_object")
            else:
                names = ("prompt_tokens", "completion_tokens", "total_tokens")
                if all(type(raw_usage.get(name)) is int and raw_usage[name] >= 0 for name in names):
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
        if choice.get("index") != 0:
            violations.append("choice_index_not_zero")
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

        logprobs = choice.get("logprobs")
        tokens = logprobs.get("tokens") if isinstance(logprobs, dict) else None
        if tokens is None:
            if text_value:
                violations.append("text_without_logprob_token_identity")
            continue
        if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
            violations.append("logprob_tokens_invalid")
            continue
        if len(tokens) == 0:
            if text_value:
                violations.append("text_without_logprob_token_identity")
            continue
        if len(tokens) != 1:
            violations.append("multiple_tokens_in_one_sse_event")
        for token in tokens:
            output_tokens.append(token)
            arrivals.append(event_ns - run_start_ns)

    stream_end_ns = clock_ns()
    if not done_seen:
        violations.append("done_event_missing")
    if finish_reasons != ["length"]:
        violations.append("finish_reason_not_exactly_length")
    if usage is None:
        violations.append("final_usage_missing")
    elif usage["completion_tokens"] != len(output_tokens):
        violations.append("usage_completion_tokens_mismatch")
    if len(output_tokens) != expected_tokens:
        violations.append("requested_completion_tokens_mismatch")
    if len(response_ids) != 1:
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
        "output_tokens": output_tokens,
        "output_token_sha256": _sha256_json(output_tokens),
        "output_text": output_text,
        "output_text_sha256": _sha256_bytes(output_text.encode("utf-8")),
        "finish_reason": finish_reasons[0] if len(finish_reasons) == 1 else None,
        "usage": usage,
        "done_seen": done_seen,
        "response_id": next(iter(response_ids)) if len(response_ids) == 1 else None,
        "response_model": next(iter(response_models)) if len(response_models) == 1 else None,
        "contract_violations": sorted(set(violations)),
        "error": None,
    }


def execute_request(
    item: WorkItem,
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
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = request.Request(
        endpoint,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    base = {
        "request_index": item.index,
        "request_id": item.request_id,
        "prompt_sha256": _sha256_bytes(item.prompt.encode("utf-8")),
        "prompt_utf8_bytes": len(item.prompt.encode("utf-8")),
        "requested_max_tokens": item.max_tokens,
        "scheduled_dispatch_offset_ns": item.arrival_offset_ns,
    }
    try:
        with opener(http_request, timeout=timeout_s) as response:
            status = int(response.getcode())
            _require(status == 200, f"HTTP status is {status}, expected 200")
            streamed = _stream_record(
                response,
                dispatch_ns=dispatch_ns,
                run_start_ns=run_start_ns,
                expected_tokens=item.max_tokens,
                clock_ns=clock_ns,
            )
            record = {**base, **streamed}
    except (ContractError, error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
        end_ns = clock_ns()
        record = {
            **base,
            "http_status": int(exc.code) if isinstance(exc, error.HTTPError) else None,
            "dispatch_offset_ns": dispatch_ns - run_start_ns,
            "token_arrival_offsets_ns": [],
            "stream_end_offset_ns": end_ns - run_start_ns,
            "output_tokens": [],
            "output_token_sha256": _sha256_json([]),
            "output_text": "",
            "output_text_sha256": _sha256_bytes(b""),
            "finish_reason": None,
            "usage": None,
            "done_seen": False,
            "response_id": None,
            "response_model": None,
            "contract_violations": ["request_or_stream_error"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    record["valid"] = not record["contract_violations"] and record["error"] is None
    return record


def run_workload(
    items: Sequence[WorkItem],
    *,
    endpoint: str,
    served_model_name: str,
    timeout_s: float,
    max_workers: int,
    seed: int,
    api_key: str | None,
) -> tuple[int, int, list[dict[str, Any]]]:
    _require(type(max_workers) is int and max_workers >= 1, "max_workers must be positive")
    run_start_ns = time.perf_counter_ns()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(
            execute_request,
            item,
            endpoint=endpoint,
            served_model_name=served_model_name,
            run_start_ns=run_start_ns,
            timeout_s=timeout_s,
            seed=seed,
            api_key=api_key,
        ) for item in items]
        for future in as_completed(futures):
            records.append(future.result())
    run_end_ns = time.perf_counter_ns()
    records.sort(key=lambda value: value["request_index"])
    return run_start_ns, run_end_ns, records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True,
                        help="vLLM base URL with no path or with /v1")
    parser.add_argument("--model", type=Path, required=True,
                        help="absolute local model directory used by the server")
    parser.add_argument("--served-model-name",
                        help="request model name; defaults to the absolute --model path")
    parser.add_argument("--workload", type=Path, required=True,
                        help="explicit UTF-8 JSONL workload path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-id", default="vllm-stream-screen")
    parser.add_argument("--default-max-tokens", type=int, default=64)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--request-rate", type=float,
                        help="open-loop requests/s; omit for a bounded burst/closed queue")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--api-key-env",
                        help="optional environment-variable name containing the API key")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        endpoint = completion_endpoint(args.base_url)
        model = args.model
        _require(model.is_absolute(), "--model must be an absolute local path")
        _require(model.is_dir(), f"local model directory does not exist: {model}")
        config_path = model / "config.json"
        _require(config_path.is_file(), f"local model config is missing: {config_path}")
        _require(isinstance(args.mode, str) and args.mode.strip(), "--mode must be nonempty")
        _require(math.isfinite(args.timeout_s) and args.timeout_s > 0.0,
                 "--timeout-s must be finite and positive")
        items, workload_sha256 = load_workload(
            args.workload,
            default_max_tokens=args.default_max_tokens,
            request_rate=args.request_rate,
        )
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        if args.api_key_env:
            _require(api_key is not None, f"API key environment variable is unset: {args.api_key_env}")
        served_name = args.served_model_name or str(model)
    except (ContractError, OSError) as exc:
        parser.error(str(exc))

    wall_start = datetime.now(timezone.utc).isoformat()
    run_start_ns, run_end_ns, records = run_workload(
        items,
        endpoint=endpoint,
        served_model_name=served_name,
        timeout_s=args.timeout_s,
        max_workers=args.max_workers,
        seed=args.seed,
        api_key=api_key,
    )
    valid = all(record["valid"] for record in records)
    artifact = {
        "schema_version": SCHEMA,
        "evidence_state": "native_vllm_client_stream",
        "run": {
            "run_id": args.run_id,
            "mode": args.mode,
            "endpoint": endpoint,
            "started_at_utc": wall_start,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "client_window_ns": run_end_ns - run_start_ns,
        },
        "model": {
            "source": "explicit_local_directory",
            "local_path": str(model),
            "served_model_name": served_name,
            "config_sha256": _sha256_bytes(config_path.read_bytes()),
            "offline_server_assumption": True,
            "offline_server_assumption_verified_by_client": False,
        },
        "endpoint_contract": {
            "api": "OpenAI-compatible POST /v1/completions",
            "stream": True,
            "logprobs": 1,
            "stream_options_include_usage": True,
            "ignore_eos": True,
            "retry_count": 0,
            "api_key_present": api_key is not None,
        },
        "clock": {
            "name": "time.perf_counter_ns",
            "scope": "single client process",
            "timestamp_semantics": "complete SSE data event observed by client",
        },
        "workload": {
            "schema_version": WORKLOAD_SCHEMA,
            "explicit_path": str(args.workload),
            "sha256": workload_sha256,
            "request_count": len(items),
            "max_workers": args.max_workers,
            "request_rate_per_s": args.request_rate,
            "default_max_tokens": args.default_max_tokens,
            "seed": args.seed,
        },
        "requests": records,
        "validation": {
            "all_requests_valid": valid,
            "valid_requests": sum(record["valid"] for record in records),
            "invalid_requests": sum(not record["valid"] for record in records),
            "performance_claim_allowed": valid,
        },
        "limitations": [
            "timestamps are client-observed SSE event arrivals, not server GPU timestamps",
            "the client cannot verify that the separately launched server is offline",
            "one logprob token per SSE event is required; batching multiple tokens fails closed",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "requests": len(records),
        "valid": valid,
    }, sort_keys=True))
    return 0 if valid else 2


if __name__ == "__main__":
    sys.exit(main())
