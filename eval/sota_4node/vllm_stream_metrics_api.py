"""Importable sidecar API for the strict vLLM streaming metrics client."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from eval.sota_4node import run_vllm_stream_metrics as client


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


def _items(
    requests: Sequence[Mapping[str, Any]],
    *,
    default_max_tokens: int,
    request_rate: float | None,
) -> list[client.WorkItem]:
    client._require(bool(requests), "requests must be nonempty")
    client._require(type(default_max_tokens) is int and default_max_tokens >= 2,
                    "default_max_tokens must be at least 2")
    if request_rate is not None:
        client._require(math.isfinite(request_rate) and request_rate > 0.0,
                        "request_rate must be finite and positive")
    explicit = ["arrival_offset_ms" in value for value in requests]
    client._require(not (request_rate is not None and any(explicit)),
                    "request_rate and arrival_offset_ms are mutually exclusive")
    client._require(not any(explicit) or all(explicit),
                    "arrival_offset_ms must be supplied for all requests or none")
    result: list[client.WorkItem] = []
    identifiers: set[str] = set()
    for index, value in enumerate(requests):
        client._require(isinstance(value, Mapping), f"requests[{index}] must be an object")
        unknown = set(value) - client._WORKLOAD_KEYS
        client._require(not unknown, f"requests[{index}] has unknown fields: {sorted(unknown)}")
        request_id = value.get("request_id")
        prompt = value.get("prompt")
        max_tokens = value.get("max_tokens", default_max_tokens)
        client._require(isinstance(request_id, str) and request_id.strip(),
                        f"requests[{index}].request_id must be nonempty")
        client._require(request_id not in identifiers, f"duplicate request_id: {request_id}")
        identifiers.add(request_id)
        client._require(isinstance(prompt, str) and prompt,
                        f"requests[{index}].prompt must be nonempty")
        client._require(type(max_tokens) is int and max_tokens >= 2,
                        f"requests[{index}].max_tokens must be at least 2")
        if all(explicit):
            offset_ms = client._nonnegative_number(
                value["arrival_offset_ms"], f"requests[{index}].arrival_offset_ms")
        elif request_rate is not None:
            offset_ms = index * 1000.0 / request_rate
        else:
            offset_ms = 0.0
        result.append(client.WorkItem(
            index=index,
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            arrival_offset_ns=round(offset_ms * 1_000_000.0),
        ))
    return result


def run_workload(
    base_url: str,
    model: str | Path,
    requests: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    run_id: str = "vllm-stream-screen",
    served_model_name: str | None = None,
    default_max_tokens: int = 64,
    max_workers: int = 4,
    request_rate: float | None = None,
    timeout_s: float = 120.0,
    seed: int = 20260814,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run one bounded request block and return a raw artifact dictionary.

    The function performs HTTP calls only.  It never starts, probes, retries,
    or stops a server and never interacts with Slurm.  Each request mapping is
    ``{request_id, prompt[, max_tokens, arrival_offset_ms]}``.
    """

    endpoint = client.completion_endpoint(base_url)
    model_path = Path(model)
    client._require(model_path.is_absolute(), "model must be an absolute local path")
    client._require(model_path.is_dir(), f"local model directory does not exist: {model_path}")
    config_path = model_path / "config.json"
    client._require(config_path.is_file(), f"local model config is missing: {config_path}")
    client._require(isinstance(mode, str) and mode.strip(), "mode must be nonempty")
    client._require(math.isfinite(timeout_s) and timeout_s > 0.0,
                    "timeout_s must be finite and positive")
    work_items = _items(
        requests,
        default_max_tokens=default_max_tokens,
        request_rate=request_rate,
    )
    canonical = [{
        "request_id": item.request_id,
        "prompt_sha256": _sha256_bytes(item.prompt.encode("utf-8")),
        "max_tokens": item.max_tokens,
        "arrival_offset_ns": item.arrival_offset_ns,
    } for item in work_items]
    started_at = datetime.now(timezone.utc).isoformat()
    start_ns, end_ns, records = client.run_workload(
        work_items,
        endpoint=endpoint,
        served_model_name=served_model_name or str(model_path),
        timeout_s=timeout_s,
        max_workers=max_workers,
        seed=seed,
        api_key=api_key,
    )
    valid = all(record["valid"] for record in records)
    return {
        "schema_version": client.SCHEMA,
        "evidence_state": "native_vllm_client_stream",
        "run": {
            "run_id": run_id,
            "mode": mode,
            "endpoint": endpoint,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "client_window_ns": end_ns - start_ns,
        },
        "model": {
            "source": "explicit_local_directory",
            "local_path": str(model_path),
            "served_model_name": served_model_name or str(model_path),
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
            "requested_tokens_are_exact": True,
            "retry_count": 0,
            "api_key_present": api_key is not None,
        },
        "clock": {
            "name": "time.perf_counter_ns",
            "scope": "single client process",
            "timestamp_semantics": "complete SSE data event observed by client",
        },
        "workload": {
            "schema_version": client.WORKLOAD_SCHEMA,
            "source": "in_memory_explicit_requests",
            "sha256": _sha256_json(canonical),
            "request_count": len(work_items),
            "max_workers": max_workers,
            "request_rate_per_s": request_rate,
            "default_max_tokens": default_max_tokens,
            "seed": seed,
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
