"""Bounded client and analyzer for a live vLLM P/D admission screen.

The two compared lifecycles use the same 2xTP8 servers, prompts, model and
potential KV payload.  The baseline always performs remote prefill.  The
controller calibrates remote-prefill and decoder-local service on separate
prompts, then admits remote prefill only when its measured E2E advantage is
larger than a frozen margin.  Validation prompts are never used for tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA = "tempo-live-vllm-pd-screen-1"
SERVED_MODEL = "tempo-tinyllama-live-pd"
BUCKET_REPETITIONS = (96, 384, 672)
OUTPUT_TOKENS = 32
REMOTE_ADVANTAGE_MARGIN_MS = 5.0
REQUEST_TIMEOUT_S = 180.0
PROMPT_UNIT = "Measured KV admission must preserve decode latency and output correctness. "


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    _require(bool(values), "percentile requires nonempty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _prompt(kind: str, bucket: int, repetitions: int) -> str:
    return f"{kind} bucket {bucket}. " + PROMPT_UNIT * repetitions


def _request_json(url: str, body: dict[str, Any], request_id: str) -> dict[str, Any]:
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Connection": "close",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        raw = response.read()
        _require(response.status == 200, f"HTTP status {response.status} from {url}")
    value = json.loads(raw)
    _require(isinstance(value, dict), "non-stream response must be an object")
    return value


def _stream_decode(
    url: str,
    body: dict[str, Any],
    request_id: str,
    origin_ns: int,
) -> dict[str, Any]:
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Connection": "close",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    arrivals_ns: list[int] = []
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        _require(response.status == 200, f"HTTP status {response.status} from {url}")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                continue
            arrivals_ns.append(time.perf_counter_ns())
            pieces.append(str(choice.get("text", "")))
    finished_ns = time.perf_counter_ns()
    _require(len(arrivals_ns) == OUTPUT_TOKENS, (
        f"expected {OUTPUT_TOKENS} token events, got {len(arrivals_ns)}"
    ))
    _require(usage is not None, "stream response did not include usage")
    _require(int(usage.get("completion_tokens", -1)) == OUTPUT_TOKENS,
             "completion token count mismatch")
    gaps_ms = [
        (right - left) / 1_000_000.0
        for left, right in zip(arrivals_ns, arrivals_ns[1:])
    ]
    output = "".join(pieces)
    return {
        "http_status": 200,
        "prompt_tokens": int(usage.get("prompt_tokens", -1)),
        "completion_tokens": int(usage["completion_tokens"]),
        "output_sha256": _sha256_text(output),
        "output_text": output,
        "ttft_ms": (arrivals_ns[0] - origin_ns) / 1_000_000.0,
        "e2e_ms": (finished_ns - origin_ns) / 1_000_000.0,
        "tpot_p50_ms": statistics.median(gaps_ms),
        "tpot_p99_ms": _percentile(gaps_ms, 0.99),
        "tpot_max_ms": max(gaps_ms),
        "token_arrival_count": len(arrivals_ns),
    }


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5.0) as response:
        _require(response.status == 200, f"metrics HTTP status {response.status}")
        return response.read().decode("utf-8")


METRIC_NAMES = (
    "vllm:nixl_bytes_transferred_sum",
    "vllm:nixl_bytes_transferred_count",
    "vllm:nixl_num_failed_transfers_total",
    "vllm:nixl_num_failed_notifications_total",
    "vllm:external_prefix_cache_queries_total",
    "vllm:external_prefix_cache_hits_total",
)


def _metric_snapshot(base_url: str) -> dict[str, float]:
    text = _fetch_text(base_url.rstrip("/") + "/metrics")
    result = {name: 0.0 for name in METRIC_NAMES}
    for raw_line in text.splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        metric, separator, raw_value = raw_line.partition(" ")
        if not separator:
            continue
        name = metric.split("{", 1)[0]
        if name not in result:
            continue
        try:
            value = float(raw_value.strip().split()[0])
        except ValueError:
            continue
        result[name] += value
    return result


def _metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: after[name] - before[name] for name in METRIC_NAMES}


def _base_decode_body(prompt: str) -> dict[str, Any]:
    return {
        "model": SERVED_MODEL,
        "prompt": prompt,
        "max_tokens": OUTPUT_TOKENS,
        "min_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 20260814,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": 1,
    }


def _run_direct(
    decode_url: str,
    prompt: str,
    request_id: str,
) -> dict[str, Any]:
    before = _metric_snapshot(decode_url)
    origin_ns = time.perf_counter_ns()
    result = _stream_decode(
        decode_url.rstrip("/") + "/v1/completions",
        _base_decode_body(prompt),
        request_id,
        origin_ns,
    )
    after = _metric_snapshot(decode_url)
    result.update({
        "route": "decoder_local_recompute_or_cache",
        "prefill_ms": 0.0,
        "metrics": {"decode": _metric_delta(before, after)},
    })
    return result


def _run_remote(
    prefill_url: str,
    decode_url: str,
    prompt: str,
    request_id: str,
) -> dict[str, Any]:
    before_prefill = _metric_snapshot(prefill_url)
    before_decode = _metric_snapshot(decode_url)
    origin_ns = time.perf_counter_ns()
    prefill_body = {
        "model": SERVED_MODEL,
        "prompt": prompt,
        "max_tokens": 1,
        "min_tokens": 1,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 20260814,
        "stream": False,
        "kv_transfer_params": {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        },
    }
    prefill = _request_json(
        prefill_url.rstrip("/") + "/v1/completions",
        prefill_body,
        request_id,
    )
    prefill_finished_ns = time.perf_counter_ns()
    transfer_params = prefill.get("kv_transfer_params")
    _require(isinstance(transfer_params, dict) and transfer_params,
             "prefill response did not return kv_transfer_params")
    decode_body = _base_decode_body(prompt)
    decode_body["kv_transfer_params"] = transfer_params
    result = _stream_decode(
        decode_url.rstrip("/") + "/v1/completions",
        decode_body,
        request_id,
        origin_ns,
    )
    after_prefill = _metric_snapshot(prefill_url)
    after_decode = _metric_snapshot(decode_url)
    result.update({
        "route": "remote_prefill_live_kv",
        "prefill_ms": (prefill_finished_ns - origin_ns) / 1_000_000.0,
        "kv_transfer_params_keys": sorted(transfer_params),
        "metrics": {
            "prefill": _metric_delta(before_prefill, after_prefill),
            "decode": _metric_delta(before_decode, after_decode),
        },
    })
    return result


def _potential_kv_bytes(prompt_tokens: int) -> dict[str, int]:
    # TinyLlama: 22 layers, 4 KV heads, head_dim 64, BF16 K+V.  TP8
    # replicates the four KV heads two ways, so physical bytes are 2x logical.
    _require(prompt_tokens > 0, "prompt token count must be positive")
    logical_per_token = 22 * 4 * 64 * 2 * 2
    return {
        "logical_bytes": prompt_tokens * logical_per_token,
        "tp8_physical_bytes": prompt_tokens * logical_per_token * 2,
    }


def run_lifecycle(
    *,
    mode: str,
    prefill_url: str,
    decode_url: str,
    model_path: Path,
) -> dict[str, Any]:
    _require(mode in {"lmcache_always_remote", "tempo_admission"}, "invalid mode")
    _require(model_path.is_absolute() and (model_path / "config.json").is_file(),
             "model must be an absolute local model directory")

    # One unmeasured request warms CUDA graphs, HTTP, connector metadata and
    # metrics plumbing identically in both fresh lifecycles.
    _run_direct(decode_url, _prompt("warmup", 0, 32), f"{mode}-warmup")

    calibration: list[dict[str, Any]] = []
    decisions: dict[int, str] = {}
    for bucket, repetitions in enumerate(BUCKET_REPETITIONS):
        remote = _run_remote(
            prefill_url, decode_url,
            _prompt("calibration-remote", bucket, repetitions),
            f"{mode}-cal-remote-{bucket}",
        )
        direct = _run_direct(
            decode_url,
            _prompt("calibration-direct", bucket, repetitions),
            f"{mode}-cal-direct-{bucket}",
        )
        remote_advantage_ms = direct["e2e_ms"] - remote["e2e_ms"]
        chosen = (
            "remote_prefill_live_kv"
            if remote_advantage_ms >= REMOTE_ADVANTAGE_MARGIN_MS
            else "decoder_local_recompute_or_cache"
        )
        decisions[bucket] = chosen
        calibration.append({
            "bucket": bucket,
            "repetitions": repetitions,
            "remote": remote,
            "direct": direct,
            "remote_advantage_ms": remote_advantage_ms,
            "frozen_decision": chosen,
        })

    validation: list[dict[str, Any]] = []
    for bucket, repetitions in enumerate(BUCKET_REPETITIONS):
        prompt = _prompt("validation", bucket, repetitions)
        request_id = f"live-pd-validation-{bucket}"
        route = (
            "remote_prefill_live_kv"
            if mode == "lmcache_always_remote"
            else decisions[bucket]
        )
        if route == "remote_prefill_live_kv":
            record = _run_remote(prefill_url, decode_url, prompt, request_id)
        else:
            record = _run_direct(decode_url, prompt, request_id)
        record.update({
            "request_id": request_id,
            "bucket": bucket,
            "repetitions": repetitions,
            "prompt_sha256": _sha256_text(prompt),
            "potential_kv": _potential_kv_bytes(record["prompt_tokens"]),
        })
        validation.append(record)

    return {
        "schema": SCHEMA,
        "mode": mode,
        "evidence": "actual_vllm_disaggregated_prefill_live_kv",
        "model": str(model_path),
        "served_model": SERVED_MODEL,
        "topology": {
            "nodes": 4,
            "gpus": 16,
            "prefill": {"nodes": 2, "tensor_parallel_size": 8},
            "decode": {"nodes": 2, "tensor_parallel_size": 8},
        },
        "controller": {
            "policy": "per-payload measured admission",
            "remote_advantage_margin_ms": REMOTE_ADVANTAGE_MARGIN_MS,
            "validation_used_for_tuning": False,
            "decisions": {str(key): value for key, value in decisions.items()},
        },
        "calibration": calibration,
        "validation": validation,
        "valid": all(item["completion_tokens"] == OUTPUT_TOKENS for item in validation),
        "limitations": [
            "one calibration observation per route and payload bucket",
            "TinyLlama is a mechanism screen, not a large-model SOTA result",
            "controller may move zero P/D bytes when it rejects remote prefill; potential KV payload is held constant",
        ],
    }


def combine(baseline: dict[str, Any], tempo: dict[str, Any]) -> dict[str, Any]:
    _require(baseline.get("schema") == SCHEMA and tempo.get("schema") == SCHEMA,
             "input schema mismatch")
    _require(baseline.get("mode") == "lmcache_always_remote", "invalid baseline mode")
    _require(tempo.get("mode") == "tempo_admission", "invalid Tempo mode")
    left = baseline["validation"]
    right = tempo["validation"]
    _require(len(left) == len(right) == len(BUCKET_REPETITIONS),
             "validation cardinality mismatch")
    paired: list[dict[str, Any]] = []
    for baseline_row, tempo_row in zip(left, right):
        _require(baseline_row["request_id"] == tempo_row["request_id"],
                 "request identity mismatch")
        _require(baseline_row["prompt_sha256"] == tempo_row["prompt_sha256"],
                 "prompt mismatch")
        _require(baseline_row["potential_kv"] == tempo_row["potential_kv"],
                 "potential KV byte mismatch")
        _require(baseline_row["output_sha256"] == tempo_row["output_sha256"],
                 "output mismatch")
        paired.append({
            "request_id": baseline_row["request_id"],
            "bucket": baseline_row["bucket"],
            "potential_kv": baseline_row["potential_kv"],
            "tempo_route": tempo_row["route"],
            "e2e_delta_ms": tempo_row["e2e_ms"] - baseline_row["e2e_ms"],
            "ttft_delta_ms": tempo_row["ttft_ms"] - baseline_row["ttft_ms"],
            "tpot_p99_delta_ms": tempo_row["tpot_p99_ms"] - baseline_row["tpot_p99_ms"],
        })
    e2e_deltas = [row["e2e_delta_ms"] for row in paired]
    baseline_tail = max(row["tpot_p99_ms"] for row in left)
    tempo_tail = max(row["tpot_p99_ms"] for row in right)
    failed_metric = "vllm:nixl_num_failed_transfers_total"
    failed_transfers = sum(
        value
        for lifecycle in (baseline, tempo)
        for row in lifecycle["validation"]
        for endpoint in row["metrics"].values()
        for name, value in endpoint.items()
        if name == failed_metric
    )
    gates = {
        "all_lifecycles_valid": bool(baseline["valid"] and tempo["valid"]),
        "identical_requests_outputs_and_potential_kv": True,
        "same_gpu_budget": baseline["topology"] == tempo["topology"],
        "no_nixl_transfer_failures": failed_transfers == 0,
        "tempo_e2e_wins_at_least_two_of_three": sum(value < 0 for value in e2e_deltas) >= 2,
        "tempo_median_e2e_improves": statistics.median(e2e_deltas) < 0,
        "tempo_tpot_p99_max_within_10_percent": tempo_tail <= baseline_tail * 1.10,
    }
    passes = all(gates.values())
    return {
        "schema": "tempo-live-vllm-pd-comparison-1",
        "evidence": "actual_vllm_disaggregated_prefill_live_kv",
        "baseline": baseline,
        "tempo": tempo,
        "paired": paired,
        "summary": {
            "e2e_delta_median_ms": statistics.median(e2e_deltas),
            "e2e_win_count": sum(value < 0 for value in e2e_deltas),
            "baseline_tpot_p99_max_ms": baseline_tail,
            "tempo_tpot_p99_max_ms": tempo_tail,
            "failed_nixl_transfers": failed_transfers,
        },
        "gates": gates,
        "screen_outcome": "live_pd_candidate_pass" if passes else "live_pd_candidate_revise",
        "promotion_valid": False,
        "claim_boundary": (
            "single-allocation TinyLlama live-P/D mechanism evidence; independent "
            "large-model replication is required for a SOTA claim"
        ),
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--mode", choices=("lmcache_always_remote", "tempo_admission"), required=True)
    run.add_argument("--prefill-url", required=True)
    run.add_argument("--decode-url", required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("combine")
    merge.add_argument("--baseline", type=Path, required=True)
    merge.add_argument("--tempo", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_lifecycle(
            mode=args.mode,
            prefill_url=args.prefill_url,
            decode_url=args.decode_url,
            model_path=args.model.resolve(),
        )
    else:
        result = combine(_load(args.baseline), _load(args.tempo))
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
