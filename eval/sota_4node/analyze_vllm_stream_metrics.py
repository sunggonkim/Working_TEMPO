#!/usr/bin/env python3
"""Analyze explicit native-vLLM streaming artifacts without file discovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

from eval.sota_4node.run_vllm_stream_metrics import SCHEMA as RAW_SCHEMA


ANALYSIS_SCHEMA = "tempo-vllm-stream-metrics-analysis-1"


class AnalysisError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing explicit raw artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _integer(value: Any, field: str) -> int:
    _require(type(value) is int and value >= 0, f"{field} must be a nonnegative integer")
    return value


def _number(value: Any, field: str) -> float:
    _require(type(value) in (int, float), f"{field} must be numeric")
    result = float(value)
    _require(math.isfinite(result) and result >= 0.0,
             f"{field} must be finite and nonnegative")
    return result


def _hex_digest(value: Any, field: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{field} must be a SHA-256 hex digest")
    _require(all(character in "0123456789abcdef" for character in value),
             f"{field} must be lowercase hexadecimal")
    return value


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _digest_bytes(raw)


def percentile(values: Sequence[float], fraction: float) -> float:
    _require(bool(values), "percentile requires samples")
    _require(0.0 < fraction <= 1.0, "percentile fraction must be in (0, 1]")
    ordered = sorted(_number(value, "percentile sample") for value in values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))]


def _distribution(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "distribution requires samples")
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def _request_metric(record: dict[str, Any], prefix: str) -> dict[str, Any]:
    request_id = record.get("request_id")
    _require(isinstance(request_id, str) and request_id, f"{prefix}.request_id must be nonempty")
    _hex_digest(record.get("prompt_sha256"), f"{prefix}.prompt_sha256")
    _integer(record.get("prompt_utf8_bytes"), f"{prefix}.prompt_utf8_bytes")
    expected = _integer(record.get("requested_max_tokens"), f"{prefix}.requested_max_tokens")
    _require(expected >= 2, f"{prefix}.requested_max_tokens must be at least 2")
    scheduled = _integer(record.get("scheduled_dispatch_offset_ns"),
                         f"{prefix}.scheduled_dispatch_offset_ns")
    dispatch = _integer(record.get("dispatch_offset_ns"), f"{prefix}.dispatch_offset_ns")
    _require(dispatch >= scheduled, f"{prefix}.dispatch precedes its scheduled time")
    stream_end = _integer(record.get("stream_end_offset_ns"), f"{prefix}.stream_end_offset_ns")
    _require(stream_end >= dispatch, f"{prefix}.stream end precedes dispatch")
    arrivals_raw = record.get("token_arrival_offsets_ns")
    _require(isinstance(arrivals_raw, list), f"{prefix}.token arrivals must be a list")
    arrivals = [_integer(value, f"{prefix}.token arrival") for value in arrivals_raw]
    _require(all(value >= dispatch for value in arrivals), f"{prefix}.token arrival precedes dispatch")
    _require(all(right >= left for left, right in zip(arrivals, arrivals[1:])),
             f"{prefix}.token arrivals are not monotonic")
    if arrivals:
        _require(stream_end >= arrivals[-1], f"{prefix}.stream end precedes last token")

    tokens = record.get("output_tokens")
    output_text = record.get("output_text")
    _require(isinstance(tokens, list) and all(isinstance(token, str) for token in tokens),
             f"{prefix}.output_tokens must be strings")
    _require(isinstance(output_text, str), f"{prefix}.output_text must be a string")
    token_digest = _hex_digest(record.get("output_token_sha256"),
                               f"{prefix}.output_token_sha256")
    text_digest = _hex_digest(record.get("output_text_sha256"), f"{prefix}.output_text_sha256")
    _require(token_digest == _digest_json(tokens), f"{prefix}.output token digest mismatch")
    _require(text_digest == _digest_bytes(output_text.encode("utf-8")),
             f"{prefix}.output text digest mismatch")
    _require(len(tokens) == len(arrivals), f"{prefix}.token/timestamp count mismatch")

    violations = record.get("contract_violations")
    _require(isinstance(violations, list) and all(isinstance(value, str) for value in violations),
             f"{prefix}.contract_violations must be strings")
    error_value = record.get("error")
    _require(error_value is None or isinstance(error_value, str), f"{prefix}.error is invalid")
    status = record.get("http_status")
    _require(status is None or type(status) is int, f"{prefix}.http_status is invalid")
    usage = record.get("usage")
    usage_valid = isinstance(usage, dict) and all(
        type(usage.get(name)) is int and usage[name] >= 0
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    if usage_valid:
        usage_valid = usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    strict_valid = (
        status == 200
        and error_value is None
        and not violations
        and record.get("done_seen") is True
        and record.get("finish_reason") == "length"
        and isinstance(record.get("response_id"), str)
        and bool(record.get("response_id"))
        and isinstance(record.get("response_model"), str)
        and bool(record.get("response_model"))
        and usage_valid
        and len(tokens) == expected
        and usage["completion_tokens"] == expected
    )
    _require(record.get("valid") is strict_valid, f"{prefix}.valid disagrees with strict contract")

    if not strict_valid:
        return {
            "request_id": request_id,
            "valid": False,
            "contract": (request_id, record["prompt_sha256"], expected, scheduled),
            "output": None,
            "metric": None,
        }
    itl_ms = [
        (right - left) / 1_000_000.0
        for left, right in zip(arrivals, arrivals[1:])
    ]
    ttft_ms = (arrivals[0] - dispatch) / 1_000_000.0
    e2e_ms = (arrivals[-1] - dispatch) / 1_000_000.0
    tpot_ms = (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1_000_000.0
    return {
        "request_id": request_id,
        "valid": True,
        "contract": (request_id, record["prompt_sha256"], expected, scheduled),
        "output": (tuple(tokens), output_text),
        "metric": {
            "request_id": request_id,
            "dispatch_offset_ns": dispatch,
            "last_token_offset_ns": arrivals[-1],
            "completion_tokens": len(tokens),
            "ttft_ms": ttft_ms,
            "itl_ms": itl_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "stream_close_overhead_ms": (stream_end - arrivals[-1]) / 1_000_000.0,
        },
    }


def _slo_pass(metric: dict[str, Any], slo: dict[str, float | None]) -> bool:
    return (
        metric["ttft_ms"] <= slo["ttft_ms"]
        and metric["tpot_ms"] <= slo["tpot_ms"]
        and (slo["itl_ms"] is None or max(metric["itl_ms"]) <= slo["itl_ms"])
        and (slo["e2e_ms"] is None or metric["e2e_ms"] <= slo["e2e_ms"])
    )


def analyze_run(path: Path, label: str, slo: dict[str, float | None]) -> dict[str, Any]:
    raw = _load(path)
    _require(raw.get("schema_version") == RAW_SCHEMA, f"{path}: unexpected schema")
    _require(raw.get("evidence_state") == "native_vllm_client_stream",
             f"{path}: unexpected evidence state")
    run = raw.get("run")
    model = raw.get("model")
    endpoint = raw.get("endpoint_contract")
    workload = raw.get("workload")
    validation = raw.get("validation")
    _require(isinstance(run, dict), f"{path}: run must be an object")
    _require(isinstance(model, dict), f"{path}: model must be an object")
    _require(isinstance(endpoint, dict), f"{path}: endpoint_contract must be an object")
    _require(isinstance(workload, dict), f"{path}: workload must be an object")
    _require(isinstance(validation, dict), f"{path}: validation must be an object")
    _require(model.get("source") == "explicit_local_directory",
             f"{path}: model source is not an explicit local directory")
    _require(isinstance(model.get("local_path"), str) and Path(model["local_path"]).is_absolute(),
             f"{path}: model.local_path must be absolute")
    config_digest = _hex_digest(model.get("config_sha256"), f"{path}: model.config_sha256")
    _require(model.get("offline_server_assumption") is True,
             f"{path}: offline server assumption is not recorded")
    _require(endpoint.get("api") == "OpenAI-compatible POST /v1/completions",
             f"{path}: unsupported endpoint contract")
    _require(endpoint.get("stream") is True and endpoint.get("logprobs") == 1,
             f"{path}: exact streaming logprob contract is missing")
    _require(endpoint.get("ignore_eos") is True and endpoint.get("retry_count") == 0,
             f"{path}: exact-length/no-retry contract is missing")
    expected_count = _integer(workload.get("request_count"), f"{path}: workload.request_count")
    _hex_digest(workload.get("sha256"), f"{path}: workload.sha256")
    records = raw.get("requests")
    _require(isinstance(records, list) and len(records) == expected_count and records,
             f"{path}: request count mismatch")
    parsed = [_request_metric(record, f"{path}: requests[{index}]")
              for index, record in enumerate(records)]
    identifiers = [value["request_id"] for value in parsed]
    _require(len(identifiers) == len(set(identifiers)), f"{path}: duplicate request records")
    all_valid = all(value["valid"] for value in parsed)
    valid_count = sum(value["valid"] for value in parsed)
    _require(validation.get("all_requests_valid") is all_valid,
             f"{path}: all_requests_valid mismatch")
    _require(validation.get("performance_claim_allowed") is all_valid,
             f"{path}: performance_claim_allowed mismatch")
    _require(validation.get("valid_requests") == valid_count,
             f"{path}: valid request count mismatch")
    _require(validation.get("invalid_requests") == len(parsed) - valid_count,
             f"{path}: invalid request count mismatch")

    performance = None
    if all_valid:
        metrics = [value["metric"] for value in parsed]
        first_dispatch = min(metric["dispatch_offset_ns"] for metric in metrics)
        last_completion = max(metric["last_token_offset_ns"] for metric in metrics)
        duration_s = (last_completion - first_dispatch) / 1_000_000_000.0
        _require(duration_s > 0.0, f"{path}: measurement window is not positive")
        passed = [metric for metric in metrics if _slo_pass(metric, slo)]
        itls = [value for metric in metrics for value in metric["itl_ms"]]
        total_tokens = sum(metric["completion_tokens"] for metric in metrics)
        passed_tokens = sum(metric["completion_tokens"] for metric in passed)
        performance = {
            "measurement_window_s": duration_s,
            "completed_requests": len(metrics),
            "completion_tokens": total_tokens,
            "request_throughput_per_s": len(metrics) / duration_s,
            "output_token_throughput_per_s": total_tokens / duration_s,
            "ttft_ms": _distribution([metric["ttft_ms"] for metric in metrics]),
            "tpot_ms": _distribution([metric["tpot_ms"] for metric in metrics]),
            "itl_ms": _distribution(itls),
            "e2e_ms": _distribution([metric["e2e_ms"] for metric in metrics]),
            "request_metrics": metrics,
            "slo_goodput": {
                "successful_requests": len(passed),
                "successful_completion_tokens": passed_tokens,
                "request_goodput_per_s": len(passed) / duration_s,
                "output_token_goodput_per_s": passed_tokens / duration_s,
                "success_fraction": len(passed) / len(metrics),
            },
        }
    return {
        "label": label,
        "path": str(path),
        "run_id": run.get("run_id"),
        "mode": run.get("mode"),
        "model_config_sha256": config_digest,
        "evidence_valid": all_valid,
        "invalid_request_ids": [value["request_id"] for value in parsed if not value["valid"]],
        "performance": performance,
        "_contracts": sorted(value["contract"] for value in parsed),
        "_outputs": {value["request_id"]: value["output"] for value in parsed},
    }


def analyze(
    runs: Sequence[tuple[str, Path]],
    *,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    itl_slo_ms: float | None = None,
    e2e_slo_ms: float | None = None,
) -> dict[str, Any]:
    _require(bool(runs), "at least one --run is required")
    labels = [label for label, _ in runs]
    _require(len(labels) == len(set(labels)), "run labels must be unique")
    slo: dict[str, float | None] = {
        "ttft_ms": _number(ttft_slo_ms, "ttft_slo_ms"),
        "tpot_ms": _number(tpot_slo_ms, "tpot_slo_ms"),
        "itl_ms": None if itl_slo_ms is None else _number(itl_slo_ms, "itl_slo_ms"),
        "e2e_ms": None if e2e_slo_ms is None else _number(e2e_slo_ms, "e2e_slo_ms"),
    }
    analyzed = [analyze_run(path, label, slo) for label, path in runs]
    all_valid = all(run["evidence_valid"] for run in analyzed)
    model_equal = len({run["model_config_sha256"] for run in analyzed}) == 1
    contract_equal = all(run["_contracts"] == analyzed[0]["_contracts"] for run in analyzed[1:])
    output_equal: bool | None
    if len(analyzed) == 1:
        output_equal = None
    elif not all_valid or not contract_equal:
        output_equal = False
    else:
        output_equal = all(run["_outputs"] == analyzed[0]["_outputs"] for run in analyzed[1:])
    correctness_met = all_valid and model_equal and contract_equal and output_equal is not False
    public_runs = []
    for run in analyzed:
        public_runs.append({key: value for key, value in run.items() if not key.startswith("_")})
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "evidence_state": "native_vllm_streaming_analysis",
        "slo": slo,
        "runs": public_runs,
        "correctness": {
            "all_raw_contracts_valid": all_valid,
            "same_model_config": model_equal,
            "same_request_contract": contract_equal,
            "cross_run_output_equivalence": output_equal,
            "correctness_met": correctness_met,
        },
        "comparison_claim_allowed": len(analyzed) >= 2 and correctness_met,
        "metric_definitions": {
            "ttft_ms": "first token SSE arrival minus request dispatch",
            "itl_ms": "each consecutive pair of output-token SSE arrivals",
            "tpot_ms": "last-minus-first token arrival divided by output_tokens-1",
            "e2e_ms": "last output-token SSE arrival minus request dispatch",
            "throughput_window": "earliest dispatch through latest last-token arrival",
            "slo_goodput": "SLO-passing requests or their output tokens divided by the full throughput window",
        },
        "limitations": [
            "client-observed SSE timing includes network and client parsing",
            "a single run checks its response contract but cannot establish cross-mode equivalence",
            "invalid or incomplete requests suppress every performance aggregate for that run",
        ],
    }


def _run_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    return label, Path(raw_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_run_argument, required=True,
                        help="explicit LABEL=RAW_JSON artifact; repeat for comparisons")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, default=1000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=100.0)
    parser.add_argument("--itl-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    args = parser.parse_args(argv)
    try:
        report = analyze(
            args.run,
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
            itl_slo_ms=args.itl_slo_ms,
            e2e_slo_ms=args.e2e_slo_ms,
        )
    except AnalysisError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "runs": len(report["runs"]),
        "correctness_met": report["correctness"]["correctness_met"],
        "comparison_claim_allowed": report["comparison_claim_allowed"],
    }, sort_keys=True))
    return 0 if report["correctness"]["correctness_met"] else 2


if __name__ == "__main__":
    sys.exit(main())
