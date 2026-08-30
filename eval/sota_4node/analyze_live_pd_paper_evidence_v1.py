"""Fail-closed summary of explicit live-vLLM P/D comparison artifacts.

This analyzer never discovers runs.  Each input is an explicitly supplied
``LABEL=PATH`` result produced by the live LMCache P/D harness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "tempo-live-vllm-pd-comparison-1"
LOCAL_ROUTE = "decoder_local_recompute_or_cache"
REMOTE_ROUTE = "remote_prefill_live_kv"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path}: result must be an object")
    return value


def _parse_labeled(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    _require(bool(separator and label and raw_path), "run must be LABEL=PATH")
    return label, Path(raw_path).resolve()


def _analyze_run(label: str, path: Path) -> dict[str, Any]:
    report = _load(path)
    _require(report.get("schema") == EXPECTED_SCHEMA, f"{label}: schema mismatch")
    _require(
        report.get("evidence") == "actual_vllm_disaggregated_prefill_live_kv",
        f"{label}: not live vLLM-owned KV evidence",
    )
    gates = report.get("gates")
    _require(isinstance(gates, dict) and gates, f"{label}: missing gates")
    _require(
        all(value is True for value in gates.values()),
        f"{label}: one or more declared gates failed",
    )
    baseline = report.get("baseline")
    tempo = report.get("tempo")
    paired = report.get("paired")
    _require(isinstance(baseline, dict) and isinstance(tempo, dict), f"{label}: modes missing")
    _require(isinstance(paired, list) and len(paired) == 3, f"{label}: expected 3 pairs")
    baseline_rows = baseline.get("validation")
    tempo_rows = tempo.get("validation")
    _require(
        isinstance(baseline_rows, list) and isinstance(tempo_rows, list)
        and len(baseline_rows) == len(tempo_rows) == 3,
        f"{label}: expected three validation rows per mode",
    )

    rows: list[dict[str, Any]] = []
    for index, (pair, base_row, tempo_row) in enumerate(
        zip(paired, baseline_rows, tempo_rows)
    ):
        _require(
            base_row.get("output_sha256") == tempo_row.get("output_sha256"),
            f"{label}: output mismatch at pair {index}",
        )
        _require(
            base_row.get("prompt_sha256") == tempo_row.get("prompt_sha256"),
            f"{label}: prompt mismatch at pair {index}",
        )
        _require(
            base_row.get("potential_kv") == tempo_row.get("potential_kv")
            == pair.get("potential_kv"),
            f"{label}: potential KV mismatch at pair {index}",
        )
        delta = float(pair["e2e_delta_ms"])
        _require(math.isfinite(delta), f"{label}: non-finite E2E delta")
        background = tempo_row.get("background_decode")
        rows.append({
            "bucket": int(pair["bucket"]),
            "prompt_tokens": int(tempo_row["prompt_tokens"]),
            "completion_tokens": int(tempo_row["completion_tokens"]),
            "potential_kv_bytes": int(pair["potential_kv"]["logical_bytes"]),
            "tempo_route": str(pair["tempo_route"]),
            "e2e_delta_ms": delta,
            "ttft_delta_ms": float(pair["ttft_delta_ms"]),
            "tpot_p99_delta_ms": float(pair["tpot_p99_delta_ms"]),
            "background_streams": (
                int(background.get("concurrent_streams", 1))
                if isinstance(background, dict) else 0
            ),
        })
    return {
        "label": label,
        "result_path": str(path),
        "screen_outcome": report.get("screen_outcome"),
        "promotion_valid": bool(report.get("promotion_valid")),
        "rows": rows,
        "e2e_win_count": sum(row["e2e_delta_ms"] < 0 for row in rows),
        "e2e_delta_median_ms": statistics.median(row["e2e_delta_ms"] for row in rows),
        "route_counts": {
            LOCAL_ROUTE: sum(row["tempo_route"] == LOCAL_ROUTE for row in rows),
            REMOTE_ROUTE: sum(row["tempo_route"] == REMOTE_ROUTE for row in rows),
        },
    }


def analyze(inputs: list[tuple[str, Path]]) -> dict[str, Any]:
    _require(bool(inputs), "at least one explicit run is required")
    labels = [label for label, _ in inputs]
    _require(len(set(labels)) == len(labels), "run labels must be unique")
    runs = [_analyze_run(label, path) for label, path in inputs]
    rows = [row for run in runs for row in run["rows"]]
    all_local = all(row["tempo_route"] == LOCAL_ROUTE for row in rows)
    all_wins = all(row["e2e_delta_ms"] < 0 for row in rows)
    return {
        "schema": "tempo-live-pd-paper-evidence-analysis-1",
        "evidence_level": "single_allocation_mechanism_screen",
        "run_count": len(runs),
        "paired_validation_count": len(rows),
        "runs": runs,
        "aggregate": {
            "e2e_win_count": sum(row["e2e_delta_ms"] < 0 for row in rows),
            "all_paired_e2e_deltas_improve": all_wins,
            "e2e_delta_median_ms": statistics.median(
                row["e2e_delta_ms"] for row in rows
            ),
            "e2e_delta_min_ms": min(row["e2e_delta_ms"] for row in rows),
            "e2e_delta_max_ms": max(row["e2e_delta_ms"] for row in rows),
            "all_observed_tempo_routes_reject_remote_pd": all_local,
            "remote_route_observed": any(
                row["tempo_route"] == REMOTE_ROUTE for row in rows
            ),
        },
        "claim": (
            "Within the supplied same-allocation live-vLLM screens, empirical "
            "admission that rejected remote P/D beat official LMCache always-remote P/D."
        ),
        "claim_boundaries": [
            "The candidate changes admission, not the LMCache/NIXL transport.",
            "No supplied validation selected the remote branch, so crossover behavior is unproven.",
            "The evidence comes from one allocation and is not an independent promotion result.",
            "Mooncake was not measured in this same live-vLLM lifecycle.",
            "Long 32-token loaded requests showed remote/local output divergence and were rejected.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze([_parse_labeled(value) for value in args.run])
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
