#!/usr/bin/env python3
"""Post-hoc analysis for the phase-gated six-arm scale-paper campaign.

The frozen campaign verifier intentionally rejects a paper claim when a policy
has no completed victims.  That is the right claim boundary, but it should not
erase a real collapse result.  This analyzer therefore preserves null p99 for
zero-completion policies and reports completion, SLO, reject, and failure
counts over the offered population.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BLOCKS = (
    "00_netkv_a",
    "01_dynamo_a",
    "02_tempo_a",
    "03_tempo_b",
    "04_dynamo_b",
    "05_netkv_b",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"artifact is not an object: {path}")
    return value


def _metrics(analysis: dict[str, Any]) -> dict[str, Any]:
    all_metrics = analysis.get("all")
    _require(isinstance(all_metrics, dict), "analysis.all is missing")
    victim = all_metrics.get("victim")
    _require(isinstance(victim, dict), "analysis.all.victim is missing")
    e2e = victim.get("e2e_ms")
    _require(isinstance(e2e, dict), "analysis.all.victim.e2e_ms is missing")
    return {
        "offered": all_metrics.get("offered_victims"),
        "completed": all_metrics.get("completed_victims"),
        "slo_good": all_metrics.get("slo_good_victims"),
        "global_rejects": all_metrics.get("global_rejects"),
        "failures": all_metrics.get("failures"),
        "slo_fraction_of_offered": all_metrics.get(
            "slo_attainment_fraction_of_offered"),
        "e2e_ms": {
            "p50": e2e.get("p50"),
            "p95": e2e.get("p95"),
            "p99": e2e.get("p99"),
        },
    }


def _regime_metrics(analysis: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("normal", "miss_hot", "remote_favorable"):
        value = analysis.get(name)
        _require(isinstance(value, dict), f"analysis.{name} is missing")
        result[name] = _metrics({"all": value})
    return result


def _cojob_summary(cojob: dict[str, Any]) -> dict[str, Any]:
    summary = cojob.get("summary")
    _require(isinstance(summary, dict), "cojob.summary is missing")
    return {
        "background_completion_p50_ms": summary.get(
            "background_completion_p50_ms"),
        "background_completion_p99_ms": summary.get(
            "background_completion_p99_ms"),
        "global_token_tail_p50_ms": summary.get("global_token_tail_p50_ms"),
        "global_token_tail_p99_ms": summary.get("global_token_tail_p99_ms"),
        "overall_correctness_met": cojob.get("overall_correctness_met"),
        "evidence_state": cojob.get("evidence_state"),
    }


def analyze(root: Path) -> dict[str, Any]:
    root = root.resolve()
    _require(root.is_dir(), f"campaign root is missing: {root}")
    arms: list[dict[str, Any]] = []
    for block in BLOCKS:
        block_root = root / block
        inference = _read(block_root / "inference" / "result.json")
        cojob = _read(block_root / "cojob" / "result.json")
        receipt = _read(block_root / "block_execution_receipt.json")
        analysis = inference.get("analysis")
        _require(isinstance(analysis, dict), f"{block}: analysis is missing")
        _require(
            analysis.get("same_population_ready_for_campaign_analysis") is True,
            f"{block}: population readiness is false",
        )
        _require(
            analysis.get("terminal_contract_valid_for_every_block") is True,
            f"{block}: terminal contract is invalid",
        )
        _require(receipt.get("inference_status") == "complete",
                 f"{block}: inference receipt is not complete")
        arms.append({
            "block": block,
            "policy": receipt.get("policy"),
            "inference": {
                "profile": inference.get("profile"),
                "performance_claim_allowed": inference.get(
                    "performance_claim_allowed"),
                "all": _metrics(analysis),
                "regimes": _regime_metrics(analysis),
            },
            "cojob": _cojob_summary(cojob),
            "receipt": {
                "cojob_outcome": receipt.get("cojob_outcome"),
                "cojob_exit_code": receipt.get("cojob_exit_code"),
                "phase_gate": receipt.get("phase_gate"),
            },
        })
    return {
        "schema": "tempo-scale-paper-causal-sota-v2-posthoc-v1",
        "root": str(root),
        "blocks": list(BLOCKS),
        "claim_boundary": {
            "performance_claim_allowed": False,
            "purpose": "discovery evidence with offered-population collapse preserved",
        },
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        output = args.output.resolve()
        _require(output.parent == args.root.resolve(),
                 "post-hoc output must be directly below campaign root")
        _require(not output.exists(), f"refusing to overwrite: {output}")
        output.write_text(encoded, encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
