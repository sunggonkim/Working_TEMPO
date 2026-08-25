#!/usr/bin/env python3
"""Analyze the same-allocation C10 paper-baseline extension of held-out C9."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "tempo-go-c10-paper-sota-analysis-v1"
REGIMES = ("normal", "miss_hot", "remote_favorable")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _metrics(analysis: dict[str, object], regime: str) -> dict[str, object]:
    value = analysis.get(regime)
    _require(isinstance(value, dict), f"missing regime {regime}")
    victim = value.get("victim")
    _require(isinstance(victim, dict), f"missing victim metrics for {regime}")
    e2e = victim.get("e2e_ms")
    ttft = victim.get("ttft_ms")
    tpot = victim.get("tpot_ms")
    _require(all(isinstance(item, dict) for item in (e2e, ttft, tpot)),
             f"latency metrics are incomplete for {regime}")
    return {
        "offered": value.get("offered_victims"),
        "completed": value.get("completed_victims"),
        "slo_good": value.get("slo_good_victims"),
        "slo_attainment": value.get("slo_attainment_fraction_of_offered"),
        "e2e_ms": {
            name: e2e[name] for name in ("mean", "p50", "p95", "p99")
        },
        "ttft_ms": {
            name: ttft[name] for name in ("mean", "p50", "p95", "p99")
        },
        "tpot_ms": {
            name: tpot[name] for name in ("mean", "p50", "p95", "p99")
        },
        "route_counts": value.get("route_counts"),
        "edge_counts": value.get("edge_counts"),
        "failures": value.get("failures"),
        "global_rejects": value.get("global_rejects"),
    }


def _execution_identity(result: dict[str, object]) -> dict[str, object]:
    raw_path = Path(str(result["raw"])).resolve()
    _require(raw_path.is_file(), "raw result bundle is missing")
    _require(_sha256(raw_path) == result.get("raw_sha256"),
             "raw result digest differs")
    raw = _load(raw_path)
    execution = raw.get("independent_validation_execution")
    _require(isinstance(execution, dict),
             "held-out workload execution receipt is missing")
    return {
        "raw": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "request_seed": execution.get("request_seed"),
        "block_order": execution.get("block_order"),
        "arrival_jitter": execution.get("arrival_jitter"),
        "p_only_prompt_namespace": execution.get("p_only_prompt_namespace"),
    }


def _paper_baseline_admission_evidence(
    raw: dict[str, object],
) -> dict[str, object]:
    """Prove the compatibility receipt did not enforce TEMPO admission.

    The frozen C8 workload requires decoder-admission lifecycle receipts.  A
    paper baseline must emit those receipts for evidence compatibility, while
    never waiting, throttling, prioritizing, or reserving capacity.  Verify
    that property from every measured artifact rather than trusting the
    adapter configuration.
    """

    artifacts = raw.get("artifacts")
    _require(isinstance(artifacts, dict) and artifacts,
             "paper baseline measured artifacts are missing")
    artifact_count = 0
    receipt_count = 0
    released_count = 0
    snapshot_count = 0
    admission_classes: dict[str, int] = {}
    for artifact_name, artifact_path_raw in artifacts.items():
        _require(isinstance(artifact_path_raw, str) and artifact_path_raw,
                 f"invalid artifact path for {artifact_name}")
        artifact_path = Path(artifact_path_raw).resolve()
        _require(artifact_path.is_file(),
                 f"paper baseline artifact is missing: {artifact_path}")
        artifact = _load(artifact_path)
        artifact_count += 1

        endpoint = artifact.get("router_decision_endpoint")
        _require(isinstance(endpoint, dict),
                 f"router endpoint evidence is missing: {artifact_name}")
        snapshot = endpoint.get("decoder_business_admission")
        _require(isinstance(snapshot, dict),
                 f"admission snapshot is missing: {artifact_name}")
        _require(snapshot.get("mode") == "evidence_only_no_throttle",
                 f"baseline admission mode enforced policy: {artifact_name}")
        _require(snapshot.get("policy_effect") == "none",
                 f"baseline admission snapshot has a policy effect: {artifact_name}")
        _require(snapshot.get("leases") == 0,
                 f"baseline admission lease leaked: {artifact_name}")
        snapshot_count += 1

        decisions = artifact.get("router_decisions")
        _require(isinstance(decisions, list),
                 f"router decisions are missing: {artifact_name}")
        for decision in decisions:
            _require(isinstance(decision, dict),
                     f"invalid router decision: {artifact_name}")
            receipt = decision.get("frontend_decoder_business_admission")
            if receipt is None:
                continue
            _require(isinstance(receipt, dict),
                     f"invalid admission receipt: {artifact_name}")
            _require(receipt.get("mode") == "evidence_only_no_throttle",
                     f"baseline receipt used an enforcing mode: {artifact_name}")
            _require(receipt.get("policy_effect") == "none",
                     f"baseline receipt has a policy effect: {artifact_name}")
            _require(receipt.get("wait_ns") == 0,
                     f"baseline admission waited: {artifact_name}")
            _require(receipt.get("starvation_escape") is False,
                     f"baseline admission altered ordering: {artifact_name}")
            _require(receipt.get("status") == "released",
                     f"baseline admission receipt was not released: {artifact_name}")
            admission_class = receipt.get("admission_class")
            _require(admission_class in {"protected", "background"},
                     f"invalid baseline evidence class: {artifact_name}")
            admission_classes[str(admission_class)] = (
                admission_classes.get(str(admission_class), 0) + 1)
            receipt_count += 1
            released_count += 1

    _require(receipt_count > 0,
             "paper baseline emitted no decoder lifecycle receipts")
    _require(admission_classes.get("protected", 0) > 0,
             "paper baseline emitted no victim admission receipts")
    _require(admission_classes.get("background", 0) > 0,
             "paper baseline emitted no background admission receipts")
    return {
        "gate": True,
        "mode": "evidence_only_no_throttle",
        "policy_effect": "none",
        "artifact_count": artifact_count,
        "snapshot_count": snapshot_count,
        "receipt_count": receipt_count,
        "released_count": released_count,
        "admission_classes": admission_classes,
        "max_wait_ns": 0,
    }


def _reduction(reference: float, value: float) -> float:
    _require(math.isfinite(reference) and reference > 0.0,
             "reference latency must be positive")
    _require(math.isfinite(value) and value > 0.0,
             "candidate latency must be positive")
    return (reference - value) / reference


def _optional_reduction(reference: object, value: object) -> float | None:
    if not isinstance(reference, (int, float)) or not isinstance(
        value, (int, float)
    ):
        return None
    if not math.isfinite(float(reference)) or float(reference) <= 0.0:
        return None
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        return None
    return _reduction(float(reference), float(value))


def analyze(
    *, manifest_path: Path, tempo_result_path: Path,
    parent_analysis_path: Path, baseline_paths: dict[str, Path],
) -> dict[str, object]:
    manifest = _load(manifest_path)
    _require(manifest.get("schema") == "tempo-go-c10-paper-sota-extension-v1",
             "paper SOTA manifest schema differs")
    _require(set(baseline_paths) == set(manifest["policies"]),
             "baseline result population differs from manifest")

    tempo_result = _load(tempo_result_path)
    tempo_analysis = tempo_result.get("analysis")
    _require(isinstance(tempo_analysis, dict), "TEMPO arm analysis is missing")
    tempo_identity = _execution_identity(tempo_result)
    job_ids = {str(tempo_result.get("slurm_job_id"))}

    baselines: dict[str, dict[str, object]] = {}
    for policy, path in sorted(baseline_paths.items()):
        result = _load(path)
        analysis = result.get("analysis")
        _require(isinstance(analysis, dict), f"{policy} analysis is missing")
        identity = _execution_identity(result)
        raw = _load(Path(str(result["raw"])).resolve())
        admission_evidence = _paper_baseline_admission_evidence(raw)
        _require(
            identity["request_seed"] == tempo_identity["request_seed"]
            and identity["block_order"] == tempo_identity["block_order"]
            and identity["arrival_jitter"] == tempo_identity["arrival_jitter"]
            and identity["p_only_prompt_namespace"]
            == tempo_identity["p_only_prompt_namespace"],
            f"{policy} held-out workload identity differs",
        )
        job_ids.add(str(result.get("slurm_job_id")))
        baselines[policy] = {
            "result": str(path),
            "result_sha256": _sha256(path),
            "execution": identity,
            "non_tempo_admission_evidence": admission_evidence,
            "metrics": {
                regime: _metrics(analysis, regime) for regime in REGIMES
            },
        }

    _require(len(job_ids) == 1 and "None" not in job_ids,
             "C10 comparisons must use one live four-node allocation")
    tempo_metrics = {
        regime: _metrics(tempo_analysis, regime) for regime in REGIMES
    }
    comparisons: dict[str, object] = {}
    for regime in REGIMES:
        comparisons[regime] = {}
        for policy, value in baselines.items():
            reference = value["metrics"][regime]
            p50_reduction = _optional_reduction(
                reference["e2e_ms"]["p50"],
                tempo_metrics[regime]["e2e_ms"]["p50"],
            )
            p95_reduction = _optional_reduction(
                reference["e2e_ms"]["p95"],
                tempo_metrics[regime]["e2e_ms"]["p95"],
            )
            p99_reduction = _optional_reduction(
                reference["e2e_ms"]["p99"],
                tempo_metrics[regime]["e2e_ms"]["p99"],
            )
            completion_delta = (
                int(tempo_metrics[regime]["completed"])
                - int(reference["completed"])
            )
            slo_good_delta = (
                int(tempo_metrics[regime]["slo_good"])
                - int(reference["slo_good"])
            )
            comparisons[regime][policy] = {
                "tempo_e2e_p50_reduction_fraction": p50_reduction,
                "tempo_e2e_p95_reduction_fraction": p95_reduction,
                "tempo_e2e_p99_reduction_fraction": p99_reduction,
                "baseline_completed_tail_defined": p99_reduction is not None,
                "tempo_completion_delta": completion_delta,
                "tempo_slo_good_delta": slo_good_delta,
                "tempo_strict_service_dominance": (
                    slo_good_delta > 0
                    or (
                        slo_good_delta == 0
                        and completion_delta >= 0
                        and p99_reduction is not None
                        and p99_reduction > 0.0
                    )
                ),
            }

    stressed = ("miss_hot", "remote_favorable")
    tempo_dominates_every_sota_service = all(
        comparisons[regime][policy]["tempo_strict_service_dominance"] is True
        for regime in stressed for policy in baselines
    )
    tempo_beats_every_defined_sota_tail = all(
        comparison["tempo_e2e_p99_reduction_fraction"] is None
        or comparison["tempo_e2e_p99_reduction_fraction"] > 0.0
        for regime in stressed
        for comparison in comparisons[regime].values()
    )
    tempo_retains_slo_good = all(
        comparisons[regime][policy]["tempo_slo_good_delta"] >= 0
        for regime in stressed for policy in baselines
    )
    normal_reductions = [
        comparison["tempo_e2e_p50_reduction_fraction"]
        for comparison in comparisons["normal"].values()
        if comparison["tempo_e2e_p50_reduction_fraction"] is not None
    ]
    _require(normal_reductions,
             "no paper baseline completed the normal control")
    normal_p50_regression = max(-float(value) for value in normal_reductions)

    parent = _load(parent_analysis_path)
    _require(parent.get("independent_validation_positive") is True,
             "parent independent C9 gate is not positive")
    motivational = parent.get("base_campaign", {}).get("effects")
    _require(isinstance(motivational, dict),
             "parent motivational fixed/predictor effects are missing")

    return {
        "schema": SCHEMA,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "slurm_job_ids": sorted(job_ids),
        "same_allocation_gate": len(job_ids) == 1,
        "same_heldout_workload_identity_gate": True,
        "paper_baseline_non_tempo_admission_gate": all(
            value["non_tempo_admission_evidence"]["gate"]
            for value in baselines.values()
        ),
        "actual_system": {
            "nodes": 4,
            "gpus": 16,
            "model": "Qwen2.5-7B-Instruct",
            "tensor_parallel_per_engine": 4,
            "carrier": "vLLM-0.26.0+cu129/LMCacheConnectorV1/NIXL-CXI/Slingshot11",
        },
        "tempo": {
            "result": str(tempo_result_path),
            "result_sha256": _sha256(tempo_result_path),
            "execution": tempo_identity,
            "metrics": tempo_metrics,
        },
        "paper_baselines": baselines,
        "comparisons": comparisons,
        "motivational_independent_c9_effects": motivational,
        "gates": {
            "tempo_strictly_dominates_every_sota_stressed_service": (
                tempo_dominates_every_sota_service),
            "tempo_beats_every_defined_sota_stressed_p99": (
                tempo_beats_every_defined_sota_tail),
            "tempo_retains_every_sota_stressed_slo_good": tempo_retains_slo_good,
            "normal_p50_regression_fraction": normal_p50_regression,
            "normal_p50_regression_within_5pct": normal_p50_regression <= 0.05,
        },
        "actual_sota_extension_positive": (
            tempo_dominates_every_sota_service
            and tempo_beats_every_defined_sota_tail
            and tempo_retains_slo_good
            and normal_p50_regression <= 0.05
        ),
        "claim_boundary": {
            "post_hoc_extension": True,
            "parent_tempo_independent_validation_positive": True,
            "sota_extension_is_actual_vllm_lmcache": True,
            "sota_extension_independent_validation_claim_allowed": False,
            "reason": (
                "baseline policies were frozen after the parent allocation "
                "started; rerun the unchanged C10 contract on a fresh "
                "allocation before an independent SOTA claim"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tempo-result", type=Path, required=True)
    parser.add_argument("--parent-analysis", type=Path, required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_paths = {}
    for raw in args.baseline:
        name, separator, path = raw.partition("=")
        _require(bool(separator) and name and path,
                 "--baseline requires POLICY=PATH")
        _require(name not in baseline_paths, "duplicate baseline policy")
        baseline_paths[name] = Path(path).resolve()
    value = analyze(
        manifest_path=args.manifest.resolve(),
        tempo_result_path=args.tempo_result.resolve(),
        parent_analysis_path=args.parent_analysis.resolve(),
        baseline_paths=baseline_paths,
    )
    args.output.resolve().write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "actual_sota_extension_positive": value[
            "actual_sota_extension_positive"],
        "gates": value["gates"],
        "comparisons": value["comparisons"],
        "claim_boundary": value["claim_boundary"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
