#!/usr/bin/env python3
"""Build current TEMPO-GO evidence from immutable and post-hoc receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "paper/tempo_go/current_evidence_manifest.json"
FULL_ARM = "full_c7_managed_background"
REGIMES = ("normal", "miss_hot", "remote_favorable")

CAMPAIGNS: dict[str, dict[str, Any]] = {
    "candidate_m": {
        "label": "Candidate M",
        "mechanism": "pressure-triggered packed-pair spill",
        "allocation": "57732862",
        "contract": (
            "results/tempo_go_c9_candidate_m_pressure_spill_v1/"
            "tempo_go_c9_candidate_m_pressure_spill_population_contract.json"
        ),
        "analysis": "results/tempo_go_c9_causal_burst_job_57732862/analysis.json",
        "posthoc": (
            "results/tempo_go_c9_causal_burst_job_57732862/"
            "analysis_failclosed_business_v3.json"
        ),
        "terminal_receipt": (
            "results/tempo_go_c9_causal_burst_job_57732862/completed_attempt.json"
        ),
    },
    "candidate_n": {
        "label": "Candidate N",
        "mechanism": "Candidate M plus pair-scoped local-receiver scalar price",
        "allocation": "57732862",
        "contract": (
            "results/tempo_go_c9_candidate_n_global_frontier_v2/"
            "tempo_go_c9_candidate_n_global_frontier_population_contract.json"
        ),
        "analysis": (
            "results/tempo_go_c9_global_frontier_job_57732862/analysis.json"
        ),
        "posthoc": (
            "results/tempo_go_c9_global_frontier_job_57732862/"
            "analysis_failclosed_business_v3.json"
        ),
        "terminal_receipt": None,
        "analysis_boundary": (
            "all seven native arms reached terminal artifacts; the initial "
            "analyzer failed on an empty completed population, so no top-level "
            "completed_attempt receipt exists"
        ),
    },
    "candidate_o": {
        "label": "Candidate O",
        "mechanism": (
            "Candidate M with telemetry-derived failure quarantine narrowed "
            "from pair scope to route scope"
        ),
        "allocation": "57736076",
        "contract": (
            "results/tempo_go_c9_candidate_o_route_liveness_v1/"
            "tempo_go_c9_candidate_o_route_liveness_population_contract.json"
        ),
        "analysis": (
            "results/tempo_go_c9_route_liveness_job_57736076_"
            "r3_canonical_outer/analysis.json"
        ),
        "posthoc": (
            "results/tempo_go_c9_route_liveness_job_57736076_"
            "r3_canonical_outer/analysis_failclosed_business_v2.json"
        ),
        "terminal_receipt": (
            "results/tempo_go_c9_route_liveness_job_57736076_"
            "r3_canonical_outer/completed_attempt.json"
        ),
        "profile": (
            "results/tempo_go_c9_candidate_o_route_liveness_v1/"
            "real_tempo_go_c9_candidate_o_route_liveness_profile_v1.json"
        ),
        "diagnosis": (
            "results/tempo_go_c9_route_liveness_job_57736076_"
            "r3_canonical_outer/candidate_o_diagnosis.json"
        ),
        "posthoc_receipt": (
            "results/tempo_go_c9_route_liveness_job_57736076_"
            "r3_canonical_outer/posthoc_business_reanalysis_receipt.json"
        ),
    },
}


def load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {relative}")
    return value


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def artifact(relative: str | None) -> dict[str, str] | None:
    if relative is None:
        return None
    return {"path": relative, "sha256": sha256(relative)}


def campaign(spec: dict[str, Any]) -> dict[str, Any]:
    native = load(spec["analysis"])
    posthoc = load(spec["posthoc"])
    aggregate = native["aggregates"][FULL_ARM]
    telemetry = posthoc["telemetry"]
    value: dict[str, Any] = {
        "label": spec["label"],
        "allocation": spec["allocation"],
        "mechanism": spec["mechanism"],
        "population_contract": artifact(spec["contract"]),
        "native_analysis": artifact(spec["analysis"]),
        "posthoc_business_analysis": artifact(spec["posthoc"]),
        "terminal_receipt": artifact(spec["terminal_receipt"]),
        "regimes": {
            regime: {
                "offered": int(aggregate[regime]["offered"]),
                "completed": int(aggregate[regime]["completed"]),
                "slo_good": int(aggregate[regime]["slo_good"]),
                "e2e_p50_ms": aggregate[regime]["mean_e2e_p50_ms"],
                "e2e_p99_ms": aggregate[regime]["mean_e2e_p99_ms"],
            }
            for regime in REGIMES
        },
        "business": posthoc["business"]["full_tempo"],
        "observer": {
            "supported": int(telemetry["full_cross_layer_supported_decisions"]),
            "total": int(telemetry["full_victim_global_decisions"]),
            "fraction": float(telemetry["full_cross_layer_supported_fraction"]),
        },
        "receipt_integrity": bool(posthoc["gates"]["correctness"]),
        "gates": posthoc["gates"],
        "causal_discovery_positive": bool(posthoc["causal_discovery_positive"]),
        "performance_claim_allowed": False,
        "analysis_boundary": spec.get(
            "analysis_boundary",
            "native analysis and completion receipt are immutable; the post-hoc "
            "analysis changes only business-terminal accounting",
        ),
    }
    for key in ("profile", "diagnosis", "posthoc_receipt"):
        if key in spec:
            value[key] = artifact(spec[key])
    return value


def noncausal_delta(new: dict[str, Any], old: dict[str, Any]) -> dict[str, Any]:
    new_business = new["business"]
    old_business = old["business"]
    return {
        "foreground_completed_delta": (
            new_business["foreground"]["completed"]
            - old_business["foreground"]["completed"]
        ),
        "background_completed_delta": (
            new_business["background"]["completed"]
            - old_business["background"]["completed"]
        ),
        "background_global_reject_delta": (
            new_business["background"]["global_rejects"]
            - old_business["background"]["global_rejects"]
        ),
        "background_failure_delta": (
            new_business["background"]["failures"]
            - old_business["background"]["failures"]
        ),
        "observer_supported_delta": (
            new["observer"]["supported"] - old["observer"]["supported"]
        ),
        "causal_comparison_allowed": False,
        "reason": (
            "M and O used separate allocations and one sample per arm; the "
            "candidate-specific changed mechanism did not activate in O"
        ),
    }


def build_payload() -> dict[str, Any]:
    campaigns = {name: campaign(spec) for name, spec in CAMPAIGNS.items()}
    diagnosis = load(CAMPAIGNS["candidate_o"]["diagnosis"])
    mechanism = diagnosis["candidate_specific_mechanism"]
    totals = mechanism["route_failure_evidence_total"]
    original_o = load(CAMPAIGNS["candidate_o"]["analysis"])

    return {
        "schema": "tempo-go-current-evidence-manifest-v2",
        "as_of_utc": "2026-08-30",
        "claim_state": {
            "actual_cross_layer_contention_reproduced": True,
            "candidate_o_changed_mechanism_activated": bool(
                mechanism["activated"]
            ),
            "current_performance_claim_allowed": False,
            "independent_sota_claim_allowed": False,
            "native_1_2_4_node_scale_complete": False,
            "launch_boundary_closed": True,
            "joint_global_orchestrator_utility_gate_closed": False,
        },
        "platform": {
            "system": "NERSC Perlmutter",
            "latest_allocation": "57736076",
            "nodes": 4,
            "gpus": 16,
            "qos": "gpu_interactive",
            "serving": "actual vLLM P/D",
            "kv_transport": "official LMCache/NIXL-UCX",
            "fabric_load": (
                "two physical-pair NCCL over Slingshot/Cassini co-jobs"
            ),
            "claim_boundary": (
                "native receiver/data-plane overload and pair asymmetry; not "
                "physical switch wire-rate saturation"
            ),
        },
        "population": {
            "foreground_offered": 210,
            "background_offered": 2748,
            "regimes": {
                "normal": 60, "miss_hot": 120, "remote_favorable": 30,
            },
            "baseline_arms": [
                name
                for name in original_o["aggregates"]
                if name != FULL_ARM
            ],
            "paired_internal_repeats": 0,
            "final_validation_minimum_repeats": 3,
        },
        "campaigns": campaigns,
        "candidate_o_changed_mechanism": {
            "changed_input": mechanism["changed_input"],
            "activated": bool(mechanism["activated"]),
            "all_global_decisions": int(totals["all_global_decisions"]),
            "nonzero_route_failure_counter_decisions": int(
                totals["all_decisions_with_nonzero_route_failure_counter"]
            ),
            "route_failure_quarantine_rejections": int(
                totals["all_route_failure_quarantine_rejections"]
            ),
            "causal_mechanism_positive": False,
        },
        "candidate_o_vs_m_cross_allocation_context": noncausal_delta(
            campaigns["candidate_o"], campaigns["candidate_m"]
        ),
        "preregistered_next_diagnostic": {
            "id": "Candidate P bounded-observer lifecycle",
            "contract": artifact(
                "results/tempo_go_c9_candidate_p_bounded_observer_v1/"
                "tempo_go_c9_candidate_p_bounded_observer_contract.json"
            ),
            "policy_delta_from_candidate_o": False,
            "native_result_exists": False,
            "performance_claim_allowed": False,
            "purpose": (
                "isolate observer-process lifetime from the 2 GiB timeout-shaped "
                "co-load; this diagnostic cannot replace the realistic overload "
                "campaign or the durable state-plane implementation"
            ),
        },
        "candidate_o_execution_boundary": {
            "failed_gpu_reserving_outer": {
                "result_root": "results/tempo_go_c9_route_liveness_job_57736076",
                "receipt": artifact(
                    "results/tempo_go_c9_route_liveness_job_57736076/"
                    "failed_attempt.json"
                ),
                "status": "failed_before_performance_result",
                "failure": (
                    "outer reserved all GPUs; child co-job failed with Requested "
                    "node configuration is not available"
                ),
            },
            "failed_inherited_gres_outer_admission": {
                "result_root": None,
                "status": "failed_before_step_and_before_performance_result",
                "failure": (
                    "CPU-only outer without explicit GPU-zero/GRES-none failed "
                    "with Insufficient GRES"
                ),
                "fix": (
                    "unset inherited GPU TRES variables and request --gpus=0 "
                    "--gres=none explicitly"
                ),
            },
            "failed_cpu_outer_with_wait": {
                "result_root": (
                    "results/tempo_go_c9_route_liveness_job_57736076_"
                    "r2_outer_cpu_only"
                ),
                "receipt": artifact(
                    "results/tempo_go_c9_route_liveness_job_57736076_"
                    "r2_outer_cpu_only/failed_attempt.json"
                ),
                "status": "execution_only_not_a_campaign_result",
                "failure": (
                    "outer --wait=600 terminated rank 0 after helper ranks exited"
                ),
            },
            "canonical_outer": {
                "wrapper": artifact(
                    "eval/sota_4node/"
                    "attach_tempo_go_c9_candidate_o_route_liveness_to_allocation.sh"
                ),
                "gpus": 0,
                "gres": "none",
                "network": "no_vni",
                "wait_option_present": False,
                "kill_on_bad_exit_present": False,
                "status": "complete",
            },
        },
        "analysis_implementation": {
            "campaign_analyzer": artifact(
                "eval/sota_4node/"
                "analyze_tempo_go_c9_causal_burst_discovery.py"
            ),
            "candidate_o_diagnosis_analyzer": artifact(
                "eval/sota_4node/"
                "analyze_tempo_go_c9_candidate_o_route_liveness.py"
            ),
            "terminal_semantics": (
                "valid is receipt integrity; completion requires HTTP 200, "
                "done, no transport error, and exact output tokens"
            ),
        },
        "lmcache_integration": {
            "upstream_commit": "227d13f5c9fdb52ddb933641d34331f678de03a0",
            "patch": artifact("eval/sota_4node/lmcache_tempo_current.patch"),
            "clean_base_apply_check": "passed",
        },
        "verification": {
            "focused_tests": 104,
            "focused_test_result": "passed",
            "python_compile": "passed",
            "posthoc_reproduction": "byte-identical",
        },
        "root_cause": diagnosis["root_cause"],
        "next_hard_gates": diagnosis["next_gate"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_payload(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
