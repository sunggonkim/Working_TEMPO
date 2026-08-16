#!/usr/bin/env python3
"""Build a non-submitting G2 two-node fabric-attribution plan.

The plan is deliberately gated on a completed G1 promotion.  It does not
submit work or synthesize fabric measurements; it only freezes the modes and
evidence split that an approved two-node run must execute.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

try:
    from tempo.domain_evidence import CounterSupport, PathStatus
    from tempo.tier_attribution import TierEvaluation
    from tempo.resource_domain import ResourceDomain
    from tempo.observation_window import observation_window_contract
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tempo.domain_evidence import CounterSupport, PathStatus
    from tempo.tier_attribution import TierEvaluation
    from tempo.resource_domain import ResourceDomain
    from tempo.observation_window import observation_window_contract


@dataclass(frozen=True)
class FabricRun:
    mode: str
    policy: str
    collective_scope: str
    auxiliary_domains: tuple[str, ...]
    requires_restore: bool


G2_MODES = (
    "fg_only",
    "open_combined",
    "causal_domain_static_cap",
    "unrelated_domain_placebo",
    "combined",
)


_G2_MANIFEST_KEYS = {
    "schema_version",
    "world_size",
    "nodes",
    "state_bytes_per_rank",
    "deadline_ns",
    "checkpoint_steps",
    "runs",
    "collective_slices",
    "fabric_splits",
    "promoted_domain",
    "placebo_domain",
    "requires_g1_promotion",
    "g1_evidence_ready",
    "g1_eligible_domains",
    "evidence_state",
    "evidence_contract",
    "observation_window_contract",
    "slurm_submitted",
}


def validate_g2_manifest(manifest: dict[str, object]) -> None:
    """Validate the promotion-gated, non-submitting G2 plan exactly."""

    if type(manifest) is not dict or set(manifest) != _G2_MANIFEST_KEYS:
        raise ValueError("G2 manifest keys are not exact")
    if manifest["schema_version"] != "tempo-rd-fabric-attribution-runner-1":
        raise ValueError("unsupported G2 manifest schema")
    if manifest["world_size"] != 8 or manifest["nodes"] != 2:
        raise ValueError("G2 fabric plan must be exactly two nodes and eight ranks")
    for key in ("state_bytes_per_rank", "deadline_ns"):
        value = manifest[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive int")
    steps = manifest["checkpoint_steps"]
    if type(steps) is not list or not steps or any(type(step) is not int for step in steps):
        raise ValueError("checkpoint_steps must be a non-empty integer list")
    if steps != sorted(set(steps)):
        raise ValueError("checkpoint_steps must be sorted and unique")
    if manifest["collective_slices"] != ["intra_node", "inter_node"]:
        raise ValueError("G2 collective slices are not exact")
    if manifest["fabric_splits"] != ["gdr_gpu_originated", "host_originated", "pfs_endpoint"]:
        raise ValueError("G2 fabric splits are not exact")
    try:
        promoted = ResourceDomain(manifest["promoted_domain"])
        placebo = ResourceDomain(manifest["placebo_domain"])
    except (TypeError, ValueError) as exc:
        raise ValueError("G2 promoted/placebo domain is invalid") from exc
    if promoted is placebo:
        raise ValueError("G2 promoted and placebo domains must differ")
    if manifest["requires_g1_promotion"] is not True:
        raise ValueError("G2 must require G1 promotion")
    if manifest["g1_evidence_ready"] is not True:
        raise ValueError("G2 requires observed G1 evidence")
    eligible = manifest["g1_eligible_domains"]
    if type(eligible) is not list or any(type(item) is not str for item in eligible):
        raise ValueError("G2 eligible domains must be a string list")
    if eligible != sorted(set(eligible)):
        raise ValueError("G2 eligible domains must be sorted and unique")
    try:
        eligible_domains = {ResourceDomain(item) for item in eligible}
    except ValueError as exc:
        raise ValueError("G2 eligible domains contain an unknown resource domain") from exc
    if promoted not in eligible_domains:
        raise ValueError("G2 eligible domains must include the promoted domain")
    if manifest["evidence_state"] != "design_only":
        raise ValueError("G2 manifest must remain design_only")
    if manifest["slurm_submitted"] is not False:
        raise ValueError("G2 runner must never submit Slurm work")
    if manifest["observation_window_contract"] != observation_window_contract():
        raise ValueError("G2 observation-window contract is not exact")
    contract = manifest["evidence_contract"]
    expected_contract = {
        "counter_support_values": sorted(item.value for item in CounterSupport),
        "path_status_values": sorted(item.value for item in PathStatus),
        "causal_requires": [
            "interventional",
            "observed_path",
            "supported_counters",
            "tail_delta_above_uncertainty",
        ],
    }
    if (
        type(contract) is not dict
        or set(contract) != set(expected_contract)
        or any(
            key in {"counter_support_values", "path_status_values", "causal_requires"}
            and type(contract[key]) is not list
            for key in expected_contract
        )
        or contract != expected_contract
    ):
        raise ValueError("G2 evidence_contract is not exact")


def build_g2_matrix(promoted_domain: ResourceDomain, placebo_domain: ResourceDomain) -> tuple[FabricRun, ...]:
    if not isinstance(promoted_domain, ResourceDomain) or not isinstance(placebo_domain, ResourceDomain):
        raise TypeError("promoted_domain and placebo_domain must be ResourceDomain values")
    if promoted_domain is placebo_domain:
        raise ValueError("promoted and placebo domains must differ")
    full_path = tuple(domain.value for domain in (
        ResourceDomain.GPU_LOCAL,
        ResourceDomain.PCIE_HOST,
        ResourceDomain.HOST_NUMA,
        ResourceDomain.NIC_FABRIC,
        ResourceDomain.SLINGSHOT_FABRIC,
        ResourceDomain.PERSISTENT_ENDPOINT,
    ))
    return (
        FabricRun("fg_only", "none", "both", (), False),
        FabricRun("open_combined", "datastates", "both", full_path, True),
        FabricRun("causal_domain_static_cap", "datastates", "both", (promoted_domain.value,), True),
        FabricRun("unrelated_domain_placebo", "datastates", "both", (placebo_domain.value,), True),
        FabricRun("combined", "datastates", "both", full_path, True),
    )


def build_g2_manifest(
    *,
    promoted_domain: ResourceDomain,
    placebo_domain: ResourceDomain,
    g1_evaluation: TierEvaluation,
    state_bytes_per_rank: int,
    deadline_ns: int,
    checkpoint_steps: list[int],
) -> dict[str, object]:
    if not isinstance(g1_evaluation, TierEvaluation):
        raise TypeError("G2 requires the in-process TierEvaluation from G1")
    if not g1_evaluation.promote_static_policy:
        raise ValueError("G2 requires a successful G1 matched-open promotion")
    if type(state_bytes_per_rank) is not int or state_bytes_per_rank <= 0:
        raise ValueError("state_bytes_per_rank must be a positive int")
    if type(deadline_ns) is not int or deadline_ns <= 0:
        raise ValueError("deadline_ns must be a positive int")
    if not checkpoint_steps or checkpoint_steps != sorted(set(checkpoint_steps)):
        raise ValueError("checkpoint_steps must be sorted and unique")
    runs = build_g2_matrix(promoted_domain, placebo_domain)
    manifest = {
        "schema_version": "tempo-rd-fabric-attribution-runner-1",
        "world_size": 8,
        "nodes": 2,
        "state_bytes_per_rank": state_bytes_per_rank,
        "deadline_ns": deadline_ns,
        "checkpoint_steps": checkpoint_steps,
        "runs": [asdict(run) for run in runs],
        "collective_slices": ["intra_node", "inter_node"],
        "fabric_splits": ["gdr_gpu_originated", "host_originated", "pfs_endpoint"],
        "promoted_domain": promoted_domain.value,
        "placebo_domain": placebo_domain.value,
        "requires_g1_promotion": True,
        "g1_evidence_ready": g1_evaluation.evidence_ready,
        "g1_eligible_domains": sorted(
            domain.value for domain in g1_evaluation.promotion.eligible_domains
        ),
        "evidence_state": "design_only",
        "evidence_contract": {
            "counter_support_values": sorted(item.value for item in CounterSupport),
            "path_status_values": sorted(item.value for item in PathStatus),
            "causal_requires": [
                "interventional",
                "observed_path",
                "supported_counters",
                "tail_delta_above_uncertainty",
            ],
        },
        "observation_window_contract": observation_window_contract(),
        "slurm_submitted": False,
    }
    validate_g2_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoted-domain", required=True, choices=[domain.value for domain in ResourceDomain])
    parser.add_argument("--placebo-domain", required=True, choices=[domain.value for domain in ResourceDomain])
    parser.add_argument(
        "--g1-promotion-eligible", action="store_true",
        help="deprecated; G2 requires an in-process TierEvaluation and cannot be built from a boolean",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-bytes-per-rank", type=int, default=402_705_672)
    parser.add_argument("--deadline-ns", type=int, default=1_000_000_000)
    parser.add_argument("--checkpoint-steps", default="16,52")
    args = parser.parse_args()
    parser.error("G2 manifest construction requires an in-process TierEvaluation; use build_g2_manifest()")


if __name__ == "__main__":
    main()
