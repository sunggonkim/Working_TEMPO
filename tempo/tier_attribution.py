"""Fail-closed contract for the TEMPO-RD tier-attribution ladder.

The contract is intentionally independent of the checkpoint backend.  It lets
the G1/G2 runner validate that foreground and auxiliary-flow modes are paired
with identical state geometry before a GPU result is treated as causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from tempo.domain_evidence import CounterSupport, PathStatus
from tempo.domain_evidence import DomainEvidence
from tempo.causal_gate import CausalModeRecord, CausalPromotion, evaluate_causal_matrix
from tempo.resource_domain import ResourceDomain


class AttributionMode(str, Enum):
    FOREGROUND_ONLY = "fg_only"
    OPEN_COMBINED = "open_combined"
    D2H_ONLY = "d2h_only"
    PERSIST_ONLY = "persist_only"
    COMBINED = "combined"
    P2P_ONLY = "p2p_only"
    HOST_PRESSURE = "host_pressure"


@dataclass(frozen=True)
class ModeSpec:
    mode: AttributionMode
    auxiliary_domains: tuple[ResourceDomain, ...]
    requires_checkpoint_endpoint: bool
    requires_gpu_transfer: bool


@dataclass(frozen=True)
class TierEvaluation:
    """Joined path-evidence and foreground-metric promotion result."""

    promotion: CausalPromotion
    evidence_ready: bool
    reasons: tuple[str, ...]

    @property
    def promote_static_policy(self) -> bool:
        return self.evidence_ready and self.promotion.promote_static_policy


MODE_SPECS: Mapping[AttributionMode, ModeSpec] = {
    AttributionMode.FOREGROUND_ONLY: ModeSpec(
        AttributionMode.FOREGROUND_ONLY, (), False, False
    ),
    AttributionMode.OPEN_COMBINED: ModeSpec(
        AttributionMode.OPEN_COMBINED,
        (ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA, ResourceDomain.NIC_FABRIC,
         ResourceDomain.SLINGSHOT_FABRIC, ResourceDomain.PERSISTENT_ENDPOINT),
        True,
        True,
    ),
    AttributionMode.D2H_ONLY: ModeSpec(
        AttributionMode.D2H_ONLY,
        (ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA),
        False,
        True,
    ),
    AttributionMode.PERSIST_ONLY: ModeSpec(
        AttributionMode.PERSIST_ONLY,
        (ResourceDomain.NIC_FABRIC, ResourceDomain.SLINGSHOT_FABRIC, ResourceDomain.PERSISTENT_ENDPOINT),
        True,
        False,
    ),
    AttributionMode.COMBINED: ModeSpec(
        AttributionMode.COMBINED,
        (ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA, ResourceDomain.NIC_FABRIC,
         ResourceDomain.SLINGSHOT_FABRIC, ResourceDomain.PERSISTENT_ENDPOINT),
        True,
        True,
    ),
    AttributionMode.P2P_ONLY: ModeSpec(
        AttributionMode.P2P_ONLY, (ResourceDomain.GPU_LOCAL, ResourceDomain.NVLINK_P2P), False, True
    ),
    AttributionMode.HOST_PRESSURE: ModeSpec(
        AttributionMode.HOST_PRESSURE, (ResourceDomain.HOST_NUMA,), False, False
    ),
}


REQUIRED_G1_MODES = frozenset(
    {
        AttributionMode.FOREGROUND_ONLY.value,
        AttributionMode.OPEN_COMBINED.value,
        AttributionMode.D2H_ONLY.value,
        AttributionMode.PERSIST_ONLY.value,
        AttributionMode.COMBINED.value,
        AttributionMode.HOST_PRESSURE.value,
    }
)


def mode_spec(mode: str | AttributionMode) -> ModeSpec:
    try:
        key = mode if isinstance(mode, AttributionMode) else AttributionMode(mode)
    except ValueError as exc:
        raise ValueError(f"unsupported attribution mode: {mode!r}") from exc
    return MODE_SPECS[key]


def validate_attribution_manifest(manifest: Mapping[str, object]) -> None:
    """Validate common geometry and mode completeness before GPU evidence use."""

    required = {
        "schema_version",
        "world_size",
        "nodes",
        "state_bytes_per_rank",
        "logical_file_extent_bytes",
        "deadline_ns",
        "checkpoint_steps",
        "modes",
        "required_domains",
        "evidence_contract",
    }
    if set(manifest) != required:
        raise ValueError("attribution manifest keys are not exact")
    if manifest["schema_version"] != "tempo-rd-tier-attribution-1":
        raise ValueError("unsupported attribution manifest schema")
    for key in ("world_size", "nodes", "state_bytes_per_rank", "logical_file_extent_bytes", "deadline_ns"):
        value = manifest[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive int")
    steps = manifest["checkpoint_steps"]
    if type(steps) is not list or not steps or any(type(step) is not int for step in steps):
        raise ValueError("checkpoint_steps must be a non-empty integer list")
    if steps != sorted(set(steps)):
        raise ValueError("checkpoint_steps must be sorted and unique")
    modes = manifest["modes"]
    if type(modes) is not list or any(type(mode) is not str for mode in modes):
        raise ValueError("modes must be a string list")
    if set(modes) != REQUIRED_G1_MODES or len(modes) != len(REQUIRED_G1_MODES):
        raise ValueError("G1 must contain each required attribution mode exactly once")
    for mode in modes:
        mode_spec(mode)
    required_domains = manifest["required_domains"]
    if type(required_domains) is not list or any(type(domain) is not str for domain in required_domains):
        raise ValueError("required_domains must be a string list")
    expected_domains = sorted(
        domain.value for domain in required_domains_for_modes(modes)
    )
    if required_domains != expected_domains:
        raise ValueError("required_domains must exactly cover the selected modes")
    evidence_contract = manifest["evidence_contract"]
    if type(evidence_contract) is not dict or set(evidence_contract) != {
        "counter_support_values", "path_status_values", "causal_requires"
    }:
        raise ValueError("evidence_contract keys are not exact")
    if evidence_contract["counter_support_values"] != sorted(item.value for item in CounterSupport):
        raise ValueError("counter support enum contract is not exact")
    if evidence_contract["path_status_values"] != sorted(item.value for item in PathStatus):
        raise ValueError("path status enum contract is not exact")
    if evidence_contract["causal_requires"] != [
        "interventional",
        "observed_path",
        "supported_counters",
        "tail_delta_above_uncertainty",
    ]:
        raise ValueError("causal evidence requirements are not exact")


def required_domains_for_modes(modes: Iterable[str]) -> frozenset[ResourceDomain]:
    domains: set[ResourceDomain] = set()
    for mode in modes:
        domains.update(mode_spec(mode).auxiliary_domains)
    return frozenset(domains)


def validate_mode_evidence(
    mode: str | AttributionMode,
    records: Iterable[DomainEvidence],
    *,
    require_observed: bool = False,
) -> Mapping[ResourceDomain, tuple[DomainEvidence, ...]]:
    """Validate that evidence records exactly cover a mode's declared path.

    A mode may have multiple observations per domain, but every auxiliary
    domain in its ``ModeSpec`` must appear at least once and no undeclared
    domain may be smuggled into the causal matrix.  ``require_observed`` is
    used only at live-result promotion time; design manifests may retain
    ``declared``/``not_collected`` records without promoting them.
    """

    spec = mode_spec(mode)
    expected = set(spec.auxiliary_domains)
    grouped: dict[ResourceDomain, list[DomainEvidence]] = {}
    for record in records:
        if not isinstance(record, DomainEvidence):
            raise TypeError("records must contain DomainEvidence values")
        if record.mode != spec.mode.value:
            raise ValueError("evidence mode does not match attribution mode")
        if record.domain not in expected:
            raise ValueError(f"evidence domain {record.domain.value} is not declared for {spec.mode.value}")
        grouped.setdefault(record.domain, []).append(record)
    missing = expected - set(grouped)
    if missing:
        raise ValueError(
            "missing evidence domains: " + ",".join(sorted(domain.value for domain in missing))
        )
    if require_observed:
        for domain, values in grouped.items():
            if any(value.path_status is not PathStatus.OBSERVED for value in values):
                raise ValueError(f"{domain.value} evidence path is not observed")
            if any(value.counter_support is not CounterSupport.SUPPORTED for value in values):
                raise ValueError(f"{domain.value} evidence counters are not supported")
    return {domain: tuple(values) for domain, values in grouped.items()}


def evaluate_tier_attribution(
    mode_evidence: Mapping[str, Iterable[DomainEvidence]],
    metrics: Iterable[CausalModeRecord],
    *,
    require_observed: bool = True,
) -> TierEvaluation:
    """Join exact path coverage with the matched-open causal metric gate."""

    reasons: list[str] = []
    metric_records = tuple(metrics)
    coverage: dict[str, Mapping[ResourceDomain, tuple[DomainEvidence, ...]]] = {}
    for mode, records in mode_evidence.items():
        try:
            coverage[mode] = validate_mode_evidence(
                mode, records, require_observed=require_observed
            )
        except (TypeError, ValueError) as exc:
            reasons.append(f"{mode}: {exc}")
    if "fg_only" not in mode_evidence:
        reasons.append("missing fg_only evidence entry")
    if "open_combined" not in mode_evidence:
        reasons.append("missing open_combined evidence entry")
    for metric in metric_records:
        if metric.mode not in mode_evidence:
            reasons.append(f"{metric.mode}: metric has no evidence entry")
        elif metric.domain is not None and metric.domain not in coverage.get(metric.mode, {}):
            reasons.append(f"{metric.mode}: metric domain lacks evidence coverage")
    promotion = evaluate_causal_matrix(metric_records)
    if not coverage:
        reasons.append("no valid mode evidence")
    if reasons:
        return TierEvaluation(promotion, False, tuple(reasons))
    return TierEvaluation(promotion, True, tuple(promotion.reasons))
