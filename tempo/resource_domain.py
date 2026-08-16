"""Resource-domain and multi-hop state-flow contracts for TEMPO-RD.

This module is deliberately policy-free.  It records which resource domains a
foreground operation or auxiliary flow stage actually claims, and aggregates
observations without inferring a route from topology alone.  Admission policy
belongs to a later stage after causal evidence exists.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


class ResourceDomain(str, Enum):
    GPU_LOCAL = "gpu_local"
    NVLINK_P2P = "nvlink_p2p"
    PCIE_HOST = "pcie_host"
    HOST_NUMA = "host_numa"
    NIC_FABRIC = "nic_fabric"  # NIC/CXI injection and queueing domain
    SLINGSHOT_FABRIC = "slingshot_fabric"  # transport/routing domain beyond the NIC
    PERSISTENT_ENDPOINT = "persistent_endpoint"
    COMPLETION_ENDPOINT = "completion_endpoint"


class EvidenceLevel(str, Enum):
    UNSUPPORTED = "unsupported"
    OBSERVATIONAL = "observational"
    INTERVENTIONAL = "interventional"


@dataclass(frozen=True)
class DomainContract:
    """The minimum path/counter labels needed to claim a domain was measured.

    These labels are evidence requirements, not topology inference.  A live
    record may still report ``not_supported``/``not_traversed``; it simply
    cannot be promoted until the declared labels are backed by observations.
    """

    domain: ResourceDomain
    path_evidence: str
    counter_family: str

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ResourceDomain):
            raise TypeError("domain must be a ResourceDomain")
        if not self.path_evidence or not self.counter_family:
            raise ValueError("domain contract labels must be non-empty")


DOMAIN_CONTRACTS: Mapping[ResourceDomain, DomainContract] = {
    ResourceDomain.GPU_LOCAL: DomainContract(
        ResourceDomain.GPU_LOCAL, "gpu_hbm_copy_engine", "gpu_copy_engine_bytes"
    ),
    ResourceDomain.NVLINK_P2P: DomainContract(
        ResourceDomain.NVLINK_P2P, "nvlink_p2p_path", "nvlink_tx_rx_bytes"
    ),
    ResourceDomain.PCIE_HOST: DomainContract(
        ResourceDomain.PCIE_HOST, "gpu_pcie_root_complex", "pcie_tx_rx_bytes"
    ),
    ResourceDomain.HOST_NUMA: DomainContract(
        ResourceDomain.HOST_NUMA, "host_numa_buffer", "numa_memory_bytes"
    ),
    ResourceDomain.NIC_FABRIC: DomainContract(
        ResourceDomain.NIC_FABRIC, "cxi_nic_injection", "cxi_tx_rx_bytes"
    ),
    ResourceDomain.SLINGSHOT_FABRIC: DomainContract(
        ResourceDomain.SLINGSHOT_FABRIC, "slingshot_transport", "slingshot_tx_rx_bytes"
    ),
    ResourceDomain.PERSISTENT_ENDPOINT: DomainContract(
        ResourceDomain.PERSISTENT_ENDPOINT, "lustre_persistent_endpoint", "lustre_ost_bytes"
    ),
    ResourceDomain.COMPLETION_ENDPOINT: DomainContract(
        ResourceDomain.COMPLETION_ENDPOINT, "global_commit_completion", "commit_completion_events"
    ),
}

# Counter scope is part of the path contract.  A host aggregate cannot be
# relabeled as a rank/slice causal measurement after the fact.
DOMAIN_COUNTER_SCOPES: Mapping[ResourceDomain, frozenset[str]] = {
    ResourceDomain.GPU_LOCAL: frozenset({"rank"}),
    ResourceDomain.NVLINK_P2P: frozenset({"pair"}),
    ResourceDomain.PCIE_HOST: frozenset({"rank"}),
    ResourceDomain.HOST_NUMA: frozenset({"rank"}),
    ResourceDomain.NIC_FABRIC: frozenset({"rank"}),
    ResourceDomain.SLINGSHOT_FABRIC: frozenset({"slice"}),
    ResourceDomain.PERSISTENT_ENDPOINT: frozenset({"rank", "endpoint"}),
    ResourceDomain.COMPLETION_ENDPOINT: frozenset({"global"}),
}


def domain_contract(domain: ResourceDomain) -> DomainContract:
    """Return the explicit evidence contract for one resource domain."""

    if not isinstance(domain, ResourceDomain):
        raise TypeError("domain must be a ResourceDomain")
    return DOMAIN_CONTRACTS[domain]


def allowed_counter_scopes(domain: ResourceDomain) -> frozenset[str]:
    if not isinstance(domain, ResourceDomain):
        raise TypeError("domain must be a ResourceDomain")
    return DOMAIN_COUNTER_SCOPES[domain]


def _check_positive(name: str, value: int, *, allow_zero: bool = False) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be positive")


def _check_domains(domains: tuple[ResourceDomain, ...]) -> None:
    if not domains:
        raise ValueError("at least one resource domain is required")
    if len(set(domains)) != len(domains):
        raise ValueError("resource domains must be unique")
    if any(not isinstance(domain, ResourceDomain) for domain in domains):
        raise TypeError("domains must contain ResourceDomain values")


@dataclass(frozen=True)
class FlowStage:
    """One stage in a checkpoint/KV transfer DAG.

    ``domains`` is an observed or explicitly declared path, not a topology
    closure.  A D2H stage may therefore contain ``PCIE_HOST`` and ``HOST_NUMA``
    without containing ``NVLINK_P2P``.
    """

    stage_id: str
    bytes: int
    domains: tuple[ResourceDomain, ...]
    deadline_ns: int
    max_residual_bytes: int = 0
    # Optional foreground-tail/SLO budget measured from stage admission.
    # ``0`` keeps only the absolute stage deadline constraint.
    tail_budget_ns: int = 0
    # Measured/reserved orchestration time before this stage can make
    # progress.  This captures controller/gather/prepare cost without
    # pretending it is bandwidth on any one route domain.
    control_overhead_ns: int = 0

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("stage_id must be non-empty")
        _check_positive("bytes", self.bytes)
        _check_domains(self.domains)
        _check_positive("deadline_ns", self.deadline_ns)
        _check_positive("max_residual_bytes", self.max_residual_bytes, allow_zero=True)
        if self.max_residual_bytes > self.bytes:
            raise ValueError("max_residual_bytes cannot exceed bytes")
        _check_positive("tail_budget_ns", self.tail_budget_ns, allow_zero=True)
        _check_positive("control_overhead_ns", self.control_overhead_ns, allow_zero=True)


@dataclass(frozen=True)
class StateFlow:
    """A versioned auxiliary flow with ordered, exact-byte stages."""

    flow_id: str
    stages: tuple[FlowStage, ...]
    deadline_ns: int
    version: str = ""

    def __post_init__(self) -> None:
        if not self.flow_id:
            raise ValueError("flow_id must be non-empty")
        if not self.stages:
            raise ValueError("a flow must contain at least one stage")
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise ValueError("stage_id values must be unique within a flow")
        _check_positive("deadline_ns", self.deadline_ns)

    @property
    def domains(self) -> frozenset[ResourceDomain]:
        return frozenset(domain for stage in self.stages for domain in stage.domains)

    @property
    def total_bytes(self) -> int:
        return sum(stage.bytes for stage in self.stages)


@dataclass(frozen=True)
class ForegroundOperation:
    """A latency-critical collective or inference request footprint."""

    operation_id: str
    kind: str
    group_id: str
    domains: tuple[ResourceDomain, ...]
    start_ns: int
    end_ns: int
    bytes: int = 1

    def __post_init__(self) -> None:
        if not self.operation_id or not self.kind or not self.group_id:
            raise ValueError("operation_id, kind, and group_id must be non-empty")
        _check_domains(self.domains)
        _check_positive("start_ns", self.start_ns, allow_zero=True)
        _check_positive("end_ns", self.end_ns, allow_zero=True)
        if self.end_ns < self.start_ns:
            raise ValueError("end_ns must not precede start_ns")
        _check_positive("bytes", self.bytes)


@dataclass(frozen=True)
class DomainObservation:
    """One measured overlap observation or controlled intervention result.

    ``uncertainty_ns`` is part of the observation rather than a property of a
    later aggregate.  This prevents a small interventional effect from being
    promoted merely because a different observational sample in the same
    bucket had a larger raw delta.
    """

    domain: ResourceDomain
    foreground_kind: str
    auxiliary_kind: str
    overlapping_bytes: int
    overlap_ns: int
    tail_delta_ns: int
    evidence: EvidenceLevel
    source: str
    uncertainty_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ResourceDomain):
            raise TypeError("domain must be a ResourceDomain")
        if not self.foreground_kind or not self.auxiliary_kind or not self.source:
            raise ValueError("observation labels and source must be non-empty")
        _check_positive("overlapping_bytes", self.overlapping_bytes, allow_zero=True)
        _check_positive("overlap_ns", self.overlap_ns, allow_zero=True)
        if type(self.tail_delta_ns) is not int:
            raise TypeError("tail_delta_ns must be an int")
        _check_positive("tail_delta_ns", abs(self.tail_delta_ns), allow_zero=True)
        _check_positive("uncertainty_ns", self.uncertainty_ns, allow_zero=True)
        if not isinstance(self.evidence, EvidenceLevel):
            raise TypeError("evidence must be an EvidenceLevel")


@dataclass(frozen=True)
class ConflictAggregate:
    domain: ResourceDomain
    foreground_kind: str
    auxiliary_kind: str
    samples: int
    interventional_samples: int
    positive_tail_samples: int
    interventional_above_uncertainty_samples: int
    total_overlap_bytes: int
    total_overlap_ns: int
    total_tail_delta_ns: int
    evidence: EvidenceLevel

    @property
    def causal_candidate(self) -> bool:
        """Whether this aggregate is eligible for a later control experiment.

        This is intentionally not a causal proof.  It only prevents an
        observational-only record from silently becoming an admission domain.
        """

        return self.interventional_above_uncertainty_samples > 0


def aggregate_observations(
    observations: Iterable[DomainObservation],
) -> Mapping[tuple[ResourceDomain, str, str], ConflictAggregate]:
    """Aggregate explicit observations without adding unmeasured domains."""

    buckets: dict[tuple[ResourceDomain, str, str], list[DomainObservation]] = defaultdict(list)
    for observation in observations:
        if not isinstance(observation, DomainObservation):
            raise TypeError("observations must contain DomainObservation values")
        buckets[(observation.domain, observation.foreground_kind, observation.auxiliary_kind)].append(
            observation
        )

    result: dict[tuple[ResourceDomain, str, str], ConflictAggregate] = {}
    for key, values in buckets.items():
        domain, foreground_kind, auxiliary_kind = key
        interventional = sum(item.evidence is EvidenceLevel.INTERVENTIONAL for item in values)
        positive = sum(item.tail_delta_ns > 0 for item in values)
        interventional_above_uncertainty = sum(
            item.evidence is EvidenceLevel.INTERVENTIONAL
            and item.tail_delta_ns > item.uncertainty_ns
            for item in values
        )
        evidence = (
            EvidenceLevel.INTERVENTIONAL
            if interventional
            else EvidenceLevel.OBSERVATIONAL
        )
        result[key] = ConflictAggregate(
            domain=domain,
            foreground_kind=foreground_kind,
            auxiliary_kind=auxiliary_kind,
            samples=len(values),
            interventional_samples=interventional,
            positive_tail_samples=positive,
            interventional_above_uncertainty_samples=interventional_above_uncertainty,
            total_overlap_bytes=sum(item.overlapping_bytes for item in values),
            total_overlap_ns=sum(item.overlap_ns for item in values),
            total_tail_delta_ns=sum(item.tail_delta_ns for item in values),
            evidence=evidence,
        )
    return result


def causal_candidate_domains(
    observations: Iterable[DomainObservation],
) -> frozenset[ResourceDomain]:
    """Return only domains eligible for a future static intervention."""

    aggregates = aggregate_observations(observations)
    return frozenset(
        aggregate.domain for aggregate in aggregates.values() if aggregate.causal_candidate
    )
