"""Explicit evidence records for the TEMPO-RD causal resource atlas."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from tempo.resource_domain import EvidenceLevel, ResourceDomain, domain_contract


class CounterSupport(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    NOT_COLLECTED = "not_collected"
    AMBIGUOUS = "ambiguous"


class PathStatus(str, Enum):
    OBSERVED = "observed"
    DECLARED = "declared"
    NOT_TRAVERSED = "not_traversed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DomainEvidence:
    domain: ResourceDomain
    mode: str
    foreground_kind: str
    auxiliary_kind: str
    overlapping_bytes: int
    overlap_ns: int
    tail_delta_ns: int
    evidence: EvidenceLevel
    counter_support: CounterSupport
    path_status: PathStatus
    uncertainty_ns: int
    source: str
    path_evidence: str = ""
    counter_family: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ResourceDomain):
            raise TypeError("domain must be a ResourceDomain")
        if not self.mode or not self.foreground_kind or not self.auxiliary_kind or not self.source:
            raise ValueError("mode, operation labels, and source are required")
        for name, value in (
            ("overlapping_bytes", self.overlapping_bytes),
            ("overlap_ns", self.overlap_ns),
            ("uncertainty_ns", self.uncertainty_ns),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if type(self.tail_delta_ns) is not int:
            raise TypeError("tail_delta_ns must be an int")
        if not isinstance(self.evidence, EvidenceLevel):
            raise TypeError("evidence must be EvidenceLevel")
        if not isinstance(self.counter_support, CounterSupport):
            raise TypeError("counter_support must be CounterSupport")
        if not isinstance(self.path_status, PathStatus):
            raise TypeError("path_status must be PathStatus")
        if self.evidence is EvidenceLevel.INTERVENTIONAL and self.path_status is not PathStatus.OBSERVED:
            raise ValueError("an intervention requires an observed path")
        contract = domain_contract(self.domain)
        if self.path_status is PathStatus.OBSERVED:
            if self.path_evidence != contract.path_evidence:
                raise ValueError(
                    f"observed path label does not match {self.domain.value} contract"
                )
        if self.counter_support is CounterSupport.SUPPORTED:
            if self.counter_family != contract.counter_family:
                raise ValueError(
                    f"counter family does not match {self.domain.value} contract"
                )


@dataclass(frozen=True)
class AtlasEntry:
    domain: ResourceDomain
    samples: int
    interventional_samples: int
    supported_counter_samples: int
    positive_tail_samples: int
    total_overlap_bytes: int
    total_tail_delta_ns: int
    causal_candidate: bool


@dataclass(frozen=True)
class DomainCoverage:
    """Explicit coverage result for a declared route/domain inventory.

    Missing or unsupported domains remain visible instead of disappearing
    from an atlas built only from records that happened to be emitted.  The
    result is descriptive; ``causal_ready`` is the only field that may feed a
    later promotion gate.
    """

    required_domains: frozenset[ResourceDomain]
    observed_domains: frozenset[ResourceDomain]
    supported_domains: frozenset[ResourceDomain]
    missing_domains: frozenset[ResourceDomain]
    causal_domains: frozenset[ResourceDomain]

    @property
    def coverage_complete(self) -> bool:
        return not self.missing_domains and self.required_domains <= self.observed_domains

    @property
    def causal_ready(self) -> bool:
        # Observed counters establish attribution coverage, but do not by
        # themselves establish causality.  Every required domain must also
        # have an interventional tail change above its uncertainty before a
        # controller or paper claim may consume this flag.
        return (
            self.coverage_complete
            and self.required_domains <= self.supported_domains
            and self.required_domains <= self.causal_domains
        )


def assess_domain_coverage(
    records: Iterable[DomainEvidence],
    required_domains: Iterable[ResourceDomain],
) -> DomainCoverage:
    """Report exact observed/counter coverage for a declared path.

    This deliberately does not infer a path from topology.  A required domain
    with no record is ``missing``; a record with only declared/not-collected
    status is not promoted to observed/supported.  A domain is ``causal`` only
    when at least one interventional sample exceeds its own uncertainty.
    """

    required = frozenset(required_domains)
    if any(not isinstance(domain, ResourceDomain) for domain in required):
        raise TypeError("required_domains must contain ResourceDomain values")
    values = tuple(records)
    if any(not isinstance(record, DomainEvidence) for record in values):
        raise TypeError("records must contain DomainEvidence values")
    observed = frozenset(
        record.domain for record in values if record.path_status is PathStatus.OBSERVED
    )
    supported = frozenset(
        record.domain for record in values
        if record.path_status is PathStatus.OBSERVED
        and record.counter_support is CounterSupport.SUPPORTED
    )
    causal = frozenset(
        record.domain for record in values
        if record.evidence is EvidenceLevel.INTERVENTIONAL
        and record.path_status is PathStatus.OBSERVED
        and record.counter_support is CounterSupport.SUPPORTED
        and record.tail_delta_ns > record.uncertainty_ns
    )
    return DomainCoverage(
        required_domains=required,
        observed_domains=observed & required,
        supported_domains=supported & required,
        missing_domains=required - observed,
        causal_domains=causal & required,
    )


def build_atlas(records: Iterable[DomainEvidence]) -> Mapping[ResourceDomain, AtlasEntry]:
    grouped: dict[ResourceDomain, list[DomainEvidence]] = {}
    for record in records:
        grouped.setdefault(record.domain, []).append(record)
    atlas: dict[ResourceDomain, AtlasEntry] = {}
    for domain, values in grouped.items():
        interventional = sum(value.evidence is EvidenceLevel.INTERVENTIONAL for value in values)
        supported = sum(value.counter_support is CounterSupport.SUPPORTED for value in values)
        positive = sum(value.tail_delta_ns > 0 for value in values)
        causal = any(
            value.evidence is EvidenceLevel.INTERVENTIONAL
            and value.counter_support is CounterSupport.SUPPORTED
            and value.path_status is PathStatus.OBSERVED
            and value.tail_delta_ns > value.uncertainty_ns
            for value in values
        )
        atlas[domain] = AtlasEntry(
            domain=domain,
            samples=len(values),
            interventional_samples=interventional,
            supported_counter_samples=supported,
            positive_tail_samples=positive,
            total_overlap_bytes=sum(value.overlapping_bytes for value in values),
            total_tail_delta_ns=sum(value.tail_delta_ns for value in values),
            causal_candidate=causal,
        )
    return atlas


def controller_candidates(records: Iterable[DomainEvidence]) -> frozenset[ResourceDomain]:
    return frozenset(domain for domain, entry in build_atlas(records).items() if entry.causal_candidate)


def detect_bottleneck_shift(
    before: Mapping[ResourceDomain, int],
    after: Mapping[ResourceDomain, int],
    *,
    controlled_domain: ResourceDomain,
) -> tuple[ResourceDomain, ...]:
    """Return domains whose exposure increased after another domain was capped.

    A lower foreground tail alone is not enough to promote a controller: the
    intervention may simply move work to a different shared resource.  This
    helper is intentionally an offline accounting predicate.  It requires an
    exact domain snapshot (no missing or invented domains), requires the
    controlled domain's exposure to decrease, and reports only *other* domains
    that increased.  Callers should record a non-empty result as a
    bottleneck-shift failure rather than treating it as a causal win.
    """

    if not isinstance(controlled_domain, ResourceDomain):
        raise TypeError("controlled_domain must be a ResourceDomain")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise TypeError("before and after must be mappings")
    if set(before) != set(after):
        raise ValueError("before and after must cover the exact same domains")
    if controlled_domain not in before:
        raise ValueError("controlled_domain is missing from the snapshots")
    for snapshot_name, snapshot in (("before", before), ("after", after)):
        for domain, value in snapshot.items():
            if not isinstance(domain, ResourceDomain):
                raise TypeError(f"{snapshot_name} contains a non-domain key")
            if type(value) is not int or value < 0:
                raise ValueError(f"{snapshot_name} exposure values must be non-negative ints")
    if after[controlled_domain] >= before[controlled_domain]:
        return ()
    return tuple(
        domain for domain in sorted(before, key=lambda item: item.value)
        if domain is not controlled_domain and after[domain] > before[domain]
    )
