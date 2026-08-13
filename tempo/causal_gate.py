"""Fail-closed causal-promotion gate for TEMPO-RD attribution modes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from collections.abc import Mapping
from typing import Iterable

from tempo.resource_domain import ResourceDomain


@dataclass(frozen=True)
class CausalModeRecord:
    mode: str
    domain: ResourceDomain | None
    tail_p99_ns: int
    skew_p99_ns: int
    deadline_met: bool
    correctness_met: bool
    samples: int
    domain_exposure_ns: Mapping[ResourceDomain, int] | None = None

    def __post_init__(self) -> None:
        if not self.mode:
            raise ValueError("mode must be non-empty")
        if self.domain is not None and not isinstance(self.domain, ResourceDomain):
            raise TypeError("domain must be a ResourceDomain or None")
        for name, value in (
            ("tail_p99_ns", self.tail_p99_ns),
            ("skew_p99_ns", self.skew_p99_ns),
            ("samples", self.samples),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if type(self.deadline_met) is not bool or type(self.correctness_met) is not bool:
            raise TypeError("deadline_met and correctness_met must be bool")
        if self.domain_exposure_ns is not None:
            if not isinstance(self.domain_exposure_ns, Mapping):
                raise TypeError("domain_exposure_ns must be a mapping or None")
            for domain, exposure in self.domain_exposure_ns.items():
                if not isinstance(domain, ResourceDomain):
                    raise TypeError("domain_exposure_ns keys must be ResourceDomain values")
                if type(exposure) is not int or exposure < 0:
                    raise ValueError("domain_exposure_ns values must be non-negative ints")


@dataclass(frozen=True)
class InferenceModeRecord:
    """Inference analogue of ``CausalModeRecord`` for KV-flow screens.

    ``slo_goodput_milli`` is an integer fixed-point fraction (0..1_000_000),
    avoiding floating-point acceptance decisions in the promotion gate.
    """

    mode: str
    domain: ResourceDomain | None
    ttft_p99_ns: int
    itl_p99_ns: int
    slo_goodput_milli: int
    deadline_met: bool
    correctness_met: bool
    samples: int
    max_domain_exposure_ns: int = 0
    # Per-route exposure is the authoritative bottleneck-shift check.  The
    # scalar above remains as a compact summary/backward-compatible fallback
    # for CPU-only callers that do not yet carry a route map.
    domain_exposure_ns: Mapping[ResourceDomain, int] | None = None

    def __post_init__(self) -> None:
        if not self.mode:
            raise ValueError("mode must be non-empty")
        if self.domain is not None and not isinstance(self.domain, ResourceDomain):
            raise TypeError("domain must be a ResourceDomain or None")
        for name, value in (
            ("ttft_p99_ns", self.ttft_p99_ns),
            ("itl_p99_ns", self.itl_p99_ns),
            ("slo_goodput_milli", self.slo_goodput_milli),
            ("samples", self.samples),
            ("max_domain_exposure_ns", self.max_domain_exposure_ns),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.slo_goodput_milli > 1_000_000:
            raise ValueError("slo_goodput_milli must be at most 1_000_000")
        if type(self.deadline_met) is not bool or type(self.correctness_met) is not bool:
            raise TypeError("deadline_met and correctness_met must be bool")
        if self.domain_exposure_ns is not None:
            if not isinstance(self.domain_exposure_ns, Mapping):
                raise TypeError("domain_exposure_ns must be a mapping or None")
            for domain, exposure in self.domain_exposure_ns.items():
                if not isinstance(domain, ResourceDomain):
                    raise TypeError("domain_exposure_ns keys must be ResourceDomain values")
                if type(exposure) is not int or exposure < 0:
                    raise ValueError("domain_exposure_ns values must be non-negative ints")


@dataclass(frozen=True)
class CausalGateConfig:
    practical_tail_margin: float = 0.05
    practical_skew_margin: float = 0.05
    minimum_samples: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.practical_tail_margin) or not 0.0 <= self.practical_tail_margin:
            raise ValueError("practical_tail_margin must be non-negative")
        if not math.isfinite(self.practical_skew_margin) or not 0.0 <= self.practical_skew_margin:
            raise ValueError("practical_skew_margin must be non-negative")
        if type(self.minimum_samples) is not int or self.minimum_samples <= 0:
            raise ValueError("minimum_samples must be positive")


@dataclass(frozen=True)
class CausalPromotion:
    eligible_domains: frozenset[ResourceDomain]
    headroom: bool
    placebo_clean: bool
    reasons: tuple[str, ...]

    @property
    def promote_static_policy(self) -> bool:
        return bool(self.headroom and self.placebo_clean and self.eligible_domains)


def _margin_fraction(margin: float) -> tuple[int, int]:
    """Return an exact decimal fraction for a configured practical margin."""

    value = Fraction(str(margin))
    return value.numerator, value.denominator


def _meets_worsening_margin(value: int, baseline: int, margin: float) -> bool:
    """Compare ``value >= baseline * (1 + margin)`` without float rounding."""

    numerator, denominator = _margin_fraction(margin)
    return value * denominator >= baseline * (denominator + numerator)


def _meets_goodput_drop(value: int, baseline: int, margin: float) -> bool:
    """Compare ``value <= baseline * (1 - margin)`` exactly."""

    numerator, denominator = _margin_fraction(margin)
    return value * denominator <= baseline * (denominator - numerator)


def evaluate_causal_matrix(
    records: Iterable[CausalModeRecord],
    *,
    config: CausalGateConfig = CausalGateConfig(),
) -> CausalPromotion:
    """Evaluate one paired G1 matrix without inferring unmeasured domains.

    ``open_combined`` is the required optimized-open counterfactual.  A domain
    becomes eligible only if its intervention improves both tail and skew over
    open, while preserving deadline/correctness.  Placebo records are required
    to be no better than open; otherwise attribution is not isolated.
    """

    record_list = tuple(records)
    modes = tuple(record.mode for record in record_list)
    duplicates = sorted({mode for mode in modes if modes.count(mode) > 1})
    if duplicates:
        return CausalPromotion(
            frozenset(),
            False,
            False,
            (f"duplicate modes: {duplicates}",),
        )
    by_mode: Mapping[str, CausalModeRecord] = {record.mode: record for record in record_list}
    required = {"fg_only", "open_combined"}
    missing = required - set(by_mode)
    if missing:
        return CausalPromotion(frozenset(), False, False, (f"missing modes: {sorted(missing)}",))
    fg = by_mode["fg_only"]
    opened = by_mode["open_combined"]
    reasons: list[str] = []
    baseline_domain_valid = fg.domain is None and opened.domain is None
    if not baseline_domain_valid:
        reasons.append("foreground/open baseline must not name an intervention domain")
    baseline_samples_valid = (
        fg.samples >= config.minimum_samples
        and opened.samples >= config.minimum_samples
    )
    if not baseline_samples_valid:
        reasons.append("insufficient foreground/open samples")
    baseline_valid = True
    if not fg.deadline_met or not fg.correctness_met:
        reasons.append("foreground-only baseline deadline/correctness failed")
        baseline_valid = False
    if not opened.deadline_met or not opened.correctness_met:
        reasons.append("optimized-open baseline deadline/correctness failed")
        baseline_valid = False
    headroom = (
        baseline_valid
        and baseline_domain_valid
        and baseline_samples_valid
        and (
            _meets_worsening_margin(
                opened.tail_p99_ns, fg.tail_p99_ns, config.practical_tail_margin
            )
            or _meets_worsening_margin(
                opened.skew_p99_ns, fg.skew_p99_ns, config.practical_skew_margin
            )
        )
    )
    if not headroom:
        reasons.append("optimized-open has no preregistered practical headroom")

    placebo_clean = True
    for record in by_mode.values():
        if record.mode in {"fg_only", "open_combined", "combined"} or record.domain is not None:
            continue
        if record.tail_p99_ns < opened.tail_p99_ns or record.skew_p99_ns < opened.skew_p99_ns:
            placebo_clean = False
    if not placebo_clean:
        reasons.append("placebo improves over open; domain attribution is ambiguous")

    eligible: set[ResourceDomain] = set()
    for record in by_mode.values():
        if record.domain is None or record.mode in {"fg_only", "open_combined"}:
            continue
        if record.samples < config.minimum_samples:
            reasons.append(f"{record.mode}: insufficient samples")
            continue
        if not record.deadline_met or not record.correctness_met:
            reasons.append(f"{record.mode}: deadline/correctness failed")
            continue
        if record.domain_exposure_ns is not None and opened.domain_exposure_ns is not None:
            candidate_domains = set(record.domain_exposure_ns)
            opened_domains = set(opened.domain_exposure_ns)
            if record.domain not in candidate_domains or record.domain not in opened_domains:
                reasons.append(
                    f"{record.mode}: intervention domain exposure is missing from candidate or optimized-open"
                )
                continue
            extra_positive = tuple(
                domain
                for domain, exposure in record.domain_exposure_ns.items()
                if domain not in opened.domain_exposure_ns and exposure > 0
            )
            if extra_positive:
                reasons.append(
                    f"{record.mode}: introduces unpaired exposed domains "
                    f"{tuple(domain.value for domain in extra_positive)}"
                )
                continue
            common_domains = candidate_domains & opened_domains
            if any(
                record.domain_exposure_ns[domain] > opened.domain_exposure_ns[domain]
                for domain in common_domains
            ):
                reasons.append(f"{record.mode}: domain exposure exceeds optimized-open")
                continue
        if record.tail_p99_ns < opened.tail_p99_ns and record.skew_p99_ns < opened.skew_p99_ns:
            eligible.add(record.domain)
        else:
            reasons.append(f"{record.mode}: does not improve both tail and skew over open")
    return CausalPromotion(frozenset(eligible), headroom, placebo_clean, tuple(reasons))


def evaluate_inference_matrix(
    records: Iterable[InferenceModeRecord],
    *,
    config: CausalGateConfig = CausalGateConfig(),
) -> CausalPromotion:
    """Promote a KV-domain intervention only under a matched-open SLO screen.

    The open lane must first exhibit practical TTFT/ITL degradation or a
    practical goodput loss relative to foreground-only.  An intervention must
    improve both latency tails and must not reduce SLO goodput, deadline, or
    correctness.  This is an offline gate; it does not imply a live inference
    backend exists.
    """

    record_list = tuple(records)
    modes = tuple(record.mode for record in record_list)
    duplicates = sorted({mode for mode in modes if modes.count(mode) > 1})
    if duplicates:
        return CausalPromotion(
            frozenset(),
            False,
            False,
            (f"duplicate modes: {duplicates}",),
        )
    by_mode: Mapping[str, InferenceModeRecord] = {record.mode: record for record in record_list}
    required = {"fg_only", "open_combined"}
    missing = required - set(by_mode)
    if missing:
        return CausalPromotion(frozenset(), False, False, (f"missing modes: {sorted(missing)}",))
    fg = by_mode["fg_only"]
    opened = by_mode["open_combined"]
    reasons: list[str] = []
    baseline_domain_valid = fg.domain is None and opened.domain is None
    if not baseline_domain_valid:
        reasons.append("foreground/open baseline must not name an intervention domain")
    baseline_samples_valid = (
        fg.samples >= config.minimum_samples
        and opened.samples >= config.minimum_samples
    )
    if not baseline_samples_valid:
        reasons.append("insufficient foreground/open samples")
    baseline_valid = True
    if not fg.deadline_met or not fg.correctness_met:
        reasons.append("foreground-only baseline deadline/correctness failed")
        baseline_valid = False
    if not opened.deadline_met or not opened.correctness_met:
        reasons.append("optimized-open baseline deadline/correctness failed")
        baseline_valid = False
    ttft_headroom = _meets_worsening_margin(
        opened.ttft_p99_ns, fg.ttft_p99_ns, config.practical_tail_margin
    )
    itl_headroom = _meets_worsening_margin(
        opened.itl_p99_ns, fg.itl_p99_ns, config.practical_skew_margin
    )
    goodput_headroom = (
        fg.slo_goodput_milli > 0
        and opened.slo_goodput_milli
        and _meets_goodput_drop(
            opened.slo_goodput_milli, fg.slo_goodput_milli, config.practical_tail_margin
        )
    )
    headroom = (
        baseline_valid
        and baseline_domain_valid
        and baseline_samples_valid
        and (ttft_headroom or itl_headroom or goodput_headroom)
    )
    if not headroom:
        reasons.append("optimized-open has no preregistered inference headroom")

    placebo_clean = True
    for record in by_mode.values():
        if record.mode in {"fg_only", "open_combined", "combined"} or record.domain is not None:
            continue
        if (
            record.ttft_p99_ns < opened.ttft_p99_ns
            or record.itl_p99_ns < opened.itl_p99_ns
            or record.slo_goodput_milli > opened.slo_goodput_milli
        ):
            placebo_clean = False
    if not placebo_clean:
        reasons.append("placebo improves over open; inference attribution is ambiguous")

    eligible: set[ResourceDomain] = set()
    for record in by_mode.values():
        if record.domain is None or record.mode in {"fg_only", "open_combined"}:
            continue
        if record.samples < config.minimum_samples:
            reasons.append(f"{record.mode}: insufficient samples")
            continue
        if not record.deadline_met or not record.correctness_met:
            reasons.append(f"{record.mode}: deadline/correctness failed")
            continue
        # An intervention that improves TTFT/ITL by keeping a larger active
        # footprint on another declared domain is not a causal promotion. A
        # mode must stay no more exposed than the matched open lane across
        # every resource domain represented by its route evidence.  The
        # scalar check is retained only for legacy CPU callers; live result
        # validators always supply the exact per-domain map.
        exposure_shift = record.max_domain_exposure_ns > opened.max_domain_exposure_ns
        if record.domain_exposure_ns is not None and opened.domain_exposure_ns is not None:
            # Promotion is a causal scheduling claim, so the candidate must
            # traverse the same measured route as optimized-open.  Comparing
            # only common keys would let an endpoint/fabric change masquerade
            # as an orchestration win (for example, a remote NIC route with a
            # lower scalar maximum than a local host route).  A separate
            # matched-open record is required before a new endpoint/domain
            # route can be promoted.
            if set(record.domain_exposure_ns) != set(opened.domain_exposure_ns):
                reasons.append(
                    f"{record.mode}: route-domain exposure set differs from optimized-open; "
                    "matched endpoint baseline required"
                )
                continue
            domains = set(record.domain_exposure_ns) & set(opened.domain_exposure_ns)
            exposure_shift = any(
                record.domain_exposure_ns.get(domain, 0)
                > opened.domain_exposure_ns.get(domain, 0)
                for domain in domains
            )
        if exposure_shift:
            reasons.append(
                f"{record.mode}: domain exposure exceeds optimized-open"
            )
            continue
        improves_latency = (
            record.ttft_p99_ns < opened.ttft_p99_ns
            and record.itl_p99_ns < opened.itl_p99_ns
        )
        preserves_goodput = record.slo_goodput_milli >= opened.slo_goodput_milli
        if improves_latency and preserves_goodput:
            eligible.add(record.domain)
        else:
            reasons.append(f"{record.mode}: does not improve both latency tails and preserve SLO goodput")
    return CausalPromotion(frozenset(eligible), headroom, placebo_clean, tuple(reasons))
