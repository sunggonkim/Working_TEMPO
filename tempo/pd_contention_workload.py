"""Preregistered multi-tenant workload contract for TEMPO Elastic-PD.

This module is deliberately independent of the routing controller.  It
materializes arm-invariant request timing for foreground inference plus three
route-pinned inference tenants:

* decoder-local-prefill-hot: long unique-cold prompt, two-token output,
  always local;
* remote-hot: long unique-cold prompt, two-token output, official remote P/D.
* KV-remote-hot: pre-seeded P_ONLY long prompt, two-token output, official
  remote retrieval/transfer/install path.

The workload is not eligible for controller tuning until fixed local and
fixed remote show the opposite C1/C2 crossovers defined below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
import statistics
from typing import Iterable, Mapping, Sequence


SCHEMA = "tempo-pd-contention-workload-v4"
OBSERVATION_SCHEMA = "tempo-pd-contention-fixed-observation-v4"
GATE_SCHEMA = "tempo-pd-contention-crossover-gate-v4"
CROSSOVER_MIN_GAIN = 0.05
CALIBRATION_FRACTIONS = (0.50, 0.70, 0.85, 1.00)
CALIBRATION_REPLICATES = 2
PHASE_DURATION_MS = 30_000.0
OVERLOAD_MULTIPLIER = 1.15
KV_REMOTE_RATE_PER_S = 12.0


class ContentionState(str, Enum):
    C0 = "c0_cool"
    C1 = "c1_decoder_hot"
    C2 = "c2_remote_hot"
    C2_KV = "c2_kv_remote_hot"
    C3 = "c3_both_hot"
    RECOVERY = "recovery"


class Tenant(str, Enum):
    FOREGROUND = "foreground"
    DECODER_HOT = "decoder_hot"
    REMOTE_HOT = "remote_hot"
    KV_REMOTE_HOT = "kv_remote_hot"


class ForegroundArm(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"
    PREDICTOR = "predictor"
    QUEUE_ONLY = "queue_only"
    OLD_SCALAR = "old_scalar"
    TEMPO = "tempo"


class TrafficShape(str, Enum):
    STABLE = "stable"
    BURST = "burst"
    OVERLOAD = "overload"


class CacheState(str, Enum):
    MISS = "miss"
    P_ONLY = "p_only"
    D_ONLY = "d_only"
    BOTH = "both"


@dataclass(frozen=True)
class TokenGeometry:
    prompt_tokens: int
    output_tokens: int
    cache_state: CacheState

    def __post_init__(self) -> None:
        if type(self.prompt_tokens) is not int or self.prompt_tokens < 2:
            raise ValueError("prompt_tokens must be an int >= 2")
        if type(self.output_tokens) is not int or self.output_tokens < 2:
            raise ValueError("output_tokens must be an int >= 2")
        if not isinstance(self.cache_state, CacheState):
            raise TypeError("cache_state must be CacheState")


VALIDATION_FOREGROUND_GEOMETRIES = (
    TokenGeometry(512, 16, CacheState.MISS),
    TokenGeometry(2048, 128, CacheState.P_ONLY),
    TokenGeometry(4094, 256, CacheState.D_ONLY),
    TokenGeometry(512, 128, CacheState.BOTH),
    TokenGeometry(2048, 256, CacheState.MISS),
    TokenGeometry(4094, 16, CacheState.P_ONLY),
)
CROSSOVER_FOREGROUND_GEOMETRIES = (
    TokenGeometry(4094, 2, CacheState.MISS),
)
# Backward-compatible name for the broad C4/final-validation mix.
FOREGROUND_GEOMETRIES = VALIDATION_FOREGROUND_GEOMETRIES
DECODER_HOT_GEOMETRY = TokenGeometry(4094, 2, CacheState.MISS)
REMOTE_HOT_GEOMETRY = TokenGeometry(4094, 2, CacheState.MISS)
KV_REMOTE_HOT_GEOMETRY = TokenGeometry(4094, 2, CacheState.P_ONLY)


@dataclass(frozen=True)
class LoadSelection:
    """Capacity-normalized background rates frozen before controller work."""

    decoder_reference_rate_per_s: float
    remote_reference_rate_per_s: float
    decoder_fraction: float
    remote_fraction: float
    kv_remote_rate_per_s: float = KV_REMOTE_RATE_PER_S

    def __post_init__(self) -> None:
        for name, value in (
            ("decoder_reference_rate_per_s", self.decoder_reference_rate_per_s),
            ("remote_reference_rate_per_s", self.remote_reference_rate_per_s),
            ("kv_remote_rate_per_s", self.kv_remote_rate_per_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("decoder_fraction", self.decoder_fraction),
            ("remote_fraction", self.remote_fraction),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) not in CALIBRATION_FRACTIONS
            ):
                raise ValueError(
                    f"{name} must be one preregistered calibration fraction")

    @property
    def decoder_rate_per_s(self) -> float:
        return self.decoder_reference_rate_per_s * self.decoder_fraction

    @property
    def remote_rate_per_s(self) -> float:
        return self.remote_reference_rate_per_s * self.remote_fraction


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    phase: ContentionState
    tenant: Tenant
    arm: ForegroundArm
    arrival_offset_ms: float
    geometry: TokenGeometry
    ordinal: int

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("request_id must be nonempty")
        if not isinstance(self.phase, ContentionState):
            raise TypeError("phase must be ContentionState")
        if not isinstance(self.tenant, Tenant):
            raise TypeError("tenant must be Tenant")
        if not isinstance(self.arm, ForegroundArm):
            raise TypeError("arm must be ForegroundArm")
        if (
            isinstance(self.arrival_offset_ms, bool)
            or not isinstance(self.arrival_offset_ms, (int, float))
            or not math.isfinite(float(self.arrival_offset_ms))
            or float(self.arrival_offset_ms) < 0.0
        ):
            raise ValueError("arrival_offset_ms must be finite and non-negative")
        if not isinstance(self.geometry, TokenGeometry):
            raise TypeError("geometry must be TokenGeometry")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative int")

    def semantic_dict(self) -> dict[str, object]:
        """Arm- and trial-independent schedule identity."""

        return {
            "phase": self.phase.value,
            "tenant": self.tenant.value,
            "arrival_offset_ms": round(float(self.arrival_offset_ms), 6),
            "prompt_tokens": self.geometry.prompt_tokens,
            "output_tokens": self.geometry.output_tokens,
            "cache_state": self.geometry.cache_state.value,
            "ordinal": self.ordinal,
        }


def _request_arm_marker(arm: ForegroundArm) -> str:
    if arm is ForegroundArm.QUEUE_ONLY:
        return "predictor"
    if arm is ForegroundArm.OLD_SCALAR:
        return "tempo"
    return arm.value


def _uniform_offsets(start_ms: float, duration_ms: float, rate_per_s: float) -> list[float]:
    if rate_per_s <= 0.0:
        return []
    count = int(math.floor(rate_per_s * duration_ms / 1000.0))
    if count < 1:
        return []
    spacing_ms = 1000.0 / rate_per_s
    return [start_ms + (index + 0.5) * spacing_ms for index in range(count)]


def _arrival_offsets(
    start_ms: float,
    duration_ms: float,
    rate_per_s: float,
    shape: TrafficShape,
) -> list[float]:
    multiplier = OVERLOAD_MULTIPLIER if shape is TrafficShape.OVERLOAD else 1.0
    offsets = _uniform_offsets(start_ms, duration_ms, rate_per_s * multiplier)
    if shape is not TrafficShape.BURST:
        return offsets
    # Keep the same average offered rate while packing each one-second epoch
    # into its first 250 ms.  The transform is deterministic and does not
    # synchronize clocks across hosts; clients use their own run-start clock.
    transformed = []
    for offset in offsets:
        relative = offset - start_ms
        epoch = math.floor(relative / 1000.0)
        within = relative - epoch * 1000.0
        transformed.append(start_ms + epoch * 1000.0 + within * 0.25)
    return transformed


def build_schedule(
    *,
    states: Sequence[ContentionState],
    selection: LoadSelection,
    foreground_arm: ForegroundArm,
    foreground_rate_per_s: float,
    trial_id: str,
    shape: TrafficShape = TrafficShape.STABLE,
    phase_duration_ms: float = PHASE_DURATION_MS,
    foreground_geometries: Sequence[TokenGeometry] = FOREGROUND_GEOMETRIES,
    passive_endpoint_feedback: bool = False,
) -> tuple[ScheduledRequest, ...]:
    """Materialize one C-state block or the C4 phase-changing trace."""

    if not states or any(not isinstance(state, ContentionState) for state in states):
        raise ValueError("states must be a nonempty ContentionState sequence")
    if len(set(states)) != len(states):
        raise ValueError("states cannot repeat within one schedule")
    if not isinstance(selection, LoadSelection):
        raise TypeError("selection must be LoadSelection")
    if not isinstance(foreground_arm, ForegroundArm):
        raise TypeError("foreground_arm must be ForegroundArm")
    if type(trial_id) is not str or not trial_id.strip():
        raise ValueError("trial_id must be nonempty")
    if not isinstance(shape, TrafficShape):
        raise TypeError("shape must be TrafficShape")
    if type(passive_endpoint_feedback) is not bool:
        raise TypeError("passive_endpoint_feedback must be bool")
    for name, value in (
        ("foreground_rate_per_s", foreground_rate_per_s),
        ("phase_duration_ms", phase_duration_ms),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    geometries = tuple(foreground_geometries)
    if not geometries or any(not isinstance(item, TokenGeometry) for item in geometries):
        raise ValueError("foreground_geometries must contain TokenGeometry values")

    requests: list[ScheduledRequest] = []
    ordinals = {tenant: 0 for tenant in Tenant}
    for phase_index, state in enumerate(states):
        start_ms = phase_index * float(phase_duration_ms)
        rates = {
            Tenant.FOREGROUND: float(foreground_rate_per_s),
            Tenant.DECODER_HOT: (
                selection.decoder_rate_per_s
                if state in {ContentionState.C1, ContentionState.C3}
                else 0.0
            ),
            Tenant.REMOTE_HOT: (
                selection.remote_rate_per_s
                if state is ContentionState.C2
                else 0.0
            ),
            Tenant.KV_REMOTE_HOT: (
                selection.kv_remote_rate_per_s
                if state in {ContentionState.C2_KV, ContentionState.C3}
                else 0.0
            ),
        }
        for tenant in Tenant:
            for offset in _arrival_offsets(
                start_ms, float(phase_duration_ms), rates[tenant], shape
            ):
                ordinal = ordinals[tenant]
                ordinals[tenant] += 1
                if tenant is Tenant.FOREGROUND:
                    arm = foreground_arm
                    geometry = geometries[ordinal % len(geometries)]
                elif tenant is Tenant.DECODER_HOT:
                    arm = ForegroundArm.LOCAL
                    geometry = DECODER_HOT_GEOMETRY
                elif tenant is Tenant.REMOTE_HOT:
                    arm = ForegroundArm.REMOTE
                    geometry = REMOTE_HOT_GEOMETRY
                else:
                    arm = ForegroundArm.REMOTE
                    geometry = KV_REMOTE_HOT_GEOMETRY
                marker = _request_arm_marker(arm)
                evidence_markers = [
                    f"cache-{geometry.cache_state.value.replace('_', '-')}-measured"
                ]
                if tenant is Tenant.KV_REMOTE_HOT:
                    if evidence_markers != ["cache-p-only-measured"]:
                        raise RuntimeError(
                            "KV-remote-hot cache marker is inconsistent")
                if (
                    passive_endpoint_feedback
                    and tenant is not Tenant.FOREGROUND
                ):
                    evidence_markers.append("endpoint-observed")
                evidence = (
                    "-" + "-".join(evidence_markers)
                    if evidence_markers else "")
                request_id = (
                    f"epd-{marker}-ct{evidence}-{trial_id}-{state.value}-"
                    f"{tenant.value}-{ordinal:06d}"
                )
                requests.append(ScheduledRequest(
                    request_id=request_id,
                    phase=state,
                    tenant=tenant,
                    arm=arm,
                    arrival_offset_ms=offset,
                    geometry=geometry,
                    ordinal=ordinal,
                ))
    requests.sort(key=lambda item: (
        item.arrival_offset_ms, tuple(Tenant).index(item.tenant), item.ordinal))
    identifiers = [item.request_id for item in requests]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("materialized request IDs are not unique")
    return tuple(requests)


def semantic_schedule_sha256(requests: Sequence[ScheduledRequest]) -> str:
    if not requests:
        raise ValueError("schedule must be nonempty")
    payload = [item.semantic_dict() for item in requests]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ForegroundObservation:
    pair_key: str
    e2e_ms: float
    output_sha256: str

    def __post_init__(self) -> None:
        if type(self.pair_key) is not str or not self.pair_key:
            raise ValueError("pair_key must be nonempty")
        if (
            isinstance(self.e2e_ms, bool)
            or not isinstance(self.e2e_ms, (int, float))
            or not math.isfinite(float(self.e2e_ms))
            or float(self.e2e_ms) <= 0.0
        ):
            raise ValueError("e2e_ms must be finite and positive")
        if (
            type(self.output_sha256) is not str
            or len(self.output_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.output_sha256)
        ):
            raise ValueError("output_sha256 must be lowercase SHA-256")


@dataclass(frozen=True)
class FixedArmObservation:
    phase: ContentionState
    load_fraction: float
    replicate: int
    arm: ForegroundArm
    semantic_schedule_sha256: str
    foreground: tuple[ForegroundObservation, ...]
    background_offered: int
    background_completed: int
    background_errors: int
    schema: str = OBSERVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OBSERVATION_SCHEMA:
            raise ValueError("fixed observation schema is not canonical")
        if self.phase not in {ContentionState.C1, ContentionState.C2}:
            raise ValueError("fixed crossover observation must be C1 or C2")
        if (
            isinstance(self.load_fraction, bool)
            or not isinstance(self.load_fraction, (int, float))
            or not math.isfinite(float(self.load_fraction))
            or float(self.load_fraction) not in CALIBRATION_FRACTIONS
        ):
            raise ValueError("load_fraction is outside the preregistered ladder")
        if type(self.replicate) is not int or self.replicate < 0:
            raise ValueError("replicate must be a non-negative int")
        if self.arm not in {ForegroundArm.LOCAL, ForegroundArm.REMOTE}:
            raise ValueError("fixed observation arm must be local or remote")
        if (
            type(self.semantic_schedule_sha256) is not str
            or len(self.semantic_schedule_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.semantic_schedule_sha256
            )
        ):
            raise ValueError("semantic_schedule_sha256 must be lowercase SHA-256")
        if type(self.foreground) is not tuple or not self.foreground:
            raise ValueError("foreground observations must be a nonempty tuple")
        if any(not isinstance(item, ForegroundObservation) for item in self.foreground):
            raise TypeError("foreground must contain ForegroundObservation")
        keys = [item.pair_key for item in self.foreground]
        if len(keys) != len(set(keys)):
            raise ValueError("foreground pair keys must be unique")
        for name, value in (
            ("background_offered", self.background_offered),
            ("background_completed", self.background_completed),
            ("background_errors", self.background_errors),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")


def _paired_gain(
    local: FixedArmObservation,
    remote: FixedArmObservation,
    *,
    winner: ForegroundArm,
) -> tuple[float, float, int]:
    if local.semantic_schedule_sha256 != remote.semantic_schedule_sha256:
        raise ValueError("fixed arms used different semantic schedules")
    if (
        local.background_errors
        or remote.background_errors
        or local.background_completed != local.background_offered
        or remote.background_completed != remote.background_offered
        or local.background_offered < 1
        or remote.background_offered < 1
    ):
        raise ValueError("background inference was incomplete or invalid")
    local_rows = {item.pair_key: item for item in local.foreground}
    remote_rows = {item.pair_key: item for item in remote.foreground}
    if set(local_rows) != set(remote_rows):
        raise ValueError("fixed arms have different foreground pair keys")
    if any(
        local_rows[key].output_sha256 != remote_rows[key].output_sha256
        for key in local_rows
    ):
        raise ValueError("fixed-arm outputs differ")
    local_values = [float(local_rows[key].e2e_ms) for key in sorted(local_rows)]
    remote_values = [float(remote_rows[key].e2e_ms) for key in sorted(local_rows)]
    if winner is ForegroundArm.REMOTE:
        paired = [
            (local_value - remote_value) / local_value
            for local_value, remote_value in zip(local_values, remote_values, strict=True)
        ]
        pooled = (
            statistics.median(local_values) - statistics.median(remote_values)
        ) / statistics.median(local_values)
    elif winner is ForegroundArm.LOCAL:
        paired = [
            (remote_value - local_value) / remote_value
            for local_value, remote_value in zip(local_values, remote_values, strict=True)
        ]
        pooled = (
            statistics.median(remote_values) - statistics.median(local_values)
        ) / statistics.median(remote_values)
    else:
        raise ValueError("winner must be a fixed arm")
    return pooled, statistics.median(paired), len(paired)


def evaluate_crossover(
    observations: Iterable[FixedArmObservation],
    *,
    load_fraction: float,
    minimum_gain: float = CROSSOVER_MIN_GAIN,
) -> dict[str, object]:
    """Require reproducible opposite fixed-arm crossovers at one load level."""

    if (
        isinstance(load_fraction, bool)
        or not isinstance(load_fraction, (int, float))
        or not math.isfinite(float(load_fraction))
        or float(load_fraction) not in CALIBRATION_FRACTIONS
    ):
        raise ValueError("load_fraction is outside the preregistered ladder")
    if (
        isinstance(minimum_gain, bool)
        or not isinstance(minimum_gain, (int, float))
        or not 0.0 < float(minimum_gain) < 1.0
    ):
        raise ValueError("minimum_gain must be in (0, 1)")
    rows = tuple(observations)
    if any(not isinstance(item, FixedArmObservation) for item in rows):
        raise TypeError("observations must contain FixedArmObservation")
    selected = [item for item in rows if item.load_fraction == load_fraction]
    expected_keys = {
        (phase, replicate, arm)
        for phase in (ContentionState.C1, ContentionState.C2)
        for replicate in range(CALIBRATION_REPLICATES)
        for arm in (ForegroundArm.LOCAL, ForegroundArm.REMOTE)
    }
    indexed: dict[
        tuple[ContentionState, int, ForegroundArm], FixedArmObservation
    ] = {}
    for item in selected:
        key = (item.phase, item.replicate, item.arm)
        if key in indexed:
            raise ValueError("duplicate fixed-arm observation")
        indexed[key] = item
    if set(indexed) != expected_keys:
        missing = sorted(
            (phase.value, replicate, arm.value)
            for phase, replicate, arm in expected_keys - set(indexed)
        )
        extra = sorted(
            (phase.value, replicate, arm.value)
            for phase, replicate, arm in set(indexed) - expected_keys
        )
        raise ValueError(f"fixed crossover matrix is not exact: missing={missing}, extra={extra}")

    phase_results: dict[str, object] = {}
    all_pass = True
    for phase, winner in (
        (ContentionState.C1, ForegroundArm.REMOTE),
        (ContentionState.C2, ForegroundArm.LOCAL),
    ):
        pooled_gains = []
        paired_gains = []
        request_count = 0
        replicate_direction = []
        for replicate in range(CALIBRATION_REPLICATES):
            pooled, paired, count = _paired_gain(
                indexed[(phase, replicate, ForegroundArm.LOCAL)],
                indexed[(phase, replicate, ForegroundArm.REMOTE)],
                winner=winner,
            )
            pooled_gains.append(pooled)
            paired_gains.append(paired)
            request_count += count
            replicate_direction.append(pooled > 0.0 and paired > 0.0)
        pooled_gain = statistics.median(pooled_gains)
        paired_gain = statistics.median(paired_gains)
        passed = (
            pooled_gain >= float(minimum_gain)
            and paired_gain >= float(minimum_gain)
            and all(replicate_direction)
        )
        all_pass = all_pass and passed
        phase_results[phase.value] = {
            "winner": winner.value,
            "pooled_median_gain": pooled_gain,
            "paired_median_gain": paired_gain,
            "replicate_direction_correct": replicate_direction,
            "paired_request_count": request_count,
            "pass": passed,
        }
    return {
        "schema": GATE_SCHEMA,
        "load_fraction": float(load_fraction),
        "minimum_gain": float(minimum_gain),
        "calibration_replicates": CALIBRATION_REPLICATES,
        "phase_results": phase_results,
        "workload_valid_for_controller_tuning": all_pass,
    }


def choose_first_valid_fraction(
    observations: Iterable[FixedArmObservation],
) -> tuple[float | None, tuple[dict[str, object], ...]]:
    """Evaluate the fixed ladder in order and stop at the first valid level."""

    rows = tuple(observations)
    if any(not isinstance(item, FixedArmObservation) for item in rows):
        raise TypeError("observations must contain FixedArmObservation")
    reports = []
    for fraction in CALIBRATION_FRACTIONS:
        candidates = [item for item in rows if item.load_fraction == fraction]
        if not candidates:
            break
        report = evaluate_crossover(candidates, load_fraction=fraction)
        reports.append(report)
        if report["workload_valid_for_controller_tuning"] is True:
            return fraction, tuple(reports)
    return None, tuple(reports)


def default_preregistration() -> dict[str, object]:
    """JSON-compatible workload decisions frozen before node calibration."""

    return {
        "schema": SCHEMA,
        "phase_changing_trace": [
            state.value for state in (
                ContentionState.C0,
                ContentionState.C1,
                ContentionState.C2,
                ContentionState.C2_KV,
                ContentionState.C3,
                ContentionState.RECOVERY,
            )
        ],
        "phase_duration_ms": PHASE_DURATION_MS,
        "calibration_fractions": list(CALIBRATION_FRACTIONS),
        "calibration_replicates": CALIBRATION_REPLICATES,
        "selection_rule": "first_level_with_both_opposite_crossovers",
        "crossover_min_gain": CROSSOVER_MIN_GAIN,
        "calibration_traffic_shape": TrafficShape.STABLE.value,
        "validation_traffic_shapes": [shape.value for shape in TrafficShape],
        "overload_multiplier": OVERLOAD_MULTIPLIER,
        "decoder_hot": {
            "route": ForegroundArm.LOCAL.value,
            "prompt_tokens": DECODER_HOT_GEOMETRY.prompt_tokens,
            "output_tokens": DECODER_HOT_GEOMETRY.output_tokens,
            "cache_state": DECODER_HOT_GEOMETRY.cache_state.value,
            "pressure_scope": "decoder_local_prefill_engine",
            "legacy_tenant_label": Tenant.DECODER_HOT.value,
        },
        "remote_hot": {
            "route": ForegroundArm.REMOTE.value,
            "prompt_tokens": REMOTE_HOT_GEOMETRY.prompt_tokens,
            "output_tokens": REMOTE_HOT_GEOMETRY.output_tokens,
            "cache_state": REMOTE_HOT_GEOMETRY.cache_state.value,
            "path": "actual_prefill_plus_official_lmcache_transfer_and_install",
        },
        "kv_remote_hot": {
            "route": ForegroundArm.REMOTE.value,
            "prompt_tokens": KV_REMOTE_HOT_GEOMETRY.prompt_tokens,
            "output_tokens": KV_REMOTE_HOT_GEOMETRY.output_tokens,
            "cache_state": KV_REMOTE_HOT_GEOMETRY.cache_state.value,
            "offered_rate_per_s": KV_REMOTE_RATE_PER_S,
            "path": "preseeded_p_only_official_lmcache_retrieval_transfer_install",
            "long_producer_prefill_removed": True,
            "zero_producer_compute_claim_allowed": False,
        },
        "foreground_geometries": [
            {
                "prompt_tokens": item.prompt_tokens,
                "output_tokens": item.output_tokens,
                "cache_state": item.cache_state.value,
            }
            for item in FOREGROUND_GEOMETRIES
        ],
        "crossover_foreground_geometries": [
            {
                "prompt_tokens": item.prompt_tokens,
                "output_tokens": item.output_tokens,
                "cache_state": item.cache_state.value,
            }
            for item in CROSSOVER_FOREGROUND_GEOMETRIES
        ],
        "arm_order": ["local", "remote", "remote", "local"],
        "schedule_identity": "semantic_hash_excludes_arm_and_trial_markers",
        "controller_tuning_before_crossover": False,
        "v1_shared_decode_negative": {
            "geometry": {
                "prompt_tokens": 512,
                "output_tokens": 256,
            },
            "fractions_completed": list(CALIBRATION_FRACTIONS),
            "c1_crossover_observed": False,
            "reason_for_v2": (
                "pure decode work is downstream of both routes and cannot be "
                "escaped by local-versus-remote prefill admission"
            ),
        },
        "v2_capacity_normalization_negative": {
            "local_reference_rate_per_s": 8,
            "remote_reference_rate_per_s": 16,
            "local_background_median_p99_ms_at_full": [149.810429, 239.978661],
            "remote_background_median_p99_ms_at_full": [4301.383079, 6215.054621],
            "remote_background_median_p99_ms_at_8_per_s": [366.340205, 707.661529],
            "v3_reference_rates_per_s": {"local": 16, "remote": 8},
            "reason_for_v3": (
                "v2 compared different load regimes and mixed foreground "
                "decode work that is shared downstream of both routes"
            ),
        },
        "v3_capacity_bracket": {
            "local_total_rate_per_s_at_full": 18,
            "local_background_median_p99_ms_at_full": [134.724935, 281.960982],
            "local_block_seconds_at_full": 15.124918372,
            "local_capacity_knee_observed": False,
            "remote_rate_per_s_with_exact_failure": 8,
            "remote_failure": "one_actual_http_502_fail_closed",
            "highest_validated_remote_rate_per_s": 6.8,
            "v4_reference_rates_per_s": {"local": 32, "remote": 6.8},
        },
    }


__all__ = [
    "CALIBRATION_FRACTIONS",
    "CALIBRATION_REPLICATES",
    "CROSSOVER_MIN_GAIN",
    "CacheState",
    "CROSSOVER_FOREGROUND_GEOMETRIES",
    "ContentionState",
    "FixedArmObservation",
    "ForegroundArm",
    "ForegroundObservation",
    "KV_REMOTE_HOT_GEOMETRY",
    "KV_REMOTE_RATE_PER_S",
    "LoadSelection",
    "ScheduledRequest",
    "Tenant",
    "TokenGeometry",
    "TrafficShape",
    "VALIDATION_FOREGROUND_GEOMETRIES",
    "build_schedule",
    "choose_first_valid_fraction",
    "default_preregistration",
    "evaluate_crossover",
    "semantic_schedule_sha256",
]
