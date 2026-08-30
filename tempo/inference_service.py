"""Deterministic service-aware inference admission compiler.

This is the compact v2 model used to screen epoch calendars before a GPU run.
Token timestamps are absolute baseline times from the epoch origin.  A width's
predicted foreground penalty shifts later token timestamps, while transfers
issued at the same token start at that token's already-shifted timestamp.

Transfer quanta are assigned in their canonical input order.  Different lanes
can run in parallel; quanta on one lane run serially.  Feasibility is therefore
based on simulated service completion and start lag, rather than token capacity
alone.  The runtime still consumes only an immutable local calendar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping


_SCHEMA = "tempo-inference-service-plan-2"


def _strict_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def _exact_fields(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} fields are not exact")


@dataclass(frozen=True)
class ServiceQuantum:
    """One exact transfer unit on a serial service lane."""

    lane: int
    bytes: int
    service_ns: int

    def __post_init__(self) -> None:
        _strict_int("lane", self.lane)
        _strict_int("bytes", self.bytes, minimum=1)
        _strict_int("service_ns", self.service_ns, minimum=1)

    def to_dict(self) -> dict[str, int]:
        return {
            "lane": self.lane,
            "bytes": self.bytes,
            "service_ns": self.service_ns,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServiceQuantum":
        _exact_fields(payload, {"lane", "bytes", "service_ns"}, "service quantum")
        return cls(
            lane=payload["lane"],
            bytes=payload["bytes"],
            service_ns=payload["service_ns"],
        )


@dataclass(frozen=True)
class ServiceProfile:
    """Frozen calibration and workload inputs for one inference epoch.

    ``width_penalties_ns[w]`` is the foreground penalty for admitting ``w``
    canonical quanta at one token.  The curve is exact for every width from
    zero through ``max_width``.  ``deadline_ns`` and ``start_lag_cap_ns`` are
    hard constraints, not objective hints.
    """

    token_base_times_ns: tuple[int, ...]
    width_penalties_ns: tuple[int, ...]
    deadline_ns: int
    start_lag_cap_ns: int
    max_width: int
    protect_prefix_tokens: int
    protect_prefix_max_width: int
    quanta: tuple[ServiceQuantum, ...]

    def __post_init__(self) -> None:
        if not self.token_base_times_ns:
            raise ValueError("token_base_times_ns must not be empty")
        previous = -1
        for index, timestamp in enumerate(self.token_base_times_ns):
            _strict_int(f"token_base_times_ns[{index}]", timestamp)
            if timestamp <= previous:
                raise ValueError("token base times must be strictly increasing")
            previous = timestamp

        _strict_int("deadline_ns", self.deadline_ns, minimum=1)
        _strict_int("start_lag_cap_ns", self.start_lag_cap_ns)
        _strict_int("max_width", self.max_width, minimum=1)
        _strict_int("protect_prefix_tokens", self.protect_prefix_tokens)
        _strict_int(
            "protect_prefix_max_width", self.protect_prefix_max_width
        )
        if self.protect_prefix_tokens > len(self.token_base_times_ns):
            raise ValueError("protect prefix exceeds the token horizon")
        if self.protect_prefix_max_width > self.max_width:
            raise ValueError("protect prefix width exceeds max_width")
        if len(self.width_penalties_ns) != self.max_width + 1:
            raise ValueError("width penalties must cover zero through max_width")
        for width, penalty in enumerate(self.width_penalties_ns):
            _strict_int(f"width_penalties_ns[{width}]", penalty)
        if self.width_penalties_ns[0] != 0:
            raise ValueError("zero width must have zero penalty")
        if not self.quanta:
            raise ValueError("quanta must not be empty")
        for index, quantum in enumerate(self.quanta):
            if not isinstance(quantum, ServiceQuantum):
                raise ValueError(f"quanta[{index}] must be a ServiceQuantum")

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_base_times_ns": list(self.token_base_times_ns),
            "width_penalties_ns": list(self.width_penalties_ns),
            "deadline_ns": self.deadline_ns,
            "start_lag_cap_ns": self.start_lag_cap_ns,
            "max_width": self.max_width,
            "protect_prefix_tokens": self.protect_prefix_tokens,
            "protect_prefix_max_width": self.protect_prefix_max_width,
            "quanta": [quantum.to_dict() for quantum in self.quanta],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServiceProfile":
        expected = {
            "token_base_times_ns",
            "width_penalties_ns",
            "deadline_ns",
            "start_lag_cap_ns",
            "max_width",
            "protect_prefix_tokens",
            "protect_prefix_max_width",
            "quanta",
        }
        _exact_fields(payload, expected, "service profile")
        times = payload["token_base_times_ns"]
        penalties = payload["width_penalties_ns"]
        quanta = payload["quanta"]
        if not isinstance(times, list) or not isinstance(penalties, list):
            raise ValueError("profile times and width penalties must be lists")
        if not isinstance(quanta, list):
            raise ValueError("profile quanta must be a list")
        parsed_quanta: list[ServiceQuantum] = []
        for item in quanta:
            if not isinstance(item, dict):
                raise ValueError("each service quantum must be an object")
            parsed_quanta.append(ServiceQuantum.from_dict(item))
        return cls(
            token_base_times_ns=tuple(times),
            width_penalties_ns=tuple(penalties),
            deadline_ns=payload["deadline_ns"],
            start_lag_cap_ns=payload["start_lag_cap_ns"],
            max_width=payload["max_width"],
            protect_prefix_tokens=payload["protect_prefix_tokens"],
            protect_prefix_max_width=payload["protect_prefix_max_width"],
            quanta=tuple(parsed_quanta),
        )


@dataclass(frozen=True)
class ServicePlan:
    """Immutable local calendar produced by :func:`compile_service`."""

    feasible: bool
    reason: str
    width_by_token: tuple[int, ...]
    quantum_indices_by_token: tuple[tuple[int, ...], ...]
    predicted_completion_ns: int | None
    predicted_max_start_lag_ns: int
    total_predicted_penalty_ns: int
    peak_predicted_penalty_ns: int
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "width_by_token": list(self.width_by_token),
            "quantum_indices_by_token": [
                list(indices) for indices in self.quantum_indices_by_token
            ],
            "predicted_completion_ns": self.predicted_completion_ns,
            "predicted_max_start_lag_ns": self.predicted_max_start_lag_ns,
            "total_predicted_penalty_ns": self.total_predicted_penalty_ns,
            "peak_predicted_penalty_ns": self.peak_predicted_penalty_ns,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServicePlan":
        expected = {
            "feasible",
            "reason",
            "width_by_token",
            "quantum_indices_by_token",
            "predicted_completion_ns",
            "predicted_max_start_lag_ns",
            "total_predicted_penalty_ns",
            "peak_predicted_penalty_ns",
            "signature",
        }
        _exact_fields(payload, expected, "service plan")
        widths = payload["width_by_token"]
        assignments = payload["quantum_indices_by_token"]
        if not isinstance(widths, list) or not isinstance(assignments, list):
            raise ValueError("plan widths and assignments must be lists")
        if not all(isinstance(item, list) for item in assignments):
            raise ValueError("each token assignment must be a list")
        return cls(
            feasible=payload["feasible"],
            reason=payload["reason"],
            width_by_token=tuple(widths),
            quantum_indices_by_token=tuple(tuple(item) for item in assignments),
            predicted_completion_ns=payload["predicted_completion_ns"],
            predicted_max_start_lag_ns=payload["predicted_max_start_lag_ns"],
            total_predicted_penalty_ns=payload["total_predicted_penalty_ns"],
            peak_predicted_penalty_ns=payload["peak_predicted_penalty_ns"],
            signature=payload["signature"],
        )


def _signature_payload(profile: ServiceProfile, plan: ServicePlan) -> dict[str, Any]:
    plan_payload = plan.to_dict()
    plan_payload.pop("signature")
    return {
        "schema_version": _SCHEMA,
        "profile": profile.to_dict(),
        "plan": plan_payload,
    }


def _signature(profile: ServiceProfile, plan: ServicePlan) -> str:
    encoded = json.dumps(
        _signature_payload(profile, plan),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assignments(widths: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    cursor = 0
    result: list[tuple[int, ...]] = []
    for width in widths:
        result.append(tuple(range(cursor, cursor + width)))
        cursor += width
    return tuple(result)


def _make_plan(
    profile: ServiceProfile,
    *,
    feasible: bool,
    reason: str,
    widths: tuple[int, ...],
    completion_ns: int | None,
    max_lag_ns: int,
    total_penalty_ns: int,
    peak_penalty_ns: int,
) -> ServicePlan:
    unsigned = ServicePlan(
        feasible=feasible,
        reason=reason,
        width_by_token=widths,
        quantum_indices_by_token=_assignments(widths),
        predicted_completion_ns=completion_ns,
        predicted_max_start_lag_ns=max_lag_ns,
        total_predicted_penalty_ns=total_penalty_ns,
        peak_predicted_penalty_ns=peak_penalty_ns,
        signature="",
    )
    return replace(unsigned, signature=_signature(profile, unsigned))


@dataclass(frozen=True)
class _State:
    cursor: int
    lane_ready_ns: tuple[int, ...]
    total_penalty_ns: int
    peak_penalty_ns: int
    max_lag_ns: int


def compile_service(profile: ServiceProfile) -> ServicePlan:
    """Compile the globally preferred feasible service calendar.

    The exact objective is total penalty, peak penalty, actual service
    completion, maximum start lag, then lexicographically stable widths.  In
    particular, compilation does not stop at the first or earliest issue
    schedule when a later schedule has lower foreground cost.
    """

    lanes = tuple(sorted({quantum.lane for quantum in profile.quanta}))
    lane_position = {lane: index for index, lane in enumerate(lanes)}
    initial = _State(0, (0,) * len(lanes), 0, 0, 0)
    # Identical simulator states have identical futures; retain the stable
    # lexicographically smallest prefix only.
    states: dict[_State, tuple[int, ...]] = {initial: ()}

    for token, base_time_ns in enumerate(profile.token_base_times_ns):
        token_cap = (
            profile.protect_prefix_max_width
            if token < profile.protect_prefix_tokens
            else profile.max_width
        )
        next_states: dict[_State, tuple[int, ...]] = {}
        for state, prefix in states.items():
            remaining = len(profile.quanta) - state.cursor
            for width in range(min(token_cap, remaining) + 1):
                penalty_ns = profile.width_penalties_ns[width]
                issue_ns = base_time_ns + state.total_penalty_ns
                ready = list(state.lane_ready_ns)
                max_lag_ns = state.max_lag_ns
                valid = True
                for quantum_index in range(state.cursor, state.cursor + width):
                    quantum = profile.quanta[quantum_index]
                    position = lane_position[quantum.lane]
                    start_ns = max(issue_ns, ready[position])
                    lag_ns = start_ns - issue_ns
                    finish_ns = start_ns + quantum.service_ns
                    if (
                        lag_ns > profile.start_lag_cap_ns
                        or finish_ns > profile.deadline_ns
                    ):
                        valid = False
                        break
                    ready[position] = finish_ns
                    max_lag_ns = max(max_lag_ns, lag_ns)
                if not valid:
                    continue
                candidate = _State(
                    cursor=state.cursor + width,
                    lane_ready_ns=tuple(ready),
                    total_penalty_ns=state.total_penalty_ns + penalty_ns,
                    peak_penalty_ns=max(state.peak_penalty_ns, penalty_ns),
                    max_lag_ns=max_lag_ns,
                )
                candidate_widths = prefix + (width,)
                incumbent = next_states.get(candidate)
                if incumbent is None or candidate_widths < incumbent:
                    next_states[candidate] = candidate_widths
        states = next_states
        if not states:
            break

    candidates: list[tuple[tuple[Any, ...], _State, tuple[int, ...]]] = []
    for state, widths in states.items():
        if state.cursor != len(profile.quanta):
            continue
        completion_ns = max(state.lane_ready_ns)
        objective = (
            state.total_penalty_ns,
            state.peak_penalty_ns,
            completion_ns,
            state.max_lag_ns,
            widths,
        )
        candidates.append((objective, state, widths))

    if not candidates:
        return _make_plan(
            profile,
            feasible=False,
            reason="deadline_service_shortfall",
            widths=(0,) * len(profile.token_base_times_ns),
            completion_ns=None,
            max_lag_ns=0,
            total_penalty_ns=0,
            peak_penalty_ns=0,
        )

    _, winner, widths = min(candidates, key=lambda item: item[0])
    plan = _make_plan(
        profile,
        feasible=True,
        reason="compiled",
        widths=widths,
        completion_ns=max(winner.lane_ready_ns),
        max_lag_ns=winner.max_lag_ns,
        total_penalty_ns=winner.total_penalty_ns,
        peak_penalty_ns=winner.peak_penalty_ns,
    )
    validate_service_plan(profile, plan)
    return plan


def validate_service_plan(profile: ServiceProfile, plan: ServicePlan) -> None:
    """Strictly replay a plan and reject stale or tampered payloads."""

    if not isinstance(plan.feasible, bool):
        raise ValueError("plan feasible must be bool")
    expected_reason = "compiled" if plan.feasible else "deadline_service_shortfall"
    if plan.reason != expected_reason:
        raise ValueError("plan reason is inconsistent")
    token_count = len(profile.token_base_times_ns)
    if len(plan.width_by_token) != token_count:
        raise ValueError("plan token count differs from profile")
    if len(plan.quantum_indices_by_token) != token_count:
        raise ValueError("plan assignments differ from token count")

    lanes = tuple(sorted({quantum.lane for quantum in profile.quanta}))
    ready = {lane: 0 for lane in lanes}
    cursor = 0
    total_penalty_ns = 0
    peak_penalty_ns = 0
    max_lag_ns = 0
    completion_ns = 0
    for token, (width, indices) in enumerate(
        zip(plan.width_by_token, plan.quantum_indices_by_token, strict=True)
    ):
        _strict_int(f"width_by_token[{token}]", width)
        if width > profile.max_width:
            raise ValueError("plan exceeds max_width")
        if (
            token < profile.protect_prefix_tokens
            and width > profile.protect_prefix_max_width
        ):
            raise ValueError("plan violates the protected prefix")
        if len(indices) != width:
            raise ValueError("assignment count differs from width")
        expected_indices = tuple(range(cursor, cursor + width))
        if indices != expected_indices:
            raise ValueError("plan assignments are not canonical")
        if cursor + width > len(profile.quanta):
            raise ValueError("plan assigns nonexistent quanta")

        issue_ns = profile.token_base_times_ns[token] + total_penalty_ns
        for quantum_index in indices:
            _strict_int("quantum index", quantum_index)
            quantum = profile.quanta[quantum_index]
            start_ns = max(issue_ns, ready[quantum.lane])
            lag_ns = start_ns - issue_ns
            finish_ns = start_ns + quantum.service_ns
            if lag_ns > profile.start_lag_cap_ns:
                raise ValueError("plan exceeds the start-lag cap")
            if finish_ns > profile.deadline_ns:
                raise ValueError("plan completes after the deadline")
            ready[quantum.lane] = finish_ns
            max_lag_ns = max(max_lag_ns, lag_ns)
            completion_ns = max(completion_ns, finish_ns)
        cursor += width
        penalty_ns = profile.width_penalties_ns[width]
        total_penalty_ns += penalty_ns
        peak_penalty_ns = max(peak_penalty_ns, penalty_ns)

    if plan.feasible:
        if cursor != len(profile.quanta):
            raise ValueError("feasible plan must assign every quantum exactly once")
        if plan.predicted_completion_ns != completion_ns:
            raise ValueError("predicted completion is inconsistent")
    else:
        if cursor or any(plan.width_by_token):
            raise ValueError("infeasible plan must not issue transfers")
        if plan.predicted_completion_ns is not None:
            raise ValueError("infeasible plan must not predict completion")
    _strict_int("predicted_max_start_lag_ns", plan.predicted_max_start_lag_ns)
    _strict_int("total_predicted_penalty_ns", plan.total_predicted_penalty_ns)
    _strict_int("peak_predicted_penalty_ns", plan.peak_predicted_penalty_ns)
    if plan.predicted_max_start_lag_ns != max_lag_ns:
        raise ValueError("predicted max start lag is inconsistent")
    if plan.total_predicted_penalty_ns != total_penalty_ns:
        raise ValueError("total predicted penalty is inconsistent")
    if plan.peak_predicted_penalty_ns != peak_penalty_ns:
        raise ValueError("peak predicted penalty is inconsistent")
    if plan.signature != _signature(profile, plan):
        raise ValueError("service plan signature mismatch")


def make_service_artifact(
    profile: ServiceProfile, plan: ServicePlan
) -> dict[str, Any]:
    validate_service_plan(profile, plan)
    return {
        "schema_version": _SCHEMA,
        "evidence_state": "offline_compiled_candidate",
        "hot_path_global_control": False,
        "adaptive": False,
        "service_model": "deterministic_lane_serial",
        "profile": profile.to_dict(),
        "plan": plan.to_dict(),
    }


def load_service_artifact(
    payload: Mapping[str, Any],
) -> tuple[ServiceProfile, ServicePlan]:
    expected = {
        "schema_version",
        "evidence_state",
        "hot_path_global_control",
        "adaptive",
        "service_model",
        "profile",
        "plan",
    }
    _exact_fields(payload, expected, "service artifact")
    if payload["schema_version"] != _SCHEMA:
        raise ValueError("unsupported service artifact schema")
    if payload["evidence_state"] != "offline_compiled_candidate":
        raise ValueError("unsupported service artifact evidence state")
    if payload["hot_path_global_control"] is not False:
        raise ValueError("service artifact must use local hot-path control")
    if payload["adaptive"] is not False:
        raise ValueError("service artifact must be non-adaptive")
    if payload["service_model"] != "deterministic_lane_serial":
        raise ValueError("unsupported service model")
    if not isinstance(payload["profile"], dict) or not isinstance(
        payload["plan"], dict
    ):
        raise ValueError("service artifact profile and plan must be objects")
    profile = ServiceProfile.from_dict(payload["profile"])
    plan = ServicePlan.from_dict(payload["plan"])
    validate_service_plan(profile, plan)
    return profile, plan
