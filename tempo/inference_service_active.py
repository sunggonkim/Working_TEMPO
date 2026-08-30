"""Offline compiler for service-residence-aware inference calendars.

This research-prototype model charges each foreground token according to the
number of service lanes still outstanding *after* that token's new issues.
Consequently, a token with issue width zero still pays for transfers issued by
earlier tokens.  Token timestamps come from a frozen foreground-only trace;
accumulated predicted interference shifts later issue timestamps.

Service quanta retain canonical input order.  Lanes run in parallel and each
lane is serial.  Deadline and start-lag constraints are replayed against the
same absolute service timeline used during compilation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping

from tempo.inference_service import ServiceQuantum


_SCHEMA = "tempo-inference-active-service-plan-1"
_SERVICE_MODEL = "deterministic_lane_serial_active_interference"


def _strict_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


def _exact_fields(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{name} fields are not exact")


@dataclass(frozen=True)
class ActiveServiceProfile:
    """Frozen calibration and workload inputs for one inference epoch.

    ``active_lane_penalties_ns[n]`` is the foreground penalty at a token with
    ``n`` service lanes outstanding after new issues.  The tuple must cover
    zero through every distinct lane in ``quanta``.
    """

    token_base_times_ns: tuple[int, ...]
    active_lane_penalties_ns: tuple[int, ...]
    deadline_ns: int
    start_lag_cap_ns: int
    max_issue_width: int
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
        _strict_int("max_issue_width", self.max_issue_width, minimum=1)
        _strict_int("protect_prefix_tokens", self.protect_prefix_tokens)
        _strict_int(
            "protect_prefix_max_width", self.protect_prefix_max_width
        )
        if self.protect_prefix_tokens > len(self.token_base_times_ns):
            raise ValueError("protect prefix exceeds the token horizon")
        if self.protect_prefix_max_width > self.max_issue_width:
            raise ValueError("protect prefix width exceeds max_issue_width")
        if not self.quanta:
            raise ValueError("quanta must not be empty")
        for index, quantum in enumerate(self.quanta):
            if not isinstance(quantum, ServiceQuantum):
                raise ValueError(f"quanta[{index}] must be a ServiceQuantum")

        lane_count = len({quantum.lane for quantum in self.quanta})
        if len(self.active_lane_penalties_ns) != lane_count + 1:
            raise ValueError(
                "active lane penalties must cover zero through every lane"
            )
        for active_lanes, penalty in enumerate(self.active_lane_penalties_ns):
            _strict_int(f"active_lane_penalties_ns[{active_lanes}]", penalty)
        if self.active_lane_penalties_ns[0] != 0:
            raise ValueError("zero active lanes must have zero penalty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_base_times_ns": list(self.token_base_times_ns),
            "active_lane_penalties_ns": list(self.active_lane_penalties_ns),
            "deadline_ns": self.deadline_ns,
            "start_lag_cap_ns": self.start_lag_cap_ns,
            "max_issue_width": self.max_issue_width,
            "protect_prefix_tokens": self.protect_prefix_tokens,
            "protect_prefix_max_width": self.protect_prefix_max_width,
            "quanta": [quantum.to_dict() for quantum in self.quanta],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveServiceProfile":
        expected = {
            "token_base_times_ns",
            "active_lane_penalties_ns",
            "deadline_ns",
            "start_lag_cap_ns",
            "max_issue_width",
            "protect_prefix_tokens",
            "protect_prefix_max_width",
            "quanta",
        }
        _exact_fields(payload, expected, "active service profile")
        times = payload["token_base_times_ns"]
        penalties = payload["active_lane_penalties_ns"]
        quanta = payload["quanta"]
        if not isinstance(times, list) or not isinstance(penalties, list):
            raise ValueError("profile times and active lane penalties must be lists")
        if not isinstance(quanta, list):
            raise ValueError("profile quanta must be a list")
        parsed_quanta: list[ServiceQuantum] = []
        for item in quanta:
            if not isinstance(item, dict):
                raise ValueError("each service quantum must be an object")
            parsed_quanta.append(ServiceQuantum.from_dict(item))
        return cls(
            token_base_times_ns=tuple(times),
            active_lane_penalties_ns=tuple(penalties),
            deadline_ns=payload["deadline_ns"],
            start_lag_cap_ns=payload["start_lag_cap_ns"],
            max_issue_width=payload["max_issue_width"],
            protect_prefix_tokens=payload["protect_prefix_tokens"],
            protect_prefix_max_width=payload["protect_prefix_max_width"],
            quanta=tuple(parsed_quanta),
        )


@dataclass(frozen=True)
class ActiveServicePlan:
    """Immutable local calendar produced by :func:`compile_active_service`."""

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
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActiveServicePlan":
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
        _exact_fields(payload, expected, "active service plan")
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


def _signature_payload(
    profile: ActiveServiceProfile, plan: ActiveServicePlan
) -> dict[str, Any]:
    plan_payload = plan.to_dict()
    plan_payload.pop("signature")
    return {
        "schema_version": _SCHEMA,
        "profile": profile.to_dict(),
        "plan": plan_payload,
    }


def _signature(profile: ActiveServiceProfile, plan: ActiveServicePlan) -> str:
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
    profile: ActiveServiceProfile,
    *,
    feasible: bool,
    reason: str,
    widths: tuple[int, ...],
    completion_ns: int | None,
    max_lag_ns: int,
    total_penalty_ns: int,
    peak_penalty_ns: int,
) -> ActiveServicePlan:
    unsigned = ActiveServicePlan(
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


def compile_active_service(profile: ActiveServiceProfile) -> ActiveServicePlan:
    """Compile the lowest-interference feasible active-service calendar."""

    lanes = tuple(sorted({quantum.lane for quantum in profile.quanta}))
    lane_position = {lane: index for index, lane in enumerate(lanes)}
    initial = _State(0, (0,) * len(lanes), 0, 0, 0)
    states: dict[_State, tuple[int, ...]] = {initial: ()}

    for token, base_time_ns in enumerate(profile.token_base_times_ns):
        token_cap = (
            profile.protect_prefix_max_width
            if token < profile.protect_prefix_tokens
            else profile.max_issue_width
        )
        next_states: dict[_State, tuple[int, ...]] = {}
        for state, prefix in states.items():
            remaining = len(profile.quanta) - state.cursor
            for width in range(min(token_cap, remaining) + 1):
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

                active_lanes = sum(
                    lane_ready_ns > issue_ns for lane_ready_ns in ready
                )
                penalty_ns = profile.active_lane_penalties_ns[active_lanes]
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

    candidates: list[
        tuple[tuple[Any, ...], _State, tuple[int, ...]]
    ] = []
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
    validate_active_service_plan(profile, plan)
    return plan


def validate_active_service_plan(
    profile: ActiveServiceProfile, plan: ActiveServicePlan
) -> None:
    """Strictly replay active residence, service, and artifact identity."""

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
        if width > profile.max_issue_width:
            raise ValueError("plan exceeds max_issue_width")
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

        active_lanes = sum(
            lane_ready_ns > issue_ns for lane_ready_ns in ready.values()
        )
        penalty_ns = profile.active_lane_penalties_ns[active_lanes]
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
        raise ValueError("active service plan signature mismatch")


def make_active_service_artifact(
    profile: ActiveServiceProfile, plan: ActiveServicePlan
) -> dict[str, Any]:
    validate_active_service_plan(profile, plan)
    return {
        "schema_version": _SCHEMA,
        "evidence_state": "offline_compiled_candidate",
        "hot_path_global_control": False,
        "adaptive": False,
        "service_model": _SERVICE_MODEL,
        "profile": profile.to_dict(),
        "plan": plan.to_dict(),
    }


def load_active_service_artifact(
    payload: Mapping[str, Any],
) -> tuple[ActiveServiceProfile, ActiveServicePlan]:
    expected = {
        "schema_version",
        "evidence_state",
        "hot_path_global_control",
        "adaptive",
        "service_model",
        "profile",
        "plan",
    }
    _exact_fields(payload, expected, "active service artifact")
    if payload["schema_version"] != _SCHEMA:
        raise ValueError("unsupported active service artifact schema")
    if payload["evidence_state"] != "offline_compiled_candidate":
        raise ValueError("unsupported active service artifact evidence state")
    if payload["hot_path_global_control"] is not False:
        raise ValueError("active service artifact must use local hot-path control")
    if payload["adaptive"] is not False:
        raise ValueError("active service artifact must be non-adaptive")
    if payload["service_model"] != _SERVICE_MODEL:
        raise ValueError("unsupported active service model")
    if not isinstance(payload["profile"], dict) or not isinstance(
        payload["plan"], dict
    ):
        raise ValueError("active service profile and plan must be objects")
    profile = ActiveServiceProfile.from_dict(payload["profile"])
    plan = ActiveServicePlan.from_dict(payload["plan"])
    validate_active_service_plan(profile, plan)
    return profile, plan
