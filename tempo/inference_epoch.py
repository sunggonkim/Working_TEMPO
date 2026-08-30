"""Compile deterministic epoch-level admission calendars for inference traffic.

The compiler is deliberately transport agnostic.  A calibration step supplies
the measured foreground penalty for each supported concurrent transfer width,
and the runtime consumes only the resulting immutable per-token calendar.  No
global decision or feedback exchange is required in the token hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


def _strict_int(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}")
    return value


@dataclass(frozen=True, order=True)
class WidthPoint:
    """One calibrated concurrency point.

    ``width`` is the number of canonical transfer quanta issued at a token and
    ``penalty_ns`` is the conservative measured foreground-latency increment.
    """

    width: int
    penalty_ns: int

    def __post_init__(self) -> None:
        _strict_int("width", self.width)
        _strict_int("penalty_ns", self.penalty_ns)
        if self.width == 0 and self.penalty_ns != 0:
            raise ValueError("zero width must have zero penalty")

    def to_dict(self) -> dict[str, int]:
        return {"width": self.width, "penalty_ns": self.penalty_ns}


@dataclass(frozen=True)
class EpochProfile:
    """Inputs frozen before an experiment epoch.

    ``deadline_tokens`` is exclusive.  Every transfer quantum must be assigned
    before that token boundary.  ``token_slack_ns`` is a hard per-token
    interference budget, not an objective hint.
    """

    total_quanta: int
    deadline_tokens: int
    token_slack_ns: tuple[int, ...]
    width_points: tuple[WidthPoint, ...]
    max_width: int
    protect_prefix_tokens: int = 0
    protect_prefix_max_width: int = 1

    def __post_init__(self) -> None:
        _strict_int("total_quanta", self.total_quanta, minimum=1)
        _strict_int("deadline_tokens", self.deadline_tokens, minimum=1)
        _strict_int("max_width", self.max_width, minimum=1)
        _strict_int("protect_prefix_tokens", self.protect_prefix_tokens)
        _strict_int(
            "protect_prefix_max_width", self.protect_prefix_max_width
        )
        if self.deadline_tokens > len(self.token_slack_ns):
            raise ValueError("deadline_tokens exceeds token_slack_ns")
        if self.protect_prefix_tokens > self.deadline_tokens:
            raise ValueError("protect prefix exceeds the deadline horizon")
        if self.protect_prefix_max_width > self.max_width:
            raise ValueError("protect prefix width exceeds max_width")
        for index, slack in enumerate(self.token_slack_ns):
            _strict_int(f"token_slack_ns[{index}]", slack)
        if not self.width_points:
            raise ValueError("width_points must not be empty")
        widths = tuple(point.width for point in self.width_points)
        if widths != tuple(sorted(widths)) or len(set(widths)) != len(widths):
            raise ValueError("width_points must have unique increasing widths")
        if self.width_points[0] != WidthPoint(0, 0):
            raise ValueError("width_points must start with WidthPoint(0, 0)")
        if not any(point.width == self.max_width for point in self.width_points):
            raise ValueError("max_width must be a calibrated width point")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_quanta": self.total_quanta,
            "deadline_tokens": self.deadline_tokens,
            "token_slack_ns": list(self.token_slack_ns),
            "width_points": [point.to_dict() for point in self.width_points],
            "max_width": self.max_width,
            "protect_prefix_tokens": self.protect_prefix_tokens,
            "protect_prefix_max_width": self.protect_prefix_max_width,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpochProfile":
        expected = {
            "total_quanta",
            "deadline_tokens",
            "token_slack_ns",
            "width_points",
            "max_width",
            "protect_prefix_tokens",
            "protect_prefix_max_width",
        }
        if set(payload) != expected:
            raise ValueError("epoch profile fields are not exact")
        slack = payload["token_slack_ns"]
        points = payload["width_points"]
        if not isinstance(slack, list) or not isinstance(points, list):
            raise ValueError("profile slack and width_points must be lists")
        return cls(
            total_quanta=payload["total_quanta"],
            deadline_tokens=payload["deadline_tokens"],
            token_slack_ns=tuple(slack),
            width_points=tuple(
                WidthPoint(point["width"], point["penalty_ns"])
                for point in points
                if isinstance(point, dict) and set(point) == {"width", "penalty_ns"}
            ),
            max_width=payload["max_width"],
            protect_prefix_tokens=payload["protect_prefix_tokens"],
            protect_prefix_max_width=payload["protect_prefix_max_width"],
        )


@dataclass(frozen=True)
class EpochPlan:
    """Immutable local-execution calendar produced by :func:`compile_epoch`."""

    feasible: bool
    reason: str
    width_by_token: tuple[int, ...]
    quantum_indices_by_token: tuple[tuple[int, ...], ...]
    completion_token_exclusive: int | None
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
            "completion_token_exclusive": self.completion_token_exclusive,
            "total_predicted_penalty_ns": self.total_predicted_penalty_ns,
            "peak_predicted_penalty_ns": self.peak_predicted_penalty_ns,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EpochPlan":
        expected = {
            "feasible",
            "reason",
            "width_by_token",
            "quantum_indices_by_token",
            "completion_token_exclusive",
            "total_predicted_penalty_ns",
            "peak_predicted_penalty_ns",
            "signature",
        }
        if set(payload) != expected:
            raise ValueError("epoch plan fields are not exact")
        widths = payload["width_by_token"]
        assignments = payload["quantum_indices_by_token"]
        if not isinstance(widths, list) or not isinstance(assignments, list):
            raise ValueError("plan widths and assignments must be lists")
        return cls(
            feasible=payload["feasible"],
            reason=payload["reason"],
            width_by_token=tuple(widths),
            quantum_indices_by_token=tuple(tuple(item) for item in assignments),
            completion_token_exclusive=payload["completion_token_exclusive"],
            total_predicted_penalty_ns=payload["total_predicted_penalty_ns"],
            peak_predicted_penalty_ns=payload["peak_predicted_penalty_ns"],
            signature=payload["signature"],
        )


def _signature_payload(profile: EpochProfile, plan: EpochPlan) -> dict[str, Any]:
    plan_payload = plan.to_dict()
    plan_payload.pop("signature")
    return {"profile": profile.to_dict(), "plan": plan_payload}


def _signature(profile: EpochProfile, plan: EpochPlan) -> str:
    encoded = json.dumps(
        _signature_payload(profile, plan),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_plan(
    profile: EpochProfile,
    *,
    feasible: bool,
    reason: str,
    widths: tuple[int, ...],
    completion: int | None,
    total_penalty: int,
    peak_penalty: int,
) -> EpochPlan:
    cursor = 0
    assignments: list[tuple[int, ...]] = []
    for width in widths:
        assignments.append(tuple(range(cursor, cursor + width)))
        cursor += width
    unsigned = EpochPlan(
        feasible=feasible,
        reason=reason,
        width_by_token=widths,
        quantum_indices_by_token=tuple(assignments),
        completion_token_exclusive=completion,
        total_predicted_penalty_ns=total_penalty,
        peak_predicted_penalty_ns=peak_penalty,
        signature="",
    )
    return EpochPlan(**{**unsigned.__dict__, "signature": _signature(profile, unsigned)})


def compile_epoch(profile: EpochProfile) -> EpochPlan:
    """Compile the earliest feasible, minimum-total-penalty calendar.

    The hard slack and prefix constraints protect foreground latency.  Among
    schedules that finish at the earliest token boundary, the compiler chooses
    minimum total predicted penalty, then minimum peak penalty, then a stable
    lexicographic width sequence.
    """

    penalty_by_width = {point.width: point.penalty_ns for point in profile.width_points}
    # moved -> (total penalty, peak penalty, width sequence)
    states: dict[int, tuple[int, int, tuple[int, ...]]] = {0: (0, 0, ())}
    for token in range(profile.deadline_tokens):
        slack = profile.token_slack_ns[token]
        token_cap = (
            profile.protect_prefix_max_width
            if token < profile.protect_prefix_tokens
            else profile.max_width
        )
        choices = tuple(
            point
            for point in profile.width_points
            if point.width <= token_cap and point.penalty_ns <= slack
        )
        next_states: dict[int, tuple[int, int, tuple[int, ...]]] = {}
        for moved, (total_penalty, peak_penalty, widths) in states.items():
            remaining = profile.total_quanta - moved
            for point in choices:
                if point.width > remaining:
                    continue
                candidate = (
                    total_penalty + point.penalty_ns,
                    max(peak_penalty, point.penalty_ns),
                    widths + (point.width,),
                )
                new_moved = moved + point.width
                incumbent = next_states.get(new_moved)
                if incumbent is None or candidate < incumbent:
                    next_states[new_moved] = candidate
        states = next_states
        completed = states.get(profile.total_quanta)
        if completed is not None:
            total_penalty, peak_penalty, prefix = completed
            widths = prefix + (0,) * (len(profile.token_slack_ns) - len(prefix))
            plan = _make_plan(
                profile,
                feasible=True,
                reason="compiled",
                widths=widths,
                completion=token + 1,
                total_penalty=total_penalty,
                peak_penalty=peak_penalty,
            )
            validate_epoch_plan(profile, plan)
            return plan
        if not states:
            break

    return _make_plan(
        profile,
        feasible=False,
        reason="deadline_capacity_shortfall",
        widths=(0,) * len(profile.token_slack_ns),
        completion=None,
        total_penalty=0,
        peak_penalty=0,
    )


def validate_epoch_plan(profile: EpochProfile, plan: EpochPlan) -> None:
    """Fail closed on a stale or tampered serialized plan."""

    if not isinstance(plan.feasible, bool):
        raise ValueError("plan feasible must be bool")
    if not isinstance(plan.reason, str) or not plan.reason:
        raise ValueError("plan reason must be a non-empty string")
    if len(plan.width_by_token) != len(profile.token_slack_ns):
        raise ValueError("plan token count differs from profile")
    if len(plan.quantum_indices_by_token) != len(plan.width_by_token):
        raise ValueError("plan assignments differ from width count")
    penalty_by_width = {point.width: point.penalty_ns for point in profile.width_points}
    flattened: list[int] = []
    total_penalty = 0
    peak_penalty = 0
    completion = None
    for token, (width, assignments) in enumerate(
        zip(plan.width_by_token, plan.quantum_indices_by_token, strict=True)
    ):
        _strict_int(f"width_by_token[{token}]", width)
        if width not in penalty_by_width or width > profile.max_width:
            raise ValueError("plan uses an uncalibrated width")
        if token >= profile.deadline_tokens and width:
            raise ValueError("plan issues transfers after its deadline")
        if token < profile.protect_prefix_tokens and width > profile.protect_prefix_max_width:
            raise ValueError("plan violates the protected prefix")
        penalty = penalty_by_width[width]
        if penalty > profile.token_slack_ns[token]:
            raise ValueError("plan exceeds token slack")
        if len(assignments) != width:
            raise ValueError("assignment count differs from width")
        for index in assignments:
            _strict_int("quantum index", index)
        flattened.extend(assignments)
        total_penalty += penalty
        peak_penalty = max(peak_penalty, penalty)
        if assignments:
            completion = token + 1
    if plan.feasible:
        if flattened != list(range(profile.total_quanta)):
            raise ValueError("plan must assign every canonical quantum exactly once")
        if completion != plan.completion_token_exclusive:
            raise ValueError("completion token is inconsistent")
    elif flattened or plan.completion_token_exclusive is not None:
        raise ValueError("infeasible plan must not issue transfers")
    if total_penalty != plan.total_predicted_penalty_ns:
        raise ValueError("total predicted penalty is inconsistent")
    if peak_penalty != plan.peak_predicted_penalty_ns:
        raise ValueError("peak predicted penalty is inconsistent")
    if plan.signature != _signature(profile, plan):
        raise ValueError("epoch plan signature mismatch")


def make_epoch_artifact(profile: EpochProfile, plan: EpochPlan) -> dict[str, Any]:
    validate_epoch_plan(profile, plan)
    return {
        "schema_version": "tempo-inference-epoch-plan-1",
        "evidence_state": "offline_compiled_candidate",
        "hot_path_global_control": False,
        "adaptive": False,
        "profile": profile.to_dict(),
        "plan": plan.to_dict(),
    }


def load_epoch_artifact(payload: Mapping[str, Any]) -> tuple[EpochProfile, EpochPlan]:
    expected = {
        "schema_version",
        "evidence_state",
        "hot_path_global_control",
        "adaptive",
        "profile",
        "plan",
    }
    if set(payload) != expected:
        raise ValueError("epoch artifact fields are not exact")
    if payload["schema_version"] != "tempo-inference-epoch-plan-1":
        raise ValueError("unsupported epoch artifact schema")
    if payload["hot_path_global_control"] is not False or payload["adaptive"] is not False:
        raise ValueError("epoch artifact semantics are not local AOT")
    if not isinstance(payload["profile"], dict) or not isinstance(payload["plan"], dict):
        raise ValueError("epoch artifact profile and plan must be objects")
    profile = EpochProfile.from_dict(payload["profile"])
    plan = EpochPlan.from_dict(payload["plan"])
    validate_epoch_plan(profile, plan)
    return profile, plan
