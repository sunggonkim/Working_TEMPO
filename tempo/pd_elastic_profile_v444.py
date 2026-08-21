"""Strict calibration contract for the TEMPO Elastic-PD ingress policy.

The profile is deliberately small and exact.  It binds latency estimates and
credit weights to one model revision, topology, backend, classifier, and exact
prompt/output geometry.  Missing or mismatched evidence disables the remote
route instead of extrapolating.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from tempo.pd_elastic_controller_v443 import ElasticConfig, ElasticEstimate


SCHEMA = "tempo-elastic-pd-profile-444"
_TOP_KEYS = {"schema", "profile_id", "deployment_scope", "identity", "controller", "rows"}
_IDENTITY_KEYS = {
    "model_id", "model_revision", "topology_id", "remote_backend",
    "classifier_version", "kv_bytes_per_token",
}
_CONTROLLER_KEYS = {
    "local_compute_budget_us", "remote_kv_budget_bytes", "arrival_window",
    "enter_high_gap_ns", "exit_high_gap_ns", "exit_consecutive_windows",
    "route_margin_ms", "spill_regression_budget_ms",
}
_ROW_KEYS = {
    "prompt_tokens", "output_tokens", "local_upper_bound_ms",
    "remote_upper_bound_ms", "uncertainty_ms", "local_tbt_safe",
    "remote_evidence_valid", "local_compute_cost_us", "remote_kv_bytes",
    "samples_local", "samples_remote", "outputs_equivalent",
    "remote_transfer_failures",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} keys must be exact")


def _positive_int(name: str, value: Any) -> int:
    _require(type(value) is int and value > 0, f"{name} must be a positive int")
    return value


def _finite_nonnegative(name: str, value: Any) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0,
        f"{name} must be finite and nonnegative",
    )
    return float(value)


@dataclass(frozen=True)
class ElasticProfileIdentity:
    model_id: str
    model_revision: str
    topology_id: str
    remote_backend: str
    classifier_version: str
    kv_bytes_per_token: int


@dataclass(frozen=True)
class ElasticProfileRow:
    prompt_tokens: int
    output_tokens: int
    local_upper_bound_ms: float
    remote_upper_bound_ms: float
    uncertainty_ms: float
    local_tbt_safe: bool
    remote_evidence_valid: bool
    local_compute_cost_us: int
    remote_kv_bytes: int
    samples_local: int
    samples_remote: int
    outputs_equivalent: bool
    remote_transfer_failures: int

    @property
    def key(self) -> tuple[int, int]:
        return self.prompt_tokens, self.output_tokens

    @property
    def evidence_safe(self) -> bool:
        return (
            self.remote_evidence_valid
            and self.outputs_equivalent
            and self.remote_transfer_failures == 0
            and self.samples_local > 0
            and self.samples_remote > 0
        )

    def estimate(self, remaining_deadline_ms: float) -> ElasticEstimate:
        return ElasticEstimate(
            local_upper_bound_ms=self.local_upper_bound_ms,
            remote_upper_bound_ms=self.remote_upper_bound_ms,
            uncertainty_ms=self.uncertainty_ms,
            remaining_deadline_ms=remaining_deadline_ms,
            local_tbt_safe=self.local_tbt_safe,
            remote_backend_available=True,
            remote_evidence_valid=self.evidence_safe,
        )


@dataclass(frozen=True)
class ElasticPDProfile:
    profile_id: str
    deployment_scope: str
    identity: ElasticProfileIdentity
    controller: ElasticConfig
    rows: tuple[ElasticProfileRow, ...]
    fingerprint_sha256: str

    def __post_init__(self) -> None:
        _require(self.deployment_scope in {"screen_only", "replicated"},
                 "deployment_scope must be screen_only or replicated")
        keys = [row.key for row in self.rows]
        _require(bool(keys) and len(keys) == len(set(keys)),
                 "profile rows must be nonempty and unique")
        for row in self.rows:
            expected = row.prompt_tokens * self.identity.kv_bytes_per_token
            _require(row.remote_kv_bytes == expected,
                     "row remote_kv_bytes must match exact prompt geometry")

    def exact_row(self, prompt_tokens: int, output_tokens: int) -> ElasticProfileRow | None:
        for row in self.rows:
            if row.key == (prompt_tokens, output_tokens):
                return row
        return None

    def validate_identity(
        self, *, model_id: str, model_revision: str, topology_id: str,
        remote_backend: str, classifier_version: str, kv_bytes_per_token: int,
    ) -> None:
        observed = ElasticProfileIdentity(
            model_id=model_id,
            model_revision=model_revision,
            topology_id=topology_id,
            remote_backend=remote_backend,
            classifier_version=classifier_version,
            kv_bytes_per_token=kv_bytes_per_token,
        )
        _require(observed == self.identity, "elastic profile identity mismatch")


def _parse_row(payload: Any, index: int) -> ElasticProfileRow:
    _require(isinstance(payload, dict), f"rows[{index}] must be an object")
    _exact_keys(payload, _ROW_KEYS, f"rows[{index}]")
    bool_names = ("local_tbt_safe", "remote_evidence_valid", "outputs_equivalent")
    for name in bool_names:
        _require(type(payload[name]) is bool, f"rows[{index}].{name} must be bool")
    failures = payload["remote_transfer_failures"]
    _require(type(failures) is int and failures >= 0,
             f"rows[{index}].remote_transfer_failures must be nonnegative")
    return ElasticProfileRow(
        prompt_tokens=_positive_int(f"rows[{index}].prompt_tokens", payload["prompt_tokens"]),
        output_tokens=_positive_int(f"rows[{index}].output_tokens", payload["output_tokens"]),
        local_upper_bound_ms=_finite_nonnegative(
            f"rows[{index}].local_upper_bound_ms", payload["local_upper_bound_ms"]),
        remote_upper_bound_ms=_finite_nonnegative(
            f"rows[{index}].remote_upper_bound_ms", payload["remote_upper_bound_ms"]),
        uncertainty_ms=_finite_nonnegative(
            f"rows[{index}].uncertainty_ms", payload["uncertainty_ms"]),
        local_tbt_safe=payload["local_tbt_safe"],
        remote_evidence_valid=payload["remote_evidence_valid"],
        local_compute_cost_us=_positive_int(
            f"rows[{index}].local_compute_cost_us", payload["local_compute_cost_us"]),
        remote_kv_bytes=_positive_int(
            f"rows[{index}].remote_kv_bytes", payload["remote_kv_bytes"]),
        samples_local=_positive_int(f"rows[{index}].samples_local", payload["samples_local"]),
        samples_remote=_positive_int(f"rows[{index}].samples_remote", payload["samples_remote"]),
        outputs_equivalent=payload["outputs_equivalent"],
        remote_transfer_failures=failures,
    )


def load_elastic_profile(path: Path) -> ElasticPDProfile:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path")
    raw = path.read_bytes()
    payload = json.loads(raw)
    _require(isinstance(payload, dict), "profile must be an object")
    _exact_keys(payload, _TOP_KEYS, "profile")
    _require(payload["schema"] == SCHEMA, "unsupported elastic profile schema")
    _require(isinstance(payload["profile_id"], str) and payload["profile_id"].strip(),
             "profile_id must be nonempty")
    _require(isinstance(payload["deployment_scope"], str),
             "deployment_scope must be a string")

    identity = payload["identity"]
    controller = payload["controller"]
    rows = payload["rows"]
    _require(isinstance(identity, dict), "identity must be an object")
    _require(isinstance(controller, dict), "controller must be an object")
    _require(isinstance(rows, list), "rows must be a list")
    _exact_keys(identity, _IDENTITY_KEYS, "identity")
    _exact_keys(controller, _CONTROLLER_KEYS, "controller")
    for name in (
        "model_id", "model_revision", "topology_id", "remote_backend",
        "classifier_version",
    ):
        _require(isinstance(identity[name], str) and identity[name].strip(),
                 f"identity.{name} must be nonempty")

    parsed_identity = ElasticProfileIdentity(
        model_id=identity["model_id"],
        model_revision=identity["model_revision"],
        topology_id=identity["topology_id"],
        remote_backend=identity["remote_backend"],
        classifier_version=identity["classifier_version"],
        kv_bytes_per_token=_positive_int(
            "identity.kv_bytes_per_token", identity["kv_bytes_per_token"]),
    )
    parsed_controller = ElasticConfig(**controller)
    parsed_rows = tuple(_parse_row(row, index) for index, row in enumerate(rows))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return ElasticPDProfile(
        profile_id=payload["profile_id"],
        deployment_scope=payload["deployment_scope"],
        identity=parsed_identity,
        controller=parsed_controller,
        rows=parsed_rows,
        fingerprint_sha256=hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "ElasticPDProfile", "ElasticProfileIdentity", "ElasticProfileRow",
    "SCHEMA", "load_elastic_profile",
]
