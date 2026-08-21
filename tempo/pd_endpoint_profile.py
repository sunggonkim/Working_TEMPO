"""Frozen endpoint-service profile for the TEMPO feedback controller."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from tempo.pd_elastic_controller_v443 import CacheResidency
from tempo.pd_endpoint_controller import EndpointAdmissionConfig, EndpointRoute


SCHEMA_V1 = "tempo-pd-endpoint-service-profile-v1"
SCHEMA_V2 = "tempo-pd-endpoint-service-profile-v2"
SCHEMA = SCHEMA_V1
SEMANTIC_EPOCH_POLICY = "semantic_epoch_v1"
_DEPLOYMENT_SCOPES = frozenset({"calibration_only", "frozen_validation"})
_V1_TOP_LEVEL_KEYS = frozenset({
    "schema",
    "profile_id",
    "elastic_profile_fingerprint_sha256",
    "workload_manifest_sha256",
    "deployment_scope",
    "default_e2e_deadline_ms",
    "controller",
    "rows",
    "fingerprint_sha256",
})
_V2_TOP_LEVEL_KEYS = _V1_TOP_LEVEL_KEYS | {"routing_policy"}
_SEMANTIC_EPOCH_POLICY_KEYS = frozenset({
    "policy",
    "pair_local",
    "decoder_load_scope",
    "endpoint_credit_scope",
    "decoder_high_water_numerator",
    "decoder_high_water_denominator",
    "decoder_low_water_numerator",
    "decoder_low_water_denominator",
    "epoch_confirmation_requests",
    "remote_overload_service_stretch",
    "remote_external_credit_close_fraction",
    "phase_label_policy_input",
    "physical_switch_label_policy_input",
})
_SEMANTIC_CREDIT_EPOCH_POLICY_KEYS = _SEMANTIC_EPOCH_POLICY_KEYS | {
    "local_external_credit_opens_epoch",
    "frontend_decoder_watermarks_policy_input",
}
_ROW_KEYS = frozenset({
    "prompt_tokens",
    "output_tokens",
    "cache_residency",
    "local_ttft_prior_ms",
    "remote_ttft_prior_ms",
    "local_token_ms",
    "remote_prefill_token_ms",
    "samples_local",
    "samples_remote",
    "outputs_equivalent",
    "evidence_valid",
})


def _sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def endpoint_service_profile_fingerprint(raw: Mapping[str, object]) -> str:
    if not isinstance(raw, Mapping):
        raise TypeError("endpoint service profile must be a mapping")
    payload = dict(raw)
    payload.pop("fingerprint_sha256", None)
    return _sha256(payload)


def _canonical_sha(name: str, value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class SemanticEpochRoutingPolicy:
    policy: str
    pair_local: bool
    decoder_load_scope: str
    endpoint_credit_scope: str
    decoder_high_water_numerator: int
    decoder_high_water_denominator: int
    decoder_low_water_numerator: int
    decoder_low_water_denominator: int
    epoch_confirmation_requests: int
    remote_overload_service_stretch: float
    remote_external_credit_close_fraction: float
    phase_label_policy_input: bool
    physical_switch_label_policy_input: bool
    local_external_credit_opens_epoch: bool | None = None
    frontend_decoder_watermarks_policy_input: bool | None = None

    def __post_init__(self) -> None:
        if self.policy != SEMANTIC_EPOCH_POLICY:
            raise ValueError("semantic routing policy ID differs")
        if self.pair_local is not True:
            raise ValueError("semantic routing policy must be pair-local")
        if self.decoder_load_scope != "frontend_request_start_to_http_eof":
            raise ValueError("semantic decoder-load scope differs")
        if (
            self.endpoint_credit_scope
            != "all_route_pinned_and_tempo_work_to_first_response"
        ):
            raise ValueError("semantic endpoint-credit scope differs")
        for name, value in (
            ("decoder_high_water_numerator", self.decoder_high_water_numerator),
            ("decoder_high_water_denominator", self.decoder_high_water_denominator),
            ("decoder_low_water_numerator", self.decoder_low_water_numerator),
            ("decoder_low_water_denominator", self.decoder_low_water_denominator),
            ("epoch_confirmation_requests", self.epoch_confirmation_requests),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int")
        if (
            self.decoder_high_water_numerator
            > self.decoder_high_water_denominator
        ):
            raise ValueError("semantic decoder high watermark exceeds capacity")
        if (
            self.decoder_low_water_numerator
            * self.decoder_high_water_denominator
            >= self.decoder_high_water_numerator
            * self.decoder_low_water_denominator
        ):
            raise ValueError("semantic decoder low watermark must be below high")
        for name, value, minimum, maximum in (
            (
                "remote_overload_service_stretch",
                self.remote_overload_service_stretch,
                1.0,
                math.inf,
            ),
            (
                "remote_external_credit_close_fraction",
                self.remote_external_credit_close_fraction,
                0.0,
                1.0,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < minimum
                or float(value) > maximum
                or (name == "remote_external_credit_close_fraction"
                    and float(value) == 0.0)
            ):
                raise ValueError(f"{name} is outside its frozen safe range")
        if self.phase_label_policy_input is not False:
            raise ValueError("phase labels cannot be semantic policy input")
        if self.physical_switch_label_policy_input is not False:
            raise ValueError(
                "physical-switch labels cannot be semantic policy input")
        credit_variant = (
            self.local_external_credit_opens_epoch,
            self.frontend_decoder_watermarks_policy_input,
        )
        if credit_variant not in {
            (None, None),
            (True, False),
        }:
            raise ValueError(
                "semantic credit epoch requires external-credit routing and "
                "observational-only frontend watermarks")

    @property
    def uses_local_external_credit_epoch(self) -> bool:
        return self.local_external_credit_opens_epoch is True

    def as_dict(self) -> dict[str, object]:
        keys = (
            _SEMANTIC_CREDIT_EPOCH_POLICY_KEYS
            if self.uses_local_external_credit_epoch
            else _SEMANTIC_EPOCH_POLICY_KEYS
        )
        return {
            name: getattr(self, name)
            for name in sorted(keys)
        }


@dataclass(frozen=True)
class EndpointServiceRow:
    prompt_tokens: int
    output_tokens: int
    cache_residency: CacheResidency
    local_ttft_prior_ms: float
    remote_ttft_prior_ms: float
    local_token_ms: int
    remote_prefill_token_ms: int
    samples_local: int
    samples_remote: int
    outputs_equivalent: bool
    evidence_valid: bool

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("prompt_tokens", self.prompt_tokens, 2),
            ("output_tokens", self.output_tokens, 2),
            ("local_token_ms", self.local_token_ms, 1),
            ("remote_prefill_token_ms", self.remote_prefill_token_ms, 1),
            ("samples_local", self.samples_local, 2),
            ("samples_remote", self.samples_remote, 2),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an int >= {minimum}")
        if not isinstance(self.cache_residency, CacheResidency):
            raise TypeError("cache_residency must be CacheResidency")
        if self.cache_residency is CacheResidency.UNKNOWN:
            raise ValueError("profile rows cannot use unknown cache residency")
        for name, value in (
            ("local_ttft_prior_ms", self.local_ttft_prior_ms),
            ("remote_ttft_prior_ms", self.remote_ttft_prior_ms),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if type(self.outputs_equivalent) is not bool or type(self.evidence_valid) is not bool:
            raise TypeError("row evidence flags must be bool")
        if not self.outputs_equivalent or not self.evidence_valid:
            raise ValueError("endpoint service rows require valid equivalent evidence")

    @property
    def key(self) -> tuple[int, int, CacheResidency]:
        return self.prompt_tokens, self.output_tokens, self.cache_residency


@dataclass(frozen=True)
class EndpointExternalServiceProxy:
    """Auditable profile-row proxy for route-pinned external work only.

    Adaptive TEMPO requests must continue to use :meth:`exact_row`.  Fixed
    background tenants can contain short diagnostic geometries that were not
    part of the foreground calibration grid.  For those requests this object
    records the measured row used to weight endpoint occupancy and normalize
    passive first-response feedback; it never turns the proxy into an exact
    service-profile claim.
    """

    row: EndpointServiceRow
    lookup_mode: str
    requested_prompt_tokens: int
    requested_output_tokens: int
    requested_cache_residency: CacheResidency

    def __post_init__(self) -> None:
        if not isinstance(self.row, EndpointServiceRow):
            raise TypeError("external service proxy row is invalid")
        if self.lookup_mode not in {
            "exact",
            "same_residency_geometry_ceiling",
            "miss_via_prefill_only_geometry_ceiling",
        }:
            raise ValueError("external service proxy lookup mode is invalid")
        for name, value in (
            ("requested_prompt_tokens", self.requested_prompt_tokens),
            ("requested_output_tokens", self.requested_output_tokens),
        ):
            if type(value) is not int or value < 2:
                raise ValueError(f"{name} must be an int >= 2")
        if not isinstance(self.requested_cache_residency, CacheResidency):
            raise TypeError("requested cache residency is invalid")
        if self.requested_cache_residency is CacheResidency.UNKNOWN:
            raise ValueError("external service proxy cannot retain unknown residency")


@dataclass(frozen=True)
class EndpointServiceProfile:
    profile_id: str
    elastic_profile_fingerprint_sha256: str
    workload_manifest_sha256: str
    deployment_scope: str
    default_e2e_deadline_ms: float
    controller: EndpointAdmissionConfig
    rows: tuple[EndpointServiceRow, ...]
    fingerprint_sha256: str
    routing_policy: SemanticEpochRoutingPolicy | None = None
    schema: str = SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema not in {SCHEMA_V1, SCHEMA_V2}:
            raise ValueError("endpoint service profile schema mismatch")
        if self.schema == SCHEMA_V1 and self.routing_policy is not None:
            raise ValueError("v1 endpoint profile cannot carry routing policy")
        if self.schema == SCHEMA_V2 and not isinstance(
            self.routing_policy, SemanticEpochRoutingPolicy
        ):
            raise ValueError("v2 endpoint profile requires routing policy")
        if type(self.profile_id) is not str or not self.profile_id.strip():
            raise ValueError("profile_id must be nonempty")
        _canonical_sha(
            "elastic_profile_fingerprint_sha256",
            self.elastic_profile_fingerprint_sha256,
        )
        _canonical_sha("workload_manifest_sha256", self.workload_manifest_sha256)
        _canonical_sha("fingerprint_sha256", self.fingerprint_sha256)
        if self.deployment_scope not in _DEPLOYMENT_SCOPES:
            raise ValueError("deployment_scope is invalid")
        if (
            isinstance(self.default_e2e_deadline_ms, bool)
            or not isinstance(self.default_e2e_deadline_ms, (int, float))
            or not math.isfinite(float(self.default_e2e_deadline_ms))
            or float(self.default_e2e_deadline_ms) <= 0.0
        ):
            raise ValueError("default_e2e_deadline_ms must be finite and positive")
        if not isinstance(self.controller, EndpointAdmissionConfig):
            raise TypeError("controller must be EndpointAdmissionConfig")
        if type(self.rows) is not tuple or not self.rows:
            raise ValueError("rows must be a nonempty tuple")
        if any(not isinstance(row, EndpointServiceRow) for row in self.rows):
            raise TypeError("rows must contain EndpointServiceRow")
        keys = [row.key for row in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("endpoint service profile row keys must be unique")
        if keys != sorted(keys, key=lambda key: (key[0], key[1], key[2].value)):
            raise ValueError("endpoint service profile rows must be canonically sorted")

    def exact_row(
        self,
        prompt_tokens: int,
        output_tokens: int,
        cache_residency: CacheResidency,
        *,
        cold_unknown_as_miss: bool = False,
    ) -> EndpointServiceRow:
        if cache_residency is CacheResidency.UNKNOWN and cold_unknown_as_miss:
            cache_residency = CacheResidency.MISS
        key = (prompt_tokens, output_tokens, cache_residency)
        for row in self.rows:
            if row.key == key:
                return row
        raise ValueError(
            "no exact endpoint service profile row: "
            f"prompt={prompt_tokens} output={output_tokens} "
            f"cache={cache_residency.value}"
        )

    def external_credit_proxy(
        self,
        prompt_tokens: int,
        output_tokens: int,
        cache_residency: CacheResidency,
        *,
        route: EndpointRoute,
        cold_unknown_as_miss: bool = False,
    ) -> EndpointExternalServiceProxy:
        """Resolve a measured geometry ceiling for fixed external tenants.

        The fallback is deliberately narrower than interpolation: a measured
        row must dominate both requested token dimensions.  Same-residency
        evidence is preferred.  A confirmed miss may use remote-eligible
        P_ONLY evidence as an explicitly labelled proxy because both local
        prefill and remote P/D occupy the same endpoints, while D_ONLY/BOTH
        are never treated as remotely eligible.  Requests outside the frozen
        profile envelope still fail closed.
        """

        if route not in {EndpointRoute.LOCAL, EndpointRoute.REMOTE}:
            raise ValueError("external credit proxy route must be local or remote")
        if cache_residency is CacheResidency.UNKNOWN and cold_unknown_as_miss:
            cache_residency = CacheResidency.MISS
        if not isinstance(cache_residency, CacheResidency):
            raise TypeError("cache_residency must be CacheResidency")
        if cache_residency is CacheResidency.UNKNOWN:
            raise ValueError("external credit proxy requires known residency")
        if type(prompt_tokens) is not int or prompt_tokens < 2:
            raise ValueError("prompt_tokens must be an int >= 2")
        if type(output_tokens) is not int or output_tokens < 2:
            raise ValueError("output_tokens must be an int >= 2")

        key = (prompt_tokens, output_tokens, cache_residency)
        for row in self.rows:
            if row.key == key:
                return EndpointExternalServiceProxy(
                    row=row,
                    lookup_mode="exact",
                    requested_prompt_tokens=prompt_tokens,
                    requested_output_tokens=output_tokens,
                    requested_cache_residency=cache_residency,
                )

        candidates = [
            row for row in self.rows
            if row.cache_residency is cache_residency
            and row.prompt_tokens >= prompt_tokens
            and row.output_tokens >= output_tokens
        ]
        mode = "same_residency_geometry_ceiling"
        if not candidates and cache_residency is CacheResidency.MISS:
            candidates = [
                row for row in self.rows
                if row.cache_residency is CacheResidency.P_ONLY
                and row.prompt_tokens >= prompt_tokens
                and row.output_tokens >= output_tokens
            ]
            mode = "miss_via_prefill_only_geometry_ceiling"
        if not candidates:
            raise ValueError(
                "no safe external endpoint service proxy row: "
                f"prompt={prompt_tokens} output={output_tokens} "
                f"cache={cache_residency.value} route={route.value}"
            )

        minimum_prompt = min(row.prompt_tokens for row in candidates)
        candidates = [
            row for row in candidates if row.prompt_tokens == minimum_prompt]
        minimum_output = min(row.output_tokens for row in candidates)
        candidates = [
            row for row in candidates if row.output_tokens == minimum_output]
        weight = (
            (lambda row: row.local_token_ms)
            if route is EndpointRoute.LOCAL
            else (lambda row: row.remote_prefill_token_ms)
        )
        row = max(candidates, key=lambda value: (weight(value), value.key))
        return EndpointExternalServiceProxy(
            row=row,
            lookup_mode=mode,
            requested_prompt_tokens=prompt_tokens,
            requested_output_tokens=output_tokens,
            requested_cache_residency=cache_residency,
        )


def load_endpoint_service_profile(path: Path) -> EndpointServiceProfile:
    if not isinstance(path, Path):
        raise TypeError("path must be Path")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("failed to read endpoint service profile") from exc
    if not isinstance(raw, dict):
        raise ValueError("endpoint service profile must be an object")
    schema = raw.get("schema")
    expected_keys = {
        SCHEMA_V1: _V1_TOP_LEVEL_KEYS,
        SCHEMA_V2: _V2_TOP_LEVEL_KEYS,
    }.get(schema)
    if expected_keys is None:
        raise ValueError("endpoint service profile schema mismatch")
    if set(raw) != expected_keys:
        raise ValueError("endpoint service profile top-level inventory is not exact")
    declared = _canonical_sha("fingerprint_sha256", raw["fingerprint_sha256"])
    computed = endpoint_service_profile_fingerprint(raw)
    if declared != computed:
        raise ValueError("endpoint service profile fingerprint mismatch")
    controller_raw = raw["controller"]
    if not isinstance(controller_raw, dict):
        raise TypeError("endpoint controller profile must be an object")
    try:
        controller = EndpointAdmissionConfig(**controller_raw)
    except TypeError as exc:
        raise ValueError("endpoint controller profile inventory is invalid") from exc
    routing_policy = None
    if schema == SCHEMA_V2:
        routing_raw = raw["routing_policy"]
        if (
            not isinstance(routing_raw, dict)
            or frozenset(routing_raw) not in {
                _SEMANTIC_EPOCH_POLICY_KEYS,
                _SEMANTIC_CREDIT_EPOCH_POLICY_KEYS,
            }
        ):
            raise ValueError("endpoint routing-policy inventory is not exact")
        try:
            routing_policy = SemanticEpochRoutingPolicy(**routing_raw)
        except TypeError as exc:
            raise ValueError("endpoint routing-policy inventory is invalid") from exc
    rows_raw = raw["rows"]
    if not isinstance(rows_raw, list) or not rows_raw:
        raise ValueError("endpoint service profile rows must be nonempty")
    rows = []
    for item in rows_raw:
        if not isinstance(item, dict) or set(item) != _ROW_KEYS:
            raise ValueError("endpoint service row inventory is not exact")
        values = dict(item)
        try:
            values["cache_residency"] = CacheResidency(values["cache_residency"])
        except (TypeError, ValueError) as exc:
            raise ValueError("endpoint service row cache residency is invalid") from exc
        rows.append(EndpointServiceRow(**values))
    return EndpointServiceProfile(
        profile_id=raw["profile_id"],
        elastic_profile_fingerprint_sha256=_canonical_sha(
            "elastic_profile_fingerprint_sha256",
            raw["elastic_profile_fingerprint_sha256"],
        ),
        workload_manifest_sha256=_canonical_sha(
            "workload_manifest_sha256", raw["workload_manifest_sha256"]
        ),
        deployment_scope=raw["deployment_scope"],
        default_e2e_deadline_ms=raw["default_e2e_deadline_ms"],
        controller=controller,
        rows=tuple(rows),
        fingerprint_sha256=declared,
        routing_policy=routing_policy,
        schema=schema,
    )


__all__ = [
    "EndpointExternalServiceProxy",
    "EndpointServiceProfile",
    "EndpointServiceRow",
    "SCHEMA",
    "SCHEMA_V1",
    "SCHEMA_V2",
    "SEMANTIC_EPOCH_POLICY",
    "SemanticEpochRoutingPolicy",
    "endpoint_service_profile_fingerprint",
    "load_endpoint_service_profile",
]
