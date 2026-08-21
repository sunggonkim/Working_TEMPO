"""Strict JSON manifest for frozen TEMPO-PD calibration profiles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tempo.pd_admission import (
    POLICY_SCHEMA,
    FrozenPDAdmissionPolicy,
    PDCalibrationProfile,
    PDEvidenceLevel,
    PDPolicyConfig,
    PDWorkloadClass,
)


MANIFEST_SCHEMA = "tempo-live-pd-profile-manifest-1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _closed(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    _require(actual == expected, f"{field} keys mismatch: {sorted(actual ^ expected)}")


@dataclass(frozen=True)
class PDPolicyManifest:
    classifier_version: str
    policy_epoch: int
    deployment_scope: str
    config: PDPolicyConfig
    profiles: tuple[PDCalibrationProfile, ...]

    def __post_init__(self) -> None:
        _require(bool(self.classifier_version.strip()), "classifier_version must be nonempty")
        _require(type(self.policy_epoch) is int and self.policy_epoch >= 0,
                 "policy_epoch must be a nonnegative int")
        _require(self.deployment_scope in {"screen_only", "production"},
                 "deployment_scope must be screen_only or production")
        _require(isinstance(self.config, PDPolicyConfig), "config type mismatch")
        _require(bool(self.profiles), "manifest requires at least one profile")
        _require(all(isinstance(value, PDCalibrationProfile) for value in self.profiles),
                 "profiles type mismatch")
        fingerprints = [value.workload.fingerprint for value in self.profiles]
        _require(len(fingerprints) == len(set(fingerprints)),
                 "manifest contains duplicate workload profiles")
        _require(all(value.valid_from_epoch <= self.policy_epoch <= value.valid_through_epoch
                     for value in self.profiles),
                 "manifest policy_epoch is outside a profile validity range")
        if self.deployment_scope == "production":
            _require(self.config.require_replicated_evidence,
                     "production manifest must require replicated evidence")
            _require(all(value.evidence_level is PDEvidenceLevel.REPLICATED
                         for value in self.profiles),
                     "production manifest contains screen-only evidence")

    def build_policy(self, *, allow_screen_profiles: bool = False) -> FrozenPDAdmissionPolicy:
        if self.deployment_scope == "screen_only" and not allow_screen_profiles:
            raise ValueError("screen-only manifest requires explicit allow_screen_profiles")
        return FrozenPDAdmissionPolicy(self.profiles, self.config)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "policy_schema": POLICY_SCHEMA,
            "classifier_version": self.classifier_version,
            "policy_epoch": self.policy_epoch,
            "deployment_scope": self.deployment_scope,
            "config": {
                "remote_advantage_margin_ms": self.config.remote_advantage_margin_ms,
                "minimum_samples_per_route": self.config.minimum_samples_per_route,
                "require_replicated_evidence": self.config.require_replicated_evidence,
                "remote_deadline_reserve_ms": self.config.remote_deadline_reserve_ms,
            },
            "profiles": [value.canonical_dict() for value in self.profiles],
        }

    @property
    def manifest_id(self) -> str:
        raw = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
        return f"pd-manifest-{hashlib.sha256(raw).hexdigest()[:20]}"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PDPolicyManifest":
        _closed(value, {
            "schema", "policy_schema", "classifier_version", "policy_epoch",
            "deployment_scope", "config", "profiles",
        }, "manifest")
        _require(value["schema"] == MANIFEST_SCHEMA, "manifest schema mismatch")
        _require(value["policy_schema"] == POLICY_SCHEMA, "policy schema mismatch")
        config_raw = value["config"]
        _require(isinstance(config_raw, Mapping), "config must be an object")
        _closed(config_raw, {
            "remote_advantage_margin_ms", "minimum_samples_per_route",
            "require_replicated_evidence", "remote_deadline_reserve_ms",
        }, "config")
        profiles_raw = value["profiles"]
        _require(isinstance(profiles_raw, list), "profiles must be a list")
        profiles: list[PDCalibrationProfile] = []
        for index, raw in enumerate(profiles_raw):
            _require(isinstance(raw, Mapping), f"profiles[{index}] must be an object")
            _closed(raw, {
                "workload", "evidence_level", "local_samples", "remote_samples",
                "local_latency_p50_ms", "remote_latency_p50_ms",
                "local_latency_lower_bound_ms", "remote_latency_upper_bound_ms",
                "outputs_equivalent", "remote_transfer_failures",
                "valid_from_epoch", "valid_through_epoch",
            }, f"profiles[{index}]")
            workload_raw = raw["workload"]
            _require(isinstance(workload_raw, Mapping),
                     f"profiles[{index}].workload must be an object")
            _closed(workload_raw, {
                "model_id", "model_revision", "topology_id", "remote_backend",
                "prompt_bucket", "output_bucket", "decoder_load_bucket",
                "kv_bytes_bucket",
            }, f"profiles[{index}].workload")
            profiles.append(PDCalibrationProfile(
                workload=PDWorkloadClass(**dict(workload_raw)),
                evidence_level=PDEvidenceLevel(raw["evidence_level"]),
                local_samples=raw["local_samples"],
                remote_samples=raw["remote_samples"],
                local_latency_p50_ms=raw["local_latency_p50_ms"],
                remote_latency_p50_ms=raw["remote_latency_p50_ms"],
                local_latency_lower_bound_ms=raw["local_latency_lower_bound_ms"],
                remote_latency_upper_bound_ms=raw["remote_latency_upper_bound_ms"],
                outputs_equivalent=raw["outputs_equivalent"],
                remote_transfer_failures=raw["remote_transfer_failures"],
                valid_from_epoch=raw["valid_from_epoch"],
                valid_through_epoch=raw["valid_through_epoch"],
            ))
        return cls(
            classifier_version=value["classifier_version"],
            policy_epoch=value["policy_epoch"],
            deployment_scope=value["deployment_scope"],
            config=PDPolicyConfig(**dict(config_raw)),
            profiles=tuple(profiles),
        )


def load_manifest(path: Path) -> PDPolicyManifest:
    _require(path.is_file(), f"manifest is not an explicit file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "manifest root must be an object")
    return PDPolicyManifest.from_dict(value)


def write_manifest(path: Path, manifest: PDPolicyManifest) -> None:
    _require(not path.exists(), f"refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.canonical_dict(), sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
