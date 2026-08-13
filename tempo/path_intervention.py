"""Matched path-intervention evidence for TEMPO-RD resource domains.

This is intentionally separate from :class:`DomainEvidence`.  A path
intervention can establish that a declared resource path changes foreground
tail while still lacking a byte-level counter suitable for admission control.
The distinction prevents a strong route experiment (such as disabling NCCL
P2P) from being silently promoted as a byte-accounted scheduler domain.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from tempo.resource_domain import (
    DomainObservation,
    EvidenceLevel,
    ResourceDomain,
    allowed_counter_scopes,
)


@dataclass(frozen=True)
class PathInterventionEvidence:
    """One matched baseline/intervention path experiment.

    ``baseline`` and ``intervention`` must share workload geometry, auxiliary
    bytes, placement, and executable identity.  ``path_status`` is a string
    rather than the broader DomainEvidence enum so this record can be loaded
    directly from a runtime JSON artifact without coercion.
    """

    domain: ResourceDomain
    intervention_id: str
    control_name: str
    control_value: str
    baseline_mode: str
    intervention_mode: str
    baseline_step_p99_ns: int
    intervention_step_p99_ns: int
    baseline_window_p99_ns: int
    intervention_window_p99_ns: int
    baseline_skew_p99_ns: int | None
    intervention_skew_p99_ns: int | None
    sample_count: int
    uncertainty_ns: int
    source: str
    baseline_workload_digest: str
    intervention_workload_digest: str
    baseline_placement_digest: str
    intervention_placement_digest: str
    auxiliary_bytes_baseline: int
    auxiliary_bytes_intervention: int
    path_status: str = "observed"
    counter_scope: str = "none"
    byte_attribution: bool = False
    observed_path: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ResourceDomain):
            raise TypeError("domain must be a ResourceDomain")
        for name in (
            "intervention_id", "control_name", "control_value", "baseline_mode",
            "intervention_mode", "source", "baseline_workload_digest",
            "intervention_workload_digest", "baseline_placement_digest",
            "intervention_placement_digest",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.observed_path) is not str:
            raise TypeError("observed_path must be a string")
        for name in (
            "baseline_step_p99_ns", "intervention_step_p99_ns",
            "baseline_window_p99_ns", "intervention_window_p99_ns",
            "sample_count", "uncertainty_ns", "auxiliary_bytes_baseline",
            "auxiliary_bytes_intervention",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        for name in ("baseline_skew_p99_ns", "intervention_skew_p99_ns"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be None or a non-negative int")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.path_status not in {"observed", "declared", "unknown"}:
            raise ValueError("path_status must be observed, declared, or unknown")
        if not self.counter_scope:
            raise ValueError("counter_scope must be non-empty")
        if type(self.byte_attribution) is not bool:
            raise TypeError("byte_attribution must be bool")
        if self.byte_attribution and (
            self.counter_scope == "none"
            or self.counter_scope not in allowed_counter_scopes(self.domain)
        ):
            raise ValueError(
                f"byte-attributed {self.domain.value} evidence requires an allowed counter scope"
            )
        if self.path_status != "observed" and self.causal_path_candidate:
            raise ValueError("a causal path candidate requires an observed path")
        if self.auxiliary_bytes_baseline != self.auxiliary_bytes_intervention:
            raise ValueError("matched path experiments must preserve auxiliary bytes")
        if not self.observed_path and self.path_status == "observed":
            raise ValueError("observed path evidence must be labeled")

    @property
    def step_delta_ns(self) -> int:
        return self.intervention_step_p99_ns - self.baseline_step_p99_ns

    @property
    def window_delta_ns(self) -> int:
        return self.intervention_window_p99_ns - self.baseline_window_p99_ns

    @property
    def skew_delta_ns(self) -> int | None:
        if self.baseline_skew_p99_ns is None or self.intervention_skew_p99_ns is None:
            return None
        return self.intervention_skew_p99_ns - self.baseline_skew_p99_ns

    @property
    def causal_path_candidate(self) -> bool:
        """Whether the path intervention changes tail above uncertainty."""

        return (
            self.path_status == "observed"
            and self.baseline_workload_digest == self.intervention_workload_digest
            and self.baseline_placement_digest == self.intervention_placement_digest
            and self.step_delta_ns > self.uncertainty_ns
            and self.window_delta_ns > self.uncertainty_ns
        )

    @property
    def controller_ready(self) -> bool:
        """Whether this path record may feed byte-admission control."""

        return (
            self.causal_path_candidate
            and self.byte_attribution
            and self.counter_scope in allowed_counter_scopes(self.domain)
        )

    def to_domain_observation(
        self,
        *,
        foreground_kind: str,
        auxiliary_kind: str,
        overlap_ns: int,
    ) -> DomainObservation:
        """Materialize a causal aggregate only with explicit overlap metadata.

        The path record does not pretend that a p99 interval is a traffic
        overlap interval.  Callers must provide the measured interval from the
        same source-bound trace, and only byte-attributed records can enter the
        shared observation aggregate.
        """

        if not self.controller_ready:
            raise ValueError("path intervention is not byte-attributed/controller-ready")
        if type(overlap_ns) is not int or overlap_ns <= 0:
            raise ValueError("overlap_ns must be a positive int")
        return DomainObservation(
            domain=self.domain,
            foreground_kind=foreground_kind,
            auxiliary_kind=auxiliary_kind,
            overlapping_bytes=self.auxiliary_bytes_intervention,
            overlap_ns=overlap_ns,
            tail_delta_ns=self.step_delta_ns,
            evidence=EvidenceLevel.INTERVENTIONAL,
            source=self.source,
            uncertainty_ns=self.uncertainty_ns,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, source-bound representation."""

        return {
            "schema_version": "tempo-rd-path-intervention-1",
            "domain": self.domain.value,
            "intervention_id": self.intervention_id,
            "control_name": self.control_name,
            "control_value": self.control_value,
            "baseline_mode": self.baseline_mode,
            "intervention_mode": self.intervention_mode,
            "baseline_step_p99_ns": self.baseline_step_p99_ns,
            "intervention_step_p99_ns": self.intervention_step_p99_ns,
            "baseline_window_p99_ns": self.baseline_window_p99_ns,
            "intervention_window_p99_ns": self.intervention_window_p99_ns,
            "baseline_skew_p99_ns": self.baseline_skew_p99_ns,
            "intervention_skew_p99_ns": self.intervention_skew_p99_ns,
            "sample_count": self.sample_count,
            "uncertainty_ns": self.uncertainty_ns,
            "source": self.source,
            "baseline_workload_digest": self.baseline_workload_digest,
            "intervention_workload_digest": self.intervention_workload_digest,
            "baseline_placement_digest": self.baseline_placement_digest,
            "intervention_placement_digest": self.intervention_placement_digest,
            "auxiliary_bytes_baseline": self.auxiliary_bytes_baseline,
            "auxiliary_bytes_intervention": self.auxiliary_bytes_intervention,
            "path_status": self.path_status,
            "counter_scope": self.counter_scope,
            "byte_attribution": self.byte_attribution,
            "observed_path": self.observed_path,
            "causal_path_candidate": self.causal_path_candidate,
            "controller_ready": self.controller_ready,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PathInterventionEvidence":
        """Parse the strict machine-readable evidence schema."""

        if not isinstance(payload, Mapping):
            raise TypeError("path intervention payload must be a mapping")
        if payload.get("schema_version") != "tempo-rd-path-intervention-1":
            raise ValueError("unsupported path intervention schema")
        values = dict(payload)
        values.pop("schema_version", None)
        values.pop("causal_path_candidate", None)
        values.pop("controller_ready", None)
        # Provenance is an optional artifact envelope; the evidence record
        # itself remains stable when a report adds job IDs or source hashes.
        values.pop("provenance", None)
        try:
            values["domain"] = ResourceDomain(values["domain"])
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid path intervention domain") from exc
        record = cls(**values)
        if payload.get("causal_path_candidate") is not record.causal_path_candidate:
            raise ValueError("causal_path_candidate does not match evidence")
        if payload.get("controller_ready") is not record.controller_ready:
            raise ValueError("controller_ready does not match evidence")
        return record


def path_intervention_candidates(
    records: Iterable[PathInterventionEvidence],
) -> frozenset[ResourceDomain]:
    """Return domains with causal path evidence, not necessarily byte control."""

    values = tuple(records)
    if any(not isinstance(record, PathInterventionEvidence) for record in values):
        raise TypeError("records must contain PathInterventionEvidence values")
    return frozenset(record.domain for record in values if record.causal_path_candidate)


def controller_ready_domains(
    records: Iterable[PathInterventionEvidence],
) -> frozenset[ResourceDomain]:
    """Return only causal path domains with explicit byte-level attribution."""

    values = tuple(records)
    if any(not isinstance(record, PathInterventionEvidence) for record in values):
        raise TypeError("records must contain PathInterventionEvidence values")
    return frozenset(record.domain for record in values if record.controller_ready)


def build_causal_domain_controller(
    records: Iterable[PathInterventionEvidence],
    budgets: Mapping[ResourceDomain, Any],
    *,
    catch_up_slack_ns: int,
) -> Any:
    """Build the shared controller only from promoted, byte-attributed domains.

    This is the explicit bridge from causal atlas evidence to orchestration.
    A topology label or a path-only intervention cannot silently enable a
    controller budget.  The returned object is the same controller used by
    checkpoint and KV adapters.
    """

    from tempo.domain_admission import DomainAdmissionController, DomainBudget

    values = tuple(records)
    if any(not isinstance(record, PathInterventionEvidence) for record in values):
        raise TypeError("records must contain PathInterventionEvidence values")
    if type(budgets) is not dict or not budgets:
        raise TypeError("budgets must be a non-empty dict")
    ready = controller_ready_domains(values)
    missing = set(budgets) - set(ready)
    if missing:
        raise ValueError(
            "cannot enable domains without causal byte evidence: "
            + ",".join(sorted(domain.value for domain in missing))
        )
    if any(not isinstance(domain, ResourceDomain) for domain in budgets):
        raise TypeError("budget keys must be ResourceDomain values")
    if any(not isinstance(budget, DomainBudget) for budget in budgets.values()):
        raise TypeError("budgets must contain DomainBudget values")
    return DomainAdmissionController(budgets, catch_up_slack_ns=catch_up_slack_ns)


def _resolve_artifact_root(
    repo_root: Path, value: object, *, field_name: str
) -> Path:
    """Resolve one artifact root without allowing repository escape."""

    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field_name} must be a relative path inside repo_root")
    candidate = (repo_root / relative).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{field_name} must stay inside repo_root") from exc
    return candidate


def validate_path_intervention_artifact(
    payload: Mapping[str, Any], *, repo_root: str | Path
) -> PathInterventionEvidence:
    """Validate a path record against its paired raw job manifests.

    The numeric record alone is insufficient: a recomputed JSON value must be
    tied to the baseline/intervention execution directories, job IDs, source
    bundles, five matched modes, and complete raw matrix.  This validator does
    not grant byte-level controller readiness; it only makes the path result
    source-bound and reproducible.
    """

    record = PathInterventionEvidence.from_dict(payload)
    provenance = payload.get("provenance")
    if type(provenance) is not dict:
        raise ValueError("path intervention provenance is required")
    required = {
        "baseline_job", "intervention_job", "baseline_artifact_root",
        "intervention_artifact_root", "baseline_source_bundle_sha256",
        "intervention_source_bundle_sha256", "matched_modes",
        "groups_per_mode", "restores_per_mode",
    }
    if set(provenance) != required:
        raise ValueError("path intervention provenance keys are not exact")
    for name in ("baseline_job", "intervention_job", "groups_per_mode", "restores_per_mode"):
        if type(provenance[name]) is not int or provenance[name] <= 0:
            raise ValueError(f"{name} must be a positive int")
    for name in ("baseline_source_bundle_sha256", "intervention_source_bundle_sha256"):
        value = provenance[name]
        if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} must be lowercase SHA-256 hex")
    modes = provenance["matched_modes"]
    expected_modes = ["combined", "d2h_only", "fg_only", "open_combined", "persist_only"]
    if modes != expected_modes:
        raise ValueError("matched_modes are not the exact raw matrix")
    root = Path(repo_root).resolve()
    baseline_root = _resolve_artifact_root(
        root,
        provenance["baseline_artifact_root"],
        field_name="baseline_artifact_root",
    )
    intervention_root = _resolve_artifact_root(
        root,
        provenance["intervention_artifact_root"],
        field_name="intervention_artifact_root",
    )
    if not baseline_root.is_dir() or not intervention_root.is_dir():
        raise ValueError("paired raw artifact directory is missing")
    baseline_manifest_path = baseline_root / "raw_manifest.json"
    intervention_manifest_path = intervention_root / "raw_manifest.json"
    try:
        baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
        intervention_manifest = json.loads(intervention_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("paired raw manifest is unreadable") from exc
    if baseline_manifest.get("job_id") != provenance["baseline_job"]:
        raise ValueError("baseline job ID does not match provenance")
    if intervention_manifest.get("job_id") != provenance["intervention_job"]:
        raise ValueError("intervention job ID does not match provenance")
    for manifest in (baseline_manifest, intervention_manifest):
        if manifest.get("nodes") != 2 or manifest.get("world_size") != 8:
            raise ValueError("paired raw geometry is not 2-node/8-rank")
        if manifest.get("promotion_eligible") is not False:
            raise ValueError("raw path artifacts must remain non-promotable")
    if baseline_manifest.get("source_bundle_sha256") != provenance["baseline_source_bundle_sha256"]:
        raise ValueError("baseline source bundle does not match provenance")
    if intervention_manifest.get("source_bundle_sha256") != provenance["intervention_source_bundle_sha256"]:
        raise ValueError("intervention source bundle does not match provenance")
    intervention = intervention_manifest.get("path_intervention")
    if intervention != {"name": record.control_name, "value": record.control_value}:
        raise ValueError("intervention control does not match raw manifest")
    for artifact_root in (baseline_root, intervention_root):
        for mode in modes:
            if not (artifact_root / f"fabric_observation_{mode}.json").is_file():
                raise ValueError(f"raw mode artifact is missing: {mode}")
    return record
