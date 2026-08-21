"""Canonical frozen-profile loader for TEMPO Elastic-PD."""

from __future__ import annotations

from pathlib import Path

from tempo.pd_elastic_profile_v444 import (
    ElasticPDProfile,
    ElasticProfileIdentity,
    ElasticProfileRow,
    SCHEMA,
    load_elastic_profile as _load_versioned_profile,
)


def load_elastic_profile(path: Path) -> ElasticPDProfile:
    """Load the exact JSON profile without filling missing fields."""

    return _load_versioned_profile(path)


def require_replicated_profile(profile: ElasticPDProfile) -> None:
    """Reject screen-only or under-sampled profiles for final validation."""

    if profile.deployment_scope != "replicated":
        raise ValueError("final validation requires a replicated profile")
    for row in profile.rows:
        if row.samples_local < 3 or row.samples_remote < 3:
            raise ValueError("final validation requires three samples per route")
        if not row.evidence_safe:
            raise ValueError("profile row lacks exact remote evidence")


__all__ = [
    "ElasticPDProfile",
    "ElasticProfileIdentity",
    "ElasticProfileRow",
    "SCHEMA",
    "load_elastic_profile",
    "require_replicated_profile",
]
