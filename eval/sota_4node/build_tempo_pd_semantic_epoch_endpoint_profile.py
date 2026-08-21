#!/usr/bin/env python3
"""Derive the frozen semantic-epoch v2 profile without tuning service rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from tempo.pd_endpoint_profile import (
    SCHEMA_V1,
    SCHEMA_V2,
    SEMANTIC_EPOCH_POLICY,
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)


PROFILE_ID = "tempo-pd-endpoint-qwen25-ucx-c4-semantic-epoch-v2"
CREDIT_EPOCH_PROFILE_ID = (
    "tempo-pd-endpoint-qwen25-ucx-c4-semantic-credit-epoch-v2")
SEMANTIC_ROUTING_POLICY: dict[str, object] = {
    "policy": SEMANTIC_EPOCH_POLICY,
    "pair_local": True,
    "decoder_load_scope": "frontend_request_start_to_http_eof",
    "endpoint_credit_scope": (
        "all_route_pinned_and_tempo_work_to_first_response"),
    "decoder_high_water_numerator": 1,
    "decoder_high_water_denominator": 2,
    "decoder_low_water_numerator": 1,
    "decoder_low_water_denominator": 4,
    "epoch_confirmation_requests": 2,
    "remote_overload_service_stretch": 2.0,
    "remote_external_credit_close_fraction": 1.0,
    "phase_label_policy_input": False,
    "physical_switch_label_policy_input": False,
}
CREDIT_EPOCH_POLICY_ADDITIONS: dict[str, object] = {
    # Candidate C replaces the confounded active-request route signal with
    # exact route-pinned local endpoint ownership.  Any positive owned
    # external credit is already weighted by the frozen service profile, so
    # this rule introduces no fitted scalar threshold.
    "local_external_credit_opens_epoch": True,
    "frontend_decoder_watermarks_policy_input": False,
}


def routing_policy_for_profile_id(profile_id: str) -> dict[str, object]:
    value = copy.deepcopy(SEMANTIC_ROUTING_POLICY)
    if profile_id == CREDIT_EPOCH_PROFILE_ID:
        value.update(CREDIT_EPOCH_POLICY_ADDITIONS)
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256",
    )
    return value


def build_profile(
    base_path: Path, *, expected_base_sha256: str,
    profile_id: str = PROFILE_ID,
) -> dict[str, object]:
    base_path = base_path.resolve()
    _require(base_path.is_file(), "base endpoint profile is missing")
    _require(
        _sha256(base_path)
        == _canonical_sha(expected_base_sha256, name="base profile SHA-256"),
        "base endpoint profile digest differs",
    )
    loaded = load_endpoint_service_profile(base_path)
    _require(
        loaded.schema == SCHEMA_V1
        and loaded.routing_policy is None
        and loaded.deployment_scope == "calibration_only",
        "semantic profile requires a calibration-only v1 source",
    )
    _require(type(profile_id) is str and profile_id.strip(),
             "semantic profile_id must be nonempty")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    value = copy.deepcopy(base)
    value["schema"] = SCHEMA_V2
    value["profile_id"] = profile_id
    value["routing_policy"] = routing_policy_for_profile_id(profile_id)
    value["fingerprint_sha256"] = endpoint_service_profile_fingerprint(value)
    _require(
        value["controller"] == base["controller"]
        and value["rows"] == base["rows"]
        and value["default_e2e_deadline_ms"]
        == base["default_e2e_deadline_ms"]
        and value["elastic_profile_fingerprint_sha256"]
        == base["elastic_profile_fingerprint_sha256"]
        and value["workload_manifest_sha256"]
        == base["workload_manifest_sha256"]
        and value["deployment_scope"] == base["deployment_scope"],
        "semantic derivation changed service evidence or controller values",
    )
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--profile-id", default=PROFILE_ID)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite semantic profile")
    value = build_profile(
        args.base_profile,
        expected_base_sha256=args.expected_base_sha256,
        profile_id=args.profile_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_endpoint_service_profile(args.output.resolve())
    _require(
        loaded.schema == SCHEMA_V2
        and loaded.routing_policy is not None
        and loaded.routing_policy.as_dict()
        == routing_policy_for_profile_id(args.profile_id),
        "published semantic endpoint profile differs",
    )
    print(json.dumps({
        "schema": loaded.schema,
        "profile_id": loaded.profile_id,
        "fingerprint_sha256": loaded.fingerprint_sha256,
        "sha256": _sha256(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
