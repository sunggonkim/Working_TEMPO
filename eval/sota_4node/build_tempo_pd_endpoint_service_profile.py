#!/usr/bin/env python3
"""Build a fail-closed endpoint-service profile from paired live-vLLM traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Iterable, Mapping

from tempo.pd_elastic_controller_v443 import CacheResidency
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import (
    SCHEMA,
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)


MANIFEST_SCHEMA = "tempo-pd-c4-phase-screen-manifest-v1"
_LOCAL_ROUTE = "decoder_local_chunked_prefill"
_REMOTE_ROUTE = "official_lmcache_remote_prefill"
_PAIR_ARM = re.compile(r"^epd-(?:local|remote)-(r\d+-measured-item-\d+)$")


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
        f"{name} must be lowercase SHA-256",
    )
    return value


def _ttft_ms(row: Mapping[str, object]) -> float:
    dispatch = row.get("dispatch_offset_ns")
    arrivals = row.get("token_arrival_offsets_ns")
    _require(type(dispatch) is int and dispatch >= 0, "dispatch timestamp missing")
    _require(
        isinstance(arrivals, list)
        and arrivals
        and type(arrivals[0]) is int
        and arrivals[0] > dispatch,
        "first-token timestamp missing",
    )
    return (arrivals[0] - dispatch) / 1_000_000.0


def _pair_key(request_id: object) -> str:
    _require(type(request_id) is str, "request ID missing")
    match = _PAIR_ARM.fullmatch(request_id)
    _require(match is not None, "request ID is not a paired calibration ID")
    return match.group(1)


def collect_service_rows(
    artifacts: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[
        tuple[int, int, CacheResidency], dict[str, list[float]]
    ] = defaultdict(lambda: {"local": [], "remote": []})
    paired_outputs: dict[
        tuple[str, int, int, CacheResidency], dict[str, str]
    ] = defaultdict(dict)

    for artifact in artifacts:
        validation = artifact.get("validation")
        _require(
            isinstance(validation, Mapping)
            and validation.get("all_streams_valid") is True
            and validation.get("router_decisions_exact") is True
            and validation.get("performance_claim_allowed") is True,
            "endpoint calibration artifact is not performance-valid",
        )
        requests = artifact.get("requests")
        decisions = artifact.get("router_decisions")
        _require(isinstance(requests, list), "calibration requests missing")
        _require(isinstance(decisions, list), "calibration decisions missing")
        request_index = {row.get("request_id"): row for row in requests}
        decision_index = {row.get("request_id"): row for row in decisions}
        _require(
            len(request_index) == len(requests)
            and len(decision_index) == len(decisions)
            and set(request_index) == set(decision_index),
            "calibration request/decision IDs are not exact",
        )
        for request_id, row in request_index.items():
            _require(isinstance(row, Mapping), "calibration request is not an object")
            _require(row.get("valid") is True, "invalid endpoint calibration request")
            decision = decision_index[request_id]
            _require(isinstance(decision, Mapping), "calibration decision is not an object")
            route = decision.get("route")
            if route == _LOCAL_ROUTE:
                label = "local"
            elif route == _REMOTE_ROUTE:
                label = "remote"
            else:
                raise ValueError("endpoint calibration route is not canonical")
            try:
                residency = CacheResidency(decision.get("cache_residency"))
            except (TypeError, ValueError) as exc:
                raise ValueError("endpoint calibration cache residency is invalid") from exc
            _require(
                residency is not CacheResidency.UNKNOWN,
                "endpoint calibration cannot use unknown cache residency",
            )
            prompt_tokens = decision.get("prompt_tokens")
            output_tokens = decision.get("output_tokens")
            _require(
                type(prompt_tokens) is int and prompt_tokens >= 2,
                "calibration prompt geometry missing",
            )
            _require(
                type(output_tokens) is int and output_tokens >= 2,
                "calibration output geometry missing",
            )
            _require(
                row.get("requested_max_tokens") == output_tokens,
                "request/decision output geometry differs",
            )
            key = (prompt_tokens, output_tokens, residency)
            groups[key][label].append(_ttft_ms(row))
            output_hash = row.get("output_text_sha256")
            _require(
                type(output_hash) is str and len(output_hash) == 64,
                "calibration output hash missing",
            )
            paired_outputs[
                (_pair_key(request_id), prompt_tokens, output_tokens, residency)
            ][label] = output_hash

    rows: list[dict[str, object]] = []
    for key, samples in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value)
    ):
        prompt_tokens, output_tokens, residency = key
        local = samples["local"]
        remote = samples["remote"]
        _require(
            len(local) >= 2 and len(remote) >= 2,
            "each endpoint service row needs two samples per route",
        )
        comparable = [
            value
            for pair_key, value in paired_outputs.items()
            if pair_key[1:] == key and set(value) == {"local", "remote"}
        ]
        outputs_equivalent = bool(comparable) and all(
            value["local"] == value["remote"] for value in comparable
        )
        _require(outputs_equivalent, "paired endpoint outputs are not equivalent")
        local_prior = float(statistics.median(local))
        remote_prior = float(statistics.median(remote))
        rows.append({
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "cache_residency": residency.value,
            "local_ttft_prior_ms": local_prior,
            "remote_ttft_prior_ms": remote_prior,
            "local_token_ms": math.ceil(prompt_tokens * local_prior),
            "remote_prefill_token_ms": math.ceil(prompt_tokens * remote_prior),
            "samples_local": len(local),
            "samples_remote": len(remote),
            "outputs_equivalent": True,
            "evidence_valid": True,
        })
    _require(rows, "no endpoint service rows were collected")
    return rows


def _resolve_bound_path(
    repo_root: Path, entry: Mapping[str, object], *, name: str,
) -> Path:
    _require(set(entry) == {"path", "sha256"}, f"{name} binding is not exact")
    path_value = entry["path"]
    _require(type(path_value) is str and path_value, f"{name} path missing")
    path = (repo_root / path_value).resolve()
    _require(path.is_file(), f"{name} artifact missing")
    expected = _canonical_sha(entry["sha256"], name=f"{name}.sha256")
    _require(_sha256(path) == expected, f"{name} artifact hash differs")
    return path


def build_from_manifest(manifest_path: Path, *, profile_id: str) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        isinstance(manifest, dict) and manifest.get("schema") == MANIFEST_SCHEMA,
        "C4 screen manifest schema differs",
    )
    _require(manifest.get("controller_profile_scope") == "calibration_only",
             "endpoint builder requires calibration-only scope")
    _require(manifest.get("performance_claim_allowed") is False,
             "C4 screen manifest cannot permit a performance claim")
    repo_root = Path(__file__).resolve().parents[2]
    parents = manifest.get("calibration_parents")
    _require(isinstance(parents, Mapping), "calibration parent bindings missing")
    gate_path = _resolve_bound_path(repo_root, parents["c3_gate"], name="c3_gate")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    _require(
        gate.get("authorizes_c4_phase_trace") is True
        and gate.get("c3_coupled_characterization_valid") is True,
        "C3 gate does not authorize C4",
    )
    _resolve_bound_path(repo_root, parents["c3_raw"], name="c3_raw")
    elastic_binding = parents.get("elastic_profile")
    _require(isinstance(elastic_binding, Mapping), "elastic profile binding missing")
    elastic_path = _resolve_bound_path(
        repo_root,
        {"path": elastic_binding.get("path"), "sha256": elastic_binding.get("sha256")},
        name="elastic_profile",
    )
    elastic = load_elastic_profile(elastic_path)
    _require(
        elastic.fingerprint_sha256
        == _canonical_sha(
            elastic_binding.get("fingerprint_sha256"),
            name="elastic_profile.fingerprint_sha256",
        ),
        "elastic profile fingerprint differs",
    )
    raw_bindings = parents.get("endpoint_service_raw")
    _require(
        isinstance(raw_bindings, list) and len(raw_bindings) == 4,
        "four paired endpoint calibration artifacts are required",
    )
    artifacts = []
    for index, binding in enumerate(raw_bindings):
        _require(isinstance(binding, Mapping), "raw binding is not an object")
        path = _resolve_bound_path(repo_root, binding, name=f"endpoint_raw[{index}]")
        artifacts.append(json.loads(path.read_text(encoding="utf-8")))
    rows = collect_service_rows(artifacts)
    elastic_keys = {(row.prompt_tokens, row.output_tokens) for row in elastic.rows}
    _require(
        {(row["prompt_tokens"], row["output_tokens"]) for row in rows}
        == elastic_keys,
        "endpoint and elastic profile geometries differ",
    )
    max_local = max(int(row["local_token_ms"]) for row in rows)
    max_remote = max(int(row["remote_prefill_token_ms"]) for row in rows)
    max_kv = max(row.remote_kv_bytes for row in elastic.rows)
    measurement = manifest.get("measurement")
    _require(isinstance(measurement, Mapping), "measurement contract missing")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "profile_id": profile_id,
        "elastic_profile_fingerprint_sha256": elastic.fingerprint_sha256,
        "workload_manifest_sha256": _sha256(manifest_path),
        "deployment_scope": "calibration_only",
        "default_e2e_deadline_ms": float(measurement["e2e_slo_ms"]),
        "controller": {
            "local_token_ms_window": 6 * max_local,
            "remote_prefill_token_ms_window": 4 * max_remote,
            "remote_kv_bytes_window": 4 * max_kv,
            "remote_semantic_ops_window": 4,
            "feedback_history": 16,
            "feedback_quantile": 0.9,
            "minimum_feedback": 2,
            "route_margin_ms": 5.0,
            "feedback_fresh_ns": 5_000_000_000,
            "probe_after_ns": 5_000_000_000,
            "denied_probe_after_ns": 10_000_000_000
        },
        "rows": rows,
    }
    payload["fingerprint_sha256"] = endpoint_service_profile_fingerprint(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite endpoint profile")
    payload = build_from_manifest(args.manifest, profile_id=args.profile_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    loaded = load_endpoint_service_profile(args.output.resolve())
    print(json.dumps({
        "fingerprint_sha256": loaded.fingerprint_sha256,
        "profile_id": loaded.profile_id,
        "rows": len(loaded.rows),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
