#!/usr/bin/env python3
"""Promote an evidence-bound held-out endpoint proxy into frozen profiles.

This command is deliberately offline.  It validates the held-out workload,
the four named C4 source streams, the official LMCache route identity, and the
current calibration profile before writing a new receipt plus frozen endpoint
and global profiles.  It never edits an existing artifact and never grants a
performance claim to a proxy row.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import (
    CacheResidency,
    endpoint_service_profile_fingerprint,
    load_endpoint_service_profile,
)
from tempo.pd_global_profile import (
    SERVICE_PROXY_POLICY_ID,
    global_profile_fingerprint,
    load_global_profile,
)


SCHEMA = "tempo-go-heldout-frozen-proxy-receipt-v1"
HELDOUT_SCHEMA = "tempo-go-c5-heldout-manifest-v1"
TRANSPORT = "LMCacheConnectorV1:UCX"
LOCAL_ROUTE = "decoder_local_chunked_prefill"
REMOTE_ROUTE = "official_lmcache_remote_prefill"
P_ONLY = "prefill_only"
MISS = "confirmed_miss"
REMOTE_HEAD_PROOF = "official_lmcache_proxy_single_prefill_token"
REQUIRED_RAW_ROLES = {
    "00_local_r0.raw.json": ("local", "always_local", LOCAL_ROUTE),
    "01_remote_r0.raw.json": (
        "remote", "official_lmcache_remote", REMOTE_ROUTE),
    "06_remote_r1.raw.json": (
        "remote", "official_lmcache_remote", REMOTE_ROUTE),
    "07_local_r1.raw.json": ("local", "always_local", LOCAL_ROUTE),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, *, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest_and_workload(
    manifest_path: Path, workload_path: Path, model_config: Path,
) -> tuple[dict[str, object], list[dict[str, object]], dict[tuple[int, int], int]]:
    manifest = _json(manifest_path, name="held-out manifest")
    _require(manifest.get("schema") == "tempo-go-contention-manifest-v1",
             "held-out manifest schema differs")
    _require(manifest.get("transport") == TRANSPORT,
             "held-out manifest transport differs")
    _require(manifest.get("native_only") is True,
             "held-out manifest is not native-only")
    _require(manifest.get("performance_claim_allowed") is False,
             "held-out manifest authorizes a performance claim")
    heldout = manifest.get("heldout")
    _require(isinstance(heldout, Mapping)
             and heldout.get("schema") == HELDOUT_SCHEMA,
             "held-out manifest provenance is missing")
    _require(Path(str(manifest["validation_workload"]["path"])).resolve()
             == workload_path.resolve(),
             "held-out workload path differs from manifest")
    _require(manifest["validation_workload"]["sha256"] == _sha256(workload_path),
             "held-out workload SHA differs from manifest")
    _require(manifest.get("model_config_sha256") == _sha256(model_config),
             "held-out model-config SHA differs")
    rows = []
    with workload_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, dict),
                     f"held-out workload row {line_number} is not an object")
            _require(
                set(value) == {
                    "request_id", "prompt", "max_tokens", "arrival_offset_ms",
                },
                f"held-out workload row {line_number} fields differ",
            )
            rows.append(value)
    _require(rows, "held-out workload is empty")
    geometry_counts: dict[tuple[int, int], int] = defaultdict(int)
    for row in rows:
        _require(type(row["max_tokens"]) is int and row["max_tokens"] >= 2,
                 "held-out output geometry is invalid")
        geometry_counts[(0, int(row["max_tokens"]))] += 1
    return manifest, rows, dict(geometry_counts)


def _derive_geometries(
    rows: list[dict[str, object]], model_path: Path,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], dict[str, int]]]:
    """Derive actual prompt/output geometries with the pinned local tokenizer."""

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "held-out promotion requires the pinned transformers environment") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_path.resolve(), local_files_only=True, trust_remote_code=False)
    geometries: set[tuple[int, int]] = set()
    states: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"miss": 0, "p_only": 0})
    for row in rows:
        request_id = str(row["request_id"])
        prompt_tokens = len(tokenizer.encode(
            str(row["prompt"]), add_special_tokens=False))
        output_tokens = int(row["max_tokens"])
        geometry = (prompt_tokens, output_tokens)
        geometries.add(geometry)
        if "-cache-miss-measured-" in request_id:
            states[geometry]["miss"] += 1
        elif "-cache-p-only-measured-" in request_id:
            states[geometry]["p_only"] += 1
        else:
            raise ValueError(f"held-out request lacks cache contract: {request_id}")
    _require(geometries, "held-out geometry set is empty")
    return geometries, dict(states)


def _validate_c4_source_manifest(
    repo_root: Path, c4_manifest_path: Path, raw_paths: list[Path],
) -> None:
    manifest = _json(c4_manifest_path, name="C4 source manifest")
    _require(manifest.get("transport") == TRANSPORT,
             "C4 source manifest transport differs")
    _require(manifest.get("unchanged_pd_data_plane") is True,
             "C4 source manifest data-plane contract differs")
    parents = manifest.get("calibration_parents")
    _require(isinstance(parents, Mapping), "C4 source parent bindings are missing")
    entries = parents.get("endpoint_service_raw")
    _require(isinstance(entries, list), "C4 endpoint raw bindings are missing")
    expected = {
        (repo_root / str(item["path"])).resolve(): str(item["sha256"])
        for item in entries if isinstance(item, Mapping)
    }
    _require(len(expected) == len(entries), "C4 endpoint raw bindings are malformed")
    for path in raw_paths:
        _require(path.resolve() in expected,
                 f"source raw is not bound by the C4 manifest: {path}")
        _require(_sha256(path.resolve()) == expected[path.resolve()],
                 f"C4 source raw SHA differs: {path}")


def _raw_records(
    path: Path, *, model_config_sha256: str,
) -> tuple[str, str, dict[tuple[int, int], list[dict[str, str]]]]:
    raw = _json(path, name=f"source raw {path.name}")
    _require(raw.get("schema") == "tempo-pd-stream-metrics-raw-1",
             f"source raw schema differs: {path}")
    _require(raw.get("evidence") == "actual_vllm_pd_router_client_stream",
             f"source raw evidence differs: {path}")
    validation = raw.get("validation")
    _require(isinstance(validation, Mapping)
             and validation.get("all_streams_valid") is True
             and validation.get("router_decisions_exact") is True
             and validation.get("performance_claim_allowed") is True,
             f"source raw validation is not complete: {path}")
    model = raw.get("model")
    _require(isinstance(model, Mapping)
             and model.get("config_sha256") == model_config_sha256,
             f"source raw model identity differs: {path}")
    contract = raw.get("elastic_balanced_contract")
    _require(isinstance(contract, Mapping)
             and contract.get("one_live_server_epoch") is True,
             f"source raw live-epoch contract differs: {path}")
    metric_contract = raw.get("metric_contract")
    _require(isinstance(metric_contract, Mapping)
             and metric_contract.get("remote_first_token")
             == "official LMCache proxy max_tokens=1 head event",
             f"source raw remote-head contract differs: {path}")
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(requests, list) and isinstance(decisions, list),
             f"source raw request/decision lists are missing: {path}")
    request_map = {item.get("request_id"): item for item in requests
                   if isinstance(item, Mapping)}
    decision_map = {item.get("request_id"): item for item in decisions
                   if isinstance(item, Mapping)}
    _require(len(request_map) == len(requests)
             and len(decision_map) == len(decisions)
             and set(request_map) == set(decision_map),
             f"source raw request/decision IDs are not exact: {path}")
    name = path.name
    _require(name in REQUIRED_RAW_ROLES,
             f"source raw filename is not one of the four frozen roles: {path}")
    expected_role, expected_arm, expected_route = REQUIRED_RAW_ROLES[name]
    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for request_id, request in request_map.items():
        decision = decision_map[request_id]
        _require(decision.get("arm") == expected_arm
                 and decision.get("route") == expected_route,
                 f"source raw route identity differs: {path} {request_id}")
        _require(decision.get("cache_residency") == P_ONLY
                 and decision.get("cache_residency_source")
                 == "confirmed_completion_event",
                 f"source raw cache evidence is not P_ONLY: {path} {request_id}")
        _require(request.get("valid") is True,
                 f"source request is invalid: {path} {request_id}")
        prompt_tokens = decision.get("prompt_tokens")
        output_tokens = decision.get("output_tokens")
        _require(type(prompt_tokens) is int and type(output_tokens) is int
                 and prompt_tokens >= 2 and output_tokens >= 2,
                 f"source router geometry is invalid: {path} {request_id}")
        _require(request.get("requested_max_tokens") == output_tokens,
                 f"source requested output differs from router: {path} {request_id}")
        usage = request.get("usage")
        _require(isinstance(usage, Mapping)
                 and usage.get("prompt_tokens") in {prompt_tokens, prompt_tokens + 1},
                 f"source prompt geometry differs: {path} {request_id}")
        if expected_role == "remote" and usage.get("prompt_tokens") == prompt_tokens + 1:
            proofs = request.get("output_token_proofs")
            _require(isinstance(proofs, list)
                     and REMOTE_HEAD_PROOF in proofs,
                     f"source remote head-token proof is missing: {path} {request_id}")
        _require(isinstance(request.get("prompt_sha256"), str)
                 and isinstance(request.get("output_text_sha256"), str),
                 f"source output identity is missing: {path} {request_id}")
        grouped[(prompt_tokens, output_tokens)].append({
            "prompt_sha256": str(request["prompt_sha256"]),
            "output_text_sha256": str(request["output_text_sha256"]),
        })
    return expected_role, expected_route, dict(grouped)


def _validate_source_coverage(
    raw_paths: list[Path], required_geometries: set[tuple[int, int]],
    *, model_config_sha256: str,
) -> dict[str, object]:
    coverage: dict[tuple[int, int], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"local": [], "remote": []})
    source_receipts = []
    for path in raw_paths:
        role, route, grouped = _raw_records(
            path, model_config_sha256=model_config_sha256)
        source_receipts.append({
            "path": str(path.resolve()),
            "sha256": _sha256(path.resolve()),
            "role": role,
            "route": route,
        })
        for geometry, records in grouped.items():
            coverage[geometry][role].extend(records)
    report = {}
    for geometry in sorted(required_geometries):
        local = coverage[geometry]["local"]
        remote = coverage[geometry]["remote"]
        _require(len(local) >= 2 and len(remote) >= 2,
                 f"source coverage needs two local and remote samples: {geometry}")
        local_hashes = {item["output_text_sha256"] for item in local}
        remote_hashes = {item["output_text_sha256"] for item in remote}
        _require(local_hashes == remote_hashes,
                 f"cross-route output hashes differ: {geometry}")
        report[f"{geometry[0]}x{geometry[1]}"] = {
            "local_samples": len(local),
            "remote_samples": len(remote),
            "local_output_hashes": sorted(local_hashes),
            "remote_output_hashes": sorted(remote_hashes),
            "cache_residency": P_ONLY,
        }
    return {"raw": source_receipts, "geometry": report}


def _frozen_endpoint(
    base_path: Path, heldout_manifest_sha256: str, profile_id: str,
) -> dict[str, object]:
    base = _json(base_path, name="calibration endpoint profile")
    _require(base.get("deployment_scope") == "calibration_only",
             "source endpoint profile is not calibration-only")
    rows = base.get("rows")
    _require(isinstance(rows, list) and rows,
             "source endpoint profile rows are missing")
    _require(all(row.get("cache_residency") == P_ONLY for row in rows),
             "source endpoint profile contains an unapproved residency")
    value = dict(base)
    value["profile_id"] = profile_id
    value["deployment_scope"] = "frozen_validation"
    value["workload_manifest_sha256"] = heldout_manifest_sha256
    value.pop("fingerprint_sha256", None)
    value["fingerprint_sha256"] = endpoint_service_profile_fingerprint(value)
    return value


def _frozen_global(
    base_path: Path, *, endpoint: Mapping[str, object], elastic_fingerprint: str,
    heldout_manifest_sha256: str, model_config_sha256: str,
    policy: Mapping[str, object], profile_id: str,
) -> dict[str, object]:
    value = _json(base_path, name="discovery global profile")
    _require(value.get("deployment_scope") == "discovery",
             "source global profile is not discovery scope")
    identity = dict(value["identity"])
    identity.update({
        "endpoint_profile_id": endpoint["profile_id"],
        "endpoint_profile_fingerprint_sha256": endpoint["fingerprint_sha256"],
        "endpoint_profile_deployment_scope": "frozen_validation",
        "elastic_profile_fingerprint_sha256": elastic_fingerprint,
        "workload_manifest_sha256": heldout_manifest_sha256,
        "model_config_sha256": model_config_sha256,
    })
    controller = dict(value["controller"])
    controller["frozen_service_proxy_policy"] = dict(policy)
    value["profile_id"] = profile_id
    value["deployment_scope"] = "frozen_validation"
    value["identity"] = identity
    value["controller"] = controller
    value.pop("fingerprint_sha256", None)
    value["fingerprint_sha256"] = global_profile_fingerprint(value)
    return value


def build_artifacts(args: argparse.Namespace) -> dict[str, object]:
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest.resolve()
    workload_path = args.workload.resolve()
    model_path = args.model.resolve()
    model_config = (model_path / "config.json").resolve()
    _require(repo_root in manifest_path.parents and repo_root in workload_path.parents,
             "held-out inputs must be below repository")
    _require(repo_root in model_path.parents and model_config.is_file(),
             "held-out model must be below repository and contain config.json")
    _require(args.output_dir.resolve().parent == repo_root / "results",
             "frozen proxy output must be a direct child of results")
    if args.output_dir.exists():
        _require(args.output_dir.is_dir() and not any(args.output_dir.iterdir()),
                 f"refusing to overwrite nonempty output directory {args.output_dir}")
    raw_paths = [path.resolve() for path in args.raw]
    _require(sorted(path.name for path in raw_paths) == sorted(REQUIRED_RAW_ROLES),
             "exactly the four named C4 source raw files are required")
    _validate_c4_source_manifest(repo_root, args.c4_manifest.resolve(), raw_paths)

    manifest, rows, _ = _manifest_and_workload(
        manifest_path, workload_path, model_config)
    geometries, state_counts = _derive_geometries(rows, model_path)
    _require(set(geometries) == {
        (512, 16), (2048, 256), (4094, 16), (4094, 128),
    }, "held-out geometry set is not the pinned output=128 workload")
    _require(state_counts[(4094, 128)]["p_only"] > 0,
             "held-out workload lacks the P_ONLY validation stream")

    elastic = load_elastic_profile(args.elastic_profile.resolve())
    endpoint = load_endpoint_service_profile(args.endpoint_profile.resolve())
    global_profile = load_global_profile(args.global_profile.resolve())
    _require(elastic.identity.remote_backend
             == "official-lmcacheconnectorv1-nixl-ucx",
             "Elastic profile remote backend is not official LMCache UCX")
    _require(elastic.identity.topology_id
             == "perlmutter-4n-2replica-tp4-prefill-tp4-decode",
             "Elastic profile topology identity differs")
    _require(elastic.identity.model_revision == _sha256(model_config),
             "Elastic profile model revision differs from model config")
    _require(endpoint.elastic_profile_fingerprint_sha256
             == elastic.fingerprint_sha256,
             "endpoint and Elastic profile identities differ")
    _require(global_profile.identity.elastic_profile_fingerprint_sha256
             == elastic.fingerprint_sha256,
             "global and Elastic profile identities differ")
    for prompt_tokens, output_tokens in sorted(geometries):
        _require(elastic.exact_row(prompt_tokens, output_tokens) is not None,
                 f"Elastic profile lacks held-out geometry {(prompt_tokens, output_tokens)}")
        _require(endpoint.exact_row(
            prompt_tokens, output_tokens, CacheResidency.P_ONLY),
            f"endpoint profile lacks P_ONLY source row {(prompt_tokens, output_tokens)}")
    coverage = _validate_source_coverage(
        raw_paths, geometries, model_config_sha256=_sha256(model_config))

    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "transport": TRANSPORT,
        "deployment_scope": {
            "receipt": "calibration_only",
            "endpoint_profile": "frozen_validation",
            "global_profile": "frozen_validation",
            "performance_claim_allowed": False,
        },
        "heldout_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "workload_path": str(workload_path),
            "workload_sha256": _sha256(workload_path),
            "request_count": len(rows),
            "geometry_set": [list(item) for item in sorted(geometries)],
            "cache_state_counts": {
                f"{geometry[0]}x{geometry[1]}": counts
                for geometry, counts in sorted(state_counts.items())
            },
        },
        "model": {
            "path": str(model_path),
            "config_path": str(model_config),
            "config_sha256": _sha256(model_config),
        },
        "calibration_endpoint_profile": {
            "path": str(args.endpoint_profile.resolve()),
            "sha256": _sha256(args.endpoint_profile.resolve()),
            "fingerprint_sha256": endpoint.fingerprint_sha256,
            "numeric_rows_unchanged_in_frozen_clone": True,
        },
        "calibration_elastic_profile": {
            "path": str(args.elastic_profile.resolve()),
            "sha256": _sha256(args.elastic_profile.resolve()),
            "fingerprint_sha256": elastic.fingerprint_sha256,
        },
        "c4_source_manifest": {
            "path": str(args.c4_manifest.resolve()),
            "sha256": _sha256(args.c4_manifest.resolve()),
        },
        "source_evidence": coverage,
        "proxy_contract": {
            "policy_id": SERVICE_PROXY_POLICY_ID,
            "allowed_lookup_modes": [
                "exact", "same_residency_geometry_ceiling",
                "miss_via_prefill_only_geometry_ceiling",
            ],
            "allowed_cache_residencies": [MISS, P_ONLY],
            "allowed_remote_cache_residencies": [P_ONLY],
            "allowed_geometries": [list(item) for item in sorted(geometries)],
            "miss_remote_route": "deny",
            "p_only_remote_route": "allow_only_with_exact_row",
            "numeric_rows_unchanged": True,
            "proxy_is_not_exact": True,
            "performance_claim_allowed": False,
        },
    }
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    receipt_path = args.output_dir / "heldout_proxy_source_receipt.json"
    _write_new(receipt_path, receipt)
    receipt_sha256 = _sha256(receipt_path)

    endpoint_value = _frozen_endpoint(
        args.endpoint_profile.resolve(),
        _sha256(manifest_path),
        args.endpoint_profile_id,
    )
    endpoint_path = args.output_dir / "frozen_endpoint_service_profile.json"
    _write_new(endpoint_path, endpoint_value)
    frozen_endpoint = load_endpoint_service_profile(endpoint_path)
    _require(
        [row.key for row in frozen_endpoint.rows]
        == [row.key for row in endpoint.rows]
        and [row.local_token_ms for row in frozen_endpoint.rows]
        == [row.local_token_ms for row in endpoint.rows]
        and [row.remote_prefill_token_ms for row in frozen_endpoint.rows]
        == [row.remote_prefill_token_ms for row in endpoint.rows],
        "frozen endpoint clone changed numeric service rows",
    )
    policy = {
        "policy_id": SERVICE_PROXY_POLICY_ID,
        "allowed_lookup_modes": receipt["proxy_contract"][
            "allowed_lookup_modes"],
        "allowed_cache_residencies": receipt["proxy_contract"][
            "allowed_cache_residencies"],
        "allowed_remote_cache_residencies": receipt["proxy_contract"][
            "allowed_remote_cache_residencies"],
        "allowed_geometries": receipt["proxy_contract"]["allowed_geometries"],
        "proxy_is_not_exact": True,
        "numeric_rows_unchanged": True,
        "performance_claim_allowed": False,
        "endpoint_profile_id": frozen_endpoint.profile_id,
        "endpoint_profile_fingerprint_sha256": frozen_endpoint.fingerprint_sha256,
        "calibration_receipt_sha256": receipt_sha256,
    }
    global_value = _frozen_global(
        args.global_profile.resolve(),
        endpoint={
            "profile_id": frozen_endpoint.profile_id,
            "fingerprint_sha256": frozen_endpoint.fingerprint_sha256,
        },
        elastic_fingerprint=elastic.fingerprint_sha256,
        heldout_manifest_sha256=_sha256(manifest_path),
        model_config_sha256=_sha256(model_config),
        policy=policy,
        profile_id=args.global_profile_id,
    )
    global_path = args.output_dir / "frozen_global_profile.json"
    _write_new(global_path, global_value)
    frozen_global = load_global_profile(global_path)
    _require(frozen_global.service_proxy_policy() is not None,
             "frozen global profile lost service proxy policy")
    receipt["promoted_profiles"] = {
        "endpoint": {
            "path": str(endpoint_path),
            "sha256": _sha256(endpoint_path),
            "fingerprint_sha256": frozen_endpoint.fingerprint_sha256,
        },
        "global": {
            "path": str(global_path),
            "sha256": _sha256(global_path),
            "fingerprint_sha256": frozen_global.fingerprint_sha256,
        },
        "calibration_receipt_sha256": receipt_sha256,
    }
    _write_new(args.output_dir / "promotion_summary.json", receipt)
    return {
        "output_dir": str(args.output_dir),
        "receipt": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "endpoint_profile": str(endpoint_path),
        "endpoint_fingerprint_sha256": frozen_endpoint.fingerprint_sha256,
        "global_profile": str(global_path),
        "global_fingerprint_sha256": frozen_global.fingerprint_sha256,
        "geometry_count": len(geometries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--elastic-profile", type=Path, required=True)
    parser.add_argument("--endpoint-profile", type=Path, required=True)
    parser.add_argument("--global-profile", type=Path, required=True)
    parser.add_argument("--c4-manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint-profile-id",
        default="tempo-pd-endpoint-qwen25-heldout-output128-frozen-v1",
    )
    parser.add_argument(
        "--global-profile-id",
        default="tempo-go-qwen25-perlmutter-heldout-output128-frozen-v1",
    )
    args = parser.parse_args()
    result = build_artifacts(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
