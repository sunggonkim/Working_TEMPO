#!/usr/bin/env python3
"""Fail closed unless the frozen C4 implementation and environment match."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess
from typing import Mapping


SCHEMA = "tempo-pd-c4-implementation-contract-v1"
REQUIRED_FILES = frozenset({
    "eval/sota_4node/build_tempo_pd_c4_phase_manifest.py",
    "eval/sota_4node/c4_fixed_phase_pd_node_entry.sh",
    "eval/sota_4node/require_perlmutter_4node_4h_interactive.sh",
    "eval/sota_4node/run_tempo_pd_c4_fixed_phase_in_allocation.sh",
    "eval/sota_4node/run_tempo_pd_c4_fixed_phase_client.py",
    "eval/sota_4node/run_tempo_pd_contention_fixed_client.py",
    "eval/sota_4node/run_tempo_pd_elastic_stream_metrics.py",
    "eval/sota_4node/run_tempo_pd_elastic_stream_metrics_cache_protocol.py",
    "eval/sota_4node/run_tempo_pd_stream_metrics_v1.py",
    "eval/sota_4node/tempo_pd_elastic_frontend.py",
    "eval/sota_4node/tempo_pd_elastic_router.py",
    "eval/sota_4node/tempo_pd_endpoint_probe.py",
    "eval/sota_4node/tempo_pd_frontend_v1.py",
    "eval/sota_4node/tempo_pd_proxy_cache_control.py",
    "eval/sota_4node/vllm_lmcache_chunk256_node_v7.py",
    "eval/sota_4node/vllm_lmcache_elastic_pd_node.py",
    "eval/sota_4node/vllm_lmcache_live_pd_node_v1.py",
    "eval/sota_4node/vllm_lmcache_live_pd_node_v2.py",
    "eval/sota_4node/vllm_lmcache_pd_c4_fixed_phase_node.py",
    "eval/sota_4node/vllm_lmcache_pd_contention_node.py",
    "eval/sota_4node/vllm_lmcache_tempo_pd_perf_node_v1.py",
    "eval/sota_4node/vllm_tempo_cache_control.py",
    "tempo/cassini_endpoint.py",
    "tempo/pd_cache_state_protocol.py",
    "tempo/pd_contention_workload.py",
    "tempo/pd_decoder_cache_evidence.py",
    "tempo/pd_endpoint_evidence.py",
    "third_party/lmcache/examples/disagg_prefill/disagg_proxy_server.py",
    "third_party/lmcache/lmcache/integration/vllm/vllm_v1_adapter.py",
    "third_party/lmcache/lmcache/v1/cache_engine.py",
    "third_party/lmcache/lmcache/v1/storage_backend/pd_backend_async.py",
})
PACKAGE_NAMES = ("vllm", "lmcache", "torch", "transformers")


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


def contract_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_file(repo_root: Path, raw: object, *, name: str) -> Path:
    _require(type(raw) is str and raw, f"{name} path is missing")
    pure = PurePosixPath(raw)
    _require(
        not pure.is_absolute()
        and ".." not in pure.parts
        and str(pure) == raw,
        f"{name} path is not canonical and relative",
    )
    path = (repo_root / raw).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{name} escapes the repository") from exc
    _require(path.is_file(), f"{name} is missing")
    return path


def _git_head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    value = completed.stdout.strip()
    _require(
        len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        "Git HEAD is invalid",
    )
    return value


def _environment_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in PACKAGE_NAMES:
        result[name] = importlib.metadata.version(name)
    return result


def verify_contract(
    *, repo_root: Path, contract_path: Path, expected_sha256: str,
    phase_manifest: Path,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    contract_path = contract_path.resolve()
    phase_manifest = phase_manifest.resolve()
    _require(repo_root.is_dir(), "repository root is missing")
    _require(contract_path.is_file(), "C4 implementation contract is missing")
    _require(
        _sha256(contract_path)
        == _canonical_sha(expected_sha256, name="contract SHA-256"),
        "C4 implementation contract digest differs",
    )
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(
        isinstance(value, dict)
        and value.get("schema") == SCHEMA
        and value.get("fingerprint_sha256") == contract_fingerprint(value),
        "C4 implementation contract schema or fingerprint differs",
    )
    _require(value.get("purpose") == "frozen C4 characterization only",
             "C4 implementation contract purpose differs")
    manifest = value.get("phase_manifest")
    _require(isinstance(manifest, dict) and set(manifest) == {"path", "sha256"},
             "C4 phase-manifest binding differs")
    bound_manifest = _relative_file(
        repo_root, manifest["path"], name="phase manifest")
    _require(
        bound_manifest == phase_manifest
        and _sha256(bound_manifest)
        == _canonical_sha(manifest["sha256"], name="phase manifest SHA-256"),
        "C4 phase-manifest implementation binding differs",
    )

    files = value.get("files")
    _require(isinstance(files, list) and bool(files),
             "C4 implementation file bindings are missing")
    observed: set[str] = set()
    for index, entry in enumerate(files):
        _require(isinstance(entry, dict) and set(entry) == {"path", "sha256"},
                 f"C4 implementation file[{index}] binding differs")
        raw_path = entry["path"]
        _require(type(raw_path) is str and raw_path not in observed,
                 "C4 implementation file paths are invalid or duplicated")
        path = _relative_file(
            repo_root, raw_path, name=f"implementation file[{index}]")
        _require(
            _sha256(path) == _canonical_sha(
                entry["sha256"], name=f"implementation file[{index}] SHA-256"),
            f"C4 implementation file drifted: {raw_path}",
        )
        observed.add(raw_path)
    _require(REQUIRED_FILES.issubset(observed),
             "C4 implementation contract omits a required file")

    repositories = value.get("git_heads")
    _require(
        isinstance(repositories, dict)
        and set(repositories) == {"repository", "third_party_lmcache"}
        and repositories["repository"] == _git_head(repo_root)
        and repositories["third_party_lmcache"]
        == _git_head(repo_root / "third_party/lmcache"),
        "C4 implementation Git provenance differs",
    )
    _require(value.get("environment_versions") == _environment_versions(),
             "C4 implementation environment versions differ")
    _require(value.get("performance_claim_allowed") is False,
             "C4 implementation contract permits a performance claim")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--phase-manifest", type=Path, required=True)
    args = parser.parse_args()
    value = verify_contract(
        repo_root=args.repo_root,
        contract_path=args.contract,
        expected_sha256=args.expected_sha256,
        phase_manifest=args.phase_manifest,
    )
    print(json.dumps({
        "schema": SCHEMA,
        "fingerprint_sha256": value["fingerprint_sha256"],
        "files": len(value["files"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
