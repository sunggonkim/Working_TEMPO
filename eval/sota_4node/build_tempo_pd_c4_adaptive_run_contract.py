#!/usr/bin/env python3
"""Bind all post-C4 calibration artifacts into one live-screen contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

from eval.sota_4node import analyze_tempo_pd_c4_fixed_phase as analyzer
from eval.sota_4node import build_tempo_pd_c4_adaptive_screen_manifest as screen
from eval.sota_4node import build_tempo_pd_c4_calibrated_profiles as profiles
from eval.sota_4node import replay_tempo_pd_c4_calibrated_controller as replay_module
from eval.sota_4node import verify_tempo_pd_c4_adaptive_implementation as implementation
from eval.sota_4node import verify_tempo_pd_c4_implementation as fixed_implementation
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile


SCHEMA = "tempo-pd-c4-adaptive-screen-run-contract-v2"
ADAPTIVE_FIXED_RUNTIME_ENVIRONMENT = {
    "TEMPO_PD_C4_ADAPTIVE_APPROVED": "YES",
    "TEMPO_PD_C4_PHASE_DURATION_MS": "8000",
    "TEMPO_PD_C4_COOLDOWN_S": "2",
    "TEMPO_PD_BENCHMARK_COLD_MEASURED": "0",
    "TEMPO_PD_BENCHMARK_RESET_DECODER_APC": "1",
    "TEMPO_VLLM_DECODER_PREFIX_CACHING": "1",
    "TEMPO_PD_FRONTEND_PAIR_POLICY": (
        "tempo-min-outstanding-decode-tokens-v1"),
    "TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY": "1",
    "TEMPO_PD_DECODER_REUSE_ITEMS": "all",
    "TEMPO_PD_FORWARD_TOKEN_IDS": "0",
    "TEMPO_PD_PROXY_KV_CONTROL_OVERLAP": "0",
    "TEMPO_PD_REMOTE_DECODE_PLACEMENT": "paired",
    "TEMPO_PD_PROXY_TOKENIZER_PLACEMENT": "round_robin",
    "TEMPO_LMCACHE_NIXL_BACKEND": "UCX",
    "TEMPO_LMCACHE_LOCAL_CPU_GB": "16",
    "TEMPO_LMCACHE_PD_BUFFER_BYTES": "2147483648",
    "TEMPO_ELASTIC_PD_PROFILE_SCOPE": "screen_only",
    "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "adaptive",
    "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "instant_score_v1",
    "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "0",
    "TEMPO_PD_PRESSURE_MODE": "disabled",
    "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "disabled",
    "TEMPO_VLLM_MAX_NUM_SEQS": "16",
    "TEMPO_VLLM_ASYNC_SCHEDULING": "0",
    "TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS": "32768",
    "TEMPO_VLLM_SCHEDULING_POLICY": "fcfs",
    "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "0",
    "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY": "0",
    "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY": "0",
    "TEMPO_PD_LONG_REMOTE_CATCHUP_MIN_PROMPT_TOKENS": "0",
    "TEMPO_PD_MEDIAN_GUARD_PRIORITY": "0",
    "TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY": "0",
    "TEMPO_PD_REMOTE_CATCHUP_MIN_OUTPUT_TOKENS": "256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_bound(
    path: Path, expected_sha256: str, *, name: str,
) -> tuple[Path, dict[str, object]]:
    path = path.resolve()
    fixed_implementation._canonical_sha(
        expected_sha256, name=f"{name} SHA-256")
    _require(path.is_file(), f"{name} is missing")
    _require(_sha256(path) == expected_sha256, f"{name} digest differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be an object")
    return path, value


def _binding(
    path: Path, *, fingerprint_sha256: str | None = None,
) -> dict[str, str]:
    value = {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}
    if fingerprint_sha256 is not None:
        value["fingerprint_sha256"] = fingerprint_sha256
    return value


def _resolve_manifest_artifact(
    entry: object, *, repo_root: Path, name: str,
) -> Path:
    _require(isinstance(entry, dict) and set(entry) == {"path", "sha256"},
             f"adaptive manifest {name} binding differs")
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    _require(path.is_file() and _sha256(path) == entry["sha256"],
             f"adaptive manifest {name} digest differs")
    return path


def build_run_contract(
    *, analysis_path: Path, analysis_sha256: str,
    manifest_path: Path, manifest_sha256: str,
    elastic_path: Path, elastic_sha256: str,
    endpoint_path: Path, endpoint_sha256: str,
    receipt_path: Path, receipt_sha256: str,
    replay_path: Path, replay_sha256: str,
    implementation_path: Path, implementation_sha256: str,
    repo_root: Path,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    analysis_path, analysis = _load_bound(
        analysis_path, analysis_sha256, name="C4 analysis")
    _require(
        analysis.get("schema") == analyzer.SCHEMA
        and analysis.get("fingerprint_sha256")
        == analyzer._analysis_fingerprint(analysis)
        and analysis.get("authorizes_profile_fit") is True
        and analyzer.analyze(
            Path(str(analysis["source_node_result"]["path"])),
            expected_result_sha256=analysis["source_node_result"]["sha256"],
        ) == analysis,
        "C4 analysis is not reproducible or does not authorize profile fit",
    )
    manifest_path, manifest = _load_bound(
        manifest_path, manifest_sha256, name="adaptive screen manifest")
    _require(
        manifest == screen.build_manifest(
            analysis_path, expected_analysis_sha256=analysis_sha256)
        and manifest.get("fingerprint_sha256")
        == screen.manifest_fingerprint(manifest),
        "adaptive screen manifest does not reproduce",
    )
    elastic_path, elastic_raw = _load_bound(
        elastic_path, elastic_sha256, name="calibrated Elastic profile")
    endpoint_path, endpoint_raw = _load_bound(
        endpoint_path, endpoint_sha256, name="calibrated endpoint profile")
    receipt_path, receipt = _load_bound(
        receipt_path, receipt_sha256, name="calibrated profile receipt")
    elastic = load_elastic_profile(elastic_path)
    endpoint = load_endpoint_service_profile(endpoint_path)
    _require(
        receipt.get("schema") == profiles.SCHEMA
        and receipt.get("fingerprint_sha256")
        == profiles._receipt_fingerprint(receipt)
        and receipt.get("formula_id") == profiles.FORMULA_ID
        and elastic.deployment_scope == "screen_only"
        and endpoint.deployment_scope == "calibration_only"
        and endpoint.elastic_profile_fingerprint_sha256
        == elastic.fingerprint_sha256
        and endpoint.workload_manifest_sha256 == manifest_sha256,
        "calibrated profile lineage differs",
    )
    rebuilt = profiles.build_profiles(
        analysis_path=analysis_path,
        expected_analysis_sha256=analysis_sha256,
        workload_manifest_path=manifest_path,
        expected_workload_manifest_sha256=manifest_sha256,
        elastic_profile_id=elastic.profile_id,
        endpoint_profile_id=endpoint.profile_id,
    )
    _require(rebuilt == (elastic_raw, endpoint_raw, receipt),
             "calibrated profiles or receipt do not reproduce")
    replay_path, replay_value = _load_bound(
        replay_path, replay_sha256, name="offline controller replay")
    reproduced_replay = replay_module.replay(
        analysis_path=analysis_path,
        analysis_sha256=analysis_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        elastic_path=elastic_path,
        elastic_sha256=elastic_sha256,
        endpoint_path=endpoint_path,
        endpoint_sha256=endpoint_sha256,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha256,
    )
    _require(
        replay_value == reproduced_replay
        and replay_value.get("fingerprint_sha256")
        == replay_module.replay_fingerprint(replay_value)
        and replay_value.get("live_adaptive_screen_authorized") is True
        and all(replay_value.get("screen_gates", {}).values()),
        "offline replay does not reproducibly authorize a live screen",
    )

    fixed_binding = analysis.get("implementation_contract")
    _require(isinstance(fixed_binding, dict),
             "C4 analysis fixed implementation binding is missing")
    fixed_path = Path(str(fixed_binding["path"])).resolve()
    _require(fixed_path.is_file() and _sha256(fixed_path) == fixed_binding["sha256"],
             "C4 fixed implementation digest differs")
    implementation_path, _raw_implementation = _load_bound(
        implementation_path, implementation_sha256,
        name="adaptive implementation contract",
    )
    implementation_value = implementation.verify_contract(
        repo_root=repo_root,
        contract_path=implementation_path,
        expected_sha256=implementation_sha256,
        fixed_c4_contract=fixed_path,
    )
    source_workload = _resolve_manifest_artifact(
        manifest.get("source_workload"), repo_root=repo_root,
        name="source workload")
    _require(
        analysis.get("source_workload") == {
            "path": str(source_workload),
            "sha256": _sha256(source_workload),
        },
        "adaptive source workload differs from C4 analysis",
    )
    source_result = Path(str(analysis["source_node_result"]["path"])).resolve()
    value: dict[str, object] = {
        "schema": SCHEMA,
        "purpose": "calibration-only four-arm adaptive endpoint-feedback screen",
        "source_node_result": _binding(source_result),
        "source_workload": _binding(source_workload),
        "analysis": _binding(
            analysis_path,
            fingerprint_sha256=str(analysis["fingerprint_sha256"])),
        "phase_manifest": _binding(
            manifest_path,
            fingerprint_sha256=str(manifest["fingerprint_sha256"])),
        "elastic_profile": _binding(
            elastic_path,
            fingerprint_sha256=elastic.fingerprint_sha256),
        "endpoint_service_profile": _binding(
            endpoint_path,
            fingerprint_sha256=endpoint.fingerprint_sha256),
        "profile_receipt": _binding(
            receipt_path,
            fingerprint_sha256=str(receipt["fingerprint_sha256"])),
        "offline_replay": _binding(
            replay_path,
            fingerprint_sha256=str(replay_value["fingerprint_sha256"])),
        "fixed_c4_implementation_contract": {
            "path": str(fixed_path),
            "sha256": fixed_binding["sha256"],
            "fingerprint_sha256": fixed_binding["fingerprint_sha256"],
        },
        "adaptive_implementation_contract": _binding(
            implementation_path,
            fingerprint_sha256=str(
                implementation_value["fingerprint_sha256"])),
        "profile_fit_formula": profiles.FORMULA_ID,
        "fixed_runtime_environment": dict(sorted(
            ADAPTIVE_FIXED_RUNTIME_ENVIRONMENT.items())),
        "slurm": manifest["slurm"],
        "transport": "LMCacheConnectorV1:UCX",
        "unchanged_pd_data_plane": True,
        "offline_replay_authorized": True,
        "controller_parameter_search_allowed": False,
        "calibration_only": True,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "independent_validation_required": True,
    }
    value["fingerprint_sha256"] = contract_fingerprint(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "analysis", "manifest", "elastic", "endpoint", "receipt", "replay",
        "implementation",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite run contract")
    value = build_run_contract(
        analysis_path=args.analysis,
        analysis_sha256=args.analysis_sha256,
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        elastic_path=args.elastic,
        elastic_sha256=args.elastic_sha256,
        endpoint_path=args.endpoint,
        endpoint_sha256=args.endpoint_sha256,
        receipt_path=args.receipt,
        receipt_sha256=args.receipt_sha256,
        replay_path=args.replay,
        replay_sha256=args.replay_sha256,
        implementation_path=args.implementation,
        implementation_sha256=args.implementation_sha256,
        repo_root=args.repo_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "fingerprint_sha256": value["fingerprint_sha256"],
        "sha256": _sha256(args.output),
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
