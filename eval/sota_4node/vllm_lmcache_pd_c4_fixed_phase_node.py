#!/usr/bin/env python3
"""Run completion-backed C4 fixed phases on four actual vLLM/LMCache nodes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.sota_4node import build_tempo_pd_c4_phase_manifest as builder
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256
from eval.sota_4node import vllm_lmcache_elastic_pd_node as canonical
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as elastic
from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf
from eval.sota_4node import vllm_lmcache_pd_contention_node as contention
from eval.sota_4node import verify_tempo_pd_c4_implementation as implementation


SCHEMA = "tempo-pd-c4-fixed-phase-node-v1"
CLIENT_SCHEMA = "tempo-pd-c4-fixed-phase-client-v1"
CLIENT_MODULE = "eval.sota_4node.run_tempo_pd_c4_fixed_phase_client"
STAGE_NAME = "tempo_pd_c4_fixed_phase"
MANIFEST_ENV = "TEMPO_PD_C4_PHASE_MANIFEST"
MANIFEST_SHA_ENV = "TEMPO_PD_C4_PHASE_MANIFEST_SHA256"
IMPLEMENTATION_ENV = "TEMPO_PD_C4_IMPLEMENTATION_CONTRACT"
IMPLEMENTATION_SHA_ENV = "TEMPO_PD_C4_IMPLEMENTATION_CONTRACT_SHA256"
DEFAULT_READINESS_S = 3600.0
_FIXED_RUNTIME_ENVIRONMENT = dict(
    builder.C4_FIXED_RUNTIME_ENVIRONMENT)
_DYNAMIC_RUNTIME_ENVIRONMENT = frozenset({
    MANIFEST_ENV,
    MANIFEST_SHA_ENV,
    IMPLEMENTATION_ENV,
    IMPLEMENTATION_SHA_ENV,
    "TEMPO_PD_C4_READINESS_S",
    "TEMPO_ELASTIC_PD_PROFILE",
})
_EXPECTED_BLOCKS = (
    ("00_local_r0", "local", 0),
    ("01_remote_r0", "remote", 0),
    ("02_remote_r1", "remote", 1),
    ("03_local_r1", "local", 1),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_client_artifacts(
    artifact: object, *, client_raw_path: Path,
) -> dict[str, dict[str, str]]:
    perf._require(isinstance(artifact, dict), "C4 client artifact is malformed")
    expected_keys = [name for name, _arm, _replicate in _EXPECTED_BLOCKS]
    expected_order = [
        {"arm": arm, "replicate": replicate}
        for _name, arm, replicate in _EXPECTED_BLOCKS
    ]
    artifacts = artifact.get("artifacts")
    contracts = artifact.get("contracts")
    perf._require(
        isinstance(artifacts, dict)
        and list(artifacts) == expected_keys
        and isinstance(contracts, dict)
        and list(contracts) == expected_keys
        and artifact.get("block_order") == expected_order,
        "C4 client ABBA artifact inventory differs",
    )
    result_root = client_raw_path.resolve().parent
    validated: dict[str, dict[str, str]] = {}
    for sequence, (key, arm, replicate) in enumerate(_EXPECTED_BLOCKS):
        entry = artifacts[key]
        perf._require(
            isinstance(entry, dict) and set(entry) == {"path", "sha256"},
            f"C4 block artifact binding differs: {key}",
        )
        raw_path = entry.get("path")
        expected_sha = entry.get("sha256")
        perf._require(
            isinstance(raw_path, str) and Path(raw_path).is_absolute(),
            f"C4 block artifact path is not absolute: {key}",
        )
        path = Path(raw_path).resolve()
        try:
            path.relative_to(result_root)
        except ValueError as exc:
            raise ValueError(
                f"C4 block artifact escapes the result directory: {key}"
            ) from exc
        perf._require(path.is_file(), f"C4 block artifact is missing: {key}")
        perf._require(
            type(expected_sha) is str
            and len(expected_sha) == 64
            and all(character in "0123456789abcdef" for character in expected_sha)
            and _sha256(path) == expected_sha,
            f"C4 block artifact digest differs: {key}",
        )
        contract = contracts[key]
        perf._require(
            isinstance(contract, dict)
            and contract.get("sequence") == sequence
            and contract.get("foreground_arm") == arm
            and contract.get("replicate") == replicate
            and contract.get("all_requests_valid") is True
            and contract.get("completion_cache_evidence_exact") is True
            and contract.get("phase_aligned_endpoint_evidence") is True,
            f"C4 block contract differs: {key}",
        )
        validated[key] = {"path": str(path), "sha256": expected_sha}
    return validated


def _resolve_artifact(
    repo_root: Path, entry: object, *, name: str,
) -> Path:
    perf._require(isinstance(entry, dict), f"C4 manifest lacks {name}")
    raw_path = entry.get("path")
    perf._require(isinstance(raw_path, str) and raw_path,
                  f"C4 {name} path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), f"C4 {name} is missing")
    perf._require(_sha256(path) == entry.get("sha256"),
                  f"C4 {name} digest differs")
    return path


def _load_manifest(
    args, *, workload: Path, elastic_profile: Path,
) -> tuple[Path, dict[str, object]]:
    raw = os.environ.get(MANIFEST_ENV)
    expected_sha = os.environ.get(MANIFEST_SHA_ENV)
    perf._require(bool(raw) and bool(expected_sha),
                  "frozen C4 phase manifest is required")
    path = Path(str(raw))
    if not path.is_absolute():
        path = args.repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), "frozen C4 phase manifest is missing")
    perf._require(_sha256(path) == expected_sha,
                  "frozen C4 phase manifest digest differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    perf._require(
        value.get("schema") == builder.SCHEMA
        and value.get("fingerprint_sha256")
        == builder.manifest_fingerprint(value),
        "frozen C4 phase manifest fingerprint differs",
    )
    perf._require(
        value.get("performance_claim_allowed") is False
        and value.get("controller_tuning_allowed") is False
        and value.get("physical_switch_bottleneck_claim_allowed") is False,
        "C4 characterization manifest permits an invalid claim",
    )
    perf._require(
        value.get("transport") == "LMCacheConnectorV1:UCX"
        and value.get("fixed_arm_order")
        == ["local", "remote", "remote", "local"],
        "C4 transport/order contract differs",
    )
    source = _resolve_artifact(
        args.repo_root, value.get("source_workload"), name="source workload")
    profile = _resolve_artifact(
        args.repo_root, value.get("elastic_profile"), name="Elastic profile")
    perf._require(source == workload, "runtime C4 source workload differs")
    perf._require(profile == elastic_profile,
                  "runtime C4 Elastic profile differs")
    parents = value.get("parent_evidence")
    perf._require(isinstance(parents, dict) and len(parents) == 8,
                  "C4 parent-evidence set differs")
    for name, entry in parents.items():
        _resolve_artifact(args.repo_root, entry, name=f"parent {name}")
    protocol = value.get("cache_state_protocol")
    route_contracts = (
        protocol.get("measured_decoder_route_contracts")
        if isinstance(protocol, dict) else None)
    remote_contract = (
        route_contracts.get("official_lmcache_remote_prefill")
        if isinstance(route_contracts, dict) else None)
    perf._require(
        isinstance(protocol, dict)
        and protocol.get("fixed_arm_pair_placement")
        == "terminal_item_modulo_two_pairs"
        and protocol.get("decoder_usage_breakdown_required") is True
        and protocol.get(
            "stock_cached_tokens_without_source_breakdown_allowed") is False
        and protocol.get(
            "request_id_labels_without_completion_evidence_allowed") is False,
        "C4 manifest does not require physical cache evidence",
    )
    perf._require(
        isinstance(remote_contract, dict)
        and remote_contract.get("decoder_residency_basis")
        == "exact_local_preparation_hit_on_original_P_token_prompt"
        and remote_contract.get("local_cached_tokens_by_state", {}).get(
            "d_only") == "floor((P-1)/16)*16"
        and remote_contract.get("local_cached_tokens_by_state", {}).get(
            "both") == "floor((P-1)/16)*16",
        "C4 manifest does not require physical cache evidence",
    )
    endpoint_contract = value.get("endpoint_evidence_contract")
    perf._require(
        isinstance(endpoint_contract, dict)
        and endpoint_contract.get("schema")
        == builder.ENDPOINT_EVIDENCE_CONTRACT_SCHEMA
        and endpoint_contract.get("measurement_start_marker_required") is True
        and endpoint_contract.get(
            "publisher_pid_matches_measured_child") is True
        and endpoint_contract.get("sampling_policy")
        == "workload_start_boundary_midpoint_and_end_boundary"
        and endpoint_contract.get("phase_boundary_samples") == 7
        and endpoint_contract.get("phase_midpoint_samples") == 6
        and endpoint_contract.get("cross_host_clock_subtraction_allowed")
        is False,
        "C4 manifest does not require phase-aligned endpoint evidence",
    )
    perf._require(
        value.get("fixed_runtime_environment")
        == dict(sorted(_FIXED_RUNTIME_ENVIRONMENT.items())),
        "C4 manifest fixed runtime environment differs",
    )
    return path, value


def _client_command(
    python: Path, *, base_url: str, model: Path, workload: Path,
    output: Path, mode: str, run_id: str, request_rate: float,
    max_workers: int,
) -> list[str]:
    command = [
        str(python), "-m", CLIENT_MODULE,
        "--base-url", base_url,
        "--model", str(model),
        "--served-model-name", perf.SERVED_MODEL,
        "--workload", str(workload),
        "--output", str(output),
        "--mode", mode,
        "--run-id", run_id,
        "--max-workers", str(max_workers),
        "--request-rate", str(request_rate),
        "--timeout-s", "600",
        "--phase-duration-ms", os.environ.get(
            "TEMPO_PD_C4_PHASE_DURATION_MS", "8000"),
        "--cooldown-s", os.environ.get("TEMPO_PD_C4_COOLDOWN_S", "2"),
    ]
    probe_urls = os.environ.get(contention.PROBE_URLS_ENV, "").split(",")
    perf._require(
        len(probe_urls) == 4
        and all(value.startswith("http://") for value in probe_urls),
        "four endpoint evidence probe URLs are required",
    )
    for value in probe_urls:
        command.extend(("--endpoint-evidence-url", value))
    return command


def _load_implementation_contract(
    args, *, phase_manifest: Path,
) -> tuple[Path, dict[str, object]]:
    raw = os.environ.get(IMPLEMENTATION_ENV)
    expected_sha = os.environ.get(IMPLEMENTATION_SHA_ENV)
    perf._require(bool(raw) and bool(expected_sha),
                  "frozen C4 implementation contract is required")
    path = Path(str(raw))
    if not path.is_absolute():
        path = args.repo_root / path
    path = path.resolve()
    value = implementation.verify_contract(
        repo_root=args.repo_root,
        contract_path=path,
        expected_sha256=str(expected_sha),
        phase_manifest=phase_manifest,
    )
    return path, value


def _readiness_timeout() -> float:
    try:
        value = float(os.environ.get(
            "TEMPO_PD_C4_READINESS_S", str(DEFAULT_READINESS_S)))
    except ValueError as exc:
        raise RuntimeError("TEMPO_PD_C4_READINESS_S must be numeric") from exc
    perf._require(common.READINESS_S <= value <= 3600.0,
                  "C4 readiness must be in [600, 3600] seconds")
    return value


def _validate_environment() -> None:
    for name, expected_value in _FIXED_RUNTIME_ENVIRONMENT.items():
        perf._require(os.environ.get(name) == expected_value,
                      f"C4 requires {name}={expected_value}")
    perf._require(
        not os.environ.get("TEMPO_PD_ENDPOINT_SERVICE_PROFILE")
        and not os.environ.get("TEMPO_CXI_BACKGROUND_DUTY_CYCLE")
        and not os.environ.get("TEMPO_CXI_BACKGROUND_START_FILE"),
        "C4 fixed characterization forbids a controller profile or synthetic load",
    )
    allowed = set(_FIXED_RUNTIME_ENVIRONMENT) | set(
        _DYNAMIC_RUNTIME_ENVIRONMENT)
    unexpected = sorted(
        name for name in os.environ
        if name.startswith("TEMPO_") and name not in allowed
    )
    perf._require(
        not unexpected,
        f"C4 refuses inherited experiment variables: {unexpected}",
    )


def main() -> int:
    args = elastic.capacity._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    perf._require(args.repo_root in args.result_dir.parents,
                  "result directory must be below repository")
    perf._require(args.request_rate > 0 and args.max_workers > 0,
                  "request rate and workers must be positive")
    _validate_environment()
    readiness = _readiness_timeout()
    common.READINESS_S = readiness

    workload = args.scout_root
    if workload.is_dir():
        workload = workload / "workloads/validation.jsonl"
    workload = workload.resolve()
    perf._require(workload.is_file(), "explicit source workload is missing")
    hosts = args.hosts.split(",")
    perf._require(len(hosts) == 4 and len(set(hosts)) == 4,
                  "four unique hosts are required")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    perf._require((model / "config.json").is_file(), "Qwen model is missing")
    revision = _sha256(model / "config.json")
    profile = contention._profile(args)
    manifest_path, manifest = _load_manifest(
        args, workload=workload, elastic_profile=profile)
    implementation_path, implementation_value = (
        _load_implementation_contract(
            args, phase_manifest=manifest_path))
    perf._require(float(args.request_rate)
                  == float(manifest["foreground_rate_per_s"]),
                  "runtime C4 foreground rate differs")
    perf._require(float(os.environ["TEMPO_PD_C4_PHASE_DURATION_MS"])
                  == float(manifest["phase_duration_ms"]),
                  "runtime C4 phase duration differs")

    probe_port = contention._probe_port(args.port_slot)
    probe_urls = contention._probe_urls(hosts, probe_port)
    os.environ[contention.PROBE_URLS_ENV] = ",".join(probe_urls)

    probe = probe_handle = None
    try:
        probe, probe_handle = common._spawn(
            contention._probe_command(
                python, node_index=args.node_index, hosts=hosts,
                port_slot=args.port_slot),
            args.result_dir / f"node-{args.node_index}-endpoint-probe.log",
            dict(os.environ),
        )
        common._wait_url(probe_urls[args.node_index] + "/health", [probe])
        old_client = perf._client_command
        old_router = perf._router_command
        old_frontend = perf._frontend_command
        old_vllm = perf._vllm_command
        old_config = perf._config_text
        old_chunk_config = chunk256._config_text
        old_proxy = legacy._proxy_command
        perf._client_command = _client_command
        perf._router_command = canonical._router_command
        perf._frontend_command = canonical._frontend_command
        perf._vllm_command = canonical._vllm_command
        perf._config_text = canonical._config_text
        chunk256._config_text = canonical._config_text
        legacy._proxy_command = chunk256._proxy_command
        try:
            raw_path = perf._lifecycle(
                args,
                lifecycle=0,
                stage_name=STAGE_NAME,
                router_mode="tempo_auto",
                workload_kind="validation",
                workload=workload,
                manifest=profile,
                hosts=hosts,
                model=model,
                python=python,
                model_revision=revision,
            )
        finally:
            perf._client_command = old_client
            perf._router_command = old_router
            perf._frontend_command = old_frontend
            perf._vllm_command = old_vllm
            perf._config_text = old_config
            chunk256._config_text = old_chunk_config
            legacy._proxy_command = old_proxy
    finally:
        common._stop(probe)
        if probe_handle is not None:
            probe_handle.close()

    marker = args.result_dir / f"node-{args.node_index}-complete"
    with marker.open("x", encoding="utf-8") as stream:
        stream.write("complete\n")
    result_path = args.result_dir / "result.json"
    if args.node_index == 0:
        for index in range(4):
            common._wait_file(args.result_dir / f"node-{index}-complete", [])
        artifact = json.loads(raw_path.read_text(encoding="utf-8"))
        block_artifacts = _validate_client_artifacts(
            artifact, client_raw_path=raw_path)
        gate = artifact.get("gate")
        perf._require(
            artifact.get("schema") == CLIENT_SCHEMA
            and artifact.get("performance_claim_allowed") is False
            and artifact.get("controller_tuning_allowed") is True
            and isinstance(gate, dict)
            and gate.get("all_blocks_valid") is True
            and gate.get("paired_semantic_schedules_exact") is True
            and gate.get("paired_output_digests_exact") is True
            and gate.get("phase_aligned_endpoint_evidence") is True
            and gate.get("phase_geometry_cells_complete") is True
            and len(gate.get("phase_service_rows", [])) == 36
            and len(gate.get("phase_route_summaries", [])) == 6
            and gate.get("performance_claim_allowed") is False,
            "C4 fixed-phase characterization gate failed",
        )
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump({
                "schema": SCHEMA,
                "raw": str(raw_path.resolve()),
                "raw_sha256": _sha256(raw_path),
                "phase_manifest": str(manifest_path),
                "phase_manifest_sha256": _sha256(manifest_path),
                "phase_manifest_fingerprint_sha256": manifest[
                    "fingerprint_sha256"],
                "implementation_contract": str(implementation_path),
                "implementation_contract_sha256": _sha256(
                    implementation_path),
                "implementation_fingerprint_sha256": implementation_value[
                    "fingerprint_sha256"],
                "implementation_file_count": len(
                    implementation_value["files"]),
                "implementation_git_heads": implementation_value[
                    "git_heads"],
                "implementation_environment_versions": implementation_value[
                    "environment_versions"],
                "fixed_runtime_environment": {
                    name: os.environ[name]
                    for name in sorted(_FIXED_RUNTIME_ENVIRONMENT)
                },
                "transport_environment": {
                    name: value for name, value in sorted(os.environ.items())
                    if name.startswith((
                        "FI_", "UCX_", "NIXL_", "LMCACHE_", "VLLM_",
                        "NCCL_", "CUDA_",
                    ))
                },
                "elastic_profile": str(profile),
                "elastic_profile_sha256": _sha256(profile),
                "source_workload": str(workload),
                "source_workload_sha256": _sha256(workload),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "startup_readiness_timeout_s": readiness,
                "block_count": len(block_artifacts),
                "paired_output_count": gate.get("paired_output_count"),
                "phase_service_row_count": len(
                    gate.get("phase_service_rows", [])),
                "phase_route_summary_count": len(
                    gate.get("phase_route_summaries", [])),
                "cache_state_protocol_completion_backed": True,
                "decoder_cache_source_breakdown_exact": True,
                "phase_aligned_endpoint_evidence": True,
                "decoder_residency_basis": (
                    "exact_local_preparation_hit_on_original_P_token_prompt"),
                "characterization_gate_pass": True,
                "controller_tuning_allowed": True,
                "performance_claim_allowed": False,
                "physical_switch_bottleneck_claim_allowed": False,
                "unchanged_pd_data_plane": True,
                "transport": "LMCacheConnectorV1:UCX",
            }, stream, indent=2, sort_keys=True)
            stream.write("\n")
    else:
        common._wait_file(result_path, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
