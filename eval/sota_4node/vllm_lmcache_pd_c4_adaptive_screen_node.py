#!/usr/bin/env python3
"""Run the post-C4 adaptive four-arm screen on four real P/D nodes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.sota_4node import build_tempo_pd_c4_adaptive_run_contract as contract_builder
from eval.sota_4node import build_tempo_pd_c4_adaptive_screen_manifest as manifest_builder
from eval.sota_4node import build_tempo_pd_c4_semantic_integration_run_contract as semantic_contract_builder
from eval.sota_4node import replay_tempo_pd_c4_calibrated_controller as replay_module
from eval.sota_4node import run_tempo_pd_c4_adaptive_screen_client as client
from eval.sota_4node import verify_tempo_pd_c4_adaptive_implementation as implementation
from eval.sota_4node import verify_tempo_pd_c4_semantic_integration_implementation as semantic_implementation
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256
from eval.sota_4node import vllm_lmcache_elastic_pd_node as canonical
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as elastic
from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf
from eval.sota_4node import vllm_lmcache_pd_contention_node as contention
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile


SCHEMA = "tempo-pd-c4-adaptive-screen-node-v2"
SEMANTIC_SCHEMA = "tempo-pd-c4-semantic-integration-screen-node-v1"
CLIENT_SCHEMA = client.SCHEMA
SEMANTIC_CLIENT_SCHEMA = client.SEMANTIC_SCHEMA
CLIENT_MODULE = "eval.sota_4node.run_tempo_pd_c4_adaptive_screen_client"
STAGE_NAME = "tempo_pd_c4_adaptive_screen"
RUN_CONTRACT_ENV = client.RUN_CONTRACT_ENV
RUN_CONTRACT_SHA_ENV = client.RUN_CONTRACT_SHA_ENV
CONTROLLER_URLS_ENV = client.CONTROLLER_URLS_ENV
DEFAULT_READINESS_S = 3600.0
_FIXED_RUNTIME_ENVIRONMENT = dict(
    contract_builder.ADAPTIVE_FIXED_RUNTIME_ENVIRONMENT)
_SEMANTIC_FIXED_RUNTIME_ENVIRONMENT = dict(
    semantic_contract_builder.SEMANTIC_FIXED_RUNTIME_ENVIRONMENT)
_PRESTART_DYNAMIC_ENVIRONMENT = frozenset({
    RUN_CONTRACT_ENV,
    RUN_CONTRACT_SHA_ENV,
    "TEMPO_PD_C4_READINESS_S",
})
_EXPECTED_BLOCKS = tuple(
    (f"{sequence:02d}_{arm}_r{replicate}", arm, replicate)
    for sequence, (arm, replicate) in enumerate(
        (value, replicate)
        for replicate, values in enumerate(
            manifest_builder.ARM_ORDER_BY_REPLICATE)
        for value in values
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_entry(
    repo_root: Path, contract: dict[str, object], name: str,
) -> tuple[Path, dict[str, object]]:
    entry = contract.get(name)
    perf._require(isinstance(entry, dict),
                  f"adaptive run contract lacks {name}")
    raw_path = entry.get("path")
    perf._require(type(raw_path) is str and raw_path,
                  f"adaptive {name} path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), f"adaptive {name} is missing")
    perf._require(_sha256(path) == entry.get("sha256"),
                  f"adaptive {name} digest differs")
    return path, entry


def _validate_prestart_environment() -> bool:
    _raw_path, _expected_sha, semantic = client._runtime_contract_binding()
    fixed_environment = (
        _SEMANTIC_FIXED_RUNTIME_ENVIRONMENT
        if semantic else _FIXED_RUNTIME_ENVIRONMENT)
    for name, expected in fixed_environment.items():
        perf._require(os.environ.get(name) == expected,
                      f"adaptive screen requires {name}={expected}")
    dynamic = set(_PRESTART_DYNAMIC_ENVIRONMENT)
    if semantic:
        dynamic.remove(RUN_CONTRACT_ENV)
        dynamic.remove(RUN_CONTRACT_SHA_ENV)
        dynamic.update({
            client.SEMANTIC_RUN_CONTRACT_ENV,
            client.SEMANTIC_RUN_CONTRACT_SHA_ENV,
        })
    allowed = set(fixed_environment) | dynamic
    unexpected = sorted(
        name for name in os.environ
        if name.startswith("TEMPO_") and name not in allowed
    )
    perf._require(
        not unexpected,
        f"adaptive screen refuses inherited experiment variables: {unexpected}",
    )
    return semantic


def _load_run_contract(args, *, semantic: bool | None = None):
    raw_path, expected_sha, observed_semantic = client._runtime_contract_binding()
    if semantic is None:
        semantic = observed_semantic
    perf._require(semantic is observed_semantic,
                  "C4 screen runtime variant changed after prestart")
    builder = semantic_contract_builder if semantic else contract_builder
    fixed_environment = (
        _SEMANTIC_FIXED_RUNTIME_ENVIRONMENT
        if semantic else _FIXED_RUNTIME_ENVIRONMENT)
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = args.repo_root / path
    path = path.resolve()
    perf._require(path == args.scout_root,
                  "adaptive positional run contract differs")
    perf._require(path.is_file() and _sha256(path) == expected_sha,
                  "adaptive run contract digest differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    perf._require(
        value.get("schema") == builder.SCHEMA
        and value.get("fingerprint_sha256")
        == builder.contract_fingerprint(value)
        and value.get("fixed_runtime_environment")
        == dict(sorted(fixed_environment.items()))
        and value.get("transport") == "LMCacheConnectorV1:UCX"
        and value.get("unchanged_pd_data_plane") is True
        and value.get("offline_replay_authorized") is True
        and value.get("performance_claim_allowed") is False
        and value.get("physical_switch_bottleneck_claim_allowed") is False,
        "adaptive run contract schema, environment, or claim differs",
    )
    if semantic:
        perf._require(
            value.get("semantic_policy_authorized") is True
            and value.get("same_allocation_calibration_required") is True
            and value.get("endpoint_routing_policy") == "semantic_epoch_v1"
            and value.get("passive_external_credit") is True,
            "semantic integration authorization differs",
        )
    slurm = value.get("slurm")
    perf._require(
        isinstance(slurm, dict)
        and slurm.get("nodes") == 4
        and slurm.get("gpus") == 16
        and slurm.get("interactive_time_limit") == "04:00:00"
        and slurm.get("persistent_allocation_reuse_required") is True
        and slurm.get("login_node_experiment_execution_allowed") is False,
        "adaptive run contract Slurm scope differs",
    )
    manifest_path, manifest_entry = _resolve_entry(
        args.repo_root, value, "phase_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    perf._require(
        manifest.get("schema") == manifest_builder.SCHEMA
        and manifest.get("fingerprint_sha256")
        == manifest_builder.manifest_fingerprint(manifest)
        == manifest_entry.get("fingerprint_sha256")
        and manifest.get("arm_order_by_replicate") == [
            list(order)
            for order in manifest_builder.ARM_ORDER_BY_REPLICATE
        ],
        "adaptive workload manifest differs",
    )
    source, _ = _resolve_entry(args.repo_root, value, "source_workload")
    elastic_path, elastic_entry = _resolve_entry(
        args.repo_root, value, "elastic_profile")
    endpoint_path, endpoint_entry = _resolve_entry(
        args.repo_root, value, "endpoint_service_profile")
    elastic_profile = load_elastic_profile(elastic_path)
    endpoint_profile = load_endpoint_service_profile(endpoint_path)
    perf._require(
        elastic_profile.fingerprint_sha256
        == elastic_entry.get("fingerprint_sha256")
        and endpoint_profile.fingerprint_sha256
        == endpoint_entry.get("fingerprint_sha256")
        and endpoint_profile.elastic_profile_fingerprint_sha256
        == elastic_profile.fingerprint_sha256
        and endpoint_profile.workload_manifest_sha256
        == _sha256(manifest_path),
        "adaptive profile binding differs",
    )
    if semantic:
        source_endpoint_path, source_endpoint_entry = _resolve_entry(
            args.repo_root, value, "source_endpoint_service_profile")
        endpoint_raw = json.loads(endpoint_path.read_text(encoding="utf-8"))
        reproduced = semantic_contract_builder.profile_builder.build_profile(
            source_endpoint_path,
            expected_base_sha256=str(source_endpoint_entry["sha256"]),
            profile_id=str(endpoint_raw.get("profile_id", "")),
        )
        perf._require(
            endpoint_raw == reproduced
            and endpoint_profile.routing_policy is not None
            and endpoint_profile.routing_policy.as_dict()
            == value.get("semantic_credit_contract")
            and endpoint_entry.get("derived_from_sha256")
            == source_endpoint_entry.get("sha256"),
            "semantic integration endpoint derivation differs",
        )
    replay_path, replay_entry = _resolve_entry(
        args.repo_root, value, "offline_replay")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    perf._require(
        replay.get("schema") == replay_module.SCHEMA
        and replay.get("fingerprint_sha256")
        == replay_module.replay_fingerprint(replay)
        == replay_entry.get("fingerprint_sha256")
        and replay.get("live_adaptive_screen_authorized") is True
        and all(replay.get("screen_gates", {}).values()),
        "adaptive offline replay authorization differs",
    )
    fixed_path, _ = _resolve_entry(
        args.repo_root, value, "fixed_c4_implementation_contract")
    adaptive_path, adaptive_entry = _resolve_entry(
        args.repo_root, value, "adaptive_implementation_contract")
    adaptive_value = implementation.verify_contract(
        repo_root=args.repo_root,
        contract_path=adaptive_path,
        expected_sha256=str(adaptive_entry["sha256"]),
        fixed_c4_contract=fixed_path,
    )
    perf._require(
        adaptive_value["fingerprint_sha256"]
        == adaptive_entry.get("fingerprint_sha256"),
        "adaptive implementation fingerprint differs",
    )
    selected_path = adaptive_path
    selected_value = adaptive_value
    if semantic:
        semantic_path, semantic_entry = _resolve_entry(
            args.repo_root, value,
            "semantic_integration_implementation_contract")
        selected_value = semantic_implementation.verify_contract(
            repo_root=args.repo_root,
            contract_path=semantic_path,
            expected_sha256=str(semantic_entry["sha256"]),
            adaptive_contract=adaptive_path,
        )
        perf._require(
            selected_value["fingerprint_sha256"]
            == semantic_entry.get("fingerprint_sha256"),
            "semantic integration implementation fingerprint differs",
        )
        selected_path = semantic_path
    return {
        "path": path,
        "value": value,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "source": source,
        "elastic": elastic_path,
        "endpoint": endpoint_path,
        "implementation_path": selected_path,
        "implementation": selected_value,
        "semantic": semantic,
        "fixed_runtime_environment": fixed_environment,
        "client_schema": (
            SEMANTIC_CLIENT_SCHEMA if semantic else CLIENT_SCHEMA),
        "block_schema": (
            client.SEMANTIC_BLOCK_SCHEMA if semantic else client.BLOCK_SCHEMA),
        "node_schema": SEMANTIC_SCHEMA if semantic else SCHEMA,
        "stage_name": (
            "tempo_pd_c4_semantic_integration_screen"
            if semantic else STAGE_NAME),
    }


def _configure_dynamic_environment(loaded: dict[str, object]) -> None:
    os.environ["TEMPO_ELASTIC_PD_PROFILE"] = str(loaded["elastic"])
    os.environ["TEMPO_PD_ENDPOINT_SERVICE_PROFILE"] = str(
        loaded["endpoint"])
    os.environ[client.WORKLOAD_SHA_ENV] = _sha256(
        loaded["manifest_path"])


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
        "--phase-duration-ms", os.environ[
            "TEMPO_PD_C4_PHASE_DURATION_MS"],
        "--cooldown-s", os.environ["TEMPO_PD_C4_COOLDOWN_S"],
    ]
    probe_urls = os.environ.get(contention.PROBE_URLS_ENV, "").split(",")
    controller_urls = os.environ.get(CONTROLLER_URLS_ENV, "").split(",")
    perf._require(
        len(probe_urls) == 4
        and all(value.startswith("http://") for value in probe_urls),
        "four adaptive endpoint probes are required",
    )
    perf._require(
        len(controller_urls) == 2
        and all(value.startswith("http://") for value in controller_urls),
        "two adaptive endpoint controllers are required",
    )
    for value in probe_urls:
        command.extend(("--endpoint-evidence-url", value))
    for value in controller_urls:
        command.extend(("--endpoint-controller-url", value))
    return command


def _readiness_timeout() -> float:
    try:
        value = float(os.environ.get(
            "TEMPO_PD_C4_READINESS_S", str(DEFAULT_READINESS_S)))
    except ValueError as exc:
        raise RuntimeError("TEMPO_PD_C4_READINESS_S must be numeric") from exc
    perf._require(common.READINESS_S <= value <= 3600.0,
                  "adaptive readiness must be in [600, 3600] seconds")
    return value


def _validate_client_artifacts(
    artifact: object, *, client_raw_path: Path,
    block_schema: str = client.BLOCK_SCHEMA,
) -> dict[str, dict[str, str]]:
    perf._require(isinstance(artifact, dict),
                  "adaptive client artifact is malformed")
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
        "adaptive child artifact inventory differs",
    )
    result_root = client_raw_path.resolve().parent
    validated = {}
    for sequence, (key, arm, replicate) in enumerate(_EXPECTED_BLOCKS):
        entry = artifacts[key]
        perf._require(isinstance(entry, dict)
                      and set(entry) == {"path", "sha256"},
                      f"adaptive child binding differs: {key}")
        path = Path(str(entry.get("path", "")))
        perf._require(path.is_absolute(),
                      f"adaptive child path is not absolute: {key}")
        path = path.resolve()
        try:
            path.relative_to(result_root)
        except ValueError as exc:
            raise ValueError(
                f"adaptive child escapes the result directory: {key}") from exc
        perf._require(
            path.is_file() and _sha256(path) == entry.get("sha256"),
            f"adaptive child digest differs: {key}",
        )
        contract = contracts[key]
        child = json.loads(path.read_text(encoding="utf-8"))
        perf._require(
            isinstance(contract, dict)
            and child.get("c4_adaptive_screen_contract") == contract
            and contract.get("schema") == block_schema
            and contract.get("sequence") == sequence
            and contract.get("arm") == arm
            and contract.get("replicate") == replicate
            and contract.get("all_requests_valid") is True
            and contract.get("completion_cache_evidence_exact") is True
            and contract.get("phase_aligned_endpoint_evidence") is True
            and contract.get("controller_reset_before_block_exact") is True
            and contract.get("controller_quiescent_after_block") is True,
            f"adaptive child contract differs: {key}",
        )
        validated[key] = {"path": str(path), "sha256": entry["sha256"]}
    return validated


def main() -> int:
    args = elastic.capacity._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    perf._require(args.repo_root in args.result_dir.parents,
                  "adaptive result directory must be below repository")
    perf._require(args.request_rate > 0 and args.max_workers > 0,
                  "adaptive rate and workers must be positive")
    semantic = _validate_prestart_environment()
    loaded = _load_run_contract(args, semantic=semantic)
    _configure_dynamic_environment(loaded)
    readiness = _readiness_timeout()
    common.READINESS_S = readiness

    workload = loaded["source"]
    hosts = args.hosts.split(",")
    perf._require(len(hosts) == 4 and len(set(hosts)) == 4,
                  "adaptive screen requires four unique hosts")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    perf._require((model / "config.json").is_file(),
                  "Qwen model is missing")
    revision = _sha256(model / "config.json")
    manifest = loaded["manifest"]
    perf._require(
        float(args.request_rate) == float(manifest["foreground_rate_per_s"])
        and float(os.environ["TEMPO_PD_C4_PHASE_DURATION_MS"])
        == float(manifest["phase_duration_ms"])
        and int(args.max_workers)
        == int(manifest["measurement"]["max_workers"]),
        "adaptive runtime workload parameters differ",
    )

    probe_port = contention._probe_port(args.port_slot)
    probe_urls = contention._probe_urls(hosts, probe_port)
    os.environ[contention.PROBE_URLS_ENV] = ",".join(probe_urls)
    ports = perf._ports(args.port_slot, 0)
    controller_urls = [
        f"http://{hosts[index]}:{ports['pair_router']}" for index in (0, 2)
    ]
    os.environ[CONTROLLER_URLS_ENV] = ",".join(controller_urls)

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
                stage_name=str(loaded["stage_name"]),
                router_mode="tempo_auto",
                workload_kind="validation",
                workload=workload,
                manifest=loaded["elastic"],
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
            artifact, client_raw_path=raw_path,
            block_schema=str(loaded["block_schema"]))
        paired = artifact.get("paired_output_gate")
        perf._require(
            artifact.get("schema") == loaded["client_schema"]
            and artifact.get("blocks_completed") == 8
            and artifact.get("live_screen_correctness_pass") is True
            and artifact.get("performance_claim_allowed") is False
            and isinstance(paired, dict)
            and paired.get("all_four_arms_present") is True
            and paired.get("semantic_schedules_exact_within_replicate") is True
            and paired.get("prompt_and_output_digests_exact") is True,
            "adaptive client correctness gate failed",
        )
        with result_path.open("x", encoding="utf-8") as stream:
            payload = {
                "schema": loaded["node_schema"],
                "raw": str(raw_path.resolve()),
                "raw_sha256": _sha256(raw_path),
                "run_contract": str(loaded["path"]),
                "run_contract_sha256": _sha256(loaded["path"]),
                "run_contract_fingerprint_sha256": loaded["value"][
                    "fingerprint_sha256"],
                "adaptive_implementation_contract": str(
                    loaded["implementation_path"]),
                "adaptive_implementation_contract_sha256": _sha256(
                    loaded["implementation_path"]),
                "adaptive_implementation_fingerprint_sha256": loaded[
                    "implementation"]["fingerprint_sha256"],
                "adaptive_implementation_file_count": len(
                    loaded["implementation"]["files"]),
                "source_workload": str(workload),
                "source_workload_sha256": _sha256(workload),
                "phase_manifest": str(loaded["manifest_path"]),
                "phase_manifest_sha256": _sha256(loaded["manifest_path"]),
                "elastic_profile": str(loaded["elastic"]),
                "elastic_profile_sha256": _sha256(loaded["elastic"]),
                "endpoint_service_profile": str(loaded["endpoint"]),
                "endpoint_service_profile_sha256": _sha256(
                    loaded["endpoint"]),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "startup_readiness_timeout_s": readiness,
                "block_count": len(block_artifacts),
                "block_artifacts": block_artifacts,
                "tempo_both_routes_exercised": paired.get(
                    "tempo_both_routes_exercised"),
                "fixed_runtime_environment": {
                    name: os.environ[name]
                    for name in sorted(loaded["fixed_runtime_environment"])
                },
                "transport_environment": {
                    name: value for name, value in sorted(os.environ.items())
                    if name.startswith((
                        "FI_", "UCX_", "NIXL_", "LMCACHE_", "VLLM_",
                        "NCCL_", "CUDA_",
                    ))
                },
                "correctness_gate_pass": True,
                "calibration_only": True,
                "performance_claim_allowed": False,
                "physical_switch_bottleneck_claim_allowed": False,
                "independent_validation_required": True,
                "unchanged_pd_data_plane": True,
                "transport": "LMCacheConnectorV1:UCX",
            }
            if semantic:
                payload["semantic_integration_implementation_contract"] = (
                    payload.pop("adaptive_implementation_contract"))
                payload[
                    "semantic_integration_implementation_contract_sha256"
                ] = payload.pop("adaptive_implementation_contract_sha256")
                payload[
                    "semantic_integration_implementation_fingerprint_sha256"
                ] = payload.pop(
                    "adaptive_implementation_fingerprint_sha256")
                payload["semantic_integration_implementation_file_count"] = (
                    payload.pop("adaptive_implementation_file_count"))
                payload.update({
                    "endpoint_routing_policy": "semantic_epoch_v1",
                    "endpoint_service_profile_fingerprint_sha256":
                        loaded["value"]["endpoint_service_profile"][
                            "fingerprint_sha256"],
                    "semantic_credit_contract": loaded["value"][
                        "semantic_credit_contract"],
                    "passive_external_endpoint_credit": True,
                    "semantic_policy_authorized": True,
                })
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    else:
        common._wait_file(result_path, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
