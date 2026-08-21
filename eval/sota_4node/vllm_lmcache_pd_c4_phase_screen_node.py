#!/usr/bin/env python3
"""Run one frozen four-node actual-vLLM/LMCache C4 phase screen."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256
from eval.sota_4node import vllm_lmcache_elastic_pd_node as canonical
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as elastic
from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf
from eval.sota_4node import vllm_lmcache_pd_contention_node as contention
from eval.sota_4node import run_tempo_pd_c4_phase_screen_client as client
from tempo.pd_endpoint_profile import SCHEMA_V2, load_endpoint_service_profile


SCHEMA = "tempo-pd-c4-phase-screen-node-v1"
CLIENT_SCHEMA = "tempo-pd-c4-phase-screen-client-v1"
CLIENT_MODULE = "eval.sota_4node.run_tempo_pd_c4_phase_screen_client"
STAGE_NAME = "tempo_pd_c4_phase_screen"
RUN_CONTRACT_SCHEMA = "tempo-pd-c4-phase-screen-run-contract-v1"
SEMANTIC_RUN_CONTRACT_SCHEMA = client.SEMANTIC_RUN_CONTRACT_SCHEMA
RUN_CONTRACT_ENV = "TEMPO_PD_C4_RUN_CONTRACT"
RUN_CONTRACT_SHA_ENV = "TEMPO_PD_C4_RUN_CONTRACT_SHA256"
CONTROLLER_URLS_ENV = "TEMPO_PD_ENDPOINT_CONTROLLER_URLS"
DEFAULT_READINESS_S = 3600.0
DEFAULT_LIFECYCLE_S = 3600.0
OVERLAY_SCHEMA = "tempo-pd-c4-python-overlay-v1"
PROBE_METRICS_TIMEOUT_S = 1.0
PROBE_METRICS_ATTEMPTS = 2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(repo_root: Path, entry: object, *, name: str) -> Path:
    perf._require(isinstance(entry, dict), f"C4 contract lacks {name}")
    raw = entry.get("path")
    perf._require(isinstance(raw, str) and raw, f"C4 {name} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), f"C4 {name} is missing")
    perf._require(_sha256(path) == entry.get("sha256"),
                  f"C4 {name} digest differs")
    return path


def _load_contract(args, *, workload: Path, elastic_profile: Path):
    raw = os.environ.get(RUN_CONTRACT_ENV)
    expected_sha = os.environ.get(RUN_CONTRACT_SHA_ENV)
    perf._require(bool(raw) and bool(expected_sha),
                  "frozen C4 run contract is required")
    path = Path(str(raw))
    if not path.is_absolute():
        path = args.repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), "frozen C4 run contract is missing")
    perf._require(_sha256(path) == expected_sha,
                  "frozen C4 run contract digest differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    perf._require(
        value.get("schema") in {
            RUN_CONTRACT_SCHEMA, SEMANTIC_RUN_CONTRACT_SCHEMA},
        "frozen C4 run contract schema differs",
    )
    if value.get("schema") == SEMANTIC_RUN_CONTRACT_SCHEMA:
        perf._require(
            value.get("fingerprint_sha256")
            == client._semantic_contract_fingerprint(value)
            and value.get("endpoint_routing_policy") == "semantic_epoch_v1"
            and value.get("passive_external_credit") is True,
            "semantic C4 run contract differs",
        )
        client._validate_semantic_implementation(value)
        client._validate_semantic_runtime(value)
        base, base_entry = _resolve(
            args.repo_root, value.get("base_c4_run_contract"),
            name="base C4 run contract"), value["base_c4_run_contract"]
        base_value = json.loads(base.read_text(encoding="utf-8"))
        perf._require(
            base_value.get("schema") == RUN_CONTRACT_SCHEMA
            and base_entry.get("schema") == RUN_CONTRACT_SCHEMA,
            "semantic base C4 run contract differs",
        )
        _resolve(
            args.repo_root, value.get("semantic_observer_result"),
            name="semantic observer result")
        _resolve(
            args.repo_root, value.get("semantic_observer_analysis"),
            name="semantic observer analysis")
    perf._require(value.get("performance_claim_allowed") is False
                  and value.get("physical_switch_bottleneck_claim_allowed") is False,
                  "C4 screen contract permits an invalid claim")
    perf._require(value.get("transport") == "LMCacheConnectorV1:UCX"
                  and value.get("unchanged_pd_data_plane") is True,
                  "C4 screen data-plane contract differs")
    source = _resolve(args.repo_root, value.get("source_workload"),
                      name="source workload")
    profile = _resolve(args.repo_root, value.get("elastic_profile"),
                       name="elastic profile")
    manifest = _resolve(args.repo_root, value.get("phase_manifest"),
                        name="phase manifest")
    endpoint = _resolve(args.repo_root, value.get("endpoint_service_profile"),
                        name="endpoint service profile")
    replay = _resolve(args.repo_root, value.get("offline_replay"),
                      name="offline replay")
    perf._require(source == workload, "C4 source workload path differs")
    perf._require(profile == elastic_profile, "C4 Elastic profile path differs")
    endpoint_value = json.loads(endpoint.read_text(encoding="utf-8"))
    endpoint_entry = value["endpoint_service_profile"]
    perf._require(endpoint_value.get("fingerprint_sha256")
                  == endpoint_entry.get("fingerprint_sha256"),
                  "C4 endpoint profile fingerprint differs")
    perf._require(endpoint_value.get("workload_manifest_sha256")
                  == _sha256(manifest),
                  "C4 endpoint profile workload binding differs")
    loaded_endpoint = load_endpoint_service_profile(endpoint)
    if value.get("schema") == SEMANTIC_RUN_CONTRACT_SCHEMA:
        source_endpoint = _resolve(
            args.repo_root, value.get("source_endpoint_service_profile"),
            name="source endpoint service profile")
        source_entry = value["source_endpoint_service_profile"]
        perf._require(
            loaded_endpoint.schema == SCHEMA_V2
            and loaded_endpoint.routing_policy is not None
            and loaded_endpoint.routing_policy.as_dict()
            == value.get("semantic_credit_contract")
            and endpoint_entry.get("schema") == SCHEMA_V2
            and endpoint_entry.get("derived_from_sha256")
            == source_entry.get("sha256")
            and source_endpoint
            == client.semantic_contract_builder._resolve_base_entry(
                base_value, "endpoint_service_profile"),
            "semantic endpoint profile is not profile-bound",
        )
    else:
        perf._require(
            loaded_endpoint.routing_policy is None,
            "instant-score C4 cannot carry a semantic routing profile",
        )
    perf._require(os.environ.get(
        "TEMPO_PD_ENDPOINT_SERVICE_PROFILE") == str(endpoint),
        "runtime endpoint profile path differs")
    perf._require(os.environ.get(
        "TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256") == _sha256(manifest),
        "runtime endpoint workload binding differs")
    replay_value = json.loads(replay.read_text(encoding="utf-8"))
    perf._require(replay_value.get("schema")
                  == value["offline_replay"].get("schema")
                  and replay_value.get("live_c4_screen_authorized") is True,
                  "offline replay did not authorize live C4")
    slurm = value.get("slurm")
    perf._require(isinstance(slurm, dict)
                  and slurm.get("nodes") == 4 and slurm.get("gpus") == 16
                  and slurm.get("interactive_time_limit") == "04:00:00",
                  "C4 Slurm contract differs")
    return path, value, manifest, endpoint, replay


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
            "TEMPO_PD_C4_PHASE_DURATION_MS", "15000"),
        "--cooldown-s", os.environ.get("TEMPO_PD_C4_COOLDOWN_S", "2"),
    ]
    probe_urls = os.environ.get(contention.PROBE_URLS_ENV, "").split(",")
    perf._require(len(probe_urls) == 4
                  and all(value.startswith("http://") for value in probe_urls),
                  "four endpoint evidence probe URLs are required")
    for value in probe_urls:
        command.extend(("--endpoint-evidence-url", value))
    controller_urls = os.environ.get(CONTROLLER_URLS_ENV, "").split(",")
    perf._require(len(controller_urls) == 2
                  and all(value.startswith("http://") for value in controller_urls),
                  "two endpoint controller URLs are required")
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
                  "C4 readiness must be in [600, 3600] seconds")
    return value


def _lifecycle_timeout() -> float:
    try:
        value = float(os.environ.get(
            "TEMPO_PD_C4_LIFECYCLE_S", str(DEFAULT_LIFECYCLE_S)))
    except ValueError as exc:
        raise RuntimeError("TEMPO_PD_C4_LIFECYCLE_S must be numeric") from exc
    perf._require(common.LIFECYCLE_S <= value <= 7200.0,
                  "C4 lifecycle must be in [900, 7200] seconds")
    return value


def _validate_environment() -> None:
    policy = os.environ.get(
        "TEMPO_PD_ENDPOINT_ROUTING_POLICY", "instant_score_v1")
    perf._require(
        policy in {"instant_score_v1", "semantic_epoch_v1"},
        "C4 endpoint routing policy differs")
    expected = {
        "TEMPO_PD_C4_APPROVED": "YES",
        "TEMPO_PD_BENCHMARK_COLD_MEASURED": "1",
        "TEMPO_VLLM_DECODER_PREFIX_CACHING": "0",
        "TEMPO_LMCACHE_NIXL_BACKEND": "UCX",
        "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "adaptive",
        "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": (
            "1" if policy == "semantic_epoch_v1" else "0"),
        "TEMPO_PD_PRESSURE_MODE": "disabled",
        "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "disabled",
        "TEMPO_VLLM_SCHEDULING_POLICY": "fcfs",
        "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "0",
        "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY": "0",
        "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY": "0",
        "TEMPO_PD_MEDIAN_GUARD_PRIORITY": "0",
        "TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY": "0",
    }
    for name, value in expected.items():
        perf._require(os.environ.get(name) == value,
                      f"C4 requires {name}={value}")
    perf._require(not os.environ.get("TEMPO_CXI_BACKGROUND_DUTY_CYCLE")
                  and not os.environ.get("TEMPO_CXI_BACKGROUND_START_FILE"),
                  "C4 forbids synthetic network traffic")


def _python_overlay_provenance(args) -> tuple[Path, dict]:
    required = (
        "TEMPO_C4_PYTHON_OVERLAY",
        "TEMPO_C4_PYTHON_OVERLAY_ARCHIVE_SHA256",
        "TEMPO_C4_PYTHON_OVERLAY_SCHEMA",
        "TEMPO_C4_PYTHON_OVERLAY_STAGE_ELAPSED_NS",
        "TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT",
    )
    perf._require(all(os.environ.get(name) is not None for name in required),
                  "C4 node-local Python overlay provenance is incomplete")
    overlay = Path(os.environ["TEMPO_C4_PYTHON_OVERLAY"]).resolve()
    expected = Path(f"/tmp/tempo-c4-{os.environ.get('SLURM_JOB_ID')}-py312")
    perf._require(overlay == expected and overlay.is_dir(),
                  "C4 Python overlay is not the job-local /tmp directory")
    perf._require(os.environ["TEMPO_C4_PYTHON_OVERLAY_SCHEMA"]
                  == "tempo-pd-c4-python-overlay-prepare-v2",
                  "unexpected C4 Python overlay schema")
    helper = args.repo_root / "eval/sota_4node/stage_c4_python_overlay.sh"
    entry_script = (
        args.repo_root / "eval/sota_4node/c4_phase_screen_pd_node_entry.sh")
    prepare_path = Path(os.environ[
        "TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT"]).resolve()
    perf._require(prepare_path.is_file(),
                  "C4 sbcast overlay prepare artifact is missing")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    perf._require(
        prepare.get("schema") == "tempo-pd-c4-python-overlay-prepare-v2"
        and prepare.get("slurm_job_id") == os.environ.get("SLURM_JOB_ID")
        and prepare.get("overlay") == str(overlay)
        and prepare.get("archive_sha256")
        == os.environ["TEMPO_C4_PYTHON_OVERLAY_ARCHIVE_SHA256"]
        and prepare.get("controller_or_workload_changed") is False
        and prepare.get("pd_data_plane_changed") is False,
        "C4 sbcast overlay prepare provenance differs",
    )
    package_names = [entry.get("name") for entry in prepare.get("packages", [])]
    perf._require(package_names == ["transformers", "vllm", "lmcache"],
                  "C4 Python overlay package set differs")
    resolved_origins = {}
    for package in package_names:
        package_entry = next(value for value in prepare["packages"]
                             if value["name"] == package)
        for relative, expected_sha in package_entry["sentinels"].items():
            perf._require(_sha256(overlay / relative) == expected_sha,
                          f"C4 overlay {package} sentinel differs")
        spec = importlib.util.find_spec(package)
        perf._require(spec is not None and spec.origin is not None,
                      f"C4 overlay {package} import cannot be resolved")
        origin = Path(spec.origin).resolve()
        perf._require(overlay in origin.parents,
                      f"C4 {package} does not resolve from node-local overlay")
        resolved_origins[package] = str(origin)
    value = {
        "schema": OVERLAY_SCHEMA,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_id": args.node_index,
        "hostname": os.uname().nodename,
        "overlay": str(overlay),
        "resolved_package_origins": resolved_origins,
        "packages": package_names,
        "archive_sha256": os.environ[
            "TEMPO_C4_PYTHON_OVERLAY_ARCHIVE_SHA256"],
        "stage_elapsed_ns": int(os.environ[
            "TEMPO_C4_PYTHON_OVERLAY_STAGE_ELAPSED_NS"]),
        "helper_script_sha256": _sha256(helper),
        "entry_script_sha256": _sha256(entry_script),
        "prepare_artifact": str(prepare_path),
        "prepare_artifact_sha256": _sha256(prepare_path),
        "delivery_only": True,
        "controller_or_workload_changed": False,
        "pd_data_plane_changed": False,
    }
    path = args.result_dir / f"node-{args.node_index}-python-overlay.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path, value


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
    overlay_path, _overlay = _python_overlay_provenance(args)
    readiness = _readiness_timeout()
    lifecycle_timeout = _lifecycle_timeout()
    common.READINESS_S = readiness
    common.LIFECYCLE_S = lifecycle_timeout
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
    contract_path, _contract, manifest, endpoint, replay = _load_contract(
        args, workload=workload, elastic_profile=profile)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    perf._require(float(args.request_rate)
                  == float(manifest_value["foreground"]["offered_rate_per_s"]),
                  "runtime C4 foreground rate differs")
    perf._require(float(os.environ["TEMPO_PD_C4_PHASE_DURATION_MS"])
                  == float(manifest_value["measurement"]["phase_duration_ms"]),
                  "runtime C4 phase duration differs")
    perf._require(float(os.environ["TEMPO_PD_C4_COOLDOWN_S"])
                  == float(manifest_value["cooldown_s"]),
                  "runtime C4 cooldown differs")
    probe_port = contention._probe_port(args.port_slot)
    probe_urls = contention._probe_urls(hosts, probe_port)
    os.environ[contention.PROBE_URLS_ENV] = ",".join(probe_urls)
    ports = perf._ports(args.port_slot, 0)
    controller_urls = [
        f"http://{hosts[0]}:{ports['pair_router']}",
        f"http://{hosts[2]}:{ports['pair_router']}",
    ]
    os.environ[CONTROLLER_URLS_ENV] = ",".join(controller_urls)

    probe = probe_handle = None
    try:
        probe_command = contention._probe_command(
            python, node_index=args.node_index, hosts=hosts,
            port_slot=args.port_slot)
        probe_command.extend((
            "--metrics-timeout-s", str(PROBE_METRICS_TIMEOUT_S),
            "--metrics-attempts", str(PROBE_METRICS_ATTEMPTS),
        ))
        probe, probe_handle = common._spawn(
            probe_command,
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
        overlay_records = []
        for index in range(4):
            path = args.result_dir / f"node-{index}-python-overlay.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            perf._require(value.get("schema") == OVERLAY_SCHEMA
                          and value.get("slurm_node_id") == index
                          and value.get("controller_or_workload_changed") is False
                          and value.get("pd_data_plane_changed") is False,
                          "C4 Python overlay provenance differs")
            overlay_records.append({
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            })
        artifact = json.loads(raw_path.read_text(encoding="utf-8"))
        perf._require(artifact.get("schema") == CLIENT_SCHEMA,
                      "C4 client schema differs")
        perf._require(artifact.get("performance_claim_allowed") is False,
                      "C4 client permits a performance claim")
        perf._require(artifact.get("live_screen_correctness_pass") is True,
                      "C4 live screen correctness gate failed")
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump({
                "schema": SCHEMA,
                "raw": str(raw_path.resolve()),
                "raw_sha256": _sha256(raw_path),
                "run_contract": str(contract_path),
                "run_contract_sha256": _sha256(contract_path),
                "phase_manifest": str(manifest),
                "phase_manifest_sha256": _sha256(manifest),
                "elastic_profile": str(profile),
                "elastic_profile_sha256": _sha256(profile),
                "endpoint_service_profile": str(endpoint),
                "endpoint_service_profile_sha256": _sha256(endpoint),
                "offline_replay": str(replay),
                "offline_replay_sha256": _sha256(replay),
                "source_workload": str(workload),
                "source_workload_sha256": _sha256(workload),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "startup_readiness_timeout_s": readiness,
                "lifecycle_timeout_s": lifecycle_timeout,
                "endpoint_probe_metrics_timeout_s": PROBE_METRICS_TIMEOUT_S,
                "endpoint_probe_metrics_attempts": PROBE_METRICS_ATTEMPTS,
                "python_overlay_records": overlay_records,
                "blocks_completed": artifact.get("blocks_completed"),
                "live_screen_correctness_pass": True,
                "performance_claim_allowed": False,
                "physical_switch_bottleneck_claim_allowed": False,
                "unchanged_pd_data_plane": True,
                "transport": "LMCacheConnectorV1:UCX",
                "endpoint_routing_policy": os.environ.get(
                    "TEMPO_PD_ENDPOINT_ROUTING_POLICY", "instant_score_v1"),
                "passive_external_endpoint_credit": (
                    os.environ.get("TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK") == "1"),
            }, stream, indent=2, sort_keys=True)
            stream.write("\n")
    else:
        common._wait_file(result_path, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
