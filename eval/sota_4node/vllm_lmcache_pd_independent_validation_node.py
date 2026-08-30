#!/usr/bin/env python3
"""Run one frozen independent TEMPO validation on four real P/D nodes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.sota_4node import build_tempo_pd_independent_validation_manifest as manifest_builder
from eval.sota_4node import build_tempo_pd_independent_validation_run_contract as contract_builder
from eval.sota_4node import run_tempo_pd_independent_validation_client as client
from eval.sota_4node import verify_tempo_pd_c4_semantic_integration_implementation as semantic_implementation
from eval.sota_4node import verify_tempo_pd_independent_validation_implementation as implementation
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256
from eval.sota_4node import vllm_lmcache_elastic_pd_node as canonical
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as elastic
from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf
from eval.sota_4node import vllm_lmcache_pd_contention_node as contention
from tempo.pd_elastic_profile import load_elastic_profile, require_replicated_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile


SCHEMA = "tempo-pd-independent-validation-node-v1"
CLIENT_SCHEMA = client.SCHEMA
CLIENT_MODULE = "eval.sota_4node.run_tempo_pd_independent_validation_client"
STAGE_NAME = "tempo_pd_independent_validation"
RUN_CONTRACT_ENV = client.RUN_CONTRACT_ENV
RUN_CONTRACT_SHA_ENV = client.RUN_CONTRACT_SHA_ENV
CONTROLLER_URLS_ENV = client.CONTROLLER_URLS_ENV
DEFAULT_READINESS_S = 3600.0
_FIXED_RUNTIME_ENVIRONMENT = dict(
    contract_builder.INDEPENDENT_FIXED_RUNTIME_ENVIRONMENT)
_PRESTART_DYNAMIC_ENVIRONMENT = frozenset({
    RUN_CONTRACT_ENV,
    RUN_CONTRACT_SHA_ENV,
    "TEMPO_PD_C4_READINESS_S",
})
_ORDER = (
    ("local", 2), ("predictor", 2), ("tempo", 2), ("remote", 2),
    ("remote", 3), ("tempo", 3), ("predictor", 3), ("local", 3),
    ("predictor", 4), ("local", 4), ("remote", 4), ("tempo", 4),
    ("tempo", 5), ("remote", 5), ("local", 5), ("predictor", 5),
)
_EXPECTED_BLOCKS = tuple(
    (f"{sequence:02d}_{arm}_r{replicate}", arm, replicate)
    for sequence, (arm, replicate) in enumerate(_ORDER)
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_entry(
    repo_root: Path, contract: dict[str, object], name: str,
) -> tuple[Path, dict[str, object]]:
    entry = contract.get(name)
    perf._require(isinstance(entry, dict),
                  f"independent run contract lacks {name}")
    raw_path = entry.get("path")
    perf._require(type(raw_path) is str and raw_path,
                  f"independent {name} path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    perf._require(path.is_file() and _sha256(path) == entry.get("sha256"),
                  f"independent {name} digest differs")
    return path, entry


def _runtime_environment_from_contract() -> dict[str, str]:
    raw_path = os.environ.get(RUN_CONTRACT_ENV)
    expected_sha = os.environ.get(RUN_CONTRACT_SHA_ENV)
    perf._require(bool(raw_path) and bool(expected_sha),
                  "frozen independent run contract is required")
    path = Path(str(raw_path)).resolve()
    perf._require(path.is_file() and _sha256(path) == expected_sha,
                  "independent run contract digest differs at prestart")
    value = json.loads(path.read_text(encoding="utf-8"))
    candidate = value.get("candidate")
    perf._require(isinstance(candidate, dict),
                  "independent prestart candidate is missing")
    return contract_builder.independent_runtime_environment(candidate)


def _validate_prestart_environment() -> dict[str, str]:
    fixed_environment = _runtime_environment_from_contract()
    for name, expected in fixed_environment.items():
        perf._require(os.environ.get(name) == expected,
                      f"independent validation requires {name}={expected}")
    allowed = set(fixed_environment) | set(
        _PRESTART_DYNAMIC_ENVIRONMENT)
    unexpected = sorted(
        name for name in os.environ
        if name.startswith("TEMPO_") and name not in allowed
    )
    perf._require(
        not unexpected,
        "independent validation refuses inherited experiment variables: "
        f"{unexpected}",
    )
    return fixed_environment


def _load_run_contract(args):
    raw_path = os.environ.get(RUN_CONTRACT_ENV)
    expected_sha = os.environ.get(RUN_CONTRACT_SHA_ENV)
    perf._require(bool(raw_path) and bool(expected_sha),
                  "frozen independent run contract is required")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = args.repo_root / path
    path = path.resolve()
    perf._require(path == args.scout_root,
                  "independent positional run contract differs")
    perf._require(path.is_file() and _sha256(path) == expected_sha,
                  "independent run contract digest differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    candidate = value.get("candidate")
    perf._require(isinstance(candidate, dict),
                  "independent run contract candidate is missing")
    fixed_environment = contract_builder.independent_runtime_environment(
        candidate)
    perf._require(
        value.get("schema") == contract_builder.SCHEMA
        and value.get("fingerprint_sha256")
        == contract_builder.contract_fingerprint(value)
        and value.get("fixed_runtime_environment")
        == dict(sorted(fixed_environment.items()))
        and value.get("transport") == "LMCacheConnectorV1:UCX"
        and value.get("unchanged_pd_data_plane") is True
        and value.get("controller_parameters_unchanged") is True
        and value.get("independent_validation_authorized") is True
        and value.get("post_validation_tuning_allowed") is False
        and value.get("performance_claim_allowed") is False
        and value.get("physical_switch_bottleneck_claim_allowed") is False,
        "independent run contract environment or claim differs",
    )
    slurm = value.get("slurm")
    perf._require(
        isinstance(slurm, dict)
        and slurm.get("nodes") == 4
        and slurm.get("gpus") == 16
        and slurm.get("interactive_time_limit") == "04:00:00"
        and slurm.get("one_persistent_allocation_for_entire_validation") is True
        and slurm.get("must_use_different_slurm_job_from_calibration") is True
        and slurm.get("login_node_gpu_or_inference_execution_allowed") is False,
        "independent run contract Slurm scope differs",
    )
    current_job = os.environ.get("SLURM_JOB_ID")
    perf._require(
        type(current_job) is str and current_job.strip()
        and value.get("validation_must_use_different_slurm_job") is True
        and current_job != value.get("calibration_slurm_job_id"),
        "independent validation reused the calibration allocation",
    )

    manifest_path, manifest_entry = _resolve_entry(
        args.repo_root, value, "independent_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    perf._require(
        manifest.get("schema") == manifest_builder.SCHEMA
        and manifest.get("fingerprint_sha256")
        == manifest_builder.manifest_fingerprint(manifest)
        == manifest_entry.get("fingerprint_sha256")
        and manifest.get("arm_order_by_replicate") == [
            {"replicate": 2,
             "arms": ["local", "predictor", "tempo", "remote"]},
            {"replicate": 3,
             "arms": ["remote", "tempo", "predictor", "local"]},
            {"replicate": 4,
             "arms": ["predictor", "local", "remote", "tempo"]},
            {"replicate": 5,
             "arms": ["tempo", "remote", "local", "predictor"]},
        ]
        and manifest.get("traffic_shape") == "burst",
        "independent workload manifest differs",
    )
    perf._require(manifest.get("candidate") == candidate,
                  "independent manifest candidate differs")
    source, _ = _resolve_entry(args.repo_root, value, "source_workload")
    elastic_path, elastic_entry = _resolve_entry(
        args.repo_root, value, "promoted_elastic_profile")
    endpoint_path, endpoint_entry = _resolve_entry(
        args.repo_root, value, "promoted_endpoint_service_profile")
    elastic_profile = load_elastic_profile(elastic_path)
    require_replicated_profile(elastic_profile)
    endpoint_profile = load_endpoint_service_profile(endpoint_path)
    perf._require(
        elastic_profile.fingerprint_sha256
        == elastic_entry.get("fingerprint_sha256")
        and endpoint_profile.fingerprint_sha256
        == endpoint_entry.get("fingerprint_sha256")
        and endpoint_profile.deployment_scope == "frozen_validation"
        and endpoint_profile.elastic_profile_fingerprint_sha256
        == elastic_profile.fingerprint_sha256
        and endpoint_profile.workload_manifest_sha256
        == _sha256(manifest_path),
        "independent promoted profile binding differs",
    )
    if candidate.get("kind") == "candidate_b_semantic_epoch_v1":
        perf._require(
            endpoint_profile.routing_policy is not None
            and endpoint_profile.routing_policy.policy == "semantic_epoch_v1",
            "semantic independent endpoint policy differs",
        )
    else:
        perf._require(endpoint_profile.routing_policy is None,
                      "instant independent endpoint policy differs")
    adaptive_implementation_path, _ = _resolve_entry(
        args.repo_root, value, "adaptive_implementation_contract")
    independent_path, independent_entry = _resolve_entry(
        args.repo_root, value, "independent_implementation_contract")
    independent_value = implementation.verify_contract(
        repo_root=args.repo_root,
        contract_path=independent_path,
        expected_sha256=str(independent_entry["sha256"]),
        adaptive_contract=adaptive_implementation_path,
    )
    perf._require(
        independent_value["fingerprint_sha256"]
        == independent_entry.get("fingerprint_sha256"),
        "independent implementation fingerprint differs",
    )

    candidate_implementation_path, candidate_implementation_entry = (
        _resolve_entry(
            args.repo_root, value, "candidate_implementation_contract"))
    if candidate.get("kind") == "candidate_b_semantic_epoch_v1":
        candidate_implementation = semantic_implementation.verify_contract(
            repo_root=args.repo_root,
            contract_path=candidate_implementation_path,
            expected_sha256=str(candidate_implementation_entry["sha256"]),
            adaptive_contract=adaptive_implementation_path,
        )
        perf._require(
            candidate_implementation["fingerprint_sha256"]
            == candidate_implementation_entry.get("fingerprint_sha256"),
            "semantic candidate implementation fingerprint differs",
        )
    else:
        perf._require(
            candidate_implementation_path == adaptive_implementation_path
            and candidate_implementation_entry.get("fingerprint_sha256")
            == value["adaptive_implementation_contract"].get(
                "fingerprint_sha256"),
            "instant-score candidate implementation differs",
        )
    analysis_path, _ = _resolve_entry(
        args.repo_root, value, "candidate_screen_analysis")
    preregistration_path, _ = _resolve_entry(
        args.repo_root, value, "preregistration")
    promotion_receipt_path, _ = _resolve_entry(
        args.repo_root, value, "profile_promotion_receipt")
    rebuilt = contract_builder.build_run_contract(
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        adaptive_analysis_path=analysis_path,
        adaptive_analysis_sha256=_sha256(analysis_path),
        preregistration_path=preregistration_path,
        preregistration_sha256=_sha256(preregistration_path),
        elastic_path=elastic_path,
        elastic_sha256=_sha256(elastic_path),
        endpoint_path=endpoint_path,
        endpoint_sha256=_sha256(endpoint_path),
        promotion_receipt_path=promotion_receipt_path,
        promotion_receipt_sha256=_sha256(promotion_receipt_path),
        implementation_path=independent_path,
        implementation_sha256=_sha256(independent_path),
        repo_root=args.repo_root,
    )
    perf._require(rebuilt == value,
                  "independent run contract does not reproduce")
    return {
        "path": path,
        "value": value,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "source": source,
        "elastic": elastic_path,
        "endpoint": endpoint_path,
        "implementation_path": independent_path,
        "implementation": independent_value,
        "fixed_runtime_environment": fixed_environment,
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
        "four independent endpoint probes are required",
    )
    perf._require(
        len(controller_urls) == 2
        and all(value.startswith("http://") for value in controller_urls),
        "two independent endpoint controllers are required",
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
                  "independent readiness must be in [600, 3600] seconds")
    return value


def _validate_client_artifacts(
    artifact: object, *, client_raw_path: Path,
) -> dict[str, dict[str, str]]:
    perf._require(isinstance(artifact, dict),
                  "independent client artifact is malformed")
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
        "independent child artifact inventory differs",
    )
    result_root = client_raw_path.resolve().parent
    validated = {}
    for sequence, (key, arm, replicate) in enumerate(_EXPECTED_BLOCKS):
        entry = artifacts[key]
        perf._require(
            isinstance(entry, dict) and set(entry) == {"path", "sha256"},
            f"independent child binding differs: {key}",
        )
        path = Path(str(entry.get("path", "")))
        perf._require(path.is_absolute(),
                      f"independent child path is not absolute: {key}")
        path = path.resolve()
        try:
            path.relative_to(result_root)
        except ValueError as exc:
            raise ValueError(
                f"independent child escapes result directory: {key}") from exc
        perf._require(
            path.is_file() and _sha256(path) == entry.get("sha256"),
            f"independent child digest differs: {key}",
        )
        contract = contracts[key]
        child = json.loads(path.read_text(encoding="utf-8"))
        perf._require(
            isinstance(contract, dict)
            and child.get("independent_validation_contract") == contract
            and contract.get("schema") == client.BLOCK_SCHEMA
            and contract.get("sequence") == sequence
            and contract.get("arm") == arm
            and contract.get("replicate") == replicate
            and contract.get("all_requests_valid") is True
            and contract.get("completion_cache_evidence_exact") is True
            and contract.get("phase_aligned_endpoint_evidence") is True
            and contract.get("controller_reset_before_block_exact") is True
            and contract.get("controller_quiescent_after_block") is True
            and contract.get("held_out_burst_workload") is True
            and contract.get("calibration_only") is False,
            f"independent child contract differs: {key}",
        )
        validated[key] = {"path": str(path), "sha256": entry["sha256"]}
    return validated


def main() -> int:
    args = elastic.capacity._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    perf._require(args.repo_root in args.result_dir.parents,
                  "independent result directory must be below repository")
    perf._require(args.request_rate > 0 and args.max_workers > 0,
                  "independent rate and workers must be positive")
    prestart_environment = _validate_prestart_environment()
    loaded = _load_run_contract(args)
    perf._require(
        prestart_environment == loaded["fixed_runtime_environment"],
        "independent runtime policy changed after prestart",
    )
    _configure_dynamic_environment(loaded)
    readiness = _readiness_timeout()
    common.READINESS_S = readiness

    workload = loaded["source"]
    hosts = args.hosts.split(",")
    perf._require(len(hosts) == 4 and len(set(hosts)) == 4,
                  "independent validation requires four unique hosts")
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
        and int(args.max_workers) == 128,
        "independent runtime workload parameters differ",
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
                stage_name=STAGE_NAME,
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
            artifact, client_raw_path=raw_path)
        paired = artifact.get("paired_output_gate")
        perf._require(
            artifact.get("schema") == CLIENT_SCHEMA
            and artifact.get("candidate") == loaded["value"]["candidate"]
            and artifact.get("blocks_completed") == 16
            and artifact.get("independent_correctness_pass") is True
            and artifact.get("held_out_burst_workload") is True
            and artifact.get("calibration_only") is False
            and artifact.get("performance_claim_allowed") is False
            and isinstance(paired, dict)
            and paired.get("all_four_arms_present") is True
            and paired.get("semantic_schedules_exact_within_replicate") is True
            and paired.get("prompt_and_output_digests_exact") is True,
            "independent client correctness gate failed",
        )
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump({
                "schema": SCHEMA,
                "raw": str(raw_path.resolve()),
                "raw_sha256": _sha256(raw_path),
                "run_contract": str(loaded["path"]),
                "run_contract_sha256": _sha256(loaded["path"]),
                "run_contract_fingerprint_sha256": loaded["value"][
                    "fingerprint_sha256"],
                "independent_implementation_contract": str(
                    loaded["implementation_path"]),
                "independent_implementation_contract_sha256": _sha256(
                    loaded["implementation_path"]),
                "independent_implementation_fingerprint_sha256": loaded[
                    "implementation"]["fingerprint_sha256"],
                "independent_implementation_file_count": len(
                    loaded["implementation"]["files"]),
                "source_workload": str(workload),
                "source_workload_sha256": _sha256(workload),
                "independent_manifest": str(loaded["manifest_path"]),
                "independent_manifest_sha256": _sha256(
                    loaded["manifest_path"]),
                "promoted_elastic_profile": str(loaded["elastic"]),
                "promoted_elastic_profile_sha256": _sha256(
                    loaded["elastic"]),
                "promoted_endpoint_service_profile": str(
                    loaded["endpoint"]),
                "promoted_endpoint_service_profile_sha256": _sha256(
                    loaded["endpoint"]),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "calibration_slurm_job_id": loaded["value"][
                    "calibration_slurm_job_id"],
                "separate_validation_allocation": True,
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
                "held_out_burst_workload": True,
                "calibration_only": False,
                "post_validation_tuning_allowed": False,
                "performance_claim_allowed": False,
                "physical_switch_bottleneck_claim_allowed": False,
                "unchanged_pd_data_plane": True,
                "transport": "LMCacheConnectorV1:UCX",
                "candidate": loaded["value"]["candidate"],
            }, stream, indent=2, sort_keys=True)
            stream.write("\n")
    else:
        common._wait_file(result_path, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
