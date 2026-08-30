#!/usr/bin/env python3
"""Build and verify an immutable native TEMPO-GO C5 run contract.

The C5 discovery launcher previously accepted profile and runtime overrides and
recorded ``native-c5-discovery-unfrozen``.  This module makes the experiment
identity explicit: workload, manifest, all three profiles, model config, the
launcher/node/analyzer source inventory, arm order, and fixed runtime
parameters must agree before a native step is spawned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from tempo.pd_elastic_profile import SCHEMA as ELASTIC_PROFILE_SCHEMA
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile
from tempo.pd_global_profile import load_global_profile


SCHEMA = "tempo-go-c5-native-run-contract-v1"
ARMS = ("local", "remote", "predictor", "queue_gpu", "tempo")
CROSS_LAYER_ARMS = (
    "local", "remote", "predictor", "queue_gpu",
    "network_request_only", "app_global_only", "tempo",
)
TRANSPORT = "LMCacheConnectorV1:UCX"

# These are the exact non-arm-specific values set by the native C5 launcher.
# They are part of the contract so a hidden inherited environment cannot alter
# the data plane or scheduler configuration.
BASE_ENVIRONMENT = {
    "TEMPO_LMCACHE_NIXL_BACKEND": "UCX",
    "TEMPO_LMCACHE_LOCAL_CPU_GB": "16",
    "TEMPO_LMCACHE_PD_BUFFER_BYTES": "2147483648",
    "TEMPO_PD_PRESSURE_MODE": "disabled",
    "TEMPO_VLLM_DECODER_PREFIX_CACHING": "0",
    "TEMPO_PD_FRONTEND_PAIR_POLICY": "tempo-min-outstanding-decode-tokens-v1",
    "TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY": "1",
    "TEMPO_PD_BENCHMARK_COLD_MEASURED": "1",
    "TEMPO_PD_BENCHMARK_RESET_DECODER_APC": "0",
    "TEMPO_PD_DECODER_REUSE_ITEMS": "all",
    "TEMPO_VLLM_MAX_NUM_SEQS": "16",
    "TEMPO_VLLM_ASYNC_SCHEDULING": "0",
    "TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS": "32768",
    "TEMPO_VLLM_SCHEDULING_POLICY": "fcfs",
    "TEMPO_PD_REMOTE_DECODE_PLACEMENT": "paired",
    "TEMPO_PD_PROXY_TOKENIZER_PLACEMENT": "round_robin",
}

ARM_ENVIRONMENT = {
    "local": {
        "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "disabled",
        "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "0",
        "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "instant_score_v1",
        "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "disabled",
        "TEMPO_GO_ABLATION": "disabled",
    },
    "remote": {
        "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "disabled",
        "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "0",
        "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "instant_score_v1",
        "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "disabled",
        "TEMPO_GO_ABLATION": "disabled",
    },
    "predictor": {
        "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "disabled",
        "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "0",
        "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "instant_score_v1",
        "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "disabled",
        "TEMPO_GO_ABLATION": "disabled",
    },
    "queue_gpu": {
        "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "disabled",
        "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "0",
        "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "instant_score_v1",
        "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "observe_only",
        "TEMPO_GO_ABLATION": "disabled",
    },
    "network_request_only": {
        "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "disabled",
        "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "0",
        "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "instant_score_v1",
        "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "disabled",
        "TEMPO_GO_ABLATION": "disabled",
    },
    "app_global_only": {
        "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "adaptive",
        "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "1",
        "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "semantic_epoch_v1",
        "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "disabled",
        "TEMPO_GO_ABLATION": "app_global_only",
    },
    "tempo": {
        "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "adaptive",
        "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "1",
        "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "semantic_epoch_v1",
        "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "disabled",
        "TEMPO_GO_ABLATION": "disabled",
    },
}

SOURCE_PATHS = (
    "tempo/pd_global_orchestrator.py",
    "tempo/pd_global_hierarchy.py",
    "tempo/pd_global_coordinator.py",
    "tempo/pd_global_profile.py",
    "tempo/pd_global_candidates.py",
    "tempo/pd_global_telemetry.py",
    "tempo/cross_layer_observer.py",
    "tempo/cassini_endpoint.py",
    "tempo/pd_endpoint_controller.py",
    "tempo/pd_endpoint_profile.py",
    "tempo/pd_cache_state_protocol.py",
    "eval/sota_4node/tempo_pd_elastic_frontend.py",
    "eval/sota_4node/tempo_pd_elastic_router_v444.py",
    "eval/sota_4node/build_tempo_go_c5_heldout_manifest.py",
    "eval/sota_4node/build_tempo_go_heldout_frozen_proxy.py",
    "eval/sota_4node/validate_tempo_go_manifest.py",
    "eval/sota_4node/replay_tempo_go_c5_five_arm.py",
    "eval/sota_4node/tempo_pd_elastic_router.py",
    "eval/sota_4node/vllm_lmcache_tempo_go_c5_node.py",
    "eval/sota_4node/run_tempo_go_c5_stream_client.py",
    "eval/sota_4node/run_tempo_pd_elastic_stream_metrics.py",
    "eval/sota_4node/run_tempo_pd_elastic_stream_metrics_v445.py",
    "eval/sota_4node/run_tempo_pd_stream_metrics_forced_v32.py",
    "eval/sota_4node/vllm_lmcache_elastic_pd_node.py",
    "eval/sota_4node/vllm_lmcache_elastic_pd_node_v445.py",
    "eval/sota_4node/vllm_lmcache_live_pd_node_v1.py",
    "eval/sota_4node/vllm_lmcache_live_pd_node_v2.py",
    "eval/sota_4node/vllm_lmcache_tempo_pd_perf_node_v1.py",
    "eval/sota_4node/train.py",
    "eval/sota_4node/run_lmcache_nixl_contention_2node.py",
    "eval/sota_4node/run_lmcache_nixl_contention_2node_in_allocation.sh",
    "eval/sota_4node/run_tempo_go_cross_layer_with_cojob_in_allocation.sh",
    "eval/sota_4node/run_tempo_go_cxi_background_with_c5_in_allocation.sh",
    "eval/sota_4node/cxi_background_traffic.c",
    "eval/sota_4node/rebind_tempo_go_workload_profiles.py",
    "eval/sota_4node/probe_tempo_go_cross_layer_capability.py",
    "third_party/lmcache/lmcache/v1/transfer_channel/nixl_channel.py",
    "eval/sota_4node/vllm_lmcache_chunk256_node_v7.py",
    "eval/sota_4node/prepare_c4_python_overlay.sh",
    "eval/sota_4node/stage_c4_python_overlay.sh",
    "eval/sota_4node/require_perlmutter_4node_4h_interactive.sh",
    "eval/sota_4node/c5_tempo_go_node_entry.sh",
    "eval/sota_4node/run_tempo_go_c5_five_arm_in_allocation.sh",
    "eval/sota_4node/analyze_tempo_go_c5_five_arm.py",
    "eval/sota_4node/tempo_go_c5_run_contract.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract_fingerprint(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_path(repo_root: Path, value: Path, *, name: str) -> Path:
    repo_root = repo_root.resolve()
    path = value.expanduser().resolve()
    _require(repo_root in path.parents, f"{name} must be below repository")
    _require(path.is_file(), f"{name} is missing: {path}")
    return path


def _binding(path: Path, *, fingerprint_sha256: str | None = None) -> dict[str, str]:
    value = {"path": str(path.resolve()), "sha256": sha256(path.resolve())}
    if fingerprint_sha256 is not None:
        value["fingerprint_sha256"] = fingerprint_sha256
    return value


def _profile_binding(path: Path, profile: object) -> dict[str, str]:
    value = _binding(path, fingerprint_sha256=str(profile.fingerprint_sha256))
    value.update({
        "schema": str(getattr(profile, "schema", ELASTIC_PROFILE_SCHEMA)),
        "profile_id": str(profile.profile_id),
        "deployment_scope": str(profile.deployment_scope),
    })
    return value


def _load_json(path: Path, *, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} is not a JSON object")
    return value


def _resolve_workload(workload_input: Path) -> Path:
    value = workload_input.resolve()
    if value.is_dir():
        value = value / "workloads/validation.jsonl"
    _require(value.is_file(), f"C5 workload is missing: {value}")
    return value.resolve()


def _manifest_for_workload(workload: Path) -> Path:
    manifest = (workload.parent.parent / "tempo_go_workload_manifest.json").resolve()
    _require(manifest.is_file(), f"C5 workload sidecar manifest is missing: {manifest}")
    return manifest


def _source_inventory(repo_root: Path) -> dict[str, dict[str, str]]:
    value = {}
    for relative in SOURCE_PATHS:
        path = _repo_path(repo_root, repo_root / relative, name=relative)
        value[relative] = _binding(path)
    return value


def _arm_order(
    value: str | tuple[str, ...] | list[str],
    *,
    allowed: tuple[str, ...] = ARMS,
) -> tuple[str, ...]:
    items = tuple(value.split(",") if isinstance(value, str) else value)
    _require(items and all(item in allowed for item in items),
             "C5 arm order contains an unsupported arm")
    _require(len(items) == len(set(items)), "C5 arm order contains duplicates")
    _require(set(items) == set(allowed),
             "frozen C5 contract must contain exactly its declared arms")
    return items


def _validate_manifest_bindings(
    *, manifest: Mapping[str, object], manifest_path: Path,
    workload: Path, model_config: Path,
) -> None:
    _require(manifest.get("schema") == "tempo-go-contention-manifest-v1",
             "C5 manifest schema differs")
    _require(manifest.get("transport") == TRANSPORT,
             "C5 manifest transport differs")
    _require(manifest.get("native_only") is True,
             "C5 manifest is not native-only")
    _require(manifest.get("model_config_sha256") == sha256(model_config),
             "C5 manifest model config binding differs")
    execution = manifest.get("execution_contract")
    _require(isinstance(execution, Mapping), "C5 execution contract is missing")
    _require(execution.get("client_request_rate_flag") == "must_be_omitted",
             "C5 workload must use explicit absolute arrivals")
    _require(execution.get("warmup_outside_measurement") is True,
             "C5 warmup contract differs")
    validation = manifest.get("validation_workload")
    _require(isinstance(validation, Mapping), "C5 validation workload is missing")
    _require(Path(str(validation.get("path"))).resolve() == workload,
             "C5 workload path differs from manifest")
    _require(validation.get("sha256") == sha256(workload),
             "C5 validation workload SHA differs")
    _require(Path(str(validation.get("path"))).resolve().is_file(),
             "C5 validation workload path is missing")
    _require(manifest_path.name == "tempo_go_workload_manifest.json",
             "C5 manifest filename differs")


def build_run_contract(
    *,
    repo_root: Path,
    workload_input: Path,
    global_profile_path: Path,
    elastic_profile_path: Path,
    endpoint_profile_path: Path,
    model_config_path: Path,
    output_path: Path,
    candidate_id: str,
    candidate_revision: str,
    arm_order: str | tuple[str, ...] | list[str],
    step_time: str = "00:40:00",
    timeout_seconds: int = 7200,
    request_rate: int = 8,
    max_workers: int = 128,
    output_tokens: int = 2,
    samples_per_bucket: int = 3,
    ttft_slo_ms: int = 3000,
    tpot_slo_ms: int = 250,
    e2e_slo_ms: int = 16000,
    cross_layer_cojob: bool = False,
    seven_arm_cross_layer: bool = False,
    cxi_background_cojob: bool = False,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    workload = _repo_path(repo_root, _resolve_workload(workload_input), name="C5 workload")
    manifest_path = _repo_path(
        repo_root, _manifest_for_workload(workload), name="C5 manifest")
    global_path = _repo_path(repo_root, global_profile_path, name="global profile")
    elastic_path = _repo_path(repo_root, elastic_profile_path, name="Elastic profile")
    endpoint_path = _repo_path(repo_root, endpoint_profile_path, name="endpoint profile")
    model_config = _repo_path(repo_root, model_config_path, name="model config")
    output_path = output_path.resolve()
    _require(repo_root in output_path.parents, "contract output must be below repository")
    _require(not output_path.exists(), "refusing to overwrite C5 run contract")
    _require(candidate_id.strip() and candidate_revision.strip(),
             "candidate identity is required")
    allowed_arms = CROSS_LAYER_ARMS if seven_arm_cross_layer else ARMS
    _require(
        not seven_arm_cross_layer or cross_layer_cojob,
        "seven-arm cross-layer contract requires the co-job runner",
    )
    _require(
        not cxi_background_cojob or cross_layer_cojob,
        "CXI background co-job requires the cross-layer contract",
    )
    order = _arm_order(arm_order, allowed=allowed_arms)

    manifest = _load_json(manifest_path, name="C5 manifest")
    _validate_manifest_bindings(
        manifest=manifest, manifest_path=manifest_path,
        workload=workload, model_config=model_config,
    )
    global_profile = load_global_profile(global_path)
    elastic_profile = load_elastic_profile(elastic_path)
    endpoint_profile = load_endpoint_service_profile(endpoint_path)
    _require(global_profile.transport == TRANSPORT,
             "global profile transport differs")
    _require(global_profile.identity.workload_manifest_sha256 == sha256(manifest_path),
             "global profile manifest binding differs")
    _require(global_profile.identity.model_config_sha256 == sha256(model_config),
             "global profile model binding differs")
    _require(global_profile.identity.elastic_profile_fingerprint_sha256
             == elastic_profile.fingerprint_sha256,
             "global/Elastic profile fingerprint differs")
    _require(global_profile.identity.endpoint_profile_fingerprint_sha256
             == endpoint_profile.fingerprint_sha256,
             "global/endpoint profile fingerprint differs")
    _require(endpoint_profile.elastic_profile_fingerprint_sha256
             == elastic_profile.fingerprint_sha256,
             "endpoint/Elastic profile fingerprint differs")
    _require(endpoint_profile.workload_manifest_sha256 == sha256(manifest_path),
             "endpoint profile manifest binding differs")
    _require(endpoint_profile.deployment_scope
             == global_profile.identity.endpoint_profile_deployment_scope,
             "endpoint profile deployment scope differs")
    _require(global_profile.telemetry.scheduler_observation_required is True,
             "native C5 contract requires scheduler observation")
    manifest_arms = manifest.get("comparison_arms")
    _require(isinstance(manifest_arms, list),
             "C5 manifest comparison arms are missing")
    required_manifest_arms = {
        "always_local", "official_always_remote", "predictor_only",
        "queue_gpu_only", "tempo_go",
    }
    _require(required_manifest_arms.issubset(set(manifest_arms)),
             "C5 manifest lacks required comparison arms")

    if cxi_background_cojob:
        runner_relative = (
            "eval/sota_4node/"
            "run_tempo_go_cxi_background_with_c5_in_allocation.sh"
        )
    elif cross_layer_cojob:
        runner_relative = (
            "eval/sota_4node/"
            "run_tempo_go_cross_layer_with_cojob_in_allocation.sh"
        )
    else:
        runner_relative = (
            "eval/sota_4node/run_tempo_go_c5_five_arm_in_allocation.sh"
        )
    value: dict[str, object] = {
        "schema": SCHEMA,
        "contract_version": 1,
        "purpose": "frozen native C5 discovery/receipt validation identity",
        "candidate": {
            "id": candidate_id,
            "revision": candidate_revision,
            "controller_parameters_unchanged": True,
            "post_validation_tuning_allowed": False,
        },
        "topology": {
            "node_count": 4,
            "gpu_count": 16,
            "prewarmed_pair_count": 2,
            "native_only": True,
            "transport": TRANSPORT,
        },
        "arm_order": list(order),
        "arms": list(allowed_arms),
        "artifacts": {
            "workload": _binding(workload),
            "manifest": _binding(manifest_path),
            "global_profile": _profile_binding(global_path, global_profile),
            "elastic_profile": _profile_binding(elastic_path, elastic_profile),
            "endpoint_profile": _profile_binding(endpoint_path, endpoint_profile),
            "model_config": _binding(model_config),
        },
        "source_inventory": _source_inventory(repo_root),
        "launcher": {
            "runner": _binding(repo_root / runner_relative),
            "node_entry": _binding(repo_root / "eval/sota_4node/c5_tempo_go_node_entry.sh"),
            "analyzer": _binding(repo_root / "eval/sota_4node/analyze_tempo_go_c5_five_arm.py"),
            "step_time": step_time,
            "timeout_seconds": timeout_seconds,
            "srun_nodes": 4,
            "srun_tasks": 4,
            "gpus_per_task": 4,
            "cpus_per_task": 128,
            "node_parameters": {
                "request_rate": request_rate,
                "max_workers": max_workers,
                "output_tokens": output_tokens,
                "samples_per_bucket": samples_per_bucket,
                "ttft_slo_ms": ttft_slo_ms,
                "tpot_slo_ms": tpot_slo_ms,
                "e2e_slo_ms": e2e_slo_ms,
            },
            "cross_layer_cojob": cross_layer_cojob,
            "seven_arm_cross_layer": seven_arm_cross_layer,
            "cxi_background_cojob": cxi_background_cojob,
        },
        "fixed_environment": {
            "base": dict(sorted(BASE_ENVIRONMENT.items())),
            "per_arm": {
                arm: dict(sorted(ARM_ENVIRONMENT[arm].items()))
                for arm in allowed_arms
            },
        },
        "gates": {
            "performance_claim_allowed": False,
            "independent_validation": False,
            "post_validation_tuning_allowed": False,
            "physical_switch_bottleneck_claim_allowed": False,
        },
        "manifest_fingerprint_context": {
            "manifest_sha256": sha256(manifest_path),
            "workload_sha256": sha256(workload),
            "model_config_sha256": sha256(model_config),
            "global_profile_fingerprint_sha256": global_profile.fingerprint_sha256,
            "elastic_profile_fingerprint_sha256": elastic_profile.fingerprint_sha256,
            "endpoint_profile_fingerprint_sha256": endpoint_profile.fingerprint_sha256,
        },
    }
    value["fingerprint_sha256"] = contract_fingerprint(value)
    return value


def _binding_value(value: Mapping[str, object], *, name: str) -> tuple[Path, str]:
    path_value = value.get("path")
    expected = value.get("sha256")
    _require(isinstance(path_value, str) and isinstance(expected, str)
             and len(expected) == 64, f"{name} binding is invalid")
    path = Path(path_value).resolve()
    _require(path.is_file() and sha256(path) == expected,
             f"{name} digest differs: {path}")
    return path, expected


def verify_contract(
    contract_path: Path,
    expected_sha256: str,
    *,
    repo_root: Path,
    workload_input: Path | None = None,
    arm_only: str | None = None,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    path = _repo_path(repo_root, contract_path, name="C5 run contract")
    _require(len(expected_sha256) == 64 and all(
        character in "0123456789abcdef" for character in expected_sha256),
        "C5 run-contract SHA is not lowercase hexadecimal")
    _require(sha256(path) == expected_sha256,
             "C5 run-contract digest differs")
    value = _load_json(path, name="C5 run contract")
    _require(value.get("schema") == SCHEMA, "C5 run-contract schema differs")
    _require(value.get("fingerprint_sha256") == contract_fingerprint(value),
             "C5 run-contract fingerprint differs")
    topology = value.get("topology")
    _require(isinstance(topology, Mapping)
             and topology.get("node_count") == 4
             and topology.get("gpu_count") == 16
             and topology.get("native_only") is True
             and topology.get("transport") == TRANSPORT,
             "C5 run-contract topology differs")
    order = value.get("arm_order")
    declared_arms = value.get("arms")
    _require(
        isinstance(declared_arms, list)
        and tuple(declared_arms) in {ARMS, CROSS_LAYER_ARMS},
        "C5 run-contract declared arms differ",
    )
    allowed_arms = tuple(declared_arms)
    _require(isinstance(order, list)
             and tuple(order) == _arm_order(tuple(order), allowed=allowed_arms),
             "C5 run-contract arm order differs")
    if arm_only is not None:
        _require(arm_only in order, "single C5 arm is not in the frozen order")

    artifacts = value.get("artifacts")
    _require(isinstance(artifacts, Mapping), "C5 run-contract artifacts missing")
    resolved: dict[str, Path] = {}
    for name in (
        "workload", "manifest", "global_profile", "elastic_profile",
        "endpoint_profile", "model_config",
    ):
        binding = artifacts.get(name)
        _require(isinstance(binding, Mapping), f"C5 contract lacks {name}")
        resolved[name], _ = _binding_value(binding, name=name)
        _require(repo_root in resolved[name].parents,
                 f"C5 {name} escapes repository")
    if workload_input is not None:
        supplied = _resolve_workload(workload_input)
        _require(supplied == resolved["workload"],
                 "supplied C5 workload differs from frozen contract")
    _require(resolved["manifest"].name == "tempo_go_workload_manifest.json",
             "frozen C5 manifest filename differs")
    manifest = _load_json(resolved["manifest"], name="frozen C5 manifest")
    _validate_manifest_bindings(
        manifest=manifest, manifest_path=resolved["manifest"],
        workload=resolved["workload"], model_config=resolved["model_config"],
    )
    global_profile = load_global_profile(resolved["global_profile"])
    elastic_profile = load_elastic_profile(resolved["elastic_profile"])
    endpoint_profile = load_endpoint_service_profile(resolved["endpoint_profile"])
    bindings = {str(name): value for name, value in artifacts.items()
                if isinstance(value, Mapping)}
    for name, profile in (
        ("global_profile", global_profile),
        ("elastic_profile", elastic_profile),
        ("endpoint_profile", endpoint_profile),
    ):
        _require(bindings[name].get("fingerprint_sha256")
                 == profile.fingerprint_sha256,
                 f"{name} fingerprint differs from contract")
        _require(bindings[name].get("schema")
                 == getattr(profile, "schema", ELASTIC_PROFILE_SCHEMA)
                 and bindings[name].get("profile_id") == profile.profile_id
                 and bindings[name].get("deployment_scope") == profile.deployment_scope,
                 f"{name} identity differs from contract")
    _require(global_profile.transport == TRANSPORT,
             "frozen global profile transport differs")
    _require(global_profile.identity.workload_manifest_sha256
             == sha256(resolved["manifest"]),
             "frozen global profile manifest binding differs")
    _require(global_profile.identity.model_config_sha256
             == sha256(resolved["model_config"]),
             "frozen global profile model binding differs")
    _require(global_profile.identity.elastic_profile_fingerprint_sha256
             == elastic_profile.fingerprint_sha256,
             "frozen global/Elastic profile binding differs")
    _require(global_profile.identity.endpoint_profile_fingerprint_sha256
             == endpoint_profile.fingerprint_sha256,
             "frozen global/endpoint profile binding differs")
    _require(endpoint_profile.elastic_profile_fingerprint_sha256
             == elastic_profile.fingerprint_sha256
             and endpoint_profile.workload_manifest_sha256
             == sha256(resolved["manifest"]),
             "frozen endpoint profile binding differs")
    _require(global_profile.telemetry.scheduler_observation_required is True,
             "frozen C5 contract does not require scheduler observation")

    sources = value.get("source_inventory")
    _require(isinstance(sources, Mapping), "C5 source inventory is missing")
    for name, binding in sources.items():
        _require(isinstance(binding, Mapping), f"C5 source binding is invalid: {name}")
        source, _ = _binding_value(binding, name=f"source {name}")
        _require(repo_root in source.parents, f"C5 source escapes repository: {name}")

    launcher = value.get("launcher")
    _require(isinstance(launcher, Mapping), "C5 launcher contract is missing")
    for name in ("runner", "node_entry", "analyzer"):
        binding = launcher.get(name)
        _require(isinstance(binding, Mapping), f"C5 launcher lacks {name}")
        _binding_value(binding, name=f"launcher {name}")
    cross_layer_cojob = launcher.get("cross_layer_cojob", False)
    _require(type(cross_layer_cojob) is bool,
             "C5 cross-layer co-job flag is invalid")
    if cross_layer_cojob:
        cxi_background_cojob = launcher.get(
            "cxi_background_cojob", False)
        _require(type(cxi_background_cojob) is bool,
                 "C5 CXI background co-job flag is invalid")
        runner_path = Path(str(launcher["runner"].get("path"))).name
        expected_runner = (
            "run_tempo_go_cxi_background_with_c5_in_allocation.sh"
            if cxi_background_cojob
            else "run_tempo_go_cross_layer_with_cojob_in_allocation.sh"
        )
        _require(
            runner_path == expected_runner,
            "cross-layer contract does not bind the co-job runner",
        )
    seven_arm_cross_layer = launcher.get("seven_arm_cross_layer", False)
    _require(type(seven_arm_cross_layer) is bool,
             "C5 seven-arm flag is invalid")
    _require(seven_arm_cross_layer == (allowed_arms == CROSS_LAYER_ARMS),
             "C5 declared arms and seven-arm flag differ")
    params = launcher.get("node_parameters")
    _require(isinstance(params, Mapping), "C5 node parameters are missing")
    _require(params.get("max_workers") == 128
             and params.get("samples_per_bucket") == 3,
             "C5 node parameter contract differs")
    gates = value.get("gates")
    _require(isinstance(gates, Mapping)
             and gates.get("performance_claim_allowed") is False
             and gates.get("post_validation_tuning_allowed") is False,
             "C5 run-contract gate boundary was relaxed")
    return value


def expected_environment(contract: Mapping[str, object], arm: str) -> dict[str, str]:
    declared_arms = contract.get("arms")
    _require(isinstance(declared_arms, list) and arm in declared_arms,
             f"unsupported C5 arm: {arm}")
    fixed = contract.get("fixed_environment")
    _require(isinstance(fixed, Mapping), "C5 fixed environment is missing")
    base = fixed.get("base")
    per_arm = fixed.get("per_arm")
    _require(isinstance(base, Mapping) and isinstance(per_arm, Mapping),
             "C5 fixed environment is malformed")
    arm_value = per_arm.get(arm)
    _require(isinstance(arm_value, Mapping), f"C5 environment lacks arm {arm}")
    return {str(key): str(value) for key, value in {**base, **arm_value}.items()}


def validate_environment(
    contract: Mapping[str, object], arm: str, environment: Mapping[str, str],
) -> None:
    for name, expected in expected_environment(contract, arm).items():
        _require(environment.get(name) == expected,
                 f"C5 environment differs for {name}: expected {expected!r}")


def _build_command(args: argparse.Namespace) -> int:
    value = build_run_contract(
        repo_root=args.repo_root,
        workload_input=args.workload_input,
        global_profile_path=args.global_profile,
        elastic_profile_path=args.elastic_profile,
        endpoint_profile_path=args.endpoint_profile,
        model_config_path=args.model_config,
        output_path=args.output,
        candidate_id=args.candidate_id,
        candidate_revision=args.candidate_revision,
        arm_order=args.arm_order,
        step_time=args.step_time,
        timeout_seconds=args.timeout_seconds,
        request_rate=args.request_rate,
        max_workers=args.max_workers,
        output_tokens=args.output_tokens,
        samples_per_bucket=args.samples_per_bucket,
        ttft_slo_ms=args.ttft_slo_ms,
        tpot_slo_ms=args.tpot_slo_ms,
        e2e_slo_ms=args.e2e_slo_ms,
        cross_layer_cojob=args.cross_layer_cojob,
        seven_arm_cross_layer=args.seven_arm_cross_layer,
        cxi_background_cojob=args.cxi_background_cojob,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": SCHEMA,
        "fingerprint_sha256": value["fingerprint_sha256"],
        "sha256": sha256(args.output),
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--workload-input", type=Path, required=True)
    build.add_argument("--global-profile", type=Path, required=True)
    build.add_argument("--elastic-profile", type=Path, required=True)
    build.add_argument("--endpoint-profile", type=Path, required=True)
    build.add_argument("--model-config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--candidate-id", required=True)
    build.add_argument("--candidate-revision", required=True)
    build.add_argument("--arm-order", required=True)
    build.add_argument("--step-time", default="00:40:00")
    build.add_argument("--timeout-seconds", type=int, default=7200)
    build.add_argument("--request-rate", type=int, default=8)
    build.add_argument("--max-workers", type=int, default=128)
    build.add_argument("--output-tokens", type=int, default=2)
    build.add_argument("--samples-per-bucket", type=int, default=3)
    build.add_argument("--ttft-slo-ms", type=int, default=3000)
    build.add_argument("--tpot-slo-ms", type=int, default=250)
    build.add_argument("--e2e-slo-ms", type=int, default=16000)
    build.add_argument(
        "--cross-layer-cojob", action="store_true",
        help="bind the same-allocation NCCL/LMCache co-job wrapper",
    )
    build.add_argument(
        "--seven-arm-cross-layer", action="store_true",
        help="include NETWORK_REQUEST_ONLY and APP_GLOBAL_ONLY ablations",
    )
    build.add_argument(
        "--cxi-background-cojob", action="store_true",
        help="bind the four-node CPU/Cassini incast co-job wrapper",
    )
    verify = sub.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--contract", type=Path, required=True)
    verify.add_argument("--sha256", required=True)
    verify.add_argument("--workload-input", type=Path)
    verify.add_argument("--arm-only")
    args = parser.parse_args()
    if args.command == "build":
        return _build_command(args)
    value = verify_contract(
        args.contract, args.sha256, repo_root=args.repo_root,
        workload_input=args.workload_input, arm_only=args.arm_only,
    )
    print(json.dumps({
        "schema": value["schema"],
        "candidate": value["candidate"],
        "fingerprint_sha256": value["fingerprint_sha256"],
        "arm_order": value["arm_order"],
        "performance_claim_allowed": value["gates"]["performance_claim_allowed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
