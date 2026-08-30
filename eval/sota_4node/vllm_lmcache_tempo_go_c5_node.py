#!/usr/bin/env python3
"""Run one native 4-node TEMPO-GO C5 vLLM/LMCache lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import signal
from pathlib import Path

from eval.sota_4node import vllm_lmcache_elastic_pd_node as canonical
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as elastic
from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256
from eval.sota_4node import tempo_go_c5_run_contract as run_contract
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile
from tempo.pd_global_profile import load_global_profile


SCHEMA = "tempo-go-c5-native-node-v2"
GLOBAL_PROFILE = (
    "results/tempo_go_c5_anchor_priors_c12_v3_retry1/"
    "real_tempo_go_profile_c12_anchor_v3.json"
)
ELASTIC_PROFILE = (
    "results/tempo_go_c5_anchor_priors_c12_v3_retry1/"
    "real_tempo_pd_elastic_profile_c12_anchor_output2_screen_v3.json"
)
ENDPOINT_PROFILE = (
    "results/tempo_go_c5_anchor_priors_c12_v3_retry1/"
    "real_tempo_pd_endpoint_service_profile_c12_anchor_output2_"
    "calibration_v3.json"
)
GLOBAL_PROFILE_ENV = "TEMPO_GO_GLOBAL_PROFILE"
ELASTIC_PROFILE_ENV = "TEMPO_GO_ELASTIC_PROFILE_PATH"
ENDPOINT_PROFILE_ENV = "TEMPO_GO_ENDPOINT_PROFILE_PATH"
RUN_CONTRACT_ENV = "TEMPO_GO_C5_RUN_CONTRACT"
RUN_CONTRACT_SHA_ENV = "TEMPO_GO_C5_RUN_CONTRACT_SHA256"
NATIVE_GLOBAL_PROFILE_SCOPES = frozenset({"discovery", "frozen_validation"})
_ORIGINAL_CLIENT_COMMAND = perf._client_command


class _NativeNodeSignal(Exception):
    """Convert Slurm TERM into an exception so lifecycle finally blocks run."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"native node received signal {signum}")


def _raise_native_node_signal(signum, _frame) -> None:
    raise _NativeNodeSignal(int(signum))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _profile_path(repo_root: Path, env_name: str, default: str) -> Path:
    """Resolve an explicitly selected profile without leaving the repository."""
    raw = os.environ.get(env_name, default)
    value = Path(raw).expanduser()
    if not value.is_absolute():
        value = repo_root / value
    value = value.resolve()
    _require(
        repo_root == value or repo_root in value.parents,
        f"{env_name} must resolve below the repository",
    )
    _require(value.is_file(), f"{env_name} profile is missing: {value}")
    return value


def _validate_profile_bindings(
    *, global_path: Path, elastic_path: Path, endpoint_path: Path,
    workload_manifest: Path,
):
    """Validate the complete profile/manifest identity before spawning vLLM."""
    global_profile = load_global_profile(global_path)
    elastic_profile = load_elastic_profile(elastic_path)
    endpoint_profile = load_endpoint_service_profile(endpoint_path)
    manifest_sha = _sha256(workload_manifest)
    identity = global_profile.identity
    _require(global_profile.deployment_scope in NATIVE_GLOBAL_PROFILE_SCOPES,
             "C5 native node requires discovery or frozen_validation global profile")
    _require(global_profile.transport == "LMCacheConnectorV1:UCX",
             "C5 transport identity differs")
    _require(identity.workload_manifest_sha256 == manifest_sha,
             "C5 workload manifest is not bound to the global profile")
    _require(
        identity.elastic_profile_fingerprint_sha256
        == elastic_profile.fingerprint_sha256,
        "C5 global and Elastic-PD profile fingerprints differ",
    )
    _require(
        identity.endpoint_profile_fingerprint_sha256
        == endpoint_profile.fingerprint_sha256,
        "C5 global and endpoint profile fingerprints differ",
    )
    _require(
        identity.endpoint_profile_id == endpoint_profile.profile_id
        and identity.endpoint_profile_schema == endpoint_profile.schema
        and identity.endpoint_profile_deployment_scope
        == endpoint_profile.deployment_scope,
        "C5 global and endpoint profile identity differs",
    )
    _require(
        endpoint_profile.elastic_profile_fingerprint_sha256
        == elastic_profile.fingerprint_sha256,
        "C5 endpoint and Elastic-PD profile fingerprints differ",
    )
    _require(
        endpoint_profile.workload_manifest_sha256 == manifest_sha,
        "C5 endpoint profile workload binding differs",
    )
    return global_profile


def _validate_native_environment() -> None:
    forbidden = ("SHIFTER", "UDI", "CRAY_ROOTFS", "SLURM_CONTAINER")
    present = [name for name in forbidden if os.environ.get(name)]
    _require(not present, f"C5 refuses container environment: {present}")
    _require(os.getuid() != 0, "C5 refuses uid 0")
    _require(os.environ.get("TEMPO_LMCACHE_NIXL_BACKEND") == "UCX",
             "C5 requires official LMCache UCX")
    arm = os.environ.get("TEMPO_GO_C5_ARM", "tempo")
    if arm in {"tempo", "app_global_only"}:
        _require(os.environ.get("TEMPO_PD_ENDPOINT_FEEDBACK_MODE") == "adaptive",
                 "TEMPO arm requires adaptive endpoint feedback")
        _require(os.environ.get("TEMPO_PD_ENDPOINT_ROUTING_POLICY")
                 == "semantic_epoch_v1",
                 "TEMPO arm requires semantic endpoint policy")
        _require(os.environ.get("TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK") == "1",
                 "TEMPO arm requires passive endpoint feedback")
        _require(os.environ.get("TEMPO_VLLM_LOAD_SNAPSHOT_MODE") == "disabled",
                 "TEMPO arm must not use request-start scheduler routing")
        expected_ablation = (
            "app_global_only" if arm == "app_global_only" else "disabled")
        _require(os.environ.get("TEMPO_GO_ABLATION", "disabled")
                 == expected_ablation,
                 "global ablation mode differs from C5 arm")
    else:
        _require(os.environ.get("TEMPO_PD_ENDPOINT_FEEDBACK_MODE") == "disabled",
                 "baseline arms require endpoint feedback disabled")
        _require(os.environ.get("TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK") == "0",
                 "baseline arms require passive endpoint feedback disabled")
        _require(os.environ.get("TEMPO_PD_ENDPOINT_ROUTING_POLICY")
                 == "instant_score_v1",
                 "baseline arms require instant endpoint policy")
        expected_load_mode = (
            "observe_only" if arm == "queue_gpu" else "disabled")
        _require(os.environ.get("TEMPO_VLLM_LOAD_SNAPSHOT_MODE")
                 == expected_load_mode,
                 "baseline scheduler observation mode differs")
        _require(os.environ.get("TEMPO_GO_ABLATION", "disabled") == "disabled",
                 "baseline arms must not enable a global ablation")


def _load_frozen_run_contract(
    *, repo_root: Path, workload: Path, arm: str,
) -> tuple[dict[str, object], Path, str]:
    raw_path = os.environ.get(RUN_CONTRACT_ENV)
    expected_sha = os.environ.get(RUN_CONTRACT_SHA_ENV)
    _require(raw_path is not None and expected_sha is not None,
             "frozen C5 run contract is required")
    contract_path = Path(raw_path).expanduser().resolve()
    contract = run_contract.verify_contract(
        contract_path,
        expected_sha,
        repo_root=repo_root,
        workload_input=workload,
        arm_only=arm,
    )
    run_contract.validate_environment(contract, arm, os.environ)
    _require(contract.get("fingerprint_sha256") is not None,
             "frozen C5 run-contract fingerprint is missing")
    return contract, contract_path, expected_sha


def _contract_artifact_path(
    contract: dict[str, object], name: str,
) -> Path:
    artifacts = contract.get("artifacts")
    _require(isinstance(artifacts, dict), "C5 run-contract artifacts are missing")
    value = artifacts.get(name)
    _require(isinstance(value, dict) and isinstance(value.get("path"), str),
             f"C5 run-contract artifact is missing: {name}")
    return Path(str(value["path"])).resolve()


def _bind_profile_environment(
    contract: dict[str, object], *, global_path: Path,
    elastic_path: Path, endpoint_path: Path,
) -> None:
    for name, expected in (
        (GLOBAL_PROFILE_ENV, global_path),
        (ELASTIC_PROFILE_ENV, elastic_path),
        (ENDPOINT_PROFILE_ENV, endpoint_path),
    ):
        supplied = os.environ.get(name)
        if supplied is not None:
            _require(Path(supplied).expanduser().resolve() == expected,
                     f"{name} differs from frozen C5 run contract")
        os.environ[name] = str(expected)
    supplied_endpoint = os.environ.get("TEMPO_PD_ENDPOINT_SERVICE_PROFILE")
    if supplied_endpoint is not None:
        _require(Path(supplied_endpoint).expanduser().resolve() == endpoint_path,
                 "TEMPO_PD_ENDPOINT_SERVICE_PROFILE differs from contract")
    os.environ["TEMPO_PD_ENDPOINT_SERVICE_PROFILE"] = str(endpoint_path)


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_stream_metrics_v1"
    if old in command:
        command[command.index(old)] = (
            "eval.sota_4node.run_tempo_go_c5_stream_client")
    else:
        raise ValueError("C5 client command lost canonical stream seam")
    workload = Path(command[command.index("--workload") + 1])
    first = next(
        (line for line in workload.read_text(encoding="utf-8").splitlines()
         if line.strip()),
        None,
    )
    if first is not None and "arrival_offset_ms" in json.loads(first):
        # Explicit phase schedules and --request-rate are mutually exclusive
        # in the native client.  The manifest's absolute offsets are the
        # authoritative arrival clock for C1/C2/C3 contention.
        marker = "--request-rate"
        if marker not in command:
            raise ValueError("canonical client lost request-rate seam")
        del command[command.index(marker):command.index(marker) + 2]
    return command


def main() -> int:
    signal.signal(signal.SIGTERM, _raise_native_node_signal)
    signal.signal(signal.SIGINT, _raise_native_node_signal)
    args = elastic.capacity._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    _require(args.repo_root in args.result_dir.parents,
             "C5 result directory must be below repository")
    arm = os.environ.get("TEMPO_GO_C5_ARM", "tempo")
    _require(
        arm in {
            "local", "remote", "predictor", "queue_gpu",
            "network_request_only", "app_global_only", "tempo",
        },
        "TEMPO_GO_C5_ARM is invalid",
    )

    hosts = args.hosts.split(",")
    _require(len(hosts) == 4 and len(set(hosts)) == 4,
             "C5 requires four unique hosts")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    _require((model / "config.json").is_file(), "C5 Qwen model is missing")
    workload = args.scout_root
    if workload.is_dir():
        workload = workload / "workloads/validation.jsonl"
    workload = workload.resolve()
    _require(workload.is_file(), "C5 workload is missing")
    frozen_contract, contract_path, contract_sha = _load_frozen_run_contract(
        repo_root=args.repo_root, workload=workload, arm=arm,
    )
    # The canonical perf lifecycle rewrites the warmup request IDs before
    # invoking the client.  Keep the original measured workload available so
    # the client can retain the explicit MISS/P_ONLY cache contract.
    os.environ["TEMPO_GO_C5_SOURCE_WORKLOAD"] = str(workload)

    global_path = _contract_artifact_path(frozen_contract, "global_profile")
    elastic_path = _contract_artifact_path(frozen_contract, "elastic_profile")
    endpoint_path = _contract_artifact_path(frozen_contract, "endpoint_profile")
    _bind_profile_environment(
        frozen_contract, global_path=global_path,
        elastic_path=elastic_path, endpoint_path=endpoint_path,
    )
    _validate_native_environment()
    sidecar_manifest = _contract_artifact_path(frozen_contract, "manifest")
    _require(sidecar_manifest.is_file(),
             "C5 workload sidecar manifest is required")
    profile = _validate_profile_bindings(
        global_path=global_path,
        elastic_path=elastic_path,
        endpoint_path=endpoint_path,
        workload_manifest=sidecar_manifest,
    )
    os.environ["TEMPO_GO_PROFILE"] = str(global_path)
    os.environ["TEMPO_GO_PROFILE_SHA256"] = profile.fingerprint_sha256
    os.environ["TEMPO_GO_ELASTIC_PROFILE"] = str(elastic_path)
    os.environ["TEMPO_GO_ENDPOINT_PROFILE"] = str(endpoint_path)
    # Pair 0's decoder is also the tokenizer endpoint.  Tokenization is an
    # application request to the vLLM API, never a cross-node clock probe.
    ports = perf._ports(args.port_slot, 0)
    os.environ["TEMPO_GO_TOKENIZER_URL"] = (
        f"http://{hosts[1]}:{ports['decode_api']}")

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
        try:
            raw_path = perf._lifecycle(
                args, lifecycle=0, stage_name="tempo_go_c5_discovery",
                router_mode="tempo_auto", workload_kind="validation",
                workload=workload, manifest=elastic_path, hosts=hosts,
                model=model, python=python,
                model_revision=_sha256(model / "config.json"),
            )
        except _NativeNodeSignal as exc:
            return 128 + exc.signum
    finally:
        perf._client_command = old_client
        perf._router_command = old_router
        perf._frontend_command = old_frontend
        perf._vllm_command = old_vllm
        perf._config_text = old_config
        chunk256._config_text = old_chunk_config
        legacy._proxy_command = old_proxy

    marker = args.result_dir / f"node-{args.node_index}-complete"
    marker.write_text("complete\n", encoding="utf-8")
    result = args.result_dir / "result.json"
    if args.node_index == 0:
        for index in range(4):
            common._wait_file(args.result_dir / f"node-{index}-complete", [])
        raw_value = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_workload = raw_value.get("workload")
        _require(isinstance(raw_workload, dict),
                 "native raw workload receipt is missing")
        raw_workload_path = raw_workload.get("explicit_path")
        raw_workload_sha256 = raw_workload.get("sha256")
        _require(
            isinstance(raw_workload_path, str)
            and isinstance(raw_workload_sha256, str)
            and len(raw_workload_sha256) == 64,
            "native raw rewritten workload identity is invalid",
        )
        resolved_raw_workload = Path(raw_workload_path).resolve()
        _require(resolved_raw_workload.is_file(),
                 "native raw rewritten workload is missing")
        _require(_sha256(resolved_raw_workload) == raw_workload_sha256,
                 "native raw rewritten workload SHA mismatch")
        value = {
            "schema": SCHEMA,
            "raw": str(raw_path.resolve()),
            "raw_sha256": _sha256(raw_path),
            "global_profile": str(global_path),
            "global_profile_sha256": profile.fingerprint_sha256,
            "global_profile_fingerprint_sha256": profile.fingerprint_sha256,
            "global_profile_file_sha256": _sha256(global_path),
            "elastic_profile": str(elastic_path),
            "elastic_profile_sha256": _sha256(elastic_path),
            "endpoint_profile": str(endpoint_path),
            "endpoint_profile_sha256": _sha256(endpoint_path),
            "workload": str(workload),
            "workload_sha256": _sha256(workload),
            "raw_workload": str(resolved_raw_workload),
            "raw_workload_sha256": raw_workload_sha256,
            "workload_manifest": (
                str(sidecar_manifest) if sidecar_manifest.is_file() else None),
            "workload_manifest_sha256": (
                _sha256(sidecar_manifest) if sidecar_manifest.is_file() else None),
            "run_contract": str(contract_path),
            "run_contract_sha256": contract_sha,
            "run_contract_fingerprint_sha256": frozen_contract[
                "fingerprint_sha256"],
            "transport": "LMCacheConnectorV1:UCX",
            "native_only": True,
            "arm": arm,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node_count": 4,
            "gpu_count": 16,
        }
        result.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                           encoding="utf-8")
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
