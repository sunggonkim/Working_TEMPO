#!/usr/bin/env python3
"""Run the four-node P-only KV-path attribution campaign."""

from __future__ import annotations

import hashlib
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


SCHEMA = "tempo-pd-kv-only-attribution-node-v1"
CLIENT_SCHEMA = "tempo-pd-kv-only-attribution-client-v2"
CLIENT_MODULE = "eval.sota_4node.run_tempo_pd_kv_only_attribution_client"
STAGE_NAME = "tempo_pd_kv_only_attribution"
DEFAULT_READINESS_S = 3600.0
COUPLED_MANIFEST_ENV = "TEMPO_PD_C3_COUPLED_MANIFEST"


def _client_command(
    python: Path,
    *,
    base_url: str,
    model: Path,
    workload: Path,
    output: Path,
    mode: str,
    run_id: str,
    request_rate: float,
    max_workers: int,
) -> list[str]:
    command = [
        str(python),
        "-m",
        CLIENT_MODULE,
        "--base-url",
        base_url,
        "--model",
        str(model),
        "--served-model-name",
        perf.SERVED_MODEL,
        "--workload",
        str(workload),
        "--output",
        str(output),
        "--mode",
        mode,
        "--run-id",
        run_id,
        "--max-workers",
        str(max_workers),
        "--request-rate",
        str(request_rate),
        "--timeout-s",
        "600",
        "--phase-duration-ms",
        os.environ.get("TEMPO_PD_KV_ATTR_PHASE_DURATION_MS", "8000"),
        "--cooldown-s",
        os.environ.get("TEMPO_PD_KV_ATTR_COOLDOWN_S", "2"),
    ]
    probe_urls = os.environ.get(contention.PROBE_URLS_ENV, "").split(",")
    perf._require(
        len(probe_urls) == 4 and all(value.startswith("http://") for value in probe_urls),
        "four endpoint evidence probe URLs are required",
    )
    for value in probe_urls:
        command.extend(("--endpoint-evidence-url", value))
    return command


def _validate_environment() -> None:
    perf._require(os.environ.get("TEMPO_PD_KV_ATTR_APPROVED") == "YES",
                  "explicit P-only attribution approval is required")
    perf._require(os.environ.get("TEMPO_PD_BENCHMARK_COLD_MEASURED") == "1",
                  "cold foreground validation must be enabled")
    perf._require(os.environ.get("TEMPO_VLLM_DECODER_PREFIX_CACHING") == "0",
                  "decoder prefix caching must be disabled")
    perf._require(os.environ.get("TEMPO_LMCACHE_NIXL_BACKEND") == "UCX",
                  "P-only attribution requires UCX")
    perf._require(
        not os.environ.get("TEMPO_CXI_BACKGROUND_DUTY_CYCLE")
        and not os.environ.get("TEMPO_CXI_BACKGROUND_START_FILE"),
        "P-only attribution forbids synthetic network traffic",
    )


def _readiness_timeout_from_environment() -> float:
    try:
        value = float(os.environ.get(
            "TEMPO_PD_KV_ATTR_READINESS_S", str(DEFAULT_READINESS_S)))
    except ValueError as exc:
        raise RuntimeError(
            "TEMPO_PD_KV_ATTR_READINESS_S must be numeric") from exc
    perf._require(
        value >= common.READINESS_S and value <= 3600.0,
        "P-only attribution readiness must be in [600, 3600] seconds",
    )
    return value


def _coupled_manifest(
    args, *, workload: Path, profile: Path,
) -> tuple[Path | None, str | None]:
    try:
        decoder_hot_rate = float(os.environ.get(
            "TEMPO_PD_KV_ATTR_DECODER_HOT_RATE", "0"))
    except ValueError as exc:
        raise ValueError("decoder-hot rate is not numeric") from exc
    if decoder_hot_rate == 0.0:
        perf._require(not os.environ.get(COUPLED_MANIFEST_ENV),
                      "uncoupled attribution cannot declare a C3 manifest")
        return None, None
    perf._require(os.environ.get("TEMPO_PD_C3_APPROVED") == "YES",
                  "explicit coupled C3 approval is required")
    raw_path = os.environ.get(COUPLED_MANIFEST_ENV)
    perf._require(bool(raw_path), "coupled C3 manifest is required")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = args.repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), "coupled C3 manifest is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    manifest_schema = value.get("schema")
    perf._require(
        manifest_schema in {
            "tempo-pd-c3-coupled-pilot-manifest-v1",
            "tempo-pd-c3-coupled-abba-manifest-v2",
        },
        "coupled C3 manifest schema mismatch",
    )
    perf._require(value.get("performance_claim_allowed") is False,
                  "coupled C3 pilot cannot permit a performance claim")
    try:
        runtime_repetitions = int(os.environ.get(
            "TEMPO_PD_KV_ATTR_REPETITIONS", "1"))
    except ValueError as exc:
        raise ValueError("coupled C3 runtime repetitions are invalid") from exc
    runtime_arm_order = os.environ.get(
        "TEMPO_PD_KV_ATTR_ARM_ORDER", "local_remote")
    perf._require(runtime_repetitions == int(value["replicates"]),
                  "coupled C3 repetitions differ from manifest")
    perf._require(
        runtime_arm_order == value.get("arm_order_policy", "local_remote"),
        "coupled C3 arm order differs from manifest",
    )
    if manifest_schema == "tempo-pd-c3-coupled-pilot-manifest-v1":
        perf._require(
            runtime_repetitions == 1 and runtime_arm_order == "local_remote",
            "coupled C3 v1 pilot geometry differs",
        )
    else:
        perf._require(
            runtime_repetitions == 2 and runtime_arm_order == "paired_abba"
            and value.get("within_rate_block_order")
            == ["local", "remote", "remote", "local"],
            "coupled C3 ABBA geometry differs",
        )
    expected_rates = tuple(float(item) for item in value["p_only_rates_per_s"])
    try:
        observed_rates = tuple(float(item) for item in os.environ[
            "TEMPO_PD_KV_ATTR_RATES"].split(","))
    except (KeyError, ValueError) as exc:
        raise ValueError("coupled C3 runtime rates are invalid") from exc
    perf._require(observed_rates == expected_rates,
                  "coupled C3 rates differ from manifest")
    perf._require(
        decoder_hot_rate == float(value["decoder_hot_rate_per_s"]),
        "coupled C3 decoder-hot rate differs from manifest",
    )
    perf._require(
        float(args.request_rate) == float(value["foreground_rate_per_s"]),
        "coupled C3 foreground rate differs from manifest",
    )
    perf._require(
        float(os.environ["TEMPO_PD_KV_ATTR_PHASE_DURATION_MS"])
        == float(value["phase_duration_ms"]),
        "coupled C3 phase duration differs from manifest",
    )
    perf._require(
        float(os.environ["TEMPO_PD_KV_ATTR_COOLDOWN_S"])
        == float(value["cooldown_s"]),
        "coupled C3 cooldown differs from manifest",
    )
    source = (args.repo_root / value["source_workload"]["path"]).resolve()
    frozen_profile = (args.repo_root / value["profile"]["path"]).resolve()
    perf._require(workload == source,
                  "coupled C3 source workload path differs")
    perf._require(profile == frozen_profile,
                  "coupled C3 profile path differs")
    perf._require(
        hashlib.sha256(workload.read_bytes()).hexdigest()
        == value["source_workload"]["sha256"],
        "coupled C3 source workload digest differs",
    )
    perf._require(
        hashlib.sha256(profile.read_bytes()).hexdigest()
        == value["profile"]["sha256"],
        "coupled C3 profile digest differs",
    )
    perf._require(value.get("transport") == "LMCacheConnectorV1:UCX",
                  "coupled C3 transport contract differs")
    if manifest_schema == "tempo-pd-c3-coupled-abba-manifest-v2":
        parent = value.get("parent_pilot")
        perf._require(isinstance(parent, dict),
                      "coupled C3 ABBA parent pilot is missing")
        for name in ("result", "characterization"):
            item = parent.get(name)
            perf._require(isinstance(item, dict),
                          f"coupled C3 ABBA parent {name} is missing")
            artifact = (args.repo_root / item["path"]).resolve()
            perf._require(artifact.is_file(),
                          f"coupled C3 ABBA parent {name} file is missing")
            perf._require(
                hashlib.sha256(artifact.read_bytes()).hexdigest()
                == item["sha256"],
                f"coupled C3 ABBA parent {name} digest differs",
            )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


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
    readiness_timeout_s = _readiness_timeout_from_environment()
    # Perlmutter cold imports from the shared project filesystem can exceed
    # the generic 600 s timeout before a GPU context is created.  This changes
    # startup admission only; startup remains outside every measured block.
    common.READINESS_S = readiness_timeout_s
    workload = args.scout_root
    if workload.is_dir():
        workload = workload / "workloads/validation.jsonl"
    perf._require(workload.is_file(), "explicit source workload is missing")
    hosts = args.hosts.split(",")
    perf._require(len(hosts) == 4 and len(set(hosts)) == 4,
                  "four unique hosts are required")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    perf._require((model / "config.json").is_file(), "Qwen model is missing")
    revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
    profile = contention._profile(args)
    coupled_manifest, coupled_manifest_sha256 = _coupled_manifest(
        args, workload=workload, profile=profile)
    probe_port = contention._probe_port(args.port_slot)
    probe_urls = contention._probe_urls(hosts, probe_port)
    os.environ[contention.PROBE_URLS_ENV] = ",".join(probe_urls)

    probe = probe_handle = None
    try:
        probe, probe_handle = common._spawn(
            contention._probe_command(
                python,
                node_index=args.node_index,
                hosts=hosts,
                port_slot=args.port_slot,
            ),
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
        perf._require(artifact.get("schema") == CLIENT_SCHEMA,
                      "P-only attribution client schema mismatch")
        perf._require(artifact.get("performance_claim_allowed") is False,
                      "P-only attribution cannot permit a performance claim")
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump({
                "schema": SCHEMA,
                "raw": str(raw_path.resolve()),
                "profile": str(profile.resolve()),
                "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                "source_workload": str(workload.resolve()),
                "source_workload_sha256": hashlib.sha256(
                    workload.read_bytes()).hexdigest(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "block_count": len(artifact.get("summaries", [])),
                "stopped_after_first_invalid_block": artifact.get(
                    "stopped_after_first_invalid_block"),
                "performance_claim_allowed": False,
                "physical_switch_bottleneck_claim_allowed": False,
                "purpose": "P-only KV-transfer/receiver component attribution",
                "workload_mode": artifact.get("workload_mode"),
                "decoder_hot_rate_per_s": artifact.get(
                    "decoder_hot_rate_per_s"),
                "repetitions_per_rate": artifact.get(
                    "repetitions_per_rate"),
                "arm_order_policy": artifact.get("arm_order_policy"),
                "paired_semantic_schedules_exact": artifact.get(
                    "paired_semantic_schedules_exact"),
                "startup_readiness_timeout_s": readiness_timeout_s,
                "coupled_manifest": (
                    str(coupled_manifest) if coupled_manifest else None),
                "coupled_manifest_sha256": coupled_manifest_sha256,
            }, stream, indent=2, sort_keys=True)
            stream.write("\n")
    else:
        common._wait_file(result_path, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
