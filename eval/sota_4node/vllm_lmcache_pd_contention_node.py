#!/usr/bin/env python3
"""Run one four-node actual-vLLM fixed-arm contention crossover screen."""

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


SCHEMA = "tempo-pd-contention-node-result-v7"
STAGE_NAME = "tempo_pd_contention_fixed"
CLIENT_MODULE = "eval.sota_4node.run_tempo_pd_contention_fixed_client"
PROBE_MODULE = "eval.sota_4node.tempo_pd_endpoint_probe"
PROBE_URLS_ENV = "TEMPO_PD_ENDPOINT_EVIDENCE_URLS"
FROZEN_MANIFEST_ENV = "TEMPO_PD_CONTENTION_FROZEN_MANIFEST"


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
        "--decoder-reference-rate", os.environ.get(
            "TEMPO_PD_CONTENTION_DECODER_REFERENCE_RATE", "32"),
        "--remote-reference-rate", os.environ.get(
            "TEMPO_PD_CONTENTION_REMOTE_REFERENCE_RATE", "6.8"),
        "--load-fraction", os.environ.get(
            "TEMPO_PD_CONTENTION_LOAD_FRACTION", "0.50"),
        "--phase-duration-ms", os.environ.get(
            "TEMPO_PD_CONTENTION_PHASE_DURATION_MS", "15000"),
        "--cooldown-s", os.environ.get(
            "TEMPO_PD_CONTENTION_COOLDOWN_S", "2"),
    ]
    probe_urls = os.environ.get(PROBE_URLS_ENV, "").split(",")
    perf._require(
        len(probe_urls) == 4 and all(url.startswith("http://") for url in probe_urls),
        "four endpoint evidence probe URLs are required",
    )
    for url in probe_urls:
        command.extend(("--endpoint-evidence-url", url))
    return command


def _probe_port(port_slot: int) -> int:
    port = 30_000 + port_slot
    perf._require(1024 <= port < 32768, "endpoint probe port is invalid")
    return port


def _probe_urls(hosts: list[str], port: int) -> list[str]:
    perf._require(
        len(hosts) == 4 and len(set(hosts)) == 4,
        "four unique probe hosts required",
    )
    return [f"http://{host}:{port}" for host in hosts]


def _probe_command(
    python: Path,
    *,
    node_index: int,
    hosts: list[str],
    port_slot: int,
) -> list[str]:
    perf._require(node_index in range(4), "probe node index is invalid")
    ports = perf._ports(port_slot, 0)
    is_prefill = node_index % 2 == 0
    role = "prefill" if is_prefill else "decoder"
    pair = node_index // 2
    engine_port = ports["prefill_api"] if is_prefill else ports["decode_api"]
    return [
        str(python), "-m", PROBE_MODULE,
        "--host", "0.0.0.0",
        "--port", str(_probe_port(port_slot)),
        "--endpoint-id", f"pair{pair}-{role}",
        "--role", role,
        "--pair-index", str(pair),
        "--vllm-metrics-url", f"http://{hosts[node_index]}:{engine_port}",
        "--served-model-name", perf.SERVED_MODEL,
    ]


def _profile(args) -> Path:
    raw = os.environ.get(
        "TEMPO_ELASTIC_PD_PROFILE", canonical._DEFAULT_PROFILE)
    path = Path(raw)
    if not path.is_absolute():
        path = args.repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), "frozen Elastic-PD profile is missing")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _frozen_workload_manifest(
    args, *, workload: Path, profile: Path,
) -> tuple[Path, str]:
    raw_path = os.environ.get(
        FROZEN_MANIFEST_ENV,
        "eval/sota_4node/tempo_pd_contention_workload_v4_frozen.json",
    )
    path = Path(raw_path)
    if not path.is_absolute():
        path = args.repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), "frozen contention manifest is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    perf._require(
        value.get("schema") == "tempo-pd-contention-frozen-manifest-v1",
        "frozen contention manifest schema mismatch",
    )
    perf._require(value.get("performance_claim_allowed") is False,
                  "contention calibration cannot permit a performance claim")
    perf._require(value.get("controller_tuning_allowed") is True,
                  "contention calibration is not authorized for tuning")
    expected = {
        "TEMPO_PD_CONTENTION_DECODER_REFERENCE_RATE":
            value["load"]["decoder_reference_rate_per_s"],
        "TEMPO_PD_CONTENTION_REMOTE_REFERENCE_RATE":
            value["load"]["remote_reference_rate_per_s"],
        "TEMPO_PD_CONTENTION_LOAD_FRACTION": value["load"]["fraction"],
        "TEMPO_PD_CONTENTION_PHASE_DURATION_MS": value["phase_duration_ms"],
        "TEMPO_PD_CONTENTION_COOLDOWN_S": value["cooldown_s"],
    }
    for name, frozen in expected.items():
        try:
            observed = float(os.environ[name])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"missing or invalid frozen value: {name}") from exc
        perf._require(observed == float(frozen),
                      f"runtime value differs from frozen manifest: {name}")
    perf._require(
        float(args.request_rate) == float(value["foreground_rate_per_s"]),
        "foreground rate differs from frozen manifest",
    )
    source = (args.repo_root / value["source_workload"]["path"]).resolve()
    frozen_profile = (args.repo_root / value["profile"]["path"]).resolve()
    perf._require(workload == source, "source workload path differs from manifest")
    perf._require(profile == frozen_profile, "profile path differs from manifest")
    perf._require(_sha256(workload) == value["source_workload"]["sha256"],
                  "source workload digest differs from manifest")
    perf._require(_sha256(profile) == value["profile"]["sha256"],
                  "profile digest differs from manifest")
    perf._require(os.environ.get("TEMPO_LMCACHE_NIXL_BACKEND") == "UCX",
                  "frozen contention transport requires UCX")
    return path, _sha256(path)


def _validate_environment() -> None:
    perf._require(
        os.environ.get("TEMPO_PD_BENCHMARK_COLD_MEASURED") == "1",
        "contention screen requires TEMPO_PD_BENCHMARK_COLD_MEASURED=1",
    )
    perf._require(
        not os.environ.get("TEMPO_CXI_BACKGROUND_DUTY_CYCLE")
        and not os.environ.get("TEMPO_CXI_BACKGROUND_START_FILE"),
        "contention screen forbids synthetic CXI background traffic",
    )


def main() -> int:
    args = elastic.capacity._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    perf._require(
        args.repo_root in args.result_dir.parents,
        "result directory must be below repository",
    )
    perf._require(
        args.request_rate > 0 and args.max_workers > 0,
        "request rate and workers must be positive",
    )
    _validate_environment()
    workload = args.scout_root
    if workload.is_dir():
        workload = workload / "workloads/validation.jsonl"
    perf._require(workload.is_file(), "explicit source workload is missing")
    hosts = args.hosts.split(",")
    perf._require(
        len(hosts) == 4 and len(set(hosts)) == 4,
        "four unique hosts required",
    )
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    perf._require((model / "config.json").is_file(), "Qwen model is missing")
    revision = hashlib.sha256(
        (model / "config.json").read_bytes()).hexdigest()
    profile = _profile(args)
    frozen_manifest, frozen_manifest_sha256 = _frozen_workload_manifest(
        args, workload=workload, profile=profile)
    probe_port = _probe_port(args.port_slot)
    probe_urls = _probe_urls(hosts, probe_port)
    os.environ[PROBE_URLS_ENV] = ",".join(probe_urls)

    probe = probe_handle = None
    try:
        probe, probe_handle = common._spawn(
            _probe_command(
                python,
                node_index=args.node_index,
                hosts=hosts,
                port_slot=args.port_slot,
            ),
            args.result_dir / f"node-{args.node_index}-endpoint-probe.log",
            dict(os.environ),
        )
        common._wait_url(
            probe_urls[args.node_index] + "/health", [probe])

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
            common._wait_file(
                args.result_dir / f"node-{index}-complete", [])
        artifact = json.loads(raw_path.read_text(encoding="utf-8"))
        perf._require(
            artifact.get("schema") == "tempo-pd-contention-fixed-client-v7",
            "contention client artifact schema mismatch",
        )
        perf._require(
            isinstance(artifact.get("crossover_gate"), dict),
            "contention crossover gate is missing",
        )
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump({
                "schema": SCHEMA,
                "raw": str(raw_path.resolve()),
                "profile": str(profile.resolve()),
                "source_workload": str(workload.resolve()),
                "frozen_workload_manifest": str(frozen_manifest),
                "frozen_workload_manifest_sha256": frozen_manifest_sha256,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "crossover_gate": artifact["crossover_gate"],
                "controller_tuning_allowed": artifact.get(
                    "controller_tuning_allowed") is True,
                "performance_claim_allowed": False,
                "purpose": "workload_characterization_only",
            }, stream, indent=2, sort_keys=True)
            stream.write("\n")
    else:
        common._wait_file(result_path, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
