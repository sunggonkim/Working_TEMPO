#!/usr/bin/env python3
"""Launch one four-node actual-vLLM C6 decoder-victim qualification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from eval.sota_4node import run_tempo_go_c6_decoder_victim_client as client
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256
from eval.sota_4node import vllm_lmcache_elastic_pd_node as canonical
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as elastic
from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_pd_contention_node as contention
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf


SCHEMA = "tempo-go-c6-decoder-victim-node-result-v1"
STAGE_NAME = "tempo_go_c6_decoder_victim"
CLIENT_MODULE = "eval.sota_4node.run_tempo_go_c6_decoder_victim_client"
CONTRACT_ENV = "TEMPO_GO_C6_QUALIFICATION_CONTRACT"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qualification(repo_root: Path) -> tuple[Path, dict[str, object]]:
    raw = os.environ.get(
        CONTRACT_ENV,
        "eval/sota_4node/tempo_go_c6_qualification_contract_v1.json",
    )
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    perf._require(path.is_file(), "C6 qualification contract is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    perf._require(
        value.get("schema") == "tempo-go-c6-qualification-contract-v1",
        "C6 qualification contract schema differs",
    )
    perf._require(
        value.get("claim_boundary", {}).get("controller_performance_claim_allowed")
        is False,
        "C6 qualification cannot authorize a controller performance claim",
    )
    return path, value


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
    contract_path = Path(os.environ[CONTRACT_ENV]).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    decoder = client._decoder_contract(contract)
    if run_id.endswith("-warmup"):
        # The generic lifecycle materializes a renamed warmup JSONL, but the C6
        # client intentionally accepts only the frozen source workload.  Its
        # warmup branch uses that file solely as prompt templates and emits a
        # separate two-route correctness preflight, so preserve the strict
        # source path/digest contract here instead of weakening the client.
        repo_root = Path(__file__).resolve().parents[2]
        workload = (repo_root / decoder["source_workload"]["path"]).resolve()
        perf._require(workload.is_file(), "frozen C6 source workload is missing")
        perf._require(
            _sha256(workload) == decoder["source_workload"]["sha256"],
            "frozen C6 source workload digest differs during warmup",
        )
    command = [
        str(python), "-m", CLIENT_MODULE,
        "--base-url", base_url,
        "--model", str(model),
        "--served-model-name", perf.SERVED_MODEL,
        "--workload", str(workload),
        "--output", str(output),
        "--mode", mode,
        "--run-id", run_id,
        "--qualification-contract", str(contract_path),
        "--default-max-tokens", "128",
        "--max-workers", str(max_workers),
        "--request-rate", str(request_rate),
        "--timeout-s", "1200",
        "--decoder-reference-rate", str(
            decoder["aggressor"]["reference_rate_per_s"]
        ),
        "--remote-reference-rate", "6.8",
        "--load-fraction", str(decoder["aggressor"]["load_fraction"]),
        "--phase-duration-ms", str(decoder["phase_duration_ms"]),
        "--cooldown-s", str(decoder["cooldown_s"]),
    ]
    ingress = decoder.get("ingress", {})
    perf._require(isinstance(ingress, dict), "ingress contract section is malformed")
    command.extend((
        "--ingress-policy", str(ingress.get("policy", "shared_pool")),
        "--interactive-reserved-workers", str(
            ingress.get("interactive_reserved_workers", 0)),
    ))
    probe_urls = os.environ.get(contention.PROBE_URLS_ENV, "").split(",")
    perf._require(
        len(probe_urls) == 4 and all(url.startswith("http://") for url in probe_urls),
        "four endpoint evidence probe URLs are required",
    )
    for url in probe_urls:
        command.extend(("--endpoint-evidence-url", url))
    return command


def _validate_environment() -> None:
    perf._require(
        os.environ.get("TEMPO_PD_BENCHMARK_COLD_MEASURED") == "1",
        "C6 decoder victim requires exact cold measurement",
    )
    perf._require(
        os.environ.get("TEMPO_LMCACHE_NIXL_BACKEND") == "UCX",
        "C6 decoder victim requires official LMCache UCX",
    )
    perf._require(
        not os.environ.get("TEMPO_CXI_BACKGROUND_DUTY_CYCLE")
        and not os.environ.get("TEMPO_CXI_BACKGROUND_START_FILE"),
        "synthetic CXI background is forbidden in C6 qualification",
    )


def main() -> int:
    args = elastic.capacity._parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    perf._require(
        args.repo_root in args.result_dir.parents,
        "result directory must be below the repository",
    )
    _validate_environment()
    qualification_path, qualification = _qualification(args.repo_root)
    os.environ[CONTRACT_ENV] = str(qualification_path)
    decoder = client._decoder_contract(qualification)
    perf._require(
        args.request_rate == decoder["victim"]["offered_rate_per_s"]
        and args.max_workers == decoder["max_workers"],
        "node launch differs from frozen victim load",
    )

    workload = args.scout_root
    if workload.is_dir():
        workload = workload / "workloads/validation.jsonl"
    expected_workload = (
        args.repo_root / decoder["source_workload"]["path"]
    ).resolve()
    perf._require(workload == expected_workload, "source workload path differs")
    perf._require(
        _sha256(workload) == decoder["source_workload"]["sha256"],
        "source workload digest differs",
    )
    profile = (args.repo_root / decoder["profile"]["path"]).resolve()
    perf._require(profile.is_file(), "frozen C6 qualification profile is missing")
    perf._require(_sha256(profile) == decoder["profile"]["sha256"], "profile digest differs")

    hosts = args.hosts.split(",")
    perf._require(len(hosts) == 4 and len(set(hosts)) == 4, "four unique hosts required")
    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    perf._require((model / "config.json").is_file(), "Qwen model is missing")
    revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()

    # C6 performance uses the same native lifecycle for two immutable server
    # epochs: fixed-cross and full P-by-D mesh.  Let the selected client bind
    # its frozen profile/environment before any topology-dependent process is
    # spawned, while keeping qualification clients unchanged.
    configure = getattr(client, "configure_node_environment", None)
    if configure is not None:
        configure(
            repo_root=args.repo_root,
            qualification=qualification,
            hosts=hosts,
            port_slot=args.port_slot,
            elastic_profile=profile,
        )
    frozen_placement = decoder.get("remote_decode_placement")
    if frozen_placement is not None:
        perf._require(
            os.environ.get("TEMPO_PD_REMOTE_DECODE_PLACEMENT")
            == frozen_placement,
            "remote decoder placement differs from C6 contract",
        )

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
        perf._require(artifact.get("schema") == client.SCHEMA, "C6 client schema differs")
        analysis = artifact.get("analysis")
        perf._require(isinstance(analysis, dict), "C6 decoder analysis is missing")
        claim_boundary = qualification.get("claim_boundary", {})
        controller_allowed = (
            claim_boundary.get("controller_performance_claim_allowed") is True
        )
        performance_allowed = (
            claim_boundary.get("performance_claim_allowed") is True
            and artifact.get("performance_claim_allowed") is True
        )
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump({
                "schema": SCHEMA,
                "raw": str(raw_path.resolve()),
                "raw_sha256": _sha256(raw_path),
                "profile": str(profile),
                "profile_sha256": _sha256(profile),
                "source_workload": str(workload),
                "qualification_contract": str(qualification_path),
                "qualification_contract_sha256": _sha256(qualification_path),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "analysis": analysis,
                "controller_performance_run_allowed": controller_allowed,
                "performance_claim_allowed": performance_allowed,
                "purpose": qualification.get(
                    "purpose",
                    qualification.get("claim_boundary", {}).get(
                        "purpose", "C6_P2_decoder_victim_qualification_only"
                    ),
                ),
            }, stream, indent=2, sort_keys=True)
            stream.write("\n")
    else:
        common._wait_file(result_path, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
