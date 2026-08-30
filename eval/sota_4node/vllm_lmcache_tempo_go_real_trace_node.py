#!/usr/bin/env python3
"""One actual four-node TEMPO-PD epoch for a frozen Mooncake population.

This is a thin carrier adapter.  The P/D, official LMCache/NIXL, router,
admission and stream lifecycle remain the proven Elastic-PD lifecycle; only
the frontend module and client module are replaced so the source-bound
token-ID population can be sent without reconstructing private prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from eval.sota_4node import run_tempo_go_real_trace_stream as real_client
from eval.sota_4node import tempo_pd_real_trace_frontend as real_frontend
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256
from eval.sota_4node import vllm_lmcache_elastic_pd_node as elastic
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as v445
from eval.sota_4node import vllm_lmcache_live_pd_node_v1 as common
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--scout-root", type=Path, required=True)
    parser.add_argument("--node-index", type=int, choices=range(4), required=True)
    parser.add_argument("--hosts", required=True)
    parser.add_argument("--port-slot", type=int, required=True)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--ttft-slo-ms", type=float, default=3000)
    parser.add_argument("--tpot-slo-ms", type=float, default=250)
    parser.add_argument("--e2e-slo-ms", type=float, default=12000)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--business-profile", type=Path, required=True)
    parser.add_argument("--wire-arm", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    return parser.parse_args()


def _client_command(
    python: Path, *, base_url: str, model: Path, workload: Path,
    output: Path, mode: str, run_id: str, request_rate: float,
    max_workers: int,
) -> list[str]:
    del request_rate
    args = _parse_state
    # The generic lifecycle writes a renamed warmup JSONL.  A source-bound
    # population is immutable, so warmup must use the verified original
    # manifest/workload rather than silently creating a second population.
    if run_id.endswith("-warmup"):
        workload = Path(args["source_workload"])
    command = [
        str(python), "-m", "eval.sota_4node.run_tempo_go_real_trace_stream",
        "--base-url", base_url,
        "--model", str(model),
        "--served-model-name", perf.SERVED_MODEL,
        "--workload", str(workload),
        "--output", str(output),
        "--mode", mode,
        "--run-id", run_id,
        "--default-max-tokens", "128",
        "--max-workers", str(max_workers),
        "--ingress-policy", "shared_pool",
        "--timeout-s", "1200",
        "--population-manifest", str(args["population_manifest"]),
        "--wire-arm", str(args["wire_arm"]),
        "--business-profile", str(args["business_profile"]),
    ]
    if run_id.endswith("-warmup"):
        command.extend(["--wire-namespace", "warmup"])
    return command


_parse_state: dict[str, str] = {}


def _frontend_command(
    python: Path, *, host0: str, host2: str, ports: dict[str, int],
) -> list[str]:
    return [
        str(python), "-m", "eval.sota_4node.tempo_pd_real_trace_frontend",
        "--host", "0.0.0.0", "--port", str(ports["frontend"]),
        "--pair-url", f"http://{host0}:{ports['pair_router']}",
        "--pair-url", f"http://{host2}:{ports['pair_router']}",
    ]


def main() -> int:
    global _parse_state
    args = _parse()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.scout_root = args.scout_root.resolve()
    args.population_manifest = args.population_manifest.resolve()
    args.business_profile = args.business_profile.resolve()
    args.profile = args.profile.resolve()
    perf._require(args.repo_root in args.result_dir.parents,
                  "result directory must be below repository")
    perf._require(args.scout_root.is_file(), "real-trace workload is missing")
    perf._require(args.population_manifest.is_file(),
                  "real-trace population manifest is missing")
    perf._require(args.business_profile.is_file(),
                  "real-trace business profile is missing")
    perf._require(args.profile.is_file(), "Elastic-PD profile is missing")
    hosts = args.hosts.split(",")
    perf._require(len(hosts) == 4 and len(set(hosts)) == 4,
                  "four unique hosts required")
    perf._require(args.max_workers >= 2, "max-workers must be positive")
    perf._require(args.wire_arm in real_client.WIRE_ARMS,
                  "unsupported real-trace wire arm")

    model = args.repo_root / "models/Qwen2.5-7B-Instruct"
    python = args.repo_root / ".vllm_venv/bin/python"
    perf._require((model / "config.json").is_file(), "Qwen model is missing")
    model_revision = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()

    _parse_state = {
        "source_workload": str(args.scout_root),
        "population_manifest": str(args.population_manifest),
        "business_profile": str(args.business_profile),
        "wire_arm": args.wire_arm,
    }
    old_client = perf._client_command
    old_router = perf._router_command
    old_frontend = perf._frontend_command
    old_vllm = perf._vllm_command
    old_config = perf._config_text
    old_proxy = legacy._proxy_command
    old_chunk_config = chunk256._config_text
    perf._client_command = _client_command
    perf._router_command = elastic._router_command
    perf._frontend_command = _frontend_command
    perf._vllm_command = elastic._vllm_command
    perf._config_text = elastic._config_text
    chunk256._config_text = elastic._config_text
    legacy._proxy_command = chunk256._proxy_command
    # The official population contains the exact prefixes whose residency is
    # measured; generic lifecycle warmup would poison the cold-start arm.
    os.environ["TEMPO_GO_SKIP_WARMUP"] = "1"
    if args.wire_arm in {"remote", "predictor"}:
        os.environ["TEMPO_REAL_TRACE_NATURAL_CACHE"] = "1"
    try:
        raw = perf._lifecycle(
            args,
            lifecycle=0,
            stage_name=f"real_trace_{args.wire_arm}",
            router_mode="tempo_auto",
            workload_kind="validation",
            workload=args.scout_root,
            manifest=args.profile,
            hosts=hosts,
            model=model,
            python=python,
            model_revision=model_revision,
        )
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
        for node_index in range(4):
            common._wait_file(args.result_dir / f"node-{node_index}-complete", [])
        artifact = json.loads(raw.read_text(encoding="utf-8"))
        result.write_text(json.dumps({
            "schema": "tempo-go-real-trace-native-result-v1",
            "raw": str(raw.resolve()),
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "workload": str(args.scout_root),
            "population_manifest": str(args.population_manifest),
            "business_profile": str(args.business_profile),
            "profile": str(args.profile),
            "wire_arm": args.wire_arm,
            "request_count": len(artifact.get("requests", [])),
            "performance_claim_allowed": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        common._wait_file(result, [])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
