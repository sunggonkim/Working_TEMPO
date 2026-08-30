#!/usr/bin/env python3
"""Node entrypoint for safe polling-loop local-rescue950 E12."""

from __future__ import annotations
from pathlib import Path
from eval.sota_4node import vllm_lmcache_tp16_hybrid_boost_node_v5 as runtime
from eval.sota_4node import vllm_lmcache_tp16_quiescence_scout_node_v1 as base

base.RUNNER_MODULE = (
    "eval.sota_4node.run_vllm_lmcache_tp16_deadline_e12_localrescue950_safe_entry"
)
base.PLAN_RELATIVE = Path(
    "eval/sota_4node/real_tp16_deadline_e12_localrescue950_safe.json"
)
base.PINNED_SITE_RELATIVE = Path(
    "eval/sota_4node/vllm_quiescence_sitecustomize_async_v8"
)

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    runtime._configure_nixl_runtime(repo_root)
    base.main()

if __name__ == "__main__":
    main()
