#!/usr/bin/env python3
"""Explicit output64 diagnostic selecting the force-local router."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_same_server_balanced_node_v72 as balanced
from eval.sota_4node import vllm_lmcache_same_server_production_explicit_output64_node_v91 as prior


_ORIGINAL_ROUTER_COMMAND = balanced._ORIGINAL_ROUTER_COMMAND


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER_COMMAND(*args, **kwargs)
    command[command.index("eval.sota_4node.tempo_pd_capacity_router_v13")] = (
        "eval.sota_4node.tempo_pd_same_server_balanced_force_local_router_v94")
    return command


def main() -> int:
    balanced._router_command = _router_command
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
