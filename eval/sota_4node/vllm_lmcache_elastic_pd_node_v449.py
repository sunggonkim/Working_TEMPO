#!/usr/bin/env python3
"""Run v447 weighted credits with first-response lease release."""

from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as base
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v447 as weighted


_ORIGINAL_ROUTER_COMMAND = base._router_command


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER_COMMAND(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_elastic_router_v445"
    command[command.index(old)] = "eval.sota_4node.tempo_pd_elastic_router_v449"
    return command


def main() -> int:
    old = base._router_command
    base._router_command = _router_command
    try:
        return weighted.main()
    finally:
        base._router_command = old


if __name__ == "__main__":
    raise SystemExit(main())
