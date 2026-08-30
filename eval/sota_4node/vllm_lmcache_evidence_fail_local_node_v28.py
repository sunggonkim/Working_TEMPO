#!/usr/bin/env python3
"""Final evidence-gated fail-local candidate lifecycle."""

from eval.sota_4node import vllm_lmcache_capacity_candidate_node_v13 as base


_ORIGINAL_ROUTER_COMMAND = base._router_command


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER_COMMAND(*args, **kwargs)
    index = command.index("eval.sota_4node.tempo_pd_capacity_router_v13")
    command[index] = "eval.sota_4node.tempo_pd_evidence_fail_local_router_v28"
    return command


def main() -> int:
    base._router_command = _router_command
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
