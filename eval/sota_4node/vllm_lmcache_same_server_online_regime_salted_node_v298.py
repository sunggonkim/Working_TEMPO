#!/usr/bin/env python3
"""Use the online regime router with collision-isolated measured requests."""

from eval.sota_4node import vllm_lmcache_same_server_online_regime_mixed_node_v293 as base


_BASE_CLIENT = base._client_command


def _client_command(*args, **kwargs):
    command = _BASE_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_same_server_mixed_only_client_v265"
    command[command.index(old)] = (
        "eval.sota_4node.run_tempo_pd_same_server_mixed_only_client_salted_v297")
    return command


def main() -> int:
    original = base._client_command
    base._client_command = _client_command
    try:
        return base.main()
    finally:
        base._client_command = original


if __name__ == "__main__":
    raise SystemExit(main())
