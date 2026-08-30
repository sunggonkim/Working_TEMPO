"""Loaded crossover node routing to the callback-normalized client."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v10 as loaded_node


_ORIGINAL_RUN = loaded_node._ORIGINAL_RUN


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v9_loaded_fix"
            if value == "eval.sota_4node.live_pd_controller_lmcache_v8_loaded"
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old = loaded_node._ORIGINAL_RUN
    loaded_node._ORIGINAL_RUN = _run
    try:
        return loaded_node.main()
    finally:
        loaded_node._ORIGINAL_RUN = old


if __name__ == "__main__":
    raise SystemExit(main())
