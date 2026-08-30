"""Loaded Qwen7B 3K crossover entry over the audited v19 server config."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v19 as loaded_long_node


_ORIGINAL_RUN = loaded_long_node._ORIGINAL_RUN


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v16_qwen7b_loaded_3k"
            if value == "eval.sota_4node.live_pd_controller_lmcache_v15_qwen7b_long_loaded"
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old = loaded_long_node._ORIGINAL_RUN
    loaded_long_node._ORIGINAL_RUN = _run
    try:
        return loaded_long_node.main()
    finally:
        loaded_long_node._ORIGINAL_RUN = old


if __name__ == "__main__":
    raise SystemExit(main())
