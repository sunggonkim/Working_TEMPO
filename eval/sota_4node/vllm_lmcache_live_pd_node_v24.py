"""Long-context TTFT-oriented Qwen7B entry over saturated v23."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v23 as saturated_node


_ORIGINAL_RUN = saturated_node._ORIGINAL_RUN


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v20_qwen7b_long_ttft"
            if value == "eval.sota_4node.live_pd_controller_lmcache_v19_qwen7b_loaded_saturated"
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old = saturated_node._ORIGINAL_RUN
    saturated_node._ORIGINAL_RUN = _run
    try:
        return saturated_node.main()
    finally:
        saturated_node._ORIGINAL_RUN = old


if __name__ == "__main__":
    raise SystemExit(main())
