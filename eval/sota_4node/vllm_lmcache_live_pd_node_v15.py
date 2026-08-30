"""Qwen7B unloaded node routing to same-prompt calibration."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v14 as qwen_node


_ORIGINAL_RUN = qwen_node._ORIGINAL_RUN


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v13_qwen7b_sameprompt"
            if value == "eval.sota_4node.live_pd_controller_lmcache_v12_qwen7b_unloaded"
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old = qwen_node._ORIGINAL_RUN
    qwen_node._ORIGINAL_RUN = _run
    try:
        return qwen_node.main()
    finally:
        qwen_node._ORIGINAL_RUN = old


if __name__ == "__main__":
    raise SystemExit(main())
