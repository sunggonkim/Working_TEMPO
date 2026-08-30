"""Saturated-loaded Qwen7B entry with max_num_seqs eight."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v22 as heavy_node


_ORIGINAL_COMMAND = heavy_node._vllm_command
_ORIGINAL_RUN = heavy_node._ORIGINAL_RUN


def _vllm_command(*args, **kwargs):
    command = _ORIGINAL_COMMAND(*args, **kwargs)
    command[command.index("--max-num-seqs") + 1] = "8"
    return command


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v19_qwen7b_loaded_saturated"
            if value == "eval.sota_4node.live_pd_controller_lmcache_v18_qwen7b_loaded_heavy"
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old_command = heavy_node._vllm_command
    old_run = heavy_node._ORIGINAL_RUN
    heavy_node._vllm_command = _vllm_command
    heavy_node._ORIGINAL_RUN = _run
    try:
        return heavy_node.main()
    finally:
        heavy_node._vllm_command = old_command
        heavy_node._ORIGINAL_RUN = old_run


if __name__ == "__main__":
    raise SystemExit(main())
