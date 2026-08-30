"""Heavy-loaded Qwen7B entry with max_num_seqs four."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v19 as server_node
from eval.sota_4node import vllm_lmcache_live_pd_node_v21 as short_node


_ORIGINAL_COMMAND = server_node._vllm_command
_ORIGINAL_RUN = short_node._ORIGINAL_RUN


def _vllm_command(*args, **kwargs):
    command = _ORIGINAL_COMMAND(*args, **kwargs)
    command[command.index("--max-num-seqs") + 1] = "4"
    return command


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v18_qwen7b_loaded_heavy"
            if value == "eval.sota_4node.live_pd_controller_lmcache_v17_qwen7b_loaded_short"
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old_command = server_node._vllm_command
    old_run = short_node._ORIGINAL_RUN
    server_node._vllm_command = _vllm_command
    short_node._ORIGINAL_RUN = _run
    try:
        return short_node.main()
    finally:
        server_node._vllm_command = old_command
        short_node._ORIGINAL_RUN = old_run


if __name__ == "__main__":
    raise SystemExit(main())
