"""Loaded Qwen7B long-context node with one-pass prefill batches."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v10 as loaded_node
from eval.sota_4node import vllm_lmcache_live_pd_node_v13 as qwen_stream_node
from eval.sota_4node import vllm_lmcache_live_pd_node_v8 as gpu_v3


_ORIGINAL_COMMAND = loaded_node._vllm_command
_ORIGINAL_CONFIG = gpu_v3._config_text
_ORIGINAL_RUN = qwen_stream_node._ORIGINAL_RUN


def _vllm_command(*args, **kwargs):
    command = _ORIGINAL_COMMAND(*args, **kwargs)
    command[command.index("--max-model-len") + 1] = "8192"
    if "--max-num-batched-tokens" in command:
        command[command.index("--max-num-batched-tokens") + 1] = "8192"
    else:
        command.extend(("--max-num-batched-tokens", "8192"))
    return command


def _config_text(*args, **kwargs) -> str:
    text = _ORIGINAL_CONFIG(*args, **kwargs)
    return text.replace("pd_max_prefill_len: 2048", "pd_max_prefill_len: 8192")


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v15_qwen7b_long_loaded"
            if value == "eval.sota_4node.live_pd_controller_lmcache_v11_qwen7b"
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old_command = loaded_node._vllm_command
    old_config = gpu_v3._config_text
    old_run = qwen_stream_node._ORIGINAL_RUN
    loaded_node._vllm_command = _vllm_command
    gpu_v3._config_text = _config_text
    qwen_stream_node._ORIGINAL_RUN = _run
    try:
        return qwen_stream_node.main()
    finally:
        loaded_node._vllm_command = old_command
        gpu_v3._config_text = old_config
        qwen_stream_node._ORIGINAL_RUN = old_run


if __name__ == "__main__":
    raise SystemExit(main())
