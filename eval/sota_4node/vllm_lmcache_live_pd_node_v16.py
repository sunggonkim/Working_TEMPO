"""Qwen7B 8K-context live-P/D node."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v15 as qwen_node
from eval.sota_4node import vllm_lmcache_live_pd_node_v4 as cli_compatible
from eval.sota_4node import vllm_lmcache_live_pd_node_v8 as gpu_v3


_ORIGINAL_RUN = qwen_node._ORIGINAL_RUN
_ORIGINAL_COMMAND = cli_compatible._vllm_command
_ORIGINAL_CONFIG = gpu_v3._config_text


def _vllm_command(*args, **kwargs):
    command = _ORIGINAL_COMMAND(*args, **kwargs)
    index = command.index("--max-model-len") + 1
    command[index] = "8192"
    return command


def _config_text(*args, **kwargs) -> str:
    text = _ORIGINAL_CONFIG(*args, **kwargs)
    return text.replace("pd_max_prefill_len: 2048", "pd_max_prefill_len: 8192")


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v14_qwen7b_longcontext"
            if value == "eval.sota_4node.live_pd_controller_lmcache_v13_qwen7b_sameprompt"
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old_run = qwen_node._ORIGINAL_RUN
    old_command = cli_compatible._vllm_command
    old_config = gpu_v3._config_text
    qwen_node._ORIGINAL_RUN = _run
    cli_compatible._vllm_command = _vllm_command
    gpu_v3._config_text = _config_text
    try:
        return qwen_node.main()
    finally:
        qwen_node._ORIGINAL_RUN = old_run
        cli_compatible._vllm_command = old_command
        gpu_v3._config_text = old_config


if __name__ == "__main__":
    raise SystemExit(main())
