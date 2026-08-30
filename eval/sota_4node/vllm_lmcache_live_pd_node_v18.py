"""Qwen7B long-context node with the batching override at the final hook."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v16 as long_node


_ORIGINAL_FINAL_COMMAND = long_node._vllm_command


def _vllm_command(*args, **kwargs):
    command = _ORIGINAL_FINAL_COMMAND(*args, **kwargs)
    if "--max-num-batched-tokens" in command:
        command[command.index("--max-num-batched-tokens") + 1] = "8192"
    else:
        command.extend(("--max-num-batched-tokens", "8192"))
    return command


def main() -> int:
    old = long_node._vllm_command
    long_node._vllm_command = _vllm_command
    try:
        return long_node.main()
    finally:
        long_node._vllm_command = old


if __name__ == "__main__":
    raise SystemExit(main())
