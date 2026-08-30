"""Qwen7B 8K P/D node with single-pass prefill batching.

LMCache PDBackendAsync reserves the request's total chunk count once.  vLLM's
default 2048-token chunked prefill can present cumulative 2048, then N-token
stores for one request, violating that reservation.  Match the frozen 8192
context and PD limits so each measured prompt is presented in one prefill.
"""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v16 as long_node
from eval.sota_4node import vllm_lmcache_live_pd_node_v4 as cli_compatible


_ORIGINAL_COMMAND = cli_compatible._vllm_command


def _vllm_command(*args, **kwargs):
    command = _ORIGINAL_COMMAND(*args, **kwargs)
    if "--max-num-batched-tokens" in command:
        command[command.index("--max-num-batched-tokens") + 1] = "8192"
    else:
        command.extend(("--max-num-batched-tokens", "8192"))
    return command


def main() -> int:
    old = cli_compatible._vllm_command
    cli_compatible._vllm_command = _vllm_command
    try:
        return long_node.main()
    finally:
        cli_compatible._vllm_command = old


if __name__ == "__main__":
    raise SystemExit(main())
