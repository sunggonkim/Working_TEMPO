"""Qwen2.5-7B stream-synchronized loaded crossover node."""

from __future__ import annotations

from pathlib import Path

from eval.sota_4node import vllm_lmcache_live_pd_node_v12 as stream_node


_ORIGINAL_DIV = Path.__truediv__
_ORIGINAL_RUN = stream_node._ORIGINAL_RUN
_TINY_RELATIVE = "models/TinyLlama-1.1B-Chat-v1.0"
_QWEN_RELATIVE = "models/Qwen2.5-7B-Instruct"


def _div(self: Path, key):
    if key == _TINY_RELATIVE:
        key = _QWEN_RELATIVE
    return _ORIGINAL_DIV(self, key)


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v11_qwen7b"
            if value == "eval.sota_4node.live_pd_controller_lmcache_v10_streamsync"
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old_div = Path.__truediv__
    old_run = stream_node._ORIGINAL_RUN
    Path.__truediv__ = _div
    stream_node._ORIGINAL_RUN = _run
    try:
        return stream_node.main()
    finally:
        Path.__truediv__ = old_div
        stream_node._ORIGINAL_RUN = old_run


if __name__ == "__main__":
    raise SystemExit(main())
