"""vLLM 0.26 CLI-compatible wrapper for the race-safe live-P/D node."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as impl
from eval.sota_4node import vllm_lmcache_live_pd_node_v3 as race_safe


_ORIGINAL_COMMAND = impl._vllm_command


def _vllm_command(*args, **kwargs):
    command = _ORIGINAL_COMMAND(*args, **kwargs)
    return [value for value in command if value != "--disable-log-requests"]


def main() -> int:
    old = impl._vllm_command
    impl._vllm_command = _vllm_command
    try:
        return race_safe.main()
    finally:
        impl._vllm_command = old


if __name__ == "__main__":
    raise SystemExit(main())
