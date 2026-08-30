"""Diagnostic live-P/D node that logs the official proxy's HTTP error body."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as impl
from eval.sota_4node import vllm_lmcache_live_pd_node_v5 as token_accurate


_ORIGINAL_PROXY_COMMAND = impl._proxy_command


def _proxy_command(*args, **kwargs):
    command = _ORIGINAL_PROXY_COMMAND(*args, **kwargs)
    command[1] = "-m"
    command.insert(2, "eval.sota_4node.lmcache_disagg_proxy_diagnostic_v1")
    return command


def main() -> int:
    old = impl._proxy_command
    impl._proxy_command = _proxy_command
    try:
        return token_accurate.main()
    finally:
        impl._proxy_command = old


if __name__ == "__main__":
    raise SystemExit(main())
