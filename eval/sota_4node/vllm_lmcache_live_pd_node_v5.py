"""Live-P/D node using the token-accurate v5 SSE client."""

from __future__ import annotations

from pathlib import Path

from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as impl
from eval.sota_4node import vllm_lmcache_live_pd_node_v4 as cli_compatible


_ORIGINAL_MKDIR = Path.mkdir
_ORIGINAL_RUN = impl.subprocess.run


def _mkdir(self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False):
    if self.name in {"lmcache_always_remote", "tempo_admission"}:
        exist_ok = True
    return _ORIGINAL_MKDIR(self, mode=mode, parents=parents, exist_ok=exist_ok)


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v5"
            if value in {
                "eval.sota_4node.live_pd_controller_lmcache_v3",
                "eval.sota_4node.live_pd_controller_lmcache_v4",
            }
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old_mkdir = Path.mkdir
    old_run = impl.subprocess.run
    old_command = impl._vllm_command
    Path.mkdir = _mkdir
    impl.subprocess.run = _run
    impl._vllm_command = cli_compatible._vllm_command
    try:
        return impl.main()
    finally:
        Path.mkdir = old_mkdir
        impl.subprocess.run = old_run
        impl._vllm_command = old_command


if __name__ == "__main__":
    raise SystemExit(main())
