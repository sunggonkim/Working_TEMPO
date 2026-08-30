"""Race-safe entry that routes the v2 node harness to client v4."""

from __future__ import annotations

from pathlib import Path

from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as impl


_ORIGINAL_MKDIR = Path.mkdir
_ORIGINAL_RUN = impl.subprocess.run


def _mkdir(self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False):
    if self.name in {"lmcache_always_remote", "tempo_admission"}:
        exist_ok = True
    return _ORIGINAL_MKDIR(self, mode=mode, parents=parents, exist_ok=exist_ok)


def _run(command, *args, **kwargs):
    if isinstance(command, list):
        command = [
            "eval.sota_4node.live_pd_controller_lmcache_v4"
            if value == "eval.sota_4node.live_pd_controller_lmcache_v3"
            else value
            for value in command
        ]
    return _ORIGINAL_RUN(command, *args, **kwargs)


def main() -> int:
    old_mkdir = Path.mkdir
    old_run = impl.subprocess.run
    Path.mkdir = _mkdir
    impl.subprocess.run = _run
    try:
        return impl.main()
    finally:
        Path.mkdir = old_mkdir
        impl.subprocess.run = old_run


if __name__ == "__main__":
    raise SystemExit(main())
