#!/usr/bin/env python3
"""Node entrypoint for the TP16 single-flight hybrid boost campaign."""

from __future__ import annotations

import os
from pathlib import Path

from eval.sota_4node import vllm_lmcache_tp16_quiescence_scout_node_v1 as base


base.RUNNER_MODULE = "eval.sota_4node.run_vllm_lmcache_tp16_hybrid_boost_v5"
base.PLAN_RELATIVE = Path("eval/sota_4node/real_tp16_hybrid_boost_v5.json")
base.PINNED_SITE_RELATIVE = Path(
    "eval/sota_4node/vllm_quiescence_sitecustomize_v5_hybrid"
)


def _configure_nixl_runtime(repo_root: Path) -> None:
    backend = os.environ.get("TEMPO_NIXL_BACKEND", "")
    if backend not in {"UCX", "LIBFABRIC"}:
        raise RuntimeError("TEMPO_NIXL_BACKEND must be UCX or LIBFABRIC")
    if backend != "LIBFABRIC":
        os.environ.pop("NIXL_PLUGIN_DIR", None)
        return
    site = repo_root / ".vllm_venv/lib/python3.12/site-packages"
    nixl_lib = site / ".nixl_cu12.mesonpy.libs"
    plugin_dir = nixl_lib / "plugins"
    plugin = plugin_dir / "libplugin_LIBFABRIC.so"
    cuda_lib = site / "nvidia/cuda_runtime/lib"
    fabric_lib = Path("/opt/cray/libfabric/1.22.0/lib64")
    for required in (
        plugin,
        cuda_lib / "libcudart.so.12",
        fabric_lib / "libfabric.so.1",
    ):
        if not required.is_file():
            raise RuntimeError(f"required LIBFABRIC runtime is missing: {required}")
    os.environ["NIXL_PLUGIN_DIR"] = str(plugin_dir)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    prefixes = [str(nixl_lib), str(cuda_lib), str(fabric_lib)]
    os.environ["LD_LIBRARY_PATH"] = ":".join(prefixes + ([existing] if existing else []))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    _configure_nixl_runtime(repo_root)
    base.main()


if __name__ == "__main__":
    main()
