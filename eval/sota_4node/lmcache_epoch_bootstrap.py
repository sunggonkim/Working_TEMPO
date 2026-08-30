"""Keep the validated module PyTorch while adding LMCache runtime dependencies."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected_site = (
        repo_root / ".sota_venv" / "lib" / "python3.12" / "site-packages"
    ).resolve()
    raw_site = os.environ.get("TEMPO_LMCACHE_EXTRA_SITE")
    if not raw_site or Path(raw_site).resolve() != expected_site or not expected_site.is_dir():
        raise SystemExit("TEMPO_LMCACHE_EXTRA_SITE must name the repository venv site-packages")

    # Import the validated module build first.  Appending the dependency site
    # afterwards prevents its newer torch wheel from shadowing PyTorch 2.8.
    import torch

    if torch.__version__ != "2.8.0+cu129" or torch.version.cuda != "12.9":
        raise SystemExit(
            f"expected module torch 2.8.0+cu129/CUDA 12.9, got "
            f"{torch.__version__}/{torch.version.cuda}"
        )
    sys.path.append(str(expected_site))

    if os.environ.get("TEMPO_LMCACHE_PREFLIGHT") == "YES":
        import nixl
        from eval.sota_4node import run_lmcache_nixl_contention_2node as official

        NixlChannel, _, _, _ = official._load_official_lmcache(repo_root)
        import lmcache.c_ops as c_ops

        print(
            json.dumps(
                {
                    "preflight": "ok",
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "nixl": importlib.metadata.version("nixl"),
                    "nixl_module": str(Path(nixl.__file__).resolve()),
                    "lmcache_commit": official.LMCACHE_COMMIT,
                    "nixl_channel_module": str(
                        Path(sys.modules[NixlChannel.__module__].__file__).resolve()
                    ),
                    "lmcache_c_ops_shim": type(c_ops).__name__,
                },
                sort_keys=True,
            )
        )
        return

    runpy.run_module(
        "eval.sota_4node.run_lmcache_epoch_2node",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
