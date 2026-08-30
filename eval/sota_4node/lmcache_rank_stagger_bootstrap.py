"""Launch rank-stagger LMCache with module PyTorch and appended NIXL deps."""

from __future__ import annotations

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
        raise SystemExit("TEMPO_LMCACHE_EXTRA_SITE must name repository venv site-packages")
    import torch

    if torch.__version__ != "2.8.0+cu129" or torch.version.cuda != "12.9":
        raise SystemExit("rank-stagger requires module PyTorch 2.8.0+cu129/CUDA 12.9")
    sys.path.append(str(expected_site))
    runpy.run_module(
        "eval.sota_4node.run_lmcache_rank_stagger_2node",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
