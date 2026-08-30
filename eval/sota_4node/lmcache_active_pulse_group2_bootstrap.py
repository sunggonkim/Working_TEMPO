"""Launch the coalesced group-two screen with module PyTorch."""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    site = (repo_root / ".sota_venv/lib/python3.12/site-packages").resolve()
    if Path(os.environ.get("TEMPO_LMCACHE_EXTRA_SITE", "")).resolve() != site or not site.is_dir():
        raise SystemExit("invalid TEMPO_LMCACHE_EXTRA_SITE")
    import torch

    if torch.__version__ != "2.8.0+cu129" or torch.version.cuda != "12.9":
        raise SystemExit("requires module PyTorch 2.8.0+cu129/CUDA 12.9")
    sys.path.append(str(site))
    runpy.run_module(
        "eval.sota_4node.run_lmcache_active_pulse_group2_2node",
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
