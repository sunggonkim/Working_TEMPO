#!/usr/bin/env python3
"""Corrected launch entry for M20's candidate dispatch."""
from __future__ import annotations
from typing import Any
from eval.sota_4node import run_vllm_lmcache_tp16_predecode_phase_m20_entry as m20

def _run_block(*args: Any, mode: str, **kwargs: Any) -> dict[str, Any]:
    if mode == m20.CANDIDATE_MODE:
        return m20._run_candidate(*args, **kwargs)
    return m20.c9._run_block(*args, mode=mode, **kwargs)

def main() -> None:
    m20._run_block = _run_block
    m20.main()

if __name__ == "__main__":
    main()
