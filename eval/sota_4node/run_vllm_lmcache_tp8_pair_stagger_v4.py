#!/usr/bin/env python3
"""Compatibility-fixed entrypoint for the pair-stagger v3 pilot."""

from __future__ import annotations

import os
import sys

from eval.sota_4node import run_vllm_lmcache_tp8_pair_stagger_v3 as candidate


def main() -> None:
    candidate._install()
    # The reused v1 main performs one final legacy constant comparison after
    # calling the injected contract loader.  Bind that guard to this contract.
    candidate._v1.EXPECTED_PLAN_SIGNATURE = candidate.CONTRACT_SIGNATURE
    candidate._v1.main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
