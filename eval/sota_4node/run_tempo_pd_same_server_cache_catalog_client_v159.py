#!/usr/bin/env python3
"""Warm remote first; measured crossover remains unchanged and balanced."""

from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as balanced
from eval.sota_4node import run_tempo_pd_same_server_cache_catalog_client_v136 as catalog


def main() -> int:
    original = balanced._WARM_ORDER
    balanced._WARM_ORDER = ("lmcache_remote", "fixed_local", "tempo")
    try:
        return catalog.main()
    finally:
        balanced._WARM_ORDER = original


if __name__ == "__main__": raise SystemExit(main())
