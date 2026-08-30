#!/usr/bin/env python3
"""Experimental dispersed affinity profile with equal request count/work."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_same_server_hybrid_controller_router_v150 as production
from tempo import pd_cache_affinity as affinity


REMOTE_BUCKETS = frozenset({(512, 32), (512, 64), (1230, 32), (1230, 64)})


def main(argv=None) -> int:
    original = affinity.REMOTE_BUCKETS
    affinity.REMOTE_BUCKETS = REMOTE_BUCKETS
    try:
        return production.main(argv)
    finally:
        affinity.REMOTE_BUCKETS = original


if __name__ == "__main__": raise SystemExit(main())
