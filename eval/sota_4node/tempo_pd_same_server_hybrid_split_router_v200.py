#!/usr/bin/env python3
"""Rate-saturation router with a stable split of the 2048/64 bucket."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_same_server_hybrid_saturation_router_v191 as saturation
from tempo.pd_cache_affinity_split import SplitCacheAffinityCatalog


class SplitHybridCore(saturation.SaturationHybridCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._hybrid._catalog = SplitCacheAffinityCatalog()


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = SplitHybridCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
