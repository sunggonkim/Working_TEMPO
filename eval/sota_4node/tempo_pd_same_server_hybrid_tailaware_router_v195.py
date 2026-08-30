#!/usr/bin/env python3
"""Rate-saturation router removing only the observed 2048/64 tail bucket."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_same_server_hybrid_saturation_router_v191 as saturation
from tempo.pd_cache_affinity_tailaware import TailAwareCacheAffinityCatalog


class TailAwareHybridCore(saturation.SaturationHybridCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._hybrid._catalog = TailAwareCacheAffinityCatalog()


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = TailAwareHybridCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
