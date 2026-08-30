#!/usr/bin/env python3
"""Rate-saturation client isolating TEMPO and fixed-local availability."""

from __future__ import annotations

from eval.sota_4node import run_tempo_pd_same_server_balanced_client_v70 as balanced
from eval.sota_4node import run_tempo_pd_same_server_cache_catalog_client_v136 as catalog
from eval.sota_4node import run_tempo_pd_same_server_hybrid_phase_client_v182 as phase


_WARM = ("fixed_local", "tempo", "tempo")
_MEASURED = (
    "fixed_local", "tempo", "fixed_local",
    "fixed_local", "tempo", "fixed_local",
)


def main() -> int:
    args = balanced._parse()
    if args.run_id.endswith("-warmup"):
        phase._transport_then_cold()
    old_warm = balanced._WARM_ORDER
    old_measured = balanced._MEASURED_ORDER
    balanced._WARM_ORDER = _WARM
    balanced._MEASURED_ORDER = _MEASURED
    try:
        return catalog.main()
    finally:
        balanced._WARM_ORDER = old_warm
        balanced._MEASURED_ORDER = old_measured


if __name__ == "__main__":
    raise SystemExit(main())
