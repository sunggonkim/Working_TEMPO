#!/usr/bin/env python3
"""Canonical stream client for an already-preseeded P-only cache pool.

Unlike ``run_tempo_pd_elastic_stream_metrics``, this module never issues an
implicit seed request.  It is intentionally fail-closed behind an environment
marker so a warm request cannot silently include cold prefill work inside a
measurement window.
"""

from __future__ import annotations

import os

from eval.sota_4node import run_tempo_pd_elastic_stream_metrics_v445 as _prior


ROUTER_SCHEMA = "tempo-elastic-pd-router-canonical"
PRESEEDED_ENV = "TEMPO_PD_P_ONLY_PRESEEDED"


def main() -> int:
    if os.environ.get(PRESEEDED_ENV) != "1":
        raise RuntimeError(
            f"{PRESEEDED_ENV}=1 is required for the preseeded client")
    old_schema = _prior.ROUTER_SCHEMA
    _prior.ROUTER_SCHEMA = ROUTER_SCHEMA
    try:
        return _prior.main()
    finally:
        _prior.ROUTER_SCHEMA = old_schema


if __name__ == "__main__":
    raise SystemExit(main())
