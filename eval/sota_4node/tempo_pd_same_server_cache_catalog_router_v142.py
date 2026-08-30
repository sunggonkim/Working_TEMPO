#!/usr/bin/env python3
"""Keep 16 remote requests while minimizing warm-hit remote KV bytes."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_same_server_cache_catalog_router_v136 as catalog
from tempo.pd_admission import PDRoute


def _selected_route(prompt_tokens: int, output_tokens: int) -> PDRoute:
    if prompt_tokens not in (512, 1230, 2048):
        raise ValueError("cache-catalog policy is validated only for prompt 512/1230/2048")
    if output_tokens not in (16, 32, 64, 128):
        raise ValueError("cache-catalog policy output length is unvalidated")
    return (PDRoute.REMOTE_PREFILL if prompt_tokens == 512
            else PDRoute.DECODER_LOCAL)


def main(argv=None) -> int:
    original = catalog._selected_route
    catalog._selected_route = _selected_route
    try:
        return catalog.main(argv)
    finally:
        catalog._selected_route = original


if __name__ == "__main__":
    raise SystemExit(main())
