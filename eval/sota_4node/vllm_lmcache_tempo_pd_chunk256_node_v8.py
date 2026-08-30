#!/usr/bin/env python3
"""Full TEMPO-PD calibration/validation using the stronger chunk256 baseline."""

from __future__ import annotations

from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as legacy
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as base
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v4 as validated
from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256


def main() -> int:
    base._config_text = chunk256._config_text
    legacy._proxy_command = chunk256._proxy_command
    return validated.main()


if __name__ == "__main__":
    raise SystemExit(main())
