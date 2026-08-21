#!/usr/bin/env python3
"""One-factor weighted-local-credit revision of Elastic-PD."""

from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as base
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v446 as isolated


PROFILE = "eval/sota_4node/real_tempo_pd_elastic_profile_v447.json"


def main():
    old = base.PROFILE
    base.PROFILE = PROFILE
    try:
        return isolated.main()
    finally:
        base.PROFILE = old


if __name__ == "__main__":
    raise SystemExit(main())
