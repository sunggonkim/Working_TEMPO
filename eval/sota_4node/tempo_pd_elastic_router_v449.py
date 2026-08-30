#!/usr/bin/env python3
"""Launch wrapper for first-response credit release with frozen wire schema."""

from eval.sota_4node import tempo_pd_elastic_router_v444 as wire
from eval.sota_4node import tempo_pd_elastic_router_v448 as runtime


# The metrics client intentionally consumes the frozen v444 public header.
# Runtime provenance remains available from v448 health/decision payloads.
runtime._headers = wire._headers

ElasticPDRouterCore = runtime.ElasticPDRouterCore
build_app = runtime.build_app
main = runtime.main
ROUTER_SCHEMA = runtime.ROUTER_SCHEMA


if __name__ == "__main__":
    raise SystemExit(main())
