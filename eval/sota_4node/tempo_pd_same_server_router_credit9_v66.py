#!/usr/bin/env python3
"""Same-server diagnostic with nine high-regime local credits at 32 tokens."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_same_server_router_v61 as v61
from tempo.pd_workload_policy import FrozenPDPolicy


class CreditNinePolicy(FrozenPDPolicy):
    def high_local_credit(self, output_tokens: int) -> int:
        if output_tokens == 32:
            return 9
        return super().high_local_credit(output_tokens)


def main(argv=None) -> int:
    original = v61.FrozenPDPolicy
    v61.FrozenPDPolicy = CreditNinePolicy
    try:
        return v61.main(argv)
    finally:
        v61.FrozenPDPolicy = original


if __name__ == "__main__":
    raise SystemExit(main())
