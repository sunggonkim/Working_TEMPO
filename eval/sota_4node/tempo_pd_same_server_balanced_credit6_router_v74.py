#!/usr/bin/env python3
"""Balanced same-server diagnostic with six local credits at 32 tokens."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_same_server_balanced_router_v70 as balanced
from eval.sota_4node import tempo_pd_same_server_router_v61 as prior
from tempo.pd_workload_policy import FrozenPDPolicy


class CreditSixPolicy(FrozenPDPolicy):
    def high_local_credit(self, output_tokens: int) -> int:
        if output_tokens == 32:
            return 6
        return super().high_local_credit(output_tokens)


def main(argv=None) -> int:
    original = prior.FrozenPDPolicy
    prior.FrozenPDPolicy = CreditSixPolicy
    try:
        return balanced.main(argv)
    finally:
        prior.FrozenPDPolicy = original


if __name__ == "__main__":
    raise SystemExit(main())
