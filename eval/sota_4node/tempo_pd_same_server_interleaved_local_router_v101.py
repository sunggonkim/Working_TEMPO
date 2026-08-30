#!/usr/bin/env python3
"""Request-interleaved output16 all-local candidate."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_same_server_router_v61 as prior
from eval.sota_4node import tempo_pd_same_server_balanced_router_v70 as balanced
from eval.sota_4node.tempo_pd_same_server_interleaved_router_v100 import (
    InterleavedSplitCore,
)
from eval.sota_4node.tempo_pd_same_server_output16_diagnostic_router_v98 import (
    DiagnosticOutput16Policy,
)


class InterleavedLocalCore(balanced.BalancedSameServerCore):
    _arm = staticmethod(InterleavedSplitCore._arm)


def main(argv=None) -> int:
    original_core = credit.CreditCore
    original_policy = prior.FrozenPDPolicy
    credit.CreditCore = InterleavedLocalCore
    prior.FrozenPDPolicy = DiagnosticOutput16Policy
    try:
        return credit.main(argv)
    finally:
        prior.FrozenPDPolicy = original_policy
        credit.CreditCore = original_core


if __name__ == "__main__":
    raise SystemExit(main())
