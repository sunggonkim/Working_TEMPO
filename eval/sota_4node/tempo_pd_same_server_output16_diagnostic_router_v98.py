#!/usr/bin/env python3
"""Balanced router for the mixed-prompt/output16 local-guard diagnostic."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_same_server_router_v61 as prior
from eval.sota_4node import tempo_pd_same_server_balanced_router_v70 as balanced


class DiagnosticOutput16Policy(prior.FrozenPDPolicy):
    def force_local(self, prompt_tokens: int, output_tokens: int) -> bool:
        if output_tokens == 16:
            if type(prompt_tokens) is not int or not 0 < prompt_tokens <= 2048:
                raise ValueError("output16 diagnostic requires prompt_tokens <= 2048")
            return True
        return super().force_local(prompt_tokens, output_tokens)

    def high_pair_interval(self, output_tokens: int) -> int:
        return 58_000_000 if output_tokens == 16 else super().high_pair_interval(output_tokens)

    def high_local_credit(self, output_tokens: int) -> int:
        return 8 if output_tokens == 16 else super().high_local_credit(output_tokens)


def main(argv=None) -> int:
    original = prior.FrozenPDPolicy
    prior.FrozenPDPolicy = DiagnosticOutput16Policy
    try:
        return balanced.main(argv)
    finally:
        prior.FrozenPDPolicy = original


if __name__ == "__main__":
    raise SystemExit(main())
