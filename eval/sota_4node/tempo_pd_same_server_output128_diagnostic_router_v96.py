#!/usr/bin/env python3
"""Balanced same-server router for the output128 local-guard diagnostic."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_same_server_router_v61 as prior
from eval.sota_4node import tempo_pd_same_server_balanced_router_v70 as balanced


class DiagnosticOutput128Policy(prior.FrozenPDPolicy):
    def force_local(self, prompt_tokens: int, output_tokens: int) -> bool:
        if output_tokens == 128:
            if type(prompt_tokens) is not int or prompt_tokens <= 0:
                raise ValueError("prompt_tokens must be a positive int")
            return prompt_tokens <= 512
        return super().force_local(prompt_tokens, output_tokens)

    def high_pair_interval(self, output_tokens: int) -> int:
        if output_tokens == 128:
            return 80_000_000
        return super().high_pair_interval(output_tokens)

    def high_local_credit(self, output_tokens: int) -> int:
        if output_tokens == 128:
            return 10
        return super().high_local_credit(output_tokens)


def main(argv=None) -> int:
    original = prior.FrozenPDPolicy
    prior.FrozenPDPolicy = DiagnosticOutput128Policy
    try:
        return balanced.main(argv)
    finally:
        prior.FrozenPDPolicy = original


if __name__ == "__main__":
    raise SystemExit(main())
