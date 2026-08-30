#!/usr/bin/env python3
"""Accept request-level interleaved arm identities for the output16 split."""

from __future__ import annotations

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_same_server_output16_split_router_v99 as split


class InterleavedSplitCore(split.Output16SplitCore):
    @staticmethod
    def _arm(request_id: str) -> tuple[str, str]:
        for arm in ("local", "tempo", "remote"):
            for replicate in (0, 1):
                if request_id.startswith(f"ssi-{arm}-r{replicate}-measured-"):
                    return arm, "measured"
        return split.Output16SplitCore._arm(request_id)


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = InterleavedSplitCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
