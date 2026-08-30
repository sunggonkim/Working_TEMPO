#!/usr/bin/env python3
"""Corrected low client with context-preserving two-token cold-key words."""

from __future__ import annotations

from eval.sota_4node import run_tempo_pd_same_server_balanced_low_client_v81 as prior


_CONTEXT_PRESERVING_WORDS = {
    100: "Observed", 200: "Evaluated", 300: "Assessed",
    400: "Tracked", 500: "Bounded", 600: "Calibrated",
    700: "Audited", 800: "Monitored", 900: "Optimized",
}


def main() -> int:
    prior._WORD = _CONTEXT_PRESERVING_WORDS
    return prior.main()


if __name__ == "__main__":
    raise SystemExit(main())
