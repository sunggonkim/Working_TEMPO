#!/usr/bin/env python3
"""Bursty paired trace with same-length request-unique first chunks."""

from eval.sota_4node import run_tempo_pd_same_server_bursty_client_v322 as base
from eval.sota_4node.tempo_pd_prefixswap_common_v372 import annotate, rewrite_rows


_ORIGINAL_ROWS = base._rows
_ORIGINAL_RUN = base._run_phase


def _rows(source, phase):
    return rewrite_rows(_ORIGINAL_ROWS(source, phase),
                        base._argument("--model"), phase)


def _run_phase(root, source, phase, workers):
    return annotate(_ORIGINAL_RUN(root, source, phase, workers))


def main():
    old = base._rows, base._run_phase
    base._rows, base._run_phase = _rows, _run_phase
    try:
        return base.main()
    finally:
        base._rows, base._run_phase = old


if __name__ == "__main__":
    raise SystemExit(main())
