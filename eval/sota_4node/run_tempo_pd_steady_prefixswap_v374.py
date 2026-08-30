#!/usr/bin/env python3
"""Steady paired trace with same-length request-unique first chunks."""

from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_unique_chunks_v305 as underlying
from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_unique_chunks_v308 as punctuation
from eval.sota_4node.tempo_pd_prefixswap_common_v372 import annotate, rewrite_rows


_ORIGINAL_ROWS = underlying._rows
_ORIGINAL_RUN = underlying._run_phase


def _rows(source, phase):
    return rewrite_rows(_ORIGINAL_ROWS(source, phase),
                        underlying._argument("--model"), phase)


def _run_phase(root, source, phase, workers):
    return annotate(_ORIGINAL_RUN(root, source, phase, workers))


def main():
    old = underlying._rows, underlying._run_phase
    underlying._rows, underlying._run_phase = _rows, _run_phase
    try:
        return punctuation.main()
    finally:
        underlying._rows, underlying._run_phase = old


if __name__ == "__main__":
    raise SystemExit(main())
