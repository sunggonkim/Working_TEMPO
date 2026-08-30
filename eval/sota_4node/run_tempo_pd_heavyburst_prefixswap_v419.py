#!/usr/bin/env python3
"""Heavier paired burst trace with request-unique, same-length prefixes."""

import json

from eval.sota_4node import run_tempo_pd_same_server_bursty_client_v322 as base
from eval.sota_4node.tempo_pd_prefixswap_common_v372 import annotate, rewrite_rows


PAIR_GAP_MS = 8.0
INTER_BURST_GAP_MS = 100.0
_ORIGINAL_ROWS = base._rows
_ORIGINAL_RUN = base._run_phase


def _rows(source, phase):
    return rewrite_rows(_ORIGINAL_ROWS(source, phase),
                        base._argument("--model"), phase)


def _run_phase(root, source, phase, workers):
    path = annotate(_ORIGINAL_RUN(root, source, phase, workers))
    value = json.loads(path.read_text())
    contract = value["mixed_crossover_contract"]
    contract["arrival_trace"] = "six_bursts_four_pairs_8ms_with_100ms_idle_v419"
    contract["pair_gap_ms"] = PAIR_GAP_MS
    contract["inter_burst_gap_ms"] = INTER_BURST_GAP_MS
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def main():
    old = (base.PAIR_GAP_MS, base.INTER_BURST_GAP_MS,
           base.BURST_STRIDE_MS, base._rows, base._run_phase)
    base.PAIR_GAP_MS = PAIR_GAP_MS
    base.INTER_BURST_GAP_MS = INTER_BURST_GAP_MS
    base.BURST_STRIDE_MS = ((base.PAIRS_PER_BURST - 1) * PAIR_GAP_MS
                            + INTER_BURST_GAP_MS)
    base._rows, base._run_phase = _rows, _run_phase
    try:
        return base.main()
    finally:
        (base.PAIR_GAP_MS, base.INTER_BURST_GAP_MS,
         base.BURST_STRIDE_MS, base._rows, base._run_phase) = old


if __name__ == "__main__":
    raise SystemExit(main())
