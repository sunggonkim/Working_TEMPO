#!/usr/bin/env python3
"""Paired trace that changes from microbursts to sparse steady traffic."""

import json

from eval.sota_4node import run_tempo_pd_same_server_phasechange_client_v353 as base


_ORIGINAL_ROWS = base._rows
_ORIGINAL_RUN_PHASE = base._run_phase
BURST_PAIRS = 4
BURSTS = 4
BURST_PAIR_GAP_MS = 14.0
INTER_BURST_GAP_MS = 220.0
BURST_STRIDE_MS = ((BURST_PAIRS - 1) * BURST_PAIR_GAP_MS
                   + INTER_BURST_GAP_MS)
STEADY_PAIRS = 8
STEADY_PAIR_GAP_MS = 100.0
INTER_PHASE_GAP_MS = 220.0
LAST_BURST_ITEM_MS = ((BURSTS - 1) * BURST_STRIDE_MS
                      + (BURST_PAIRS - 1) * BURST_PAIR_GAP_MS)
STEADY_ORIGIN_MS = LAST_BURST_ITEM_MS + INTER_PHASE_GAP_MS


def _rows(source, phase):
    rows = _ORIGINAL_ROWS(source, phase)
    for row in rows:
        item = int(row["request_id"].rsplit("-item-", 1)[1])
        if item < BURSTS * BURST_PAIRS:
            burst, slot = divmod(item, BURST_PAIRS)
            arrival_ms = burst * BURST_STRIDE_MS + slot * BURST_PAIR_GAP_MS
        else:
            arrival_ms = (STEADY_ORIGIN_MS
                          + (item - BURSTS * BURST_PAIRS) * STEADY_PAIR_GAP_MS)
        row["arrival_offset_ms"] = arrival_ms
    return rows


def _run_phase(root, source, phase, workers):
    path = _ORIGINAL_RUN_PHASE(root, source, phase, workers)
    value = json.loads(path.read_text())
    contract = value["mixed_crossover_contract"]
    contract.update({
        "arrival_trace": "four_bursts4_14ms_idle220_then_steady8_100ms_v394",
        "burst_pairs": BURST_PAIRS,
        "bursts": BURSTS,
        "burst_pair_gap_ms": BURST_PAIR_GAP_MS,
        "inter_burst_gap_ms": INTER_BURST_GAP_MS,
        "inter_phase_gap_ms": INTER_PHASE_GAP_MS,
        "steady_pairs": STEADY_PAIRS,
        "steady_pair_gap_ms": STEADY_PAIR_GAP_MS,
    })
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def main():
    old_rows, old_run = base._rows, base._run_phase
    base._rows, base._run_phase = _rows, _run_phase
    try:
        return base.main()
    finally:
        base._rows, base._run_phase = old_rows, old_run


if __name__ == "__main__":
    raise SystemExit(main())
