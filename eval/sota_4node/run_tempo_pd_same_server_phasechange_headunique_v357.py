#!/usr/bin/env python3
"""Phase-change trace with a request-unique leading LMCache chunk marker."""

import json

from eval.sota_4node import run_tempo_pd_same_server_phasechange_client_v353 as base
from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_unique_chunks_v308 as unique


_ORIGINAL_ROWS = base._rows
_ORIGINAL_RUN_PHASE = base._run_phase


def _rows(source, phase):
    rows = _ORIGINAL_ROWS(source, phase)
    phase_index = {"warm": 0, "measured": 1}[phase]
    for row_index, row in enumerate(rows):
        # The leading region ensures two simultaneous remote requests never
        # contend for LMCache's first common chunk. All markers have identical
        # token geometry; only A/B identities differ.
        marker_id = (phase_index << 10) | row_index
        row["prompt"] = unique._marker(marker_id) + " " + row["prompt"]
    return rows


def _run_phase(root, source, phase, workers):
    path = _ORIGINAL_RUN_PHASE(root, source, phase, workers)
    value = json.loads(path.read_text())
    value["mixed_crossover_contract"]["leading_unique_region"] = (
        "request_unique_18_bit_marker_plus_punctuation_v357"
    )
    value["mixed_crossover_contract"]["paired_prompt_token_geometry_equal"] = True
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
