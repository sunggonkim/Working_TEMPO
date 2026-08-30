#!/usr/bin/env python3
"""Phase-change trace with same-length request-unique prefix substitution."""

import json

from transformers import AutoTokenizer

from eval.sota_4node import run_tempo_pd_same_server_phasechange_client_v353 as base
from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_unique_chunks_v308 as unique


_ORIGINAL_ROWS = base._rows
_ORIGINAL_RUN_PHASE = base._run_phase
_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(
            base._argument("--model"), local_files_only=True)
    return _TOKENIZER


def _rows(source, phase):
    rows = _ORIGINAL_ROWS(source, phase)
    tokenizer = _tokenizer()
    phase_index = {"warm": 0, "measured": 1}[phase]
    rewritten = []
    first_chunks = set()
    lengths_by_item = {}
    for row_index, row in enumerate(rows):
        original_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        marker_ids = tokenizer.encode(
            unique._marker((phase_index << 10) | row_index),
            add_special_tokens=False,
        )
        candidate_ids = marker_ids + original_ids[len(marker_ids):]
        prompt = tokenizer.decode(
            candidate_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        checked_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(checked_ids) != len(original_ids):
            raise ValueError("prefix substitution changed prompt token geometry")
        first_chunk = tuple(checked_ids[:256])
        if first_chunk in first_chunks:
            raise ValueError("leading LMCache chunk is not request-unique")
        first_chunks.add(first_chunk)
        item = int(row["request_id"].rsplit("-item-", 1)[1])
        lengths_by_item.setdefault(item, set()).add(len(checked_ids))
        value = dict(row)
        value["prompt"] = prompt
        rewritten.append(value)
    if len(first_chunks) != 48:
        raise ValueError("expected 48 unique leading chunks")
    if any(len(lengths) != 1 for lengths in lengths_by_item.values()):
        raise ValueError("paired prompt token geometry diverged")
    return rewritten


def _run_phase(root, source, phase, workers):
    path = _ORIGINAL_RUN_PHASE(root, source, phase, workers)
    value = json.loads(path.read_text())
    contract = value["mixed_crossover_contract"]
    contract["leading_unique_region"] = (
        "same_length_first_19_token_prefix_substitution_v361"
    )
    contract["leading_unique_chunk_count"] = 48
    contract["paired_prompt_token_geometry_equal"] = True
    contract["frozen_prompt_token_buckets_preserved"] = [512, 1230, 2048, 4094]
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
