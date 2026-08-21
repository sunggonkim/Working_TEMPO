#!/usr/bin/env python3
"""Cache-isolated correction of the four-arm Elastic-PD client."""

from __future__ import annotations

from pathlib import Path
import sys

from eval.sota_4node import run_tempo_pd_elastic_balanced_client_v445 as prior
from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_unique_chunks_v308 as unique


_ORIGINAL_DERIVE = prior._derive
_TOKENIZER = None


def _derive(rows, *, arm, replicate, phase, offset):
    derived = _ORIGINAL_DERIVE(
        rows, arm=arm, replicate=replicate, phase=phase, offset=offset)
    if _TOKENIZER is None:
        raise RuntimeError("cache-isolation tokenizer is not initialized")
    phase_index = {"warm": 0, "measured": 1}[phase]
    arm_index = prior._ARMS.index(arm)
    first_chunks = set()
    rewritten = []
    for item, row in enumerate(derived):
        original_ids = _TOKENIZER.encode(row["prompt"], add_special_tokens=False)
        marker_id = phase_index * 100_000 + arm_index * 10_000 + replicate * 1_000 + item
        marker_ids = _TOKENIZER.encode(unique._marker(marker_id), add_special_tokens=False)
        candidate_ids = marker_ids + original_ids[len(marker_ids):]
        prompt = _TOKENIZER.decode(
            candidate_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        checked = _TOKENIZER.encode(prompt, add_special_tokens=False)
        if len(checked) != len(original_ids):
            raise ValueError("cache-isolation prefix changed prompt length")
        chunk = tuple(checked[:256])
        if chunk in first_chunks:
            raise ValueError("duplicate first LMCache chunk within block")
        first_chunks.add(chunk)
        value = dict(row)
        value["prompt"] = prompt
        rewritten.append(value)
    if len(first_chunks) != len(derived):
        raise ValueError("first-chunk uniqueness count mismatch")
    return rewritten


def main() -> int:
    global _TOKENIZER
    from transformers import AutoTokenizer
    model = Path(sys.argv[sys.argv.index("--model") + 1]).resolve()
    _TOKENIZER = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    old_derive = prior._derive
    prior._derive = _derive
    max_index = sys.argv.index("--max-workers") + 1
    old_workers = sys.argv[max_index]
    if sys.argv[sys.argv.index("--run-id") + 1].endswith("-warmup"):
        sys.argv[max_index] = "1"
    try:
        return prior.main()
    finally:
        prior._derive = old_derive
        sys.argv[max_index] = old_workers


if __name__ == "__main__":
    raise SystemExit(main())
