"""Same-token-count, request-unique leading-chunk transformation."""

from transformers import AutoTokenizer


_TOKENIZERS = {}


def rewrite_rows(rows, model_path, phase):
    tokenizer = _TOKENIZERS.get(str(model_path))
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        _TOKENIZERS[str(model_path)] = tokenizer
    phase_index = {"warm": 0, "measured": 1}[phase]
    from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_unique_chunks_v308 as unique
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
            candidate_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        checked = tokenizer.encode(prompt, add_special_tokens=False)
        if len(checked) != len(original_ids):
            raise ValueError("prefix swap changed prompt length")
        chunk = tuple(checked[:256])
        if chunk in first_chunks:
            raise ValueError("duplicate first LMCache chunk")
        first_chunks.add(chunk)
        item = int(row["request_id"].rsplit("-item-", 1)[1])
        lengths_by_item.setdefault(item, set()).add(len(checked))
        value = dict(row)
        value["prompt"] = prompt
        rewritten.append(value)
    if len(first_chunks) != 48:
        raise ValueError("expected 48 unique first chunks")
    if any(len(lengths) != 1 for lengths in lengths_by_item.values()):
        raise ValueError("paired token geometry differs")
    return rewritten


def annotate(path):
    import json
    value = json.loads(path.read_text())
    contract = value["mixed_crossover_contract"]
    contract["leading_unique_region"] = (
        "same_length_first_19_token_prefix_substitution_v372"
    )
    contract["leading_unique_chunk_count"] = 48
    contract["paired_prompt_token_geometry_equal"] = True
    contract["frozen_prompt_token_buckets_preserved"] = [512, 1230, 2048, 4094]
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path
