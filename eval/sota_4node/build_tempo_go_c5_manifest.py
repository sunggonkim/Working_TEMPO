#!/usr/bin/env python3
"""Build the explicit-arrival C1/C2/C3 TEMPO-GO contention workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.sota_4node import (
    run_tempo_pd_same_server_mixed_only_client_unique_chunks_v308 as unique,
)
from tempo.pd_global_workload import (
    build_contention_workload,
    canonical_contention_phases,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_source(path: Path, *, expected_tokens: int, tokenizer) -> list[str]:
    if not path.is_file():
        raise ValueError(f"source workload is missing: {path}")
    prompts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        prompt = value.get("prompt") if isinstance(value, dict) else None
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"source line {line_number} has no prompt")
        observed = len(tokenizer.encode(prompt, add_special_tokens=False))
        if observed != expected_tokens:
            raise ValueError(
                f"{path}:{line_number} has {observed} tokens, expected {expected_tokens}")
        prompts.append(prompt)
    if not prompts:
        raise ValueError(f"source workload is empty: {path}")
    return prompts


def _unique_miss_prompt(tokenizer, prompt: str, marker_id: int) -> str:
    """Make one source-template MISS row cache-unique without changing geometry.

    The source pool remains the semantic/template source.  Only the proven
    token-preserving marker used by the earlier C1/C2 contention client is
    placed in the leading tokens, so repeated source rows cannot silently
    become warm MISS rows while prompt length and model remain frozen.
    """
    original = tuple(tokenizer.encode(prompt, add_special_tokens=False))
    marker_ids = tuple(tokenizer.encode(
        unique._marker(marker_id), add_special_tokens=False))
    if not marker_ids or len(marker_ids) >= len(original):
        raise ValueError("MISS uniqueness marker does not fit source prompt")
    candidate = list(marker_ids) + list(original[len(marker_ids):])
    rewritten = tokenizer.decode(
        candidate, skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    checked = tuple(tokenizer.encode(rewritten, add_special_tokens=False))
    if checked != tuple(candidate):
        raise ValueError("token-preserving MISS marker changed token geometry")
    return rewritten


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-512", type=Path, required=True)
    parser.add_argument("--source-2048", type=Path, required=True)
    parser.add_argument("--source-4094", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--duration-ms", type=float, default=15_000.0)
    parser.add_argument("--cooldown-ms", type=float, default=2_000.0)
    parser.add_argument("--foreground-rate", type=float, default=2.0)
    parser.add_argument("--decoder-hot-rate", type=float, default=22.4)
    parser.add_argument("--remote-hot-rate", type=float, default=4.76)
    parser.add_argument("--kv-remote-hot-rate", type=float, default=12.0)
    parser.add_argument("--anchor-output-tokens", type=int, default=2)
    parser.add_argument("--background-output-tokens", type=int, default=128)
    args = parser.parse_args()
    if not (args.model / "config.json").is_file():
        raise ValueError("local model config.json is missing")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads_dir = output_dir / "workloads"
    workloads_dir.mkdir()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model.resolve(), local_files_only=True, trust_remote_code=False)
    source_paths = {512: args.source_512.resolve(), 2048: args.source_2048.resolve(),
                    4094: args.source_4094.resolve()}
    source_pools = {
        tokens: _read_source(path, expected_tokens=tokens, tokenizer=tokenizer)
        for tokens, path in source_paths.items()
    }
    phases = canonical_contention_phases(
        duration_ms=args.duration_ms,
        foreground_rate_per_s=args.foreground_rate,
        decoder_hot_rate_per_s=args.decoder_hot_rate,
        remote_hot_rate_per_s=args.remote_hot_rate,
        kv_remote_hot_rate_per_s=args.kv_remote_hot_rate,
        cooldown_ms=args.cooldown_ms,
    )
    rows, manifest = build_contention_workload(
        source_pools,
        phases=phases,
        replicates=args.replicates,
        anchor_output_tokens=args.anchor_output_tokens,
        background_output_tokens=args.background_output_tokens,
    )
    miss_rows = 0
    miss_first_chunks: set[tuple[int, ...]] = set()
    for row_index, row in enumerate(rows):
        if "-cache-miss-measured-" not in str(row["request_id"]):
            continue
        if row_index >= (1 << 18) - 100_000:
            raise ValueError("C5 MISS marker space exhausted")
        prompt = _unique_miss_prompt(
            tokenizer, str(row["prompt"]), 100_000 + row_index)
        token_ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
        first_chunk = token_ids[:256]
        if first_chunk in miss_first_chunks:
            raise ValueError("C5 MISS first LMCache chunk is not unique")
        miss_first_chunks.add(first_chunk)
        row["prompt"] = prompt
        miss_rows += 1
    if miss_rows != len(miss_first_chunks):
        raise ValueError("C5 MISS uniqueness accounting mismatch")
    manifest["cache_contracts"].update({
        "miss_prompt_namespace":
            "token_preserving_unique_first_chunk_v1",
        "miss_unique_prompt_count": miss_rows,
    })
    validation = workloads_dir / "validation.jsonl"
    validation.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows),
        encoding="utf-8",
    )
    for phase in manifest["phases"]:
        start = int(phase["row_start"])
        end = int(phase["row_end"])
        phase_path = workloads_dir / (
            f"{phase['name']}_r{int(phase['replicate']):02d}.jsonl")
        phase_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in rows[start:end]),
            encoding="utf-8",
        )
    manifest.update({
        "transport": "LMCacheConnectorV1:UCX",
        "native_only": True,
        "source_pools": {
            str(tokens): {
                "path": str(path),
                "sha256": _sha256(path),
                "count": len(source_pools[tokens]),
            }
            for tokens, path in sorted(source_paths.items())
        },
        "model_config_sha256": _sha256(args.model.resolve() / "config.json"),
        "validation_workload": {
            "path": str(validation),
            "sha256": _sha256(validation),
            "request_count": len(rows),
        },
        "tenant_contract": {
            "latency": {"weight": 4.0, "ttft_slo_ms": 1000.0,
                        "tpot_slo_ms": 100.0, "e2e_slo_ms": 4000.0,
                        "maximum_queue_wait_ns": 500_000_000,
                        "minimum_service_fraction": 0.15},
            "interactive": {"weight": 2.0, "ttft_slo_ms": 2000.0,
                             "tpot_slo_ms": 150.0, "e2e_slo_ms": 8000.0,
                             "maximum_queue_wait_ns": 1_000_000_000,
                             "minimum_service_fraction": 0.15},
            "batch": {"weight": 1.0, "ttft_slo_ms": 3000.0,
                      "tpot_slo_ms": 250.0, "e2e_slo_ms": 16000.0,
                      "maximum_queue_wait_ns": 2_000_000_000,
                      "minimum_service_fraction": 0.10},
            "background": {"weight": 0.5, "ttft_slo_ms": 5000.0,
                            "tpot_slo_ms": 400.0, "e2e_slo_ms": 30000.0,
                            "maximum_queue_wait_ns": 5_000_000_000,
                            "minimum_service_fraction": 0.05},
        },
        "execution_contract": {
            "client_request_rate_flag": "must_be_omitted",
            "max_workers": 128,
            "warmup_outside_measurement": True,
            "synthetic_network_background": False,
            "performance_claim_allowed": False,
        },
    })
    manifest_path = output_dir / "tempo_go_workload_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
