#!/usr/bin/env python3
"""Build a held-out C5 manifest with an explicit production output geometry.

The historical C5 v3 builder is intentionally left immutable: its measured hot
streams use output=2 and are the C1/C2 anchor discovery trace.  This builder
creates a separate held-out artifact with replicate IDs 02/03, a disjoint MISS
marker namespace, deterministic within-phase arrival salt, and output=128 for
all non-foreground pressure streams.  It never changes the source pools or
pretends that phase metadata is an online policy input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from eval.sota_4node.build_tempo_go_c5_manifest import (
    _read_source,
    _unique_miss_prompt,
)
from tempo.pd_global_workload import (
    build_contention_workload,
    canonical_contention_phases,
)


HELDOUT_SCHEMA = "tempo-go-c5-heldout-manifest-v1"
DEFAULT_REPLICATE_START = 2
DEFAULT_MISS_MARKER_BASE = 200_000
STREAM_OFFSET_MS = {
    "foreground": 0.250,
    "decoder-hot": 0.750,
    "remote-hot": 1.250,
    "kv-remote-hot": 1.750,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stream_name(request_id: str) -> str:
    for name in ("kv-remote-hot", "decoder-hot", "remote-hot", "foreground"):
        if f"-{name}-" in request_id:
            return name
    raise ValueError(f"held-out request ID lacks a known stream: {request_id}")


def _transform_rows(
    rows: list[dict[str, object]],
    phases: list[dict[str, object]],
    *,
    tokenizer: Any,
    hot_output_tokens: int,
    replicate_start: int,
    miss_marker_base: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Apply held-out identity, geometry, schedule and MISS transformations."""

    if type(hot_output_tokens) is not int or hot_output_tokens < 2:
        raise ValueError("hot_output_tokens must be an int >= 2")
    if type(replicate_start) is not int or replicate_start < 0:
        raise ValueError("replicate_start must be non-negative")
    if type(miss_marker_base) is not int or miss_marker_base < 0:
        raise ValueError("miss_marker_base must be non-negative")

    transformed = [dict(row) for row in rows]
    miss_count = 0
    miss_first_chunks: set[tuple[int, ...]] = set()
    for row_index, row in enumerate(transformed):
        request_id = str(row["request_id"])
        stream = _stream_name(request_id)
        row["request_id"] = request_id.replace("r00", "r02", 1).replace(
            "r01", "r03", 1)
        if stream != "foreground":
            row["max_tokens"] = hot_output_tokens
        row["arrival_offset_ms"] = round(
            float(row["arrival_offset_ms"]) + STREAM_OFFSET_MS[stream], 6)
        if "-cache-miss-measured-" in request_id:
            prompt = _unique_miss_prompt(
                tokenizer, str(row["prompt"]), miss_marker_base + row_index)
            token_ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
            first_chunk = token_ids[:256]
            if first_chunk in miss_first_chunks:
                raise ValueError("held-out MISS first chunk is not unique")
            miss_first_chunks.add(first_chunk)
            row["prompt"] = prompt
            miss_count += 1

    # Preserve per-phase contiguous row ranges while making the salted stream
    # order explicit.  All salts are below the available phase headroom.
    for phase in phases:
        start = int(phase["row_start"])
        end = int(phase["row_end"])
        transformed[start:end] = sorted(
            transformed[start:end],
            key=lambda value: (
                float(value["arrival_offset_ms"]),
                str(value["request_id"]),
            ),
        )
        phase["replicate"] = int(phase["replicate"]) + replicate_start

    return transformed, {
        "miss_count": miss_count,
        "miss_unique_first_chunk_count": len(miss_first_chunks),
    }


def build_heldout_artifact(
    *,
    source_pools: dict[int, list[str]],
    tokenizer: Any,
    phases,
    hot_output_tokens: int,
    replicate_start: int = DEFAULT_REPLICATE_START,
    miss_marker_base: int = DEFAULT_MISS_MARKER_BASE,
    parent_manifest_sha256: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build rows and a held-out sidecar without touching the filesystem."""

    rows, manifest = build_contention_workload(
        source_pools,
        phases=phases,
        replicates=2,
        # Keep this historical input explicit; all pressure-stream rows are
        # rewritten below to the held-out production geometry.
        anchor_output_tokens=2,
        background_output_tokens=2,
    )
    rows, counts = _transform_rows(
        rows,
        manifest["phases"],
        tokenizer=tokenizer,
        hot_output_tokens=hot_output_tokens,
        replicate_start=replicate_start,
        miss_marker_base=miss_marker_base,
    )
    manifest = dict(manifest)
    manifest["heldout"] = {
        "schema": HELDOUT_SCHEMA,
        "parent_manifest_sha256": parent_manifest_sha256,
        "replicate_ids": [replicate_start, replicate_start + 1],
        "miss_marker_base": miss_marker_base,
        "stream_arrival_offset_ms": dict(STREAM_OFFSET_MS),
        "hot_output_tokens": hot_output_tokens,
        "source_pool_geometry_reused": [512, 2048, 4094],
        "policy_inputs_excluded": [
            "phase_name", "future_arrivals", "physical_switch_label",
            "oracle_route",
        ],
    }
    manifest["anchor_output_tokens"] = 2
    manifest["background_output_tokens"] = hot_output_tokens
    manifest["hot_output_tokens"] = hot_output_tokens
    manifest["execution_contract"] = {
        "client_request_rate_flag": "must_be_omitted",
        "warmup_outside_measurement": True,
        "synthetic_network_background": False,
        "performance_claim_allowed": False,
    }
    manifest["cache_contracts"] = dict(manifest["cache_contracts"])
    manifest["cache_contracts"]["miss_unique_prompt_count"] = counts[
        "miss_unique_first_chunk_count"]
    if counts["miss_count"] != counts["miss_unique_first_chunk_count"]:
        raise ValueError("held-out MISS uniqueness accounting mismatch")
    return rows, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-512", type=Path, required=True)
    parser.add_argument("--source-2048", type=Path, required=True)
    parser.add_argument("--source-4094", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-ms", type=float, default=15_000.0)
    parser.add_argument("--cooldown-ms", type=float, default=2_000.0)
    parser.add_argument("--foreground-rate", type=float, default=2.0)
    parser.add_argument("--decoder-hot-rate", type=float, default=22.4)
    parser.add_argument("--remote-hot-rate", type=float, default=4.76)
    parser.add_argument("--kv-remote-hot-rate", type=float, default=12.0)
    parser.add_argument("--hot-output-tokens", type=int, default=128)
    parser.add_argument(
        "--replicate-start", type=int, default=DEFAULT_REPLICATE_START)
    parser.add_argument(
        "--miss-marker-base", type=int, default=DEFAULT_MISS_MARKER_BASE)
    args = parser.parse_args()

    if not (args.model / "config.json").is_file():
        raise ValueError("local model config.json is missing")
    parent_manifest = args.parent_manifest.resolve()
    if not parent_manifest.is_file():
        raise ValueError("parent manifest is missing")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    workloads_dir = output_dir / "workloads"
    workloads_dir.mkdir()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model.resolve(), local_files_only=True, trust_remote_code=False)
    source_paths = {
        512: args.source_512.resolve(),
        2048: args.source_2048.resolve(),
        4094: args.source_4094.resolve(),
    }
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
    rows, manifest = build_heldout_artifact(
        source_pools=source_pools,
        tokenizer=tokenizer,
        phases=phases,
        hot_output_tokens=args.hot_output_tokens,
        replicate_start=args.replicate_start,
        miss_marker_base=args.miss_marker_base,
        parent_manifest_sha256=_sha256(parent_manifest),
    )
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
    })
    manifest_path = output_dir / "tempo_go_workload_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "workload": str(validation),
        "workload_sha256": _sha256(validation),
        "request_count": len(rows),
        "hot_output_tokens": args.hot_output_tokens,
        "replicate_ids": [args.replicate_start, args.replicate_start + 1],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
