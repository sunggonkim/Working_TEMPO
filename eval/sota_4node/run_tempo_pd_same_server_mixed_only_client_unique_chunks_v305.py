#!/usr/bin/env python3
"""Mixed crossover whose every LMCache-sized prompt region is request-unique."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_v265 as base


PHRASE = (
    "Measured admission must preserve output correctness, decode latency, and "
    "the exact live KV routing contract."
)


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def _marker(marker_id: int) -> str:
    if not 0 <= marker_id < (1 << 18):
        raise ValueError("marker id outside frozen 18-token encoding")
    return " ".join("B" if marker_id & (1 << bit) else "A"
                    for bit in range(17, -1, -1))


def _rows(source: Path, phase: str) -> list[dict]:
    rows = base._rows(source, phase)
    phase_index = {"warm": 0, "measured": 1}[phase]
    rewritten = []
    for row_index, row in enumerate(rows):
        occurrence = 0

        def replacement() -> str:
            nonlocal occurrence
            if occurrence >= 256:
                raise ValueError("more than 256 repeated regions in one prompt")
            marker_id = ((phase_index * 48 + row_index) << 8) | occurrence
            occurrence += 1
            return _marker(marker_id)

        prompt = row["prompt"]
        parts = prompt.split(PHRASE)
        if len(parts) <= 1:
            raise ValueError("frozen repeated phrase missing")
        built = parts[0]
        for suffix in parts[1:]:
            built += replacement() + suffix
        value = dict(row)
        value["prompt"] = built
        value["unique_chunk_marker_count"] = occurrence
        rewritten.append(value)
    return rewritten


def _write(path: Path, rows: list[dict]) -> None:
    # The metrics parser rejects auxiliary workload keys.
    public = []
    for row in rows:
        value = dict(row)
        value.pop("unique_chunk_marker_count")
        public.append(value)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                            for row in public))


def _run_phase(root: Path, source: Path, phase: str, workers: str) -> Path:
    rows = _rows(source, phase)
    workload = root / f"{phase}.jsonl"
    raw = root / f"{phase}.raw.json"
    _write(workload, rows)
    command = [
        sys.executable, "-m",
        "eval.sota_4node.run_tempo_pd_stream_metrics_forced_drain_salted_v296",
        "--base-url", _argument("--base-url"),
        "--model", _argument("--model"),
        "--served-model-name", _argument("--served-model-name"),
        "--workload", str(workload), "--output", str(raw),
        "--mode", "tempo_auto", "--run-id", f"mixed-only-unique-{phase}",
        "--default-max-tokens", "32", "--max-workers", workers,
        "--request-rate", _argument("--request-rate"),
        "--timeout-s", "120", "--seed", "20260815",
    ]
    if "--api-key-env" in sys.argv:
        command.extend(("--api-key-env", _argument("--api-key-env")))
    subprocess.run(command, check=True, timeout=300.0)
    value = json.loads(raw.read_text())
    value["mixed_crossover_contract"] = {
        "schema": "tempo-pd-mixed-request-crossover-260",
        "phase": phase,
        "base_items": 24,
        "requests": 48,
        "tempo_requests": 24,
        "lmcache_remote_requests": 24,
        "same_client_window": True,
        "paired_by_geometry_and_base_item": True,
        "nonce_offsets": [930, 960],
        "variant_assignment_counterbalanced_by_item_parity": True,
        "warm_max_workers": 1,
        "measured_max_workers": int(_argument("--max-workers")),
        "cache_isolation": "vllm_cache_salt_plus_unique_18_token_regions_v305",
        "unique_region_markers": sum(row["unique_chunk_marker_count"] for row in rows),
        "marker_tokens": 18,
    }
    raw.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return raw


def main() -> int:
    if _argument("--run-id").endswith("-warmup"):
        return base.serial.main()
    output = Path(_argument("--output")).resolve()
    source = Path(_argument("--workload")).resolve()
    root = output.parent / "mixed_request_crossover_unique_chunks_v305"
    root.mkdir()
    _run_phase(root, source, "warm", "1")
    measured = _run_phase(root, source, "measured", _argument("--max-workers"))
    output.write_text(measured.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
