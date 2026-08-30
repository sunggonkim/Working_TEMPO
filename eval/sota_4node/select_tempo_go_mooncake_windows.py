#!/usr/bin/env python3
"""Select four deterministic FAST'25 windows without performance feedback."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Sequence


SCHEMA = "tempo-go-mooncake-fast25-window-selection-v1"
WINDOW_SIZE = 64
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT / "eval/sota_4node/data/mooncake_fast25/source_manifest_v1.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "eval/sota_4node/tempo_go_mooncake_windows_v1.json"
)
_CORE_PATH = REPO_ROOT / "tempo/mooncake_fast25_workload.py"
_SPEC = importlib.util.spec_from_file_location(
    "_tempo_mooncake_window_selection_core", _CORE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load Mooncake workload module: {_CORE_PATH}")
_CORE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CORE
_SPEC.loader.exec_module(_CORE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prior_reuse(rows: Sequence[Any]) -> list[int]:
    trie: dict[int, Any] = {}
    values: list[int] = []
    for row in rows:
        complete_blocks = row.input_tokens // _CORE.BLOCK_SIZE_TOKENS
        node = trie
        matched = 0
        for block_id in row.hash_ids[:complete_blocks]:
            child = node.get(block_id)
            if child is None:
                break
            node = child
            matched += 1
        values.append(matched * _CORE.BLOCK_SIZE_TOKENS)
        node = trie
        for block_id in row.hash_ids[:complete_blocks]:
            node = node.setdefault(block_id, {})
    return values


def _nearest_rank(values: Sequence[int], probability: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def _window_metrics(rows: Sequence[Any], start_index: int) -> dict[str, Any]:
    selected = tuple(rows[start_index:start_index + WINDOW_SIZE])
    _require(len(selected) == WINDOW_SIZE, "window is incomplete")
    input_tokens = [row.input_tokens for row in selected]
    output_tokens = [row.output_tokens for row in selected]
    reuse_tokens = _prior_reuse(selected)
    duration_ms = selected[-1].timestamp_ms - selected[0].timestamp_ms
    offered_rate = (
        (WINDOW_SIZE - 1) * 1000.0 / duration_ms
        if duration_ms > 0 else None
    )
    return {
        "start_index": start_index,
        "stop_index_exclusive": start_index + WINDOW_SIZE,
        "request_count": WINDOW_SIZE,
        "first_timestamp_ms": selected[0].timestamp_ms,
        "last_timestamp_ms": selected[-1].timestamp_ms,
        "duration_ms": duration_ms,
        "base_offered_rate_per_s": offered_rate,
        "input_tokens": {
            "mean": statistics.fmean(input_tokens),
            "p50": statistics.median(input_tokens),
            "p90": _nearest_rank(input_tokens, 0.90),
            "max": max(input_tokens),
        },
        "output_tokens": {
            "mean": statistics.fmean(output_tokens),
            "p50": statistics.median(output_tokens),
            "p90": _nearest_rank(output_tokens, 0.90),
            "max": max(output_tokens),
        },
        "reuse": {
            "requests_with_prior_reusable_prefix": sum(
                value > 0 for value in reuse_tokens
            ),
            "request_fraction": sum(value > 0 for value in reuse_tokens)
            / WINDOW_SIZE,
            "mean_prior_reusable_prefix_tokens": statistics.fmean(reuse_tokens),
            "mean_prior_reusable_input_fraction": statistics.fmean(
                reuse / source.input_tokens
                for reuse, source in zip(reuse_tokens, selected)
            ),
        },
    }


def _aligned_windows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        _window_metrics(rows, start)
        for start in range(0, len(rows) - WINDOW_SIZE + 1, WINDOW_SIZE)
    ]


def _select(
    windows: Sequence[dict[str, Any]],
    *,
    eligible: Callable[[dict[str, Any]], bool],
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> dict[str, Any]:
    candidates = [window for window in windows if eligible(window)]
    _require(bool(candidates), "window objective has no eligible candidate")
    # Every objective includes ``-start_index`` as its final key, making the
    # earliest source window the deterministic tie break.
    return max(candidates, key=key)


def select_windows(source_manifest: Path) -> dict[str, Any]:
    traces: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    windows: dict[str, list[dict[str, Any]]] = {}
    for trace_name in ("conversation", "toolagent", "synthetic"):
        rows, receipt = _CORE.load_trace(source_manifest, trace_name)
        traces[trace_name] = (rows, receipt)
        windows[trace_name] = _aligned_windows(rows)

    selected = {
        "conversation_long_decode": {
            "trace": "conversation",
            "objective": (
                "maximize aligned-window median input tokens, then median "
                "output tokens; earliest index breaks ties"
            ),
            "metrics": _select(
                windows["conversation"],
                eligible=lambda _window: True,
                key=lambda window: (
                    window["input_tokens"]["p50"],
                    window["output_tokens"]["p50"],
                    -window["start_index"],
                ),
            ),
        },
        "toolagent_prefix_reuse": {
            "trace": "toolagent",
            "objective": (
                "maximize aligned-window mean prior reusable input fraction, "
                "then native offered rate; earliest index breaks ties"
            ),
            "metrics": _select(
                windows["toolagent"],
                eligible=lambda _window: True,
                key=lambda window: (
                    window["reuse"]["mean_prior_reusable_input_fraction"],
                    window["base_offered_rate_per_s"],
                    -window["start_index"],
                ),
            ),
        },
        "toolagent_native_burst": {
            "trace": "toolagent",
            "objective": (
                "maximize aligned-window native offered rate, then mean prior "
                "reusable input fraction; earliest index breaks ties"
            ),
            "metrics": _select(
                windows["toolagent"],
                eligible=lambda window: window["base_offered_rate_per_s"] is not None,
                key=lambda window: (
                    window["base_offered_rate_per_s"],
                    window["reuse"]["mean_prior_reusable_input_fraction"],
                    -window["start_index"],
                ),
            ),
        },
        "synthetic_zero_reuse_burst": {
            "trace": "synthetic",
            "objective": (
                "among aligned windows with zero prior reusable-prefix requests, "
                "maximize native offered rate, then median input; earliest index "
                "breaks ties"
            ),
            "metrics": _select(
                windows["synthetic"],
                eligible=lambda window: (
                    window["reuse"]["requests_with_prior_reusable_prefix"] == 0
                    and window["base_offered_rate_per_s"] is not None
                ),
                key=lambda window: (
                    window["base_offered_rate_per_s"],
                    window["input_tokens"]["p50"],
                    -window["start_index"],
                ),
            ),
        },
    }
    source_receipts = {
        name: {
            "source_sha256": receipt["source_sha256"],
            "source_git_blob_sha1": receipt["source_git_blob_sha1"],
            "upstream_commit": receipt["upstream_commit"],
            "aligned_candidate_windows": len(windows[name]),
        }
        for name, (_rows, receipt) in traces.items()
    }
    return {
        "schema": SCHEMA,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": _sha256(source_manifest),
        "window_contract": {
            "window_size": WINDOW_SIZE,
            "alignment": "non-overlapping source-index multiples of 64",
            "selection_uses_inference_performance": False,
            "selection_uses_policy_output": False,
            "prefix_reuse_semantics": (
                "complete leading 512-token hash blocks shared with a prior "
                "request in the same window; no eviction or placement assumed"
            ),
        },
        "source_receipts": source_receipts,
        "selected": selected,
        "capacity_normalization": {
            "load_multipliers_frozen": False,
            "required_next_step": (
                "measure per-scale fixed-carrier capacity before freezing normal, "
                "knee, overload, and microburst multipliers"
            ),
        },
        "performance_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite: {args.output}")
    try:
        value = select_windows(args.source_manifest)
    except (OSError, ValueError, _CORE.TraceContractError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "selected": {
            name: entry["metrics"]["start_index"]
            for name, entry in value["selected"].items()
        },
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
