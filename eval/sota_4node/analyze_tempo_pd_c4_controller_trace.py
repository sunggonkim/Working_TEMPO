#!/usr/bin/env python3
"""Analyze pair-local C4 endpoint-controller dynamics without cross-node clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SCHEMA = "tempo-pd-c4-controller-trace-analysis-v1"
LOCAL = "decoder_local_chunked_prefill"
REMOTE = "official_lmcache_remote_prefill"
_PHASE = re.compile(r"-measured-(?P<phase>[^-]+)-foreground-")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object, *, name: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value)),
        f"{name} must be finite",
    )
    return float(value)


def _nearest_rank(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    copied = [float(value) for value in values]
    return {
        "count": len(copied),
        "median": _nearest_rank(copied, 0.5),
        "p90": _nearest_rank(copied, 0.9),
        "p99": _nearest_rank(copied, 0.99),
        "maximum": max(copied) if copied else None,
    }


def _phase(request_id: object) -> str | None:
    if not isinstance(request_id, str):
        return None
    match = _PHASE.search(request_id)
    return match.group("phase") if match is not None else None


def _foreground_rows(raw: dict[str, object]) -> list[dict[str, object]]:
    decisions = raw.get("router_decisions")
    _require(isinstance(decisions, list), "router_decisions must be a list")
    rows = []
    for decision in decisions:
        _require(isinstance(decision, dict), "router decision must be an object")
        phase = _phase(decision.get("request_id"))
        if decision.get("arm") != "tempo" or phase is None:
            continue
        route = decision.get("endpoint_decision_route")
        _require(route in {LOCAL, REMOTE}, "TEMPO foreground route differs")
        pair = decision.get("frontend_pair_index")
        _require(type(pair) is int and pair in {0, 1}, "pair index differs")
        decided_ns = decision.get("endpoint_decision_decided_ns")
        _require(type(decided_ns) is int and decided_ns >= 0,
                 "decision timestamp differs")
        local_prior = _finite(
            decision.get("endpoint_request_local_e2e_prior_ms"),
            name="local E2E prior",
        )
        remote_prior = _finite(
            decision.get("endpoint_request_remote_e2e_prior_ms"),
            name="remote E2E prior",
        )
        uncertainty = _finite(
            decision.get("endpoint_request_uncertainty_ms"),
            name="uncertainty",
        )
        rows.append({
            "request_id": decision["request_id"],
            "phase": phase,
            "pair": pair,
            "decided_ns": decided_ns,
            "route": route,
            "prompt_tokens": decision.get("prompt_tokens"),
            "output_tokens": decision.get("output_tokens"),
            "local_multiplier": _finite(
                decision.get("endpoint_decision_local_multiplier"),
                name="local multiplier",
            ),
            "remote_multiplier": _finite(
                decision.get("endpoint_decision_remote_multiplier"),
                name="remote multiplier",
            ),
            "local_score_ms": _finite(
                decision.get("endpoint_decision_local_score_ms"),
                name="local score",
            ),
            "remote_score_ms": _finite(
                decision.get("endpoint_decision_remote_score_ms"),
                name="remote score",
            ),
            "static_winner": (
                LOCAL if local_prior + uncertainty <= remote_prior + uncertainty
                else REMOTE
            ),
            "service_stretch": (
                _finite(
                    decision.get("endpoint_feedback_service_stretch"),
                    name="service stretch",
                )
                if decision.get("endpoint_feedback_service_stretch") is not None
                else None
            ),
            "observed_ttft_ms": (
                _finite(
                    decision.get("endpoint_feedback_observed_ttft_ms"),
                    name="observed TTFT",
                )
                if decision.get("endpoint_feedback_observed_ttft_ms") is not None
                else None
            ),
            "probe": decision.get("endpoint_decision_probe") is True,
            "attempts": decision.get("endpoint_decision_attempts"),
        })
    _require(len(rows) == 180, "each TEMPO block must have 180 foreground rows")
    _require(len({row["request_id"] for row in rows}) == len(rows),
             "foreground request IDs must be unique")
    return rows


def _group_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: int(row["decided_ns"]))
    routes = [str(row["route"]) for row in ordered]
    route_counts = Counter(routes)
    transitions = sum(
        left != right for left, right in zip(routes, routes[1:])
    )
    static_overrides = sum(
        row["route"] != row["static_winner"] for row in ordered
    )
    stretches_by_route: dict[str, list[float]] = defaultdict(list)
    for row in ordered:
        if row["service_stretch"] is not None:
            stretches_by_route[str(row["route"])].append(
                float(row["service_stretch"])
            )
    first = ordered[0]
    last = ordered[-1]
    return {
        "requests": len(ordered),
        "route_counts": {
            "local": route_counts[LOCAL],
            "remote": route_counts[REMOTE],
        },
        "route_transitions": transitions,
        "static_prior_overrides": static_overrides,
        "probes": sum(bool(row["probe"]) for row in ordered),
        "queued_retries": sum(int(row["attempts"]) > 1 for row in ordered),
        "entry": {
            "route": first["route"],
            "local_multiplier": first["local_multiplier"],
            "remote_multiplier": first["remote_multiplier"],
        },
        "exit": {
            "route": last["route"],
            "local_multiplier": last["local_multiplier"],
            "remote_multiplier": last["remote_multiplier"],
        },
        "local_multiplier": _distribution(
            float(row["local_multiplier"]) for row in ordered
        ),
        "remote_multiplier": _distribution(
            float(row["remote_multiplier"]) for row in ordered
        ),
        "selected_service_stretch": {
            "local": _distribution(stretches_by_route[LOCAL]),
            "remote": _distribution(stretches_by_route[REMOTE]),
        },
    }


def analyze(paths: list[Path]) -> dict[str, object]:
    _require(len(paths) == 2, "exactly two TEMPO block paths are required")
    blocks = []
    pooled: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for replicate, source in enumerate(paths):
        source = source.resolve()
        _require(source.is_file(), "TEMPO block is missing")
        raw = json.loads(source.read_text(encoding="utf-8"))
        _require(isinstance(raw, dict), "TEMPO block must be an object")
        rows = _foreground_rows(raw)
        groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            key = (str(row["phase"]), int(row["pair"]))
            groups[key].append(row)
            pooled[key].append(row)
        _require(
            set(groups) == {
                (phase, pair)
                for phase in (
                    "c0_cool", "c1_decoder_hot", "c2_remote_hot",
                    "c2_kv_remote_hot", "c3_both_hot", "recovery",
                )
                for pair in (0, 1)
            },
            "phase/pair coverage differs",
        )
        blocks.append({
            "replicate": replicate,
            "source": {"path": str(source), "sha256": _sha256(source)},
            "phase_pair": {
                f"{phase}:pair{pair}": _group_summary(group_rows)
                for (phase, pair), group_rows in sorted(groups.items())
            },
        })
    value: dict[str, object] = {
        "schema": SCHEMA,
        "clock_scope": "pair-local-only",
        "cross_pair_timestamp_comparisons": False,
        "blocks": blocks,
        "pooled_phase_pair": {
            f"{phase}:pair{pair}": _group_summary(group_rows)
            for (phase, pair), group_rows in sorted(pooled.items())
        },
        "interpretation_limits": {
            "calibration_only": True,
            "performance_claim_allowed": False,
            "physical_switch_bottleneck_claim_allowed": False,
        },
    }
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tempo-block", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite trace analysis")
    value = analyze(args.tempo_block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "sha256": _sha256(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
