#!/usr/bin/env python3
"""Compile a signed local TEMPO epoch plan from explicit calibration inputs."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path

from tempo.inference_epoch import (
    EpochProfile,
    WidthPoint,
    compile_epoch,
    make_epoch_artifact,
)


NS_PER_MS = 1_000_000


def milliseconds_to_ns(text: str) -> int:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid millisecond value: {text}") from exc
    if not value.is_finite() or value < 0:
        raise ValueError("millisecond values must be finite and non-negative")
    nanoseconds = value * NS_PER_MS
    integral = nanoseconds.to_integral_value()
    if nanoseconds != integral:
        raise ValueError("millisecond values must resolve to whole nanoseconds")
    return int(integral)


def parse_repeated_milliseconds(text: str) -> tuple[int, ...]:
    """Parse ``1x4,3x6,0x6`` or a plain comma-separated millisecond list."""

    values: list[int] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            raise ValueError("empty token slack item")
        if "x" in item:
            value_text, count_text = item.rsplit("x", 1)
            try:
                count = int(count_text)
            except ValueError as exc:
                raise ValueError(f"invalid repeat count: {count_text}") from exc
            if count <= 0 or str(count) != count_text:
                raise ValueError("repeat counts must be canonical positive ints")
        else:
            value_text, count = item, 1
        values.extend([milliseconds_to_ns(value_text)] * count)
    if not values:
        raise ValueError("at least one token slack value is required")
    return tuple(values)


def parse_width_penalties(text: str) -> tuple[WidthPoint, ...]:
    """Parse an increasing curve such as ``0:0,1:1,2:3,4:9``."""

    points: list[WidthPoint] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if item.count(":") != 1:
            raise ValueError(f"invalid width penalty item: {item}")
        width_text, penalty_text = item.split(":", 1)
        try:
            width = int(width_text)
        except ValueError as exc:
            raise ValueError(f"invalid width: {width_text}") from exc
        if str(width) != width_text:
            raise ValueError("widths must be canonical decimal ints")
        points.append(WidthPoint(width, milliseconds_to_ns(penalty_text)))
    return tuple(points)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-quanta", type=int, default=16)
    parser.add_argument("--deadline-tokens", type=int, required=True)
    parser.add_argument("--token-slack-ms", required=True)
    parser.add_argument("--width-penalty-ms", required=True)
    parser.add_argument("--max-width", type=int, required=True)
    parser.add_argument("--protect-prefix-tokens", type=int, default=0)
    parser.add_argument("--protect-prefix-max-width", type=int, default=1)
    return parser.parse_args()


def _resolve_output(path: Path) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidate = path if path.is_absolute() else repo_root / path
    resolved = candidate.resolve()
    if resolved != repo_root and repo_root not in resolved.parents:
        raise ValueError("output must resolve inside the repository")
    if resolved == repo_root:
        raise ValueError("output must be a file below the repository root")
    return resolved


def main() -> None:
    args = _parse_args()
    try:
        profile = EpochProfile(
            total_quanta=args.total_quanta,
            deadline_tokens=args.deadline_tokens,
            token_slack_ns=parse_repeated_milliseconds(args.token_slack_ms),
            width_points=parse_width_penalties(args.width_penalty_ms),
            max_width=args.max_width,
            protect_prefix_tokens=args.protect_prefix_tokens,
            protect_prefix_max_width=args.protect_prefix_max_width,
        )
        output = _resolve_output(args.output)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    plan = compile_epoch(profile)
    artifact = make_epoch_artifact(profile, plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "feasible": plan.feasible,
                "reason": plan.reason,
                "completion_token_exclusive": plan.completion_token_exclusive,
                "width_by_token": list(plan.width_by_token),
                "signature": plan.signature,
            },
            sort_keys=True,
        )
    )
    if not plan.feasible:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
