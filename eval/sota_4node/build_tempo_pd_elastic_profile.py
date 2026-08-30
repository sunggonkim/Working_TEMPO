#!/usr/bin/env python3
"""Build a frozen Elastic-PD profile with explicit evidence scope."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import tempfile

from eval.sota_4node import build_tempo_pd_elastic_profile_v446 as _normalizer
from eval.sota_4node import build_tempo_pd_elastic_profile_v445 as _builder
from tempo.pd_elastic_profile import load_elastic_profile


def _linear_quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _paired_route_gaps(
    raw_paths: list[Path],
) -> dict[tuple[int, int], list[float]]:
    paired = defaultdict(dict)
    for raw_path in raw_paths:
        artifact = json.loads(raw_path.resolve().read_text())
        source_cohort = artifact.get("_tempo_profile_source_cohort")
        if not isinstance(source_cohort, str) or not source_cohort:
            raise ValueError("paired route evidence lacks a source cohort")
        for row in artifact.get("requests", []):
            router = row.get("router")
            usage = row.get("usage")
            if not isinstance(router, dict) or not isinstance(usage, dict):
                continue
            route = router.get("route")
            label = (
                "local" if route in _builder._LOCAL_ROUTES else
                "remote" if route in _builder._REMOTE_ROUTES else None
            )
            if label is None:
                continue
            match = _builder._BALANCED_ITEM.fullmatch(
                str(row.get("request_id", "")))
            if match is None:
                continue
            prompt_tokens = usage.get("prompt_tokens")
            output_tokens = row.get("requested_max_tokens")
            key = (
                source_cohort,
                int(match.group(1)), int(match.group(2)),
                prompt_tokens, output_tokens,
            )
            if label in paired[key]:
                raise ValueError("duplicate paired route latency")
            paired[key][label] = _builder._latency_ms(row)
    gaps = defaultdict(list)
    for (_cohort, _replicate, _item, prompt_tokens, output_tokens), routes in paired.items():
        if set(routes) == {"local", "remote"}:
            gaps[(prompt_tokens, output_tokens)].append(
                routes["local"] - routes["remote"])
    return dict(gaps)


def _apply_paired_gap_uncertainty(
    payload, raw_paths, *, minimum_pairs, lower_quantile,
):
    gaps = _paired_route_gaps(raw_paths)
    for row in payload["rows"]:
        key = row["prompt_tokens"], row["output_tokens"]
        values = gaps.get(key, [])
        if len(values) < minimum_pairs:
            raise ValueError(
                f"profile row {key} requires {minimum_pairs} paired route gaps")
        representative_gap = (
            row["local_upper_bound_ms"] - row["remote_upper_bound_ms"])
        lower_gap = _linear_quantile(values, lower_quantile)
        row["uncertainty_ms"] = max(
            1.0, representative_gap - lower_gap)


def build_profile(
    raw_paths: list[Path], *, deployment_scope: str,
    paired_gap_lower_quantile: float = 0.0, **kwargs,
):
    if deployment_scope not in {"screen_only", "replicated"}:
        raise ValueError("deployment_scope must be screen_only or replicated")
    if not 0.0 <= paired_gap_lower_quantile <= 0.5:
        raise ValueError("paired gap lower quantile must be between 0 and 0.5")
    with tempfile.TemporaryDirectory(prefix="tempo-elastic-profile-canonical-") as tmp:
        root = Path(tmp)
        normalized_paths = []
        for index, raw_path in enumerate(raw_paths):
            artifact = json.loads(raw_path.resolve().read_text())
            normalized, _ = _normalizer.normalize_artifact(artifact)
            normalized["_tempo_profile_source_cohort"] = str(
                raw_path.resolve().parent)
            path = root / f"normalized-{index}.json"
            path.write_text(json.dumps(normalized))
            normalized_paths.append(path)
        payload = _builder.build_profile(normalized_paths, **kwargs)
        _apply_paired_gap_uncertainty(
            payload, normalized_paths,
            minimum_pairs=3 if deployment_scope == "replicated" else 2,
            lower_quantile=paired_gap_lower_quantile,
        )
    payload["deployment_scope"] = deployment_scope
    if deployment_scope == "replicated":
        for row in payload["rows"]:
            if row["samples_local"] < 3 or row["samples_remote"] < 3:
                raise ValueError("replicated profile requires three samples per route")
            if not row["outputs_equivalent"] or row["remote_transfer_failures"] != 0:
                raise ValueError("replicated profile requires exact successful evidence")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--deployment-scope", choices=("screen_only", "replicated"),
                        default="replicated")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--topology-id", required=True)
    parser.add_argument("--remote-backend", required=True)
    parser.add_argument("--classifier-version", required=True)
    parser.add_argument("--kv-bytes-per-token", type=int, required=True)
    parser.add_argument("--local-capacity-equivalent", type=int, default=6)
    parser.add_argument("--remote-capacity-equivalent", type=int, default=1)
    parser.add_argument("--latency-estimator", choices=("max", "median"),
                        default="max")
    parser.add_argument("--spill-regression-budget-ms", type=float,
                        default=5.0)
    parser.add_argument(
        "--paired-gap-lower-quantile", type=float, default=0.0,
        help="empirical lower quantile of paired local-minus-remote gaps")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite profile: {args.output}")
    payload = build_profile(
        args.raw,
        deployment_scope=args.deployment_scope,
        paired_gap_lower_quantile=args.paired_gap_lower_quantile,
        profile_id=args.profile_id,
        model_id=args.model_id,
        model_revision=args.model_revision,
        topology_id=args.topology_id,
        remote_backend=args.remote_backend,
        classifier_version=args.classifier_version,
        kv_bytes_per_token=args.kv_bytes_per_token,
        local_capacity_equivalent=args.local_capacity_equivalent,
        remote_capacity_equivalent=args.remote_capacity_equivalent,
        latency_estimator=args.latency_estimator,
        spill_regression_budget_ms=args.spill_regression_budget_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    loaded = load_elastic_profile(args.output)
    print(json.dumps({
        "profile_id": loaded.profile_id,
        "deployment_scope": loaded.deployment_scope,
        "fingerprint_sha256": loaded.fingerprint_sha256,
        "rows": len(loaded.rows),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
