#!/usr/bin/env python3
"""Characterize a preseeded P-only LMCache transfer/receiver path.

The cache pool is populated before endpoint ``before`` snapshots.  Measured
background requests then hit that pool on the same producer pair, so their
critical path contains source-cache retrieval, official LMCacheConnectorV1
transfer, decoder install, and one decode token, but no long producer prefill.
This is a component-attribution experiment and cannot make a TEMPO performance
claim.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from eval.sota_4node import run_tempo_pd_contention_fixed_client as fixed


SCHEMA = "tempo-pd-kv-only-attribution-client-v2"
BLOCK_SCHEMA = "tempo-pd-kv-only-attribution-block-v2"
PRESEED_SCHEMA = "tempo-pd-kv-only-preseed-v1"
PRESEEDED_MODULE = (
    "eval.sota_4node.run_tempo_pd_elastic_stream_metrics_preseeded")
CANONICAL_MODULE = "eval.sota_4node.run_tempo_pd_elastic_stream_metrics"
DEFAULT_RATES = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0)
POOL_SIZE = 32
PROMPT_TOKENS = 4094
OUTPUT_TOKENS = 2
PRESEED_RATE = 4.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rates_from_environment() -> tuple[float, ...]:
    raw = os.environ.get(
        "TEMPO_PD_KV_ATTR_RATES", ",".join(str(value) for value in DEFAULT_RATES))
    try:
        rates = tuple(float(item) for item in raw.split(","))
    except ValueError as exc:
        raise ValueError("TEMPO_PD_KV_ATTR_RATES contains a non-number") from exc
    _require(bool(rates), "attribution rate ladder is empty")
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in rates),
        "attribution rates must be finite and non-negative",
    )
    _require(
        tuple(sorted(set(rates))) == rates,
        "attribution rates must be unique and strictly increasing",
    )
    return rates


def _repetitions_from_environment() -> int:
    raw = os.environ.get("TEMPO_PD_KV_ATTR_REPETITIONS", "1")
    try:
        repetitions = int(raw)
    except ValueError as exc:
        raise ValueError(
            "TEMPO_PD_KV_ATTR_REPETITIONS must be an integer") from exc
    _require(1 <= repetitions <= 4,
             "attribution repetitions must be in [1, 4]")
    return repetitions


def _arm_order_policy_from_environment() -> str:
    policy = os.environ.get(
        "TEMPO_PD_KV_ATTR_ARM_ORDER", "local_remote")
    _require(
        policy in {"local_remote", "remote_local", "paired_abba"},
        "attribution arm order must be local_remote, remote_local, or paired_abba",
    )
    return policy


def _arm_order(policy: str, replicate_index: int) -> tuple[str, str]:
    if policy == "local_remote":
        return ("local", "remote")
    if policy == "remote_local":
        return ("remote", "local")
    _require(policy == "paired_abba", "unknown arm-order policy")
    return (
        ("local", "remote")
        if replicate_index % 2 == 0 else ("remote", "local")
    )


def _stream_command(
    args: argparse.Namespace,
    *,
    module: str,
    workload: Path,
    output: Path,
    run_id: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        module,
        "--base-url",
        args.base_url,
        "--model",
        str(args.model),
        "--served-model-name",
        args.served_model_name,
        "--workload",
        str(workload),
        "--output",
        str(output),
        "--mode",
        "tempo_auto",
        "--run-id",
        run_id,
        "--default-max-tokens",
        str(args.default_max_tokens),
        "--max-workers",
        str(args.max_workers),
        "--timeout-s",
        str(args.timeout_s),
        "--seed",
        str(args.seed),
    ]
    if args.api_key_env:
        command.extend(("--api-key-env", args.api_key_env))
    return command


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _pool_prompts(tokenizer, template: tuple[int, ...]) -> tuple[str, ...]:
    prompts = tuple(
        fixed._unique_prompt(tokenizer, template, 200_000 + index)
        for index in range(POOL_SIZE)
    )
    hashes = {hashlib.sha256(value.encode()).hexdigest() for value in prompts}
    _require(len(hashes) == POOL_SIZE, "P-only pool prompts are not unique")
    return prompts


def _preseed(
    args: argparse.Namespace,
    *,
    root: Path,
    pool: tuple[str, ...],
) -> dict[str, object]:
    workload = root / "p_only_preseed.jsonl"
    raw_path = root / "p_only_preseed.raw.json"
    rows = []
    for index, prompt in enumerate(pool):
        rows.append({
            "request_id": (
                f"epd-remote-kvattr-preseed-warm-item-{index:06d}"),
            "prompt": prompt,
            "max_tokens": OUTPUT_TOKENS,
            "arrival_offset_ms": round(
                (index + 0.5) * 1000.0 / PRESEED_RATE, 6),
        })
    _write_rows(workload, rows)
    completed = subprocess.run(
        _stream_command(
            args,
            module=CANONICAL_MODULE,
            workload=workload,
            output=raw_path,
            run_id=f"{args.run_id}-p-only-preseed",
        ),
        check=False,
    )
    _require(completed.returncode == 0, "P-only pool preseed failed")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(requests, list) and len(requests) == POOL_SIZE,
             "P-only preseed request count differs")
    _require(isinstance(decisions, list) and len(decisions) == POOL_SIZE,
             "P-only preseed decision count differs")
    _require(all(row.get("valid") is True for row in requests),
             "P-only preseed stream is invalid")
    _require(all(
        isinstance(row.get("p_only_cache_seed"), dict)
        and row["p_only_cache_seed"].get("valid") is True
        and row["p_only_cache_seed"].get("route")
        == "official_lmcache_remote_prefill"
        for row in requests
    ), "P-only preseed evidence is incomplete")
    _require(all(
        row.get("route") == "official_lmcache_remote_prefill"
        and row.get("lmcache_source_cached_tokens") == PROMPT_TOKENS
        and row.get("lmcache_source_full_hit_observed") is True
        for row in decisions
    ), "P-only post-seed probes did not observe exact source hits")
    pair_counts = collections.Counter(
        int(row["frontend_pair_index"]) for row in decisions)
    _require(pair_counts == {0: POOL_SIZE // 2, 1: POOL_SIZE // 2},
             "P-only pool is not balanced across producer pairs")
    return {
        "schema": PRESEED_SCHEMA,
        "workload": str(workload.resolve()),
        "raw": str(raw_path.resolve()),
        "pool_size": POOL_SIZE,
        "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "preseed_rate_per_s": PRESEED_RATE,
        "pair_counts": {str(key): value for key, value in sorted(pair_counts.items())},
        "all_post_seed_source_hits_exact": True,
        "measurement_window_includes_seed_requests": False,
    }


def _uniform_offsets(rate: float, duration_ms: float) -> list[float]:
    if rate == 0.0:
        return []
    count = int(math.floor(rate * duration_ms / 1000.0))
    return [(index + 0.5) * 1000.0 / rate for index in range(count)]


def _block_rows(
    *,
    tokenizer,
    template: tuple[int, ...],
    pool: tuple[str, ...],
    rate: float,
    rate_index: int,
    replicate_index: int = 0,
    arm: str,
    duration_ms: float,
    foreground_rate: float,
    decoder_hot_rate: float = 0.0,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], str]:
    _require(arm in {"local", "remote"}, "foreground arm is invalid")
    rows: list[dict[str, object]] = []
    index: dict[str, dict[str, object]] = {}
    semantic: list[dict[str, object]] = []
    for ordinal, offset in enumerate(_uniform_offsets(rate, duration_ms)):
        pool_index = ordinal % POOL_SIZE
        request_id = (
            f"epd-remote-kvattr-cache-p-only-measured-rate{rate_index:02d}-"
            f"rep{replicate_index:02d}-"
            f"{arm}-occ-{ordinal:06d}-item-{pool_index:06d}")
        row = {
            "request_id": request_id,
            "prompt": pool[pool_index],
            "max_tokens": OUTPUT_TOKENS,
            "arrival_offset_ms": round(offset, 6),
        }
        rows.append(row)
        index[request_id] = {
            "tenant": "p_only_remote_background",
            "arm": "remote",
            "pool_index": pool_index,
            "arrival_offset_ms": round(offset, 6),
            "expected_cache": "p_only_full_source_hit",
        }
        semantic.append({
            "tenant": "p_only_remote_background",
            "arrival_offset_ms": round(offset, 6),
            "pool_index": pool_index,
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
        })
    for ordinal, offset in enumerate(
        _uniform_offsets(foreground_rate, duration_ms)
    ):
        marker = (
            220_000 + replicate_index * 8_192
            + rate_index * 1_024 + ordinal
        )
        _require(marker < (1 << 18), "foreground marker space exhausted")
        prompt = fixed._unique_prompt(tokenizer, template, marker)
        request_id = (
            f"epd-{arm}-kvattr-measured-rate{rate_index:02d}-"
            f"rep{replicate_index:02d}-"
            f"foreground-{ordinal:06d}")
        row = {
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": OUTPUT_TOKENS,
            "arrival_offset_ms": round(offset, 6),
        }
        rows.append(row)
        index[request_id] = {
            "tenant": "foreground",
            "arm": arm,
            "arrival_offset_ms": round(offset, 6),
            "expected_cache": "cold_miss",
        }
        semantic.append({
            "tenant": "foreground",
            "arrival_offset_ms": round(offset, 6),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
        })
    for ordinal, offset in enumerate(
        _uniform_offsets(decoder_hot_rate, duration_ms)
    ):
        # Decoder-hot prompts must remain cold in every block.  The arm-local
        # marker offset changes prompt contents across paired blocks while the
        # semantic schedule deliberately binds only geometry and timing.
        marker = (
            20_000 + replicate_index * 32_768 + rate_index * 4_096
            + (0 if arm == "local" else 1_024) + ordinal
        )
        _require(marker < (1 << 18), "decoder-hot marker space exhausted")
        prompt = fixed._unique_prompt(tokenizer, template, marker)
        request_id = (
            f"epd-local-kvattr-decoderhot-measured-rate{rate_index:02d}-"
            f"rep{replicate_index:02d}-"
            f"{arm}-{ordinal:06d}")
        row = {
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": OUTPUT_TOKENS,
            "arrival_offset_ms": round(offset, 6),
        }
        rows.append(row)
        index[request_id] = {
            "tenant": "decoder_hot_background",
            "arm": "local",
            "arrival_offset_ms": round(offset, 6),
            "expected_cache": "cold_miss",
        }
        semantic.append({
            "tenant": "decoder_hot_background",
            "arrival_offset_ms": round(offset, 6),
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
        })
    rows.sort(key=lambda row: (
        float(row["arrival_offset_ms"]),
        0 if "p_only_remote_background" in index[str(row["request_id"])]["tenant"] else 1,
        str(row["request_id"]),
    ))
    semantic.sort(key=lambda row: (
        float(row["arrival_offset_ms"]), str(row["tenant"]),
        int(row.get("pool_index", -1)),
    ))
    encoded = json.dumps(
        semantic, sort_keys=True, separators=(",", ":")).encode()
    return rows, index, hashlib.sha256(encoded).hexdigest()


def _run_with_evidence(
    command: list[str], *, args: argparse.Namespace,
) -> tuple[dict[str, object], int]:
    before = fixed._capture_endpoint_evidence(
        args.endpoint_evidence_url, stage="before", require_valid_delta=False)
    child = subprocess.Popen(command)
    try:
        time.sleep(args.phase_duration_ms / 2_000.0)
        _require(child.poll() is None,
                 "P-only attribution child exited before midpoint")
        midpoint = fixed._capture_endpoint_evidence(
            args.endpoint_evidence_url,
            stage="midpoint",
            require_valid_delta=True,
        )
        return_code = child.wait(timeout=1200.0)
        _require(return_code in {0, 2},
                 f"P-only attribution child returned {return_code}")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10.0)
        raise
    # Under overload, draining the open-loop block can legitimately take more
    # than the Cassini sampler's bounded 10 s delta window.  Preserve that
    # sample as explicitly invalid/missing instead of converting it to zero or
    # discarding otherwise exact request and vLLM cumulative evidence.
    after = fixed._capture_endpoint_evidence(
        args.endpoint_evidence_url, stage="after", require_valid_delta=False)
    evidence = {
        "schema": fixed.ENDPOINT_EVIDENCE_SCHEMA,
        "sampling_policy": "on_demand_block_boundary_and_midpoint",
        "cross_endpoint_clock_subtraction_allowed": False,
        "before": before,
        "midpoint": midpoint,
        "after": after,
    }
    fixed._validate_endpoint_evidence_bundle(evidence)
    return evidence, return_code


def _cassini_quality(endpoint_evidence: dict[str, object]) -> dict[str, object]:
    invalid = []
    total = 0
    for stage_name in ("before", "midpoint", "after"):
        stage = endpoint_evidence[stage_name]
        for row in stage["snapshots"]:
            total += 1
            cassini = row["probe"]["cassini"]
            if cassini.get("valid") is not True:
                invalid.append({
                    "stage": stage_name,
                    "endpoint_id": cassini.get("endpoint_id"),
                    "invalid_reason": cassini.get("invalid_reason"),
                    "window_ms": cassini.get("window_ms"),
                })
    return {
        "samples_total": total,
        "samples_valid": total - len(invalid),
        "all_valid": not invalid,
        "invalid_samples": invalid,
        "invalid_is_missing_not_zero": True,
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _latency_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    valid = [row for row in rows if row.get("valid") is True]
    e2e = [
        (int(row["stream_end_offset_ns"]) - int(row["dispatch_offset_ns"]))
        / 1_000_000.0
        for row in valid
    ]
    ttft = [
        (int(row["token_arrival_offsets_ns"][0]) - int(row["dispatch_offset_ns"]))
        / 1_000_000.0
        for row in valid if row.get("token_arrival_offsets_ns")
    ]
    return {
        "offered": len(rows),
        "completed_valid": len(valid),
        "errors": sum(row.get("error") is not None for row in rows),
        "non_200": sum(row.get("http_status") != 200 for row in rows),
        "e2e_median_ms": statistics.median(e2e) if e2e else None,
        "e2e_p99_ms": _percentile(e2e, 0.99),
        "ttft_median_ms": statistics.median(ttft) if ttft else None,
        "ttft_p99_ms": _percentile(ttft, 0.99),
    }


def _augment_block(
    raw_path: Path,
    *,
    request_index: dict[str, dict[str, object]],
    endpoint_evidence: dict[str, object],
    rate: float,
    replicate_index: int,
    block_sequence_index: int,
    arm_order_policy: str,
    arm: str,
    schedule_sha256: str,
    return_code: int,
) -> tuple[dict[str, object], dict[str, object]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(requests, list), "attribution requests are missing")
    _require(isinstance(decisions, list), "attribution decisions are missing")
    request_ids = {row.get("request_id") for row in requests}
    _require(request_ids == set(request_index),
             "attribution request IDs differ")
    decision_index = {row.get("request_id"): row for row in decisions}
    _require(len(decision_index) == len(decisions),
             "attribution decisions contain duplicate IDs")
    _require(set(decision_index) == set(request_index),
             "attribution decision IDs differ")
    for request_id, metadata in request_index.items():
        decision = decision_index[request_id]
        expected_route = (
            "decoder_local_chunked_prefill"
            if metadata["arm"] == "local"
            else "official_lmcache_remote_prefill"
        )
        _require(decision.get("route") == expected_route,
                 "attribution route differs from pinned route")
        if metadata["tenant"] == "p_only_remote_background":
            _require(
                decision.get("decision_cache_residency") == "prefill_only"
                and decision.get("completion_cache_residency") == "prefill_only"
                and decision.get("request_cache_contract") == "p_only"
                and decision.get("reason") == "fixed_official_lmcache_remote"
                and decision.get("lmcache_source_cached_tokens") == PROMPT_TOKENS
                and decision.get("lmcache_source_full_hit_observed") is True,
                "measured P-only background lost its full source hit",
            )
        elif metadata["arm"] == "remote":
            _require(decision.get("lmcache_source_cached_tokens") == 0,
                     "cold remote foreground observed a source hit")
        else:
            _require(decision.get("lmcache_source_cached_tokens") is None,
                     "local foreground has remote cache evidence")
    by_id = {row["request_id"]: row for row in requests}
    background = [
        by_id[request_id] for request_id, metadata in request_index.items()
        if metadata["tenant"] == "p_only_remote_background"
    ]
    foreground = [
        by_id[request_id] for request_id, metadata in request_index.items()
        if metadata["tenant"] == "foreground"
    ]
    decoder_hot = [
        by_id[request_id] for request_id, metadata in request_index.items()
        if metadata["tenant"] == "decoder_hot_background"
    ]
    contract = {
        "schema": BLOCK_SCHEMA,
        "background_rate_per_s": rate,
        "replicate_index": replicate_index,
        "block_sequence_index": block_sequence_index,
        "arm_order_policy": arm_order_policy,
        "foreground_arm": arm,
        "semantic_schedule_sha256": schedule_sha256,
        "request_counts": {
            "p_only_remote_background": len(background),
            "decoder_hot_background": len(decoder_hot),
            "foreground": len(foreground),
        },
        "preseed_completed_before_endpoint_before_snapshot": True,
        "measurement_window_includes_seed_requests": False,
        "background_full_source_hits_exact": True,
        "background_decision_cache_contract_exact": True,
        "pinned_routes_exact": True,
        "decoder_prefix_caching_disabled_exact": all(
            row.get("decoder_prefix_caching") is False
            for row in decisions
        ),
        "background_decision_cache_contract_exact": True,
        "background_same_pair_by_terminal_pool_index": True,
        "actual_inference_background_only": True,
        "official_lmcache_connector_v1": True,
        "synthetic_network_background": False,
        "cross_endpoint_clock_subtraction_allowed": False,
        "endpoint_evidence_exact": True,
        "cassini_quality": _cassini_quality(endpoint_evidence),
        "child_return_code": return_code,
        "all_requests_valid": all(row.get("valid") is True for row in requests),
    }
    summary = {
        "background_rate_per_s": rate,
        "replicate_index": replicate_index,
        "block_sequence_index": block_sequence_index,
        "foreground_arm": arm,
        "background": _latency_summary(background),
        "decoder_hot_background": _latency_summary(decoder_hot),
        "foreground": _latency_summary(foreground),
        "all_requests_valid": contract["all_requests_valid"],
        "child_return_code": return_code,
    }
    raw["kv_only_attribution_contract"] = contract
    raw["endpoint_evidence"] = endpoint_evidence
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract, summary


def _warmup(args: argparse.Namespace, tokenizer, templates) -> int:
    return fixed._warmup(args, tokenizer, templates)


def _measured(args: argparse.Namespace, tokenizer, templates) -> int:
    _require(len(args.endpoint_evidence_url) == 4,
             "P-only attribution requires four endpoint probes")
    _require(args.phase_duration_ms >= 4_000.0,
             "P-only attribution phase is too short")
    rates = _rates_from_environment()
    repetitions = _repetitions_from_environment()
    arm_order_policy = _arm_order_policy_from_environment()
    if arm_order_policy == "paired_abba":
        _require(repetitions >= 2,
                 "paired_abba requires at least two repetitions")
    try:
        decoder_hot_rate = float(os.environ.get(
            "TEMPO_PD_KV_ATTR_DECODER_HOT_RATE", "0"))
    except ValueError as exc:
        raise ValueError(
            "TEMPO_PD_KV_ATTR_DECODER_HOT_RATE is not numeric") from exc
    _require(math.isfinite(decoder_hot_rate) and decoder_hot_rate >= 0.0,
             "decoder-hot rate must be finite and non-negative")
    root = args.output.parent / "kv_only_attribution"
    root.mkdir()
    pool = _pool_prompts(tokenizer, templates[PROMPT_TOKENS])
    preseed = _preseed(args, root=root, pool=pool)
    time.sleep(args.cooldown_s)
    os.environ["TEMPO_PD_P_ONLY_PRESEEDED"] = "1"

    artifacts: dict[str, str] = {}
    contracts: dict[str, dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    stopped_after: str | None = None
    sequence = 0
    try:
        for rate_index, rate in enumerate(rates):
            for replicate_index in range(repetitions):
                pair_hashes: dict[str, str] = {}
                for arm in _arm_order(arm_order_policy, replicate_index):
                    key = (
                        f"{sequence:02d}_rate{rate:g}_"
                        f"rep{replicate_index:02d}_{arm}"
                    )
                    rows, request_index, schedule_sha = _block_rows(
                        tokenizer=tokenizer,
                        template=templates[PROMPT_TOKENS],
                        pool=pool,
                        rate=rate,
                        rate_index=rate_index,
                        replicate_index=replicate_index,
                        arm=arm,
                        duration_ms=args.phase_duration_ms,
                        foreground_rate=args.request_rate,
                        decoder_hot_rate=decoder_hot_rate,
                    )
                    pair_hashes[arm] = schedule_sha
                    workload_path = root / f"{key}.jsonl"
                    raw_path = root / f"{key}.raw.json"
                    _write_rows(workload_path, rows)
                    evidence, return_code = _run_with_evidence(
                        _stream_command(
                            args,
                            module=PRESEEDED_MODULE,
                            workload=workload_path,
                            output=raw_path,
                            run_id=f"{args.run_id}-{key}",
                        ),
                        args=args,
                    )
                    contract, summary = _augment_block(
                        raw_path,
                        request_index=request_index,
                        endpoint_evidence=evidence,
                        rate=rate,
                        replicate_index=replicate_index,
                        block_sequence_index=sequence,
                        arm_order_policy=arm_order_policy,
                        arm=arm,
                        schedule_sha256=schedule_sha,
                        return_code=return_code,
                    )
                    artifacts[key] = str(raw_path.resolve())
                    contracts[key] = contract
                    summaries.append(summary)
                    sequence += 1
                    if return_code != 0:
                        stopped_after = key
                        break
                    time.sleep(args.cooldown_s)
                if len(pair_hashes) == 2:
                    _require(pair_hashes["local"] == pair_hashes["remote"],
                             "local/remote semantic schedules differ")
                if stopped_after is not None:
                    break
            if stopped_after is not None:
                break
    finally:
        os.environ.pop("TEMPO_PD_P_ONLY_PRESEEDED", None)

    public = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "purpose": "P-only KV-transfer/receiver component attribution",
        "workload_mode": (
            "coupled_decoder_hot_and_p_only_remote"
            if decoder_hot_rate > 0.0 else "p_only_remote_only"),
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "source_workload_sha256": hashlib.sha256(
            args.workload.read_bytes()).hexdigest(),
        "rates_per_s": list(rates),
        "repetitions_per_rate": repetitions,
        "arm_order_policy": arm_order_policy,
        "paired_semantic_schedules_exact": True,
        "phase_duration_ms": args.phase_duration_ms,
        "foreground_rate_per_s": args.request_rate,
        "decoder_hot_rate_per_s": decoder_hot_rate,
        "cooldown_s": args.cooldown_s,
        "preseed": preseed,
        "artifacts": artifacts,
        "contracts": contracts,
        "summaries": summaries,
        "stopped_after_first_invalid_block": stopped_after,
    }
    args.output.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "blocks": len(summaries),
        "stopped_after": stopped_after,
    }, sort_keys=True))
    return 0


def main() -> int:
    args = fixed._parse()
    _require(args.mode == "tempo_auto", "attribution client requires tempo_auto")
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    _require(args.model.is_absolute(), "model path must be absolute")
    _require(args.request_rate > 0.0, "foreground rate must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True)
    templates = fixed._load_templates(args.workload, tokenizer)
    if args.run_id.endswith("-warmup"):
        return _warmup(args, tokenizer, templates)
    return _measured(args, tokenizer, templates)


if __name__ == "__main__":
    raise SystemExit(main())
