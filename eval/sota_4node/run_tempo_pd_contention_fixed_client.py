#!/usr/bin/env python3
"""Run the preregistered C1/C2 fixed-arm contention screen.

Foreground and background inference requests share one open-loop client clock.
The background tenants are route-pinned by their canonical ``epd-local`` or
``epd-remote`` request IDs.  This client never starts or discovers servers.
"""

from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import urlopen

from tempo.cassini_endpoint import validate_cassini_endpoint_sample
from tempo.domain_evidence import CounterSupport
from tempo.pd_endpoint_evidence import PDEndpointRole, endpoint_metric_names
from eval.sota_4node.tempo_pd_endpoint_probe import (
    SCHEMA as ENDPOINT_PROBE_SCHEMA,
    validate_vllm_endpoint_cumulative,
)

from tempo.pd_contention_workload import (
    CacheState,
    ContentionState,
    CROSSOVER_FOREGROUND_GEOMETRIES,
    FixedArmObservation,
    ForegroundArm,
    ForegroundObservation,
    LoadSelection,
    Tenant,
    TokenGeometry,
    TrafficShape,
    build_schedule,
    evaluate_crossover,
    semantic_schedule_sha256,
)
from eval.sota_4node import (
    run_tempo_pd_same_server_mixed_only_client_unique_chunks_v308 as unique,
)


SCHEMA = "tempo-pd-contention-fixed-client-v7"
BLOCK_SCHEMA = "tempo-pd-contention-fixed-block-v7"
ENDPOINT_EVIDENCE_SCHEMA = "tempo-pd-contention-endpoint-evidence-v1"
LOCAL_ROUTE = "decoder_local_chunked_prefill"
REMOTE_ROUTE = "official_lmcache_remote_prefill"
FOREGROUND_GEOMETRIES = CROSSOVER_FOREGROUND_GEOMETRIES
BLOCK_ORDER = (
    (ContentionState.C1, ForegroundArm.LOCAL, 0),
    (ContentionState.C1, ForegroundArm.REMOTE, 0),
    (ContentionState.C2, ForegroundArm.REMOTE, 0),
    (ContentionState.C2, ForegroundArm.LOCAL, 0),
    (ContentionState.C2, ForegroundArm.LOCAL, 1),
    (ContentionState.C2, ForegroundArm.REMOTE, 1),
    (ContentionState.C1, ForegroundArm.REMOTE, 1),
    (ContentionState.C1, ForegroundArm.LOCAL, 1),
)
PREFLIGHT_REQUESTS = (
    (ForegroundArm.LOCAL, "epd-local-ct-preflight-local"),
    (ForegroundArm.REMOTE, "epd-remote-ct-preflight-remote"),
)


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--default-max-tokens", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--api-key-env")
    parser.add_argument("--decoder-reference-rate", type=float, default=32.0)
    parser.add_argument("--remote-reference-rate", type=float, default=6.8)
    parser.add_argument("--load-fraction", type=float, default=0.50)
    parser.add_argument("--phase-duration-ms", type=float, default=15_000.0)
    parser.add_argument("--cooldown-s", type=float, default=2.0)
    parser.add_argument("--endpoint-evidence-url", action="append", default=[])
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_templates(path: Path, tokenizer) -> dict[int, tuple[int, ...]]:
    _require(path.is_file(), "explicit source workload is missing")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(bool(rows), "source workload is empty")
    templates: dict[int, tuple[int, ...]] = {}
    for row in rows:
        prompt = row.get("prompt")
        _require(isinstance(prompt, str) and prompt, "source prompt is invalid")
        token_ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
        templates.setdefault(len(token_ids), token_ids)
    required = {
        geometry.prompt_tokens for geometry in FOREGROUND_GEOMETRIES
    } | {512, 4094}
    missing = sorted(required - set(templates))
    _require(not missing, f"source workload lacks prompt templates: {missing}")
    return {length: templates[length] for length in sorted(required)}


def _unique_prompt(tokenizer, token_ids: tuple[int, ...], marker_id: int) -> str:
    marker_ids = tokenizer.encode(
        unique._marker(marker_id), add_special_tokens=False)
    _require(bool(marker_ids), "unique marker did not tokenize")
    _require(len(marker_ids) < len(token_ids), "marker exceeds prompt template")
    candidate = list(marker_ids) + list(token_ids[len(marker_ids):])
    prompt = tokenizer.decode(
        candidate,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    checked = tokenizer.encode(prompt, add_special_tokens=False)
    _require(len(checked) == len(token_ids), "marker changed prompt geometry")
    return prompt


def _write_workload(
    path: Path,
    *,
    requests,
    templates: dict[int, tuple[int, ...]],
    tokenizer,
    marker_base: int,
) -> dict[str, dict[str, object]]:
    _require(not path.exists(), f"refusing to overwrite {path}")
    rows = []
    index = {}
    first_chunks = set()
    for item, request in enumerate(requests):
        marker_id = marker_base + item
        _require(marker_id < (1 << 18), "contention marker space exhausted")
        prompt = _unique_prompt(
            tokenizer,
            templates[request.geometry.prompt_tokens],
            marker_id,
        )
        chunk = tuple(tokenizer.encode(
            prompt, add_special_tokens=False)[:256])
        _require(chunk not in first_chunks, "first LMCache chunk is not unique")
        first_chunks.add(chunk)
        rows.append({
            "request_id": request.request_id,
            "prompt": prompt,
            "max_tokens": request.geometry.output_tokens,
            "arrival_offset_ms": round(request.arrival_offset_ms, 6),
        })
        index[request.request_id] = {
            **request.semantic_dict(),
            "arm": request.arm.value,
            "pair_key": (
                f"{request.phase.value}:foreground:{request.ordinal:06d}"
                if request.tenant is Tenant.FOREGROUND else None
            ),
        }
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return index


def _child_command(
    args: argparse.Namespace, *, workload: Path, output: Path, run_id: str,
    max_workers: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "eval.sota_4node.run_tempo_pd_elastic_stream_metrics",
        "--base-url", args.base_url,
        "--model", str(args.model),
        "--served-model-name", args.served_model_name,
        "--workload", str(workload),
        "--output", str(output),
        "--mode", "tempo_auto",
        "--run-id", run_id,
        "--default-max-tokens", str(args.default_max_tokens),
        "--max-workers", str(max_workers or args.max_workers),
        "--timeout-s", str(args.timeout_s),
        "--seed", str(args.seed),
    ]
    if args.api_key_env:
        command.extend(("--api-key-env", args.api_key_env))
    return command


def _fetch_endpoint_snapshot(url: str) -> dict[str, object]:
    started_ns = time.perf_counter_ns()
    try:
        with urlopen(url.rstrip("/") + "/snapshot", timeout=10.0) as response:
            _require(response.status == 200, "endpoint probe returned non-200")
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"endpoint probe HTTP {exc.code}: {body[:2000]}") from exc
    received_ns = time.perf_counter_ns()
    _require(isinstance(value, dict), "endpoint probe payload is not an object")
    _require(value.get("schema") == ENDPOINT_PROBE_SCHEMA,
             "endpoint probe schema mismatch")
    endpoint = value.get("endpoint")
    cumulative = value.get("vllm_cumulative")
    cassini = value.get("cassini")
    _require(isinstance(endpoint, dict), "endpoint load evidence is missing")
    validate_vllm_endpoint_cumulative(cumulative)
    validate_cassini_endpoint_sample(cassini)
    _require(endpoint.get("endpoint_id") == cassini.get("endpoint_id"),
             "endpoint and Cassini identities differ")
    _require(endpoint.get("role") == cassini.get("role"),
             "endpoint and Cassini roles differ")
    _require(endpoint.get("pair_index") == cassini.get("pair_index"),
             "endpoint and Cassini pair indices differ")
    return {
        "source_url": url,
        "client_fetch_started_monotonic_ns": started_ns,
        "client_received_monotonic_ns": received_ns,
        "probe": value,
    }


def _capture_endpoint_evidence(
    urls: list[str], *, stage: str, require_valid_delta: bool,
) -> dict[str, object]:
    _require(stage in {"before", "midpoint", "after"},
             "endpoint evidence stage is invalid")
    _require(len(urls) == 4 and len(set(urls)) == 4,
             "exactly four unique endpoint evidence URLs are required")
    with ThreadPoolExecutor(max_workers=4) as pool:
        snapshots = list(pool.map(_fetch_endpoint_snapshot, urls))
    identities = {}
    for row in snapshots:
        probe = row["probe"]
        endpoint = probe["endpoint"]
        endpoint_id = endpoint.get("endpoint_id")
        _require(isinstance(endpoint_id, str) and endpoint_id,
                 "endpoint identity is missing")
        _require(endpoint_id not in identities, "duplicate endpoint identity")
        try:
            role = PDEndpointRole(endpoint.get("role"))
        except (TypeError, ValueError) as exc:
            raise ValueError("endpoint role is invalid") from exc
        pair_index = endpoint.get("pair_index")
        _require(type(pair_index) is int and pair_index in (0, 1),
                 "endpoint pair index is invalid")
        metrics = endpoint.get("metrics")
        _require(isinstance(metrics, dict), "endpoint metrics are missing")
        _require(set(metrics) == set(endpoint_metric_names(role)),
                 "endpoint metric inventory is not exact")
        for metric in metrics.values():
            _require(isinstance(metric, dict), "endpoint metric is malformed")
            try:
                support = CounterSupport(metric.get("support"))
            except (TypeError, ValueError) as exc:
                raise ValueError("endpoint metric support is invalid") from exc
            _require(
                (support is CounterSupport.SUPPORTED) ==
                (metric.get("value") is not None),
                "endpoint metric support/value contract is invalid",
            )
        cassini = probe["cassini"]
        if require_valid_delta:
            _require(cassini.get("valid") is True,
                     "measured Cassini endpoint delta is invalid")
        identities[endpoint_id] = (role.value, pair_index)
    expected = {
        "pair0-prefill": ("prefill", 0),
        "pair0-decoder": ("decoder", 0),
        "pair1-prefill": ("prefill", 1),
        "pair1-decoder": ("decoder", 1),
    }
    _require(identities == expected, "P/D endpoint identity set is not exact")
    return {
        "schema": ENDPOINT_EVIDENCE_SCHEMA,
        "stage": stage,
        "snapshots": sorted(
            snapshots,
            key=lambda row: row["probe"]["endpoint"]["endpoint_id"],
        ),
    }


def _run_child_with_midpoint_evidence(
    command: list[str], *, args: argparse.Namespace,
) -> dict[str, object]:
    before = _capture_endpoint_evidence(
        args.endpoint_evidence_url,
        stage="before",
        require_valid_delta=False,
    )
    child = subprocess.Popen(command)
    try:
        time.sleep(args.phase_duration_ms / 2_000.0)
        _require(child.poll() is None,
                 "contention child exited before midpoint evidence")
        midpoint = _capture_endpoint_evidence(
            args.endpoint_evidence_url,
            stage="midpoint",
            require_valid_delta=True,
        )
        return_code = child.wait(timeout=1200.0)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10.0)
        raise
    after = _capture_endpoint_evidence(
        args.endpoint_evidence_url,
        stage="after",
        require_valid_delta=True,
    )
    result = {
        "schema": ENDPOINT_EVIDENCE_SCHEMA,
        "sampling_policy": "on_demand_block_boundary_and_midpoint",
        "cross_endpoint_clock_subtraction_allowed": False,
        "before": before,
        "midpoint": midpoint,
        "after": after,
    }
    _validate_endpoint_evidence_bundle(result)
    return result


def _validate_endpoint_evidence_bundle(raw: object) -> None:
    _require(isinstance(raw, dict), "endpoint evidence bundle is not an object")
    _require(raw.get("schema") == ENDPOINT_EVIDENCE_SCHEMA,
             "endpoint evidence bundle schema mismatch")
    _require(raw.get("sampling_policy") ==
             "on_demand_block_boundary_and_midpoint",
             "endpoint evidence sampling policy mismatch")
    _require(raw.get("cross_endpoint_clock_subtraction_allowed") is False,
             "cross-endpoint clock subtraction must be forbidden")
    histories: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for expected_stage in ("before", "midpoint", "after"):
        stage = raw.get(expected_stage)
        _require(isinstance(stage, dict), "endpoint evidence stage is missing")
        _require(stage.get("schema") == ENDPOINT_EVIDENCE_SCHEMA,
                 "endpoint evidence stage schema mismatch")
        _require(stage.get("stage") == expected_stage,
                 "endpoint evidence stage label mismatch")
        snapshots = stage.get("snapshots")
        _require(isinstance(snapshots, list) and len(snapshots) == 4,
                 "endpoint evidence stage requires four snapshots")
        for row in snapshots:
            endpoint = row["probe"]["endpoint"]
            cassini = row["probe"]["cassini"]
            histories[endpoint["endpoint_id"]].append((
                endpoint["sequence"], cassini["sequence"]
            ))
    _require(len(histories) == 4, "endpoint evidence identity count differs")
    for history in histories.values():
        _require(len(history) == 3, "endpoint evidence history is incomplete")
        endpoint_sequences = [item[0] for item in history]
        cassini_sequences = [item[1] for item in history]
        _require(endpoint_sequences == sorted(set(endpoint_sequences)),
                 "endpoint evidence sequence did not increase")
        _require(cassini_sequences == sorted(set(cassini_sequences)),
                 "Cassini evidence sequence did not increase")


def _expected_route(metadata: dict[str, object]) -> str:
    arm = metadata["arm"]
    if arm == ForegroundArm.LOCAL.value:
        return LOCAL_ROUTE
    if arm == ForegroundArm.REMOTE.value:
        return REMOTE_ROUTE
    raise ValueError("fixed screen contains a non-fixed route")


def _cold_completion_valid(decision: dict[str, object]) -> bool:
    if (
        decision.get("benchmark_cold_measured") is not True
        or decision.get("decision_cache_residency") != "unknown"
    ):
        return False
    route = decision.get("route")
    if route == LOCAL_ROUTE:
        return (
            decision.get("cache_residency") == "confirmed_miss"
            and decision.get("completion_cache_residency") == "confirmed_miss"
            and decision.get("lmcache_source_cached_tokens") is None
            and decision.get("lmcache_source_full_hit_observed") is None
        )
    if route == REMOTE_ROUTE:
        return (
            decision.get("cache_residency") == "prefill_only"
            and decision.get("completion_cache_residency") == "prefill_only"
            and decision.get("lmcache_source_cached_tokens") == 0
            and decision.get("lmcache_source_full_hit_observed") is False
        )
    return False


def _augment_block(
    raw_path: Path,
    *,
    phase: ContentionState,
    arm: ForegroundArm,
    replicate: int,
    load_fraction: float,
    schedule_sha256: str,
    request_index: dict[str, dict[str, object]],
    endpoint_evidence: dict[str, object],
) -> dict[str, object]:
    _validate_endpoint_evidence_bundle(endpoint_evidence)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    _require(
        raw.get("validation", {}).get("performance_claim_allowed") is True,
        "fixed contention child failed correctness",
    )
    requests = raw.get("requests")
    _require(isinstance(requests, list), "fixed child requests are missing")
    observed_ids = {row.get("request_id") for row in requests}
    _require(observed_ids == set(request_index), "fixed child request IDs differ")
    for row in requests:
        metadata = request_index[row["request_id"]]
        _require(row.get("valid") is True, "fixed child request is invalid")
        router = row.get("router")
        _require(isinstance(router, dict), "fixed child router evidence is missing")
        _require(
            router.get("route") == _expected_route(metadata),
            "fixed child route differs from pinned route",
        )
    decisions = raw.get("router_decisions")
    _require(isinstance(decisions, list), "fixed child decisions are missing")
    decision_index = {row.get("request_id"): row for row in decisions}
    _require(
        len(decision_index) == len(decisions)
        and set(decision_index) == set(request_index),
        "fixed child decision IDs differ",
    )
    for request_id, metadata in request_index.items():
        decision = decision_index[request_id]
        _require(
            decision.get("route") == _expected_route(metadata),
            "fixed decision route differs from pinned route",
        )
        _require(
            _cold_completion_valid(decision),
            "fixed decision lacks an exact cold completion",
        )
    counts = collections.Counter(
        metadata["tenant"] for metadata in request_index.values())
    contract = {
        "schema": BLOCK_SCHEMA,
        "phase": phase.value,
        "foreground_arm": arm.value,
        "replicate": replicate,
        "load_fraction": load_fraction,
        "semantic_schedule_sha256": schedule_sha256,
        "request_counts": dict(sorted(counts.items())),
        "request_index": request_index,
        "same_client_clock_for_foreground_and_background": True,
        "actual_inference_background_only": True,
        "synthetic_network_background": False,
        "cold_unique_first_lmcache_chunks": True,
        "benchmark_cold_measured": True,
        "cold_completion_exact_for_every_request": True,
        "endpoint_evidence_exact": True,
        "endpoint_evidence_stages": ["before", "midpoint", "after"],
        "cross_endpoint_clock_subtraction_allowed": False,
    }
    raw["contention_fixed_contract"] = contract
    raw["endpoint_evidence"] = endpoint_evidence
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _warmup(args: argparse.Namespace, tokenizer, templates) -> int:
    root = args.output.parent / "contention_fixed_preflight"
    root.mkdir()
    workload = root / "preflight.jsonl"
    raw = root / "preflight.raw.json"
    rows = []
    for index, (_arm, request_id) in enumerate(PREFLIGHT_REQUESTS):
        rows.append({
            "request_id": request_id,
            "prompt": _unique_prompt(tokenizer, templates[512], index),
            "max_tokens": 2,
            "arrival_offset_ms": index * 500.0,
        })
    workload.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    subprocess.run(
        _child_command(
            args, workload=workload, output=raw,
            run_id=args.run_id, max_workers=2),
        check=True,
        timeout=1200.0,
    )
    artifact = json.loads(raw.read_text(encoding="utf-8"))
    _require(
        artifact.get("validation", {}).get("performance_claim_allowed") is True,
        "contention preflight failed",
    )
    expected = {
        request_id: (
            LOCAL_ROUTE if arm is ForegroundArm.LOCAL else REMOTE_ROUTE)
        for arm, request_id in PREFLIGHT_REQUESTS
    }
    requests = artifact.get("requests")
    _require(isinstance(requests, list), "contention preflight requests missing")
    observed = {
        row.get("request_id"): row.get("router", {}).get("route")
        for row in requests if isinstance(row.get("router"), dict)
    }
    _require(observed == expected, "contention preflight routes differ")
    decisions = artifact.get("router_decisions")
    _require(isinstance(decisions, list), "contention preflight decisions missing")
    decision_index = {row.get("request_id"): row for row in decisions}
    _require(set(decision_index) == set(expected), "preflight decision IDs differ")
    _require(
        all(_cold_completion_valid(decision) for decision in decisions),
        "contention preflight lacks exact cold completions",
    )
    artifact["contention_preflight"] = {
        "schema": SCHEMA,
        "routes": [arm.value for arm, _request_id in PREFLIGHT_REQUESTS],
        "measured": False,
        "p_only_warm_seed_used": False,
        "cold_completion_exact": True,
    }
    raw.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return 0


def _observation(raw_path: Path) -> FixedArmObservation:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    contract = raw["contention_fixed_contract"]
    foreground = []
    background_offered = 0
    background_completed = 0
    background_errors = 0
    for row in raw["requests"]:
        metadata = contract["request_index"][row["request_id"]]
        if metadata["tenant"] == Tenant.FOREGROUND.value:
            foreground.append(ForegroundObservation(
                pair_key=metadata["pair_key"],
                e2e_ms=(
                    row["stream_end_offset_ns"] - row["dispatch_offset_ns"]
                ) / 1_000_000.0,
                output_sha256=row["output_text_sha256"],
            ))
        else:
            background_offered += 1
            if row.get("valid") is True:
                background_completed += 1
            else:
                background_errors += 1
    return FixedArmObservation(
        phase=ContentionState(contract["phase"]),
        load_fraction=contract["load_fraction"],
        replicate=contract["replicate"],
        arm=ForegroundArm(contract["foreground_arm"]),
        semantic_schedule_sha256=contract["semantic_schedule_sha256"],
        foreground=tuple(sorted(foreground, key=lambda item: item.pair_key)),
        background_offered=background_offered,
        background_completed=background_completed,
        background_errors=background_errors,
    )


def _measured(args: argparse.Namespace, tokenizer, templates) -> int:
    for name, value in (
        ("phase_duration_ms", args.phase_duration_ms),
        ("cooldown_s", args.cooldown_s),
    ):
        _require(math.isfinite(value) and value > 0.0, f"{name} must be positive")
    _require(len(args.endpoint_evidence_url) == 4,
             "measured contention requires four endpoint probes")
    selection = LoadSelection(
        decoder_reference_rate_per_s=args.decoder_reference_rate,
        remote_reference_rate_per_s=args.remote_reference_rate,
        decoder_fraction=args.load_fraction,
        remote_fraction=args.load_fraction,
    )
    root = args.output.parent / "contention_fixed_measured"
    workload_root = args.output.parent / "contention_fixed_measured_workloads"
    root.mkdir()
    workload_root.mkdir()
    artifacts = {}
    contracts = {}
    for sequence, (phase, arm, replicate) in enumerate(BLOCK_ORDER):
        key = f"{sequence:02d}_{phase.name.lower()}_{arm.value}_r{replicate}"
        trial_id = f"{key}-measured"
        schedule = build_schedule(
            states=(phase,),
            selection=selection,
            foreground_arm=arm,
            foreground_rate_per_s=args.request_rate,
            trial_id=trial_id,
            shape=TrafficShape.STABLE,
            phase_duration_ms=args.phase_duration_ms,
            foreground_geometries=FOREGROUND_GEOMETRIES,
        )
        schedule_sha = semantic_schedule_sha256(schedule)
        workload_path = workload_root / f"{key}.jsonl"
        raw_path = root / f"{key}.raw.json"
        request_index = _write_workload(
            workload_path,
            requests=schedule,
            templates=templates,
            tokenizer=tokenizer,
            marker_base=(sequence + 1) * 8192,
        )
        endpoint_evidence = _run_child_with_midpoint_evidence(
            _child_command(
                args, workload=workload_path, output=raw_path,
                run_id=f"{args.run_id}-{key}",
            ),
            args=args,
        )
        contracts[key] = _augment_block(
            raw_path,
            phase=phase,
            arm=arm,
            replicate=replicate,
            load_fraction=args.load_fraction,
            schedule_sha256=schedule_sha,
            request_index=request_index,
            endpoint_evidence=endpoint_evidence,
        )
        artifacts[key] = str(raw_path.resolve())
        if sequence + 1 < len(BLOCK_ORDER):
            time.sleep(args.cooldown_s)

    observations = tuple(_observation(Path(path)) for path in artifacts.values())
    gate = evaluate_crossover(observations, load_fraction=args.load_fraction)
    public = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "phase": "measured",
        "block_order": [
            {
                "phase": phase.value,
                "foreground_arm": arm.value,
                "replicate": replicate,
            }
            for phase, arm, replicate in BLOCK_ORDER
        ],
        "artifacts": artifacts,
        "contracts": contracts,
        "load": {
            "foreground_rate_per_s": args.request_rate,
            "decoder_reference_rate_per_s": args.decoder_reference_rate,
            "remote_reference_rate_per_s": args.remote_reference_rate,
            "load_fraction": args.load_fraction,
            "decoder_offered_rate_per_s": selection.decoder_rate_per_s,
            "remote_offered_rate_per_s": selection.remote_rate_per_s,
            "phase_duration_ms": args.phase_duration_ms,
            "cooldown_s": args.cooldown_s,
        },
        "crossover_gate": gate,
        "controller_tuning_allowed": gate[
            "workload_valid_for_controller_tuning"],
        "source_workload_sha256": hashlib.sha256(
            args.workload.read_bytes()).hexdigest(),
    }
    args.output.write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "crossover_gate": gate,
    }, sort_keys=True))
    return 0


def main() -> int:
    args = _parse()
    _require(args.mode == "tempo_auto", "contention client requires tempo_auto router")
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    _require(args.model.is_absolute(), "model path must be absolute")
    _require(args.max_workers > 0, "max_workers must be positive")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True)
    templates = _load_templates(args.workload, tokenizer)
    if args.run_id.endswith("-warmup"):
        return _warmup(args, tokenizer, templates)
    return _measured(args, tokenizer, templates)


if __name__ == "__main__":
    raise SystemExit(main())
