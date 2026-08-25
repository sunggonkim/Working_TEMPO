#!/usr/bin/env python3
"""Run one server-epoch ABBA screen for an actual vLLM decoder victim."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from eval.sota_4node import analyze_tempo_go_c6_decoder_victim_abba as analyzer
from eval.sota_4node import run_tempo_pd_contention_fixed_client as fixed
from tempo.pd_contention_workload import (
    CacheState,
    ContentionState,
    ForegroundArm,
    LoadSelection,
    Tenant,
    TokenGeometry,
    TrafficShape,
    build_schedule,
    semantic_schedule_sha256,
)


SCHEMA = analyzer.BUNDLE_SCHEMA
BLOCK_SCHEMA = analyzer.BLOCK_SCHEMA
CONTRACT_SCHEMA = analyzer.CONTRACT_SCHEMA
CASSINI_BRIDGE_INTERVAL_S = 5.0


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--qualification-contract", type=Path, required=True)
    parser.add_argument("--default-max-tokens", type=int, default=128)
    parser.add_argument("--max-workers", type=int, required=True)
    parser.add_argument(
        "--ingress-policy",
        choices=("shared_pool", "interactive_reserved"),
        default="shared_pool",
    )
    parser.add_argument("--interactive-reserved-workers", type=int, default=0)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--api-key-env")
    parser.add_argument("--decoder-reference-rate", type=float, required=True)
    parser.add_argument("--remote-reference-rate", type=float, default=6.8)
    parser.add_argument("--load-fraction", type=float, required=True)
    parser.add_argument("--phase-duration-ms", type=float, required=True)
    parser.add_argument("--cooldown-s", type=float, required=True)
    parser.add_argument("--endpoint-evidence-url", action="append", default=[])
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _decoder_contract(value: dict[str, object]) -> dict[str, object]:
    """Return the frozen decoder workload selected by the C6 contract kind."""
    kind = value.get("qualification_kind", "decoder_victim_abba")
    section_name = {
        "decoder_victim_abba": "decoder_victim_abba",
        "fixed_cross_edge_recovery": "fixed_cross_edge_recovery",
    }.get(kind)
    _require(section_name is not None, "unsupported C6 qualification kind")
    section = value.get(section_name)
    _require(isinstance(section, dict), "C6 decoder qualification section is missing")
    return section


def _load_contract(args: argparse.Namespace) -> dict[str, object]:
    path = args.qualification_contract.resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("schema") == CONTRACT_SCHEMA, "qualification contract differs")
    decoder = _decoder_contract(value)
    victim = decoder["victim"]
    aggressor = decoder["aggressor"]
    expected = {
        "request_rate": victim["offered_rate_per_s"],
        "decoder_reference_rate": aggressor["reference_rate_per_s"],
        "load_fraction": aggressor["load_fraction"],
        "phase_duration_ms": decoder["phase_duration_ms"],
        "cooldown_s": decoder["cooldown_s"],
        "max_workers": decoder["max_workers"],
    }
    for name, frozen in expected.items():
        _require(getattr(args, name) == frozen, f"runtime differs from frozen: {name}")
    source = (Path(__file__).resolve().parents[2] / decoder["source_workload"]["path"]).resolve()
    _require(args.workload.resolve() == source, "source workload path differs")
    _require(
        hashlib.sha256(source.read_bytes()).hexdigest()
        == decoder["source_workload"]["sha256"],
        "source workload digest differs",
    )
    _require(float(args.remote_reference_rate) == 6.8, "remote reference prior differs")
    return value


def _augment_block(
    path: Path,
    *,
    name: str,
    aggressor: bool,
    replicate: int,
    phase_duration_ms: float,
    schedule_sha256: str,
    request_index: dict[str, dict[str, object]],
    endpoint_evidence: dict[str, object],
) -> dict[str, object]:
    fixed._validate_endpoint_evidence_bundle(endpoint_evidence)
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require(
        raw.get("validation", {}).get("performance_claim_allowed") is True,
        f"{name} native child correctness failed",
    )
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(requests, list), f"{name} requests are missing")
    _require(isinstance(decisions, list), f"{name} decisions are missing")
    rows = {row.get("request_id"): row for row in requests}
    decision_rows = {row.get("request_id"): row for row in decisions}
    _require(
        len(rows) == len(requests)
        and len(decision_rows) == len(decisions)
        and set(rows) == set(request_index) == set(decision_rows),
        f"{name} request/decision identities differ",
    )
    cache_namespaces = set()
    for request_id, metadata in request_index.items():
        row = rows[request_id]
        decision = decision_rows[request_id]
        expected_route = fixed._expected_route(metadata)
        _require(
            metadata.get("cache_state") == CacheState.MISS.value,
            f"{name} request is not frozen MISS",
        )
        _require(row.get("valid") is True, f"{name} request is invalid")
        _require(
            row.get("router", {}).get("route") == expected_route
            and decision.get("route") == expected_route,
            f"{name} request escaped its fixed route",
        )
        _require(
            fixed._cold_completion_valid(
                decision, require_explicit_miss=True,
            ),
            f"{name} cold evidence differs",
        )
        namespace = decision.get("cache_namespace")
        _require(
            isinstance(namespace, str) and namespace,
            f"{name} cache namespace is missing",
        )
        _require(namespace not in cache_namespaces,
                 f"{name} cache namespace was reused")
        cache_namespaces.add(namespace)
        expected_tokens = 128 if metadata["tenant"] == Tenant.FOREGROUND.value else 2
        _require(
            len(row.get("output_token_values", [])) == expected_tokens,
            f"{name} output count differs",
        )
    counts = collections.Counter(
        metadata["tenant"] for metadata in request_index.values()
    )
    contract = {
        "schema": BLOCK_SCHEMA,
        "name": name,
        "aggressor": aggressor,
        "replicate": replicate,
        "phase_duration_ms": phase_duration_ms,
        "semantic_schedule_sha256": schedule_sha256,
        "request_counts": dict(sorted(counts.items())),
        "request_index": request_index,
        "same_client_clock_for_victim_and_aggressor": True,
        "actual_route_pinned_vllm_aggressor": True,
        "synthetic_network_background": False,
        "cold_completion_exact_for_every_request": True,
        "decision_explicit_miss_exact_for_every_request": True,
        "cache_namespace_unique_within_arm": True,
        "endpoint_evidence_exact": True,
        "endpoint_evidence_stages": [
            "before", "cassini_bridges", "midpoint", "after",
        ],
        "cassini_bridge_interval_s": CASSINI_BRIDGE_INTERVAL_S,
        "cross_endpoint_clock_subtraction_allowed": False,
    }
    raw["c6_decoder_victim_contract"] = contract
    raw["endpoint_evidence"] = endpoint_evidence
    path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _run_child_with_cadenced_endpoint_evidence(
    command: list[str], *, args: argparse.Namespace,
) -> dict[str, object]:
    """Keep Cassini's hardware-counter windows valid during a 60 s arm."""
    before = fixed._capture_endpoint_evidence(
        args.endpoint_evidence_url,
        stage="before",
        require_valid_delta=False,
    )
    child = subprocess.Popen(command)
    started = time.monotonic()
    midpoint_target_s = args.phase_duration_ms / 2_000.0
    hard_deadline = started + args.timeout_s
    bridges: list[dict[str, object]] = []

    def wait_until(target_elapsed_s: float) -> int | None:
        deadline = min(started + target_elapsed_s, hard_deadline)
        timeout_s = max(0.0, deadline - time.monotonic())
        try:
            return child.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if time.monotonic() >= hard_deadline:
                raise
            return None

    def bridge(segment: str) -> None:
        ordinal = len(bridges)
        evidence = fixed._capture_endpoint_evidence(
            args.endpoint_evidence_url,
            stage=f"bridge-{ordinal:03d}",
            require_valid_delta=True,
        )
        bridges.append({
            "ordinal": ordinal,
            "segment": segment,
            "captured_elapsed_s": time.monotonic() - started,
            "evidence": evidence,
        })

    try:
        target_s = CASSINI_BRIDGE_INTERVAL_S
        while target_s < midpoint_target_s:
            return_code = wait_until(target_s)
            _require(return_code is None,
                     "decoder victim child exited before midpoint evidence")
            bridge("before_midpoint")
            target_s += CASSINI_BRIDGE_INTERVAL_S

        return_code = wait_until(midpoint_target_s)
        _require(return_code is None,
                 "decoder victim child exited before midpoint evidence")
        midpoint = fixed._capture_endpoint_evidence(
            args.endpoint_evidence_url,
            stage="midpoint",
            require_valid_delta=True,
        )

        target_s = midpoint_target_s + CASSINI_BRIDGE_INTERVAL_S
        while True:
            return_code = wait_until(target_s)
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
                break
            bridge("after_midpoint")
            target_s += CASSINI_BRIDGE_INTERVAL_S
        after = fixed._capture_endpoint_evidence(
            args.endpoint_evidence_url,
            stage="after",
            require_valid_delta=True,
        )
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10.0)
        raise

    result = {
        "schema": fixed.ENDPOINT_EVIDENCE_SCHEMA,
        "sampling_policy": (
            "on_demand_block_boundary_midpoint_and_cassini_bridges"
        ),
        "cross_endpoint_clock_subtraction_allowed": False,
        "cassini_bridge_interval_s": CASSINI_BRIDGE_INTERVAL_S,
        "midpoint_target_elapsed_s": midpoint_target_s,
        "child_elapsed_s": time.monotonic() - started,
        "before": before,
        "cassini_bridges": bridges,
        "midpoint": midpoint,
        "after": after,
    }
    fixed._validate_endpoint_evidence_bundle(result)
    return result


def _measured(
    args: argparse.Namespace,
    tokenizer,
    templates,
    qualification: dict[str, object],
) -> int:
    decoder = qualification["decoder_victim_abba"]
    selection = LoadSelection(
        decoder_reference_rate_per_s=args.decoder_reference_rate,
        remote_reference_rate_per_s=args.remote_reference_rate,
        decoder_fraction=args.load_fraction,
        remote_fraction=args.load_fraction,
    )
    victim_geometry = TokenGeometry(4094, 128, CacheState.MISS)
    root = args.output.parent / "c6_decoder_victim_measured"
    workload_root = args.output.parent / "c6_decoder_victim_workloads"
    root.mkdir()
    workload_root.mkdir()
    artifacts: dict[str, str] = {}
    contracts: dict[str, object] = {}
    for sequence, arm_spec in enumerate(decoder["arms"]):
        name = arm_spec["name"]
        aggressor = bool(arm_spec["aggressor"])
        state = ContentionState.C1 if aggressor else ContentionState.C0
        schedule = build_schedule(
            states=(state,),
            selection=selection,
            foreground_arm=ForegroundArm.REMOTE,
            foreground_rate_per_s=args.request_rate,
            trial_id=f"c6-decoder-{name}",
            shape=TrafficShape.STABLE,
            phase_duration_ms=args.phase_duration_ms,
            foreground_geometries=(victim_geometry,),
        )
        workload_path = workload_root / f"{name}.jsonl"
        raw_path = root / f"{name}.raw.json"
        request_index = fixed._write_workload(
            workload_path,
            requests=schedule,
            templates=templates,
            tokenizer=tokenizer,
            marker_base=(sequence + 1) * 32768,
        )
        endpoint_evidence = _run_child_with_cadenced_endpoint_evidence(
            fixed._child_command(
                args,
                workload=workload_path,
                output=raw_path,
                run_id=f"{args.run_id}-{name}",
            ),
            args=args,
        )
        evidence_path = root / f"{name}.endpoint-evidence.json"
        evidence_path.write_text(
            json.dumps(endpoint_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        contracts[name] = _augment_block(
            raw_path,
            name=name,
            aggressor=aggressor,
            replicate=int(arm_spec["replicate"]),
            phase_duration_ms=args.phase_duration_ms,
            schedule_sha256=semantic_schedule_sha256(schedule),
            request_index=request_index,
            endpoint_evidence=endpoint_evidence,
        )
        artifacts[name] = str(raw_path.resolve())
        if sequence + 1 < len(decoder["arms"]):
            time.sleep(args.cooldown_s)

    bundle = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "artifacts": artifacts,
        "contracts": contracts,
        "qualification_contract": str(args.qualification_contract.resolve()),
        "qualification_contract_sha256": hashlib.sha256(
            args.qualification_contract.read_bytes()
        ).hexdigest(),
        "source_workload": str(args.workload.resolve()),
        "source_workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        "controller_performance_run_allowed": False,
        "performance_claim_allowed": False,
    }
    bundle["analysis"] = analyzer.analyze_bundle(bundle, args.qualification_contract)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "q1_decoder_output_completion_victim_pass": bundle["analysis"][
            "q1_decoder_output_completion_victim_pass"
        ],
        "q3_service_horizon_pass": bundle["analysis"]["q3_service_horizon_pass"],
    }, sort_keys=True))
    return 0


def main() -> int:
    args = _parse()
    _require(args.mode == "tempo_auto", "decoder victim client requires tempo_auto")
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    _require(args.model.is_absolute(), "model path must be absolute")
    _require(args.max_workers > 0, "max-workers must be positive")
    _require(math.isfinite(args.phase_duration_ms) and args.phase_duration_ms >= 30_000.0,
             "phase duration must be at least 30 seconds")
    _require(len(args.endpoint_evidence_url) == 4, "four endpoint probes are required")
    qualification = _load_contract(args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    templates = fixed._load_templates(args.workload, tokenizer)
    if args.run_id.endswith("-warmup"):
        return fixed._warmup(args, tokenizer, templates)
    if qualification.get("qualification_kind") == "fixed_cross_edge_recovery":
        from eval.sota_4node import run_tempo_go_c6_fixed_cross_edge_recovery

        return run_tempo_go_c6_fixed_cross_edge_recovery.measured(
            args,
            tokenizer,
            templates,
            qualification,
            evidence_runner=_run_child_with_cadenced_endpoint_evidence,
            bundle_schema=SCHEMA,
        )
    return _measured(args, tokenizer, templates, qualification)


if __name__ == "__main__":
    raise SystemExit(main())
