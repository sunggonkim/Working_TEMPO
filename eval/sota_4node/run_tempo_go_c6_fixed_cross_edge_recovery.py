#!/usr/bin/env python3
"""Qualify fixed alternate P-to-D edge recovery under decoder asymmetry."""

from __future__ import annotations

import collections
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable

from eval.sota_4node import analyze_tempo_go_c6_decoder_victim_abba as decoder_analysis
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


ANALYSIS_SCHEMA = "tempo-go-c6-fixed-cross-edge-recovery-analysis-v1"
BLOCK_SCHEMA = "tempo-go-c6-fixed-cross-edge-recovery-block-v1"
BUNDLE_SCHEMA = decoder_analysis.BUNDLE_SCHEMA


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _routed_schedule(
    schedule,
    *,
    hot_decoder_index: int,
) -> tuple[tuple[object, ...], dict[str, dict[str, object]]]:
    """Pin local aggressors to the hot D and split victims over both cross edges."""
    _require(hot_decoder_index in (0, 1), "hot decoder index differs")
    routed = []
    identities: dict[str, dict[str, object]] = {}
    for request in schedule:
        if request.tenant is Tenant.FOREGROUND:
            source = request.ordinal % 2
            destination = 1 - source
            edge_id = f"remote:p{source}->d{destination}"
            destination_hot = destination == hot_decoder_index
        else:
            _require(
                request.tenant is Tenant.DECODER_HOT,
                "cross-edge schedule contains an unexpected aggressor",
            )
            source = hot_decoder_index
            destination = hot_decoder_index
            edge_id = f"local:d{destination}"
            destination_hot = True
        request_id = (
            f"{request.request_id}-fixed-source{source}-"
            f"destination{destination}-{source}"
        )
        _require(request_id not in identities, "routed request ID was reused")
        routed.append(replace(request, request_id=request_id))
        identities[request_id] = {
            "source_prefill_index": source,
            "expected_decoder_index": destination,
            "edge_id": edge_id,
            "destination_hot": destination_hot,
        }
    _require(bool(routed), "cross-edge schedule is empty")
    return tuple(routed), identities


def _augment_block(
    path: Path,
    *,
    name: str,
    hot_decoder_index: int,
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
        source = metadata["source_prefill_index"]
        destination = metadata["expected_decoder_index"]
        _require(metadata["cache_state"] == CacheState.MISS.value,
                 f"{name} request is not frozen MISS")
        _require(row.get("valid") is True, f"{name} request is invalid")
        _require(
            row.get("router", {}).get("route") == expected_route
            and decision.get("route") == expected_route,
            f"{name} request escaped its fixed route",
        )
        _require(
            fixed._cold_completion_valid(decision, require_explicit_miss=True),
            f"{name} cold evidence differs",
        )
        namespace = decision.get("cache_namespace")
        _require(isinstance(namespace, str) and namespace,
                 f"{name} cache namespace is missing")
        _require(namespace not in cache_namespaces,
                 f"{name} cache namespace was reused")
        cache_namespaces.add(namespace)
        _require(decision.get("frontend_pair_index") == source,
                 f"{name} frontend source pin differs")
        _require(decision.get("local_decoder_index") == source,
                 f"{name} router source identity differs")
        _require(decision.get("remote_decode_placement") == "cross",
                 f"{name} is not using immutable cross proxies")

        values = row.get("output_token_values")
        _require(isinstance(values, list), f"{name} output values are missing")
        if metadata["tenant"] == Tenant.FOREGROUND.value:
            _require(expected_route == fixed.REMOTE_ROUTE,
                     f"{name} victim route differs")
            _require(destination == 1 - source,
                     f"{name} victim edge is not cross-pair")
            _require(decision.get("remote_decoder_index") == destination,
                     f"{name} physical remote decoder differs")
            _require(
                decision.get("remote_decoder_index_source")
                == "fixed_cross_proxy_topology",
                f"{name} physical decoder receipt source differs",
            )
            _require(decision.get("remote_decoder_crossed") is True,
                     f"{name} cross-edge receipt differs")
            _require(len(values) == 128, f"{name} victim output count differs")
        else:
            _require(expected_route == fixed.LOCAL_ROUTE,
                     f"{name} aggressor route differs")
            _require(source == destination == hot_decoder_index,
                     f"{name} aggressor escaped the hot decoder")
            _require(decision.get("remote_decoder_index") is None,
                     f"{name} local aggressor has a remote decoder")
            _require(decision.get("remote_decoder_crossed") is False,
                     f"{name} local aggressor crossed decoders")
            _require(len(values) == 2, f"{name} aggressor output count differs")

    counts = collections.Counter(
        metadata["tenant"] for metadata in request_index.values()
    )
    contract = {
        "schema": BLOCK_SCHEMA,
        "name": name,
        "hot_decoder_index": hot_decoder_index,
        "phase_duration_ms": phase_duration_ms,
        "semantic_schedule_sha256": schedule_sha256,
        "request_counts": dict(sorted(counts.items())),
        "request_index": request_index,
        "same_client_clock_for_both_edges_and_aggressor": True,
        "actual_route_pinned_vllm_aggressor": True,
        "actual_official_lmcache_cross_edges": True,
        "immutable_proxy_topology": {
            "remote:p0->d1": True,
            "remote:p1->d0": True,
        },
        "synthetic_network_background": False,
        "cold_completion_exact_for_every_request": True,
        "decision_explicit_miss_exact_for_every_request": True,
        "cache_namespace_unique_within_phase": True,
        "endpoint_evidence_exact": True,
        "endpoint_evidence_stages": [
            "before", "cassini_bridges", "midpoint", "after",
        ],
        "cross_endpoint_clock_subtraction_allowed": False,
    }
    raw["c6_fixed_cross_edge_contract"] = contract
    raw["endpoint_evidence"] = endpoint_evidence
    path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract


def _metric_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    _require(bool(items), "cross edge has no victim samples")
    return {
        "victim_count": len(items),
        "ttft_ms": decoder_analysis._summary([row["ttft_ms"] for row in items]),
        "decode_completion_ms": decoder_analysis._summary(
            [row["decode_completion_ms"] for row in items]
        ),
        "tpot_ms": decoder_analysis._summary([row["tpot_ms"] for row in items]),
        "e2e_ms": decoder_analysis._summary([row["e2e_ms"] for row in items]),
        "slo_attainment_fraction": (
            sum(row["slo_pass"] for row in items) / len(items)
        ),
    }


def _phase_result(
    name: str,
    path: Path,
    spec: dict[str, Any],
    section: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[object, ...]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    block = raw.get("c6_fixed_cross_edge_contract")
    _require(isinstance(block, dict) and block.get("schema") == BLOCK_SCHEMA,
             f"{name} cross-edge block contract differs")
    _require(block.get("name") == name, f"{name} block name differs")
    _require(block.get("hot_decoder_index") == spec["hot_decoder_index"],
             f"{name} hot decoder differs")
    fixed._validate_endpoint_evidence_bundle(raw.get("endpoint_evidence"))
    request_index = block.get("request_index")
    requests = raw.get("requests")
    _require(isinstance(request_index, dict), f"{name} request index is missing")
    _require(isinstance(requests, list), f"{name} requests are missing")
    rows = {row.get("request_id"): row for row in requests}
    _require(len(rows) == len(requests) and set(rows) == set(request_index),
             f"{name} terminal identities differ")

    by_destination: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    victim_signature = []
    aggressor_count = 0
    for request_id, metadata in request_index.items():
        if metadata["tenant"] == Tenant.FOREGROUND.value:
            destination = int(metadata["expected_decoder_index"])
            by_destination[destination].append(decoder_analysis._victim_metric(
                rows[request_id],
                tpot_slo_ms=float(section["slo"]["tpot_ms"]),
                e2e_slo_ms=float(section["slo"]["e2e_ms"]),
            ))
            victim_signature.append((
                metadata["ordinal"],
                metadata["arrival_offset_ms"],
                metadata["prompt_tokens"],
                metadata["output_tokens"],
                metadata["cache_state"],
                metadata["source_prefill_index"],
                metadata["expected_decoder_index"],
            ))
        else:
            aggressor_count += 1

    expected_per_edge = int(
        section["victim"]["offered_rate_per_edge_per_s"]
        * section["phase_duration_ms"] / 1000.0
    )
    expected_aggressors = int(
        section["aggressor"]["offered_rate_per_s"]
        * section["phase_duration_ms"] / 1000.0
    )
    _require(
        all(len(rows_for_edge) == expected_per_edge
            for rows_for_edge in by_destination.values()),
        f"{name} per-edge victim population differs",
    )
    _require(aggressor_count == expected_aggressors,
             f"{name} aggressor population differs")
    edges = {}
    for destination, rows_for_edge in by_destination.items():
        source = 1 - destination
        edges[f"remote:p{source}->d{destination}"] = {
            "source_prefill_index": source,
            "decoder_index": destination,
            "destination_hot": destination == spec["hot_decoder_index"],
            "metrics": _metric_summary(rows_for_edge),
        }
    return ({
        "name": name,
        "hot_decoder_index": spec["hot_decoder_index"],
        "aggressor_count": aggressor_count,
        "edges": edges,
        "raw": str(path.resolve()),
        "raw_sha256": _sha256(path),
    }, sorted(victim_signature))


def analyze_bundle(
    bundle: dict[str, Any], contract_path: Path,
) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    qualification = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(
        qualification.get("qualification_kind") == "fixed_cross_edge_recovery",
        "cross-edge qualification kind differs",
    )
    section = qualification["fixed_cross_edge_recovery"]
    specs = section.get("phases")
    _require(isinstance(specs, list) and len(specs) == 2,
             "cross-edge phase list differs")
    _require([row.get("hot_decoder_index") for row in specs] == [0, 1],
             "cross-edge hotspot order differs")
    _require(bundle.get("schema") == BUNDLE_SCHEMA,
             "cross-edge bundle schema differs")
    artifacts = bundle.get("artifacts")
    _require(isinstance(artifacts, dict), "cross-edge artifacts are missing")
    phase_pairs = [
        _phase_result(
            spec["name"], Path(artifacts[spec["name"]]), spec, section
        )
        for spec in specs
    ]
    phases = [row[0] for row in phase_pairs]
    signatures = [row[1] for row in phase_pairs]
    _require(signatures[0] == signatures[1],
             "cross-edge victim populations differ between phases")

    effects = []
    winner_edges = []
    for phase in phases:
        hot_decoder = phase["hot_decoder_index"]
        hot_edge = f"remote:p{1 - hot_decoder}->d{hot_decoder}"
        healthy_decoder = 1 - hot_decoder
        healthy_edge = f"remote:p{1 - healthy_decoder}->d{healthy_decoder}"
        hot_metrics = phase["edges"][hot_edge]["metrics"]
        healthy_metrics = phase["edges"][healthy_edge]["metrics"]
        hot_p50 = hot_metrics["decode_completion_ms"]["p50"]
        healthy_p50 = healthy_metrics["decode_completion_ms"]["p50"]
        hot_p99 = hot_metrics["decode_completion_ms"]["p99"]
        healthy_p99 = healthy_metrics["decode_completion_ms"]["p99"]
        winner_edges.append(healthy_edge)
        effects.append({
            "phase": phase["name"],
            "losing_hot_edge": hot_edge,
            "winning_alternate_edge": healthy_edge,
            "winner_p50_margin_fraction": hot_p50 / healthy_p50 - 1.0,
            "alternate_p50_latency_recovery_fraction": 1.0 - healthy_p50 / hot_p50,
            "alternate_p99_latency_recovery_fraction": 1.0 - healthy_p99 / hot_p99,
            "alternate_slo_recovery_percentage_points": 100.0 * (
                healthy_metrics["slo_attainment_fraction"]
                - hot_metrics["slo_attainment_fraction"]
            ),
        })

    thresholds = section["thresholds"]
    margins = [row["winner_p50_margin_fraction"] for row in effects]
    recoveries = [
        row["alternate_p50_latency_recovery_fraction"] for row in effects
    ]
    measured_p95_first_response_ms = max(
        edge["metrics"]["ttft_ms"]["p95"]
        for phase in phases for edge in phase["edges"].values()
    )
    required_phase_duration_ms = max(
        30_000.0, 3.0 * measured_p95_first_response_ms
    )
    phase_duration_ms = float(section["phase_duration_ms"])
    gates = {
        "same_offered_victim_population": True,
        "all_native_requests_correct_cold_and_physically_pinned": True,
        "two_phases_have_opposite_fixed_edge_winners": (
            len(set(winner_edges)) == 2
        ),
        "each_winner_margin_at_least_15pct": min(margins) >= float(
            thresholds["minimum_each_phase_winner_margin_fraction"]
        ),
        "one_overload_winner_margin_at_least_30pct": max(margins) >= float(
            thresholds["minimum_one_overload_winner_margin_fraction"]
        ),
        "alternate_edge_recovers_at_least_20pct_each_phase": min(
            recoveries
        ) >= float(
            thresholds["alternate_completion_capacity_recovery_fraction"]
        ),
        "phase_at_least_30s_and_3xp95_first_response": (
            phase_duration_ms >= required_phase_duration_ms
        ),
        "recovery_gap_at_least_5s": float(section["cooldown_s"]) >= 5.0,
    }
    q2_keys = (
        "two_phases_have_opposite_fixed_edge_winners",
        "each_winner_margin_at_least_15pct",
        "one_overload_winner_margin_at_least_30pct",
        "alternate_edge_recovers_at_least_20pct_each_phase",
    )
    q2_pass = all(gates[name] for name in q2_keys)
    q3_pass = (
        gates["phase_at_least_30s_and_3xp95_first_response"]
        and gates["recovery_gap_at_least_5s"]
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "phases": phases,
        "fixed_edge_effects": effects,
        "aggregate_effect": {
            "median_winner_p50_margin_fraction": statistics.median(margins),
            "median_alternate_p50_latency_recovery_fraction": statistics.median(
                recoveries
            ),
            "measured_p95_first_response_ms": measured_p95_first_response_ms,
            "required_phase_duration_ms": required_phase_duration_ms,
            "frozen_phase_duration_ms": phase_duration_ms,
        },
        "gates": gates,
        "q2_opposite_action_opportunity_pass": q2_pass,
        "q3_service_horizon_pass": q3_pass,
        "controller_performance_run_allowed": False,
        "performance_claim_allowed": False,
    }


def measured(
    args,
    tokenizer,
    templates,
    qualification: dict[str, object],
    *,
    evidence_runner: Callable[..., dict[str, object]],
    bundle_schema: str,
) -> int:
    _require(bundle_schema == BUNDLE_SCHEMA, "C6 client bundle schema differs")
    section = qualification["fixed_cross_edge_recovery"]
    _require(os.environ.get("TEMPO_PD_REMOTE_DECODE_PLACEMENT") == "cross",
             "fixed cross-edge run requires cross proxy placement")
    selection = LoadSelection(
        decoder_reference_rate_per_s=args.decoder_reference_rate,
        remote_reference_rate_per_s=args.remote_reference_rate,
        decoder_fraction=args.load_fraction,
        remote_fraction=args.load_fraction,
    )
    victim = section["victim"]
    victim_geometry = TokenGeometry(
        int(victim["prompt_tokens"]),
        int(victim["output_tokens"]),
        CacheState(victim["cache_state"]),
    )
    root = args.output.parent / "c6_fixed_cross_edge_measured"
    workload_root = args.output.parent / "c6_fixed_cross_edge_workloads"
    root.mkdir()
    workload_root.mkdir()
    artifacts: dict[str, str] = {}
    contracts: dict[str, object] = {}
    for sequence, phase_spec in enumerate(section["phases"]):
        name = phase_spec["name"]
        hot_decoder_index = int(phase_spec["hot_decoder_index"])
        schedule = build_schedule(
            states=(ContentionState.C1,),
            selection=selection,
            foreground_arm=ForegroundArm.REMOTE,
            foreground_rate_per_s=args.request_rate,
            trial_id=f"c6-fixed-cross-{name}",
            shape=TrafficShape.STABLE,
            phase_duration_ms=args.phase_duration_ms,
            foreground_geometries=(victim_geometry,),
        )
        routed_schedule, routing = _routed_schedule(
            schedule, hot_decoder_index=hot_decoder_index
        )
        workload_path = workload_root / f"{name}.jsonl"
        raw_path = root / f"{name}.raw.json"
        request_index = fixed._write_workload(
            workload_path,
            requests=routed_schedule,
            templates=templates,
            tokenizer=tokenizer,
            marker_base=(sequence + 1) * 32768,
        )
        for request_id, identity in routing.items():
            request_index[request_id].update(identity)
        endpoint_evidence = evidence_runner(
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
            hot_decoder_index=hot_decoder_index,
            phase_duration_ms=args.phase_duration_ms,
            schedule_sha256=semantic_schedule_sha256(routed_schedule),
            request_index=request_index,
            endpoint_evidence=endpoint_evidence,
        )
        artifacts[name] = str(raw_path.resolve())
        if sequence + 1 < len(section["phases"]):
            time.sleep(args.cooldown_s)

    bundle = {
        "schema": bundle_schema,
        "run_id": args.run_id,
        "qualification_kind": "fixed_cross_edge_recovery",
        "artifacts": artifacts,
        "contracts": contracts,
        "qualification_contract": str(args.qualification_contract.resolve()),
        "qualification_contract_sha256": _sha256(args.qualification_contract),
        "source_workload": str(args.workload.resolve()),
        "source_workload_sha256": _sha256(args.workload),
        "controller_performance_run_allowed": False,
        "performance_claim_allowed": False,
    }
    bundle["analysis"] = analyze_bundle(bundle, args.qualification_contract)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "schema": bundle_schema,
        "output": str(args.output.resolve()),
        "q2_opposite_action_opportunity_pass": bundle["analysis"][
            "q2_opposite_action_opportunity_pass"
        ],
        "q3_service_horizon_pass": bundle["analysis"][
            "q3_service_horizon_pass"
        ],
    }, sort_keys=True))
    return 0
