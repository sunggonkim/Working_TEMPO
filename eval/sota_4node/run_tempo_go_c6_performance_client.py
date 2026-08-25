#!/usr/bin/env python3
"""Run the same-population fixed-cross versus full C6 native campaign."""

from __future__ import annotations

import collections
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

from eval.sota_4node import run_tempo_go_c6_decoder_victim_client as decoder
from eval.sota_4node import run_tempo_pd_contention_fixed_client as fixed
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf
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
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile
from tempo.pd_global_profile import load_global_profile


SCHEMA = "tempo-go-c6-performance-client-v1"
BLOCK_SCHEMA = "tempo-go-c6-performance-block-v1"
ANALYSIS_SCHEMA = "tempo-go-c6-performance-arm-analysis-v1"
CONTRACT_SCHEMA = "tempo-go-c6-performance-contract-v1"
ARM_ENV = "TEMPO_GO_C6_PERFORMANCE_ARM"
FIXED_POLICY_ENV = "TEMPO_GO_C6_FIXED_POLICY"
FIXED_ARM = "fixed"
FULL_ARM = "full_c6"
PREDICTOR_ARM = "predictor"
QUEUE_GPU_ARM = "queue_gpu"
NETWORK_REQUEST_ONLY_ARM = "network_request_only"
APP_GLOBAL_ONLY_ARM = "app_global_only"
GLOBAL_ARMS = frozenset({FULL_ARM, APP_GLOBAL_ONLY_ARM})
BASELINE_ARMS = frozenset({
    PREDICTOR_ARM,
    QUEUE_GPU_ARM,
    NETWORK_REQUEST_ONLY_ARM,
})
ARMS = frozenset({FIXED_ARM, *GLOBAL_ARMS, *BASELINE_ARMS})
FIXED_POLICIES = frozenset({"fixed_p0d1", "fixed_p1d0"})
LOCAL_ROUTE = fixed.LOCAL_ROUTE
REMOTE_ROUTE = fixed.REMOTE_ROUTE


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm() -> str:
    value = os.environ.get(ARM_ENV, "")
    _require(value in ARMS, f"{ARM_ENV} selects an unsupported C6 arm")
    return value


def _fixed_policy() -> str | None:
    value = os.environ.get(FIXED_POLICY_ENV)
    if _arm() != FIXED_ARM:
        _require(value in {None, ""},
                 f"{FIXED_POLICY_ENV} must be unset for dynamic C6 arms")
        return None
    _require(value in FIXED_POLICIES,
             f"{FIXED_POLICY_ENV} must select one fixed policy")
    return value


def _selected_policy() -> str:
    return _fixed_policy() or _arm()


def _decoder_contract(value: dict[str, object]) -> dict[str, object]:
    _require(value.get("schema") == CONTRACT_SCHEMA,
             "C6 performance contract schema differs")
    section = value.get("c6_performance")
    _require(isinstance(section, dict), "C6 performance section is missing")
    result = dict(section)
    result["remote_decode_placement"] = (
        "cross"
        if _arm() == FIXED_ARM else
        "global_mesh"
        if _arm() in GLOBAL_ARMS else
        "paired"
    )
    return result


def _resolve_artifact(
    repo_root: Path, spec: dict[str, object], *, label: str,
) -> Path:
    raw = spec.get("path")
    digest = spec.get("sha256")
    _require(isinstance(raw, str) and raw, f"{label} path is missing")
    _require(isinstance(digest, str) and len(digest) == 64,
             f"{label} digest is invalid")
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    _require(repo_root == path or repo_root in path.parents,
             f"{label} must be below the repository")
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(_sha256(path) == digest, f"{label} digest differs")
    return path


def configure_node_environment(
    *, repo_root: Path, qualification: dict[str, object], hosts: list[str],
    port_slot: int, elastic_profile: Path,
) -> None:
    """Bind one immutable native topology/profile epoch before process spawn."""
    section = _decoder_contract(qualification)
    _require(len(hosts) == 4 and len(set(hosts)) == 4,
             "C6 performance requires four unique hosts")
    expected_elastic = _resolve_artifact(
        repo_root, section["elastic_profile"], label="elastic profile")
    _require(elastic_profile.resolve() == expected_elastic,
             "lifecycle Elastic profile differs from C6 contract")
    elastic = load_elastic_profile(expected_elastic)

    arm = _arm()
    common = {
        "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": (
            "observe_only" if arm == QUEUE_GPU_ARM else "disabled"),
        "TEMPO_GO_ABLATION": (
            "app_global_only" if arm == APP_GLOBAL_ONLY_ARM else "disabled"),
        "TEMPO_GO_C5_ARM": "tempo" if arm == FULL_ARM else arm,
    }
    os.environ.update(common)
    if arm not in GLOBAL_ARMS:
        os.environ.update({
            "TEMPO_PD_REMOTE_DECODE_PLACEMENT": (
                "cross" if arm == FIXED_ARM else "paired"),
            "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "disabled",
            "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "instant_score_v1",
            "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "0",
        })
        for name in (
            "TEMPO_GO_PROFILE", "TEMPO_GO_PROFILE_SHA256",
            "TEMPO_GO_ELASTIC_PROFILE", "TEMPO_GO_ENDPOINT_PROFILE",
            "TEMPO_PD_ENDPOINT_SERVICE_PROFILE", "TEMPO_GO_TOKENIZER_URL",
            "TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256",
        ):
            os.environ.pop(name, None)
        return

    global_path = _resolve_artifact(
        repo_root, section["global_profile"], label="global profile")
    endpoint_path = _resolve_artifact(
        repo_root, section["endpoint_profile"], label="endpoint profile")
    global_profile = load_global_profile(global_path)
    endpoint = load_endpoint_service_profile(endpoint_path)
    expected_fingerprint = section["global_profile"].get(
        "fingerprint_sha256")
    _require(global_profile.fingerprint_sha256 == expected_fingerprint,
             "global profile fingerprint differs")
    _require(
        global_profile.identity.elastic_profile_fingerprint_sha256
        == elastic.fingerprint_sha256,
        "global/Elastic profile binding differs",
    )
    _require(
        global_profile.identity.endpoint_profile_fingerprint_sha256
        == endpoint.fingerprint_sha256,
        "global/endpoint profile binding differs",
    )
    ports = perf._ports(port_slot, 0)
    os.environ.update({
        "TEMPO_PD_REMOTE_DECODE_PLACEMENT": "global_mesh",
        "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "adaptive",
        "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "semantic_epoch_v1",
        "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "1",
        "TEMPO_GO_ABLATION": (
            "app_global_only" if arm == APP_GLOBAL_ONLY_ARM else "disabled"),
        "TEMPO_GO_PROFILE": str(global_path),
        "TEMPO_GO_PROFILE_SHA256": global_profile.fingerprint_sha256,
        "TEMPO_GO_ELASTIC_PROFILE": str(expected_elastic),
        "TEMPO_GO_ENDPOINT_PROFILE": str(endpoint_path),
        "TEMPO_PD_ENDPOINT_SERVICE_PROFILE": str(endpoint_path),
        "TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256": (
            endpoint.workload_manifest_sha256
        ),
        "TEMPO_GO_TOKENIZER_URL": (
            f"http://{hosts[1]}:{ports['decode_api']}"
        ),
    })


def _load_contract(args) -> tuple[dict[str, object], dict[str, object]]:
    path = args.qualification_contract.resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    section = _decoder_contract(value)
    victim = section["victim"]
    aggressor = section["aggressor"]
    expected = {
        "request_rate": victim["offered_rate_per_s"],
        "decoder_reference_rate": aggressor["reference_rate_per_s"],
        "load_fraction": aggressor["load_fraction"],
        "phase_duration_ms": section["phase_duration_ms"],
        "cooldown_s": section["cooldown_s"],
        "max_workers": section["max_workers"],
    }
    for name, frozen in expected.items():
        _require(getattr(args, name) == frozen,
                 f"runtime differs from frozen C6 value: {name}")
    repo_root = Path(__file__).resolve().parents[2]
    source = _resolve_artifact(
        repo_root, section["source_workload"], label="source workload")
    _require(args.workload.resolve() == source,
             "source workload path differs")
    _require(float(args.remote_reference_rate) == 6.8,
             "remote reference prior differs")
    return value, section


def _logical_phase(hot_decoder_index: int | None) -> str:
    return "normal" if hot_decoder_index is None else f"hot_d{hot_decoder_index}"


def _marker_base(hot_decoder_index: int | None) -> int:
    return {None: 32768, 0: 65536, 1: 98304}[hot_decoder_index]


def _materialize_schedule(
    *, spec: dict[str, object], section: dict[str, object], args,
) -> tuple[tuple[object, ...], dict[str, dict[str, object]]]:
    policy = str(spec["policy"])
    _require(policy == _selected_policy(),
             "C6 block policy differs from selected server epoch")
    hot = spec["hot_decoder_index"]
    _require(hot is None or hot in (0, 1), "hot decoder index differs")
    state = ContentionState.C0 if hot is None else ContentionState.C1
    selection = LoadSelection(
        decoder_reference_rate_per_s=args.decoder_reference_rate,
        remote_reference_rate_per_s=args.remote_reference_rate,
        decoder_fraction=args.load_fraction,
        remote_fraction=args.load_fraction,
    )
    victim = section["victim"]
    geometry = TokenGeometry(
        int(victim["prompt_tokens"]), int(victim["output_tokens"]),
        CacheState(str(victim["cache_state"])),
    )
    foreground_arm = {
        FULL_ARM: ForegroundArm.TEMPO,
        APP_GLOBAL_ONLY_ARM: ForegroundArm.TEMPO,
        PREDICTOR_ARM: ForegroundArm.PREDICTOR,
        QUEUE_GPU_ARM: ForegroundArm.QUEUE_ONLY,
        NETWORK_REQUEST_ONLY_ARM: ForegroundArm.TEMPO,
    }.get(policy, ForegroundArm.REMOTE)
    schedule = build_schedule(
        states=(state,),
        selection=selection,
        foreground_arm=foreground_arm,
        foreground_rate_per_s=args.request_rate,
        trial_id=f"c6-performance-{_logical_phase(hot)}",
        shape=TrafficShape.STABLE,
        phase_duration_ms=args.phase_duration_ms,
        foreground_geometries=(geometry,),
    )
    routed = []
    identities: dict[str, dict[str, object]] = {}
    for request in schedule:
        if request.tenant is Tenant.FOREGROUND:
            business_tenant = (
                "interactive" if request.ordinal % 2 == 0 else "batch"
            )
            if policy in GLOBAL_ARMS:
                request_arm = (
                    "tempo" if policy == FULL_ARM else APP_GLOBAL_ONLY_ARM)
                request_id = (
                    f"epd-{request_arm}-{business_tenant}-cache-miss-measured-"
                    f"c6-performance-{_logical_phase(hot)}-foreground-"
                    f"{request.ordinal:06d}"
                )
                source = destination = edge_id = None
            elif policy in FIXED_POLICIES:
                frozen = section["policies"][policy]
                source = int(frozen["prefill_index"])
                destination = int(frozen["decoder_index"])
                edge_id = str(frozen["edge_id"])
                request_id = (
                    f"epd-remote-{business_tenant}-cache-miss-measured-"
                    f"c6-performance-{_logical_phase(hot)}-foreground-"
                    f"{request.ordinal:06d}-{policy}-source{source}-"
                    f"destination{destination}-{source}"
                )
            else:
                _require(policy in BASELINE_ARMS,
                         "C6 dynamic baseline policy differs")
                request_id = (
                    f"epd-{policy}-{business_tenant}-cache-miss-measured-"
                    f"c6-performance-{_logical_phase(hot)}-foreground-"
                    f"{request.ordinal:06d}"
                )
                source = destination = edge_id = None
        else:
            _require(request.tenant is Tenant.DECODER_HOT and hot in (0, 1),
                     "C6 performance has an unexpected aggressor")
            business_tenant = "background"
            source = destination = int(hot)
            edge_id = f"local:d{hot}"
            request_id = (
                f"epd-local-background-cache-miss-measured-c6-performance-"
                f"{_logical_phase(hot)}-decoder-hot-{request.ordinal:06d}-"
                f"{policy}-{hot}"
            )
        _require(request_id not in identities, "C6 request ID was reused")
        routed.append(replace(request, request_id=request_id))
        identities[request_id] = {
            "policy": policy,
            "logical_phase": _logical_phase(hot),
            "hot_decoder_index": hot,
            "business_tenant": business_tenant,
            "expected_prefill_index": source,
            "expected_decoder_index": destination,
            "expected_edge_id": edge_id,
        }
    return tuple(routed), identities


def _child_command(args, *, workload: Path, output: Path, run_id: str) -> list[str]:
    command = fixed._child_command(
        args, workload=workload, output=output, run_id=run_id)
    if _arm() in GLOBAL_ARMS:
        canonical = "eval.sota_4node.run_tempo_pd_elastic_stream_metrics"
        _require(canonical in command, "canonical stream client seam is missing")
        command[command.index(canonical)] = (
            "eval.sota_4node.run_tempo_go_c6_stream_client")
    return command


def _warmup(args, tokenizer, templates) -> int:
    root = args.output.parent / "c6_performance_preflight"
    root.mkdir()
    workload = root / "preflight.jsonl"
    raw = root / "preflight.raw.json"
    rows = []
    epoch = _selected_policy()
    for index in range(2):
        rows.append({
            "request_id": (
                f"epd-local-background-cache-miss-measured-"
                f"c6-performance-preflight-{epoch}-{index}"
            ),
            "prompt": fixed._unique_prompt(
                tokenizer, templates[512], 4096 + index),
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
            run_id=f"{args.run_id}-local-preflight",
        ),
        check=True,
        timeout=1200.0,
    )
    artifact = json.loads(raw.read_text(encoding="utf-8"))
    _require(artifact.get("validation", {}).get("terminal_contract_valid") is True,
             "C6 local preflight failed")
    _require(
        all(row.get("valid") is True
            and row.get("router", {}).get("route") == LOCAL_ROUTE
            for row in artifact.get("requests", [])),
        "C6 preflight escaped the local route",
    )
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _canonical_edge(route: str, prefill: int, destination: int) -> str:
    return (
        f"local:d{destination}" if route == LOCAL_ROUTE
        else f"remote:p{prefill}->d{destination}"
    )


def _augment_block(
    raw_path: Path, *, spec: dict[str, object], section: dict[str, object],
    schedule_sha256: str, request_index: dict[str, dict[str, object]],
    endpoint_evidence: dict[str, object],
) -> dict[str, object]:
    fixed._validate_endpoint_evidence_bundle(endpoint_evidence)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    validation = raw.get("validation", {})
    _require(validation.get("terminal_contract_valid") is True,
             f"{spec['name']} terminal contract failed")
    _require(validation.get("router_decisions_exact") is True,
             f"{spec['name']} router decisions are incomplete")
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(requests, list) and isinstance(decisions, list),
             f"{spec['name']} terminal rows are missing")
    rows = {row.get("request_id"): row for row in requests}
    receipts = {row.get("request_id"): row for row in decisions}
    _require(
        len(rows) == len(requests) and len(receipts) == len(decisions)
        and set(rows) == set(receipts) == set(request_index),
        f"{spec['name']} request identities differ",
    )
    namespaces = set()
    admitted = rejected = 0
    for request_id, metadata in request_index.items():
        row = rows[request_id]
        receipt = receipts[request_id]
        _require(row.get("valid") is True,
                 f"{spec['name']} has an invalid terminal request")
        is_reject = row.get("terminal_kind") == "global_reject"
        is_foreground = metadata["tenant"] == Tenant.FOREGROUND.value
        if is_reject:
            _require(_arm() in GLOBAL_ARMS and is_foreground,
                     f"{spec['name']} has an ineligible global reject")
            _require(receipt.get("tempo_go_global_commit_applied") is False,
                     f"{spec['name']} reject incorrectly has a commit")
            rejected += 1
            continue
        admitted += 1
        route = row.get("router", {}).get("route")
        _require(route in {LOCAL_ROUTE, REMOTE_ROUTE}
                 and receipt.get("route") == route,
                 f"{spec['name']} admitted route differs")
        _require(fixed._cold_completion_valid(
            receipt, require_explicit_miss=True),
            f"{spec['name']} admitted request lacks exact MISS completion")
        namespace = receipt.get("cache_namespace")
        _require(isinstance(namespace, str) and namespace
                 and namespace not in namespaces,
                 f"{spec['name']} cache namespace is missing or reused")
        namespaces.add(namespace)
        expected_tokens = 128 if is_foreground else 2
        _require(len(row.get("output_token_values", [])) == expected_tokens,
                 f"{spec['name']} output token count differs")

        if is_foreground and _arm() in GLOBAL_ARMS:
            _require(receipt.get("tempo_go_global_commit_applied") is True,
                     f"{spec['name']} full C6 request lacks global commit")
            prefill_index = receipt.get("tempo_go_global_commit_prefill_index")
            decoder_index = receipt.get("tempo_go_global_commit_decoder_index")
            edge_id = receipt.get("tempo_go_global_commit_edge_id")
            _require(prefill_index in (0, 1) and decoder_index in (0, 1),
                     f"{spec['name']} P-by-D identity is missing")
            _require(edge_id == _canonical_edge(
                route, int(prefill_index), int(decoder_index)),
                f"{spec['name']} edge identity is not canonical")
            _require(receipt.get("tempo_go_global_commit_pair_index")
                     == decoder_index,
                     f"{spec['name']} pair compatibility alias differs")
            _require(receipt.get("frontend_pair_index") == prefill_index
                     and receipt.get("local_decoder_index") == prefill_index,
                     f"{spec['name']} selected prefill ingress differs")
            _require(
                receipt.get("tempo_go_global_commit_phase_label_policy_input")
                is False
                and receipt.get(
                    "tempo_go_global_commit_physical_switch_label_policy_input"
                ) is False
                and receipt.get(
                    "tempo_go_global_commit_future_arrivals_policy_input"
                ) is False,
                f"{spec['name']} used an oracle/phase policy input",
            )
            if route == REMOTE_ROUTE:
                _require(receipt.get("remote_decoder_index") == decoder_index,
                         f"{spec['name']} physical decoder differs from commit")
            else:
                _require(prefill_index == decoder_index
                         and receipt.get("remote_decoder_index") is None,
                         f"{spec['name']} local mesh actuation differs")
        elif is_foreground and _arm() == FIXED_ARM:
            source = int(metadata["expected_prefill_index"])
            destination = int(metadata["expected_decoder_index"])
            _require(route == REMOTE_ROUTE
                     and receipt.get("frontend_pair_index") == source
                     and receipt.get("local_decoder_index") == source
                     and receipt.get("remote_decoder_index") == destination,
                     f"{spec['name']} fixed edge escaped its pin")
            _require(receipt.get("remote_decode_placement") == "cross",
                     f"{spec['name']} fixed topology differs")
        elif is_foreground:
            _require(_arm() in BASELINE_ARMS,
                     f"{spec['name']} dynamic baseline arm differs")
            _require(receipt.get("tempo_go_global_commit_applied") is not True,
                     f"{spec['name']} baseline unexpectedly used global commit")
            local_index = receipt.get("local_decoder_index")
            frontend_index = receipt.get("frontend_pair_index")
            _require(local_index in (0, 1) and frontend_index == local_index,
                     f"{spec['name']} baseline pair identity differs")
            if route == REMOTE_ROUTE:
                _require(receipt.get("remote_decoder_index") == local_index,
                         f"{spec['name']} paired remote decoder differs")
            else:
                _require(receipt.get("remote_decoder_index") is None,
                         f"{spec['name']} local baseline has remote decoder")
        else:
            hot = int(metadata["hot_decoder_index"])
            _require(route == LOCAL_ROUTE
                     and receipt.get("local_decoder_index") == hot
                     and receipt.get("remote_decoder_index") is None,
                     f"{spec['name']} aggressor escaped hot decoder")

    counts = collections.Counter(
        metadata["tenant"] for metadata in request_index.values())
    contract = {
        "schema": BLOCK_SCHEMA,
        "name": spec["name"],
        "policy": spec["policy"],
        "logical_phase": _logical_phase(spec["hot_decoder_index"]),
        "hot_decoder_index": spec["hot_decoder_index"],
        "phase_duration_ms": section["phase_duration_ms"],
        "semantic_schedule_sha256": schedule_sha256,
        "request_counts": dict(sorted(counts.items())),
        "request_index": request_index,
        "admitted_count": admitted,
        "global_rejected_count": rejected,
        "same_population_contract": (
            "logical-phase marker, arrival, geometry, tenant ordinal"
        ),
        "actual_vllm_lmcache_native": True,
        "synthetic_network_background": False,
        "explicit_miss_for_every_admitted_request": True,
        "phase_or_oracle_policy_input": False,
        "endpoint_evidence_exact": True,
    }
    raw["c6_performance_contract"] = contract
    raw["endpoint_evidence"] = endpoint_evidence
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _metric(row: dict[str, Any], tenant: str,
            slos: dict[str, dict[str, float]]) -> dict[str, Any] | None:
    if row.get("terminal_kind") == "global_reject":
        return None
    arrivals = row.get("token_arrival_offsets_ns")
    dispatch = row.get("dispatch_offset_ns")
    end = row.get("stream_end_offset_ns")
    _require(isinstance(arrivals, list) and arrivals,
             "completed victim lacks token arrivals")
    _require(isinstance(dispatch, int) and isinstance(end, int),
             "completed victim lacks client timestamps")
    ttft_ms = (arrivals[0] - dispatch) / 1e6
    e2e_ms = (end - dispatch) / 1e6
    tpot_ms = (
        (arrivals[-1] - arrivals[0]) / (len(arrivals) - 1) / 1e6
        if len(arrivals) > 1 else 0.0
    )
    slo = slos[tenant]
    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "slo_pass": e2e_ms <= float(slo["e2e_ms"])
        and tpot_ms <= float(slo["tpot_ms"]),
    }


def _summarize_block(path: Path, contract: dict[str, object],
                     section: dict[str, object]) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["request_id"]: row for row in raw["requests"]}
    receipts = {row["request_id"]: row for row in raw["router_decisions"]}
    metrics = []
    rejects = failures = 0
    routes: collections.Counter[str] = collections.Counter()
    edges: collections.Counter[str] = collections.Counter()
    tenants: collections.Counter[str] = collections.Counter()
    for request_id, metadata in contract["request_index"].items():
        if metadata["tenant"] != Tenant.FOREGROUND.value:
            continue
        row = rows[request_id]
        tenant = metadata["business_tenant"]
        tenants[tenant] += 1
        if row.get("terminal_kind") == "global_reject":
            rejects += 1
            continue
        if row.get("valid") is not True:
            failures += 1
            continue
        metric = _metric(row, tenant, section["tenant_slos"])
        _require(metric is not None, "completed metric is missing")
        metrics.append(metric)
        receipt = receipts[request_id]
        route = receipt["route"]
        routes[route] += 1
        if contract["policy"] in GLOBAL_ARMS:
            edge = receipt["tempo_go_global_commit_edge_id"]
        elif contract["policy"] in FIXED_POLICIES:
            edge = metadata["expected_edge_id"]
        elif route == LOCAL_ROUTE:
            edge = f"local:d{int(receipt['local_decoder_index'])}"
        else:
            edge = (
                f"remote:p{int(receipt['frontend_pair_index'])}"
                f"->d{int(receipt['remote_decoder_index'])}"
            )
        edges[str(edge)] += 1
    offered = len(metrics) + rejects + failures
    duration_s = float(section["phase_duration_ms"]) / 1000.0
    e2e = [row["e2e_ms"] for row in metrics]
    tpot = [row["tpot_ms"] for row in metrics]
    slo_good = sum(row["slo_pass"] for row in metrics)
    return {
        "name": contract["name"],
        "policy": contract["policy"],
        "logical_phase": contract["logical_phase"],
        "hot_decoder_index": contract["hot_decoder_index"],
        "offered_victims": offered,
        "completed_victims": len(metrics),
        "slo_good_victims": slo_good,
        "slo_goodput_per_s": slo_good / duration_s,
        "slo_attainment_fraction_of_offered": slo_good / offered,
        "global_rejects": rejects,
        "failures": failures,
        "e2e_ms": {
            "p50": _quantile(e2e, 0.50),
            "p99": _quantile(e2e, 0.99),
            "mean": statistics.fmean(e2e) if e2e else None,
        },
        "tpot_ms": {
            "p50": _quantile(tpot, 0.50),
            "p99": _quantile(tpot, 0.99),
            "mean": statistics.fmean(tpot) if tpot else None,
        },
        "route_counts": dict(sorted(routes.items())),
        "edge_counts": dict(sorted(edges.items())),
        "business_tenant_offered": dict(sorted(tenants.items())),
        "raw": str(path.resolve()),
        "raw_sha256": _sha256(path),
    }


def _arm_analysis(artifacts: dict[str, str], contracts: dict[str, object],
                  section: dict[str, object]) -> dict[str, object]:
    blocks = [
        _summarize_block(Path(artifacts[name]), contracts[name], section)
        for name in artifacts
    ]
    return {
        "schema": ANALYSIS_SCHEMA,
        "arm": _arm(),
        "fixed_policy": _fixed_policy(),
        "blocks": blocks,
        "terminal_contract_valid_for_every_block": True,
        "same_population_ready_for_campaign_analysis": True,
        "actual_native_transport": True,
    }


def _measured(args, tokenizer, templates, section: dict[str, object]) -> int:
    if _arm() == FIXED_ARM:
        policy = _fixed_policy()
        epochs = section.get("fixed_server_epochs")
        _require(isinstance(epochs, list) and len(epochs) == 2,
                 "fixed server epoch contract differs")
        epoch = next(
            (row for row in epochs if row.get("policy") == policy), None)
        _require(isinstance(epoch, dict)
                 and epoch.get("fresh_vllm_lmcache_epoch") is True,
                 "selected fixed policy lacks a fresh server epoch")
        specs = epoch.get("block_order")
        _require(isinstance(specs, list) and len(specs) == 3,
                 "fixed policy block order differs")
        epoch_name = str(policy)
    elif _arm() == FULL_ARM:
        specs = section["full_block_order"]
        epoch_name = FULL_ARM
    else:
        epochs = section.get("ablation_server_epochs")
        _require(isinstance(epochs, list),
                 "C6 ablation server epochs are missing")
        epoch_name = _arm()
        epoch = next(
            (row for row in epochs if row.get("policy") == epoch_name), None)
        _require(isinstance(epoch, dict)
                 and epoch.get("fresh_vllm_lmcache_epoch") is True,
                 "selected C6 ablation lacks a fresh server epoch")
        specs = epoch.get("block_order")
        _require(isinstance(specs, list) and len(specs) == 3,
                 "C6 ablation block order differs")
    root = args.output.parent / f"c6_performance_{epoch_name}_measured"
    workload_root = args.output.parent / f"c6_performance_{epoch_name}_workloads"
    root.mkdir()
    workload_root.mkdir()
    artifacts: dict[str, str] = {}
    contracts: dict[str, object] = {}
    for sequence, spec in enumerate(specs):
        name = str(spec["name"])
        schedule, identities = _materialize_schedule(
            spec=spec, section=section, args=args)
        workload_path = workload_root / f"{name}.jsonl"
        raw_path = root / f"{name}.raw.json"
        request_index = fixed._write_workload(
            workload_path,
            requests=schedule,
            templates=templates,
            tokenizer=tokenizer,
            marker_base=_marker_base(spec["hot_decoder_index"]),
        )
        for request_id, identity in identities.items():
            request_index[request_id].update(identity)
        endpoint_evidence = decoder._run_child_with_cadenced_endpoint_evidence(
            _child_command(
                args, workload=workload_path, output=raw_path,
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
            spec=spec,
            section=section,
            schedule_sha256=semantic_schedule_sha256(schedule),
            request_index=request_index,
            endpoint_evidence=endpoint_evidence,
        )
        artifacts[name] = str(raw_path.resolve())
        if sequence + 1 < len(specs):
            time.sleep(args.cooldown_s)

    bundle = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "arm": _arm(),
        "fixed_policy": _fixed_policy(),
        "block_order": list(specs),
        "artifacts": artifacts,
        "contracts": contracts,
        "qualification_contract": str(args.qualification_contract.resolve()),
        "qualification_contract_sha256": _sha256(args.qualification_contract),
        "source_workload": str(args.workload.resolve()),
        "source_workload_sha256": _sha256(args.workload),
        "controller_performance_run_allowed": True,
        "performance_claim_allowed": True,
    }
    bundle["analysis"] = _arm_analysis(artifacts, contracts, section)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "arm": _arm(),
        "output": str(args.output.resolve()),
        "blocks": len(artifacts),
    }, sort_keys=True))
    return 0


def main() -> int:
    args = decoder._parse()
    _require(args.mode == "tempo_auto", "C6 performance requires tempo_auto")
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    _require(args.model.is_absolute(), "model path must be absolute")
    _require(args.max_workers > 0, "max-workers must be positive")
    _require(math.isfinite(args.phase_duration_ms)
             and args.phase_duration_ms >= 30_000.0,
             "phase duration must be at least 30 seconds")
    _require(len(args.endpoint_evidence_url) == 4,
             "four endpoint probes are required")
    _qualification, section = _load_contract(args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True)
    templates = fixed._load_templates(args.workload, tokenizer)
    if args.run_id.endswith("-warmup"):
        return _warmup(args, tokenizer, templates)
    return _measured(args, tokenizer, templates, section)


if __name__ == "__main__":
    raise SystemExit(main())
