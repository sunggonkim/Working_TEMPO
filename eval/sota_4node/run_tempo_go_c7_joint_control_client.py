#!/usr/bin/env python3
"""Run one C7 whole-system arm against frozen receiver-incast phases."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from eval.sota_4node import analyze_tempo_go_c7_joint_control as analyzer
from eval.sota_4node import run_tempo_go_c6_decoder_victim_client as decoder
from eval.sota_4node import run_tempo_go_c6_performance_client as c6
from eval.sota_4node import run_tempo_pd_contention_fixed_client as fixed
from tempo.pd_contention_workload import (
    CacheState,
    ContentionState,
    ForegroundArm,
    ScheduledRequest,
    Tenant,
    TokenGeometry,
    semantic_schedule_sha256,
)


SCHEMA = analyzer.BUNDLE_SCHEMA
CONTRACT_SCHEMA = analyzer.CONTRACT_SCHEMA
BLOCK_SCHEMA = "tempo-go-c7-joint-control-block-v1"
CONTRACT_ENV = "TEMPO_GO_C7_JOINT_CONTROL_CONTRACT"
ARM_ENV = "TEMPO_GO_C7_JOINT_CONTROL_ARM"
REMOTE_ROUTE = fixed.REMOTE_ROUTE
LOCAL_ROUTE = fixed.LOCAL_ROUTE
MANAGED_BACKGROUND_ARM = "full_c7_managed_background"
GLOBAL_ARMS = frozenset({
    "full_c7", "app_global_only", MANAGED_BACKGROUND_ARM,
})
PAIRED_BASELINES = frozenset({
    "predictor", "queue_gpu", "network_request_only",
})
FIXED_ARMS = frozenset({
    "fixed_local_d0", "fixed_local_d1",
    "fixed_remote_p0d1", "fixed_remote_p1d0",
})
ARMS = frozenset({*GLOBAL_ARMS, *PAIRED_BASELINES, *FIXED_ARMS})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm() -> str:
    value = os.environ.get(ARM_ENV, "")
    _require(value in ARMS, f"{ARM_ENV} selects an unsupported arm")
    return value


def _decoder_contract(value: dict[str, object]) -> dict[str, object]:
    _require(value.get("schema") == CONTRACT_SCHEMA,
             "C7 joint contract schema differs")
    section = value.get("joint_control")
    _require(isinstance(section, dict), "C7 joint section is missing")
    return section


def configure_node_environment(
    *, repo_root: Path, qualification: dict[str, object], hosts: list[str],
    port_slot: int, elastic_profile: Path,
) -> None:
    """Bind every arm to one multi-decoder topology before process spawn."""
    section = _decoder_contract(qualification)
    arm = _arm()
    if arm in {"full_c7", MANAGED_BACKGROUND_ARM}:
        c6_arm = c6.FULL_ARM
    elif arm in FIXED_ARMS:
        c6_arm = c6.FIXED_ARM
    else:
        c6_arm = arm
    os.environ[c6.ARM_ENV] = c6_arm
    if c6_arm == c6.FIXED_ARM:
        os.environ[c6.FIXED_POLICY_ENV] = (
            "fixed_p0d1" if arm.endswith("d1") else "fixed_p1d0")
    else:
        os.environ.pop(c6.FIXED_POLICY_ENV, None)
    wrapped = {"schema": c6.CONTRACT_SCHEMA, "c6_performance": section}
    c6.configure_node_environment(
        repo_root=repo_root,
        qualification=wrapped,
        hosts=hosts,
        port_slot=port_slot,
        elastic_profile=elastic_profile,
    )
    # Aggressors from both producer nodes must reach either frozen receiver in
    # every arm.  Keep the physical server topology identical; the router
    # retains paired semantics for predictor/queue/network baselines and
    # requires a global commit only for full/app-global requests.
    os.environ["TEMPO_PD_REMOTE_DECODE_PLACEMENT"] = "global_mesh"


def _load_contract(args) -> tuple[dict[str, object], dict[str, object]]:
    path = args.qualification_contract.resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    section = _decoder_contract(value)
    victim = section["victim"]
    expected = {
        "request_rate": victim["offered_rate_per_s"],
        "phase_duration_ms": section["phase_duration_ms"],
        "cooldown_s": section["cooldown_s"],
        "max_workers": section["max_workers"],
    }
    ingress = section.get("ingress", {})
    _require(isinstance(ingress, dict), "C7 ingress section is malformed")
    expected.update({
        "ingress_policy": ingress.get("policy", "shared_pool"),
        "interactive_reserved_workers": ingress.get(
            "interactive_reserved_workers", 0),
    })
    for name, frozen in expected.items():
        _require(getattr(args, name) == frozen,
                 f"runtime differs from frozen C7 joint value: {name}")
    repo_root = Path(__file__).resolve().parents[2]
    source = c6._resolve_artifact(
        repo_root, section["source_workload"], label="source workload")
    _require(args.workload.resolve() == source,
             "C7 joint source workload differs")
    return value, section


def _uniform_offsets(duration_ms: float, rate_per_s: float) -> list[float]:
    if rate_per_s == 0.0:
        return []
    count = int(math.floor(rate_per_s * duration_ms / 1000.0))
    spacing_ms = 1000.0 / rate_per_s
    return [(index + 0.5) * spacing_ms for index in range(count)]


def _aggressor_id(
    *, block: str, target: int, ordinal: int, source: int,
) -> str:
    return (
        "epd-remote-background-cache-miss-measured-endpoint-observed-"
        f"tempo-go-exogenous-fixed-remote-d{target}-c7-joint-{block}-"
        f"aggressor-{ordinal:06d}-{source}"
    )


def _local_aggressor_id(
    *, block: str, ordinal: int, decoder_index: int,
) -> str:
    return (
        "epd-local-background-cache-miss-measured-endpoint-observed-"
        f"c7-joint-{block}-local-aggressor-{ordinal:06d}-{decoder_index}"
    )


def _managed_remote_aggressor_id(
    *, block: str, target: int, ordinal: int, source: int,
) -> str:
    return (
        "epd-tempo-background-cache-miss-measured-endpoint-observed-"
        f"tempo-go-managed-remote-d{target}-c7-joint-{block}-"
        f"aggressor-{ordinal:06d}-{source}"
    )


def _managed_local_aggressor_id(
    *, block: str, ordinal: int, decoder_index: int,
) -> str:
    return (
        "epd-tempo-background-cache-miss-measured-endpoint-observed-"
        f"tempo-go-managed-local-d{decoder_index}-c7-joint-{block}-"
        f"aggressor-{ordinal:06d}-{decoder_index}"
    )


def _victim_identity(block: str, ordinal: int) -> tuple[str, dict[str, object]]:
    arm = _arm()
    base = f"c7-joint-{block}-victim-{ordinal:06d}"
    if arm == "fixed_local_d0":
        return (
            f"epd-local-interactive-cache-miss-measured-{base}-0",
            {"expected_route": LOCAL_ROUTE, "expected_source": 0,
             "expected_decoder": 0},
        )
    if arm == "fixed_local_d1":
        return (
            f"epd-local-interactive-cache-miss-measured-{base}-1",
            {"expected_route": LOCAL_ROUTE, "expected_source": 1,
             "expected_decoder": 1},
        )
    if arm == "fixed_remote_p0d1":
        return (
            "epd-remote-interactive-cache-miss-measured-"
            f"tempo-go-exogenous-fixed-remote-d1-{base}-0",
            {"expected_route": REMOTE_ROUTE, "expected_source": 0,
             "expected_decoder": 1},
        )
    if arm == "fixed_remote_p1d0":
        return (
            "epd-remote-interactive-cache-miss-measured-"
            f"tempo-go-exogenous-fixed-remote-d0-{base}-1",
            {"expected_route": REMOTE_ROUTE, "expected_source": 1,
             "expected_decoder": 0},
        )
    marker = {
        "full_c7": "tempo",
        MANAGED_BACKGROUND_ARM: "tempo",
        "app_global_only": "app_global_only",
        "predictor": "predictor",
        "queue_gpu": "queue_gpu",
        "network_request_only": "network_request_only",
    }[arm]
    return (
        f"epd-{marker}-interactive-cache-miss-measured-{base}-{ordinal % 2}",
        {"expected_route": None, "expected_source": None,
         "expected_decoder": None},
    )


def _materialize_schedule(
    *, spec: dict[str, object], section: dict[str, object],
) -> tuple[tuple[ScheduledRequest, ...], dict[str, dict[str, object]]]:
    name = str(spec["name"])
    hot = spec["hot_decoder_index"]
    _require(hot is None or hot in (0, 1), "hot decoder index differs")
    duration_ms = float(section["phase_duration_ms"])
    # The phase schedule is exogenous workload input, not a controller hint.  The
    # original C7 contract keeps the section-wide rates; activation-matrix
    # contracts may override them per block to separate remote/fabric pressure
    # from local decoder pressure.
    rate = 0.0 if hot is None else float(
        spec.get("remote_aggressor_rate_per_s",
                 section["aggressor"]["rate_per_s"])
    )
    raw_source_indices = spec.get("remote_source_indices", (0, 1))
    _require(
        isinstance(raw_source_indices, (list, tuple))
        and bool(raw_source_indices),
        "remote source index set is empty",
    )
    source_indices = tuple(int(item) for item in raw_source_indices)
    _require(
        all(item in (0, 1) for item in source_indices),
        "remote source index differs",
    )
    state = ContentionState.C0 if hot is None else ContentionState.C2
    victim = section["victim"]
    aggressor = section["aggressor"]
    local_aggressor = section["local_aggressor"]
    victim_geometry = TokenGeometry(
        int(victim["prompt_tokens"]), int(victim["output_tokens"]),
        CacheState(str(victim["cache_state"])),
    )
    aggressor_geometry = TokenGeometry(
        int(aggressor["prompt_tokens"]), int(aggressor["output_tokens"]),
        CacheState(str(aggressor["cache_state"])),
    )
    requests: list[ScheduledRequest] = []
    identities: dict[str, dict[str, object]] = {}
    for ordinal, offset in enumerate(_uniform_offsets(
        duration_ms, float(victim["offered_rate_per_s"]))):
        request_id, expected = _victim_identity(name, ordinal)
        requests.append(ScheduledRequest(
            request_id=request_id,
            phase=state,
            tenant=Tenant.FOREGROUND,
            arm=(
                ForegroundArm.TEMPO if _arm() in GLOBAL_ARMS
                else ForegroundArm.PREDICTOR if _arm() == "predictor"
                else ForegroundArm.QUEUE_ONLY if _arm() == "queue_gpu"
                else ForegroundArm.TEMPO if _arm() == "network_request_only"
                else ForegroundArm.LOCAL if _arm().startswith("fixed_local")
                else ForegroundArm.REMOTE
            ),
            arrival_offset_ms=offset,
            geometry=victim_geometry,
            ordinal=ordinal,
        ))
        identities[request_id] = {
            "role": "victim", "business_tenant": "interactive",
            "hot_decoder_index": hot, "block": name, **expected,
        }
    if hot is not None:
        for ordinal, offset in enumerate(_uniform_offsets(duration_ms, rate)):
            source = source_indices[ordinal % len(source_indices)]
            managed = _arm() == MANAGED_BACKGROUND_ARM
            request_id = (
                _managed_remote_aggressor_id(
                    block=name, target=int(hot), ordinal=ordinal, source=source)
                if managed else _aggressor_id(
                    block=name, target=int(hot), ordinal=ordinal, source=source)
            )
            requests.append(ScheduledRequest(
                request_id=request_id,
                phase=state,
                tenant=Tenant.REMOTE_HOT,
                arm=ForegroundArm.TEMPO if managed else ForegroundArm.REMOTE,
                arrival_offset_ms=offset,
                geometry=aggressor_geometry,
                ordinal=ordinal,
            ))
            identities[request_id] = {
                "role": "aggressor", "business_tenant": "background",
                "source_prefill_index": source,
                "target_decoder_index": int(hot),
                "hot_decoder_index": hot, "block": name,
                "managed_by_tempo_go": managed,
            }
        local_rate = float(spec.get(
            "local_aggressor_rate_per_s", local_aggressor["rate_per_s"]))
        for ordinal, offset in enumerate(_uniform_offsets(duration_ms, local_rate)):
            source = int(hot)
            managed = _arm() == MANAGED_BACKGROUND_ARM
            request_id = (
                _managed_local_aggressor_id(
                    block=name, ordinal=ordinal, decoder_index=source)
                if managed else _local_aggressor_id(
                    block=name, ordinal=ordinal, decoder_index=source)
            )
            requests.append(ScheduledRequest(
                request_id=request_id,
                phase=state,
                tenant=Tenant.DECODER_HOT,
                arm=ForegroundArm.TEMPO if managed else ForegroundArm.LOCAL,
                arrival_offset_ms=offset,
                geometry=TokenGeometry(
                    int(local_aggressor["prompt_tokens"]),
                    int(local_aggressor["output_tokens"]),
                    CacheState(str(local_aggressor["cache_state"])),
                ),
                ordinal=ordinal,
            ))
            identities[request_id] = {
                "role": "local_aggressor", "business_tenant": "background_local_decoder",
                "source_prefill_index": source,
                "target_decoder_index": source,
                "hot_decoder_index": hot, "block": name,
                "managed_by_tempo_go": managed,
            }
    requests.sort(key=lambda row: (
        row.arrival_offset_ms,
        0 if row.tenant is Tenant.FOREGROUND else 1,
        row.ordinal,
    ))
    _require(len(requests) == len(identities),
             "joint request identities are not unique")
    return tuple(requests), identities


def _child_command(args, *, workload: Path, output: Path, run_id: str) -> list[str]:
    command = fixed._child_command(
        args, workload=workload, output=output, run_id=run_id)
    canonical = "eval.sota_4node.run_tempo_pd_elastic_stream_metrics"
    _require(canonical in command, "canonical stream client seam is missing")
    command[command.index(canonical)] = (
        "eval.sota_4node.run_tempo_go_c6_stream_client")
    return command


def _warmup(args, tokenizer, templates) -> int:
    root = args.output.parent / "c7_joint_preflight"
    root.mkdir()
    workload = root / "preflight.jsonl"
    raw = root / "preflight.raw.json"
    rows = []
    for source in (0, 1):
        rows.append({
            "request_id": _aggressor_id(
                block="preflight", target=0, ordinal=source, source=source),
            "prompt": fixed._unique_prompt(
                tokenizer, templates[512], 12288 + source),
            "max_tokens": 2,
            "arrival_offset_ms": source * 500.0,
        })
    workload.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    subprocess.run(
        _child_command(
            args, workload=workload, output=raw,
            run_id=f"{args.run_id}-joint-preflight"),
        check=True, timeout=1200.0,
    )
    artifact = json.loads(raw.read_text(encoding="utf-8"))
    decisions = {row["request_id"]: row for row in artifact["router_decisions"]}
    _require(set(decisions) == {row["request_id"] for row in rows},
             "joint preflight identities differ")
    for source, row in enumerate(rows):
        decision = decisions[row["request_id"]]
        _require(
            decision.get("route") == REMOTE_ROUTE
            and decision.get("frontend_pair_index") == source
            and decision.get("remote_decoder_index") == 0,
            "joint preflight escaped P0/P1-to-D0 fan-in",
        )
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _canonical_edge(route: str, source: int, decoder_index: int) -> str:
    return (
        f"local:d{decoder_index}" if route == LOCAL_ROUTE
        else f"remote:p{source}->d{decoder_index}"
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
    rows = raw["requests"]
    decisions = raw["router_decisions"]
    workload = raw.get("workload", {})
    ingress = section.get("ingress", {})
    _require(isinstance(workload, dict), f"{spec['name']} workload receipt is missing")
    _require(workload.get("ingress_policy") == ingress.get("policy", "shared_pool"),
             f"{spec['name']} ingress policy receipt differs")
    _require(
        workload.get("interactive_reserved_workers") == ingress.get(
            "interactive_reserved_workers", 0),
        f"{spec['name']} interactive reservation receipt differs",
    )
    row_index = {row["request_id"]: row for row in rows}
    decision_index = {row["request_id"]: row for row in decisions}
    _require(
        len(row_index) == len(rows)
        and len(decision_index) == len(decisions)
        and set(row_index) == set(decision_index) == set(request_index),
        f"{spec['name']} request identities differ",
    )
    namespaces = set()
    counts = collections.Counter(
        metadata["role"] for metadata in request_index.values())
    for request_id, metadata in request_index.items():
        row = row_index[request_id]
        decision = decision_index[request_id]
        role = metadata["role"]
        managed_background = _arm() == MANAGED_BACKGROUND_ARM
        is_reject = row.get("terminal_kind") == "global_reject"
        if is_reject:
            _require(
                _arm() in GLOBAL_ARMS
                and (role == "victim" or managed_background),
                     f"{spec['name']} has an ineligible global reject")
            _require(decision.get("tempo_go_global_commit_applied") is False,
                     f"{spec['name']} reject incorrectly has a commit")
            continue
        _require(row.get("valid") is True,
                 f"{spec['name']} has an invalid terminal request")
        route = decision.get("route")
        if row.get("terminal_kind") == "service_lane_failure":
            _require(
                isinstance(row.get("terminal_error_kind"), str)
                and row["terminal_error_kind"].startswith("endpoint_"),
                f"{spec['name']} has an unclassified service-lane failure",
            )
            if row["terminal_error_kind"] == (
                    "endpoint_service_lane_preflight_unavailable"):
                global_decision = decision.get(
                    "frontend_tempo_go_decision")
                reservation_failure = decision.get(
                    "frontend_tempo_go_reservation_failure")
                queue_promotion = decision.get(
                    "frontend_tempo_go_service_lane_queue_promotion")
                _require(
                    route == "bounded_ingress_queue",
                    f"{spec['name']} preflight failure escaped endpoint queue",
                )
                _require(
                    isinstance(global_decision, dict)
                    and global_decision.get("route") in {
                        LOCAL_ROUTE, REMOTE_ROUTE
                    },
                    f"{spec['name']} preflight failure lacks global candidate",
                )
                _require(
                    isinstance(reservation_failure, dict)
                    and reservation_failure.get("route")
                    == global_decision.get("route")
                    and reservation_failure.get("terminal_phase") == "failed",
                    f"{spec['name']} preflight release differs from candidate",
                )
                _require(
                    isinstance(queue_promotion, dict)
                    and queue_promotion.get("status") == "rejected"
                    and queue_promotion.get("route")
                    == global_decision.get("route")
                    and queue_promotion.get("reason")
                    == reservation_failure.get("reason"),
                    f"{spec['name']} preflight rejection lacks promotion receipt",
                )
                _require(
                    decision.get("tempo_go_global_commit_applied") is False,
                    f"{spec['name']} preflight failure has a global commit",
                )
            else:
                _require(
                    route == "bounded_ingress_queue",
                    f"{spec['name']} service-lane failure escaped bounded queue",
                )
            _require(
                isinstance(decision.get("tempo_go_global_commit_applied"), bool),
                f"{spec['name']} service-lane failure lacks commit state",
            )
            continue
        _require(route in {LOCAL_ROUTE, REMOTE_ROUTE},
                 f"{spec['name']} has an invalid route")
        _require(fixed._cold_completion_valid(
            decision, require_explicit_miss=True),
            f"{spec['name']} lacks exact MISS completion")
        namespace = decision.get("cache_namespace")
        _require(isinstance(namespace, str) and namespace
                 and namespace not in namespaces,
                 f"{spec['name']} cache namespace is missing or reused")
        namespaces.add(namespace)
        expected_tokens = (
            int(section["victim"]["output_tokens"])
            if role == "victim"
            else int(section["aggressor"]["output_tokens"])
        )
        _require(len(row.get("output_token_values", [])) == expected_tokens,
                 f"{spec['name']} output token count differs")
        source = int(decision["frontend_pair_index"])
        decoder_index = int(
            decision["local_decoder_index"]
            if route == LOCAL_ROUTE else decision["remote_decoder_index"])
        if managed_background:
            committed_source = decision.get(
                "tempo_go_global_commit_prefill_index")
            committed_decoder = decision.get(
                "tempo_go_global_commit_decoder_index")
            _require(
                decision.get("tempo_go_global_commit_applied") is True,
                f"{spec['name']} managed request lacks a global commit",
            )
            _require(
                isinstance(committed_source, int)
                and committed_source >= 0
                and committed_decoder == decoder_index
                and decision.get("tempo_go_global_commit_edge_id") == _canonical_edge(
                    route, committed_source, decoder_index),
                f"{spec['name']} managed global edge commitment differs",
            )
        elif role in {"aggressor", "local_aggressor"}:
            _require(
                route == (
                    REMOTE_ROUTE if role == "aggressor" else LOCAL_ROUTE)
                and source == metadata["source_prefill_index"]
                and decoder_index == metadata["target_decoder_index"]
                and decision.get("tempo_go_global_commit_applied") is not True,
                f"{spec['name']} aggressor escaped its frozen edge",
            )
        elif _arm() in FIXED_ARMS:
            _require(
                route == metadata["expected_route"]
                and source == metadata["expected_source"]
                and decoder_index == metadata["expected_decoder"],
                f"{spec['name']} fixed victim escaped its edge",
            )
        elif _arm() in PAIRED_BASELINES:
            _require(
                decision.get("tempo_go_global_commit_applied") is not True
                and decoder_index == source,
                f"{spec['name']} paired baseline escaped its pair",
            )
        else:
            _require(_arm() in GLOBAL_ARMS,
                     f"{spec['name']} victim arm differs")
            _require(decision.get("tempo_go_global_commit_applied") is True,
                     f"{spec['name']} global victim lacks a commit")
            committed_source = decision.get(
                "tempo_go_global_commit_prefill_index")
            committed_decoder = decision.get(
                "tempo_go_global_commit_decoder_index")
            committed_edge = decision.get("tempo_go_global_commit_edge_id")
            _require(
                committed_source == source
                and committed_decoder == decoder_index
                and committed_edge == _canonical_edge(
                    route, source, decoder_index),
                f"{spec['name']} global edge commitment differs",
            )
    hot = spec["hot_decoder_index"]
    rate = 0.0 if hot is None else float(
        spec.get("remote_aggressor_rate_per_s",
                 section["aggressor"]["rate_per_s"])
    )
    local_rate = 0.0 if hot is None else float(spec.get(
        "local_aggressor_rate_per_s", section["local_aggressor"]["rate_per_s"]))
    contract = {
        "schema": BLOCK_SCHEMA,
        "name": spec["name"],
        "arm": _arm(),
        "hot_decoder_index": hot,
        "aggressor_rate_per_s": rate,
        "local_aggressor_rate_per_s": local_rate,
        "phase_duration_ms": section["phase_duration_ms"],
        "semantic_schedule_sha256": schedule_sha256,
        "request_counts": dict(sorted(counts.items())),
        "request_index": request_index,
        "same_client_clock_for_victim_and_aggressor": True,
        "actual_two_prefill_receiver_incast": hot is None or True,
        "exogenous_aggressor_not_controller_movable": not managed_background,
        "managed_background_global_admission": managed_background,
        "actual_vllm_lmcache_native": True,
        "explicit_miss_for_every_admitted_request": True,
        "endpoint_evidence_exact": True,
        "phase_or_future_arrival_policy_input": False,
        "ingress_policy": workload["ingress_policy"],
        "interactive_reserved_workers": workload[
            "interactive_reserved_workers"],
    }
    raw["c7_joint_control_contract"] = contract
    raw["endpoint_evidence"] = endpoint_evidence
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _measured(args, tokenizer, templates, section: dict[str, object]) -> int:
    root = args.output.parent / f"c7_joint_{_arm()}_measured"
    workload_root = args.output.parent / f"c7_joint_{_arm()}_workloads"
    root.mkdir()
    workload_root.mkdir()
    artifacts: dict[str, str] = {}
    contracts: dict[str, object] = {}
    for sequence, spec in enumerate(section["blocks"]):
        name = str(spec["name"])
        schedule, identities = _materialize_schedule(
            spec=spec, section=section)
        workload_path = workload_root / f"{name}.jsonl"
        raw_path = root / f"{name}.raw.json"
        request_index = fixed._write_workload(
            workload_path,
            requests=schedule,
            templates=templates,
            tokenizer=tokenizer,
            marker_base=(sequence + 1) * 32768,
        )
        for request_id, identity in identities.items():
            request_index[request_id].update(identity)
        endpoint_evidence = decoder._run_child_with_cadenced_endpoint_evidence(
            _child_command(
                args, workload=workload_path, output=raw_path,
                run_id=f"{args.run_id}-{name}"),
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
        if sequence + 1 < len(section["blocks"]):
            time.sleep(args.cooldown_s)
    bundle: dict[str, object] = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "arm": _arm(),
        "block_order": list(section["blocks"]),
        "artifacts": artifacts,
        "contracts": contracts,
        "qualification_contract": str(args.qualification_contract.resolve()),
        "qualification_contract_sha256": _sha256(args.qualification_contract),
        "source_workload": str(args.workload.resolve()),
        "source_workload_sha256": _sha256(args.workload),
        "performance_claim_allowed": False,
    }
    bundle["analysis"] = analyzer.analyze_arm_bundle(
        bundle, args.qualification_contract)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA, "arm": _arm(),
        "output": str(args.output.resolve()),
        "hot_slo_good": bundle["analysis"]["hot"]["slo_good_victims"],
        "hot_p99_ms": bundle["analysis"]["hot"]["victim"]["e2e_ms"]["p99"],
    }, sort_keys=True))
    return 0


def main() -> int:
    args = decoder._parse()
    _require(args.mode == "tempo_auto", "C7 joint requires tempo_auto")
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
