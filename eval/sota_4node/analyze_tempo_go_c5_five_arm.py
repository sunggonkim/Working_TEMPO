#!/usr/bin/env python3
"""Analyze one native TEMPO-GO C5 five-arm discovery directory.

The analyzer is deliberately a receipt checker plus descriptive summary.  It
does not convert one allocation into a performance claim; independent native
validation remains a separate gate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from eval.sota_4node import tempo_go_c5_run_contract as run_contract


ARMS = ("local", "remote", "predictor", "queue_gpu", "tempo")
SEVEN_ARMS = (
    "local", "remote", "predictor", "queue_gpu",
    "network_request_only", "app_global_only", "tempo",
)
SUPPORTED_ARMS = (ARMS, SEVEN_ARMS)
# These arms claim to exercise the TEMPO admission/control path.  A failed
# request in either arm must therefore carry the controller's signed terminal
# receipt; an ordinary HTTP/process failure is not a substitute for it.
_GLOBAL_RECEIPT_ARMS = frozenset({"app_global_only", "tempo"})
_TENANT = re.compile(
    r"^epd-(?:local|remote|predictor|queue_gpu|network_request_only|"
    r"app_global_only|tempo)-"
    r"(latency|interactive|batch|background)-"
)
_ROUTES = {
    "decoder_local_chunked_prefill": "local",
    "official_lmcache_remote_prefill": "remote",
}
_GLOBAL_ROUTE_NAMES = frozenset(_ROUTES)
# The native launcher may execute this analyzer from an immutable source
# snapshot under ``results/.../source`` while its contract remains in the
# repository's results tree.  The wrapper exports the worktree root so
# contract validation uses the same bounded root in both layouts.
_REPO_ROOT = Path(
    os.environ.get("TEMPO_GO_REPO_ROOT", Path(__file__).resolve().parents[2])
).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _tenant(request_id: str) -> str:
    match = _TENANT.match(request_id)
    if match is None:
        raise ValueError(f"native request lacks canonical tenant: {request_id}")
    return match.group(1)


def _phase(request_id: str) -> str:
    match = _TENANT.match(request_id)
    if match is None:
        raise ValueError(f"native request lacks canonical phase: {request_id}")
    remainder = request_id[match.end():]
    return remainder.split("-cache-", 1)[0]


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
    }


def _timing(row: dict[str, Any]) -> dict[str, float] | None:
    if row.get("valid") is not True:
        return None
    dispatch = row.get("dispatch_offset_ns")
    arrivals = row.get("token_arrival_offsets_ns")
    finished = row.get("stream_end_offset_ns")
    if (
        type(dispatch) is not int
        or not isinstance(arrivals, list)
        or not arrivals
        or any(type(value) is not int for value in arrivals)
        or type(finished) is not int
        or arrivals[0] < dispatch
        or finished < dispatch
    ):
        return None
    ttft_ms = (arrivals[0] - dispatch) / 1_000_000.0
    e2e_ms = (finished - dispatch) / 1_000_000.0
    tpot_ms = (
        (arrivals[-1] - arrivals[0]) / 1_000_000.0 / (len(arrivals) - 1)
        if len(arrivals) > 1 else None
    )
    return {"ttft_ms": ttft_ms, "tpot_ms": tpot_ms, "e2e_ms": e2e_ms}


def _terminal_rejected(
    row: dict[str, Any], decision: dict[str, Any] | None,
) -> bool:
    return bool(
        row.get("terminal_kind") == "global_reject"
        or isinstance(decision, dict)
        and decision.get("phase") == "rejected"
        and decision.get("global_decision_kind") == "reject"
    )


def _terminal_failed(
    row: dict[str, Any], decision: dict[str, Any] | None,
) -> bool:
    failure = _global_failure_receipt(decision)
    return bool(
        isinstance(failure, dict)
        and failure.get("schema") in {
            "tempo-go-global-failure-v1",
            "tempo-go-service-lane-reservation-v1",
        }
        and failure.get("terminal_phase") == "failed"
        and (
            row.get("valid") is not True
            or row.get("terminal_kind") == "service_lane_failure"
        )
    )


def _global_failure_receipt(
    decision: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(decision, dict):
        return None
    for key in (
        "frontend_tempo_go_failure",
        "frontend_tempo_go_reservation_failure",
    ):
        value = decision.get(key)
        if isinstance(value, dict):
            return value
    return None


def _aggregate_items(
    items: list[tuple[dict[str, Any], dict[str, Any] | None]],
    *,
    tenant_contract: dict[str, Any],
    client_window_s: float | None,
    total_output_tokens: int | None = None,
) -> dict[str, Any]:
    queue_wait = []
    global_admission_wait = []
    ttft = []
    tpot = []
    e2e = []
    output_tokens = 0
    completed = 0
    rejected = 0
    failed = 0
    service_lane_reservation_failures = 0
    terminal_valid = 0
    slo_good = 0
    for row, decision in items:
        scheduled = _number(row.get("scheduled_dispatch_offset_ns"))
        dispatched = _number(row.get("dispatch_offset_ns"))
        if scheduled is not None and dispatched is not None:
            queue_wait.append(max(0.0, dispatched - scheduled) / 1_000_000.0)
        admission_wait = _number(
            decision.get("frontend_tempo_go_admission_wait_ns")
            if isinstance(decision, dict) else None
        )
        if admission_wait is not None:
            global_admission_wait.append(max(0.0, admission_wait) / 1_000_000.0)
        if _terminal_rejected(row, decision):
            rejected += 1
        elif _terminal_failed(row, decision):
            failed += 1
            failure = _global_failure_receipt(decision)
            if (
                isinstance(failure, dict)
                and failure.get("schema")
                == "tempo-go-service-lane-reservation-v1"
            ):
                service_lane_reservation_failures += 1
        if row.get("valid") is True:
            terminal_valid += 1
        timing = _timing(row)
        if timing is None:
            continue
        completed += 1
        ttft.append(timing["ttft_ms"])
        e2e.append(timing["e2e_ms"])
        if timing["tpot_ms"] is not None:
            tpot.append(timing["tpot_ms"])
        values = row.get("output_token_values")
        count = len(values) if isinstance(values, list) else 0
        output_tokens += count
        tenant = _tenant(str(row["request_id"]))
        contract = tenant_contract.get(tenant, {})
        if (
            timing["ttft_ms"] <= float(contract.get("ttft_slo_ms", math.inf))
            and timing["tpot_ms"] is not None
            and timing["tpot_ms"] <= float(contract.get("tpot_slo_ms", math.inf))
            and timing["e2e_ms"] <= float(contract.get("e2e_slo_ms", math.inf))
        ):
            slo_good += 1
    denominator = client_window_s if client_window_s and client_window_s > 0 else None
    output_share = (
        output_tokens / total_output_tokens
        if total_output_tokens and total_output_tokens > 0 else None
    )
    return {
        "request_count": len(items),
        "terminal_valid_count": terminal_valid,
        "completed_count": completed,
        "rejected_count": rejected,
        "failed_count": failed,
        "service_lane_reservation_failure_count": (
            service_lane_reservation_failures),
        "starvation": bool(items) and completed == 0
        and rejected == 0 and failed == 0,
        "rejection_only_no_completion": bool(items) and completed == 0
        and rejected == len(items),
        "slo_good_count": slo_good,
        "slo_goodput_per_s": slo_good / denominator if denominator else None,
        "request_goodput_per_s": completed / denominator if denominator else None,
        "output_tokens": output_tokens,
        "output_token_goodput_per_s": output_tokens / denominator if denominator else None,
        "output_token_share": output_share,
        "queue_wait_ms": _summary(queue_wait),
        "global_admission_wait_ms": _summary(global_admission_wait),
        "ttft_ms": _summary(ttft),
        "tpot_ms": _summary(tpot),
        "e2e_ms": _summary(e2e),
    }


def _load_manifest(receipt: dict[str, Any]) -> dict[str, Any]:
    path_value = receipt.get("workload_manifest")
    if not isinstance(path_value, str) or not path_value:
        return {}
    path = Path(path_value).resolve()
    if not path.is_file():
        return {}
    expected = receipt.get("workload_manifest_sha256")
    if isinstance(expected, str) and expected != _sha256(path):
        raise ValueError(f"native workload manifest SHA mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"native workload manifest is not an object: {path}")
    return value


def _validate_receipt_contract(
    receipt: dict[str, Any], *, arm: str, workload: Path,
) -> dict[str, str] | None:
    """Validate a frozen contract when a new runner supplied one.

    Historical C5 receipts predate the contract schema and remain analyzable
    as historical discovery evidence.  New receipts must carry all three
    contract identity fields; a partial binding is never accepted.
    """

    names = (
        "run_contract", "run_contract_sha256",
        "run_contract_fingerprint_sha256",
    )
    present = [name for name in names if name in receipt]
    if not present:
        return None
    if len(present) != len(names):
        raise ValueError("native receipt has a partial C5 run-contract binding")
    contract_path = Path(str(receipt["run_contract"])).resolve()
    contract = run_contract.verify_contract(
        contract_path,
        str(receipt["run_contract_sha256"]),
        repo_root=_REPO_ROOT,
        workload_input=workload,
        arm_only=arm,
    )
    if receipt["run_contract_fingerprint_sha256"] != contract[
        "fingerprint_sha256"]:
        raise ValueError("native receipt/run-contract fingerprint differs")
    return {
        "path": str(contract_path),
        "sha256": str(receipt["run_contract_sha256"]),
        "fingerprint_sha256": str(contract["fingerprint_sha256"]),
    }


def _validate_workload_identity(
    receipt: dict[str, Any], raw: dict[str, Any], raw_path: Path,
) -> None:
    """Validate both canonical source and arm-rewritten client workload SHA."""

    source_path_value = receipt.get("workload")
    source_sha = receipt.get("workload_sha256")
    if not (
        isinstance(source_path_value, str)
        and isinstance(source_sha, str)
        and len(source_sha) == 64
    ):
        raise ValueError(f"native canonical workload identity is missing: {raw_path}")
    source_path = Path(source_path_value).resolve()
    if not source_path.is_file() or _sha256(source_path) != source_sha:
        raise ValueError(f"native canonical workload SHA mismatch: {raw_path}")

    raw_workload = raw.get("workload")
    if not isinstance(raw_workload, dict):
        raise ValueError(f"native raw workload identity is missing: {raw_path}")
    raw_path_value = raw_workload.get("explicit_path")
    raw_sha = raw_workload.get("sha256")
    if not (
        isinstance(raw_path_value, str)
        and isinstance(raw_sha, str)
        and len(raw_sha) == 64
    ):
        raise ValueError(f"native raw rewritten workload identity is invalid: {raw_path}")
    rewritten_path = Path(raw_path_value).resolve()
    if not rewritten_path.is_file() or _sha256(rewritten_path) != raw_sha:
        raise ValueError(f"native raw rewritten workload SHA mismatch: {raw_path}")
    receipt_raw_sha = receipt.get("raw_workload_sha256")
    if receipt_raw_sha is not None and receipt_raw_sha != raw_sha:
        raise ValueError(f"native receipt/raw workload SHA mismatch: {raw_path}")


def _analyze_native_arm_failure(
    root: Path, arm: str, failure_path: Path,
) -> dict[str, Any]:
    receipt = json.loads(failure_path.read_text(encoding="utf-8"))
    failure_schema = receipt.get("schema")
    if not (
        failure_schema in {
            "tempo-go-c5-native-arm-failure-v1",
            "tempo-go-c5-native-arm-signal-failure-v1",
        }
        and receipt.get("arm") == arm
        and receipt.get("native_only") is True
        and receipt.get("node_count") == 4
        and receipt.get("gpu_count") == 16
        and receipt.get("transport") == "LMCacheConnectorV1:UCX"
    ):
        raise ValueError(f"native failure receipt is not eligible: {failure_path}")
    workload = Path(str(receipt.get("workload", ""))).resolve()
    if not workload.is_file() or receipt.get("workload_sha256") != _sha256(workload):
        raise ValueError(f"native failure workload SHA mismatch: {failure_path}")
    manifest = Path(str(receipt.get("workload_manifest", ""))).resolve()
    if not manifest.is_file() or receipt.get("workload_manifest_sha256") != _sha256(manifest):
        raise ValueError(f"native failure manifest SHA mismatch: {failure_path}")
    contract_identity = _validate_receipt_contract(
        receipt, arm=arm, workload=workload,
    )
    return {
        "arm": arm,
        "result": str(failure_path.resolve()),
        "raw": None,
        "raw_sha256": None,
        "request_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "route_counts": {},
        "tenant_counts": {},
        "tenant_valid_counts": {},
        "scheduler_observation_modes": {},
        "endpoint_feedback_events": {},
        "terminal_phase_counts": {},
        "queue_gpu_pair_observations": 0,
        "tempo_endpoint_completion_receipts": 0,
        "latency_summary": {
            "ttft_p50_ms": None, "ttft_p99_ms": None,
            "e2e_p50_ms": None, "e2e_p99_ms": None,
        },
        "service_metrics": {
            "global": {"request_count": 0, "completed_count": 0,
                       "rejected_count": 0},
            "by_tenant": {}, "by_phase": {},
            "pair_assignment": {}, "telemetry": {},
            "selected_route_counterfactual": {},
        },
        "raw_validation": {
            "native_arm_failure": receipt.get("failure"),
            "failure_schema": failure_schema,
            "signal": receipt.get("signal"),
            "exit_code": receipt.get("exit_code"),
            "performance_claim_allowed": False,
        },
        "execution_failure": receipt.get("failure"),
        "run_contract": contract_identity,
        "performance_claim_allowed": False,
    }


def _completion_tokens(row: dict[str, Any]) -> int:
    values = row.get("output_token_values")
    if isinstance(values, list):
        return len(values)
    usage = row.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        return max(0, usage["completion_tokens"])
    return 0


def _latencies(requests: list[dict[str, Any]]) -> dict[str, float | int | None]:
    ttft = []
    tpot = []
    e2e = []
    for row in requests:
        timing = _timing(row)
        if timing is None:
            continue
        ttft.append(timing["ttft_ms"])
        e2e.append(timing["e2e_ms"])
        if timing["tpot_ms"] is not None:
            tpot.append(timing["tpot_ms"])
    return {
        "ttft_p50_ms": _percentile(ttft, 50),
        "ttft_p95_ms": _percentile(ttft, 95),
        "ttft_p99_ms": _percentile(ttft, 99),
        "tpot_p50_ms": _percentile(tpot, 50),
        "tpot_p95_ms": _percentile(tpot, 95),
        "tpot_p99_ms": _percentile(tpot, 99),
        "e2e_p50_ms": _percentile(e2e, 50),
        "e2e_p95_ms": _percentile(e2e, 95),
        "e2e_p99_ms": _percentile(e2e, 99),
        "completed_count": len(e2e),
    }


def _workload_group(request_id: str) -> str:
    marker = "-cache-"
    if marker not in request_id:
        return "unknown"
    value = request_id.split(marker, 1)[1]
    return value.split("-measured-", 1)[0].split("-", 1)[0]


def _route_counterfactual_summary(
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    route_counts = Counter()
    selected_prior = []
    alternative_minus_selected = []
    comparable = 0
    for row in decisions:
        route = row.get("route")
        if route not in _ROUTES:
            continue
        route_name = _ROUTES[route]
        route_counts[route_name] += 1
        local = _number(row.get("endpoint_request_local_e2e_prior_ms"))
        remote = _number(row.get("endpoint_request_remote_e2e_prior_ms"))
        if local is None or remote is None:
            continue
        chosen = local if route_name == "local" else remote
        alternative = remote if route_name == "local" else local
        selected_prior.append(chosen)
        alternative_minus_selected.append(alternative - chosen)
        comparable += 1
    return {
        "basis": "decision_time_endpoint_service_prior",
        "measured_same_request_counterfactual": False,
        "comparable_decision_count": comparable,
        "selected_route_counts": dict(sorted(route_counts.items())),
        "selected_prior_ms": _summary(selected_prior),
        "alternative_minus_selected_prior_ms": _summary(
            alternative_minus_selected),
    }


def _global_scheduler_observation_summary(
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate scheduler snapshots carried by TEMPO-GO provenance.

    TEMPO-GO intentionally disables the request-start ``/metrics`` fetch when
    adaptive endpoint feedback is enabled.  Its actual vLLM scheduler
    observation arrives through the allocation-scoped telemetry agent and is
    copied into ``frontend_tempo_go_decision.telemetry_provenance``.  Keep
    that evidence separate from the request-start ablation field so a valid
    TEMPO run is not misclassified as scheduler-blind.
    """
    source_counts = Counter()
    pair_counts = Counter()
    payload_count = 0
    observation_count = 0
    invalid_count = 0
    for decision in decisions:
        payload = decision.get("frontend_tempo_go_decision")
        if not isinstance(payload, dict):
            continue
        payload_count += 1
        provenance = payload.get("telemetry_provenance")
        if not isinstance(provenance, dict):
            invalid_count += 1
            continue
        for pair in ("0", "1"):
            # A reject carries only aggregate ``-1`` provenance and has no
            # selected endpoint pair.  Missing pair entries are therefore
            # expected, not invalid scheduler snapshots.
            if pair not in provenance:
                continue
            endpoint = provenance.get(pair)
            scheduler = (
                endpoint.get("scheduler")
                if isinstance(endpoint, dict) else None
            )
            if not isinstance(scheduler, dict):
                invalid_count += 1
                continue
            source = scheduler.get("source")
            schema = scheduler.get("schema")
            running = scheduler.get("running_requests")
            waiting = scheduler.get("waiting_requests")
            usage = scheduler.get("kv_cache_usage_fraction")
            valid = (
                schema == "tempo-go-vllm-scheduler-snapshot-v1"
                and source == "router_local_vllm_prometheus_observe_only"
                and type(running) is int and running >= 0
                and type(waiting) is int and waiting >= 0
                and isinstance(usage, (int, float))
                and not isinstance(usage, bool)
                and math.isfinite(float(usage))
                and 0.0 <= float(usage) <= 1.0
            )
            if not valid:
                invalid_count += 1
                continue
            source_counts[str(source)] += 1
            pair_counts[pair] += 1
            observation_count += 1
    return {
        "payload_count": payload_count,
        "observation_count": observation_count,
        "invalid_count": invalid_count,
        "source_counts": dict(sorted(source_counts.items())),
        "pair_observation_counts": dict(sorted(pair_counts.items())),
    }


def _cross_layer_observer_provenance_summary(
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Count cross-layer observer envelopes independently of scheduler gates.

    TEMPO-GO may intentionally disable request-start ``/metrics`` fetches, so
    scheduler observation and cross-layer observer provenance are separate
    claims.  A valid envelope must retain its schema, communicator, epoch,
    topology identity, and a signal list; unsupported signals are evidence,
    not an invalid observation.
    """
    payload_count = 0
    observation_count = 0
    invalid_count = 0
    pair_counts = Counter()
    schema_counts = Counter()
    source_epochs = Counter()
    communicators = Counter()
    for decision in decisions:
        payload = decision.get("frontend_tempo_go_decision")
        if not isinstance(payload, dict):
            continue
        payload_count += 1
        provenance = payload.get("telemetry_provenance")
        if not isinstance(provenance, dict):
            invalid_count += 1
            continue
        for pair, endpoint in provenance.items():
            # ``-1`` is the aggregate shared-fabric provenance used when no
            # endpoint pair was selected (for example, a global admission
            # rejection).  It intentionally contains ``groups`` rather than
            # an endpoint scheduler/cross-layer envelope, so it is not an
            # invalid endpoint observation and must not poison the gate.
            if pair == "-1":
                continue
            cross_layer = (
                endpoint.get("cross_layer")
                if isinstance(endpoint, dict) else None
            )
            valid = (
                isinstance(cross_layer, dict)
                and cross_layer.get("schema")
                == "tempo-go-cross-layer-envelope-v1"
                and isinstance(cross_layer.get("communicator_id"), str)
                and bool(cross_layer.get("communicator_id"))
                and isinstance(cross_layer.get("source_epoch"), str)
                and bool(cross_layer.get("source_epoch"))
                and isinstance(
                    cross_layer.get("topology_fingerprint_sha256"), str)
                and len(cross_layer["topology_fingerprint_sha256"]) == 64
                and isinstance(cross_layer.get("signals"), list)
            )
            if not valid:
                invalid_count += 1
                continue
            observation_count += 1
            pair_counts[str(pair)] += 1
            schema_counts[str(cross_layer["schema"])] += 1
            source_epochs[str(cross_layer["source_epoch"])] += 1
            communicators[str(cross_layer["communicator_id"])] += 1
    return {
        "payload_count": payload_count,
        "observation_count": observation_count,
        "invalid_count": invalid_count,
        "pair_observation_counts": dict(sorted(pair_counts.items())),
        "schema_counts": dict(sorted(schema_counts.items())),
        "source_epoch_counts": dict(sorted(source_epochs.items())),
        "communicator_counts": dict(sorted(communicators.items())),
    }


def _service_metrics(
    requests: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    client_window_ns: object,
) -> dict[str, Any]:
    decision_by_id = {
        str(row.get("request_id")): row
        for row in decisions
        if isinstance(row.get("request_id"), str)
    }
    contract = manifest.get("tenant_contract", {})
    if not isinstance(contract, dict):
        contract = {}
    window_s = (
        float(client_window_ns) / 1_000_000_000.0
        if isinstance(client_window_ns, (int, float))
        and not isinstance(client_window_ns, bool)
        and float(client_window_ns) > 0 else None
    )
    items = [
        (row, decision_by_id.get(str(row.get("request_id"))))
        for row in requests
    ]
    total_output = sum(
        len(row.get("output_token_values", []))
        for row in requests
        if row.get("valid") is True
        and isinstance(row.get("output_token_values"), list)
    )
    by_tenant: dict[str, list[tuple[dict[str, Any], dict[str, Any] | None]]] = defaultdict(list)
    by_phase: dict[str, list[tuple[dict[str, Any], dict[str, Any] | None]]] = defaultdict(list)
    for item in items:
        request_id = item[0].get("request_id")
        if isinstance(request_id, str):
            by_tenant[_tenant(request_id)].append(item)
            by_phase[_phase(request_id)].append(item)
    tenant_metrics = {}
    for tenant, values in sorted(by_tenant.items()):
        tenant_metrics[tenant] = _aggregate_items(
            values, tenant_contract=contract, client_window_s=window_s,
            total_output_tokens=total_output,
        )
    phase_metrics = {
        phase: _aggregate_items(
            values, tenant_contract=contract, client_window_s=window_s,
            total_output_tokens=total_output,
        )
        for phase, values in sorted(by_phase.items())
    }

    all_items = _aggregate_items(
        items, tenant_contract=contract, client_window_s=window_s,
        total_output_tokens=total_output,
    )
    duration_s = window_s
    all_latencies = []
    all_ttft = []
    all_tpot = []
    queue_waits = []
    route_counts = Counter()
    pair_counts = Counter()
    phase_counts = Counter()
    group_counts = Counter()
    for request_row, decision in items:
        timing = _timing(request_row)
        if timing is not None:
            all_latencies.append(timing["e2e_ms"])
            all_ttft.append(timing["ttft_ms"])
            if timing["tpot_ms"] is not None:
                all_tpot.append(timing["tpot_ms"])
        scheduled = _number(request_row.get("scheduled_dispatch_offset_ns"))
        dispatched = _number(request_row.get("dispatch_offset_ns"))
        if scheduled is not None and dispatched is not None:
            queue_waits.append(max(0.0, dispatched - scheduled) / 1_000_000.0)
        request_id = request_row.get("request_id")
        if isinstance(request_id, str):
            phase_counts[_phase(request_id)] += 1
            group_counts[_workload_group(request_id)] += 1
        if isinstance(request_row.get("router"), dict):
            route = request_row["router"].get("route")
            if route in _ROUTES:
                route_counts[_ROUTES[route]] += 1
        if isinstance(decision, dict):
            pair = decision.get("frontend_pair_index")
            if isinstance(pair, int):
                pair_counts[str(pair)] += 1

    global_payloads = [
        row.get("frontend_tempo_go_decision")
        for row in decisions
        if isinstance(row.get("frontend_tempo_go_decision"), dict)
    ]
    active_before = Counter()
    active_after = Counter()
    pair_activated_count = 0
    for payload in global_payloads:
        before = payload.get("active_pairs_before")
        after = payload.get("active_pairs_after")
        if isinstance(before, list):
            active_before[tuple(before)] += 1
        if isinstance(after, list):
            active_after[tuple(after)] += 1
        if payload.get("pair_activated") is True:
            pair_activated_count += 1
    scheduler_summary = _global_scheduler_observation_summary(decisions)
    cross_layer_summary = _cross_layer_observer_provenance_summary(decisions)
    telemetry_fetch = [
        float(row["vllm_load_fetch_ms"])
        for row in decisions
        if isinstance(row.get("vllm_load_fetch_ms"), (int, float))
        and math.isfinite(float(row["vllm_load_fetch_ms"]))
        and float(row["vllm_load_fetch_ms"]) >= 0
    ]
    tokenizer = [
        float(row["frontend_tempo_go_tokenizer_ms"])
        for row in decisions
        if isinstance(row.get("frontend_tempo_go_tokenizer_ms"), (int, float))
        and math.isfinite(float(row["frontend_tempo_go_tokenizer_ms"]))
        and float(row["frontend_tempo_go_tokenizer_ms"]) >= 0
    ]
    feedback = [
        float(row["endpoint_feedback_service_stretch"])
        for row in decisions
        if isinstance(row.get("endpoint_feedback_service_stretch"), (int, float))
        and math.isfinite(float(row["endpoint_feedback_service_stretch"]))
        and float(row["endpoint_feedback_service_stretch"]) >= 0
    ]
    return {
        "client_window_s": duration_s,
        "global": all_items,
        "completed_count": all_items["completed_count"],
        "output_tokens": all_items["output_tokens"],
        "request_goodput_per_s": all_items["request_goodput_per_s"],
        "output_token_goodput_per_s": all_items["output_token_goodput_per_s"],
        "latency_summary": _summary(all_latencies),
        "ttft_summary": _summary(all_ttft),
        "tpot_summary": _summary(all_tpot),
        "queue_wait_summary": _summary(queue_waits),
        "route_counts": dict(sorted(route_counts.items())),
        "pair_counts": dict(sorted(pair_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "workload_group_counts": dict(sorted(group_counts.items())),
        "by_tenant": tenant_metrics,
        "by_phase": phase_metrics,
        "pair_activation": {
            "pair_activated_count": pair_activated_count,
            "active_pairs_before": {
                ",".join(map(str, key)): value
                for key, value in sorted(active_before.items())
            },
            "active_pairs_after": {
                ",".join(map(str, key)): value
                for key, value in sorted(active_after.items())
            },
            "observed_pair_count": len(pair_counts),
        },
        "telemetry_overhead": {
            "scheduler_source_counts": scheduler_summary["source_counts"],
            "scheduler_observation_count": scheduler_summary[
                "observation_count"],
            "scheduler_observation_payload_count": scheduler_summary[
                "payload_count"],
            "scheduler_observation_invalid_count": scheduler_summary[
                "invalid_count"],
            "scheduler_pair_observation_counts": scheduler_summary[
                "pair_observation_counts"],
            "cross_layer_observer_provenance": cross_layer_summary,
            "vllm_load_fetch_ms": _summary(telemetry_fetch),
            "tokenizer_ms": _summary(tokenizer),
            "endpoint_feedback_service_stretch": {
                "count": len(feedback),
                "p50": _percentile(feedback, 50),
                "p99": _percentile(feedback, 99),
            },
        },
        "selected_route_counterfactual": _route_counterfactual_summary(
            decisions),
    }


def _analyze_raw_arm(
    root: Path,
    arm: str,
    receipt: dict[str, Any],
    receipt_path: Path,
    *,
    allow_execution_failure: bool = False,
    execution_failure: str | None = None,
) -> dict[str, Any]:
    if receipt.get("arm") != arm:
        raise ValueError(f"native arm receipt mismatch: {receipt_path}")
    if not (
        receipt.get("native_only") is True
        and receipt.get("node_count") == 4
        and receipt.get("gpu_count") == 16
        and receipt.get("transport") == "LMCacheConnectorV1:UCX"
    ):
        raise ValueError(f"native topology receipt is not eligible: {receipt_path}")
    workload_value = receipt.get("workload")
    workload = Path(str(workload_value)).resolve()
    if not workload.is_file() or receipt.get("workload_sha256") != _sha256(workload):
        raise ValueError(f"native canonical workload SHA mismatch: {receipt_path}")
    contract_identity = _validate_receipt_contract(
        receipt, arm=arm, workload=workload,
    )
    raw_path = Path(str(receipt.get("raw", ""))).resolve()
    if not raw_path.is_file():
        raise ValueError(f"native raw artifact is missing: {raw_path}")
    if receipt.get("raw_sha256") != _sha256(raw_path):
        raise ValueError(f"native raw SHA mismatch: {raw_path}")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    validation = raw.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"native raw validation is missing: {raw_path}")
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    if not isinstance(requests, list) or not isinstance(decisions, list):
        raise ValueError(f"native raw request/decision lists are missing: {raw_path}")
    _validate_workload_identity(receipt, raw, raw_path)
    if receipt.get("workload_manifest_sha256") is None:
        raise ValueError(f"native workload manifest binding is missing: {receipt_path}")
    manifest = _load_manifest(receipt)

    route_counts = Counter()
    tenant_counts = Counter()
    tenant_valid = Counter()
    scheduler_modes = Counter()
    endpoint_feedback_events = Counter()
    terminal_phases = Counter()
    global_decision_reasons = Counter()
    rejected_candidate_reasons = Counter()
    failure_receipts = 0
    failure_kinds = Counter()
    failure_scopes = Counter()
    router_execution_failure_receipts = 0
    router_execution_failure_kinds = Counter()
    reservation_failure_receipts = 0
    reservation_failure_kinds = Counter()
    hierarchy_reduction_receipts = 0
    hierarchy_raw_candidates = 0
    hierarchy_forwarded_candidates = 0
    hierarchy_omitted_pairs = 0
    decision_by_id = {row.get("request_id"): row for row in decisions}
    for row in requests:
        request_id = row.get("request_id")
        if not isinstance(request_id, str):
            raise ValueError(f"native request ID is missing: {raw_path}")
        tenant = _tenant(request_id)
        tenant_counts[tenant] += 1
        if row.get("valid") is True:
            tenant_valid[tenant] += 1
        router = row.get("router")
        if isinstance(router, dict):
            route = router.get("route")
            if route not in _ROUTES:
                raise ValueError(f"native route is invalid: {route}")
            route_counts[_ROUTES[route]] += 1
        decision = decision_by_id.get(request_id)
        if isinstance(decision, dict):
            terminal_phases[str(decision.get("phase"))] += 1
            decision_payload = decision.get("frontend_tempo_go_decision")
            if not isinstance(decision_payload, dict):
                decision_payload = decision
            reason = decision_payload.get("reason")
            if isinstance(reason, str):
                global_decision_reasons[reason] += 1
            candidates = decision_payload.get("rejected_candidates")
            if isinstance(candidates, list):
                for rejected in candidates:
                    if isinstance(rejected, dict):
                        reason = rejected.get("reason")
                        if isinstance(reason, str):
                            rejected_candidate_reasons[reason] += 1
            mode = decision.get("vllm_load_decision_mode")
            if mode is not None:
                scheduler_modes[str(mode)] += 1
            failure = decision.get("frontend_tempo_go_failure")
            if isinstance(failure, dict):
                if not (
                    failure.get("schema") == "tempo-go-global-failure-v1"
                    and failure.get("request_id") == request_id
                    and failure.get("terminal_phase") == "failed"
                    and isinstance(failure.get("failure_kind"), str)
                    and isinstance(failure.get("quarantine_scope"), str)
                ):
                    raise ValueError(
                        f"native global failure receipt is invalid: {raw_path}")
                failure_receipts += 1
                failure_kinds[str(failure["failure_kind"])] += 1
                failure_scopes[str(failure["quarantine_scope"])] += 1
            reservation_failure = decision.get(
                "frontend_tempo_go_reservation_failure")
            if isinstance(reservation_failure, dict):
                if not (
                    reservation_failure.get("schema")
                    == "tempo-go-service-lane-reservation-v1"
                    and reservation_failure.get("request_id") == request_id
                    and reservation_failure.get("terminal_phase") == "failed"
                    and isinstance(reservation_failure.get("failure_kind"), str)
                    and reservation_failure.get("pair_index")
                    == decision.get("frontend_pair_index")
                ):
                    raise ValueError(
                        "native service-lane reservation receipt is invalid: "
                        f"{raw_path}")
                reservation_failure_receipts += 1
                reservation_failure_kinds[str(
                    reservation_failure["failure_kind"])] += 1
            hierarchy = decision.get(
                "frontend_tempo_go_hierarchy_reduction")
            if hierarchy is not None:
                if not isinstance(hierarchy, dict):
                    raise ValueError(
                        f"native hierarchy reduction receipt is invalid: {raw_path}")
                receipt = hierarchy.get("receipt")
                fingerprint = hierarchy.get("fingerprint_sha256")
                if not (
                    isinstance(receipt, dict)
                    and receipt.get("schema") == "tempo-go-reduction-receipt-v1"
                    and receipt.get("request_id") == request_id
                    and isinstance(fingerprint, str)
                    and len(fingerprint) == 64
                    and isinstance(receipt.get("raw_candidate_count"), int)
                    and isinstance(receipt.get("forwarded_candidate_count"), int)
                    and isinstance(receipt.get("omitted_pair_count"), int)
                    and receipt["raw_candidate_count"] >= receipt[
                        "forwarded_candidate_count"]
                    and receipt["omitted_pair_count"] >= 0
                ):
                    raise ValueError(
                        f"native hierarchy reduction receipt is invalid: {raw_path}")
                hierarchy_reduction_receipts += 1
                hierarchy_raw_candidates += int(receipt["raw_candidate_count"])
                hierarchy_forwarded_candidates += int(
                    receipt["forwarded_candidate_count"])
                hierarchy_omitted_pairs += int(receipt["omitted_pair_count"])
            event = decision.get("endpoint_feedback_event")
            if event is not None:
                endpoint_feedback_events[str(event)] += 1

    if len(decision_by_id) != len(decisions):
        raise ValueError(f"native router decisions contain duplicate IDs: {raw_path}")
    request_ids = {row.get("request_id") for row in requests}
    if set(decision_by_id) != request_ids:
        raise ValueError(f"native router decisions do not cover requests: {raw_path}")
    if validation.get("router_decisions_exact") is not True and not (
        allow_execution_failure and execution_failure is not None
    ):
        raise ValueError(f"native router decision gate failed: {raw_path}")
    for request_row in requests:
        request_id = request_row["request_id"]
        decision = decision_by_id[request_id]
        if decision.get("phase") == "rejected":
            if not (
                decision.get("tempo_go_rejected") is True
                and decision.get("global_decision_kind") == "reject"
                and request_row.get("terminal_kind") == "global_reject"
                and request_row.get("valid") is True
            ):
                raise ValueError(
                    f"native global reject lacks terminal receipt: {raw_path}")
        elif decision.get("phase") == "failed":
            failure = decision.get("frontend_tempo_go_failure")
            reservation_failure = decision.get(
                "frontend_tempo_go_reservation_failure")
            router_execution_failure = (
                request_row.get("valid") is not True
                and isinstance(decision.get("error"), str)
                and bool(decision.get("error"))
            )
            baseline_execution_failure = (
                arm not in _GLOBAL_RECEIPT_ARMS
                and allow_execution_failure
                and execution_failure is not None
                and router_execution_failure
            )
            # A client-side terminal receipt is a valid measured outcome, not
            # a transport failure.  The v2 stream client deliberately marks a
            # matching service-lane reservation receipt ``valid=True`` after
            # closing it against the router ledger.  Preserve compatibility
            # with older raw artifacts (which used ``valid=False``), while
            # requiring the explicit terminal kind for the new semantics.
            service_lane_receipt_valid = (
                request_row.get("valid") is not True
                or request_row.get("terminal_kind") == "service_lane_failure"
            )
            if not (
                isinstance(failure, dict)
                and failure.get("schema") == "tempo-go-global-failure-v1"
                and failure.get("terminal_phase") == "failed"
                and service_lane_receipt_valid
            ) and not (
                isinstance(reservation_failure, dict)
                and reservation_failure.get("schema")
                == "tempo-go-service-lane-reservation-v1"
                and reservation_failure.get("terminal_phase") == "failed"
                and service_lane_receipt_valid
            ) and not baseline_execution_failure:
                raise ValueError(
                    "native failed request lacks global or service-lane "
                    f"failure receipt: {raw_path}")
            if (
                router_execution_failure
                and arm not in _GLOBAL_RECEIPT_ARMS
                and not isinstance(failure, dict)
                and not isinstance(reservation_failure, dict)
            ):
                router_execution_failure_receipts += 1
                router_execution_failure_kinds[str(
                    decision["error"])] += 1
        elif not (
            decision.get("phase") == "complete"
            and decision.get("error") is None
        ):
            raise ValueError(f"native decision phase is not terminal: {raw_path}")

    queue_gpu_observations = sum(
        1 for row in decisions
        if row.get("frontend_pair_queue_gpu_selection") is True
    )
    tempo_completion_receipts = sum(
        1 for row in decisions
        if row.get("endpoint_feedback_event") in {
            "first_response_chunk", "stream_completion_fallback",
        }
        and row.get("endpoint_feedback_released_ns") is not None
    )
    tempo_reject_receipts = sum(
        row.get("phase") == "rejected"
        and row.get("tempo_go_rejected") is True
        for row in decisions
    )
    service_metrics = _service_metrics(
        requests,
        decisions,
        manifest=manifest,
        client_window_ns=raw.get("run", {}).get("client_window_ns")
        if isinstance(raw.get("run"), dict) else None,
    )
    return {
        "arm": arm,
        "result": str(receipt_path.resolve()),
        "raw": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "request_count": len(requests),
        "valid_count": sum(row.get("valid") is True for row in requests),
        "invalid_count": sum(row.get("valid") is not True for row in requests),
        "route_counts": dict(sorted(route_counts.items())),
        "tenant_counts": dict(sorted(tenant_counts.items())),
        "tenant_valid_counts": dict(sorted(tenant_valid.items())),
        "scheduler_observation_modes": dict(sorted(scheduler_modes.items())),
        "endpoint_feedback_events": dict(sorted(endpoint_feedback_events.items())),
        "terminal_phase_counts": dict(sorted(terminal_phases.items())),
        "global_failure_receipts": failure_receipts,
        "global_failure_kinds": dict(sorted(failure_kinds.items())),
        "global_failure_scopes": dict(sorted(failure_scopes.items())),
        "router_execution_failure_receipts": router_execution_failure_receipts,
        "router_execution_failure_kinds": dict(sorted(
            router_execution_failure_kinds.items())),
        "service_lane_reservation_failure_receipts": (
            reservation_failure_receipts),
        "service_lane_reservation_failure_kinds": dict(sorted(
            reservation_failure_kinds.items())),
        "hierarchy_reduction_receipts": hierarchy_reduction_receipts,
        "hierarchy_raw_candidate_count": hierarchy_raw_candidates,
        "hierarchy_forwarded_candidate_count": hierarchy_forwarded_candidates,
        "hierarchy_omitted_pair_count": hierarchy_omitted_pairs,
        "global_decision_reasons": dict(sorted(global_decision_reasons.items())),
        "rejected_candidate_reasons": dict(
            sorted(rejected_candidate_reasons.items())),
        "queue_gpu_pair_observations": queue_gpu_observations,
        "tempo_endpoint_completion_receipts": tempo_completion_receipts,
        "tempo_global_reject_receipts": tempo_reject_receipts,
        "global_scheduler_observation": service_metrics[
            "telemetry_overhead"],
        "latency_summary": _latencies(requests),
        "service_metrics": service_metrics,
        "raw_validation": validation,
        "execution_failure": execution_failure,
        "run_contract": contract_identity,
        "performance_claim_allowed": False,
    }


def _native_failure_raw_path(receipt: dict[str, Any]) -> Path | None:
    """Locate raw evidence retained beside a native execution-failure receipt."""

    direct = receipt.get("raw")
    if isinstance(direct, str) and direct:
        path = Path(direct).resolve()
        return path if path.is_file() else None
    result_dir = receipt.get("result_dir")
    if not isinstance(result_dir, str) or not result_dir:
        return None
    path = Path(result_dir).resolve() / "tempo_go_c5_discovery" / "raw.json"
    return path if path.is_file() else None


def _analyze_arm(root: Path, arm: str) -> dict[str, Any]:
    result_path = root / arm / "result.json"
    if result_path.is_file():
        receipt = json.loads(result_path.read_text(encoding="utf-8"))
        return _analyze_raw_arm(root, arm, receipt, result_path)

    failure_path = root / arm / "failure.json"
    if not failure_path.is_file():
        raise ValueError(f"native arm result is missing: {result_path}")
    failure_receipt = json.loads(failure_path.read_text(encoding="utf-8"))
    failure_summary = _analyze_native_arm_failure(root, arm, failure_path)
    raw_path = _native_failure_raw_path(failure_receipt)
    if raw_path is None:
        return failure_summary

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_workload = raw.get("workload")
    if not isinstance(raw_workload, dict):
        raise ValueError(f"native raw workload identity is missing: {raw_path}")
    raw_workload_sha = raw_workload.get("sha256")
    if not isinstance(raw_workload_sha, str):
        raise ValueError(f"native raw workload SHA is missing: {raw_path}")
    enriched_receipt = dict(failure_receipt)
    enriched_receipt.update({
        "raw": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "raw_workload_sha256": raw_workload_sha,
    })
    value = _analyze_raw_arm(
        root,
        arm,
        enriched_receipt,
        failure_path,
        allow_execution_failure=True,
        execution_failure=failure_summary["execution_failure"],
    )
    value["failure_artifact"] = str(failure_path.resolve())
    value["failure_artifact_sha256"] = _sha256(failure_path)
    value["raw_validation"] = dict(value["raw_validation"])
    value["raw_validation"].update({
        "native_arm_failure": failure_receipt.get("failure"),
        "failure_schema": failure_receipt.get("schema"),
        "exit_code": failure_receipt.get("exit_code"),
        "execution_failure_raw_backed": True,
        "performance_claim_allowed": False,
    })
    return value


def analyze(result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    if not root.is_dir():
        raise ValueError(f"native five-arm result root is missing: {root}")
    order_path = root / "arm_order.txt"
    order = [line.strip() for line in order_path.read_text(encoding="utf-8").splitlines()
             if line.strip()] if order_path.is_file() else []
    declared = next((allowed for allowed in SUPPORTED_ARMS
                     if set(order) == set(allowed)), None)
    if len(order) == 1 and order[0] in SEVEN_ARMS:
        arm = order[0]
        value = _analyze_arm(root, arm)
        return {
            "schema": "tempo-go-c5-native-single-arm-analysis-v1",
            "result_root": str(root),
            "arm_order": order,
            "arms": {arm: value},
            "gates": {
                "single_arm_receipt": True,
                "frozen_run_contract_valid": value.get("run_contract") is not None,
                "native_4node_16gpu_ucx": True,
                "router_decisions_exact": (
                    value["raw_validation"].get("router_decisions_exact") is True),
                "terminal_contract_valid": (
                    value["raw_validation"].get("terminal_contract_valid") is True),
                "performance_claim_allowed": False,
            },
            "claim_boundary": (
                "native single-arm receipt closure only; no five-arm comparison, "
                "independent validation, or production performance claim",
            ),
        }
    if declared is None:
        raise ValueError("native arm order is not an exact supported permutation")
    arms = {arm: _analyze_arm(root, arm) for arm in declared}
    tempo_telemetry = arms["tempo"].get("service_metrics", {}).get(
        "telemetry_overhead", {})
    request_counts = {
        value["request_count"] for value in arms.values()
        if value.get("execution_failure") is None
    }
    workload_shas = set()
    run_contract_shas = set()
    run_contract_fingerprints = set()
    for value in arms.values():
        receipt = json.loads(Path(value["result"]).read_text(encoding="utf-8"))
        if isinstance(receipt.get("workload_sha256"), str):
            workload_shas.add(receipt["workload_sha256"])
        identity = value.get("run_contract")
        if isinstance(identity, dict):
            if isinstance(identity.get("sha256"), str):
                run_contract_shas.add(identity["sha256"])
            if isinstance(identity.get("fingerprint_sha256"), str):
                run_contract_fingerprints.add(identity["fingerprint_sha256"])
    return {
        "schema": (
            "tempo-go-c5-native-seven-arm-analysis-v1"
            if declared == SEVEN_ARMS
            else "tempo-go-c5-native-five-arm-analysis-v2"
        ),
        "result_root": str(root),
        "arm_order": order,
        "arms": arms,
        "gates": {
            "all_declared_arms_present": True,
            "all_five_arms_present": declared == ARMS,
            "all_seven_arms_present": declared == SEVEN_ARMS,
            "same_request_count": len(request_counts) == 1,
            "same_workload_sha": len(workload_shas) == 1,
            "frozen_run_contract_valid": (
                len(run_contract_shas) == 1
                and len(run_contract_fingerprints) == 1
                and all(value.get("run_contract") is not None
                        for value in arms.values())
            ),
            "queue_gpu_failure_receipted": (
                arms["queue_gpu"].get("execution_failure") is not None
            ),
            "native_4node_16gpu_ucx": True,
            "queue_gpu_has_scheduler_observation": (
                arms["queue_gpu"]["queue_gpu_pair_observations"] > 0
                and arms["queue_gpu"]["scheduler_observation_modes"].get(
                    "observe_only", 0) > 0
            ),
            "tempo_has_endpoint_completion_receipt": (
                arms["tempo"]["tempo_endpoint_completion_receipts"] > 0
            ),
            "tempo_has_global_scheduler_observation": (
                tempo_telemetry.get("scheduler_observation_count", 0) > 0
                and tempo_telemetry.get(
                    "scheduler_observation_invalid_count", 1) == 0
            ),
            "app_global_only_present": "app_global_only" in arms,
            "network_request_only_present": "network_request_only" in arms,
            "performance_claim_allowed": False,
        },
        "claim_boundary": (
            "native four-node descriptive discovery only; no independent "
            "validation or production performance claim",
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite analysis: {args.output}")
    value = analyze(args.result_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
