#!/usr/bin/env python3
"""Replay one explicit C5 trace through the five TEMPO-GO policy arms.

This is a deterministic control-plane/invariant replay.  It does not model a
GPU, LMCache, or a physical network and it never authorizes a performance
claim.  The same arrival trace and frozen calibration-only priors are used for
all arms so that queue ownership, route commitment, scheduler pressure,
endpoint completion residual, pair activation, and tenant accounting can be
checked before a native allocation.  Queued requests receive an explicit
per-request timeout at the same effective budget used by the async admission
coordinator: global cap, tenant cap, and request deadline.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from eval.sota_4node.run_vllm_stream_metrics import WorkItem, load_workload
from eval.sota_4node.validate_tempo_go_manifest import validate_manifest
from tempo.pd_elastic_controller import CacheResidency
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile
from tempo.pd_global_candidates import GlobalCandidateBuilder, PairCacheState
from tempo.pd_global_orchestrator import (
    CrossLayerSignal,
    CrossLayerTelemetry,
    GlobalDecision,
    GlobalDecisionKind,
    GlobalOrchestrator,
    GlobalRequest,
    GlobalRoute,
    PairTelemetry,
    PathHealth,
    ResourceVector,
    global_failure_dict,
    global_failure_fingerprint,
)
from tempo.pd_global_profile import load_global_profile
from eval.sota_4node import tempo_go_c5_run_contract as run_contract


SCHEMA = "tempo-go-global-five-arm-replay-v2"
ARM_NAMES = (
    "always_local",
    "official_always_remote",
    "predictor_only",
    "queue_gpu_only",
    "tempo_go",
)
_TENANT = re.compile(r"^epd-tempo-(latency|interactive|batch|background)-")
_STREAM_PHASE = re.compile(
    r"^epd-tempo-(?:latency|interactive|batch|background)-"
    r"(?P<phase>[^-]+(?:_[^-]+)*)-r\d+-"
)
_SCHEDULER_SCHEMA = "tempo-go-vllm-scheduler-snapshot-v1"
_SCHEDULER_SOURCE = "router_local_vllm_prometheus_observe_only"
_COMPLETION_SCHEMA = "tempo-go-endpoint-completion-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _tenant(request_id: str) -> str:
    match = _TENANT.match(request_id)
    if match is None:
        raise ValueError(f"request ID has no canonical tenant: {request_id}")
    return match.group(1)


def _phase_for(index: int, manifest: dict[str, Any]) -> str:
    for phase in manifest["phases"]:
        if int(phase["row_start"]) <= index < int(phase["row_end"]):
            return str(phase["name"])
    raise ValueError(f"workload row has no phase: {index}")


def _cache_evidence(item: WorkItem) -> tuple[CacheResidency, str]:
    """Resolve externally prepared cache evidence, not a policy phase label."""

    # The manifest builder creates the P_ONLY stream from a separately seeded
    # source pool.  This mapping is replay metadata; the phase name is never
    # passed to GlobalCandidateBuilder or GlobalOrchestrator.  Use the
    # explicit cache contract, not a stream/phase substring, so a future
    # workload cannot accidentally turn an unrelated row into P_ONLY.
    if "-cache-p-only-measured-" in item.request_id:
        return CacheResidency.P_ONLY, "completed_frontend_affinity_evidence"
    if "-cache-miss-measured-" in item.request_id:
        return CacheResidency.MISS, "explicit_cache_reset_miss"
    raise ValueError(
        "replay request lacks an explicit measured cache contract: "
        f"{item.request_id}")


def _filtered_request(request: GlobalRequest, arm: str) -> GlobalRequest:
    candidates = request.candidates
    if arm == "always_local":
        candidates = tuple(item for item in candidates if item.route is GlobalRoute.LOCAL)
    elif arm == "official_always_remote":
        candidates = tuple(item for item in candidates if item.route is GlobalRoute.REMOTE)
    elif arm == "predictor_only":
        if not candidates:
            raise ValueError("predictor arm received no candidates")
        selected = min(
            candidates,
            key=lambda item: (
                item.predicted_e2e_ms + item.uncertainty_ms,
                item.route is GlobalRoute.REMOTE,
                item.pair_index,
            ),
        )
        candidates = (selected,)
    if not candidates:
        raise ValueError(f"{arm} has no legal route candidates for {request.request_id}")
    return GlobalRequest(
        request_id=request.request_id,
        tenant_id=request.tenant_id,
        arrival_ns=request.arrival_ns,
        deadline_ns=request.deadline_ns,
        candidates=tuple(candidates),
    )


@dataclass(frozen=True)
class ReplaySpec:
    index: int
    item: WorkItem
    prompt_tokens: int
    phase: str
    residency: CacheResidency
    residency_source: str
    request: GlobalRequest
    baseline_request: GlobalRequest


@dataclass
class _Scheduled:
    request_id: str
    pair_index: int
    route: GlobalRoute
    admitted_ns: int
    first_response_ns: int
    eof_ns: int
    predicted_ttft_ms: float
    predicted_e2e_ms: float


class _ArmReplay:
    def __init__(
        self,
        *,
        arm: str,
        specs: list[ReplaySpec],
        profile,
        include_completion_feedback: bool,
        include_cross_layer_pressure: bool = False,
        failure_index: int | None = None,
        telemetry_failure_index: int | None = None,
    ) -> None:
        self.arm = arm
        self.specs = specs
        self.profile = profile
        self.include_completion_feedback = include_completion_feedback
        self.include_cross_layer_pressure = include_cross_layer_pressure
        if failure_index is not None and failure_index < 0:
            raise ValueError("failure_index must be non-negative")
        self.failure_index = failure_index
        if (
            telemetry_failure_index is not None
            and telemetry_failure_index < 0
        ):
            raise ValueError("telemetry_failure_index must be non-negative")
        self.telemetry_failure_index = telemetry_failure_index
        self._telemetry_failure_active = False
        self._telemetry_failure_target_request_id: str | None = None
        self.orchestrator = GlobalOrchestrator(profile.orchestrator_config())
        self._requests: dict[str, GlobalRequest] = {
            spec.item.request_id: spec.request for spec in specs
        }
        self.records: dict[str, dict[str, Any]] = {
            spec.item.request_id: {
                "index": spec.index,
                "tenant_id": spec.item.request_id.split("-")[2],
                "phase": spec.phase,
                "arrival_ns": spec.item.arrival_offset_ns + 1_000_000,
                "max_tokens": spec.item.max_tokens,
                "status": "not_arrived",
                "queue_decisions": 0,
                "admission_timeout_scheduled": False,
                "decision_reasons": [],
                # Keep the rejected-candidate receipt at the request boundary
                # so a whole-system replay can explain why an expired waiter
                # was not leased into the native endpoint queue.  Aggregate
                # counts alone cannot distinguish endpoint capacity, fabric,
                # cache, fairness, or deadline control.
                "last_decision_kind": None,
                "last_queue_lease": False,
                "last_rejected_candidates": [],
                "pair": None,
                "route": None,
                "pair_activated": False,
                "admitted_ns": None,
                "first_response_ns": None,
                "eof_ns": None,
                "error": None,
                "rejection_reason": None,
                "failure_kind": None,
                "failure_receipt": None,
                "failure_receipt_sha256": None,
            }
            for spec in specs
        }
        self._scheduled: dict[str, _Scheduled] = {}
        self._events: list[tuple[int, int, int, str, str]] = []
        self._event_serial = 0
        self._active = [0, 0]
        self._endpoint_active = [0, 0]
        self._remote_active = [0, 0]
        self._completed_first_responses = [0, 0]
        self._telemetry_sequence = 1
        self._telemetry_updates = 0
        self._errors: list[str] = []
        self._failed_requests: set[str] = set()
        self._failure_receipts: list[dict[str, Any]] = []
        self._failure_target_admitted = False
        self._failure_target_request_id: str | None = None
        self._failure_target_route: str | None = None

    def _telemetry(self, now_ns: int) -> tuple[PairTelemetry, ...]:
        pairs = []
        for pair in range(2):
            capacity = self.profile.capacities[pair].resources
            remote_multiplier = 1.0
            if self.include_completion_feedback:
                # This is a deterministic endpoint service model for control
                # invariants only.  It is not a measured latency claim.
                denominator = max(1, capacity.endpoint_requests // 4)
                remote_multiplier += min(
                    3.0, self._remote_active[pair] / denominator)
            failure_observed = (
                self.include_completion_feedback
                and self._telemetry_failure_active
                and pair == 0
            )
            cross_layer = None
            if self.include_cross_layer_pressure:
                signals = (
                    CrossLayerSignal(
                        name="nccl_collective_p99_ms",
                        value=25.0,
                        unit="milliseconds",
                        support="supported",
                        source="offline_replay_cross_layer_fixture",
                        scope="communicator",
                    ),
                    CrossLayerSignal(
                        name="lmcache_transfer_p99_ms",
                        value=80.0,
                        unit="milliseconds",
                        support="supported",
                        source="offline_replay_cross_layer_fixture",
                        scope="pair",
                    ),
                    CrossLayerSignal(
                        name="lmcache_remote_kv_bytes_inflight",
                        value=768 * 1024 * 1024,
                        unit="bytes",
                        support="supported",
                        source="offline_replay_cross_layer_fixture",
                        scope="pair",
                    ),
                    CrossLayerSignal(
                        name="cassini_rx_pause_fraction_max",
                        value=0.25,
                        unit="fraction",
                        support="supported",
                        source="offline_replay_cross_layer_fixture",
                        scope="node",
                    ),
                )
                cross_layer = CrossLayerTelemetry(
                    pair_index=pair,
                    node_id=f"replay-node-{pair}",
                    endpoint_id=f"replay-pair-{pair}",
                    communicator_id="replay-shared-communicator",
                    source_epoch="tempo-go-offline-replay-epoch",
                    topology_fingerprint_sha256="e" * 64,
                    sequence=self._telemetry_sequence,
                    sampled_ns=now_ns,
                    window_ms=20.0,
                    signals=signals,
                    # Exercise the same node-local 4 NIC x 8 traffic-class
                    # shape emitted by CassiniEndpointSampler.  Pair 0 has
                    # one hot NIC while pair 1 is cool; this is deliberately
                    # a mechanism fixture, never a performance claim.
                    cassini_by_nic=tuple(
                        tuple(
                            (
                                traffic_class,
                                (
                                    0.35
                                    if pair == 0 and nic_index == 0
                                    else 0.05
                                    if pair == 1
                                    else 0.0
                                ),
                                (
                                    0.10
                                    if pair == 0 and nic_index == 0
                                    else 0.02
                                    if pair == 1
                                    else 0.0
                                ),
                            )
                            for traffic_class in range(8)
                        )
                        for nic_index in range(4)
                    ),
                )
            pairs.append(PairTelemetry(
                pair_index=pair,
                sequence=self._telemetry_sequence,
                sampled_ns=now_ns,
                collected_ns=now_ns + 1,
                agent_epoch="tempo-go-offline-replay-epoch",
                profile_fingerprint_sha256=self.profile.fingerprint_sha256,
                controller_generation=self.profile.telemetry.controller_generation,
                observed_total=ResourceVector(),
                local_health=PathHealth.GOOD,
                remote_health=PathHealth.GOOD,
                local_service_multiplier=1.0,
                remote_service_multiplier=remote_multiplier,
                remote_failure_count=1 if failure_observed else 0,
                remote_last_failure_kind=(
                    "observed_lmcache_engine_failure"
                    if failure_observed else None
                ),
                scheduler_running_requests=self._active[pair],
                scheduler_waiting_requests=0,
                scheduler_kv_cache_usage_fraction=min(
                    1.0, self._active[pair] / capacity.active_sequences),
                scheduler_schema=_SCHEDULER_SCHEMA,
                scheduler_source=_SCHEDULER_SOURCE,
                endpoint_completed_first_responses=(
                    self._completed_first_responses[pair]
                    if self.include_completion_feedback else None),
                endpoint_residual_inflight=(
                    self._endpoint_active[pair]
                    if self.include_completion_feedback else None),
                completion_schema=(
                    _COMPLETION_SCHEMA if self.include_completion_feedback else None),
                cross_layer=cross_layer,
            ))
        self._telemetry_sequence += 1
        self._telemetry_updates += 1
        return tuple(pairs)

    def _refresh(self, now_ns: int) -> None:
        self.orchestrator.update_telemetry_batch(self._telemetry(now_ns))

    def _push_event(self, when_ns: int, kind: str, request_id: str) -> None:
        order = {
            # A queue timeout wins ties with a completion event.  This is the
            # deterministic replay equivalent of the frontend's
            # wait_for(timeout) boundary; it keeps the control-plane contract
            # explicit when a queued request and an EOF share a timestamp.
            "admission_timeout": 0,
            "route_failure": 1,
            "first_response": 2,
            "eof": 3,
        }.get(kind)
        if order is None:
            raise ValueError(f"unsupported replay event kind: {kind}")
        self._event_serial += 1
        heapq.heappush(
            self._events, (when_ns, order, self._event_serial, kind, request_id))

    def _record_decision(self, decision: GlobalDecision) -> None:
        record = self.records[decision.request_id]
        record["last_decision_kind"] = decision.kind.value
        record["last_queue_lease"] = decision.queue_lease
        record["last_rejected_candidates"] = [
            {
                "pair_index": item.pair_index,
                "route": item.route.value,
                "reason": item.reason,
                "binding_resources": list(item.binding_resources),
            }
            for item in decision.rejected_candidates
        ]
        record["decision_reasons"].append(decision.reason)
        if decision.kind is GlobalDecisionKind.QUEUE:
            record["queue_decisions"] += 1
            record["status"] = "queued"
            if not record["admission_timeout_scheduled"]:
                request = self._requests[decision.request_id]
                wait_budget_ns = min(
                    self.orchestrator.admission_wait_budget_ns(
                        request.tenant_id),
                    max(0, request.deadline_ns - request.arrival_ns),
                )
                self._push_event(
                    request.arrival_ns + wait_budget_ns,
                    "admission_timeout",
                    decision.request_id,
                )
                record["admission_timeout_scheduled"] = True
            return
        if decision.kind is GlobalDecisionKind.REJECT:
            record["status"] = "rejected"
            record["rejection_reason"] = decision.reason
            return
        if decision.kind is not GlobalDecisionKind.ADMIT:
            self._errors.append(
                f"unsupported decision kind {decision.kind} for {decision.request_id}")
            return
        if decision.request_id in self._scheduled:
            self._errors.append(f"duplicate route commitment: {decision.request_id}")
            return
        if decision.pair_index is None or decision.route is None:
            self._errors.append(f"admission lacks route: {decision.request_id}")
            return
        multiplier = 1.0
        if decision.route is GlobalRoute.REMOTE and self.include_completion_feedback:
            denominator = max(1, self.profile.capacities[decision.pair_index].resources.endpoint_requests // 4)
            multiplier += min(3.0, self._remote_active[decision.pair_index] / denominator)
        start_ns = decision.decided_ns
        first_ns = start_ns + max(
            1, round(float(decision.predicted_ttft_ms) * multiplier * 1_000_000))
        eof_ns = start_ns + max(
            first_ns - start_ns,
            round(float(decision.predicted_e2e_ms) * multiplier * 1_000_000),
        )
        scheduled = _Scheduled(
            request_id=decision.request_id,
            pair_index=decision.pair_index,
            route=decision.route,
            admitted_ns=start_ns,
            first_response_ns=first_ns,
            eof_ns=eof_ns,
            predicted_ttft_ms=float(decision.predicted_ttft_ms),
            predicted_e2e_ms=float(decision.predicted_e2e_ms),
        )
        self._scheduled[decision.request_id] = scheduled
        self._active[decision.pair_index] += 1
        self._endpoint_active[decision.pair_index] += 1
        if decision.route is GlobalRoute.REMOTE:
            self._remote_active[decision.pair_index] += 1
        record["status"] = "admitted"
        record["pair"] = decision.pair_index
        record["route"] = decision.route.value
        record["pair_activated"] = decision.pair_activated
        record["admitted_ns"] = start_ns
        if (
            self.failure_index is not None
            and int(record["index"]) == self.failure_index
        ):
            if decision.route is not GlobalRoute.REMOTE:
                self._errors.append(
                    "failure injection target was admitted on a non-remote route: "
                    f"{decision.request_id} -> {decision.route.value}"
                )
            else:
                self._failure_target_admitted = True
                self._failure_target_request_id = decision.request_id
                self._failure_target_route = decision.route.value
                # The injected event is ordered immediately before the normal
                # first-response event.  The request therefore exercises the
                # committed-route failure path and never reaches first response.
                self._push_event(first_ns, "route_failure", decision.request_id)
        self._push_event(first_ns, "first_response", decision.request_id)
        self._push_event(eof_ns, "eof", decision.request_id)

    def _dispatch(self, decisions: Iterable[GlobalDecision]) -> None:
        for decision in decisions:
            self._record_decision(decision)

    def _arrive(self, spec: ReplaySpec) -> None:
        now_ns = spec.item.arrival_offset_ns + 1_000_000
        record = self.records[spec.item.request_id]
        record["status"] = "arrived"
        if (
            self.telemetry_failure_index is not None
            and spec.index >= self.telemetry_failure_index
            and not self._telemetry_failure_active
        ):
            self._telemetry_failure_active = True
            self._telemetry_failure_target_request_id = spec.item.request_id
        self._refresh(now_ns)
        try:
            decision = self.orchestrator.submit(spec.request, now_ns=now_ns)
        except Exception as exc:  # record a terminal replay failure, never hide it
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            self._errors.append(f"{spec.item.request_id}: {record['error']}")
            return
        self._record_decision(decision)

    def _event(self) -> None:
        now_ns, _, _, kind, request_id = heapq.heappop(self._events)
        if kind == "admission_timeout":
            record = self.records[request_id]
            # The queued request may have been dispatched just before its
            # timeout event.  Its timeout event remains in the heap for audit,
            # but must then be inert rather than producing a duplicate
            # terminal decision.
            if record["status"] != "queued":
                return
            # Match GlobalAdmissionCoordinator._timeout exactly at the
            # replay boundary: refresh the allocation-wide telemetry first,
            # let any newly feasible waiter dispatch, then give the tenant's
            # frozen endpoint_queue_lease policy one explicit opportunity.
            # The old replay called reject_queued directly, which made every
            # queued request a terminal reject and therefore undercounted the
            # work-conserving native path.
            self._refresh(now_ns)
            if self.records[request_id]["status"] != "queued":
                return
            try:
                decision = self.orchestrator.lease_queued_to_endpoint(
                    request_id, now_ns=now_ns)
                if decision is None:
                    decision = self.orchestrator.reject_queued(
                        request_id,
                        now_ns=now_ns,
                        reason="global_admission_queue_timeout",
                    )
            except Exception as exc:
                self._errors.append(f"admission timeout {request_id}: {exc}")
                return
            self._record_decision(decision)
            return
        if request_id in self._failed_requests:
            # The failed request's normal first-response/EOF events remain in
            # the heap for auditability, but are intentionally inert.
            return
        scheduled = self._scheduled.get(request_id)
        if scheduled is None:
            self._errors.append(f"event has no scheduled request: {request_id}")
            return
        pair = scheduled.pair_index
        if kind == "route_failure":
            if self.failure_index is None:
                self._errors.append(
                    f"unexpected route failure event for {request_id}")
                return
            if scheduled.route is not GlobalRoute.REMOTE:
                self._errors.append(
                    f"route failure injection was not remote: {request_id}")
                return
            try:
                report = self.orchestrator.report_route_failure(
                    request_id,
                    failure_kind="injected_remote_route_failure",
                    now_ns=now_ns,
                    scope="route",
                    route=scheduled.route,
                )
            except Exception as exc:
                self._errors.append(f"route failure {request_id}: {exc}")
                raise RuntimeError(
                    "failure injection could not produce a terminal receipt"
                ) from exc
            # Release replay-side credits only after the orchestrator has
            # accepted the failure and produced its terminal receipt.  This
            # prevents an unsupported injection from turning the subsequent
            # normal first-response event into negative telemetry.
            self._active[pair] -= 1
            self._endpoint_active[pair] -= 1
            self._remote_active[pair] -= 1
            self._refresh(now_ns)
            receipt = global_failure_dict(report.receipt)
            receipt_sha256 = global_failure_fingerprint(report.receipt)
            record = self.records[request_id]
            record["status"] = "failed"
            record["failure_kind"] = report.receipt.failure_kind
            record["failure_receipt"] = receipt
            record["failure_receipt_sha256"] = receipt_sha256
            self._failed_requests.add(request_id)
            self._failure_receipts.append({
                "receipt": receipt,
                "sha256": receipt_sha256,
            })
            self._dispatch(report.dispatched)
            return
        if kind == "first_response":
            self._endpoint_active[pair] -= 1
            if scheduled.route is GlobalRoute.REMOTE:
                self._remote_active[pair] -= 1
            self._refresh(now_ns)
            try:
                decisions = self.orchestrator.mark_first_response(
                    request_id, now_ns=now_ns)
            except Exception as exc:
                self._errors.append(f"first response {request_id}: {exc}")
                return
            self._completed_first_responses[pair] += 1
            self.records[request_id]["first_response_ns"] = now_ns
            self.records[request_id]["status"] = "first_response"
            self._dispatch(decisions)
            return
        if kind != "eof":
            self._errors.append(f"unknown replay event: {kind}")
            return
        self._active[pair] -= 1
        self._refresh(now_ns)
        try:
            decisions = self.orchestrator.complete(request_id, now_ns=now_ns)
        except Exception as exc:
            self._errors.append(f"EOF {request_id}: {exc}")
            return
        self.records[request_id]["eof_ns"] = now_ns
        self.records[request_id]["status"] = "complete"
        self._dispatch(decisions)

    def run(self) -> dict[str, Any]:
        next_index = 0
        while next_index < len(self.specs) or self._events:
            next_arrival = (
                self.specs[next_index].item.arrival_offset_ns + 1_000_000
                if next_index < len(self.specs) else None)
            next_event = self._events[0][0] if self._events else None
            if next_event is not None and (
                next_arrival is None or next_event <= next_arrival):
                self._event()
                continue
            self._arrive(self.specs[next_index])
            next_index += 1
        # Every queued request has a per-request timeout event.  Keep a final
        # bounded sweep only as a leak guard for future replay changes; normal
        # execution should have no queued entries left here.
        drain_ns = max(
            (spec.item.arrival_offset_ns for spec in self.specs), default=0
        ) + self.orchestrator.config.maximum_queue_wait_ns + 1_000_000
        queued = [
            request_id
            for request_id, phase in self.orchestrator.snapshot(
                now_ns=drain_ns)["phases"].items()
            if phase == "queued"
        ]
        for request_id in queued:
            decision = self.orchestrator.lease_queued_to_endpoint(
                request_id, now_ns=drain_ns)
            if decision is None:
                decision = self.orchestrator.reject_queued(
                    request_id,
                    now_ns=drain_ns,
                    reason="global_admission_queue_timeout",
                )
            self._record_decision(decision)
        snapshot = self.orchestrator.snapshot(now_ns=drain_ns)
        return self._report(snapshot)

    def _report(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        records = list(self.records.values())
        complete = [item for item in records if item["status"] == "complete"]
        terminal = [
            item for item in records
            if item["status"] in {"complete", "failed", "rejected"}
        ]
        e2e_ms = [
            (item["eof_ns"] - item["arrival_ns"]) / 1_000_000
            for item in complete
        ]
        ttft_ms = [
            (item["first_response_ns"] - item["arrival_ns"]) / 1_000_000
            for item in complete
        ]
        tpot_ms = []
        for item in complete:
            output = int(item["max_tokens"])
            if output > 1:
                tpot_ms.append(
                    (item["eof_ns"] - item["first_response_ns"])
                    / 1_000_000 / (output - 1))
        by_tenant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in records:
            by_tenant[str(item["tenant_id"])].append(item)
        tenant_report = {}
        for tenant, values in sorted(by_tenant.items()):
            policy = next(
                item for item in self.profile.tenants if item.tenant_id == tenant)
            done = [item for item in values if item["status"] == "complete"]
            good = 0
            waits = []
            for item in done:
                ttft = (item["first_response_ns"] - item["arrival_ns"]) / 1_000_000
                e2e = (item["eof_ns"] - item["arrival_ns"]) / 1_000_000
                output = int(item["max_tokens"])
                tpot = (
                    (item["eof_ns"] - item["first_response_ns"])
                    / 1_000_000 / (output - 1)
                    if output > 1 else math.inf)
                good += int(
                    ttft <= policy.ttft_slo_ms
                    and tpot <= policy.tpot_slo_ms
                    and e2e <= policy.e2e_slo_ms)
                if item["admitted_ns"] is not None:
                    waits.append((item["admitted_ns"] - item["arrival_ns"]) / 1_000_000)
            tenant_report[tenant] = {
                "arrivals": len(values),
                "completed": len(done),
                "failed": sum(item["status"] == "failed" for item in values),
                "rejected": sum(item["status"] == "rejected" for item in values),
                "queued_or_unfinished": sum(
                    item["status"] not in {"complete", "failed", "rejected"}
                    for item in values),
                "slo_goodput_requests": good,
                "minimum_service_fraction": policy.minimum_service_fraction,
                "weight": policy.weight,
                "max_queue_wait_ms": max(waits, default=None),
                "starved": bool(values) and not done,
            }
        routes = Counter(item["route"] for item in complete if item["route"])
        pairs = Counter(str(item["pair"]) for item in complete if item["pair"] is not None)
        predicted_counterfactual = []
        for spec in self.specs:
            item = self.records[spec.item.request_id]
            if item["status"] != "complete" or item["pair"] is None:
                continue
            selected = next(
                candidate for candidate in spec.request.candidates
                if candidate.pair_index == item["pair"]
                and candidate.route.value == item["route"])
            opposite = [
                candidate for candidate in spec.request.candidates
                if candidate.pair_index == item["pair"]
                and candidate.route is not selected.route
            ]
            if opposite:
                predicted_counterfactual.append({
                    "request_id": spec.item.request_id,
                    "selected_route": selected.route.value,
                    "selected_predicted_e2e_ms": selected.predicted_e2e_ms,
                    "opposite_predicted_e2e_ms": opposite[0].predicted_e2e_ms,
                })
        return {
            "arm": self.arm,
            "request_count": len(records),
            "completed": len(complete),
            "failed": sum(item["status"] == "failed" for item in records),
            "rejected": sum(item["status"] == "rejected" for item in records),
            "queued_or_unfinished": len(records) - len(terminal),
            "queue_decisions": sum(int(item["queue_decisions"]) for item in records),
            "routes": dict(sorted(routes.items())),
            "pairs": dict(sorted(pairs.items())),
            "e2e_ms": {name: _percentile(e2e_ms, value) for name, value in (
                ("p50", 50), ("p95", 95), ("p99", 99))},
            "ttft_ms": {name: _percentile(ttft_ms, value) for name, value in (
                ("p50", 50), ("p95", 95), ("p99", 99))},
            "tpot_ms": {name: _percentile(tpot_ms, value) for name, value in (
                ("p50", 50), ("p95", 95), ("p99", 99))},
            "tenant": tenant_report,
            "pair_activation_count": sum(
                bool(item["pair_activated"]) for item in records),
            "selected_route_counterfactual": {
                "schema": "tempo-go-predicted-counterfactual-v1",
                "claim_allowed": False,
                "rows": predicted_counterfactual,
            },
            "failure_injection": {
                "enabled": self.failure_index is not None,
                "target_index": self.failure_index,
                "target_admitted_on_remote": self._failure_target_admitted,
                "target_request_id": self._failure_target_request_id,
                "target_route": self._failure_target_route,
                "triggered": len(self._failure_receipts) == 1,
                "expected_failure_kind": (
                    "injected_remote_route_failure"
                    if self.failure_index is not None else None
                ),
            },
            "telemetry_failure_injection": {
                "enabled": self.telemetry_failure_index is not None,
                "target_index": self.telemetry_failure_index,
                "target_request_id": self._telemetry_failure_target_request_id,
                "triggered": self._telemetry_failure_active,
                "scope": (
                    self.profile.controller.get(
                        "telemetry_failure_quarantine_scope")
                    if self.telemetry_failure_index is not None else None
                ),
            },
            "failure_receipts": list(self._failure_receipts),
            "telemetry": {
                "updates": self._telemetry_updates,
                "scheduler_observation": True,
                "completion_observation": self.include_completion_feedback,
                "cpu_overhead_measured": False,
                "overhead_claim_allowed": False,
            },
            "snapshot": snapshot,
            "records": records,
            "errors": list(self._errors),
            "invariants": {
                "all_requests_terminal": len(terminal) == len(records),
                "no_errors": not self._errors,
                "no_owned_resource_leak": not any(
                    any(value.values()) for value in snapshot["owned_by_pair"].values()),
                "no_inflight": snapshot["inflight"] == 0,
                "no_queued": snapshot["queued"] == 0,
                "phase_name_policy_input": False,
                "physical_switch_policy_input": False,
                "performance_claim_allowed": False,
            },
        }


def _make_specs(
    *,
    workload_path: Path,
    manifest: dict[str, Any],
    global_profile,
    elastic_path: Path,
    endpoint_path: Path,
    tokenizer,
) -> list[ReplaySpec]:
    items, workload_sha = load_workload(
        workload_path, default_max_tokens=64, request_rate=None)
    declared = manifest.get("validation_workload", {})
    if declared.get("sha256") != workload_sha:
        raise ValueError("manifest validation workload SHA does not match replay input")
    elastic = load_elastic_profile(elastic_path)
    endpoint = load_endpoint_service_profile(endpoint_path)
    if endpoint.elastic_profile_fingerprint_sha256 != elastic.fingerprint_sha256:
        raise ValueError("elastic and endpoint replay profiles are not identical")
    proxy_policy = global_profile.service_proxy_policy()
    tempo_builder = GlobalCandidateBuilder(
        elastic,
        endpoint,
        pair_count=2,
        allow_service_proxy=(
            global_profile.deployment_scope == "discovery"
            and proxy_policy is None),
        service_proxy_policy=proxy_policy,
    )
    # Fixed comparison arms are baselines, not TEMPO-GO policy consumers.  A
    # frozen policy may intentionally deny MISS->remote for adaptive TEMPO,
    # while the official always-remote baseline still needs its separately
    # labelled calibration proxy to remain executable.  Keep the two
    # candidate sets distinct so a controller guard cannot silently change a
    # comparison arm.
    baseline_builder = GlobalCandidateBuilder(
        elastic,
        endpoint,
        pair_count=2,
        allow_service_proxy=True,
    )
    specs = []
    for item in items:
        prompt_tokens = len(tokenizer.encode(item.prompt, add_special_tokens=False))
        residency, source = _cache_evidence(item)
        tenant = _tenant(item.request_id)
        policy = next(
            value for value in global_profile.tenants if value.tenant_id == tenant)
        arrival_ns = item.arrival_offset_ns + 1_000_000
        request_kwargs = {
            "request_id": item.request_id,
            "tenant_id": tenant,
            "arrival_ns": arrival_ns,
            "deadline_ns": arrival_ns + int(policy.e2e_slo_ms * 1_000_000),
            "prompt_tokens": prompt_tokens,
            "output_tokens": item.max_tokens,
            "cache_states": tuple(
                PairCacheState(
                    pair_index=pair,
                    residency=residency,
                    source=source,
                ) for pair in range(2)),
        }
        request = tempo_builder.build(**request_kwargs)
        baseline_request = baseline_builder.build(
            **request_kwargs,
        )
        specs.append(ReplaySpec(
            index=item.index,
            item=item,
            prompt_tokens=prompt_tokens,
            phase=_phase_for(item.index, manifest),
            residency=residency,
            residency_source=source,
            request=request,
            baseline_request=baseline_request,
        ))
    return specs


def replay(
    *,
    manifest_path: Path,
    workload_path: Path,
    model_path: Path,
    global_profile_path: Path,
    baseline_global_profile_path: Path | None = None,
    elastic_profile_path: Path,
    endpoint_profile_path: Path,
    failure_index: int | None = None,
    telemetry_failure_index: int | None = None,
    include_cross_layer_pressure: bool = False,
    run_contract_path: Path | None = None,
    run_contract_sha256: str | None = None,
) -> dict[str, Any]:
    frozen_contract = None
    if run_contract_path is not None or run_contract_sha256 is not None:
        if run_contract_path is None or run_contract_sha256 is None:
            raise ValueError("C5 replay contract path and SHA are both required")
        repo_root = Path(__file__).resolve().parents[2]
        frozen_contract = run_contract.verify_contract(
            run_contract_path.resolve(),
            run_contract_sha256,
            repo_root=repo_root,
            workload_input=workload_path,
        )
        artifacts = frozen_contract["artifacts"]
        expected_paths = {
            "manifest": manifest_path.resolve(),
            "workload": workload_path.resolve(),
            "global_profile": global_profile_path.resolve(),
            "elastic_profile": elastic_profile_path.resolve(),
            "endpoint_profile": endpoint_profile_path.resolve(),
            "model_config": (model_path / "config.json").resolve(),
        }
        for name, expected in expected_paths.items():
            actual = Path(str(artifacts[name]["path"])).resolve()
            if actual != expected:
                raise ValueError(
                    f"C5 replay {name} differs from frozen run contract")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = validate_manifest(manifest_path, workload_path)
    global_profile = load_global_profile(global_profile_path)
    baseline_global_profile = load_global_profile(
        baseline_global_profile_path
        if baseline_global_profile_path is not None
        else global_profile_path
    )
    if (
        baseline_global_profile.identity.workload_manifest_sha256
        != global_profile.identity.workload_manifest_sha256
        or baseline_global_profile.identity.elastic_profile_fingerprint_sha256
        != global_profile.identity.elastic_profile_fingerprint_sha256
        or baseline_global_profile.identity.endpoint_profile_fingerprint_sha256
        != global_profile.identity.endpoint_profile_fingerprint_sha256
    ):
        raise ValueError(
            "baseline and TEMPO global profiles do not share frozen identities"
        )
    if failure_index is not None and (
        global_profile.controller.get("route_failure_quarantine_mode")
        != "deny_until_probe"
    ):
        raise ValueError(
            "failure injection requires a global profile with "
            "route_failure_quarantine_mode=deny_until_probe"
        )
    if telemetry_failure_index is not None and (
        global_profile.controller.get("telemetry_failure_quarantine_mode")
        != "deny_until_probe"
    ):
        raise ValueError(
            "telemetry failure injection requires a global profile with "
            "telemetry_failure_quarantine_mode=deny_until_probe"
        )
    proxy_policy = global_profile.service_proxy_policy()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False)
    specs = _make_specs(
        workload_path=workload_path,
        manifest=manifest,
        global_profile=global_profile,
        elastic_path=elastic_profile_path,
        endpoint_path=endpoint_profile_path,
        tokenizer=tokenizer,
    )
    arms = {}
    for arm in ARM_NAMES:
        arm_specs = [
            ReplaySpec(
                index=spec.index,
                item=spec.item,
                prompt_tokens=spec.prompt_tokens,
                phase=spec.phase,
                residency=spec.residency,
                residency_source=spec.residency_source,
                request=_filtered_request(
                    spec.request if arm == "tempo_go" else spec.baseline_request,
                    arm,
                ),
                baseline_request=spec.baseline_request,
            ) for spec in specs
        ]
        arms[arm] = _ArmReplay(
            arm=arm,
            specs=arm_specs,
            profile=(global_profile if arm == "tempo_go" else baseline_global_profile),
            include_completion_feedback=arm == "tempo_go",
            failure_index=failure_index if arm == "tempo_go" else None,
            telemetry_failure_index=(
                telemetry_failure_index if arm == "tempo_go" else None
            ),
            include_cross_layer_pressure=(
                include_cross_layer_pressure and arm == "tempo_go"
            ),
        ).run()
    manifest_sha = _sha256(manifest_path)
    binding_match = global_profile.identity.workload_manifest_sha256 == manifest_sha
    return {
        "schema": SCHEMA,
        "performance_claim_allowed": False,
        "replay_authorized_for_native": False,
        "native_gpu_run_allowed": False,
        "policy_inputs_excluded": [
            "phase_name", "future_arrivals", "oracle_route", "physical_switch_label",
        ],
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "workload": str(workload_path.resolve()),
        "workload_sha256": validation["workload_sha256"],
        "model": str(model_path.resolve()),
        "global_profile": str(global_profile_path.resolve()),
        "global_profile_sha256": global_profile.fingerprint_sha256,
        "baseline_global_profile": str(
            (baseline_global_profile_path or global_profile_path).resolve()),
        "baseline_global_profile_sha256": baseline_global_profile.fingerprint_sha256,
        "elastic_profile": str(elastic_profile_path.resolve()),
        "elastic_profile_sha256": _sha256(elastic_profile_path),
        "endpoint_profile": str(endpoint_profile_path.resolve()),
        "endpoint_profile_sha256": _sha256(endpoint_profile_path),
        "profile_workload_binding_match": binding_match,
        "service_proxy_policy": (
            proxy_policy.as_dict() if proxy_policy is not None else None),
        "calibration_only_prior": True,
        "run_contract": (
            str(run_contract_path.resolve()) if run_contract_path is not None else None
        ),
        "run_contract_sha256": run_contract_sha256,
        "run_contract_fingerprint_sha256": (
            frozen_contract.get("fingerprint_sha256")
            if frozen_contract is not None else None
        ),
        "failure_injection": {
            "schema": "tempo-go-replay-failure-injection-v1",
            "enabled": failure_index is not None,
            "target_index": failure_index,
            "applied_arm": "tempo_go" if failure_index is not None else None,
            "failure_kind": (
                "injected_remote_route_failure"
                if failure_index is not None else None
            ),
            "claim_allowed": False,
        },
        "telemetry_failure_injection": {
            "schema": "tempo-go-replay-telemetry-failure-injection-v1",
            "enabled": telemetry_failure_index is not None,
            "target_index": telemetry_failure_index,
            "applied_arm": "tempo_go" if telemetry_failure_index is not None else None,
            "failure_kind": (
                "observed_lmcache_engine_failure"
                if telemetry_failure_index is not None else None
            ),
            "claim_allowed": False,
        },
        "cross_layer_pressure_fixture": {
            "enabled": include_cross_layer_pressure,
            "claim_allowed": False,
            "scope": "offline_control_plane_only",
        },
        "validation": validation,
        "arms": arms,
        "replay_gates": {
            "manifest_valid": True,
            "all_arms_have_same_request_count": len({
                value["request_count"] for value in arms.values()}) == 1,
            "all_arms_have_same_trace_sha": True,
            "all_arms_no_phase_policy_input": all(
                value["invariants"]["phase_name_policy_input"] is False
                for value in arms.values()),
            "all_arms_no_physical_switch_input": all(
                value["invariants"]["physical_switch_policy_input"] is False
                for value in arms.values()),
            "all_arms_terminal_and_leak_free": all(
                value["invariants"]["all_requests_terminal"]
                and value["invariants"]["no_owned_resource_leak"]
                and value["invariants"]["no_inflight"]
                and value["invariants"]["no_queued"]
                for value in arms.values()),
            "failure_injection_triggered": (
                failure_index is None
                or arms["tempo_go"]["failure_injection"]["triggered"]
            ),
            "telemetry_failure_injection_triggered": (
                telemetry_failure_index is None
                or arms["tempo_go"]["telemetry_failure_injection"]["triggered"]
            ),
            "failure_receipt_schema_validated": (
                failure_index is None
                or bool(arms["tempo_go"]["failure_receipts"])
            ),
            "profile_binding_required_before_native": not binding_match,
            "frozen_run_contract_valid": frozen_contract is not None,
            "performance_claim_allowed": False,
        },
        "claim_boundary": (
            "deterministic control-plane replay only; no GPU, LMCache, fabric, "
            "latency, goodput, or production claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--global-profile", type=Path,
        default=Path("eval/sota_4node/real_tempo_go_discovery_profile_v1.json"),
    )
    parser.add_argument(
        "--baseline-global-profile", type=Path,
        help=(
            "frozen global profile for fixed arms; defaults to the TEMPO "
            "profile when omitted"),
    )
    parser.add_argument(
        "--elastic-profile", type=Path,
        default=Path(
            "results/tempo_elastic_pd_canonical_discovery_57133688/profiles/"
            "real_tempo_pd_elastic_profile_run17_v452.json"),
    )
    parser.add_argument(
        "--endpoint-profile", type=Path,
        default=Path(
            "eval/sota_4node/real_tempo_pd_endpoint_service_profile_c4_"
            "semantic_credit_epoch_v2.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path)
    parser.add_argument("--run-contract-sha256")
    parser.add_argument(
        "--inject-failure-index",
        type=int,
        default=None,
        help=(
            "inject one explicit remote route failure in TEMPO-GO for the "
            "specified workload row index; control-plane evidence only"
        ),
    )
    parser.add_argument(
        "--inject-telemetry-failure-index",
        type=int,
        default=None,
        help=(
            "expose one cumulative endpoint failure observation from the "
            "specified workload row onward; control-plane evidence only"
        ),
    )
    parser.add_argument(
        "--include-cross-layer-pressure",
        action="store_true",
        help=(
            "inject deterministic cross-layer signals into the TEMPO arm "
            "to exercise shared remote-budget control; no performance claim"
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite replay output: {args.output}")
    result = replay(
        manifest_path=args.manifest.resolve(),
        workload_path=args.workload.resolve(),
        model_path=args.model.resolve(),
        global_profile_path=args.global_profile.resolve(),
        baseline_global_profile_path=(
            args.baseline_global_profile.resolve()
            if args.baseline_global_profile is not None else None
        ),
        elastic_profile_path=args.elastic_profile.resolve(),
        endpoint_profile_path=args.endpoint_profile.resolve(),
        failure_index=args.inject_failure_index,
        telemetry_failure_index=args.inject_telemetry_failure_index,
        include_cross_layer_pressure=args.include_cross_layer_pressure,
        run_contract_path=(
            args.run_contract.resolve() if args.run_contract is not None else None
        ),
        run_contract_sha256=args.run_contract_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
