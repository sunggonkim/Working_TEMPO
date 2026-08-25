"""Async HTTP-lifecycle coordinator around the TEMPO-GO state machine."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
import time

from tempo.pd_global_agent import RequestTriggeredTelemetryAgent
from tempo.pd_global_hierarchy import (
    HierarchicalCandidateReducer,
    HierarchyCandidateUnavailableError,
    HierarchyIdentityError,
    HierarchyTelemetryStaleError,
)
from tempo.pd_global_orchestrator import (
    GlobalDecision,
    GlobalDecisionKind,
    GlobalFailureReport,
    GlobalOrchestrator,
    GlobalRequest,
    GlobalServiceLaneQueuePromotionReport,
    GlobalServiceLaneReservationFailureReport,
)
from tempo.pd_global_telemetry import GlobalTelemetryBatch


@dataclass(frozen=True)
class GlobalAdmissionPreparation:
    """One request-triggered telemetry result prepared before admission.

    Tokenization and the bounded all-pair scrape are independent prerequisites
    of a native admission.  The frontend may therefore overlap them and pass
    this immutable result back to :meth:`admit`.  This does not relax the
    orchestrator's admission-time freshness/identity checks and does not
    create a persistent poller.
    """

    batch: GlobalTelemetryBatch | None
    refresh_reason: str | None
    collection_started_ns: int
    collection_finished_ns: int
    attempts_used: int
    retry_triggered: bool

    def __post_init__(self) -> None:
        if (self.batch is None) == (self.refresh_reason is None):
            raise ValueError(
                "exactly one of batch or refresh_reason must be present")
        if self.batch is not None and not isinstance(
            self.batch, GlobalTelemetryBatch
        ):
            raise TypeError("batch must be GlobalTelemetryBatch")
        if self.refresh_reason is not None and (
            not isinstance(self.refresh_reason, str)
            or not self.refresh_reason.strip()
        ):
            raise ValueError("refresh_reason must be nonempty")
        for name in ("collection_started_ns", "collection_finished_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative int")
        if self.collection_finished_ns < self.collection_started_ns:
            raise ValueError("telemetry collection timestamps are inverted")
        if type(self.attempts_used) is not int or self.attempts_used <= 0:
            raise ValueError("attempts_used must be a positive int")
        if type(self.retry_triggered) is not bool:
            raise TypeError("retry_triggered must be bool")

    def as_dict(self) -> dict[str, object]:
        """Return bounded per-request telemetry-control-plane evidence."""

        return {
            "schema": "tempo-go-admission-preparation-v1",
            "status": "batch" if self.batch is not None else "refresh_failed",
            "batch_sequence": (
                self.batch.sequence if self.batch is not None else None),
            "refresh_reason": self.refresh_reason,
            "collection_started_ns": self.collection_started_ns,
            "collection_finished_ns": self.collection_finished_ns,
            "collection_elapsed_ns": (
                self.collection_finished_ns - self.collection_started_ns),
            "attempts_used": self.attempts_used,
            "retry_triggered": self.retry_triggered,
        }


class GlobalAdmissionCoordinator:
    """Join request-triggered telemetry, fair queueing, and stream events."""

    def __init__(
        self,
        orchestrator: GlobalOrchestrator,
        telemetry_agent: RequestTriggeredTelemetryAgent,
        *,
        admission_wait_ns: int,
        hierarchical_reducer: HierarchicalCandidateReducer | None = None,
        clock_ns=time.perf_counter_ns,
    ) -> None:
        if not isinstance(orchestrator, GlobalOrchestrator):
            raise TypeError("orchestrator must be GlobalOrchestrator")
        if not isinstance(telemetry_agent, RequestTriggeredTelemetryAgent):
            raise TypeError(
                "telemetry_agent must be RequestTriggeredTelemetryAgent")
        if type(admission_wait_ns) is not int or admission_wait_ns <= 0:
            raise ValueError("admission_wait_ns must be a positive int")
        if hierarchical_reducer is not None and not isinstance(
            hierarchical_reducer, HierarchicalCandidateReducer
        ):
            raise TypeError(
                "hierarchical_reducer must be HierarchicalCandidateReducer")
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self.orchestrator = orchestrator
        self.telemetry_agent = telemetry_agent
        self.admission_wait_ns = admission_wait_ns
        self.hierarchical_reducer = hierarchical_reducer
        self.clock_ns = clock_ns
        self._lock = asyncio.Lock()
        self._waiters: dict[str, asyncio.Future[GlobalDecision]] = {}
        self._installed_sequence: int | None = None
        self._admitted = 0
        self._queued = 0
        self._queue_timeouts = 0
        self._queue_leases = 0
        self._service_lane_queue_promotions = 0
        self._service_lane_queue_promotion_rejections = 0
        self._service_lane_reservation_failures = 0
        self._telemetry_rejections = 0
        self._telemetry_validation_retries = 0
        self._telemetry_validation_retry_failures = 0
        self._telemetry_preparations = 0
        self._prepared_admission_uses = 0
        self._prepared_telemetry_superseded = 0
        self._stale_snapshot_fallbacks = 0
        self._delivered_from_queue = 0
        self._route_failures = 0
        self._hierarchy_reductions = 0
        self._hierarchy_identity_rejections = 0
        self._hierarchy_stale_rejections = 0
        self._hierarchy_receipts: dict[str, dict[str, object]] = {}

    @staticmethod
    def _refresh_failure_reason(exc: RuntimeError) -> str | None:
        return {
            "global telemetry refresh timed out": (
                "global_telemetry_refresh_timeout"),
            "global telemetry refresh failed": (
                "global_telemetry_refresh_failed"),
            "global telemetry validation failed": (
                "global_telemetry_validation_failed"),
        }.get(str(exc))

    def _use_stale_snapshot_locked(self, *, tenant_id: str | None = None) -> bool:
        """Dispatch from the last bounded snapshot after refresh failure.

        The orchestrator decides whether the installed snapshot is within its
        configured grace and still applies every resource/health/transport
        guard.  The coordinator only handles the request-triggered refresh
        failure boundary and records that the degraded path was used.
        """

        if not self.orchestrator.telemetry_admission_available(
            now_ns=self.clock_ns(), tenant_id=tenant_id,
        ):
            return False
        self._stale_snapshot_fallbacks += 1
        self._deliver_locked(
            self.orchestrator.dispatch(now_ns=self.clock_ns()))
        return True

    async def _collect_telemetry(
        self, *, force: bool = False,
    ) -> tuple[GlobalTelemetryBatch | None, str | None, int, bool]:
        """Collect one causal batch with one bounded validation-only retry.

        A collection that exceeded its causal span or failed schema validation
        is never installed.  Under a loaded frontend event loop that isolated
        attempt can miss the wall-clock span even though the next
        request-triggered scrape is valid.  Retry exactly once inside the
        foreground admission path; fetch timeout/failure behavior remains
        fail-closed and there is no background polling or unbounded loop.
        """

        try:
            return await self.telemetry_agent.get(force=force), None, 1, False
        except RuntimeError as exc:
            reason = self._refresh_failure_reason(exc)
            if reason is None:
                raise
        if reason != "global_telemetry_validation_failed":
            return None, reason, 1, False
        self._telemetry_validation_retries += 1
        try:
            return await self.telemetry_agent.get(force=True), None, 2, True
        except RuntimeError as exc:
            retry_reason = self._refresh_failure_reason(exc)
            if retry_reason is None:
                raise
            self._telemetry_validation_retry_failures += 1
            return None, retry_reason, 2, True

    async def prepare_admission(self) -> GlobalAdmissionPreparation:
        """Prepare one bounded telemetry result for a pending request.

        Callers may run this coroutine concurrently with tokenization.  The
        agent still coalesces concurrent refreshes and performs no background
        work after the request-scoped coroutine completes.
        """

        started_ns = self.clock_ns()
        batch, refresh_reason, attempts_used, retry_triggered = (
            await self._collect_telemetry())
        finished_ns = self.clock_ns()
        self._telemetry_preparations += 1
        return GlobalAdmissionPreparation(
            batch=batch,
            refresh_reason=refresh_reason,
            collection_started_ns=started_ns,
            collection_finished_ns=finished_ns,
            attempts_used=attempts_used,
            retry_triggered=retry_triggered,
        )

    def _deliver_locked(self, decisions: Iterable[GlobalDecision]) -> None:
        for decision in decisions:
            waiter = self._waiters.pop(decision.request_id, None)
            if waiter is None:
                raise RuntimeError(
                    "global dispatch has no matching frontend waiter")
            if waiter.done():
                raise RuntimeError("global frontend waiter completed twice")
            waiter.set_result(decision)
            self._delivered_from_queue += 1

    def _install_batch_locked(
        self,
        batch: GlobalTelemetryBatch,
        *,
        allow_superseded: bool = False,
    ) -> None:
        """Atomically install one already-collected batch and dispatch."""

        if not isinstance(batch, GlobalTelemetryBatch):
            raise TypeError("batch must be GlobalTelemetryBatch")
        if self._installed_sequence is None:
            self.orchestrator.update_telemetry_batch(batch.pairs)
            self._installed_sequence = batch.sequence
        elif batch.sequence > self._installed_sequence:
            self.orchestrator.update_telemetry_batch(batch.pairs)
            self._installed_sequence = batch.sequence
        elif batch.sequence < self._installed_sequence:
            if not allow_superseded:
                raise RuntimeError("global telemetry sequence moved backwards")
            # A request-scoped preparation can be overtaken while its caller
            # tokenizes by another request or a forced queue-boundary refresh.
            # Never reinstall the older batch; dispatch from the already
            # installed newer generation instead.
            self._prepared_telemetry_superseded += 1
        dispatched = self.orchestrator.dispatch(now_ns=self.clock_ns())
        self._deliver_locked(dispatched)

    async def admit(
        self,
        request: GlobalRequest,
        *,
        preparation: GlobalAdmissionPreparation | None = None,
    ) -> GlobalDecision:
        if not isinstance(request, GlobalRequest):
            raise TypeError("request must be GlobalRequest")
        if preparation is None:
            batch, refresh_reason, _, _ = await self._collect_telemetry()
        else:
            if not isinstance(preparation, GlobalAdmissionPreparation):
                raise TypeError(
                    "preparation must be GlobalAdmissionPreparation")
            batch = preparation.batch
            refresh_reason = preparation.refresh_reason
            self._prepared_admission_uses += 1
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[GlobalDecision] = loop.create_future()
        async with self._lock:
            if request.request_id in self._waiters:
                raise ValueError("duplicate global frontend waiter")
            if batch is not None:
                self._install_batch_locked(
                    batch,
                    allow_superseded=preparation is not None,
                )
            else:
                assert refresh_reason is not None
                if not self._use_stale_snapshot_locked(
                    tenant_id=request.tenant_id):
                    self._telemetry_rejections += 1
                    return self.orchestrator.reject_unadmitted(
                        request,
                        now_ns=self.clock_ns(),
                        reason=refresh_reason,
                    )
            now_ns = self.clock_ns()
            if self.hierarchical_reducer is None:
                decision = self.orchestrator.submit(request, now_ns=now_ns)
            else:
                try:
                    decision, reduction = self.orchestrator.submit_hierarchical(
                        request,
                        reducer=self.hierarchical_reducer,
                        now_ns=now_ns,
                    )
                except HierarchyTelemetryStaleError:
                    self._hierarchy_identity_rejections += 1
                    self._hierarchy_stale_rejections += 1
                    return self.orchestrator.reject_unadmitted(
                        request,
                        now_ns=now_ns,
                        reason="global_telemetry_stale",
                    )
                except HierarchyIdentityError:
                    self._hierarchy_identity_rejections += 1
                    return self.orchestrator.reject_unadmitted(
                        request,
                        now_ns=now_ns,
                        reason="global_hierarchy_identity_reject",
                    )
                except HierarchyCandidateUnavailableError:
                    return self.orchestrator.reject_unadmitted(
                        request,
                        now_ns=now_ns,
                        reason="global_hierarchy_no_candidate",
                    )
                self._hierarchy_reductions += 1
                self._hierarchy_receipts[request.request_id] = {
                    "receipt": reduction.receipt.as_dict(),
                    "fingerprint_sha256": reduction.fingerprint,
                }
            if decision.kind is GlobalDecisionKind.ADMIT:
                self._admitted += 1
                return decision
            if decision.kind is GlobalDecisionKind.REJECT:
                return decision
            if (
                self.orchestrator.config.endpoint_queue_admission_mode
                == "headroom_first_v1"
                and not self._waiters
            ):
                # The request is the head of the global queue.  An immediate
                # lease is allowed only for a tenant that explicitly opted
                # into endpoint queue debt and only when the orchestrator's
                # current scheduler/completion/fabric guards accept it.
                # Existing waiters stay ahead, preserving business fairness.
                leased = self.orchestrator.lease_queued_to_endpoint(
                    request.request_id, now_ns=now_ns)
                if leased is not None:
                    self._queue_leases += 1
                    self._admitted += 1
                    return leased
            self._waiters[request.request_id] = waiter
            self._queued += 1
            wait_ns = min(
                self.orchestrator.admission_wait_budget_ns(request.tenant_id),
                self.admission_wait_ns,
                max(0, request.deadline_ns - now_ns),
            )
        if wait_ns == 0:
            decision = await self._timeout(request, waiter)
            if decision is None:
                raise RuntimeError("global admission queue timed out")
            if decision.kind is GlobalDecisionKind.REJECT:
                return decision
            self._admitted += 1
            return decision
        try:
            decision = await asyncio.wait_for(
                asyncio.shield(waiter), wait_ns / 1_000_000_000)
        except asyncio.TimeoutError as exc:
            decision = await self._timeout(request, waiter)
            if decision is None:
                raise RuntimeError("global admission queue timed out") from exc
            if decision.kind is GlobalDecisionKind.REJECT:
                return decision
            self._admitted += 1
            return decision
        self._admitted += 1
        return decision

    async def _timeout(
        self, request: GlobalRequest, waiter: asyncio.Future[GlobalDecision]
    ) -> GlobalDecision | None:
        request_id = request.request_id
        async with self._lock:
            if waiter.done():
                return waiter.result()

        # Telemetry HTTP must not hold the lifecycle lock: first-response and
        # EOF releases are the events that free capacity for this waiter.  A
        # forced request is still single-flight/coalesced by the agent.
        batch, refresh_reason, _, _ = await self._collect_telemetry(force=True)

        async with self._lock:
            if waiter.done():
                return waiter.result()
            if batch is not None:
                self._install_batch_locked(batch)
            else:
                assert refresh_reason is not None
                if not self._use_stale_snapshot_locked(
                    tenant_id=request.tenant_id):
                    current = self._waiters.pop(request_id, None)
                    if current is not waiter:
                        raise RuntimeError("global queue waiter ownership changed")
                    self._telemetry_rejections += 1
                    decision = self.orchestrator.reject_queued(
                        request_id,
                        now_ns=self.clock_ns(),
                        reason=refresh_reason,
                    )
                    waiter.cancel()
                    return decision
            if waiter.done():
                return waiter.result()
            current = self._waiters.pop(request_id, None)
            if current is not waiter:
                raise RuntimeError("global queue waiter ownership changed")
            now_ns = self.clock_ns()
            decision = self.orchestrator.lease_queued_to_endpoint(
                request_id, now_ns=now_ns)
            if decision is not None:
                self._queue_leases += 1
                waiter.cancel()
                return decision
            decision = self.orchestrator.reject_queued(
                request_id,
                now_ns=now_ns,
                reason="global_admission_queue_timeout",
            )
            waiter.cancel()
            self._queue_timeouts += 1
            return decision

    async def mark_first_response(
        self, request_id: str
    ) -> tuple[GlobalDecision, ...]:
        async with self._lock:
            decisions = self.orchestrator.mark_first_response(
                request_id, now_ns=self.clock_ns())
            self._deliver_locked(decisions)
            return decisions

    async def complete(
        self, request_id: str
    ) -> tuple[GlobalDecision, ...]:
        async with self._lock:
            decisions = self.orchestrator.complete(
                request_id, now_ns=self.clock_ns())
            self._deliver_locked(decisions)
            return decisions

    async def fail(self, request_id: str) -> tuple[GlobalDecision, ...]:
        async with self._lock:
            decisions = self.orchestrator.fail(
                request_id, now_ns=self.clock_ns())
            self._deliver_locked(decisions)
            return decisions

    async def fail_service_lane_reservation(
        self,
        request_id: str,
        *,
        failure_kind: str,
        reason: str,
    ) -> GlobalServiceLaneReservationFailureReport:
        """Close a global lease when the endpoint cannot reserve service.

        This path releases global ownership without quarantining the route;
        the endpoint was never started and the failure is capacity, not path
        health.  Any other global waiters are dispatched after the release.
        """

        async with self._lock:
            report = self.orchestrator.fail_service_lane_reservation(
                request_id,
                failure_kind=failure_kind,
                reason=reason,
                now_ns=self.clock_ns(),
            )
            self._deliver_locked(report.dispatched)
            self._service_lane_reservation_failures += 1
            return report

    async def promote_service_lane_queue_lease(
        self, request_id: str
    ) -> GlobalServiceLaneQueuePromotionReport:
        """Reconcile one endpoint queue offer under the lifecycle lock."""

        async with self._lock:
            report = self.orchestrator.promote_service_lane_queue_lease(
                request_id, now_ns=self.clock_ns())
            if report.decision is None:
                self._service_lane_queue_promotion_rejections += 1
            else:
                self._queue_leases += 1
                self._service_lane_queue_promotions += 1
            return report

    async def fail_route(
        self,
        request_id: str,
        *,
        failure_kind: str,
        scope: str = "route",
    ) -> GlobalFailureReport | None:
        """Record an endpoint failure and dispatch surviving queued work.

        Legacy profiles with the feature disabled retain the exact generic
        failure lifecycle.  Candidate C returns a signed failure receipt and
        never migrates the failed request under its existing ID.
        """

        async with self._lock:
            if (
                self.orchestrator.config.route_failure_quarantine_mode
                == "disabled"
            ):
                decisions = self.orchestrator.fail(
                    request_id, now_ns=self.clock_ns())
                self._deliver_locked(decisions)
                return None
            report = self.orchestrator.report_route_failure(
                request_id,
                failure_kind=failure_kind,
                now_ns=self.clock_ns(),
                scope=scope,
            )
            self._deliver_locked(report.dispatched)
            self._route_failures += 1
            return report

    def status(self) -> dict[str, object]:
        return {
            "mode": "tempo_go_request_lifecycle_coordinator_v1",
            "installed_telemetry_sequence": self._installed_sequence,
            "waiters": len(self._waiters),
            "admitted": self._admitted,
            "queued": self._queued,
            "delivered_from_queue": self._delivered_from_queue,
            "queue_timeouts": self._queue_timeouts,
            "queue_leases": self._queue_leases,
            "service_lane_queue_promotions": (
                self._service_lane_queue_promotions),
            "service_lane_queue_promotion_rejections": (
                self._service_lane_queue_promotion_rejections),
            "service_lane_reservation_failures": (
                self._service_lane_reservation_failures),
            "telemetry_rejections": self._telemetry_rejections,
            "telemetry_validation_retries": (
                self._telemetry_validation_retries),
            "telemetry_validation_retry_failures": (
                self._telemetry_validation_retry_failures),
            "telemetry_preparations": self._telemetry_preparations,
            "prepared_admission_uses": self._prepared_admission_uses,
            "prepared_telemetry_superseded": (
                self._prepared_telemetry_superseded),
            "stale_snapshot_fallbacks": self._stale_snapshot_fallbacks,
            "route_failures": self._route_failures,
            "hierarchical_fan_in": {
                "enabled": self.hierarchical_reducer is not None,
                "reductions": self._hierarchy_reductions,
                "identity_rejections": self._hierarchy_identity_rejections,
                "telemetry_stale_rejections": self._hierarchy_stale_rejections,
                "pending_receipts": len(self._hierarchy_receipts),
            },
            "admission_wait_ns": self.admission_wait_ns,
            "telemetry": self.telemetry_agent.status(),
        }

    def take_hierarchy_receipt(
        self, request_id: str
    ) -> dict[str, object] | None:
        """Consume the immutable fan-in receipt after frontend recording."""

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be nonempty")
        value = self._hierarchy_receipts.pop(request_id, None)
        return dict(value) if value is not None else None


__all__ = ["GlobalAdmissionCoordinator", "GlobalAdmissionPreparation"]
