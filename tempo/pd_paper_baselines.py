"""Paper-faithful baseline policies on TEMPO's common vLLM/LMCache carrier.

This module deliberately contains *baseline* policy, not TEMPO policy.  It
reuses the common candidate/lifecycle safety machinery so every comparison
executes the same vLLM, LMCache, NIXL, model, and physical P-by-D mesh, while
removing TEMPO's business reservations, priority lane, pair scaling, shared
fabric budget, and cross-layer actuation.

Two policies are supported:

``netkv``
    Reproduces Algorithm 1 of NetKV for the topology available in one
    Perlmutter allocation.  It considers only P->D candidates and minimizes
    transfer + decoder queue + first-step cost.  The static oracle bandwidth
    is the documented 25 GB/s per Slingshot 11 NIC; TP=4 shards are charged
    against four NICs.  Live Cassini congestion and LMCache self-inflight
    signals are used only when explicitly supported.

``kairos_x512``
    Reproduces the Kairos deflection decision with a conservative candidate
    chunk set X={512}.  The corresponding frontend launches decoder engines
    with a 512-token batching ceiling.  A local decoder-prefill candidate is
    eligible only when the analytical step estimate is TBT-safe and its TTFT
    is within alpha=1.3 of the best ordinary P/D path.

The restricted Kairos chunk set is explicit because the public paper's code
URL was unavailable during this evaluation and stock vLLM does not expose a
per-request chunk schedule.  It is a real mechanism run, not an attribution
claim for the authors' unreleased implementation.
"""

from __future__ import annotations

from dataclasses import replace
import math
import os

from tempo.pd_global_coordinator import GlobalAdmissionCoordinator
from tempo.pd_global_orchestrator import (
    GlobalDecisionKind,
    GlobalOrchestrator,
    GlobalOrchestratorConfig,
    GlobalRequest,
    GlobalRoute,
    PairTelemetry,
    RejectedCandidate,
    RouteCandidate,
    _CandidateEvaluation,
)


POLICY_ENV = "TEMPO_PAPER_BASELINE_POLICY"
NETKV = "netkv"
KAIROS_X512 = "kairos_x512"
POLICIES = frozenset({NETKV, KAIROS_X512})

# NERSC documents four 200 Gbit/s (25 GB/s) Slingshot 11 NICs per GPU node.
PERLMUTTER_NIC_BYTES_PER_S = 25_000_000_000.0
DEFAULT_TP_DEGREE = 4
KAIROS_CHUNK_TOKENS = 512
KAIROS_ALPHA = 1.3
KAIROS_TBT_SAFETY_FACTOR = 0.9


def _require_finite(name: str, value: float, *, minimum: float = 0.0) -> float:
    if not math.isfinite(float(value)) or float(value) < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _supported_signal(telemetry: PairTelemetry, name: str) -> float | None:
    cross = telemetry.cross_layer
    if cross is None:
        return None
    signal = cross.signal(name)
    if signal is None or signal.support != "supported" or signal.value is None:
        return None
    return float(signal.value)


def _external_congestion_fraction(telemetry: PairTelemetry) -> float:
    """Map only supported dimensionless Cassini signals to NetKV's c_tau."""

    values: list[float] = []
    cross = telemetry.cross_layer
    if cross is not None:
        pause = cross.cassini_nic_pause_max()
        if pause is not None:
            values.append(float(pause))
    for name in (
        "cassini_rx_pause_fraction_max",
        "cassini_tx_pause_fraction_max",
        "cassini_ecn_fraction_max",
        "cassini_oxe_channel_active_fraction_max",
    ):
        value = _supported_signal(telemetry, name)
        if value is not None:
            values.append(value)
    # NetKV requires c < 1.  Unsupported is unknown, not a fabricated zero;
    # the static oracle remains usable and the receipt exposes zero coverage.
    return min(0.95, max(values, default=0.0))


def _self_inflight(telemetry: PairTelemetry) -> int:
    value = _supported_signal(
        telemetry, "lmcache_remote_semantic_ops_inflight")
    if value is None:
        value = _supported_signal(telemetry, "lmcache_remote_ops_inflight")
    if value is None:
        return 0
    return max(0, int(math.ceil(value)))


def _iteration_ms(candidate: RouteCandidate) -> float:
    decode_steps = max(1, candidate.work.decode_tokens - 1)
    residual = max(0.0, candidate.predicted_e2e_ms - candidate.predicted_ttft_ms)
    return max(0.05, residual / decode_steps)


def _decoder_cost_ms(
    candidate: RouteCandidate,
    telemetry: PairTelemetry,
    *, beta_max: int,
) -> tuple[float, float, float]:
    """NetKV equations (6)-(7) using the live vLLM batch/queue gauges."""

    running = telemetry.scheduler_running_requests
    waiting = telemetry.scheduler_waiting_requests
    if running is None or waiting is None:
        raise ValueError("paper baseline requires complete scheduler telemetry")
    base_iteration = _iteration_ms(candidate)
    blocked = max(0, waiting - (beta_max - running))
    queue_ms = blocked * base_iteration
    first_step_ms = base_iteration * (1.0 + running / max(1, beta_max))
    return queue_ms + first_step_ms, queue_ms, first_step_ms


def netkv_score_ms(
    candidate: RouteCandidate,
    *, source: PairTelemetry,
    destination: PairTelemetry,
    beta_max: int,
    tp_degree: int = DEFAULT_TP_DEGREE,
) -> tuple[float, dict[str, float | int]]:
    """Return NetKV post-prefill cost and a complete scoring receipt."""

    if candidate.route is not GlobalRoute.REMOTE:
        raise ValueError("NetKV scores only remote P/D candidates")
    if type(tp_degree) is not int or tp_degree <= 0:
        raise ValueError("tp_degree must be positive")
    congestion = max(
        _external_congestion_fraction(source),
        _external_congestion_fraction(destination),
    )
    inflight = _self_inflight(source)
    effective_bytes_per_s = (
        PERLMUTTER_NIC_BYTES_PER_S
        * tp_degree
        * (1.0 - congestion)
        / (1.0 + inflight)
    )
    transfer_ms = (
        candidate.work.remote_kv_bytes / effective_bytes_per_s * 1000.0
        + 0.01
    )
    decoder_ms, queue_ms, first_step_ms = _decoder_cost_ms(
        candidate, destination, beta_max=beta_max)
    total = transfer_ms + decoder_ms
    return _require_finite("NetKV score", total, minimum=1e-12), {
        "transfer_ms": transfer_ms,
        "queue_ms": queue_ms,
        "first_decode_step_ms": first_step_ms,
        "external_congestion_fraction": congestion,
        "source_self_inflight": inflight,
        "effective_bytes_per_s": effective_bytes_per_s,
        "tp_degree": tp_degree,
    }


def kairos_score_ms(
    candidate: RouteCandidate,
    *, destination: PairTelemetry,
    beta_max: int,
) -> tuple[float, dict[str, float | int | bool]]:
    """Estimate Kairos TTFT and fixed-X TBT feasibility for one candidate."""

    decoder_ms, queue_ms, first_step_ms = _decoder_cost_ms(
        candidate, destination, beta_max=beta_max)
    # The profile TTFT already contains the first response step.  Add only
    # currently visible queue debt to its hardware-calibrated path estimate.
    score = candidate.predicted_ttft_ms + queue_ms
    feasible = True
    mixed_step_ms = first_step_ms
    prompt_tokens_estimate = 0.0
    if candidate.route is GlobalRoute.LOCAL:
        prompt_tokens_estimate = (
            candidate.work.local_prefill_token_ms
            / max(candidate.predicted_ttft_ms, 1e-12)
        )
        prefill_chunk_ms = (
            candidate.predicted_ttft_ms
            * min(KAIROS_CHUNK_TOKENS, prompt_tokens_estimate)
            / max(1.0, prompt_tokens_estimate)
        )
        mixed_step_ms = first_step_ms + prefill_chunk_ms
    return _require_finite("Kairos score", score, minimum=1e-12), {
        "queue_ms": queue_ms,
        "first_decode_step_ms": first_step_ms,
        "mixed_step_ms": mixed_step_ms,
        "prompt_tokens_estimate": prompt_tokens_estimate,
        "chunk_tokens": KAIROS_CHUNK_TOKENS,
        "feasible": feasible,
    }


def _neutral_baseline_config(config: GlobalOrchestratorConfig) -> GlobalOrchestratorConfig:
    pair_count = len(config.capacities)
    tenants = tuple(replace(
        tenant,
        weight=1.0,
        minimum_service_fraction=0.0,
        queue_reservation_slots=0,
        telemetry_stale_grace_ns=0,
        admission_priority=0,
        protected_capacity_fraction=0.0,
        pair_spread_limit=None,
    ) for tenant in config.tenants)
    return replace(
        config,
        tenants=tenants,
        minimum_active_pairs=pair_count,
        maximum_active_pairs=pair_count,
        utilization_penalty_ms=0.0,
        activation_penalty_ms=0.0,
        probe_penalty_ms=0.0,
        endpoint_queue_admission_mode="after_timeout",
        priority_service_lane_mode="disabled",
        priority_service_lane_capacity=0,
        priority_service_lane_min_admission_priority=0,
        priority_service_lane_priority=0,
        decoder_business_admission_mode="disabled",
        decoder_business_background_max_wait_ns=0,
        survivor_capacity_reserve_fraction=0.0,
        survivor_reserve_bypass_min_weight=0.0,
        cross_layer_stagger_max_us=0,
        shared_fabric_control_mode="disabled",
        mesh_receiver_stagger_max_us=0,
        mesh_near_tie_source_balance_mode="disabled",
        mesh_near_tie_source_balance_uncertainty_fraction=0.0,
    )


class PaperBaselineOrchestrator(GlobalOrchestrator):
    """Global carrier whose route score is restricted to one paper policy."""

    def __init__(
        self,
        config: GlobalOrchestratorConfig,
        *,
        policy: str | None = None,
    ) -> None:
        selected = policy or os.environ.get(POLICY_ENV, "")
        if selected not in POLICIES:
            raise ValueError(f"{POLICY_ENV} must select {sorted(POLICIES)}")
        self.paper_policy = selected
        self.paper_score_receipts: dict[str, dict[str, dict[str, object]]] = {}
        super().__init__(_neutral_baseline_config(config))

    def _joint_actuation_plan(self, *args, **kwargs):
        # A baseline may observe its paper-allowed signals but cannot inherit
        # TEMPO's cross-layer throttling, stagger, or shared-fabric budget.
        return None

    def _paper_score(
        self, candidate: RouteCandidate, *, request: GlobalRequest,
    ) -> tuple[float, dict[str, object]]:
        pair = int(candidate.decoder_index)
        source_index = int(candidate.prefill_index)
        destination = self._telemetry[pair]
        beta_max = self._capacities[pair].active_sequences
        if self.paper_policy == NETKV:
            return netkv_score_ms(
                candidate,
                source=self._telemetry[source_index],
                destination=destination,
                beta_max=beta_max,
            )
        score, receipt = kairos_score_ms(
            candidate, destination=destination, beta_max=beta_max)
        receipt = dict(receipt)
        tenant = self._tenants[request.tenant_id]
        receipt["tbt_limit_ms"] = (
            KAIROS_TBT_SAFETY_FACTOR * tenant.tpot_slo_ms)
        receipt["feasible"] = (
            float(receipt["mixed_step_ms"])
            <= float(receipt["tbt_limit_ms"])
        )
        return score, receipt

    def _evaluate_candidate(
        self, candidate: RouteCandidate, *, request: GlobalRequest, now_ns: int,
    ) -> _CandidateEvaluation | RejectedCandidate:
        if self.paper_policy == NETKV and candidate.route is GlobalRoute.LOCAL:
            return RejectedCandidate.from_candidate(
                candidate, "netkv_remote_decode_candidates_only")
        try:
            score, receipt = self._paper_score(candidate, request=request)
        except (KeyError, ValueError):
            return RejectedCandidate.from_candidate(
                candidate, "paper_baseline_required_telemetry_missing")
        self.paper_score_receipts.setdefault(request.request_id, {})[
            str(candidate.edge_id)
        ] = {"score_ms": score, **receipt}

        if self.paper_policy == KAIROS_X512 and candidate.route is GlobalRoute.LOCAL:
            remote_scores = []
            for alternative in request.candidates:
                if alternative.route is not GlobalRoute.REMOTE:
                    continue
                try:
                    remote_score, _ = self._paper_score(
                        alternative, request=request)
                except (KeyError, ValueError):
                    continue
                remote_scores.append(remote_score)
            tenant = self._tenants[request.tenant_id]
            if receipt.get("feasible") is not True:
                return RejectedCandidate.from_candidate(
                    candidate, "kairos_no_tbt_safe_x512_schedule")
            if score > tenant.ttft_slo_ms:
                return RejectedCandidate.from_candidate(
                    candidate, "kairos_decoder_path_ttft_slo")
            if remote_scores and score > KAIROS_ALPHA * min(remote_scores):
                return RejectedCandidate.from_candidate(
                    candidate, "kairos_alpha_margin_keep_prefill_path")

        evaluated = super()._evaluate_candidate(
            candidate, request=request, now_ns=now_ns)
        if isinstance(evaluated, RejectedCandidate):
            return evaluated
        deadline_ms = (
            self._effective_deadline_ns(request) - now_ns) / 1_000_000
        slack_ms = deadline_ms - score
        if slack_ms < 0.0:
            return RejectedCandidate.from_candidate(
                candidate, "paper_baseline_deadline")
        return replace(
            evaluated,
            score_ms=score,
            slack_ms=slack_ms,
            joint_actuation=None,
            cross_layer_scale_required=False,
            shared_scale_suppressed=False,
            receiver_stagger_us=0,
            priority_service_lane=False,
            mesh_near_tie_source_balanced=False,
            mesh_near_tie_score_window_ms=None,
            mesh_near_tie_score_delta_ms=None,
            mesh_source_virtual_service_before=None,
            mesh_edge_virtual_service_before=None,
        )

    def submit(self, request: GlobalRequest, *, now_ns: int):
        decision = super().submit(request, now_ns=now_ns)
        if decision.kind is GlobalDecisionKind.ADMIT:
            decision = replace(
                decision,
                reason=f"{self.paper_policy}_paper_reproduction_route_committed",
                joint_actuation=None,
                receiver_stagger_us=0,
                mesh_near_tie_source_balanced=False,
                mesh_near_tie_score_window_ms=None,
                mesh_near_tie_score_delta_ms=None,
                mesh_source_virtual_service_before=None,
                mesh_edge_virtual_service_before=None,
            )
        return decision

    def snapshot(self, *, now_ns: int) -> dict[str, object]:
        value = super().snapshot(now_ns=now_ns)
        value["paper_baseline"] = {
            "schema": "tempo-paper-baseline-policy-v1",
            "policy": self.paper_policy,
            "netkv_static_nic_bytes_per_s": PERLMUTTER_NIC_BYTES_PER_S,
            "netkv_tp_degree": DEFAULT_TP_DEGREE,
            "kairos_chunk_tokens": KAIROS_CHUNK_TOKENS,
            "kairos_alpha": KAIROS_ALPHA,
            "kairos_tbt_safety_factor": KAIROS_TBT_SAFETY_FACTOR,
            "tempo_business_and_cross_layer_actuation_enabled": False,
        }
        return value


class PaperBaselineCoordinator(GlobalAdmissionCoordinator):
    """Disable TEMPO's hierarchical frontier reduction for SOTA baselines."""

    def __init__(self, orchestrator, telemetry_agent, **kwargs) -> None:
        kwargs["hierarchical_reducer"] = None
        super().__init__(orchestrator, telemetry_agent, **kwargs)

    def status(self) -> dict[str, object]:
        value = super().status()
        value["paper_baseline_policy"] = self.orchestrator.paper_policy
        value["hierarchical_reducer_inherited_from_tempo"] = False
        return value


__all__ = [
    "KAIROS_X512",
    "NETKV",
    "POLICIES",
    "POLICY_ENV",
    "PaperBaselineCoordinator",
    "PaperBaselineOrchestrator",
    "kairos_score_ms",
    "netkv_score_ms",
]
