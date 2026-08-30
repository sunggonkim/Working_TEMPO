"""Unified evidence-backed cold/miss and warm/hit P/D controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading

from tempo.pd_admission import PDRoute
from tempo.pd_cache_affinity import CacheAffinityCatalog, POLICY_ID as AFFINITY_POLICY_ID
from tempo.pd_regime_controller import AdmissionRoute, PairArrivalRegimeController
from tempo.pd_workload_policy import FrozenPDPolicy


CONTROLLER_ID = "tempo-pd-hybrid-controller-2"


class CachePhase(str, Enum):
    MISS = "miss"
    WARM_SEED = "warm_seed"
    WARM_HIT = "warm_hit"


@dataclass(frozen=True)
class HybridDecision:
    request_id: str
    route: PDRoute
    reason: str
    cache_phase: CachePhase
    policy_id: str


class HybridPDController:
    """Pair-local controller; caller supplies monotonic arrival time and cache phase."""

    def __init__(self) -> None:
        self._cold_policy = FrozenPDPolicy()
        self._cold_controllers: dict[int, PairArrivalRegimeController] = {}
        self._catalog = CacheAffinityCatalog()
        self._active: dict[str, PairArrivalRegimeController | None] = {}
        self._lock = threading.Lock()

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               now_ns: int, cache_phase: CachePhase,
               cache_item: str | None = None) -> HybridDecision:
        if not isinstance(cache_phase, CachePhase):
            raise TypeError("cache_phase must be CachePhase")
        with self._lock:
            if request_id in self._active:
                raise ValueError("duplicate active request_id")
            if cache_phase is CachePhase.WARM_SEED:
                if cache_item is None:
                    raise ValueError("warm seed requires cache_item")
                placement = self._catalog.seed(cache_item, prompt_tokens, output_tokens)
                decision = HybridDecision(
                    request_id, placement.route, "cache_affinity_warm_seed",
                    cache_phase, AFFINITY_POLICY_ID)
                owner = None
            elif cache_phase is CachePhase.WARM_HIT:
                if cache_item is None:
                    raise ValueError("warm hit requires cache_item")
                placement = self._catalog.hit(cache_item, prompt_tokens, output_tokens)
                decision = HybridDecision(
                    request_id, placement.route, "cache_affinity_warm_hit",
                    cache_phase, AFFINITY_POLICY_ID)
                owner = None
            else:
                if cache_item is not None:
                    raise ValueError("cache miss must not claim a cache_item")
                if self._cold_policy.direct_local(prompt_tokens, output_tokens):
                    decision = HybridDecision(
                        request_id, PDRoute.DECODER_LOCAL,
                        f"output{output_tokens}_direct_local_fast_path",
                        cache_phase, CONTROLLER_ID)
                    owner = None
                else:
                    self._cold_policy.validate_controller_workload(
                        prompt_tokens, output_tokens)
                    controller = self._cold_controllers.setdefault(
                        output_tokens, self._cold_policy.controller(output_tokens))
                    core = controller.decide(
                        request_id, now_ns,
                        force_local=self._cold_policy.force_local(
                            prompt_tokens, output_tokens))
                    route = (PDRoute.REMOTE_PREFILL
                             if core.route is AdmissionRoute.REMOTE
                             else PDRoute.DECODER_LOCAL)
                    decision = HybridDecision(
                        request_id, route, core.reason, cache_phase, CONTROLLER_ID)
                    owner = controller
            self._active[request_id] = owner
            return decision

    def complete(self, request_id: str) -> None:
        with self._lock:
            if request_id not in self._active:
                raise KeyError(request_id)
            owner = self._active.pop(request_id)
            if owner is not None:
                owner.release(request_id)
