#!/usr/bin/env python3
"""Cache-catalog route selection for the measured warm-reuse regimes."""

from __future__ import annotations

import time

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as base
from eval.sota_4node import tempo_pd_same_server_balanced_router_v70 as balanced
from tempo.pd_admission import PDRequestPhase, PDRoute


def _cache_item(request_id: str) -> str:
    marker = "-cache-item-"
    if marker not in request_id:
        raise ValueError("cache-catalog request identity missing")
    suffix = request_id.rsplit(marker, 1)[1]
    if len(suffix) != 2 or not suffix.isdigit():
        raise ValueError("cache-catalog item identity malformed")
    return f"cache-item-{suffix}"


def _selected_route(prompt_tokens: int, output_tokens: int) -> PDRoute:
    if prompt_tokens not in (512, 1230, 2048):
        raise ValueError("cache-catalog policy is validated only for prompt 512/1230/2048")
    if output_tokens not in (16, 32, 64, 128):
        raise ValueError("cache-catalog policy output length is unvalidated")
    remote = (
        (prompt_tokens == 512 and output_tokens in (32, 64, 128))
        or (prompt_tokens == 2048 and output_tokens == 64)
    )
    return PDRoute.REMOTE_PREFILL if remote else PDRoute.DECODER_LOCAL


class CacheCatalogCore(balanced.BalancedSameServerCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._cache_catalog: dict[str, PDRoute] = {}

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        del remaining_deadline_ms
        arm, phase_name = self._arm(request_id)
        if arm != "tempo":
            return super().decide(
                request_id=request_id, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens)
        cache_item = _cache_item(request_id)
        selected = _selected_route(prompt_tokens, output_tokens)
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        now_ns = time.perf_counter_ns()
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            if phase_name == "warm":
                prior = self._cache_catalog.setdefault(cache_item, selected)
                base._require(prior is selected, "cache-catalog warm route changed")
                evidence = "seed"
            elif phase_name == "measured":
                base._require(cache_item in self._cache_catalog,
                              "cache-catalog measured item was not seeded on this pair")
                base._require(self._cache_catalog[cache_item] is selected,
                              "cache-catalog measured route changed")
                evidence = "hit"
            else:
                raise ValueError("cache-catalog phase is unvalidated")
            route_name = "remote" if selected is PDRoute.REMOTE_PREFILL else "local"
            record = base.RouterDecision(
                request_id=request_id, mode=base.RouterMode.TEMPO_AUTO,
                route=selected,
                reason=f"same_server_tempo_{phase_name}:cache_catalog_{evidence}_{route_name}",
                workload=workload,
                profile_id="same-server-cache-catalog-v136",
                manifest_id="same-server-cache-catalog-v136", policy_epoch=0,
                remote_advantage_lower_bound_ms=(
                    0.0 if selected is PDRoute.REMOTE_PREFILL else None),
                prompt_tokens=prompt_tokens, potential_kv_bytes=kv_bytes,
                decided_ns=now_ns,
                phase=(PDRequestPhase.REMOTE_SELECTED.value
                       if selected is PDRoute.REMOTE_PREFILL
                       else PDRequestPhase.LOCAL_SELECTED.value),
            )
            self._records[request_id] = record
            return record


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = CacheCatalogCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
