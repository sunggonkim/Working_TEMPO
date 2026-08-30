#!/usr/bin/env python3
"""Output16 prompt-length crossover: remote <=1536, otherwise local."""

from __future__ import annotations

import time

from eval.sota_4node import tempo_pd_router_v1 as base
from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_same_server_balanced_router_v70 as balanced
from tempo.pd_admission import PDRequestPhase, PDRoute


class Output16SplitCore(balanced.BalancedSameServerCore):
    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        arm, phase_name = self._arm(request_id)
        if arm != "tempo" or output_tokens != 16:
            return super().decide(
                request_id=request_id, prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                remaining_deadline_ms=remaining_deadline_ms)
        del remaining_deadline_ms
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        route = (PDRoute.REMOTE_PREFILL if prompt_tokens <= 1536
                 else PDRoute.DECODER_LOCAL)
        reason = ("output16_prompt_le_1536_remote" if route is PDRoute.REMOTE_PREFILL
                  else "output16_prompt_gt_1536_local")
        now_ns = time.perf_counter_ns()
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            record = base.RouterDecision(
                request_id=request_id, mode=base.RouterMode.TEMPO_AUTO,
                route=route, reason=f"same_server_tempo_{phase_name}:{reason}",
                workload=workload, profile_id="output16-prompt-split-v99",
                manifest_id="output16-prompt-split-v99", policy_epoch=0,
                remote_advantage_lower_bound_ms=None,
                prompt_tokens=prompt_tokens, potential_kv_bytes=kv_bytes,
                decided_ns=now_ns,
                phase=(PDRequestPhase.REMOTE_SELECTED.value
                       if route is PDRoute.REMOTE_PREFILL
                       else PDRequestPhase.LOCAL_SELECTED.value),
            )
            self._records[request_id] = record
            return record


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = Output16SplitCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
