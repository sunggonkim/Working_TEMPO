#!/usr/bin/env python3
"""Evidence-gated TEMPO router: fail local when remote proof does not pass."""

from __future__ import annotations

import time

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_admission import PDRequestPhase, PDRoute


EVIDENCE_ID = "unique-head-r16-r32-remote-noninferiority-failed-v28"


class EvidenceFailLocalCore(credit.CreditCore):
    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        del remaining_deadline_ms
        base._require(isinstance(request_id, str) and request_id.strip(),
                      "request_id must be nonempty")
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens
        )
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            record = base.RouterDecision(
                request_id=request_id,
                mode=base.RouterMode.TEMPO_AUTO,
                route=PDRoute.DECODER_LOCAL,
                reason="fail_local_remote_correctness_or_5ms_gate_unproven",
                workload=workload,
                profile_id=EVIDENCE_ID,
                manifest_id=EVIDENCE_ID,
                policy_epoch=0,
                remote_advantage_lower_bound_ms=-57.0,
                prompt_tokens=prompt_tokens,
                potential_kv_bytes=kv_bytes,
                decided_ns=time.perf_counter_ns(),
                phase=PDRequestPhase.LOCAL_SELECTED.value,
            )
            self._records[request_id] = record
            return record


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = EvidenceFailLocalCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
