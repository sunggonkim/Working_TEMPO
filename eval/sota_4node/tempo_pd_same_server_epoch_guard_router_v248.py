#!/usr/bin/env python3
"""Live router with a pre-measurement epoch circuit breaker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as router_base
from eval.sota_4node import tempo_pd_same_server_hybrid_phase_router_v181 as phase
from tempo.pd_admission import PDRequestPhase, PDRoute


MODE_SCHEMA = "tempo-pd-epoch-mode-248"
MODE_ENV = "TEMPO_PD_EPOCH_MODE_FILE"


def load_epoch_mode(path: Path) -> str:
    value = json.loads(path.read_text())
    if value.get("schema") != MODE_SCHEMA:
        raise ValueError("epoch mode schema changed")
    selected = value.get("selected_mode")
    if selected not in ("policy8", "fixed_local"):
        raise ValueError("epoch selected mode invalid")
    if value.get("calibration_replicates_per_candidate") != 3:
        raise ValueError("epoch calibration replicate contract changed")
    return selected


class EpochGuardCore(phase.FullPhaseHybridCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        raw_path = os.environ.get(MODE_ENV)
        if not raw_path:
            raise ValueError(f"{MODE_ENV} is required")
        self._epoch_mode_path = Path(raw_path).resolve()
        self._epoch_mode: str | None = None

    @staticmethod
    def _arm(request_id: str) -> tuple[str, str]:
        if request_id.startswith("ssb-tempo-r0-cold-"):
            return "tempo", "cold"
        for arm in ("local", "tempo", "remote"):
            for replicate in (0, 1, 2):
                for phase_name in ("warm", "measured"):
                    if request_id.startswith(
                        f"ssb-{arm}-r{replicate}-{phase_name}-"
                    ):
                        return arm, phase_name
        raise ValueError("epoch-guard request ID has no explicit arm/phase prefix")

    def _selected_mode(self) -> str:
        if self._epoch_mode is None:
            self._epoch_mode = load_epoch_mode(self._epoch_mode_path)
        return self._epoch_mode

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        arm, phase_name = self._arm(request_id)
        if arm != "tempo" or phase_name != "measured":
            return super().decide(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                remaining_deadline_ms=remaining_deadline_ms,
            )
        if self._selected_mode() == "policy8":
            return super().decide(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                remaining_deadline_ms=remaining_deadline_ms,
            )

        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        now_ns = time.perf_counter_ns()
        with self._lock:
            router_base._require(
                request_id not in self._records, "duplicate request_id")
            record = router_base.RouterDecision(
                request_id=request_id,
                mode=router_base.RouterMode.TEMPO_AUTO,
                route=PDRoute.DECODER_LOCAL,
                reason="same_server_tempo_measured:epoch_guard_fixed_local",
                workload=workload,
                profile_id="tempo-pd-epoch-guard-v248",
                manifest_id="tempo-pd-epoch-guard-v248",
                policy_epoch=0,
                remote_advantage_lower_bound_ms=None,
                prompt_tokens=prompt_tokens,
                potential_kv_bytes=kv_bytes,
                decided_ns=now_ns,
                phase=PDRequestPhase.LOCAL_SELECTED.value,
            )
            self._records[request_id] = record
            return record


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = EpochGuardCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
