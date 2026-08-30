#!/usr/bin/env python3
"""Per-request fixed-local, TEMPO, and LMCache routing in one server epoch."""

from __future__ import annotations

import time

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as base
from tempo.pd_admission import PDRequestPhase, PDRoute
from tempo.pd_regime_controller import AdmissionRoute
from tempo.pd_workload_policy import FrozenPDPolicy


class SameServerCore(credit.CreditCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._tempo_controllers = {}
        self._tempo_owned: dict[str, str] = {}

    @staticmethod
    def _arm(request_id: str) -> tuple[str, str]:
        for arm in ("local", "tempo", "remote"):
            for phase in ("warm", "measured"):
                prefix = f"ss-{arm}-{phase}-"
                if request_id.startswith(prefix):
                    return arm, phase
        raise ValueError("same-server request ID has no explicit arm/phase prefix")

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        del remaining_deadline_ms
        base._require(isinstance(request_id, str) and request_id.strip(),
                      "request_id must be nonempty")
        workload, kv_bytes = self.classify(
            prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        arm, phase_name = self._arm(request_id)
        now_ns = time.perf_counter_ns()
        with self._lock:
            base._require(request_id not in self._records, "duplicate request_id")
            if arm == "local":
                route = PDRoute.DECODER_LOCAL
                reason = f"same_server_fixed_local_{phase_name}"
            elif arm == "remote":
                route = PDRoute.REMOTE_PREFILL
                reason = f"same_server_lmcache_remote_{phase_name}"
            else:
                policy = FrozenPDPolicy()
                if policy.direct_local(prompt_tokens, output_tokens):
                    route = PDRoute.DECODER_LOCAL
                    reason = (
                        f"same_server_tempo_{phase_name}:"
                        f"output{output_tokens}_direct_local_fast_path"
                    )
                else:
                    policy.validate_controller_workload(
                        prompt_tokens, output_tokens)
                    epoch = f"{phase_name}:{output_tokens}"
                    controller = self._tempo_controllers.get(epoch)
                    if controller is None:
                        controller = policy.controller(output_tokens)
                        self._tempo_controllers[epoch] = controller
                    admission = controller.decide(
                        request_id,
                        now_ns,
                        force_local=policy.force_local(prompt_tokens, output_tokens),
                    )
                    route = (PDRoute.REMOTE_PREFILL
                             if admission.route is AdmissionRoute.REMOTE
                             else PDRoute.DECODER_LOCAL)
                    reason = (
                        f"same_server_tempo_{phase_name}:{admission.reason}:"
                        f"mean_pair_interval_ns={admission.observed_mean_pair_interval_ns}"
                    )
                    self._tempo_owned[request_id] = epoch
            record = base.RouterDecision(
                request_id=request_id, mode=base.RouterMode.TEMPO_AUTO,
                route=route, reason=reason, workload=workload,
                profile_id="same-server-arrival-regime-v61",
                manifest_id="same-server-arrival-regime-v61", policy_epoch=0,
                remote_advantage_lower_bound_ms=(37.0 if route is PDRoute.REMOTE_PREFILL else None),
                prompt_tokens=prompt_tokens, potential_kv_bytes=kv_bytes,
                decided_ns=now_ns,
                phase=(PDRequestPhase.REMOTE_SELECTED.value
                       if route is PDRoute.REMOTE_PREFILL
                       else PDRequestPhase.LOCAL_SELECTED.value),
            )
            self._records[request_id] = record
            return record

    def _release(self, request_id: str) -> None:
        with self._lock:
            epoch = self._tempo_owned.pop(request_id, None)
            if epoch is not None:
                base._require(epoch in self._tempo_controllers,
                              "TEMPO controller epoch missing during release")

                self._tempo_controllers[epoch].release(request_id)

def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = SameServerCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
