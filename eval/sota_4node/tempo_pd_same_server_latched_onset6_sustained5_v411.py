#!/usr/bin/env python3
"""Latched bypass with cap six at burst onset and cap five when sustained."""

from dataclasses import replace

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_router_v1 as router_base
from eval.sota_4node import tempo_pd_same_server_latched_router_v381 as latched
from tempo.pd_admission import PDRequestPhase, PDRoute


POLICY_ID = "tempo-pd-latched-onset6-sustained5-411"
ONSET_CAP = 6
SUSTAINED_CAP = 5
ONSET_TEMPO_REQUESTS = 8
MICROBURST_THRESHOLD_NS = 25_000_000


class OnsetSustainedCreditCore(latched.LatchedOnlineRegimeCore):
    def __init__(self, config, manifest=None, *, allow_screen_profiles=False):
        super().__init__(config, manifest, allow_screen_profiles=allow_screen_profiles)
        self._local_owned = set()
        self._microburst_tempo_count = 0

    def decide(self, *, request_id: str, prompt_tokens: int, output_tokens: int,
               remaining_deadline_ms: float | None = None):
        record = super().decide(
            request_id=request_id, prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            remaining_deadline_ms=remaining_deadline_ms,
        )
        arm, phase_name = self._arm(request_id)
        if arm != "tempo" or phase_name != "measured":
            return record
        median_text = record.reason.rsplit("median_gap_ns=", 1)[1].split(":", 1)[0]
        median_ns = None if median_text == "none" else int(median_text)
        microburst = median_ns is not None and median_ns <= MICROBURST_THRESHOLD_NS
        with self._lock:
            router_base._require(self._records.get(request_id) == record,
                                 "onset credit record changed concurrently")
            if microburst:
                self._microburst_tempo_count += 1
            else:
                self._microburst_tempo_count = 0
            count = self._microburst_tempo_count
            effective_cap = (ONSET_CAP if microburst and count <= ONSET_TEMPO_REQUESTS
                             else SUSTAINED_CAP)
            before = len(self._local_owned)
            route = record.route
            capped = (microburst and route is PDRoute.DECODER_LOCAL
                      and before >= effective_cap)
            if capped:
                route = PDRoute.REMOTE_PREFILL
            elif route is PDRoute.DECODER_LOCAL:
                self._local_owned.add(request_id)
            marker = ":online_regime="
            router_base._require(marker in record.reason, "latched provenance")
            prefix, suffix = record.reason.rsplit(marker, 1)
            annotated = replace(
                record,
                route=route,
                reason=(f"{prefix}:local_inflight_before={before}:"
                        f"local_cap={effective_cap}:"
                        f"onset_microburst_tempo_count={count}:"
                        f"microburst_threshold_ns={MICROBURST_THRESHOLD_NS}:"
                        f"microburst_credit_active={str(microburst).lower()}:"
                        f"local_capped={str(capped).lower()}{marker}{suffix}"),
                profile_id=POLICY_ID,
                manifest_id=POLICY_ID,
                remote_advantage_lower_bound_ms=(
                    0.0 if capped else record.remote_advantage_lower_bound_ms),
                phase=(PDRequestPhase.REMOTE_SELECTED.value if capped else record.phase),
            )
            self._records[request_id] = annotated
            return annotated

    def _release(self, request_id: str) -> None:
        with self._lock:
            self._local_owned.discard(request_id)
        super()._release(request_id)


def main(argv=None):
    original = credit.CreditCore
    credit.CreditCore = OnsetSustainedCreditCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
