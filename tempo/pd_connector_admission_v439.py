"""Pure scheduler-side admission state for a vLLM P/D KV connector.

The state machine makes exactly one decision per request.  It latches the
opportunity to recompute on the decoder after high load is observed, while the
local-compute credit is enforced only during a current microburst.
"""

from dataclasses import dataclass
import threading

from tempo.pd_online_regime_latched_v380 import (
    LatchedOnlineRegimeClassifier,
    OnlineRegime,
)


POLICY_ID = "tempo-pd-connector-latched-credit6-439"
DEFAULT_MICROBURST_THRESHOLD_NS = 25_000_000
DEFAULT_LOCAL_INFLIGHT_CAP = 6


@dataclass(frozen=True)
class ConnectorAdmissionDecision:
    request_id: str
    route: str
    regime: str
    raw_regime: str
    observations: int
    median_gap_ns: int | None
    high_load_latched: bool
    microburst_credit_active: bool
    local_inflight_before: int
    local_cap: int
    local_capped: bool
    policy_id: str = POLICY_ID


class ConnectorAdmissionState:
    def __init__(
        self,
        *,
        local_inflight_cap: int = DEFAULT_LOCAL_INFLIGHT_CAP,
        microburst_threshold_ns: int = DEFAULT_MICROBURST_THRESHOLD_NS,
    ) -> None:
        if type(local_inflight_cap) is not int or local_inflight_cap <= 0:
            raise ValueError("local_inflight_cap must be a positive integer")
        if type(microburst_threshold_ns) is not int or microburst_threshold_ns <= 0:
            raise ValueError("microburst_threshold_ns must be a positive integer")
        self.local_inflight_cap = local_inflight_cap
        self.microburst_threshold_ns = microburst_threshold_ns
        self._classifier = LatchedOnlineRegimeClassifier()
        self._local_owned: set[str] = set()
        self._decisions: dict[str, ConnectorAdmissionDecision] = {}
        self._lock = threading.Lock()

    def decide(self, request_id: str, now_ns: int) -> ConnectorAdmissionDecision:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        with self._lock:
            if cached := self._decisions.get(request_id):
                return cached
            snapshot = self._classifier.observe(now_ns)
            median = snapshot.median_gap_ns
            microburst = median is not None and median <= self.microburst_threshold_ns
            before = len(self._local_owned)
            high = snapshot.regime is OnlineRegime.HIGH_LOAD_LOCAL_BYPASS
            capped = high and microburst and before >= self.local_inflight_cap
            local = high and not capped
            route = "decoder_local_recompute" if local else "remote_kv_pull"
            if local:
                self._local_owned.add(request_id)
            decision = ConnectorAdmissionDecision(
                request_id=request_id,
                route=route,
                regime=snapshot.regime.value,
                raw_regime=snapshot.raw_regime.value,
                observations=snapshot.observations,
                median_gap_ns=median,
                high_load_latched=snapshot.high_load_latched,
                microburst_credit_active=microburst,
                local_inflight_before=before,
                local_cap=self.local_inflight_cap,
                local_capped=capped,
            )
            self._decisions[request_id] = decision
            return decision

    def finish(self, request_id: str) -> None:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        with self._lock:
            self._local_owned.discard(request_id)

    @property
    def local_inflight(self) -> int:
        with self._lock:
            return len(self._local_owned)
