"""Monotone local-bypass opportunity with a continuously updated raw window."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading


CLASSIFIER_ID = "tempo-pd-online-regime-latched-380"
WINDOW_GAPS = 4
HIGH_LOAD_THRESHOLD_NS = 39_000_000


class OnlineRegime(str, Enum):
    PENDING = "pending"
    AFFINITY = "affinity"
    HIGH_LOAD_LOCAL_BYPASS = "high_load_local_bypass"


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: OnlineRegime
    raw_regime: OnlineRegime
    observations: int
    gap_count: int
    median_gap_ns: int | None
    high_load_latched: bool
    threshold_ns: int = HIGH_LOAD_THRESHOLD_NS
    classifier_id: str = CLASSIFIER_ID


class LatchedOnlineRegimeClassifier:
    def __init__(self) -> None:
        self._last_ns = None
        self._gaps = deque(maxlen=WINDOW_GAPS)
        self._observations = 0
        self._high_load_latched = False
        self._lock = threading.Lock()

    def observe(self, now_ns: int) -> RegimeSnapshot:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative integer")
        with self._lock:
            if self._last_ns is not None:
                gap = now_ns - self._last_ns
                if gap <= 0:
                    raise ValueError("arrival clock must increase")
                self._gaps.append(gap)
            self._last_ns = now_ns
            self._observations += 1
            if len(self._gaps) < WINDOW_GAPS:
                raw = OnlineRegime.PENDING
                median = None
            else:
                ordered = sorted(self._gaps)
                median = (ordered[1] + ordered[2]) // 2
                raw = (OnlineRegime.HIGH_LOAD_LOCAL_BYPASS
                       if median <= HIGH_LOAD_THRESHOLD_NS
                       else OnlineRegime.AFFINITY)
                if raw is OnlineRegime.HIGH_LOAD_LOCAL_BYPASS:
                    self._high_load_latched = True
            regime = (OnlineRegime.HIGH_LOAD_LOCAL_BYPASS
                      if self._high_load_latched else raw)
            return RegimeSnapshot(
                regime=regime,
                raw_regime=raw,
                observations=self._observations,
                gap_count=len(self._gaps),
                median_gap_ns=median,
                high_load_latched=self._high_load_latched,
            )
