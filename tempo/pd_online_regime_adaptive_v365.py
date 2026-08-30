"""Continuously updated four-gap pair-local arrival classifier."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading


CLASSIFIER_ID = "tempo-pd-online-regime-adaptive-365"
WINDOW_GAPS = 4
HIGH_LOAD_THRESHOLD_NS = 39_000_000


class OnlineRegime(str, Enum):
    PENDING = "pending"
    AFFINITY = "affinity"
    HIGH_LOAD_LOCAL_BYPASS = "high_load_local_bypass"


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: OnlineRegime
    observations: int
    gap_count: int
    median_gap_ns: int | None
    threshold_ns: int = HIGH_LOAD_THRESHOLD_NS
    classifier_id: str = CLASSIFIER_ID


class AdaptiveOnlineRegimeClassifier:
    def __init__(self) -> None:
        self._last_ns = None
        self._gaps = deque(maxlen=WINDOW_GAPS)
        self._observations = 0
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
                return RegimeSnapshot(
                    OnlineRegime.PENDING, self._observations,
                    len(self._gaps), None,
                )
            ordered = sorted(self._gaps)
            median = (ordered[1] + ordered[2]) // 2
            regime = (
                OnlineRegime.HIGH_LOAD_LOCAL_BYPASS
                if median <= HIGH_LOAD_THRESHOLD_NS
                else OnlineRegime.AFFINITY
            )
            return RegimeSnapshot(
                regime, self._observations, len(self._gaps), median,
            )
