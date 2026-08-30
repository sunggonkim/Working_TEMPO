"""One-shot, pair-local arrival-regime classifier for warm P/D routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading


CLASSIFIER_ID = "tempo-pd-online-regime-1"
WINDOW_GAPS = 6
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
    threshold_ns: int
    classifier_id: str = CLASSIFIER_ID


class OnlineRegimeClassifier:
    """Classify once from six local-clock gaps, then remain frozen."""

    def __init__(self) -> None:
        self._last_ns: int | None = None
        self._gaps: list[int] = []
        self._regime = OnlineRegime.PENDING
        self._median_gap_ns: int | None = None
        self._lock = threading.Lock()

    def observe(self, now_ns: int) -> RegimeSnapshot:
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a nonnegative integer")
        with self._lock:
            if self._regime is not OnlineRegime.PENDING:
                return self._snapshot()
            if self._last_ns is not None:
                gap = now_ns - self._last_ns
                if gap <= 0:
                    raise ValueError("arrival clock must increase")
                self._gaps.append(gap)
            self._last_ns = now_ns
            if len(self._gaps) == WINDOW_GAPS:
                ordered = sorted(self._gaps)
                self._median_gap_ns = (ordered[2] + ordered[3]) // 2
                self._regime = (
                    OnlineRegime.HIGH_LOAD_LOCAL_BYPASS
                    if self._median_gap_ns <= HIGH_LOAD_THRESHOLD_NS
                    else OnlineRegime.AFFINITY
                )
            return self._snapshot()

    def snapshot(self) -> RegimeSnapshot:
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> RegimeSnapshot:
        return RegimeSnapshot(
            regime=self._regime,
            observations=1 + len(self._gaps) if self._last_ns is not None else 0,
            gap_count=len(self._gaps),
            median_gap_ns=self._median_gap_ns,
            threshold_ns=HIGH_LOAD_THRESHOLD_NS,
        )
