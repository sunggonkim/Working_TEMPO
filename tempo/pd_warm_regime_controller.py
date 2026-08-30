"""Frozen load-regime controller for validated Qwen2.5-7B warm P/D hits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from tempo.pd_admission import PDRoute
from tempo.pd_cache_affinity import CacheAffinityCatalog


POLICY_ID = "qwen25-7b-tp4x2-warm-regime-controller-12"
VALIDATED_OFFERED_RATES = frozenset({16.0, 32.0, 48.0, 52.0})
HIGH_LOAD_RATE = 52.0


class WarmLoadRegime(str, Enum):
    AFFINITY = "affinity"
    HIGH_LOAD_LOCAL_BYPASS = "high_load_local_bypass"


@dataclass(frozen=True)
class WarmRegimeDecision:
    cache_item: str
    prompt_tokens: int
    output_tokens: int
    affinity_route: PDRoute
    route: PDRoute
    regime: WarmLoadRegime
    reason: str
    policy_id: str = POLICY_ID


class WarmRegimeController:
    """Choose one immutable regime for an explicitly validated offered load.

    This is deliberately not an online rate estimator.  The caller freezes the
    offered-load contract before an epoch, avoiding request-to-request route
    flapping.  Unknown rates and geometries fail closed.
    """

    def __init__(self, offered_rate_per_s: float) -> None:
        if isinstance(offered_rate_per_s, bool) or not isinstance(
                offered_rate_per_s, (int, float)):
            raise TypeError("offered_rate_per_s must be numeric")
        rate = float(offered_rate_per_s)
        if not math.isfinite(rate) or rate not in VALIDATED_OFFERED_RATES:
            raise ValueError("offered load is outside the validated frontier")
        self.offered_rate_per_s = rate
        self.regime = (WarmLoadRegime.HIGH_LOAD_LOCAL_BYPASS
                       if rate == HIGH_LOAD_RATE else WarmLoadRegime.AFFINITY)
        self._catalog = CacheAffinityCatalog()

    def _decision(self, placement) -> WarmRegimeDecision:
        high = self.regime is WarmLoadRegime.HIGH_LOAD_LOCAL_BYPASS
        return WarmRegimeDecision(
            cache_item=placement.cache_item,
            prompt_tokens=placement.prompt_tokens,
            output_tokens=placement.output_tokens,
            affinity_route=placement.route,
            route=(PDRoute.DECODER_LOCAL if high else placement.route),
            regime=self.regime,
            reason=("validated_rate52_remote_warm_hit_bypass"
                    if high else "validated_rate16_32_48_cache_affinity"),
        )

    def seed(self, cache_item: str, prompt_tokens: int,
             output_tokens: int) -> WarmRegimeDecision:
        return self._decision(
            self._catalog.seed(cache_item, prompt_tokens, output_tokens))

    def hit(self, cache_item: str, prompt_tokens: int,
            output_tokens: int) -> WarmRegimeDecision:
        return self._decision(
            self._catalog.hit(cache_item, prompt_tokens, output_tokens))
