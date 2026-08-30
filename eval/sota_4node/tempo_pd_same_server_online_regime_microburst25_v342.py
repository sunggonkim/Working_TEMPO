#!/usr/bin/env python3
"""25ms microburst threshold binding for the integrated controller."""

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_same_server_online_regime_microburst_credit_v335 as core

POLICY_ID = "tempo-pd-online-regime-microburst25-credit5-342"

def main(argv=None):
    old_core = credit.CreditCore
    old_threshold = core.MICROBURST_THRESHOLD_NS
    old_policy = core.POLICY_ID
    credit.CreditCore = core.MicroburstCreditCore
    core.MICROBURST_THRESHOLD_NS = 25_000_000
    core.POLICY_ID = POLICY_ID
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = old_core
        core.MICROBURST_THRESHOLD_NS = old_threshold
        core.POLICY_ID = old_policy

if __name__ == "__main__": raise SystemExit(main())
