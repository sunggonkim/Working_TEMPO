#!/usr/bin/env python3
"""Experimental cap=4 binding for the local-credit online router."""

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_same_server_online_regime_local_credit_v326 as cap5


POLICY_ID = "tempo-pd-online-regime-local-credit4-330"


def main(argv=None):
    old_core = credit.CreditCore
    old_cap = cap5.LOCAL_INFLIGHT_CAP
    old_policy = cap5.POLICY_ID
    credit.CreditCore = cap5.LocalCreditCore
    cap5.LOCAL_INFLIGHT_CAP = 4
    cap5.POLICY_ID = POLICY_ID
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = old_core
        cap5.LOCAL_INFLIGHT_CAP = old_cap
        cap5.POLICY_ID = old_policy


if __name__ == "__main__":
    raise SystemExit(main())
