#!/usr/bin/env python3
"""One-factor cap-six variant of the frozen latched controller."""

from eval.sota_4node import tempo_pd_same_server_latched_microburst25_v382 as base


POLICY_ID = "tempo-pd-latched-bypass-rolling-credit6-401"
LOCAL_INFLIGHT_CAP = 6


def main(argv=None):
    old_policy, old_cap = base.POLICY_ID, base.LOCAL_INFLIGHT_CAP
    base.POLICY_ID, base.LOCAL_INFLIGHT_CAP = POLICY_ID, LOCAL_INFLIGHT_CAP
    try:
        return base.main(argv)
    finally:
        base.POLICY_ID, base.LOCAL_INFLIGHT_CAP = old_policy, old_cap


if __name__ == "__main__":
    raise SystemExit(main())
