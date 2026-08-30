#!/usr/bin/env python3
"""Pressure router v26: rearm its one-remote epoch only after full idle."""

from eval.sota_4node import tempo_pd_capacity_router_v13 as credit
from eval.sota_4node import tempo_pd_pressure_router_v25 as v25


class PressureEpochCore(v25.PressureCore):
    def _release(self, request_id: str) -> None:
        super()._release(request_id)
        with self._lock:
            if not self._local_owned and self._remote_owner is None:
                self._remote_ever_used = False


def main(argv=None) -> int:
    original = credit.CreditCore
    credit.CreditCore = PressureEpochCore
    try:
        return credit.main(argv)
    finally:
        credit.CreditCore = original


if __name__ == "__main__":
    raise SystemExit(main())
