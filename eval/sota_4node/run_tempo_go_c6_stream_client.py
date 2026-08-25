#!/usr/bin/env python3
"""C6 stream seam with tenant headers and pre-scoped request identities."""

from __future__ import annotations

from eval.sota_4node import run_tempo_go_c5_stream_client as c5


def main() -> int:
    # The C6 contract materializes the final arm name, business tenant, phase,
    # and ordinal directly into every request ID.  Reuse C5's exact tenant
    # header and terminal handling, but never rewrite those frozen identities.
    original = c5._rewrite_measured_arm_workload
    c5._rewrite_measured_arm_workload = lambda argv: list(argv)
    try:
        return c5.main()
    finally:
        c5._rewrite_measured_arm_workload = original


if __name__ == "__main__":
    raise SystemExit(main())
