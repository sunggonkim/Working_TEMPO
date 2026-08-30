#!/usr/bin/env python3
"""Run a capacity-matched 16-chunk TEMPO calendar on official LMCache/NIXL.

The canonical LMCache epoch runner uses four MiB-scale chunks.  This research
variant deliberately changes only transfer granularity: each request is split
into sixteen KiB-addressed registered objects, allowing the compiled calendar
to issue sub-token microbursts instead of queueing long indivisible writes.
All correctness, actual-start, deadline, and backlog gates remain those of the
canonical runner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.sota_4node import run_lmcache_epoch_2node as base


KIB = 1 << 10
MICROBURST_CHUNKS_PER_REQUEST = 16
MICROBURST_QUANTA = tuple(
    (pair, chunk)
    for chunk in range(MICROBURST_CHUNKS_PER_REQUEST)
    for pair in range(base.PAIR_COUNT)
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--kv-kib", type=int, default=4096)
    parser.add_argument("--chunk-kib", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--port-base", type=int, default=30100)
    args = parser.parse_args()
    numeric = (
        args.requests,
        args.kv_kib,
        args.chunk_kib,
        args.tokens,
        args.layers,
        args.hidden_size,
        args.context,
    )
    if any(isinstance(value, bool) or value <= 0 for value in numeric):
        parser.error("all workload values must be positive")
    if args.tokens < len(MICROBURST_QUANTA):
        parser.error("tokens must cover all 64 static-serial quanta")
    if args.kv_kib != MICROBURST_CHUNKS_PER_REQUEST * args.chunk_kib:
        parser.error("kv-kib must equal sixteen chunk-kib chunks")
    if args.hidden_size % 8:
        parser.error("hidden-size must be divisible by 8")
    if not 1024 <= args.port_base <= 65535 - base.PAIR_COUNT:
        parser.error("port-base must leave four valid ports")
    return argparse.Namespace(
        output_dir=args.output_dir,
        requests=args.requests,
        kv_mib=args.kv_kib,
        chunk_mib=args.chunk_kib,
        tokens=args.tokens,
        layers=args.layers,
        hidden_size=args.hidden_size,
        context=args.context,
        port_base=args.port_base,
    )


def install_microburst_geometry() -> None:
    base.MIB = KIB
    base.CHUNKS_PER_REQUEST = MICROBURST_CHUNKS_PER_REQUEST
    base.CANONICAL_QUANTA = MICROBURST_QUANTA
    base._parse_args = _parse_args


def main() -> None:
    install_microburst_geometry()
    base.main()


if __name__ == "__main__":
    main()
