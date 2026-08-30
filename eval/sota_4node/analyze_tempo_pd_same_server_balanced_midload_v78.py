#!/usr/bin/env python3
"""Finalize the order-balanced 24 request/s mid-regime verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    old = "tempo_routes_32_local_16_remote"
    if old not in value["gates"]:
        raise ValueError("base high-regime route gate missing")
    del value["gates"][old]
    value["gates"]["tempo_mid_routes_8_local_40_remote"] = value["tempo"]["routes"] == {
        "decoder_local_recompute_or_cache": 8,
        "remote_prefill_live_kv": 40,
    }
    value["schema"] = "tempo-pd-same-server-balanced-midload-analysis-78"
    value["controller_variant"] = {"request_rate_per_s": 24, "arrival_regime": "mid"}
    value["passes"] = all(value["gates"].values())
    value["verdict"] = (
        "promising_order_balanced_midload" if value["passes"]
        else "reject_order_balanced_midload"
    )
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": value["verdict"], "gates": value["gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
