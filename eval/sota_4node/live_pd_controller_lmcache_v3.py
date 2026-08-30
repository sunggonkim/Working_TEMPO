"""Recursion-safe launch entry for the LMCache live-P/D controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.sota_4node import live_pd_controller_v1 as base
from eval.sota_4node import live_pd_controller_lmcache_v2 as wire


_ORIGINAL_DIRECT = base._run_direct


def _run_direct(decoder_urls_csv: str, prompt: str, request_id: str) -> dict[str, Any]:
    decoder_urls = decoder_urls_csv.split(",")
    base._require(len(decoder_urls) == 2, "two decoder URLs are required")
    pair = wire._pair_index(request_id)
    result = _ORIGINAL_DIRECT(decoder_urls[pair], prompt, request_id)
    result["pair_index"] = pair
    result["live_kv_proof"] = {
        "connector": "LMCacheConnectorV1",
        "conditional_route": "no disagg_spec, no PD transfer",
        "backend": "decoder local computation or LMCache lookup",
    }
    return result


def _install() -> None:
    base.BUCKET_REPETITIONS = wire.BUCKET_REPETITIONS
    base._run_remote = wire._run_remote
    base._run_direct = _run_direct


def main() -> int:
    _install()
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--mode", choices=("lmcache_always_remote", "tempo_admission"), required=True)
    run.add_argument("--proxy-urls", required=True)
    run.add_argument("--decoder-urls", required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("combine")
    merge.add_argument("--baseline", type=Path, required=True)
    merge.add_argument("--tempo", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = base.run_lifecycle(
            mode=args.mode,
            prefill_url=args.proxy_urls,
            decode_url=args.decoder_urls,
            model_path=args.model.resolve(),
        )
        result["implementation"] = {
            "connector": "LMCacheConnectorV1",
            "proxy": "pinned LMCache examples/disagg_prefill/disagg_proxy_server.py",
            "topology": "two replicas of single-node TP4 prefill -> TP4 decode",
        }
    else:
        result = base.combine(base._load(args.baseline), base._load(args.tempo))
        result["baseline_name"] = "official LMCacheConnectorV1 always-remote P/D"
        result["candidate_name"] = "Tempo measured conditional P/D admission"
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
