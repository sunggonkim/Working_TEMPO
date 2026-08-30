"""Launch-safe LMCache P/D client with exact original prompt token accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.sota_4node import live_pd_controller_v1 as base
from eval.sota_4node import live_pd_controller_lmcache_v2 as wire


_ORIGINAL_DIRECT = base._run_direct
_ORIGINAL_BODY = base._base_decode_body


def _token_count(decoder_url: str, prompt: str, request_id: str) -> int:
    value = base._request_json(
        decoder_url.rstrip("/") + "/tokenize",
        {"prompt": prompt},
        request_id + "-tokenize",
    )
    tokens = value.get("tokens")
    base._require(isinstance(tokens, list) and tokens, "tokenize returned no tokens")
    return len(tokens)


def _run_direct(decoder_urls_csv: str, prompt: str, request_id: str) -> dict[str, Any]:
    decoder_urls = decoder_urls_csv.split(",")
    base._require(len(decoder_urls) == 2, "two decoder URLs are required")
    pair = wire._pair_index(request_id)
    expected_tokens = _token_count(decoder_urls[pair], prompt, request_id)
    result = _ORIGINAL_DIRECT(decoder_urls[pair], prompt, request_id)
    base._require(result["prompt_tokens"] == expected_tokens, "direct prompt token mismatch")
    result["pair_index"] = pair
    result["live_kv_proof"] = {
        "connector": "LMCacheConnectorV1",
        "conditional_route": "no disagg_spec, no PD transfer",
        "backend": "decoder local computation or LMCache lookup",
        "original_prompt_tokens": expected_tokens,
    }
    return result


def _run_remote(
    proxy_urls_csv: str,
    decoder_urls_csv: str,
    prompt: str,
    request_id: str,
) -> dict[str, Any]:
    decoder_urls = decoder_urls_csv.split(",")
    base._require(len(decoder_urls) == 2, "two decoder URLs are required")
    pair = wire._pair_index(request_id)
    expected_tokens = _token_count(decoder_urls[pair], prompt, request_id)

    # The pinned official proxy forces prefill max_tokens=1 and decode
    # max_tokens=N-1 but does not rewrite min_tokens.  Omit min_tokens only on
    # the proxy request; direct decode retains the exact-token contract.
    def proxy_body(value: str) -> dict[str, Any]:
        body = _ORIGINAL_BODY(value)
        body.pop("min_tokens", None)
        return body

    old_body = base._base_decode_body
    base._base_decode_body = proxy_body
    try:
        result = wire._run_remote(proxy_urls_csv, decoder_urls_csv, prompt, request_id)
    finally:
        base._base_decode_body = old_body
    result["prompt_tokens"] = expected_tokens
    result["live_kv_proof"]["original_prompt_tokens"] = expected_tokens
    return result


def _install() -> None:
    base.BUCKET_REPETITIONS = wire.BUCKET_REPETITIONS
    base._run_remote = _run_remote
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
            "original_prompt_tokens_from_tokenize": True,
        }
    else:
        result = base.combine(base._load(args.baseline), base._load(args.tempo))
        result["baseline_name"] = "official LMCacheConnectorV1 always-remote P/D"
        result["candidate_name"] = "Tempo measured conditional P/D admission"
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
