"""Official LMCacheConnectorV1 live-P/D adapter for the v1 admission client."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from eval.sota_4node import live_pd_controller_v1 as base


BUCKET_REPETITIONS = (16, 64, 128)


def _pair_index(request_id: str) -> int:
    try:
        return int(request_id.rsplit("-", 1)[1]) % 2
    except (ValueError, IndexError):
        return 0


def _zero_metrics() -> dict[str, float]:
    return {name: 0.0 for name in base.METRIC_NAMES}


def _stream_proxy(
    url: str,
    prompt: str,
    request_id: str,
    origin_ns: int,
) -> dict[str, Any]:
    body = base._base_decode_body(prompt)
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Connection": "close",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    arrivals_ns: list[int] = []
    pieces: list[str] = []
    usage: dict[str, Any] | None = None
    with urllib.request.urlopen(request, timeout=base.REQUEST_TIMEOUT_S) as response:
        base._require(response.status == 200, f"proxy HTTP status {response.status}")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason") is not None:
                continue
            arrivals_ns.append(time.perf_counter_ns())
            pieces.append(str(choice.get("text", "")))
    finished_ns = time.perf_counter_ns()
    base._require(len(arrivals_ns) == base.OUTPUT_TOKENS, (
        f"LMCache proxy expected {base.OUTPUT_TOKENS} events, got {len(arrivals_ns)}"
    ))
    base._require(usage is not None, "LMCache proxy stream omitted usage")
    # The proxy emits the prefiller's first token itself; decoder usage reports
    # the remaining OUTPUT_TOKENS-1 tokens.
    base._require(int(usage.get("completion_tokens", -1)) in {
        base.OUTPUT_TOKENS - 1, base.OUTPUT_TOKENS
    }, "LMCache proxy completion usage mismatch")
    gaps_ms = [
        (right - left) / 1_000_000.0
        for left, right in zip(arrivals_ns, arrivals_ns[1:])
    ]
    output = "".join(pieces)
    return {
        "http_status": 200,
        "prompt_tokens": int(usage.get("prompt_tokens", -1)),
        "completion_tokens": len(arrivals_ns),
        "output_sha256": base._sha256_text(output),
        "output_text": output,
        "ttft_ms": (arrivals_ns[0] - origin_ns) / 1_000_000.0,
        "e2e_ms": (finished_ns - origin_ns) / 1_000_000.0,
        "tpot_p50_ms": __import__("statistics").median(gaps_ms),
        "tpot_p99_ms": base._percentile(gaps_ms, 0.99),
        "tpot_max_ms": max(gaps_ms),
        "token_arrival_count": len(arrivals_ns),
    }


def _run_remote(
    proxy_urls_csv: str,
    decoder_urls_csv: str,
    prompt: str,
    request_id: str,
) -> dict[str, Any]:
    del decoder_urls_csv
    proxy_urls = proxy_urls_csv.split(",")
    base._require(len(proxy_urls) == 2, "two proxy URLs are required")
    pair = _pair_index(request_id)
    origin_ns = time.perf_counter_ns()
    result = _stream_proxy(proxy_urls[pair], prompt, request_id, origin_ns)
    result.update({
        "route": "official_lmcache_connector_v1_live_pd",
        "pair_index": pair,
        "prefill_ms": None,
        "metrics": {"proxy": _zero_metrics()},
        "live_kv_proof": {
            "connector": "LMCacheConnectorV1",
            "transfer_channel": "nixl",
            "backend": "UCX",
            "disagg_spec_injected_by_official_proxy": True,
        },
    })
    return result


def _run_direct(decoder_urls_csv: str, prompt: str, request_id: str) -> dict[str, Any]:
    decoder_urls = decoder_urls_csv.split(",")
    base._require(len(decoder_urls) == 2, "two decoder URLs are required")
    pair = _pair_index(request_id)
    result = base._run_direct(decoder_urls[pair], prompt, request_id)
    result["pair_index"] = pair
    result["live_kv_proof"] = {
        "connector": "LMCacheConnectorV1",
        "conditional_route": "no disagg_spec, no PD transfer",
        "backend": "decoder local computation or LMCache lookup",
    }
    return result


def _install() -> None:
    base.BUCKET_REPETITIONS = BUCKET_REPETITIONS
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
        }
    else:
        result = base.combine(base._load(args.baseline), base._load(args.tempo))
        result["baseline_name"] = "official LMCacheConnectorV1 always-remote P/D"
        result["candidate_name"] = "Tempo measured conditional P/D admission"
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
