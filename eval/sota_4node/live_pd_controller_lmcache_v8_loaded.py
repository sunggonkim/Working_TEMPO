"""Live-P/D admission experiment under a controlled decoder background load."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from eval.sota_4node import live_pd_controller_lmcache_v2 as wire
from eval.sota_4node import live_pd_controller_lmcache_v7 as fair
from eval.sota_4node import live_pd_controller_v1 as base


BACKGROUND_TOKENS = 128
BACKGROUND_HEADSTART_S = 0.15


def _with_background(
    decoder_urls_csv: str,
    pair: int,
    bucket: int,
    tag: str,
    foreground: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    decoder_urls = decoder_urls_csv.split(",")
    base._require(len(decoder_urls) == 2, "two decoder URLs are required")
    holder: dict[str, Any] = {}

    def run_background() -> None:
        started_ns = time.perf_counter_ns()
        try:
            value = base._request_json(
                decoder_urls[pair].rstrip("/") + "/v1/completions",
                {
                    "model": base.SERVED_MODEL,
                    "prompt": base._prompt("decoder-background", bucket, 16),
                    "max_tokens": BACKGROUND_TOKENS,
                    "min_tokens": BACKGROUND_TOKENS,
                    "ignore_eos": True,
                    "temperature": 0.0,
                    "seed": 20260814,
                    "stream": False,
                },
                f"background-{tag}-{pair}",
            )
            holder["response"] = value
        except BaseException as exc:  # propagated on the foreground thread
            holder["error"] = exc
        finally:
            holder["e2e_ms"] = (time.perf_counter_ns() - started_ns) / 1_000_000.0

    worker = threading.Thread(target=run_background, name=f"live-pd-bg-{tag}")
    worker.start()
    time.sleep(BACKGROUND_HEADSTART_S)
    overlap_at_foreground_start = worker.is_alive()
    base._require(overlap_at_foreground_start, "background completed before foreground start")
    foreground_result = foreground()
    worker.join(timeout=base.REQUEST_TIMEOUT_S)
    base._require(not worker.is_alive(), "background request did not finish")
    if "error" in holder:
        raise holder["error"]
    usage = holder["response"].get("usage")
    base._require(isinstance(usage, dict), "background response omitted usage")
    base._require(
        int(usage.get("completion_tokens", -1)) == BACKGROUND_TOKENS,
        "background completion token mismatch",
    )
    foreground_result["background_decode"] = {
        "pair_index": pair,
        "completion_tokens": BACKGROUND_TOKENS,
        "e2e_ms": holder["e2e_ms"],
        "overlap_at_foreground_start": overlap_at_foreground_start,
    }
    return foreground_result


def _run_lifecycle(
    *, mode: str, prefill_url: str, decode_url: str, model_path: Path
) -> dict[str, Any]:
    base._require(mode in {"lmcache_always_remote", "tempo_admission"}, "invalid mode")
    base._require(
        model_path.is_absolute() and (model_path / "config.json").is_file(),
        "model must be an absolute local model directory",
    )

    warmup: list[dict[str, Any]] = []
    for pair in range(2):
        prompt = base._prompt("connector-warmup", pair, 16)
        direct = base._run_direct(decode_url, prompt, f"{mode}-warm-direct-{pair}")
        base._run_remote(prefill_url, decode_url, prompt, f"{mode}-warm-handshake-{pair}")
        verified = base._run_remote(
            prefill_url, decode_url, prompt, f"{mode}-warm-verify-{pair}"
        )
        base._require(
            direct["output_sha256"] == verified["output_sha256"],
            f"pair {pair} connector warmup output mismatch",
        )
        warmup.append({
            "pair_index": pair,
            "direct_output_sha256": direct["output_sha256"],
            "verified_remote_output_sha256": verified["output_sha256"],
            "verified": True,
        })

    calibration: list[dict[str, Any]] = []
    decisions: dict[int, str] = {}
    for bucket, repetitions in enumerate(base.BUCKET_REPETITIONS):
        pair = bucket % 2
        remote = _with_background(
            decode_url, pair, bucket, f"{mode}-cal-remote-{bucket}",
            lambda b=bucket, r=repetitions: base._run_remote(
                prefill_url,
                decode_url,
                base._prompt("calibration-remote", b, r),
                f"{mode}-cal-remote-{b}",
            ),
        )
        direct = _with_background(
            decode_url, pair, bucket, f"{mode}-cal-direct-{bucket}",
            lambda b=bucket, r=repetitions: base._run_direct(
                decode_url,
                base._prompt("calibration-direct", b, r),
                f"{mode}-cal-direct-{b}",
            ),
        )
        base._require(
            remote["output_sha256"] == direct["output_sha256"],
            f"bucket {bucket} calibration output mismatch",
        )
        advantage = direct["e2e_ms"] - remote["e2e_ms"]
        chosen = (
            "remote_prefill_live_kv"
            if advantage >= base.REMOTE_ADVANTAGE_MARGIN_MS
            else "decoder_local_recompute_or_cache"
        )
        decisions[bucket] = chosen
        calibration.append({
            "bucket": bucket,
            "repetitions": repetitions,
            "remote": remote,
            "direct": direct,
            "remote_advantage_ms": advantage,
            "frozen_decision": chosen,
            "output_equivalent": True,
        })

    validation: list[dict[str, Any]] = []
    for bucket, repetitions in enumerate(base.BUCKET_REPETITIONS):
        prompt = base._prompt("validation", bucket, repetitions)
        request_id = f"live-pd-validation-{bucket}"
        route = (
            "remote_prefill_live_kv"
            if mode == "lmcache_always_remote"
            else decisions[bucket]
        )
        pair = bucket % 2
        record = _with_background(
            decode_url,
            pair,
            bucket,
            f"{mode}-validation-{bucket}",
            (
                lambda p=prompt, q=request_id: base._run_remote(
                    prefill_url, decode_url, p, q
                )
                if route == "remote_prefill_live_kv"
                else lambda p=prompt, q=request_id: base._run_direct(decode_url, p, q)
            ),
        )
        record.update({
            "request_id": request_id,
            "bucket": bucket,
            "repetitions": repetitions,
            "prompt_sha256": base._sha256_text(prompt),
            "potential_kv": fair._potential_kv_bytes_tp4(record["prompt_tokens"]),
        })
        validation.append(record)

    valid = (
        all(item["verified"] for item in warmup)
        and all(item["output_equivalent"] for item in calibration)
        and all(item["completion_tokens"] == base.OUTPUT_TOKENS for item in validation)
        and all(item["background_decode"]["overlap_at_foreground_start"] for item in validation)
    )
    return {
        "schema": base.SCHEMA,
        "revision": "loaded-crossover-v8",
        "mode": mode,
        "evidence": "actual_vllm_disaggregated_prefill_live_kv",
        "model": str(model_path),
        "served_model": base.SERVED_MODEL,
        "topology": {
            "nodes": 4,
            "gpus": 16,
            "replicas": 2,
            "per_replica": {
                "prefill": {"nodes": 1, "gpus": 4, "tensor_parallel_size": 4},
                "decode": {"nodes": 1, "gpus": 4, "tensor_parallel_size": 4},
            },
        },
        "background_workload": {
            "max_num_seqs": 2,
            "completion_tokens": BACKGROUND_TOKENS,
            "headstart_ms": BACKGROUND_HEADSTART_S * 1000.0,
            "same_for_every_measured_route": True,
        },
        "connector_warmup": warmup,
        "controller": {
            "policy": "per-payload measured admission under decoder load",
            "remote_advantage_margin_ms": base.REMOTE_ADVANTAGE_MARGIN_MS,
            "validation_used_for_tuning": False,
            "decisions": {str(key): value for key, value in decisions.items()},
        },
        "calibration": calibration,
        "validation": validation,
        "valid": valid,
        "limitations": [
            "one post-warmup calibration observation per route and payload bucket",
            "synthetic controlled decoder load with one concurrent request",
            "TinyLlama is a mechanism screen, not a large-model SOTA result",
        ],
    }


def main() -> int:
    old = base.run_lifecycle
    base.run_lifecycle = _run_lifecycle
    try:
        return fair.previous.main()
    finally:
        base.run_lifecycle = old


if __name__ == "__main__":
    raise SystemExit(main())
