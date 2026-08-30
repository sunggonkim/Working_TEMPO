"""Fair live-P/D controller with verified connector warmup and TP4 metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eval.sota_4node import live_pd_controller_lmcache_v6 as previous
from eval.sota_4node import live_pd_controller_v1 as base


def _potential_kv_bytes_tp4(prompt_tokens: int) -> dict[str, int]:
    base._require(prompt_tokens > 0, "prompt token count must be positive")
    # TinyLlama: 22 layers, 4 KV heads, head_dim 64, BF16 K+V.
    # TP4 shards four KV heads exactly, so aggregate physical == logical.
    logical = prompt_tokens * 22 * 4 * 64 * 2 * 2
    return {"logical_bytes": logical, "tp4_physical_bytes": logical}


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
        # The first P/D request initializes endpoint and GPU-layout state and is
        # explicitly outside measurement. The second remote request must be
        # output-identical before calibration is allowed to begin.
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
        remote = base._run_remote(
            prefill_url,
            decode_url,
            base._prompt("calibration-remote", bucket, repetitions),
            f"{mode}-cal-remote-{bucket}",
        )
        direct = base._run_direct(
            decode_url,
            base._prompt("calibration-direct", bucket, repetitions),
            f"{mode}-cal-direct-{bucket}",
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
        record = (
            base._run_remote(prefill_url, decode_url, prompt, request_id)
            if route == "remote_prefill_live_kv"
            else base._run_direct(decode_url, prompt, request_id)
        )
        record.update({
            "request_id": request_id,
            "bucket": bucket,
            "repetitions": repetitions,
            "prompt_sha256": base._sha256_text(prompt),
            "potential_kv": _potential_kv_bytes_tp4(record["prompt_tokens"]),
        })
        validation.append(record)

    valid = (
        all(item["verified"] for item in warmup)
        and all(item["output_equivalent"] for item in calibration)
        and all(item["completion_tokens"] == base.OUTPUT_TOKENS for item in validation)
    )
    return {
        "schema": base.SCHEMA,
        "revision": "verified-warmup-tp4-metadata-v7",
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
        "connector_warmup": warmup,
        "controller": {
            "policy": "per-payload measured admission",
            "remote_advantage_margin_ms": base.REMOTE_ADVANTAGE_MARGIN_MS,
            "validation_used_for_tuning": False,
            "decisions": {str(key): value for key, value in decisions.items()},
        },
        "calibration": calibration,
        "validation": validation,
        "valid": valid,
        "limitations": [
            "one post-warmup calibration observation per route and payload bucket",
            "TinyLlama is a mechanism screen, not a large-model SOTA result",
            "admission may choose local recomputation and transfer zero live-P/D bytes",
        ],
    }


def main() -> int:
    old = base.run_lifecycle
    base.run_lifecycle = _run_lifecycle
    try:
        return previous.main()
    finally:
        base.run_lifecycle = old


if __name__ == "__main__":
    raise SystemExit(main())
