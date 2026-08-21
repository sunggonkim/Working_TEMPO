#!/usr/bin/env python3
"""Canonical order-balanced four-arm client with canonical wire validation."""

import json
import os
import re
import urllib.request
from pathlib import Path
import sys

from eval.sota_4node import run_tempo_pd_elastic_balanced_client_v446 as _prior


ROUTER_SCHEMA = "tempo-elastic-pd-router-canonical"
_CANONICAL_STREAM_MODULE = "eval.sota_4node.run_tempo_pd_elastic_stream_metrics"
FRONTEND_SCHEMA = "tempo-elastic-pd-frontend-canonical-replicated-affinity-3"
_RESET_ENV = "TEMPO_PD_BENCHMARK_RESET_DECODER_APC"
_COLD_MEASURED_ENV = "TEMPO_PD_BENCHMARK_COLD_MEASURED"
_PAIRED_MEASURED_ORDER = (
    "local", "local", "remote", "remote",
    "tempo", "tempo", "predictor", "predictor",
)
_RESET_RUN_ID = re.compile(
    r"-[0-9]{2}_(?:local|remote|predictor|tempo)_r0$")


def _cold_measured_enabled() -> bool:
    raw = os.environ.get(_COLD_MEASURED_ENV, "0")
    if raw not in ("0", "1"):
        raise ValueError(f"{_COLD_MEASURED_ENV} must be 0 or 1")
    return raw == "1"



def _derive(rows, *, arm, replicate, phase, offset):
    """Derive either warm-reused or explicitly cold measured prompt keys."""
    del offset
    if _prior._TOKENIZER is None:
        raise RuntimeError("cache-reuse tokenizer is not initialized")
    if phase not in {"warm", "measured"}:
        raise ValueError("phase must be warm or measured")
    arm_index = _prior.prior._ARMS.index(arm)
    first_chunks = set()
    rewritten = []
    for item, row in enumerate(rows):
        original_ids = _prior._TOKENIZER.encode(
            row["prompt"], add_special_tokens=False)
        if _cold_measured_enabled() and phase == "measured":
            if not 0 <= replicate < 128:
                raise ValueError("cold measured replicate exceeds marker encoding")
            if not 0 <= item < 256:
                raise ValueError("cold measured item exceeds marker encoding")
            marker_id = (1 << 17) | (replicate << 10) | (arm_index << 8) | item
        else:
            marker_id = arm_index * 10_000 + item
        marker_ids = _prior._TOKENIZER.encode(
            _prior.unique._marker(marker_id), add_special_tokens=False)
        if len(marker_ids) > len(original_ids):
            raise ValueError("cache-reuse marker exceeds prompt length")
        candidate_ids = marker_ids + original_ids[len(marker_ids):]
        prompt = _prior._TOKENIZER.decode(
            candidate_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=False)
        checked = _prior._TOKENIZER.encode(prompt, add_special_tokens=False)
        if len(checked) != len(original_ids):
            raise ValueError("cache-reuse prefix changed prompt length")
        chunk = tuple(checked[:256])
        if chunk in first_chunks:
            raise ValueError("duplicate first LMCache chunk within arm block")
        first_chunks.add(chunk)
        value = dict(row)
        value["prompt"] = prompt
        value["request_id"] = (
            f"epd-{arm}-r{replicate}-{phase}-item-{item:02d}")
        rewritten.append(value)
    if len(first_chunks) != len(rows):
        raise ValueError("first-chunk uniqueness count mismatch")
    return rewritten


def _canonicalize_child_command(command):
    """Replace the inherited stream child with the canonical schema wrapper."""
    if not isinstance(command, (list, tuple)):
        return command
    value = list(command)
    try:
        index = value.index("eval.sota_4node.run_tempo_pd_elastic_stream_metrics_v445")
    except ValueError:
        return command
    value[index] = _CANONICAL_STREAM_MODULE
    return value
def _reset_decoder_prefix_cache(base_url: str) -> dict:
    if not isinstance(base_url, str) or not base_url.startswith(
        ("http://", "https://")
    ):
        raise ValueError("reset base URL must be HTTP(S)")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/tempo/reset_decoder_prefix_cache",
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:
        value = json.loads(response.read())
    if not (
        isinstance(value, dict)
        and value.get("schema") == FRONTEND_SCHEMA
        and value.get("success") is True
        and value.get("pair_decoder_resets") == 2
        and value.get("external_cache_reset") is False
    ):
        raise ValueError("decoder APC reset evidence mismatch")
    return value



def _correct_cache_contract(
    *, reset_events: list[dict], paired_measured_order: bool,
    cold_measured: bool,
) -> None:
    """Replace inherited cache-isolation metadata with the reuse contract."""
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    run_id = sys.argv[sys.argv.index("--run-id") + 1]
    phase = "warm" if run_id.endswith("-warmup") else "measured"
    artifact_root = output.parent / f"elastic_balanced_{phase}"
    paths = sorted(artifact_root.glob("*.raw.json")) + [output]
    reset_evidence = [dict(value) for value in reset_events]
    for path in paths:
        artifact = json.loads(path.read_text())
        contract = artifact.get("elastic_balanced_contract")
        if not isinstance(contract, dict):
            raise ValueError(f"{path}: balanced cache contract missing")
        if cold_measured:
            contract["cache_keys_disjoint_across_blocks"] = True
            contract["cache_keys_stable_across_warm_and_measured"] = False
            contract["cache_key_isolation_scope"] = "phase_arm_replicate_and_item"
            contract["warm_preparation"] = "unmeasured_only_no_measured_key_reuse"
            contract["measured_cache_residency"] = "cold_disjoint_prompt_keys"
        else:
            contract["cache_keys_disjoint_across_blocks"] = False
            contract["cache_keys_stable_across_warm_and_measured"] = True
            contract["cache_key_isolation_scope"] = "arm_and_item"
            contract["warm_preparation"] = (
                "official_remote_seed_then_full_source_hit_probe")
            contract["measured_cache_residency"] = "prefill_only_warm"
        contract["decoder_apc_reset_between_arm_pairs"] = (
            paired_measured_order)
        contract["decoder_apc_reset_events"] = reset_evidence
        contract["measured_order_mode"] = (
            "paired_with_quiescent_decoder_apc_reset"
            if paired_measured_order else "balanced_abba")
        orchestration = artifact.get("elastic_balanced_orchestration")
        if isinstance(orchestration, dict):
            orchestration["warm_preparation_is_unmeasured"] = True
            orchestration["cache_keys_stable_across_phases"] = not cold_measured
            orchestration["decoder_apc_reset_between_arm_pairs"] = (
                paired_measured_order)
            orchestration["decoder_apc_reset_events"] = reset_evidence
            orchestration["measured_order_mode"] = contract[
                "measured_order_mode"]
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def main() -> int:
    # v446 delegates each arm to a fresh Python process. A parent-process
    # monkey-patch cannot cross that boundary, so patch the exact subprocess
    # command builder at the boundary and restore it after the run.
    raw_reset = os.environ.get(_RESET_ENV, "0")
    if raw_reset not in ("0", "1"):
        raise ValueError(f"{_RESET_ENV} must be 0 or 1")
    reset_requested = raw_reset == "1"
    cold_measured = _cold_measured_enabled()
    run_id = sys.argv[sys.argv.index("--run-id") + 1]
    measured = not run_id.endswith("-warmup")
    paired_measured_order = reset_requested and measured
    if reset_requested and os.environ.get(
        "TEMPO_VLLM_DECODER_PREFIX_CACHING", "0"
    ) != "1":
        raise ValueError("decoder APC reset requires decoder prefix caching")

    child_owner = _prior.prior
    old_derive = _prior._derive
    old_run = child_owner.subprocess.run
    old_order = child_owner._MEASURED_ORDER
    reset_events = []
    _prior._derive = _derive
    if paired_measured_order:
        child_owner._MEASURED_ORDER = _PAIRED_MEASURED_ORDER
    base_url = sys.argv[sys.argv.index("--base-url") + 1]

    def canonical_run(command, *args, **kwargs):
        value = _canonicalize_child_command(command)
        if paired_measured_order and isinstance(value, list):
            child_run_id = value[value.index("--run-id") + 1]
            if _RESET_RUN_ID.search(child_run_id):
                evidence = dict(_reset_decoder_prefix_cache(base_url))
                evidence["child_run_id"] = child_run_id
                reset_events.append(evidence)
        return old_run(value, *args, **kwargs)

    child_owner.subprocess.run = canonical_run
    try:
        status = _prior.main()
        if paired_measured_order and len(reset_events) != 4:
            raise ValueError("expected one decoder APC reset per arm pair")
        _correct_cache_contract(
            reset_events=reset_events,
            paired_measured_order=paired_measured_order,
            cold_measured=cold_measured,
        )
        return status
    finally:
        _prior._derive = old_derive
        child_owner.subprocess.run = old_run
        child_owner._MEASURED_ORDER = old_order


if __name__ == "__main__":
    raise SystemExit(main())
