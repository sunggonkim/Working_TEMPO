#!/usr/bin/env python3
"""Canonical four-node actual-vLLM Elastic-PD lifecycle entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from eval.sota_4node import vllm_lmcache_chunk256_node_v7 as chunk256
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v445 as v445
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v446 as v446
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v447 as v447
from eval.sota_4node import vllm_lmcache_elastic_pd_node_v449 as v449
from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v1 as perf


_BACKGROUND_START_ENV = "TEMPO_CXI_BACKGROUND_START_FILE"
_CACHE_CONTROL_MODULE = "eval.sota_4node.vllm_tempo_cache_control"
_ORIGINAL_ROUTER_COMMAND = v449._ORIGINAL_ROUTER_COMMAND
_ORIGINAL_FRONTEND_COMMAND = v445._frontend_command
_DEFAULT_PROFILE = v447.PROFILE
_ORIGINAL_VLLM_COMMAND = perf._vllm_command
_ORIGINAL_CONFIG_TEXT = chunk256._config_text
_REMOTE_BACKEND_IDENTITIES = {
    "UCX": "official-lmcacheconnectorv1-nixl-ucx",
    "LIBFABRIC": "official-lmcacheconnectorv1-nixl-libfabric-cxi",
}


def _remote_backend_identity() -> str:
    backend = os.environ.get("TEMPO_LMCACHE_NIXL_BACKEND", "UCX")
    if backend not in _REMOTE_BACKEND_IDENTITIES:
        raise ValueError(
            "TEMPO_LMCACHE_NIXL_BACKEND must be UCX or LIBFABRIC")
    if backend == "LIBFABRIC" and os.environ.get("FI_PROVIDER") != "cxi":
        raise ValueError(
            "LIBFABRIC Elastic-PD profiles require FI_PROVIDER=cxi")
    return _REMOTE_BACKEND_IDENTITIES[backend]


def _signal_background_start(*, output, run_id) -> None:
    raw_start = os.environ.get(_BACKGROUND_START_ENV)
    if raw_start is None:
        return
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("background-gated client requires run_id")
    if run_id.endswith("-warmup"):
        return
    start_file = Path(raw_start)
    if not start_file.is_absolute():
        raise ValueError(f"{_BACKGROUND_START_ENV} must be absolute")
    if output is None:
        raise ValueError("background-gated client requires output")
    output_path = Path(output).resolve()
    expected_result_dir = output_path.parent.parent
    if start_file.parent.resolve() != expected_result_dir:
        raise ValueError(
            f"{_BACKGROUND_START_ENV} must share the client result directory")
    with start_file.open("x", encoding="utf-8") as marker:
        marker.write("start\n")


def _client_command(*args, **kwargs):
    command = v445._ORIGINAL_CLIENT(*args, **kwargs)
    old = "eval.sota_4node.run_tempo_pd_stream_metrics_v1"
    command[command.index(old)] = "eval.sota_4node.run_tempo_pd_elastic"
    _signal_background_start(
        output=kwargs.get("output"), run_id=kwargs.get("run_id"))
    return command


def _router_command(*args, **kwargs):
    command = _ORIGINAL_ROUTER_COMMAND(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_elastic_router_v445"
    command[command.index(old)] = "eval.sota_4node.tempo_pd_elastic_router"
    backend_marker = "--remote-backend"
    if command.count(backend_marker) != 1:
        raise ValueError("unexpected inherited remote-backend seam")
    command[command.index(backend_marker) + 1] = (
        _remote_backend_identity())
    scope = os.environ.get("TEMPO_ELASTIC_PD_PROFILE_SCOPE", "screen_only")
    if scope == "replicated":
        command.remove("--allow-screen-profile")
        command.append("--require-replicated-profile")
    elif scope != "screen_only":
        raise ValueError("TEMPO_ELASTIC_PD_PROFILE_SCOPE must be screen_only or replicated")
    queue_marker = "--queue-wait-ms"
    queue_index = command.index(queue_marker) + 1
    if command[queue_index] != "250":
        raise ValueError("unexpected inherited queue-wait-ms seam")
    command[queue_index] = "1000"
    return command


def _frontend_command(*args, **kwargs):
    command = _ORIGINAL_FRONTEND_COMMAND(*args, **kwargs)
    old = "eval.sota_4node.tempo_pd_elastic_frontend_v445"
    command[command.index(old)] = "eval.sota_4node.tempo_pd_elastic_frontend"
    return command


def _vllm_command(*args, **kwargs):
    """Use role-specific chunking without splitting LMCache source work.

    Producer requests retain a 32768-token scheduler budget: the workload has
    eight concurrent 4094+1-token source requests and LMCache PD reservations
    are request-scoped. Decoder-local prefills do not use that source
    reservation, so an explicit smaller decoder budget may interleave local
    prefill with active decode. The default remains the proven 32768 control.
    """
    is_prefill = kwargs.get("is_prefill")
    if type(is_prefill) is not bool:
        raise ValueError("canonical vLLM command requires is_prefill")
    command = _ORIGINAL_VLLM_COMMAND(*args, **kwargs)
    prefix_marker = "--no-enable-prefix-caching"
    if command.count(prefix_marker) != 1:
        raise ValueError("unexpected inherited prefix-caching seam")
    raw_decoder_prefix_caching = os.environ.get(
        "TEMPO_VLLM_DECODER_PREFIX_CACHING", "0")
    if raw_decoder_prefix_caching not in ("0", "1"):
        raise ValueError(
            "TEMPO_VLLM_DECODER_PREFIX_CACHING must be 0 or 1")
    if not is_prefill and raw_decoder_prefix_caching == "1":
        command[command.index(prefix_marker)] = "--enable-prefix-caching"
        if "--enable-prompt-tokens-details" in command:
            raise ValueError("unexpected inherited prompt-token-details seam")
        command.append("--enable-prompt-tokens-details")
        if "--block-size" in command:
            raise ValueError("unexpected inherited decoder block-size seam")
        command.extend(["--block-size", "16"])
        executable = Path(command[0])
        if executable.name != "vllm":
            raise ValueError("unexpected inherited vLLM executable seam")
        command[:1] = [
            str(executable.with_name("python")),
            "-m",
            _CACHE_CONTROL_MODULE,
        ]
    async_marker = "--no-async-scheduling"
    if command.count(async_marker) != 1:
        raise ValueError("unexpected inherited async-scheduling seam")
    raw_async_scheduling = os.environ.get(
        "TEMPO_VLLM_ASYNC_SCHEDULING", "0")
    if raw_async_scheduling not in ("0", "1"):
        raise ValueError(
            "TEMPO_VLLM_ASYNC_SCHEDULING must be 0 or 1")
    if raw_async_scheduling == "1":
        command[command.index(async_marker)] = "--async-scheduling"

    seq_marker = "--max-num-seqs"
    seq_index = command.index(seq_marker) + 1
    if command[seq_index] != "8":
        raise ValueError("unexpected inherited max-num-seqs seam")
    raw_max_num_seqs = os.environ.get("TEMPO_VLLM_MAX_NUM_SEQS", "8")
    try:
        max_num_seqs = int(raw_max_num_seqs)
    except ValueError as exc:
        raise ValueError("TEMPO_VLLM_MAX_NUM_SEQS must be 8 or 16") from exc
    if max_num_seqs not in (8, 16):
        raise ValueError("TEMPO_VLLM_MAX_NUM_SEQS must be 8 or 16")
    command[seq_index] = str(max_num_seqs)
    if "--scheduling-policy" in command:
        raise ValueError("unexpected inherited scheduling-policy seam")
    scheduling_policy = os.environ.get(
        "TEMPO_VLLM_SCHEDULING_POLICY", "fcfs")
    if scheduling_policy not in ("fcfs", "priority"):
        raise ValueError(
            "TEMPO_VLLM_SCHEDULING_POLICY must be fcfs or priority")
    command.extend(["--scheduling-policy", scheduling_policy])
    marker = "--max-num-batched-tokens"
    index = command.index(marker) + 1
    if command[index] != "8192":
        raise ValueError("unexpected inherited max-num-batched-tokens seam")
    raw_decoder_batch_tokens = os.environ.get(
        "TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS", "32768")
    try:
        decoder_batch_tokens = int(raw_decoder_batch_tokens)
    except ValueError as exc:
        raise ValueError(
            "TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS must be "
            "8192, 16384, or 32768") from exc
    if decoder_batch_tokens not in (8192, 16384, 32768):
        raise ValueError(
            "TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS must be "
            "8192, 16384, or 32768")
    command[index] = (
        "32768" if is_prefill else str(decoder_batch_tokens))
    connector_marker = "--kv-transfer-config"
    connector_index = command.index(connector_marker) + 1
    connector = json.loads(command[connector_index])
    extra = connector.get("kv_connector_extra_config")
    if not isinstance(extra, dict):
        raise ValueError("unexpected inherited kv connector config seam")
    # LMCache returns the number of source-side cached prompt tokens in
    # kv_transfer_params. The proxy promotes it to an HTTP header so P_ONLY
    # comes from an observed hit, never from prompt identity.
    lmcache_extra = extra.get("lmcache.extra_config")
    if lmcache_extra is None:
        lmcache_extra = {}
        extra["lmcache.extra_config"] = lmcache_extra
    if not isinstance(lmcache_extra, dict):
        raise ValueError("unexpected inherited LMCache extra config seam")
    lmcache_extra["enable_cache_usage_details_in_response"] = True
    command[connector_index] = json.dumps(
        connector, sort_keys=True, separators=(",", ":"))
    return command


def _config_text(*args, **kwargs):
    """Retain prefill KV in LMCache LocalCPU while preserving PDBackend."""
    is_prefill = kwargs.get("is_prefill")
    if is_prefill is None and args:
        raise ValueError("canonical config requires keyword arguments")
    text = _ORIGINAL_CONFIG_TEXT(*args, **kwargs)
    raw_pd_buffer_bytes = os.environ.get(
        "TEMPO_LMCACHE_PD_BUFFER_BYTES", "2147483648")
    try:
        pd_buffer_bytes = int(raw_pd_buffer_bytes)
    except ValueError as exc:
        raise ValueError(
            "TEMPO_LMCACHE_PD_BUFFER_BYTES must be 536870912, "
            "1073741824, or 2147483648") from exc
    if pd_buffer_bytes not in (536870912, 1073741824, 2147483648):
        raise ValueError(
            "TEMPO_LMCACHE_PD_BUFFER_BYTES must be 536870912, "
            "1073741824, or 2147483648")
    pd_buffer_marker = "pd_buffer_size: 2147483648\n"
    if text.count(pd_buffer_marker) != 1:
        raise ValueError("unexpected inherited PD buffer config seam")
    text = text.replace(
        pd_buffer_marker, f"pd_buffer_size: {pd_buffer_bytes}\n", 1)
    if is_prefill:
        local_cpu_gb = float(os.environ.get(
            "TEMPO_LMCACHE_LOCAL_CPU_GB", "16"))
        if not 0 < local_cpu_gb <= 128:
            raise ValueError(
                "TEMPO_LMCACHE_LOCAL_CPU_GB must be in (0, 128]")
        marker = "local_cpu: False\n"
        if text.count(marker) != 1:
            raise ValueError("unexpected inherited local_cpu config seam")
        text = text.replace(
            marker,
            "local_cpu: True\n"
            f"max_local_cpu_size: {local_cpu_gb:g}\n"
            'retrieve_locations: ["LocalCPUBackend"]\n'
            "save_unfull_chunk: true\n",
            1,
        )
    backend = os.environ.get("TEMPO_LMCACHE_NIXL_BACKEND", "UCX")
    if backend not in ("UCX", "LIBFABRIC"):
        raise ValueError(
            "TEMPO_LMCACHE_NIXL_BACKEND must be UCX or LIBFABRIC")
    backend_marker = "nixl_backends: [UCX]\n"
    if text.count(backend_marker) != 1:
        raise ValueError("unexpected inherited NIXL backend config seam")
    if backend == "LIBFABRIC":
        text = text.replace(
            backend_marker, "nixl_backends: [LIBFABRIC]\n", 1)
    return text


def _analyze(args) -> None:
    stage_root = args.result_dir / "tempo_elastic_pd_v445"
    final = args.result_dir / "elastic_pd_final.json"
    legacy_final = args.result_dir / "elastic_pd_final_v445.json"
    if final.exists():
        final.rename(legacy_final)
    python = args.repo_root / ".vllm_venv/bin/python"
    subprocess.run([
        str(python), "-m", "eval.sota_4node.analyze_tempo_pd_elastic",
        "--stage-root", str(stage_root), "--output", str(final),
    ], cwd=args.repo_root, check=True, timeout=120.0)
    result = args.result_dir / "result.json"
    result.write_text(json.dumps({
        "schema": "tempo-elastic-pd-result-canonical",
        "final": str(final.resolve()),
        "legacy_screen": str(legacy_final.resolve()) if legacy_final.exists() else None,
    }, sort_keys=True) + "\n")


def main() -> int:
    old_client = v446._client_command
    old_router = v449._router_command
    old_frontend = v445._frontend_command
    old_profile = v447.PROFILE
    old_vllm = perf._vllm_command
    old_config = perf._config_text
    old_chunk_config = chunk256._config_text
    v446._client_command = _client_command
    v449._router_command = _router_command
    v445._frontend_command = _frontend_command
    v447.PROFILE = os.environ.get("TEMPO_ELASTIC_PD_PROFILE", _DEFAULT_PROFILE)
    perf._vllm_command = _vllm_command
    perf._config_text = _config_text
    chunk256._config_text = _config_text
    try:
        status = v449.main()
        args = v445.capacity._parse()
        if args.node_index == 0:
            _analyze(args)
        return status
    finally:
        v446._client_command = old_client
        v449._router_command = old_router
        v445._frontend_command = old_frontend
        v447.PROFILE = old_profile
        perf._vllm_command = old_vllm
        perf._config_text = old_config
        chunk256._config_text = old_chunk_config


if __name__ == "__main__":
    raise SystemExit(main())
