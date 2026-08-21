#!/usr/bin/env python3
"""Run the frozen calibration-only C4 phase-changing four-arm screen.

This client owns no servers.  One server lifecycle first executes an
unmeasured P_ONLY seed/catalog phase and then eight open-loop traces:
four arms in a counterbalanced order for each of two replicates.  Every trace
contains the same C0/C1/cold-C2/KV-C2/C3/recovery semantic schedule.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from tempo.pd_contention_workload import (
    CacheState,
    ContentionState,
    ForegroundArm,
    LoadSelection,
    ScheduledRequest,
    Tenant,
    TokenGeometry,
    TrafficShape,
    build_schedule,
    semantic_schedule_sha256,
)
from eval.sota_4node import run_tempo_pd_contention_fixed_client as fixed
from eval.sota_4node import (
    build_tempo_pd_c4_semantic_epoch_run_contract
    as semantic_contract_builder,
)
from eval.sota_4node import verify_tempo_pd_c4_implementation as fixed_implementation
from tempo.pd_endpoint_profile import SCHEMA_V2, load_endpoint_service_profile


SCHEMA = "tempo-pd-c4-phase-screen-client-v1"
BLOCK_SCHEMA = "tempo-pd-c4-phase-screen-block-v1"
PRESEED_SCHEMA = "tempo-pd-c4-phase-screen-preseed-v1"
ENDPOINT_SERIES_SCHEMA = "tempo-pd-c4-endpoint-series-v1"
RUN_CONTRACT_SCHEMA = "tempo-pd-c4-phase-screen-run-contract-v1"
SEMANTIC_RUN_CONTRACT_SCHEMA = (
    "tempo-pd-c4-semantic-epoch-screen-run-contract-v2")
MANIFEST_SCHEMA = "tempo-pd-c4-phase-screen-manifest-v1"
ROUTER_SCHEMA = "tempo-elastic-pd-router-canonical"
CANONICAL_MODULE = "eval.sota_4node.run_tempo_pd_elastic_stream_metrics"
PRESEEDED_MODULE = (
    "eval.sota_4node.run_tempo_pd_elastic_stream_metrics_preseeded"
)
PRESEEDED_ENV = "TEMPO_PD_P_ONLY_PRESEEDED"
CONTRACT_ENV = "TEMPO_PD_C4_RUN_CONTRACT"
CONTRACT_SHA_ENV = "TEMPO_PD_C4_RUN_CONTRACT_SHA256"
WORKLOAD_SHA_ENV = "TEMPO_PD_ENDPOINT_WORKLOAD_MANIFEST_SHA256"
LOCAL_ROUTE = "decoder_local_chunked_prefill"
REMOTE_ROUTE = "official_lmcache_remote_prefill"
PHASES = (
    ContentionState.C0,
    ContentionState.C1,
    ContentionState.C2,
    ContentionState.C2_KV,
    ContentionState.C3,
    ContentionState.RECOVERY,
)
ARMS = (
    ForegroundArm.LOCAL,
    ForegroundArm.REMOTE,
    ForegroundArm.PREDICTOR,
    ForegroundArm.TEMPO,
)
FOREGROUND_GEOMETRIES = (
    TokenGeometry(512, 16, CacheState.P_ONLY),
    TokenGeometry(2048, 256, CacheState.P_ONLY),
    TokenGeometry(4094, 16, CacheState.P_ONLY),
)
FOREGROUND_POOL_PER_GEOMETRY = 2
KV_BACKGROUND_POOL_SIZE = 32
KV_BACKGROUND_GEOMETRY = TokenGeometry(4094, 2, CacheState.P_ONLY)
ENDPOINT_SAMPLE_INTERVAL_S = 7.5


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_contract_fingerprint(value: dict[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_implementation_file(raw: object, *, name: str) -> Path:
    _require(type(raw) is str and raw, f"{name} path is missing")
    pure = PurePosixPath(raw)
    _require(
        not pure.is_absolute() and ".." not in pure.parts and str(pure) == raw,
        f"{name} path is not canonical and relative",
    )
    path = (_repo_root() / raw).resolve()
    _require(path.is_file(), f"{name} is missing")
    return path


def _validate_semantic_implementation(contract: dict[str, object]) -> None:
    entries = contract.get("implementation")
    _require(isinstance(entries, list) and entries,
             "semantic implementation inventory is missing")
    seen = set()
    for index, entry in enumerate(entries):
        _require(isinstance(entry, dict)
                 and set(entry) == {"path", "sha256"},
                 f"semantic implementation[{index}] binding differs")
        raw = entry["path"]
        _require(type(raw) is str and raw not in seen,
                 "semantic implementation paths are duplicated")
        path = _resolve_implementation_file(
            raw, name=f"semantic implementation[{index}]")
        _require(_sha256(path) == entry["sha256"],
                 f"semantic implementation drifted: {raw}")
        seen.add(raw)
    _require(
        {
            "tempo/pd_endpoint_controller.py",
            "tempo/pd_endpoint_profile.py",
            "tempo/pd_endpoint_profile.py",
            "eval/sota_4node/tempo_pd_elastic_frontend.py",
            "eval/sota_4node/tempo_pd_elastic_router.py",
            "eval/sota_4node/run_tempo_pd_c4_phase_screen_client.py",
            "eval/sota_4node/vllm_lmcache_pd_c4_phase_screen_node.py",
            "eval/sota_4node/analyze_tempo_pd_c4_phase_screen.py",
            "eval/sota_4node/analyze_tempo_pd_c4_semantic_epoch_screen.py",
            "eval/sota_4node/run_tempo_pd_c4_semantic_epoch_screen_in_allocation.sh",
            "eval/sota_4node/build_tempo_pd_semantic_epoch_endpoint_profile.py",
        } <= seen,
        "semantic implementation omits a verdict or live-path file",
    )
    fixed_entry = contract.get("fixed_c4_implementation_contract")
    _require(
        isinstance(fixed_entry, dict)
        and set(fixed_entry) == {
            "path", "sha256", "schema", "fingerprint_sha256"},
        "semantic fixed-C4 implementation binding differs",
    )
    fixed_path = _resolve_implementation_file(
        fixed_entry["path"], name="semantic fixed-C4 implementation")
    _require(
        _sha256(fixed_path) == fixed_entry["sha256"]
        and fixed_entry["schema"] == fixed_implementation.SCHEMA,
        "semantic fixed-C4 implementation digest differs",
    )
    fixed_value = json.loads(fixed_path.read_text(encoding="utf-8"))
    _require(
        fixed_value.get("fingerprint_sha256")
        == fixed_entry["fingerprint_sha256"],
        "semantic fixed-C4 implementation fingerprint differs",
    )
    manifest_entry = fixed_value.get("phase_manifest")
    _require(isinstance(manifest_entry, dict),
             "semantic fixed-C4 phase manifest is missing")
    semantic_contract_builder.verify_fixed_baseline_for_semantic(
        repo_root=_repo_root(),
        contract_path=fixed_path,
        expected_sha256=str(fixed_entry["sha256"]),
        phase_manifest=_resolve_implementation_file(
            manifest_entry.get("path"),
            name="semantic fixed-C4 phase manifest",
        ),
        semantic_contract=contract,
    )


def _validate_semantic_runtime(contract: dict[str, object]) -> None:
    runtime = contract.get("runtime_environment")
    _require(isinstance(runtime, dict) and runtime,
             "semantic runtime environment is missing")
    for name, expected in runtime.items():
        _require(
            type(name) is str and name.startswith("TEMPO_")
            and type(expected) is str
            and os.environ.get(name) == expected,
            f"semantic runtime requires {name}={expected}",
        )


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--default-max-tokens", type=int, default=32)
    parser.add_argument("--max-workers", type=int, default=128)
    parser.add_argument("--request-rate", type=float, required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--api-key-env")
    parser.add_argument("--phase-duration-ms", type=float, required=True)
    parser.add_argument("--cooldown-s", type=float, required=True)
    parser.add_argument("--endpoint-evidence-url", action="append", default=[])
    parser.add_argument("--endpoint-controller-url", action="append", default=[])
    return parser.parse_args()


def _resolve_entry(
    contract: dict[str, object], name: str,
) -> tuple[Path, dict[str, object]]:
    entry = contract.get(name)
    _require(isinstance(entry, dict), f"run contract lacks {name}")
    raw_path = entry.get("path")
    _require(isinstance(raw_path, str) and raw_path, f"{name} path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = _repo_root() / path
    path = path.resolve()
    _require(path.is_file(), f"{name} artifact is missing")
    _require(_sha256(path) == entry.get("sha256"), f"{name} digest differs")
    return path, entry


def _load_contract() -> tuple[Path, dict[str, object], Path, dict[str, object]]:
    raw_path = os.environ.get(CONTRACT_ENV)
    expected_sha = os.environ.get(CONTRACT_SHA_ENV)
    _require(bool(raw_path) and bool(expected_sha), "frozen C4 run contract is required")
    path = Path(str(raw_path)).resolve()
    _require(path.is_file(), "frozen C4 run contract is missing")
    _require(_sha256(path) == expected_sha, "frozen C4 run contract digest differs")
    contract = json.loads(path.read_text(encoding="utf-8"))
    schema = contract.get("schema")
    _require(
        schema in {RUN_CONTRACT_SCHEMA, SEMANTIC_RUN_CONTRACT_SCHEMA},
        "run contract schema differs",
    )
    if schema == SEMANTIC_RUN_CONTRACT_SCHEMA:
        _require(
            contract.get("fingerprint_sha256")
            == _semantic_contract_fingerprint(contract)
            and contract.get("endpoint_routing_policy") == "semantic_epoch_v1"
            and contract.get("passive_external_credit") is True
            and contract.get("calibration_only") is True,
            "semantic C4 contract binding differs",
        )
        _require(
            contract.get("controller_reset_before_each_measured_block")
            is True,
            "semantic C4 contract permits cross-block controller state",
        )
        _validate_semantic_implementation(contract)
        _validate_semantic_runtime(contract)
        base_path, base_entry = _resolve_entry(
            contract, "base_c4_run_contract")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        _require(
            base.get("schema") == RUN_CONTRACT_SCHEMA
            and base_entry.get("schema") == RUN_CONTRACT_SCHEMA,
            "semantic C4 base contract differs",
        )
        _resolve_entry(contract, "semantic_observer_result")
        _resolve_entry(contract, "semantic_observer_analysis")
    _require(contract.get("performance_claim_allowed") is False,
             "C4 screen cannot permit a performance claim")
    _require(contract.get("transport") == "LMCacheConnectorV1:UCX",
             "C4 run contract transport differs")
    _require(contract.get("unchanged_pd_data_plane") is True,
             "C4 run contract changed the P/D data plane")
    expected_policy = (
        "semantic_epoch_v1"
        if schema == SEMANTIC_RUN_CONTRACT_SCHEMA else "instant_score_v1"
    )
    _require(
        os.environ.get(
            "TEMPO_PD_ENDPOINT_ROUTING_POLICY", "instant_score_v1")
        == expected_policy,
        "runtime endpoint routing policy differs",
    )
    _require(
        os.environ.get("TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK")
        == ("1" if schema == SEMANTIC_RUN_CONTRACT_SCHEMA else "0"),
        "runtime passive endpoint feedback differs",
    )
    manifest_path, _ = _resolve_entry(contract, "phase_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == MANIFEST_SCHEMA, "C4 manifest schema differs")
    _require(manifest.get("performance_claim_allowed") is False,
             "C4 manifest cannot permit a performance claim")
    _require(os.environ.get(WORKLOAD_SHA_ENV) == _sha256(manifest_path),
             "runtime C4 workload binding differs")
    _resolve_entry(contract, "source_workload")
    _resolve_entry(contract, "elastic_profile")
    endpoint_path, endpoint_entry = _resolve_entry(
        contract, "endpoint_service_profile")
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    _require(endpoint.get("fingerprint_sha256") == endpoint_entry.get(
        "fingerprint_sha256"), "endpoint profile fingerprint differs")
    _require(endpoint.get("workload_manifest_sha256") == _sha256(manifest_path),
             "endpoint profile is not bound to the C4 manifest")
    loaded_endpoint = load_endpoint_service_profile(endpoint_path)
    if schema == SEMANTIC_RUN_CONTRACT_SCHEMA:
        source_endpoint_path, source_endpoint_entry = _resolve_entry(
            contract, "source_endpoint_service_profile")
        _require(
            loaded_endpoint.schema == SCHEMA_V2
            and loaded_endpoint.routing_policy is not None
            and loaded_endpoint.routing_policy.as_dict()
            == contract.get("semantic_credit_contract")
            and endpoint_entry.get("schema") == SCHEMA_V2
            and endpoint_entry.get("derived_from_sha256")
            == source_endpoint_entry.get("sha256")
            and source_endpoint_path
            == semantic_contract_builder._resolve_base_entry(
                base, "endpoint_service_profile"),
            "semantic endpoint profile is not profile-bound",
        )
    else:
        _require(
            loaded_endpoint.routing_policy is None,
            "instant-score C4 cannot carry a semantic routing profile",
        )
    replay_path, replay_entry = _resolve_entry(contract, "offline_replay")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    _require(replay.get("schema") == replay_entry.get("schema")
             and replay.get("live_c4_screen_authorized") is True,
             "offline replay did not authorize the C4 screen")
    return path, contract, manifest_path, manifest


def _load_templates(path: Path, tokenizer) -> dict[int, tuple[int, ...]]:
    _require(path.is_file(), "explicit source workload is missing")
    required = {512, 2048, 4094}
    templates: dict[int, tuple[int, ...]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row.get("prompt")
        _require(isinstance(prompt, str) and prompt, "source prompt is invalid")
        token_ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
        if len(token_ids) in required:
            templates.setdefault(len(token_ids), token_ids)
    missing = sorted(required - set(templates))
    _require(not missing, f"source workload lacks templates: {missing}")
    return templates


def _pool_prompts(tokenizer, templates) -> tuple[dict[tuple[int, int, int], str], tuple[str, ...]]:
    foreground: dict[tuple[int, int, int], str] = {}
    for geometry_index, geometry in enumerate(FOREGROUND_GEOMETRIES):
        for owner in range(FOREGROUND_POOL_PER_GEOMETRY):
            prompt = fixed._unique_prompt(
                tokenizer,
                templates[geometry.prompt_tokens],
                240_000 + geometry_index * 16 + owner,
            )
            foreground[(
                geometry.prompt_tokens, geometry.output_tokens, owner,
            )] = prompt
    background = tuple(
        fixed._unique_prompt(tokenizer, templates[4094], 200_000 + index)
        for index in range(KV_BACKGROUND_POOL_SIZE)
    )
    values = list(foreground.values()) + list(background)
    hashes = {hashlib.sha256(value.encode()).hexdigest() for value in values}
    _require(len(hashes) == len(values), "C4 cache pool prompts are not unique")
    return foreground, background


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ), encoding="utf-8")


def _stream_command(
    args: argparse.Namespace, *, module: str, workload: Path, output: Path,
    run_id: str,
) -> list[str]:
    command = [
        sys.executable, "-m", module,
        "--base-url", args.base_url,
        "--model", str(args.model),
        "--served-model-name", args.served_model_name,
        "--workload", str(workload),
        "--output", str(output),
        "--mode", "tempo_auto",
        "--run-id", run_id,
        "--default-max-tokens", str(args.default_max_tokens),
        "--max-workers", str(args.max_workers),
        "--timeout-s", str(args.timeout_s),
        "--seed", str(args.seed),
    ]
    if args.api_key_env:
        command.extend(("--api-key-env", args.api_key_env))
    return command


def _run_stream(
    args: argparse.Namespace, *, module: str, workload: Path, output: Path,
    run_id: str, preseeded: bool,
) -> dict[str, object]:
    env = dict(os.environ)
    if preseeded:
        env[PRESEEDED_ENV] = "1"
    else:
        env.pop(PRESEEDED_ENV, None)
    completed = subprocess.run(
        _stream_command(
            args, module=module, workload=workload, output=output,
            run_id=run_id),
        cwd=_repo_root(), env=env, check=False, timeout=1200.0,
    )
    _require(completed.returncode == 0, f"unmeasured stream failed: {run_id}")
    value = json.loads(output.read_text(encoding="utf-8"))
    _require(value.get("validation", {}).get("all_streams_valid") is True,
             f"unmeasured stream validation failed: {run_id}")
    return value


def _validate_preseed_artifact(
    artifact: dict[str, object], expected: dict[str, dict[str, object]], *,
    implicit_seed: bool,
) -> None:
    requests = artifact.get("requests")
    decisions = artifact.get("router_decisions")
    _require(isinstance(requests, list) and isinstance(decisions, list),
             "P_ONLY preparation rows are missing")
    _require({row.get("request_id") for row in requests} == set(expected),
             "P_ONLY preparation request IDs differ")
    index = {row.get("request_id"): row for row in decisions}
    _require(len(index) == len(decisions) and set(index) == set(expected),
             "P_ONLY preparation decision IDs differ")
    for row in requests:
        _require(row.get("valid") is True, "P_ONLY preparation stream is invalid")
        if implicit_seed:
            seed = row.get("p_only_cache_seed")
            _require(isinstance(seed, dict) and seed.get("valid") is True
                     and seed.get("route") == REMOTE_ROUTE,
                     "P_ONLY physical seed evidence is missing")
    for request_id, metadata in expected.items():
        decision = index[request_id]
        _require(decision.get("route") == REMOTE_ROUTE,
                 "P_ONLY preparation was not routed remotely")
        _require(decision.get("lmcache_source_cached_tokens")
                 == metadata["prompt_tokens"],
                 "P_ONLY preparation probe lost its exact source hit")
        _require(decision.get("lmcache_source_full_hit_observed") is True,
                 "P_ONLY preparation lacks full-hit evidence")
        _require(decision.get("completion_cache_residency") == "prefill_only",
                 "P_ONLY preparation did not establish prefill residency")
        _require(int(decision.get("frontend_pair_index"))
                 == int(metadata["owner"]),
                 "P_ONLY preparation used the wrong producer pair")
        if (
            implicit_seed
            and request_id.startswith("epd-tempo-")
            and "-physical-" in request_id
        ):
            _require(
                decision.get("frontend_pair_physical_seed_pin") is True,
                "P_ONLY physical seed pair pin evidence is missing",
            )


def _warmup(
    args: argparse.Namespace, tokenizer, templates,
    contract_path: Path, manifest_path: Path,
) -> int:
    root = args.output.parent / "c4_preseed"
    root.mkdir()
    foreground, background = _pool_prompts(tokenizer, templates)
    physical_rows: list[dict[str, object]] = []
    physical_index: dict[str, dict[str, object]] = {}
    for geometry_index, geometry in enumerate(FOREGROUND_GEOMETRIES):
        for owner in range(FOREGROUND_POOL_PER_GEOMETRY):
            request_id = (
                "epd-tempo-c4-cache-p-only-warm-physical-"
                f"g{geometry_index:02d}-item-{owner:06d}"
            )
            physical_rows.append({
                "request_id": request_id,
                "prompt": foreground[(
                    geometry.prompt_tokens, geometry.output_tokens, owner)],
                "max_tokens": geometry.output_tokens,
                "arrival_offset_ms": round(len(physical_rows) * 250.0, 6),
            })
            physical_index[request_id] = {
                "prompt_tokens": geometry.prompt_tokens, "owner": owner,
            }
    for item, prompt in enumerate(background):
        request_id = (
            "epd-remote-c4-cache-p-only-warm-kv-background-"
            f"item-{item:06d}"
        )
        physical_rows.append({
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": KV_BACKGROUND_GEOMETRY.output_tokens,
            "arrival_offset_ms": round(len(physical_rows) * 250.0, 6),
        })
        physical_index[request_id] = {
            "prompt_tokens": KV_BACKGROUND_GEOMETRY.prompt_tokens,
            "owner": item % 2,
        }
    physical_workload = root / "physical_seed.jsonl"
    physical_raw = root / "physical_seed.raw.json"
    _write_rows(physical_workload, physical_rows)
    physical = _run_stream(
        args, module=CANONICAL_MODULE, workload=physical_workload,
        output=physical_raw, run_id=f"{args.run_id}-physical-seed",
        preseeded=False,
    )
    _validate_preseed_artifact(
        physical, physical_index, implicit_seed=True)

    catalog_rows: list[dict[str, object]] = []
    catalog_index: dict[str, dict[str, object]] = {}
    for arm in (ForegroundArm.LOCAL, ForegroundArm.REMOTE,
                ForegroundArm.PREDICTOR):
        for geometry_index, geometry in enumerate(FOREGROUND_GEOMETRIES):
            for owner in range(FOREGROUND_POOL_PER_GEOMETRY):
                request_id = (
                    f"epd-{arm.value}-c4-cache-p-only-warm-catalog-"
                    f"g{geometry_index:02d}-item-{owner:06d}"
                )
                catalog_rows.append({
                    "request_id": request_id,
                    "prompt": foreground[(
                        geometry.prompt_tokens, geometry.output_tokens, owner)],
                    "max_tokens": geometry.output_tokens,
                    "arrival_offset_ms": round(len(catalog_rows) * 250.0, 6),
                })
                catalog_index[request_id] = {
                    "prompt_tokens": geometry.prompt_tokens, "owner": owner,
                }
    catalog_workload = root / "arm_catalog_probe.jsonl"
    catalog_raw = root / "arm_catalog_probe.raw.json"
    _write_rows(catalog_workload, catalog_rows)
    catalog = _run_stream(
        args, module=PRESEEDED_MODULE, workload=catalog_workload,
        output=catalog_raw, run_id=f"{args.run_id}-arm-catalog-probe",
        preseeded=True,
    )
    _validate_preseed_artifact(
        catalog, catalog_index, implicit_seed=False)

    payload = {
        "schema": PRESEED_SCHEMA,
        "run_id": args.run_id,
        "measured": False,
        "performance_claim_allowed": False,
        "run_contract": str(contract_path),
        "run_contract_sha256": _sha256(contract_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "physical_seed": str(physical_raw.resolve()),
        "physical_seed_sha256": _sha256(physical_raw),
        "arm_catalog_probe": str(catalog_raw.resolve()),
        "arm_catalog_probe_sha256": _sha256(catalog_raw),
        "physical_prompt_count": len(physical_rows),
        "arm_catalog_probe_count": len(catalog_rows),
        "foreground_pool_prompts": len(foreground),
        "kv_background_pool_prompts": len(background),
        "preseed_completed_before_measurement": True,
        "all_full_source_hits_exact": True,
        "physical_seed_pair_pin_exact": True,
        "all_arm_residency_namespaces_confirmed": True,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return 0


def _foreground_prompt(
    pools: dict[tuple[int, int, int], str], request: ScheduledRequest,
) -> str:
    owner = request.ordinal % FOREGROUND_POOL_PER_GEOMETRY
    return pools[(
        request.geometry.prompt_tokens,
        request.geometry.output_tokens,
        owner,
    )]


def _trace_rows(
    *, schedule: tuple[ScheduledRequest, ...], foreground_pool,
    background_pool: tuple[str, ...], tokenizer, templates,
    marker_base: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    index: dict[str, dict[str, object]] = {}
    cold_chunks: set[tuple[int, ...]] = set()
    cold_item = 0
    for request in schedule:
        if request.tenant is Tenant.FOREGROUND:
            prompt = _foreground_prompt(foreground_pool, request)
            pair_key = f"{request.phase.value}:foreground:{request.ordinal:06d}"
            expected_cache = "p_only"
        elif request.tenant is Tenant.KV_REMOTE_HOT:
            prompt = background_pool[request.ordinal % len(background_pool)]
            pair_key = None
            expected_cache = "p_only"
        else:
            marker = marker_base + cold_item
            cold_item += 1
            _require(marker < (1 << 18), "C4 cold marker space exhausted")
            prompt = fixed._unique_prompt(
                tokenizer, templates[request.geometry.prompt_tokens], marker)
            chunk = tuple(tokenizer.encode(
                prompt, add_special_tokens=False)[:256])
            _require(chunk not in cold_chunks,
                     "C4 cold first LMCache chunk is not unique")
            cold_chunks.add(chunk)
            pair_key = None
            expected_cache = "miss"
        rows.append({
            "request_id": request.request_id,
            "prompt": prompt,
            "max_tokens": request.geometry.output_tokens,
            "arrival_offset_ms": round(request.arrival_offset_ms, 6),
        })
        index[request.request_id] = {
            **request.semantic_dict(),
            "arm": request.arm.value,
            "pair_key": pair_key,
            "expected_cache": expected_cache,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
    return rows, index


def _fetch_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=10.0) as response:
        _require(response.status == 200, "telemetry endpoint returned non-200")
        value = json.loads(response.read().decode("utf-8"))
    _require(isinstance(value, dict), "telemetry endpoint is not an object")
    return value


def _controller_snapshot(urls: list[str]) -> list[dict[str, object]]:
    _require(len(urls) == 2 and len(set(urls)) == 2,
             "two unique endpoint controller URLs are required")
    values = [_fetch_json(url.rstrip("/") + "/tempo/endpoint_controller")
              for url in urls]
    for value in values:
        _require(value.get("schema") == ROUTER_SCHEMA,
                 "endpoint controller router schema differs")
        _require(value.get("endpoint_feedback_mode") == "adaptive",
                 "endpoint controller is not adaptive")
    return values


def _controller_reset(url: str) -> dict[str, object]:
    request = Request(
        url.rstrip("/") + "/tempo/reset_endpoint_controller",
        data=b"",
        method="POST",
    )
    with urlopen(request, timeout=10.0) as response:
        _require(response.status == 200, "endpoint controller reset failed")
        value = json.loads(response.read().decode("utf-8"))
    _validate_controller_reset_evidence([value])
    return value


def _validate_controller_reset_evidence(
    values: object,
) -> list[int]:
    _require(isinstance(values, list) and values,
             "endpoint controller reset evidence is missing")
    expected_resources = {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    generations: list[int] = []
    for value in values:
        _require(isinstance(value, dict) and value.get("success") is True,
                 "endpoint controller reset did not succeed")
        generation = value.get("controller_generation")
        controller = value.get("controller")
        _require(
            type(generation) is int and generation >= 1
            and isinstance(controller, dict)
            and controller.get("inflight") == 0
            and controller.get("external_inflight", 0) == 0
            and controller.get("resources") == expected_resources
            and controller.get("owned_resources", expected_resources)
            == expected_resources
            and controller.get("external_resources", expected_resources)
            == expected_resources,
            "endpoint controller reset evidence is not quiescent",
        )
        generations.append(generation)
    return generations


def _run_trace_with_evidence(
    command: list[str], *, args: argparse.Namespace, trace_duration_s: float,
) -> tuple[dict[str, object], int]:
    before = fixed._capture_endpoint_evidence(
        args.endpoint_evidence_url, stage="before", require_valid_delta=False)
    controller_before = _controller_snapshot(args.endpoint_controller_url)
    env = dict(os.environ)
    env[PRESEEDED_ENV] = "1"
    child = subprocess.Popen(command, cwd=_repo_root(), env=env)
    periodic: list[dict[str, object]] = []
    started = time.monotonic()
    target = ENDPOINT_SAMPLE_INTERVAL_S
    try:
        while target < trace_duration_s:
            remaining = started + target - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            _require(child.poll() is None,
                     "C4 child exited before its open-loop trace ended")
            sample = fixed._capture_endpoint_evidence(
                args.endpoint_evidence_url,
                stage="midpoint",
                require_valid_delta=True,
            )
            sample["stage"] = f"periodic-{len(periodic):02d}"
            sample["client_trace_elapsed_ms"] = (
                time.monotonic() - started) * 1000.0
            periodic.append(sample)
            target += ENDPOINT_SAMPLE_INTERVAL_S
        return_code = child.wait(timeout=1200.0)
        _require(return_code in {0, 2}, f"C4 child returned {return_code}")
    except BaseException:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10.0)
        raise
    after = fixed._capture_endpoint_evidence(
        args.endpoint_evidence_url, stage="after", require_valid_delta=False)
    controller_after = _controller_snapshot(args.endpoint_controller_url)
    frontend_after = _fetch_json(args.base_url.rstrip("/") + "/health")
    evidence = {
        "schema": ENDPOINT_SERIES_SCHEMA,
        "sampling_policy": "bounded_on_demand_7p5s_plus_boundaries",
        "sample_interval_s": ENDPOINT_SAMPLE_INTERVAL_S,
        "cross_endpoint_clock_subtraction_allowed": False,
        "before": before,
        "periodic": periodic,
        "after": after,
        "endpoint_controller_before": controller_before,
        "endpoint_controller_after": controller_after,
        "frontend_after": frontend_after,
    }
    return evidence, return_code


def _nearest_rank(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _latencies(row: dict[str, object]) -> tuple[float, float, float]:
    dispatch = int(row["dispatch_offset_ns"])
    end = int(row["stream_end_offset_ns"])
    arrivals = [int(value) for value in row["token_arrival_offsets_ns"]]
    _require(bool(arrivals), "valid stream has no token arrivals")
    e2e = (end - dispatch) / 1_000_000.0
    ttft = (arrivals[0] - dispatch) / 1_000_000.0
    output_tokens = len(row.get("output_token_values", []))
    tpot = ((end - arrivals[0]) / 1_000_000.0
            / max(1, output_tokens - 1))
    return e2e, ttft, tpot


def _latency_summary(
    rows: list[dict[str, object]], *, ttft_slo: float, tpot_slo: float,
    e2e_slo: float,
) -> dict[str, object]:
    valid = [row for row in rows if row.get("valid") is True]
    triples = [_latencies(row) for row in valid]
    e2e = [value[0] for value in triples]
    ttft = [value[1] for value in triples]
    tpot = [value[2] for value in triples]
    good = [
        a <= e2e_slo and b <= ttft_slo and c <= tpot_slo
        for a, b, c in triples
    ]
    return {
        "offered": len(rows),
        "completed_valid": len(valid),
        "goodput_requests": sum(good),
        "goodput_fraction": sum(good) / len(rows) if rows else None,
        "e2e_median_ms": statistics.median(e2e) if e2e else None,
        "e2e_p99_ms": _nearest_rank(e2e, 0.99),
        "ttft_median_ms": statistics.median(ttft) if ttft else None,
        "ttft_p99_ms": _nearest_rank(ttft, 0.99),
        "tpot_median_ms": statistics.median(tpot) if tpot else None,
        "tpot_p99_ms": _nearest_rank(tpot, 0.99),
    }


def _controller_quiescent(values: list[dict[str, object]]) -> bool:
    expected_resources = {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    return all(
        isinstance(value.get("controller"), dict)
        and value["controller"].get("inflight") == 0
        and value["controller"].get("external_inflight", 0) == 0
        and value["controller"].get("resources") == expected_resources
        and value.get("queued_requests") == 0
        for value in values
    )


def _augment_trace(
    raw_path: Path, *, request_index: dict[str, dict[str, object]],
    evidence: dict[str, object], arm: ForegroundArm, replicate: int,
    sequence: int, schedule_sha256: str, return_code: int,
    manifest: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    requests = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(requests, list) and isinstance(decisions, list),
             "C4 child rows are missing")
    _require({row.get("request_id") for row in requests} == set(request_index),
             "C4 request IDs differ")
    decision_index = {row.get("request_id"): row for row in decisions}
    _require(len(decision_index) == len(decisions)
             and set(decision_index) == set(request_index),
             "C4 decision IDs differ")
    _require(all(row.get("valid") is True for row in requests),
             "C4 contains an invalid stream")
    reset_generations = _validate_controller_reset_evidence(
        evidence.get("endpoint_controller_reset_before_block"))
    _require(len(reset_generations) == 2,
             "C4 requires one controller reset per pair")
    route_counts: collections.Counter[str] = collections.Counter()
    foreground_by_phase: dict[str, list[dict[str, object]]] = {
        phase.value: [] for phase in PHASES
    }
    for row in requests:
        metadata = request_index[str(row["request_id"])]
        decision = decision_index[str(row["request_id"])]
        route = decision.get("route")
        _require(route in {LOCAL_ROUTE, REMOTE_ROUTE},
                 "C4 request lacks one committed upstream route")
        route_counts[str(route)] += 1
        if metadata["arm"] == ForegroundArm.LOCAL.value:
            _require(route == LOCAL_ROUTE, "route-pinned local request escaped")
        elif metadata["arm"] == ForegroundArm.REMOTE.value:
            _require(route == REMOTE_ROUTE, "route-pinned remote request escaped")
        cache = metadata["expected_cache"]
        _require(decision.get("request_cache_contract") == cache,
                 "C4 explicit cache contract differs")
        if cache == "p_only":
            _require(decision.get("decision_cache_residency") == "prefill_only"
                     and decision.get("completion_cache_residency") == "prefill_only",
                     "C4 P_ONLY residency differs")
            if route == REMOTE_ROUTE:
                _require(decision.get("lmcache_source_cached_tokens")
                         == metadata["prompt_tokens"]
                         and decision.get("lmcache_source_full_hit_observed") is True,
                         "C4 P_ONLY remote route lost its full source hit")
            else:
                _require(decision.get("lmcache_source_cached_tokens") is None,
                         "C4 local route has remote source-cache evidence")
        else:
            if route == REMOTE_ROUTE:
                _require(decision.get("lmcache_source_cached_tokens") == 0
                         and decision.get("lmcache_source_full_hit_observed") is False,
                         "C4 cold remote request observed a source hit")
            else:
                _require(decision.get("lmcache_source_cached_tokens") is None,
                         "C4 cold local request has remote cache evidence")
        if metadata["tenant"] == Tenant.FOREGROUND.value:
            foreground_by_phase[str(metadata["phase"])].append(row)
            if arm is ForegroundArm.TEMPO:
                _require(decision.get("endpoint_policy_applied") is True
                         and decision.get("endpoint_decision_route") == route
                         and int(decision.get("endpoint_decision_attempts")) >= 1,
                         "TEMPO endpoint decision provenance differs")
                _require(decision.get("admission_credit_release_event")
                         == "first_response_chunk"
                         and decision.get("admission_credit_released_ns") is not None,
                         "TEMPO endpoint credit was not released on first response")
                _require(decision.get("endpoint_feedback_event")
                         == "first_response_chunk"
                         and decision.get("endpoint_feedback_accepted") is True,
                         "TEMPO endpoint completion feedback differs")
            else:
                _require(decision.get("endpoint_policy_applied") is False,
                         "non-TEMPO foreground used endpoint policy")
    controller_after = evidence["endpoint_controller_after"]
    _require(_controller_quiescent(controller_after),
             "C4 endpoint controller leaked credit or queued work")
    frontend_after = evidence["frontend_after"]
    _require(frontend_after.get("active_pair_reservations") == 0
             and frontend_after.get("pair_loads") == [0, 0],
             "C4 frontend decode reservation leaked")
    measurement = manifest["measurement"]
    phase_summaries = {
        phase: _latency_summary(
            rows,
            ttft_slo=float(measurement["ttft_slo_ms"]),
            tpot_slo=float(measurement["tpot_slo_ms"]),
            e2e_slo=float(measurement["e2e_slo_ms"]),
        )
        for phase, rows in foreground_by_phase.items()
    }
    contract = {
        "schema": BLOCK_SCHEMA,
        "arm": arm.value,
        "replicate": replicate,
        "block_sequence_index": sequence,
        "semantic_schedule_sha256": schedule_sha256,
        "request_index": request_index,
        "request_counts": dict(collections.Counter(
            value["tenant"] for value in request_index.values())),
        "route_counts": dict(route_counts),
        "all_requests_valid": True,
        "router_decisions_exact": True,
        "one_way_route_commit_exact": True,
        "p_only_full_source_hits_exact": True,
        "cold_remote_source_misses_exact": True,
        "credit_release_and_quiescence_exact": True,
        "controller_reset_before_block_exact": True,
        "preseed_outside_measurement": True,
        "actual_inference_background_only": True,
        "synthetic_network_background": False,
        "passive_external_endpoint_credit": (
            os.environ.get("TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK") == "1"),
        "endpoint_routing_policy": os.environ.get(
            "TEMPO_PD_ENDPOINT_ROUTING_POLICY", "instant_score_v1"),
        "official_lmcache_connector_v1_ucx": True,
        "cross_endpoint_clock_subtraction_allowed": False,
        "child_return_code": return_code,
    }
    raw["c4_phase_screen_contract"] = contract
    raw["c4_endpoint_series"] = evidence
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "arm": arm.value,
        "replicate": replicate,
        "block_sequence_index": sequence,
        "route_counts": dict(route_counts),
        "foreground_by_phase": phase_summaries,
        "all_requests_valid": True,
        "credit_quiescent": True,
        "controller_reset_generations": reset_generations,
    }
    return contract, summary


def _paired_output_gate(
    artifacts: dict[str, str], contracts: dict[str, dict[str, object]],
) -> dict[str, object]:
    values: dict[tuple[int, str], dict[str, tuple[str, tuple[str, ...], str]]] = {}
    schedules: dict[int, set[str]] = collections.defaultdict(set)
    for key, raw_name in artifacts.items():
        contract = contracts[key]
        replicate = int(contract["replicate"])
        arm = str(contract["arm"])
        schedules[replicate].add(str(contract["semantic_schedule_sha256"]))
        raw = json.loads(Path(raw_name).read_text(encoding="utf-8"))
        rows = {row["request_id"]: row for row in raw["requests"]}
        for request_id, metadata in contract["request_index"].items():
            pair_key = metadata.get("pair_key")
            if pair_key is None:
                continue
            row = rows[request_id]
            values.setdefault((replicate, pair_key), {})[arm] = (
                str(row["output_text_sha256"]),
                tuple(str(value) for value in row["output_token_values"]),
                str(row["prompt_sha256"]),
            )
    expected_arms = {arm.value for arm in ARMS}
    failures = []
    for (replicate, pair_key), by_arm in sorted(values.items()):
        if set(by_arm) != expected_arms or len(set(by_arm.values())) != 1:
            failures.append({
                "replicate": replicate, "pair_key": pair_key,
                "observed_arms": sorted(by_arm),
            })
    schedule_exact = all(len(values) == 1 for values in schedules.values())
    _require(schedule_exact, "C4 arm semantic schedules differ")
    _require(not failures, "C4 paired foreground outputs differ")
    return {
        "paired_foreground_requests": len(values),
        "all_four_arms_present": True,
        "prompt_output_and_token_digests_exact": True,
        "semantic_schedules_exact_within_replicate": True,
        "failures": failures,
    }


def _measured(
    args: argparse.Namespace, tokenizer, templates,
    contract_path: Path, manifest_path: Path, manifest: dict[str, object],
) -> int:
    preseed_path = args.output.parent / "warmup.raw.json"
    _require(preseed_path.is_file(), "C4 preseed artifact is missing")
    preseed = json.loads(preseed_path.read_text(encoding="utf-8"))
    _require(preseed.get("schema") == PRESEED_SCHEMA
             and preseed.get("preseed_completed_before_measurement") is True
             and preseed.get("all_full_source_hits_exact") is True,
             "C4 preseed artifact is invalid")
    _require(preseed.get("run_contract_sha256") == _sha256(contract_path)
             and preseed.get("manifest_sha256") == _sha256(manifest_path),
             "C4 preseed binding differs")
    _require(len(args.endpoint_evidence_url) == 4,
             "C4 requires four endpoint probes")
    _require(len(args.endpoint_controller_url) == 2,
             "C4 requires two endpoint controllers")
    _require(tuple(manifest["phase_order"]) == tuple(
        phase.value for phase in PHASES), "C4 phase order differs")
    _require(float(manifest["measurement"]["phase_duration_ms"])
             == args.phase_duration_ms, "C4 phase duration differs")
    _require(float(manifest["cooldown_s"]) == args.cooldown_s,
             "C4 cooldown differs")
    _require(float(manifest["foreground"]["offered_rate_per_s"])
             == args.request_rate, "C4 foreground rate differs")
    foreground_pool, background_pool = _pool_prompts(tokenizer, templates)
    selection = LoadSelection(
        decoder_reference_rate_per_s=float(
            manifest["load"]["decoder_hot_rate_per_s"]),
        remote_reference_rate_per_s=float(
            manifest["load"]["remote_hot_rate_per_s"]),
        decoder_fraction=1.0,
        remote_fraction=1.0,
        kv_remote_rate_per_s=float(
            manifest["load"]["kv_remote_hot_rate_per_s"]),
    )
    root = args.output.parent / "c4_phase_screen"
    workload_root = args.output.parent / "c4_phase_screen_workloads"
    root.mkdir()
    workload_root.mkdir()
    artifacts: dict[str, str] = {}
    contracts: dict[str, dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    stopped_after: str | None = None
    sequence = 0
    passive_endpoint_feedback = (
        os.environ.get("TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK") == "1")
    for replicate, raw_order in enumerate(manifest["arm_order_by_replicate"]):
        order = tuple(ForegroundArm(value) for value in raw_order)
        _require(set(order) == set(ARMS) and len(order) == len(ARMS),
                 "C4 arm order is not a permutation")
        for arm in order:
            key = f"{sequence:02d}_rep{replicate:02d}_{arm.value}"
            schedule = build_schedule(
                states=PHASES,
                selection=selection,
                foreground_arm=arm,
                foreground_rate_per_s=args.request_rate,
                trial_id=f"c4-{key}-measured",
                shape=TrafficShape.STABLE,
                phase_duration_ms=args.phase_duration_ms,
                foreground_geometries=FOREGROUND_GEOMETRIES,
                passive_endpoint_feedback=passive_endpoint_feedback,
            )
            schedule_sha = semantic_schedule_sha256(schedule)
            rows, request_index = _trace_rows(
                schedule=schedule,
                foreground_pool=foreground_pool,
                background_pool=background_pool,
                tokenizer=tokenizer,
                templates=templates,
                marker_base=10_000 + sequence * 8_192,
            )
            workload_path = workload_root / f"{key}.jsonl"
            raw_path = root / f"{key}.raw.json"
            _write_rows(workload_path, rows)
            controller_reset = [
                _controller_reset(url)
                for url in args.endpoint_controller_url
            ]
            evidence, return_code = _run_trace_with_evidence(
                _stream_command(
                    args, module=PRESEEDED_MODULE, workload=workload_path,
                    output=raw_path, run_id=f"{args.run_id}-{key}"),
                args=args,
                trace_duration_s=(
                    args.phase_duration_ms * len(PHASES) / 1000.0),
            )
            evidence["endpoint_controller_reset_before_block"] = (
                controller_reset)
            block_contract, summary = _augment_trace(
                raw_path,
                request_index=request_index,
                evidence=evidence,
                arm=arm,
                replicate=replicate,
                sequence=sequence,
                schedule_sha256=schedule_sha,
                return_code=return_code,
                manifest=manifest,
            )
            artifacts[key] = str(raw_path.resolve())
            contracts[key] = block_contract
            summaries.append(summary)
            sequence += 1
            if return_code != 0:
                stopped_after = key
                break
            if sequence < 8:
                time.sleep(args.cooldown_s)
        if stopped_after is not None:
            break
    paired = (
        _paired_output_gate(artifacts, contracts)
        if stopped_after is None and len(artifacts) == 8 else None
    )
    payload = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "purpose": manifest["purpose"],
        "calibration_only": True,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "endpoint_routing_policy": os.environ.get(
            "TEMPO_PD_ENDPOINT_ROUTING_POLICY", "instant_score_v1"),
        "passive_external_endpoint_credit": passive_endpoint_feedback,
        "run_contract": str(contract_path),
        "run_contract_sha256": _sha256(contract_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "preseed": str(preseed_path.resolve()),
        "preseed_sha256": _sha256(preseed_path),
        "artifacts": artifacts,
        "contracts": contracts,
        "summaries": summaries,
        "paired_output_gate": paired,
        "blocks_completed": len(artifacts),
        "stopped_after_first_invalid_block": stopped_after,
        "live_screen_correctness_pass": (
            stopped_after is None and len(artifacts) == 8
            and paired is not None
        ),
        "controller_reset_before_each_block_exact": (
            stopped_after is None and len(summaries) == 8
            and all(
                len(summary.get("controller_reset_generations", [])) == 2
                for summary in summaries
            )
        ),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "blocks_completed": len(artifacts),
        "correctness_pass": payload["live_screen_correctness_pass"],
        "stopped_after": stopped_after,
    }, sort_keys=True))
    return 0 if payload["live_screen_correctness_pass"] else 2


def main() -> int:
    args = _parse()
    _require(args.mode == "tempo_auto", "C4 client requires tempo_auto")
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    _require(args.model.is_absolute(), "model path must be absolute")
    _require(args.max_workers > 0 and args.request_rate > 0,
             "C4 workers/rate must be positive")
    contract_path, _contract, manifest_path, manifest = _load_contract()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True)
    templates = _load_templates(args.workload, tokenizer)
    if args.run_id.endswith("-warmup"):
        return _warmup(
            args, tokenizer, templates, contract_path, manifest_path)
    return _measured(
        args, tokenizer, templates, contract_path, manifest_path, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
