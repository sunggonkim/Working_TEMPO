#!/usr/bin/env python3
"""Run the frozen four-arm TEMPO validation on held-out burst prompts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Mapping

from eval.sota_4node import build_tempo_pd_independent_validation_manifest as manifest_module
from eval.sota_4node import build_tempo_pd_independent_validation_run_contract as run_contract_module
from eval.sota_4node import run_tempo_pd_c4_adaptive_screen_client as adaptive
from eval.sota_4node import run_tempo_pd_c4_fixed_phase_client as c4
from tempo.pd_cache_state_protocol import build_cache_preparation_plan
from tempo.pd_contention_workload import (
    CacheState,
    ForegroundArm,
    LoadSelection,
    Tenant,
    TrafficShape,
    VALIDATION_FOREGROUND_GEOMETRIES,
    build_schedule,
    semantic_schedule_sha256,
)
from tempo.pd_elastic_profile import load_elastic_profile, require_replicated_profile
from tempo.pd_endpoint_profile import load_endpoint_service_profile


SCHEMA = "tempo-pd-independent-validation-client-v1"
BLOCK_SCHEMA = "tempo-pd-independent-validation-block-v1"
RUN_CONTRACT_ENV = "TEMPO_PD_INDEPENDENT_VALIDATION_RUN_CONTRACT"
RUN_CONTRACT_SHA_ENV = (
    "TEMPO_PD_INDEPENDENT_VALIDATION_RUN_CONTRACT_SHA256")
WORKLOAD_SHA_ENV = adaptive.WORKLOAD_SHA_ENV
CONTROLLER_URLS_ENV = adaptive.CONTROLLER_URLS_ENV
PROTOCOL_MODULE = adaptive.PROTOCOL_MODULE
SOURCE_MODULE = adaptive.SOURCE_MODULE
ARMS = adaptive.ARMS
HELD_OUT_PROMPT_MARKER_START = 180_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse() -> argparse.Namespace:
    return adaptive._parse()


def _resolve_entry(
    contract: Mapping[str, object], name: str,
) -> tuple[Path, Mapping[str, object]]:
    entry = contract.get(name)
    _require(isinstance(entry, Mapping),
             f"independent run contract lacks {name}")
    raw_path = entry.get("path")
    _require(type(raw_path) is str and raw_path,
             f"independent {name} path is missing")
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path = path.resolve()
    _require(path.is_file() and _sha256(path) == entry.get("sha256"),
             f"independent {name} digest differs")
    return path, entry


def _load_contract():
    raw_path = os.environ.get(RUN_CONTRACT_ENV)
    expected_sha = os.environ.get(RUN_CONTRACT_SHA_ENV)
    _require(bool(raw_path) and bool(expected_sha),
             "frozen independent run contract is required")
    path = Path(str(raw_path)).resolve()
    _require(path.is_file() and _sha256(path) == expected_sha,
             "independent run contract digest differs")
    contract = json.loads(path.read_text(encoding="utf-8"))
    _require(
        contract.get("schema") == run_contract_module.SCHEMA
        and contract.get("fingerprint_sha256")
        == run_contract_module.contract_fingerprint(contract)
        and contract.get("independent_validation_authorized") is True
        and contract.get("controller_parameters_unchanged") is True
        and contract.get("controller_parameter_search_allowed") is False
        and contract.get("post_validation_tuning_allowed") is False
        and contract.get("performance_claim_allowed") is False
        and contract.get("physical_switch_bottleneck_claim_allowed") is False
        and contract.get("transport") == "LMCacheConnectorV1:UCX"
        and contract.get("unchanged_pd_data_plane") is True,
        "independent run contract claim or transport differs",
    )
    candidate = contract.get("candidate")
    _require(
        isinstance(candidate, Mapping)
        and candidate == contract.get("candidate"),
        "independent candidate binding is missing",
    )
    manifest_path, manifest_entry = _resolve_entry(
        contract, "independent_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(
        manifest.get("schema") == manifest_module.SCHEMA
        and manifest.get("fingerprint_sha256")
        == manifest_module.manifest_fingerprint(manifest)
        == manifest_entry.get("fingerprint_sha256")
        and manifest.get("traffic_shape") == "burst"
        and manifest.get("replicate_ids") == [2, 3, 4, 5]
        and manifest.get("post_validation_tuning_allowed") is False
        and manifest.get("performance_claim_allowed") is False
        and os.environ.get(WORKLOAD_SHA_ENV) == _sha256(manifest_path),
        "independent workload manifest binding differs",
    )
    source_path, source_entry = _resolve_entry(contract, "source_workload")
    manifest_source = manifest.get("source_workload")
    _require(
        isinstance(manifest_source, Mapping)
        and Path(str(manifest_source.get("path", ""))).resolve()
        == source_path
        and manifest_source.get("sha256") == source_entry.get("sha256"),
        "independent source workload binding differs",
    )
    _require(manifest.get("candidate") == candidate,
             "independent manifest candidate differs")
    analysis_path, analysis_entry = _resolve_entry(
        contract, "candidate_screen_analysis")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    _require(
        analysis.get("fingerprint_sha256")
        == analysis_entry.get("fingerprint_sha256")
        and analysis.get("authorizes_independent_validation") is True
        and analysis.get("performance_claim_allowed") is False,
        "candidate analysis authorization differs",
    )
    elastic_path, elastic_entry = _resolve_entry(
        contract, "promoted_elastic_profile")
    endpoint_path, endpoint_entry = _resolve_entry(
        contract, "promoted_endpoint_service_profile")
    elastic = load_elastic_profile(elastic_path)
    require_replicated_profile(elastic)
    endpoint = load_endpoint_service_profile(endpoint_path)
    _require(
        elastic.fingerprint_sha256 == elastic_entry.get("fingerprint_sha256")
        and endpoint.fingerprint_sha256
        == endpoint_entry.get("fingerprint_sha256")
        and endpoint.deployment_scope == "frozen_validation"
        and endpoint.elastic_profile_fingerprint_sha256
        == elastic.fingerprint_sha256
        and endpoint.workload_manifest_sha256 == _sha256(manifest_path),
        "promoted independent profile binding differs",
    )
    if candidate.get("kind") == "candidate_b_semantic_epoch_v1":
        _require(
            endpoint.routing_policy is not None
            and endpoint.routing_policy.policy == "semantic_epoch_v1",
            "semantic candidate endpoint policy differs",
        )
    else:
        _require(
            candidate.get("kind") == "candidate_a_instant_score_v1"
            and endpoint.routing_policy is None,
            "instant-score candidate endpoint policy differs",
        )
    receipt_path, receipt_entry = _resolve_entry(
        contract, "profile_promotion_receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(
        receipt.get("fingerprint_sha256")
        == receipt_entry.get("fingerprint_sha256")
        and receipt.get("controller_parameters_unchanged") is True
        and receipt.get("post_validation_tuning_allowed") is False,
        "independent promotion receipt differs",
    )
    _resolve_entry(contract, "adaptive_implementation_contract")
    _resolve_entry(contract, "candidate_implementation_contract")
    _resolve_entry(contract, "independent_implementation_contract")
    fixed_environment = run_contract_module.independent_runtime_environment(
        candidate)
    _require(
        contract.get("fixed_runtime_environment") == dict(sorted(
            fixed_environment.items())),
        "independent fixed runtime environment differs",
    )
    for name, expected in (
        ("TEMPO_ELASTIC_PD_PROFILE", str(elastic_path)),
        ("TEMPO_PD_ENDPOINT_SERVICE_PROFILE", str(endpoint_path)),
        (WORKLOAD_SHA_ENV, _sha256(manifest_path)),
    ):
        actual = os.environ.get(name)
        if name.endswith("PROFILE"):
            actual = str(Path(actual).resolve()) if actual else actual
        _require(actual == expected, f"independent runtime {name} differs")
    for name, expected in fixed_environment.items():
        _require(os.environ.get(name) == expected,
                 f"independent runtime {name} differs")
    return path, contract, manifest_path, manifest, source_path


def _block_order(
    manifest: Mapping[str, object],
) -> tuple[tuple[ForegroundArm, int], ...]:
    raw_orders = manifest.get("arm_order_by_replicate")
    expected = [
        {"replicate": 2, "arms": ["local", "predictor", "tempo", "remote"]},
        {"replicate": 3, "arms": ["remote", "tempo", "predictor", "local"]},
        {"replicate": 4, "arms": ["predictor", "local", "remote", "tempo"]},
        {"replicate": 5, "arms": ["tempo", "remote", "local", "predictor"]},
    ]
    _require(raw_orders == expected,
             "independent arm order differs from preregistration")
    order = tuple(
        (ForegroundArm(arm), int(row["replicate"]))
        for row in raw_orders for arm in row["arms"]
    )
    _require(
        len(order) == 16
        and set(order) == {
            (arm, replicate) for arm in ARMS for replicate in (2, 3, 4, 5)
        },
        "independent block inventory differs",
    )
    return order


def _materialize_block(
    *, sequence: int, arm: ForegroundArm, replicate: int,
    manifest: Mapping[str, object], factory: c4._PromptFactory,
) -> dict[str, object]:
    rates = manifest.get("background_rates_per_s")
    _require(isinstance(rates, Mapping),
             "independent background rates are missing")
    selection = LoadSelection(
        decoder_reference_rate_per_s=float(rates["decoder_hot"]),
        remote_reference_rate_per_s=float(rates["cold_remote_hot"]),
        decoder_fraction=1.0,
        remote_fraction=1.0,
        kv_remote_rate_per_s=float(rates["kv_remote_hot"]),
    )
    schedule = build_schedule(
        states=c4.manifest_builder.PHASES,
        selection=selection,
        foreground_arm=arm,
        foreground_rate_per_s=float(manifest["foreground_rate_per_s"]),
        trial_id=f"independent-r{replicate}-{arm.value}",
        shape=TrafficShape.BURST,
        phase_duration_ms=float(manifest["phase_duration_ms"]),
        foreground_geometries=VALIDATION_FOREGROUND_GEOMETRIES,
        passive_endpoint_feedback=True,
    )
    rows: list[dict[str, object]] = []
    items = []
    request_index: dict[str, dict[str, object]] = {}
    for request in schedule:
        geometry = request.geometry
        geometry_index = (
            VALIDATION_FOREGROUND_GEOMETRIES.index(geometry)
            if request.tenant is Tenant.FOREGROUND else -1
        )
        terminal_item = c4._terminal_item(
            tenant=request.tenant,
            ordinal=request.ordinal,
            geometry_index=geometry_index,
            cache_state=geometry.cache_state,
        )
        key = c4._prompt_key(
            sequence=sequence,
            replicate=replicate,
            tenant=request.tenant,
            ordinal=request.ordinal,
            geometry_index=geometry_index,
            cache_state=geometry.cache_state,
            terminal_item=terminal_item,
        )
        prompt, prompt_key = factory.prompt(key, geometry.prompt_tokens)
        request_id = c4._request_id(
            sequence=sequence,
            arm=request.arm,
            replicate=replicate,
            phase=request.phase,
            tenant=request.tenant,
            ordinal=request.ordinal,
            state=geometry.cache_state,
            terminal_item=terminal_item,
        )
        rows.append({
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": geometry.output_tokens,
            "arrival_offset_ms": round(request.arrival_offset_ms, 6),
        })
        items.append(c4.CacheProtocolItem(
            request_id=request_id,
            prompt=prompt,
            prompt_token_sha256=prompt_key,
            prompt_tokens=geometry.prompt_tokens,
            output_tokens=geometry.output_tokens,
            cache_state=geometry.cache_state,
            terminal_item=terminal_item,
        ))
        request_index[request_id] = {
            **request.semantic_dict(),
            "arm": request.arm.value,
            "prompt_token_sha256": prompt_key,
            "terminal_item": terminal_item,
            "pair_key": (
                f"r{replicate}:{request.phase.value}:"
                f"foreground:{request.ordinal:06d}"
                if request.tenant is Tenant.FOREGROUND else None
            ),
        }
    rows.sort(key=lambda row: (
        float(row["arrival_offset_ms"]), str(row["request_id"])))
    return {
        "sequence": sequence,
        "arm": arm,
        "replicate": replicate,
        "schedule_sha256": semantic_schedule_sha256(schedule),
        "rows": rows,
        "items": items,
        "request_index": request_index,
    }


def _validate_block(
    raw_path: Path, block: Mapping[str, object], endpoint_evidence: object,
    *, controller_reset: list[Mapping[str, object]],
    controller_before: list[Mapping[str, object]],
    controller_after: list[Mapping[str, object]],
    semantic_contract: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    adaptive_contract, summary = adaptive._validate_block(
        raw_path,
        block,
        endpoint_evidence,
        controller_reset=controller_reset,
        controller_before=controller_before,
        controller_after=controller_after,
        semantic_contract=semantic_contract,
    )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    _require(raw.pop("c4_adaptive_screen_contract", None) == adaptive_contract,
             "adaptive validator did not publish the expected child contract")
    contract = dict(adaptive_contract)
    contract["schema"] = BLOCK_SCHEMA
    contract["held_out_burst_workload"] = True
    contract["calibration_only"] = False
    raw["independent_validation_contract"] = contract
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract, summary


def _paired_gate(
    *, block_paths: Mapping[str, Path],
    contracts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    _require(
        set(block_paths) == set(contracts) and len(contracts) == 16,
        "independent child artifact inventory differs",
    )
    samples: dict[tuple[int, str], dict[str, tuple[str, str]]] = defaultdict(dict)
    schedules: dict[int, set[str]] = defaultdict(set)
    tempo_routes = Counter()
    observed_blocks = set()
    for key, contract in contracts.items():
        _require(
            contract.get("schema") == BLOCK_SCHEMA
            and contract.get("all_requests_valid") is True
            and contract.get("completion_cache_evidence_exact") is True
            and contract.get("phase_aligned_endpoint_evidence") is True
            and contract.get("controller_reset_before_block_exact") is True
            and contract.get("controller_quiescent_after_block") is True
            and contract.get("held_out_burst_workload") is True
            and contract.get("calibration_only") is False,
            f"independent block contract differs: {key}",
        )
        replicate = int(contract["replicate"])
        arm = str(contract["arm"])
        observed_blocks.add((arm, replicate))
        schedules[replicate].add(str(contract["semantic_schedule_sha256"]))
        raw = json.loads(block_paths[key].read_text(encoding="utf-8"))
        requests = {row["request_id"]: row for row in raw["requests"]}
        decisions = {row["request_id"]: row for row in raw["router_decisions"]}
        for request_id, metadata in contract["request_index"].items():
            if metadata["tenant"] != Tenant.FOREGROUND.value:
                continue
            pair_key = str(metadata["pair_key"])
            samples[(replicate, pair_key)][arm] = (
                str(requests[request_id]["output_text_sha256"]),
                str(metadata["prompt_token_sha256"]),
            )
            if arm == ForegroundArm.TEMPO.value:
                tempo_routes[str(decisions[request_id]["route"])] += 1
    expected_arms = {arm.value for arm in ARMS}
    _require(
        observed_blocks == {
            (arm.value, replicate)
            for arm in ARMS for replicate in (2, 3, 4, 5)
        },
        "independent arm/replicate inventory differs",
    )
    _require(
        set(schedules) == {2, 3, 4, 5}
        and all(len(value) == 1 for value in schedules.values()),
        "independent semantic schedules differ within a replicate",
    )
    _require(bool(samples), "independent run has no foreground pairs")
    _require(all(
        set(by_arm) == expected_arms and len(set(by_arm.values())) == 1
        for by_arm in samples.values()
    ), "independent paired prompt/output digests differ")
    return {
        "paired_foreground_requests": len(samples),
        "all_four_arms_present": True,
        "semantic_schedules_exact_within_replicate": True,
        "prompt_and_output_digests_exact": True,
        "held_out_prompt_marker_start": HELD_OUT_PROMPT_MARKER_START,
        "tempo_route_counts": {
            c4._LOCAL_ROUTE: tempo_routes[c4._LOCAL_ROUTE],
            c4._REMOTE_ROUTE: tempo_routes[c4._REMOTE_ROUTE],
        },
        "tempo_both_routes_exercised": all(
            tempo_routes[route] > 0
            for route in (c4._LOCAL_ROUTE, c4._REMOTE_ROUTE)),
        "performance_claim_allowed": False,
    }


def _measured(
    args: argparse.Namespace, tokenizer, templates,
    contract_path: Path, manifest_path: Path, manifest: Mapping[str, object],
    contract: Mapping[str, object],
) -> int:
    _require(os.environ.get("TEMPO_VLLM_DECODER_PREFIX_CACHING") == "1",
             "independent validation requires decoder prefix caching")
    _require(os.environ.get("TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY") == "1",
             "independent validation requires replicated warm affinity")
    _require(len(args.endpoint_evidence_url) == 4,
             "independent validation requires four endpoint probes")
    _require(len(args.endpoint_controller_url) == 2,
             "independent validation requires two endpoint controllers")
    _require(
        args.phase_duration_ms == float(manifest["phase_duration_ms"])
        and args.request_rate == float(manifest["foreground_rate_per_s"])
        and args.cooldown_s == float(manifest["cooldown_s"]),
        "independent runtime workload differs from its manifest",
    )
    order = _block_order(manifest)
    semantic_contract = (
        contract
        if contract["candidate"]["kind"]
        == "candidate_b_semantic_epoch_v1" else None)
    root = args.output.parent / "independent_validation"
    workload_root = root / "workloads"
    root.mkdir()
    workload_root.mkdir()
    factory = c4._PromptFactory(tokenizer, templates)
    factory._next_marker = HELD_OUT_PROMPT_MARKER_START
    blocks = [
        _materialize_block(
            sequence=sequence,
            arm=arm,
            replicate=replicate,
            manifest=manifest,
            factory=factory,
        )
        for sequence, (arm, replicate) in enumerate(order)
    ]
    plan = build_cache_preparation_plan(
        item for block in blocks for item in block["items"])
    plan_path = root / "cache_preparation_plan.json"
    plan_path.write_text(
        json.dumps(plan.manifest_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_workload = workload_root / "source_prepare.jsonl"
    source_raw = root / "source_prepare.raw.json"
    c4._write_rows(source_workload, list(plan.source_probe_rows))
    c4._run(c4._stream_command(
        args, module=SOURCE_MODULE, workload=source_workload,
        output=source_raw, run_id=f"{args.run_id}-source-prepare",
        max_workers=1))
    source_evidence = c4.validate_source_preparation(source_raw, plan)
    reset = c4._reset_decoder_prefix_cache(args.base_url)

    decoder_workload = workload_root / "decoder_prepare.jsonl"
    decoder_raw = root / "decoder_prepare.raw.json"
    c4._write_rows(decoder_workload, list(plan.decoder_prepare_rows))
    decoder_env = dict(os.environ)
    decoder_env[c4.protocol_client.PHASE_ENV] = "decoder_prepare"
    decoder_env[c4.protocol_client.PLAN_ENV] = str(plan_path.resolve())
    decoder_env.pop(c4.protocol_client.EVIDENCE_ENV, None)
    c4._run(c4._stream_command(
        args, module=PROTOCOL_MODULE, workload=decoder_workload,
        output=decoder_raw, run_id=f"{args.run_id}-decoder-prepare",
        max_workers=1), env=decoder_env)
    decoder_evidence = c4.validate_decoder_preparation(decoder_raw, plan)
    runtime_evidence = c4._runtime_evidence(
        plan_path=plan_path,
        plan=plan,
        source=source_evidence,
        reset=reset,
        decoder=decoder_evidence,
    )
    evidence_path = root / "cache_runtime_evidence.json"
    evidence_path.write_text(
        json.dumps(runtime_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    measured_env = dict(os.environ)
    measured_env[c4.protocol_client.PHASE_ENV] = "measured"
    measured_env[c4.protocol_client.PLAN_ENV] = str(plan_path.resolve())
    measured_env[c4.protocol_client.EVIDENCE_ENV] = str(evidence_path.resolve())
    artifacts = {}
    contracts = {}
    summaries = []
    block_paths = {}
    evidence_args = argparse.Namespace(**vars(args))
    for block in blocks:
        sequence = int(block["sequence"])
        arm = block["arm"]
        replicate = int(block["replicate"])
        key = f"{sequence:02d}_{arm.value}_r{replicate}"
        controller_reset = [
            adaptive._controller_reset(url)
            for url in args.endpoint_controller_url]
        controller_before = [
            adaptive._controller_get(url)
            for url in args.endpoint_controller_url]
        workload = workload_root / f"{key}.jsonl"
        raw_path = root / f"{key}.raw.json"
        c4._write_rows(workload, block["rows"])
        endpoint_evidence = c4._run_with_endpoint_evidence(
            c4._stream_command(
                args, module=PROTOCOL_MODULE, workload=workload,
                output=raw_path, run_id=f"{args.run_id}-{key}",
                max_workers=args.max_workers),
            args=evidence_args,
            env=measured_env,
            start_marker=(root / f"{key}.measurement-start.json").resolve(),
            first_arrival_offset_ms=min(
                float(row["arrival_offset_ms"]) for row in block["rows"]),
        )
        controller_after = [
            adaptive._controller_get(url)
            for url in args.endpoint_controller_url]
        block_contract, summary = _validate_block(
            raw_path,
            block,
            endpoint_evidence,
            controller_reset=controller_reset,
            controller_before=controller_before,
            controller_after=controller_after,
            semantic_contract=semantic_contract,
        )
        artifacts[key] = c4._artifact_binding(raw_path)
        contracts[key] = block_contract
        summaries.append(summary)
        block_paths[key] = raw_path.resolve()
        if sequence + 1 < len(blocks):
            time.sleep(args.cooldown_s)

    paired = _paired_gate(block_paths=block_paths, contracts=contracts)
    payload = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "run_contract": str(contract_path),
        "run_contract_sha256": _sha256(contract_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "cache_plan": str(plan_path.resolve()),
        "cache_plan_sha256": _sha256(plan_path),
        "cache_runtime_evidence": str(evidence_path.resolve()),
        "cache_runtime_evidence_sha256": _sha256(evidence_path),
        "block_order": [
            {"arm": arm.value, "replicate": replicate}
            for arm, replicate in order
        ],
        "artifacts": artifacts,
        "contracts": contracts,
        "summaries": summaries,
        "paired_output_gate": paired,
        "blocks_completed": len(artifacts),
        "held_out_burst_workload": True,
        "independent_correctness_pass": (
            len(artifacts) == 16
            and paired["all_four_arms_present"] is True
            and paired["prompt_and_output_digests_exact"] is True
        ),
        "independent_route_diversity_pass":
            paired["tempo_both_routes_exercised"],
        "calibration_only": False,
        "post_validation_tuning_allowed": False,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "unchanged_pd_data_plane": True,
        "candidate": contract["candidate"],
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    args = _parse()
    _require(args.mode == "tempo_auto",
             "independent validation requires tempo_auto")
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    _require(args.model.is_absolute(), "model path must be absolute")
    contract_path, contract, manifest_path, manifest, source_path = (
        _load_contract())
    is_warmup = args.run_id.endswith("-warmup")
    if is_warmup:
        _require(
            args.workload.resolve().parent == args.output.resolve().parent
            and args.workload.name == "warmup.jsonl",
            "independent warmup must be lifecycle-local",
        )
    else:
        _require(args.workload.resolve() == source_path,
                 "independent runtime source workload differs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True)
    templates = c4._load_templates(args.workload, tokenizer)
    if is_warmup:
        return c4.fixed._warmup(args, tokenizer, templates)
    return _measured(
        args, tokenizer, templates, contract_path, manifest_path, manifest,
        contract)


if __name__ == "__main__":
    raise SystemExit(main())
