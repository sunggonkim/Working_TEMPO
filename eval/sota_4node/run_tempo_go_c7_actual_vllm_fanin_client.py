#!/usr/bin/env python3
"""Run one actual-vLLM P0/P1-to-D0 receiver-incast qualification epoch."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from eval.sota_4node import analyze_tempo_go_c7_actual_vllm_fanin as analyzer
from eval.sota_4node import run_tempo_go_c6_decoder_victim_client as decoder
from eval.sota_4node import run_tempo_go_c6_performance_client as c6
from eval.sota_4node import run_tempo_pd_contention_fixed_client as fixed
from tempo.pd_contention_workload import (
    CacheState,
    ContentionState,
    ForegroundArm,
    ScheduledRequest,
    Tenant,
    TokenGeometry,
    semantic_schedule_sha256,
)


SCHEMA = analyzer.BUNDLE_SCHEMA
CONTRACT_SCHEMA = analyzer.CONTRACT_SCHEMA
BLOCK_SCHEMA = "tempo-go-c7-actual-vllm-fanin-block-v1"
CONTRACT_ENV = "TEMPO_GO_C7_ACTUAL_VLLM_FANIN_CONTRACT"
TARGET_DECODER = 0
REMOTE_ROUTE = fixed.REMOTE_ROUTE


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decoder_contract(value: dict[str, object]) -> dict[str, object]:
    _require(value.get("schema") == CONTRACT_SCHEMA,
             "C7 fan-in contract schema differs")
    section = value.get("actual_vllm_fanin")
    _require(isinstance(section, dict), "C7 fan-in section is missing")
    return section


def configure_node_environment(
    *, repo_root: Path, qualification: dict[str, object], hosts: list[str],
    port_slot: int, elastic_profile: Path,
) -> None:
    """Reuse C6's proven full-mesh lifecycle with a C7 frozen contract."""
    section = _decoder_contract(qualification)
    os.environ[c6.ARM_ENV] = c6.FULL_ARM
    wrapped = {"schema": c6.CONTRACT_SCHEMA, "c6_performance": section}
    c6.configure_node_environment(
        repo_root=repo_root,
        qualification=wrapped,
        hosts=hosts,
        port_slot=port_slot,
        elastic_profile=elastic_profile,
    )


def _load_contract(args) -> tuple[dict[str, object], dict[str, object]]:
    path = args.qualification_contract.resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    section = _decoder_contract(value)
    victim = section["victim"]
    expected = {
        "request_rate": victim["offered_rate_per_s"],
        "phase_duration_ms": section["phase_duration_ms"],
        "cooldown_s": section["cooldown_s"],
        "max_workers": section["max_workers"],
    }
    for name, frozen in expected.items():
        _require(getattr(args, name) == frozen,
                 f"runtime differs from frozen C7 value: {name}")
    repo_root = Path(__file__).resolve().parents[2]
    source = c6._resolve_artifact(
        repo_root, section["source_workload"], label="source workload")
    _require(args.workload.resolve() == source,
             "C7 source workload path differs")
    return value, section


def _uniform_offsets(duration_ms: float, rate_per_s: float) -> list[float]:
    if rate_per_s == 0.0:
        return []
    count = int(math.floor(rate_per_s * duration_ms / 1000.0))
    spacing_ms = 1000.0 / rate_per_s
    return [(index + 0.5) * spacing_ms for index in range(count)]


def _request_id(
    *, name: str, role: str, ordinal: int, source: int,
) -> str:
    tenant = "interactive" if role == "victim" else "background"
    return (
        f"epd-remote-{tenant}-cache-miss-measured-endpoint-observed-"
        f"tempo-go-exogenous-fixed-remote-d{TARGET_DECODER}-"
        f"c7-fanin-{name}-{role}-{ordinal:06d}-{source}"
    )


def _materialize_schedule(
    *, spec: dict[str, object], section: dict[str, object],
) -> tuple[tuple[ScheduledRequest, ...], dict[str, dict[str, object]]]:
    name = str(spec["name"])
    duration_ms = float(section["phase_duration_ms"])
    victim = section["victim"]
    aggressor = section["aggressor"]
    rate = float(spec["aggressor_rate_per_s"])
    state = ContentionState.C0 if rate == 0.0 else ContentionState.C2
    requests: list[ScheduledRequest] = []
    identities: dict[str, dict[str, object]] = {}
    victim_geometry = TokenGeometry(
        int(victim["prompt_tokens"]), int(victim["output_tokens"]),
        CacheState(str(victim["cache_state"])),
    )
    aggressor_geometry = TokenGeometry(
        int(aggressor["prompt_tokens"]), int(aggressor["output_tokens"]),
        CacheState(str(aggressor["cache_state"])),
    )

    for ordinal, offset in enumerate(_uniform_offsets(
        duration_ms, float(victim["offered_rate_per_s"]))):
        source = int(victim["source_prefill_index"])
        request_id = _request_id(
            name=name, role="victim", ordinal=ordinal, source=source)
        requests.append(ScheduledRequest(
            request_id=request_id,
            phase=state,
            tenant=Tenant.FOREGROUND,
            arm=ForegroundArm.REMOTE,
            arrival_offset_ms=offset,
            geometry=victim_geometry,
            ordinal=ordinal,
        ))
        identities[request_id] = {
            "role": "victim",
            "business_tenant": "interactive",
            "source_prefill_index": source,
            "target_decoder_index": TARGET_DECODER,
            "block": name,
        }

    for ordinal, offset in enumerate(_uniform_offsets(duration_ms, rate)):
        source = ordinal % 2
        request_id = _request_id(
            name=name, role="aggressor", ordinal=ordinal, source=source)
        requests.append(ScheduledRequest(
            request_id=request_id,
            phase=state,
            tenant=Tenant.REMOTE_HOT,
            arm=ForegroundArm.REMOTE,
            arrival_offset_ms=offset,
            geometry=aggressor_geometry,
            ordinal=ordinal,
        ))
        identities[request_id] = {
            "role": "aggressor",
            "business_tenant": "background",
            "source_prefill_index": source,
            "target_decoder_index": TARGET_DECODER,
            "block": name,
        }
    requests.sort(key=lambda row: (
        row.arrival_offset_ms,
        0 if row.tenant is Tenant.FOREGROUND else 1,
        row.ordinal,
    ))
    _require(len(requests) == len(identities), "C7 request IDs are not unique")
    return tuple(requests), identities


def _child_command(args, *, workload: Path, output: Path, run_id: str) -> list[str]:
    command = fixed._child_command(
        args, workload=workload, output=output, run_id=run_id)
    canonical = "eval.sota_4node.run_tempo_pd_elastic_stream_metrics"
    _require(canonical in command, "canonical stream client seam is missing")
    command[command.index(canonical)] = (
        "eval.sota_4node.run_tempo_go_c6_stream_client")
    return command


def _warmup(args, tokenizer, templates) -> int:
    root = args.output.parent / "c7_actual_vllm_fanin_preflight"
    root.mkdir()
    workload = root / "preflight.jsonl"
    raw = root / "preflight.raw.json"
    rows = []
    for source in (0, 1):
        rows.append({
            "request_id": _request_id(
                name="preflight", role="aggressor", ordinal=source,
                source=source),
            "prompt": fixed._unique_prompt(
                tokenizer, templates[512], 8192 + source),
            "max_tokens": 2,
            "arrival_offset_ms": source * 500.0,
        })
    workload.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    subprocess.run(
        _child_command(
            args, workload=workload, output=raw,
            run_id=f"{args.run_id}-fanin-preflight"),
        check=True,
        timeout=1200.0,
    )
    artifact = json.loads(raw.read_text(encoding="utf-8"))
    decisions = {
        row["request_id"]: row for row in artifact.get("router_decisions", [])
    }
    _require(set(decisions) == {row["request_id"] for row in rows},
             "C7 preflight decisions differ")
    for source, row in enumerate(rows):
        decision = decisions[row["request_id"]]
        _require(
            decision.get("route") == REMOTE_ROUTE
            and decision.get("frontend_pair_index") == source
            and decision.get("remote_decoder_index") == TARGET_DECODER,
            "C7 preflight escaped P0/P1-to-D0 fan-in",
        )
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def _augment_block(
    raw_path: Path, *, spec: dict[str, object], section: dict[str, object],
    schedule_sha256: str, request_index: dict[str, dict[str, object]],
    endpoint_evidence: dict[str, object],
) -> dict[str, object]:
    fixed._validate_endpoint_evidence_bundle(endpoint_evidence)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    validation = raw.get("validation", {})
    _require(validation.get("terminal_contract_valid") is True,
             f"{spec['name']} terminal contract failed")
    rows = raw.get("requests")
    decisions = raw.get("router_decisions")
    _require(isinstance(rows, list) and isinstance(decisions, list),
             f"{spec['name']} terminal evidence is missing")
    row_index = {row.get("request_id"): row for row in rows}
    decision_index = {row.get("request_id"): row for row in decisions}
    _require(
        len(row_index) == len(rows)
        and len(decision_index) == len(decisions)
        and set(row_index) == set(decision_index) == set(request_index),
        f"{spec['name']} request identities differ",
    )
    namespaces = set()
    source_counts = {"0": 0, "1": 0}
    role_counts = {"victim": 0, "aggressor": 0}
    for request_id, metadata in request_index.items():
        row = row_index[request_id]
        decision = decision_index[request_id]
        source = int(metadata["source_prefill_index"])
        role = str(metadata["role"])
        _require(row.get("valid") is True,
                 f"{spec['name']} has an invalid terminal request")
        _require(
            decision.get("route") == REMOTE_ROUTE
            and decision.get("frontend_pair_index") == source
            and decision.get("local_decoder_index") == source
            and decision.get("remote_decoder_index") == TARGET_DECODER,
            f"{spec['name']} escaped its frozen P-to-D edge",
        )
        _require(decision.get("tempo_go_global_commit_applied") is not True,
                 f"{spec['name']} exogenous request consumed a global commit")
        _require(fixed._cold_completion_valid(
            decision, require_explicit_miss=True),
            f"{spec['name']} lacks exact MISS completion")
        namespace = decision.get("cache_namespace")
        _require(isinstance(namespace, str) and namespace
                 and namespace not in namespaces,
                 f"{spec['name']} cache namespace is missing or reused")
        namespaces.add(namespace)
        expected_tokens = (
            int(section["victim"]["output_tokens"])
            if role == "victim"
            else int(section["aggressor"]["output_tokens"])
        )
        _require(len(row.get("output_token_values", [])) == expected_tokens,
                 f"{spec['name']} output token count differs")
        source_counts[str(source)] += role == "aggressor"
        role_counts[role] += 1
    rate = float(spec["aggressor_rate_per_s"])
    _require(
        rate == 0.0 or source_counts["0"] > 0 and source_counts["1"] > 0,
        f"{spec['name']} did not materialize two-source fan-in",
    )
    contract = {
        "schema": BLOCK_SCHEMA,
        "name": spec["name"],
        "aggressor_rate_per_s": rate,
        "phase_duration_ms": section["phase_duration_ms"],
        "semantic_schedule_sha256": schedule_sha256,
        "request_counts": role_counts,
        "aggressor_source_counts": source_counts,
        "request_index": request_index,
        "actual_vllm_lmcache_native": True,
        "actual_two_prefill_to_one_decoder_fanin": rate == 0.0 or all(
            source_counts[str(index)] > 0 for index in (0, 1)),
        "target_decoder_index": TARGET_DECODER,
        "same_client_clock_for_victim_and_aggressor": True,
        "exogenous_aggressor_not_controller_movable": True,
        "explicit_miss_for_every_request": True,
        "endpoint_evidence_exact": True,
        "phase_or_future_arrival_policy_input": False,
    }
    raw["c7_actual_vllm_fanin_contract"] = contract
    raw["endpoint_evidence"] = endpoint_evidence
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _measured(args, tokenizer, templates, section: dict[str, object]) -> int:
    specs = section["blocks"]
    root = args.output.parent / "c7_actual_vllm_fanin_measured"
    workload_root = args.output.parent / "c7_actual_vllm_fanin_workloads"
    root.mkdir()
    workload_root.mkdir()
    artifacts: dict[str, str] = {}
    contracts: dict[str, object] = {}
    for sequence, spec in enumerate(specs):
        name = str(spec["name"])
        schedule, identities = _materialize_schedule(
            spec=spec, section=section)
        workload_path = workload_root / f"{name}.jsonl"
        raw_path = root / f"{name}.raw.json"
        request_index = fixed._write_workload(
            workload_path,
            requests=schedule,
            templates=templates,
            tokenizer=tokenizer,
            marker_base=(sequence + 1) * 32768,
        )
        for request_id, identity in identities.items():
            request_index[request_id].update(identity)
        endpoint_evidence = decoder._run_child_with_cadenced_endpoint_evidence(
            _child_command(
                args, workload=workload_path, output=raw_path,
                run_id=f"{args.run_id}-{name}"),
            args=args,
        )
        evidence_path = root / f"{name}.endpoint-evidence.json"
        evidence_path.write_text(
            json.dumps(endpoint_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        contracts[name] = _augment_block(
            raw_path,
            spec=spec,
            section=section,
            schedule_sha256=semantic_schedule_sha256(schedule),
            request_index=request_index,
            endpoint_evidence=endpoint_evidence,
        )
        artifacts[name] = str(raw_path.resolve())
        if sequence + 1 < len(specs):
            time.sleep(args.cooldown_s)

    bundle: dict[str, object] = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "block_order": list(specs),
        "artifacts": artifacts,
        "contracts": contracts,
        "qualification_contract": str(args.qualification_contract.resolve()),
        "qualification_contract_sha256": _sha256(args.qualification_contract),
        "source_workload": str(args.workload.resolve()),
        "source_workload_sha256": _sha256(args.workload),
        "controller_performance_run_allowed": False,
        "performance_claim_allowed": False,
    }
    bundle["analysis"] = analyzer.analyze_bundle(
        bundle, args.qualification_contract)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "qualification_pass": bundle["analysis"][
            "c7_actual_vllm_fanin_qualification_pass"],
        "first_material_knee_rate_per_s": bundle["analysis"][
            "first_material_knee_rate_per_s"],
    }, sort_keys=True))
    return 0


def main() -> int:
    args = decoder._parse()
    _require(args.mode == "tempo_auto", "C7 fan-in requires tempo_auto")
    _require(not args.output.exists(), f"refusing to overwrite: {args.output}")
    _require(args.model.is_absolute(), "model path must be absolute")
    _require(args.max_workers > 0, "max-workers must be positive")
    _require(math.isfinite(args.phase_duration_ms)
             and args.phase_duration_ms >= 30_000.0,
             "phase duration must be at least 30 seconds")
    _require(len(args.endpoint_evidence_url) == 4,
             "four endpoint probes are required")
    _qualification, section = _load_contract(args)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True)
    templates = fixed._load_templates(args.workload, tokenizer)
    if args.run_id.endswith("-warmup"):
        return _warmup(args, tokenizer, templates)
    return _measured(args, tokenizer, templates, section)


if __name__ == "__main__":
    raise SystemExit(main())
