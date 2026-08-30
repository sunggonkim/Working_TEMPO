#!/usr/bin/env python3
"""Run C7 local protection plus a physically preseeded C8 remote regime."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from eval.sota_4node import analyze_tempo_go_c8_dual_regime as analyzer
from eval.sota_4node import run_tempo_go_c7_joint_control_client as c7
from tempo.pd_global_profile import load_global_profile


SCHEMA = analyzer.BUNDLE_SCHEMA
CONTRACT_SCHEMA = analyzer.CONTRACT_SCHEMA
BLOCK_SCHEMA = "tempo-go-c8-dual-regime-block-v1"
CONTRACT_ENV = "TEMPO_GO_C8_DUAL_REGIME_CONTRACT"
ARM_ENV = "TEMPO_GO_C8_DUAL_REGIME_ARM"
REMOTE_REGIME = analyzer.REMOTE_REGIME
P_ONLY = "p_only"
P_ONLY_VALUE = "prefill_only"
REMOTE_ROUTE = c7.REMOTE_ROUTE
LOCAL_ROUTE = c7.LOCAL_ROUTE
PRIORITY_SERVICE_LANE_MODES = frozenset({
    "vllm_priority_remote_cache_v1",
    "vllm_priority_business_dual_route_v2",
})


def _patch_c7_seams() -> None:
    """Reuse frozen C7 mechanics without changing the C7 source snapshot."""
    c7.SCHEMA = SCHEMA
    c7.CONTRACT_SCHEMA = CONTRACT_SCHEMA
    c7.CONTRACT_ENV = CONTRACT_ENV
    c7.ARM_ENV = ARM_ENV
    c7.analyzer = analyzer


_patch_c7_seams()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm() -> str:
    _patch_c7_seams()
    return c7._arm()


def _validated_global_edge(
    decision: dict[str, object], *, route: str,
) -> tuple[int, int]:
    """Resolve a global edge without confusing destination with source.

    ``frontend_pair_index`` is the globally committed destination pair.  A
    replicated P_ONLY request can be served by either prefill owner, carried
    separately in the global commit.  Treating the destination as the source
    rejects valid p0->d1/p1->d0 mesh placements after successful execution.
    """

    source = decision.get("tempo_go_global_commit_prefill_index")
    decoder = decision.get("tempo_go_global_commit_decoder_index")
    pair = decision.get("tempo_go_global_commit_pair_index")
    ingress_pair = decision.get("frontend_pair_index")
    _require(
        all(type(value) is int for value in (
            source, decoder, pair, ingress_pair))
        and pair == ingress_pair == decoder,
        "C8 global destination commitment differs",
    )
    assert type(source) is int and type(decoder) is int
    actual_decoder = decision.get(
        "local_decoder_index" if route == LOCAL_ROUTE
        else "remote_decoder_index")
    _require(actual_decoder == decoder,
             "C8 global decoder actuation differs")
    _require(
        decision.get("tempo_go_global_commit_edge_id")
        == c7._canonical_edge(route, source, decoder),
        "C8 global edge commitment differs",
    )
    owners = decision.get("frontend_pair_affinity_owner_indices")
    _require(
        isinstance(owners, list)
        and all(type(value) is int for value in owners)
        and source in owners,
        "C8 global prefill source lacks replicated affinity evidence",
    )
    if route == LOCAL_ROUTE:
        _require(source == decoder,
                 "C8 local global edge crosses a pair")
    return source, decoder


def _decoder_contract(value: dict[str, object]) -> dict[str, object]:
    _patch_c7_seams()
    return c7._decoder_contract(value)


def configure_node_environment(**kwargs) -> None:
    _patch_c7_seams()
    # Every P_ONLY prompt is physically seeded and probed on both producer
    # pairs before the measured block.  The frontend registers only completed
    # shadow+primary EOF receipts, so this flag cannot invent cache residency.
    os.environ["TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY"] = "1"
    c7.configure_node_environment(**kwargs)
    repo_root = kwargs.get("repo_root")
    qualification = kwargs.get("qualification")
    _require(isinstance(repo_root, Path), "C8 repo root is missing")
    section = _decoder_contract(qualification)
    profile_spec = section.get("global_profile")
    _require(isinstance(profile_spec, dict), "C8 global profile is missing")
    profile_path = (repo_root / str(profile_spec.get("path", ""))).resolve()
    profile = load_global_profile(profile_path)
    config = profile.orchestrator_config()
    remote_activation = section.get("remote_activation")
    _require(isinstance(remote_activation, dict),
             "C8 remote activation contract is missing")
    _require(
        config.priority_service_lane_mode
        == remote_activation.get("priority_service_lane_mode")
        and config.priority_service_lane_mode in PRIORITY_SERVICE_LANE_MODES
        and config.priority_service_lane_capacity
        == remote_activation.get("priority_service_lane_capacity_per_decoder")
        and config.priority_service_lane_min_admission_priority
        == remote_activation.get("priority_service_lane_min_admission_priority")
        and config.priority_service_lane_priority
        == remote_activation.get("managed_remote_priority"),
        "C8 global priority service-lane binding differs",
    )
    expected_background_limits = [
        item.resources.active_sequences
        - config.priority_service_lane_capacity
        for item in sorted(
            profile.capacities, key=lambda value: value.pair_index)
    ]
    _require(
        config.decoder_business_admission_mode
        == remote_activation.get("decoder_business_admission_mode")
        == "priority_drain_v1"
        and expected_background_limits
        == remote_activation.get("decoder_background_concurrency_limits")
        and config.decoder_business_background_max_wait_ns
        == remote_activation.get("decoder_background_max_wait_ns")
        and remote_activation.get(
            "decoder_background_requests_are_delayed_not_dropped") is True,
        "C8 decoder business admission binding differs",
    )
    _require(
        profile.telemetry.freshness_ns
        == remote_activation.get("telemetry_freshness_ns")
        and profile.telemetry.refresh_timeout_ns
        == remote_activation.get("telemetry_refresh_timeout_ns")
        and profile.telemetry.maximum_collection_span_ns
        == remote_activation.get("telemetry_maximum_collection_span_ns")
        and min(
            profile.telemetry.refresh_timeout_ns,
            max(
                1_000_000,
                profile.telemetry.maximum_collection_span_ns // 2,
            ),
        ) == remote_activation.get("telemetry_per_fetch_timeout_ns")
        and remote_activation.get("all_endpoint_fetch_failure_policy")
        == "preserve_last_complete_batch_then_tenant_stale_grace",
        "C8 control-plane liveness binding differs",
    )
    # Every arm uses the same vLLM priority scheduler.  Baseline requests keep
    # priority zero; only a globally committed TEMPO request that is eligible
    # for the selected service-lane mode receives the frozen negative priority
    # in the canonical router.
    os.environ.update({
        "TEMPO_VLLM_SCHEDULING_POLICY": "priority",
        "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "0",
        "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY": str(
            config.priority_service_lane_priority),
        "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY": "0",
        "TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY": "0",
        "TEMPO_PD_MEDIAN_GUARD_PRIORITY": "0",
    })


def _load_contract(args):
    _patch_c7_seams()
    return c7._load_contract(args)


def _is_remote_regime(spec: dict[str, object]) -> bool:
    return spec.get("pressure_regime") == REMOTE_REGIME


def _victim_identity(
    *, block: str, ordinal: int,
) -> tuple[str, dict[str, object]]:
    arm = _arm()
    owner = ordinal % 2
    base = f"c8-dual-{block}-victim-{ordinal:06d}"
    marker = "cache-p-only-measured"
    if arm == "fixed_local_d0":
        owner = 0
        return (
            f"epd-local-interactive-{marker}-{base}-0",
            {"expected_route": LOCAL_ROUTE, "expected_source": 0,
             "expected_decoder": 0, "p_only_owner": owner},
        )
    if arm == "fixed_local_d1":
        owner = 1
        return (
            f"epd-local-interactive-{marker}-{base}-1",
            {"expected_route": LOCAL_ROUTE, "expected_source": 1,
             "expected_decoder": 1, "p_only_owner": owner},
        )
    if arm == "fixed_remote_p0d1":
        owner = 0
        return (
            "epd-remote-interactive-cache-p-only-measured-"
            f"tempo-go-exogenous-fixed-remote-d1-{base}-0",
            {"expected_route": REMOTE_ROUTE, "expected_source": 0,
             "expected_decoder": 1, "p_only_owner": owner},
        )
    if arm == "fixed_remote_p1d0":
        owner = 1
        return (
            "epd-remote-interactive-cache-p-only-measured-"
            f"tempo-go-exogenous-fixed-remote-d0-{base}-1",
            {"expected_route": REMOTE_ROUTE, "expected_source": 1,
             "expected_decoder": 0, "p_only_owner": owner},
        )
    arm_marker = {
        "full_c7": "tempo",
        c7.MANAGED_BACKGROUND_ARM: "tempo",
        "app_global_only": "app_global_only",
        "predictor": "predictor",
        "queue_gpu": "queue_gpu",
        "network_request_only": "network_request_only",
    }[arm]
    return (
        f"epd-{arm_marker}-interactive-{marker}-{base}-{owner}",
        {"expected_route": None, "expected_source": None,
         "expected_decoder": None, "p_only_owner": owner},
    )


def _materialize_remote_schedule(
    *, spec: dict[str, object], section: dict[str, object],
) -> tuple[tuple[c7.ScheduledRequest, ...], dict[str, dict[str, object]]]:
    name = str(spec["name"])
    targets = tuple(int(value) for value in spec["hot_decoder_indices"])
    _require(targets == (0, 1), "C8 remote regime requires both decoders hot")
    _require(spec.get("managed_background") is False,
             "C8 remote pressure must remain exogenous")
    duration_ms = float(section["phase_duration_ms"])
    victim = section["victim"]
    local = section["local_aggressor"]
    victim_geometry = c7.TokenGeometry(
        int(victim["prompt_tokens"]), int(victim["output_tokens"]),
        c7.CacheState.P_ONLY,
    )
    local_geometry = c7.TokenGeometry(
        int(local["prompt_tokens"]), int(local["output_tokens"]),
        c7.CacheState.MISS,
    )
    requests: list[c7.ScheduledRequest] = []
    identities: dict[str, dict[str, object]] = {}
    for ordinal, offset in enumerate(c7._uniform_offsets(
        duration_ms, float(victim["offered_rate_per_s"]))):
        request_id, expected = _victim_identity(
            block=name, ordinal=ordinal)
        requests.append(c7.ScheduledRequest(
            request_id=request_id,
            phase=c7.ContentionState.C2,
            tenant=c7.Tenant.FOREGROUND,
            arm=(
                c7.ForegroundArm.TEMPO if _arm() in c7.GLOBAL_ARMS
                else c7.ForegroundArm.PREDICTOR if _arm() == "predictor"
                else c7.ForegroundArm.QUEUE_ONLY if _arm() == "queue_gpu"
                else c7.ForegroundArm.TEMPO
                if _arm() == "network_request_only"
                else c7.ForegroundArm.LOCAL
                if _arm().startswith("fixed_local")
                else c7.ForegroundArm.REMOTE
            ),
            arrival_offset_ms=offset,
            geometry=victim_geometry,
            ordinal=ordinal,
        ))
        identities[request_id] = {
            "role": "victim",
            "business_tenant": "interactive",
            "hot_decoder_index": 0,
            "hot_decoder_indices": list(targets),
            "block": name,
            "expected_cache": P_ONLY,
            "prompt_tokens": victim_geometry.prompt_tokens,
            "output_tokens": victim_geometry.output_tokens,
            **expected,
        }
    local_rate = float(spec["local_aggressor_rate_per_s"])
    for target in targets:
        for ordinal, offset in enumerate(c7._uniform_offsets(
            duration_ms, local_rate)):
            request_id = c7._local_aggressor_id(
                block=name, ordinal=ordinal, decoder_index=target)
            requests.append(c7.ScheduledRequest(
                request_id=request_id,
                phase=c7.ContentionState.C2,
                tenant=c7.Tenant.DECODER_HOT,
                arm=c7.ForegroundArm.LOCAL,
                arrival_offset_ms=offset,
                geometry=local_geometry,
                ordinal=ordinal,
            ))
            identities[request_id] = {
                "role": "local_aggressor",
                "business_tenant": "background_local_decoder",
                "source_prefill_index": target,
                "target_decoder_index": target,
                "hot_decoder_index": 0,
                "hot_decoder_indices": list(targets),
                "block": name,
                "managed_by_tempo_go": False,
                "expected_cache": "miss",
                "prompt_tokens": local_geometry.prompt_tokens,
                "output_tokens": local_geometry.output_tokens,
            }
    requests.sort(key=lambda row: (
        row.arrival_offset_ms,
        0 if row.tenant is c7.Tenant.FOREGROUND else 1,
        row.ordinal,
        row.request_id,
    ))
    _require(len(requests) == len(identities),
             "C8 remote-regime identities are not unique")
    return tuple(requests), identities


def _p_only_prompt(
    tokenizer, template: tuple[int, ...], *, sequence: int,
    owner: int, pool_index: int, pool_size: int,
) -> str:
    marker = 240_000 + sequence * 32 + owner * pool_size + pool_index
    _require(marker < (1 << 18), "C8 P_ONLY marker space exhausted")
    return c7.fixed._unique_prompt(tokenizer, template, marker)


def _write_remote_workload(
    path: Path, *, schedule: tuple[c7.ScheduledRequest, ...],
    identities: dict[str, dict[str, object]], templates, tokenizer,
    sequence: int, pool_size: int,
) -> tuple[dict[str, dict[str, object]], dict[tuple[int, int], str]]:
    _require(not path.exists(), f"refusing to overwrite {path}")
    rows = []
    index: dict[str, dict[str, object]] = {}
    pools: dict[tuple[int, int], str] = {}
    prompt_chunks: dict[tuple[int, ...], tuple[int, int] | str] = {}
    cold_item = 0
    for request in schedule:
        identity = identities[request.request_id]
        if identity["role"] == "victim":
            owner = int(identity["p_only_owner"])
            pool_index = (
                request.ordinal // 2 if _arm() not in c7.FIXED_ARMS
                else request.ordinal
            ) % pool_size
            key = (owner, pool_index)
            prompt = pools.setdefault(key, _p_only_prompt(
                tokenizer,
                templates[request.geometry.prompt_tokens],
                sequence=sequence,
                owner=owner,
                pool_index=pool_index,
                pool_size=pool_size,
            ))
            identity["p_only_pool_key"] = f"p{owner}:item{pool_index:02d}"
        else:
            marker = (sequence + 1) * 32768 + cold_item
            cold_item += 1
            _require(marker < 240_000, "C8 cold marker overlaps P_ONLY pool")
            prompt = c7.fixed._unique_prompt(
                tokenizer, templates[request.geometry.prompt_tokens], marker)
            key = f"cold:{cold_item:06d}"
        chunk = tuple(tokenizer.encode(
            prompt, add_special_tokens=False)[:256])
        prior = prompt_chunks.get(chunk)
        if identity["role"] == "victim":
            _require(prior in (None, key),
                     "C8 P_ONLY pool aliases another prompt")
        else:
            _require(prior is None, "C8 cold prompt chunk is reused")
        prompt_chunks[chunk] = key
        rows.append({
            "request_id": request.request_id,
            "prompt": prompt,
            "max_tokens": request.geometry.output_tokens,
            "arrival_offset_ms": round(request.arrival_offset_ms, 6),
        })
        index[request.request_id] = {
            **request.semantic_dict(),
            "arm": request.arm.value,
            "pair_key": (
                f"{request.phase.value}:foreground:{request.ordinal:06d}"
                if request.tenant is c7.Tenant.FOREGROUND else None
            ),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            **identity,
        }
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return index, pools


def _physical_preseed_request_id(
    *, name: str, pool_index: int, owner: int,
) -> str:
    _require(bool(name), "C8 preseed block name must be nonempty")
    _require(pool_index >= 0, "C8 preseed pool index must be nonnegative")
    _require(owner in {0, 1}, "C8 preseed owner must be a physical pair")
    arm_marker = {
        "fixed_local_d0": "local",
        "fixed_local_d1": "local",
        "fixed_remote_p0d1": "remote",
        "fixed_remote_p1d0": "remote",
        "predictor": "predictor",
        "queue_gpu": "queue_gpu",
        "network_request_only": "network_request_only",
        "app_global_only": "app_global_only",
        "full_c7": "tempo",
        c7.MANAGED_BACKGROUND_ARM: "tempo",
    }[_arm()]
    return (
        f"epd-{arm_marker}-interactive-c4-cache-p-only-warm-physical-"
        f"c8-{name}-pool-{pool_index:02d}-owner-{owner}-"
        f"item-{owner:06d}"
    )


def _preseed_remote_pool(
    args, *, root: Path, name: str, pools: dict[tuple[int, int], str],
    output_tokens: int,
) -> dict[str, object]:
    root.mkdir(exist_ok=False)
    workload = root / f"{name}.preseed.jsonl"
    raw_path = root / f"{name}.preseed.raw.json"
    rows = []
    expected = {}
    for sequence, ((owner, pool_index), prompt) in enumerate(sorted(pools.items())):
        request_id = _physical_preseed_request_id(
            name=name, pool_index=pool_index, owner=owner)
        rows.append({
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "arrival_offset_ms": round(sequence * 250.0, 6),
        })
        expected[request_id] = owner
    workload.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    subprocess.run(
        c7._child_command(
            args, workload=workload, output=raw_path,
            run_id=f"{args.run_id}-{name}-p-only-preseed"),
        check=True,
        timeout=1200.0,
    )
    artifact = json.loads(raw_path.read_text(encoding="utf-8"))
    validation = artifact.get("validation", {})
    _require(
        validation.get("terminal_contract_valid") is True
        or validation.get("all_streams_valid") is True,
        "C8 P_ONLY preseed terminal contract failed",
    )
    request_index = {row["request_id"]: row for row in artifact["requests"]}
    decisions = {row["request_id"]: row for row in artifact["router_decisions"]}
    _require(set(request_index) == set(decisions) == set(expected),
             "C8 P_ONLY preseed identities differ")
    for request_id, owner in expected.items():
        row = request_index[request_id]
        decision = decisions[request_id]
        seed = row.get("p_only_cache_seed")
        _require(row.get("valid") is True
                 and isinstance(seed, dict) and seed.get("valid") is True
                 and seed.get("route") == REMOTE_ROUTE,
                 "C8 physical cache seed evidence is missing")
        _require(
            decision.get("route") == REMOTE_ROUTE
            and decision.get("frontend_pair_index") == owner
            and decision.get("lmcache_source_cached_tokens") == 4094
            and decision.get("lmcache_source_full_hit_observed") is True
            and decision.get("completion_cache_residency") == P_ONLY_VALUE,
            "C8 P_ONLY probe lacks exact replicated source-hit evidence",
        )
    return {
        "schema": "tempo-go-c8-p-only-preseed-v1",
        "measured": False,
        "performance_claim_allowed": False,
        "workload": str(workload.resolve()),
        "workload_sha256": _sha256(workload),
        "raw": str(raw_path.resolve()),
        "raw_sha256": _sha256(raw_path),
        "prompt_count": len(rows),
        "preseed_completed_before_measurement": True,
        "all_full_source_hits_exact": True,
        "replicated_affinity_required_for_measurement": True,
    }


def _augment_remote_block(
    raw_path: Path, *, spec: dict[str, object], section: dict[str, object],
    schedule_sha256: str, request_index: dict[str, dict[str, object]],
    endpoint_evidence: dict[str, object], preseed: dict[str, object],
) -> dict[str, object]:
    c7.fixed._validate_endpoint_evidence_bundle(endpoint_evidence)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    validation = raw.get("validation", {})
    _require(validation.get("terminal_contract_valid") is True,
             "C8 remote block terminal contract failed")
    rows = raw["requests"]
    decisions = raw["router_decisions"]
    row_index = {row["request_id"]: row for row in rows}
    decision_index = {row["request_id"]: row for row in decisions}
    _require(
        len(row_index) == len(rows)
        and len(decision_index) == len(decisions)
        and set(row_index) == set(decision_index) == set(request_index),
        "C8 remote block request identities differ",
    )
    counts = collections.Counter(
        metadata["role"] for metadata in request_index.values())
    cold_namespaces = set()
    warm_namespaces: dict[str, str] = {}
    remote_full_hits = 0
    remote_victim_completions = 0
    for request_id, metadata in request_index.items():
        row = row_index[request_id]
        decision = decision_index[request_id]
        role = str(metadata["role"])
        terminal = row.get("terminal_kind")
        if terminal == "global_reject":
            _require(role == "victim" and _arm() in c7.GLOBAL_ARMS,
                     "C8 exogenous background was globally rejected")
            _require(decision.get("tempo_go_global_commit_applied") is False,
                     "C8 rejected victim has a global commit")
            continue
        if terminal == "service_lane_failure":
            _require(isinstance(row.get("terminal_error_kind"), str)
                     and row["terminal_error_kind"].startswith("endpoint_"),
                     "C8 service-lane failure is unclassified")
            continue
        _require(row.get("valid") is True,
                 "C8 remote block has an invalid terminal request")
        route = decision.get("route")
        _require(route in {LOCAL_ROUTE, REMOTE_ROUTE},
                 "C8 completion lacks a native route")
        namespace = decision.get("cache_namespace")
        _require(isinstance(namespace, str) and namespace,
                 "C8 completion lacks cache namespace")
        if role == "victim":
            pool_key = str(metadata["p_only_pool_key"])
            prior = warm_namespaces.setdefault(pool_key, namespace)
            _require(prior == namespace,
                     "C8 P_ONLY prompt namespace changed across reuse")
            _require(
                decision.get("request_cache_contract") == P_ONLY
                and decision.get("decision_cache_residency") == P_ONLY_VALUE
                and decision.get("cache_residency") == P_ONLY_VALUE
                and decision.get("completion_cache_residency") == P_ONLY_VALUE,
                "C8 victim lacks completed P_ONLY residency",
            )
            if route == REMOTE_ROUTE:
                remote_victim_completions += 1
                _require(
                    decision.get("lmcache_source_cached_tokens")
                    == int(metadata["prompt_tokens"])
                    and decision.get("lmcache_source_full_hit_observed") is True,
                    "C8 remote victim lost its official LMCache full source hit",
                )
                remote_full_hits += 1
            else:
                _require(decision.get("lmcache_source_cached_tokens") is None,
                         "C8 local victim has remote source-cache evidence")
        else:
            _require(c7.fixed._cold_completion_valid(
                decision, require_explicit_miss=True),
                "C8 local aggressor lacks exact MISS completion")
            _require(namespace not in cold_namespaces
                     and namespace not in warm_namespaces.values(),
                     "C8 cold aggressor namespace was reused")
            cold_namespaces.add(namespace)
        _require(
            len(row.get("output_token_values", []))
            == int(metadata["output_tokens"]),
            "C8 output-token count differs",
        )
        source = int(decision["frontend_pair_index"])
        decoder_index = int(
            decision["local_decoder_index"]
            if route == LOCAL_ROUTE else decision["remote_decoder_index"])
        if role == "local_aggressor":
            _require(
                route == LOCAL_ROUTE
                and source == metadata["source_prefill_index"]
                and decoder_index == metadata["target_decoder_index"]
                and decision.get("tempo_go_global_commit_applied") is not True,
                "C8 exogenous local aggressor escaped its decoder",
            )
            if _arm() in c7.GLOBAL_ARMS:
                admission = decision.get(
                    "frontend_decoder_business_admission")
                _require(
                    isinstance(admission, dict)
                    and admission.get("status") == "released"
                    and admission.get("admission_class") == "background"
                    and admission.get("pair_index") == decoder_index,
                    "C8 background lacks decoder admission receipt",
                )
        elif _arm() in c7.FIXED_ARMS:
            _require(
                route == metadata["expected_route"]
                and source == metadata["expected_source"]
                and decoder_index == metadata["expected_decoder"],
                "C8 fixed victim escaped its edge",
            )
        elif _arm() in c7.PAIRED_BASELINES:
            _require(
                decision.get("tempo_go_global_commit_applied") is not True
                and decoder_index == source,
                "C8 paired baseline escaped its pair",
            )
        else:
            _require(_arm() in c7.GLOBAL_ARMS
                     and decision.get("tempo_go_global_commit_applied") is True,
                     "C8 global victim lacks a commit")
            source, decoder_index = _validated_global_edge(
                decision, route=route)
            admission = decision.get("frontend_decoder_business_admission")
            _require(
                isinstance(admission, dict)
                and admission.get("status") == "released"
                and admission.get("admission_class") == "protected"
                and admission.get("pair_index") == decoder_index,
                "C8 global victim lacks protected decoder admission",
            )
    workload = raw.get("workload", {})
    ingress = section.get("ingress", {})
    _require(workload.get("ingress_policy")
             == ingress.get("policy", "shared_pool")
             and workload.get("interactive_reserved_workers")
             == ingress.get("interactive_reserved_workers", 0),
             "C8 ingress receipt differs")
    contract = {
        "schema": BLOCK_SCHEMA,
        "name": spec["name"],
        "arm": _arm(),
        "hot_decoder_index": spec["hot_decoder_index"],
        "hot_decoder_indices": list(spec["hot_decoder_indices"]),
        "pressure_regime": spec["pressure_regime"],
        "victim_cache_state": "p_only",
        "aggressor_rate_per_s": 0.0,
        "local_aggressor_rate_per_s": spec["local_aggressor_rate_per_s"],
        "local_aggressor_total_rate_per_s": (
            float(spec["local_aggressor_rate_per_s"])
            * len(spec["hot_decoder_indices"])),
        "phase_duration_ms": section["phase_duration_ms"],
        "semantic_schedule_sha256": schedule_sha256,
        "request_counts": dict(sorted(counts.items())),
        "request_index": request_index,
        "same_client_clock_for_victim_and_aggressor": True,
        "exogenous_aggressor_not_controller_movable": True,
        "managed_background_global_admission": False,
        "actual_vllm_lmcache_native": True,
        "explicit_p_only_for_every_victim": True,
        "explicit_miss_for_every_aggressor": True,
        "remote_victim_completions": remote_victim_completions,
        "remote_victim_exact_full_source_hits": remote_full_hits,
        "p_only_preseed": preseed,
        "endpoint_evidence_exact": True,
        "phase_or_future_arrival_policy_input": False,
        "ingress_policy": workload["ingress_policy"],
        "interactive_reserved_workers": workload[
            "interactive_reserved_workers"],
    }
    raw["c8_dual_regime_contract"] = contract
    raw["endpoint_evidence"] = endpoint_evidence
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _measured(args, tokenizer, templates, section: dict[str, object]) -> int:
    root = args.output.parent / f"c8_dual_{_arm()}_measured"
    workload_root = args.output.parent / f"c8_dual_{_arm()}_workloads"
    preseed_root = args.output.parent / f"c8_dual_{_arm()}_preseed"
    root.mkdir()
    workload_root.mkdir()
    preseed_root.mkdir()
    artifacts: dict[str, str] = {}
    contracts: dict[str, object] = {}
    for sequence, spec in enumerate(section["blocks"]):
        name = str(spec["name"])
        remote_regime = _is_remote_regime(spec)
        if remote_regime:
            schedule, identities = _materialize_remote_schedule(
                spec=spec, section=section)
        else:
            schedule, identities = c7._materialize_schedule(
                spec=spec, section=section)
        workload_path = workload_root / f"{name}.jsonl"
        raw_path = root / f"{name}.raw.json"
        if remote_regime:
            request_index, pools = _write_remote_workload(
                workload_path,
                schedule=schedule,
                identities=identities,
                templates=templates,
                tokenizer=tokenizer,
                sequence=sequence,
                pool_size=int(spec["p_only_pool_per_owner"]),
            )
            preseed = _preseed_remote_pool(
                args,
                root=preseed_root / name,
                name=name,
                pools=pools,
                output_tokens=int(section["victim"]["output_tokens"]),
            )
        else:
            request_index = c7.fixed._write_workload(
                workload_path,
                requests=schedule,
                templates=templates,
                tokenizer=tokenizer,
                marker_base=(sequence + 1) * 32768,
            )
            for request_id, identity in identities.items():
                request_index[request_id].update(identity)
            preseed = None
        endpoint_evidence = c7.decoder._run_child_with_cadenced_endpoint_evidence(
            c7._child_command(
                args, workload=workload_path, output=raw_path,
                run_id=f"{args.run_id}-{name}"),
            args=args,
        )
        evidence_path = root / f"{name}.endpoint-evidence.json"
        evidence_path.write_text(
            json.dumps(endpoint_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if remote_regime:
            assert preseed is not None
            contracts[name] = _augment_remote_block(
                raw_path,
                spec=spec,
                section=section,
                schedule_sha256=c7.semantic_schedule_sha256(schedule),
                request_index=request_index,
                endpoint_evidence=endpoint_evidence,
                preseed=preseed,
            )
        else:
            contracts[name] = c7._augment_block(
                raw_path,
                spec=spec,
                section=section,
                schedule_sha256=c7.semantic_schedule_sha256(schedule),
                request_index=request_index,
                endpoint_evidence=endpoint_evidence,
            )
        artifacts[name] = str(raw_path.resolve())
        if sequence + 1 < len(section["blocks"]):
            time.sleep(args.cooldown_s)
    bundle: dict[str, object] = {
        "schema": SCHEMA,
        "run_id": args.run_id,
        "arm": _arm(),
        "block_order": list(section["blocks"]),
        "artifacts": artifacts,
        "contracts": contracts,
        "qualification_contract": str(args.qualification_contract.resolve()),
        "qualification_contract_sha256": _sha256(args.qualification_contract),
        "source_workload": str(args.workload.resolve()),
        "source_workload_sha256": _sha256(args.workload),
        "performance_claim_allowed": False,
    }
    bundle["analysis"] = analyzer.analyze_arm_bundle(
        bundle, args.qualification_contract)
    args.output.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": SCHEMA,
        "arm": _arm(),
        "output": str(args.output.resolve()),
        "miss_hot_slo_good": bundle["analysis"]["miss_hot"][
            "slo_good_victims"],
        "miss_hot_p99_ms": bundle["analysis"]["miss_hot"]["victim"][
            "e2e_ms"]["p99"],
        "remote_favorable_slo_good": bundle["analysis"][
            "remote_favorable"]["slo_good_victims"],
        "remote_favorable_p99_ms": bundle["analysis"][
            "remote_favorable"]["victim"]["e2e_ms"]["p99"],
        "remote_favorable_routes": bundle["analysis"][
            "remote_favorable"]["route_counts"],
    }, sort_keys=True))
    return 0


def main() -> int:
    _patch_c7_seams()
    args = c7.decoder._parse()
    _require(args.mode == "tempo_auto", "C8 requires tempo_auto")
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
    templates = c7.fixed._load_templates(args.workload, tokenizer)
    if args.run_id.endswith("-warmup"):
        return c7._warmup(args, tokenizer, templates)
    return _measured(args, tokenizer, templates, section)


if __name__ == "__main__":
    raise SystemExit(main())
