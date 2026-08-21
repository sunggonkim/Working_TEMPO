#!/usr/bin/env python3
"""Freeze the exploratory C4 semantic-epoch implementation and inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.sota_4node import build_tempo_pd_semantic_epoch_endpoint_profile as semantic_profile_builder
from eval.sota_4node import verify_tempo_pd_c4_implementation as fixed_implementation
from tempo.pd_endpoint_profile import SCHEMA_V2, load_endpoint_service_profile


SCHEMA = "tempo-pd-c4-semantic-epoch-screen-run-contract-v2"
BASE_SCHEMA = "tempo-pd-c4-phase-screen-run-contract-v1"
OBSERVER_RESULT_SCHEMA = "tempo-pd-c4-phase-screen-node-v1"
OBSERVER_ANALYSIS_SCHEMA = "tempo-pd-c4-semantic-load-analysis-v1"
IMPLEMENTATION_FILES = (
    "tempo/pd_endpoint_controller.py",
    "tempo/pd_endpoint_profile.py",
    "eval/sota_4node/analyze_tempo_pd_c4_phase_screen.py",
    "eval/sota_4node/analyze_tempo_pd_c4_semantic_epoch_screen.py",
    "eval/sota_4node/tempo_pd_elastic_frontend.py",
    "eval/sota_4node/tempo_pd_elastic_router.py",
    "eval/sota_4node/tempo_pd_elastic_router_v448.py",
    "eval/sota_4node/c4_phase_screen_pd_node_entry.sh",
    "eval/sota_4node/prepare_c4_python_overlay.sh",
    "eval/sota_4node/stage_c4_python_overlay.sh",
    "eval/sota_4node/run_tempo_pd_c4_phase_screen_client.py",
    "eval/sota_4node/vllm_lmcache_pd_c4_phase_screen_node.py",
    "eval/sota_4node/run_tempo_pd_c4_semantic_epoch_screen_in_allocation.sh",
    "eval/sota_4node/build_tempo_pd_c4_semantic_epoch_run_contract.py",
    "eval/sota_4node/build_tempo_pd_semantic_epoch_endpoint_profile.py",
)
SEMANTIC_BASELINE_OVERRIDES = frozenset({
    "eval/sota_4node/tempo_pd_elastic_router.py",
    "tempo/pd_endpoint_profile.py",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _display_path(path: Path) -> str:
    path = path.resolve()
    root = _repo_root()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _entry(path: Path, **extra: object) -> dict[str, object]:
    value = {"path": _display_path(path), "sha256": _sha256(path)}
    value.update(extra)
    return value


def contract_fingerprint(value: dict[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _resolve_base_entry(base: dict[str, object], name: str) -> Path:
    entry = base.get(name)
    _require(isinstance(entry, dict), f"base contract lacks {name}")
    raw = entry.get("path")
    _require(isinstance(raw, str) and raw, f"base {name} path is missing")
    path = Path(raw)
    if not path.is_absolute():
        path = _repo_root() / path
    path = path.resolve()
    _require(path.is_file() and _sha256(path) == entry.get("sha256"),
             f"base {name} differs")
    return path


def verify_fixed_baseline_for_semantic(
    *, repo_root: Path, contract_path: Path, expected_sha256: str,
    phase_manifest: Path, semantic_contract: dict[str, object] | None = None,
) -> dict[str, object]:
    """Verify fixed C4 except files explicitly owned by this scheduler.

    The original fixed contract remains immutable.  Every non-overridden
    baseline file, both Git heads, package versions, and the phase manifest
    are still checked by the original verifier.  The two overridden files
    are admission/profile code and must instead be hash-bound by the semantic
    implementation inventory and the explicit override receipt.
    """

    repo_root = repo_root.resolve()
    contract_path = contract_path.resolve()
    phase_manifest = phase_manifest.resolve()
    _require(_sha256(contract_path) == expected_sha256,
             "fixed C4 implementation contract digest differs")
    value = json.loads(contract_path.read_text(encoding="utf-8"))
    _require(
        value.get("schema") == fixed_implementation.SCHEMA
        and value.get("fingerprint_sha256")
        == fixed_implementation.contract_fingerprint(value)
        and value.get("purpose") == "frozen C4 characterization only",
        "fixed C4 implementation contract differs",
    )
    manifest = value.get("phase_manifest")
    _require(isinstance(manifest, dict)
             and set(manifest) == {"path", "sha256"},
             "fixed C4 phase-manifest binding differs")
    bound_manifest = (repo_root / str(manifest["path"])).resolve()
    _require(
        bound_manifest == phase_manifest
        and bound_manifest.is_file()
        and _sha256(bound_manifest) == manifest["sha256"],
        "fixed C4 phase-manifest implementation binding differs",
    )
    entries = value.get("files")
    _require(isinstance(entries, list) and entries,
             "fixed C4 implementation file bindings are missing")
    observed: set[str] = set()
    baseline_sha: dict[str, str] = {}
    for index, entry in enumerate(entries):
        _require(isinstance(entry, dict)
                 and set(entry) == {"path", "sha256"},
                 f"fixed C4 implementation[{index}] binding differs")
        raw = entry["path"]
        _require(type(raw) is str and raw not in observed,
                 "fixed C4 implementation paths are invalid or duplicated")
        path = (repo_root / raw).resolve()
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError("fixed C4 implementation escapes repository") from exc
        _require(path.is_file(), f"fixed C4 implementation is missing: {raw}")
        baseline_sha[raw] = str(entry["sha256"])
        if raw not in SEMANTIC_BASELINE_OVERRIDES:
            _require(_sha256(path) == entry["sha256"],
                     f"fixed C4 implementation drifted: {raw}")
        observed.add(raw)
    _require(fixed_implementation.REQUIRED_FILES <= observed,
             "fixed C4 implementation omits a required file")
    _require(SEMANTIC_BASELINE_OVERRIDES <= observed,
             "semantic override is absent from the fixed C4 contract")
    _require(
        value.get("git_heads") == {
            "repository": fixed_implementation._git_head(repo_root),
            "third_party_lmcache": fixed_implementation._git_head(
                repo_root / "third_party/lmcache"),
        },
        "fixed C4 Git provenance differs",
    )
    _require(value.get("environment_versions")
             == fixed_implementation._environment_versions(),
             "fixed C4 environment versions differ")
    _require(value.get("performance_claim_allowed") is False,
             "fixed C4 implementation permits a performance claim")

    if semantic_contract is not None:
        implementation = semantic_contract.get("implementation")
        _require(isinstance(implementation, list),
                 "semantic implementation inventory is missing")
        candidate_sha = {
            str(entry.get("path")): str(entry.get("sha256"))
            for entry in implementation if isinstance(entry, dict)
        }
        receipt = semantic_contract.get("fixed_baseline_overrides")
        _require(isinstance(receipt, list)
                 and len(receipt) == len(SEMANTIC_BASELINE_OVERRIDES),
                 "semantic baseline override receipt differs")
        receipt_index = {
            str(entry.get("path")): entry
            for entry in receipt if isinstance(entry, dict)
        }
        _require(set(receipt_index) == SEMANTIC_BASELINE_OVERRIDES,
                 "semantic baseline override paths differ")
        for raw in SEMANTIC_BASELINE_OVERRIDES:
            entry = receipt_index[raw]
            _require(
                set(entry) == {
                    "path", "baseline_sha256", "candidate_sha256", "scope"}
                and entry["baseline_sha256"] == baseline_sha[raw]
                and entry["candidate_sha256"] == candidate_sha.get(raw)
                and entry["candidate_sha256"]
                == _sha256((repo_root / raw).resolve())
                and entry["scope"] == "admission_or_service_profile_only",
                f"semantic baseline override binding differs: {raw}",
            )
    return value


def build(
    *, base_path: Path, observer_result_path: Path,
    observer_analysis_path: Path, fixed_implementation_path: Path,
    semantic_endpoint_profile_path: Path,
) -> dict[str, object]:
    base_path = base_path.resolve()
    observer_result_path = observer_result_path.resolve()
    observer_analysis_path = observer_analysis_path.resolve()
    fixed_implementation_path = fixed_implementation_path.resolve()
    semantic_endpoint_profile_path = semantic_endpoint_profile_path.resolve()
    for path in (
        base_path, observer_result_path, observer_analysis_path,
        fixed_implementation_path, semantic_endpoint_profile_path,
    ):
        _require(path.is_file(), f"required artifact is missing: {path}")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    observer_result = json.loads(
        observer_result_path.read_text(encoding="utf-8"))
    observer_analysis = json.loads(
        observer_analysis_path.read_text(encoding="utf-8"))
    _require(
        base.get("schema") == BASE_SCHEMA
        and base.get("performance_claim_allowed") is False
        and base.get("transport") == "LMCacheConnectorV1:UCX"
        and base.get("unchanged_pd_data_plane") is True,
        "base C4 contract differs",
    )
    _require(
        observer_result.get("schema") == OBSERVER_RESULT_SCHEMA
        and observer_result.get("live_screen_correctness_pass") is True
        and observer_result.get("blocks_completed") == 8
        and observer_result.get("performance_claim_allowed") is False,
        "semantic observer result differs",
    )
    _require(
        observer_analysis.get("schema") == OBSERVER_ANALYSIS_SCHEMA
        and len(observer_analysis.get("blocks", [])) == 8
        and observer_analysis.get("policy_input_used") is False
        and observer_analysis.get("interpretation_limits", {}).get(
            "threshold_selected") is False,
        "semantic observer analysis differs",
    )
    fixed_value = json.loads(
        fixed_implementation_path.read_text(encoding="utf-8"))
    fixed_manifest_entry = fixed_value.get("phase_manifest")
    _require(isinstance(fixed_manifest_entry, dict),
             "fixed implementation phase manifest is missing")
    fixed_manifest = _repo_root() / str(fixed_manifest_entry.get("path"))
    fixed_value = verify_fixed_baseline_for_semantic(
        repo_root=_repo_root(),
        contract_path=fixed_implementation_path,
        expected_sha256=_sha256(fixed_implementation_path),
        phase_manifest=fixed_manifest,
    )
    copied_entries = {}
    for name in (
        "source_workload", "phase_manifest", "elastic_profile",
        "offline_replay",
    ):
        path = _resolve_base_entry(base, name)
        copied_entries[name] = {
            **_entry(path),
            **{
                key: value for key, value in base[name].items()
                if key not in {"path", "sha256"}
            },
        }
    source_endpoint_path = _resolve_base_entry(
        base, "endpoint_service_profile")
    source_endpoint_entry = base["endpoint_service_profile"]
    _require(isinstance(source_endpoint_entry, dict),
             "base endpoint profile binding differs")
    source_endpoint = load_endpoint_service_profile(source_endpoint_path)
    _require(
        source_endpoint.fingerprint_sha256
        == source_endpoint_entry.get("fingerprint_sha256"),
        "base endpoint profile fingerprint differs",
    )
    semantic_endpoint_raw = json.loads(
        semantic_endpoint_profile_path.read_text(encoding="utf-8"))
    reproduced_semantic_endpoint = semantic_profile_builder.build_profile(
        source_endpoint_path,
        expected_base_sha256=str(source_endpoint_entry["sha256"]),
        profile_id=str(semantic_endpoint_raw.get("profile_id", "")),
    )
    semantic_endpoint = load_endpoint_service_profile(
        semantic_endpoint_profile_path)
    expected_routing_policy = (
        semantic_profile_builder.routing_policy_for_profile_id(
            str(semantic_endpoint_raw.get("profile_id", ""))))
    _require(
        semantic_endpoint_raw == reproduced_semantic_endpoint
        and semantic_endpoint.schema == SCHEMA_V2
        and semantic_endpoint.routing_policy is not None
        and semantic_endpoint.routing_policy.as_dict()
        == expected_routing_policy,
        "semantic endpoint profile is not an exact frozen derivation",
    )
    implementation = []
    for relative in IMPLEMENTATION_FILES:
        path = (_repo_root() / relative).resolve()
        _require(path.is_file(), f"implementation file is missing: {relative}")
        implementation.append(_entry(path))
    fixed_file_index = {
        entry["path"]: entry for entry in fixed_value["files"]}
    fixed_baseline_overrides = [
        {
            "path": relative,
            "baseline_sha256": fixed_file_index[relative]["sha256"],
            "candidate_sha256": _sha256((_repo_root() / relative).resolve()),
            "scope": "admission_or_service_profile_only",
        }
        for relative in sorted(SEMANTIC_BASELINE_OVERRIDES)
    ]
    value: dict[str, object] = {
        "schema": SCHEMA,
        "purpose": (
            "exploratory live screen of profile-bound pair-local semantic "
            "evidence, all-tenant endpoint credits, and a hysteretic route "
            "epoch"),
        "calibration_only": True,
        "performance_claim_allowed": False,
        "physical_switch_bottleneck_claim_allowed": False,
        "controller_parameter_search_allowed": False,
        "frozen_validation_allowed": False,
        "transport": "LMCacheConnectorV1:UCX",
        "unchanged_pd_data_plane": True,
        "endpoint_routing_policy": "semantic_epoch_v1",
        "passive_external_credit": True,
        "controller_reset_before_each_measured_block": True,
        "base_c4_run_contract": _entry(
            base_path, schema=BASE_SCHEMA),
        "fixed_c4_implementation_contract": _entry(
            fixed_implementation_path,
            schema=fixed_implementation.SCHEMA,
            fingerprint_sha256=fixed_value["fingerprint_sha256"],
        ),
        "semantic_observer_result": _entry(
            observer_result_path, schema=OBSERVER_RESULT_SCHEMA),
        "semantic_observer_analysis": _entry(
            observer_analysis_path, schema=OBSERVER_ANALYSIS_SCHEMA),
        "source_endpoint_service_profile": _entry(
            source_endpoint_path,
            schema=source_endpoint.schema,
            fingerprint_sha256=source_endpoint.fingerprint_sha256,
        ),
        "endpoint_service_profile": _entry(
            semantic_endpoint_profile_path,
            schema=semantic_endpoint.schema,
            fingerprint_sha256=semantic_endpoint.fingerprint_sha256,
            derived_from_sha256=str(source_endpoint_entry["sha256"]),
        ),
        **copied_entries,
        "implementation": implementation,
        "fixed_baseline_overrides": fixed_baseline_overrides,
        "semantic_credit_contract": semantic_endpoint.routing_policy.as_dict(),
        "external_service_proxy_contract": {
            "scope": "route_pinned_external_tenants_only",
            "tempo_adaptive_requests_require_exact_profile_rows": True,
            "geometry_rule": (
                "minimum_measured_prompt_and_output_ceiling"),
            "same_residency_preferred": True,
            "confirmed_miss_prefill_only_fallback_explicitly_labeled": True,
            "outside_profile_envelope_fails_closed": True,
            "proxy_is_exact_service_evidence": False,
        },
        "runtime_environment": {
            "TEMPO_PD_C4_APPROVED": "YES",
            "TEMPO_PD_C4_PHASE_DURATION_MS": "15000",
            "TEMPO_PD_C4_COOLDOWN_S": "2",
            "TEMPO_PD_ENDPOINT_FEEDBACK_MODE": "adaptive",
            "TEMPO_PD_ENDPOINT_ROUTING_POLICY": "semantic_epoch_v1",
            "TEMPO_PD_ENDPOINT_PASSIVE_FEEDBACK": "1",
            "TEMPO_PD_PRESSURE_MODE": "disabled",
            "TEMPO_VLLM_LOAD_SNAPSHOT_MODE": "disabled",
            "TEMPO_PD_BENCHMARK_COLD_MEASURED": "1",
            "TEMPO_PD_BENCHMARK_RESET_DECODER_APC": "0",
            "TEMPO_VLLM_DECODER_PREFIX_CACHING": "0",
            "TEMPO_PD_FRONTEND_PAIR_POLICY": (
                "tempo-min-outstanding-decode-tokens-v1"),
            "TEMPO_PD_FRONTEND_REPLICATE_WARM_AFFINITY": "1",
            "TEMPO_PD_DECODER_REUSE_ITEMS": "all",
            "TEMPO_PD_FORWARD_TOKEN_IDS": "0",
            "TEMPO_PD_PROXY_KV_CONTROL_OVERLAP": "0",
            "TEMPO_PD_REMOTE_DECODE_PLACEMENT": "paired",
            "TEMPO_PD_PROXY_TOKENIZER_PLACEMENT": "round_robin",
            "TEMPO_ELASTIC_PD_PROFILE_SCOPE": "screen_only",
            "TEMPO_VLLM_MAX_NUM_SEQS": "16",
            "TEMPO_VLLM_ASYNC_SCHEDULING": "0",
            "TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS": "32768",
            "TEMPO_VLLM_SCHEDULING_POLICY": "fcfs",
            "TEMPO_PD_REMOTE_CATCHUP_PRIORITY": "0",
            "TEMPO_PD_STRONG_REMOTE_CATCHUP_PRIORITY": "0",
            "TEMPO_PD_LONG_REMOTE_CATCHUP_PRIORITY": "0",
            "TEMPO_PD_LONG_REMOTE_CATCHUP_MIN_PROMPT_TOKENS": "0",
            "TEMPO_PD_MEDIAN_GUARD_PRIORITY": "0",
            "TEMPO_PD_MEDIUM_REMOTE_CATCHUP_PRIORITY": "0",
            "TEMPO_PD_REMOTE_CATCHUP_MIN_OUTPUT_TOKENS": "256",
            "TEMPO_LMCACHE_NIXL_BACKEND": "UCX",
            "TEMPO_LMCACHE_LOCAL_CPU_GB": "16",
            "TEMPO_LMCACHE_PD_BUFFER_BYTES": "2147483648",
        },
        "slurm": {
            "nodes": 4,
            "gpus": 16,
            "interactive_time_limit": "04:00:00",
            "perlmutter_only": True,
            "login_node_experiment_execution_allowed": False,
        },
        "claim_boundary": (
            "request-level admission and routing under moving endpoint "
            "contention with the official vLLM/LMCache P/D data plane unchanged"),
    }
    verify_fixed_baseline_for_semantic(
        repo_root=_repo_root(),
        contract_path=fixed_implementation_path,
        expected_sha256=_sha256(fixed_implementation_path),
        phase_manifest=fixed_manifest,
        semantic_contract=value,
    )
    value["fingerprint_sha256"] = contract_fingerprint(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--observer-result", type=Path, required=True)
    parser.add_argument("--observer-analysis", type=Path, required=True)
    parser.add_argument(
        "--fixed-implementation", type=Path,
        default=_repo_root()
        / "eval/sota_4node/tempo_pd_c4_implementation_contract_v1.json",
    )
    parser.add_argument(
        "--semantic-endpoint-profile", type=Path,
        default=_repo_root()
        / "eval/sota_4node/real_tempo_pd_endpoint_service_profile_c4_semantic_epoch_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite run contract")
    value = build(
        base_path=args.base_contract,
        observer_result_path=args.observer_result,
        observer_analysis_path=args.observer_analysis,
        fixed_implementation_path=args.fixed_implementation,
        semantic_endpoint_profile_path=args.semantic_endpoint_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "sha256": _sha256(args.output.resolve()),
        "fingerprint_sha256": value["fingerprint_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
