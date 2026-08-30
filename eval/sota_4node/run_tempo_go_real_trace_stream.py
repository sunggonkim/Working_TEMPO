#!/usr/bin/env python3
"""Run a source-bound Mooncake token-ID population through TEMPO-PD.

This adapter reuses the canonical strict SSE, router-decision, terminal-state,
and forced-token client.  It changes only two seams that the frozen C9 client
does not support: prompts may be token-ID lists, and semantic trace IDs are
mapped to arm/tenant-bearing wire IDs.  The mapping and business assignment
are frozen before dispatch and appended to the raw artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence
from urllib import error, request

from eval.sota_4node import run_tempo_pd_elastic_stream_metrics as canonical
from eval.sota_4node import run_tempo_pd_stream_metrics_forced_v32 as forced
from eval.sota_4node import run_tempo_pd_stream_metrics_v1 as v1
from eval.sota_4node import run_vllm_stream_metrics as base


SCHEMA = "tempo-go-real-trace-stream-adapter-v1"
BUSINESS_PROFILE_SCHEMA = "tempo-go-real-trace-business-profile-v1"
POPULATION_MANIFEST_ENV = "TEMPO_GO_REAL_TRACE_POPULATION_MANIFEST"
WIRE_ARM_ENV = "TEMPO_GO_REAL_TRACE_WIRE_ARM"
BUSINESS_PROFILE_ENV = "TEMPO_GO_REAL_TRACE_BUSINESS_PROFILE"
WIRE_NAMESPACE_ENV = "TEMPO_GO_REAL_TRACE_WIRE_NAMESPACE"
WIRE_ARMS = frozenset({
    "local", "remote", "predictor", "queue_gpu", "network_request_only",
    "app_global_only", "tempo",
})
TENANTS = frozenset({"latency", "interactive", "batch", "background"})
_WORKLOAD_KEYS = frozenset({
    "request_id", "prompt", "max_tokens", "arrival_offset_ms",
})
_REQUEST_METADATA: dict[str, dict[str, Any]] = {}
_MODEL_TOKEN_RECEIPT: dict[str, Any] | None = None


REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_PATH = REPO_ROOT / "tempo/mooncake_fast25_workload.py"
_CORE_SPEC = importlib.util.spec_from_file_location(
    "_tempo_mooncake_fast25_real_stream", _CORE_PATH,
)
if _CORE_SPEC is None or _CORE_SPEC.loader is None:
    raise RuntimeError(f"cannot load Mooncake workload module: {_CORE_PATH}")
_CORE = importlib.util.module_from_spec(_CORE_SPEC)
sys.modules[_CORE_SPEC.name] = _CORE
_CORE_SPEC.loader.exec_module(_CORE)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise base.ContractError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_business_profile(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"business profile is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise base.ContractError(f"cannot read business profile: {path}") from exc
    _require(isinstance(value, dict), "business profile must be an object")
    _require(
        value.get("schema") == BUSINESS_PROFILE_SCHEMA,
        "business profile schema differs",
    )
    assignment = value.get("assignment")
    _require(isinstance(assignment, dict), "business assignment is missing")
    _require(
        assignment.get("mode") == "weighted_cycle_v1",
        "business assignment mode differs",
    )
    cycle = assignment.get("tenant_cycle")
    _require(
        isinstance(cycle, list)
        and bool(cycle)
        and all(tenant in TENANTS for tenant in cycle),
        "tenant_cycle contains an unsupported business class",
    )
    deadlines = assignment.get("remaining_deadline_ms")
    _require(isinstance(deadlines, dict), "remaining_deadline_ms is missing")
    _require(set(deadlines) == TENANTS, "deadline tenant set differs")
    for tenant, deadline in deadlines.items():
        _require(
            deadline is None
            or (
                type(deadline) in (int, float)
                and not isinstance(deadline, bool)
                and math.isfinite(float(deadline))
                and float(deadline) > 0.0
            ),
            f"deadline is invalid for tenant {tenant}",
        )
    _require(
        value.get("policy_inputs_excluded") == [
            "future_arrivals", "oracle_route", "future_cache_evictions",
            "physical_switch_label",
        ],
        "business profile policy-input exclusion differs",
    )
    return value


def _wire_request_id(
    *, semantic_request_id: str, wire_arm: str, tenant: str,
    namespace: str = "",
) -> str:
    _require(wire_arm in WIRE_ARMS, "wire arm is unsupported")
    _require(tenant in TENANTS, "wire tenant is unsupported")
    _require(
        re.fullmatch(r"mooncake-[a-z0-9]+-[0-9]{6,}", semantic_request_id)
        is not None
        or re.fullmatch(r"warm-[a-z0-9-]+", semantic_request_id) is not None,
        "semantic request ID is malformed",
    )
    suffix = semantic_request_id.replace("_", "-")
    if namespace:
        _require(
            re.fullmatch(r"[a-z0-9-]+", namespace) is not None,
            "wire namespace is malformed",
        )
        suffix = f"{namespace}-{suffix}"
    return (
        f"epd-{wire_arm}-{tenant}-cache-natural-measured-real-{suffix}"
    )


def _configuration() -> tuple[Path, str, Path, dict[str, Any]]:
    raw_manifest = os.environ.get(POPULATION_MANIFEST_ENV)
    wire_arm = os.environ.get(WIRE_ARM_ENV)
    raw_business = os.environ.get(BUSINESS_PROFILE_ENV)
    _require(bool(raw_manifest), f"{POPULATION_MANIFEST_ENV} is unset")
    _require(wire_arm in WIRE_ARMS, f"{WIRE_ARM_ENV} is invalid")
    _require(bool(raw_business), f"{BUSINESS_PROFILE_ENV} is unset")
    manifest_path = Path(str(raw_manifest)).resolve()
    business_path = Path(str(raw_business)).resolve()
    return manifest_path, str(wire_arm), business_path, _load_business_profile(
        business_path,
    )


def validate_model_token_contract(
    model_path: Path,
    population_manifest_path: Path,
) -> dict[str, Any]:
    """Prove the materialized IDs and context fit the exact local model."""

    model_path = model_path.resolve()
    config_path = model_path / "config.json"
    tokenizer_path = model_path / "tokenizer_config.json"
    _require(config_path.is_file(), "model config.json is missing")
    _require(tokenizer_path.is_file(), "model tokenizer_config.json is missing")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        population = json.loads(
            population_manifest_path.read_text(encoding="utf-8"),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise base.ContractError("model/token population metadata is unreadable") from exc
    _require(isinstance(config, dict), "model config must be an object")
    _require(isinstance(tokenizer, dict), "tokenizer config must be an object")
    vocab_size = config.get("vocab_size")
    model_context = config.get("max_position_embeddings")
    _require(type(vocab_size) is int and vocab_size > 0, "model vocab_size is invalid")
    _require(
        type(model_context) is int and model_context > 0,
        "model max_position_embeddings is invalid",
    )
    token_contract = population.get("token_materialization")
    context = population.get("context")
    _require(isinstance(token_contract, dict), "population token contract is missing")
    _require(isinstance(context, dict), "population context contract is missing")
    token_min = token_contract.get("token_id_min_inclusive")
    token_max = token_contract.get("token_id_max_exclusive")
    requested_context = context.get("max_model_len")
    _require(
        type(token_min) is int and 0 <= token_min < vocab_size,
        "population token_id_min is outside the model vocabulary",
    )
    _require(
        type(token_max) is int and token_min < token_max <= vocab_size,
        "population token_id_max is outside the model vocabulary",
    )
    _require(
        type(requested_context) is int
        and 0 < requested_context <= model_context,
        "population max_model_len exceeds model position capacity",
    )
    added = tokenizer.get("added_tokens_decoder", {})
    _require(isinstance(added, dict), "added_tokens_decoder must be an object")
    special_ids = sorted(
        int(token_id)
        for token_id, description in added.items()
        if isinstance(description, dict) and description.get("special") is True
    )
    _require(
        all(not (token_min <= token_id < token_max) for token_id in special_ids),
        "population token interval intersects a tokenizer special token",
    )
    return {
        "schema": "tempo-go-real-trace-model-token-contract-v1",
        "model_path": str(model_path),
        "model_config_sha256": _sha256(config_path),
        "tokenizer_config_sha256": _sha256(tokenizer_path),
        "vocab_size": vocab_size,
        "max_position_embeddings": model_context,
        "population_max_model_len": requested_context,
        "token_id_min_inclusive": token_min,
        "token_id_max_exclusive": token_max,
        "special_token_ids": special_ids,
        "token_interval_in_vocabulary": True,
        "token_interval_excludes_special_tokens": True,
    }


def load_token_workload(
    path: Path,
    *,
    default_max_tokens: int,
    request_rate: float | None,
) -> tuple[list[base.WorkItem], str]:
    """Validate one token population and bind arm/tenant wire identities."""

    manifest_path, wire_arm, _business_path, business = _configuration()
    _require(request_rate is None, "real trace requires explicit source arrivals")
    _require(
        type(default_max_tokens) is int and default_max_tokens >= 2,
        "default_max_tokens must be at least two",
    )
    _CORE.load_and_verify_population(path, manifest_path)
    raw = path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    token_contract = manifest["token_materialization"]
    token_min = int(token_contract["token_id_min_inclusive"])
    token_max = int(token_contract["token_id_max_exclusive"])
    cycle = business["assignment"]["tenant_cycle"]
    deadlines = business["assignment"]["remaining_deadline_ms"]
    namespace = os.environ.get(WIRE_NAMESPACE_ENV, "")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        value = json.loads(line)
        _require(
            isinstance(value, dict) and set(value) == _WORKLOAD_KEYS,
            f"token workload line {line_number} fields differ",
        )
        rows.append(value)
    items: list[base.WorkItem] = []
    _REQUEST_METADATA.clear()
    for index, value in enumerate(rows):
        semantic_id = value["request_id"]
        prompt = value["prompt"]
        output_tokens = value["max_tokens"]
        arrival_ms = value["arrival_offset_ms"]
        _require(
            isinstance(semantic_id, str) and bool(semantic_id),
            f"semantic request ID is invalid at row {index}",
        )
        _require(
            isinstance(prompt, list)
            and bool(prompt)
            and all(type(token) is int and token_min <= token < token_max for token in prompt),
            f"prompt token IDs are invalid at row {index}",
        )
        _require(
            type(output_tokens) is int and output_tokens >= 2,
            f"output tokens are invalid at row {index}",
        )
        _require(
            type(arrival_ms) in (int, float)
            and not isinstance(arrival_ms, bool)
            and math.isfinite(float(arrival_ms))
            and float(arrival_ms) >= 0.0,
            f"arrival is invalid at row {index}",
        )
        tenant = str(cycle[index % len(cycle)])
        wire_id = _wire_request_id(
            semantic_request_id=semantic_id,
            wire_arm=wire_arm,
            tenant=tenant,
            namespace=namespace,
        )
        deadline = deadlines[tenant]
        _REQUEST_METADATA[wire_id] = {
            "semantic_request_id": semantic_id,
            "wire_request_id": wire_id,
            "wire_arm": wire_arm,
            "business_tenant_id": tenant,
            "remaining_deadline_ms": deadline,
            "prompt_tokens": len(prompt),
            "output_tokens": output_tokens,
            "arrival_offset_ms": float(arrival_ms),
        }
        # WorkItem's historical type annotation is text-only, but its runtime
        # container is intentionally reused here with a validated token list.
        items.append(base.WorkItem(
            index=index,
            request_id=wire_id,
            prompt=list(prompt),  # type: ignore[arg-type]
            max_tokens=output_tokens,
            arrival_offset_ns=round(float(arrival_ms) * 1_000_000.0),
        ))
    _require(
        len(items) == len(_REQUEST_METADATA),
        "wire request identities are not unique",
    )
    return items, hashlib.sha256(raw).hexdigest()


def execute_token_request(
    item: base.WorkItem,
    *,
    endpoint: str,
    served_model_name: str,
    run_start_ns: int,
    timeout_s: float,
    seed: int,
    api_key: str | None,
    opener: Callable[..., Any] = request.urlopen,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Canonical request execution with token-list prompt provenance."""

    prompt = item.prompt
    _require(
        isinstance(prompt, list)
        and bool(prompt)
        and all(type(token) is int and token >= 0 for token in prompt),
        "real-trace WorkItem prompt is not a token-ID list",
    )
    metadata = _REQUEST_METADATA.get(item.request_id)
    _require(isinstance(metadata, dict), "real-trace request metadata is missing")
    target_ns = run_start_ns + item.arrival_offset_ns
    delay_ns = target_ns - clock_ns()
    if delay_ns > 0:
        sleeper(delay_ns / 1_000_000_000.0)
    dispatch_ns = clock_ns()
    body = {
        "model": served_model_name,
        "prompt": prompt,
        "max_tokens": item.max_tokens,
        "temperature": 0.0,
        "seed": seed + item.index,
        "n": 1,
        "echo": False,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": 1,
    }
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "X-Tempo-Request-Id": item.request_id,
        "X-Tempo-Tenant-Id": metadata["business_tenant_id"],
    }
    deadline = metadata["remaining_deadline_ms"]
    if deadline is not None:
        headers["X-Tempo-Remaining-Deadline-Ms"] = str(float(deadline))
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = request.Request(
        endpoint,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    prompt_json = json.dumps(prompt, separators=(",", ":")).encode("utf-8")
    common = {
        "request_index": item.index,
        "request_id": item.request_id,
        "semantic_request_id": metadata["semantic_request_id"],
        "ingress_lane": v1._ingress_lane(item),
        "prompt_sha256": base._sha256_bytes(prompt_json),
        "prompt_token_count": len(prompt),
        "prompt_utf8_bytes": None,
        "requested_max_tokens": item.max_tokens,
        "scheduled_dispatch_offset_ns": item.arrival_offset_ns,
        "tempo_business_tenant_id": metadata["business_tenant_id"],
        "remaining_deadline_ms": deadline,
    }
    try:
        with opener(http_request, timeout=timeout_s) as response:
            status = int(response.getcode())
            _require(status == 200, f"HTTP status {status}")
            router = v1._router_headers(response, item.request_id)
            streamed = v1._stream_record(
                response,
                dispatch_ns=dispatch_ns,
                run_start_ns=run_start_ns,
                expected_tokens=item.max_tokens,
                route=router["route"],
                clock_ns=clock_ns,
            )
            record = {**common, "router": router, **streamed}
    except (
        base.ContractError,
        error.HTTPError,
        error.URLError,
        TimeoutError,
        OSError,
    ) as exc:
        end_ns = clock_ns()
        http_detail = ""
        if isinstance(exc, error.HTTPError):
            try:
                http_detail = exc.read(16 * 1024).decode(
                    "utf-8", errors="replace")
            except (OSError, ValueError):
                http_detail = ""
        terminal_error_kind = (
            v1._classify_http_error(exc)
            if isinstance(exc, error.HTTPError)
            else "request_or_stream_error"
        )
        global_reject_candidate = (
            terminal_error_kind in v1._GLOBAL_REJECTION_KINDS
        )
        service_lane_failure_candidate = (
            terminal_error_kind in v1._SERVICE_LANE_FAILURE_KINDS
        )
        record = {
            **common,
            "router": None,
            "http_status": int(exc.code) if isinstance(exc, error.HTTPError) else None,
            "dispatch_offset_ns": dispatch_ns - run_start_ns,
            "token_arrival_offsets_ns": [],
            "stream_end_offset_ns": end_ns - run_start_ns,
            "output_token_values": [],
            "output_token_proofs": [],
            "output_text": "",
            "output_text_sha256": base._sha256_bytes(b""),
            "finish_reason": None,
            "usage": None,
            "done_seen": False,
            "response_ids": [],
            "response_models": [],
            "contract_violations": (
                []
                if global_reject_candidate or service_lane_failure_candidate
                else [terminal_error_kind]
            ),
            "error": (
                None
                if global_reject_candidate or service_lane_failure_candidate
                else f"{type(exc).__name__}: {exc}"
            ),
            "terminal_reject_candidate": global_reject_candidate,
            "terminal_service_lane_failure_candidate": service_lane_failure_candidate,
            "terminal_error_kind": terminal_error_kind,
            "transport_error": f"{type(exc).__name__}: {exc}",
        }
        if http_detail:
            record["http_error_detail"] = http_detail[:2048]
    record["valid"] = not record["contract_violations"] and record["error"] is None
    return record


def _custom_parser(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--wire-arm", choices=sorted(WIRE_ARMS), required=True)
    parser.add_argument("--business-profile", type=Path, required=True)
    parser.add_argument("--wire-namespace", default="")
    return parser.parse_known_args(argv)


def _argument_path(arguments: Sequence[str], name: str) -> Path:
    _require(name in arguments, f"delegated argument is missing: {name}")
    index = arguments.index(name)
    _require(index + 1 < len(arguments), f"delegated argument has no value: {name}")
    return Path(arguments[index + 1])


def _argument_value(arguments: Sequence[str], name: str) -> str:
    _require(name in arguments, f"delegated argument is missing: {name}")
    index = arguments.index(name)
    _require(index + 1 < len(arguments), f"delegated argument has no value: {name}")
    return arguments[index + 1]


def _augment_output(
    output_path: Path,
    *,
    population_manifest_path: Path,
    wire_arm: str,
    business_profile_path: Path,
) -> None:
    _require(output_path.is_file(), "canonical real-trace raw output is missing")
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    population = json.loads(population_manifest_path.read_text(encoding="utf-8"))
    business = _load_business_profile(business_profile_path)
    requests = raw.get("requests")
    _require(isinstance(requests, list), "canonical request rows are missing")
    _require(
        {row.get("request_id") for row in requests} == set(_REQUEST_METADATA),
        "canonical request IDs differ from real-trace wire mapping",
    )
    mapping_rows = [
        _REQUEST_METADATA[request_id]
        for request_id in sorted(_REQUEST_METADATA)
    ]
    raw["real_trace_contract"] = {
        "schema": SCHEMA,
        "population_manifest": str(population_manifest_path),
        "population_manifest_sha256": _sha256(population_manifest_path),
        "population_semantic_sha256": population["population_semantic_sha256"],
        "source_trace_name": population["source"]["trace_name"],
        "source_trace_sha256": population["source"]["source_sha256"],
        "source_upstream_commit": population["source"]["upstream_commit"],
        "wire_arm": wire_arm,
        "business_profile": str(business_profile_path),
        "business_profile_sha256": _sha256(business_profile_path),
        "business_assignment_mode": business["assignment"]["mode"],
        "wire_mapping_sha256": hashlib.sha256(json.dumps(
            mapping_rows, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "request_count": len(mapping_rows),
        "token_ids_forwarded_by_client": True,
        "raw_prompt_content_available": False,
        "retry_count": 0,
        "model_token_contract": _MODEL_TOKEN_RECEIPT,
        "performance_claim_allowed": False,
    }
    raw["workload"]["schema"] = population["workload_schema"]
    output_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    global _MODEL_TOKEN_RECEIPT
    supplied = list(sys.argv[1:] if argv is None else argv)
    custom, delegated = _custom_parser(supplied)
    population_manifest = custom.population_manifest.resolve()
    business_profile = custom.business_profile.resolve()
    _CORE.load_and_verify_population(
        _argument_path(delegated, "--workload"), population_manifest,
    )
    _load_business_profile(business_profile)
    output_path = _argument_path(delegated, "--output")
    model_path = _argument_path(delegated, "--model")
    population = json.loads(population_manifest.read_text(encoding="utf-8"))
    request_count = int(population["selection"]["request_count"])
    max_workers = int(_argument_value(delegated, "--max-workers"))
    _require(
        max_workers >= request_count,
        "real-trace synchronous ingress requires max_workers >= request_count",
    )
    ingress_policy = (
        _argument_value(delegated, "--ingress-policy")
        if "--ingress-policy" in delegated
        else "shared_pool"
    )
    _require(
        ingress_policy == "shared_pool",
        "real-trace client ingress must be shared_pool; server policy owns fairness",
    )
    _MODEL_TOKEN_RECEIPT = validate_model_token_contract(
        model_path, population_manifest,
    )
    old_load = base.load_workload
    old_original_execute = forced._ORIGINAL_EXECUTE
    old_argv = sys.argv
    old_environment = {
        name: os.environ.get(name)
        for name in (
            POPULATION_MANIFEST_ENV, WIRE_ARM_ENV, BUSINESS_PROFILE_ENV,
            WIRE_NAMESPACE_ENV,
        )
    }
    os.environ[POPULATION_MANIFEST_ENV] = str(population_manifest)
    os.environ[WIRE_ARM_ENV] = custom.wire_arm
    os.environ[BUSINESS_PROFILE_ENV] = str(business_profile)
    os.environ[WIRE_NAMESPACE_ENV] = custom.wire_namespace
    base.load_workload = load_token_workload
    forced._ORIGINAL_EXECUTE = execute_token_request
    sys.argv = [old_argv[0], *delegated]
    try:
        status = canonical.main()
        if output_path.is_file():
            _augment_output(
                output_path,
                population_manifest_path=population_manifest,
                wire_arm=custom.wire_arm,
                business_profile_path=business_profile,
            )
        return status
    finally:
        sys.argv = old_argv
        base.load_workload = old_load
        forced._ORIGINAL_EXECUTE = old_original_execute
        _MODEL_TOKEN_RECEIPT = None
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
