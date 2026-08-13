#!/usr/bin/env python3
"""Build a non-submitting single-GPU inference KV attribution matrix.

This module freezes the adapter contract and endpoint/path split for a future
approved run.  It deliberately does not import vLLM/SGLang/LMCache and never
submits work; a live backend must bind its native KV identity and counters to
the resulting manifest before any performance number is publishable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

try:
    from tempo.kv_flow import KVOperation
    from tempo.resource_domain import ResourceDomain
    from tempo.observation_window import observation_window_contract
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tempo.kv_flow import KVOperation
    from tempo.resource_domain import ResourceDomain
    from tempo.observation_window import observation_window_contract


@dataclass(frozen=True)
class KVRun:
    mode: str
    policy: str
    operation: str
    endpoint: str
    route: tuple[str, ...]
    requires_output_equivalence: bool
    requires_version_check: bool
    requires_prefetch_before_use: bool


KV_MODES = ("fg_only", "open_combined", "d2h_only", "remote_fabric", "persistent_tier", "combined")
_MANIFEST_KEYS = {
    "schema_version",
    "world_size",
    "nodes",
    "kv_bytes_per_request",
    "deadline_ns",
    "offered_load_requests",
    "operation",
    "session_ids",
    "runs",
    "domain_footprints",
    "evidence_state",
    "live_backend",
    "slurm_submitted",
    "correctness_contract",
    "metric_contract",
    "observation_window_contract",
}

_METRIC_CONTRACT = {
    "required_fields": [
        "ttft_p99_ns",
        "itl_p99_ns",
        "slo_goodput_milli",
        "deadline_met",
        "correctness_met",
        "samples",
        "max_domain_exposure_ns",
        "domain_exposure_ns",
    ],
    "exposure_definition": "max_route_overlap_ns",
    "promotion_rule": "matched_route_domain_set_and_exposure_le_open_combined",
}


def _domain_footprints(runs: tuple[KVRun, ...] | list[KVRun]) -> dict[str, dict[str, list[str]]]:
    """Return explicit foreground/auxiliary overlap for every KV mode.

    The one-GPU G1 inference screen declares GPU-local execution as the
    foreground footprint.  A live backend may only replace this with measured
    domains in its result; it may not infer overlap from the endpoint name.
    """
    foreground = (ResourceDomain.GPU_LOCAL.value,)
    footprints: dict[str, dict[str, list[str]]] = {}
    for run in runs:
        auxiliary = list(run.route)
        shared = sorted(set(foreground).intersection(auxiliary))
        footprints[run.mode] = {
            "foreground_domains": list(foreground),
            "auxiliary_domains": auxiliary,
            "shared_domains": shared,
        }
    return footprints


def _route(*domains: ResourceDomain) -> tuple[str, ...]:
    return tuple(domain.value for domain in domains)


def build_kv_matrix() -> tuple[KVRun, ...]:
    # Every auxiliary mode uses the same endpoint/path.  The mode names denote
    # which domain admission intervention is isolated, not a different
    # endpoint.  Otherwise a remote/persistent route could win simply because
    # it changed the data plane, which is not a scheduler result.
    matched_route = _route(
        ResourceDomain.PERSISTENT_ENDPOINT,
        ResourceDomain.SLINGSHOT_FABRIC,
        ResourceDomain.NIC_FABRIC,
        ResourceDomain.HOST_NUMA,
        ResourceDomain.PCIE_HOST,
        ResourceDomain.GPU_LOCAL,
    )
    common = {
        "operation": KVOperation.PREFETCH.value,
        "requires_output_equivalence": True,
        "requires_version_check": True,
        "requires_prefetch_before_use": True,
    }
    return (
        KVRun("fg_only", "none", KVOperation.PREFETCH.value, "none", (), True, True, False),
        KVRun("open_combined", "kv_open", **common, endpoint="persistent_endpoint", route=matched_route),
        KVRun("d2h_only", "kv_d2h_only", **common, endpoint="persistent_endpoint", route=matched_route),
        KVRun("remote_fabric", "kv_remote", **common, endpoint="persistent_endpoint", route=matched_route),
        KVRun("persistent_tier", "kv_persistent", **common, endpoint="persistent_endpoint", route=matched_route),
        KVRun("combined", "kv_combined", **common, endpoint="persistent_endpoint", route=matched_route),
    )


def validate_kv_matrix(runs: tuple[KVRun, ...] | list[KVRun]) -> None:
    if tuple(run.mode for run in runs) != KV_MODES:
        raise ValueError("KV matrix must contain each mode exactly once in order")
    matched_open_route = tuple(runs[1].route)
    for run in runs:
        if run.operation != KVOperation.PREFETCH.value:
            raise ValueError("KV matrix operation must be prefetch")
        if run.mode == "fg_only":
            if run.policy != "none" or run.route or run.endpoint != "none":
                raise ValueError("fg_only must have no auxiliary route")
            continue
        if not run.route or run.route[0] != ResourceDomain.PERSISTENT_ENDPOINT.value:
            raise ValueError(f"{run.mode} prefetch must originate at persistent_endpoint")
        if run.route[-1] != ResourceDomain.GPU_LOCAL.value:
            raise ValueError(f"{run.mode} prefetch must terminate at gpu_local")
        if tuple(run.route) != matched_open_route:
            raise ValueError(f"{run.mode} must use the matched-open route")


def validate_kv_manifest(manifest: dict[str, object]) -> None:
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("KV manifest keys are not exact")
    if manifest["schema_version"] != "tempo-rd-inference-kv-runner-1":
        raise ValueError("unsupported KV manifest schema")
    if manifest["world_size"] != 1 or manifest["nodes"] != 1:
        raise ValueError("initial KV attribution must be one GPU on one node")
    for key in ("kv_bytes_per_request", "deadline_ns", "offered_load_requests"):
        value = manifest[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive int")
    if manifest["operation"] != KVOperation.PREFETCH.value:
        raise ValueError("KV operation is not the frozen prefetch contract")
    sessions = manifest["session_ids"]
    if type(sessions) is not list or not sessions or any(type(item) is not str or not item for item in sessions):
        raise ValueError("session_ids must be a non-empty string list")
    if manifest["evidence_state"] != "design_only" or manifest["live_backend"] is not False:
        raise ValueError("KV manifest cannot claim live backend evidence")
    if manifest["slurm_submitted"] is not False:
        raise ValueError("KV runner must never submit Slurm work")
    if manifest["observation_window_contract"] != observation_window_contract():
        raise ValueError("KV observation-window contract is not exact")
    contract = manifest["correctness_contract"]
    expected_contract = {
        "native_version_identity": True,
        "output_token_equivalence": True,
        "stale_version_rejection": True,
        "prefetch_before_use": True,
        "exact_completion_bytes": True,
        "slo_goodput_fixed_offered_load": True,
    }
    if (
        type(contract) is not dict
        or set(contract) != set(expected_contract)
        or any(type(contract[key]) is not bool for key in expected_contract)
        or contract != expected_contract
    ):
        raise ValueError("KV correctness contract is not exact")
    if manifest["metric_contract"] != _METRIC_CONTRACT:
        raise ValueError("KV metric contract is not exact")
    raw_runs = manifest["runs"]
    if type(raw_runs) is not list:
        raise ValueError("KV runs must be a list")
    # JSON turns the dataclass route tuples into lists; compare the exact
    # wire-shaped representation rather than Python tuple/list identity.
    expected = json.loads(json.dumps([asdict(run) for run in build_kv_matrix()]))
    if raw_runs != expected:
        raise ValueError("KV runs do not match the frozen adapter matrix")
    expected_footprints = _domain_footprints(build_kv_matrix())
    if manifest["domain_footprints"] != expected_footprints:
        raise ValueError("KV domain footprints do not match the frozen route contract")


def build_manifest(
    *,
    kv_bytes_per_request: int = 64 * 1024 * 1024,
    deadline_ns: int = 250_000_000,
    offered_load_requests: int = 64,
    session_ids: list[str] | None = None,
) -> dict[str, object]:
    if type(kv_bytes_per_request) is not int or kv_bytes_per_request <= 0:
        raise ValueError("kv_bytes_per_request must be a positive int")
    if type(deadline_ns) is not int or deadline_ns <= 0:
        raise ValueError("deadline_ns must be a positive int")
    if type(offered_load_requests) is not int or offered_load_requests <= 0:
        raise ValueError("offered_load_requests must be a positive int")
    sessions = session_ids if session_ids is not None else ["session-0", "session-1"]
    runs = build_kv_matrix()
    validate_kv_matrix(runs)
    manifest = {
        "schema_version": "tempo-rd-inference-kv-runner-1",
        "world_size": 1,
        "nodes": 1,
        "kv_bytes_per_request": kv_bytes_per_request,
        "deadline_ns": deadline_ns,
        "offered_load_requests": offered_load_requests,
        "operation": KVOperation.PREFETCH.value,
        "session_ids": list(sessions),
        "runs": json.loads(json.dumps([asdict(run) for run in runs])),
        "domain_footprints": _domain_footprints(runs),
        "evidence_state": "design_only",
        "live_backend": False,
        "slurm_submitted": False,
        "correctness_contract": {
            "native_version_identity": True,
            "output_token_equivalence": True,
            "stale_version_rejection": True,
            "prefetch_before_use": True,
            "exact_completion_bytes": True,
            "slo_goodput_fixed_offered_load": True,
        },
        "metric_contract": dict(_METRIC_CONTRACT),
        "observation_window_contract": observation_window_contract(),
    }
    validate_kv_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--kv-bytes-per-request", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--deadline-ns", type=int, default=250_000_000)
    parser.add_argument("--offered-load-requests", type=int, default=64)
    args = parser.parse_args()
    encoded = json.dumps(
        build_manifest(
            kv_bytes_per_request=args.kv_bytes_per_request,
            deadline_ns=args.deadline_ns,
            offered_load_requests=args.offered_load_requests,
        ),
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
