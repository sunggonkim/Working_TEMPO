#!/usr/bin/env python3
"""Build a fail-closed causal-readiness record for a G1 raw artifact.

The one-node tier runner deliberately separates functional evidence from
causal resource evidence.  This tool makes that boundary machine-readable:
it validates the raw five-mode tree, checks that logical stage timing exists,
and looks for *actual* domain-counter records.  It never infers PCIe,
NVLink, CXI, Slingshot, or OST activity from a logical byte count or from a
topology label.  Missing evidence therefore produces ``not_ready`` rather
than a synthetic promotion sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

try:
    from eval.sota_4node.validate_g1_tier_raw import MODES, validate_g1_tier_raw
    from tempo.domain_counters import CounterSnapshot, validate_counter_series
    from tempo.domain_evidence import CounterSupport, PathStatus
    from tempo.foreground_path import validate_foreground_path
    from tempo.resource_domain import ResourceDomain, allowed_counter_scopes, domain_contract
    from tempo.tier_attribution import REQUIRED_G1_MODES, mode_spec, required_domains_for_modes
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, os.environ.get("TEMPO_RD_REPO_ROOT", str(Path(__file__).resolve().parents[2])))
    from eval.sota_4node.validate_g1_tier_raw import MODES, validate_g1_tier_raw
    from tempo.domain_counters import CounterSnapshot, validate_counter_series
    from tempo.domain_evidence import CounterSupport, PathStatus
    from tempo.foreground_path import validate_foreground_path
    from tempo.resource_domain import ResourceDomain, allowed_counter_scopes, domain_contract
    from tempo.tier_attribution import REQUIRED_G1_MODES, mode_spec, required_domains_for_modes


SCHEMA = "tempo-rd-g1-causal-readiness-1"
COUNTER_SCHEMA = "tempo-rd-domain-counter-record-2"
DOMAIN_MODES = tuple(MODES) + ("host_pressure",)
# Slingshot transport is deliberately deferred to the two-node fabric slice.
# A one-node run can observe NIC/CXI injection, but it cannot separate
# intra-node traffic from the inter-node transport domain without a peer node.
G1_DEFERRED_DOMAINS = frozenset({ResourceDomain.SLINGSHOT_FABRIC})
FOREGROUND_PATH_FILENAME = "foreground_path.json"

G1_COUNTER_COLLECTION_PLAN = {
    ResourceDomain.GPU_LOCAL.value: {
        "stage": "g1",
        "strategy": "separate_nsys_cuda_gpu_mem_size_sum_diagnostic",
        "timed_metrics_eligible": False,
        "requires": "unambiguous_rank_bound_DtoH_rows",
    },
    ResourceDomain.PCIE_HOST.value: {
        "stage": "g1",
        "strategy": "site_supported_gpu_endpoint_or_gdr_byte_counter",
        "timed_metrics_eligible": True,
        "requires": "endpoint_identity_and_monotonic_bytes",
    },
    ResourceDomain.HOST_NUMA.value: {
        "stage": "g1",
        "strategy": "pinned_buffer_residency_plus_supported_numa_bytes_counter",
        "timed_metrics_eligible": True,
        "requires": "buffer_identity_and_monotonic_bytes",
    },
    ResourceDomain.NIC_FABRIC.value: {
        "stage": "g1",
        "strategy": "hsn_sysfs_or_site_supported_rank_bound_counter",
        "timed_metrics_eligible": True,
        "requires": "rank_or_mode_bound_injection_path",
    },
    ResourceDomain.PERSISTENT_ENDPOINT.value: {
        "stage": "g1",
        "strategy": "site_supported_lustre_ost_or_client_byte_counter",
        "timed_metrics_eligible": True,
        "requires": "OST_or_client_scope_and_completion_interval",
    },
    ResourceDomain.SLINGSHOT_FABRIC.value: {
        "stage": "g2",
        "strategy": "two_node_transport_or_traffic_class_counter",
        "timed_metrics_eligible": True,
        "requires": "intra_inter_node_split_and_route_witness",
    },
}

# A host-wide counter is useful diagnostics but cannot attribute a foreground
# tail to one rank/slice.  The scope contract is therefore checked separately
# from the counter family/path labels.

def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load %s: %s" % (path, exc)) from exc


def _bound_raw_validator(root: Path):
    """Use the allocation snapshot only for a new extended manifest.

    Historical raw G1 artifacts predate the readiness-builder source key and
    were validated under a slightly older compatibility contract.  Preserve
    their legacy interpretation, while a new runner binds readiness to the
    copied validator that it records in its manifest.
    """

    manifest_path = root / "g1_tier_runtime_manifest.json"
    try:
        manifest = _load(manifest_path)
    except ValueError:
        return validate_g1_tier_raw
    sources = manifest.get("source_sha256") if isinstance(manifest, dict) else None
    snapshot = root / "validate_g1_tier_raw_executed.py"
    if not isinstance(sources, dict) or "build_g1_causal_readiness_executed.py" not in sources:
        return validate_g1_tier_raw
    if not snapshot.is_file():
        return validate_g1_tier_raw
    spec = importlib.util.spec_from_file_location("tempo_rd_g1_raw_validator_snapshot", snapshot)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load G1 raw-validator snapshot")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_g1_tier_raw


def _logical_record_present(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    start = record.get("tier_stage_stats_start")
    end = record.get("tier_stage_stats_end")
    return (
        isinstance(start, dict)
        and isinstance(end, dict)
        and isinstance(start.get("logical_stage"), dict)
        and isinstance(end.get("logical_stage"), dict)
    )


def _logical_stage_inventory(root: Path, steps: Iterable[int]) -> Dict[str, int]:
    """Count logical stage pairs, without treating them as hardware counters."""

    expected = 0
    present = 0
    missing: List[str] = []
    for mode in MODES:
        if mode == "fg_only":
            continue
        mode_root = root / mode
        for rank in range(4):
            final_path = mode_root / ("checkpoint_rank%d.json" % rank)
            expected += 1
            if final_path.is_file() and _logical_record_present(_load(final_path)):
                present += 1
            else:
                missing.append("%s/checkpoint_rank%d.json" % (mode, rank))
            events_path = mode_root / ("checkpoint_events_rank%d.json" % rank)
            events = _load(events_path) if events_path.is_file() else None
            if not isinstance(events, list):
                missing.append("%s/checkpoint_events_rank%d.json" % (mode, rank))
                expected += len(tuple(steps))
                continue
            expected += len(tuple(steps))
            for index, event in enumerate(events):
                if _logical_record_present(event):
                    present += 1
                else:
                    missing.append("%s/checkpoint_events_rank%d[%d]" % (mode, rank, index))
    return {"expected": expected, "present": present, "missing": missing}


def _required_pairs() -> List[Tuple[str, ResourceDomain]]:
    pairs: List[Tuple[str, ResourceDomain]] = []
    for mode in DOMAIN_MODES:
        for domain in sorted(mode_spec(mode).auxiliary_domains, key=lambda item: item.value):
            if domain in G1_DEFERRED_DOMAINS:
                continue
            pairs.append((mode, domain))
    return pairs


def _raw_mode_summary(root: Path) -> Dict[str, Dict[str, Any]]:
    """Copy only raw timing/durability summaries; never interpret them causally."""

    result: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        step_values: List[float] = []
        window_values: List[float] = []
        for rank in range(4):
            summary = _load(root / mode / ("summary_rank%d.json" % rank))
            for field, values in (("step_p99_ms", step_values), ("window_step_p99_ms", window_values)):
                value = summary.get(field)
                if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                    raise ValueError("%s/summary rank %d: %s is not a finite non-negative number" % (mode, rank, field))
                values.append(float(value))
        durable_values: List[float] = []
        if mode != "fg_only":
            for rank in range(4):
                events = _load(root / mode / ("checkpoint_events_rank%d.json" % rank))
                if not isinstance(events, list):
                    raise ValueError("%s/checkpoint_events_rank%d.json is not a list" % (mode, rank))
                for event in events:
                    value = event.get("durable_ms")
                    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                        raise ValueError("%s/rank %d: durable_ms is not a finite non-negative number" % (mode, rank))
                    durable_values.append(float(value))
        result[mode] = {
            "rank_max_step_p99_ms": max(step_values),
            "rank_max_window_step_p99_ms": max(window_values),
            "rank_max_durable_ms": max(durable_values) if durable_values else None,
        }
    return result


def _foreground_path_status(root: Path) -> Dict[str, Any]:
    """Validate the observed foreground route before causal readiness.

    The raw tier runner historically collected auxiliary counters first and
    left foreground route evidence to the metric sidecar.  That is unsafe for
    a readiness record: auxiliary bytes alone cannot prove that the
    foreground used the same domain.  A future live runner must therefore
    publish a source-bound ``foreground_path.json``; missing or malformed
    evidence keeps the artifact ``not_ready`` and is never replaced by a
    topology label.
    """

    path = root / FOREGROUND_PATH_FILENAME
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file():
        return {
            "status": "missing",
            "path": str(path),
            "domains": [],
            "reasons": ["foreground_path.json is missing"],
        }
    if not digest_path.is_file():
        return {
            "status": "invalid",
            "path": str(path),
            "domains": [],
            "reasons": ["foreground_path.json.sha256 is missing"],
        }
    encoded = path.read_bytes()
    expected_digest = hashlib.sha256(encoded).hexdigest()
    actual_digest = digest_path.read_text(encoding="utf-8").strip()
    if actual_digest != expected_digest or len(actual_digest) != 64:
        return {
            "status": "invalid",
            "path": str(path),
            "domains": [],
            "reasons": ["foreground_path.json.sha256 does not match exact bytes"],
        }
    try:
        raw = json.loads(encoded.decode("utf-8"))
        normalized = validate_foreground_path(raw)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "path": str(path),
            "domains": [],
            "reasons": [f"foreground path is invalid: {exc}"],
        }
    return {
        "status": "observed_supported",
        "path": str(path),
        "sha256": expected_digest,
        "domains": list(normalized["domains"]),
        "sample_counts": {
            domain: len(series) for domain, series in normalized["counters"].items()
        },
        "reasons": [],
    }


def _validate_counter_file(path: Path, mode: str, domain: ResourceDomain) -> Dict[str, Any]:
    raw = _load(path)
    expected_keys = {
        "schema", "mode", "domain", "scope", "scope_id", "intervention_id",
        "path_evidence", "counter_family", "path_status", "counter_support",
        "source", "hardware_counter", "samples",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise ValueError("%s: counter record keys are not exact" % path)
    if raw["schema"] != COUNTER_SCHEMA or raw["mode"] != mode or raw["domain"] != domain.value:
        raise ValueError("%s: counter schema/mode/domain mismatch" % path)
    if (
        type(raw["scope"]) is not str
        or not raw["scope"]
        or type(raw["scope_id"]) is not str
        or not raw["scope_id"]
        or raw["scope"] not in allowed_counter_scopes(domain)
    ):
        raise ValueError("%s: counter scope is not rank/slice/endpoint bound for %s" % (path, domain.value))
    if raw["intervention_id"] != mode:
        raise ValueError("%s: counter intervention binding does not match mode" % path)
    contract = domain_contract(domain)
    if raw["path_evidence"] != contract.path_evidence or raw["counter_family"] != contract.counter_family:
        raise ValueError("%s: counter contract labels mismatch" % path)
    if raw["path_status"] != PathStatus.OBSERVED.value or raw["counter_support"] != CounterSupport.SUPPORTED.value:
        raise ValueError("%s: causal record is not observed/supported" % path)
    if raw["hardware_counter"] is not True or not isinstance(raw["source"], str) or not raw["source"]:
        raise ValueError("%s: hardware source marker is invalid" % path)
    samples = raw["samples"]
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("%s: at least two counter samples are required" % path)
    parsed: List[CounterSnapshot] = []
    for item in samples:
        if not isinstance(item, dict) or set(item) != {
            "sample_id", "source", "timestamp_ns", "cumulative_bytes",
            "cumulative_busy_ns", "support",
        }:
            raise ValueError("%s: counter sample keys are not exact" % path)
        if (
            type(item["sample_id"]) is not str
            or not item["sample_id"]
            or type(item["source"]) is not str
            or not item["source"]
        ):
            raise ValueError("%s: counter sample_id/source must be non-empty strings" % path)
        if item["source"] != raw["source"] or item["support"] != raw["counter_support"]:
            raise ValueError("%s: counter sample provenance mismatch" % path)
        parsed.append(CounterSnapshot(
            domain=domain,
            sample_id=item["sample_id"],
            source=item["source"],
            timestamp_ns=item["timestamp_ns"],
            cumulative_bytes=item["cumulative_bytes"],
            cumulative_busy_ns=item["cumulative_busy_ns"],
            support=CounterSupport(item["support"]),
        ))
    validate_counter_series(parsed)
    return {
        "path": str(path),
        "mode": mode,
        "domain": domain.value,
        "scope": raw["scope"],
        "scope_id": raw["scope_id"],
        "intervention_id": raw["intervention_id"],
        "samples": len(parsed),
        "source": raw["source"],
    }


def build_g1_causal_readiness(root: Path) -> Dict[str, Any]:
    root = root.resolve()
    raw = _bound_raw_validator(root)(root)
    manifest = _load(root / "g1_tier_runtime_manifest.json")
    logical = _logical_stage_inventory(root, manifest["checkpoint_steps"])
    raw_mode_summary = _raw_mode_summary(root)
    foreground_path = _foreground_path_status(root)
    required_domains = sorted(
        domain.value for domain in required_domains_for_modes(REQUIRED_G1_MODES)
        if domain not in G1_DEFERRED_DOMAINS
    )
    required_pairs = _required_pairs()
    observed: List[Dict[str, Any]] = []
    missing_pairs: List[Dict[str, str]] = []
    invalid_pairs: List[Dict[str, str]] = []
    for mode, domain in required_pairs:
        path = root / "domain_counters" / mode / (domain.value + ".json")
        if not path.is_file():
            missing_pairs.append({"mode": mode, "domain": domain.value})
            continue
        try:
            observed.append(_validate_counter_file(path, mode, domain))
        except (TypeError, ValueError, KeyError) as exc:
            invalid_pairs.append({"mode": mode, "domain": domain.value, "reason": str(exc)})
    missing_domains = sorted({item["domain"] for item in missing_pairs})
    reasons: List[str] = []
    if logical["missing"]:
        reasons.append("logical stage timing is missing from %d record(s)" % len(logical["missing"]))
    if missing_pairs:
        reasons.append("domain counter records are missing for %d mode/domain pair(s)" % len(missing_pairs))
    if invalid_pairs:
        reasons.append("domain counter records are malformed or unsupported")
    if foreground_path["status"] != "observed_supported":
        reasons.extend(foreground_path["reasons"])
    ready = (
        not logical["missing"]
        and not missing_pairs
        and not invalid_pairs
        and foreground_path["status"] == "observed_supported"
    )
    return {
        "schema_version": SCHEMA,
        "status": "ready" if ready else "not_ready",
        "promotion_ready": False,
        "raw_status": raw["status"],
        "live_external_execution": raw["live_external_execution"],
        "required_domains": required_domains,
        "required_mode_domain_pairs": [
            {"mode": mode, "domain": domain.value} for mode, domain in required_pairs
        ],
        "deferred_domains": sorted(domain.value for domain in G1_DEFERRED_DOMAINS),
        "collection_plan": G1_COUNTER_COLLECTION_PLAN,
        "observed_counter_records": observed,
        "missing_domains": missing_domains,
        "missing_mode_domain_pairs": missing_pairs,
        "invalid_mode_domain_pairs": invalid_pairs,
        "logical_stage": logical,
        "raw_mode_summary": raw_mode_summary,
        "foreground_path": foreground_path,
        "reasons": reasons,
        "next_gate": "compose_g1_result_with_independent_causal_sidecar" if ready else "collect_actual_domain_counters_and_rerun",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_g1_causal_readiness(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
