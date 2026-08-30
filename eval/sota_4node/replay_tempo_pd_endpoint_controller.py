#!/usr/bin/env python3
"""Calibration-only replay for the endpoint-completion TEMPO controller."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import statistics
from typing import Mapping

from tempo.pd_elastic_controller_v443 import CacheResidency
from tempo.pd_elastic_profile import load_elastic_profile
from tempo.pd_endpoint_controller import (
    EndpointFeedbackController,
    EndpointRequest,
    EndpointRoute,
    EndpointWork,
)
from tempo.pd_endpoint_profile import load_endpoint_service_profile


SCHEMA = "tempo-pd-endpoint-controller-replay-v1"
_ITEM = re.compile(r"^epd-(?:local|remote)-r(\d+)-measured-item-(\d+)$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _e2e_ms(row: Mapping[str, object]) -> float:
    return (
        int(row["stream_end_offset_ns"]) - int(row["dispatch_offset_ns"])
    ) / 1_000_000.0


def _ttft_ms(row: Mapping[str, object]) -> float:
    return (
        int(row["token_arrival_offsets_ns"][0])
        - int(row["dispatch_offset_ns"])
    ) / 1_000_000.0


def _nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summary(values: list[float], *, deadline_ms: float) -> dict[str, object]:
    _require(bool(values), "latency summary cannot be empty")
    return {
        "count": len(values),
        "goodput_fraction": sum(value <= deadline_ms for value in values) / len(values),
        "mean_e2e_ms": statistics.mean(values),
        "median_e2e_ms": statistics.median(values),
        "p99_e2e_ms": _nearest_rank(values, 0.99),
    }


def _artifact_index(artifact: Mapping[str, object]) -> dict[tuple[int, int], dict[str, object]]:
    validation = artifact.get("validation")
    _require(
        isinstance(validation, Mapping)
        and validation.get("all_streams_valid") is True
        and validation.get("router_decisions_exact") is True,
        "replay artifact is invalid",
    )
    result = {}
    for row in artifact["requests"]:
        match = _ITEM.fullmatch(str(row.get("request_id")))
        _require(match is not None, "replay request ID is not paired")
        key = (int(match.group(1)), int(match.group(2)))
        _require(key not in result and row.get("valid") is True,
                 "replay request is duplicate or invalid")
        result[key] = row
    return result


def replay_idle_pairs(
    *, elastic, endpoint, local_artifacts, remote_artifacts, deadline_ms: float,
    foreground_rate_per_s: float,
) -> dict[str, object]:
    local = {}
    remote = {}
    for artifact in local_artifacts:
        local.update(_artifact_index(artifact))
    for artifact in remote_artifacts:
        remote.update(_artifact_index(artifact))
    _require(set(local) == set(remote), "local/remote replay pairs differ")
    controller = EndpointFeedbackController(endpoint.controller)
    events: list[tuple[int, str, float]] = []
    endpoint_values = []
    fixed_local = []
    fixed_remote = []
    predictor = []
    oracle = []
    decisions = []

    ordered = sorted(
        local,
        key=lambda key: (key[0], int(local[key]["dispatch_offset_ns"]), key[1]),
    )
    _require(foreground_rate_per_s > 0.0, "foreground replay rate must be positive")
    spacing_ns = round(1_000_000_000 / foreground_rate_per_s)
    replicate_offsets = {
        replicate: replicate * 20_000_000_000
        for replicate in {k[0] for k in ordered}
    }
    for replicate, item in ordered:
        local_row = local[(replicate, item)]
        remote_row = remote[(replicate, item)]
        _require(
            local_row["output_text_sha256"] == remote_row["output_text_sha256"],
            "paired replay outputs differ",
        )
        # The source calibration intentionally dispatched at 200 rps.  Keep
        # its paired service counterfactuals, but replay admission at the C4
        # manifest's frozen foreground rate rather than relabeling that burst
        # as an idle controller trace.
        now_ns = replicate_offsets[replicate] + item * spacing_ns
        while events and events[0][0] <= now_ns:
            completed_ns, request_id, observed_ttft_ms = heapq.heappop(events)
            controller.observe_first_response(
                request_id, observed_ttft_ms=observed_ttft_ms, now_ns=completed_ns)
        prompt_tokens = int(local_row["usage"]["prompt_tokens"])
        output_tokens = int(local_row["requested_max_tokens"])
        elastic_row = elastic.exact_row(prompt_tokens, output_tokens)
        _require(elastic_row is not None, "replay geometry lacks elastic profile row")
        service = endpoint.exact_row(
            prompt_tokens, output_tokens, CacheResidency.P_ONLY)
        request_id = f"replay-r{replicate}-item-{item:03d}"
        request = EndpointRequest(
            request_id=request_id,
            local_e2e_prior_ms=elastic_row.local_upper_bound_ms,
            remote_e2e_prior_ms=elastic_row.remote_upper_bound_ms,
            local_ttft_prior_ms=service.local_ttft_prior_ms,
            remote_ttft_prior_ms=service.remote_ttft_prior_ms,
            uncertainty_ms=elastic_row.uncertainty_ms,
            e2e_deadline_ms=deadline_ms,
            work=EndpointWork(
                local_token_ms=service.local_token_ms,
                remote_prefill_token_ms=service.remote_prefill_token_ms,
                remote_kv_bytes=elastic_row.remote_kv_bytes,
                remote_semantic_ops=1,
            ),
        )
        decision = controller.submit(request, now_ns=now_ns)
        _require(decision.route is not EndpointRoute.QUEUE,
                 "idle calibration replay unexpectedly queued")
        observed = local_row if decision.route is EndpointRoute.LOCAL else remote_row
        observed_ttft = _ttft_ms(observed)
        heapq.heappush(
            events,
            (now_ns + math.ceil(observed_ttft * 1_000_000), request_id, observed_ttft),
        )
        local_e2e = _e2e_ms(local_row)
        remote_e2e = _e2e_ms(remote_row)
        chosen_e2e = local_e2e if decision.route is EndpointRoute.LOCAL else remote_e2e
        endpoint_values.append(chosen_e2e)
        fixed_local.append(local_e2e)
        fixed_remote.append(remote_e2e)
        predictor.append(
            local_e2e
            if elastic_row.local_upper_bound_ms <= elastic_row.remote_upper_bound_ms
            else remote_e2e
        )
        oracle.append(min(local_e2e, remote_e2e))
        decisions.append({
            "request_id": request_id,
            "route": decision.route.value,
            "reason": decision.reason,
            "local_score_ms": decision.local_score_ms,
            "remote_score_ms": decision.remote_score_ms,
            "local_e2e_ms": local_e2e,
            "remote_e2e_ms": remote_e2e,
            "chosen_e2e_ms": chosen_e2e,
        })
    while events:
        completed_ns, request_id, observed_ttft_ms = heapq.heappop(events)
        controller.observe_first_response(
            request_id, observed_ttft_ms=observed_ttft_ms, now_ns=completed_ns)
    snapshot = controller.snapshot(now_ns=max(replicate_offsets.values()) + 30_000_000_000)
    route_counts = {
        route.value: sum(row["route"] == route.value for row in decisions)
        for route in (EndpointRoute.LOCAL, EndpointRoute.REMOTE)
    }
    return {
        "paired_requests": len(decisions),
        "route_counts": route_counts,
        "endpoint": _summary(endpoint_values, deadline_ms=deadline_ms),
        "predictor": _summary(predictor, deadline_ms=deadline_ms),
        "fixed_local": _summary(fixed_local, deadline_ms=deadline_ms),
        "fixed_remote": _summary(fixed_remote, deadline_ms=deadline_ms),
        "per_request_oracle": _summary(oracle, deadline_ms=deadline_ms),
        "controller_final": snapshot,
        "decisions": decisions,
    }


def replay_measured_stretch_transition(*, elastic, endpoint, deadline_ms: float) -> dict[str, object]:
    """Exercise deflection/recovery with the measured C3 rate-12 stretch."""

    prompt_tokens, output_tokens = 512, 16
    elastic_row = elastic.exact_row(prompt_tokens, output_tokens)
    _require(elastic_row is not None, "transition geometry missing")
    service = endpoint.exact_row(
        prompt_tokens, output_tokens, CacheResidency.P_ONLY)
    controller = EndpointFeedbackController(endpoint.controller)
    routes = []
    now_ns = 1

    def one(name: str, remote_stretch: float) -> None:
        nonlocal now_ns
        request = EndpointRequest(
            request_id=name,
            local_e2e_prior_ms=elastic_row.local_upper_bound_ms,
            remote_e2e_prior_ms=elastic_row.remote_upper_bound_ms,
            local_ttft_prior_ms=service.local_ttft_prior_ms,
            remote_ttft_prior_ms=service.remote_ttft_prior_ms,
            uncertainty_ms=elastic_row.uncertainty_ms,
            e2e_deadline_ms=deadline_ms,
            work=EndpointWork(
                local_token_ms=service.local_token_ms,
                remote_prefill_token_ms=service.remote_prefill_token_ms,
                remote_kv_bytes=elastic_row.remote_kv_bytes,
                remote_semantic_ops=1,
            ),
        )
        decision = controller.submit(request, now_ns=now_ns)
        routes.append({
            "request_id": name,
            "route": decision.route.value,
            "probe": decision.probe,
            "local_multiplier": decision.local_multiplier,
            "remote_multiplier": decision.remote_multiplier,
        })
        _require(decision.route is not EndpointRoute.QUEUE, "transition replay queued")
        observed = (
            service.remote_ttft_prior_ms * remote_stretch
            if decision.route is EndpointRoute.REMOTE
            else service.local_ttft_prior_ms
        )
        now_ns += max(1, math.ceil(observed * 1_000_000))
        controller.observe_first_response(name, observed_ttft_ms=observed, now_ns=now_ns)
        now_ns += 1

    one("control-0", 1.0)
    one("control-1", 1.0)
    one("overload-evidence", 4.63485363490015)
    one("after-overload", 4.63485363490015)
    now_ns += endpoint.controller.feedback_fresh_ns + 1
    one("recovery-probe", 1.0)
    one("local-refresh-probe", 1.0)
    one("post-recovery", 1.0)
    _require(
        routes[0]["route"] == EndpointRoute.REMOTE.value
        and routes[1]["route"] == EndpointRoute.REMOTE.value
        and routes[3]["route"] == EndpointRoute.LOCAL.value
        and routes[4]["route"] == EndpointRoute.REMOTE.value
        and routes[4]["probe"] is True
        and routes[5]["route"] == EndpointRoute.LOCAL.value
        and routes[5]["probe"] is True
        and routes[6]["route"] == EndpointRoute.REMOTE.value
        and routes[6]["probe"] is False,
        f"measured-stretch replay did not deflect and recover: {routes}",
    )
    return {
        "measured_remote_inflation_at_rate12": 4.63485363490015,
        "routes": routes,
        "deflection_and_bounded_recovery_valid": True,
    }


def replay(*, manifest_path: Path, endpoint_profile_path: Path) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    endpoint = load_endpoint_service_profile(endpoint_profile_path.resolve())
    _require(
        endpoint.workload_manifest_sha256 == _sha256(manifest_path),
        "endpoint replay workload binding differs",
    )
    parents = manifest["calibration_parents"]
    elastic_path = (repo_root / parents["elastic_profile"]["path"]).resolve()
    elastic = load_elastic_profile(elastic_path)
    _require(
        endpoint.elastic_profile_fingerprint_sha256 == elastic.fingerprint_sha256,
        "endpoint replay elastic binding differs",
    )
    artifacts = []
    for entry in parents["endpoint_service_raw"]:
        path = (repo_root / entry["path"]).resolve()
        _require(_sha256(path) == entry["sha256"], "replay parent hash differs")
        artifacts.append(json.loads(path.read_text(encoding="utf-8")))
    local_artifacts = [
        artifact for artifact in artifacts
        if all(row["route"] == "decoder_local_chunked_prefill"
               for row in artifact["router_decisions"])
    ]
    remote_artifacts = [
        artifact for artifact in artifacts
        if all(row["route"] == "official_lmcache_remote_prefill"
               for row in artifact["router_decisions"])
    ]
    deadline_ms = float(manifest["measurement"]["e2e_slo_ms"])
    idle = replay_idle_pairs(
        elastic=elastic,
        endpoint=endpoint,
        local_artifacts=local_artifacts,
        remote_artifacts=remote_artifacts,
        deadline_ms=deadline_ms,
        foreground_rate_per_s=float(manifest["foreground"]["offered_rate_per_s"]),
    )
    transition = replay_measured_stretch_transition(
        elastic=elastic, endpoint=endpoint, deadline_ms=deadline_ms)
    resources = idle["controller_final"]["resources"]
    live_authorized = (
        idle["paired_requests"] == 48
        and all(value == 0 for value in resources.values())
        and transition["deflection_and_bounded_recovery_valid"] is True
    )
    _require(live_authorized, "offline endpoint replay did not authorize live C4")
    return {
        "schema": SCHEMA,
        "calibration_only": True,
        "performance_claim_allowed": False,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "endpoint_profile": str(endpoint_profile_path.resolve()),
        "endpoint_profile_sha256": _sha256(endpoint_profile_path.resolve()),
        "idle_paired_counterfactual": idle,
        "measured_stretch_transition": transition,
        "live_c4_screen_authorized": live_authorized,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--endpoint-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "refusing to overwrite replay")
    payload = replay(
        manifest_path=args.manifest,
        endpoint_profile_path=args.endpoint_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "live_c4_screen_authorized": payload["live_c4_screen_authorized"],
        "output": str(args.output.resolve()),
        "schema": payload["schema"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
