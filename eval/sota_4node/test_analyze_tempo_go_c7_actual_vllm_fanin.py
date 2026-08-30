from __future__ import annotations

import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_go_c7_actual_vllm_fanin as analyzer


def _row(request_id: str, e2e_ms: float) -> dict[str, object]:
    return {
        "request_id": request_id,
        "valid": True,
        "terminal_kind": "complete",
        "dispatch_offset_ns": 0,
        "token_arrival_offsets_ns": [
            int(0.2 * e2e_ms * 1e6), int(0.8 * e2e_ms * 1e6)],
        "stream_end_offset_ns": int(e2e_ms * 1e6),
    }


def _decision(request_id: str, source: int) -> dict[str, object]:
    return {
        "request_id": request_id,
        "route": "official_lmcache_remote_prefill",
        "frontend_pair_index": source,
        "remote_decoder_index": 0,
    }


def test_material_two_prefill_receiver_incast_passes(tmp_path: Path) -> None:
    specs = [
        {"name": "00_control_a", "aggressor_rate_per_s": 0.0},
        {"name": "01_hot", "aggressor_rate_per_s": 10.0},
        {"name": "02_control_b", "aggressor_rate_per_s": 0.0},
    ]
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "schema": analyzer.CONTRACT_SCHEMA,
        "actual_vllm_fanin": {
            "phase_duration_ms": 30000,
            "blocks": specs,
            "victim": {"slo": {"e2e_ms": 2000, "tpot_ms": 1000}},
            "qualification_gates": {
                "victim_e2e_p50_degradation_fraction": 0.25,
                "victim_e2e_p99_ratio": 2.0,
                "victim_slo_attainment_drop_percentage_points": 20.0,
            },
        },
    }))
    artifacts = {}
    contracts = {}
    for spec in specs:
        name = spec["name"]
        hot = spec["aggressor_rate_per_s"] > 0
        request_index = {}
        rows = []
        decisions = []
        for ordinal in range(2):
            request_id = f"{name}-victim-{ordinal}"
            request_index[request_id] = {
                "role": "victim",
                "source_prefill_index": 0,
            }
            rows.append(_row(request_id, 3000.0 if hot else 1000.0))
            decisions.append(_decision(request_id, 0))
        if hot:
            for source in (0, 1):
                request_id = f"{name}-aggressor-{source}"
                request_index[request_id] = {
                    "role": "aggressor",
                    "source_prefill_index": source,
                    "arrival_offset_ms": source * 29000.0,
                }
                rows.append(_row(request_id, 1000.0))
                decisions.append(_decision(request_id, source))
        raw = tmp_path / f"{name}.json"
        raw.write_text(json.dumps({
            "requests": rows, "router_decisions": decisions,
        }))
        artifacts[name] = str(raw)
        contracts[name] = {
            "aggressor_rate_per_s": spec["aggressor_rate_per_s"],
            "request_index": request_index,
        }
    bundle = {
        "schema": analyzer.BUNDLE_SCHEMA,
        "artifacts": artifacts,
        "contracts": contracts,
    }
    result = analyzer.analyze_bundle(bundle, contract)
    assert result["actual_two_prefill_to_one_decoder_fanin"] is True
    assert result["material_independent_victim_degradation"] is True
    assert result["c7_actual_vllm_fanin_qualification_pass"] is True
    assert result["first_material_knee_rate_per_s"] == 10.0
