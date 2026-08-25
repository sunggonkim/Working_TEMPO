from __future__ import annotations

import pytest
import json
from pathlib import Path

from tempo.pd_global_workload import (
    ContentionPhase,
    build_contention_workload,
    canonical_contention_phases,
)
from eval.sota_4node.validate_tempo_go_manifest import validate_manifest


def pools() -> dict[int, list[str]]:
    return {
        512: ["p512-a", "p512-b"],
        2048: ["p2048-a", "p2048-b"],
        4094: ["p4094-a", "p4094-b"],
    }


def test_canonical_phases_preserve_c1_c2_c3_anchor_rates() -> None:
    phases = canonical_contention_phases()
    assert [item.name for item in phases] == [
        "c0_cool", "c1_decoder_hot", "c2_remote_hot",
        "c2_kv_remote_hot", "c3_both_hot", "recovery",
    ]
    assert phases[1].decoder_hot_rate_per_s == 22.4
    assert phases[2].remote_hot_rate_per_s == 4.76
    assert phases[3].kv_remote_hot_rate_per_s == 12.0
    assert phases[4].decoder_hot_rate_per_s == 22.4


def test_workload_has_only_native_client_fields_and_explicit_absolute_arrivals() -> None:
    rows, manifest = build_contention_workload(
        pools(),
        phases=(ContentionPhase(
            "c1_decoder_hot", duration_ms=1_000.0,
            foreground_rate_per_s=2.0, decoder_hot_rate_per_s=2.0,
            cooldown_ms=100.0,
        ),),
        replicates=1,
        background_output_tokens=16,
    )
    assert len(rows) == 4
    assert all(set(row) == {
        "request_id", "prompt", "max_tokens", "arrival_offset_ms",
    } for row in rows)
    assert [row["arrival_offset_ms"] for row in rows] == [0.0, 0.0, 500.0, 500.0]
    assert all(str(row["request_id"]).startswith("epd-tempo-") for row in rows)
    assert manifest["workload_fields"] == [
        "request_id", "prompt", "max_tokens", "arrival_offset_ms",
    ]
    assert manifest["performance_claim_allowed"] is False


def test_phase_ranges_are_stable_across_replicates() -> None:
    rows, manifest = build_contention_workload(
        pools(),
        phases=(ContentionPhase("c0_cool", duration_ms=500.0,
                                foreground_rate_per_s=2.0, cooldown_ms=50.0),),
        replicates=2,
    )
    assert len(rows) == 2
    assert manifest["phases"][0]["row_start"] == 0
    assert manifest["phases"][0]["row_end"] == 1
    assert manifest["phases"][1]["row_start"] == 1
    assert manifest["phases"][1]["row_end"] == 2
    assert len({row["request_id"] for row in rows}) == 2


def test_missing_background_geometry_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing geometry 4094"):
        build_contention_workload(
            {512: ["a"], 2048: ["b"]},
            phases=(ContentionPhase("hot", duration_ms=1_000.0,
                                    foreground_rate_per_s=0.0,
                                    decoder_hot_rate_per_s=1.0),),
        )


def test_manifest_validator_is_the_cpu_stop_gate(tmp_path: Path) -> None:
    rows, manifest = build_contention_workload(
        pools(),
        phases=(ContentionPhase("c0_cool", duration_ms=1_000.0,
                                foreground_rate_per_s=2.0, cooldown_ms=100.0),),
    )
    manifest_path = tmp_path / "manifest.json"
    workload_path = tmp_path / "validation.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    workload_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = validate_manifest(manifest_path, workload_path)
    assert report["arrival_offsets_monotonic"] is True
    assert report["native_client_fields_only"] is True
