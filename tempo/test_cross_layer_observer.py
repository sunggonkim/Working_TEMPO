from __future__ import annotations

import json
from pathlib import Path

import pytest

from tempo.cross_layer_observer import (
    NCCLObserverSnapshot,
    publish_observer_snapshot,
    read_observer_snapshot,
    snapshot_age_ms,
)


def _snapshot(**overrides: object) -> NCCLObserverSnapshot:
    values: dict[str, object] = {
        "source_epoch": "slurm-123",
        "sequence": 4,
        "sampled_unix_ns": 10_000_000_000,
        "window_ms": 25.0,
        "communicator_id": "cojob-world",
        "topology_fingerprint_sha256": "a" * 64,
        "nccl_collective_p99_ms": 12.0,
        "nccl_arrival_spread_ms": None,
        "lmcache_transfer_p99_ms": 31.0,
        "uncertainty_ms": 0.0,
        "rank_count": 8,
        "background_mode": "nixl_ucx",
        "producer_state": "active",
        "correctness_met": True,
    }
    values.update(overrides)
    return NCCLObserverSnapshot(**values)


def test_snapshot_round_trip_is_exact(tmp_path: Path) -> None:
    path = tmp_path / "observer.json"
    original = _snapshot()
    publish_observer_snapshot(path, original)
    assert read_observer_snapshot(path) == original
    assert json.loads(path.read_text(encoding="utf-8")) == original.as_dict()


def test_snapshot_rejects_partial_inventory(tmp_path: Path) -> None:
    path = tmp_path / "observer.json"
    payload = _snapshot().as_dict()
    payload.pop("source_epoch")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        read_observer_snapshot(path)


def test_snapshot_age_uses_unix_clock_only() -> None:
    snapshot = _snapshot(sampled_unix_ns=10_000_000_000)
    assert snapshot_age_ms(snapshot, now_unix_ns=10_025_000_000) == 25.0
    with pytest.raises(ValueError, match="future"):
        snapshot_age_ms(snapshot, now_unix_ns=9_999_999_999)


def test_active_failed_snapshot_is_not_publishable() -> None:
    with pytest.raises(ValueError, match="failed correctness"):
        _snapshot(correctness_met=False)


def test_completed_snapshot_is_valid_but_not_active() -> None:
    snapshot = _snapshot(producer_state="complete")
    assert snapshot.producer_state == "complete"
