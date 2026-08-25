"""Strict, unprivileged observer snapshots for the TEMPO cross-layer loop.

The producer is an experiment workload (for example, an official LMCache/NIXL
co-job) and the consumer is a vLLM pair router.  The snapshot is deliberately
small and action-relevant: it carries local-window NCCL/transfer evidence,
identity, and publication state.  It is not a physical-switch claim and it
never uses cross-host monotonic-clock subtraction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping


NCCL_OBSERVER_SCHEMA = "tempo-nccl-observer-v1"
NCCL_OBSERVER_SOURCE = "cuda_collective_observer_cojob"


def _positive_int(name: str, value: object, *, zero: bool = False) -> int:
    if type(value) is not int or value < (0 if zero else 1):
        qualifier = "non-negative" if zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} int")
    return value


def _finite_nonnegative(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _optional_ms(name: str, value: object) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(name, value)


def _nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _sha256(name: str, value: object) -> str:
    result = _nonempty(name, value)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


@dataclass(frozen=True)
class NCCLObserverSnapshot:
    """One producer window published to the vLLM observer consumer."""

    source_epoch: str
    sequence: int
    sampled_unix_ns: int
    window_ms: float
    communicator_id: str
    topology_fingerprint_sha256: str
    nccl_collective_p99_ms: float | None
    nccl_arrival_spread_ms: float | None
    lmcache_transfer_p99_ms: float | None
    uncertainty_ms: float
    rank_count: int
    background_mode: str
    producer_state: str
    correctness_met: bool
    source: str = NCCL_OBSERVER_SOURCE
    schema: str = NCCL_OBSERVER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != NCCL_OBSERVER_SCHEMA:
            raise ValueError("NCCL observer schema mismatch")
        if self.source != NCCL_OBSERVER_SOURCE:
            raise ValueError("NCCL observer source mismatch")
        _nonempty("source_epoch", self.source_epoch)
        _positive_int("sequence", self.sequence)
        _positive_int("sampled_unix_ns", self.sampled_unix_ns)
        _finite_nonnegative("window_ms", self.window_ms)
        if self.window_ms <= 0.0:
            raise ValueError("window_ms must be positive")
        _nonempty("communicator_id", self.communicator_id)
        _sha256(
            "topology_fingerprint_sha256",
            self.topology_fingerprint_sha256,
        )
        _optional_ms("nccl_collective_p99_ms", self.nccl_collective_p99_ms)
        _optional_ms("nccl_arrival_spread_ms", self.nccl_arrival_spread_ms)
        _optional_ms("lmcache_transfer_p99_ms", self.lmcache_transfer_p99_ms)
        _finite_nonnegative("uncertainty_ms", self.uncertainty_ms)
        _positive_int("rank_count", self.rank_count)
        if self.producer_state not in {"active", "complete"}:
            raise ValueError("producer_state must be active or complete")
        _nonempty("background_mode", self.background_mode)
        if type(self.correctness_met) is not bool:
            raise TypeError("correctness_met must be bool")
        if not self.correctness_met and self.producer_state == "active":
            raise ValueError("active observer cannot publish failed correctness")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source": self.source,
            "source_epoch": self.source_epoch,
            "sequence": self.sequence,
            "sampled_unix_ns": self.sampled_unix_ns,
            "window_ms": self.window_ms,
            "communicator_id": self.communicator_id,
            "topology_fingerprint_sha256": self.topology_fingerprint_sha256,
            "nccl_collective_p99_ms": self.nccl_collective_p99_ms,
            "nccl_arrival_spread_ms": self.nccl_arrival_spread_ms,
            "lmcache_transfer_p99_ms": self.lmcache_transfer_p99_ms,
            "uncertainty_ms": self.uncertainty_ms,
            "rank_count": self.rank_count,
            "background_mode": self.background_mode,
            "producer_state": self.producer_state,
            "correctness_met": self.correctness_met,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NCCLObserverSnapshot":
        required = {
            "schema", "source", "source_epoch", "sequence", "sampled_unix_ns",
            "window_ms", "communicator_id", "topology_fingerprint_sha256",
            "nccl_collective_p99_ms", "nccl_arrival_spread_ms",
            "lmcache_transfer_p99_ms", "uncertainty_ms", "rank_count",
            "background_mode", "producer_state", "correctness_met",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("NCCL observer snapshot inventory is not exact")
        return cls(
            schema=value["schema"],
            source=value["source"],
            source_epoch=value["source_epoch"],
            sequence=value["sequence"],
            sampled_unix_ns=value["sampled_unix_ns"],
            window_ms=float(value["window_ms"]),
            communicator_id=value["communicator_id"],
            topology_fingerprint_sha256=value["topology_fingerprint_sha256"],
            nccl_collective_p99_ms=_optional_ms(
                "nccl_collective_p99_ms", value["nccl_collective_p99_ms"]),
            nccl_arrival_spread_ms=_optional_ms(
                "nccl_arrival_spread_ms", value["nccl_arrival_spread_ms"]),
            lmcache_transfer_p99_ms=_optional_ms(
                "lmcache_transfer_p99_ms", value["lmcache_transfer_p99_ms"]),
            uncertainty_ms=float(value["uncertainty_ms"]),
            rank_count=value["rank_count"],
            background_mode=value["background_mode"],
            producer_state=value["producer_state"],
            correctness_met=value["correctness_met"],
        )


def publish_observer_snapshot(
    path: str | os.PathLike[str],
    snapshot: NCCLObserverSnapshot,
) -> None:
    """Atomically publish a snapshot without exposing a partial JSON file."""

    if not isinstance(snapshot, NCCLObserverSnapshot):
        raise TypeError("snapshot must be NCCLObserverSnapshot")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                snapshot.as_dict(),
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_observer_snapshot(path: str | os.PathLike[str]) -> NCCLObserverSnapshot:
    """Read and validate one complete observer snapshot."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("NCCL observer snapshot is not an object")
    return NCCLObserverSnapshot.from_mapping(raw)


def snapshot_age_ms(snapshot: NCCLObserverSnapshot, *, now_unix_ns: int | None = None) -> float:
    """Return producer wall-clock age for freshness, never a cross-host monotonic delta."""

    now = time.time_ns() if now_unix_ns is None else _positive_int(
        "now_unix_ns", now_unix_ns)
    age_ms = (now - snapshot.sampled_unix_ns) / 1_000_000.0
    if age_ms < 0.0:
        raise ValueError("NCCL observer snapshot is from the future")
    return age_ms


__all__ = [
    "NCCL_OBSERVER_SCHEMA",
    "NCCL_OBSERVER_SOURCE",
    "NCCLObserverSnapshot",
    "publish_observer_snapshot",
    "read_observer_snapshot",
    "snapshot_age_ms",
]
