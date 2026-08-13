"""Run-level replication gates for TEMPO-RD paper claims.

Steps and collectives are not independent replicates.  These small immutable
records require independent complete blocks before a training or inference
win can be promoted into a paper-level claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _nonnegative(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative int")


def _fingerprint(name: str, value: str) -> None:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")


@dataclass(frozen=True)
class TrainingReplicationBlock:
    block_id: str
    source_bundle_sha256: str
    workload_fingerprint: str
    open_tail_ns: int
    candidate_tail_ns: int
    open_skew_ns: int
    candidate_skew_ns: int
    deadline_met: bool
    correctness_met: bool

    def __post_init__(self) -> None:
        if type(self.block_id) is not str or not self.block_id:
            raise ValueError("block_id must be non-empty")
        _fingerprint("source_bundle_sha256", self.source_bundle_sha256)
        _fingerprint("workload_fingerprint", self.workload_fingerprint)
        for name in (
            "open_tail_ns", "candidate_tail_ns", "open_skew_ns", "candidate_skew_ns"
        ):
            _nonnegative(name, getattr(self, name))
        if type(self.deadline_met) is not bool or type(self.correctness_met) is not bool:
            raise TypeError("deadline_met and correctness_met must be bool")


@dataclass(frozen=True)
class InferenceReplicationBlock:
    block_id: str
    source_bundle_sha256: str
    workload_fingerprint: str
    open_ttft_ns: int
    candidate_ttft_ns: int
    open_itl_ns: int
    candidate_itl_ns: int
    open_slo_goodput_milli: int
    candidate_slo_goodput_milli: int
    deadline_met: bool
    correctness_met: bool

    def __post_init__(self) -> None:
        if type(self.block_id) is not str or not self.block_id:
            raise ValueError("block_id must be non-empty")
        _fingerprint("source_bundle_sha256", self.source_bundle_sha256)
        _fingerprint("workload_fingerprint", self.workload_fingerprint)
        for name in (
            "open_ttft_ns", "candidate_ttft_ns", "open_itl_ns", "candidate_itl_ns",
            "open_slo_goodput_milli", "candidate_slo_goodput_milli",
        ):
            _nonnegative(name, getattr(self, name))
        for name in ("open_slo_goodput_milli", "candidate_slo_goodput_milli"):
            if getattr(self, name) > 1_000_000:
                raise ValueError(f"{name} must be at most 1000000")
        if type(self.deadline_met) is not bool or type(self.correctness_met) is not bool:
            raise TypeError("deadline_met and correctness_met must be bool")


@dataclass(frozen=True)
class ReplicationResult:
    complete_blocks: int
    wins: int
    minimum_blocks: int
    required_wins: int
    eligible: bool
    reasons: tuple[str, ...]


def _validate_thresholds(minimum_blocks: int, required_wins: int) -> None:
    if type(minimum_blocks) is not int or minimum_blocks <= 0:
        raise ValueError("minimum_blocks must be positive")
    if type(required_wins) is not int or required_wins <= 0:
        raise ValueError("required_wins must be positive")
    if required_wins > minimum_blocks:
        raise ValueError("required_wins cannot exceed minimum_blocks")


def evaluate_training_replication(
    blocks: Iterable[TrainingReplicationBlock],
    *,
    minimum_blocks: int = 5,
    required_wins: int = 4,
) -> ReplicationResult:
    """Require 4/5 independent complete blocks winning both tail metrics."""

    _validate_thresholds(minimum_blocks, required_wins)
    values = tuple(blocks)
    if any(not isinstance(block, TrainingReplicationBlock) for block in values):
        raise TypeError("blocks must contain TrainingReplicationBlock values")
    if len({block.block_id for block in values}) != len(values):
        raise ValueError("block_id values must be unique")
    reasons: list[str] = []
    if len({block.source_bundle_sha256 for block in values}) > 1:
        reasons.append("independent blocks use different source bundles")
    if len({block.workload_fingerprint for block in values}) > 1:
        reasons.append("independent blocks use different workload fingerprints")
    complete = [block for block in values if block.deadline_met and block.correctness_met]
    wins = sum(
        block.candidate_tail_ns < block.open_tail_ns
        and block.candidate_skew_ns < block.open_skew_ns
        for block in complete
    )
    if len(complete) < minimum_blocks:
        reasons.append("insufficient independent complete blocks")
    if len(complete) != len(values):
        reasons.append("at least one block failed deadline or correctness")
    if wins < required_wins:
        reasons.append("candidate does not win both training tail metrics often enough")
    return ReplicationResult(
        len(complete), wins, minimum_blocks, required_wins,
        not reasons, tuple(reasons),
    )


def evaluate_inference_replication(
    blocks: Iterable[InferenceReplicationBlock],
    *,
    minimum_blocks: int = 5,
    required_wins: int = 4,
) -> ReplicationResult:
    """Require 4/5 independent blocks winning TTFT/ITL and preserving goodput."""

    _validate_thresholds(minimum_blocks, required_wins)
    values = tuple(blocks)
    if any(not isinstance(block, InferenceReplicationBlock) for block in values):
        raise TypeError("blocks must contain InferenceReplicationBlock values")
    if len({block.block_id for block in values}) != len(values):
        raise ValueError("block_id values must be unique")
    reasons: list[str] = []
    if len({block.source_bundle_sha256 for block in values}) > 1:
        reasons.append("independent blocks use different source bundles")
    if len({block.workload_fingerprint for block in values}) > 1:
        reasons.append("independent blocks use different workload fingerprints")
    complete = [block for block in values if block.deadline_met and block.correctness_met]
    wins = sum(
        block.candidate_ttft_ns < block.open_ttft_ns
        and block.candidate_itl_ns < block.open_itl_ns
        and block.candidate_slo_goodput_milli >= block.open_slo_goodput_milli
        for block in complete
    )
    if len(complete) < minimum_blocks:
        reasons.append("insufficient independent complete blocks")
    if len(complete) != len(values):
        reasons.append("at least one block failed deadline or correctness")
    if wins < required_wins:
        reasons.append("candidate does not win inference latency and preserve goodput often enough")
    return ReplicationResult(
        len(complete), wins, minimum_blocks, required_wins,
        not reasons, tuple(reasons),
    )
