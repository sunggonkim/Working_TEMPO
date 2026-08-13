#!/usr/bin/env python3
"""Build (but never submit) the TEMPO-RD G1 attribution matrix.

The production four-node runner is intentionally not reused here: its strict
v4_open/tempo_v4 comparison has a different correctness contract.  This
module only emits the exact policy/mode tuples that a future approved one-node
allocation may execute and validates that all tuples share the same geometry.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Sequence

try:
    from tempo.domain_evidence import CounterSupport, DomainEvidence, PathStatus
    from tempo.resource_domain import EvidenceLevel, ResourceDomain, domain_contract
    from tempo.observation_window import observation_window_contract
    from tempo.tier_attribution import (
        AttributionMode,
        mode_spec,
        required_domains_for_modes,
        validate_mode_evidence,
    )
except ModuleNotFoundError:  # direct ``python eval/.../tier_attribution_runner.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tempo.domain_evidence import CounterSupport, DomainEvidence, PathStatus
    from tempo.resource_domain import EvidenceLevel, ResourceDomain, domain_contract
    from tempo.observation_window import observation_window_contract
    from tempo.tier_attribution import (
        AttributionMode,
        mode_spec,
        required_domains_for_modes,
        validate_mode_evidence,
    )


@dataclass(frozen=True)
class TierRun:
    mode: str
    policy: str
    tier_mode: str
    endpoint: str
    requires_restore: bool
    requires_gpu_transfer: bool


@dataclass(frozen=True)
class TierCommand:
    """One deterministic train invocation for the future approved G1 run.

    This is deliberately an argv/env description rather than a subprocess
    launcher.  Keeping construction separate from execution prevents a
    design manifest or a unit test from accidentally submitting work.
    """

    mode: str
    policy: str
    tier_mode: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    requires_restore: bool


def build_g1_command_plan(
    *,
    repo_root: Path,
    result_root: Path,
    checkpoint_root: Path,
    train_snapshot: str = "train_executed.py",
    checkpoint_steps: Sequence[int] = (16, 52),
    steps: int = 72,
    warmup_steps: int = 12,
    world_size: int = 4,
    layers: int = 2,
    hidden_size: int = 2048,
    ffn_size: int = 8192,
    heads: int = 16,
    sequence_length: int = 64,
    batch_size: int = 1,
    window_steps: int = 16,
    probe_mb: int = 64,
    deadline_seconds: float = 1.0,
    datastates_cache_gb: float = 1.0,
    seed: int = 20260811,
) -> tuple[TierCommand, ...]:
    """Build the exact five-mode G1 argv/env plan without executing it.

    All modes use the same model geometry and checkpoint schedule.  The only
    intended differences are the declared tier mode, endpoint, and whether a
    persistent restore is meaningful.  Paths are absolute and rank-independent
    roots; the eventual Slurm wrapper appends its rank-local output path.
    """

    repo_root = Path(repo_root).resolve()
    result_root = Path(result_root).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    if not repo_root.is_dir():
        raise ValueError("repo_root must be an existing directory")
    if not result_root.is_absolute() or not checkpoint_root.is_absolute():
        raise ValueError("G1 result/checkpoint roots must be absolute")
    if not train_snapshot or Path(train_snapshot).name != train_snapshot:
        raise ValueError("train_snapshot must be a basename")
    if any(type(step) is not int for step in checkpoint_steps):
        raise ValueError("checkpoint_steps must contain only integers")
    run_steps = tuple(checkpoint_steps)
    if not run_steps or list(run_steps) != sorted(set(run_steps)):
        raise ValueError("checkpoint_steps must be sorted and unique")
    if type(world_size) is not int or world_size != 4:
        raise ValueError("G1 command plan requires world_size=4")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if type(warmup_steps) is not int or warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError("warmup_steps must be within the run")
    for name, value in {
        "layers": layers,
        "hidden_size": hidden_size,
        "ffn_size": ffn_size,
        "heads": heads,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "window_steps": window_steps,
        "probe_mb": probe_mb,
        "seed": seed,
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(deadline_seconds) not in (int, float) or deadline_seconds <= 0:
        raise ValueError("deadline_seconds must be positive")
    if type(datastates_cache_gb) not in (int, float) or datastates_cache_gb <= 0:
        raise ValueError("datastates_cache_gb must be positive")
    if hidden_size % heads:
        raise ValueError("hidden_size must be divisible by heads")
    # The intentionally simple constraints below are the ones also enforced
    # by train.py's parser, expressed without importing the training process.
    if any(step < warmup_steps or step + window_steps + 1 >= steps for step in run_steps):
        raise ValueError("checkpoint schedule does not fit the G1 run")

    commands: list[TierCommand] = []
    for run in build_g1_matrix():
        mode_root = result_root / run.mode
        checkpoint_dir = checkpoint_root / run.mode
        env = {
            "TEMPO_RD_TIER_MODE": run.tier_mode,
            "TEMPO_RD_ENDPOINT": run.endpoint,
            "TEMPO_RD_G1_MODE": run.mode,
        }
        if run.mode == AttributionMode.D2H_ONLY.value:
            env["TEMPO_RD_LOCAL_SINK_ROOT"] = str(mode_root / "local_sink")
        ordered_env = tuple(sorted(env.items()))
        argv = (
            "python",
            str(repo_root / train_snapshot),
            "--policy",
            run.policy,
            "--tier-mode",
            run.tier_mode,
            "--output-dir",
            str(mode_root),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--steps",
            str(steps),
            "--warmup-steps",
            str(warmup_steps),
            "--layers",
            str(layers),
            "--hidden-size",
            str(hidden_size),
            "--ffn-size",
            str(ffn_size),
            "--heads",
            str(heads),
            "--sequence-length",
            str(sequence_length),
            "--batch-size",
            str(batch_size),
            "--window-steps",
            str(window_steps),
            "--probe-mb",
            str(probe_mb),
            "--deadline-seconds",
            str(deadline_seconds),
            "--datastates-cache-gb",
            str(datastates_cache_gb),
            "--seed",
            str(seed),
            "--checkpoint-steps",
            ",".join(str(step) for step in run_steps),
        )
        commands.append(
            TierCommand(
                mode=run.mode,
                policy=run.policy,
                tier_mode=run.tier_mode,
                argv=argv,
                env=ordered_env,
                requires_restore=run.requires_restore,
            )
        )
    return tuple(commands)


EXECUTABLE_G1_MODES: tuple[AttributionMode, ...] = (
    AttributionMode.FOREGROUND_ONLY,
    AttributionMode.OPEN_COMBINED,
    AttributionMode.D2H_ONLY,
    AttributionMode.PERSIST_ONLY,
    AttributionMode.COMBINED,
)


_RUNNER_MANIFEST_KEYS = {
    "schema_version",
    "world_size",
    "nodes",
    "state_bytes_per_rank",
    "deadline_ns",
    "checkpoint_steps",
    "runs",
    "domain_footprints",
    "optional_modes",
    "evidence_state",
    "evidence_records",
    "required_domains",
    "evidence_contract",
    "slurm_submitted",
    "inference_adapter",
    "observation_window_contract",
}

G1_FOREGROUND_DOMAINS = tuple(
    sorted(
        (
            ResourceDomain.GPU_LOCAL.value,
            ResourceDomain.NVLINK_P2P.value,
            ResourceDomain.PCIE_HOST.value,
            ResourceDomain.HOST_NUMA.value,
            ResourceDomain.NIC_FABRIC.value,
            ResourceDomain.SLINGSHOT_FABRIC.value,
        )
    )
)


def _domain_footprints(runs: Sequence[TierRun]) -> dict[str, dict[str, list[str]]]:
    footprints: dict[str, dict[str, list[str]]] = {}
    for run in runs:
        auxiliary = tuple(domain.value for domain in mode_spec(run.tier_mode).auxiliary_domains)
        shared = sorted(set(G1_FOREGROUND_DOMAINS).intersection(auxiliary))
        footprints[run.mode] = {
            "foreground_domains": list(G1_FOREGROUND_DOMAINS),
            "auxiliary_domains": list(auxiliary),
            "shared_domains": shared,
        }
    return footprints


def validate_runner_manifest(manifest: dict[str, object]) -> None:
    """Fail closed on the non-submitting G1 manifest itself.

    ``validate_attribution_manifest`` validates the older generic schema.  G1
    uses a richer runner schema, so it needs its own exact validator; otherwise
    a caller could accidentally edit a design manifest (for example changing
    the endpoint or adding a mode) without the matrix contract noticing.
    """

    if type(manifest) is not dict or set(manifest) != _RUNNER_MANIFEST_KEYS:
        raise ValueError("G1 runner manifest keys are not exact")
    if manifest["schema_version"] != "tempo-rd-tier-attribution-runner-1":
        raise ValueError("unsupported G1 runner manifest schema")
    if manifest["world_size"] != 4 or manifest["nodes"] != 1:
        raise ValueError("G1 runner must be exactly one node and four ranks")
    for key in ("state_bytes_per_rank", "deadline_ns"):
        value = manifest[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive int")
    steps = manifest["checkpoint_steps"]
    if type(steps) is not list or not steps or any(type(step) is not int for step in steps):
        raise ValueError("checkpoint_steps must be a non-empty integer list")
    if steps != sorted(set(steps)):
        raise ValueError("checkpoint_steps must be sorted and unique")
    if manifest["evidence_state"] != "design_only":
        raise ValueError("G1 runner manifest must remain design_only")
    if manifest["slurm_submitted"] is not False:
        raise ValueError("G1 runner must never submit Slurm work")
    if manifest["inference_adapter"] != "not_implemented_in_g1":
        raise ValueError("G1 inference adapter marker is not exact")
    if manifest["observation_window_contract"] != observation_window_contract():
        raise ValueError("G1 observation-window contract is not exact")

    runs = manifest["runs"]
    if type(runs) is not list or len(runs) != len(EXECUTABLE_G1_MODES):
        raise ValueError("G1 runs must contain the exact executable mode set")
    expected_runs = [asdict(run) for run in build_g1_matrix()]
    if runs != expected_runs:
        raise ValueError("G1 runs do not match the frozen executable matrix")

    footprints = manifest["domain_footprints"]
    expected_footprints = _domain_footprints(build_g1_matrix())
    if footprints != expected_footprints:
        raise ValueError("G1 domain footprints do not match the frozen route contract")

    optional_modes = manifest["optional_modes"]
    if type(optional_modes) is not list:
        raise ValueError("G1 optional_modes must be a list")
    expected_optional = {
        "p2p_only": "NVLink/P2P is enabled only after the observed path is proven",
        "host_pressure": "host-pressure placebo requires tempo.host_pressure and a live NUMA counter; path is not traversed",
    }
    if len(optional_modes) != len(expected_optional):
        raise ValueError("G1 optional_modes are not exact")
    seen_optional: set[str] = set()
    for item in optional_modes:
        if type(item) is not dict or set(item) != {"mode", "path_status", "reason"}:
            raise ValueError("G1 optional mode keys are not exact")
        mode = item["mode"]
        if mode not in expected_optional or mode in seen_optional:
            raise ValueError("G1 optional mode set is not exact")
        if item["path_status"] != "not_traversed" or item["reason"] != expected_optional[mode]:
            raise ValueError("G1 optional modes must remain not_traversed placeholders")
        seen_optional.add(mode)
    if seen_optional != set(expected_optional):
        raise ValueError("G1 optional mode set is not exact")

    required_domains = manifest["required_domains"]
    expected_domains = sorted(
        domain.value for domain in required_domains_for_modes(run.mode for run in build_g1_matrix())
    )
    if required_domains != expected_domains:
        raise ValueError("G1 required_domains do not match the executable matrix")

    contract = manifest["evidence_contract"]
    expected_contract = {
        "counter_support_values": sorted(item.value for item in CounterSupport),
        "path_status_values": sorted(item.value for item in PathStatus),
        "causal_requires": [
            "interventional",
            "observed_path",
            "supported_counters",
            "tail_delta_above_uncertainty",
        ],
    }
    if contract != expected_contract:
        raise ValueError("G1 evidence_contract is not exact")

    evidence_records = manifest["evidence_records"]
    if type(evidence_records) is not list:
        raise ValueError("G1 evidence_records must be a list")
    expected_count = sum(
        len(mode_spec(run.mode).auxiliary_domains) for run in build_g1_matrix()
    )
    if len(evidence_records) != expected_count:
        raise ValueError("G1 evidence_records do not cover each declared path")
    grouped: dict[str, list[DomainEvidence]] = {}
    for raw in evidence_records:
        if type(raw) is not dict or set(raw) != {
            "domain", "mode", "foreground_kind", "auxiliary_kind",
            "overlapping_bytes", "overlap_ns", "tail_delta_ns", "evidence",
            "counter_support", "path_status", "uncertainty_ns", "source",
            "path_evidence", "counter_family",
        }:
            raise ValueError("G1 evidence record keys are not exact")
        try:
            record = DomainEvidence(
                domain=ResourceDomain(raw["domain"]),
                mode=raw["mode"],
                foreground_kind=raw["foreground_kind"],
                auxiliary_kind=raw["auxiliary_kind"],
                overlapping_bytes=raw["overlapping_bytes"],
                overlap_ns=raw["overlap_ns"],
                tail_delta_ns=raw["tail_delta_ns"],
                evidence=EvidenceLevel(raw["evidence"]),
                counter_support=CounterSupport(raw["counter_support"]),
                path_status=PathStatus(raw["path_status"]),
                uncertainty_ns=raw["uncertainty_ns"],
                source=raw["source"],
                path_evidence=raw["path_evidence"],
                counter_family=raw["counter_family"],
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"invalid G1 evidence record: {exc}") from exc
        if (
            record.evidence is not EvidenceLevel.UNSUPPORTED
            or record.counter_support is not CounterSupport.NOT_COLLECTED
            or record.path_status is not PathStatus.DECLARED
            or record.overlapping_bytes != 0
            or record.overlap_ns != 0
            or record.tail_delta_ns != 0
            or record.uncertainty_ns != 0
            or record.source != "g1_design_manifest"
        ):
            raise ValueError("G1 design evidence records must remain explicit placeholders")
        grouped.setdefault(record.mode, []).append(record)
    for run in build_g1_matrix():
        if run.mode == "fg_only":
            if grouped.get(run.mode):
                raise ValueError("foreground-only must have no auxiliary evidence records")
            continue
        validate_mode_evidence(run.mode, grouped.get(run.mode, []))


def build_g1_matrix() -> tuple[TierRun, ...]:
    """Return the executable one-node matrix without touching Slurm."""

    runs: list[TierRun] = []
    for mode in EXECUTABLE_G1_MODES:
        spec = mode_spec(mode)
        if mode is AttributionMode.FOREGROUND_ONLY:
            runs.append(
                TierRun(
                    mode=mode.value,
                    policy="none",
                    tier_mode=mode.value,
                    endpoint="none",
                    requires_restore=False,
                    requires_gpu_transfer=False,
                )
            )
            continue
        endpoint = "node_local_sink" if mode is AttributionMode.D2H_ONLY else "persistent_endpoint"
        runs.append(
            TierRun(
                mode=mode.value,
                policy="datastates",
                tier_mode=mode.value,
                endpoint=endpoint,
                requires_restore=spec.requires_checkpoint_endpoint,
                requires_gpu_transfer=spec.requires_gpu_transfer,
            )
        )
    return tuple(runs)


def validate_matrix(runs: Sequence[TierRun]) -> None:
    if tuple(run.mode for run in runs) != tuple(mode.value for mode in EXECUTABLE_G1_MODES):
        raise ValueError("G1 matrix must contain each executable mode exactly once in order")
    if any(run.policy == "none" and run.tier_mode != "fg_only" for run in runs):
        raise ValueError("foreground-only must use policy=none")
    for run in runs:
        if run.policy != "none":
            mode_spec(run.tier_mode)
        if run.mode == "d2h_only" and run.endpoint != "node_local_sink":
            raise ValueError("d2h_only must terminate at a node-local sink")
        if run.mode != "d2h_only" and run.mode != "fg_only" and run.endpoint != "persistent_endpoint":
            raise ValueError("persistent modes must use the declared persistent endpoint")


def build_declared_evidence_records(runs: Sequence[TierRun]) -> tuple[DomainEvidence, ...]:
    """Emit explicit placeholders; these are not live causal observations."""

    records: list[DomainEvidence] = []
    for run in runs:
        for domain in mode_spec(run.tier_mode).auxiliary_domains:
            records.append(
                DomainEvidence(
                    domain=domain,
                    mode=run.tier_mode,
                    foreground_kind="fsdp_collective",
                    auxiliary_kind="checkpoint_flow",
                    overlapping_bytes=0,
                    overlap_ns=0,
                    tail_delta_ns=0,
                    evidence=EvidenceLevel.UNSUPPORTED,
                    counter_support=CounterSupport.NOT_COLLECTED,
                    path_status=PathStatus.DECLARED,
                    uncertainty_ns=0,
                    source="g1_design_manifest",
                    path_evidence=domain_contract(domain).path_evidence,
                    counter_family=domain_contract(domain).counter_family,
                )
            )
    return tuple(records)


def serialize_evidence_record(record: DomainEvidence) -> dict[str, object]:
    return {
        "domain": record.domain.value,
        "mode": record.mode,
        "foreground_kind": record.foreground_kind,
        "auxiliary_kind": record.auxiliary_kind,
        "overlapping_bytes": record.overlapping_bytes,
        "overlap_ns": record.overlap_ns,
        "tail_delta_ns": record.tail_delta_ns,
        "evidence": record.evidence.value,
        "counter_support": record.counter_support.value,
        "path_status": record.path_status.value,
        "uncertainty_ns": record.uncertainty_ns,
        "source": record.source,
        "path_evidence": record.path_evidence,
        "counter_family": record.counter_family,
    }


def build_manifest(*, world_size: int, nodes: int, state_bytes_per_rank: int, deadline_ns: int,
                   checkpoint_steps: Sequence[int]) -> dict[str, object]:
    runs = build_g1_matrix()
    validate_matrix(runs)
    evidence_records = build_declared_evidence_records(runs)
    for run in runs:
        validate_mode_evidence(
            run.tier_mode,
            [record for record in evidence_records if record.mode == run.tier_mode],
        )
    if world_size != 4 or nodes != 1:
        raise ValueError("G1 attribution is deliberately limited to one node and four ranks")
    if type(state_bytes_per_rank) is not int or state_bytes_per_rank <= 0:
        raise ValueError("state_bytes_per_rank must be a positive integer")
    if type(deadline_ns) is not int or deadline_ns <= 0:
        raise ValueError("deadline_ns must be a positive integer")
    steps = list(checkpoint_steps)
    if not steps or steps != sorted(set(steps)):
        raise ValueError("checkpoint_steps must be sorted and unique")
    manifest = {
        "schema_version": "tempo-rd-tier-attribution-runner-1",
        "world_size": world_size,
        "nodes": nodes,
        "state_bytes_per_rank": state_bytes_per_rank,
        "deadline_ns": deadline_ns,
        "checkpoint_steps": steps,
        "runs": [asdict(run) for run in runs],
        "domain_footprints": _domain_footprints(runs),
        "optional_modes": [
            {
                "mode": "p2p_only",
                "path_status": "not_traversed",
                "reason": "NVLink/P2P is enabled only after the observed path is proven",
            },
            {
                "mode": "host_pressure",
                "path_status": "not_traversed",
                "reason": "host-pressure placebo requires tempo.host_pressure and a live NUMA counter; path is not traversed",
            },
        ],
        "evidence_state": "design_only",
        "evidence_records": [serialize_evidence_record(record) for record in evidence_records],
        "required_domains": sorted(
            domain.value for domain in required_domains_for_modes(run.mode for run in runs)
        ),
        "evidence_contract": {
            "counter_support_values": sorted(
                item.value for item in CounterSupport
            ),
            "path_status_values": sorted(
                item.value for item in PathStatus
            ),
            "causal_requires": [
                "interventional",
                "observed_path",
                "supported_counters",
                "tail_delta_above_uncertainty",
            ],
        },
        "slurm_submitted": False,
        "inference_adapter": "not_implemented_in_g1",
        "observation_window_contract": observation_window_contract(),
    }
    validate_runner_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--command-plan-output",
        type=Path,
        help="write the non-submitting five-mode argv/env plan instead of a design manifest",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--result-root", type=Path, default=Path("/tmp/tempo-rd-g1-results"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/tmp/tempo-rd-g1-checkpoints"))
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--state-bytes-per-rank", type=int, default=402_705_672)
    parser.add_argument("--deadline-ns", type=int, default=1_000_000_000)
    parser.add_argument("--checkpoint-steps", default="16,52")
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--warmup-steps", type=int, default=12)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ffn-size", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--window-steps", type=int, default=16)
    parser.add_argument("--probe-mb", type=int, default=64)
    parser.add_argument("--deadline-seconds", type=float, default=1.0)
    parser.add_argument("--datastates-cache-gb", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()
    steps = [int(item) for item in args.checkpoint_steps.split(",") if item]
    if args.command_plan_output:
        commands = build_g1_command_plan(
            repo_root=args.repo_root,
            result_root=args.result_root,
            checkpoint_root=args.checkpoint_root,
            checkpoint_steps=steps,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            world_size=args.world_size,
            layers=args.layers,
            hidden_size=args.hidden_size,
            ffn_size=args.ffn_size,
            heads=args.heads,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            window_steps=args.window_steps,
            probe_mb=args.probe_mb,
            deadline_seconds=args.deadline_seconds,
            datastates_cache_gb=args.datastates_cache_gb,
            seed=args.seed,
        )
        command_plan = {
            "schema_version": "tempo-rd-g1-command-plan-1",
            "submitting": False,
            "world_size": args.world_size,
            "commands": [asdict(command) for command in commands],
        }
        encoded = json.dumps(command_plan, indent=2, sort_keys=True) + "\n"
        args.command_plan_output.write_text(encoded, encoding="utf-8")
        return
    manifest = build_manifest(
        world_size=args.world_size,
        nodes=args.nodes,
        state_bytes_per_rank=args.state_bytes_per_rank,
        deadline_ns=args.deadline_ns,
        checkpoint_steps=steps,
    )
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
