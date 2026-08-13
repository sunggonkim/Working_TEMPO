from __future__ import annotations

import hashlib
import json
import copy
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.validate_g1_tier_raw import MODES, validate_g1_tier_raw
from eval.sota_4node.host_pressure_placebo import _record_digest


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _make_fixture(root: Path) -> None:
    train = root / "train_executed.py"
    runner = root / "tier_attribution_runner_executed.py"
    validator = root / "validate_g1_tier_raw_executed.py"
    causal_validator = root / "validate_g1_result_executed.py"
    composer = root / "compose_g1_result_executed.py"
    readiness_builder = root / "build_g1_causal_readiness_executed.py"
    pressure_wrapper = root / "host_pressure_train_wrapper_executed.py"
    pressure_helper = root / "host_pressure_placebo.py"
    collector = root / "capture_g1_domain_counters_executed.py"
    foreground_publisher = root / "prepare_foreground_path_executed.py"
    pcie_probe = root / "capture_nvml_pcie_observation_executed.py"
    train.write_text("# frozen train\n", encoding="utf-8")
    runner.write_text("# frozen runner\n", encoding="utf-8")
    validator.write_text("# frozen validator\n", encoding="utf-8")
    causal_validator.write_text("# frozen causal validator\n", encoding="utf-8")
    composer.write_text("# frozen composer\n", encoding="utf-8")
    readiness_builder.write_text("# frozen readiness builder\n", encoding="utf-8")
    pressure_wrapper.write_text("# frozen pressure wrapper\n", encoding="utf-8")
    pressure_helper.write_text("# frozen pressure helper\n", encoding="utf-8")
    collector.write_text("# frozen counter collector\n", encoding="utf-8")
    foreground_publisher.write_text("# frozen foreground publisher\n", encoding="utf-8")
    pcie_probe.write_text("# frozen NVML/PCIe probe\n", encoding="utf-8")
    sources = {
        "train_executed.py": hashlib.sha256(train.read_bytes()).hexdigest(),
        "tier_attribution_runner_executed.py": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "validate_g1_tier_raw_executed.py": hashlib.sha256(validator.read_bytes()).hexdigest(),
        "validate_g1_result_executed.py": hashlib.sha256(causal_validator.read_bytes()).hexdigest(),
        "compose_g1_result_executed.py": hashlib.sha256(composer.read_bytes()).hexdigest(),
        "build_g1_causal_readiness_executed.py": hashlib.sha256(readiness_builder.read_bytes()).hexdigest(),
        "host_pressure_train_wrapper_executed.py": hashlib.sha256(pressure_wrapper.read_bytes()).hexdigest(),
        "host_pressure_placebo.py": hashlib.sha256(pressure_helper.read_bytes()).hexdigest(),
        "capture_g1_domain_counters_executed.py": hashlib.sha256(collector.read_bytes()).hexdigest(),
        "prepare_foreground_path_executed.py": hashlib.sha256(foreground_publisher.read_bytes()).hexdigest(),
        "capture_nvml_pcie_observation_executed.py": hashlib.sha256(pcie_probe.read_bytes()).hexdigest(),
    }
    manifest = {
        "schema_version": "tempo-rd-g1-tier-runtime-1",
        "stage": "g1_tier",
        "nodes": 1,
        "world_size": 4,
        "state_bytes_per_rank": 402714696,
        "logical_file_extent_bytes": 402890752,
        "checkpoint_steps": [16, 52],
        "steps": 72,
        "warmup_steps": 12,
        "mode_outer_seconds": 35,
        "restore_outer_seconds": 20,
        "phase_budget_seconds": 235,
        "cleanup_reserve_seconds": 65,
        "geometry": {"layers": 2, "hidden_size": 2048, "ffn_size": 8192, "heads": 16, "sequence_length": 64, "batch_size": 1},
        "host_pressure_placebo": {
            "mode": "fg_only",
            "buffer_bytes": 64 * 1024 * 1024,
            "source": "proc_self_numa_maps_plus_touch_loop",
            "rank_files": [f"fg_only/host_pressure_rank_{rank}.json" for rank in range(4)],
        },
        "modes": list(MODES),
        "source_sha256": sources,
        "slurm_submitted": True,
        "no_retry": True,
    }
    _write_json(root / "g1_tier_runtime_manifest.json", manifest)
    _write_json(root / "g1_command_plan.json", {
        "schema_version": "tempo-rd-g1-command-plan-1",
        "submitting": False,
        "world_size": 4,
        "commands": [{"mode": mode} for mode in MODES],
    })
    (root / "execution_status.env").write_text("status=raw_complete\n", encoding="utf-8")
    (root / "placement.txt").write_text(
        "".join(f"rank={rank} local_rank={rank} host=nid numa={rank} cuda={rank}\n" for rank in range(4)),
        encoding="utf-8",
    )
    for mode in MODES:
        mode_dir = root / mode
        mode_dir.mkdir()
        policy = "none" if mode == "fg_only" else "datastates"
        for rank in range(4):
            _write_json(mode_dir / f"summary_rank{rank}.json", {
                "rank": rank, "world_size": 4, "policy": policy,
                "tier_mode": mode, "source_sha256": sources["train_executed.py"],
                "tier_endpoint": "" if mode == "fg_only" else (
                    "node_local_sink" if mode == "d2h_only" else "persistent_endpoint"
                ),
                "tier_host_preloaded": mode == "persist_only",
                "tier_gpu_transfer": mode in {"d2h_only", "open_combined", "combined"},
            })
            (mode_dir / f"steps_rank{rank}.csv").write_text("rank,step\n0,12\n", encoding="utf-8")
            (mode_dir / f"collectives_rank{rank}.csv").write_text("rank,step\n0,12\n", encoding="utf-8")
        if mode == "fg_only":
            for rank in range(4):
                pressure_record = {
                    "schema_version": "tempo-rd-host-pressure-run-1",
                    "spec": {
                        "rank": rank,
                        "world_size": 4,
                        "numa_node": rank,
                        "buffer_bytes": 64 * 1024 * 1024,
                        "duration_ns": 100,
                        "sample_period_ns": 10,
                        "source": "proc_self_numa_maps_plus_touch_loop",
                    },
                    "samples": [
                        {"sample_id": "start", "timestamp_ns": 1, "cumulative_touched_bytes": 0, "cumulative_busy_ns": 0, "numa_node_bytes": 0},
                        {"sample_id": "finish", "timestamp_ns": 2, "cumulative_touched_bytes": 64 * 1024 * 1024, "cumulative_busy_ns": 1, "numa_node_bytes": 4096},
                    ],
                    "output_sha256": "",
                }
                pressure_record["output_sha256"] = _record_digest(pressure_record)
                _write_json(mode_dir / f"host_pressure_rank_{rank}.json", pressure_record)
            continue
        stage = {
            "d2h": {
                "total_bytes": 1024,
                "queued_bytes": 0,
                "ready_bytes": 0,
                "admitted_bytes": 1024,
                "completed_bytes": 1024,
                "inflight_bytes": 0,
                "inflight_requests": 0,
                "admitted_requests": 1,
                "max_request_bytes": 1024,
                "peak_inflight_bytes": 1024,
                "peak_inflight_requests": 1,
                "last_progress_monotonic_ns": 20,
                "last_completion_monotonic_ns": 20,
            },
            "pfs": {
                "total_bytes": 2048,
                "queued_bytes": 0,
                "ready_bytes": 0,
                "admitted_bytes": 2048,
                "completed_bytes": 2048,
                "inflight_bytes": 0,
                "inflight_requests": 0,
                "admitted_requests": 1,
                "max_request_bytes": 2048,
                "peak_inflight_bytes": 2048,
                "peak_inflight_requests": 1,
                "last_progress_monotonic_ns": 30,
                "last_completion_monotonic_ns": 30,
            },
        }
        for rank in range(4):
            stage_start = copy.deepcopy(stage)
            stage_end = copy.deepcopy(stage)
            metrics = {
                "tier_stage_stats_start": stage_start,
                "tier_stage_stats_end": stage_end,
                "state_bytes_local": manifest["state_bytes_per_rank"],
                "checkpoint_path": f"/tmp/step-52/rank_{rank:05d}.ds",
                "durable_ms": 10.0,
                "deadline_met": True,
            }
            if mode in {"open_combined", "persist_only", "combined"}:
                metrics["tier_stage_stats_start"].update({
                    "pfs_fsync_complete": False,
                    "pfs_fsync_monotonic_ns": 0,
                    "pfs_odirect_verified": False,
                })
                metrics["tier_stage_stats_end"].update({
                    "pfs_fsync_complete": True,
                    "pfs_fsync_monotonic_ns": 40,
                    "pfs_odirect_verified": True,
                })
                metrics.update({
                    "checkpoint_file_bytes": manifest["logical_file_extent_bytes"],
                    "checkpoint_allocated_bytes": manifest["logical_file_extent_bytes"],
                    "logical_file_extent_bytes": manifest["logical_file_extent_bytes"],
                    "commit_marker_path": "/tmp/step-52/GLOBAL_COMMIT.json",
                    "commit_manifest_sha256": "a" * 64,
                    "commit_validated": True,
                    "fsync_evidence_valid": True,
                })
            else:
                metrics.update({
                    "checkpoint_file_bytes": 0,
                    "checkpoint_allocated_bytes": 0,
                    "logical_file_extent_bytes": 0,
                    "commit_validated": False,
                    "fsync_evidence_valid": False,
                })
            events = []
            for step in (16, 52):
                event = dict(metrics)
                event["checkpoint_path"] = f"/tmp/step-{step}/rank_{rank:05d}.ds"
                events.append(event)
            _write_json(mode_dir / f"checkpoint_events_rank{rank}.json", events)
            _write_json(mode_dir / f"checkpoint_rank{rank}.json", {
                **metrics,
            })
            if mode in {"open_combined", "persist_only", "combined"}:
                _write_json(mode_dir / f"fresh_restore_rank{rank}.json", {"rank": rank, "passed": True})


class G1TierRawValidatorTests(unittest.TestCase):
    def test_complete_raw_structure_passes_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            result = validate_g1_tier_raw(root)
            self.assertEqual(result["status"], "pass")
            self.assertFalse(result["promotion_ready"])
            self.assertTrue(result["live_external_execution"])

    def test_interleaved_placement_output_is_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            lines = (root / "placement.txt").read_text(encoding="utf-8").splitlines()
            (root / "placement.txt").write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
            result = validate_g1_tier_raw(root)
            self.assertEqual(result["status"], "pass")

    def test_missing_stage_snapshot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            metrics = json.loads((root / "combined/checkpoint_rank0.json").read_text(encoding="utf-8"))
            del metrics["tier_stage_stats_end"]
            _write_json(root / "combined/checkpoint_rank0.json", metrics)
            with self.assertRaisesRegex(ValueError, "engine snapshots must be objects"):
                validate_g1_tier_raw(root)

    def test_validator_snapshot_is_source_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            (root / "validate_g1_tier_raw_executed.py").unlink()
            with self.assertRaisesRegex(ValueError, "source hash mismatch: validate_g1_tier_raw_executed.py"):
                validate_g1_tier_raw(root)

    def test_host_pressure_digest_is_bound_to_record_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            record_path = root / "fg_only/host_pressure_rank_0.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["samples"][1]["numa_node_bytes"] += 4096
            _write_json(record_path, record)
            with self.assertRaisesRegex(ValueError, "output digest mismatch"):
                validate_g1_tier_raw(root)

    def test_stage_counter_regression_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            metrics = json.loads((root / "combined/checkpoint_rank0.json").read_text(encoding="utf-8"))
            metrics["tier_stage_stats_end"]["d2h"]["completed_bytes"] = 0
            _write_json(root / "combined/checkpoint_rank0.json", metrics)
            with self.assertRaisesRegex(ValueError, "completed_bytes regresses"):
                validate_g1_tier_raw(root)

    def test_logical_stage_timing_is_optional_but_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            path = root / "combined/checkpoint_rank0.json"
            metrics = json.loads(path.read_text(encoding="utf-8"))
            logical = {
                "schema": "tempo-rd-logical-stage-timing-1",
                "counter_semantics": "logical_bytes_and_wait_interval",
                "hardware_counter": False,
                "d2h": {**metrics["tier_stage_stats_start"]["d2h"], "busy_ns": 11},
                "pfs": {**metrics["tier_stage_stats_start"]["pfs"], "busy_ns": 22},
            }
            metrics["tier_stage_stats_start"]["logical_stage"] = logical
            metrics["tier_stage_stats_end"]["logical_stage"] = logical
            _write_json(path, metrics)
            validate_g1_tier_raw(root)

            logical["hardware_counter"] = True
            metrics["tier_stage_stats_end"]["logical_stage"] = logical
            _write_json(path, metrics)
            with self.assertRaisesRegex(ValueError, "must not self-attest hardware"):
                validate_g1_tier_raw(root)

    def test_persistent_endpoint_requires_odirect_and_fsync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            metrics = json.loads((root / "combined/checkpoint_rank0.json").read_text(encoding="utf-8"))
            del metrics["tier_stage_stats_end"]["pfs_odirect_verified"]
            _write_json(root / "combined/checkpoint_rank0.json", metrics)
            with self.assertRaisesRegex(ValueError, "O_DIRECT evidence"):
                validate_g1_tier_raw(root)

    def test_checkpoint_metrics_bind_state_and_extent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            metrics = json.loads((root / "combined/checkpoint_rank0.json").read_text(encoding="utf-8"))
            metrics["state_bytes_local"] -= 1
            _write_json(root / "combined/checkpoint_rank0.json", metrics)
            with self.assertRaisesRegex(ValueError, "state_bytes_local"):
                validate_g1_tier_raw(root)

    def test_checkpoint_physical_extent_must_match_logical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            metrics = json.loads((root / "combined/checkpoint_rank0.json").read_text(encoding="utf-8"))
            metrics["checkpoint_file_bytes"] -= 4096
            _write_json(root / "combined/checkpoint_rank0.json", metrics)
            with self.assertRaisesRegex(ValueError, "physical checkpoint extent"):
                validate_g1_tier_raw(root)

    def test_restore_is_required_for_persistent_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            (root / "persist_only/fresh_restore_rank0.json").unlink()
            with self.assertRaisesRegex(ValueError, "persistent mode lacks"):
                validate_g1_tier_raw(root)


if __name__ == "__main__":
    unittest.main()
