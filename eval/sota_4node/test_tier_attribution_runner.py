from __future__ import annotations

import unittest
from pathlib import Path

from eval.sota_4node.tier_attribution_runner import (
    build_g1_command_plan,
    build_declared_evidence_records,
    build_g1_matrix,
    build_manifest,
    validate_matrix,
    validate_runner_manifest,
)


class TierAttributionRunnerTests(unittest.TestCase):
    def test_command_plan_is_five_mode_and_non_submitting(self) -> None:
        commands = build_g1_command_plan(
            repo_root=Path("."),
            result_root=Path("/tmp/tempo-rd-results"),
            checkpoint_root=Path("/tmp/tempo-rd-checkpoints"),
        )
        self.assertEqual(
            [command.mode for command in commands],
            ["fg_only", "open_combined", "d2h_only", "persist_only", "combined"],
        )
        self.assertTrue(all(command.argv[0] == "python" for command in commands))
        self.assertTrue(all("sbatch" not in command.argv and "srun" not in command.argv for command in commands))
        self.assertEqual(commands[0].argv[commands[0].argv.index("--layers") + 1], "2")
        self.assertEqual(commands[0].argv[commands[0].argv.index("--hidden-size") + 1], "2048")
        self.assertIn("TEMPO_RD_LOCAL_SINK_ROOT", dict(commands[2].env))
        self.assertFalse(commands[0].requires_restore)
        self.assertTrue(commands[3].requires_restore)
        for flag in ("--window-steps", "--probe-mb", "--deadline-seconds", "--datastates-cache-gb", "--seed"):
            self.assertIn(flag, commands[0].argv)

    def test_command_plan_rejects_non_integer_or_short_schedule(self) -> None:
        kwargs = {
            "repo_root": Path("."),
            "result_root": Path("/tmp/results"),
            "checkpoint_root": Path("/tmp/checkpoints"),
        }
        with self.assertRaisesRegex(ValueError, "only integers"):
            build_g1_command_plan(**kwargs, checkpoint_steps=[16.0])
        with self.assertRaisesRegex(ValueError, "does not fit"):
            build_g1_command_plan(**kwargs, checkpoint_steps=[60], steps=72)
    def test_matrix_is_cpu_only_and_has_exact_executable_modes(self) -> None:
        runs = build_g1_matrix()
        validate_matrix(runs)
        self.assertEqual(
            [run.mode for run in runs],
            ["fg_only", "open_combined", "d2h_only", "persist_only", "combined"],
        )
        self.assertEqual(runs[0].policy, "none")
        self.assertEqual(runs[1].policy, "datastates")
        self.assertEqual(runs[2].endpoint, "node_local_sink")

    def test_manifest_is_explicitly_non_submitting(self) -> None:
        manifest = build_manifest(
            world_size=4,
            nodes=1,
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16, 52],
        )
        self.assertFalse(manifest["slurm_submitted"])
        script = Path(__file__).with_name("run_g1_tier_1node.slurm").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --mail-type=NONE", script)
        self.assertIn("trap 'exit 143' USR1", script)

        self.assertIn("TEMPO_RD_APPROVE_G1:-", script)
        self.assertIn("TEMPO_RD_G1_EXPECTED_SOURCE_BUNDLE_SHA256:-", script)
        self.assertIn("build_g1_causal_readiness.py", script)
        self.assertIn("build_g1_causal_readiness_executed.py", script)
        self.assertIn('"causal_readiness_builder"', script)
        self.assertIn("causal_readiness.json", script)
        self.assertEqual(manifest["inference_adapter"], "not_implemented_in_g1")
        self.assertEqual(len(manifest["runs"]), 5)
        self.assertEqual(
            manifest["domain_footprints"]["d2h_only"]["shared_domains"],
            ["gpu_local", "host_numa", "pcie_host"],
        )
        self.assertEqual(manifest["domain_footprints"]["fg_only"]["shared_domains"], [])
        self.assertEqual(
            [(item["mode"], item["path_status"]) for item in manifest["optional_modes"]],
            [("p2p_only", "not_traversed"), ("host_pressure", "not_traversed")],
        )
        self.assertEqual(manifest["evidence_state"], "design_only")
        self.assertEqual(len(manifest["evidence_records"]), len(manifest["required_domains"]) * 3)
        self.assertTrue(all(record["path_status"] == "declared" for record in manifest["evidence_records"]))
        self.assertIn("pcie_host", manifest["required_domains"])
        self.assertEqual(
            manifest["evidence_contract"]["causal_requires"],
            ["interventional", "observed_path", "supported_counters", "tail_delta_above_uncertainty"],
        )

    def test_g1_runner_binds_lustre_rpc_diagnostic_capture(self) -> None:
        script = Path(__file__).with_name("run_g1_tier_1node.slurm").read_text(encoding="utf-8")
        self.assertIn("capture_lustre_rpc_observation.py", script)
        self.assertIn("domain_observations/${mode}/lustre_rpc", script)
        self.assertIn('"capture_lustre_rpc_observation_executed.py"', script)

    def test_manifest_rejects_non_g1_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "one node"):
            build_manifest(
                world_size=8,
                nodes=2,
                state_bytes_per_rank=402_705_672,
                deadline_ns=1_000_000_000,
                checkpoint_steps=[16],
            )

    def test_declared_evidence_is_not_a_causal_candidate(self) -> None:
        records = build_declared_evidence_records(build_g1_matrix())
        self.assertTrue(records)
        self.assertTrue(all(record.path_status.value == "declared" for record in records))
        self.assertTrue(all(record.counter_support.value == "not_collected" for record in records))

    def test_manifest_build_validates_every_mode_coverage(self) -> None:
        manifest = build_manifest(
            world_size=4,
            nodes=1,
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16],
        )
        by_mode = {}
        for record in manifest["evidence_records"]:
            by_mode.setdefault(record["mode"], []).append(record)
        self.assertEqual(set(by_mode), {"open_combined", "d2h_only", "persist_only", "combined"})

    def test_runner_validator_rejects_endpoint_or_mode_edits(self) -> None:
        candidate = build_manifest(
            world_size=4,
            nodes=1,
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16],
        )
        candidate["runs"] = [dict(item) for item in candidate["runs"]]
        candidate["runs"][1]["endpoint"] = "node_local_sink"
        with self.assertRaisesRegex(ValueError, "frozen executable matrix"):
            validate_runner_manifest(candidate)

        candidate = build_manifest(
            world_size=4,
            nodes=1,
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16],
        )
        candidate["domain_footprints"] = {
            mode: dict(value) for mode, value in candidate["domain_footprints"].items()
        }
        candidate["domain_footprints"]["d2h_only"]["shared_domains"] = []
        with self.assertRaisesRegex(ValueError, "domain footprints"):
            validate_runner_manifest(candidate)

    def test_runner_validator_rejects_declared_record_coercion(self) -> None:
        candidate = build_manifest(
            world_size=4,
            nodes=1,
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16],
        )
        candidate["evidence_records"] = [dict(item) for item in candidate["evidence_records"]]
        candidate["evidence_records"][0]["overlap_ns"] = 0.0
        with self.assertRaisesRegex(ValueError, "invalid G1 evidence record"):
            validate_runner_manifest(candidate)

    def test_runner_validator_rejects_submission_or_live_evidence(self) -> None:
        candidate = build_manifest(
            world_size=4,
            nodes=1,
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16],
        )
        candidate["slurm_submitted"] = True
        with self.assertRaisesRegex(ValueError, "never submit"):
            validate_runner_manifest(candidate)

    def test_runner_validator_rejects_placeholder_relabeling(self) -> None:
        candidate = build_manifest(
            world_size=4,
            nodes=1,
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16],
        )
        candidate["evidence_records"] = [dict(item) for item in candidate["evidence_records"]]
        candidate["evidence_records"][0]["path_status"] = "observed"
        with self.assertRaisesRegex(ValueError, "explicit placeholders"):
            validate_runner_manifest(candidate)

    def test_runner_validator_rejects_unobserved_optional_mode_as_live(self) -> None:
        candidate = build_manifest(
            world_size=4,
            nodes=1,
            state_bytes_per_rank=402_705_672,
            deadline_ns=1_000_000_000,
            checkpoint_steps=[16],
        )
        candidate["optional_modes"] = [dict(item) for item in candidate["optional_modes"]]
        candidate["optional_modes"][0]["path_status"] = "observed"
        with self.assertRaisesRegex(ValueError, "not_traversed placeholders"):
            validate_runner_manifest(candidate)


if __name__ == "__main__":
    unittest.main()
