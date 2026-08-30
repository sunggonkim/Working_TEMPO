from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import analyze_tempo_go_c7_lmcache_victim_abba as analyzer


CONTRACT = Path(__file__).with_name("tempo_go_c7_lmcache_victim_contract_v1.json")


class C7LMCacheVictimABBAAnalysisTests(unittest.TestCase):
    def _write_arm(
        self,
        root: Path,
        *,
        name: str,
        load: str,
        completion_ms: float,
    ) -> None:
        arm = root / name
        arm.mkdir()
        token_iters = 1 if load == "control" else 8192
        blocks = 16
        expected_bytes = 4 * 8 * 128 * 1024 * 1024
        result = {
            "schema_version": analyzer.RESULT_SCHEMA,
            "evidence_state": "live_official_component",
            "baseline": {"backend": "NIXL UCX", "proxy": False},
            "world_size": 8,
            "nodes": 2,
            "pair_count": 4,
            "config": {
                "requests": 8,
                "kv_bytes": 128 * 1024 * 1024,
                "token_iters": token_iters,
                "blocks": blocks,
                "foreground_bytes": 32 * 1024 * 1024,
                "block_delay_s": 0,
                "minimum_active_duration_s": 30,
                "maximum_blocks": 17,
                "process_group_timeout_s": 120,
                "nixl_transfer_timeout_s": 120,
                "background_mode": "nixl_ucx",
                "traffic_pattern": "paired_1to1",
            },
            "blocks": [
                {
                    "block_index": index,
                    "correctness_met": True,
                    "expected_background_bytes": expected_bytes,
                    "source_completed_bytes": expected_bytes,
                    "receiver_verified_bytes": expected_bytes,
                    "full_bytes_completed": True,
                    "full_bytes_verified": True,
                    "background_completion_ms": completion_ms,
                }
                for index in range(blocks)
            ],
            "active_loop": {
                "rank_min_elapsed_ms": 31_000.0,
                "rank_max_elapsed_ms": 31_100.0,
                "minimum_requested_ms": 30_000.0,
                "horizon_met": True,
            },
            "summary": {
                "global_token_tail_p50_ms": 1.0,
                "global_token_tail_p99_ms": 2.0,
                "background_completion_p50_ms": completion_ms,
                "background_completion_p99_ms": completion_ms,
            },
            "rank_diagnostics": [
                {
                    "rank": rank,
                    "hostname": "nid000001" if rank < 4 else "nid000002",
                    "blocks": [
                        {
                            "attempted_objects": 8 if rank < 4 else 0,
                            "returned_objects": 8 if rank < 4 else 0,
                            "started": rank < 4,
                            "finished": rank < 4,
                            "worker_alive_after_join": False,
                            "elapsed_ms": completion_ms if rank < 4 else 0.0,
                            "error": None,
                        }
                        for _ in range(blocks)
                    ],
                }
                for rank in range(8)
            ],
            "overall_correctness_met": True,
        }
        transport = {
            "schema": analyzer.TRANSPORT_SCHEMA,
            "production_transport_verified": True,
            "transport": {
                "nccl_net": "AWS Libfabric",
                "fixed_nodelist": ["nid000001", "nid000002"],
            },
        }
        (arm / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (arm / "native_transport_receipt.json").write_text(json.dumps(transport), encoding="utf-8")

    def _root(self, path: Path, *, hot_completion_ms: float) -> Path:
        for name, load in (
            ("00_lmcache_control_a", "control"),
            ("01_lmcache_nccl_hot_a", "hot"),
            ("02_lmcache_nccl_hot_b", "hot"),
            ("03_lmcache_control_b", "control"),
        ):
            self._write_arm(
                path,
                name=name,
                load=load,
                completion_ms=3000.0 if load == "control" else hot_completion_ms,
            )
        receipt = {
            "schema": analyzer.EXECUTION_SCHEMA,
            "contract_sha256": analyzer._sha256(CONTRACT),
            "batch_submission": False,
            "privileged_or_container_configuration": False,
        }
        (path / "execution_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        return path

    def test_passes_material_real_lmcache_victim_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            value = analyzer.analyze(self._root(Path(tmp), hot_completion_ms=4500.0), CONTRACT)
        self.assertTrue(value["c7_real_lmcache_victim_pass"])
        self.assertTrue(value["actual_vllm_joint_control_run_allowed"])
        self.assertAlmostEqual(value["aggregate_effect"]["median_p50_degradation_fraction"], 0.5)

    def test_rejects_non_material_effect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            value = analyzer.analyze(self._root(Path(tmp), hot_completion_ms=3300.0), CONTRACT)
        self.assertFalse(value["c7_real_lmcache_victim_pass"])
        self.assertFalse(value["performance_claim_allowed"])

    def test_rejects_wrong_verified_victim_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(Path(tmp), hot_completion_ms=4500.0)
            path = root / "01_lmcache_nccl_hot_a" / "result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["blocks"][0]["receiver_verified_bytes"] -= 1
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bytes did not complete"):
                analyzer.analyze(root, CONTRACT)


class C7LMCacheVictimRunnerStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = Path(__file__).with_name(
            "run_tempo_go_c7_lmcache_victim_abba_in_allocation.sh"
        ).read_text(encoding="utf-8")

    def test_requires_one_existing_approved_interactive_allocation(self) -> None:
        self.assertIn("TEMPO_GO_C7_QUALIFICATION_APPROVED", self.text)
        self.assertIn("SLURM_JOB_ID", self.text)
        self.assertIn("SLURM_JOB_NUM_NODES", self.text)
        self.assertNotIn("sbatch", self.text)
        self.assertNotIn("scancel", self.text)

    def test_runs_official_lmcache_in_every_abba_arm(self) -> None:
        self.assertIn("control hot hot control", self.text)
        self.assertIn("TEMPO_GO_CROSS_LAYER_NO_BACKGROUND_TRANSFER=0", self.text)
        self.assertIn("run_lmcache_nixl_contention_2node_in_allocation.sh", self.text)
        self.assertIn("ARM_TOKEN_ITERS", self.text)

    def test_contains_no_privileged_or_container_path(self) -> None:
        for forbidden in ("sudo", "udiRoot", "CAP_NET_ADMIN", "--image"):
            self.assertNotIn(forbidden, self.text)


if __name__ == "__main__":
    unittest.main()
