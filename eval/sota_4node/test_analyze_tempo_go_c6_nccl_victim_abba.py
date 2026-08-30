from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import analyze_tempo_go_c6_nccl_victim_abba as analyzer


CONTRACT = Path(__file__).with_name("tempo_go_c6_qualification_contract_v1.json")


class C6NCCLVictimABBAAnalysisTests(unittest.TestCase):
    def _write_arm(
        self,
        root: Path,
        *,
        name: str,
        mode: str,
        p50: float,
        p99: float,
    ) -> None:
        arm = root / name
        arm.mkdir()
        expected_bytes = 0 if mode == "nccl_only" else 4 * 8 * 128 * 1024 * 1024
        block = {
            "correctness_met": True,
            "expected_background_bytes": expected_bytes,
            "source_completed_bytes": expected_bytes,
            "receiver_verified_bytes": expected_bytes,
            "full_bytes_completed": True,
            "full_bytes_verified": True,
        }
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
                "token_iters": 128,
                "blocks": 4,
                "foreground_bytes": 16 * 1024 * 1024,
                "block_delay_s": 0.0,
                "minimum_active_duration_s": 30.0,
                "maximum_blocks": 512,
                "process_group_timeout_s": 120,
                "nixl_transfer_timeout_s": 60,
                "background_mode": mode,
            },
            "blocks": [dict(block) for _ in range(4)],
            "active_loop": {
                "rank_min_elapsed_ms": 30_100.0,
                "rank_max_elapsed_ms": 30_200.0,
                "minimum_requested_ms": 30_000.0,
                "horizon_met": True,
            },
            "summary": {
                "global_token_tail_p50_ms": p50,
                "global_token_tail_p99_ms": p99,
                "background_completion_p99_ms": 0.0 if mode == "nccl_only" else 9.0,
            },
            "rank_diagnostics": [
                {"rank": rank, "hostname": "nid000001" if rank < 4 else "nid000002"}
                for rank in range(8)
            ],
            "overall_correctness_met": True,
        }
        transport = {
            "schema": analyzer.TRANSPORT_SCHEMA,
            "production_transport_verified": True,
            "transport": {"nccl_net": "AWS Libfabric"},
        }
        (arm / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (arm / "native_transport_receipt.json").write_text(
            json.dumps(transport), encoding="utf-8"
        )

    def _root(self, path: Path, *, loaded_p50: float, loaded_p99: float) -> Path:
        self._write_arm(
            path, name="00_nccl_only_a", mode="nccl_only", p50=10.0, p99=20.0
        )
        self._write_arm(
            path, name="01_lmcache_on_a", mode="nixl_ucx", p50=loaded_p50, p99=loaded_p99
        )
        self._write_arm(
            path, name="02_lmcache_on_b", mode="nixl_ucx", p50=loaded_p50, p99=loaded_p99
        )
        self._write_arm(
            path, name="03_nccl_only_b", mode="nccl_only", p50=10.0, p99=20.0
        )
        return path

    def test_passes_material_real_nccl_victim_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            value = analyzer.analyze(
                self._root(Path(tmp), loaded_p50=13.0, loaded_p99=30.0), CONTRACT
            )
        self.assertTrue(value["q1_real_nccl_victim_pass"])
        self.assertTrue(value["q3_service_horizon_pass"])
        self.assertAlmostEqual(
            value["aggregate_effect"]["median_p50_degradation_fraction"], 0.3
        )

    def test_rejects_counter_only_non_material_effect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            value = analyzer.analyze(
                self._root(Path(tmp), loaded_p50=11.0, loaded_p99=30.0), CONTRACT
            )
        self.assertFalse(value["q1_real_nccl_victim_pass"])
        self.assertFalse(value["performance_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
