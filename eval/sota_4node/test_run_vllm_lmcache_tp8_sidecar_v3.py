from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import unittest
from unittest import mock

from eval.sota_4node import run_vllm_lmcache_tp8_sidecar_v3 as screen


HERE = Path(__file__).parent
CONTRACT = HERE / "real_tp8_pair_stagger_v1.json"
LAUNCHER = HERE / "run_vllm_lmcache_tp8_stagger_v3_in_allocation.sh"
NODE_ENTRY = HERE / "vllm_lmcache_tp8_stagger_node_v3.sh"
RUNNER = HERE / "run_vllm_lmcache_tp8_sidecar_v3.py"


class VllmLmcacheTp8PairStaggerV3Tests(unittest.TestCase):
    def test_signed_contract_and_exact_schedule(self) -> None:
        screen.validate_pair_stagger_schedule()
        payload, signature = screen.load_pair_stagger_contract(CONTRACT)
        self.assertEqual(signature, screen.PAIR_STAGGER_SIGNATURE)
        self.assertEqual(payload["provenance"], screen.EXPECTED_PROVENANCE)
        for pair in range(screen.PAIR_COUNT):
            calls = []
            flattened = []
            for token in screen.SCHEDULED_TOKENS:
                indices = screen.schedule_object_indices(
                    screen.PAIR_MODE, token, pair_index=pair
                )
                if indices:
                    calls.append(token)
                    flattened.extend(indices)
                    self.assertEqual(len(indices), 4)
            self.assertEqual(len(calls), 8)
            self.assertTrue(all(b - a == 8 for a, b in zip(calls, calls[1:])))
            self.assertEqual(sorted(flattened), list(range(32)))

    def test_one_pair_admitted_per_slot_but_not_global_single_flight(self) -> None:
        for token in screen.SCHEDULED_TOKENS:
            active = [
                pair
                for pair in range(screen.PAIR_COUNT)
                if screen.schedule_object_indices(
                    screen.PAIR_MODE, token, pair_index=pair
                )
            ]
            self.assertEqual(active, [((token - 1) // 2) % 4])
        self.assertFalse(screen.EXPECTED_PROVENANCE["global_single_flight"])
        self.assertTrue(screen.EXPECTED_PROVENANCE["physical_transfer_overlap_possible"])

    def test_request_start_probe_is_suppressed_then_event_zero_admits_t1(self) -> None:
        screen._install_corrections()
        screen._pair_runtime = True
        screen._suppress_initial_pair_lookup = True
        try:
            self.assertEqual(
                screen._v1.schedule_object_indices(
                    "tempo_group2", 0, pair_index=0
                ),
                (),
            )
            self.assertEqual(
                screen._v1.schedule_object_indices(
                    "tempo_group2", 0, pair_index=0
                ),
                (0, 16, 1, 17),
            )
            self.assertEqual(
                screen._v1.schedule_object_indices(
                    "tempo_group2", 0, pair_index=1
                ),
                (),
            )
        finally:
            screen._pair_runtime = False
            screen._suppress_initial_pair_lookup = False

    def test_latin_balance_and_aggregate_gates(self) -> None:
        for mode in screen.MODES:
            self.assertEqual(
                sorted(row.index(mode) for row in screen.LATIN_ROWS),
                [0, 1, 2],
            )
        candidate = {
            "mode": screen.PAIR_MODE,
            "correctness_met": True,
            "expected_background_bytes": 64 * (1 << 20),
            "background_completed_bytes": 64 * (1 << 20),
            "receiver_verified_bytes": 64 * (1 << 20),
            "schedule_start_adherence_met": True,
            "absolute_service_deadline_met": True,
            "post_foreground_drain_ms": 0.0,
        }
        base = {
            "blocks": [dict(candidate) for _ in range(3)],
            "overall_correctness_met": True,
            "honesty_boundary": "registered sidecar buffers are not live vLLM KV",
            "frozen_group2": {"must": "be removed"},
            "candidate_start_lag_cap_met": False,
        }
        with mock.patch.object(
            screen, "_original_aggregate_rank_records", return_value=base
        ):
            result = screen.aggregate_rank_records([])
        self.assertTrue(result["candidate_exact_bytes_met"])
        self.assertTrue(result["candidate_schedule_adherence_met"])
        self.assertTrue(result["candidate_absolute_deadline_met"])
        self.assertTrue(result["candidate_no_post_foreground_drain_met"])
        self.assertFalse(result["promotion_valid"])
        self.assertFalse(result["real_tp8_pair_stagger_v1"]["global_single_flight"])
        self.assertNotIn("frozen_group2", result)

    def test_shell_and_forced_success_exit_contracts(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        subprocess.run(["bash", "-n", str(NODE_ENTRY)], check=True)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        node = NODE_ENTRY.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"\bsrun\b", launcher)), 1)
        self.assertIn("TEMPO_VLLM_LMCACHE_STAGGER_APPROVED", launcher)
        self.assertIn("TEMPO_VLLM_STARTUP_GRACE_S=90", launcher)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=15s 1200s", launcher)
        self.assertIsNone(re.search(r"(?m)^\s*(?:salloc|sbatch|scancel)\b", launcher))
        self.assertIn("/tmp/tempo-vllm-${SLURM_JOB_ID}-node${SLURM_NODEID}", node)
        self.assertIn("--tensor-parallel-size 8", node)
        self.assertIn("--no-enable-prefix-caching", node)
        self.assertIn("run_vllm_lmcache_tp8_sidecar_v3", node)
        self.assertIn("real_tp8_pair_stagger_v1.json", node)
        self.assertIn("sys.stdout.flush()", runner)
        self.assertIn("sys.stderr.flush()", runner)
        self.assertIn("os._exit(0)", runner)
        self.assertNotIn("channel.close", runner)


if __name__ == "__main__":
    unittest.main()
