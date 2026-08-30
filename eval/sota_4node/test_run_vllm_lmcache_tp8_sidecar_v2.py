from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from eval.sota_4node import run_vllm_lmcache_tp8_sidecar_v2 as screen


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    REPO_ROOT
    / "results"
    / "lmcache_active_pulse_group2_job_56929977"
    / "active_pulse_group2_plan.json"
)
LAUNCHER = Path(__file__).parent / "run_vllm_lmcache_tp8_screen_v2_in_allocation.sh"
NODE_ENTRY = Path(__file__).parent / "vllm_lmcache_tp8_screen_node_v2.sh"


class VllmLmcacheTp8SidecarV2Tests(unittest.TestCase):
    def test_signed_plan_and_runtime_boundary_mapping(self) -> None:
        screen.validate_frozen_schedule()
        _, signature = screen.load_frozen_plan(PLAN)
        self.assertEqual(signature, screen.EXPECTED_PLAN_SIGNATURE)
        screen._install_corrections()
        screen._shift_group2_schedule = True
        try:
            # Event 3 is the observable boundary immediately before decode
            # token 4, whose frozen pulse contains chunks 0 and 1 per pair.
            self.assertEqual(
                screen._v1.schedule_object_indices(
                    "tempo_group2", 3, pair_index=0
                ),
                (0, 16, 1, 17),
            )
            self.assertEqual(
                screen._v1.schedule_object_indices(
                    "tempo_group2", 63, pair_index=0
                ),
                (),
            )
        finally:
            screen._shift_group2_schedule = False

    def test_changed_envelope_fails_before_nested_decode(self) -> None:
        payload = json.loads(PLAN.read_text(encoding="utf-8"))
        payload["artifact_signature_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact signature changed"):
                screen.load_frozen_plan(path)

    def test_default_model_is_resolved_to_local_absolute_path(self) -> None:
        argv = [
            "screen",
            "--output-dir",
            "results/test-v2",
            "--api-host",
            "nid000001",
            "--api-port",
            "42000",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = screen._parse_args()
        self.assertEqual(
            args.model,
            str((REPO_ROOT / "models/TinyLlama-1.1B-Chat-v1.0").resolve()),
        )

    def test_v2_shell_syntax_and_single_bounded_srun(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        subprocess.run(["bash", "-n", str(NODE_ENTRY)], check=True)
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"\bsrun\b", launcher)), 1)
        self.assertIn(
            "timeout --foreground --signal=TERM --kill-after=15s 1200s",
            launcher,
        )
        self.assertIsNone(
            re.search(r"(?m)^\s*(?:salloc|sbatch|scancel)\b", launcher)
        )

    def test_cold_start_and_same_step_sidecar_contract(self) -> None:
        launcher = LAUNCHER.read_text(encoding="utf-8")
        node = NODE_ENTRY.read_text(encoding="utf-8")
        self.assertIn("TEMPO_VLLM_STARTUP_GRACE_S=180", launcher)
        self.assertIn("TEMPO_VLLM_STARTUP_GRACE_S <= 240", node)
        self.assertIn("/tmp/tempo-vllm-${SLURM_JOB_ID}-node${SLURM_NODEID}", node)
        self.assertIn("--tensor-parallel-size 8", node)
        self.assertIn("--no-enable-prefix-caching", node)
        self.assertIn("--nproc-per-node=4", node)
        self.assertIn(
            "-m eval.sota_4node.run_vllm_lmcache_tp8_sidecar_v2",
            node,
        )


if __name__ == "__main__":
    unittest.main()
