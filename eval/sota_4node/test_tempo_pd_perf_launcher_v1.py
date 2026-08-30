from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
NODE = ROOT / "eval/sota_4node/vllm_lmcache_tempo_pd_perf_node_v1.py"
ENTRY = ROOT / "eval/sota_4node/tempo_pd_perf_node_entry_v1.sh"
LAUNCHER = ROOT / "eval/sota_4node/run_tempo_pd_perf_v1_in_allocation.sh"


class TempoPDPerfLauncherTests(unittest.TestCase):
    def test_python_syntax_and_five_fresh_lifecycles(self) -> None:
        text = NODE.read_text(encoding="utf-8")
        ast.parse(text)
        self.assertIn('("calibration_local", "fixed_local", "calibration")', text)
        self.assertIn('("calibration_remote", "lmcache_always_remote", "calibration")', text)
        self.assertIn('("validation_tempo", "tempo_auto", "validation")', text)
        self.assertIn("use_gpu_connector_v3: True", text)
        self.assertIn("--max-num-batched-tokens", text)
        self.assertIn("build_tempo_pd_profile_manifest_v1", text)
        self.assertIn("analyze_tempo_pd_performance_v1", text)

    def test_launcher_is_existing_allocation_only_and_bounded(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("TEMPO_PD_PERF_APPROVED", text)
        self.assertIn("SLURM_JOB_NUM_NODES", text)
        self.assertEqual(text.count(" srun "), 1)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("salloc", text)
        self.assertIn("timeout --foreground", text)
        self.assertIn("--kill-on-bad-exit=1", text)

    def test_entry_targets_canonical_node(self) -> None:
        text = ENTRY.read_text(encoding="utf-8")
        self.assertIn("eval.sota_4node.vllm_lmcache_tempo_pd_perf_node_v1", text)
        self.assertIn("SLURM_NODEID", text)


if __name__ == "__main__":
    unittest.main()
