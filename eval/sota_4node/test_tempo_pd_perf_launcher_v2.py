from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

from eval.sota_4node import vllm_lmcache_tempo_pd_perf_node_v2 as node


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "eval/sota_4node/tempo_pd_perf_node_entry_v2.sh"
LAUNCHER = ROOT / "eval/sota_4node/run_tempo_pd_perf_v2_in_allocation.sh"
CONTRACT = ROOT / "eval/sota_4node/tempo_pd_performance_contract_v2.json"


class TempoPDPerfLauncherV2Tests(unittest.TestCase):
    def test_frozen_contract_is_context_safe_and_measurement_sized(self) -> None:
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        workload = value["workload"]
        self.assertEqual(workload["prompt_repetitions"], [64, 192, 384])
        self.assertEqual(workload["output_tokens"], 64)
        self.assertLessEqual(max(workload["expected_prompt_tokens_for_frozen_model"]) + 64,
                             value["model_max_length"])
        self.assertEqual(len(value["lifecycles"]), 5)
        self.assertTrue(value["policy"]["manifest_frozen_before_validation"])

    def test_workload_wrapper_passes_frozen_repetitions_and_checks_context(self) -> None:
        class Args:
            result_dir = Path("/repo/results/run")
            repo_root = Path("/repo")
            node_index = 0
            samples_per_bucket = 3
            output_tokens = 64

        manifest = {
            "buckets": [
                {"repetitions": 64, "prompt_tokens": 1220},
                {"repetitions": 192, "prompt_tokens": 3652},
                {"repetitions": 384, "prompt_tokens": 7300},
            ]
        }
        with mock.patch.object(node.subprocess, "run") as run, \
             mock.patch.object(Path, "read_text", return_value=json.dumps(manifest)):
            calibration, validation = node._prepare_workloads(
                Args(), Path("/repo/models/Qwen2.5-7B-Instruct"), Path("/repo/python")
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--repetitions") + 1], "64,192,384")
        self.assertEqual(command[command.index("--output-tokens") + 1], "64")
        self.assertEqual(calibration.name, "calibration.jsonl")
        self.assertEqual(validation.name, "validation.jsonl")

    def test_launcher_is_one_bounded_existing_allocation_step(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        entry = ENTRY.read_text(encoding="utf-8")
        self.assertEqual(text.count(" srun "), 1)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("salloc", text)
        self.assertIn("TEMPO_PD_PERF_APPROVED", text)
        self.assertIn(" 2.0 4 64 3 1000 100 5000", text)
        self.assertIn("vllm_lmcache_tempo_pd_perf_node_v2", entry)
        self.assertIn("--e2e-slo-ms \"${11}\"", entry)


if __name__ == "__main__":
    unittest.main()
