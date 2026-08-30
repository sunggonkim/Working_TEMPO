from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).parent
LAUNCHER = HERE / "run_vllm_lmcache_tp16_quiescence_scout_in_allocation.sh"
NODE = HERE / "vllm_lmcache_tp16_quiescence_scout_node_v1.py"


class QuiescenceScoutLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.node = NODE.read_text(encoding="utf-8")

    def test_static_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        ast.parse(self.node, filename=str(NODE))

    def test_existing_allocation_one_bounded_n4_step(self) -> None:
        self.assertIn("SLURM_JOB_ID", self.launcher)
        self.assertIn("SLURM_JOB_NUM_NODES", self.launcher)
        self.assertIn("TEMPO_VLLM_LMCACHE_TP16_APPROVED", self.launcher)
        self.assertEqual(len(re.findall(r"\bsrun\b", self.launcher)), 1)
        self.assertIn("--nodes=4 --ntasks=4 --ntasks-per-node=1", self.launcher)
        self.assertIn("--distribution=block:block --gpus-per-task=4", self.launcher)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=20s 1800s", self.launcher)
        self.assertIsNone(re.search(r"(?m)^\s*(?:salloc|sbatch|scancel)\b", self.launcher))

    def test_runner_cli_and_frozen_plan(self) -> None:
        for fragment in (
            "eval.sota_4node.run_vllm_lmcache_tp16_quiescence_scout_v1",
            "real_tp16_quiescence_scout_v1.json",
            '"--quiescence-socket"',
            '"--quiescence-trace"',
            '"--campaign-index"',
            '"--allocation-id"',
        ):
            self.assertIn(fragment, self.node)
        self.assertIn("0|1|2", self.launcher)

    def test_node0_only_pinned_hook_and_sync_scheduler(self) -> None:
        self.assertIn("vllm_quiescence_sitecustomize_v3_pinned", self.node)
        self.assertIn('if node_id == 0:', self.node)
        self.assertIn('"TEMPO_VLLM_QUIESCENCE_ENABLED": "YES"', self.node)
        self.assertIn('"TEMPO_VLLM_QUIESCENCE_NODE_RANK": "0"', self.node)
        self.assertIn('"--no-async-scheduling"', self.node)
        self.assertIn('"--max-num-seqs",\n        "1"', self.node)
        self.assertIn("_scrub_hook_environment(sidecar_environment", self.node.replace("sidecar_environment = common_environment.copy()", "_scrub_hook_environment(sidecar_environment"))

    def test_preflight_health_provenance_then_sidecar(self) -> None:
        self.assertIn("0.26.0+cu129", self.node)
        self.assertIn("41295db73bb85ebda9cee7c4f32d944e5f973b6bcc0433ff6b152a9368b175b9", self.node)
        health = self.node.index("base._wait_for_health(")
        provenance = self.node.index("_trace_provenance(gate_trace)")
        sidecar = self.node.index("sidecar_process = subprocess.Popen(")
        self.assertLess(health, provenance)
        self.assertLess(provenance, sidecar)

    def test_unique_tmp_paths_low_ports_and_retry_stride(self) -> None:
        self.assertIn("TEMPO_TP16_QUIESCENCE_PORT_STRIDE", self.launcher)
        self.assertIn("CAMPAIGN_INDEX * 32", self.launcher)
        self.assertIn("/tmp/tempo-vllm-quiescence-", self.launcher)
        self.assertIn("/tmp/tempo-step-gate-", self.launcher)
        for base in ("9000", "13000", "17000", "21000"):
            self.assertIn(base, self.launcher)

    def test_term_kill_and_trace_copy_after_vllm_teardown(self) -> None:
        self.assertIn("os.killpg(process_group, signal.SIGTERM)", Path(HERE / "vllm_lmcache_tp16_campaign_node_v1.py").read_text())
        self.assertIn("os.killpg(process_group, signal.SIGKILL)", Path(HERE / "vllm_lmcache_tp16_campaign_node_v1.py").read_text())
        terminate = self.node.index('base._terminate_process_group(vllm_process')
        copy = self.node.index("copied_trace = _copy_trace")
        self.assertLess(terminate, copy)
        self.assertIn("vllm-quiescence-trace-node-0.jsonl", self.launcher)

    def test_no_monitor_or_external_activity(self) -> None:
        combined = self.launcher + self.node
        for forbidden in ("squeue", "sacct", "ssh ", "find ", "curl "):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
