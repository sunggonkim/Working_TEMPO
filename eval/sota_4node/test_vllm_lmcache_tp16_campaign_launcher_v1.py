from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path


HERE = Path(__file__).parent
LAUNCHER = HERE / "run_vllm_lmcache_tp16_campaign_in_allocation.sh"
NODE_DRIVER = HERE / "vllm_lmcache_tp16_campaign_node_v1.py"


class VllmLmcacheTp16CampaignLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.node_driver = NODE_DRIVER.read_text(encoding="utf-8")

    def test_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        ast.parse(self.node_driver, filename=str(NODE_DRIVER))

    def test_existing_allocation_only_and_explicit_approval(self) -> None:
        self.assertIn("SLURM_JOB_ID", self.launcher)
        self.assertIn("SLURM_JOB_NUM_NODES", self.launcher)
        self.assertIn("TEMPO_VLLM_LMCACHE_TP16_APPROVED", self.launcher)
        self.assertIsNone(
            re.search(r"(?m)^\s*(?:salloc|sbatch|scancel)\b", self.launcher)
        )

    def test_one_bounded_four_node_step(self) -> None:
        self.assertEqual(len(re.findall(r"\bsrun\b", self.launcher)), 1)
        self.assertIn(
            "timeout --foreground --signal=TERM --kill-after=20s 1800s",
            self.launcher,
        )
        self.assertIn("--nodes=4 --ntasks=4 --ntasks-per-node=1", self.launcher)
        self.assertIn("--distribution=block:block --gpus-per-task=4", self.launcher)
        self.assertIn("--time=00:29:30", self.launcher)

    def test_campaign_and_port_isolation(self) -> None:
        self.assertIn("0|1|2", self.launcher)
        self.assertIn("CAMPAIGN_INDEX * 32", self.launcher)
        for base in ("20000", "30000", "40000", "50000"):
            self.assertIn(base, self.launcher)
        self.assertIn("NIXL_PORT_BASE + 7 <= 65535", self.launcher)

    def test_stale_result_fails_closed(self) -> None:
        stale_check = self.launcher.index('[[ -e "${RESULT_DIR}/result.json" ]]')
        create_directory = self.launcher.index('mkdir -p -- "${RESULT_DIR}"')
        self.assertLess(stale_check, create_directory)
        self.assertIn("refusing to overwrite stale result", self.launcher)
        self.assertIn("refusing to overwrite stale result", self.node_driver)

    def test_native_tp16_and_four_by_four_sidecar(self) -> None:
        for fragment in (
            '"--tensor-parallel-size"',
            '"--distributed-executor-backend"',
            '"--nnodes"',
            '"--node-rank"',
            '"--headless"',
            'f"--nnodes={NODES}"',
            'f"--nproc-per-node={LOCAL_RANKS}"',
            '"--max-restarts=0"',
            "run_vllm_lmcache_tp16_pair_stagger_coalesced_v2",
            "real_tp16_pair_stagger_coalesced_v2.json",
            '"--allocation-id"',
        ):
            self.assertIn(fragment, self.node_driver)

    def test_bounded_readiness_and_process_group_cleanup(self) -> None:
        self.assertIn('f"http://{api_host}:{api_port}/health"', self.node_driver)
        self.assertIn("process.poll()", self.node_driver)
        self.assertIn("time.monotonic() + timeout_s", self.node_driver)
        self.assertIn("start_new_session=True", self.node_driver)
        self.assertIn("os.killpg(process_group, signal.SIGTERM)", self.node_driver)
        self.assertIn("os.killpg(process_group, signal.SIGKILL)", self.node_driver)
        self.assertIn("sidecar_process.wait(timeout=args.sidecar_timeout_s)", self.node_driver)

    def test_offline_local_paths_and_no_scheduler_monitoring(self) -> None:
        for fragment in (
            "models/TinyLlama-1.1B-Chat-v1.0",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "/tmp/tempo-vllm-tp16-",
            "FLASHINFER_WORKSPACE_BASE",
        ):
            self.assertIn(fragment, self.node_driver)
        combined = self.launcher + self.node_driver
        for forbidden in ("squeue", "sacct", "ssh ", "find "):
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
