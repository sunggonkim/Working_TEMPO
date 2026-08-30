from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "eval/sota_4node/run_tempo_pd_c3_coupled_in_allocation.sh"
ABBA = ROOT / "eval/sota_4node/run_tempo_pd_c3_coupled_abba_in_allocation.sh"
ALLOCATION_GUARD = (
    ROOT / "eval/sota_4node/require_perlmutter_4node_4h_interactive.sh")


class CoupledC3WrapperStaticTest(unittest.TestCase):
    def _wrapper(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        self.assertIn(
            'source "${SCRIPT_DIR}/require_perlmutter_4node_4h_interactive.sh"',
            text,
        )
        self.assertIn("--nodes=4", text)
        self.assertIn("--gpus-per-task=4", text)
        expected_readiness = (
            "3600" if path == ABBA else "1800")
        self.assertIn(
            f"TEMPO_PD_KV_ATTR_READINESS_S:-{expected_readiness}", text)
        self.assertNotIn("salloc ", text)
        approval_export = text.index("export TEMPO_PD_C3_APPROVED")
        srun = text.index("srun --overlap")
        self.assertLess(approval_export, srun)
        return text

    def test_guard_requires_exact_interactive_allocation(self):
        text = ALLOCATION_GUARD.read_text(encoding="utf-8")
        self.assertIn('${SLURM_JOB_ID:?existing Slurm allocation required}', text)
        self.assertIn('${SLURM_JOB_NODELIST:?Slurm allocation nodelist required}', text)
        self.assertIn('SLURM_JOB_NUM_NODES:-${SLURM_JOB_NODES:-}', text)
        self.assertIn('TEMPO_PD_ALLOCATION_RECORD_SOURCE="inherited"', text)
        self.assertIn('/usr/bin/timeout --foreground', text)
        self.assertIn('scontrol show job "${SLURM_JOB_ID}" --oneliner', text)
        self.assertIn('JobState=RUNNING', text)
        self.assertIn('QOS=interactive', text)
        self.assertIn('TimeLimit=04:00:00', text)
        self.assertIn('NumNodes=4', text)
        self.assertIn('gres/gpu=16', text)

    def test_pilot_requires_existing_four_node_gpu_allocation(self):
        text = self._wrapper(PILOT)
        self.assertIn("TEMPO_PD_KV_ATTR_REPETITIONS=1", text)
        self.assertIn("TEMPO_PD_KV_ATTR_ARM_ORDER=local_remote", text)

    def test_abba_requires_existing_four_node_gpu_allocation(self):
        text = self._wrapper(ABBA)
        self.assertIn("TEMPO_PD_KV_ATTR_REPETITIONS=2", text)
        self.assertIn("TEMPO_PD_KV_ATTR_ARM_ORDER=paired_abba", text)
        self.assertIn("kill-after=30s 7200s", text)
        self.assertIn("--time=01:58:00", text)
        self.assertIn(
            "eval.sota_4node.analyze_tempo_pd_kv_only_attribution", text)
        self.assertIn(
            "eval.sota_4node.analyze_tempo_pd_c3_coupled_abba", text)
        self.assertIn("kv_only_characterization_v3.json", text)
        self.assertIn("c3_abba_gate_v1.json", text)


if __name__ == "__main__":
    unittest.main()
