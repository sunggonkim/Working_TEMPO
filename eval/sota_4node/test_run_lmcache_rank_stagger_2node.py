from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest

from eval.sota_4node import run_lmcache_rank_stagger_2node as stagger
from tempo.inference_epoch import EpochProfile, WidthPoint, compile_epoch


def _plan():
    return compile_epoch(
        EpochProfile(
            total_quanta=4,
            deadline_tokens=10,
            token_slack_ns=(3,) * 16,
            width_points=(WidthPoint(0, 0), WidthPoint(1, 1)),
            max_width=1,
        )
    )


class LMCacheRankStaggerPolicyTest(unittest.TestCase):
    def test_plan_is_four_ordered_rank_service_calls(self) -> None:
        plan = _plan()
        self.assertTrue(plan.feasible)
        self.assertEqual(plan.completion_token_exclusive, 4)
        self.assertEqual(plan.width_by_token, (1, 1, 1, 1) + (0,) * 12)
        self.assertEqual(
            tuple(index for group in plan.quantum_indices_by_token for index in group),
            (0, 1, 2, 3),
        )

    def test_candidate_coalesces_one_full_batch_per_source_rank(self) -> None:
        plan = _plan()
        expected = (0, 4, 1, 5, 2, 6, 3, 7)
        for pair in range(4):
            calls = [
                stagger._pair_batch_object_indices(
                    plan,
                    "tempo_epoch",
                    token,
                    pair_index=pair,
                    requests=2,
                )
                for token in range(16)
            ]
            nonempty = [call for call in calls if call]
            self.assertEqual(nonempty, [expected])
            self.assertEqual(calls[pair], expected)

    def test_greedy_and_candidate_have_same_four_global_call_count(self) -> None:
        plan = _plan()
        for mode in ("lmcache_greedy", "tempo_epoch"):
            calls = sum(
                bool(
                    stagger._pair_batch_object_indices(
                        plan,
                        mode,
                        token,
                        pair_index=pair,
                        requests=2,
                    )
                )
                for pair in range(4)
                for token in range(16)
            )
            self.assertEqual(calls, 4)


class LMCacheRankStaggerLauncherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = Path(__file__).with_name(
            "run_lmcache_rank_stagger_in_allocation.sh"
        )
        cls.text = cls.script.read_text(encoding="utf-8")

    def test_shell_and_existing_allocation_contract(self) -> None:
        subprocess.run(["bash", "-n", str(self.script)], check=True)
        self.assertIn("TEMPO_LMCACHE_RANK_STAGGER_APPROVED", self.text)
        self.assertNotRegex(self.text, r"\b(?:sbatch|salloc|scancel|retry|while|sleep)\b")

    def test_one_bounded_step_and_exact_candidate(self) -> None:
        self.assertEqual(len(re.findall(r"^\s*srun\b", self.text, re.MULTILINE)), 1)
        for literal in (
            "timeout --foreground --signal=TERM --kill-after=5s 240s",
            "--total-quanta 4",
            "--deadline-tokens 10",
            "--max-width 1",
            "--kv-mib 4",
            "--chunk-mib 1",
            "--tokens 16",
            "--layers 8",
            "lmcache_rank_stagger_bootstrap",
        ):
            self.assertIn(literal, self.text)


if __name__ == "__main__":
    unittest.main()
