from pathlib import Path
import re
import unittest
SCRIPT=Path(__file__).with_name("run_lmcache_active_pulse_hybrid_in_allocation.sh").read_text()
class HybridLauncherTests(unittest.TestCase):
    def test_one_bounded_approved_srun(self):
        self.assertEqual(len(re.findall(r"\bsrun\b",SCRIPT)),1)
        self.assertIn("TEMPO_LMCACHE_ACTIVE_HYBRID_APPROVED",SCRIPT)
        self.assertIn("timeout --foreground --signal=TERM --kill-after=5s 240s",SCRIPT)
        self.assertIn("--nodes=2 --ntasks=8 --ntasks-per-node=4",SCRIPT)
        self.assertLess(SCRIPT.index("compile_lmcache_active_pulse_hybrid_plan"),SCRIPT.index("srun --exact"))
        for word in ("salloc","sbatch","scancel","pip install","git clone","while ","until "):
            self.assertNotIn(word,SCRIPT)
if __name__ == "__main__": unittest.main()
