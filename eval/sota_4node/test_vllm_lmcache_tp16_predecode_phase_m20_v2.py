import unittest
from eval.sota_4node import run_vllm_lmcache_tp16_predecode_phase_m20_entry as m20
from eval.sota_4node import run_vllm_lmcache_tp16_predecode_phase_m20_v2_entry as v2

class M20V2DispatchTest(unittest.TestCase):
    def test_candidate_dispatch_does_not_forward_mode(self):
        seen = {}
        original = m20._run_candidate
        try:
            def fake(*args, **kwargs):
                seen.update(kwargs)
                return {"ok": True}
            m20._run_candidate = fake
            self.assertEqual(v2._run_block(mode=m20.CANDIDATE_MODE, marker=7), {"ok": True})
            self.assertEqual(seen, {"marker": 7})
        finally:
            m20._run_candidate = original

if __name__ == "__main__":
    unittest.main()
