import unittest
from eval.sota_4node import run_vllm_lmcache_tp16_predecode_phase_m20_entry as m20
from eval.sota_4node import run_vllm_lmcache_tp16_predecode_phase_m20_v3_entry as v3

class M20V3WorkerTest(unittest.TestCase):
    def test_entry_restores_entered_aware_worker(self):
        original_old = m20.old._transfer_worker
        original_main = m20.main
        seen = {}
        try:
            m20.old._transfer_worker = lambda **kwargs: None
            m20.main = lambda: seen.update(worker=m20.old._transfer_worker)
            v3.main()
            self.assertIs(seen["worker"], m20.fixed._transfer_worker)
        finally:
            m20.old._transfer_worker = original_old
            m20.main = original_main

if __name__ == "__main__": unittest.main()
