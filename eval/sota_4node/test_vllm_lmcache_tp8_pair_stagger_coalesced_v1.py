from pathlib import Path
import unittest
from eval.sota_4node import run_vllm_lmcache_tp8_pair_stagger_coalesced_v1 as c

class CoalescedTests(unittest.TestCase):
    def test_contract_and_coverage(self):
        c.validate_schedule(); payload, contract = c.load_contract(Path("eval/sota_4node/real_tp8_pair_stagger_coalesced_v1.json"))
        self.assertEqual(contract, c.CONTRACT_ID); self.assertEqual(payload["schedule"]["global_bytes"], 67_108_864)
        for pair, token in enumerate(c.TOKENS):
            self.assertEqual(c.coalesced_indices("tempo_group2", token, pair_index=pair), tuple(range(32)))

if __name__ == "__main__": unittest.main()
