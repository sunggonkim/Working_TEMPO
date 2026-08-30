import json
import unittest
from pathlib import Path
from eval.sota_4node import run_vllm_lmcache_tp16_predecode_phase_m20_entry as m20

class M20ContractTest(unittest.TestCase):
    def test_contract_matches_file(self):
        path = Path(__file__).with_name("real_tp16_predecode_phase_m20.json")
        self.assertEqual(json.loads(path.read_text()), m20._expected_contract())

    def test_frozen_phase_contract(self):
        contract = m20._expected_contract()
        self.assertEqual(contract["algorithm"]["phase_order"],
                         ["remote_kv_transfer", "receiver_verify", "tp16_decode"])
        self.assertFalse(contract["algorithm"]["transfer_decode_overlap"])
        self.assertTrue(contract["algorithm"]["admission_to_response_includes_transfer"])
        self.assertEqual(contract["transfer"]["global_bytes"], 128 << 20)
        self.assertEqual(contract["transfer"]["physical_descriptors_global"], 8)

if __name__ == "__main__":
    unittest.main()
