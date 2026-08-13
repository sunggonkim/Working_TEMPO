import ast
import unittest
from pathlib import Path


class KVLiveSmokeStaticTests(unittest.TestCase):
    def test_script_has_exact_contract_markers(self):
        path = Path(__file__).with_name("run_inference_kv_live_smoke.py")
        tree = ast.parse(path.read_text())
        self.assertIn("KVVersion", path.read_text())
        self.assertIn("admit_via_domain_controller", path.read_text())
        self.assertIn("correctness_met", path.read_text())
        self.assertIn("causal_claim_allowed", path.read_text())
        self.assertIsInstance(tree, ast.Module)
