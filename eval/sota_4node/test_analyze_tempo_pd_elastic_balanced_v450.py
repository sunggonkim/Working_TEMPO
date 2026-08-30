import ast
from pathlib import Path
import unittest


class ElasticAnalyzerV450StaticTest(unittest.TestCase):
    def test_credit_lifecycle_is_fail_closed_and_part_of_pass(self):
        path = Path(__file__).with_name(
            "analyze_tempo_pd_elastic_balanced_v450.py")
        source = path.read_text()
        ast.parse(source)
        self.assertIn('len(rows) == 48', source)
        self.assertIn('started <= released == response < finished', source)
        self.assertIn('"first_response_chunk"', source)
        self.assertIn('"bounded_ingress_queue"', source)
        self.assertIn('all(result["candidate_gates"].values())', source)


if __name__ == "__main__":
    unittest.main()
