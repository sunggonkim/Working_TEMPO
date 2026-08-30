import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.run_tempo_pd_same_server_mixed_only_client_v265 import _rows


class MixedOnlyTest(unittest.TestCase):
    def test_variants_are_counterbalanced(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "base.jsonl"
            path.write_text("".join(json.dumps({
                "request_id": f"cache-item-{i:02d}",
                "prompt": f"nonce {i:03d}.", "max_tokens": 16}) + "\n"
                for i in range(24)))
            rows = _rows(path, "measured")
        tempo_a = sum("ssb-tempo" in x["request_id"] and "mixA" in x["request_id"]
                      for x in rows)
        tempo_b = sum("ssb-tempo" in x["request_id"] and "mixB" in x["request_id"]
                      for x in rows)
        self.assertEqual((tempo_a, tempo_b), (12, 12))


if __name__ == "__main__":
    unittest.main()
