import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.run_tempo_pd_same_server_mixed_crossover_client_v260 import _rows


class MixedCrossoverTest(unittest.TestCase):
    def test_exact_balanced_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "base.jsonl"
            rows = [{"request_id": f"cache-item-{i:02d}",
                     "prompt": f"nonce {i:03d}.", "max_tokens": 16}
                    for i in range(24)]
            path.write_text("".join(json.dumps(x) + "\n" for x in rows))
            mixed = _rows(path, "measured")
        self.assertEqual(len(mixed), 48)
        self.assertEqual(sum("ssb-tempo-" in x["request_id"] for x in mixed), 24)
        self.assertEqual(sum("ssb-remote-" in x["request_id"] for x in mixed), 24)
        self.assertEqual(len({x["prompt"] for x in mixed}), 48)


if __name__ == "__main__":
    unittest.main()
