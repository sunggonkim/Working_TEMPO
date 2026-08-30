from pathlib import Path
import json
import tempfile
import unittest

from eval.sota_4node import tempo_pd_unique_short_workload_v21 as unique


class UniqueShortV21Tests(unittest.TestCase):
    def test_rewrite_produces_nine_unique_prompts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_text("".join(
                json.dumps({"request_id": str(i), "prompt": "same", "max_tokens": 32}) + "\n"
                for i in range(9)
            ))
            unique._rewrite(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len({row["prompt"] for row in rows}), 9)

    def test_launcher_one_bounded_step(self):
        root = Path(__file__).resolve().parent
        text = (root / "run_tempo_pd_unique_short_v21_in_allocation.sh").read_text()
        self.assertEqual(text.count("srun "), 1)
        self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
