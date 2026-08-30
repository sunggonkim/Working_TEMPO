from pathlib import Path
import json
import tempfile
import unittest

from eval.sota_4node import tempo_pd_unique_head_many_workload_v37 as workload


class ForcedManyV37Tests(unittest.TestCase):
    def test_rewrite_24_unique_prefixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_text("".join(json.dumps({"prompt": "same"}) + "\n" for _ in range(24)))
            workload._rewrite(path, 24)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len({row["prompt"] for row in rows}), 24)

    def test_one_step_and_exact_load(self):
        text = Path(__file__).with_name(
            "run_tempo_pd_forced_many_v37_in_allocation.sh"
        ).read_text()
        self.assertEqual(text.count("srun "), 1)
        self.assertIn("32 32 32 8 3000 250 12000", text)


if __name__ == "__main__":
    unittest.main()
