from pathlib import Path
import json
import tempfile
import unittest

from eval.sota_4node import tempo_pd_unique_head_short_workload_v23 as unique


class UniqueHeadV23Tests(unittest.TestCase):
    def test_nonce_precedes_shared_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            path.write_text("".join(
                json.dumps({"request_id": str(i), "prompt": "shared", "max_tokens": 32}) + "\n"
                for i in range(9)
            ))
            unique._rewrite(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len({row["prompt"] for row in rows}), 9)
            self.assertTrue(all(row["prompt"].startswith("Cold-cache request nonce") for row in rows))

    def test_launcher_one_step(self):
        text = Path(__file__).with_name(
            "run_tempo_pd_unique_head_v23_in_allocation.sh"
        ).read_text()
        self.assertEqual(text.count("srun "), 1)
        self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
