import json
from pathlib import Path
import unittest

from eval.sota_4node import run_tempo_pd_elastic_balanced_client_v446 as client


class CacheIsolationTest(unittest.TestCase):
    def test_exact_prior_workload_has_unique_length_preserving_first_chunks(self):
        from transformers import AutoTokenizer
        root = Path(__file__).resolve().parents[2]
        source = root / (
            "results/tempo_pd_latched_cap6_bursty_v407_job_57078464/"
            "tempo_credit_admission/warmup.jsonl")
        if not source.is_file():
            self.skipTest("exact evidence workload unavailable")
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        client._TOKENIZER = AutoTokenizer.from_pretrained(
            str(root / "models/Qwen2.5-7B-Instruct"), local_files_only=True)
        original = [len(client._TOKENIZER.encode(row["prompt"], add_special_tokens=False))
                    for row in rows]
        derived = client._derive(
            rows, arm="predictor", replicate=0, phase="warm", offset=600)
        observed = [len(client._TOKENIZER.encode(row["prompt"], add_special_tokens=False))
                    for row in derived]
        chunks = {tuple(client._TOKENIZER.encode(
            row["prompt"], add_special_tokens=False)[:256]) for row in derived}
        self.assertEqual(observed, original)
        self.assertEqual(len(chunks), len(rows))


if __name__ == "__main__":
    unittest.main()
