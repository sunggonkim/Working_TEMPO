from pathlib import Path
import unittest

from transformers import AutoTokenizer

from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_v265 as original
from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_unique_chunks_v308 as revised


class RevisedUniqueChunksTests(unittest.TestCase):
    def test_all_prompt_geometries_are_exactly_preserved(self):
        source = Path(
            "results/tempo_pd_cross_geometry_input_v216/workloads/validation.jsonl")
        tokenizer = AutoTokenizer.from_pretrained(
            "models/Qwen2.5-7B-Instruct", local_files_only=True)
        for phase in ("warm", "measured"):
            before = original._rows(source, phase)
            after = revised._rows(source, phase)
            self.assertEqual(len(before), len(after))
            for left, right in zip(before, after, strict=True):
                self.assertEqual(
                    len(tokenizer.encode(left["prompt"], add_special_tokens=False)),
                    len(tokenizer.encode(right["prompt"], add_special_tokens=False)),
                )

    def test_marker_encoding_is_unique_and_punctuation_delimited(self):
        values = [revised._marker(index) for index in range(1 << 14)]
        self.assertEqual(len(set(values)), len(values))
        self.assertTrue(all(value.endswith(".") for value in values))
        self.assertTrue(all(len(value[:-1].split()) == 18 for value in values))


if __name__ == "__main__":
    unittest.main()
