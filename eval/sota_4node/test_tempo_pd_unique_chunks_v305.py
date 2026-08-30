import json
from pathlib import Path
import unittest

from transformers import AutoTokenizer

from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_v265 as base
from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_unique_chunks_v305 as unique


class UniqueChunkWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(
            "results/tempo_pd_cross_geometry_input_v216/workloads/validation.jsonl")
        cls.tokenizer = AutoTokenizer.from_pretrained(
            "models/Qwen2.5-7B-Instruct", local_files_only=True)

    def test_rewrite_preserves_every_prompt_token_count(self):
        old = base._rows(self.source, "measured")
        new = unique._rows(self.source, "measured")
        self.assertEqual(len(old), 48)
        self.assertEqual(len(new), 48)
        for before, after in zip(old, new, strict=True):
            self.assertEqual(
                len(self.tokenizer.encode(before["prompt"], add_special_tokens=False)),
                len(self.tokenizer.encode(after["prompt"], add_special_tokens=False)),
            )

    def test_every_region_marker_is_globally_unique(self):
        prompts = [row["prompt"] for phase in ("warm", "measured")
                   for row in unique._rows(self.source, phase)]
        markers = []
        for prompt in prompts:
            words = prompt.split()
            for index in range(len(words) - 17):
                window = words[index:index + 18]
                if all(word in {"A", "B"} for word in window):
                    markers.append(" ".join(window))
        expected = sum(row["unique_chunk_marker_count"]
                       for phase in ("warm", "measured")
                       for row in unique._rows(self.source, phase))
        self.assertEqual(len(markers), expected)
        self.assertEqual(len(set(markers)), expected)


if __name__ == "__main__":
    unittest.main()
