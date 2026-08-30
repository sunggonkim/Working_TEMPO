from pathlib import Path
import json
import unittest
from unittest.mock import Mock

from eval.sota_4node import run_tempo_pd_stream_metrics_forced_v32 as forced


class ForcedV36Tests(unittest.TestCase):
    def test_forced_opener_injects_exact_token_bias(self):
        captured = {}

        def opener(req, **kwargs):
            captured.update(json.loads(req.data))
            raise RuntimeError("captured")

        item = Mock(arrival_offset_ns=0, max_tokens=32, index=0,
                    prompt="p", request_id="r")
        record = forced.execute_request(
            item, endpoint="http://x", served_model_name="m", run_start_ns=0,
            timeout_s=1, seed=1, api_key=None, opener=opener,
            clock_ns=lambda: 1, sleeper=lambda _: None,
        )
        self.assertEqual(captured["logit_bias"], {"362": 100.0})
        self.assertIn("captured", record["error"])

    def test_two_launchers_each_use_one_step(self):
        root = Path(__file__).resolve().parent
        for name in ("run_tempo_pd_forced_crossover_v33_in_allocation.sh",
                     "run_tempo_pd_final_forced_v35_in_allocation.sh"):
            text = (root / name).read_text()
            self.assertEqual(text.count("srun "), 1)
            self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
