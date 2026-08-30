from pathlib import Path
import unittest
from unittest.mock import patch

from eval.sota_4node import tempo_pd_pressure_router_v25 as pressure
from eval.sota_4node import tempo_pd_router_v1 as base


class PressureV25Tests(unittest.TestCase):
    def _core(self):
        config = base.RouterConfig(
            mode=base.RouterMode.TEMPO_AUTO, local_url="http://l",
            remote_url="http://r", tokenizer_url="http://t",
            served_model_name="m", model_id="m", model_revision="x",
            topology_id="t", remote_backend="b", classifier_version="c",
            decoder_load_bucket="high", kv_bytes_per_token=1,
        )
        return pressure.PressureCore(config)

    def test_remote_once_only_after_two_local_inflight(self):
        core = self._core()
        rows = [core.decide(request_id=str(i), prompt_tokens=100,
                            output_tokens=32) for i in range(5)]
        self.assertEqual([row.route.value for row in rows], [
            "decoder_local_recompute_or_cache", "decoder_local_recompute_or_cache",
            "remote_prefill_live_kv", "decoder_local_recompute_or_cache",
            "decoder_local_recompute_or_cache",
        ])
        core.complete("2")
        row = core.decide(request_id="5", prompt_tokens=100, output_tokens=32)
        self.assertEqual(row.route.value, "decoder_local_recompute_or_cache")

    def test_launcher_one_step(self):
        text = Path(__file__).with_name(
            "run_tempo_pd_pressure_v25_in_allocation.sh"
        ).read_text()
        self.assertEqual(text.count("srun "), 1)
        self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
