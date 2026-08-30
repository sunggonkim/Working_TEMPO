from __future__ import annotations

from pathlib import Path
import unittest

from eval.sota_4node import tempo_pd_capacity_router_v13 as capacity
from eval.sota_4node import tempo_pd_router_v1 as base


class CapacityTests(unittest.TestCase):
    def _core(self):
        return capacity.CreditCore(base.RouterConfig(
            mode=base.RouterMode.TEMPO_AUTO,
            local_url="http://local", remote_url="http://remote",
            tokenizer_url="http://tokenizer", served_model_name="m",
            model_id="m", model_revision="r", topology_id="t",
            remote_backend="b", classifier_version="v",
            decoder_load_bucket="load", kv_bytes_per_token=1,
        ))

    def test_one_remote_credit_then_local_and_release(self):
        core = self._core()
        first = core.decide(request_id="a", prompt_tokens=8, output_tokens=2)
        second = core.decide(request_id="b", prompt_tokens=8, output_tokens=2)
        self.assertEqual(first.route.value, "remote_prefill_live_kv")
        self.assertEqual(second.route.value, "decoder_local_recompute_or_cache")
        core.complete("a")
        third = core.decide(request_id="c", prompt_tokens=8, output_tokens=2)
        self.assertEqual(third.route.value, "remote_prefill_live_kv")

    def test_launcher_is_one_bounded_step(self):
        root = Path(__file__).resolve().parent
        launcher = (root / "run_tempo_pd_capacity_candidate_v13_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)


if __name__ == "__main__":
    unittest.main()
