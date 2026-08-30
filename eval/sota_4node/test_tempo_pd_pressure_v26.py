import unittest
from eval.sota_4node.test_tempo_pd_pressure_v25 import PressureV25Tests
from eval.sota_4node import tempo_pd_pressure_router_v26 as v26


class PressureV26Tests(PressureV25Tests):
    def _core(self):
        core = super()._core()
        core.__class__ = v26.PressureEpochCore
        return core

    def test_rearms_only_after_all_requests_finish(self):
        core = self._core()
        rows = [core.decide(request_id=str(i), prompt_tokens=100,
                            output_tokens=32) for i in range(3)]
        self.assertEqual(rows[2].route.value, "remote_prefill_live_kv")
        for i in range(3):
            core.complete(str(i))
        rows = [core.decide(request_id=str(i), prompt_tokens=100,
                            output_tokens=32) for i in range(3, 6)]
        self.assertEqual(rows[2].route.value, "remote_prefill_live_kv")


if __name__ == "__main__":
    unittest.main()
