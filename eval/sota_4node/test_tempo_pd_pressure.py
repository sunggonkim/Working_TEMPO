from pathlib import Path
import unittest

from eval.sota_4node import tempo_pd_elastic_router as router


class TempoPDPressureTest(unittest.TestCase):
    def test_penalties_scale_on_separate_pressure_axes(self):
        local, remote = router.pressure_penalties_ms(
            prompt_tokens=2048,
            local_pressure=0.5,
            fabric_pressure=0.25,
            local_ms_per_prompt_token=0.04,
            remote_ms_per_prompt_token=0.08,
        )
        self.assertAlmostEqual(local, 40.96)
        self.assertAlmostEqual(remote, 40.96)

    def test_penalties_reject_nonphysical_pressure(self):
        with self.assertRaisesRegex(ValueError, "fabric_pressure"):
            router.pressure_penalties_ms(
                prompt_tokens=512,
                local_pressure=0.0,
                fabric_pressure=1.1,
                local_ms_per_prompt_token=0.04,
                remote_ms_per_prompt_token=0.08,
            )

    def test_predictor_keeps_static_estimate_and_tempo_uses_hook(self):
        source = (
            Path(__file__).resolve().parent
            / "tempo_pd_elastic_router_v444.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "route, reason = self._predictor_route(row, estimate)", source)
        self.assertIn(
            "estimate = self._tempo_estimate(", source)
        self.assertIn(
            "self._tempo_estimate(request_id, row, remaining_deadline_ms)",
            source,
        )


if __name__ == "__main__":
    unittest.main()
