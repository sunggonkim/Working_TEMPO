from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import analyze_tempo_pd_c4_adaptive_screen as analyzer
from eval.sota_4node import build_tempo_pd_c4_adaptive_screen_manifest as manifest_builder
from eval.sota_4node import run_tempo_pd_c4_fixed_phase_client as c4
from tempo.pd_contention_workload import (
    ForegroundArm,
    VALIDATION_FOREGROUND_GEOMETRIES,
)


def _manifest():
    return {
        "phase_order": [phase.value for phase in c4.manifest_builder.PHASES],
        "measurement": {
            "e2e_slo_ms": 16_000.0,
            "ttft_slo_ms": 3_000.0,
            "tpot_slo_ms": 250.0,
        },
    }


def _samples(*, tempo_e2e_ms: float = 85.0):
    result = []
    ordinal = 0
    for replicate in (0, 1):
        for phase in c4.manifest_builder.PHASES:
            for geometry in VALIDATION_FOREGROUND_GEOMETRIES:
                arms = {
                    ForegroundArm.LOCAL.value: {
                        "request_id": f"local-{ordinal}",
                        "route": c4._LOCAL_ROUTE,
                        "ttft_ms": 20.0,
                        "e2e_ms": 100.0,
                        "tpot_ms": 5.0,
                    },
                    ForegroundArm.REMOTE.value: {
                        "request_id": f"remote-{ordinal}",
                        "route": c4._REMOTE_ROUTE,
                        "ttft_ms": 25.0,
                        "e2e_ms": 110.0,
                        "tpot_ms": 5.0,
                    },
                    ForegroundArm.PREDICTOR.value: {
                        "request_id": f"predictor-{ordinal}",
                        "route": c4._LOCAL_ROUTE,
                        "ttft_ms": 19.0,
                        "e2e_ms": 95.0,
                        "tpot_ms": 5.0,
                    },
                    ForegroundArm.TEMPO.value: {
                        "request_id": f"tempo-{ordinal}",
                        "route": (
                            c4._LOCAL_ROUTE
                            if ordinal % 2 == 0 else c4._REMOTE_ROUTE),
                        "ttft_ms": 18.0,
                        "e2e_ms": tempo_e2e_ms,
                        "tpot_ms": 5.0,
                    },
                }
                result.append({
                    "pair_key": f"r{replicate}:pair-{ordinal}",
                    "replicate": replicate,
                    "phase": phase.value,
                    "arrival_offset_ms": float(ordinal),
                    "prompt_tokens": geometry.prompt_tokens,
                    "output_tokens": geometry.output_tokens,
                    "cache_state": geometry.cache_state.value,
                    "ordinal": ordinal,
                    "output_text_sha256": "a" * 64,
                    "arms": arms,
                })
                ordinal += 1
    return result


class C4AdaptiveScreenAnalysisTest(unittest.TestCase):
    def test_screen_metrics_require_both_routes_and_frozen_gains(self):
        metrics, groups = analyzer._screen_metrics(_samples(), _manifest())
        self.assertEqual(len(groups), 36)
        self.assertEqual(
            metrics["strongest_fixed_name_calibration_only"], "local")
        self.assertGreaterEqual(
            metrics["mean_gain_vs_strongest_fixed"], 0.03)
        self.assertGreaterEqual(metrics["mean_gain_vs_predictor"], 0.02)
        self.assertTrue(metrics["authorizes_independent_validation"])
        self.assertTrue(all(metrics["screen_gates"].values()))

        failed, _ = analyzer._screen_metrics(
            _samples(tempo_e2e_ms=99.0), _manifest())
        self.assertFalse(
            failed["screen_gates"][
                "mean_gain_vs_predictor_at_least_2pct"])
        self.assertFalse(failed["authorizes_independent_validation"])

    def test_four_arm_pairing_fails_on_output_drift(self):
        blocks = {}
        for replicate in (0, 1):
            metadata = {
                "phase": "c0_cool",
                "tenant": "foreground",
                "arrival_offset_ms": 1.0,
                "prompt_tokens": 512,
                "output_tokens": 16,
                "cache_state": "miss",
                "ordinal": 0,
                "pair_key": f"r{replicate}:pair",
                "prompt_token_sha256": "b" * 64,
                "terminal_item": 0,
            }
            for arm in analyzer._ARMS:
                blocks[(replicate, arm)] = {
                    metadata["pair_key"]: {
                        "request_id": f"{arm}-{replicate}",
                        "metadata": dict(metadata),
                        "output_text_sha256": "a" * 64,
                        "route": (
                            c4._LOCAL_ROUTE
                            if arm != "remote" else c4._REMOTE_ROUTE),
                        "ttft_ms": 1.0,
                        "e2e_ms": 2.0,
                        "tpot_ms": 1.0,
                    },
                }
        paired = analyzer._paired_samples(blocks)
        self.assertEqual(len(paired), 2)
        blocks[(0, "tempo")]["r0:pair"]["output_text_sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "semantics/output"):
            analyzer._paired_samples(blocks)

    def test_bound_artifact_digest_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifact.json"
            path.write_text("{}\n", encoding="utf-8")
            expected = analyzer._sha256(path)
            self.assertEqual(
                analyzer._bound_path(
                    str(path.resolve()), expected,
                    name="test artifact", within=root),
                path.resolve(),
            )
            path.write_text('{"drift": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                analyzer._bound_path(
                    str(path.resolve()), expected,
                    name="test artifact", within=root)

    def test_preregistered_arm_order_constant_is_unchanged(self):
        self.assertEqual(
            manifest_builder.ARM_ORDER_BY_REPLICATE,
            (
                ("local", "remote", "predictor", "tempo"),
                ("tempo", "predictor", "remote", "local"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
