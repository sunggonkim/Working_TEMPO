from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node import analyze_tempo_pd_c3_coupled_abba as gate


class CoupledC3ABBAGateTest(unittest.TestCase):
    def _fixture(self, root: Path):
        raw = root / "raw.json"
        raw.write_text("{}\n", encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": gate.MANIFEST_SCHEMA,
            "performance_claim_allowed": False,
            "replicates": 2,
            "arm_order_policy": "paired_abba",
            "within_rate_block_order": [
                "local", "remote", "remote", "local"],
            "p_only_rates_per_s": [0.0, 4.0, 8.0, 12.0],
            "decoder_hot_rate_per_s": 22.4,
        }), encoding="utf-8")
        rows = []
        for rate in gate.RATES:
            for replicate in range(gate.REPETITIONS):
                winner = "remote" if rate == 0.0 else "local"
                rows.append({
                    "background_rate_per_s": rate,
                    "replicate_index": replicate,
                    "measured_arm_order": (
                        "local_remote" if replicate == 0 else "remote_local"),
                    "winner": winner,
                    "remote_gain_over_local": 0.10 if rate == 0.0 else -0.10,
                    "local_gain_over_remote": 0.10 if rate != 0.0 else -0.10,
                })
        characterization = root / "characterization.json"
        characterization.write_text(json.dumps({
            "schema": gate.CHARACTERIZATION_SCHEMA,
            "source": str(raw.resolve()),
            "repetitions_per_rate": 2,
            "arm_order_policy": "paired_abba",
            "all_measured_requests_valid": True,
            "p_only_source_compute_attribution": {
                "long_producer_prefill_removed": True,
                "expected_residual_recompute_tokens_per_request": 1,
                "zero_producer_compute_claim_allowed": False,
            },
            "paired_replicate_summary": rows,
        }), encoding="utf-8")
        result = root / "result.json"
        result.write_text(json.dumps({
            "schema": gate.RESULT_SCHEMA,
            "raw": str(raw.resolve()),
            "block_count": 16,
            "stopped_after_first_invalid_block": None,
            "performance_claim_allowed": False,
            "physical_switch_bottleneck_claim_allowed": False,
            "repetitions_per_rate": 2,
            "arm_order_policy": "paired_abba",
            "paired_semantic_schedules_exact": True,
            "coupled_manifest": str(manifest.resolve()),
            "coupled_manifest_sha256": hashlib.sha256(
                manifest.read_bytes()).hexdigest(),
        }), encoding="utf-8")
        return result, characterization, manifest

    def test_valid_abba_crossover_authorizes_c4(self):
        with tempfile.TemporaryDirectory() as directory:
            result, characterization, manifest = self._fixture(Path(directory))
            value = gate.evaluate(
                result_path=result,
                characterization_path=characterization,
                manifest_path=manifest,
            )
            self.assertTrue(value["c3_coupled_characterization_valid"])
            self.assertTrue(value["authorizes_c4_phase_trace"])
            self.assertFalse(value["performance_claim_allowed"])

    def test_arm_order_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result, characterization, manifest = self._fixture(Path(directory))
            value = json.loads(characterization.read_text(encoding="utf-8"))
            value["paired_replicate_summary"][1][
                "measured_arm_order"] = "local_remote"
            characterization.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "replicate 1 order differs"):
                gate.evaluate(
                    result_path=result,
                    characterization_path=characterization,
                    manifest_path=manifest,
                )


if __name__ == "__main__":
    unittest.main()
