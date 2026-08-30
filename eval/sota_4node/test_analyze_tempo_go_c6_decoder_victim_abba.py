from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from eval.sota_4node import analyze_tempo_go_c6_decoder_victim_abba as analyzer


class C6DecoderVictimABBAAnalysisTests(unittest.TestCase):
    def _contract(self, root: Path) -> Path:
        source = Path(__file__).with_name("tempo_go_c6_qualification_contract_v1.json")
        value = json.loads(source.read_text(encoding="utf-8"))
        value["decoder_victim_abba"]["victim"]["offered_rate_per_s"] = 0.1
        value["decoder_victim_abba"]["aggressor"]["offered_rate_per_s"] = 0.1
        path = root / "contract.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _block(
        self,
        root: Path,
        *,
        name: str,
        aggressor: bool,
        replicate: int,
        decode_completion_ms: float,
    ) -> Path:
        request_index = {}
        requests = []
        decisions = []
        for ordinal in range(6):
            request_id = f"victim-{name}-{ordinal}"
            request_index[request_id] = {
                "tenant": "foreground",
                "ordinal": ordinal,
                "arrival_offset_ms": (ordinal + 0.5) * 10_000.0,
                "prompt_tokens": 4094,
                "output_tokens": 128,
                "cache_state": "miss",
                "arm": "remote",
            }
            first_ns = 100_000_000
            step_ns = int(decode_completion_ms * 1_000_000 / 127)
            arrivals = [first_ns + index * step_ns for index in range(128)]
            requests.append({
                "request_id": request_id,
                "valid": True,
                "router": {"route": analyzer.fixed.REMOTE_ROUTE},
                "dispatch_offset_ns": 0,
                "stream_end_offset_ns": arrivals[-1] + 1_000_000,
                "token_arrival_offsets_ns": arrivals,
                "output_token_values": ["x"] * 128,
            })
            decisions.append({"request_id": request_id, "route": analyzer.fixed.REMOTE_ROUTE})
        if aggressor:
            for ordinal in range(6):
                request_id = f"aggressor-{name}-{ordinal}"
                request_index[request_id] = {
                    "tenant": "decoder_hot",
                    "ordinal": ordinal,
                    "arrival_offset_ms": (ordinal + 0.5) * 10_000.0,
                    "prompt_tokens": 4094,
                    "output_tokens": 2,
                    "cache_state": "miss",
                    "arm": "local",
                }
                requests.append({
                    "request_id": request_id,
                    "valid": True,
                    "router": {"route": analyzer.fixed.LOCAL_ROUTE},
                    "output_token_values": ["x", "x"],
                })
                decisions.append({"request_id": request_id, "route": analyzer.fixed.LOCAL_ROUTE})
        raw = {
            "schema": analyzer.RAW_SCHEMA,
            "validation": {"performance_claim_allowed": True},
            "requests": requests,
            "router_decisions": decisions,
            "endpoint_evidence": {},
            "c6_decoder_victim_contract": {
                "schema": analyzer.BLOCK_SCHEMA,
                "name": name,
                "aggressor": aggressor,
                "replicate": replicate,
                "request_index": request_index,
            },
        }
        path = root / f"{name}.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def _bundle(self, root: Path, loaded_ms: float) -> dict:
        specs = [
            ("00_decoder_victim_clean_a", False, 0, 1270.0),
            ("01_decoder_victim_hot_a", True, 0, loaded_ms),
            ("02_decoder_victim_hot_b", True, 1, loaded_ms),
            ("03_decoder_victim_clean_b", False, 1, 1270.0),
        ]
        return {
            "schema": analyzer.BUNDLE_SCHEMA,
            "artifacts": {
                name: str(self._block(
                    root,
                    name=name,
                    aggressor=aggressor,
                    replicate=replicate,
                    decode_completion_ms=duration,
                ))
                for name, aggressor, replicate, duration in specs
            },
        }

    @mock.patch.object(analyzer.fixed, "_cold_completion_valid", return_value=True)
    @mock.patch.object(analyzer.fixed, "_validate_endpoint_evidence_bundle")
    def test_material_output_completion_slowdown_passes(self, _evidence, _cold) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = analyzer.analyze_bundle(
                self._bundle(root, 2540.0), self._contract(root)
            )
        self.assertTrue(value["q1_decoder_output_completion_victim_pass"])
        self.assertTrue(value["q3_service_horizon_pass"])
        self.assertGreaterEqual(
            value["aggregate_effect"]["median_p50_degradation_fraction"], 0.99
        )

    @mock.patch.object(analyzer.fixed, "_cold_completion_valid", return_value=True)
    @mock.patch.object(analyzer.fixed, "_validate_endpoint_evidence_bundle")
    def test_non_material_slowdown_does_not_promote(self, _evidence, _cold) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = analyzer.analyze_bundle(
                self._bundle(root, 1397.0), self._contract(root)
            )
        self.assertFalse(value["q1_decoder_output_completion_victim_pass"])
        self.assertFalse(value["controller_performance_run_allowed"])


if __name__ == "__main__":
    unittest.main()
