from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node import run_tempo_pd_c4_phase_screen_client as c4


class PairedOutputGateTest(unittest.TestCase):
    def test_controller_reset_evidence_requires_owned_and_external_quiescence(
        self,
    ) -> None:
        resources = {
            "local_token_ms": 0,
            "remote_prefill_token_ms": 0,
            "remote_kv_bytes": 0,
            "remote_semantic_ops": 0,
        }
        value = {
            "success": True,
            "controller_generation": 3,
            "controller": {
                "inflight": 0,
                "external_inflight": 0,
                "resources": dict(resources),
                "owned_resources": dict(resources),
                "external_resources": dict(resources),
            },
        }
        self.assertEqual(
            c4._validate_controller_reset_evidence([value, value]), [3, 3])
        value["controller"]["external_inflight"] = 1
        with self.assertRaisesRegex(ValueError, "not quiescent"):
            c4._validate_controller_reset_evidence([value])

    def test_stream_token_strings_are_compared_without_integer_coercion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts: dict[str, str] = {}
            contracts: dict[str, dict[str, object]] = {}
            for arm in c4.ARMS:
                key = arm.value
                request_id = f"request-{key}"
                path = root / f"{key}.json"
                path.write_text(json.dumps({
                    "requests": [{
                        "request_id": request_id,
                        "output_text_sha256": "output-digest",
                        "output_token_values": [" A", " B"],
                        "prompt_sha256": "prompt-digest",
                    }],
                }), encoding="utf-8")
                artifacts[key] = str(path)
                contracts[key] = {
                    "replicate": 0,
                    "arm": key,
                    "semantic_schedule_sha256": "schedule-digest",
                    "request_index": {
                        request_id: {"pair_key": "paired-0"},
                    },
                }

            result = c4._paired_output_gate(artifacts, contracts)

            self.assertEqual(result["paired_foreground_requests"], 1)
            self.assertTrue(result["all_four_arms_present"])
            self.assertEqual(result["failures"], [])


if __name__ == "__main__":
    unittest.main()
