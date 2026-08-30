from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from urllib import error

from eval.sota_4node import run_tempo_go_real_trace_stream as client


_CORE_PATH = Path(__file__).resolve().parents[2] / "tempo/mooncake_fast25_workload.py"
_SPEC = importlib.util.spec_from_file_location("_tempo_real_stream_test_core", _CORE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
core = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = core
_SPEC.loader.exec_module(core)


class RealTraceStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rows = (
            core.TraceRow(0, 1_000, 512, 2, (1,)),
            core.TraceRow(1, 1_250, 768, 3, (1, 2)),
        )
        receipt = {
            "trace_name": "tiny",
            "source_sha256": "a" * 64,
            "source_git_blob_sha1": "b" * 40,
            "source_bytes": 10,
            "source_requests": 2,
            "upstream_commit": "c" * 40,
            "upstream_url": "https://example.invalid/tiny.jsonl",
            "source_manifest_path": "/bounded/source.json",
            "source_manifest_sha256": "d" * 64,
        }
        spec = core.MaterializationSpec(
            trace_name="tiny",
            start_index=0,
            request_count=2,
            max_model_len=2_048,
            max_output_tokens=32,
            token_id_min=100,
            token_id_max_exclusive=2_000,
        )
        _workload, manifest, workload_bytes = core.build_population(
            rows, receipt, spec,
        )
        self.workload_path = self.root / "population.jsonl"
        self.manifest_path = self.root / "population.manifest.json"
        self.business_path = self.root / "business.json"
        self.workload_path.write_bytes(workload_bytes)
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.business_path.write_text(json.dumps({
            "schema": client.BUSINESS_PROFILE_SCHEMA,
            "assignment": {
                "mode": "weighted_cycle_v1",
                "tenant_cycle": ["latency", "background"],
                "remaining_deadline_ms": {
                    "latency": 1_000.0,
                    "interactive": None,
                    "batch": None,
                    "background": 8_000.0,
                },
            },
            "policy_inputs_excluded": [
                "future_arrivals", "oracle_route", "future_cache_evictions",
                "physical_switch_label",
            ],
        }), encoding="utf-8")
        self.old_environment = {
            name: os.environ.get(name)
            for name in (
                client.POPULATION_MANIFEST_ENV,
                client.WIRE_ARM_ENV,
                client.BUSINESS_PROFILE_ENV,
            )
        }
        os.environ[client.POPULATION_MANIFEST_ENV] = str(self.manifest_path)
        os.environ[client.WIRE_ARM_ENV] = "tempo"
        os.environ[client.BUSINESS_PROFILE_ENV] = str(self.business_path)

    def tearDown(self) -> None:
        for name, value in self.old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        client._REQUEST_METADATA.clear()
        self.temporary.cleanup()

    def test_population_maps_to_arm_and_business_wire_ids(self) -> None:
        items, workload_sha = client.load_token_workload(
            self.workload_path,
            default_max_tokens=2,
            request_rate=None,
        )
        self.assertEqual(workload_sha, hashlib.sha256(
            self.workload_path.read_bytes(),
        ).hexdigest())
        self.assertEqual(len(items), 2)
        self.assertTrue(items[0].request_id.startswith(
            "epd-tempo-latency-cache-natural-measured-real-mooncake-tiny-",
        ))
        self.assertIn("-background-", items[1].request_id)
        self.assertIsInstance(items[0].prompt, list)
        self.assertEqual(
            client._REQUEST_METADATA[items[0].request_id]["remaining_deadline_ms"],
            1_000.0,
        )

    def test_token_ids_and_deadline_reach_http_request(self) -> None:
        items, _digest = client.load_token_workload(
            self.workload_path,
            default_max_tokens=2,
            request_rate=None,
        )
        captured = {}

        def opener(http_request, **_kwargs):
            captured["body"] = json.loads(http_request.data)
            captured["headers"] = dict(http_request.headers)
            raise error.URLError("bounded test stop")

        record = client.execute_token_request(
            items[0],
            endpoint="http://127.0.0.1:1/v1/completions",
            served_model_name="test-model",
            run_start_ns=10,
            timeout_s=1.0,
            seed=7,
            api_key=None,
            opener=opener,
            clock_ns=lambda: 10,
            sleeper=lambda _seconds: None,
        )
        self.assertEqual(captured["body"]["prompt"], items[0].prompt)
        self.assertEqual(captured["headers"]["X-tempo-tenant-id"], "latency")
        self.assertEqual(
            captured["headers"]["X-tempo-remaining-deadline-ms"], "1000.0",
        )
        self.assertEqual(record["prompt_token_count"], len(items[0].prompt))
        self.assertFalse(record["valid"])

    def test_request_rate_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(client.base.ContractError, "explicit source arrivals"):
            client.load_token_workload(
                self.workload_path,
                default_max_tokens=2,
                request_rate=4.0,
            )

    def test_model_vocabulary_context_and_special_ids_are_checked(self) -> None:
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({
            "vocab_size": 2_048,
            "max_position_embeddings": 2_048,
        }), encoding="utf-8")
        tokenizer = model / "tokenizer_config.json"
        tokenizer.write_text(json.dumps({
            "added_tokens_decoder": {
                "2040": {"special": True},
            },
        }), encoding="utf-8")
        receipt = client.validate_model_token_contract(
            model, self.manifest_path,
        )
        self.assertTrue(receipt["token_interval_in_vocabulary"])
        self.assertTrue(receipt["token_interval_excludes_special_tokens"])
        tokenizer.write_text(json.dumps({
            "added_tokens_decoder": {
                "150": {"special": True},
            },
        }), encoding="utf-8")
        with self.assertRaisesRegex(client.base.ContractError, "special token"):
            client.validate_model_token_contract(model, self.manifest_path)


if __name__ == "__main__":
    unittest.main()
