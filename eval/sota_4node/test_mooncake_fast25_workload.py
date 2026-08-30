from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


# The repository's historical ``tempo.__init__`` eagerly imports the PyTorch
# backend.  Load this pure-stdlib module directly so its CPU contract tests can
# run on a login node without treating torch availability as trace evidence.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "tempo/mooncake_fast25_workload.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_tempo_mooncake_fast25_workload", _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
MaterializationSpec = _MODULE.MaterializationSpec
TraceContractError = _MODULE.TraceContractError
TraceRow = _MODULE.TraceRow
build_population = _MODULE.build_population
load_trace = _MODULE.load_trace
verify_population = _MODULE.verify_population


def _canonical_lines(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        for row in rows
    )


def _git_blob_sha1(value: bytes) -> str:
    return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()


class MooncakeSourceTest(unittest.TestCase):
    def _source(self, root: Path) -> tuple[Path, Path]:
        rows = [
            {
                "timestamp": 100,
                "input_length": 512,
                "output_length": 2,
                "hash_ids": [7],
            },
            {
                "timestamp": 200,
                "input_length": 513,
                "output_length": 3,
                "hash_ids": [7, 8],
            },
        ]
        raw = _canonical_lines(rows)
        trace_path = root / "tiny.jsonl"
        trace_path.write_bytes(raw)
        manifest = {
            "schema": "tempo-go-mooncake-fast25-source-manifest-v1",
            "upstream": {"commit": "a" * 40},
            "trace_contract": {
                "block_size_tokens": 512,
                "fields": [
                    "timestamp", "input_length", "output_length", "hash_ids",
                ],
            },
            "files": {
                "tiny": {
                    "bytes": len(raw),
                    "git_blob_sha1": _git_blob_sha1(raw),
                    "local_path": "tiny.jsonl",
                    "requests": len(rows),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "url": "https://example.invalid/tiny.jsonl",
                },
            },
        }
        manifest_path = root / "source.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, trace_path

    def test_source_identity_and_geometry_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, trace_path = self._source(Path(temporary))
            rows, receipt = load_trace(manifest_path, "tiny")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1].hash_ids, (7, 8))
            self.assertEqual(receipt["source_git_blob_sha1"], _git_blob_sha1(
                trace_path.read_bytes(),
            ))
            trace_path.write_bytes(trace_path.read_bytes() + b" ")
            with self.assertRaisesRegex(TraceContractError, "byte count"):
                load_trace(manifest_path, "tiny")


class MooncakePopulationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = (
            TraceRow(10, 1_000, 1_024, 1, (1, 2)),
            TraceRow(11, 1_100, 1_280, 700, (1, 2, 3)),
            TraceRow(12, 1_300, 512, 8, (4,)),
        )
        self.receipt = {
            "trace_name": "tiny",
            "source_sha256": "b" * 64,
            "source_git_blob_sha1": "c" * 40,
            "source_bytes": 1,
            "source_requests": 3,
            "upstream_commit": "d" * 40,
            "upstream_url": "https://example.invalid/tiny.jsonl",
            "source_manifest_path": "/bounded/source.json",
            "source_manifest_sha256": "e" * 64,
        }
        self.spec = MaterializationSpec(
            trace_name="tiny",
            start_index=0,
            request_count=3,
            arrival_load_multiplier=2.0,
            max_model_len=1_200,
            min_output_tokens=2,
            max_output_tokens=128,
            context_policy="prefix_clip",
            token_id_min=100,
            token_id_max_exclusive=2_000,
            max_materialized_tokens=10_000,
        )

    def test_prefix_relation_arrival_and_adjustments_are_preserved(self) -> None:
        workload, manifest, raw = build_population(
            self.rows, self.receipt, self.spec,
        )
        self.assertEqual(
            [row["arrival_offset_ms"] for row in workload], [0.0, 50.0, 150.0],
        )
        self.assertEqual(workload[0]["max_tokens"], 2)
        self.assertEqual(workload[1]["max_tokens"], 128)
        self.assertEqual(len(workload[0]["prompt"]), 1_024)
        self.assertEqual(len(workload[1]["prompt"]), 1_072)
        self.assertEqual(
            workload[0]["prompt"], workload[1]["prompt"][:1_024],
        )
        self.assertNotEqual(
            workload[0]["prompt"][:512], workload[2]["prompt"],
        )
        self.assertEqual(manifest["context"]["clipped_requests"], 1)
        self.assertEqual(manifest["context"]["clipped_input_tokens"], 208)
        self.assertEqual(manifest["output"]["floor_adjusted_requests"], 1)
        self.assertEqual(manifest["output"]["cap_adjusted_requests"], 1)
        self.assertEqual(
            manifest["request_index"]["mooncake-tiny-000011"]
            ["prior_reusable_prefix_tokens_upper_bound"],
            1_024,
        )
        self.assertTrue(verify_population(raw, manifest)["all_rows_valid"])

    def test_materialization_is_byte_deterministic(self) -> None:
        first = build_population(self.rows, self.receipt, self.spec)
        second = build_population(self.rows, self.receipt, self.spec)
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[1], second[1])

    def test_reject_policy_does_not_silently_clip(self) -> None:
        spec = MaterializationSpec(
            trace_name="tiny",
            start_index=1,
            request_count=1,
            max_model_len=1_200,
            max_output_tokens=128,
            context_policy="reject",
            token_id_min=100,
            token_id_max_exclusive=2_000,
        )
        with self.assertRaisesRegex(TraceContractError, "exceeds max_model_len"):
            build_population(self.rows, self.receipt, spec)

    def test_workload_tamper_fails_sha_gate(self) -> None:
        _workload, manifest, raw = build_population(
            self.rows, self.receipt, self.spec,
        )
        with self.assertRaisesRegex(TraceContractError, "SHA-256"):
            verify_population(raw + b" ", manifest)


class MooncakePinnedDataTest(unittest.TestCase):
    def test_all_downloaded_official_sources_match_manifest(self) -> None:
        root = Path(__file__).resolve().parent / "data/mooncake_fast25"
        manifest = root / "source_manifest_v1.json"
        expected = {
            "conversation": 12_031,
            "toolagent": 23_608,
            "synthetic": 3_993,
        }
        for trace_name, count in expected.items():
            with self.subTest(trace=trace_name):
                rows, _receipt = load_trace(manifest, trace_name)
                self.assertEqual(len(rows), count)

    def test_source_only_window_selection_is_deterministic(self) -> None:
        from eval.sota_4node.select_tempo_go_mooncake_windows import (
            DEFAULT_SOURCE_MANIFEST,
            WINDOW_SIZE,
            select_windows,
        )

        result = select_windows(DEFAULT_SOURCE_MANIFEST)
        self.assertFalse(
            result["window_contract"]["selection_uses_inference_performance"],
        )
        expected = {
            "conversation_long_decode": ("conversation", 896),
            "toolagent_prefix_reuse": ("toolagent", 4992),
            "toolagent_native_burst": ("toolagent", 22912),
            "synthetic_zero_reuse_burst": ("synthetic", 2304),
        }
        for name, (trace, start) in expected.items():
            with self.subTest(window=name):
                entry = result["selected"][name]
                self.assertEqual(entry["trace"], trace)
                self.assertEqual(entry["metrics"]["start_index"], start)
                self.assertEqual(entry["metrics"]["request_count"], WINDOW_SIZE)
        synthetic = result["selected"]["synthetic_zero_reuse_burst"]["metrics"]
        self.assertEqual(
            synthetic["reuse"]["requests_with_prior_reusable_prefix"], 0,
        )


if __name__ == "__main__":
    unittest.main()
