#!/usr/bin/env python3
"""CPU-only contract tests for the three-run 4-node campaign analyzer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from eval.sota_4node.analyze_vllm_lmcache_4node_campaign_v1 import (
    BACKGROUND_BYTES,
    ContractError,
    RunInput,
    analyze_runs,
    load_runs,
    main,
    parse_run_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "eval/sota_4node/fixture_vllm_lmcache_4node_campaign_run_v1.json"
SCHEMA_PATH = REPO_ROOT / "eval/sota_4node/vllm_lmcache_4node_campaign_analysis_schema_v1.json"


def fixture_payload() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def three_runs() -> list[RunInput]:
    return [
        RunInput(label=label, path=f"fixture-{label}.json", payload=copy.deepcopy(fixture_payload()))
        for label in ("campaign_a", "campaign_b", "campaign_c")
    ]


class CampaignAnalyzerTests(unittest.TestCase):
    def test_three_runs_pool_nine_and_pair_same_prompt(self) -> None:
        result = analyze_runs(three_runs())

        self.assertFalse(result["promotion_valid"])
        self.assertEqual(result["analysis_method"], "descriptive_robust_no_bootstrap")
        self.assertTrue(result["experimental_design"]["same_allocation_three_campaigns"])
        self.assertFalse(result["experimental_design"]["allocation_independence"])
        self.assertEqual(result["experimental_design"]["pooled_samples_per_mode"], 9)

        tempo = result["pooled_modes"]["tempo_group2"]
        self.assertEqual(tempo["samples"], 9)
        self.assertEqual(tempo["metrics"]["ttft_ms"]["median"], 65.0)
        self.assertEqual(tempo["metrics"]["ttft_ms"]["min"], 64.0)
        self.assertEqual(tempo["metrics"]["ttft_ms"]["max"], 66.0)
        self.assertEqual(tempo["metrics"]["ttft_ms"]["mad"], 1.0)

        for run in result["runs"]:
            summary = run["mode_summaries"]["tempo_group2"]["metrics"]["ttft_ms"]
            self.assertEqual(summary, {"samples": 3, "p50": 65.0, "max": 66.0})
            self.assertEqual(run["checks"]["background_bytes_per_block"], BACKGROUND_BYTES)
            self.assertEqual(run["checks"]["source_calls_per_background_block"], 8)

        greedy = result["paired_comparisons"]["tempo_group2_vs_lmcache_greedy"]["metrics"]
        self.assertEqual(greedy["ttft_ms"]["paired_samples"], 9)
        self.assertEqual(greedy["ttft_ms"]["win_rate"], 1.0)
        self.assertEqual(greedy["ttft_ms"]["tempo_minus_baseline"]["median"], -6.0)
        self.assertEqual(greedy["background_finish_from_request_start_ms"]["win_rate"], 1.0)

        foreground_finish = result["paired_comparisons"]["tempo_group2_vs_fg_only"]["metrics"][
            "background_finish_from_request_start_ms"
        ]
        self.assertIsNone(foreground_finish["win_rate"])
        self.assertIn("not_comparable", foreground_finish["comparison_interpretation"])
        self.assertFalse(result["validation"]["all_tempo_no_post_foreground_drain_met"])
        self.assertFalse(result["validation"]["all_tempo_runtime_gates_met"])

    def test_exactly_three_distinct_runs_are_required(self) -> None:
        with self.assertRaisesRegex(ContractError, "exactly 3"):
            analyze_runs(three_runs()[:2])
        duplicate = three_runs()
        duplicate[2] = RunInput("campaign_a", "different.json", duplicate[2].payload)
        with self.assertRaisesRegex(ContractError, "labels must be unique"):
            analyze_runs(duplicate)

    def test_rejects_mixed_allocations(self) -> None:
        runs = three_runs()
        changed = copy.deepcopy(runs[2].payload)
        changed["allocation_id"] = "different-allocation"
        runs[2] = RunInput(runs[2].label, runs[2].path, changed)
        with self.assertRaisesRegex(ContractError, "not from one allocation"):
            analyze_runs(runs)

    def test_rejects_wrong_bytes(self) -> None:
        runs = three_runs()
        changed = copy.deepcopy(runs[0].payload)
        changed["blocks"][1]["receiver_verified_bytes"] -= 1
        runs[0] = RunInput(runs[0].label, runs[0].path, changed)
        with self.assertRaisesRegex(ContractError, "receiver_verified_bytes"):
            analyze_runs(runs)

    def test_rejects_missing_source_call_evidence(self) -> None:
        runs = three_runs()
        changed = copy.deepcopy(runs[0].payload)
        del changed["blocks"][1]["background_source_calls"]
        runs[0] = RunInput(runs[0].label, runs[0].path, changed)
        with self.assertRaisesRegex(ContractError, "source-call field"):
            analyze_runs(runs)

    def test_rejects_output_mismatch(self) -> None:
        runs = three_runs()
        changed = copy.deepcopy(runs[0].payload)
        changed["blocks"][2]["output_token_sha256"] = "different-output"
        runs[0] = RunInput(runs[0].label, runs[0].path, changed)
        with self.assertRaisesRegex(ContractError, "output mismatch"):
            analyze_runs(runs)

    def test_rejects_declared_gate_that_disagrees_with_blocks(self) -> None:
        runs = three_runs()
        changed = copy.deepcopy(runs[0].payload)
        changed["candidate_no_post_foreground_drain_met"] = True
        runs[0] = RunInput(runs[0].label, runs[0].path, changed)
        with self.assertRaisesRegex(ContractError, "disagrees"):
            analyze_runs(runs)

    def test_run_spec_and_file_loader_require_explicit_three_paths(self) -> None:
        self.assertEqual(parse_run_spec("r1=/tmp/result.json"), ("r1", Path("/tmp/result.json")))
        with self.assertRaisesRegex(ContractError, "LABEL=PATH"):
            parse_run_spec("result.json")
        with self.assertRaisesRegex(ContractError, "exactly 3"):
            load_runs([])

    def test_cli_writes_schema_conforming_core(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["promotion_valid"]["const"], False)
        self.assertEqual(schema["properties"]["contract"]["properties"]["nodes"]["const"], 4)

        with tempfile.TemporaryDirectory() as raw_dir:
            directory = Path(raw_dir)
            specs = []
            for label in ("a", "b", "c"):
                path = directory / f"{label}.json"
                path.write_text(json.dumps(fixture_payload()), encoding="utf-8")
                specs.extend(("--run", f"{label}={path}"))
            output = directory / "analysis.json"
            return_code = main([*specs, "--output", str(output)])
            self.assertEqual(return_code, 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["schema_version"], schema["properties"]["schema_version"]["const"])
            self.assertEqual(len(result["runs"]), 3)
            self.assertEqual(result["pooled_modes"]["lmcache_greedy"]["samples"], 9)


if __name__ == "__main__":
    unittest.main()
