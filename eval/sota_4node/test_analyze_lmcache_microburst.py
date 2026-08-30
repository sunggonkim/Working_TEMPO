from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node import analyze_lmcache_microburst as analyzer


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _write_run(
    directory: Path,
    *,
    greedy_tails=(3.9, 4.2, 4.7, 5.2),
    tempo_tails=(3.0, 3.5, 4.0, 4.5),
    tempo_drain=0.0,
) -> Path:
    directory.mkdir()
    tails = {
        "fg_only": (2.0, 2.1, 2.2, 2.3),
        "lmcache_greedy": tuple(greedy_tails),
        "lmcache_static_serial": (4.0, 4.1, 4.2, 4.3),
        "tempo_epoch": tuple(tempo_tails),
    }
    finishes = {
        "fg_only": 0.0,
        "lmcache_greedy": 20.0,
        "lmcache_static_serial": 24.0,
        "tempo_epoch": 18.0,
    }
    config = {
        "tokens": 4,
        "requests": 2,
        "kv_bytes": 32 << 20,
        "hot_path_global_control": False,
    }
    rank_blocks = []
    for block_index, mode in enumerate(analyzer.BLOCK_MODES):
        rank_blocks.append(
            {
                "block_index": block_index,
                "mode": mode,
                "token_latency_ms": list(tails[mode]),
                "correctness_met": True,
                "transfer_errors": [],
                "schedule_start_adherence_met": True,
                "plan_deadline_met": True,
                "post_foreground_drain_ms": tempo_drain if mode == "tempo_epoch" else 0.0,
            }
        )
    for rank in range(analyzer.WORLD_SIZE):
        record = {
            "schema_version": analyzer.RANK_SCHEMA,
            "rank": rank,
            "world_size": analyzer.WORLD_SIZE,
            "nodes": analyzer.NODES,
            "config": config,
            "blocks": rank_blocks,
        }
        (directory / f"rank_{rank}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    result_blocks = []
    for block_index, mode in enumerate(analyzer.BLOCK_MODES):
        values = list(tails[mode])
        result_blocks.append(
            {
                "block_index": block_index,
                "mode": mode,
                "global_token_tail_p50_ms": _median(values),
                "global_token_tail_p99_ms": max(values),
                "background_finish_from_block_start_ms": finishes[mode],
                "post_foreground_drain_ms": tempo_drain if mode == "tempo_epoch" else 0.0,
                "schedule_start_adherence_met": True,
                "plan_deadline_met": True,
                "transfer_errors": [],
                "correctness_met": True,
            }
        )
    result_modes = {}
    for mode in analyzer.MODE_ORDER:
        values = list(tails[mode]) * 4
        result_modes[mode] = {
            "global_token_tail_p50_ms": _median(values),
            "global_token_tail_p99_ms": max(values),
            "background_finish_p50_ms": finishes[mode],
            "background_finish_p99_ms": finishes[mode],
            "post_foreground_drain_p99_ms": tempo_drain if mode == "tempo_epoch" else 0.0,
            "schedule_start_adherence_met": True,
            "plan_deadline_met": True,
            "correctness_met": True,
        }
    result = {
        "schema_version": analyzer.RESULT_SCHEMA,
        "evidence_state": "live_official_component_with_compatibility_shim",
        "world_size": analyzer.WORLD_SIZE,
        "nodes": analyzer.NODES,
        "block_sequence": list(analyzer.BLOCK_MODES),
        "config": config,
        "baseline": {"name": "LMCache NixlChannel", "proxy": False},
        "scheduler_semantics": {"hot_path_global_control": False},
        "blocks": result_blocks,
        "modes": result_modes,
        "tempo_epoch_execution_valid": True,
        "screen_outcome": "valid_measurement_requires_performance_comparison",
        "overall_correctness_met": True,
    }
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


class AnalyzeLmcacheMicroburstTest(unittest.TestCase):
    def test_two_valid_runs_pass_conservative_promising_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _write_run(root / "run_a")
            second = _write_run(root / "run_b")
            report = analyzer.analyze_paths([first, second])

        self.assertEqual(report["verdict"], "promising")
        self.assertEqual(report["valid_run_count"], 2)
        self.assertTrue(report["gates"]["every_run_tail_p99_improves"])
        self.assertTrue(report["gates"]["aggregate_4ms_slo_goodput_improves"])
        self.assertGreaterEqual(
            report["runs"][0]["tempo_vs_lmcache_greedy"]["tail_p99_improvement_percent"],
            10.0,
        )
        goodput = report["runs"][0]["mode_metrics"]
        self.assertEqual(
            goodput["lmcache_greedy"]["slo_goodput"]["4.0"]["successful_global_tokens"],
            4,
        )
        self.assertEqual(
            goodput["tempo_epoch"]["slo_goodput"]["4.0"]["successful_global_tokens"],
            12,
        )
        paired = report["aggregate"]["tempo_vs_lmcache_greedy"]["paired_occurrence_wins"]
        self.assertEqual(paired["occurrences"], 8)
        self.assertEqual(paired["tempo_tail_wins"], 8)
        self.assertIn("same-allocation synthetic", report["evidence_scope"]["measurement"])

    def test_valid_but_slower_tempo_is_killed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_path = _write_run(
                Path(temporary) / "run",
                greedy_tails=(3.0, 3.2, 3.4, 3.6),
                tempo_tails=(3.1, 3.3, 3.5, 3.8),
            )
            report = analyzer.analyze_paths([result_path])

        self.assertEqual(report["verdict"], "kill")
        self.assertTrue(report["gates"]["all_runs_valid"])
        self.assertFalse(report["gates"]["every_run_tail_p99_improves"])
        self.assertFalse(report["gates"]["aggregate_4ms_slo_goodput_improves"])

    def test_rank_correctness_mismatch_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            result_path = _write_run(run_dir)
            rank_path = run_dir / "rank_3.json"
            rank = json.loads(rank_path.read_text(encoding="utf-8"))
            rank["blocks"][0]["correctness_met"] = False
            rank_path.write_text(json.dumps(rank), encoding="utf-8")
            report = analyzer.analyze_paths([result_path])

        self.assertEqual(report["verdict"], "inconclusive")
        self.assertEqual(report["valid_run_count"], 0)
        self.assertIn("correctness_met", report["runs"][0]["validation_errors"][0])

    def test_tempo_post_foreground_drain_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_path = _write_run(Path(temporary) / "run", tempo_drain=0.25)
            report = analyzer.analyze_paths([result_path])

        self.assertEqual(report["verdict"], "inconclusive")
        self.assertFalse(report["gates"]["all_tempo_zero_post_foreground_drain"])
        self.assertIn("drain", report["runs"][0]["validation_errors"][0])

    def test_cli_writes_optional_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = _write_run(root / "run")
            output = root / "analysis.json"
            with contextlib.redirect_stdout(io.StringIO()):
                status = analyzer.main([str(result_path), "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertEqual(report["schema_version"], analyzer.ANALYSIS_SCHEMA)
        self.assertEqual(report["verdict"], "promising")


if __name__ == "__main__":
    unittest.main()
