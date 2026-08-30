from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

from eval.sota_4node import run_tempo_go_c6_fixed_cross_edge_recovery as cross
from tempo.pd_contention_workload import (
    CacheState,
    ContentionState,
    ForegroundArm,
    LoadSelection,
    Tenant,
    TokenGeometry,
    TrafficShape,
    build_schedule,
)


ROOT = Path(__file__).parent


def _summary(p50: float, p95: float, p99: float) -> dict[str, float]:
    return {"mean": p50, "p50": p50, "p95": p95, "p99": p99, "max": p99}


def _edge(source: int, destination: int, p50: float, *, slo: float):
    return {
        "source_prefill_index": source,
        "decoder_index": destination,
        "destination_hot": p50 > 100.0,
        "metrics": {
            "victim_count": 60,
            "ttft_ms": _summary(300.0, 350.0, 400.0),
            "decode_completion_ms": _summary(p50, p50, p50 * 1.1),
            "tpot_ms": _summary(p50 / 127.0, p50 / 127.0, p50 * 1.1 / 127.0),
            "e2e_ms": _summary(p50 + 300.0, p50 + 350.0, p50 * 1.1 + 400.0),
            "slo_attainment_fraction": slo,
        },
    }


class C6FixedCrossEdgeRecoveryTests(unittest.TestCase):
    def test_schedule_pins_hot_local_aggressor_and_splits_cross_victims(self) -> None:
        schedule = build_schedule(
            states=(ContentionState.C1,),
            selection=LoadSelection(
                decoder_reference_rate_per_s=10.0,
                remote_reference_rate_per_s=6.8,
                decoder_fraction=0.7,
                remote_fraction=0.7,
            ),
            foreground_arm=ForegroundArm.REMOTE,
            foreground_rate_per_s=2.0,
            trial_id="unit-cross",
            shape=TrafficShape.STABLE,
            phase_duration_ms=1000.0,
            foreground_geometries=(TokenGeometry(4094, 128, CacheState.MISS),),
        )
        routed, identities = cross._routed_schedule(
            schedule, hot_decoder_index=0
        )
        foreground = [row for row in routed if row.tenant is Tenant.FOREGROUND]
        aggressors = [row for row in routed if row.tenant is Tenant.DECODER_HOT]
        self.assertEqual(len(foreground), 2)
        self.assertEqual(
            {identities[row.request_id]["edge_id"] for row in foreground},
            {"remote:p0->d1", "remote:p1->d0"},
        )
        self.assertTrue(all(row.request_id.endswith(("-0", "-1")) for row in routed))
        self.assertTrue(all(
            identities[row.request_id]["edge_id"] == "local:d0"
            for row in aggressors
        ))

    def test_q2_requires_and_accepts_opposite_large_edge_winners(self) -> None:
        contract_path = ROOT / "tempo_go_c6_fixed_cross_edge_recovery_contract_v1.json"
        phase0 = {
            "name": "00_hot_decoder_0",
            "hot_decoder_index": 0,
            "aggressor_count": 1344,
            "edges": {
                "remote:p1->d0": _edge(1, 0, 600.0, slo=0.1),
                "remote:p0->d1": _edge(0, 1, 100.0, slo=1.0),
            },
            "raw": "/tmp/a",
            "raw_sha256": "a" * 64,
        }
        phase1 = {
            "name": "01_hot_decoder_1",
            "hot_decoder_index": 1,
            "aggressor_count": 1344,
            "edges": {
                "remote:p1->d0": _edge(1, 0, 100.0, slo=1.0),
                "remote:p0->d1": _edge(0, 1, 600.0, slo=0.1),
            },
            "raw": "/tmp/b",
            "raw_sha256": "b" * 64,
        }
        bundle = {
            "schema": cross.BUNDLE_SCHEMA,
            "artifacts": {
                "00_hot_decoder_0": "/tmp/a",
                "01_hot_decoder_1": "/tmp/b",
            },
        }
        with mock.patch.object(
            cross,
            "_phase_result",
            side_effect=[(phase0, [(0,)]), (phase1, [(0,)])],
        ):
            result = cross.analyze_bundle(bundle, contract_path)
        self.assertTrue(result["q2_opposite_action_opportunity_pass"])
        self.assertTrue(result["q3_service_horizon_pass"])
        self.assertGreater(
            result["aggregate_effect"][
                "median_alternate_p50_latency_recovery_fraction"
            ],
            0.8,
        )

    def test_contract_and_wrapper_freeze_native_interactive_cross_path(self) -> None:
        contract = json.loads((
            ROOT / "tempo_go_c6_fixed_cross_edge_recovery_contract_v1.json"
        ).read_text(encoding="utf-8"))
        section = contract["fixed_cross_edge_recovery"]
        self.assertEqual(contract["qualification_kind"], "fixed_cross_edge_recovery")
        self.assertEqual(section["remote_decode_placement"], "cross")
        self.assertEqual(section["phase_duration_ms"], 60000.0)
        self.assertEqual(section["victim"]["offered_rate_per_edge_per_s"], 1.0)
        wrapper = (
            ROOT / "run_tempo_go_c6_fixed_cross_edge_recovery_in_allocation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("TEMPO_PD_REMOTE_DECODE_PLACEMENT=cross", wrapper)
        self.assertIn("--nodes=4 --ntasks=4 --ntasks-per-node=1", wrapper)
        self.assertNotIn("sbatch", wrapper)
        self.assertNotIn("scancel", wrapper)
        for forbidden in ("sudo", "udiRoot", "CAP_NET_ADMIN", "--image"):
            self.assertNotIn(forbidden, wrapper)


if __name__ == "__main__":
    unittest.main()
