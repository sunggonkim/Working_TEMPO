from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from eval.sota_4node import run_tempo_go_c6_performance_client as client
from eval.sota_4node import run_tempo_go_c6_stream_client as stream
from eval.sota_4node import vllm_lmcache_tempo_go_c6_performance_node as node
from tempo.pd_contention_workload import Tenant


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "eval/sota_4node/tempo_go_c6_ablation_contract_v1.json"


class C6PerformanceAblationTests(unittest.TestCase):
    def _contract(self) -> dict[str, object]:
        return json.loads(CONTRACT.read_text(encoding="utf-8"))

    def _args(self) -> SimpleNamespace:
        return SimpleNamespace(
            decoder_reference_rate=32.0,
            remote_reference_rate=6.8,
            load_fraction=0.7,
            request_rate=2.0,
            phase_duration_ms=60000.0,
        )

    def test_contract_freezes_all_required_ablation_epochs(self) -> None:
        value = self._contract()
        section = value["c6_performance"]
        epochs = section["ablation_server_epochs"]
        self.assertEqual(
            [row["policy"] for row in epochs],
            [
                "predictor",
                "queue_gpu",
                "network_request_only",
                "app_global_only",
            ],
        )
        self.assertTrue(all(
            row["fresh_vllm_lmcache_epoch"] is True
            and len(row["block_order"]) == 3
            for row in epochs
        ))
        self.assertFalse(
            value["claim_boundary"]["independent_validation_claim_allowed"])
        for relative, expected in value["source_inventory"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                expected,
            )

    def test_compute_node_qualification_verifies_source_inventory(self) -> None:
        with mock.patch.dict(
            os.environ,
            {node.CONTRACT_ENV: str(CONTRACT)},
            clear=False,
        ):
            path, value = node._qualification(ROOT)
        self.assertEqual(path, CONTRACT)
        self.assertEqual(
            value["qualification_kind"], "c6_performance_ablation")

    def test_dynamic_arm_schedule_preserves_population_and_arm_identity(self) -> None:
        section = self._contract()["c6_performance"]
        for arm in (
            "predictor",
            "queue_gpu",
            "network_request_only",
            "app_global_only",
        ):
            epoch = next(
                row for row in section["ablation_server_epochs"]
                if row["policy"] == arm
            )
            with self.subTest(arm=arm), mock.patch.dict(
                os.environ,
                {client.ARM_ENV: arm},
                clear=False,
            ):
                os.environ.pop(client.FIXED_POLICY_ENV, None)
                schedule, identities = client._materialize_schedule(
                    spec=epoch["block_order"][0],
                    section=section,
                    args=self._args(),
                )
                foreground = [
                    row for row in schedule
                    if row.tenant is Tenant.FOREGROUND
                ]
                self.assertEqual(len(foreground), 120)
                wire_arm = "app_global_only" if arm == "app_global_only" else arm
                self.assertTrue(all(
                    row.request_id.startswith(f"epd-{wire_arm}-")
                    for row in foreground
                ))
                self.assertTrue(all(
                    identities[row.request_id]["expected_edge_id"] is None
                    for row in foreground
                ))

    def test_launcher_remains_interactive_only_and_accepts_bounded_arms(self) -> None:
        wrapper = (
            ROOT / "eval/sota_4node/run_tempo_go_c6_performance_in_allocation.sh"
        ).read_text(encoding="utf-8")
        for arm in (
            "predictor",
            "queue_gpu",
            "network_request_only",
            "app_global_only",
        ):
            self.assertIn(arm, wrapper)
        self.assertIn("--nodes=4 --ntasks=4 --ntasks-per-node=1", wrapper)
        self.assertNotIn("sbatch", wrapper)
        self.assertNotIn("scancel", wrapper)
        for forbidden in ("sudo", "udiRoot", "CAP_NET_ADMIN", "--image"):
            self.assertNotIn(forbidden, wrapper)

    def test_global_arms_use_c6_identity_preserving_stream_seam(self) -> None:
        canonical = "eval.sota_4node.run_tempo_pd_elastic_stream_metrics"
        for arm in ("full_c6", "app_global_only"):
            with self.subTest(arm=arm), mock.patch.dict(
                os.environ, {client.ARM_ENV: arm}, clear=False,
            ), mock.patch.object(
                client.fixed,
                "_child_command",
                return_value=["python", "-m", canonical],
            ):
                os.environ.pop(client.FIXED_POLICY_ENV, None)
                command = client._child_command(
                    SimpleNamespace(),
                    workload=Path("/tmp/workload"),
                    output=Path("/tmp/output"),
                    run_id="unit",
                )
            self.assertIn(
                "eval.sota_4node.run_tempo_go_c6_stream_client", command)

        original = stream.c5._rewrite_measured_arm_workload

        def verify_inner() -> int:
            self.assertEqual(
                stream.c5._rewrite_measured_arm_workload(["unchanged"]),
                ["unchanged"],
            )
            return 0

        with mock.patch.object(stream.c5, "main", side_effect=verify_inner):
            self.assertEqual(stream.main(), 0)
        self.assertIs(stream.c5._rewrite_measured_arm_workload, original)


if __name__ == "__main__":
    unittest.main()
