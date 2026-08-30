from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from eval.sota_4node import vllm_lmcache_pd_contention_node as node


class PDContentionNodeTest(unittest.TestCase):
    def test_client_command_uses_one_combined_contention_parent(self) -> None:
        with mock.patch.dict(os.environ, {
            "TEMPO_PD_CONTENTION_DECODER_REFERENCE_RATE": "9",
            "TEMPO_PD_CONTENTION_REMOTE_REFERENCE_RATE": "17",
            "TEMPO_PD_CONTENTION_LOAD_FRACTION": "0.70",
            "TEMPO_PD_CONTENTION_PHASE_DURATION_MS": "30000",
            "TEMPO_PD_CONTENTION_COOLDOWN_S": "3",
            node.PROBE_URLS_ENV: ",".join(
                f"http://node{index}:31000" for index in range(4)),
        }, clear=True):
            command = node._client_command(
                Path("python"),
                base_url="http://frontend",
                model=Path("/model"),
                workload=Path("/workload"),
                output=Path("/output"),
                mode="tempo_auto",
                run_id="contention",
                request_rate=2.0,
                max_workers=64,
            )
        self.assertEqual(command[command.index("-m") + 1], node.CLIENT_MODULE)
        self.assertEqual(
            command[command.index("--decoder-reference-rate") + 1], "9")
        self.assertEqual(
            command[command.index("--remote-reference-rate") + 1], "17")
        self.assertEqual(command[command.index("--load-fraction") + 1], "0.70")
        self.assertEqual(command.count("--endpoint-evidence-url"), 4)

    def test_probe_command_maps_exact_interleaved_pd_roles(self) -> None:
        hosts = [f"node{index}" for index in range(4)]
        prefill = node._probe_command(
            Path("python"), node_index=2, hosts=hosts, port_slot=1680)
        decoder = node._probe_command(
            Path("python"), node_index=3, hosts=hosts, port_slot=1680)
        self.assertEqual(prefill[prefill.index("--endpoint-id") + 1],
                         "pair1-prefill")
        self.assertEqual(prefill[prefill.index("--role") + 1], "prefill")
        self.assertEqual(decoder[decoder.index("--endpoint-id") + 1],
                         "pair1-decoder")
        self.assertEqual(decoder[decoder.index("--role") + 1], "decoder")

    def test_environment_forbids_synthetic_background(self) -> None:
        valid = {"TEMPO_PD_BENCHMARK_COLD_MEASURED": "1"}
        with mock.patch.dict(os.environ, valid, clear=True):
            node._validate_environment()
        with (
            mock.patch.dict(os.environ, {
                **valid,
                "TEMPO_CXI_BACKGROUND_DUTY_CYCLE": "1.0",
            }, clear=True),
            self.assertRaisesRegex(ValueError, "synthetic"),
        ):
            node._validate_environment()

    def test_frozen_manifest_binds_rates_paths_and_digests(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        manifest = repo / "eval/sota_4node/tempo_pd_contention_workload_v4_frozen.json"
        value = json.loads(manifest.read_text(encoding="utf-8"))
        workload = repo / value["source_workload"]["path"]
        profile = repo / value["profile"]["path"]
        environment = {
            node.FROZEN_MANIFEST_ENV: str(manifest),
            "TEMPO_PD_CONTENTION_DECODER_REFERENCE_RATE": "32",
            "TEMPO_PD_CONTENTION_REMOTE_REFERENCE_RATE": "6.8",
            "TEMPO_PD_CONTENTION_LOAD_FRACTION": "0.70",
            "TEMPO_PD_CONTENTION_PHASE_DURATION_MS": "15000",
            "TEMPO_PD_CONTENTION_COOLDOWN_S": "2",
            "TEMPO_LMCACHE_NIXL_BACKEND": "UCX",
        }
        args = SimpleNamespace(repo_root=repo, request_rate=2.0)
        with mock.patch.dict(os.environ, environment, clear=True):
            path, digest = node._frozen_workload_manifest(
                args, workload=workload, profile=profile)
        self.assertEqual(path, manifest)
        self.assertEqual(digest, node._sha256(manifest))

        environment["TEMPO_PD_CONTENTION_LOAD_FRACTION"] = "0.85"
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            self.assertRaisesRegex(ValueError, "differs from frozen"),
        ):
            node._frozen_workload_manifest(
                args, workload=workload, profile=profile)
    def test_launcher_requires_existing_four_node_allocation(self) -> None:
        launcher = (
            Path(__file__).resolve().parent
            / "run_tempo_pd_contention_fixed_in_allocation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("existing allocation required", launcher)
        self.assertIn('"${SLURM_JOB_NUM_NODES:-}" == 4', launcher)
        self.assertIn("--gpus-per-task=4", launcher)
        self.assertIn("TEMPO_PD_BENCHMARK_COLD_MEASURED=1", launcher)
        self.assertNotIn("salloc", launcher)
        self.assertNotIn("sbatch", launcher)
        self.assertNotIn("cxi_background_traffic", launcher)


if __name__ == "__main__":
    unittest.main()
