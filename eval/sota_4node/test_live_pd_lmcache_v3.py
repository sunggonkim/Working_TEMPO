from __future__ import annotations

import json
from pathlib import Path
import unittest

from eval.sota_4node import live_pd_controller_v1 as base
from eval.sota_4node import live_pd_controller_lmcache_v3 as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v2 as node


ROOT = Path(__file__).resolve().parents[2]


class LivePDLMCacheV3Tests(unittest.TestCase):
    def test_official_connector_not_multiconnector(self) -> None:
        command = node._vllm_command(
            ROOT / ".vllm_venv/bin/vllm",
            ROOT / "models/TinyLlama-1.1B-Chat-v1.0",
            is_prefill=True,
            mode="lmcache_always_remote",
            pair=0,
            ports=node._ports(100, 0),
        )
        config = json.loads(command[command.index("--kv-transfer-config") + 1])
        self.assertEqual(config["kv_connector"], "LMCacheConnectorV1")
        self.assertEqual(config["kv_role"], "kv_producer")
        self.assertNotIn("MultiConnector", json.dumps(config))
        self.assertIn("--disable-hybrid-kv-cache-manager", command)

    def test_receiver_has_four_rank_ports(self) -> None:
        ports = node._ports(100, 0)
        config = node._config_text(
            is_prefill=False,
            prefill_host="p0",
            decode_host="d0",
            ports=ports,
        )
        expected_init = ", ".join(str(ports["decoder_init"] + i) for i in range(4))
        expected_alloc = ", ".join(str(ports["decoder_alloc"] + i) for i in range(4))
        self.assertIn(f"pd_peer_init_port: [{expected_init}]", config)
        self.assertIn(f"pd_peer_alloc_port: [{expected_alloc}]", config)
        self.assertIn('pd_role: "receiver"', config)

    def test_pair_selection_is_stable(self) -> None:
        self.assertEqual(client.wire._pair_index("mode-cal-remote-0"), 0)
        self.assertEqual(client.wire._pair_index("mode-cal-remote-1"), 1)
        self.assertEqual(client.wire._pair_index("live-pd-validation-2"), 0)

    def test_combine_requires_same_output_and_kv(self) -> None:
        def lifecycle(mode: str, delta: float) -> dict:
            rows = []
            for bucket in range(3):
                rows.append({
                    "request_id": f"live-pd-validation-{bucket}",
                    "bucket": bucket,
                    "prompt_sha256": f"prompt-{bucket}",
                    "output_sha256": f"output-{bucket}",
                    "potential_kv": {"logical_bytes": 10 + bucket, "tp8_physical_bytes": 20 + bucket},
                    "route": "official_lmcache_connector_v1_live_pd",
                    "e2e_ms": 100.0 + delta,
                    "ttft_ms": 20.0 + delta,
                    "tpot_p99_ms": 2.0,
                    "metrics": {"proxy": {name: 0.0 for name in base.METRIC_NAMES}},
                })
            return {
                "schema": base.SCHEMA,
                "mode": mode,
                "valid": True,
                "topology": {"nodes": 4, "gpus": 16},
                "validation": rows,
            }

        report = base.combine(
            lifecycle("lmcache_always_remote", 0.0),
            lifecycle("tempo_admission", -10.0),
        )
        self.assertEqual(report["screen_outcome"], "live_pd_candidate_pass")
        self.assertEqual(report["summary"]["e2e_win_count"], 3)

    def test_launcher_is_one_bounded_existing_allocation_step(self) -> None:
        launcher = (ROOT / "eval/sota_4node/run_vllm_lmcache_live_pd_v3_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertIn("timeout --foreground", launcher)
        self.assertIn("SLURM_JOB_ID", launcher)
        self.assertNotIn("salloc", launcher)
        self.assertNotIn("sbatch", launcher)
        self.assertIn("live_pd_node_entry_v2.sh", launcher)


if __name__ == "__main__":
    unittest.main()
