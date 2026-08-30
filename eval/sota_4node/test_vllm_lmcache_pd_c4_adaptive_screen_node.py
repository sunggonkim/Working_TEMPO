from pathlib import Path
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from eval.sota_4node import vllm_lmcache_pd_c4_adaptive_screen_node as node
from eval.sota_4node import vllm_lmcache_pd_contention_node as contention


class C4AdaptiveScreenNodeTest(unittest.TestCase):
    def test_child_artifacts_are_hash_bound_and_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client_raw = root / "raw.json"
            artifacts = {}
            contracts = {}
            block_order = []
            for sequence, (key, arm, replicate) in enumerate(
                node._EXPECTED_BLOCKS
            ):
                contract = {
                    "schema": node.client.BLOCK_SCHEMA,
                    "sequence": sequence,
                    "arm": arm,
                    "replicate": replicate,
                    "all_requests_valid": True,
                    "completion_cache_evidence_exact": True,
                    "phase_aligned_endpoint_evidence": True,
                    "controller_reset_before_block_exact": True,
                    "controller_quiescent_after_block": True,
                }
                path = root / "c4_adaptive_screen" / f"{key}.raw.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({
                    "c4_adaptive_screen_contract": contract,
                }), encoding="utf-8")
                artifacts[key] = {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                contracts[key] = contract
                block_order.append({"arm": arm, "replicate": replicate})
            artifact = {
                "artifacts": artifacts,
                "contracts": contracts,
                "block_order": block_order,
            }
            validated = node._validate_client_artifacts(
                artifact, client_raw_path=client_raw)
            self.assertEqual(
                list(validated), [item[0] for item in node._EXPECTED_BLOCKS])

            first = artifacts["00_local_r0"]
            Path(first["path"]).write_text(
                '{"drift": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest differs"):
                node._validate_client_artifacts(
                    artifact, client_raw_path=client_raw)

    def test_client_command_uses_four_probes_and_two_controllers(self):
        probes = [f"http://node-{index}:9000" for index in range(4)]
        controllers = ["http://node-0:9100", "http://node-2:9100"]
        with patch.dict(os.environ, {
            contention.PROBE_URLS_ENV: ",".join(probes),
            node.CONTROLLER_URLS_ENV: ",".join(controllers),
            "TEMPO_PD_C4_PHASE_DURATION_MS": "8000",
            "TEMPO_PD_C4_COOLDOWN_S": "2",
        }, clear=False):
            command = node._client_command(
                Path("/python"),
                base_url="http://front:8000",
                model=Path("/model"),
                workload=Path("/workload"),
                output=Path("/output"),
                mode="tempo_auto",
                run_id="adaptive",
                request_rate=2.0,
                max_workers=128,
            )
        self.assertIn(node.CLIENT_MODULE, command)
        self.assertEqual(command.count("--endpoint-evidence-url"), 4)
        self.assertEqual(command.count("--endpoint-controller-url"), 2)

    def test_prestart_environment_is_exact_and_rejects_inherited_policy(self):
        values = dict(node._FIXED_RUNTIME_ENVIRONMENT)
        values.update({
            node.RUN_CONTRACT_ENV: "/repo/results/run-contract.json",
            node.RUN_CONTRACT_SHA_ENV: "a" * 64,
            "TEMPO_PD_C4_READINESS_S": "3600",
        })
        with patch.dict(os.environ, values, clear=True):
            node._validate_prestart_environment()

        values["TEMPO_PD_PRESSURE_MODE"] = "adaptive"
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaisesRegex(ValueError, "PRESSURE_MODE=disabled"):
                node._validate_prestart_environment()

        values["TEMPO_PD_PRESSURE_MODE"] = "disabled"
        values["TEMPO_PD_REMOTE_KV_BUDGET_BYTES"] = "123"
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaisesRegex(ValueError, "inherited"):
                node._validate_prestart_environment()

    def test_dynamic_profile_environment_comes_only_from_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            elastic = root / "elastic.json"
            endpoint = root / "endpoint.json"
            manifest = root / "manifest.json"
            for path in (elastic, endpoint, manifest):
                path.write_text("{}\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                node._configure_dynamic_environment({
                    "elastic": elastic.resolve(),
                    "endpoint": endpoint.resolve(),
                    "manifest_path": manifest.resolve(),
                })
                self.assertEqual(
                    os.environ["TEMPO_ELASTIC_PD_PROFILE"], str(elastic.resolve()))
                self.assertEqual(
                    os.environ["TEMPO_PD_ENDPOINT_SERVICE_PROFILE"],
                    str(endpoint.resolve()),
                )
                self.assertEqual(
                    os.environ[node.client.WORKLOAD_SHA_ENV],
                    hashlib.sha256(manifest.read_bytes()).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
