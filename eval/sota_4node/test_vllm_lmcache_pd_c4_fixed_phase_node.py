from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from eval.sota_4node import vllm_lmcache_pd_c4_fixed_phase_node as node
from eval.sota_4node import vllm_lmcache_pd_contention_node as contention


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "eval/sota_4node/tempo_pd_c4_phase_manifest_v2.json"
IMPLEMENTATION_CONTRACT = (
    ROOT / "eval/sota_4node/tempo_pd_c4_implementation_contract_v1.json"
)
PROFILE = ROOT / "eval/sota_4node/real_tempo_pd_elastic_profile_v447.json"
WORKLOAD = (
    ROOT
    / "results/tempo_elastic_pd_canonical_discovery_57198936"
    / "run02_libfabric_cxi_peermem_pd1g_cold_q25_longp1230_cxibg100_dynamic_v544b"
    / "tempo_elastic_pd_v445/warmup.jsonl"
)


class C4FixedPhaseNodeTest(unittest.TestCase):
    def test_client_block_artifacts_are_hash_bound_and_abba_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client_raw = root / "raw.json"
            artifacts = {}
            contracts = {}
            block_order = []
            for sequence, (key, arm, replicate) in enumerate(
                node._EXPECTED_BLOCKS
            ):
                path = root / "c4_fixed_phase" / f"{key}.raw.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({"sequence": sequence}))
                artifacts[key] = {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                contracts[key] = {
                    "sequence": sequence,
                    "foreground_arm": arm,
                    "replicate": replicate,
                    "all_requests_valid": True,
                    "completion_cache_evidence_exact": True,
                    "phase_aligned_endpoint_evidence": True,
                }
                block_order.append({"arm": arm, "replicate": replicate})
            artifact = {
                "artifacts": artifacts,
                "contracts": contracts,
                "block_order": block_order,
            }
            validated = node._validate_client_artifacts(
                artifact, client_raw_path=client_raw)
            self.assertEqual(list(validated), [
                item[0] for item in node._EXPECTED_BLOCKS])

            first = artifacts["00_local_r0"]
            Path(first["path"]).write_text('{"drift": true}\n')
            with self.assertRaisesRegex(ValueError, "digest differs"):
                node._validate_client_artifacts(
                    artifact, client_raw_path=client_raw)

    def test_frozen_manifest_revalidates_all_parent_digests(self):
        args = SimpleNamespace(repo_root=ROOT)
        with patch.dict(os.environ, {
            node.MANIFEST_ENV: str(MANIFEST),
            node.MANIFEST_SHA_ENV: node._sha256(MANIFEST),
        }, clear=False):
            path, value = node._load_manifest(
                args, workload=WORKLOAD.resolve(),
                elastic_profile=PROFILE.resolve())
        self.assertEqual(path, MANIFEST.resolve())
        self.assertEqual(value["schema"], node.builder.SCHEMA)
        self.assertTrue(
            value["cache_state_protocol"][
                "decoder_usage_breakdown_required"])
        self.assertEqual(
            value["endpoint_evidence_contract"]["sampling_policy"],
            "workload_start_boundary_midpoint_and_end_boundary",
        )
        self.assertEqual(
            value["endpoint_evidence_contract"]["phase_boundary_samples"], 7)
        self.assertTrue(
            value["endpoint_evidence_contract"][
                "publisher_pid_matches_measured_child"])
        self.assertEqual(
            value["fixed_runtime_environment"],
            dict(sorted(node._FIXED_RUNTIME_ENVIRONMENT.items())),
        )

    def test_frozen_implementation_contract_revalidates_runtime(self):
        args = SimpleNamespace(repo_root=ROOT)
        with patch.dict(os.environ, {
            node.IMPLEMENTATION_ENV: str(IMPLEMENTATION_CONTRACT),
            node.IMPLEMENTATION_SHA_ENV: node._sha256(
                IMPLEMENTATION_CONTRACT),
        }, clear=False):
            path, value = node._load_implementation_contract(
                args, phase_manifest=MANIFEST.resolve())
        self.assertEqual(path, IMPLEMENTATION_CONTRACT.resolve())
        self.assertEqual(value["schema"], node.implementation.SCHEMA)
        self.assertEqual(len(value["files"]), 75)
        self.assertFalse(value["performance_claim_allowed"])

    def test_client_command_uses_new_fixed_phase_module_and_four_probes(self):
        urls = [f"http://node-{index}:9000" for index in range(4)]
        with patch.dict(os.environ, {
            contention.PROBE_URLS_ENV: ",".join(urls),
            "TEMPO_PD_C4_PHASE_DURATION_MS": "8000",
            "TEMPO_PD_C4_COOLDOWN_S": "2",
        }, clear=False):
            command = node._client_command(
                Path("/python"), base_url="http://front:8000",
                model=Path("/model"), workload=Path("/workload"),
                output=Path("/output"), mode="tempo_auto", run_id="c4",
                request_rate=2.0, max_workers=128)
        self.assertIn(node.CLIENT_MODULE, command)
        self.assertEqual(command.count("--endpoint-evidence-url"), 4)
        self.assertNotIn("--endpoint-controller-url", command)

    def test_environment_enables_apc_evidence_without_controller(self):
        values = dict(node._FIXED_RUNTIME_ENVIRONMENT)
        values.update({
            node.MANIFEST_ENV: str(MANIFEST),
            node.MANIFEST_SHA_ENV: node._sha256(MANIFEST),
            node.IMPLEMENTATION_ENV: str(IMPLEMENTATION_CONTRACT),
            node.IMPLEMENTATION_SHA_ENV: node._sha256(
                IMPLEMENTATION_CONTRACT),
            "TEMPO_PD_C4_READINESS_S": "3600",
            "TEMPO_ELASTIC_PD_PROFILE": str(PROFILE),
        })
        with patch.dict(os.environ, values, clear=True):
            node._validate_environment()

        values["TEMPO_VLLM_ASYNC_SCHEDULING"] = "1"
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaisesRegex(ValueError, "ASYNC_SCHEDULING=0"):
                node._validate_environment()

        values["TEMPO_VLLM_ASYNC_SCHEDULING"] = "0"
        values["TEMPO_PD_REMOTE_KV_BUDGET_BYTES"] = "999999999"
        with patch.dict(os.environ, values, clear=True):
            with self.assertRaisesRegex(ValueError, "inherited"):
                node._validate_environment()


if __name__ == "__main__":
    unittest.main()
