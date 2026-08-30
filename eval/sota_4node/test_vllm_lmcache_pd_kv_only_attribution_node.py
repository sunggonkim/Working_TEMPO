from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from eval.sota_4node import vllm_lmcache_pd_kv_only_attribution_node as node


class CoupledManifestTest(unittest.TestCase):
    def _fixture(self, root: Path):
        source = root / "results/source.jsonl"
        profile = root / "eval/profile.json"
        parent_result = root / "results/parent/result.json"
        parent_characterization = root / "results/parent/characterization.json"
        for path in (source, profile, parent_result, parent_characterization):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name + "\n", encoding="utf-8")

        def item(path: Path) -> dict[str, str]:
            return {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        manifest = root / "eval/manifest.json"
        manifest.write_text(json.dumps({
            "schema": "tempo-pd-c3-coupled-abba-manifest-v2",
            "performance_claim_allowed": False,
            "replicates": 2,
            "arm_order_policy": "paired_abba",
            "within_rate_block_order": [
                "local", "remote", "remote", "local"],
            "p_only_rates_per_s": [0.0, 4.0, 8.0, 12.0],
            "decoder_hot_rate_per_s": 22.4,
            "foreground_rate_per_s": 2.0,
            "phase_duration_ms": 8000.0,
            "cooldown_s": 2.0,
            "source_workload": item(source),
            "profile": item(profile),
            "parent_pilot": {
                "result": item(parent_result),
                "characterization": item(parent_characterization),
            },
            "transport": "LMCacheConnectorV1:UCX",
        }), encoding="utf-8")
        return source, profile, manifest

    def test_abba_manifest_binds_runtime_and_parent_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, profile, manifest = self._fixture(root)
            environment = {
                "TEMPO_PD_KV_ATTR_DECODER_HOT_RATE": "22.4",
                "TEMPO_PD_C3_APPROVED": "YES",
                node.COUPLED_MANIFEST_ENV: str(manifest),
                "TEMPO_PD_KV_ATTR_RATES": "0,4,8,12",
                "TEMPO_PD_KV_ATTR_REPETITIONS": "2",
                "TEMPO_PD_KV_ATTR_ARM_ORDER": "paired_abba",
                "TEMPO_PD_KV_ATTR_PHASE_DURATION_MS": "8000",
                "TEMPO_PD_KV_ATTR_COOLDOWN_S": "2",
            }
            args = SimpleNamespace(repo_root=root, request_rate=2.0)
            with patch.dict(os.environ, environment, clear=True):
                path, digest = node._coupled_manifest(
                    args, workload=source, profile=profile)
            self.assertEqual(path, manifest.resolve())
            self.assertEqual(
                digest, hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_abba_manifest_rejects_runtime_repetition_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, profile, manifest = self._fixture(root)
            environment = {
                "TEMPO_PD_KV_ATTR_DECODER_HOT_RATE": "22.4",
                "TEMPO_PD_C3_APPROVED": "YES",
                node.COUPLED_MANIFEST_ENV: str(manifest),
                "TEMPO_PD_KV_ATTR_RATES": "0,4,8,12",
                "TEMPO_PD_KV_ATTR_REPETITIONS": "1",
                "TEMPO_PD_KV_ATTR_ARM_ORDER": "paired_abba",
                "TEMPO_PD_KV_ATTR_PHASE_DURATION_MS": "8000",
                "TEMPO_PD_KV_ATTR_COOLDOWN_S": "2",
            }
            args = SimpleNamespace(repo_root=root, request_rate=2.0)
            with patch.dict(os.environ, environment, clear=True):
                with self.assertRaisesRegex(
                    ValueError, "repetitions differ from manifest"
                ):
                    node._coupled_manifest(
                        args, workload=source, profile=profile)

    def test_readiness_allows_bounded_perlmutter_cold_start(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(node._readiness_timeout_from_environment(), 3600.0)
        with patch.dict(
            os.environ, {"TEMPO_PD_KV_ATTR_READINESS_S": "3600"}, clear=True,
        ):
            self.assertEqual(node._readiness_timeout_from_environment(), 3600.0)
        with patch.dict(
            os.environ, {"TEMPO_PD_KV_ATTR_READINESS_S": "3601"}, clear=True,
        ):
            with self.assertRaisesRegex(ValueError, r"\[600, 3600\]"):
                node._readiness_timeout_from_environment()


if __name__ == "__main__":
    unittest.main()
