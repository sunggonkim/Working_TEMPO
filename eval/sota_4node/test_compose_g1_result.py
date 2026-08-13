from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.compose_g1_result import compose_g1_result
from eval.sota_4node.compose_g1_result import _host_pressure_raw_digest
from eval.sota_4node.host_pressure_placebo import _record_digest
from eval.sota_4node.test_validate_g1_result import valid_result
from eval.sota_4node.test_validate_g1_tier_raw import _make_fixture


def _source_bundle(root: Path) -> str:
    values = []
    names = [
        "train_executed.py",
        "tier_attribution_runner_executed.py",
        "validate_g1_tier_raw_executed.py",
        "validate_g1_result_executed.py",
        "compose_g1_result_executed.py",
        "build_g1_causal_readiness_executed.py",
        "host_pressure_train_wrapper_executed.py",
        "host_pressure_placebo.py",
        "capture_g1_domain_counters_executed.py",
    ]
    if (root / "prepare_foreground_path_executed.py").is_file():
        names.append("prepare_foreground_path_executed.py")
    if (root / "capture_nvml_pcie_observation_executed.py").is_file():
        names.append("capture_nvml_pcie_observation_executed.py")
    if (root / "capture_lustre_rpc_observation_executed.py").is_file():
        names.append("capture_lustre_rpc_observation_executed.py")
    for name in sorted(names):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        values.append(f"{name}:{digest}")
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


class ComposeG1ResultTests(unittest.TestCase):
    def _bind_analysis_snapshots(self, root: Path) -> None:
        mapping = {
            "validate_g1_result.py": "validate_g1_result_executed.py",
            "compose_g1_result.py": "compose_g1_result_executed.py",
            "validate_g1_tier_raw.py": "validate_g1_tier_raw_executed.py",
        }
        manifest_path = root / "g1_tier_runtime_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for source, snapshot in mapping.items():
            source_path = Path(__file__).with_name(source)
            shutil.copyfile(source_path, root / snapshot)
            manifest["source_sha256"][snapshot] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    def _write_sidecar(self, root: Path) -> Path:
        result = copy.deepcopy(valid_result())
        self._bind_analysis_snapshots(root)
        manifest = json.loads((root / "g1_tier_runtime_manifest.json").read_text(encoding="utf-8"))
        result["source_bundle_sha256"] = _source_bundle(root)
        result["world_size"] = manifest["world_size"]
        result["nodes"] = manifest["nodes"]
        result["state_bytes_per_rank"] = manifest["state_bytes_per_rank"]
        result["logical_file_extent_bytes"] = manifest["logical_file_extent_bytes"]
        result["checkpoint_steps"] = manifest["checkpoint_steps"]
        result["host_pressure_raw_digest"] = _host_pressure_raw_digest(root, manifest)
        path = root / "g1_metrics.json"
        path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def test_composes_only_after_raw_and_causal_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            output = compose_g1_result(root, self._write_sidecar(root))
            self.assertEqual(output["schema_version"], "tempo-rd-g1-composed-evaluation-1")
            self.assertTrue(output["causal_evaluation"]["promote_static_policy"])

    def test_missing_raw_structure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = root / "g1_metrics.json"
            sidecar.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest"):
                compose_g1_result(root, sidecar)

    def test_source_bundle_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            sidecar = self._write_sidecar(root)
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
            raw["source_bundle_sha256"] = "b" * 64
            sidecar.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source bundle"):
                compose_g1_result(root, sidecar)

    def test_current_analysis_code_must_match_runtime_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            sidecar = self._write_sidecar(root)
            snapshot = root / "validate_g1_result_executed.py"
            snapshot.write_text("# changed after allocation\n", encoding="utf-8")
            manifest_path = root / "g1_tier_runtime_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_sha256"][snapshot.name] = hashlib.sha256(
                snapshot.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "analysis snapshot mismatch"):
                compose_g1_result(root, sidecar)

    def test_geometry_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            sidecar = self._write_sidecar(root)
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
            raw["state_bytes_per_rank"] += 1
            sidecar.write_text(json.dumps(raw) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "state_bytes_per_rank"):
                compose_g1_result(root, sidecar)

    def test_sidecar_cannot_replace_validated_raw_host_pressure_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_fixture(root)
            sidecar = self._write_sidecar(root)
            raw_path = root / "fg_only/host_pressure_rank_0.json"
            record = json.loads(raw_path.read_text(encoding="utf-8"))
            record["samples"][1]["numa_node_bytes"] += 4096
            record["output_sha256"] = _record_digest(record)
            raw_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw host-pressure records"):
                compose_g1_result(root, sidecar)


if __name__ == "__main__":
    unittest.main()
