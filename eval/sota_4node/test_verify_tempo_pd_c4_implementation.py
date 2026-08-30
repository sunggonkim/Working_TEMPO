from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval.sota_4node import verify_tempo_pd_c4_implementation as verify


class C4ImplementationContractTest(unittest.TestCase):
    def _fixture(self, root: Path):
        for index, relative in enumerate(sorted(verify.REQUIRED_FILES)):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"file-{index}\n", encoding="utf-8")
        manifest = root / "eval/sota_4node/manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        environment = {
            "python": "3.12.11",
            "vllm": "0.26.0+cu129",
            "lmcache": "0.1.dev1",
            "torch": "2.8.0",
            "transformers": "4.55.4",
        }
        value = {
            "schema": verify.SCHEMA,
            "purpose": "frozen C4 characterization only",
            "performance_claim_allowed": False,
            "phase_manifest": {
                "path": str(manifest.relative_to(root)),
                "sha256": verify._sha256(manifest),
            },
            "git_heads": {
                "repository": "a" * 40,
                "third_party_lmcache": "b" * 40,
            },
            "environment_versions": environment,
            "files": [{
                "path": relative,
                "sha256": verify._sha256(root / relative),
            } for relative in sorted(verify.REQUIRED_FILES)],
        }
        value["fingerprint_sha256"] = verify.contract_fingerprint(value)
        contract = root / "contract.json"
        contract.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return contract, manifest, environment

    def test_exact_contract_passes_and_file_drift_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, manifest, environment = self._fixture(root)
            with (
                patch.object(
                    verify, "_git_head",
                    side_effect=("a" * 40, "b" * 40)),
                patch.object(
                    verify, "_environment_versions",
                    return_value=environment),
            ):
                value = verify.verify_contract(
                    repo_root=root,
                    contract_path=contract,
                    expected_sha256=verify._sha256(contract),
                    phase_manifest=manifest,
                )
            self.assertEqual(value["schema"], verify.SCHEMA)

            drifted = root / sorted(verify.REQUIRED_FILES)[0]
            drifted.write_text("drift\n", encoding="utf-8")
            with (
                patch.object(
                    verify, "_git_head",
                    side_effect=("a" * 40, "b" * 40)),
                patch.object(
                    verify, "_environment_versions",
                    return_value=environment),
                self.assertRaisesRegex(ValueError, "drifted"),
            ):
                verify.verify_contract(
                    repo_root=root,
                    contract_path=contract,
                    expected_sha256=verify._sha256(contract),
                    phase_manifest=manifest,
                )

    def test_contract_digest_and_required_set_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract, manifest, environment = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "digest differs"):
                verify.verify_contract(
                    repo_root=root,
                    contract_path=contract,
                    expected_sha256="0" * 64,
                    phase_manifest=manifest,
                )
            value = json.loads(contract.read_text(encoding="utf-8"))
            value["files"].pop()
            value["fingerprint_sha256"] = verify.contract_fingerprint(value)
            contract.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(
                    verify, "_git_head",
                    side_effect=("a" * 40, "b" * 40)),
                patch.object(
                    verify, "_environment_versions",
                    return_value=environment),
                self.assertRaisesRegex(ValueError, "omits"),
            ):
                verify.verify_contract(
                    repo_root=root,
                    contract_path=contract,
                    expected_sha256=verify._sha256(contract),
                    phase_manifest=manifest,
                )


if __name__ == "__main__":
    unittest.main()
