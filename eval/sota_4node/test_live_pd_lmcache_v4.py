from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from eval.sota_4node import live_pd_controller_v1 as base
from eval.sota_4node import live_pd_controller_lmcache_v2 as wire
from eval.sota_4node import live_pd_controller_lmcache_v4 as client
from eval.sota_4node import vllm_lmcache_live_pd_node_v3 as node_entry


ROOT = Path(__file__).resolve().parents[2]


class LivePDLMCacheV4Tests(unittest.TestCase):
    def test_proxy_omits_min_tokens_and_restores_direct_body(self) -> None:
        original = base._base_decode_body

        def fake_remote(proxy_urls, decoder_urls, prompt, request_id):
            del proxy_urls, decoder_urls, request_id
            self.assertNotIn("min_tokens", base._base_decode_body(prompt))
            return {
                "prompt_tokens": 999,
                "live_kv_proof": {},
                "route": "official_lmcache_connector_v1_live_pd",
            }

        with mock.patch.object(client, "_token_count", return_value=77), mock.patch.object(
            wire, "_run_remote", side_effect=fake_remote
        ):
            result = client._run_remote("p0,p1", "d0,d1", "prompt", "req-0")
        self.assertIs(base._base_decode_body, original)
        self.assertEqual(result["prompt_tokens"], 77)
        self.assertEqual(result["live_kv_proof"]["original_prompt_tokens"], 77)

    def test_mode_directory_creation_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tempo_admission"
            node_entry._mkdir(path, parents=True, exist_ok=False)
            node_entry._mkdir(path, parents=True, exist_ok=False)
            self.assertTrue(path.is_dir())

    def test_v4_launcher_targets_race_safe_entry_once(self) -> None:
        text = (ROOT / "eval/sota_4node/run_vllm_lmcache_live_pd_v4_in_allocation.sh").read_text()
        self.assertEqual(text.count("srun "), 1)
        self.assertIn("live_pd_node_entry_v3.sh", text)
        self.assertNotIn("SLURM_NODEID", text)
        self.assertNotIn("MultiConnector", text)


if __name__ == "__main__":
    unittest.main()
