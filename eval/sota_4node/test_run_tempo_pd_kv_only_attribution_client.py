from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eval.sota_4node import run_tempo_pd_kv_only_attribution_client as client


class _Tokenizer:
    prefix = "tokens:"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text.startswith(self.prefix):
            return [int(value) for value in text[len(self.prefix):].split(",")]
        digest = hashlib.sha256(text.encode()).digest()
        return [int.from_bytes(digest[:4], "big")]

    def decode(
        self, values, *, skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return self.prefix + ",".join(str(value) for value in values)


class KVOnlyAttributionClientTest(unittest.TestCase):
    def test_rate_ladder_is_fail_closed(self):
        with patch.dict(os.environ, {"TEMPO_PD_KV_ATTR_RATES": "0,4,8,12"}):
            self.assertEqual(
                client._rates_from_environment(), (0.0, 4.0, 8.0, 12.0))
        with patch.dict(os.environ, {"TEMPO_PD_KV_ATTR_RATES": "4,8,16"}):
            self.assertEqual(client._rates_from_environment(), (4.0, 8.0, 16.0))
        with patch.dict(os.environ, {"TEMPO_PD_KV_ATTR_RATES": "8,4"}):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                client._rates_from_environment()

    def test_repetitions_and_abba_order_are_fail_closed(self):
        with patch.dict(
            os.environ,
            {
                "TEMPO_PD_KV_ATTR_REPETITIONS": "2",
                "TEMPO_PD_KV_ATTR_ARM_ORDER": "paired_abba",
            },
        ):
            self.assertEqual(client._repetitions_from_environment(), 2)
            self.assertEqual(
                client._arm_order_policy_from_environment(), "paired_abba")
        self.assertEqual(
            client._arm_order("paired_abba", 0), ("local", "remote"))
        self.assertEqual(
            client._arm_order("paired_abba", 1), ("remote", "local"))
        with patch.dict(
            os.environ, {"TEMPO_PD_KV_ATTR_REPETITIONS": "0"}
        ):
            with self.assertRaisesRegex(ValueError, r"\[1, 4\]"):
                client._repetitions_from_environment()
        with patch.dict(
            os.environ, {"TEMPO_PD_KV_ATTR_ARM_ORDER": "random"}
        ):
            with self.assertRaisesRegex(ValueError, "arm order"):
                client._arm_order_policy_from_environment()
        with patch.dict(os.environ, {"TEMPO_PD_KV_ATTR_RATES": "-1,4"}):
            with self.assertRaisesRegex(ValueError, "non-negative"):
                client._rates_from_environment()

    def test_local_remote_blocks_have_identical_semantics(self):
        tokenizer = _Tokenizer()
        template = tuple(range(client.PROMPT_TOKENS))
        pool = client._pool_prompts(tokenizer, template)
        local_rows, local_index, local_sha = client._block_rows(
            tokenizer=tokenizer,
            template=template,
            pool=pool,
            rate=4.0,
            rate_index=0,
            arm="local",
            duration_ms=4_000.0,
            foreground_rate=2.0,
        )
        remote_rows, remote_index, remote_sha = client._block_rows(
            tokenizer=tokenizer,
            template=template,
            pool=pool,
            rate=4.0,
            rate_index=0,
            arm="remote",
            duration_ms=4_000.0,
            foreground_rate=2.0,
        )
        self.assertEqual(local_sha, remote_sha)
        self.assertEqual(len(local_rows), 24)
        self.assertEqual(len(remote_rows), 24)
        self.assertEqual(
            sum(value["tenant"] == "p_only_remote_background"
                for value in local_index.values()),
            16,
        )
        background_ids = [
            request_id for request_id, metadata in local_index.items()
            if metadata["tenant"] == "p_only_remote_background"
        ]
        self.assertTrue(all(
            "-cache-p-only-measured-" in request_id
            and "-warm-" not in request_id
            for request_id in background_ids
        ))
        for request_id, metadata in remote_index.items():
            if metadata["tenant"] == "p_only_remote_background":
                self.assertEqual(
                    int(request_id.rsplit("-", 1)[1]) % 2,
                    int(metadata["pool_index"]) % 2,
                )

    def test_coupled_blocks_pair_semantics_without_reusing_cold_prompts(self):
        tokenizer = _Tokenizer()
        template = tuple(range(client.PROMPT_TOKENS))
        pool = client._pool_prompts(tokenizer, template)
        local_rows, local_index, local_sha = client._block_rows(
            tokenizer=tokenizer,
            template=template,
            pool=pool,
            rate=0.0,
            rate_index=0,
            arm="local",
            duration_ms=4_000.0,
            foreground_rate=2.0,
            decoder_hot_rate=22.4,
        )
        remote_rows, remote_index, remote_sha = client._block_rows(
            tokenizer=tokenizer,
            template=template,
            pool=pool,
            rate=0.0,
            rate_index=0,
            arm="remote",
            duration_ms=4_000.0,
            foreground_rate=2.0,
            decoder_hot_rate=22.4,
        )
        self.assertEqual(local_sha, remote_sha)
        self.assertEqual(len(local_rows), 97)
        self.assertEqual(len(remote_rows), 97)
        self.assertEqual(
            sum(value["tenant"] == "decoder_hot_background"
                for value in local_index.values()),
            89,
        )
        local_hot = {
            row["prompt"] for row in local_rows
            if local_index[row["request_id"]]["tenant"]
            == "decoder_hot_background"
        }
        remote_hot = {
            row["prompt"] for row in remote_rows
            if remote_index[row["request_id"]]["tenant"]
            == "decoder_hot_background"
        }
        self.assertTrue(local_hot.isdisjoint(remote_hot))

        next_rows, _, next_sha = client._block_rows(
            tokenizer=tokenizer,
            template=template,
            pool=pool,
            rate=0.0,
            rate_index=0,
            replicate_index=1,
            arm="local",
            duration_ms=4_000.0,
            foreground_rate=2.0,
            decoder_hot_rate=22.4,
        )
        self.assertNotEqual(local_sha, next_sha)
        self.assertTrue(
            {row["prompt"] for row in local_rows}.isdisjoint(
                {row["prompt"] for row in next_rows}
            )
        )

    def test_preseeded_command_selects_explicit_module(self):
        args = SimpleNamespace(
            base_url="http://frontend",
            model=client.Path("/model"),
            served_model_name="model",
            default_max_tokens=32,
            max_workers=64,
            timeout_s=600.0,
            seed=7,
            api_key_env=None,
        )
        command = client._stream_command(
            args,
            module=client.PRESEEDED_MODULE,
            workload=client.Path("/workload.jsonl"),
            output=client.Path("/raw.json"),
            run_id="run",
        )
        self.assertEqual(command[2], client.PRESEEDED_MODULE)
        self.assertNotIn("--request-rate", command)

    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(client._percentile(list(range(1, 101)), 0.99), 99)
        self.assertIsNone(client._percentile([], 0.99))

    def test_invalid_cassini_sample_remains_explicitly_missing(self):
        def stage(valid):
            return {"snapshots": [{"probe": {"cassini": {
                "valid": valid,
                "endpoint_id": "pair0-decoder",
                "invalid_reason": None if valid else "sample_window_exceeded",
                "window_ms": 4_000.0 if valid else 12_000.0,
            }}}]}

        quality = client._cassini_quality({
            "before": stage(True),
            "midpoint": stage(True),
            "after": stage(False),
        })
        self.assertFalse(quality["all_valid"])
        self.assertEqual(quality["samples_valid"], 2)
        self.assertTrue(quality["invalid_is_missing_not_zero"])
        self.assertEqual(
            quality["invalid_samples"][0]["invalid_reason"],
            "sample_window_exceeded",
        )


if __name__ == "__main__":
    unittest.main()
