import json
import unittest

from tempo.pd_decoder_cache_evidence import (
    CACHE_BREAKDOWN_SCHEMA,
    CACHE_BREAKDOWN_SCHEMA_FIELD,
    EXTERNAL_CACHED_TOKENS_FIELD,
    LOCAL_CACHED_TOKENS_FIELD,
    VLLMDecoderCacheSSEParser,
    full_prefix_hit_tokens,
)


def event(payload):
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def usage(
    *, prompt=32, completion=2, cached=16, local=16, external=0,
):
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {
                "cached_tokens": cached,
                CACHE_BREAKDOWN_SCHEMA_FIELD: CACHE_BREAKDOWN_SCHEMA,
                LOCAL_CACHED_TOKENS_FIELD: local,
                EXTERNAL_CACHED_TOKENS_FIELD: external,
            },
        },
    }


class DecoderCacheEvidenceTest(unittest.TestCase):
    def test_full_hit_is_last_token_excluded_and_block_aligned(self):
        self.assertEqual(full_prefix_hit_tokens(512), 496)
        self.assertEqual(full_prefix_hit_tokens(2048), 2032)
        self.assertEqual(full_prefix_hit_tokens(4094), 4080)

    def test_chunk_boundaries_and_crlf_are_irrelevant(self):
        parser = VLLMDecoderCacheSSEParser()
        payload = (
            b'data: {"choices":[{"text":"x"}],"usage":null}\r\n\r\n'
            + event(usage()).replace(b"\n", b"\r\n")
            + b"data: [DONE]\r\n\r\n"
        )
        for index in range(0, len(payload), 7):
            parser.feed(payload[index:index + 7])
        evidence = parser.finish(expected_prompt_tokens=32)
        self.assertEqual(evidence.cached_tokens, 16)
        self.assertEqual(evidence.local_cached_tokens, 16)
        self.assertEqual(evidence.external_cached_tokens, 0)
        self.assertEqual(evidence.total_tokens, 34)

    def test_missing_details_done_or_exact_geometry_fails_closed(self):
        cases = {
            "details": event({
                "choices": [],
                "usage": {
                    "prompt_tokens": 32,
                    "completion_tokens": 2,
                    "total_tokens": 34,
                },
            }) + b"data: [DONE]\n\n",
            "done": event(usage()),
            "geometry": event(usage(prompt=31)) + b"data: [DONE]\n\n",
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                parser = VLLMDecoderCacheSSEParser()
                parser.feed(payload)
                with self.assertRaises(ValueError):
                    parser.finish(expected_prompt_tokens=32)

    def test_duplicate_or_nonfinal_cache_evidence_fails_closed(self):
        duplicate = event(usage()) + event(usage()) + b"data: [DONE]\n\n"
        nonfinal = event({
            **usage(), "choices": [{"text": "x"}],
        })
        for payload in (duplicate, nonfinal):
            parser = VLLMDecoderCacheSSEParser()
            with self.assertRaises(ValueError):
                parser.feed(payload)

    def test_partial_event_and_data_after_done_fail_closed(self):
        parser = VLLMDecoderCacheSSEParser()
        parser.feed(event(usage()) + b"data: [DONE]\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parser.finish(expected_prompt_tokens=32)

        parser = VLLMDecoderCacheSSEParser()
        with self.assertRaisesRegex(ValueError, "follows DONE"):
            parser.feed(
                event(usage()) + b"data: [DONE]\n\n" + event({"usage": None})
            )

    def test_missing_or_inconsistent_source_breakdown_fails_closed(self):
        missing = usage()
        del missing["usage"]["prompt_tokens_details"][
            CACHE_BREAKDOWN_SCHEMA_FIELD]
        inconsistent = usage(cached=16, local=8, external=7)
        for payload in (missing, inconsistent):
            parser = VLLMDecoderCacheSSEParser()
            with self.assertRaises(ValueError):
                parser.feed(event(payload) + b"data: [DONE]\n\n")

    def test_remote_local_and_external_hits_remain_distinct(self):
        parser = VLLMDecoderCacheSSEParser()
        parser.feed(
            event(usage(
                prompt=33, completion=1, cached=32,
                local=16, external=16,
            ))
            + b"data: [DONE]\n\n"
        )
        evidence = parser.finish(expected_prompt_tokens=33)
        self.assertEqual(evidence.cached_tokens, 32)
        self.assertEqual(evidence.local_cached_tokens, 16)
        self.assertEqual(evidence.external_cached_tokens, 16)


if __name__ == "__main__":
    unittest.main()
