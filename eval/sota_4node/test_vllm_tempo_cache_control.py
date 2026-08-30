import json
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

from eval.sota_4node import vllm_tempo_cache_control as control


class VLLMTempoCacheControlTest(unittest.TestCase):
    def test_absent_control_preserves_vllm_default(self):
        params = SimpleNamespace(skip_reading_prefix_cache=False)
        self.assertIs(control.apply_cache_read_control(params, None), params)
        self.assertFalse(params.skip_reading_prefix_cache)

    def test_exact_integer_controls_existing_vllm_field(self):
        for raw, expected in ((0, False), (1, True)):
            with self.subTest(raw=raw):
                params = SimpleNamespace(
                    skip_reading_prefix_cache=not expected,
                    extra_args={control.XARG: raw, "kept": 7},
                )
                control.apply_cache_read_control(
                    params, {control.XARG: raw})
                self.assertIs(params.skip_reading_prefix_cache, expected)
                self.assertEqual(params.extra_args, {"kept": 7})

    def test_boolean_string_and_missing_vllm_seam_fail_closed(self):
        for raw in (True, False, "1", -1, 2):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                control.apply_cache_read_control(
                    SimpleNamespace(skip_reading_prefix_cache=False),
                    {control.XARG: raw},
                )
        with self.assertRaises(RuntimeError):
            control.apply_cache_read_control(
                SimpleNamespace(), {control.XARG: 1})

    def test_existing_prefill_stats_split_is_carried_to_request_output(self):
        state = SimpleNamespace(is_prefilling=True)
        engine_output = SimpleNamespace(prefill_stats=SimpleNamespace(
            num_prompt_tokens=513,
            num_cached_tokens=512,
            num_local_cached_tokens=496,
            num_external_cached_tokens=16,
        ))
        control.capture_prefill_cache_breakdown(state, engine_output)
        request_output = SimpleNamespace()
        control.attach_cache_breakdown(state, request_output)
        observed = control.output_cache_breakdown(request_output)
        self.assertEqual(observed.prompt_tokens, 513)
        self.assertEqual(observed.local_cached_tokens, 496)
        self.assertEqual(observed.external_cached_tokens, 16)
        self.assertEqual(observed.cached_tokens, 512)

    def test_invalid_or_missing_prefill_stats_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            control.cache_breakdown_from_prefill_stats(SimpleNamespace(
                num_prompt_tokens=33,
                num_cached_tokens=32,
                num_local_cached_tokens=16,
                num_external_cached_tokens=15,
            ))
        with self.assertRaisesRegex(RuntimeError, "lacks"):
            control.output_cache_breakdown(SimpleNamespace())

    def test_installed_prefill_stats_shape_is_checked_at_startup(self):
        @dataclass
        class Compatible:
            num_prompt_tokens: int = 0
            num_cached_tokens: int = 0
            num_local_cached_tokens: int = 0
            num_external_cached_tokens: int = 0

        @dataclass
        class MissingExternal:
            num_prompt_tokens: int = 0
            num_cached_tokens: int = 0
            num_local_cached_tokens: int = 0

        control.require_prefill_stats_compatibility(Compatible)
        with self.assertRaisesRegex(RuntimeError, "num_external_cached_tokens"):
            control.require_prefill_stats_compatibility(MissingExternal)

    def test_only_final_usage_chunk_receives_exact_breakdown(self):
        observed = control.CacheBreakdown(
            prompt_tokens=33,
            cached_tokens=32,
            local_cached_tokens=16,
            external_cached_tokens=16,
        )
        token_chunk = (
            'data: {"choices":[{"text":"x"}],"usage":null}\n\n')
        unchanged, injected, done = control.inject_cache_breakdown_sse(
            token_chunk, observed)
        self.assertEqual(unchanged, token_chunk)
        self.assertFalse(injected)
        self.assertFalse(done)

        final = (
            'data: {"choices":[],"usage":{"prompt_tokens":33,'
            '"completion_tokens":1,"total_tokens":34,'
            '"prompt_tokens_details":{"cached_tokens":32}}}\n\n'
        )
        rewritten, injected, done = control.inject_cache_breakdown_sse(
            final, observed)
        self.assertTrue(injected)
        self.assertFalse(done)
        payload = json.loads(rewritten[6:-2])
        details = payload["usage"]["prompt_tokens_details"]
        self.assertEqual(
            details[control.CACHE_BREAKDOWN_SCHEMA_FIELD],
            control.CACHE_BREAKDOWN_SCHEMA,
        )
        self.assertEqual(details[control.LOCAL_CACHED_TOKENS_FIELD], 16)
        self.assertEqual(details[control.EXTERNAL_CACHED_TOKENS_FIELD], 16)
        done_chunk, injected, done = control.inject_cache_breakdown_sse(
            "data: [DONE]\n\n", observed)
        self.assertEqual(done_chunk, "data: [DONE]\n\n")
        self.assertFalse(injected)
        self.assertTrue(done)

    def test_usage_must_match_internal_prefill_stats(self):
        observed = control.CacheBreakdown(
            prompt_tokens=33,
            cached_tokens=32,
            local_cached_tokens=0,
            external_cached_tokens=32,
        )
        bad = (
            'data: {"choices":[],"usage":{"prompt_tokens":33,'
            '"completion_tokens":1,"total_tokens":34,'
            '"prompt_tokens_details":{"cached_tokens":16}}}\n\n'
        )
        with self.assertRaisesRegex(RuntimeError, "differs"):
            control.inject_cache_breakdown_sse(bad, observed)


if __name__ == "__main__":
    unittest.main()
