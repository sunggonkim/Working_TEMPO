import unittest

from eval.sota_4node import tempo_pd_proxy_cache_control as control


class TempoPDProxyCacheControlTest(unittest.TestCase):
    def test_control_is_absent_from_producer_and_present_only_on_decoder(self):
        payload = {
            "prompt": [1, 2, 3],
            control.PROXY_DECODER_CONTROL_FIELD: 1,
            "vllm_xargs": {"kept": 7},
        }
        value = control.extract_decoder_cache_read_control(payload)
        self.assertEqual(value, 1)
        self.assertNotIn(control.PROXY_DECODER_CONTROL_FIELD, payload)
        self.assertNotIn(
            control.VLLM_SKIP_LOCAL_PREFIX_READ_XARG,
            payload["vllm_xargs"],
        )

        control.apply_decoder_cache_read_control(payload, value)
        self.assertEqual(
            payload["vllm_xargs"],
            {
                "kept": 7,
                control.VLLM_SKIP_LOCAL_PREFIX_READ_XARG: 1,
            },
        )

    def test_absent_control_leaves_payload_unchanged(self):
        payload = {"prompt": [1], "vllm_xargs": {"kept": 7}}
        before = dict(payload)
        value = control.extract_decoder_cache_read_control(payload)
        self.assertIsNone(value)
        self.assertIs(
            control.apply_decoder_cache_read_control(payload, value), payload)
        self.assertEqual(payload, before)

    def test_invalid_values_and_direct_xarg_fail_closed(self):
        for value in (True, False, "1", -1, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                control.extract_decoder_cache_read_control({
                    control.PROXY_DECODER_CONTROL_FIELD: value,
                })
        with self.assertRaisesRegex(ValueError, "proxy-only"):
            control.extract_decoder_cache_read_control({
                "vllm_xargs": {
                    control.VLLM_SKIP_LOCAL_PREFIX_READ_XARG: 1,
                },
            })

    def test_decoder_injection_rejects_collision_or_unconsumed_field(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            control.apply_decoder_cache_read_control({
                "vllm_xargs": {
                    control.VLLM_SKIP_LOCAL_PREFIX_READ_XARG: 0,
                },
            }, 1)
        with self.assertRaisesRegex(ValueError, "not consumed"):
            control.apply_decoder_cache_read_control({
                control.PROXY_DECODER_CONTROL_FIELD: 1,
            }, 1)


if __name__ == "__main__":
    unittest.main()
