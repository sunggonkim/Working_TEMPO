import unittest

from eval.sota_4node import tempo_pd_cache_reuse as reuse


class CacheReuseAssignmentTest(unittest.TestCase):
    def test_explicit_item_set_is_canonical_and_fail_closed(self):
        self.assertIsNone(reuse.parse_reuse_items("all"))
        self.assertEqual(
            reuse.parse_reuse_items("12,13,14,15"),
            frozenset({12, 13, 14, 15}),
        )
        for invalid in ("", "12,12", "012", "12, 13", "-1", "1000"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    reuse.parse_reuse_items(invalid)

    def test_cache_domains_reuse_only_selected_measured_items(self):
        selected = reuse.parse_reuse_items("12,13,14,15")
        warm_r0 = "epd-tempo-r0-warm-seed-o128-item-12"
        warm_r1 = "epd-tempo-r1-warm-item-12"
        selected_r0 = "epd-tempo-r0-measured-item-12"
        selected_r1 = "epd-tempo-r1-measured-item-12"
        isolated_r0 = "epd-tempo-r0-measured-item-20"
        isolated_r1 = "epd-tempo-r1-measured-item-20"

        self.assertEqual(
            reuse.cache_salt(warm_r0, selected),
            reuse.cache_salt(warm_r1, selected),
        )
        self.assertEqual(
            reuse.cache_salt(selected_r0, selected),
            reuse.cache_salt(selected_r1, selected),
        )
        self.assertNotEqual(
            reuse.cache_salt(isolated_r0, selected),
            reuse.cache_salt(isolated_r1, selected),
        )
        self.assertTrue(reuse.reuses_decoder_cache(selected_r0, selected))
        self.assertFalse(reuse.reuses_decoder_cache(isolated_r0, selected))

    def test_namespace_salt_is_phase_stable_and_arm_isolated(self):
        prompt_key = "a" * 64
        tempo = reuse.namespace_cache_salt(
            arm="tempo", prompt_key=prompt_key)
        self.assertEqual(
            tempo,
            reuse.namespace_cache_salt(
                arm="tempo", prompt_key=prompt_key),
        )
        self.assertNotEqual(
            tempo,
            reuse.namespace_cache_salt(
                arm="always_local", prompt_key=prompt_key),
        )
        self.assertLessEqual(len(tempo), 64)
        for invalid in ("", "A" * 64, "a" * 63, "g" * 64):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                reuse.namespace_cache_salt(
                    arm="tempo", prompt_key=invalid)


if __name__ == "__main__":
    unittest.main()
