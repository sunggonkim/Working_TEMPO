from __future__ import annotations

import unittest

from eval.sota_4node import tempo_pd_same_server_production_interleaved_router_v108 as router


class ProductionInterleavedRouterTest(unittest.TestCase):
    def test_interleaved_ids_extend_production_grammar(self) -> None:
        for arm in ("local", "tempo", "remote"):
            self.assertEqual(
                router.ProductionInterleavedCore._arm(
                    f"ssi-{arm}-r1-measured-mix512-0"),
                (arm, "measured"),
            )

    def test_balanced_warmup_ids_remain_supported(self) -> None:
        self.assertEqual(
            router.ProductionInterleavedCore._arm("ssb-tempo-r0-warm-p512-0"),
            ("tempo", "warm"),
        )

    def test_unknown_id_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            router.ProductionInterleavedCore._arm("unlabeled-request")


if __name__ == "__main__":
    unittest.main()
