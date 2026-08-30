from __future__ import annotations

import unittest
from pathlib import Path

from eval.sota_4node import run_tempo_pd_same_server_hybrid_saturation_client_v190 as client
from eval.sota_4node.tempo_pd_same_server_hybrid_saturation_router_v191 import SaturationHybridCore


class SaturationV194Test(unittest.TestCase):
    def test_frozen_orders_and_extra_local_replicates(self) -> None:
        self.assertEqual(len(client._WARM), 3)
        self.assertEqual(len(client._MEASURED), 6)
        self.assertNotIn("lmcache_remote", client._WARM + client._MEASURED)
        self.assertEqual(SaturationHybridCore._arm("ssb-local-r3-measured-x"),
                         ("local", "measured"))

    def test_launcher_is_one_bounded_step(self) -> None:
        root = Path(__file__).resolve().parent
        source = (root / "run_tempo_pd_same_server_hybrid_saturation_v194_in_allocation.sh").read_text()
        self.assertEqual(source.count(" srun "), 1)
        self.assertIn(" 64 32 128 ", source)


if __name__ == "__main__":
    unittest.main()
