from __future__ import annotations

from pathlib import Path
import unittest

from eval.sota_4node import tempo_pd_same_server_cache_catalog_router_v139 as router
from tempo.pd_admission import PDRoute


class CacheCatalogRevisionTest(unittest.TestCase):
    def test_one_removed_remote_bucket(self) -> None:
        self.assertIs(router._selected_route(2048, 64), PDRoute.DECODER_LOCAL)
        self.assertIs(router._selected_route(512, 64), PDRoute.REMOTE_PREFILL)
        remote = sum(router._selected_route(p, o) is PDRoute.REMOTE_PREFILL
                     for p in (512, 1230, 2048) for o in (16, 32, 64, 128))
        self.assertEqual(remote, 3)

    def test_single_bounded_step(self) -> None:
        path = Path(__file__).with_name(
            "run_tempo_pd_same_server_cache_catalog_v141_in_allocation.sh")
        text = path.read_text(encoding="utf-8")
        self.assertEqual(text.count("srun --exact"), 1)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
