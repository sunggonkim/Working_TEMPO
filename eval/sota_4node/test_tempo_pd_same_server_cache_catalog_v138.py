from __future__ import annotations

from pathlib import Path
import unittest

from eval.sota_4node import vllm_lmcache_same_server_cache_catalog_node_v138 as node


class CacheCatalogHarnessTest(unittest.TestCase):
    def test_client_and_router_are_exactly_replaced(self) -> None:
        source = Path(node.__file__).read_text(encoding="utf-8")
        self.assertIn("run_tempo_pd_same_server_cache_catalog_client_v136", source)
        self.assertIn("tempo_pd_same_server_cache_catalog_router_v136", source)

    def test_launcher_is_single_bounded_step(self) -> None:
        root = Path(__file__).resolve().parent
        text = (root / "run_tempo_pd_same_server_cache_catalog_v138_in_allocation.sh").read_text()
        self.assertEqual(text.count("srun --exact"), 1)
        self.assertIn("--time=00:43:00", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
