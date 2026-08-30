from __future__ import annotations

from pathlib import Path
import unittest


class ChunkTests(unittest.TestCase):
    def test_only_chunk_geometry_changes(self):
        root = Path(__file__).resolve().parent
        source = (root / "vllm_lmcache_chunk256_node_v7.py").read_text()
        self.assertIn('replace("chunk_size: 64", "chunk_size: 256")', source)
        self.assertIn('command[index] = "256"', source)
        launcher = (root / "run_lmcache_chunk256_v7_in_allocation.sh").read_text()
        self.assertEqual(launcher.count("srun "), 1)
        self.assertNotIn("salloc", launcher)


if __name__ == "__main__":
    unittest.main()
