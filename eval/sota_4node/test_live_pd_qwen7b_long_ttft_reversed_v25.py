from __future__ import annotations

import unittest

from eval.sota_4node import vllm_lmcache_live_pd_node_v25 as node


class ReversedLifecycleTest(unittest.TestCase):
    def test_tempo_runs_before_lmcache(self) -> None:
        self.assertEqual(
            node.REVERSED_MODES,
            ("tempo_admission", "lmcache_always_remote"),
        )


if __name__ == "__main__":
    unittest.main()
