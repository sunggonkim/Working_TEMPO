import unittest
from unittest.mock import patch

from eval.sota_4node import vllm_lmcache_same_server_hybrid_phase_node_v233 as node


class PhaseNodeTest(unittest.TestCase):
    def test_main_delegates_through_phase_main(self):
        observed = {}

        def fake_phase_main():
            observed["client"] = node.phase._client_command
            return 17

        original_router = node.phase._router_command
        with patch.object(node.phase, "main", side_effect=fake_phase_main):
            self.assertEqual(node.main(), 17)
        self.assertIs(observed["client"], node._client_command)
        self.assertIs(node.phase._router_command, original_router)


if __name__ == "__main__":
    unittest.main()
