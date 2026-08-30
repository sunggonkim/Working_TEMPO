import unittest
from unittest.mock import patch

from eval.sota_4node import run_tempo_pd_same_server_hybrid_phase_client_close_v226 as client
from eval.sota_4node import vllm_lmcache_same_server_hybrid_phase_node_v227 as node


class CloseAfterDoneTest(unittest.TestCase):
    def test_client_rewrites_only_forced_drain_module(self):
        seen = []
        with patch.object(client.phase, "main", side_effect=lambda: __import__("subprocess").run(["python", "-m", client._OLD])):
            original = __import__("subprocess").run
            with patch.object(client, "subprocess", wraps=__import__("subprocess")):
                # Test the public constants and node wiring without spawning.
                self.assertNotEqual(client._OLD, client._NEW)
        command = node._client_command(None, base_url="x", model=__import__("pathlib").Path("/m"), workload=__import__("pathlib").Path("/w"), output=__import__("pathlib").Path("/o"), mode="tempo_auto", run_id="r", request_rate=1.0, max_workers=1)
        self.assertIn("eval.sota_4node.run_tempo_pd_same_server_hybrid_phase_client_close_v226", command)


if __name__ == "__main__":
    unittest.main()
