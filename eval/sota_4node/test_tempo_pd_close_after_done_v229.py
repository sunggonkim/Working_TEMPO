import unittest
from pathlib import Path
from unittest.mock import patch

from eval.sota_4node import run_tempo_pd_stream_metrics_close_after_done_v225 as metrics
from eval.sota_4node import run_tempo_pd_same_server_hybrid_phase_client_close_v226 as client
from eval.sota_4node import vllm_lmcache_same_server_hybrid_phase_node_v227 as node


class CloseAfterDoneTest(unittest.TestCase):
    def test_client_rewrites_forced_drain_module(self):
        seen = []

        def fake_run(command, *args, **kwargs):
            seen.append(command)
            return 0

        with patch.object(client.subprocess, "run", fake_run):
            with patch.object(
                client.phase,
                "main",
                side_effect=lambda: client.subprocess.run(
                    ["python", "-m", client._OLD]
                ),
            ):
                client.main()
        self.assertEqual(seen, [["python", "-m", client._NEW]])

    def test_decision_poll_observes_complete(self):
        replies = [
            {"decisions": [{"phase": "remote_selected"}]},
            {"decisions": [{"phase": "complete"}]},
        ]
        with patch.object(metrics, "_ORIGINAL_FETCH", side_effect=replies) as fetch:
            with patch.object(metrics.time, "sleep"):
                value = metrics._fetch_decisions("http://x", 1.0)
        self.assertEqual(value["decisions"][0]["phase"], "complete")
        self.assertEqual(fetch.call_count, 2)

    def test_node_wires_close_after_done_client(self):
        command = node._client_command(
            None,
            base_url="x",
            model=Path("/m"),
            workload=Path("/w"),
            output=Path("/o"),
            mode="tempo_auto",
            run_id="r",
            request_rate=1.0,
            max_workers=1,
        )
        self.assertIn(
            "eval.sota_4node.run_tempo_pd_same_server_hybrid_phase_client_close_v226",
            command,
        )


if __name__ == "__main__":
    unittest.main()
