import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.run_tempo_pd_same_server_tri_epoch_guard_client_v256 import select_mode
from eval.sota_4node.tempo_pd_same_server_tri_epoch_guard_router_v255 import load_epoch_mode


def arm(throughput, e2e):
    return [{"throughput_per_s": throughput, "e2e_max_ms": e2e}] * 3


class TriEpochGuardTest(unittest.TestCase):
    def test_selects_policy_only_on_clear_win(self):
        result = select_mode({"local": arm(100, 10), "remote": arm(101, 9),
                              "tempo": arm(102, 8)})
        self.assertEqual(result["selected_mode"], "policy8")

    def test_falls_back_to_remote_without_clear_win(self):
        result = select_mode({"local": arm(100, 10), "remote": arm(101, 9),
                              "tempo": arm(101.2, 8)})
        self.assertEqual(result["selected_mode"], "lmcache_remote")

    def test_selects_local_only_on_clear_remote_win(self):
        result = select_mode({"local": arm(102, 8), "remote": arm(100, 9),
                              "tempo": arm(101, 7)})
        self.assertEqual(result["selected_mode"], "fixed_local")

    def test_loader_rejects_old_schema(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "mode.json"
            path.write_text('{"schema":"old","selected_mode":"policy8",'
                            '"calibration_replicates_per_candidate":3}')
            with self.assertRaises(ValueError):
                load_epoch_mode(path)


if __name__ == "__main__":
    unittest.main()
