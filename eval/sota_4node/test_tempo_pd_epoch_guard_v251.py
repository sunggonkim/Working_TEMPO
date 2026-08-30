import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.run_tempo_pd_same_server_epoch_guard_client_v249 import select_mode
from eval.sota_4node.tempo_pd_same_server_epoch_guard_router_v248 import load_epoch_mode, MODE_SCHEMA


class EpochGuardTest(unittest.TestCase):
    def test_selects_only_with_throughput_margin_and_tail_nonregression(self):
        local = [{"throughput_per_s": 10.0, "e2e_max_ms": 100.0}] * 3
        tempo = [{"throughput_per_s": 10.1, "e2e_max_ms": 99.0}] * 3
        self.assertEqual(select_mode(local, tempo)["selected_mode"], "policy8")
        tempo = [{"throughput_per_s": 10.1, "e2e_max_ms": 101.0}] * 3
        self.assertEqual(select_mode(local, tempo)["selected_mode"], "fixed_local")

    def test_mode_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "mode.json"
            path.write_text(json.dumps({
                "schema": MODE_SCHEMA,
                "selected_mode": "fixed_local",
                "calibration_replicates_per_candidate": 3,
            }))
            self.assertEqual(load_epoch_mode(path), "fixed_local")
            path.write_text("{}")
            with self.assertRaises(ValueError):
                load_epoch_mode(path)


if __name__ == "__main__":
    unittest.main()
