import unittest

from eval.sota_4node import run_tempo_pd_same_server_hybrid_phase_client_serial_lm_warm_v230 as client


class SerialWarmTest(unittest.TestCase):
    def test_only_unmeasured_lmcache_seed_is_serialized(self):
        base = ["python", "-m", "metrics", "--run-id", "epoch-warmup-00_lmcache_remote_r0", "--max-workers", "32", "--request-rate", "48"]
        changed = client._serial_warm_command(base)
        self.assertEqual(changed[changed.index("--max-workers") + 1], "1")
        self.assertEqual(base[base.index("--max-workers") + 1], "32")
        measured = list(base)
        measured[measured.index("epoch-warmup-00_lmcache_remote_r0")] = "epoch-02_lmcache_remote_r0"
        self.assertIs(client._serial_warm_command(measured), measured)


if __name__ == "__main__":
    unittest.main()
