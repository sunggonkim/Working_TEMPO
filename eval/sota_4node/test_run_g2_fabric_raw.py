import unittest

try:
    from .run_g2_fabric_raw import MODES
except ImportError:
    from run_g2_fabric_raw import MODES


class G2FabricRawRunnerTests(unittest.TestCase):
    def test_matrix_is_exact_composite_modes(self):
        self.assertEqual([mode for mode, _, _ in MODES], [
            "fg_only", "open_combined", "d2h_only", "persist_only", "combined"
        ])
        self.assertEqual(MODES[0][1], "none")
        self.assertEqual(MODES[2][2], "local_sink")


if __name__ == "__main__":
    unittest.main()
