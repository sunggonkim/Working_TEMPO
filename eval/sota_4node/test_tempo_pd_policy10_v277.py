import unittest

from eval.sota_4node import tempo_pd_same_server_policy10_router_v274 as policy


class Policy10Test(unittest.TestCase):
    def test_single_factor_bucket_change(self):
        self.assertNotIn((512, 32), policy.REMOTE_BUCKETS)
        self.assertIn((512, 64), policy.REMOTE_BUCKETS)
        self.assertIn((512, 128), policy.REMOTE_BUCKETS)
        self.assertIn((2048, 64), policy.REMOTE_BUCKETS)
        self.assertIn((2048, 256), policy.REMOTE_BUCKETS)
        self.assertEqual(len(policy.REMOTE_BUCKETS), 4)


if __name__ == "__main__":
    unittest.main()
