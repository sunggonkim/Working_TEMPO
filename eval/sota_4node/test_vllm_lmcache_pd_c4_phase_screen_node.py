from __future__ import annotations

import os
import unittest
from unittest import mock

from eval.sota_4node import vllm_lmcache_pd_c4_phase_screen_node as c4


class C4LifecycleTimeoutTest(unittest.TestCase):
    def test_default_exceeds_inherited_fifteen_minute_timeout(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEMPO_PD_C4_LIFECYCLE_S", None)
            self.assertEqual(c4._lifecycle_timeout(), 3600.0)
            self.assertGreater(c4._lifecycle_timeout(), c4.common.LIFECYCLE_S)

    def test_explicit_timeout_is_bounded(self) -> None:
        with mock.patch.dict(os.environ,
                             {"TEMPO_PD_C4_LIFECYCLE_S": "5400"}):
            self.assertEqual(c4._lifecycle_timeout(), 5400.0)
        with mock.patch.dict(os.environ,
                             {"TEMPO_PD_C4_LIFECYCLE_S": "899"}):
            with self.assertRaisesRegex(ValueError, "C4 lifecycle"):
                c4._lifecycle_timeout()


if __name__ == "__main__":
    unittest.main()
