from __future__ import annotations

import unittest

from eval.sota_4node import vllm_decode_quiescence_gate_launch_v3 as hook
from eval.sota_4node import vllm_quiescence_wave_protocol_v4 as protocol


class WaveProtocolTests(unittest.TestCase):
    def ready(self) -> hook.ReadyEvent:
        return hook.ReadyEvent(
            2,
            "cmpl-tempo-scout-test",
            30,
            31,
            10,
            11,
            12,
            12,
        )

    def test_bulk_round_trip(self) -> None:
        event = self.ready()
        frame = protocol.ReleaseFrame.wave(
            event,
            mode="quiescent_tempo_bulk",
            completed_bytes=128 << 20,
            source_elapsed_ns=(1,) * 8,
            wave_elapsed_ns=2,
        )
        decoded = protocol.ReleaseFrame.from_payload(frame.to_payload(), event=event)
        self.assertEqual(decoded, frame)

    def test_noop_round_trip(self) -> None:
        event = self.ready()
        frame = protocol.ReleaseFrame.noop(event)
        self.assertEqual(
            protocol.ReleaseFrame.from_payload(frame.to_payload(), event=event), frame
        )

    def test_wrong_geometry_rejected(self) -> None:
        event = self.ready()
        frame = protocol.ReleaseFrame.wave(
            event,
            mode="quiescent_lmcache_bulk",
            completed_bytes=128 << 20,
            source_elapsed_ns=(1,) * 7,
            wave_elapsed_ns=2,
        )
        with self.assertRaisesRegex(ValueError, "eight"):
            frame.to_payload()


if __name__ == "__main__":
    unittest.main()
