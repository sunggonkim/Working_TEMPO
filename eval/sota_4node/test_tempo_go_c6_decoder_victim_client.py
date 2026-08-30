from __future__ import annotations

import argparse
import subprocess
import unittest
from unittest import mock

from eval.sota_4node import run_tempo_go_c6_decoder_victim_client as client


_IDENTITIES = (
    "pair0-prefill",
    "pair0-decoder",
    "pair1-prefill",
    "pair1-decoder",
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


class _Child:
    def __init__(self, clock: _Clock, exit_at_s: float) -> None:
        self.clock = clock
        self.exit_at_s = exit_at_s
        self.return_code: int | None = None

    def wait(self, timeout: float) -> int:
        deadline = self.clock.now + timeout
        if deadline < self.exit_at_s:
            self.clock.now = deadline
            raise subprocess.TimeoutExpired(["child"], timeout)
        self.clock.now = self.exit_at_s
        self.return_code = 0
        return 0

    def poll(self) -> int | None:
        return self.return_code


class C6DecoderVictimClientTests(unittest.TestCase):
    def test_cadenced_runner_bridges_both_halves_of_long_arm(self) -> None:
        clock = _Clock()
        child = _Child(clock, exit_at_s=40.0)
        sequence = 0

        def capture(_urls, *, stage: str, require_valid_delta: bool):
            nonlocal sequence
            sequence += 1
            return {
                "schema": client.fixed.ENDPOINT_EVIDENCE_SCHEMA,
                "stage": stage,
                "snapshots": [
                    {
                        "probe": {
                            "endpoint": {
                                "endpoint_id": endpoint_id,
                                "sequence": sequence,
                            },
                            "cassini": {
                                "sequence": sequence,
                                "valid": require_valid_delta,
                            },
                        }
                    }
                    for endpoint_id in _IDENTITIES
                ],
            }

        args = argparse.Namespace(
            endpoint_evidence_url=[f"http://n{i}" for i in range(4)],
            phase_duration_ms=40_000.0,
            timeout_s=120.0,
        )
        with (
            mock.patch.object(client.subprocess, "Popen", return_value=child),
            mock.patch.object(client.time, "monotonic", clock.monotonic),
            mock.patch.object(
                client.fixed, "_capture_endpoint_evidence", side_effect=capture
            ),
        ):
            evidence = client._run_child_with_cadenced_endpoint_evidence(
                ["child"], args=args
            )

        self.assertEqual(evidence["midpoint_target_elapsed_s"], 20.0)
        self.assertEqual(evidence["child_elapsed_s"], 40.0)
        self.assertEqual(
            [row["segment"] for row in evidence["cassini_bridges"]],
            [
                "before_midpoint",
                "before_midpoint",
                "before_midpoint",
                "after_midpoint",
                "after_midpoint",
                "after_midpoint",
            ],
        )


if __name__ == "__main__":
    unittest.main()
