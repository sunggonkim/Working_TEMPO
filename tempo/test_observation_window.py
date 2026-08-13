from __future__ import annotations

import unittest

from tempo.observation_window import (
    ObservationInterval,
    canonicalize_observation_windows,
    join_observation_window,
    observation_window_contract,
    serialize_joined_observation_window,
    serialize_observation_interval,
    validate_observation_windows,
)
from tempo.resource_domain import ResourceDomain


def _interval(role: str, *, start: int = 100, end: int = 300, **overrides: object) -> ObservationInterval:
    values: dict[str, object] = {
        "observation_id": "obs-1",
        "mode": "d2h_only",
        "rank": 0,
        "event_id": "event-16",
        "clock_domain": "corrected-monotonic-v1",
        "source_snapshot_id": "snapshot-a",
        "source": role + "-collector",
        "start_ns": start,
        "end_ns": end,
        "role": role,
        "uncertainty_ns": 10,
    }
    if role == "counter":
        values["domain"] = ResourceDomain.PCIE_HOST
    values.update(overrides)
    return ObservationInterval(**values)


def _raw(interval: ObservationInterval) -> dict[str, object]:
    return {
        "observation_id": interval.observation_id,
        "mode": interval.mode,
        "rank": interval.rank,
        "event_id": interval.event_id,
        "clock_domain": interval.clock_domain,
        "source_snapshot_id": interval.source_snapshot_id,
        "source": interval.source,
        "start_ns": interval.start_ns,
        "end_ns": interval.end_ns,
        "role": interval.role,
        "domain": None if interval.domain is None else interval.domain.value,
        "uncertainty_ns": interval.uncertainty_ns,
    }


class ObservationWindowTests(unittest.TestCase):
    def test_join_returns_exact_common_interval_and_domains(self) -> None:
        joined = join_observation_window(
            _interval("foreground", start=100, end=400),
            _interval("auxiliary", start=150, end=350),
            [_interval("counter", start=175, end=325)],
        )
        self.assertEqual((joined.start_ns, joined.end_ns, joined.overlap_ns), (175, 325, 150))
        self.assertEqual(joined.counter_domains, (ResourceDomain.PCIE_HOST,))
        self.assertTrue(joined.uncertainty_safe)

    def test_foreground_only_still_requires_a_measured_counter(self) -> None:
        joined = join_observation_window(_interval("foreground"), None, [_interval("counter")])
        self.assertEqual(joined.overlap_ns, 200)

        with self.assertRaisesRegex(ValueError, "counter interval"):
            join_observation_window(_interval("foreground"), None, [])

    def test_observation_id_and_clock_provenance_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "identity/clock provenance"):
            join_observation_window(
                _interval("foreground"),
                _interval("auxiliary", observation_id="other"),
                [_interval("counter")],
            )
        with self.assertRaisesRegex(ValueError, "identity/clock provenance"):
            join_observation_window(
                _interval("foreground"),
                _interval("auxiliary", clock_domain="raw-monotonic"),
                [_interval("counter")],
            )

    def test_nonoverlap_is_not_a_causal_join(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            join_observation_window(
                _interval("foreground", start=100, end=150),
                _interval("auxiliary", start=151, end=200),
                [_interval("counter", start=151, end=200)],
            )

    def test_counter_outside_common_window_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            join_observation_window(
                _interval("foreground", start=100, end=300),
                _interval("auxiliary", start=120, end=280),
                [_interval("counter", start=300, end=400)],
            )

    def test_domain_and_interval_shape_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "counter intervals require"):
            ObservationInterval(
                observation_id="obs",
                mode="m",
                rank=0,
                event_id="e",
                clock_domain="c",
                source_snapshot_id="s",
                source="x",
                start_ns=1,
                end_ns=2,
                role="counter",
            )
        with self.assertRaisesRegex(ValueError, "interval must satisfy"):
            _interval("foreground", start=10, end=10)

    def test_uncertainty_is_reported_without_rewriting_the_overlap(self) -> None:
        joined = join_observation_window(
            _interval("foreground", uncertainty_ns=250),
            _interval("auxiliary", uncertainty_ns=250),
            [_interval("counter", uncertainty_ns=250)],
        )
        self.assertEqual(joined.overlap_ns, 200)
        self.assertFalse(joined.uncertainty_safe)

    def test_json_window_groups_require_foreground_auxiliary_and_counter(self) -> None:
        raw = [
            _raw(_interval("foreground")),
            _raw(_interval("auxiliary", start=120, end=280)),
            _raw(_interval("counter", start=140, end=260)),
        ]
        joined = validate_observation_windows(
            raw,
            expected_mode="d2h_only",
            expected_observation_id="obs-1",
            require_auxiliary=True,
        )
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0].overlap_ns, 120)

        with self.assertRaisesRegex(ValueError, "requires exactly one auxiliary"):
            validate_observation_windows(
                [raw[0], raw[2]],
                expected_mode="d2h_only",
                expected_observation_id="obs-1",
                require_auxiliary=True,
            )

    def test_json_window_rejects_mismatched_observation_id(self) -> None:
        raw = [_raw(_interval("foreground"))]
        raw[0]["observation_id"] = "other"
        with self.assertRaisesRegex(ValueError, "does not match metrics"):
            validate_observation_windows(
                raw,
                expected_mode="fg_only",
                expected_observation_id="obs-1",
                require_auxiliary=False,
            )

    def test_serializers_are_exact_and_do_not_coerce(self) -> None:
        interval = _interval("counter")
        encoded = serialize_observation_interval(interval)
        self.assertEqual(set(encoded), set(
            {
                "observation_id", "mode", "rank", "event_id", "clock_domain",
                "source_snapshot_id", "source", "start_ns", "end_ns", "role",
                "domain", "uncertainty_ns",
            }
        ))
        self.assertEqual(encoded["rank"], 0)
        with self.assertRaises(TypeError):
            serialize_observation_interval(object())

        joined = join_observation_window(
            _interval("foreground"),
            _interval("auxiliary", start=120, end=280),
            [_interval("counter", start=140, end=260)],
        )
        joined_record = serialize_joined_observation_window(joined)
        self.assertEqual(joined_record["counter_domains"], [ResourceDomain.PCIE_HOST.value])
        self.assertTrue(joined_record["uncertainty_safe"])

    def test_canonicalize_materializes_sorted_joined_windows(self) -> None:
        raw = []
        for rank, event_id in ((1, "event-b"), (0, "event-a")):
            for role, start, end in (
                ("foreground", 100, 400),
                ("auxiliary", 120, 380),
                ("counter", 150, 350),
            ):
                raw.append(_raw(ObservationInterval(
                    observation_id="obs-1",
                    mode="d2h_only",
                    rank=rank,
                    event_id=event_id,
                    clock_domain="corrected-monotonic-v1",
                    source_snapshot_id="snapshot-a",
                    source=role + "-collector",
                    start_ns=start,
                    end_ns=end,
                    role=role,
                    domain=ResourceDomain.PCIE_HOST if role == "counter" else None,
                    uncertainty_ns=10,
                )))
        records = canonicalize_observation_windows(
            raw,
            expected_mode="d2h_only",
            expected_observation_id="obs-1",
            require_auxiliary=True,
        )
        self.assertEqual([(item["rank"], item["event_id"]) for item in records], [(0, "event-a"), (1, "event-b")])
        self.assertEqual(records[0]["overlap_ns"], 200)

    def test_contract_names_joined_schema(self) -> None:
        contract = observation_window_contract()
        self.assertEqual(contract["schema_version"], "tempo-rd-observation-window-1")
        self.assertIn("joined_keys", contract)


if __name__ == "__main__":
    unittest.main()
