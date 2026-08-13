from __future__ import annotations

import copy
import unittest

from tempo.foreground_path import validate_foreground_path
from tempo.resource_domain import ResourceDomain, allowed_counter_scopes, domain_contract


DOMAINS = (ResourceDomain.GPU_LOCAL, ResourceDomain.NVLINK_P2P)


def foreground_path() -> dict[str, object]:
    counters: dict[str, list[dict[str, object]]] = {}
    path_status: dict[str, str] = {}
    counter_support: dict[str, str] = {}
    path_evidence: dict[str, str] = {}
    counter_family: dict[str, str] = {}
    for domain in DOMAINS:
        name = domain.value
        scope = sorted(allowed_counter_scopes(domain))[0]
        path_status[name] = "observed"
        counter_support[name] = "supported"
        path_evidence[name] = domain_contract(domain).path_evidence
        counter_family[name] = domain_contract(domain).counter_family
        counters[name] = [
            {
                "domain": name,
                "sample_id": f"{name}-before",
                "source": "foreground-hardware-counter",
                "timestamp_ns": 1_000,
                "cumulative_bytes": 0,
                "cumulative_busy_ns": 0,
                "support": "supported",
                "scope": scope,
                "scope_id": f"{name}-pair-0",
                "intervention_id": "fg_only",
            },
            {
                "domain": name,
                "sample_id": f"{name}-after",
                "source": "foreground-hardware-counter",
                "timestamp_ns": 2_000,
                "cumulative_bytes": 4_096,
                "cumulative_busy_ns": 100,
                "support": "supported",
                "scope": scope,
                "scope_id": f"{name}-pair-0",
                "intervention_id": "fg_only",
            },
        ]
    return {
        "domains": sorted(item.value for item in DOMAINS),
        "path_status": path_status,
        "counter_support": counter_support,
        "path_evidence": path_evidence,
        "counter_family": counter_family,
        "counters": {key: counters[key] for key in sorted(counters)},
    }


class ForegroundPathTests(unittest.TestCase):
    def test_positive_scope_bound_path_is_accepted(self) -> None:
        result = validate_foreground_path(foreground_path())
        self.assertEqual(result["domains"], ["gpu_local", "nvlink_p2p"])

    def test_missing_domain_counter_is_rejected(self) -> None:
        candidate = foreground_path()
        del candidate["counters"]["nvlink_p2p"]
        with self.assertRaisesRegex(ValueError, "cover declared domains"):
            validate_foreground_path(candidate)

    def test_topology_only_zero_traffic_is_rejected(self) -> None:
        candidate = foreground_path()
        candidate["counters"]["gpu_local"][1]["cumulative_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "no positive byte traffic"):
            validate_foreground_path(candidate)

    def test_counter_scope_and_intervention_are_bound(self) -> None:
        candidate = foreground_path()
        candidate["counters"]["nvlink_p2p"][1]["intervention_id"] = "open_combined"
        with self.assertRaisesRegex(ValueError, "binding is invalid"):
            validate_foreground_path(candidate)

    def test_path_label_cannot_be_relabelled(self) -> None:
        candidate = foreground_path()
        candidate["path_evidence"]["gpu_local"] = "topology_guess"
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_foreground_path(candidate)


if __name__ == "__main__":
    unittest.main()
