#!/usr/bin/env python3
"""CPU-only tests for the G1/G2 tier-attribution manifest contract."""

from __future__ import annotations

import unittest

from tempo.domain_evidence import CounterSupport, DomainEvidence, PathStatus
from tempo.causal_gate import CausalModeRecord
from tempo.resource_domain import EvidenceLevel, ResourceDomain, domain_contract
from tempo.tier_attribution import (
    REQUIRED_G1_MODES,
    required_domains_for_modes,
    evaluate_tier_attribution,
    validate_mode_evidence,
    validate_attribution_manifest,
)


def manifest() -> dict[str, object]:
    return {
        "schema_version": "tempo-rd-tier-attribution-1",
        "world_size": 4,
        "nodes": 1,
        "state_bytes_per_rank": 384 * 1024 * 1024,
        "logical_file_extent_bytes": 385 * 1024 * 1024,
        "deadline_ns": 1_000_000_000,
        "checkpoint_steps": [16, 52],
        "modes": sorted(REQUIRED_G1_MODES),
        "required_domains": sorted(
            domain.value for domain in required_domains_for_modes(sorted(REQUIRED_G1_MODES))
        ),
        "evidence_contract": {
            "counter_support_values": [
                "ambiguous", "not_collected", "not_supported", "supported"
            ],
            "path_status_values": ["declared", "not_traversed", "observed", "unknown"],
            "causal_requires": [
                "interventional", "observed_path", "supported_counters",
                "tail_delta_above_uncertainty"
            ],
        },
    }


class TierAttributionTests(unittest.TestCase):
    def test_production_like_manifest_is_valid(self) -> None:
        validate_attribution_manifest(manifest())
        declared = set(manifest()["required_domains"])
        self.assertEqual(
            declared,
            {
                "gpu_local",
                "pcie_host",
                "host_numa",
                "nic_fabric",
                "slingshot_fabric",
                "persistent_endpoint",
            },
        )
        # NVLink is intentionally not inferred for a GPU->host checkpoint
        # path; it becomes eligible only through the explicit P2P mode.
        self.assertNotIn("nvlink_p2p", declared)
        self.assertIn(
            ResourceDomain.PCIE_HOST,
            required_domains_for_modes(manifest()["modes"]),
        )
        self.assertIn(
            ResourceDomain.PERSISTENT_ENDPOINT,
            required_domains_for_modes(manifest()["modes"]),
        )

    def test_missing_mode_fails_closed(self) -> None:
        candidate = manifest()
        candidate["modes"] = [mode for mode in candidate["modes"] if mode != "d2h_only"]
        with self.assertRaises(ValueError):
            validate_attribution_manifest(candidate)

    def test_duplicate_mode_fails_closed(self) -> None:
        candidate = manifest()
        candidate["modes"] = list(candidate["modes"]) + ["combined"]
        with self.assertRaises(ValueError):
            validate_attribution_manifest(candidate)

    def test_geometry_type_coercion_fails_closed(self) -> None:
        candidate = manifest()
        candidate["state_bytes_per_rank"] = float(candidate["state_bytes_per_rank"])
        with self.assertRaises(ValueError):
            validate_attribution_manifest(candidate)

    def test_unknown_mode_fails_closed(self) -> None:
        candidate = manifest()
        candidate["modes"] = list(candidate["modes"])
        candidate["modes"][0] = "fake_pfs_mode"
        with self.assertRaises(ValueError):
            validate_attribution_manifest(candidate)

    def test_required_domain_coverage_fails_closed(self) -> None:
        candidate = manifest()
        candidate["required_domains"] = ["persistent_endpoint"]
        with self.assertRaisesRegex(ValueError, "required_domains"):
            validate_attribution_manifest(candidate)

    def test_evidence_enum_contract_fails_closed(self) -> None:
        candidate = manifest()
        candidate["evidence_contract"] = dict(candidate["evidence_contract"])
        candidate["evidence_contract"]["counter_support_values"] = ["supported"]
        with self.assertRaisesRegex(ValueError, "counter support"):
            validate_attribution_manifest(candidate)

    def evidence(self, mode: str, domain: ResourceDomain, *, observed: bool = False) -> DomainEvidence:
        return DomainEvidence(
            domain=domain,
            mode=mode,
            foreground_kind="fsdp_all_gather",
            auxiliary_kind="checkpoint",
            overlapping_bytes=1024,
            overlap_ns=1000,
            tail_delta_ns=50,
            evidence=EvidenceLevel.OBSERVATIONAL,
            counter_support=CounterSupport.SUPPORTED if observed else CounterSupport.NOT_COLLECTED,
            path_status=PathStatus.OBSERVED if observed else PathStatus.DECLARED,
            uncertainty_ns=10,
            source="fixture",
            path_evidence=domain_contract(domain).path_evidence if observed else "",
            counter_family=domain_contract(domain).counter_family if observed else "",
        )

    def test_mode_evidence_requires_exact_declared_domain_coverage(self) -> None:
        records = [
            self.evidence("d2h_only", ResourceDomain.GPU_LOCAL),
            self.evidence("d2h_only", ResourceDomain.PCIE_HOST),
            self.evidence("d2h_only", ResourceDomain.HOST_NUMA),
        ]
        grouped = validate_mode_evidence("d2h_only", records)
        self.assertEqual(set(grouped), {
            ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA
        })

    def test_mode_evidence_rejects_missing_or_extra_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing evidence"):
            validate_mode_evidence("d2h_only", [self.evidence("d2h_only", ResourceDomain.GPU_LOCAL)])
        with self.assertRaisesRegex(ValueError, "not declared"):
            validate_mode_evidence("d2h_only", [
                self.evidence("d2h_only", ResourceDomain.GPU_LOCAL),
                self.evidence("d2h_only", ResourceDomain.PERSISTENT_ENDPOINT),
            ])

    def test_live_mode_evidence_requires_observed_supported_records(self) -> None:
        records = [
            self.evidence("persist_only", ResourceDomain.NIC_FABRIC),
            self.evidence("persist_only", ResourceDomain.SLINGSHOT_FABRIC),
            self.evidence("persist_only", ResourceDomain.PERSISTENT_ENDPOINT),
        ]
        with self.assertRaisesRegex(ValueError, "not observed"):
            validate_mode_evidence("persist_only", records, require_observed=True)
        live = [
            self.evidence("persist_only", ResourceDomain.NIC_FABRIC, observed=True),
            self.evidence("persist_only", ResourceDomain.SLINGSHOT_FABRIC, observed=True),
            self.evidence("persist_only", ResourceDomain.PERSISTENT_ENDPOINT, observed=True),
        ]
        self.assertEqual(
            set(validate_mode_evidence("persist_only", live, require_observed=True)),
            {ResourceDomain.NIC_FABRIC, ResourceDomain.SLINGSHOT_FABRIC, ResourceDomain.PERSISTENT_ENDPOINT},
        )

    def test_mode_evidence_rejects_mode_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_mode_evidence("d2h_only", [self.evidence("persist_only", ResourceDomain.NIC_FABRIC)])

    def test_joined_tier_gate_requires_path_evidence_before_metrics(self) -> None:
        metrics = [
            CausalModeRecord("fg_only", None, 100, 100, True, True, 3),
            CausalModeRecord("open_combined", None, 130, 130, True, True, 3),
            CausalModeRecord("d2h_only", ResourceDomain.PCIE_HOST, 110, 105, True, True, 3),
            CausalModeRecord("host_pressure", None, 135, 140, True, True, 3),
        ]
        declared = {
            "fg_only": [],
            "open_combined": [
                self.evidence("open_combined", ResourceDomain.GPU_LOCAL),
                self.evidence("open_combined", ResourceDomain.PCIE_HOST),
                self.evidence("open_combined", ResourceDomain.HOST_NUMA),
                self.evidence("open_combined", ResourceDomain.NIC_FABRIC),
                self.evidence("open_combined", ResourceDomain.SLINGSHOT_FABRIC),
                self.evidence("open_combined", ResourceDomain.PERSISTENT_ENDPOINT),
            ],
            "d2h_only": [
                self.evidence("d2h_only", ResourceDomain.GPU_LOCAL),
                self.evidence("d2h_only", ResourceDomain.PCIE_HOST),
                self.evidence("d2h_only", ResourceDomain.HOST_NUMA),
            ],
            "host_pressure": [
                self.evidence("host_pressure", ResourceDomain.HOST_NUMA),
            ],
        }
        rejected = evaluate_tier_attribution(declared, metrics)
        self.assertFalse(rejected.evidence_ready)
        self.assertFalse(rejected.promote_static_policy)
        live = {
            mode: [
                DomainEvidence(
                    domain=item.domain,
                    mode=item.mode,
                    foreground_kind=item.foreground_kind,
                    auxiliary_kind=item.auxiliary_kind,
                    overlapping_bytes=1,
                    overlap_ns=1,
                    tail_delta_ns=20,
                    evidence=EvidenceLevel.INTERVENTIONAL,
                    counter_support=CounterSupport.SUPPORTED,
                    path_status=PathStatus.OBSERVED,
                    uncertainty_ns=1,
                    source="live-fixture",
                    path_evidence=domain_contract(item.domain).path_evidence,
                    counter_family=domain_contract(item.domain).counter_family,
                )
                for item in values
            ]
            for mode, values in declared.items()
        }
        accepted = evaluate_tier_attribution(live, metrics)
        self.assertTrue(accepted.evidence_ready)
        self.assertTrue(accepted.promote_static_policy)

    def test_joined_tier_gate_rejects_missing_mode(self) -> None:
        result = evaluate_tier_attribution({}, [])
        self.assertFalse(result.evidence_ready)
        self.assertIn("missing fg_only", " ".join(result.reasons))

    def test_joined_tier_gate_rejects_metric_without_domain_record(self) -> None:
        result = evaluate_tier_attribution(
            {"fg_only": [], "open_combined": []},
            [
                CausalModeRecord("fg_only", None, 100, 100, True, True, 3),
                CausalModeRecord("open_combined", None, 130, 130, True, True, 3),
                CausalModeRecord("d2h_only", ResourceDomain.PCIE_HOST, 110, 105, True, True, 3),
            ],
            require_observed=False,
        )
        self.assertFalse(result.evidence_ready)
        self.assertIn("no evidence entry", " ".join(result.reasons))


if __name__ == "__main__":
    unittest.main()
