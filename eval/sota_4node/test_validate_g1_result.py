from __future__ import annotations

import copy
import unittest

from eval.sota_4node.validate_g1_result import validate_g1_result
from tempo.domain_evidence import CounterSupport, DomainEvidence, PathStatus
from tempo.resource_domain import (
    EvidenceLevel,
    ResourceDomain,
    allowed_counter_scopes,
    domain_contract,
)


FOREGROUND_DOMAINS = tuple(
    sorted(
        (
            ResourceDomain.GPU_LOCAL,
            ResourceDomain.NVLINK_P2P,
            ResourceDomain.PCIE_HOST,
            ResourceDomain.HOST_NUMA,
            ResourceDomain.NIC_FABRIC,
            ResourceDomain.SLINGSHOT_FABRIC,
        ),
        key=lambda item: item.value,
    )
)


def _evidence(mode: str, domain: ResourceDomain, *, intervention: bool) -> dict[str, object]:
    record = DomainEvidence(
        domain=domain,
        mode=mode,
        foreground_kind="fsdp_all_gather",
        auxiliary_kind="checkpoint_flow",
        overlapping_bytes=1_000,
        overlap_ns=1_000,
        tail_delta_ns=50,
        evidence=EvidenceLevel.INTERVENTIONAL if intervention else EvidenceLevel.OBSERVATIONAL,
        counter_support=CounterSupport.SUPPORTED,
        path_status=PathStatus.OBSERVED,
        uncertainty_ns=10,
        source="synthetic-live",
        path_evidence=domain_contract(domain).path_evidence,
        counter_family=domain_contract(domain).counter_family,
    )
    return {
        "observation_id": f"{mode}-obs",
        "domain": record.domain.value,
        "mode": record.mode,
        "foreground_kind": record.foreground_kind,
        "auxiliary_kind": record.auxiliary_kind,
        "overlapping_bytes": record.overlapping_bytes,
        "overlap_ns": record.overlap_ns,
        "tail_delta_ns": record.tail_delta_ns,
        "evidence": record.evidence.value,
        "counter_support": record.counter_support.value,
        "path_status": record.path_status.value,
        "uncertainty_ns": record.uncertainty_ns,
        "source": record.source,
        "path_evidence": domain_contract(domain).path_evidence,
        "counter_family": domain_contract(domain).counter_family,
        "scope": sorted(allowed_counter_scopes(domain))[0],
        "scope_id": f"{domain.value}-scope-0",
        "intervention_id": mode,
    }


def _counters(domain: ResourceDomain, mode: str) -> list[dict[str, object]]:
    scope = sorted(allowed_counter_scopes(domain))[0]
    scope_id = f"{domain.value}-scope-0"
    observation_id = f"{mode}-obs"
    return [
        {
            "observation_id": observation_id,
            "domain": domain.value,
            "sample_id": "before",
            "source": "synthetic-live",
            "timestamp_ns": 0,
            "cumulative_bytes": 0,
            "cumulative_busy_ns": 0,
            "support": "supported",
            "scope": scope,
            "scope_id": scope_id,
            "intervention_id": mode,
        },
        {
            "observation_id": observation_id,
            "domain": domain.value,
            "sample_id": "after",
            "source": "synthetic-live",
            "timestamp_ns": 1_000,
            "cumulative_bytes": 1_000,
            "cumulative_busy_ns": 100,
            "support": "supported",
            "scope": scope,
            "scope_id": scope_id,
            "intervention_id": mode,
        },
    ]


def _observation_windows(mode: str, domains: tuple[ResourceDomain, ...]) -> list[dict[str, object]]:
    observation_id = f"{mode}-obs"
    domain = domains[0] if domains else FOREGROUND_DOMAINS[0]
    base = {
        "observation_id": observation_id,
        "mode": mode,
        "rank": 0,
        "event_id": "event-16",
        "clock_domain": "corrected-monotonic-v1",
        "source_snapshot_id": "snapshot-a",
        "uncertainty_ns": 10,
    }
    rows = [
        {
            **base,
            "source": "foreground-collector",
            "start_ns": 100,
            "end_ns": 400,
            "role": "foreground",
            "domain": None,
        },
        {
            **base,
            "source": "counter-collector",
            "start_ns": 175,
            "end_ns": 325,
            "role": "counter",
            "domain": domain.value,
        },
    ]
    if domains:
        rows.insert(
            1,
            {
                **base,
                "source": "auxiliary-collector",
                "start_ns": 150,
                "end_ns": 350,
                "role": "auxiliary",
                "domain": None,
            },
        )
    return rows


def _mode(mode: str, tail: int, skew: int, domains: tuple[ResourceDomain, ...], *, intervention: bool, metric_domain: ResourceDomain | None = None) -> dict[str, object]:
    auxiliary = set(domains)
    shared = tuple(sorted((domain for domain in FOREGROUND_DOMAINS if domain in auxiliary), key=lambda item: item.value))
    return {
        "metrics": {
            "observation_id": f"{mode}-obs",
            "domain": None if metric_domain is None else metric_domain.value,
            "foreground_domains": [domain.value for domain in FOREGROUND_DOMAINS],
            "shared_domains": [domain.value for domain in shared],
            "tail_p99_ns": tail,
            "skew_p99_ns": skew,
            "deadline_met": True,
            "correctness_met": True,
            "samples": 3,
            "active_exposure_ns": 1_000,
            "active_groups": 4,
            "domain_exposure_ns": {
                domain.value: 1_000 for domain in sorted(domains, key=lambda item: item.value)
            },
        },
        "evidence": [_evidence(mode, domain, intervention=intervention) for domain in domains],
        "counters": {domain.value: _counters(domain, mode) for domain in domains},
        "observation_windows": _observation_windows(mode, domains),
    }


def _foreground_path() -> dict[str, object]:
    names = [domain.value for domain in FOREGROUND_DOMAINS]
    return {
        "domains": names,
        "path_status": {name: "observed" for name in names},
        "counter_support": {name: "supported" for name in names},
        "path_evidence": {
            name: domain_contract(ResourceDomain(name)).path_evidence
            for name in names
        },
        "counter_family": {
            name: domain_contract(ResourceDomain(name)).counter_family
            for name in names
        },
        "counters": {
            name: [
                {key: value for key, value in sample.items() if key != "observation_id"}
                for sample in _counters(ResourceDomain(name), "fg_only")
            ]
            for name in names
        },
    }


def valid_result() -> dict[str, object]:
    return {
        "schema_version": "tempo-rd-g1-result-5",
        "evidence_state": "live_observed",
        "world_size": 4,
        "nodes": 1,
        "source_bundle_sha256": "a" * 64,
        "host_pressure_raw_digest": "a" * 64,
        "state_bytes_per_rank": 384 * 1024 * 1024,
        "logical_file_extent_bytes": 385 * 1024 * 1024,
        "deadline_ns": 1_000_000_000,
        "checkpoint_steps": [16, 52],
        "foreground_path": _foreground_path(),
        "modes": {
            "fg_only": _mode("fg_only", 100, 100, (), intervention=False),
            "open_combined": _mode(
                "open_combined", 130, 130,
                (ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA,
                 ResourceDomain.NIC_FABRIC, ResourceDomain.SLINGSHOT_FABRIC, ResourceDomain.PERSISTENT_ENDPOINT),
                intervention=False,
            ),
            "d2h_only": _mode(
                "d2h_only", 110, 105,
                (ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA),
                intervention=True, metric_domain=ResourceDomain.PCIE_HOST,
            ),
            "persist_only": _mode(
                "persist_only", 115, 110,
                (ResourceDomain.NIC_FABRIC, ResourceDomain.SLINGSHOT_FABRIC, ResourceDomain.PERSISTENT_ENDPOINT),
                intervention=True, metric_domain=ResourceDomain.PERSISTENT_ENDPOINT,
            ),
            "combined": _mode(
                "combined", 108, 104,
                (ResourceDomain.GPU_LOCAL, ResourceDomain.PCIE_HOST, ResourceDomain.HOST_NUMA,
                 ResourceDomain.NIC_FABRIC, ResourceDomain.SLINGSHOT_FABRIC, ResourceDomain.PERSISTENT_ENDPOINT),
                intervention=False,
            ),
        },
        "placebo": _mode(
            "host_pressure", 135, 140,
            (ResourceDomain.HOST_NUMA,),
            intervention=True,
            metric_domain=None,
        ),
    }


class G1ResultValidatorTests(unittest.TestCase):
    def test_observed_result_recomputes_promotion(self) -> None:
        result = validate_g1_result(valid_result())
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["evidence_ready"])
        self.assertTrue(result["promote_static_policy"])
        self.assertIn("pcie_host", result["eligible_domains"])
        self.assertIn("persistent_endpoint", result["eligible_domains"])
        self.assertTrue(result["live_external_execution"])

    def test_foreground_footprint_must_match_across_modes(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["metrics"]["foreground_domains"] = ["host_numa"]
        candidate["modes"]["d2h_only"]["metrics"]["shared_domains"] = ["host_numa"]
        with self.assertRaisesRegex(ValueError, "foreground_domains do not match"):
            validate_g1_result(candidate)

    def test_design_only_and_self_attested_fields_are_rejected(self) -> None:
        candidate = valid_result()
        candidate["evidence_state"] = "design_only"
        with self.assertRaisesRegex(ValueError, "live_observed"):
            validate_g1_result(candidate)

    def test_missing_foreground_hardware_path_is_rejected(self) -> None:
        candidate = valid_result()
        del candidate["foreground_path"]
        with self.assertRaisesRegex(ValueError, "keys are not exact"):
            validate_g1_result(candidate)

    def test_host_wide_counter_scope_is_rejected(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["evidence"][0]["scope"] = "host"
        with self.assertRaisesRegex(ValueError, "scope/intervention binding"):
            validate_g1_result(candidate)

    def test_host_wide_counter_snapshot_is_rejected(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["counters"]["gpu_local"][0]["scope"] = "host"
        with self.assertRaisesRegex(ValueError, "scope/intervention binding"):
            validate_g1_result(candidate)

    def test_counter_scope_id_must_match_evidence(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["counters"]["gpu_local"][0]["scope_id"] = "rank-99"
        with self.assertRaisesRegex(ValueError, "scope/intervention binding"):
            validate_g1_result(candidate)

    def test_observation_id_must_bind_metrics_evidence_and_counters(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["evidence"][0]["observation_id"] = "other"
        with self.assertRaisesRegex(ValueError, "observation_id"):
            validate_g1_result(candidate)

        candidate = valid_result()
        candidate["modes"]["d2h_only"]["observation_windows"][1]["clock_domain"] = "raw-monotonic"
        with self.assertRaisesRegex(ValueError, "identity/clock provenance"):
            validate_g1_result(candidate)

        candidate = valid_result()
        candidate["modes"]["d2h_only"]["observation_windows"][1]["start_ns"] = 401
        candidate["modes"]["d2h_only"]["observation_windows"][1]["end_ns"] = 450
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            validate_g1_result(candidate)

    def test_missing_host_pressure_placebo_is_rejected(self) -> None:
        candidate = valid_result()
        del candidate["placebo"]
        with self.assertRaisesRegex(ValueError, "keys are not exact"):
            validate_g1_result(candidate)
        candidate = valid_result()
        candidate["promotion"] = {"promote": True}
        with self.assertRaisesRegex(ValueError, "keys are not exact"):
            validate_g1_result(candidate)

    def test_missing_path_and_counter_regression_are_rejected(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["evidence"] = candidate["modes"]["d2h_only"]["evidence"][1:]
        with self.assertRaisesRegex(ValueError, "traversed path"):
            validate_g1_result(candidate)

        candidate = valid_result()
        candidate["modes"]["d2h_only"]["counters"]["pcie_host"][0]["source"] = "other-source"
        with self.assertRaisesRegex(ValueError, "counter source is not bound"):
            validate_g1_result(candidate)

        candidate = valid_result()
        candidate["modes"]["d2h_only"]["evidence"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered traversed path"):
            validate_g1_result(candidate)

        candidate = valid_result()
        series = candidate["modes"]["persist_only"]["counters"]["nic_fabric"]
        series[1]["cumulative_bytes"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative int"):
            validate_g1_result(candidate)

    def test_open_without_headroom_is_valid_but_not_promoted(self) -> None:
        candidate = valid_result()
        candidate["modes"]["open_combined"]["metrics"]["tail_p99_ns"] = 101
        candidate["modes"]["open_combined"]["metrics"]["skew_p99_ns"] = 101
        result = validate_g1_result(candidate)
        self.assertTrue(result["evidence_ready"])
        self.assertFalse(result["promote_static_policy"])

    def test_open_path_and_counters_cannot_be_declared_only(self) -> None:
        candidate = valid_result()
        candidate["modes"]["open_combined"]["evidence"][0]["path_status"] = "declared"
        with self.assertRaisesRegex(ValueError, "path/counters must be observed and supported"):
            validate_g1_result(candidate)
        candidate = valid_result()
        candidate["modes"]["combined"]["evidence"][0]["counter_support"] = "not_collected"
        with self.assertRaisesRegex(ValueError, "path/counters must be observed and supported"):
            validate_g1_result(candidate)

    def test_intervention_must_exceed_domain_uncertainty(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["evidence"][1]["tail_delta_ns"] = 10
        with self.assertRaisesRegex(ValueError, "tail delta does not exceed uncertainty"):
            validate_g1_result(candidate)

    def test_domain_path_and_counter_labels_are_contract_bound(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["evidence"][0]["path_evidence"] = "topology_guess"
        with self.assertRaisesRegex(ValueError, "path .*contract"):
            validate_g1_result(candidate)

        candidate = valid_result()
        candidate["modes"]["persist_only"]["evidence"][0]["counter_family"] = "aggregate_bytes"
        with self.assertRaisesRegex(ValueError, "counter .*contract"):
            validate_g1_result(candidate)

    def test_intervention_domain_is_explicit_not_inferred_from_stage_name(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["metrics"]["domain"] = "host_numa"
        result = validate_g1_result(candidate)
        self.assertIn("host_numa", result["eligible_domains"])
        self.assertNotIn("pcie_host", result["eligible_domains"])

        candidate = valid_result()
        candidate["modes"]["d2h_only"]["metrics"]["domain"] = "nvlink_p2p"
        with self.assertRaisesRegex(ValueError, "not on the isolated route"):
            validate_g1_result(candidate)

    def test_combined_consistency_cannot_rescue_failed_isolated_mode(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["metrics"]["tail_p99_ns"] = 140
        candidate["modes"]["d2h_only"]["metrics"]["skew_p99_ns"] = 140
        candidate["modes"]["persist_only"]["metrics"]["tail_p99_ns"] = 140
        candidate["modes"]["persist_only"]["metrics"]["skew_p99_ns"] = 140
        result = validate_g1_result(candidate)
        self.assertFalse(result["promote_static_policy"])

    def test_exposure_or_active_group_increase_blocks_promotion(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["metrics"]["active_exposure_ns"] = 1_001
        result = validate_g1_result(candidate)
        self.assertFalse(result["promote_static_policy"])
        self.assertEqual(result["eligible_domains"], [])
        self.assertTrue(any("active exposure" in reason for reason in result["reasons"]))

        candidate = valid_result()
        candidate["modes"]["persist_only"]["metrics"]["active_groups"] = 5
        result = validate_g1_result(candidate)
        self.assertFalse(result["promote_static_policy"])
        self.assertEqual(result["eligible_domains"], [])
        self.assertTrue(any("active group count" in reason for reason in result["reasons"]))

    def test_domain_exposure_map_is_bound_to_evidence(self) -> None:
        candidate = valid_result()
        candidate["modes"]["d2h_only"]["metrics"]["domain_exposure_ns"]["pcie_host"] = 1_001
        with self.assertRaisesRegex(ValueError, "not derived from evidence"):
            validate_g1_result(candidate)

    def test_domain_exposure_shift_blocks_even_with_scalar_active_exposure_unchanged(self) -> None:
        candidate = valid_result()
        evidence = candidate["modes"]["d2h_only"]["evidence"]
        for item in evidence:
            if item["domain"] == "pcie_host":
                item["overlap_ns"] = 1_500
        candidate["modes"]["d2h_only"]["metrics"]["domain_exposure_ns"]["pcie_host"] = 1_500
        # The scalar active interval and group count remain unchanged; only
        # the common-domain exposure map reveals the bottleneck shift.
        result = validate_g1_result(candidate)
        self.assertNotIn("pcie_host", result["eligible_domains"])
        self.assertTrue(any("domain exposure" in reason for reason in result["reasons"]))

    def test_failed_combined_mode_cannot_be_promoted_by_isolated_modes(self) -> None:
        candidate = valid_result()
        candidate["modes"]["combined"]["metrics"]["deadline_met"] = False
        with self.assertRaisesRegex(ValueError, "combined: live tier mode must satisfy"):
            validate_g1_result(candidate)

    def test_shared_domains_are_an_explicit_route_intersection(self) -> None:
        candidate = valid_result()
        metrics = candidate["modes"]["d2h_only"]["metrics"]
        metrics["shared_domains"] = ["host_numa"]
        with self.assertRaisesRegex(ValueError, "route intersection"):
            validate_g1_result(candidate)

        candidate = valid_result()
        metrics = candidate["modes"]["open_combined"]["metrics"]
        metrics["foreground_domains"] = ["gpu_local", "gpu_local"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_g1_result(candidate)

    def test_foreground_domain_order_and_unknown_domain_are_rejected(self) -> None:
        candidate = valid_result()
        metrics = candidate["modes"]["persist_only"]["metrics"]
        metrics["foreground_domains"] = ["pcie_host", "gpu_local"]
        with self.assertRaisesRegex(ValueError, "sorted"):
            validate_g1_result(candidate)

        candidate = valid_result()
        metrics = candidate["modes"]["persist_only"]["metrics"]
        metrics["foreground_domains"] = ["made_up_domain"]
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_g1_result(candidate)


if __name__ == "__main__":
    unittest.main()
