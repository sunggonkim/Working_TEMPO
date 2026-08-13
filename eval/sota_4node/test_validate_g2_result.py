from __future__ import annotations

import copy
import unittest

from eval.sota_4node.test_validate_g1_result import valid_result as valid_g1_result
from eval.sota_4node.validate_g2_result import validate_g2_result
from tempo.resource_domain import ResourceDomain, domain_contract


FULL_PATH = (
    ResourceDomain.GPU_LOCAL,
    ResourceDomain.PCIE_HOST,
    ResourceDomain.HOST_NUMA,
    ResourceDomain.NIC_FABRIC,
    ResourceDomain.SLINGSHOT_FABRIC,
    ResourceDomain.PERSISTENT_ENDPOINT,
)
SLICES = ("intra_node", "inter_node")
ORIGINS = ("gdr_gpu_originated", "host_originated", "pfs_endpoint")
FOREGROUND_PATH = tuple(
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


def _fabric(mode: str, domain: ResourceDomain, intervention: bool) -> list[dict[str, object]]:
    values = []
    for index in range(6):
        values.append({
            "observation_id": f"{mode}-obs",
            "mode": mode,
            "collective_slice": SLICES[index % 2],
            "traffic_origin": ORIGINS[index % 3],
            "domain": domain.value,
            "scope": (
                "slice" if domain is ResourceDomain.SLINGSHOT_FABRIC
                else "endpoint" if domain is ResourceDomain.PERSISTENT_ENDPOINT
                else "rank"
            ),
            "scope_id": f"scope-{index % 2}",
            "intervention_id": mode,
            "overlapping_bytes": 10_000,
            "overlap_ns": 10_000,
            "tail_delta_ns": 100,
            "evidence": "interventional" if intervention else "observational",
            "counter_support": "supported",
            "path_status": "observed",
            "uncertainty_ns": 10,
            "counter_samples": 3,
            "counter_series": [
                {
                    "observation_id": f"{mode}-obs",
                    "domain": domain.value,
                    "sample_id": f"{mode}-{domain.value}-{index}-{offset}",
                    "source": "synthetic-g2-fabric",
                    "timestamp_ns": 1_000 + offset * 1_000,
                    "cumulative_bytes": offset * 10_000,
                    "cumulative_busy_ns": offset * 100,
                    "support": "supported",
                }
                for offset in range(3)
            ],
            "source": "synthetic-g2-fabric",
            "path_evidence": domain_contract(domain).path_evidence,
            "counter_family": domain_contract(domain).counter_family,
        })
    return values


def _mode(mode: str, tail: int, skew: int, domains: set[ResourceDomain], intervention: bool) -> dict[str, object]:
    evidence = []
    for domain in domains:
        evidence.extend(_fabric(mode, domain, intervention))
    shared = tuple(sorted((item for item in domains if item in FOREGROUND_PATH), key=lambda item: item.value))
    window_domain = next(iter(sorted(domains, key=lambda item: item.value)), ResourceDomain.GPU_LOCAL)
    base_window = {
        "observation_id": f"{mode}-obs",
        "mode": mode,
        "rank": 0,
        "event_id": "event-16",
        "clock_domain": "corrected-monotonic-v1",
        "source_snapshot_id": "snapshot-a",
        "uncertainty_ns": 10,
    }
    observation_windows = [
        {
            **base_window,
            "source": "foreground-collector",
            "start_ns": 100,
            "end_ns": 400,
            "role": "foreground",
            "domain": None,
        },
        {
            **base_window,
            "source": "counter-collector",
            "start_ns": 175,
            "end_ns": 325,
            "role": "counter",
            "domain": window_domain.value,
        },
    ]
    if domains:
        observation_windows.insert(1, {
            **base_window,
            "source": "auxiliary-collector",
            "start_ns": 150,
            "end_ns": 350,
            "role": "auxiliary",
            "domain": None,
        })
    return {
        "metrics": {
            "observation_id": f"{mode}-obs",
            "foreground_domains": [item.value for item in FOREGROUND_PATH],
            "shared_domains": [item.value for item in shared],
            "tail_p99_ns": tail,
            "skew_p99_ns": skew,
            "deadline_met": True,
            "correctness_met": True,
            "samples": 3,
            "active_exposure_ns": 2_000,
            "active_groups": 8,
            "domain_exposure_ns": {
                domain.value: 10_000 for domain in sorted(domains, key=lambda item: item.value)
            },
        },
        "slice_metrics": {
            "intra_node": {"tail_p99_ns": tail, "skew_p99_ns": skew, "samples": 3},
            "inter_node": {"tail_p99_ns": tail + 1, "skew_p99_ns": skew + 1, "samples": 3},
        },
        "fabric_evidence": evidence,
        "observation_windows": observation_windows,
    }


def valid_result() -> dict[str, object]:
    promoted = ResourceDomain.PCIE_HOST
    placebo = ResourceDomain.NIC_FABRIC
    return {
        "schema_version": "tempo-rd-g2-result-5",
        "evidence_state": "live_observed",
        "world_size": 8,
        "nodes": 2,
        "source_bundle_sha256": "b" * 64,
        "g1_result": valid_g1_result(),
        "promoted_domain": promoted.value,
        "placebo_domain": placebo.value,
        "state_bytes_per_rank": 384 * 1024 * 1024,
        "deadline_ns": 1_000_000_000,
        "checkpoint_steps": [16, 52],
        "collective_slices": list(SLICES),
        "fabric_splits": list(ORIGINS),
        "modes": {
            "fg_only": _mode("fg_only", 100, 100, set(), False),
            "open_combined": _mode("open_combined", 130, 130, set(FULL_PATH), False),
            "causal_domain_static_cap": _mode("causal_domain_static_cap", 110, 105, {promoted}, True),
            "unrelated_domain_placebo": _mode("unrelated_domain_placebo", 135, 135, {placebo}, True),
            "combined": _mode("combined", 108, 104, set(FULL_PATH), False),
        },
    }


class G2ResultValidatorTests(unittest.TestCase):
    def test_observed_fabric_result_requires_g1_and_promotes(self) -> None:
        result = validate_g2_result(valid_result())
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["promote_static_policy"])
        self.assertIn("pcie_host", result["eligible_domains"])
        self.assertTrue(result["live_external_execution"])

    def test_design_only_or_missing_fabric_coverage_is_rejected(self) -> None:
        candidate = valid_result()
        candidate["evidence_state"] = "design_only"
        with self.assertRaisesRegex(ValueError, "live_observed"):
            validate_g2_result(candidate)
        candidate = valid_result()
        candidate["modes"]["open_combined"]["fabric_evidence"] = [
            item for item in candidate["modes"]["open_combined"]["fabric_evidence"]
            if item["traffic_origin"] != "pfs_endpoint"
        ]
        with self.assertRaisesRegex(ValueError, "slice/origin Cartesian coverage"):
            validate_g2_result(candidate)

        candidate = valid_result()
        evidence = candidate["modes"]["open_combined"]["fabric_evidence"]
        evidence.append(copy.deepcopy(evidence[0]))
        with self.assertRaisesRegex(ValueError, "slice/origin Cartesian coverage"):
            validate_g2_result(candidate)

        candidate = valid_result()
        evidence = candidate["modes"]["open_combined"]["fabric_evidence"]
        # Keep six records and both marginal sets present, but duplicate one
        # pair while dropping another.  Marginal-only validation would accept
        # this and would not prove the required intra/inter × origin matrix.
        evidence[-1]["collective_slice"] = evidence[-2]["collective_slice"]
        evidence[-1]["traffic_origin"] = evidence[-2]["traffic_origin"]
        with self.assertRaisesRegex(ValueError, "slice/origin Cartesian coverage"):
            validate_g2_result(candidate)

    def test_unrelated_placebo_improvement_is_rejected(self) -> None:
        candidate = valid_result()
        candidate["modes"]["unrelated_domain_placebo"]["metrics"]["tail_p99_ns"] = 100
        candidate["modes"]["unrelated_domain_placebo"]["metrics"]["skew_p99_ns"] = 100
        with self.assertRaisesRegex(ValueError, "placebo"):
            validate_g2_result(candidate)

    def test_g2_cannot_bypass_g1_promotion(self) -> None:
        candidate = valid_result()
        candidate["g1_result"]["modes"]["open_combined"]["metrics"]["tail_p99_ns"] = 101
        candidate["g1_result"]["modes"]["open_combined"]["metrics"]["skew_p99_ns"] = 101
        with self.assertRaisesRegex(ValueError, "successful G1 promotion"):
            validate_g2_result(candidate)

    def test_g2_geometry_and_deadline_are_bound_to_g1(self) -> None:
        for key, value in (
            ("state_bytes_per_rank", 385 * 1024 * 1024),
            ("deadline_ns", 1_000_000_001),
            ("checkpoint_steps", [16, 76]),
        ):
            candidate = valid_result()
            candidate[key] = value
            with self.assertRaisesRegex(ValueError, f"G2 {key} does not match"):
                validate_g2_result(candidate)

    def test_promoted_fabric_cap_requires_interventional_evidence(self) -> None:
        candidate = valid_result()
        candidate["modes"]["causal_domain_static_cap"]["fabric_evidence"][0]["evidence"] = "observational"
        with self.assertRaisesRegex(ValueError, "must be interventional"):
            validate_g2_result(candidate)

    def test_fabric_path_and_counter_labels_are_contract_bound(self) -> None:
        candidate = valid_result()
        candidate["modes"]["open_combined"]["fabric_evidence"][0]["path_evidence"] = "hca_guess"
        with self.assertRaisesRegex(ValueError, "path .*contract"):
            validate_g2_result(candidate)

        candidate = valid_result()
        candidate["modes"]["causal_domain_static_cap"]["fabric_evidence"][0]["counter_family"] = "fabric_bytes"
        with self.assertRaisesRegex(ValueError, "counter .*contract"):
            validate_g2_result(candidate)

    def test_fabric_counter_count_cannot_replace_the_counter_series(self) -> None:
        candidate = valid_result()
        candidate["modes"]["open_combined"]["fabric_evidence"][0]["counter_series"] = []
        with self.assertRaisesRegex(ValueError, "counter series/count mismatch"):
            validate_g2_result(candidate)

    def test_host_wide_fabric_counter_cannot_be_promoted(self) -> None:
        candidate = valid_result()
        candidate["modes"]["open_combined"]["fabric_evidence"][0]["scope"] = "host"
        with self.assertRaisesRegex(ValueError, "scope"):
            validate_g2_result(candidate)

    def test_fabric_counter_source_is_bound_to_evidence_source(self) -> None:
        candidate = valid_result()
        candidate["modes"]["open_combined"]["fabric_evidence"][0]["counter_series"][0]["source"] = "other-source"
        with self.assertRaisesRegex(ValueError, "counter source is not bound"):
            validate_g2_result(candidate)

    def test_fabric_observation_id_is_bound_to_metrics(self) -> None:
        candidate = valid_result()
        candidate["modes"]["open_combined"]["fabric_evidence"][0]["observation_id"] = "other"
        with self.assertRaisesRegex(ValueError, "observation_id"):
            validate_g2_result(candidate)
        candidate = valid_result()
        candidate["modes"]["open_combined"]["observation_windows"][1]["clock_domain"] = "raw-monotonic"
        with self.assertRaisesRegex(ValueError, "identity/clock provenance"):
            validate_g2_result(candidate)

    def test_combined_consistency_cannot_rescue_a_failed_static_cap(self) -> None:
        candidate = valid_result()
        cap = candidate["modes"]["causal_domain_static_cap"]["metrics"]
        cap["tail_p99_ns"] = 140
        cap["skew_p99_ns"] = 140
        # The combined replicate remains artificially excellent; it must not
        # turn a failed isolated intervention into a promoted domain.
        with self.assertRaisesRegex(ValueError, "promoted domain"):
            validate_g2_result(candidate)

    def test_active_exposure_or_group_increase_is_not_a_promotion(self) -> None:
        candidate = valid_result()
        candidate["modes"]["causal_domain_static_cap"]["metrics"]["active_exposure_ns"] = 2_001
        with self.assertRaisesRegex(ValueError, "active exposure"):
            validate_g2_result(candidate)

        candidate = valid_result()
        candidate["modes"]["combined"]["metrics"]["active_groups"] = 9
        with self.assertRaisesRegex(ValueError, "active group count"):
            validate_g2_result(candidate)

    def test_foreground_footprint_must_match_across_modes(self) -> None:
        candidate = valid_result()
        candidate["modes"]["causal_domain_static_cap"]["metrics"]["foreground_domains"] = ["gpu_local"]
        candidate["modes"]["causal_domain_static_cap"]["metrics"]["shared_domains"] = []
        with self.assertRaisesRegex(ValueError, "foreground_domains do not match"):
            validate_g2_result(candidate)

    def test_domain_exposure_map_is_bound_to_fabric_evidence(self) -> None:
        candidate = valid_result()
        candidate["modes"]["causal_domain_static_cap"]["metrics"]["domain_exposure_ns"]["pcie_host"] = 2_001
        with self.assertRaisesRegex(ValueError, "not derived from fabric evidence"):
            validate_g2_result(candidate)

    def test_domain_exposure_shift_blocks_with_unchanged_scalar_interval(self) -> None:
        candidate = valid_result()
        for item in candidate["modes"]["causal_domain_static_cap"]["fabric_evidence"]:
            if item["domain"] == "pcie_host":
                item["overlap_ns"] = 15_000
        candidate["modes"]["causal_domain_static_cap"]["metrics"]["domain_exposure_ns"]["pcie_host"] = 15_000
        with self.assertRaisesRegex(ValueError, "promoted domain"):
            validate_g2_result(candidate)

    def test_failed_combined_or_placebo_cannot_complete_fabric_matrix(self) -> None:
        candidate = valid_result()
        candidate["modes"]["combined"]["metrics"]["correctness_met"] = False
        with self.assertRaisesRegex(ValueError, "combined: live fabric mode must satisfy"):
            validate_g2_result(candidate)

        candidate = valid_result()
        candidate["modes"]["unrelated_domain_placebo"]["metrics"]["deadline_met"] = False
        with self.assertRaisesRegex(ValueError, "unrelated_domain_placebo: live fabric mode must satisfy"):
            validate_g2_result(candidate)

    def test_fabric_result_requires_explicit_shared_domain_intersection(self) -> None:
        candidate = valid_result()
        candidate["modes"]["causal_domain_static_cap"]["metrics"]["shared_domains"] = []
        with self.assertRaisesRegex(ValueError, "route intersection"):
            validate_g2_result(candidate)

        candidate = valid_result()
        candidate["modes"]["open_combined"]["metrics"]["foreground_domains"] = [
            "gpu_local", "gpu_local"
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_g2_result(candidate)


if __name__ == "__main__":
    unittest.main()
