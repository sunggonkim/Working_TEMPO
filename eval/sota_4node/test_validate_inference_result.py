from __future__ import annotations

import unittest

from eval.sota_4node.inference_kv_runner import build_kv_matrix
from eval.sota_4node.validate_inference_result import validate_inference_result
from tempo.resource_domain import ResourceDomain, allowed_counter_scopes, domain_contract


def _route(mode: str, domain: str, observational: bool) -> dict[str, object]:
    return {
        "observation_id": f"{mode}-obs",
        "mode": mode,
        "domain": domain,
        "scope": (
            "slice" if ResourceDomain(domain) is ResourceDomain.SLINGSHOT_FABRIC
            else "endpoint" if ResourceDomain(domain) is ResourceDomain.PERSISTENT_ENDPOINT
            else "rank"
        ),
        "scope_id": "scope-0",
        "intervention_id": mode,
        "overlapping_bytes": 64 * 1024 * 1024,
        "overlap_ns": 100_000,
        "tail_delta_ns": 1_000,
        "evidence": "observational" if observational else "interventional",
        "counter_support": "supported",
        "path_status": "observed",
        "uncertainty_ns": 100,
        "counter_samples": 3,
        "counter_series": [
            {
                "observation_id": f"{mode}-obs",
                "domain": domain,
                "sample_id": f"{mode}-{domain}-{offset}",
                    "source": "synthetic-inference-counters",
                "timestamp_ns": 1_000 + offset * 1_000,
                "cumulative_bytes": offset * (64 * 1024 * 1024),
                "cumulative_busy_ns": offset * 100,
                "support": "supported",
            }
            for offset in range(3)
        ],
        "source": "synthetic-inference-counters",
        "path_evidence": domain_contract(ResourceDomain(domain)).path_evidence,
        "counter_family": domain_contract(ResourceDomain(domain)).counter_family,
    }


def _foreground_path() -> dict[str, object]:
    domain = ResourceDomain.GPU_LOCAL
    name = domain.value
    scope = sorted(allowed_counter_scopes(domain))[0]
    return {
        "domains": [name],
        "path_status": {name: "observed"},
        "counter_support": {name: "supported"},
        "path_evidence": {name: domain_contract(domain).path_evidence},
        "counter_family": {name: domain_contract(domain).counter_family},
        "counters": {
            name: [
                {
                    "domain": name,
                    "sample_id": "fg-before",
                    "source": "synthetic-foreground-counters",
                    "timestamp_ns": 1_000,
                    "cumulative_bytes": 0,
                    "cumulative_busy_ns": 0,
                    "support": "supported",
                    "scope": scope,
                    "scope_id": "gpu-0",
                    "intervention_id": "fg_only",
                },
                {
                    "domain": name,
                    "sample_id": "fg-after",
                    "source": "synthetic-foreground-counters",
                    "timestamp_ns": 2_000,
                    "cumulative_bytes": 4_096,
                    "cumulative_busy_ns": 100,
                    "support": "supported",
                    "scope": scope,
                    "scope_id": "gpu-0",
                    "intervention_id": "fg_only",
                },
            ]
        },
    }


def _observation_windows(mode: str, route: list[str]) -> list[dict[str, object]]:
    domain = route[0] if route else ResourceDomain.GPU_LOCAL.value
    base = {
        "observation_id": f"{mode}-obs",
        "mode": mode,
        "rank": 0,
        "event_id": "request-0",
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
            "domain": domain,
        },
    ]
    if route:
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


def valid_result() -> dict[str, object]:
    modes: dict[str, dict[str, object]] = {}
    foreground = [ResourceDomain.GPU_LOCAL.value]
    for run in build_kv_matrix():
        route_domains = [ResourceDomain(item) for item in run.route]
        shared = sorted(set(route_domains).intersection({ResourceDomain.GPU_LOCAL}), key=lambda item: item.value)
        metrics = {
            "observation_id": f"{run.mode}-obs",
            "domain": None if run.mode in {"fg_only", "open_combined", "combined"} else {
                "d2h_only": "pcie_host",
                "remote_fabric": "slingshot_fabric",
                "persistent_tier": "persistent_endpoint",
            }[run.mode],
            "foreground_domains": list(foreground),
            "shared_domains": [item.value for item in shared],
            "ttft_p99_ns": 100 if run.mode == "fg_only" else (90 if run.mode != "open_combined" else 110),
            "itl_p99_ns": 100 if run.mode == "fg_only" else (90 if run.mode != "open_combined" else 110),
            "slo_goodput_milli": 1_000_000,
            "deadline_met": True,
            "correctness_met": True,
            "samples": 3,
            "max_domain_exposure_ns": 100_000 if run.mode != "fg_only" else 0,
            "domain_exposure_ns": {
                domain: 100_000 for domain in sorted(run.route)
            },
        }
        correctness = {
            "native_version_identity": True,
            "output_token_equivalence": True,
            "stale_version_rejection": True,
            "prefetch_before_use": True,
            "exact_completion_bytes": True,
        }
        expected_bytes = 0 if run.mode == "fg_only" else 64 * 1024 * 1024 * 64
        integrity = {
            "published_versions": 64,
            "stale_rejections": 1,
            "output_token_mismatches": 0,
            "prefetch_before_use_violations": 0,
            "admitted_bytes": expected_bytes,
            "completed_bytes": expected_bytes,
        }
        route = []
        for domain in run.route:
            route.append(_route(run.mode, domain, run.mode in {"open_combined", "combined"}))
        modes[run.mode] = {
            "metrics": metrics,
            "correctness": correctness,
            "integrity": integrity,
            "route_evidence": route,
            "observation_windows": _observation_windows(run.mode, list(run.route)),
        }
    return {
        "schema_version": "tempo-rd-inference-result-5",
        "evidence_state": "live_observed",
        "world_size": 1,
        "nodes": 1,
        "source_bundle_sha256": "a" * 64,
        "backend": {
            "name": "synthetic-kv-backend",
            "version": "1.0",
            "executable_sha256": "b" * 64,
        },
        "endpoint": "persistent_endpoint",
        "kv_bytes_per_request": 64 * 1024 * 1024,
        "deadline_ns": 250_000_000,
        "offered_load_requests": 64,
        "operation": "prefetch",
        "admission_contract": {
            "controller": "DomainAdmissionController",
            "flow_adapter": "KVFlowLedger.admit_via_domain_controller",
            "shared_domain_intersection": "explicit_foreground_route_intersection",
            "completion_owner": "KVFlowLedger.complete",
        },
        "foreground_path": _foreground_path(),
        "modes": modes,
    }


class InferenceResultValidatorTests(unittest.TestCase):
    def test_observed_result_passes_and_promotes_intervention(self) -> None:
        result = validate_inference_result(valid_result())
        self.assertEqual(result["status"], "pass")
        self.assertTrue(result["promote_static_policy"])
        self.assertIn("pcie_host", result["eligible_domains"])
        self.assertTrue(result["live_external_execution"])

    def test_foreground_footprint_must_match_across_modes(self) -> None:
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["metrics"]["foreground_domains"] = ["host_numa"]
        candidate["modes"]["remote_fabric"]["metrics"]["shared_domains"] = ["host_numa"]
        with self.assertRaisesRegex(ValueError, "foreground_domains do not match"):
            validate_inference_result(candidate)

    def test_design_only_and_wrong_world_are_rejected(self) -> None:
        candidate = valid_result()
        candidate["evidence_state"] = "design_only"
        with self.assertRaisesRegex(ValueError, "live_observed"):
            validate_inference_result(candidate)
        candidate = valid_result()
        candidate["world_size"] = 2
        with self.assertRaisesRegex(ValueError, "one GPU"):
            validate_inference_result(candidate)

    def test_missing_foreground_hardware_path_is_rejected(self) -> None:
        candidate = valid_result()
        del candidate["foreground_path"]
        with self.assertRaisesRegex(ValueError, "keys are not exact"):
            validate_inference_result(candidate)

    def test_top_level_endpoint_is_bound_to_common_auxiliary_route(self) -> None:
        candidate = valid_result()
        candidate["endpoint"] = "node_local_host"
        with self.assertRaisesRegex(ValueError, "common auxiliary matched-open endpoint"):
            validate_inference_result(candidate)

    def test_scalar_kv_ledger_cannot_claim_shared_controller_orchestration(self) -> None:
        candidate = valid_result()
        candidate["admission_contract"]["controller"] = "KVFlowLedger.admit"
        with self.assertRaisesRegex(ValueError, "shared domain controller contract"):
            validate_inference_result(candidate)
        candidate = valid_result()
        del candidate["admission_contract"]
        with self.assertRaisesRegex(ValueError, "result keys are not exact"):
            validate_inference_result(candidate)

    def test_route_and_integrity_are_bound_to_frozen_matrix(self) -> None:
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["route_evidence"] = []
        with self.assertRaisesRegex(ValueError, "route domains"):
            validate_inference_result(candidate)
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["route_evidence"].reverse()
        with self.assertRaisesRegex(ValueError, "ordered route"):
            validate_inference_result(candidate)
        candidate = valid_result()
        candidate["modes"]["persistent_tier"]["integrity"]["completed_bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "bytes are not exact"):
            validate_inference_result(candidate)

    def test_correctness_and_strict_types_cannot_be_bypassed(self) -> None:
        candidate = valid_result()
        candidate["modes"]["combined"]["correctness"]["prefetch_before_use"] = False
        with self.assertRaisesRegex(ValueError, "correctness contract failed"):
            validate_inference_result(candidate)
        candidate = valid_result()
        candidate["modes"]["combined"]["correctness"]["native_version_identity"] = 1
        with self.assertRaisesRegex(ValueError, "strict bools"):
            validate_inference_result(candidate)

    def test_intervention_must_exceed_route_uncertainty(self) -> None:
        candidate = valid_result()
        for item in candidate["modes"]["d2h_only"]["route_evidence"]:
            if item["domain"] == "pcie_host":
                item["tail_delta_ns"] = item["uncertainty_ns"]
        with self.assertRaisesRegex(ValueError, "tail delta does not exceed uncertainty"):
            validate_inference_result(candidate)

    def test_intervention_domain_is_explicit_not_inferred_from_route_name(self) -> None:
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["metrics"]["domain"] = "nic_fabric"
        result = validate_inference_result(candidate)
        # All non-foreground modes now use the same persistent route; naming
        # the NIC intervention explicitly is enough to select that domain,
        # while route counters still pass the exact matched-open gate.
        self.assertIn("nic_fabric", result["eligible_domains"])
        self.assertNotIn("slingshot_fabric", result["eligible_domains"])

        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["metrics"]["domain"] = "persistent_endpoint"
        with self.assertRaisesRegex(ValueError, "not on the route"):
            validate_inference_result(candidate)

    def test_route_and_integrity_cannot_self_attest_missing_work(self) -> None:
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["route_evidence"][0]["overlapping_bytes"] -= 1
        with self.assertRaisesRegex(ValueError, "route bytes"):
            validate_inference_result(candidate)

    def test_route_counter_source_is_bound_to_evidence_source(self) -> None:
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["route_evidence"][0]["counter_series"][0]["source"] = "other-source"
        with self.assertRaisesRegex(ValueError, "route counter source is not bound"):
            validate_inference_result(candidate)

    def test_route_counter_identifiers_are_strict_strings(self) -> None:
        for field in ("sample_id", "source"):
            candidate = valid_result()
            candidate["modes"]["remote_fabric"]["route_evidence"][0]["counter_series"][0][field] = 1
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, rf"counter {field}"):
                    validate_inference_result(candidate)

    def test_route_observation_id_is_bound_to_metrics(self) -> None:
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["route_evidence"][0]["observation_id"] = "other"
        with self.assertRaisesRegex(ValueError, "observation_id"):
            validate_inference_result(candidate)
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["observation_windows"][1]["clock_domain"] = "raw-monotonic"
        with self.assertRaisesRegex(ValueError, "identity/clock provenance"):
            validate_inference_result(candidate)
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["observation_windows"][1]["start_ns"] = 401
        candidate["modes"]["remote_fabric"]["observation_windows"][1]["end_ns"] = 450
        with self.assertRaisesRegex(ValueError, "do not overlap"):
            validate_inference_result(candidate)
        candidate = valid_result()
        candidate["modes"]["persistent_tier"]["integrity"]["stale_rejections"] = 0
        with self.assertRaisesRegex(ValueError, "stale-version rejection"):
            validate_inference_result(candidate)

        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["route_evidence"][0]["counter_series"] = []
        with self.assertRaisesRegex(ValueError, "route counter series/count mismatch"):
            validate_inference_result(candidate)

    def test_host_wide_route_counter_cannot_be_promoted(self) -> None:
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["route_evidence"][0]["scope"] = "host"
        with self.assertRaisesRegex(ValueError, "scope"):
            validate_inference_result(candidate)

    def test_combined_consistency_cannot_rescue_failed_isolated_mode(self) -> None:
        candidate = valid_result()
        # Make the only isolated intervention fail its latency gate while the
        # full-flow consistency record remains attractive.  The result must
        # not promote a domain through ``combined``.
        for mode in ("d2h_only", "remote_fabric", "persistent_tier"):
            candidate["modes"][mode]["metrics"]["ttft_p99_ns"] = 120
            candidate["modes"][mode]["metrics"]["itl_p99_ns"] = 120
        candidate["modes"]["combined"]["metrics"]["ttft_p99_ns"] = 1
        candidate["modes"]["combined"]["metrics"]["itl_p99_ns"] = 1
        result = validate_inference_result(candidate)
        self.assertFalse(result["promote_static_policy"])
        self.assertNotIn("pcie_host", result["eligible_domains"])

    def test_failed_auxiliary_mode_cannot_complete_kv_matrix(self) -> None:
        candidate = valid_result()
        candidate["modes"]["combined"]["metrics"]["deadline_met"] = False
        with self.assertRaisesRegex(ValueError, "combined: live KV mode must satisfy"):
            validate_inference_result(candidate)

    def test_shared_domains_are_explicit_route_intersections(self) -> None:
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["metrics"]["shared_domains"] = []
        with self.assertRaisesRegex(ValueError, "route intersection"):
            validate_inference_result(candidate)

        candidate = valid_result()
        candidate["modes"]["persistent_tier"]["metrics"]["foreground_domains"] = [
            "gpu_local", "gpu_local"
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_inference_result(candidate)

    def test_domain_exposure_increase_cannot_be_hidden_by_latency_gain(self) -> None:
        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["metrics"]["max_domain_exposure_ns"] = 100_001
        for item in candidate["modes"]["remote_fabric"]["route_evidence"]:
            item["overlap_ns"] = 100_001
        candidate["modes"]["remote_fabric"]["metrics"]["ttft_p99_ns"] = 1
        candidate["modes"]["remote_fabric"]["metrics"]["itl_p99_ns"] = 1
        candidate["modes"]["remote_fabric"]["metrics"]["domain_exposure_ns"] = {
            domain: 100_001
            for domain in sorted(candidate["modes"]["remote_fabric"]["metrics"]["domain_exposure_ns"])
        }
        result = validate_inference_result(candidate)
        self.assertNotIn("slingshot_fabric", result["eligible_domains"])
        self.assertIn("pcie_host", result["eligible_domains"])

        candidate = valid_result()
        candidate["modes"]["remote_fabric"]["metrics"]["max_domain_exposure_ns"] = 100_001
        with self.assertRaisesRegex(ValueError, "not derived from route evidence"):
            validate_inference_result(candidate)

if __name__ == "__main__":
    unittest.main()
