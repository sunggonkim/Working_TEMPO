import unittest

from tempo import pd_contention_workload as workload


class PDContentionWorkloadTest(unittest.TestCase):
    def setUp(self):
        self.selection = workload.LoadSelection(
            decoder_reference_rate_per_s=10.0,
            remote_reference_rate_per_s=4.0,
            decoder_fraction=0.50,
            remote_fraction=0.50,
        )

    def test_arm_blocks_share_semantic_schedule_but_not_request_ids(self):
        local = workload.build_schedule(
            states=(workload.ContentionState.C1,),
            selection=self.selection,
            foreground_arm=workload.ForegroundArm.LOCAL,
            foreground_rate_per_s=2.0,
            trial_id="local-r0",
            phase_duration_ms=1000.0,
        )
        remote = workload.build_schedule(
            states=(workload.ContentionState.C1,),
            selection=self.selection,
            foreground_arm=workload.ForegroundArm.REMOTE,
            foreground_rate_per_s=2.0,
            trial_id="remote-r0",
            phase_duration_ms=1000.0,
        )
        self.assertEqual(
            workload.semantic_schedule_sha256(local),
            workload.semantic_schedule_sha256(remote),
        )
        self.assertNotEqual(
            {item.request_id for item in local},
            {item.request_id for item in remote},
        )
        decoder = [item for item in local if item.tenant is workload.Tenant.DECODER_HOT]
        remote_hot = [item for item in local if item.tenant is workload.Tenant.REMOTE_HOT]
        foreground = [item for item in local if item.tenant is workload.Tenant.FOREGROUND]
        self.assertEqual(len(decoder), 5)
        self.assertEqual(len(remote_hot), 0)
        self.assertEqual(len(foreground), 2)
        self.assertTrue(all(item.arm is workload.ForegroundArm.LOCAL for item in decoder))
        self.assertTrue(all(item.geometry.prompt_tokens == 4094 for item in decoder))
        self.assertTrue(all(item.geometry.output_tokens == 2 for item in decoder))

    def test_path_hot_tenants_use_identical_long_context_geometry(self):
        self.assertEqual(
            workload.DECODER_HOT_GEOMETRY,
            workload.REMOTE_HOT_GEOMETRY,
        )
        self.assertEqual(workload.DECODER_HOT_GEOMETRY.prompt_tokens, 4094)
        self.assertEqual(workload.DECODER_HOT_GEOMETRY.output_tokens, 2)

    def test_crossover_and_validation_foreground_geometries_are_separate(self):
        self.assertEqual(
            workload.CROSSOVER_FOREGROUND_GEOMETRIES,
            (workload.TokenGeometry(4094, 2, workload.CacheState.MISS),),
        )
        self.assertGreater(len(workload.VALIDATION_FOREGROUND_GEOMETRIES), 1)
        self.assertEqual(
            workload.FOREGROUND_GEOMETRIES,
            workload.VALIDATION_FOREGROUND_GEOMETRIES,
        )

    def test_c4_trace_activates_only_declared_inference_tenants(self):
        states = (
            workload.ContentionState.C0,
            workload.ContentionState.C1,
            workload.ContentionState.C2,
            workload.ContentionState.C2_KV,
            workload.ContentionState.C3,
            workload.ContentionState.RECOVERY,
        )
        requests = workload.build_schedule(
            states=states,
            selection=self.selection,
            foreground_arm=workload.ForegroundArm.TEMPO,
            foreground_rate_per_s=1.0,
            trial_id="trace-r0",
            phase_duration_ms=1000.0,
        )
        tenants = {
            state: {
                item.tenant for item in requests
                if item.phase is state
            }
            for state in states
        }
        self.assertEqual(tenants[workload.ContentionState.C0], {
            workload.Tenant.FOREGROUND})
        self.assertEqual(tenants[workload.ContentionState.C1], {
            workload.Tenant.FOREGROUND, workload.Tenant.DECODER_HOT})
        self.assertEqual(tenants[workload.ContentionState.C2], {
            workload.Tenant.FOREGROUND, workload.Tenant.REMOTE_HOT})
        self.assertEqual(tenants[workload.ContentionState.C2_KV], {
            workload.Tenant.FOREGROUND, workload.Tenant.KV_REMOTE_HOT})
        self.assertEqual(tenants[workload.ContentionState.C3], {
            workload.Tenant.FOREGROUND,
            workload.Tenant.DECODER_HOT,
            workload.Tenant.KV_REMOTE_HOT,
        })
        self.assertEqual(tenants[workload.ContentionState.RECOVERY], {
            workload.Tenant.FOREGROUND})

    def test_c4_background_ids_opt_into_passive_endpoint_feedback(self):
        requests = workload.build_schedule(
            states=(
                workload.ContentionState.C1,
                workload.ContentionState.C2,
                workload.ContentionState.C2_KV,
            ),
            selection=self.selection,
            foreground_arm=workload.ForegroundArm.TEMPO,
            foreground_rate_per_s=1.0,
            trial_id="passive-r0",
            phase_duration_ms=1000.0,
            passive_endpoint_feedback=True,
        )
        background = [
            item for item in requests
            if item.tenant is not workload.Tenant.FOREGROUND
        ]
        foreground = [
            item for item in requests
            if item.tenant is workload.Tenant.FOREGROUND
        ]
        self.assertTrue(background)
        self.assertTrue(all(
            "-endpoint-observed-" in item.request_id
            for item in background
        ))
        self.assertTrue(all(
            "-endpoint-observed-" not in item.request_id
            for item in foreground
        ))
        kv = [
            item for item in background
            if item.tenant is workload.Tenant.KV_REMOTE_HOT
        ]
        self.assertTrue(kv)
        self.assertTrue(all(
            "-cache-p-only-measured-" in item.request_id
            for item in kv
        ))
        cache_markers = {
            workload.CacheState.MISS: "-cache-miss-measured-",
            workload.CacheState.P_ONLY: "-cache-p-only-measured-",
            workload.CacheState.D_ONLY: "-cache-d-only-measured-",
            workload.CacheState.BOTH: "-cache-both-measured-",
        }
        self.assertTrue(all(
            cache_markers[item.geometry.cache_state] in item.request_id
            for item in requests
        ))

    def test_burst_preserves_count_and_overload_increases_count(self):
        common = dict(
            states=(workload.ContentionState.C3,),
            selection=self.selection,
            foreground_arm=workload.ForegroundArm.TEMPO,
            foreground_rate_per_s=4.0,
            trial_id="shape-r0",
            phase_duration_ms=4000.0,
        )
        stable = workload.build_schedule(
            **common, shape=workload.TrafficShape.STABLE)
        burst = workload.build_schedule(
            **common, shape=workload.TrafficShape.BURST)
        overload = workload.build_schedule(
            **common, shape=workload.TrafficShape.OVERLOAD)
        self.assertEqual(len(stable), len(burst))
        self.assertGreater(len(overload), len(stable))
        self.assertNotEqual(
            workload.semantic_schedule_sha256(stable),
            workload.semantic_schedule_sha256(burst),
        )

    @staticmethod
    def _matrix(fraction: float, *, gain: float = 0.10):
        observations = []
        for replicate in range(workload.CALIBRATION_REPLICATES):
            schedule = (f"{replicate + 1:x}" * 64)[:64]
            for phase in (workload.ContentionState.C1, workload.ContentionState.C2):
                if phase is workload.ContentionState.C1:
                    local_e2e = 100.0
                    remote_e2e = 100.0 * (1.0 - gain)
                else:
                    remote_e2e = 100.0
                    local_e2e = 100.0 * (1.0 - gain)
                for arm, e2e in (
                    (workload.ForegroundArm.LOCAL, local_e2e),
                    (workload.ForegroundArm.REMOTE, remote_e2e),
                ):
                    rows = tuple(
                        workload.ForegroundObservation(
                            pair_key=f"item-{item}",
                            e2e_ms=e2e + item,
                            output_sha256=f"{item + 10:064x}",
                        )
                        for item in range(4)
                    )
                    observations.append(workload.FixedArmObservation(
                        phase=phase,
                        load_fraction=fraction,
                        replicate=replicate,
                        arm=arm,
                        semantic_schedule_sha256=schedule,
                        foreground=rows,
                        background_offered=8,
                        background_completed=8,
                        background_errors=0,
                    ))
        return observations

    def test_opposite_fixed_crossovers_are_required(self):
        report = workload.evaluate_crossover(
            self._matrix(0.50), load_fraction=0.50)
        self.assertTrue(report["workload_valid_for_controller_tuning"])
        c1 = report["phase_results"][workload.ContentionState.C1.value]
        c2 = report["phase_results"][workload.ContentionState.C2.value]
        self.assertEqual(c1["winner"], workload.ForegroundArm.REMOTE.value)
        self.assertEqual(c2["winner"], workload.ForegroundArm.LOCAL.value)
        self.assertGreaterEqual(c1["paired_median_gain"], 0.05)
        self.assertGreaterEqual(c2["paired_median_gain"], 0.05)

    def test_schedule_output_and_background_failures_are_fail_closed(self):
        for mutation, message in (
            ("schedule", "different semantic schedules"),
            ("output", "outputs differ"),
            ("background", "background inference"),
        ):
            with self.subTest(mutation=mutation):
                observations = self._matrix(0.50)
                target = observations[1]
                values = dict(target.__dict__)
                if mutation == "schedule":
                    values["semantic_schedule_sha256"] = "f" * 64
                elif mutation == "output":
                    rows = list(target.foreground)
                    row = rows[0]
                    rows[0] = workload.ForegroundObservation(
                        pair_key=row.pair_key,
                        e2e_ms=row.e2e_ms,
                        output_sha256="f" * 64,
                    )
                    values["foreground"] = tuple(rows)
                else:
                    values["background_completed"] = 7
                observations[1] = workload.FixedArmObservation(**values)
                with self.assertRaisesRegex(ValueError, message):
                    workload.evaluate_crossover(
                        observations, load_fraction=0.50)

    def test_calibration_chooses_first_passing_level_once(self):
        rows = self._matrix(0.50, gain=0.02) + self._matrix(0.70, gain=0.08)
        selected, reports = workload.choose_first_valid_fraction(rows)
        self.assertEqual(selected, 0.70)
        self.assertEqual([item["load_fraction"] for item in reports], [0.50, 0.70])
        self.assertFalse(reports[0]["workload_valid_for_controller_tuning"])
        self.assertTrue(reports[1]["workload_valid_for_controller_tuning"])

    def test_preregistration_keeps_controller_out_of_calibration(self):
        plan = workload.default_preregistration()
        self.assertEqual(plan["schema"], workload.SCHEMA)
        self.assertEqual(
            plan["calibration_fractions"],
            list(workload.CALIBRATION_FRACTIONS),
        )
        self.assertFalse(plan["controller_tuning_before_crossover"])
        self.assertEqual(
            plan["decoder_hot"]["pressure_scope"],
            "decoder_local_prefill_engine",
        )
        self.assertFalse(
            plan["v1_shared_decode_negative"]["c1_crossover_observed"])
        self.assertEqual(
            plan["v2_capacity_normalization_negative"]
            ["v3_reference_rates_per_s"],
            {"local": 16, "remote": 8},
        )
        self.assertEqual(
            plan["v3_capacity_bracket"]["v4_reference_rates_per_s"],
            {"local": 32, "remote": 6.8},
        )
        self.assertFalse(
            plan["v3_capacity_bracket"]["local_capacity_knee_observed"])
        self.assertEqual(
            plan["remote_hot"]["path"],
            "actual_prefill_plus_official_lmcache_transfer_and_install",
        )
        self.assertEqual(
            plan["phase_changing_trace"],
            [
                "c0_cool", "c1_decoder_hot", "c2_remote_hot",
                "c2_kv_remote_hot", "c3_both_hot", "recovery",
            ],
        )
        self.assertEqual(plan["kv_remote_hot"]["offered_rate_per_s"], 12.0)
        self.assertFalse(
            plan["kv_remote_hot"]["zero_producer_compute_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
