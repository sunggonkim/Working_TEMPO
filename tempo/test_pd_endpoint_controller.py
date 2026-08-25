from __future__ import annotations

import pytest

from tempo.pd_endpoint_controller import (
    EndpointAdmissionConfig,
    EndpointFeedbackController,
    EndpointRequest,
    EndpointRoute,
    EndpointWork,
    RouteHealth,
)


def _config(**changes) -> EndpointAdmissionConfig:
    values = {
        "local_token_ms_window": 200,
        "remote_prefill_token_ms_window": 200,
        "remote_kv_bytes_window": 2_000,
        "remote_semantic_ops_window": 2,
        "feedback_history": 4,
        "feedback_quantile": 0.75,
        "minimum_feedback": 1,
        "route_margin_ms": 5.0,
        "feedback_fresh_ns": 100,
        "probe_after_ns": 100,
        "denied_probe_after_ns": 300,
    }
    values.update(changes)
    return EndpointAdmissionConfig(**values)


def _request(request_id: str, **changes) -> EndpointRequest:
    values = {
        "request_id": request_id,
        "local_e2e_prior_ms": 200.0,
        "remote_e2e_prior_ms": 220.0,
        "local_ttft_prior_ms": 100.0,
        "remote_ttft_prior_ms": 120.0,
        "uncertainty_ms": 0.0,
        "e2e_deadline_ms": 1_000.0,
        "work": EndpointWork(
            local_token_ms=100,
            remote_prefill_token_ms=100,
            remote_kv_bytes=1_000,
            remote_semantic_ops=1,
        ),
    }
    values.update(changes)
    return EndpointRequest(**values)


def test_local_completion_inflation_flips_to_remote_with_zero_queue_input() -> None:
    controller = EndpointFeedbackController(_config())
    first = controller.submit(_request("a"), now_ns=1)
    assert first.route is EndpointRoute.LOCAL
    controller.observe_first_response("a", observed_ttft_ms=250.0, now_ns=10)

    second = controller.submit(_request("b"), now_ns=11)
    assert second.local_multiplier == 2.5
    assert second.local_score_ms == 350.0
    assert second.remote_score_ms == 220.0
    assert second.route is EndpointRoute.REMOTE
    assert second.local_state is RouteHealth.SKIP


def test_remote_completion_inflation_flips_back_to_local() -> None:
    controller = EndpointFeedbackController(_config())
    remote = controller.submit(
        _request("r", local_e2e_prior_ms=300.0, remote_e2e_prior_ms=200.0),
        now_ns=1,
    )
    assert remote.route is EndpointRoute.REMOTE
    controller.observe_first_response("r", observed_ttft_ms=360.0, now_ns=10)

    decision = controller.submit(_request("next"), now_ns=11)
    assert decision.remote_multiplier == 3.0
    assert decision.remote_score_ms == 460.0
    assert decision.route is EndpointRoute.LOCAL


def test_separate_local_and_remote_windows_allow_opposite_spill() -> None:
    controller = EndpointFeedbackController(_config(local_token_ms_window=100))
    local = controller.submit(_request("local"), now_ns=1)
    assert local.route is EndpointRoute.LOCAL

    spill = controller.submit(_request("spill"), now_ns=2)
    assert spill.route is EndpointRoute.REMOTE
    snapshot = controller.snapshot(now_ns=2)
    assert snapshot["resources"] == {
        "local_token_ms": 100,
        "remote_prefill_token_ms": 100,
        "remote_kv_bytes": 1_000,
        "remote_semantic_ops": 1,
    }


@pytest.mark.parametrize(
    ("config_change", "work_change"),
    [
        ({"remote_prefill_token_ms_window": 99}, {}),
        ({"remote_kv_bytes_window": 999}, {}),
        (
            {"remote_semantic_ops_window": 1},
            {"remote_semantic_ops": 2},
        ),
    ],
)
def test_each_remote_window_is_independently_enforced(
    config_change: dict[str, int], work_change: dict[str, int]
) -> None:
    config = _config(**config_change)
    controller = EndpointFeedbackController(config)
    work_values = {
        "local_token_ms": 100,
        "remote_prefill_token_ms": 100,
        "remote_kv_bytes": 1_000,
        "remote_semantic_ops": 1,
    }
    work_values.update(work_change)
    request = _request(
        "remote-only", local_allowed=False, work=EndpointWork(**work_values)
    )
    decision = controller.submit(request, now_ns=1)
    assert decision.route is EndpointRoute.QUEUE
    assert decision.reason == "endpoint_no_fresh_deadline_safe_window"


def test_first_response_releases_exact_resource_ownership() -> None:
    controller = EndpointFeedbackController(_config())
    controller.submit(_request("a"), now_ns=1)
    controller.observe_first_response("a", observed_ttft_ms=100.0, now_ns=2)
    assert controller.snapshot(now_ns=2)["resources"] == {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    with pytest.raises(ValueError, match="no in-flight"):
        controller.observe_first_response("a", observed_ttft_ms=100.0, now_ns=3)


def test_failure_denies_route_then_bounded_probe_recovers_it() -> None:
    controller = EndpointFeedbackController(_config())
    first = controller.submit(
        _request("fail", local_allowed=False), now_ns=1
    )
    assert first.route is EndpointRoute.REMOTE
    controller.fail("fail", now_ns=10)
    assert (
        controller.snapshot(now_ns=11)["routes"][EndpointRoute.REMOTE.value]["state"]
        == RouteHealth.DENIED.value
    )
    failed_remote = controller.snapshot(now_ns=11)["routes"][
        EndpointRoute.REMOTE.value]
    assert failed_remote["failures"] == 1
    assert failed_remote["last_failure_kind"] == "active_upstream_failure"
    assert failed_remote["last_failure_ns"] == 10

    before = controller.submit(_request("before"), now_ns=309)
    assert before.route is EndpointRoute.LOCAL
    controller.observe_first_response("before", observed_ttft_ms=100.0, now_ns=310)

    probe = controller.submit(_request("probe"), now_ns=310)
    assert probe.route is EndpointRoute.REMOTE
    assert probe.probe is True
    assert probe.remote_state is RouteHealth.PROBE
    controller.observe_first_response("probe", observed_ttft_ms=120.0, now_ns=320)
    remote = controller.snapshot(now_ns=320)["routes"][EndpointRoute.REMOTE.value]
    assert remote["state"] == RouteHealth.GOOD.value
    assert remote["feedback_count"] == 1
    assert remote["service_multiplier"] == 1.0


def test_only_one_probe_can_own_a_route() -> None:
    controller = EndpointFeedbackController(_config())
    first = controller.submit(_request("first", local_allowed=False), now_ns=1)
    controller.fail("first", now_ns=10)
    probe = controller.submit(_request("probe", local_allowed=False), now_ns=310)
    assert probe.probe is True
    blocked = controller.submit(_request("blocked", local_allowed=False), now_ns=311)
    assert blocked.route is EndpointRoute.QUEUE


def test_failure_free_stale_multiplier_cannot_deadlock_recovery_probe() -> None:
    controller = EndpointFeedbackController(_config())
    first = controller.submit(_request(
        "slow-but-valid",
        local_allowed=True,
        remote_allowed=False,
        local_ttft_prior_ms=10.0,
    ), now_ns=1)
    assert first.route is EndpointRoute.LOCAL
    assert controller.observe_first_response(
        "slow-but-valid", observed_ttft_ms=900.0, now_ns=10) is True

    # The 90x stale multiplier makes the dynamic score exceed the request's
    # deadline, but there was no failure and the frozen static service prior
    # still fits.  A bounded probe must be able to re-establish liveness.
    probe = controller.submit(_request(
        "completion-liveness-probe",
        local_allowed=True,
        remote_allowed=False,
        local_ttft_prior_ms=10.0,
    ), now_ns=111)
    assert probe.route is EndpointRoute.LOCAL
    assert probe.probe is True
    assert probe.local_multiplier == 90.0
    assert probe.local_score_ms > 1_000.0


def test_pre_failure_inflight_success_cannot_recover_route() -> None:
    controller = EndpointFeedbackController(_config())
    first = controller.submit(
        _request("first", local_allowed=False), now_ns=1)
    second = controller.submit(
        _request("second", local_allowed=False), now_ns=2)
    assert first.route is EndpointRoute.REMOTE
    assert second.route is EndpointRoute.REMOTE

    controller.fail("first", now_ns=10)
    accepted = controller.observe_first_response(
        "second", observed_ttft_ms=120.0, now_ns=11)
    assert accepted is False
    remote = controller.snapshot(now_ns=11)["routes"][
        EndpointRoute.REMOTE.value]
    assert remote["state"] == RouteHealth.DENIED.value
    assert remote["failures"] == 1
    assert remote["active_samples"] == 0
    assert remote["active_ignored_while_unhealthy"] == 1
    assert controller.snapshot(now_ns=11)["resources"] == {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }

    probe = controller.submit(
        _request("probe-after-race", local_allowed=False), now_ns=310)
    assert probe.probe is True
    assert controller.observe_first_response(
        "probe-after-race", observed_ttft_ms=120.0, now_ns=320) is True
    remote = controller.snapshot(now_ns=320)["routes"][
        EndpointRoute.REMOTE.value]
    assert remote["failures"] == 0
    assert remote["probe_request_id"] is None


def test_slo_violation_uses_request_deadline_and_enters_denied() -> None:
    controller = EndpointFeedbackController(_config())
    controller.submit(
        _request(
            "slow",
            local_e2e_prior_ms=100.0,
            e2e_deadline_ms=150.0,
            remote_allowed=False,
        ),
        now_ns=1,
    )
    controller.observe_first_response("slow", observed_ttft_ms=160.0, now_ns=2)
    state = controller.snapshot(now_ns=3)["routes"][EndpointRoute.LOCAL.value]
    assert state["state"] == RouteHealth.DENIED.value
    assert state["failures"] == 1


def test_passive_feedback_flips_route_without_owning_resources() -> None:
    controller = EndpointFeedbackController(_config())
    accepted = controller.observe_passive_first_response(
        "background-local",
        route=EndpointRoute.LOCAL,
        observed_ttft_ms=250.0,
        prior_ttft_ms=100.0,
        now_ns=10,
    )
    assert accepted is True
    snapshot = controller.snapshot(now_ns=10)
    assert snapshot["resources"] == {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    assert snapshot["passive_completed"] == 1
    local = snapshot["routes"][EndpointRoute.LOCAL.value]
    assert local["passive_samples"] == 1
    assert local["active_samples"] == 0
    decision = controller.submit(_request("foreground"), now_ns=11)
    assert decision.route is EndpointRoute.REMOTE


def test_external_route_pinned_work_consumes_and_releases_shared_credit() -> None:
    controller = EndpointFeedbackController(_config())
    work = _request("template").work
    controller.observe_external_start(
        "background-remote",
        route=EndpointRoute.REMOTE,
        work=work,
        prior_ttft_ms=120.0,
        e2e_deadline_ms=1_000.0,
        now_ns=1,
    )
    snapshot = controller.snapshot(now_ns=1)
    assert snapshot["owned_resources"] == {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    assert snapshot["external_resources"] == {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 100,
        "remote_kv_bytes": 1_000,
        "remote_semantic_ops": 1,
    }
    assert snapshot["external_inflight"] == 1

    admitted = controller.submit(
        _request("foreground", local_allowed=False), now_ns=2)
    assert admitted.route is EndpointRoute.REMOTE
    blocked = controller.submit(
        _request("blocked", local_allowed=False), now_ns=3)
    assert blocked.route is EndpointRoute.QUEUE

    assert controller.observe_external_first_response(
        "background-remote", observed_ttft_ms=120.0, now_ns=4) is True
    snapshot = controller.snapshot(now_ns=4)
    assert snapshot["external_inflight"] == 0
    assert snapshot["external_resources"] == {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    controller.observe_first_response(
        "foreground", observed_ttft_ms=120.0, now_ns=5)
    retry = controller.submit(
        _request("retry", local_allowed=False), now_ns=6)
    assert retry.route is EndpointRoute.REMOTE


def test_external_failure_releases_credit_and_denies_observed_route() -> None:
    controller = EndpointFeedbackController(_config())
    controller.observe_external_start(
        "failed-local",
        route=EndpointRoute.LOCAL,
        work=_request("template").work,
        prior_ttft_ms=100.0,
        e2e_deadline_ms=1_000.0,
        now_ns=1,
    )
    controller.fail_external("failed-local", now_ns=2)
    snapshot = controller.snapshot(now_ns=2)
    assert snapshot["external_inflight"] == 0
    assert snapshot["resources"] == {
        "local_token_ms": 0,
        "remote_prefill_token_ms": 0,
        "remote_kv_bytes": 0,
        "remote_semantic_ops": 0,
    }
    local = snapshot["routes"][EndpointRoute.LOCAL.value]
    assert local["state"] == RouteHealth.DENIED.value
    assert local["passive_failures"] == 1
    with pytest.raises(ValueError, match="already observed"):
        controller.observe_external_start(
            "failed-local",
            route=EndpointRoute.LOCAL,
            work=_request("template").work,
            prior_ttft_ms=100.0,
            e2e_deadline_ms=1_000.0,
            now_ns=3,
        )


def test_passive_success_cannot_bypass_explicit_denied_probe() -> None:
    controller = EndpointFeedbackController(_config())
    controller.fail_passive(
        "background-failure", route=EndpointRoute.REMOTE, now_ns=10)
    accepted = controller.observe_passive_first_response(
        "background-success",
        route=EndpointRoute.REMOTE,
        observed_ttft_ms=120.0,
        prior_ttft_ms=120.0,
        now_ns=311,
    )
    assert accepted is False
    remote = controller.snapshot(now_ns=311)["routes"][
        EndpointRoute.REMOTE.value]
    assert remote["state"] == RouteHealth.SKIP.value
    assert remote["failures"] == 1
    assert remote["passive_samples"] == 0
    assert remote["passive_ignored_while_unhealthy"] == 1
    probe = controller.submit(
        _request("explicit-probe", local_allowed=False), now_ns=311)
    assert probe.route is EndpointRoute.REMOTE
    assert probe.probe is True


def test_passive_feedback_rejects_duplicate_or_active_ids() -> None:
    controller = EndpointFeedbackController(_config())
    controller.observe_passive_first_response(
        "duplicate",
        route=EndpointRoute.LOCAL,
        observed_ttft_ms=100.0,
        prior_ttft_ms=100.0,
        now_ns=1,
    )
    with pytest.raises(ValueError, match="already observed"):
        controller.observe_passive_first_response(
            "duplicate",
            route=EndpointRoute.LOCAL,
            observed_ttft_ms=100.0,
            prior_ttft_ms=100.0,
            now_ns=2,
        )
    with pytest.raises(ValueError, match="already owned"):
        controller.submit(_request("duplicate"), now_ns=3)
    controller.submit(_request("active"), now_ns=4)
    with pytest.raises(ValueError, match="already observed or owned"):
        controller.fail_passive(
            "active", route=EndpointRoute.LOCAL, now_ns=5)


def test_cache_route_constraints_are_fail_closed() -> None:
    controller = EndpointFeedbackController(_config())
    local = controller.submit(_request("local", remote_allowed=False), now_ns=1)
    assert local.route is EndpointRoute.LOCAL
    remote = controller.submit(
        _request(
            "remote",
            local_allowed=False,
            local_e2e_prior_ms=100.0,
            remote_e2e_prior_ms=300.0,
        ),
        now_ns=2,
    )
    assert remote.route is EndpointRoute.REMOTE


def test_invalid_configuration_and_request_are_rejected() -> None:
    with pytest.raises(ValueError, match="minimum_feedback"):
        _config(feedback_history=1, minimum_feedback=2)
    with pytest.raises(ValueError, match="probe_after_ns"):
        _config(feedback_fresh_ns=100, probe_after_ns=101)
    with pytest.raises(ValueError, match="at least one route"):
        _request("none", local_allowed=False, remote_allowed=False)
