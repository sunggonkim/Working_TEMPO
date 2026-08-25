from __future__ import annotations

import asyncio

import pytest

from tempo.pd_global_agent import RequestTriggeredTelemetryAgent
from tempo.test_pd_global_telemetry import adapter, endpoint, frontend


class Clock:
    def __init__(self, value: int = 1_000) -> None:
        self.value = value

    def __call__(self) -> int:
        result = self.value
        self.value += 1
        return result

    def advance(self, amount: int) -> None:
        self.value += amount


def build_agent(
    *, clock: Clock, counts: dict[str, int], timeout_ns: int = 1_000_000_000,
) -> RequestTriggeredTelemetryAgent:
    async def fetch_frontend():
        counts["frontend"] += 1
        await asyncio.sleep(0)
        return frontend()

    def endpoint_fetcher(pair_index: int):
        async def fetch():
            counts[f"pair{pair_index}"] += 1
            await asyncio.sleep(0)
            return endpoint(pair_index)

        return fetch

    return RequestTriggeredTelemetryAgent(
        adapter(),
        frontend_fetcher=fetch_frontend,
        endpoint_fetchers=(endpoint_fetcher(0), endpoint_fetcher(1)),
        freshness_ns=100,
        refresh_timeout_ns=timeout_ns,
        clock_ns=clock,
    )


def test_agent_has_no_background_polling_and_uses_fresh_cache() -> None:
    async def scenario() -> None:
        counts = {"frontend": 0, "pair0": 0, "pair1": 0}
        value = build_agent(clock=Clock(), counts=counts)
        assert counts == {"frontend": 0, "pair0": 0, "pair1": 0}
        first = await value.get()
        second = await value.get()
        assert first is second
        assert counts == {"frontend": 1, "pair0": 1, "pair1": 1}
        assert value.status() == {
            "mode": "request_triggered_bounded_single_flight",
            "background_polling": False,
            "requests": 2,
            "cache_hits": 1,
            "refreshes": 1,
            "failures": 0,
            "timeouts": 0,
            "quarantines": 0,
            "forced_refresh_coalesces": 0,
            "last_quarantined_pairs": [],
            "last_error": None,
            "last_sequence": 1,
            "last_sampled_ns": first.sampled_ns,
            "freshness_ns": 100,
            "refresh_timeout_ns": 1_000_000_000,
            "per_fetch_timeout_ns": 1_000_000,
        }

    asyncio.run(scenario())


def test_concurrent_stale_callers_share_one_refresh() -> None:
    async def scenario() -> None:
        counts = {"frontend": 0, "pair0": 0, "pair1": 0}
        value = build_agent(clock=Clock(), counts=counts)
        batches = await asyncio.gather(*(value.get() for _ in range(8)))
        assert {item.sequence for item in batches} == {1}
        assert counts == {"frontend": 1, "pair0": 1, "pair1": 1}
        assert value.status()["cache_hits"] == 7

    asyncio.run(scenario())


def test_failed_endpoint_quarantines_only_that_pair() -> None:
    async def scenario() -> None:
        counts = {"frontend": 0, "pair0": 0, "pair1": 0}
        clock = Clock()
        value = build_agent(clock=clock, counts=counts)
        first = await value.get()
        clock.advance(1_000)

        async def fail():
            raise OSError("synthetic endpoint failure")

        original = value.endpoint_fetchers
        value.endpoint_fetchers = (original[0], fail)
        quarantined = await value.get()
        assert quarantined.sequence == first.sequence + 1
        assert quarantined.pairs[0].local_health.value == "good"
        assert quarantined.pairs[1].local_health.value == "denied"
        assert quarantined.pairs[1].remote_health.value == "denied"
        assert quarantined.pairs[1].quarantine_reason == "endpoint_fetch:OSError"
        assert value.status()["last_sequence"] == quarantined.sequence
        assert value.status()["last_quarantined_pairs"] == [1]
        assert value.status()["quarantines"] == 1
        value.endpoint_fetchers = original
        recovered = await value.get(force=True)
        assert recovered.sequence == quarantined.sequence + 1
        assert recovered.pairs[1].local_health.value == "good"
        assert value.status()["failures"] == 0

    asyncio.run(scenario())


def test_timed_out_endpoint_quarantines_only_that_pair() -> None:
    async def scenario() -> None:
        counts = {"frontend": 0, "pair0": 0, "pair1": 0}
        value = build_agent(
            clock=Clock(), counts=counts, timeout_ns=5_000_000)

        async def blocked():
            await asyncio.Event().wait()

        original = value.endpoint_fetchers
        value.endpoint_fetchers = (original[0], blocked)
        batch = await value.get()
        assert batch.pairs[0].local_health.value == "good"
        assert batch.pairs[1].local_health.value == "denied"
        assert batch.pairs[1].remote_health.value == "denied"
        assert batch.pairs[1].quarantine_reason == (
            "endpoint_fetch:TimeoutError")
        state = value.status()
        assert state["failures"] == 0
        assert state["timeouts"] == 0
        assert state["quarantines"] == 1

    asyncio.run(scenario())


def test_all_endpoint_timeouts_preserve_last_complete_batch() -> None:
    async def scenario() -> None:
        counts = {"frontend": 0, "pair0": 0, "pair1": 0}
        clock = Clock()
        value = build_agent(
            clock=clock, counts=counts, timeout_ns=5_000_000)
        first = await value.get()
        clock.advance(1_000)

        async def blocked():
            await asyncio.Event().wait()

        original = value.endpoint_fetchers
        value.endpoint_fetchers = (blocked, blocked)
        with pytest.raises(RuntimeError, match="refresh timed out"):
            await value.get()
        state = value.status()
        assert state["last_sequence"] == first.sequence
        assert state["last_quarantined_pairs"] == []
        assert state["quarantines"] == 0
        assert state["failures"] == 1
        assert state["timeouts"] == 1
        assert state["last_error"] == "telemetry_all_endpoints_timeout"

        value.endpoint_fetchers = original
        recovered = await value.get(force=True)
        assert recovered.sequence == first.sequence + 1

    asyncio.run(scenario())


def test_invalid_snapshot_does_not_advance_adapter_sequence() -> None:
    async def scenario() -> None:
        counts = {"frontend": 0, "pair0": 0, "pair1": 0}
        value = build_agent(clock=Clock(), counts=counts)

        async def wrong_pair():
            raw = endpoint(1)
            raw["pair_index"] = 0
            return raw

        original = value.endpoint_fetchers
        value.endpoint_fetchers = (original[0], wrong_pair)
        with pytest.raises(RuntimeError, match="validation failed"):
            await value.get()
        value.endpoint_fetchers = original
        result = await value.get()
        assert result.sequence == 1

    asyncio.run(scenario())


def test_timeout_is_bounded_and_fail_closed() -> None:
    async def scenario() -> None:
        counts = {"frontend": 0, "pair0": 0, "pair1": 0}
        value = build_agent(
            clock=Clock(), counts=counts, timeout_ns=1_000_000)

        async def blocked():
            await asyncio.Event().wait()
            return endpoint(0)

        value.frontend_fetcher = blocked
        with pytest.raises(RuntimeError, match="timed out"):
            await value.get()
        state = value.status()
        assert state["last_sequence"] is None
        assert state["failures"] == 1
        assert state["timeouts"] == 1

    asyncio.run(scenario())
