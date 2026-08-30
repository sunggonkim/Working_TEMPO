"""External vLLM NIXL connector with TEMPO scheduler-side admission.

Only the scheduler facade is replaced.  The upstream NIXL pull worker,
metadata format, handshakes, transfer implementation, and completion path are
kept unchanged.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)

from tempo.pd_connector_admission_v439 import ConnectorAdmissionState


SCHEMA = "tempo-nixl-connector-440"


class TempoNixlPullScheduler(NixlPullConnectorScheduler):
    def __init__(self, vllm_config, engine_id, kv_cache_config):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._tempo_admission = ConnectorAdmissionState(
            local_inflight_cap=int(extra.get("tempo_local_inflight_cap", 6)),
            microburst_threshold_ns=int(
                extra.get("tempo_microburst_threshold_ns", 25_000_000)),
        )
        trace = extra.get("tempo_admission_trace")
        self._tempo_trace = Path(trace).resolve() if trace else None
        if self._tempo_trace is not None:
            if self._tempo_trace.exists():
                raise ValueError("tempo admission trace already exists")
            self._tempo_trace.parent.mkdir(parents=True, exist_ok=True)
            self._append_trace({
                "schema": SCHEMA,
                "event": "provenance",
                "engine_id": engine_id,
                "local_inflight_cap": self._tempo_admission.local_inflight_cap,
                "microburst_threshold_ns": (
                    self._tempo_admission.microburst_threshold_ns),
                "pid": os.getpid(),
            })

    def _append_trace(self, value):
        if self._tempo_trace is None:
            return
        payload = (json.dumps(value, sort_keys=True, separators=(",", ":"))
                   + "\n").encode()
        descriptor = os.open(
            self._tempo_trace,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params
        if params is None or not params.get("do_remote_prefill"):
            return super().get_num_new_matched_tokens(request, num_computed_tokens)
        decision = self._tempo_admission.decide(
            request.request_id, time.perf_counter_ns())
        params["tempo_admission"] = asdict(decision)
        self._append_trace({
            "schema": SCHEMA,
            "event": "decision",
            **asdict(decision),
        })
        if decision.route == "decoder_local_recompute":
            # update_state_after_alloc receives zero external tokens.  Upstream
            # NIXL then emits a notification-only receive with empty local block
            # IDs, releasing the remote prefiller's leased blocks without a KV copy.
            return 0, False
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def request_finished(self, request, block_ids):
        self._tempo_admission.finish(request.request_id)
        self._append_trace({
            "schema": SCHEMA,
            "event": "finished",
            "request_id": request.request_id,
            "local_inflight_after": self._tempo_admission.local_inflight,
        })
        return super().request_finished(request, block_ids)


class TempoNixlConnector(NixlPullConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.SCHEDULER:
            # The stock scheduler has not started its handshake thread yet at
            # connector construction, so replacing it here leaks no process or socket.
            self.connector_scheduler = TempoNixlPullScheduler(
                vllm_config, self.engine_id, kv_cache_config)
