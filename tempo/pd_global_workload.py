"""Deterministic contention workloads for the TEMPO-GO global controller.

The generated JSONL intentionally contains only the fields accepted by the
native vLLM streaming client.  Phase names, tenant/SLO intent, and the C1/C2/
C3 anchor labels live in the sidecar manifest; they are never sent as causal
inputs to the online controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


WORKLOAD_SCHEMA = "tempo-go-contention-workload-jsonl-v2"
MANIFEST_SCHEMA = "tempo-go-contention-manifest-v1"
TENANTS = ("latency", "interactive", "batch", "background")
GEOMETRY_CYCLE = ((512, 16), (2048, 256), (4094, 16))


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class ContentionPhase:
    name: str
    duration_ms: float = 15_000.0
    foreground_rate_per_s: float = 2.0
    decoder_hot_rate_per_s: float = 0.0
    remote_hot_rate_per_s: float = 0.0
    kv_remote_hot_rate_per_s: float = 0.0
    cooldown_ms: float = 2_000.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("phase name must be nonempty")
        _positive("duration_ms", self.duration_ms)
        for name in (
            "foreground_rate_per_s",
            "decoder_hot_rate_per_s",
            "remote_hot_rate_per_s",
            "kv_remote_hot_rate_per_s",
            "cooldown_ms",
        ):
            _nonnegative(name, getattr(self, name))
        if sum((
            self.foreground_rate_per_s,
            self.decoder_hot_rate_per_s,
            self.remote_hot_rate_per_s,
            self.kv_remote_hot_rate_per_s,
        )) <= 0.0:
            raise ValueError("phase must offer at least one request stream")


def canonical_contention_phases(
    *,
    duration_ms: float = 15_000.0,
    foreground_rate_per_s: float = 2.0,
    decoder_hot_rate_per_s: float = 22.4,
    remote_hot_rate_per_s: float = 4.76,
    kv_remote_hot_rate_per_s: float = 12.0,
    cooldown_ms: float = 2_000.0,
) -> tuple[ContentionPhase, ...]:
    """Return the frozen C1/C2/C3/recovery phase order."""

    common = dict(
        duration_ms=duration_ms,
        foreground_rate_per_s=foreground_rate_per_s,
        cooldown_ms=cooldown_ms,
    )
    return (
        ContentionPhase("c0_cool", **common),
        ContentionPhase(
            "c1_decoder_hot",
            decoder_hot_rate_per_s=decoder_hot_rate_per_s,
            **common,
        ),
        ContentionPhase(
            "c2_remote_hot",
            remote_hot_rate_per_s=remote_hot_rate_per_s,
            **common,
        ),
        ContentionPhase(
            "c2_kv_remote_hot",
            kv_remote_hot_rate_per_s=kv_remote_hot_rate_per_s,
            **common,
        ),
        ContentionPhase(
            "c3_both_hot",
            decoder_hot_rate_per_s=decoder_hot_rate_per_s,
            remote_hot_rate_per_s=remote_hot_rate_per_s,
            kv_remote_hot_rate_per_s=kv_remote_hot_rate_per_s,
            **common,
        ),
        ContentionPhase("recovery", **common),
    )


def _stream_count(rate_per_s: float, duration_ms: float) -> int:
    return int(math.ceil(rate_per_s * duration_ms / 1_000.0 - 1e-12))


def _prompt(
    source_pools: Mapping[int, Sequence[str]],
    prompt_tokens: int,
    index: int,
) -> str:
    pool = source_pools.get(prompt_tokens)
    if not pool:
        raise ValueError(f"source prompt pool is missing geometry {prompt_tokens}")
    prompt = pool[index % len(pool)]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("source prompt pool contains an invalid prompt")
    return prompt


def _cache_contract_marker(phase_name: str, stream_name: str) -> str:
    """Encode the only measured cache contracts admitted by the C5 gate.

    The KV-remote stream is the prepared P_ONLY case from the C2/C3 anchor.
    Every other measured stream is deliberately a cold/MISS request.  The
    marker is a checked contract for cache preparation, not an online policy
    input: the frontend still verifies the observed cache state.
    """

    if (
        stream_name == "kv-remote-hot"
        and phase_name in {"c2_kv_remote_hot", "c3_both_hot"}
    ):
        return "p-only"
    return "miss"


def _append_stream(
    rows: list[dict[str, object]],
    *,
    phase: ContentionPhase,
    phase_index: int,
    phase_start_ms: float,
    stream_name: str,
    rate_per_s: float,
    hot_output_tokens: int,
    source_pools: Mapping[int, Sequence[str]],
    replicate: int,
    sequence_offset: int,
) -> int:
    if rate_per_s <= 0.0:
        return sequence_offset
    count = _stream_count(rate_per_s, phase.duration_ms)
    interval_ms = 1_000.0 / rate_per_s
    cache_contract = _cache_contract_marker(phase.name, stream_name)
    for index in range(count):
        if stream_name == "foreground":
            prompt_tokens, stream_output = GEOMETRY_CYCLE[index % len(GEOMETRY_CYCLE)]
            tenant = TENANTS[index % len(TENANTS)]
            max_tokens = stream_output
        else:
            prompt_tokens = 4094
            tenant = "background"
            max_tokens = hot_output_tokens
        request_id = (
            f"epd-tempo-{tenant}-{phase.name}-cache-{cache_contract}-measured-"
            f"r{replicate:02d}-"
            f"{stream_name}-{sequence_offset:06d}"
        )
        rows.append({
            "request_id": request_id,
            "prompt": _prompt(
                source_pools, prompt_tokens,
                replicate * 1_000_000 + phase_index * 100_000 + index,
            ),
            "max_tokens": max_tokens,
            "arrival_offset_ms": round(phase_start_ms + index * interval_ms, 6),
        })
        sequence_offset += 1
    return sequence_offset


def build_contention_workload(
    source_pools: Mapping[int, Sequence[str]],
    *,
    phases: Sequence[ContentionPhase] | None = None,
    replicates: int = 1,
    anchor_output_tokens: int = 2,
    background_output_tokens: int = 128,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build one explicit-arrival workload and its auditable phase manifest."""

    if not isinstance(source_pools, Mapping) or not source_pools:
        raise ValueError("source_pools must be a nonempty geometry mapping")
    if type(replicates) is not int or replicates <= 0:
        raise ValueError("replicates must be a positive int")
    if type(anchor_output_tokens) is not int or anchor_output_tokens < 2:
        raise ValueError("anchor_output_tokens must be at least 2")
    if type(background_output_tokens) is not int or background_output_tokens < 2:
        raise ValueError("background_output_tokens must be at least 2")
    selected = tuple(phases or canonical_contention_phases())
    if not selected or any(not isinstance(item, ContentionPhase) for item in selected):
        raise ValueError("phases must contain ContentionPhase values")

    rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    cache_contract_counts = {"miss": 0, "p-only": 0}
    sequence = 0
    phase_start_ms = 0.0
    for replicate in range(replicates):
        for phase_index, phase in enumerate(selected):
            before = sequence
            sequence = _append_stream(
                rows, phase=phase, phase_index=phase_index,
                phase_start_ms=phase_start_ms,
                stream_name="foreground", rate_per_s=phase.foreground_rate_per_s,
                hot_output_tokens=anchor_output_tokens, source_pools=source_pools,
                replicate=replicate, sequence_offset=sequence,
            )
            for stream_name, rate in (
                ("foreground", phase.foreground_rate_per_s),
                ("decoder-hot", phase.decoder_hot_rate_per_s),
                ("remote-hot", phase.remote_hot_rate_per_s),
                ("kv-remote-hot", phase.kv_remote_hot_rate_per_s),
            ):
                if rate > 0.0:
                    cache_contract_counts[
                        _cache_contract_marker(phase.name, stream_name)
                    ] += _stream_count(rate, phase.duration_ms)
            for stream_name, rate in (
                ("decoder-hot", phase.decoder_hot_rate_per_s),
                ("remote-hot", phase.remote_hot_rate_per_s),
                ("kv-remote-hot", phase.kv_remote_hot_rate_per_s),
            ):
                sequence = _append_stream(
                    rows, phase=phase, phase_index=phase_index,
                    phase_start_ms=phase_start_ms,
                    stream_name=stream_name, rate_per_s=rate,
                    hot_output_tokens=anchor_output_tokens, source_pools=source_pools,
                    replicate=replicate, sequence_offset=sequence,
                )
            rows[before:sequence] = sorted(
                rows[before:sequence],
                key=lambda value: (float(value["arrival_offset_ms"]),
                                   str(value["request_id"])),
            )
            phase_rows.append({
                "replicate": replicate,
                "name": phase.name,
                "start_offset_ms": phase_start_ms,
                "duration_ms": phase.duration_ms,
                "cooldown_ms": phase.cooldown_ms,
                "row_start": before,
                "row_end": sequence,
                "request_count": sequence - before,
            })
            phase_start_ms += phase.duration_ms + phase.cooldown_ms

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "workload_schema": WORKLOAD_SCHEMA,
        "workload_fields": [
            "request_id", "prompt", "max_tokens", "arrival_offset_ms",
        ],
        "arrival_semantics": "explicit_absolute_offset_ms_from_one_client_epoch",
        "phase_order": [phase.name for phase in selected],
        "phases": phase_rows,
        "replicates": replicates,
        "tenant_order": list(TENANTS),
        "anchor_output_tokens": anchor_output_tokens,
        "background_output_tokens": background_output_tokens,
        "cache_contracts": {
            "encoded_in_request_id": True,
            "allowed_measured_states": ["miss", "p-only"],
            "counts": cache_contract_counts,
            "miss_prompt_namespace": (
                "token_preserving_unique_first_chunk_v1"
            ),
            "miss_unique_prompt_count": len({
                str(row["prompt"])
                for row in rows
                if "-cache-miss-measured-" in str(row["request_id"])
            }),
            "p_only_streams": ["kv-remote-hot"],
            "miss_streams": [
                "foreground", "decoder-hot", "remote-hot",
                "kv-remote-hot outside c2_kv_remote_hot/c3_both_hot",
            ],
        },
        "anchors": {
            "c1_decoder_hot_rate_per_s": selected[1].decoder_hot_rate_per_s
            if len(selected) > 1 else None,
            "c2_remote_hot_rate_per_s": selected[2].remote_hot_rate_per_s
            if len(selected) > 2 else None,
            "c2_kv_remote_hot_rate_per_s": selected[3].kv_remote_hot_rate_per_s
            if len(selected) > 3 else None,
            "c3_both_hot": "c3_both_hot" in {item.name for item in selected},
        },
        "comparison_arms": [
            "always_local", "official_always_remote", "predictor_only",
            "queue_gpu_only", "kairos_like", "tempo_go",
        ],
        "policy_inputs_excluded": [
            "phase_name", "future_arrivals", "physical_switch_label",
            "oracle_route",
        ],
        "performance_claim_allowed": False,
    }
    return rows, manifest


__all__ = [
    "ContentionPhase",
    "GEOMETRY_CYCLE",
    "MANIFEST_SCHEMA",
    "TENANTS",
    "WORKLOAD_SCHEMA",
    "build_contention_workload",
    "canonical_contention_phases",
]
