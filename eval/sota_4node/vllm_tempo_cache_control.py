#!/usr/bin/env python3
"""Launch vLLM with audited request-level APC control and evidence.

vLLM already carries ``SamplingParams.skip_reading_prefix_cache`` internally,
but the OpenAI completion protocol does not expose it directly.  TEMPO sends a
strict integer in ``vllm_xargs`` and this launcher copies only that value onto
the existing sampling parameter.

The stock OpenAI usage field combines local APC hits and external KV-transfer
hits.  vLLM's existing ``PrefillStats`` retains the exact two-way breakdown,
so this launcher also carries that already-computed request statistic through
``RequestOutput`` and appends it to the final SSE usage object.  It does not
alter KV production, LMCacheConnectorV1, transfer, install, scheduling, model
execution, or the response token stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


XARG = "tempo_skip_local_prefix_cache_read"
_PATCH_MARKER = "_tempo_cache_control_original"
_STATS_PATCH_MARKER = "_tempo_cache_breakdown_stats_original"
_OUTPUT_PATCH_MARKER = "_tempo_cache_breakdown_output_original"
_SERVING_PATCH_MARKER = "_tempo_cache_breakdown_serving_original"
_STATE_BREAKDOWN_ATTR = "_tempo_cache_breakdown"
_OUTPUT_BREAKDOWN_ATTR = "_tempo_cache_breakdown"

CACHE_BREAKDOWN_SCHEMA = "tempo-vllm-prefill-cache-breakdown-v1"
CACHE_BREAKDOWN_SCHEMA_FIELD = "tempo_cache_breakdown_schema"
LOCAL_CACHED_TOKENS_FIELD = "tempo_local_cached_tokens"
EXTERNAL_CACHED_TOKENS_FIELD = "tempo_external_cached_tokens"


@dataclass(frozen=True)
class CacheBreakdown:
    prompt_tokens: int
    cached_tokens: int
    local_cached_tokens: int
    external_cached_tokens: int

    def __post_init__(self) -> None:
        for name in (
            "prompt_tokens",
            "cached_tokens",
            "local_cached_tokens",
            "external_cached_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise RuntimeError(
                    f"vLLM PrefillStats {name} is not a non-negative integer")
        if self.cached_tokens != (
            self.local_cached_tokens + self.external_cached_tokens
        ):
            raise RuntimeError(
                "vLLM PrefillStats cached-token breakdown is inconsistent")
        if self.cached_tokens > self.prompt_tokens:
            raise RuntimeError(
                "vLLM PrefillStats cached tokens exceed prompt tokens")


def cache_breakdown_from_prefill_stats(stats: Any) -> CacheBreakdown:
    """Read the exact cache-source split already produced by vLLM V1."""

    required = (
        "num_prompt_tokens",
        "num_cached_tokens",
        "num_local_cached_tokens",
        "num_external_cached_tokens",
    )
    missing = [name for name in required if not hasattr(stats, name)]
    if missing:
        raise RuntimeError(
            "installed vLLM lacks cache-breakdown PrefillStats fields: "
            + ",".join(missing)
        )
    return CacheBreakdown(
        prompt_tokens=stats.num_prompt_tokens,
        cached_tokens=stats.num_cached_tokens,
        local_cached_tokens=stats.num_local_cached_tokens,
        external_cached_tokens=stats.num_external_cached_tokens,
    )


def require_prefill_stats_compatibility(stats_type: Any) -> None:
    """Fail decoder startup if the pinned vLLM telemetry seam has changed."""

    fields = getattr(stats_type, "__dataclass_fields__", None)
    if not isinstance(fields, dict):
        raise RuntimeError("installed vLLM PrefillStats is not a dataclass")
    required = {
        "num_prompt_tokens",
        "num_cached_tokens",
        "num_local_cached_tokens",
        "num_external_cached_tokens",
    }
    missing = sorted(required.difference(fields))
    if missing:
        raise RuntimeError(
            "installed vLLM lacks cache-breakdown PrefillStats fields: "
            + ",".join(missing)
        )


def capture_prefill_cache_breakdown(req_state: Any, engine_output: Any) -> None:
    """Capture the first-scheduled prefill split before vLLM drops it."""

    stats = getattr(engine_output, "prefill_stats", None)
    if stats is None or not getattr(req_state, "is_prefilling", False):
        return
    observed = cache_breakdown_from_prefill_stats(stats)
    prior = getattr(req_state, _STATE_BREAKDOWN_ATTR, None)
    if prior is not None and prior != observed:
        raise RuntimeError("vLLM request cache breakdown changed mid-request")
    setattr(req_state, _STATE_BREAKDOWN_ATTR, observed)


def attach_cache_breakdown(req_state: Any, request_output: Any) -> Any:
    """Attach captured telemetry to each output without changing its payload."""

    observed = getattr(req_state, _STATE_BREAKDOWN_ATTR, None)
    if observed is not None:
        setattr(request_output, _OUTPUT_BREAKDOWN_ATTR, observed)
    return request_output


def output_cache_breakdown(request_output: Any) -> CacheBreakdown:
    observed = getattr(request_output, _OUTPUT_BREAKDOWN_ATTR, None)
    if not isinstance(observed, CacheBreakdown):
        raise RuntimeError(
            "TEMPO completion output lacks vLLM cache-breakdown evidence")
    return observed


def inject_cache_breakdown_sse(
    chunk: str, observed: CacheBreakdown | None,
) -> tuple[str, bool, bool]:
    """Inject the split into exactly one stock final completion usage event.

    Returns ``(chunk, injected, done)``.  Non-final token chunks are returned
    byte-for-byte unchanged, keeping request telemetry off the token hot path.
    """

    if not isinstance(chunk, str):
        raise TypeError("vLLM completion SSE chunk must be text")
    if chunk in {"data: [DONE]\n\n", "data:[DONE]\n\n"}:
        return chunk, False, True
    # vLLM serializes the final usage event compactly with an empty choices
    # array.  Avoid parsing ordinary token chunks.
    if '"choices":[]' not in chunk or '"usage":' not in chunk:
        return chunk, False, False
    if not chunk.startswith("data: ") or not chunk.endswith("\n\n"):
        raise RuntimeError("vLLM final usage SSE framing changed")
    try:
        payload = json.loads(chunk[6:-2])
    except json.JSONDecodeError as exc:
        raise RuntimeError("vLLM final usage SSE is not JSON") from exc
    if not isinstance(payload, dict) or payload.get("choices") != []:
        return chunk, False, False
    usage = payload.get("usage")
    if usage is None:
        return chunk, False, False
    if observed is None:
        raise RuntimeError(
            "vLLM final usage preceded cache-breakdown PrefillStats")
    if not isinstance(usage, dict):
        raise RuntimeError("vLLM final usage is not an object")
    if usage.get("prompt_tokens") != observed.prompt_tokens:
        raise RuntimeError(
            "vLLM final usage prompt geometry differs from PrefillStats")
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        raise RuntimeError(
            "vLLM final usage lacks prompt_tokens_details")
    if details.get("cached_tokens") != observed.cached_tokens:
        raise RuntimeError(
            "vLLM final usage total cache count differs from PrefillStats")
    for field in (
        CACHE_BREAKDOWN_SCHEMA_FIELD,
        LOCAL_CACHED_TOKENS_FIELD,
        EXTERNAL_CACHED_TOKENS_FIELD,
    ):
        if field in details:
            raise RuntimeError(
                f"vLLM final usage already contains reserved field {field}")
    details[CACHE_BREAKDOWN_SCHEMA_FIELD] = CACHE_BREAKDOWN_SCHEMA
    details[LOCAL_CACHED_TOKENS_FIELD] = observed.local_cached_tokens
    details[EXTERNAL_CACHED_TOKENS_FIELD] = observed.external_cached_tokens
    return (
        "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n",
        True,
        False,
    )


def apply_cache_read_control(params: Any, vllm_xargs: Any) -> Any:
    if vllm_xargs is None:
        return params
    if not isinstance(vllm_xargs, dict):
        raise ValueError("vllm_xargs must be an object")
    if XARG not in vllm_xargs:
        return params
    raw = vllm_xargs[XARG]
    if type(raw) is not int or raw not in (0, 1):
        raise ValueError(f"{XARG} must be the integer 0 or 1")
    if not hasattr(params, "skip_reading_prefix_cache"):
        raise RuntimeError(
            "installed vLLM lacks skip_reading_prefix_cache")
    params.skip_reading_prefix_cache = bool(raw)
    # CompletionRequest.to_sampling_params also copies every vllm_xargs entry
    # into SamplingParams.extra_args.  Consume TEMPO's reserved control here so
    # it cannot leak into model/plugin execution as an unrelated custom arg.
    extra_args = getattr(params, "extra_args", None)
    if extra_args is not None:
        if not isinstance(extra_args, dict):
            raise RuntimeError("vLLM SamplingParams.extra_args is not an object")
        cleaned = dict(extra_args)
        cleaned.pop(XARG, None)
        params.extra_args = cleaned or None
    return params


def install_patch() -> None:
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.v1.engine.output_processor import OutputProcessor, RequestState
    from vllm.v1.metrics.stats import PrefillStats

    require_prefill_stats_compatibility(PrefillStats)

    current = CompletionRequest.to_sampling_params
    if hasattr(current, _PATCH_MARKER):
        raise RuntimeError("TEMPO cache-control patch installed twice")

    def patched(self, *args, **kwargs):
        params = current(self, *args, **kwargs)
        return apply_cache_read_control(params, self.vllm_xargs)

    setattr(patched, _PATCH_MARKER, current)
    CompletionRequest.to_sampling_params = patched

    current_stats = OutputProcessor._update_stats_from_output
    if hasattr(current_stats, _STATS_PATCH_MARKER):
        raise RuntimeError("TEMPO cache-breakdown stats patch installed twice")

    def patched_stats(self, req_state, engine_core_output, *args, **kwargs):
        capture_prefill_cache_breakdown(req_state, engine_core_output)
        return current_stats(
            self, req_state, engine_core_output, *args, **kwargs)

    setattr(patched_stats, _STATS_PATCH_MARKER, current_stats)
    OutputProcessor._update_stats_from_output = patched_stats

    current_output = RequestState._new_request_output
    if hasattr(current_output, _OUTPUT_PATCH_MARKER):
        raise RuntimeError("TEMPO cache-breakdown output patch installed twice")

    def patched_output(self, *args, **kwargs):
        output = current_output(self, *args, **kwargs)
        return attach_cache_breakdown(self, output)

    setattr(patched_output, _OUTPUT_PATCH_MARKER, current_output)
    RequestState._new_request_output = patched_output

    current_serving = OpenAIServingCompletion.completion_stream_generator
    if hasattr(current_serving, _SERVING_PATCH_MARKER):
        raise RuntimeError("TEMPO cache-breakdown serving patch installed twice")

    async def patched_serving(
        self,
        request,
        engine_inputs,
        result_generator,
        request_id,
        created_time,
        model_name,
        num_prompts,
        tokenizer,
        request_metadata,
    ):
        xargs = getattr(request, "vllm_xargs", None)
        instrument = isinstance(xargs, dict) and XARG in xargs
        if not instrument:
            async for chunk in current_serving(
                self, request, engine_inputs, result_generator, request_id,
                created_time, model_name, num_prompts, tokenizer,
                request_metadata,
            ):
                yield chunk
            return
        if num_prompts != 1 or (getattr(request, "n", 1) or 1) != 1:
            raise RuntimeError(
                "TEMPO cache evidence requires one prompt and one completion")

        observed: CacheBreakdown | None = None

        async def observing_results():
            nonlocal observed
            async for item in result_generator:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise RuntimeError("vLLM completion result shape changed")
                candidate = output_cache_breakdown(item[1])
                if observed is None:
                    observed = candidate
                elif observed != candidate:
                    raise RuntimeError(
                        "vLLM cache breakdown changed across streamed outputs")
                yield item

        injected = False
        async for chunk in current_serving(
            self, request, engine_inputs, observing_results(), request_id,
            created_time, model_name, num_prompts, tokenizer,
            request_metadata,
        ):
            chunk, did_inject, done = inject_cache_breakdown_sse(
                chunk, observed)
            if did_inject:
                if injected:
                    raise RuntimeError(
                        "vLLM emitted duplicate final cache-breakdown usage")
                injected = True
            if done and not injected:
                raise RuntimeError(
                    "vLLM stream ended without cache-breakdown evidence")
            yield chunk
        if not injected:
            raise RuntimeError(
                "vLLM stream ended without cache-breakdown evidence")

    setattr(patched_serving, _SERVING_PATCH_MARKER, current_serving)
    OpenAIServingCompletion.completion_stream_generator = patched_serving


def main() -> int:
    install_patch()
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
