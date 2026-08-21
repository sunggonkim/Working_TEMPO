"""Strict per-request decoder cache-source evidence from vLLM SSE usage.

Stock vLLM reports only the sum of local APC and external KV-transfer hits in
``cached_tokens``.  TEMPO's decoder launcher carries vLLM V1's already-existing
``PrefillStats`` split into the final usage event.  This parser requires that
canonical split and observes the upstream byte stream inside the router, so
cache residency never depends on a client label or a planned reuse schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
import json


EVIDENCE_SOURCE = "vllm_v1_prefill_stats_via_final_stream_usage"
CACHE_BREAKDOWN_SCHEMA = "tempo-vllm-prefill-cache-breakdown-v1"
CACHE_BREAKDOWN_SCHEMA_FIELD = "tempo_cache_breakdown_schema"
LOCAL_CACHED_TOKENS_FIELD = "tempo_local_cached_tokens"
EXTERNAL_CACHED_TOKENS_FIELD = "tempo_external_cached_tokens"
DEFAULT_BLOCK_SIZE = 16
MAX_SSE_LINE_BYTES = 8 * 1024 * 1024


def full_prefix_hit_tokens(
    prompt_tokens: int, *, block_size: int = DEFAULT_BLOCK_SIZE,
) -> int:
    """Return vLLM's exact maximum APC hit for one decoder-only prompt.

    vLLM deliberately recomputes the last prompt token to obtain logits and
    prefix-cache lookup is block aligned.  Therefore a full hit is the largest
    complete block not exceeding ``prompt_tokens - 1``.
    """

    if type(prompt_tokens) is not int or prompt_tokens < 2:
        raise ValueError("prompt_tokens must be an integer >= 2")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    return ((prompt_tokens - 1) // block_size) * block_size


@dataclass(frozen=True)
class DecoderCacheEvidence:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    local_cached_tokens: int
    external_cached_tokens: int
    source: str = EVIDENCE_SOURCE

    def __post_init__(self) -> None:
        for name in (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "cached_tokens", "local_cached_tokens", "external_cached_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("usage total_tokens is inconsistent")
        if self.cached_tokens > self.prompt_tokens:
            raise ValueError("cached_tokens exceeds prompt_tokens")
        if self.cached_tokens != (
            self.local_cached_tokens + self.external_cached_tokens
        ):
            raise ValueError("cached-token source breakdown is inconsistent")
        if self.source != EVIDENCE_SOURCE:
            raise ValueError("decoder cache evidence source is not canonical")


class VLLMDecoderCacheSSEParser:
    """Incrementally parse one upstream SSE stream and require exact usage."""

    def __init__(self, *, max_line_bytes: int = MAX_SSE_LINE_BYTES) -> None:
        if type(max_line_bytes) is not int or max_line_bytes <= 0:
            raise ValueError("max_line_bytes must be a positive integer")
        self._max_line_bytes = max_line_bytes
        self._line_buffer = bytearray()
        self._event_data: list[bytes] = []
        self._evidence: DecoderCacheEvidence | None = None
        self._done_seen = False
        self._finalized = False

    def feed(self, chunk: bytes) -> None:
        if self._finalized:
            raise ValueError("decoder cache evidence parser is finalized")
        if not isinstance(chunk, bytes):
            raise TypeError("SSE chunk must be bytes")
        if not chunk:
            return
        self._line_buffer.extend(chunk)
        if len(self._line_buffer) > self._max_line_bytes:
            raise ValueError("upstream SSE line exceeds evidence bound")
        while True:
            newline = self._line_buffer.find(b"\n")
            if newline < 0:
                break
            line = bytes(self._line_buffer[:newline])
            del self._line_buffer[:newline + 1]
            if line.endswith(b"\r"):
                line = line[:-1]
            self._line(line)
            if len(self._line_buffer) > self._max_line_bytes:
                raise ValueError("upstream SSE line exceeds evidence bound")

    def _line(self, line: bytes) -> None:
        if not line:
            self._dispatch()
            return
        if line.startswith(b":"):
            return
        field, separator, value = line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            self._event_data.append(value)

    def _dispatch(self) -> None:
        if not self._event_data:
            return
        raw = b"\n".join(self._event_data)
        self._event_data.clear()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("upstream SSE data is not UTF-8") from exc
        if text == "[DONE]":
            if self._done_seen:
                raise ValueError("upstream SSE contains duplicate DONE events")
            self._done_seen = True
            return
        if self._done_seen:
            raise ValueError("upstream SSE data follows DONE")
        try:
            event = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("upstream SSE data is not JSON") from exc
        if not isinstance(event, dict):
            raise ValueError("upstream SSE event must be an object")
        usage = event.get("usage")
        if usage is None:
            return
        if not isinstance(usage, dict):
            raise ValueError("upstream SSE usage must be an object")
        names = ("prompt_tokens", "completion_tokens", "total_tokens")
        if any(type(usage.get(name)) is not int or usage[name] < 0
               for name in names):
            raise ValueError("upstream SSE usage counts are invalid")
        details = usage.get("prompt_tokens_details")
        if details is None:
            return
        if not isinstance(details, dict):
            raise ValueError("prompt_tokens_details must be an object")
        cached_tokens = details.get("cached_tokens")
        if type(cached_tokens) is not int or cached_tokens < 0:
            raise ValueError("decoder cached_tokens evidence is invalid")
        if details.get(CACHE_BREAKDOWN_SCHEMA_FIELD) != CACHE_BREAKDOWN_SCHEMA:
            raise ValueError("decoder cache-breakdown schema is missing")
        local_cached_tokens = details.get(LOCAL_CACHED_TOKENS_FIELD)
        external_cached_tokens = details.get(EXTERNAL_CACHED_TOKENS_FIELD)
        if (
            type(local_cached_tokens) is not int
            or local_cached_tokens < 0
            or type(external_cached_tokens) is not int
            or external_cached_tokens < 0
        ):
            raise ValueError("decoder cache-source counts are invalid")
        if event.get("choices") != []:
            raise ValueError("decoder cache usage evidence is not a final chunk")
        if self._evidence is not None:
            raise ValueError("decoder cache usage evidence is duplicated")
        self._evidence = DecoderCacheEvidence(
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            total_tokens=usage["total_tokens"],
            cached_tokens=cached_tokens,
            local_cached_tokens=local_cached_tokens,
            external_cached_tokens=external_cached_tokens,
        )

    def finish(self, *, expected_prompt_tokens: int) -> DecoderCacheEvidence:
        if self._finalized:
            raise ValueError("decoder cache evidence parser finalized twice")
        self._finalized = True
        if self._line_buffer or self._event_data:
            raise ValueError("upstream SSE ended with an incomplete event")
        if not self._done_seen:
            raise ValueError("upstream SSE is missing the DONE event")
        if self._evidence is None:
            raise ValueError(
                "upstream SSE is missing prompt-token cache details")
        if self._evidence.prompt_tokens != expected_prompt_tokens:
            raise ValueError(
                "decoder usage prompt_tokens differs from routed geometry")
        return self._evidence


__all__ = [
    "CACHE_BREAKDOWN_SCHEMA",
    "CACHE_BREAKDOWN_SCHEMA_FIELD",
    "DEFAULT_BLOCK_SIZE",
    "DecoderCacheEvidence",
    "EVIDENCE_SOURCE",
    "EXTERNAL_CACHED_TOKENS_FIELD",
    "LOCAL_CACHED_TOKENS_FIELD",
    "MAX_SSE_LINE_BYTES",
    "VLLMDecoderCacheSSEParser",
    "full_prefix_hit_tokens",
]
