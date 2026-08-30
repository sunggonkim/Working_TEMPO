#!/usr/bin/env python3
"""Token-ID-capable frontend entrypoint for trace-driven TEMPO experiments.

The frozen C9 frontend accepts private text prompts and obtains token IDs from
the colocated vLLM tokenizer endpoint.  Mooncake releases only token lengths
and prefix-block hashes, so the real-trace harness sends deterministic token
IDs directly.  This entrypoint extends only the request-shape/tokenization
seam and delegates the complete routing, admission, telemetry, failure, and
stream lifecycle to the canonical frontend.

It is deliberately a separate module: importing it does not alter the source
identity of the frozen C9/C10 or §73 contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import httpx

from eval.sota_4node import tempo_pd_elastic_frontend as canonical


SCHEMA = "tempo-pd-real-trace-frontend-token-ids-v1"
_TEXT_COMPLETION_SHAPE = canonical._completion_shape
_TEXT_TOKENIZE_PROMPT = canonical._tokenize_prompt


def _token_prompt(value: Any, *, context: str) -> list[int] | None:
    prompt = value.get("prompt") if isinstance(value, dict) else None
    if isinstance(prompt, str):
        return None
    if (
        not isinstance(prompt, list)
        or not prompt
        or any(type(token) is not int or token < 0 for token in prompt)
    ):
        raise ValueError(f"{context} prompt must be nonempty text or token IDs")
    return list(prompt)


def _token_prompt_key(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(
        token_ids, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _completion_shape(payload: bytes) -> tuple[int, str]:
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("completion body must be JSON") from exc
    token_ids = _token_prompt(value, context="completion")
    if token_ids is None:
        return _TEXT_COMPLETION_SHAPE(payload)
    output_tokens = value.get("max_tokens") if isinstance(value, dict) else None
    if type(output_tokens) is not int or output_tokens <= 0:
        raise ValueError("completion max_tokens must be a positive integer")
    return output_tokens, _token_prompt_key(token_ids)


def _decode_tokens(payload: bytes) -> int:
    return _completion_shape(payload)[0]


async def _tokenize_prompt(
    client: httpx.AsyncClient,
    payload: bytes,
) -> tuple[int, float, str | None, str]:
    try:
        value = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("completion body must be JSON") from exc
    token_ids = _token_prompt(value, context="global completion")
    if token_ids is None:
        return await _TEXT_TOKENIZE_PROMPT(client, payload)
    prompt_key = _token_prompt_key(token_ids)
    complete_tokens = (
        len(token_ids)
        // canonical.CACHE_CHUNK_GROUP_SIZE
        * canonical.CACHE_CHUNK_GROUP_SIZE
    )
    cache_group_key = None
    if complete_tokens:
        group_payload = json.dumps(
            {
                "schema": "tempo-cache-chunk-group-v1",
                "chunk_size": canonical.CACHE_CHUNK_GROUP_SIZE,
                "complete_tokens": token_ids[:complete_tokens],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        cache_group_key = hashlib.sha256(group_payload).hexdigest()
    # Token IDs are already supplied by the source-bound materializer; no
    # cross-process tokenizer call or cross-host clock subtraction occurs.
    return len(token_ids), 0.0, cache_group_key, prompt_key


def install_token_id_seam() -> None:
    canonical._completion_shape = _completion_shape
    canonical._decode_tokens = _decode_tokens
    canonical._tokenize_prompt = _tokenize_prompt


def build_app(pair_urls: list[str]):
    install_token_id_seam()
    app = canonical.build_app(pair_urls)
    app.state.tempo_real_trace_frontend_schema = SCHEMA
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pair-url", action="append", required=True)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        build_app(args.pair_url),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
