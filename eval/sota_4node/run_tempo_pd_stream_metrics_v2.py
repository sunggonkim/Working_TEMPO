#!/usr/bin/env python3
"""Streaming metrics v2: count a token carried by the length-finish choice."""

from __future__ import annotations

import json
from typing import Any, BinaryIO, Callable

from eval.sota_4node import run_tempo_pd_stream_metrics_v1 as v1


def _stream_record(
    stream: BinaryIO,
    *,
    dispatch_ns: int,
    run_start_ns: int,
    expected_tokens: int,
    route: str,
    clock_ns: Callable[[], int],
) -> dict[str, Any]:
    arrivals: list[int] = []
    token_values: list[str] = []
    token_proofs: list[str] = []
    text_fragments: list[str] = []
    finish_reasons: list[str] = []
    response_ids: set[str] = set()
    response_models: set[str] = set()
    violations: list[str] = []
    usage: dict[str, int] | None = None
    done_seen = False
    remote = route == "remote_prefill_live_kv"

    for event_ns, payload in v1.base.iter_sse_data(stream, clock_ns=clock_ns):
        if payload == "[DONE]":
            done_seen = True
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise v1.base.ContractError(f"malformed JSON SSE event: {exc}") from exc
        v1._require(isinstance(event, dict), "SSE event must be an object")
        if isinstance(event.get("id"), str):
            response_ids.add(event["id"])
        if isinstance(event.get("model"), str):
            response_models.add(event["model"])
        raw_usage = event.get("usage")
        if raw_usage is not None:
            names = ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(raw_usage, dict) and all(
                type(raw_usage.get(name)) is int and raw_usage[name] >= 0
                for name in names
            ):
                usage = {name: int(raw_usage[name]) for name in names}
            else:
                violations.append("usage_counts_invalid")
        choices = event.get("choices", [])
        if not isinstance(choices, list):
            violations.append("choices_not_list")
            continue
        if not choices:
            continue
        if len(choices) != 1 or not isinstance(choices[0], dict):
            violations.append("not_exactly_one_choice")
            continue
        choice = choices[0]
        text_value = choice.get("text", "")
        if not isinstance(text_value, str):
            violations.append("choice_text_not_string")
            text_value = ""
        text_fragments.append(text_value)
        reason = choice.get("finish_reason")
        if reason is not None:
            if isinstance(reason, str):
                finish_reasons.append(reason)
            else:
                violations.append("finish_reason_not_string")

        logprobs = choice.get("logprobs")
        tokens = logprobs.get("tokens") if isinstance(logprobs, dict) else None
        if isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], str):
            token_values.append(tokens[0])
            token_proofs.append("vllm_logprobs_exactly_one")
            arrivals.append(event_ns - run_start_ns)
        elif remote and not arrivals and text_value and tokens is None:
            token_values.append(text_value)
            token_proofs.append("official_lmcache_proxy_single_prefill_token")
            arrivals.append(event_ns - run_start_ns)
        elif reason is not None and not text_value and tokens is None:
            # A separate finish-only choice carries no token.
            pass
        else:
            violations.append("token_identity_or_cardinality_unproven")

    stream_end_ns = clock_ns()
    if not done_seen:
        violations.append("done_event_missing")
    if finish_reasons != ["length"]:
        violations.append("finish_reason_not_exactly_length")
    if usage is None:
        violations.append("final_usage_missing")
    elif remote:
        if usage["completion_tokens"] not in {expected_tokens - 1, expected_tokens}:
            violations.append("proxy_usage_completion_tokens_mismatch")
    elif usage["completion_tokens"] != expected_tokens:
        violations.append("usage_completion_tokens_mismatch")
    if len(arrivals) != expected_tokens:
        violations.append("requested_completion_tokens_mismatch")
    if remote:
        if not (1 <= len(response_ids) <= 2):
            violations.append("proxy_response_id_count_invalid")
        if token_proofs.count("official_lmcache_proxy_single_prefill_token") != 1:
            violations.append("proxy_first_token_proof_missing")
    elif len(response_ids) != 1:
        violations.append("response_id_not_stable")
    if len(response_models) != 1:
        violations.append("response_model_not_stable")
    dispatch_offset_ns = dispatch_ns - run_start_ns
    if any(value < dispatch_offset_ns for value in arrivals):
        violations.append("token_arrival_precedes_dispatch")
    if any(right < left for left, right in zip(arrivals, arrivals[1:])):
        violations.append("token_arrivals_not_monotonic")
    output_text = "".join(text_fragments)
    return {
        "http_status": 200,
        "dispatch_offset_ns": dispatch_offset_ns,
        "token_arrival_offsets_ns": arrivals,
        "stream_end_offset_ns": stream_end_ns - run_start_ns,
        "output_token_values": token_values,
        "output_token_proofs": token_proofs,
        "output_text": output_text,
        "output_text_sha256": v1.base._sha256_bytes(output_text.encode("utf-8")),
        "finish_reason": finish_reasons[0] if len(finish_reasons) == 1 else None,
        "usage": usage,
        "done_seen": done_seen,
        "response_ids": sorted(response_ids),
        "response_models": sorted(response_models),
        "contract_violations": sorted(set(violations)),
        "error": None,
    }


def main() -> int:
    v1._stream_record = _stream_record
    return v1.main()


if __name__ == "__main__":
    raise SystemExit(main())
