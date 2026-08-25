#!/usr/bin/env python3
"""Run the canonical stream client with an explicit TEMPO-GO tenant header.

The request ID is the only workload metadata channel accepted by the frozen
JSONL client.  C5 IDs use ``epd-tempo-<tenant>-...``; this wrapper derives the
tenant from that already-recorded ID and adds the application header.  It
does not change the request body, route, transport, or retry policy.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from urllib import error
from urllib import request

from eval.sota_4node import run_tempo_pd_elastic_stream_metrics as canonical
from eval.sota_4node import run_tempo_pd_stream_metrics_forced_v32 as forced


TENANT_HEADER = "X-Tempo-Tenant-ID"
_TENANT = re.compile(
    r"^epd-(?:local|remote|predictor|queue_gpu|network_request_only|"
    r"app_global_only|tempo)-"
    r"(latency|interactive|batch|background)-"
)
_ARMS = {
    "local", "remote", "predictor", "queue_gpu",
    "network_request_only", "app_global_only", "tempo",
}


def _arm() -> str:
    value = os.environ.get("TEMPO_GO_C5_ARM", "tempo")
    if value not in _ARMS:
        raise ValueError(f"TEMPO_GO_C5_ARM is invalid: {value}")
    return value


def _tenant(request_id: str) -> str:
    match = _TENANT.match(request_id)
    if match is None:
        raise ValueError(
            "C5 request ID must be epd-<arm>-<tenant>-...: "
            f"{request_id}")
    return match.group(1)


def _rewrite_warmup_workload(argv: list[str]) -> list[str]:
    """Make perf._lifecycle's generated warmup IDs valid C5 warm IDs.

    Warmup remains outside TEMPO-GO admission but must use the same frontend
    request-ID contract so that completed affinity evidence can be recorded.
    The generated file is beside the original artifact and is never
    overwritten.
    """

    if not any(value.endswith("-warmup") for value in argv):
        return argv
    try:
        workload_index = argv.index("--workload") + 1
    except ValueError as exc:
        raise ValueError("C5 client workload argument is missing") from exc
    generated_warmup = Path(argv[workload_index]).resolve()
    # vllm_lmcache_tempo_pd_perf_node_v1 creates a warmup copy whose request
    # IDs are intentionally rewritten to ``warm-...``.  Cache-contract
    # markers therefore have to be read from the original measured workload,
    # supplied by the native node entry, while the rewritten artifact stays
    # beside the generated warmup and remains run-local.
    source = Path(os.environ.get(
        "TEMPO_GO_C5_SOURCE_WORKLOAD", str(generated_warmup))).resolve()
    arm = _arm()
    target = generated_warmup.with_name(
        f"global-c5-{arm}-warmup-rewritten.jsonl")
    if target.exists():
        raise ValueError(f"refusing to overwrite warmup workload: {target}")
    rows = []
    seen_geometry = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("warmup workload row is not an object")
        # Only the explicitly prepared P_ONLY stream may create a warmup
        # seed.  MISS rows must remain cold; warming every row silently
        # converted the old C5 smoke run into an all-P_ONLY experiment.
        if "-cache-p-only-measured-" not in str(value.get("request_id", "")):
            continue
        geometry = (str(value.get("prompt")), int(value.get("max_tokens", 0)))
        if geometry in seen_geometry:
            continue
        seen_geometry.add(geometry)
        # The canonical frontend derives the replicated-affinity shadow ID by
        # replacing the first ``-item-`` marker.  Keep warmup IDs in that
        # same request-ID contract; otherwise the frontend rejects the
        # request before it reaches either pair router.
        value["request_id"] = f"epd-{arm}-background-warm-c5-item-{len(rows)}"
        rows.append(json.dumps(value, separators=(",", ":")))
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = list(argv)
    result[workload_index] = str(target)
    return result


def _rewrite_measured_arm_workload(argv: list[str]) -> list[str]:
    """Give each native arm a distinct request-ID namespace and evidence."""

    arm = _arm()
    if arm == "tempo" or any(value.endswith("-warmup") for value in argv):
        return argv
    try:
        workload_index = argv.index("--workload") + 1
    except ValueError as exc:
        raise ValueError("C5 client workload argument is missing") from exc
    try:
        output_index = argv.index("--output") + 1
    except ValueError as exc:
        raise ValueError("C5 client output argument is missing") from exc
    source = Path(argv[workload_index]).resolve()
    # The measured source is the immutable, shared manifest workload.  Put the
    # arm-specific rewrite beside that arm's raw output instead of beside the
    # source pool; otherwise a second native campaign cannot reuse the same
    # frozen manifest and would fail on the first existing rewrite artifact.
    target = Path(argv[output_index]).resolve().parent / (
        f"global-c5-{arm}-measured-rewritten.jsonl")
    if target.exists():
        raise ValueError(f"refusing to overwrite arm workload: {target}")
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("C5 workload row is not an object")
        request_id = value.get("request_id")
        if not isinstance(request_id, str) or not request_id.startswith(
            "epd-tempo-"
        ):
            raise ValueError(f"C5 workload row lacks TEMPO source ID: {request_id}")
        value["request_id"] = f"epd-{arm}-{request_id[len('epd-tempo-'):]}"
        rows.append(json.dumps(value, separators=(",", ":")))
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = list(argv)
    result[workload_index] = str(target)
    return result


_ORIGINAL_FORCED = forced.execute_request


def _execute_with_tenant(*args, opener=request.urlopen, **kwargs):
    item = args[0] if args else kwargs.get("item")
    request_id = getattr(item, "request_id", None)
    if not isinstance(request_id, str):
        raise ValueError("C5 forced client request item is missing")
    tenant = _tenant(request_id)

    def tenant_opener(http_request, **call_kwargs):
        headers = dict(http_request.headers)
        headers[TENANT_HEADER] = tenant
        rewritten = request.Request(
            http_request.full_url,
            data=http_request.data,
            headers=headers,
            method=http_request.get_method(),
        )
        try:
            return opener(rewritten, **call_kwargs)
        except error.HTTPError as exc:
            body = exc.read(4096).decode("utf-8", errors="replace")
            raise error.HTTPError(
                exc.url, exc.code, f"{exc.reason}; body={body}",
                exc.headers, exc.fp) from exc

    def execute_for(item_value):
        if args:
            return _ORIGINAL_FORCED(
                item_value, *args[1:], opener=tenant_opener, **kwargs)
        call_kwargs = dict(kwargs)
        call_kwargs["item"] = item_value
        return _ORIGINAL_FORCED(opener=tenant_opener, **call_kwargs)

    # Seed each warm prompt with the exact output geometry carried by that
    # workload row.  In particular, the C1/C2 hot streams use the frozen
    # output=2 anchor while foreground rows retain their own 16/256 geometry.
    # Never replace the row geometry with a hard-coded seed length.
    if "-warm-" in request_id and "-warm-seed-" not in request_id:
        seed_item = replace(
            item,
            request_id=request_id.replace(
                "-warm-", f"-warm-seed-o{item.max_tokens}-", 1),
        )
        seed = execute_for(seed_item)
        if not seed.get("valid"):
            raise RuntimeError(
                "C5 P-only cache seed failed: "
                f"{seed_item.request_id}: {seed.get('error')} "
                f"{seed.get('contract_violations')}"
            )
        result = execute_for(item)
        result["p_only_cache_seed"] = {
            "request_id": seed_item.request_id,
            "valid": True,
            "route": seed["router"]["route"],
            "reason": seed["router"]["reason"],
            "output_tokens": seed_item.max_tokens,
        }
        return result
    return execute_for(item)


def main() -> int:
    original_argv = list(sys.argv)
    rewritten = _rewrite_warmup_workload(original_argv)
    sys.argv[:] = _rewrite_measured_arm_workload(rewritten)
    old_execute = forced.execute_request
    forced.execute_request = _execute_with_tenant
    try:
        # Keep the canonical Elastic router/header/EOF checks, but bypass
        # canonical.main's older P_ONLY wrapper.  _execute_with_tenant above
        # already performs the one exact-geometry seed followed by one probe;
        # stacking both wrappers sends the same warm-seed request ID twice and
        # correctly trips the frontend's duplicate-reservation guard.
        old_schema = canonical._prior.ROUTER_SCHEMA
        canonical._prior.ROUTER_SCHEMA = canonical.ROUTER_SCHEMA
        try:
            return canonical._prior.main()
        finally:
            canonical._prior.ROUTER_SCHEMA = old_schema
    finally:
        forced.execute_request = old_execute
        sys.argv[:] = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
