#!/usr/bin/env python3
"""Canonical stream client for explicit cache preparation/measurement.

Unlike the ordinary warm client, this entrypoint never inserts an implicit
P-side seed.  The caller must declare either ``decoder_prepare`` or
``measured`` and bind the workload to a cache-protocol plan/evidence artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from eval.sota_4node import run_tempo_pd_elastic_stream_metrics_v445 as _prior
from tempo.pd_cache_state_protocol import SCHEMA as PLAN_SCHEMA


ROUTER_SCHEMA = "tempo-elastic-pd-router-canonical"
PHASE_ENV = "TEMPO_PD_CACHE_PROTOCOL_PHASE"
PLAN_ENV = "TEMPO_PD_CACHE_PROTOCOL_PLAN"
EVIDENCE_ENV = "TEMPO_PD_CACHE_PROTOCOL_EVIDENCE"
START_MARKER_ENV = "TEMPO_PD_STREAM_RUN_START_FILE"
START_MARKER_SCHEMA = "tempo-pd-stream-run-start-v1"
RUNTIME_EVIDENCE_SCHEMA = "tempo-pd-cache-state-runtime-evidence-v1"
_CACHE_MARKERS = (
    "-cache-miss-measured-",
    "-cache-p-only-measured-",
    "-cache-d-only-measured-",
    "-cache-both-measured-",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rows(path: Path) -> list[dict[str, object]]:
    _require(path.is_file(), "cache-protocol workload is missing")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(bool(rows), "cache-protocol workload is empty")
    _require(all(isinstance(row, dict) for row in rows),
             "cache-protocol workload row is not an object")
    ids = [row.get("request_id") for row in rows]
    _require(
        all(isinstance(value, str) and value for value in ids)
        and len(ids) == len(set(ids)),
        "cache-protocol workload IDs are invalid",
    )
    return rows


def _load_plan(path: Path) -> dict[str, object]:
    _require(path.is_file(), "cache-protocol plan is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("schema") == PLAN_SCHEMA,
             "cache-protocol plan schema differs")
    _require(value.get("request_id_labels_establish_residency") is False,
             "cache-protocol plan permits label-derived residency")
    return value


def validate_invocation(
    *, phase: str, rows: list[dict[str, object]],
    plan: dict[str, object], evidence: dict[str, object] | None,
) -> None:
    _require(phase in {"decoder_prepare", "measured"},
             "cache-protocol phase must be decoder_prepare or measured")
    request_ids = [str(row["request_id"]) for row in rows]
    if phase == "decoder_prepare":
        expected = plan.get("decoder_prepare_request_ids")
        _require(isinstance(expected, list) and request_ids == expected,
                 "decoder preparation workload differs from frozen plan")
        _require(len(rows) % 2 == 0,
                 "decoder preparation must contain seed/probe pairs")
        for index in range(0, len(rows), 2):
            seed = rows[index]
            probe = rows[index + 1]
            _require("-cache-d-seed-" in str(seed["request_id"]),
                     "decoder preparation pair lacks a seed")
            _require("-cache-d-probe-" in str(probe["request_id"]),
                     "decoder preparation pair lacks a probe")
            _require(seed.get("prompt") == probe.get("prompt"),
                     "decoder seed/probe prompts differ")
            _require(seed.get("max_tokens") == 2
                     and probe.get("max_tokens") == 2,
                     "decoder preparation must use two output tokens")
        _require(evidence is None,
                 "decoder preparation cannot consume ready evidence")
        return

    _require(evidence is not None, "measured phase requires runtime evidence")
    _require(
        evidence.get("schema") == RUNTIME_EVIDENCE_SCHEMA
        and evidence.get("ready_for_measurement") is True,
        "cache runtime evidence is not measurement-ready",
    )
    _require(
        evidence.get("plan_fingerprint_sha256")
        == plan.get("fingerprint_sha256"),
        "cache runtime evidence is bound to another plan",
    )
    planned = {
        item.get("request_id") for item in plan.get("items", [])
        if isinstance(item, dict)
    }
    _require(set(request_ids).issubset(planned),
             "measured workload contains a request outside the plan")
    for request_id in request_ids:
        _require(sum(marker in request_id for marker in _CACHE_MARKERS) == 1,
                 "measured request lacks one exact cache-state marker")
        _require("-warm-" not in request_id,
                 "measured cache request is incorrectly marked warm")


def _path_from_env(name: str) -> Path:
    raw = os.environ.get(name)
    _require(isinstance(raw, str) and raw, f"{name} is required")
    path = Path(raw)
    _require(path.is_absolute(), f"{name} must be absolute")
    return path


def _measurement_start_observer(path: Path):
    _require(path.is_absolute(), "measurement start marker must be absolute")
    _require(path.parent.is_dir(),
             "measurement start marker parent is missing")
    _require(not path.exists(),
             "measurement start marker already exists")
    published = False

    def observe(start_ns: int) -> None:
        nonlocal published
        _require(type(start_ns) is int and start_ns > 0,
                 "measurement start clock is invalid")
        _require(not published,
                 "measurement start marker was published twice")
        temporary = path.with_name(
            f".{path.name}.tmp-{os.getpid()}")
        _require(not temporary.exists(),
                 "measurement start temporary marker already exists")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump({
                "schema": START_MARKER_SCHEMA,
                "clock": "client time.perf_counter_ns",
                "run_start_ns": start_ns,
                "publisher_pid": os.getpid(),
            }, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        published = True

    return observe


def main() -> int:
    phase = os.environ.get(PHASE_ENV, "")
    plan_path = _path_from_env(PLAN_ENV)
    _require("--workload" in sys.argv, "--workload is required")
    workload = Path(sys.argv[sys.argv.index("--workload") + 1]).resolve()
    rows = _load_rows(workload)
    plan = _load_plan(plan_path)
    evidence = None
    if phase == "measured":
        evidence_path = _path_from_env(EVIDENCE_ENV)
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        _require(isinstance(evidence, dict),
                 "cache runtime evidence is not an object")
        _require(evidence.get("plan_sha256") == _sha256(plan_path),
                 "cache runtime evidence plan digest differs")
    elif EVIDENCE_ENV in os.environ:
        raise ValueError(
            "decoder preparation must not receive measurement evidence")
    start_marker = None
    if START_MARKER_ENV in os.environ:
        _require(phase == "measured",
                 "only measured cache-protocol runs may publish a start marker")
        start_marker = _path_from_env(START_MARKER_ENV)
        _require(start_marker.parent.is_dir(),
                 "measurement start marker parent is missing")
        _require(not start_marker.exists(),
                 "measurement start marker already exists")
    validate_invocation(
        phase=phase, rows=rows, plan=plan, evidence=evidence)

    old_schema = _prior.ROUTER_SCHEMA
    old_start_observer = _prior.v1.RUN_START_OBSERVER
    if start_marker is not None:
        _require(old_start_observer is None,
                 "stream run-start observer is already installed")
        _prior.v1.RUN_START_OBSERVER = _measurement_start_observer(
            start_marker)
    _prior.ROUTER_SCHEMA = ROUTER_SCHEMA
    try:
        return _prior.main()
    finally:
        _prior.ROUTER_SCHEMA = old_schema
        _prior.v1.RUN_START_OBSERVER = old_start_observer


if __name__ == "__main__":
    raise SystemExit(main())
