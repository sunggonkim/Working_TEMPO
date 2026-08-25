#!/usr/bin/env python3
"""Run the frozen C9 held-out workload through one paper baseline policy."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from eval.sota_4node import run_tempo_go_c8_independent_validation_client as heldout
from tempo.pd_paper_baselines import KAIROS_X512, NETKV, POLICIES, POLICY_ENV


SCHEMA = heldout.SCHEMA
CONTRACT_SCHEMA = heldout.CONTRACT_SCHEMA
CONTRACT_ENV = heldout.CONTRACT_ENV
ARM_ENV = heldout.ARM_ENV
INDEPENDENT_SCHEMA = heldout.INDEPENDENT_SCHEMA
analyzer = heldout.frozen.analyzer
_decoder_contract = heldout._decoder_contract
_ORIGINAL_C7_AUGMENT_BLOCK = heldout.frozen.c7._augment_block


def _validated_cold_global_edge(
    decision: dict[str, object],
) -> tuple[int, int]:
    """Validate a cold global edge without treating D placement as P source."""

    c7 = heldout.frozen.c7
    route = decision.get("route")
    source = decision.get("tempo_go_global_commit_prefill_index")
    decoder = decision.get("tempo_go_global_commit_decoder_index")
    pair = decision.get("tempo_go_global_commit_pair_index")
    destination = decision.get("frontend_pair_index")
    if not (
        route in {c7.LOCAL_ROUTE, c7.REMOTE_ROUTE}
        and decision.get("tempo_go_global_commit_applied") is True
        and all(type(value) is int for value in (
            source, decoder, pair, destination))
        and pair == destination == decoder
    ):
        raise ValueError("C10 cold global destination commitment differs")
    assert type(source) is int and type(decoder) is int
    actual_decoder = decision.get(
        "local_decoder_index"
        if route == c7.LOCAL_ROUTE else "remote_decoder_index")
    if actual_decoder != decoder:
        raise ValueError("C10 cold global decoder actuation differs")
    if decision.get("tempo_go_global_commit_edge_id") != c7._canonical_edge(
        str(route), source, decoder,
    ):
        raise ValueError("C10 cold global edge commitment differs")
    if route == c7.LOCAL_ROUTE and source != decoder:
        raise ValueError("C10 local cold global edge crosses a pair")
    return source, decoder


def _mesh_aware_c7_augment_block(
    raw_path: Path, *, spec: dict[str, object], section: dict[str, object],
    schedule_sha256: str, request_index: dict[str, dict[str, object]],
    endpoint_evidence: dict[str, object],
) -> dict[str, object]:
    """Adapt only the frozen validator's prefill-source interpretation.

    C7 predates cross-pair global placement and uses ``frontend_pair_index``
    (the committed decoder destination) as the prefill source.  C8 already
    fixed this for its P_ONLY block.  Validate the native global commitment
    here, then feed the old validator a temporary source-resolved view.  The
    measured raw artifact retains every original field and is never rewritten
    with the compatibility view.
    """

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    patched = json.loads(json.dumps(raw))
    rows = {row["request_id"]: row for row in raw["requests"]}
    decisions = {
        row["request_id"]: row for row in patched["router_decisions"]}
    cross_pair_edges = 0
    validated_edges = 0
    for request_id, metadata in request_index.items():
        if metadata.get("role") != "victim":
            continue
        row = rows[request_id]
        if row.get("terminal_kind") in {
            "global_reject", "service_lane_failure",
        }:
            continue
        decision = decisions[request_id]
        source, decoder = _validated_cold_global_edge(decision)
        validated_edges += 1
        if source != decoder:
            cross_pair_edges += 1
        # Compatibility view only: C7 expects this field to be P, while the
        # live frontend correctly records the committed D destination here.
        decision["frontend_pair_index"] = source

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".c10-mesh-validator-",
            suffix=".json", dir=raw_path.parent, delete=False,
        ) as handle:
            json.dump(patched, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary_path = Path(handle.name)
        contract = _ORIGINAL_C7_AUGMENT_BLOCK(
            temporary_path,
            spec=spec,
            section=section,
            schedule_sha256=schedule_sha256,
            request_index=request_index,
            endpoint_evidence=endpoint_evidence,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    contract = dict(contract)
    contract["global_edge_validation"] = {
        "schema": "tempo-go-c10-mesh-aware-cold-edge-validation-v1",
        "source_field": "tempo_go_global_commit_prefill_index",
        "destination_field": "frontend_pair_index",
        "validated_edges": validated_edges,
        "cross_pair_edges": cross_pair_edges,
        "workload_or_policy_modified": False,
    }
    raw["c7_joint_control_contract"] = contract
    raw["endpoint_evidence"] = endpoint_evidence
    raw_path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


heldout.frozen.c7._augment_block = _mesh_aware_c7_augment_block


def _policy() -> str:
    value = os.environ.get(POLICY_ENV, "")
    if value not in POLICIES:
        raise ValueError(f"{POLICY_ENV} must select {sorted(POLICIES)}")
    return value


def configure_node_environment(**kwargs) -> None:
    if os.environ.get(ARM_ENV) != "app_global_only":
        raise ValueError("paper SOTA carrier wire arm must be app_global_only")
    heldout.configure_node_environment(**kwargs)
    policy = _policy()
    # NetKV receives network-oracle signals but not TEMPO actuation.  Kairos
    # receives only application/queue state, matching its paper inputs.
    os.environ["TEMPO_GO_ABLATION"] = (
        "disabled" if policy == NETKV else "app_global_only")
    os.environ["TEMPO_VLLM_DECODER_MAX_NUM_BATCHED_TOKENS"] = "32768"
    if policy == KAIROS_X512:
        os.environ["TEMPO_PAPER_BASELINE_DECODER_CHUNK_TOKENS"] = "512"
    else:
        os.environ.pop("TEMPO_PAPER_BASELINE_DECODER_CHUNK_TOKENS", None)


def main() -> int:
    _policy()
    return heldout.main()


if __name__ == "__main__":
    raise SystemExit(main())
