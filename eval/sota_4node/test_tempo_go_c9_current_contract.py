from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "results/tempo_go_c9_causal_burst_current_source.json"
RUNNER = ROOT / "eval/sota_4node/run_tempo_go_c9_causal_burst_discovery_in_allocation.sh"


def test_current_c9_contract_is_source_bound_and_parent_shape_is_safe() -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert value["claim_boundary"]["discovery_only"] is True
    assert value["claim_boundary"]["performance_claim_allowed"] is False
    assert value["system_under_test"]["node_entry"] == (
        "eval/sota_4node/c9_gate_node_entry.sh"
    )
    provenance = value["provenance"]
    assert provenance["performance_claim_allowed"] is False
    assert provenance["independent_validation_claim_allowed"] is False
    for relative, expected in provenance["source_inventory"].items():
        source = ROOT / relative
        assert source.is_file(), relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, relative

    text = RUNNER.read_text(encoding="utf-8")
    launch = text[text.index("inference_timeout="):text.index(
        'bash "${NODE_ENTRY}"', text.index("inference_timeout="))]
    assert "--gpus-per-node=4" in launch
    assert "--network=job_vni" in launch
    assert "C9 orchestration parent must use --network=no_vni" in RUNNER.read_text(encoding="utf-8")
    assert 'CURRENT_VLLM_STEP_NAME="c9-vllm-${index}-${SLURM_JOB_ID}"' in RUNNER.read_text(encoding="utf-8")
    assert "cancel_owned_steps" in RUNNER.read_text(encoding="utf-8")
    cojob = (ROOT / "eval/sota_4node/run_lmcache_nixl_contention_2node_in_allocation.sh").read_text(encoding="utf-8")
    assert 'TEMPO_GO_SRUN_NETWORK_MODE="${TEMPO_GO_SRUN_NETWORK_MODE:-job_vni}"' in cojob
    assert "--gpus-per-task=4" not in launch
    assert "--overlap --exact" not in launch
    assert "--wait=10" not in launch
    assert "RESULT_ROOT_PORT_HASH" in text
    assert "TEMPO_GO_CROSS_LAYER_NIXL_PORT_BASE=\"$((37000 + index * 16 + pair_index * 8 + RESULT_ROOT_PORT_OFFSET))\"" in text
    assert "write_preperformance_failure" in text


def test_c9_result_root_offset_keeps_all_endpoint_probe_ports_valid() -> None:
    """The final seven-arm slot must stay below the probe port limit."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "RESULT_ROOT_PORT_OFFSET=$((16#${RESULT_ROOT_PORT_HASH} % 100))" in text
    # Candidate L's largest frozen slot is 2660 and _probe_port adds 30000.
    assert 30_000 + 2_660 + 99 < 32_768
