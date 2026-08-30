from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "results/tempo_go_c8_independent_validation_contract_v10_c8v49.json"
RUNNER = ROOT / "eval/sota_4node/run_tempo_go_c8_independent_validation_in_allocation.sh"


def test_current_heldout_contract_is_source_bound_and_network_explicit() -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    heldout = value["independent_validation"]
    assert heldout["fresh_allocation_required"] is True
    assert heldout["one_shot_no_retry"] is True
    assert value["claim_boundary"]["performance_claim_allowed"] is False
    assert value["claim_boundary"]["independent_validation_claim_allowed"] is False
    for relative, expected in value["source_inventory"].items():
        source = ROOT / relative
        assert source.is_file(), relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, relative

    text = RUNNER.read_text(encoding="utf-8")
    assert "tempo_go_c8_independent_validation_contract_v10_c8v49.json" in text
    assert '"Network=job_vni"' in text
    assert "--network=job_vni" in text
