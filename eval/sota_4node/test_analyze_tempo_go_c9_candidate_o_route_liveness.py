from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT / "eval/sota_4node/analyze_tempo_go_c9_candidate_o_route_liveness.py"
)
SPEC = importlib.util.spec_from_file_location("candidate_o_diagnosis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def raw(decision: dict[str, object]) -> dict[str, object]:
    return {
        "c8_dual_regime_contract": {
            "request_index": {
                "victim": {"role": "victim"},
                "background": {"role": "aggressor"},
            }
        },
        "router_decisions": [
            {"request_id": "victim", "frontend_tempo_go_decision": decision},
            {
                "request_id": "background",
                "frontend_tempo_go_decision": {"kind": "reject"},
            },
        ],
    }


def test_route_failure_evidence_detects_candidate_mechanism_activation() -> None:
    evidence = analyzer.route_failure_evidence(raw({
        "kind": "admit",
        "telemetry_provenance": {
            "0": {
                "route_failures": {"local_count": 0, "remote_count": 1},
            }
        },
        "rejected_candidates": [
            {"reason": "route_failure_quarantine"},
            {"reason": "higher_global_score"},
        ],
    }))
    assert evidence == {
        "all_global_decisions": 2,
        "victim_global_decisions": 1,
        "all_decisions_with_nonzero_route_failure_counter": 1,
        "victim_decisions_with_nonzero_route_failure_counter": 1,
        "all_route_failure_quarantine_rejections": 1,
        "victim_route_failure_quarantine_rejections": 1,
    }


def test_route_failure_evidence_is_zero_when_mechanism_never_activates() -> None:
    evidence = analyzer.route_failure_evidence(raw({
        "kind": "admit",
        "telemetry_provenance": {
            "0": {
                "route_failures": {"local_count": 0, "remote_count": 0},
            }
        },
        "rejected_candidates": [{"reason": "deadline"}],
    }))
    assert evidence["victim_decisions_with_nonzero_route_failure_counter"] == 0
    assert evidence["victim_route_failure_quarantine_rejections"] == 0
