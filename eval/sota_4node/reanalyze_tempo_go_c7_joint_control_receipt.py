#!/usr/bin/env python3
"""Rebuild a C7 arm receipt from completed raw block artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.sota_4node import analyze_tempo_go_c7_joint_control as analyzer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--measured-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-id", default="tempo_go_c7_joint_control")
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    measured_dir = args.measured_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite: {output}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    section = contract["joint_control"]
    artifacts: dict[str, str] = {}
    contracts: dict[str, object] = {}
    for block in section["blocks"]:
        name = str(block["name"])
        raw_path = measured_dir / f"{name}.raw.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        artifacts[name] = str(raw_path)
        contracts[name] = raw["c7_joint_control_contract"]
    bundle: dict[str, object] = {
        "schema": analyzer.BUNDLE_SCHEMA,
        "run_id": args.run_id,
        "arm": args.arm,
        "block_order": [block["name"] for block in section["blocks"]],
        "artifacts": artifacts,
        "contracts": contracts,
        "qualification_contract": str(contract_path),
        "source_workload": str(
            (Path(__file__).resolve().parents[2] /
             section["source_workload"]["path"]).resolve()),
        "performance_claim_allowed": False,
    }
    bundle["analysis"] = analyzer.analyze_arm_bundle(bundle, contract_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(output)
    print(json.dumps({
        "arm": args.arm,
        "hot_slo_good": bundle["analysis"]["hot"]["slo_good_victims"],
        "hot_p99_ms": bundle["analysis"]["hot"]["victim"]["e2e_ms"]["p99"],
        "hot_failures": bundle["analysis"]["hot"]["failures"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
