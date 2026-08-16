from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.sota_4node.prepare_foreground_path import prepare_foreground_path
from tempo.resource_domain import ResourceDomain, domain_contract


def _record() -> dict[str, object]:
    domains = [ResourceDomain.GPU_LOCAL.value, ResourceDomain.NVLINK_P2P.value]
    counters: dict[str, list[dict[str, object]]] = {}
    for index, name in enumerate(domains):
        contract = domain_contract(ResourceDomain(name))
        counters[name] = [
            {
                "domain": name,
                "sample_id": "start",
                "source": "fixture-counter",
                "timestamp_ns": 100 + index,
                "cumulative_bytes": 0,
                "cumulative_busy_ns": 0,
                "support": "supported",
                "scope": "rank" if name == "gpu_local" else "pair",
                "scope_id": "rank0" if name == "gpu_local" else "0-1",
                "intervention_id": "fg_only",
            },
            {
                "domain": name,
                "sample_id": "end",
                "source": "fixture-counter",
                "timestamp_ns": 200 + index,
                "cumulative_bytes": 4096 * (index + 1),
                "cumulative_busy_ns": 1000 * (index + 1),
                "support": "supported",
                "scope": "rank" if name == "gpu_local" else "pair",
                "scope_id": "rank0" if name == "gpu_local" else "0-1",
                "intervention_id": "fg_only",
            },
        ]
    return {
        "domains": domains,
        "path_status": {name: "observed" for name in domains},
        "counter_support": {name: "supported" for name in domains},
        "path_evidence": {name: domain_contract(ResourceDomain(name)).path_evidence for name in domains},
        "counter_family": {name: domain_contract(ResourceDomain(name)).counter_family for name in domains},
        "counters": counters,
    }


class PrepareForegroundPathTests(unittest.TestCase):
    def test_canonical_publish_and_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw.json"
            output = root / "foreground_path.json"
            source.write_text(json.dumps(_record(), indent=2) + "\n", encoding="utf-8")
            result = prepare_foreground_path(source, output)
            self.assertEqual(result["domains"], ["gpu_local", "nvlink_p2p"])
            encoded = output.read_text(encoding="utf-8")
            self.assertEqual(encoded, json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            self.assertEqual(len((output.with_suffix(".json.sha256")).read_text().strip()), 64)

    def test_topology_only_record_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw.json"
            bad = _record()
            bad["counters"] = {}
            source.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_foreground_path(source, root / "foreground_path.json")

    def test_intervention_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw.json"
            bad = _record()
            bad["counters"]["gpu_local"][0]["intervention_id"] = "d2h_only"
            source.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_foreground_path(source, root / "foreground_path.json")


if __name__ == "__main__":
    unittest.main()
