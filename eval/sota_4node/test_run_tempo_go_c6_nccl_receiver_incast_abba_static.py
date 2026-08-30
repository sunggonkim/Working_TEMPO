from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parent
SCRIPT = ROOT / "run_tempo_go_c6_nccl_receiver_incast_abba_in_allocation.sh"
CONTRACT = ROOT / "tempo_go_c6_nccl_receiver_incast_contract_v1.json"


class C6NCCLReceiverIncastABBAStaticTests(unittest.TestCase):
    def test_contract_freezes_sustained_collectives_and_receiver_incast(self) -> None:
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(
            value["schema"], "tempo-go-c6-nccl-receiver-incast-contract-v1"
        )
        abba = value["nccl_victim_abba"]
        self.assertEqual(abba["traffic_pattern"], "incast_4to1")
        self.assertEqual(abba["parameters"]["token_iters"], 4096)
        self.assertEqual(abba["minimum_active_duration_s"], 30)
        self.assertEqual(
            [row["background_mode"] for row in abba["arms"]],
            ["nccl_only", "nixl_ucx", "nixl_ucx", "nccl_only"],
        )

    def test_wrapper_is_interactive_only_unprivileged_and_contract_bound(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("TEMPO_GO_C6_QUALIFICATION_APPROVED", text)
        self.assertIn("TEMPO_GO_CROSS_LAYER_TRAFFIC_PATTERN", text)
        self.assertIn("tempo_go_c6_nccl_receiver_incast_contract_v1.json", text)
        self.assertIn("analyze_tempo_go_c6_nccl_victim_abba", text)
        for forbidden in ("sbatch", "scancel", "sudo", "udiRoot", "--image"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
