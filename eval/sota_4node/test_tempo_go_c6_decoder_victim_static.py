from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).parent


class C6DecoderVictimStaticTests(unittest.TestCase):
    def test_allocation_wrapper_is_interactive_only_and_unprivileged(self) -> None:
        text = (ROOT / "run_tempo_go_c6_decoder_victim_in_allocation.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("TEMPO_GO_C6_QUALIFICATION_APPROVED", text)
        self.assertIn("--nodes=4 --ntasks=4 --ntasks-per-node=1", text)
        self.assertIn("NCCL_NET", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("scancel", text)
        for forbidden in ("sudo", "udiRoot", "CAP_NET_ADMIN", "--image"):
            self.assertNotIn(forbidden, text)

    def test_node_uses_actual_vllm_lmcache_and_frozen_client(self) -> None:
        text = (ROOT / "vllm_lmcache_tempo_go_c6_qualification_node.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("run_tempo_go_c6_decoder_victim_client", text)
        self.assertIn("canonical._vllm_command", text)
        self.assertIn("canonical._router_command", text)
        self.assertIn("LMCache UCX", text)
        self.assertIn("controller_performance_run_allowed", text)
        self.assertIn('run_id.endswith("-warmup")', text)
        self.assertIn(
            "frozen C6 source workload digest differs during warmup", text
        )

    def test_client_freezes_sixty_second_abba_and_actual_route_pins(self) -> None:
        contract = (ROOT / "tempo_go_c6_qualification_contract_v1.json").read_text(
            encoding="utf-8"
        )
        client = (ROOT / "run_tempo_go_c6_decoder_victim_client.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"phase_duration_ms": 60000.0', contract)
        self.assertIn('"offered_rate_per_s": 22.4', contract)
        self.assertIn("ForegroundArm.REMOTE", client)
        self.assertIn("ContentionState.C1", client)
        self.assertIn("_cold_completion_valid", client)
        self.assertIn("CASSINI_BRIDGE_INTERVAL_S = 5.0", client)
        self.assertIn("_run_child_with_cadenced_endpoint_evidence", client)


if __name__ == "__main__":
    unittest.main()
