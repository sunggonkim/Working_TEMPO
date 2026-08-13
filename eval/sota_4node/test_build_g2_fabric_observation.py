import json
import tempfile
import unittest
from pathlib import Path

try:
    from .build_g2_fabric_observation import build_observation
except ImportError:  # direct unittest discovery from eval/sota_4node
    from build_g2_fabric_observation import build_observation


class G2FabricObservationTests(unittest.TestCase):
    def _fixture(self) -> Path:
        root = Path(tempfile.mkdtemp())
        hosts = ["n0"] * 4 + ["n1"] * 4
        for rank, host in enumerate(hosts):
            (root / f"placement_rank{rank}.env").write_text(
                f"rank={rank}\nlocal_rank={rank % 4}\nhost={host}\n"
            )
        policy = root / "tempo_v4"
        policy.mkdir()
        header = "rank,step,phase_index,phase_signature,collective,ready_corrected_ns,completion_callback_unix_ns,gpu_ms,tensor_bytes,output_tensor_bytes\n"
        rows = []
        for rank, host in enumerate(hosts):
            rows.append(f"{rank},16,0,0:all_gather_into_tensor:1024:8192,all_gather_into_tensor,{1000+rank*10},{2000+rank*10},1.0,1024,8192\n")
        for rank in range(8):
            (policy / f"collectives_rank{rank}.csv").write_text(header + rows[rank])
        (policy / "train_nccl_n0.log").write_text("NCCL INFO GPU Direct RDMA Enabled for GPU 0 / HCA 3\n")
        return root

    def test_keeps_route_witness_noncausal(self):
        result = build_observation(self._fixture(), "tempo_v4")
        self.assertFalse(result["promotion_eligible"])
        self.assertFalse(result["counter_contract"]["causal_claim_allowed"])
        self.assertEqual(len(result["collective_observations"]), 1)
        self.assertEqual(result["collective_observations"][0]["rank_count"], 8)
        self.assertEqual(result["route_witness"]["counts"]["gdr_gpu_originated"], 1)

    def test_rejects_incomplete_placement(self):
        root = self._fixture()
        (root / "placement_rank7.env").unlink()
        with self.assertRaises(ValueError):
            build_observation(root, "tempo_v4")


if __name__ == "__main__":
    unittest.main()
