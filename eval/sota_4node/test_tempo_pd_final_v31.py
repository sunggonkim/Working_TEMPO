import unittest
from pathlib import Path

from eval.sota_4node import tempo_pd_evidence_fail_local_router_v28 as router
from eval.sota_4node import tempo_pd_router_v1 as base


class FinalV31Tests(unittest.TestCase):
    def test_every_request_fails_local_with_bound_evidence(self):
        config = base.RouterConfig(
            mode=base.RouterMode.TEMPO_AUTO, local_url="http://l",
            remote_url="http://r", tokenizer_url="http://t",
            served_model_name="m", model_id="m", model_revision="x",
            topology_id="t", remote_backend="b", classifier_version="c",
            decoder_load_bucket="high", kv_bytes_per_token=1,
        )
        core = router.EvidenceFailLocalCore(config)
        rows = [core.decide(request_id=str(i), prompt_tokens=1220,
                            output_tokens=32) for i in range(4)]
        self.assertEqual({row.route.value for row in rows},
                         {"decoder_local_recompute_or_cache"})
        self.assertEqual({row.reason for row in rows},
                         {"fail_local_remote_correctness_or_5ms_gate_unproven"})
        self.assertEqual({row.manifest_id for row in rows}, {router.EVIDENCE_ID})

    def test_final_launcher_one_step(self):
        text = Path(__file__).with_name(
            "run_tempo_pd_final_v30_in_allocation.sh"
        ).read_text()
        self.assertEqual(text.count("srun "), 1)
        self.assertIn("analyze_tempo_pd_final_v29", text)
        self.assertNotIn("salloc", text)


if __name__ == "__main__":
    unittest.main()
