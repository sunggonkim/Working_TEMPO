import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent


class ElasticHarnessStaticTest(unittest.TestCase):
    def test_python_files_parse(self):
        for name in (
            "tempo_pd_elastic_frontend_v445.py",
            "run_tempo_pd_elastic_stream_metrics_v445.py",
            "run_tempo_pd_elastic_balanced_client_v445.py",
            "analyze_tempo_pd_elastic_balanced_v445.py",
            "vllm_lmcache_elastic_pd_node_v445.py",
        ):
            ast.parse((ROOT / name).read_text(), filename=name)

    def test_launcher_is_one_bounded_existing_allocation_step(self):
        text = (ROOT / "run_tempo_pd_elastic_v445_in_allocation.sh").read_text()
        self.assertIn("existing allocation required", text)
        self.assertEqual(text.count("srun "), 1)
        self.assertNotIn("salloc", text)
        self.assertNotIn("sbatch", text)
        self.assertIn("--nodes=4", text)
        self.assertIn("--gpus-per-task=4", text)
        self.assertIn("2640s", text)

    def test_node_wires_profile_router_frontend_client_analyzer(self):
        text = (ROOT / "vllm_lmcache_elastic_pd_node_v445.py").read_text()
        for required in (
            "real_tempo_pd_elastic_profile_v445.json",
            "tempo_pd_elastic_router_v445",
            "tempo_pd_elastic_frontend_v445",
            "run_tempo_pd_elastic_balanced_client_v445",
            "analyze_tempo_pd_elastic_balanced_v445",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
