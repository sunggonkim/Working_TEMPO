import re
import subprocess
import unittest
from pathlib import Path


LAUNCHER = Path(__file__).with_name("run_mooncake_official_2node.sh")


class MooncakeOfficialLauncherStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = LAUNCHER.read_text()

    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

    def test_exactly_one_bounded_step(self):
        self.assertEqual(len(re.findall(r"\bsrun\b", self.source)), 1)
        self.assertRegex(
            self.source,
            r"timeout --signal=TERM --kill-after=3s "
            r"\"\$\{STEP_LIFETIME_SECONDS\}s\" \\\n\s+srun --exact",
        )
        for forbidden in ("sbatch", "salloc", "scancel"):
            self.assertNotIn(forbidden, self.source)

    def test_official_fixed_contract(self):
        required = (
            'EXPECTED_WHEEL_VERSION="0.3.12.post1"',
            "--nodes=2",
            "--ntasks-per-node=1",
            "--gpus-per-task=4",
            "--metadata_server=P2PHANDSHAKE",
            "--protocol=tcp",
            "--use_vram=true",
            "--gpu_id=-1",
            'BLOCK_SIZE_BYTES="33554432"',
            'THREADS="4"',
            'DURATION_SECONDS="5"',
            "--mode=target",
            "--mode=initiator",
        )
        for value in required:
            self.assertIn(value, self.source)

    def test_cuda_runtime_wheel_path_precedes_existing_library_path(self):
        self.assertIn("distribution('nvidia-cuda-runtime-cu12')", self.source)
        self.assertIn("locate_file('nvidia/cuda_runtime/lib')", self.source)
        self.assertIn('"${CUDA_RUNTIME_LIB}/libcudart.so.12"', self.source)
        self.assertIn(
            'export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}'
            '${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"',
            self.source,
        )
        self.assertIn('BENCH_HELP_OUTPUT=$("${BENCH}" --help 2>&1)', self.source)
        self.assertIn("BENCH_HELP_RC", self.source)
        self.assertIn('*"Transfer protocol:"*', self.source)

    def test_manifest_and_result_use_explicit_output_directory(self):
        self.assertIn('echo "usage: $0 RESULT_DIR"', self.source)
        self.assertIn('MANIFEST="${RESULT_DIR}/manifest.json"', self.source)
        self.assertIn('RESULT_JSON="${RESULT_DIR}/result.json"', self.source)
        self.assertIn("parse_mooncake_official.py", self.source)


if __name__ == "__main__":
    unittest.main()
