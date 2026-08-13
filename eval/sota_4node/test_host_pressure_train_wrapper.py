from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from eval.sota_4node import host_pressure_train_wrapper as wrapper


class _FakeWorker:
    def __init__(self, output: Path, returncode: int = 0) -> None:
        self.output = output
        self.returncode = returncode
        self.terminated = False

    def wait(self, timeout: float | None = None) -> int:
        self.output.write_text("{}\n", encoding="utf-8")
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True


class HostPressureTrainWrapperTests(unittest.TestCase):
    def test_infer_numa_node_uses_explicit_or_local_rank_mapping(self) -> None:
        with mock.patch.dict(os.environ, {"TEMPO_RD_NUMA_NODE": "7"}, clear=True):
            self.assertEqual(wrapper._infer_numa_node(), 7)
        with mock.patch.dict(
            os.environ,
            {"PERLMUTTER_CPU_LDOM_MAP": "3,2,1,0", "SLURM_LOCALID": "1"},
            clear=True,
        ):
            self.assertEqual(wrapper._infer_numa_node(), 2)

    def test_wrapper_starts_only_the_rank_local_helper_and_preserves_train_rc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.py"
            train.write_text("raise SystemExit(0)\n", encoding="utf-8")
            output = root / "pressure" / "rank_0.json"
            worker = _FakeWorker(output)
            with mock.patch.object(wrapper.subprocess, "Popen", return_value=worker) as popen, mock.patch.object(
                wrapper.subprocess, "run", return_value=mock.Mock(returncode=0)
            ) as run:
                self.assertEqual(
                    wrapper.run_with_pressure(
                        train_script=train,
                        pressure_output=output,
                        train_args=["--policy", "none"],
                        rank=0,
                        numa_node=3,
                    ),
                    0,
                )
            command = popen.call_args.args[0]
            self.assertIn("host_pressure_placebo.py", command[1])
            self.assertNotIn("sbatch", command)
            self.assertNotIn("srun", command)
            self.assertEqual(run.call_args.args[0][-2:], ["--policy", "none"])

    def test_worker_failure_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.py"
            train.write_text("raise SystemExit(0)\n", encoding="utf-8")
            output = root / "pressure.json"
            worker = _FakeWorker(output, returncode=9)
            with mock.patch.object(wrapper.subprocess, "Popen", return_value=worker), mock.patch.object(
                wrapper.subprocess, "run", return_value=mock.Mock(returncode=0)
            ):
                self.assertEqual(
                    wrapper.run_with_pressure(
                        train_script=train,
                        pressure_output=output,
                        train_args=["--policy", "none"],
                        rank=0,
                        numa_node=3,
                    ),
                    3,
                )

    def test_rank_placeholder_is_expanded_inside_the_rank_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.py"
            train.write_text("raise SystemExit(0)\n", encoding="utf-8")
            output = root / "pressure-%r.json"
            worker = _FakeWorker(root / "pressure-2.json")
            with mock.patch.object(wrapper.subprocess, "Popen", return_value=worker), mock.patch.object(
                wrapper.subprocess, "run", return_value=mock.Mock(returncode=0)
            ):
                self.assertEqual(
                    wrapper.run_with_pressure(
                        train_script=train,
                        pressure_output=output,
                        train_args=["--policy", "none"],
                        rank=2,
                        numa_node=1,
                    ),
                    0,
                )
            self.assertEqual(worker.output, root / "pressure-2.json")


if __name__ == "__main__":
    unittest.main()
