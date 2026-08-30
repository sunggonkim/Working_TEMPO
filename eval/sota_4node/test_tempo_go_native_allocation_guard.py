from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REQUEST = ROOT / "eval/sota_4node/request_perlmutter_4node_4h_native.sh"
REQUIRE = ROOT / "eval/sota_4node/require_perlmutter_4node_4h_interactive.sh"


def test_native_request_has_fixed_non_containerized_shape() -> None:
    text = REQUEST.read_text(encoding="utf-8")
    assert "exec /usr/bin/salloc" in text
    for fragment in (
        "-A m1248_g",
        "-C gpu",
        "-q interactive",
        "-t 04:00:00",
        "-N 4",
        "--ntasks-per-node=1",
        "--cpus-per-task=128",
        "--gpus-per-node=4",
        "--network=job_vni",
    ):
        assert fragment in text
    assert "$@" not in text
    assert "--image" not in text
    assert " shifter " not in text.lower()


def test_guards_reject_container_activation_and_privilege() -> None:
    request = REQUEST.read_text(encoding="utf-8")
    require = REQUIRE.read_text(encoding="utf-8")
    for text in (request, require):
        for variable in (
            "SHIFTER_RUNTIME",
            "SHIFTER_IMAGE",
            "UDI",
            "CRAY_ROOTFS",
            "SLURM_CONTAINER",
        ):
            assert variable in text
        assert '"$(id -u)" -ne 0' in text


def test_request_rejects_arguments_before_salloc() -> None:
    completed = subprocess.run(
        ["bash", str(REQUEST), "--image=forbidden"],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert "arguments are not accepted" in completed.stderr


def test_request_rejects_inherited_container_environment() -> None:
    env = os.environ.copy()
    env["SHIFTER_RUNTIME"] = "1"
    completed = subprocess.run(
        ["bash", str(REQUEST)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert "SHIFTER_RUNTIME" in completed.stderr


def test_sourced_allocation_guard_restores_caller_shell_options() -> None:
    text = REQUIRE.read_text(encoding="utf-8")
    assert "TEMPO_PD_GUARD_CALLER_SHELL_OPTIONS=$(set +o)" in text
    assert 'eval "${TEMPO_PD_GUARD_CALLER_SHELL_OPTIONS}"' in text
    assert text.index("TEMPO_PD_GUARD_CALLER_SHELL_OPTIONS=$(set +o)") < text.index(
        "set -euo pipefail"
    )
