from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).with_name(
    "run_tempo_go_cross_layer_with_cojob_in_allocation.sh"
)


def test_native_step_contract_matches_perlmutter_gpu_shape() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for fragment in (
        "--nodes=4 --ntasks=4 --ntasks-per-node=1",
        "--gpus-per-task=4 --gpu-bind=none",
        "--cpus-per-task=128 --cpu-bind=cores",
        '[[ "${JOB_NETWORK}" != "job_vni" ]]',
        "allocation_missing_job_vni",
        "allocation_network_not_job_vni",
    ):
        assert fragment in text


def test_observer_stale_guard_has_startup_grace_and_bounded_age() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'TEMPO_GO_NCCL_OBSERVER_MAX_AGE_MS:-60000' in text
    assert 'TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS:-180000' in text
    assert "now_ns >= C5_START_UNIX_NS + startup_grace_ns" in text
    assert "TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS >= 60000" in text
    assert "TEMPO_GO_NCCL_OBSERVER_STARTUP_GRACE_MS <= 600000" in text


def test_cojob_timeout_is_finite_and_privilege_free() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'TEMPO_GO_NCCL_TIMEOUT_SECONDS:-60' in text
    assert "nccl_collective_timeout_seconds" in text
    assert "TEMPO_GO_SCONTROL_TIMEOUT_SECONDS" in text
    assert "/usr/bin/timeout --foreground" in text
    assert 'JOB_INFO="${TEMPO_PD_ALLOCATION_RECORD:-}"' in text
    assert 'export TEMPO_PD_ALLOCATION_RECORD="${JOB_INFO}"' in text
    for forbidden in ("sudo", "udiRoot", "CAP_NET_ADMIN", "--image", "shifter"):
        assert forbidden not in text


def test_capability_probe_is_non_interconnect_and_four_node_bound() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "probe_tempo_go_cross_layer_capability.py" in text
    assert "--network=no_vni" in text
    assert 'capability-${SLURM_PROCID}.json' in text
    assert 'expected 4 node capability receipts' in text
    assert "tempo-cross-layer-capability-receipt-v1" in text


def test_python_overlay_preparation_does_not_consume_slingshot_vni() -> None:
    text = Path(__file__).with_name(
        "prepare_c4_python_overlay.sh").read_text(encoding="utf-8")
    assert "PREPARE_SRUN_NETWORK_ARGS=(--network=no_vni)" in text
    assert text.count('"${PREPARE_SRUN_NETWORK_ARGS[@]}"') == 2
