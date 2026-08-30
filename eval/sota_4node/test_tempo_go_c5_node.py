from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from eval.sota_4node import vllm_lmcache_tempo_go_c5_node as node
from eval.sota_4node import run_tempo_go_c5_stream_client as client
from eval.sota_4node.run_vllm_stream_metrics import WorkItem


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / (
    "results/tempo_go_c5_cpu_gate_20260821_anchor_v3_retry2/"
    "tempo_go_workload_manifest.json"
)


def test_default_c5_profiles_are_the_bound_output2_anchor_set() -> None:
    assert "tempo_go_c5_anchor_priors_c12_v3_retry1" in node.GLOBAL_PROFILE
    assert "tempo_go_c5_anchor_priors_c12_v3_retry1" in node.ELASTIC_PROFILE
    assert "tempo_go_c5_anchor_priors_c12_v3_retry1" in node.ENDPOINT_PROFILE
    assert MANIFEST.is_file()


def test_profile_path_resolves_only_inside_repository(monkeypatch) -> None:
    monkeypatch.setenv(node.GLOBAL_PROFILE_ENV, str(MANIFEST))
    assert node._profile_path(ROOT, node.GLOBAL_PROFILE_ENV, node.GLOBAL_PROFILE) == MANIFEST
    outside = Path("/tmp/tempo-go-profile-outside.json")
    monkeypatch.setenv(node.GLOBAL_PROFILE_ENV, str(outside))
    with pytest.raises(ValueError, match="resolve below the repository"):
        node._profile_path(ROOT, node.GLOBAL_PROFILE_ENV, node.GLOBAL_PROFILE)


def test_anchor_profile_identity_is_closed_before_native_spawn() -> None:
    global_path = ROOT / node.GLOBAL_PROFILE
    elastic_path = ROOT / node.ELASTIC_PROFILE
    endpoint_path = ROOT / node.ENDPOINT_PROFILE
    profile = node._validate_profile_bindings(
        global_path=global_path,
        elastic_path=elastic_path,
        endpoint_path=endpoint_path,
        workload_manifest=MANIFEST,
    )
    assert profile.identity.workload_manifest_sha256 == node._sha256(MANIFEST)
    assert profile.telemetry.scheduler_observation_required is True


def test_profile_binding_rejects_a_different_manifest(tmp_path: Path) -> None:
    other = tmp_path / "tempo_go_workload_manifest.json"
    other.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    other.write_text(other.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="workload manifest"):
        node._validate_profile_bindings(
            global_path=ROOT / node.GLOBAL_PROFILE,
            elastic_path=ROOT / node.ELASTIC_PROFILE,
            endpoint_path=ROOT / node.ENDPOINT_PROFILE,
            workload_manifest=other,
        )


def test_heldout_frozen_validation_profile_is_native_bindable() -> None:
    root = ROOT / "results/tempo_go_c5_heldout_frozen_proxy_v1"
    profile = node._validate_profile_bindings(
        global_path=root / "frozen_global_profile.json",
        elastic_path=ROOT / (
            "results/tempo_go_c5_anchor_priors_c12_v3_retry1/"
            "real_tempo_pd_elastic_profile_c12_anchor_output2_screen_v3.json"
        ),
        endpoint_path=root / "frozen_endpoint_service_profile.json",
        workload_manifest=ROOT / (
            "results/tempo_go_c5_heldout_output128_v1/"
            "tempo_go_workload_manifest.json"
        ),
    )
    assert profile.deployment_scope == "frozen_validation"


def test_warmup_rewrite_seeds_only_explicit_p_only_rows(
    tmp_path: Path, monkeypatch,
) -> None:
    workload = tmp_path / "validation.jsonl"
    workload.write_text(
        "\n".join(json.dumps(row) for row in (
            {
                "request_id": "epd-tempo-background-c1-cache-miss-measured-r00-remote-hot-000001",
                "prompt": "miss",
                "max_tokens": 2,
                "arrival_offset_ms": 0.0,
            },
            {
                "request_id": "epd-tempo-background-c2_kv_remote_hot-cache-p-only-measured-r00-kv-remote-hot-000002",
                "prompt": "p-only",
                "max_tokens": 2,
                "arrival_offset_ms": 0.0,
            },
            {
                "request_id": "epd-tempo-background-c3_both_hot-cache-p-only-measured-r00-kv-remote-hot-000003",
                "prompt": "p-only",
                "max_tokens": 2,
                "arrival_offset_ms": 1.0,
            },
        )) + "\n",
        encoding="utf-8",
    )
    generated_warmup = tmp_path / "generated-warmup.jsonl"
    generated_warmup.write_text(
        json.dumps({"request_id": "warm-tempo-0", "prompt": "miss", "max_tokens": 2})
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEMPO_GO_C5_SOURCE_WORKLOAD", str(workload))
    rewritten = client._rewrite_warmup_workload([
        "client", "--workload", str(generated_warmup), "c5-warmup",
    ])
    rows = [
        json.loads(line) for line in Path(rewritten[2]).read_text(
            encoding="utf-8").splitlines() if line
    ]
    assert len(rows) == 1
    assert rows[0]["prompt"] == "p-only"
    assert rows[0]["request_id"] == "epd-tempo-background-warm-c5-item-0"


def test_measured_rewrite_selects_explicit_native_arm(tmp_path: Path, monkeypatch) -> None:
    workload = tmp_path / "validation.jsonl"
    workload.write_text(json.dumps({
        "request_id": "epd-tempo-latency-c1-cache-miss-measured-r00-foreground-0",
        "prompt": "cold",
        "max_tokens": 2,
        "arrival_offset_ms": 0.0,
    }) + "\n", encoding="utf-8")
    (tmp_path / "run").mkdir()
    monkeypatch.setenv("TEMPO_GO_C5_ARM", "queue_gpu")
    rewritten = client._rewrite_measured_arm_workload([
        "client", "--workload", str(workload),
        "--output", str(tmp_path / "run" / "raw.json"), "validation",
    ])
    rows = [
        json.loads(line) for line in Path(rewritten[2]).read_text(
            encoding="utf-8").splitlines() if line
    ]
    assert rows[0]["request_id"].startswith("epd-queue_gpu-latency-")
    assert Path(rewritten[rewritten.index("--workload") + 1]).parent == tmp_path / "run"
    assert not (tmp_path / "global-c5-queue_gpu-measured-rewritten.jsonl").exists()


def test_exact_geometry_wrapper_emits_one_seed_then_one_probe(monkeypatch) -> None:
    calls = []

    def fake_execute(item, *args, **kwargs):
        calls.append((item.request_id, item.max_tokens))
        return {
            "valid": True,
            "router": {"route": "official_lmcache_remote_prefill",
                       "reason": "physical_preparation"},
        }

    monkeypatch.setattr(client, "_ORIGINAL_FORCED", fake_execute)
    item = WorkItem(
        index=0,
        request_id=(
            "epd-tempo-interactive-c4-cache-p-only-warm-physical-"
            "c8-block-owner-0-item-000000"
        ),
        prompt="prompt",
        max_tokens=128,
        arrival_offset_ns=0,
    )
    result = client._execute_with_tenant(item)
    assert calls == [
        (item.request_id.replace("-warm-", "-warm-seed-o128-", 1), 128),
        (item.request_id, 128),
    ]
    assert result["p_only_cache_seed"]["output_tokens"] == 128


def test_main_bypasses_older_nested_p_only_seed_wrapper(monkeypatch) -> None:
    original_execute = client.forced.execute_request
    original_schema = client.canonical._prior.ROUTER_SCHEMA
    original_argv = list(sys.argv)
    monkeypatch.setattr(client, "_rewrite_warmup_workload", list)
    monkeypatch.setattr(client, "_rewrite_measured_arm_workload", list)
    monkeypatch.setattr(
        client.canonical, "main",
        lambda: pytest.fail("older nested P_ONLY wrapper must not run"),
    )

    def fake_prior_main():
        assert client.forced.execute_request is client._execute_with_tenant
        assert client.canonical._prior.ROUTER_SCHEMA == client.canonical.ROUTER_SCHEMA
        return 17

    monkeypatch.setattr(client.canonical._prior, "main", fake_prior_main)
    monkeypatch.setattr(sys, "argv", ["client"])
    assert client.main() == 17
    assert client.forced.execute_request is original_execute
    assert client.canonical._prior.ROUTER_SCHEMA == original_schema
    assert sys.argv == ["client"]
    sys.argv[:] = original_argv
