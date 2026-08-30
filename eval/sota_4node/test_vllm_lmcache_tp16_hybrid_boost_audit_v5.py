from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as hybrid
from eval.sota_4node import vllm_lmcache_tp16_hybrid_boost_node_v5 as node
from eval.sota_4node import vllm_quiescence_wave_protocol_v4 as protocol_v4
from eval.sota_4node import vllm_quiescence_wave_protocol_v5 as protocol_v5


ROOT = Path(__file__).resolve().parents[2]


class _FakeAgent:
    def __init__(self) -> None:
        self.prepared: list[object] = []
        self.posted: list[object] = []

    def make_prepped_xfer(self, *_args: object) -> object:
        handle = object()
        self.prepared.append(handle)
        return handle

    def transfer(self, handle: object) -> str:
        self.posted.append(handle)
        return "PROC"

    def check_xfer_state(self, _handle: object) -> str:
        return "DONE"


class _FakeChannel:
    def __init__(self) -> None:
        self.nixl_agent = _FakeAgent()
        self.nixl_wrapper = SimpleNamespace(xfer_handler=object())
        self.remote_xfer_handlers_dict = {"rank-8": object()}

    def get_local_mem_indices(self, _objects: list[object]) -> list[int]:
        return [0]


def _client(prompt: int, mode: str) -> dict[str, object]:
    e2e = {hybrid.FG: 100.0, hybrid.LMCACHE: 120.0, hybrid.TEMPO: 104.0}[mode]
    return {
        "ttft_ms": 5.0,
        "tpot_p50_ms": 1.0,
        "tpot_p99_ms": 2.0,
        "request_e2e_ms": e2e,
        "output_token_sha256": f"prompt-{prompt}",
    }


def _valid_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for rank in range(hybrid.WORLD_SIZE):
        source = rank < hybrid.SOURCE_COUNT
        blocks: list[dict[str, object]] = []
        for index, (prompt, mode) in enumerate(hybrid.BLOCKS):
            transfer = mode != hybrid.FG
            completion_ms = 140.0 if mode == hybrid.LMCACHE else 100.0
            drain_ms = 20.0 if mode == hybrid.LMCACHE else 0.0
            blocks.append(
                {
                    "block_index": index,
                    "prompt_index": prompt,
                    "mode": mode,
                    "client": _client(prompt, mode) if rank == 0 else None,
                    "boost_hold_ns": 20_000_000 if mode == hybrid.TEMPO else 0,
                    "correctness_met": True,
                    "receiver_verified_bytes": (
                        hybrid.BYTES_PER_SOURCE if not source and transfer else 0
                    ),
                    "source_call": {
                        "rank": rank,
                        "calls": 1 if source and transfer else 0,
                        "descriptors": 1 if source and transfer else 0,
                        "bytes": hybrid.BYTES_PER_SOURCE if source and transfer else 0,
                        "completion_from_origin_ns": (
                            int(completion_ms * 1e6) if source and transfer else 0
                        ),
                        "post_foreground_drain_ns": (
                            int(drain_ms * 1e6) if source and transfer else 0
                        ),
                        "elapsed_ns": 20_000_000 if source and transfer else 0,
                        "start_lag_ns": 100_000 if source and transfer else 0,
                    },
                }
            )
        records.append({"rank": rank, "blocks": blocks})
    return records


def _args() -> SimpleNamespace:
    return SimpleNamespace(allocation_id="cpu-test", campaign_index=0, nixl_backend="UCX")


def _ast_flags(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        value.value
        for value in ast.walk(tree)
        if isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value.startswith("--")
    }


class HybridBoostAuditTests(unittest.TestCase):
    def test_hybrid_release_codec_is_installed_in_hook_and_wire(self) -> None:
        event = SimpleNamespace(event_id=7, request_id="tempo-scout-cpu")
        frame = protocol_v5.ReleaseFrame.wave(
            event,
            mode=protocol_v5.HYBRID_MODE,
            completed_bytes=hybrid.GLOBAL_BYTES,
            source_elapsed_ns=(1,) * hybrid.SOURCE_COUNT,
            wave_elapsed_ns=2,
        )
        decoded = protocol_v5.ReleaseFrame.from_payload(frame.to_payload(), event=event)
        self.assertEqual(decoded, frame)
        protocol_v5.install_generic_release_protocol()
        self.assertIn(protocol_v5.HYBRID_MODE, protocol_v4.DATA_MODES)
        self.assertIs(protocol_v4.hook.ReleaseFrame, protocol_v5.ReleaseFrame)
        self.assertIs(protocol_v4.hook.wire.ReleaseFrame, protocol_v5.ReleaseFrame)

    def test_prepared_handle_is_cached_and_reposted_after_done(self) -> None:
        Channel = hybrid._hybrid_channel_class(_FakeChannel)
        channel = Channel()
        spec = {"receiver_id": "rank-8", "remote_indexes": [0]}
        boost = __import__("threading").Event()

        first = channel.tempo_adaptive_write([object()], spec, boost)
        second = channel.tempo_adaptive_write([object()], spec, boost)

        self.assertEqual(first["completed"], 1)
        self.assertEqual(second["completed"], 1)
        self.assertEqual(len(channel.nixl_agent.prepared), 1)
        self.assertEqual(len(channel.nixl_agent.posted), 2)
        self.assertIs(channel.nixl_agent.posted[0], channel.nixl_agent.posted[1])

    def test_emergency_release_record_fails_closed_in_aggregate(self) -> None:
        records = _valid_records()
        tempo_index = next(
            index for index, (_prompt, mode) in enumerate(hybrid.BLOCKS)
            if mode == hybrid.TEMPO
        )
        records[0]["blocks"][tempo_index]["correctness_met"] = False
        result = hybrid._aggregate(records, {"validated": True}, _args())

        self.assertFalse(result["overall_correctness_met"])
        self.assertFalse(result["candidate_gates"]["correctness_output_trace"])
        self.assertEqual(result["screen_outcome"], "invalid_correctness_output_or_trace")

    def test_valid_cpu_fixture_reaches_all_candidate_gates(self) -> None:
        result = hybrid._aggregate(_valid_records(), {"validated": True}, _args())
        self.assertTrue(result["overall_correctness_met"])
        self.assertTrue(all(result["candidate_gates"].values()))
        self.assertEqual(result["screen_outcome"], "hybrid_candidate_pass")

    def test_launcher_node_and_runner_cli_are_parity_checked(self) -> None:
        launcher = (
            ROOT / "eval/sota_4node/run_vllm_lmcache_tp16_hybrid_boost_in_allocation.sh"
        ).read_text(encoding="utf-8")
        node_base_path = ROOT / "eval/sota_4node/vllm_lmcache_tp16_quiescence_scout_node_v1.py"
        runner_path = ROOT / "eval/sota_4node/run_vllm_lmcache_tp16_hybrid_boost_v5.py"
        node_flags = _ast_flags(node_base_path)
        runner_flags = _ast_flags(runner_path)
        launcher_to_node = {
            "--repo-root",
            "--result-dir",
            "--campaign-index",
            "--master-addr",
            "--vllm-master-port",
            "--sidecar-master-port",
            "--api-port",
            "--nixl-port-base",
            "--quiescence-socket",
            "--quiescence-trace",
            "--readiness-timeout-s",
            "--sidecar-timeout-s",
        }
        node_to_runner = {
            "--output-dir",
            "--plan",
            "--api-host",
            "--api-port",
            "--model",
            "--nixl-port-base",
            "--request-timeout-s",
            "--campaign-index",
            "--allocation-id",
            "--quiescence-socket",
            "--quiescence-trace",
        }

        self.assertTrue(launcher_to_node <= node_flags)
        self.assertTrue(all(flag in launcher for flag in launcher_to_node))
        self.assertTrue(node_to_runner <= runner_flags)
        self.assertTrue(node_to_runner <= node_flags)
        self.assertNotIn("--plan", launcher)
        self.assertEqual(node.base.RUNNER_MODULE, hybrid.__name__)
        self.assertEqual(
            node.base.PLAN_RELATIVE,
            Path("eval/sota_4node/real_tp16_hybrid_boost_v5.json"),
        )


if __name__ == "__main__":
    unittest.main()
