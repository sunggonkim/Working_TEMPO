#!/usr/bin/env python3
"""N21: give prepared KV transfer a measured 25ms head start before decode."""
from __future__ import annotations
import json
from pathlib import Path
import queue
import statistics
import threading
import time
from typing import Any

from eval.sota_4node import run_vllm_lmcache_tp16_deadline_c9_entry as c9
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_async_v8_entry as v8
from eval.sota_4node import run_vllm_lmcache_tp16_predecode_phase_m20_entry as m20
from eval.sota_4node import vllm_quiescence_wave_protocol_async_v8 as protocol

CANDIDATE_MODE = "tempo_predecode_headstart_25ms"
CONTRACT_ID = "tp16-predecode-headstart25-n21"
RESULT_SCHEMA = "tempo-vllm-tp16-predecode-headstart-result-21"
HEADSTART_MS = 25.0
BLOCKS = ((0, old.FG), (0, old.LMCACHE), (0, CANDIDATE_MODE),
          (1, CANDIDATE_MODE), (1, old.FG), (1, old.LMCACHE),
          (2, old.LMCACHE), (2, CANDIDATE_MODE), (2, old.FG))

def _expected_contract() -> dict[str, Any]:
    return {
        "schema_version": "tempo-tp16-predecode-headstart-contract-21",
        "contract_id": CONTRACT_ID,
        "topology": {"nodes": 4, "world_size": 16,
                     "source_ranks": list(range(8)),
                     "receiver_ranks": list(range(8, 16)),
                     "pairing": [[r, r + 8] for r in range(8)]},
        "transfer": {"bytes_per_source": 16 << 20, "global_bytes": 128 << 20,
                     "calls_global": 8, "physical_descriptors_global": 8,
                     "prepared_handle_repost": True},
        "algorithm": {"candidate_mode": CANDIDATE_MODE,
                      "predecode_headstart_ms": HEADSTART_MS,
                      "headstart_included_in_admission_latency": True,
                      "completion_wait_before_decode": False,
                      "decode_progress_sleep_ms": 1.0,
                      "hook_events_per_measured_candidate": 0,
                      "boost_enabled": False,
                      "single_factor_from": "M20 completion barrier -> 25ms head start"},
        "campaign": {"modes": [old.FG, old.LMCACHE, CANDIDATE_MODE],
                     "blocks": 9, "replicates_per_mode": 3,
                     "candidate_gates": {"exact_correctness": True,
                        "headstart_min_ms": 25.0, "post_foreground_drain_zero": True,
                        "paired_service_delta_median_le_ms": -5.0,
                        "meaningful_service_wins": 2,
                        "request_e2e_le_fg_ratio": 1.03,
                        "tpot_p99_le_lmcache_ratio": 1.10}},
    }

def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _expected_contract(): raise ValueError("N21 contract changed")
    return payload

def _run_candidate(torch: Any, dist: Any, *, channel: Any, obj: Any, rank: int,
                   pair: int, block_index: int, prompt_index: int, args: Any):
    source, receiver = rank < 8, rank >= 8
    expected = 1 + ((block_index * 37 + pair * 3) % 251)
    obj.raw_data.fill_(expected if source else 0); torch.cuda.synchronize(); dist.barrier()
    signal = torch.tensor([block_index], dtype=torch.int64, device="cpu")
    dist.broadcast(signal, src=0); origin_ns = time.perf_counter_ns()
    if int(signal.item()) != block_index: raise RuntimeError("N21 start signal changed")
    boost, entered, done = threading.Event(), threading.Event(), threading.Event()
    state = {"started_ns": 0, "finished_ns": 0, "completed": 0, "polls": 0,
             "low_priority_sleeps": 0, "boost_polls": 0, "yields": 0, "error": None}
    worker = None
    if source:
        worker = threading.Thread(target=fixed._transfer_worker, kwargs={
            "channel": channel, "obj": obj, "receiver_id": f"rank-{rank + 8}",
            "mode": CANDIDATE_MODE, "boost": boost, "entered": entered,
            "done": done, "state": state}, daemon=True)
        worker.start()
    if source and not entered.wait(5): raise RuntimeError("N21 worker did not enter")
    entered_ok = torch.tensor([int(not source or entered.is_set())], dtype=torch.int64)
    dist.all_reduce(entered_ok, op=dist.ReduceOp.MIN)
    if int(entered_ok.item()) != 1: raise RuntimeError("N21 entry handshake failed")

    target_ns = origin_ns + int(HEADSTART_MS * 1e6)
    remaining = target_ns - time.perf_counter_ns()
    if remaining > 0: time.sleep(remaining / 1e9)
    headstart_done_ns = time.perf_counter_ns()
    headstart_elapsed_ns = headstart_done_ns - origin_ns

    caller_id = f"control-{args.allocation_id}-c{args.campaign_index}-b{block_index}-{CANDIDATE_MODE}"
    events: queue.Queue[tuple[bool, Any]] = queue.Queue()
    client_started_ns = client_finished_ns = 0
    client = None
    if rank == 0:
        client_started_ns = time.perf_counter_ns() - origin_ns
        client = threading.Thread(target=old.bulk._request_thread, kwargs={
            "events": events, "args": args, "prompt": old.base.PROMPTS[prompt_index],
            "caller_id": caller_id, "tokens": old.TOKENS})
        client.start()
    control: list[Any] = [None]
    if rank == 0:
        ok, value = events.get(timeout=args.request_timeout_s)
        client_finished_ns = time.perf_counter_ns() - origin_ns
        client.join(timeout=1.0); control[0] = {"ok": ok, "value": value}
    dist.broadcast_object_list(control, src=0)
    foreground_done_ns = time.perf_counter_ns()
    if not control[0]["ok"]: raise RuntimeError(f"N21 request failed: {control[0]['value']}")
    if source and not done.wait(60): raise RuntimeError("N21 transfer timed out")
    if worker is not None: worker.join()
    dist.barrier()
    verified = old.BYTES_PER_SOURCE if receiver and bool(torch.all(obj.raw_data == expected).item()) else 0
    dist.barrier()
    elapsed = max(0, int(state["finished_ns"]) - int(state["started_ns"])) if source else 0
    completion = max(0, int(state["finished_ns"]) - origin_ns) if source else 0
    call = {"rank": rank, "source": source, "calls": int(source),
            "completed": int(state["completed"]) if source else 0,
            "descriptors": old._descriptor_count(channel) if source else 0,
            "bytes": old.BYTES_PER_SOURCE if source and state["completed"] == 1 else 0,
            "elapsed_ns": elapsed, "completion_from_origin_ns": completion,
            "post_foreground_drain_ns": max(0, int(state["finished_ns"]) - foreground_done_ns) if source else 0,
            "start_lag_ns": max(0, int(state["started_ns"]) - origin_ns) if source else 0,
            "polls": int(state["polls"]), "low_priority_sleeps": int(state["low_priority_sleeps"]),
            "boost_polls": int(state["boost_polls"]), "yields": int(state["yields"]),
            "boost_wait_timed_out": False, "error": state["error"]}
    correct = ((not source or state["completed"] == 1 and state["error"] is None)
               and (not receiver or verified == old.BYTES_PER_SOURCE))
    return {"block_index": block_index, "prompt_index": prompt_index,
            "mode": CANDIDATE_MODE,
            "client": old.scout._client_metrics(control[0]["value"]) if rank == 0 else None,
            "client_started_from_origin_ns": client_started_ns,
            "client_finished_from_origin_ns": client_finished_ns,
            "gate_ready": None, "gate_release": None, "boost_hold_ns": 0,
            "promotion_armed_sources": 0, "controller_decision": "headstart_25ms",
            "candidate_hook_invocations": 0, "rescue_armed_sources": 0,
            "headstart_elapsed_ns": headstart_elapsed_ns,
            "transfer_completed_before_decode": bool(done.is_set()) if source else True,
            "source_call": call, "receiver_verified_bytes": verified,
            "receiver_zero_ok": True, "correctness_met": correct}

def _run_block(*args: Any, mode: str, **kwargs: Any):
    return (_run_candidate(*args, **kwargs) if mode == CANDIDATE_MODE
            else c9._run_block(*args, mode=mode, **kwargs))

def _aggregate(records: list[dict[str, Any]], trace: dict[str, Any], args: Any):
    result = m20._aggregate(records, trace, args)
    ordered = sorted(records, key=lambda row: int(row["rank"]))
    candidates = [b for b in result["blocks"] if b["mode"] == CANDIDATE_MODE]
    for block in candidates:
        raw = [row["blocks"][int(block["block_index"])] for row in ordered]
        block["headstart_elapsed_ms"] = max(float(row["headstart_elapsed_ns"]) / 1e6 for row in raw)
        block["transfer_completed_before_decode_sources"] = sum(
            bool(row["transfer_completed_before_decode"]) for row in raw[:8])
    deltas = [float(row["tempo_minus_lmcache_service_makespan_ms"]) for row in result["paired"]]
    cand_e2e = result["mode_metrics"][CANDIDATE_MODE]["e2e_p50_ms"]
    fg_e2e = result["mode_metrics"][old.FG]["e2e_p50_ms"]
    cand_tpot = result["mode_metrics"][CANDIDATE_MODE]["tpot_p99_max_ms"]
    lm_tpot = result["mode_metrics"][old.LMCACHE]["tpot_p99_max_ms"]
    gates = {"correctness_output_trace": bool(result["overall_correctness_met"]),
             "headstart_at_least_25ms": all(b["headstart_elapsed_ms"] >= 25.0 for b in candidates),
             "all_candidate_post_foreground_drain_zero": all(b["post_foreground_drain_ms"] == 0 for b in candidates),
             "paired_service_median_le_minus_5ms": statistics.median(deltas) <= -5,
             "paired_service_meaningful_wins_ge_2": sum(d <= -5 for d in deltas) >= 2,
             "candidate_request_e2e_p50_le_1_03x_fg": cand_e2e <= 1.03 * fg_e2e,
             "candidate_tpot_p99_le_1_10x_lmcache": cand_tpot <= 1.10 * lm_tpot}
    result["schema_version"] = RESULT_SCHEMA; result["contract_id"] = CONTRACT_ID
    result["config"].update(candidate_mode=CANDIDATE_MODE,
        predecode_headstart_ms=HEADSTART_MS, headstart_included_in_admission_latency=True,
        completion_wait_before_decode=False)
    result["candidate_gates"] = gates
    result["screen_outcome"] = ("invalid_correctness_output_or_trace" if not result["overall_correctness_met"]
        else "headstart25_candidate_pass" if all(gates.values()) else "headstart25_candidate_revise")
    return result

def main() -> None:
    audited = fixed._transfer_worker; old._transfer_worker = audited
    for module in (c9, m20):
        module.CANDIDATE_MODE = CANDIDATE_MODE; module.CONTRACT_ID = CONTRACT_ID
        module.RESULT_SCHEMA = RESULT_SCHEMA; module.BLOCKS = BLOCKS
    c9._install_candidate_mode(); protocol.install_async_release_protocol()
    old.protocol.ReleaseFrame = protocol.ReleaseFrame; old.bulk.protocol.ReleaseFrame = protocol.ReleaseFrame
    v8.CONTRACT_ID = CONTRACT_ID; v8.RESULT_SCHEMA = RESULT_SCHEMA
    v8._load_contract = _load_contract; v8._run_block = _run_block
    v8._validate_trace = c9._validate_trace; v8._aggregate = _aggregate
    v8.main()

if __name__ == "__main__": main()
