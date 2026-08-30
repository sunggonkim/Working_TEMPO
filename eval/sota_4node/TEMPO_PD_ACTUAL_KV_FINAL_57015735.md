# TEMPO actual-vLLM P/D KV result — allocation 57015735

## Final scheme

TEMPO uses evidence-gated P/D admission.  A request is sent through the remote
KV path only after the matching model, topology, workload, and load bucket have
shown both output correctness and at least a 5 ms conservative latency
advantage.  Otherwise it fails closed to decoder-local recompute/cache.

For this measured Qwen2.5-7B, four-node, two-replica TP4 workload, official
LMCache remote transfer did not pass that gate.  The frozen v28 controller
therefore selected local for all nine validation requests with reason
`fail_local_remote_correctness_or_5ms_gate_unproven`.

## Final exact comparison

Artifact:
`results/tempo_pd_final_forced_r16_o32_v35_job_57015735/final_result.json`

- Topology: four A100 nodes, two independent TP4 P/D pairs.
- Model: local Qwen2.5-7B-Instruct, BF16, vLLM 0.26.0+cu129.
- Workload: nine unique-head cold-cache prompts, about 1,220 input tokens and
  exactly 32 output tokens, request rate 16/s, max workers 16.
- Correctness: all requests used the same forced-token logit-bias contract;
  model/workload/schedule/output token sequences matched exactly.
- Baseline: official pinned `LMCacheConnectorV1`, all nine requests through the
  live remote P/D KV path.
- TEMPO: all nine requests took the evidence-gated local path.

| Metric | LMCache remote | TEMPO | Change |
|---|---:|---:|---:|
| E2E p50 | 829.723 ms | 738.686 ms | -10.97% |
| E2E p99 | 864.148 ms | 775.860 ms | -10.22% |
| TPOT p99 | 25.528 ms | 23.088 ms | -9.56% |
| Request goodput | 7.002 req/s | 7.403 req/s | +5.73% |
| Output-token goodput | 224.071 tok/s | 236.903 tok/s | +5.73% |
| Paired E2E wins | — | 9/9 | median -88.287 ms |

All eight final gates passed: exact outputs, exact route provenance, 9/9 SLO
success, at least 5% E2E-p50 improvement, goodput improvement, TPOT
non-regression, and at least two-thirds paired wins.

## Falsified alternatives in the same allocation

- Native vLLM `NixlConnector`: exact 32-token streams passed after correcting
  the measurement contract, but it lost to LMCache 0/9 with paired median E2E
  `+239.720 ms`; rejected.
- LMCache always-remote at rate 16: lost 0/9 to local on unique cold-cache
  prompts; paired median remote penalty was `+73.836 ms`.
- LMCache always-remote at rate 32: E2E p50 was `847.143 ms` versus local
  `775.925 ms`, and goodput was `8.390` versus `9.365 req/s`; rejected.
- Mixed local/remote one-credit admission: completed correctly on the first
  unique-head screen and improved aggregate goodput by about 0.99%, but paired
  median E2E regressed `+6.463 ms`.  A later pressure-triggered version also
  produced one remote output mismatch and worse TPOT; mixed mode is disabled.
- Repeated prompts sharing the first 1,024-token LMCache hash caused a real
  `CacheEngineKey ... not found` engine failure under mixed routing.  Final
  experiments use unique prompt prefixes, not suffix-only identifiers.

## Claim boundary

This is a promising actual-vLLM-owned-KV component/system screen showing that
measured admission can beat unconditional LMCache P/D by avoiding an
unprofitable transfer.  It is not a new transport, not a universal LMCache or
Mooncake win, and not a global SOTA claim.  Mooncake was not promoted to this
same-harness comparison because the installed environment lacks the required
official P/D router and a safety-compliant explicit CXI topology integration.

The next publication-grade step is a broader request distribution with an
independent workload, concurrency sweep, and at least one load bucket where the
same controller safely admits a proven-profitable remote transfer.
