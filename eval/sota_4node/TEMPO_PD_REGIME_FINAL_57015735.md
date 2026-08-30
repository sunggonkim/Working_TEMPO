# TEMPO actual-vLLM P/D arrival-regime controller

## Outcome

The frozen v55 controller is the best candidate from allocation `57015735`.
It operates on actual vLLM-owned KV state through the pinned official
`LMCacheConnectorV1` remote branch.  It is not a sidecar-only transfer test.

The controller observes pair-local arrival intervals for a fresh admission
epoch and freezes one regime:

- low: decoder-local;
- mid: two local calibration requests per P/D pair, then remote prefill;
- high: decoder-local until eight live requests per pair, then remote spill.

Frozen parameters are a 70 ms high/mid pair-interval boundary, a 110 ms
mid/low boundary, two intervals (three arrivals) for calibration, and eight
live local credits in the high regime.  The reusable state machine is in
`tempo/pd_regime_controller.py`; the measured vLLM adapter is
`eval/sota_4node/tempo_pd_regime_router_v55.py`.

## Experimental contract

- 4 Perlmutter A100 nodes, two independent TP4 P/D pairs, 4 GPUs per node.
- Qwen2.5-7B-Instruct, 1,230 prompt tokens, 32 forced-identical output tokens.
- Sources use actual vLLM P/D routing and official pinned LMCache/NIXL/UCX.
- Unique prompt-head nonces prevent cache-key aliasing.
- SSE is drained through HTTP EOF and every request has the exact same token
  sequence across local, LMCache, and TEMPO arms.
- Fixed local and all-remote LMCache baselines use the same model, workload,
  schedule, GPU budget, KV geometry, and SLO contract.

## Final high-load result (32 requests/s, 24 requests)

Artifact:
`results/tempo_pd_production_regime_highload_v55_job_57015735/final_result.json`

| Metric | Fixed local | LMCache remote | TEMPO v55 |
|---|---:|---:|---:|
| Routes (local/remote) | 24/0 | 0/24 | 16/8 |
| E2E p50 (ms) | 847.756 | 882.269 | **806.563** |
| E2E p99 (ms) | 1179.778 | 1126.125 | **1056.630** |
| TTFT p99 (ms) | 466.802 | 79.031 | **71.298** |
| TPOT p99 (ms) | **27.546** | 34.614 | 32.309 |
| Request goodput (/s) | 13.633 | 13.861 | **14.374** |

TEMPO versus fixed local: E2E p50 -4.86%, E2E p99 -10.44%, goodput
+5.43%, and 24/24 paired E2E wins.  TPOT p99 is worse than local because the
remote transfers share decode resources, but the TTFT and end-to-end tails
more than compensate.

TEMPO versus all-remote LMCache: E2E p50 -8.58%, E2E p99 -6.17%, TPOT p99
-6.66%, goodput +3.70%, and 23/24 paired E2E wins.  All eight conservative
analysis gates passed.

## Mid-load result (24 requests/s, 24 requests)

Artifact:
`results/tempo_pd_production_regime_midload_v56_job_57015735/final_result.json`

| Metric | Fixed local | LMCache remote | TEMPO v55 |
|---|---:|---:|---:|
| Routes (local/remote) | 24/0 | 0/24 | 4/20 |
| E2E p50 (ms) | 823.893 | 838.214 | **834.038** |
| E2E p99 (ms) | 899.352 | 860.670 | **859.165** |
| TTFT p99 (ms) | 198.455 | 69.594 | **66.968** |
| TPOT p99 (ms) | 24.639 | 26.086 | **26.018** |
| Request goodput (/s) | 13.317 | **13.360** | 13.252 |

TEMPO beats all-remote LMCache on E2E p50/p99, TTFT p99, and TPOT p99 while
retaining 99.19% of its goodput.  Every predeclared mid-load gate passed.

## Low-load result (16 requests/s, 9 requests)

Artifact:
`results/tempo_pd_queue_crossover_lowload_v47_job_57015735/final_result.json`

The same eight-credit policy sent 9/9 requests local.  TEMPO E2E p50/p99 were
738.180/776.624 ms versus fixed local 745.925/780.508 ms and all-remote LMCache
829.723/864.148 ms.  Goodput was 7.424/s versus 7.300/s local and 7.002/s
remote.  All low-load bypass gates passed.

## Falsified alternatives

- Native vLLM NixlConnector lost to LMCache 0/9 with paired median E2E
  +239.720 ms.
- Fixed all-remote LMCache lost badly at low load.
- High-load local-credit thresholds 7 and 9 both failed the frozen gate;
  threshold 8 was the measured optimum.  Threshold 7 increased goodput only
  slightly but worsened E2E and TPOT; threshold 9 lost goodput and tail.
- A raw threshold-8 controller at mid load mixed 19/5 and failed due to tail
  interference.  Arrival-regime calibration corrected this.
- Warmup traffic cannot calibrate the regime: tokenizer/connection startup
  distorted pair intervals to 81--89 ms.  The controller therefore resets at
  the explicit measured admission epoch.

## Claim boundary and next paper step

This is a promising actual-vLLM P/D component/system screen against the pinned
official LMCache remote path, not a universal SOTA claim.  The baselines and
candidate were separate bounded lifecycles in the same allocation, and each
load point has one measured campaign as requested.  A publication claim still
needs frozen multi-seed workloads, longer outputs, multiple models/KV sizes,
and an interleaved same-server lifecycle to quantify run-to-run noise.

Mooncake is not claimed as beaten here.  Its safe Perlmutter CXI path requires
an explicit topology mapping and a same-harness actual-vLLM P/D adapter; the
existing Mooncake component number uses a different topology and semantics.
