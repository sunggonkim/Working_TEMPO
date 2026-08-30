# TEMPO actual-vLLM P/D controller: frozen result

This report supersedes `TEMPO_PD_REGIME_FINAL_57015735.md` by adding the
64-output-token validation and the final workload-aware credit policy.

## Frozen scheme

For each P/D pair and measured admission epoch:

1. Observe two pair-local arrival intervals (three arrivals).
2. Freeze the epoch as low, mid, or high pressure using 110 ms and 70 ms
   pair-interval boundaries.
3. Low: decoder-local. Mid: two calibration requests local, then LMCache
   remote prefill. High: admit local work to a live-credit cap, then spill to
   LMCache remote prefill.
4. Use eight high-regime local credits for 32 output tokens and nine for 64.
   Other output lengths fail closed until measured.

Reusable implementation:

- `tempo/pd_regime_controller.py`
- `tempo/pd_workload_policy.py`

Actual-vLLM adapters used for the successful GPU runs:

- 32-token production adapter: `eval/sota_4node/tempo_pd_regime_router_v55.py`
- 64-token nine-credit adapter: `eval/sota_4node/tempo_pd_regime_router_v59.py`

## Results

All runs used 4 A100 nodes, Qwen2.5-7B-Instruct, two TP4 P/D pairs, actual
vLLM-owned KV, pinned official LMCacheConnectorV1/NIXL/UCX, unique prompt-head
nonces, exact output-token equality, and the same workload/GPU budget per arm.

### High load, 32 output tokens

Artifact:
`results/tempo_pd_production_regime_highload_v55_job_57015735/final_result.json`

| Metric | Fixed local | LMCache remote | TEMPO |
|---|---:|---:|---:|
| Local/remote routes | 24/0 | 0/24 | 16/8 |
| E2E p50 (ms) | 847.756 | 882.269 | **806.563** |
| E2E p99 (ms) | 1179.778 | 1126.125 | **1056.630** |
| TTFT p99 (ms) | 466.802 | 79.031 | **71.298** |
| TPOT p99 (ms) | **27.546** | 34.614 | 32.309 |
| Goodput (req/s) | 13.633 | 13.861 | **14.374** |

Versus local, TEMPO improves p50 4.86%, p99 10.44%, and goodput 5.43%
(24/24 paired E2E wins). Versus LMCache, it improves p50 8.58%, p99 6.17%,
TPOT p99 6.66%, and goodput 3.70% (23/24 wins). All gates passed.

### High load, 64 output tokens

Artifact:
`results/tempo_pd_production_regime_r32_o64_credit9_v60_job_57015735/final_result.json`

| Metric | Fixed local | LMCache remote | TEMPO |
|---|---:|---:|---:|
| Local/remote routes | 24/0 | 0/24 | 18/6 |
| E2E p50 (ms) | 1544.250 | 1649.755 | **1528.341** |
| E2E p99 (ms) | 2580.651 | 2561.434 | **2447.710** |
| TTFT p99 (ms) | 1147.680 | **67.787** | 1078.352 |
| TPOT p99 (ms) | **24.597** | 39.832 | 37.533 |
| Goodput (req/s) | 7.506 | 7.378 | **7.734** |

Versus local, TEMPO improves p50 1.03%, p99 5.15%, and goodput 3.04%
(22/24 wins). Versus LMCache, it improves p50 7.36%, p99 4.44%, TPOT p99
5.77%, and goodput 4.83% (23/24 wins). All gates passed. The local branch
still has the best TPOT, while remote routing removes enough TTFT queueing to
win end-to-end.

### Mid and low load

Mid artifact:
`results/tempo_pd_production_regime_midload_v56_job_57015735/final_result.json`

At 24 requests/s the controller chose 4 local/20 remote. Against all-remote
LMCache, TEMPO improved E2E p50 0.50%, p99 0.17%, TTFT p99 3.77%, and TPOT p99
0.26%, while retaining 99.19% of its goodput. All predeclared gates passed.

Low artifact:
`results/tempo_pd_queue_crossover_lowload_v47_job_57015735/final_result.json`

At 16 requests/s with nine requests, all 9 routed local. TEMPO E2E p50/p99
were 738.180/776.624 ms versus LMCache 829.723/864.148 ms, and goodput was
7.424/s versus 7.002/s. All gates passed.

## What the experiments falsified

- Native NixlConnector lost 0/9 to LMCache with +239.720 ms paired median E2E.
- All-remote LMCache is wrong at low load.
- A single raw in-flight threshold fails at mid load due to local/remote tail
  interference.
- High-load credit 7 overuses remote and worsens E2E/TPOT; credit 9 underuses
  remote for 32-token output. Credit 8 is the measured 32-token optimum.
- Credit 8 narrowly loses fixed-local p50 at 64 tokens; credit 9 restores p50
  while retaining tail and goodput wins.
- Warmup timestamps cannot classify load because tokenizer/connection startup
  distorts intervals. Admission epochs must reset after warmup or phase change.

## Claim boundary

This is a positive actual-vLLM P/D system screen against the pinned official
LMCache remote path. It is not yet a universal SOTA or Mooncake win. Each load
point has one campaign, as requested, and lifecycle noise remains measurable.
A paper claim needs frozen multi-seed repetitions, more prompt/KV sizes and
models, and a same-server interleaved baseline harness.

Mooncake was not forced into an invalid comparison. Its existing result uses a
different topology/semantics, and safe Perlmutter CXI execution still needs an
explicit topology mapping plus an actual-vLLM P/D adapter in this harness.
