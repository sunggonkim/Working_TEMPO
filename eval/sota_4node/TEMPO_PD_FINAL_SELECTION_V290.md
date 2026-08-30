# Final actual-vLLM P/D controller selection

Allocation: `57057488` (four Perlmutter A100 nodes).

## Frozen structure

Use `tempo.pd_warm_regime_controller.WarmRegimeController` with a regime
frozen before an epoch:

- Offered rates 16, 32, or 48 requests/s: policy8 cache-affinity hybrid.
- Offered rate 52 requests/s: policy11 high-load circuit breaker; bypass every
  remote warm hit and recompute/use cache on the decoder.
- Every other rate, model, topology, or geometry: fail closed.

This is a calibrated controller, not an online load estimator.  Its validated
topology is Qwen2.5-7B-Instruct with actual vLLM TP4 prefill plus TP4 decode,
two replicas over four nodes, official LMCache over NIXL/UCX, and the frozen
24-item warm-cache workload.

## Evidence

Policy8 same-window results against official LMCache remote P/D:

| Rate | E2E wins | Median E2E delta | TPOT wins | Median TPOT delta |
|---:|---:|---:|---:|---:|
| 16 | 23/24 | -78.070 ms | 24/24 | -0.926 ms |
| 32 | 22/24 | -78.112 ms | 23/24 | -1.266 ms |
| 48 | 20/24 | -119.700 ms | 22/24 | -4.097 ms |
| 52 | 22/24 | -85.382 ms | 23/24 | -4.875 ms |

Across the four valid loads, policy8 won E2E for 87/96 pairs (90.625%) with
a median delta of -86.388 ms, and TPOT for 92/96 pairs (95.833%) with a median
delta of -3.118 ms.

At rate52, policy11 improved on policy8 in the same experimental design:

- E2E: 23/24 wins, median -136.620 ms.
- TPOT: 24/24 wins, median -5.236 ms.
- Exact route partition: TEMPO 24 local/0 remote; LMCache 0 local/24 remote.
- All raw stream, output, and router-decision validity gates passed.

At rate56, the official LMCache concurrent-retrieval path terminated with its
internal cache-key assertion and 16 invalid streams.  This is a stability
observation, not a performance result, so rate56 remains outside the validated
controller frontier.

## Negative results retained

Policy10 removed only `(512,32)` from policy8's remote buckets.  It still beat
LMCache in a rate48 same-window run, but was weaker than policy8: 20/24 E2E
wins and -84.048 ms median versus policy8's 20/24 and -119.700 ms.  It is not
the normal-load policy.

The arm-separated rate52 run did not establish robust standalone throughput
superiority for policy8: 3.7889 versus LMCache 3.7972 requests/s (-0.218%),
although E2E p99 improved by 49.09 ms and TPOT p99 improved by 29.20 ms.
Policy10 produced +1.890% throughput versus its LMCache arm, but that separate
lifecycle's LMCache throughput was itself 3.25% lower and policy10 lost to the
fixed-local arm.  Therefore the final claim remains a same-window request-level
latency/tail advantage, not standalone capacity superiority.

## Claim boundary and next step

Actual requests, model, GPU budget, potential KV geometry, cache warmup, and
output validation are paired.  Actual transferred bytes differ intentionally:
TEMPO's mechanism may avoid remote transfer.  All measurements share one
allocation and no independent-allocation replication was requested.

Mooncake was not integrated into this actual-vLLM harness, so no Mooncake or
universal SOTA claim is made.  The next scientific step is to replace the
externally frozen offered-rate regime with a predeclared online congestion
signal, then replicate once on an independent allocation.  A Mooncake claim
requires the same P/D topology, model, request stream, cache state, GPU budget,
and correctness contract.

Machine-readable decision:
`results/tempo_pd_final_selection_v289_job_57057488.json`.
