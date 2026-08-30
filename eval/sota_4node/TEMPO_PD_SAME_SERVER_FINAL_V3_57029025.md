# TEMPO P/D same-server final v3 (job 57029025)

This report supersedes the v1/v2 reports for job 57029025.

## Final production controller

The controller observes pair-local arrivals, calibrates from the first three
requests, then freezes the epoch regime:

- low and mid pressure route decoder-local;
- high pressure uses decoder-local ownership up to a credit cap and spills the
  rest to official LMCache remote P/D;
- output 32: high boundary 58 ms, local credit 8;
- output 64: high boundary 70 ms, local credit 9;
- unsupported output lengths fail closed.

The output-aware boundary is required by the data: at output32, 58 ms removed
an unprofitable 30 req/s spill and improved the 32 req/s local tail; at output64,
70 ms outperformed the narrower boundary.

## Experimental contract

- Four A100 nodes; TP8 prefill on two nodes and TP8 decode on two nodes.
- One live vLLM server lifecycle per campaign.
- Qwen2.5-7B-Instruct, identical GPU budget, request schedule, KV geometry, and
  official pinned LMCache transport/cache path.
- Cold, token-count-preserving, disjoint keys for every arm/block.
- Six-block crossover: local, TEMPO, LMCache, LMCache, TEMPO, local.
- Exact normalized schedule, route, streamed output, and token equivalence.

## Production output32 result

Artifact:
`results/tempo_pd_same_server_production_v86_job_57029025/production_final.json`

| Arm | Local/remote | E2E p50 ms | E2E p99 ms | TPOT p99 ms | Goodput req/s |
|---|---:|---:|---:|---:|---:|
| Fixed local | 48 / 0 | 820.574 | 1076.041 | 24.766 | 13.9079 |
| TEMPO | 40 / 8 | 816.594 | 1038.476 | 31.653 | 13.9220 |
| LMCache remote | 0 / 48 | 896.238 | 1101.024 | 33.821 | 13.5105 |

TEMPO beat local 31/48 with a -8.234 ms paired median and LMCache 47/48
with a -55.000 ms median. Goodput and E2E p50/p99 beat both baselines. All
predeclared correctness, route, paired, tail, and goodput gates passed.

## Production output64 result

Artifact:
`results/tempo_pd_same_server_production_output64_v87_job_57029025/output64_final.json`

| Arm | Local/remote | E2E p50 ms | E2E p99 ms | TPOT p99 ms | Goodput req/s |
|---|---:|---:|---:|---:|---:|
| Fixed local | 48 / 0 | 1514.953 | 2478.255 | 23.353 | 7.6970 |
| TEMPO | 36 / 12 | 1510.395 | 2497.447 | 37.986 | 7.7351 |
| LMCache remote | 0 / 48 | 1588.462 | 2635.975 | 41.010 | 7.4994 |

TEMPO beat local 37/48 with a -18.242 ms paired median and LMCache 45/48
with a -77.317 ms median. Goodput and p50 beat both; all gates passed.

## Arrival-regime evidence

At 24 req/s the old `mid -> remote` branch was falsified: it routed 44/48
remote and lost to local by +58.252 ms median. The corrected mid-local policy
retained 98.94% local goodput, had a -1.513 ms paired median, and beat LMCache
40/48 by -33.096 ms. Artifact:
`results/tempo_pd_same_server_balanced_midlocal_v79_job_57029025/midlocal_final.json`.

At 16 req/s the low-local policy had a -1.807 ms paired median versus local,
retained 99.27% local goodput, and beat LMCache 18/18 by -72.015 ms. Artifact:
`results/tempo_pd_same_server_balanced_lowlocal_v82_job_57029025/lowlocal_final.json`.

At 30 req/s the former 70 ms boundary spilled 16 requests and lost to local
(18/48 wins, +6.804 ms). The 58 ms boundary routed all local, won 39/48 by
-8.661 ms, and beat LMCache 38/48 by -30.428 ms. Artifact:
`results/tempo_pd_same_server_threshold58_rate30_v84_job_57029025/same_server_final.json`.

## Falsified alternatives

- Output32 credit 9 reduced useful spill and was slower.
- Credit 7 improved one diagnostic but failed independent production
  replication (22/48 local wins, +5.566 ms median); it was overfit.
- Credit 6 degraded tail, goodput, and paired wins.
- A universal 70 ms boundary spilled too early at 30 req/s.
- A universal 58 ms boundary at output64 was only 24/48 versus local with a
  +1.041 ms median, worse than the output64-specific 70 ms policy.

## Verification and claim boundary

The controller/policy unit suite passes 11/11 and `git diff --check` passes.
The evidence establishes a same-harness advantage over pinned official LMCache
remote P/D on the tested high/mid/low and output32/output64 workloads. It is not
a universal SOTA claim. Mooncake is excluded from the direct table because the
installed environment lacks the official same-lifecycle router/integration
needed for this topology; its available component benchmark is not comparable
to this actual-vLLM-owned-KV P/D workload.
