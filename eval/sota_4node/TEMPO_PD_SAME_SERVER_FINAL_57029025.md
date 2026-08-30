# TEMPO P/D same-server final evidence (job 57029025)

## Frozen scheme

The production controller remains the arrival-regime policy in
`tempo/pd_workload_policy.py`:

- calibrate from the first three pair-local arrivals;
- high-pressure boundary: 70 ms mean pair interval;
- mid-pressure boundary: 110 ms;
- low pressure routes decoder-local;
- mid pressure routes remote prefill;
- high pressure uses bounded decoder-local ownership and spills to remote prefill;
- the frozen high-pressure local credit is 8 for 32 output tokens and 9 for 64;
- all other output lengths fail closed.

The P/D experiment uses one live vLLM server lifecycle on four A100 nodes, with
TP8 prefill and TP8 decode, the same Qwen2.5-7B model, request schedule, KV
geometry, GPU budget, and official pinned LMCache data path. Every arm uses
cold, token-count-preserving, disjoint cache keys. The order-balanced crossover
is local, TEMPO, LMCache, LMCache, TEMPO, local, giving 48 requests per arm.

## Final output-32 verification

Artifact:
`results/tempo_pd_same_server_balanced_credit8_final_v76_job_57029025/same_server_final.json`

| Arm | Routes local/remote | E2E p50 ms | E2E p99 ms | TPOT p99 ms | Goodput req/s |
|---|---:|---:|---:|---:|---:|
| Fixed local | 48 / 0 | 825.657 | 1077.354 | 24.998 | 13.8792 |
| TEMPO | 32 / 16 | 829.119 | 1080.905 | 33.112 | 13.9518 |
| LMCache remote | 0 / 48 | 918.989 | 1241.526 | 38.376 | 13.3844 |

TEMPO beat local on 32/48 paired requests with a -12.529 ms median E2E delta,
and beat LMCache on 46/48 with a -75.961 ms median delta. All predeclared
correctness, route, output-equivalence, goodput, paired-majority, and LMCache
tail gates passed.

## Reproducibility across output-32 campaigns

Four same-server campaigns used the frozen credit-8 policy: forward order,
reverse order, the first balanced crossover, and the final balanced crossover.
Across their 144 paired requests, TEMPO beat local 107/144 with a -17.924 ms
median E2E delta, and beat LMCache 135/144 with a -70.483 ms median delta.
TEMPO goodput exceeded both baselines in every campaign:

| Campaign | vs local | vs LMCache |
|---|---:|---:|
| Forward | +2.265% | +1.820% |
| Reverse | +2.569% | +6.315% |
| Balanced 1 | +2.135% | +3.565% |
| Balanced final | +0.523% | +4.239% |

The reverse-order campaign did not beat local on aggregate unpaired p50/p99,
which is why the claim is based on paired cold-key requests, goodput, and the
order-balanced crossovers rather than on one favorable ordering.

## Output-64 verification

Artifact:
`results/tempo_pd_same_server_balanced_output64_v77_job_57029025/output64_final.json`

| Arm | Routes local/remote | E2E p50 ms | E2E p99 ms | TPOT p99 ms | Goodput req/s |
|---|---:|---:|---:|---:|---:|
| Fixed local | 48 / 0 | 1530.135 | 2517.960 | 23.741 | 7.6732 |
| TEMPO | 36 / 12 | 1543.951 | 2491.012 | 38.677 | 7.7043 |
| LMCache remote | 0 / 48 | 1632.569 | 2614.413 | 40.653 | 7.4580 |

TEMPO beat local on 32/48 paired requests with a -19.483 ms median delta and
LMCache on 46/48 with a -57.712 ms median delta. All gates passed.

## Rejected variants

- Output-32 credit 9 reduced remote spill and was slower.
- Credit 7 looked better in one diagnostic crossover, but failed independent
  production-policy replication against local (22/48 wins, +5.566 ms median,
  slightly lower goodput). It was rejected as overfit.
- Credit 6 degraded p99, TPOT, goodput, and paired local wins relative to 7.

This falsification sequence is why production stays at credit 8 rather than the
best single-run setting.

## Claim boundary

The evidence supports a same-harness win over the pinned official LMCache
remote-P/D baseline for these 32- and 64-output-token workloads. It does not
establish universal SOTA. Mooncake was not included in this same-lifecycle
comparison because the installed environment lacks the required official
router/integration path and the available component benchmark has different
topology and semantics. Its prior throughput number is therefore not used as
a direct baseline.
