# TEMPO P/D final same-server evidence v2 (job 57029025)

This report supersedes `TEMPO_PD_SAME_SERVER_FINAL_57029025.md`. A same-server
mid-load falsification changed the final controller structure after that report
was written.

## Final controller

The controller calibrates from the first three pair-local arrivals and freezes
the regime for the epoch:

- mean pair interval greater than 110 ms: decoder-local;
- mean pair interval from 70 through 110 ms: decoder-local;
- mean pair interval at most 70 ms: bounded decoder-local ownership followed by
  remote-prefill spill;
- high-regime local credit: 8 at 32 output tokens and 9 at 64 output tokens;
- other output lengths fail closed.

In short, remote P/D is admitted only under measured high arrival pressure.
This replaces the earlier `mid -> remote` branch, which the same-server
experiment falsified.

## Method

- Four A100 nodes and one live vLLM server lifecycle per campaign.
- TP8 prefill on two nodes and TP8 decode on two nodes.
- Qwen2.5-7B-Instruct, identical request schedule, output length, KV geometry,
  GPU budget, and official pinned LMCache data path.
- Cold, token-count-preserving, disjoint cache keys for every measured block.
- Order-balanced crossover: local, TEMPO, LMCache, LMCache, TEMPO, local.
- Exact normalized workload/schedule/output equivalence and route assertions.

## High load, output 32

Final artifact:
`results/tempo_pd_same_server_balanced_credit8_final_v76_job_57029025/same_server_final.json`

| Arm | Local/remote routes | E2E p50 | E2E p99 | TPOT p99 | Goodput req/s |
|---|---:|---:|---:|---:|---:|
| Fixed local | 48 / 0 | 825.657 | 1077.354 | 24.998 | 13.8792 |
| TEMPO | 32 / 16 | 829.119 | 1080.905 | 33.112 | 13.9518 |
| LMCache remote | 0 / 48 | 918.989 | 1241.526 | 38.376 | 13.3844 |

TEMPO beat local 32/48 with a -12.529 ms paired median and LMCache 46/48
with a -75.961 ms paired median. All gates passed.

Across four credit-8 same-server campaigns (forward, reverse, balanced, final),
TEMPO beat local 107/144 with a -17.924 ms paired median and LMCache 135/144
with a -70.483 ms paired median. Goodput beat both baselines in all four.

Credit 9 was slower. Credit 7 improved one diagnostic but failed independent
production replication against local (22/48 wins, +5.566 ms median, slightly
lower goodput). Credit 6 degraded tail, goodput, and paired wins. Therefore the
production high-load credit remains 8 rather than the best single-run setting.

## High load, output 64

Artifact:
`results/tempo_pd_same_server_balanced_output64_v77_job_57029025/output64_final.json`

| Arm | Local/remote routes | E2E p50 | E2E p99 | TPOT p99 | Goodput req/s |
|---|---:|---:|---:|---:|---:|
| Fixed local | 48 / 0 | 1530.135 | 2517.960 | 23.741 | 7.6732 |
| TEMPO | 36 / 12 | 1543.951 | 2491.012 | 38.677 | 7.7043 |
| LMCache remote | 0 / 48 | 1632.569 | 2614.413 | 40.653 | 7.4580 |

TEMPO beat local 32/48 with a -19.483 ms paired median and LMCache 46/48
with a -57.712 ms median. All gates passed.

## Mid load: falsification and correction

The old `mid -> remote` policy failed the same-server crossover:

- artifact:
  `results/tempo_pd_same_server_balanced_midload_v78_job_57029025/midload_final.json`;
- routes: 4 local / 44 remote;
- only 10/48 wins versus local, +58.252 ms paired median;
- goodput lower than local.

After changing mid to decoder-local, the verification passed:

- artifact:
  `results/tempo_pd_same_server_balanced_midlocal_v79_job_57029025/midlocal_final.json`;
- exact 48/48 local routes;
- local paired median -1.513 ms, p99 888.431 versus 900.316 ms;
- retained 98.94% of local goodput;
- beat LMCache 40/48 with a -33.096 ms median.

## Low load

Artifact:
`results/tempo_pd_same_server_balanced_lowlocal_v82_job_57029025/lowlocal_final.json`

TEMPO routed all 18 requests local, had a -1.807 ms paired median versus the
fixed-local arm with no p99 regression, retained 99.27% of local goodput, and
beat LMCache 18/18 with a -72.015 ms median. All gates passed.

## Verification

The controller and frozen workload policy tests pass 11/11, including low,
mid-local, high spill, release/credit reuse, 32-token credit 8, 64-token credit
9, and fail-closed unsupported output lengths. `git diff --check` passes.

## Claim boundary

These results establish a same-harness advantage over the pinned official
LMCache remote-P/D baseline for the tested high/mid/low regimes and 32/64
output-token workloads. They are not a universal SOTA claim. Mooncake is not a
direct baseline here: the installed environment lacks the required official
same-lifecycle router/integration path, while the available component benchmark
uses different topology and semantics. Its component throughput is therefore
reported separately, not compared as if it were this P/D workload.
