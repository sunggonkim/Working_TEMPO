# Output16 direct-local candidate — allocation 57029025

## Candidate

For Qwen2.5-7B TP8-prefill + TP8-decode on four A100 nodes, route output16
requests with prompt length at most 2048 directly to decoder-local. This is a
fast-path decision: it bypasses arrival-regime calibration because all three
measured prompt buckets select the same route.

Production has **not** been changed yet. Output16 remains fail-closed until an
independent allocation reproduces the frozen v103 interleaved gates.

## Request-interleaved latency evidence

The v102 run Latin-interleaves 48 local, 48 TEMPO, and 48 LMCache requests in a
single common measurement window.

| prompt | TEMPO vs local wins / median | TEMPO vs LMCache wins / median |
|---|---:|---:|
| 512 | 12/16, -11.77 ms | 16/16, -36.30 ms |
| 1230 | 9/16, -6.12 ms | 16/16, -62.78 ms |
| 2048 | 8/16, -0.78 ms | 16/16, -79.14 ms |

Aggregate TEMPO p50/p99 is 449.15/576.45 ms versus LMCache
511.89/681.39 ms. All 48 paired LMCache requests lose to TEMPO. The frozen
next-allocation analyzer is
`eval/sota_4node/analyze_tempo_pd_interleaved_local_fast_v103.py`.

## Saturated throughput evidence

The v104 run uses one live server lifecycle and order-balanced saturated arm
blocks with identical cold-key-disjoint requests.

| arm | throughput/goodput req/s | p50 ms | p99 ms | TPOT p99 ms |
|---|---:|---:|---:|---:|
| fixed local | 21.4671 | 466.80 | 549.01 | 32.51 |
| TEMPO direct-local | 21.5456 | 466.04 | 556.98 | 33.09 |
| LMCache remote | 18.2556 | 480.53 | 694.59 | 41.45 |

TEMPO improves saturated request throughput/goodput by about 18.0% and p99 by
about 19.8% versus LMCache. It retains local behavior: +0.37% throughput and
-0.30 ms paired median E2E relative to fixed local. Every predeclared v104 gate
passes.

## Evidence and next action

- interleaved raw/result:
  `results/tempo_pd_same_server_interleaved_local_fast_v102_job_57029025/`
- saturated result:
  `results/tempo_pd_same_server_output16_saturated_v104_job_57029025/saturated_final.json`
- next independent launcher:
  `eval/sota_4node/run_tempo_pd_same_server_interleaved_local_fast_v103_in_allocation.sh`
- next saturated launcher:
  `eval/sota_4node/run_tempo_pd_same_server_output16_saturated_v104_in_allocation.sh`

On the next allocation, run v103 first without changing code or thresholds. If
it passes, run v104. Only if both pass should the output16 direct-local branch
be added to `FrozenPDPolicy` and verified once through the production router.
