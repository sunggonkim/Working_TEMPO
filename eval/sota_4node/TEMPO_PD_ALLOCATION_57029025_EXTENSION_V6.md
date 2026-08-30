# TEMPO P/D continuation — allocation 57029025

## Production state retained

The production policy remains the V5 32/64-output-token policy. Output lengths
16 and 128 still fail closed. This allocation did not justify broadening that
boundary under the predeclared gates.

## Output128 falsification

The short-prompt/output128 local diagnostic initially passed and beat LMCache,
but the subsequent production-router run failed local non-inferiority:

| arm | p50 ms | p99 ms | goodput req/s |
|---|---:|---:|---:|
| fixed local | 2819.77 | 5238.00 | 4.0543 |
| TEMPO local guard | 2842.24 | 5414.21 | 4.0493 |
| LMCache remote | 2973.55 | 5597.44 | 3.9207 |

TEMPO still beat LMCache 48/48 with paired median -110.56 ms, but lost to the
same local route 12/48 with +37.37 ms median. The tentative production change
was reverted; output128 remains unsupported.

Artifacts:

- diagnostic: `results/tempo_pd_same_server_output128_diag_v96_job_57029025/output128_final.json`
- production falsification: `results/tempo_pd_same_server_output128_production_v97_job_57029025/output128_final.json`

## Why request-level interleaving was required

Sequential arm blocks showed 30–60 ms temporal drift even when TEMPO and the
fixed-local arm selected the same physical route. A new request-level Latin
interleave sends 48 requests per arm in one common 144-request measurement
window. Each semantic request has two replicates and the immediate arm order is
rotated, reducing block-position confounding.

The first output16 split candidate (remote for prompt <=1536, local otherwise)
was rejected. Interleaved evidence showed that remote routing was slower than
local for both prompt 512 and 1230, while local decisively beat LMCache at 2048.

## Frozen next candidate: output16 direct-local fast path

The final v102 experiment bypasses the arrival controller only for the measured
output16 local decision, retaining the exact P/D server, workload, GPU budget,
and router path.

| prompt | TEMPO vs local wins / median | TEMPO vs LMCache wins / median |
|---|---:|---:|
| 512 | 12/16, -11.77 ms | 16/16, -36.30 ms |
| 1230 | 9/16, -6.12 ms | 16/16, -62.78 ms |
| 2048 | 8/16, -0.78 ms | 16/16, -79.14 ms |

Aggregate p50/p99 is 449.15/576.45 ms for TEMPO, 458.60/577.04 ms for fixed
local, and 511.89/681.39 ms for LMCache. TEMPO therefore wins all 48 paired
LMCache requests while remaining non-inferior to local in every prompt bucket.

The strict artifact remains rejected for two reasons:

1. the v101 analyzer expected the earlier `workload_guard_local` reason rather
   than the truthful v102 `output16_direct_local_fast_path` reason;
2. per-arm first-to-last-window goodput is 9.8329 vs LMCache 9.8540 req/s
   (-0.21%). In an interleaved run this denominator depends on which arm owns
   the first and last scheduled request, so it is not a clean throughput
   comparison.

Do not retroactively promote v102. Freeze it as the next independent-allocation
candidate. The next run must predeclare direct-reason provenance, use the common
global measurement window for interleaved-arm goodput (or report it as equal by
construction), and keep paired E2E/TTFT/TPOT as the primary comparison. A
separate saturated single-arm lifecycle is required for a real throughput and
goodput claim.

Artifact:

- `results/tempo_pd_same_server_interleaved_local_fast_v102_job_57029025/interleaved_local_final.json`
