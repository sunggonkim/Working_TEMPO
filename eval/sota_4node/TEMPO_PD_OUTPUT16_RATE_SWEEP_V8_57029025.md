# Output16 direct-local load sweep — allocation 57029025

## Rate 32 req/s

All v104 gates pass.

- TEMPO throughput/goodput: 21.5456 req/s
- LMCache throughput/goodput: 18.2556 req/s
- TEMPO improvement: about 18.0%
- TEMPO vs LMCache p99: 556.98 vs 694.59 ms (about 19.8% lower)
- paired TEMPO-minus-LMCache median: -44.92 ms
- TEMPO vs fixed-local throughput: +0.37%
- paired TEMPO-minus-local median: -0.30 ms

Artifact:
`results/tempo_pd_same_server_output16_saturated_v104_job_57029025/saturated_final.json`

## Rate 48 req/s

All v104 gates pass without changing the candidate.

- TEMPO throughput/goodput: 23.4700 req/s
- LMCache throughput/goodput: 20.0813 req/s
- TEMPO improvement: about 16.9%
- TEMPO vs LMCache p99: 552.06 vs 779.48 ms (about 29.2% lower)
- paired TEMPO-minus-LMCache median: -90.52 ms
- TEMPO vs fixed-local p99: 552.06 vs 568.08 ms
- paired TEMPO-minus-local median: +0.14 ms

Artifact:
`results/tempo_pd_same_server_output16_saturated_rate48_v105_job_57029025/saturated_final.json`

## Rate 64 req/s capacity boundary

The bounded v106 step stopped during LMCache remote warmup, before measured
candidate blocks. Fixed-local warmup completed. In the 24-request LMCache
warmup, 22 streams completed and two timed out after 600 seconds; raw validation
is fail-closed (`all_streams_valid=false`, `performance_claim_allowed=false`).
This is a capacity-preflight failure, not a measured TEMPO performance win, and
was not retried.

Artifact:
`results/tempo_pd_same_server_output16_saturated_rate64_v106_job_57029025/tempo_credit_admission/same_server_balanced_warm/01_lmcache_remote_r0.raw.json`

## Decision

The output16 direct-local v102 candidate is strong at rates 32 and 48 and the
LMCache remote arm cannot complete the rate64 preflight in this geometry. Still,
production remains unchanged until the frozen v103 request-interleaved gates
are reproduced on an independent allocation. After that replication, rerun the
v104 rate32 saturated comparison; rate48 is supporting evidence and rate64 is a
capacity-boundary result.
