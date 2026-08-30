# TEMPO P/D cache-affinity campaign — allocation 57047614

## Validated configuration

- Four Perlmutter A100 nodes; two independent P/D pairs.
- Qwen2.5-7B-Instruct; TP4 prefill plus TP4 decode per pair.
- vLLM 0.26.0+cu129 and pinned official LMCache data path.
- Arm-isolated stable prompt keys, one warmup and two measured replicates.
- 48 requests per measured arm over prompt lengths 512/1230/2048 and output
  lengths 16/32/64/128.
- TEMPO warm-hit policy: stable per-pair cache affinity, 32 local and 16 remote
  requests. The remote catalog is `(512,32)`, `(512,64)`, `(512,128)`, and
  `(2048,64)` (two requests per bucket in each measured block).

## Reproduced eval-policy result

The initial result and one exact confirmation are pooled in
`results/tempo_pd_same_server_cache_catalog_reproduction_v149_job_57047614.json`.
Every reproduction gate passed.

| metric vs official LMCache | lifecycle 1 | lifecycle 2 | median |
|---|---:|---:|---:|
| request-throughput gain | +0.884% | +2.310% | +1.597% |
| E2E p99 reduction | 0.424% | 3.827% | 2.126% |
| TPOT p99 reduction | 21.288% | 31.145% | 26.217% |
| paired E2E wins | 25/48 | 32/48 | — |

In both lifecycles TEMPO also beat fixed-local on request throughput and E2E
p99. These are two server lifecycles in one allocation, not independent
allocations.

## Production-controller result

The eval policy was moved to `tempo.pd_cache_affinity.CacheAffinityCatalog`
and composed with the cold/miss policy in
`tempo.pd_hybrid_controller.HybridPDController`. The live-router adapter is
`tempo_pd_same_server_hybrid_controller_router_v150`.

At 48 req/s (`...hybrid_controller_v153_job_57047614`):

- throughput: 6.511 TEMPO vs 6.429 LMCache (+1.28%);
- E2E p99: 3256.1 ms TEMPO vs 3275.3 ms LMCache (-0.59%);
- TPOT p99: 30.21 ms TEMPO vs 40.23 ms LMCache (-24.9%);
- paired wins: 32/48, paired median -42.51 ms;
- fixed-local p99 was 3255.8 ms, so TEMPO missed the strict local-superiority
  gate by 0.281 ms (0.009%) while beating local throughput by 0.89%.

At 56 req/s (`...hybrid_controller_v154_rate56_job_57047614`) every gate
passed:

- throughput: 6.630 TEMPO vs 6.399 LMCache (+3.61%) and 6.542 local (+1.35%);
- E2E p99: 3237.1 ms TEMPO vs 3432.9 ms LMCache (-5.71%) and 3274.1 local
  (-1.13%);
- TPOT p99: 29.9 ms TEMPO vs 51.2 ms LMCache (-41.6%);
- paired wins: 32/48, paired median -49.68 ms.

## Rejected and invalid experiments

- 36-local/12-remote with only 10.1% of prompt work offloaded: performance
  regression; rejected.
- 32-local/16-remote with only 13.5% of prompt work offloaded: performance
  regression; rejected.
- 64 req/s with 48 workers and 60 req/s with 32 workers: the LMCache warm arm
  returned HTTP headers but one or more streams did not close. No final
  artifact was produced; these are harness/LMCache liveness failures, not
  performance evidence.
- A dispersed 32/16 catalog was attempted after repeated lifecycle failures in
  the same allocation and hit the same liveness failure. It requires a fresh
  allocation before it can be judged.

## Claim boundary

This is an actual vLLM-owned-KV 1P1D component/system screen and a reproduced
same-allocation win over the official LMCache path for the declared workload.
It is not yet a cross-allocation result or a same-topology Mooncake comparison.
Mooncake was not used as a proxy: the available environment lacks the
production router/config needed for an apples-to-apples 4-node P/D run.
