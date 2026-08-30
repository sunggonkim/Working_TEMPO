# TEMPO actual-vLLM P/D result (allocation 57057488)

## Decision

Freeze `qwen25-7b-tp4x2-warm-affinity-8` as the current controller.  The
one-factor policy10 revision was rejected: at 48 requests/s it kept the E2E
win count at 20/24 but reduced the median E2E advantage from 119.700 ms to
84.048 ms and reduced the median TPOT advantage from 4.097 ms to 3.762 ms.

The frozen policy routes a validated warm cache item immutably.  Its remote
buckets are `(512,32)`, `(512,64)`, `(512,128)`, `(2048,64)`, and
`(2048,256)`; other validated buckets use decoder-local recomputation/cache.
The mixed-composition guard may demote `(2048,256)` to local.  Unknown
geometries fail closed.

## System and comparison

- Four Perlmutter A100 nodes, four GPUs per node.
- Actual vLLM Qwen2.5-7B-Instruct P/D: TP4 prefill plus TP4 decode, two
  replicas across the allocation.
- vLLM 0.26.0, torch 2.11.0+cu129, NIXL `nixl_cu12._api`, UCX.
- Official editable LMCache `227d13f5c9fdb52ddb933641d34331f678de03a0`.
- Repository base revision `e51affeb0d9bb3c55d5f2ccc1cb6ea24cce95ba5`.
- Model config SHA256
  `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c`.

At each offered load, one server lifecycle first seeded the cache and then
issued a counterbalanced 48-request window.  Each of 24 workload items had
one TEMPO request and one official-LMCache remote-P/D request, with nonce
assignment reversed by item parity.  All reported streams, router decisions,
and output validations passed.  LMCache routed 24/24 paired requests remotely;
TEMPO routed 5 remotely and 19 to decoder-local recomputation/cache.

## Result

Negative deltas favor TEMPO.

| Offered rate | E2E wins | E2E median delta | TPOT wins | TPOT median delta |
|---:|---:|---:|---:|---:|
| 16 req/s | 23/24 | -78.070 ms | 24/24 | -0.926 ms |
| 32 req/s | 22/24 | -78.112 ms | 23/24 | -1.266 ms |
| 48 req/s | 20/24 | -119.700 ms | 22/24 | -4.097 ms |
| 52 req/s | 22/24 | -85.382 ms | 23/24 | -4.875 ms |
| Pooled | 87/96 (90.625%) | -86.388 ms | 92/96 (95.833%) | -3.118 ms |

The one-sided request-pair sign probabilities are `1.82e-17` for E2E and
`4.38e-23` for TPOT.  These values describe the sampled request pairs; they
do not eliminate within-allocation or workload dependence.

At 56 requests/s the official LMCache concurrent-retrieval path terminated
with its internal `pd_backend_async.get_blocking` cache-key assertion and 16
invalid streams.  That artifact is stability evidence only, not a performance
comparison.  The valid frontier is therefore 52 requests/s and the observed
LMCache failure point is 56 requests/s for this exact workload and build.

Canonical machine-readable result:
`results/tempo_pd_mixed_frontier_v280_job_57057488.json`.

## Claim boundary

This establishes a same-window, request-level advantage over the official
LMCache P/D baseline on the tested actual-vLLM topology.  It is a system-policy
comparison: requests and potential KV geometry are paired, while actual
transport bytes intentionally differ because TEMPO can avoid remote transfer
and recompute locally.  Mixed windows do not establish standalone throughput
superiority, the four loads share one allocation, and no independent
allocation replication was requested.

Mooncake was not run in this same harness.  Its existing component benchmark
uses a different topology and cannot support a direct claim.  Accordingly,
this result is not a Mooncake win, a universal SOTA claim, or evidence for
unvalidated models, cache geometries, or arrival processes.

## Reproduction artifacts

- Controller: `tempo/pd_cache_affinity.py`, SHA256
  `f627b934c58e68540ae9132f7be71ca05935bb1c089f092c3fb5a2ee4543ba77`.
- Hybrid controller: `tempo/pd_hybrid_controller.py`, SHA256
  `9c1b2b11326d0f59ff445b34bb68edeba8608a9ef5fb2c8d91ddb3910424a5ba`.
- Frontier analyzer: `eval/sota_4node/analyze_tempo_pd_mixed_frontier_v280.py`,
  SHA256 `81b3d132955b9956c38fac03faccf6f88a7d45beb0e305a7586e1e11e51abb08`.
- Per-load reports:
  `results/tempo_pd_mixed_only_rate{16,32,48,52}_v*_job_57057488/hybrid_controller_final.json`.
- Rate-56 failure report:
  `results/tempo_pd_mixed_only_rate56_failure_v272_job_57057488.json`.

The next publication-grade step is an independent-allocation replication and
a separate same-harness standalone capacity curve.  A Mooncake comparison is
valid only after its actual vLLM P/D connector is integrated with the same
model, requests, GPU budget, cache state, and correctness checks.
