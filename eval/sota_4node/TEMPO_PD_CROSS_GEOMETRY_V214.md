# TEMPO phase-aware P/D controller: cross-geometry evidence

Verdict: `cross_geometry_lmcache_advantage_validated`.

## Main results

| Workload | Throughput vs LMCache | E2E p99 vs LMCache | TPOT p99 vs LMCache | Paired result |
|---|---:|---:|---:|---:|
| Mixed output16-128, rate 40 | +4.59% | -2.95% | -30.66% | see artifact |
| Mixed output16-128, rate 48 | +2.56% | -1.73% | -24.37% | see artifact |
| Mixed output16-128, rate 56 | +4.48% | -4.15% | -25.11% | see artifact |
| Output256, prompts512/1230/2048 | +1.21% | -0.72% | -0.85% | 40/48, median -93.3 ms |
| Prompt4094, output16/128 | +24.34% | -21.24% | -35.90% | 43/48, median -481.6 ms |

## Frozen policy

Cold misses use the validated local fast path. Warm cache items retain stable placement. Remote buckets are `(512,32)`, `(512,64)`, `(512,128)`, `(2048,64)`, and `(2048,256)`; other validated buckets are local. Prompt4094/4096 is accepted only for output16/128 and is local.

## Boundaries

- Rate64: Tempo/local Pareto and availability screen only; the LMCache arm did not finish, so no rate64 LMCache performance win is claimed.
- Local control: An oracle/control, not the primary distributed-KV baseline. Output256 accepts a measured TPOT tradeoff while remaining within 2% of local throughput and E2E p99.
- Mooncake: No same-harness actual-vLLM Mooncake P/D result exists in this environment; no direct Mooncake superiority claim is made.
