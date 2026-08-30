# TEMPO online pair-local P/D result (v321)

## Frozen structure

The production candidate observes every measured arrival on each P/D pair's
local monotonic clock. After four gaps (five observations), it freezes once:

- median gap `<= 39 ms`: high-load local bypass;
- median gap `> 39 ms`: retain the geometry/cache-affinity decision;
- before freezing: retain the existing fail-closed geometry decision.

No timestamps are subtracted across hosts. The two pair routers may freeze to
different regimes. The implementation is
`tempo/pd_online_regime_fast_v311.py` and the actual-vLLM router binding is
`eval/sota_4node/tempo_pd_same_server_online_regime_router_fast_v311.py`.

## Workload validity fix

Official LMCache repeatedly crashed under concurrent repeated prompt chunks
with `CacheEngineKey ... not found in local data` at rates 48 and 52. vLLM
`cache_salt` alone did not change LMCache's internal chunk hash. The final
workload therefore replaces each repeated 18-token region with a globally
unique, punctuation-delimited 18-bit A/B region while preserving every prompt's
exact tokenizer length. Tempo and LMCache still receive paired prompts with the
same geometry and output contract. The tokenizer/uniqueness tests pass.

## Actual-vLLM results

All three runs used allocation `57069588`, Qwen2.5-7B-Instruct, four nodes,
two TP4 P/D pairs, 24 paired Tempo/official-LMCache requests per rate, and the
same-window counterbalanced client.

| Offered rate | Pair regimes | E2E wins | Median E2E delta | TPOT wins | Median TPOT delta |
|---:|---|---:|---:|---:|---:|
| 48 req/s | high + affinity | 22/24 | -157.732 ms | 23/24 | -4.433 ms |
| 50 req/s | high + affinity | 21/24 | -167.858 ms | 23/24 | -4.393 ms |
| 52 req/s | high + high | 22/24 | -182.980 ms | 24/24 | -4.936 ms |

Pooled: E2E `65/72` wins with median `-160.140 ms`; TPOT `70/72`
wins with median `-4.509 ms`; pooled TPOT p90 delta is `-2.607 ms`.
Every per-rate and pooled conservative gate passed.

Machine-readable final evidence:
`results/tempo_pd_online_regime_final_v320_job_57069588.json`.

## Claim boundary

This establishes a promising actual-vLLM P/D component-screen advantage over
official `LMCacheConnectorV1` for these cold, paired workloads and offered
rates. It is one allocation, not independent replication. It is not a
Mooncake comparison, a production throughput study over arbitrary traffic, or
a universal/SOTA claim. The next paper-grade step is an independent allocation
with a real request trace and an end-to-end P/D router baseline, without
retuning the frozen 39 ms threshold.
