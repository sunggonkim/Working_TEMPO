# TEMPO Elastic-PD v449 evidence

## Frozen structure

- One-way ingress route commit occurs before either upstream starts.
- Four arms share one live vLLM server epoch: always-local, official LMCache
  always-remote, profile predictor, and full TEMPO.
- Full TEMPO uses weighted local-compute and remote-KV credits, arrival-gap
  hysteresis, explicit recovery-probe state, bounded queue retry, and exact
  cache-residency branches.
- Local prefill and remote handoff credits are released on the first streamed
  response chunk. They are not held through autoregressive decode.
- Unknown/unverified cache state fails closed to `confirmed_miss`; the measured
  workload intentionally used unique first-chunk markers and exercised only
  this conservative cache state.

## Authoritative artifacts

- Profile: `eval/sota_4node/real_tempo_pd_elastic_profile_v447.json`
  (`640c8789457b7e0b3971f8f408d8b195048128c4b429b399a60904786bf2b052`).
- Raw four-node result:
  `results/tempo_elastic_pd_v449_job_57086357/tempo_elastic_pd_v445/`.
- Strengthened analysis:
  `results/tempo_elastic_pd_v449_job_57086357/elastic_pd_final_v450.json`.
- Allocation: four A100 nodes, actual Qwen2.5-7B vLLM TP4 P/D with two
  replicas, official `LMCacheConnectorV1` remote path, one live server epoch.

## Result

All measured streams, routes, geometries, and outputs were exact. TEMPO routed
42/48 requests to local chunked prefill and 6/48 to official LMCache remote
prefill. Official LMCache always-remote served all 48 remote requests.

Against the paired official-LMCache arm, TEMPO achieved:

- E2E median delta: **-209.356 ms**.
- E2E wins: **45/48 (93.75%)**.
- Worst paired E2E regression: **+79.562 ms**, below the predeclared 100 ms
  guardrail.
- Median TPOT: **21.789 ms** versus **22.087 ms**.
- Worst-request TPOT-max: **531.945 ms** versus **764.046 ms**.

The phase-correct credit invariant held for 48/48 TEMPO requests: every credit
was released at the first response chunk before stream completion, four
requests retried from the bounded queue, and no request ended queued or with a
route error. Compared with v446, the first-response lease fix improved E2E wins
from 42/48 to 45/48, reduced worst E2E regression from 192.362 ms to 79.562 ms,
and reduced worst TPOT-max regression from 92.808 ms to 3.297 ms.

## Claim boundary

This proves a same-allocation component/system screen on the stated topology:
the ingress admission policy outperformed official LMCache always-remote for
this frozen workload while preserving exact output and bounded tail
regression. It does **not** establish a new transport, independent replication,
live cache-hit behavior, generalization to other models/topologies, or a
Mooncake comparison. Cache-hit branches and hysteretic recovery are covered by
CPU invariants; the GPU workload exercised confirmed misses and the
`remote_stable`/`deflect_active` regimes, not a recovery probe.
