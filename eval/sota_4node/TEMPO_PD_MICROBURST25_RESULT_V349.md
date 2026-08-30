# TEMPO P/D microburst-credit result (v349)

## Frozen controller

The frozen policy is `tempo-pd-online-regime-microburst25-credit5-342`.
It combines three decisions, all made from pair-local observations:

1. Collect five pair-local inter-arrival gaps and classify load at 39 ms.
2. Prefer decoder-local execution in the high-load regime; otherwise retain
   geometry affinity.
3. If the initial pair median is at most 25 ms, activate a local decoder
   credit of five. A request arriving with five local requests already in
   flight is sent to the normal remote P/D path. The credit is disabled for
   non-microburst traffic.

The controller therefore does not impose a global fixed cap. It adds backpressure
only when the observed traffic shape is a microburst.

## Validation

Both measurements ran in allocation `57074923` on four Perlmutter nodes and
used actual vLLM P/D requests. The comparison arm used the official
`LMCacheConnectorV1` path. Every request used an isolated, globally unique
18-token region so that LMCache could not turn repeated internal chunks into
accidental warm-cache hits.

The fail-closed combined report is:

`results/tempo_pd_microburst25_suite_v348_job_57074923.json`

| Workload | Tempo route | LMCache route | E2E wins | Median E2E delta | TPOT wins | Median TPOT delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Six bursts, four pairs/burst | 22 local, 2 remote | 24 remote | 23/24 | -84.569 ms | 23/24 | -3.483 ms |
| Steady rate-52 crossover | 24 local, 0 remote | 24 remote | 23/24 | -112.965 ms | 24/24 | -4.789 ms |
| Pooled | 46 local-equivalent wins / 48 pairs | 48 remote | 46/48 | -108.632 ms | 47/48 | -4.219 ms |

In the burst trace the pair-local medians were 9.078 and 13.381 ms. The
microburst branch activated for 20 measured Tempo requests and capped exactly
items 14 and 15, both observed at local depth five. In the steady trace the
pair-local medians were 36.331 and 38.101 ms, so the credit branch did not
activate and no request was capped.

Both workload reports passed all predeclared gates:

- exact controller provenance and route accounting;
- negative paired E2E median with at least 80% wins;
- negative paired TPOT median;
- non-regressing paired TPOT p90;
- burst-only activation and steady deactivation of the credit rule.

The combined report additionally requires at least 44/48 E2E wins and at
least 46/48 TPOT wins. The observed counts were 46/48 and 47/48 respectively,
so its verdict is `freeze_microburst25_credit5_controller`.

## Interpretation and claim boundary

This is a validated workload-adaptive P/D controller and a same-harness
component/system-screen win over the official LMCache path for the two tested
traffic shapes. It is stronger than the earlier global-cap variants: cap five
fixed the burst losses, while automatic deactivation avoided the steady-load
TPOT regression caused by applying that cap globally.

It is not yet a universal or Mooncake SOTA claim. The active environment has
no `mooncake` or `mooncake_transfer_engine` Python runtime, and the repository
does not currently contain a safe same-vLLM explicit-topology Mooncake adapter.
The official Mooncake Python initialization also performs device discovery,
which is outside the repository's bounded Perlmutter safety contract. An
apples-to-apples Mooncake result therefore requires a separately installed,
explicit-topology adapter and the same request/KV-byte/GPU-budget harness.

The next scientific validation is one independent four-node allocation with
the frozen policy and no retuning. If that repeats the direction, the controller
can be promoted as the default TEMPO admission policy while Mooncake parity is
developed as a separate transport integration.
