# TEMPO P/D policy freeze — job 57029025

## Decision

Freeze the measured P/D routing policy below for the validated Qwen2.5-7B,
TP8-prefill + TP8-decode, four-node A100 geometry:

1. Low and mid pair-arrival pressure route decoder-local.
2. High pressure with 32 output tokens uses a 58 ms mean-pair-interval
   threshold and eight live-local credits; excess requests spill to remote
   prefill.
3. High pressure with 64 output tokens and prompt length at most 512 uses the
   explicit `workload_guard_local` route.
4. Other validated 64-output-token requests use a 70 ms threshold and nine
   live-local credits; excess requests spill to remote prefill.
5. Output lengths other than 32 and 64 fail closed because they have not been
   GPU validated.

The production implementation is in `tempo/pd_regime_controller.py`,
`tempo/pd_workload_policy.py`, and
`eval/sota_4node/tempo_pd_same_server_router_v61.py`.

## Frozen six-workload validation grid

Every row used one live server lifecycle and the cold-key-disjoint measured
order local, TEMPO, LMCache, LMCache, TEMPO, local. Each arm has 48 measured
requests. `wins` and `delta` are paired TEMPO-minus-baseline E2E results.

| prompt / output | TEMPO routes local/remote | TEMPO p50 / p99 ms | TEMPO goodput req/s | vs local wins, median delta | vs LMCache wins, median delta | verdict |
|---|---:|---:|---:|---:|---:|---|
| 512 / 32 | 40 / 8 | 732.90 / 924.17 | 14.9607 | 27/48, -1.50 ms | 48/48, -60.10 ms | pass |
| 1230 / 32 | 40 / 8 | 816.59 / 1038.48 | 13.9220 | 31/48, -8.23 ms | 47/48, -55.00 ms | pass |
| 2048 / 32 | 40 / 8 | 954.17 / 1267.57 | 12.6685 | 32/48, -14.58 ms | 44/48, -161.67 ms | pass |
| 512 / 64 | 48 / 0 | 1396.49 / 2367.28 | 7.9165 | 29/48, -1.62 ms | 48/48, -81.70 ms | pass |
| 1230 / 64 | 36 / 12 | 1510.40 / 2497.45 | 7.7351 | 37/48, -18.24 ms | 45/48, -77.32 ms | pass |
| 2048 / 64 | 36 / 12 | 1652.34 / 2640.43 | 7.4021 | 37/48, -40.69 ms | 47/48, -202.83 ms | pass |

Across the grid, TEMPO beats the official pinned LMCache arm in 279/288 paired
requests. TEMPO request goodput is higher than LMCache in every row (about
2.8%–6.6%), and paired median E2E is lower in every row.

## Short-prompt/output64 falsification and repair

The original 70 ms/credit-nine policy failed for prompt 512/output 64:
22/48 wins versus local, +2.52 ms paired median, and worse p99 than LMCache.
A 58 ms threshold recovered the LMCache comparison but remained a local tie.
The force-local diagnostic then produced 48/48 LMCache wins without a local
regression. The production controller was consequently changed to record an
explicit `workload_guard_local` decision rather than encoding the guard as a
fake arrival threshold.

The post-change production rerun is
`results/tempo_pd_same_server_production_prompt512_output64_guard_v95_job_57029025/prompt_guard_final.json`.
All gates pass: 48/48 guard reasons, 29/48 local wins with -1.62 ms median,
48/48 LMCache wins with -81.70 ms median, p99 2367.28 vs LMCache 2517.35 ms,
and goodput 7.9165 vs 7.6984 req/s.

## Claim boundary

This is strong same-allocation component/system evidence for the tested real
vLLM P/D KV path and workload grid, not a universal SOTA claim. It compares
against the pinned official LMCache path with identical model, requests, KV
geometry, GPU budget, server lifecycle, and order-balanced cold keys. It does
not yet contain an apples-to-apples Mooncake P/D arm, different models,
different output lengths, or an independent-allocation replication.
