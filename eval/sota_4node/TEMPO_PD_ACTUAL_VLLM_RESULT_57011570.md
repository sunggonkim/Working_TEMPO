# TEMPO-PD actual-vLLM result — allocation 57011570

## Valid experiment

The valid artifact is
`results/tempo_pd_perf_v5_job_57011570/result.json`.

- Model: Qwen2.5-7B-Instruct, BF16.
- Hardware: four Perlmutter A100 nodes, 16 GPUs total.
- Topology: two replicas, each TP4 prefill plus TP4 decode.
- Remote baseline: official LMCacheConnectorV1, PDBackendAsync,
  NIXL/UCX, GPU connector V3.
- Workload: prompt lengths 1,220 / 3,652 / 7,300 tokens; 32 output
  tokens; three samples per bucket; 2 requests/s; four client workers.
- Five fresh lifecycles: local calibration, remote calibration, local
  validation, remote validation, TEMPO validation.
- Correctness: same model and schedule, exact 32-token output text equality
  across all three validation modes, and all streams/router decisions valid.

## Result

TEMPO selected local decode for all 9 validation requests because every frozen
remote-advantage lower bound was negative. Relative to official LMCache
always-remote:

- paired E2E wins: 9/9;
- paired median E2E delta: -222.314 ms;
- E2E p50 reduction: 23.108%;
- E2E p99 reduction: 34.986%;
- TTFT p50 reduction: 18.195%;
- TPOT p50 reduction: 23.893%;
- SLO request-goodput increase: 8.553% (1.86749 vs 1.72035 req/s).

Relative to fixed-local, TEMPO won 7/9 paired E2E observations with a
-14.081 ms paired median and 1.109% higher measured request goodput. Since the
routes are identical, this small difference is run-to-run noise, not a claimed
controller speedup.

The three conservative calibration lower bounds for remote advantage were
-103.874, -206.726, and -419.339 ms. This deployment therefore has no observed
LMCache remote crossover in the frozen workload region.

## Follow-up falsification

`results/tempo_pd_perf_v6_loaded_job_57011570/result.json` doubled offered load
to 4 requests/s and eight workers. Remote remained slower in all three
calibration buckets (best lower bound -123.592 ms). The final comparison was
correctly invalidated because high-load floating-point/batching nondeterminism
changed exact output text, including the first token of the longest bucket.
Its performance values are diagnostic only and are not used as evidence.

Existing Qwen screens with three and seven synchronized decoder background
streams also selected local for every bucket. Repeating stronger decoder load
on this same LMCache/NIXL/UCX path is therefore not justified by current
evidence.

## Claim boundary

This result establishes an actual-vLLM component/system screen in which a
frozen, fail-closed admission controller avoids a harmful official LMCache
remote P/D path and improves aggregate client performance. It does not show a
mixed local/remote crossover, a faster KV transport, an independent replicate,
or a Mooncake comparison. Publication-grade promotion requires a deployment
where at least one exact workload class has independently replicated positive
remote advantage, or a faster live remote transport under the same GPU/byte
budget.
