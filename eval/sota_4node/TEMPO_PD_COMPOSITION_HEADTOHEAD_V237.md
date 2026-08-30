# TEMPO actual-vLLM P/D composition controller

Frozen controller: `tempo-pd-hybrid-controller-2` with warm-affinity policy
`qwen25-7b-tp4x2-warm-affinity-8`.

## Same-lifecycle head-to-head

Allocation 57052289, four A100 nodes, Qwen2.5-7B-Instruct, two TP4 P/D
replicas, request rate 48/s, 24 mixed request geometries repeated twice (48
measured requests per arm), output lengths 16/32/64/128/256 and actual prompt
lengths 512/1230/2048/4094.

| Metric | TEMPO | LMCache always-remote | Fixed local | TEMPO vs LMCache |
|---|---:|---:|---:|---:|
| Request throughput | 3.82360/s | 3.80709/s | 3.82414/s | **+0.434%** |
| E2E p99 | 5945.80 ms | 6072.95 ms | 5944.52 ms | **-2.094%** |
| TPOT p99 | 30.406 ms | 58.590 ms | 32.253 ms | **-48.104%** |
| SLO success | 100% | 100% | 100% | pass |

TEMPO routed 38/48 requests locally and 10/48 through actual LMCache P/D.
It retained 99.986% of fixed-local throughput and stayed within 0.022% of
fixed-local E2E p99 while improving its TPOT p99 by 5.73%.

The aggregate primary gates pass. The separate request-paired gate does not:
TEMPO wins 23/48 request pairs and the paired E2E median is +4.348 ms versus
LMCache. This run therefore supports an aggregate mixed-workload advantage,
not a claim that every request class or a majority of individual pairs wins.

## Policy learned from experiments

- Cold misses with outputs 16/128/256 use decoder-local execution; validated
  32/64-token misses use the bounded arrival-regime controller.
- Warm cache items receive immutable pair-local placement.
- Calibrated remote buckets are `(512,32)`, `(512,64)`, `(512,128)`,
  `(2048,64)`, and `(2048,256)`.
- The composition guard keeps `(2048,256)` remote in an output256-only epoch,
  but suppresses it to local when the bounded recent warm-seed window contains
  another output class. This one change converted the mixed controller from
  -4.51% to +0.43% throughput versus LMCache and from +3.63% to -2.09% E2E p99.
- Unvalidated prompt/output geometries fail closed.

## Validity and claim boundary

- The LMCache cache seed was serialized only in the unmeasured warmup to avoid
  an observed KV-ready liveness stall. All six measured crossover blocks kept
  the same rate 48/s, 32 workers, cache state, model, bytes, and live servers.
- Corrected metadata validation uses router-declared prompt tokens because the
  proxy usage count can include the returned prefill token. Exact output and
  route/SLO validation passed.
- The stronger cross-geometry result is one four-node allocation and one
  measured lifecycle. It is a component/system screen, not a statistically
  replicated production benchmark.
- No same-harness actual-vLLM Mooncake P/D result exists here. No Mooncake
  superiority claim is made.

## Reproducible artifacts

- Raw lifecycle: `results/tempo_pd_cross_geometry_composition_headtohead_v234_job_57052289/`
- Corrected analysis: `results/tempo_pd_composition_headtohead_v236_job_57052289.json`
- Launcher: `eval/sota_4node/run_tempo_pd_cross_geometry_composition_headtohead_v234_in_allocation.sh`
- Analyzer: `eval/sota_4node/analyze_tempo_pd_composition_headtohead_v236.py`
