# Frozen TEMPO P/D controller — final allocation result

Selected policy: `qwen25-7b-tp4x2-warm-affinity-8`.

Actual Qwen2.5-7B vLLM P/D, four A100 nodes, two TP4 P/D replicas,
mixed prompt lengths 512/1230/2048/4094 and output lengths
16/32/64/128/256. Each load point used one live-server lifecycle with two
balanced measured blocks per arm.

| Offered rate | Throughput vs LMCache | E2E p99 vs LMCache | TPOT p99 vs LMCache | Paired requests |
|---:|---:|---:|---:|---:|
| 48/s | **+0.434%** | **-2.094%** | **-48.104%** | 23/48, median +4.35ms |
| 56/s | **+3.215%** | **-4.151%** | **-46.066%** | 22/48, median +11.55ms |
| Median | **+1.824%** | **-3.122%** | **-47.085%** | paired gate fails |

Aggregate throughput, E2E-p99, TPOT-p99, SLO, route, and local-non-regression
gates pass at both rates. The separate request-paired majority gate fails at
both rates; the supported claim is aggregate mixed-workload advantage, not
per-request dominance.

The frozen scheme combines:

1. Fail-closed geometry validation.
2. Decoder-local cold paths for outputs 16/128/256 and bounded arrival-regime
   control for validated 32/64-token misses.
3. Immutable pair-local warm cache affinity.
4. A recent-composition guard that keeps `(2048,256)` remote for an
   output256-only epoch but local in a mixed-output epoch.

The final falsification, policy9, additionally suppressed `(512,128)` and
`(2048,64)` under mixed composition. It was worse on throughput, E2E p99,
TPOT p99, paired wins, and paired median, so it was rejected and the code was
restored to policy8.

Validity boundary:

- LMCache seeding was serialized only during unmeasured warmup to avoid its
  observed concurrent KV-ready liveness stall. All measured blocks retained
  the same offered rate, 32 workers, model, workload, cache state, and servers.
- This is two load points in one allocation, not a multi-allocation statistical
  benchmark.
- No same-harness actual-vLLM Mooncake P/D run exists; no Mooncake win is
  claimed.

Primary artifacts:

- `results/tempo_pd_composition_headtohead_v236_job_57052289.json`
- `results/tempo_pd_cross_geometry_policy8_rate56_v243_job_57052289.json`
- `results/tempo_pd_composition_cross_load_v244_job_57052289.json`
- `results/tempo_pd_policy_selection_v240_job_57052289.json`
