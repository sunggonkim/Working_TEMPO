# Frozen TEMPO P/D controller decision

Selected policy: `qwen25-7b-tp4x2-warm-affinity-8`.

The decisive experiment is an actual Qwen2.5-7B vLLM P/D head-to-head on four
A100 nodes, with two TP4 P/D replicas, a mixed 24-row prompt/output workload,
rate 48/s, and two balanced measured blocks per arm.

| Candidate | Routes (local/remote) | Throughput vs LMCache | E2E p99 vs LMCache | TPOT p99 vs LMCache | Paired result |
|---|---:|---:|---:|---:|---:|
| policy8 (selected) | 38/10 | **+0.434%** | **-2.094%** | **-48.104%** | 23/48, median +4.35ms |
| policy9 (rejected) | 44/4 | -0.741% | -0.335% | -38.010% | 11/48, median +72.59ms |

Policy8 passes the aggregate throughput, E2E-p99, TPOT-p99, SLO, exact-route,
and local-non-regression gates. Its request-paired majority gate does not pass,
so the claim is aggregate mixed-workload advantage, not per-request dominance.

Policy9 was the final one-factor falsification: it additionally suppressed the
mixed `(512,128)` and `(2048,64)` remote buckets. It lost throughput and was
worse than policy8 on every selected metric, so the repository was restored to
policy8 after the run.

The frozen scheme is phase-aware admission plus immutable warm cache affinity
and a bounded recent-composition guard for `(2048,256)`. Cold outputs
16/128/256 are local; validated 32/64-token misses use the arrival-regime
controller; unvalidated geometries fail closed.

Evidence:

- `results/tempo_pd_cross_geometry_composition_headtohead_v234_job_57052289/`
- `results/tempo_pd_composition_headtohead_v236_job_57052289.json`
- `results/tempo_pd_cross_geometry_composition_policy9_v238_job_57052289/`
- `results/tempo_pd_policy_selection_v240_job_57052289.json`

No same-harness actual-vLLM Mooncake P/D result exists, so no Mooncake win is
claimed. This is one-allocation system evidence and not yet a multi-allocation
statistical result.
