# TEMPO C4 frozen negative conclusion

All three candidates passed the live correctness/data-plane checks, but none jointly passed the original 10% median and tail bundle. All three diagnostic phase-oracle policies also failed the full gate. The preregistered stop condition is therefore satisfied without weakening a threshold.

## Candidate summary

| Candidate | Mechanism | Fixed median gain | Predictor median gain | Goodput gain | Paired wins | TPOT p99 regression | Worst regression |
|---|---|---:|---:|---:|---:|---:|---:|
| A | `instant_score_v1` | -2.92% | +3.48% | +10.17% | 68.89% | +44.53% | +2506.4 ms |
| B | `frontend_active_watermark_epoch` | +7.10% | +17.46% | +7.67% | 76.11% | +64.28% | +997.9 ms |
| C | `local_external_endpoint_credit_epoch` | +7.92% | +21.30% | +4.58% | 75.56% | +49.41% | +2278.7 ms |

## Candidate C workload groups

| Workload | Fixed E2E median | TEMPO E2E median | Paired wins | Fixed/TEMPO TPOT p99 | Fixed/TEMPO goodput |
|---|---:|---:|---:|---:|---:|
| C0 cool | 603.7 ms | 445.8 ms | 90.0% | 88.8/71.7 ms | 93.3%/93.3% |
| C1 D-hot | 2128.1 ms | 2189.8 ms | 33.3% | 135.7/136.9 ms | 93.3%/93.3% |
| C2 remote-hot | 817.0 ms | 499.0 ms | 90.0% | 30.3/30.0 ms | 100.0%/100.0% |
| C2 KV-hot | 2380.3 ms | 573.6 ms | 88.3% | 177.8/82.9 ms | 85.0%/86.7% |
| C3 both-hot | 6075.6 ms | 4412.2 ms | 58.3% | 189.0/283.6 ms | 48.3%/60.0% |
| recovery | 1854.2 ms | 465.4 ms | 93.3% | 34.4/32.9 ms | 90.0%/100.0% |

## Verdict

- Independent mechanisms: `True`
- Median+tail joint passes: `0`
- Full phase-oracle passes: `0`
- Reproducible negative conclusion allowed: `True`
- Scope: dynamic contention admission/routing on the frozen four-node C4 workload with unchanged vLLM/LMCache P/D data plane.
- Not claimed: universal LMCache inferiority, a physical switch bottleneck, or impossibility of production-scale orchestration.

## Bound inputs

- Negative analysis: `/pscratch/sd/s/sgkim/Skim-Tempo/results/tempo_pd_c4_semantic_credit_epoch_candidate_v7_job_57362947/negative_conclusion_analysis_v2.json` (`c8cb985aba33724b22c16d1501d9cdbd057d95ea5231b64de23a88d2572cd1f3`)
- Candidate C phase analysis: `/pscratch/sd/s/sgkim/Skim-Tempo/results/tempo_pd_c4_semantic_credit_epoch_candidate_v7_job_57362947/phase_screen_analysis.json` (`f5b9e8994d97974cddde1fc9529ee364f029278226f3fb3a5d1aa5d21374a773`)
- Plots: `candidate_c_pooled_metrics.svg`, `candidate_c_phase_metrics.svg`
