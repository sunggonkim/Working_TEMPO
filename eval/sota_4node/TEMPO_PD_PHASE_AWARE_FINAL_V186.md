# TEMPO phase-aware P/D result (v186)

## Outcome

The production `HybridPDController` is validated on actual Qwen2.5-7B vLLM
P/D execution across four A100 nodes.  Its frozen policy is:

1. On a cold cache miss, apply the workload/decode-pressure admission policy.
   The validated heterogeneous workload selected decoder-local execution.
2. On a warm seed, bind each stable cache item to a calibrated local or remote
   placement.
3. On every warm hit, reuse that placement; never silently convert a nominal
   hit into a miss on another engine pair.

The same-epoch run used one live server lifecycle and observed all three states:
24 cold misses, 24 warm seeds, and 48 measured warm hits.  All v186 gates pass.

## Same-epoch rate-56 result

| Metric | TEMPO | LMCache remote | Fixed local |
|---|---:|---:|---:|
| Request throughput (req/s) | 6.5571 | 6.2761 | 6.5214 |
| E2E p99 (ms) | 3298.12 | 3440.80 | 3299.58 |
| TPOT p99 (ms) | 30.37 | 40.55 | 26.86 |

TEMPO improves throughput by 4.48% and E2E p99 by 4.15% over the pinned
LMCache always-remote baseline.  It wins 34/48 paired requests with a median
E2E delta of -68.83 ms.  Throughput and E2E p99 also narrowly beat the
fixed-local oracle.  The cold transition completed 24/24 requests without an
error and routed all 24 locally.  Warm seed and both hit replicates preserved
the exact 16-local/8-remote placement per 24-item catalog.

Primary artifact:
`results/tempo_pd_same_server_hybrid_phase_v186_job_57051102.json`.

## Reproduction evidence

The rate-48 warm-hit production result was independently reproduced on two
four-node allocations.  Across those allocations, the median improvements over
LMCache were 2.56% throughput, 1.73% E2E p99, and 24.37% TPOT p99.  Every run
also remained within 0.1% of the fixed-local E2E p99.

Cross-allocation artifact:
`results/tempo_pd_production_cross_allocation_v169.json`.

The separate cold-miss screen selected local for 48/48 measured requests.  It
beat LMCache remote by 4.74% throughput, 4.35% E2E p99, and 42.75% TPOT p99,
while remaining within 2% of the fixed-local oracle on all three metrics.

Unified cold/warm artifact:
`results/tempo_pd_hybrid_phase_v180.json`.

## Claim boundary

This is an actual-vLLM, actual-model, actual-KV P/D result, not the earlier
sidecar component screen.  The comparison holds model, prompt/output workload,
GPU budget, server lifecycle, and request order constant.  The LMCache remote
arm uses the pinned official LMCache checkout and the same UCX/NIXL path.

This does not establish a universal SOTA claim.  It is a four-node A100,
Qwen2.5-7B, TP4+TP4 result for the frozen workload and load points.  Mooncake
was not included as a direct system baseline because the installed environment
lacks a compatible production router/config for an apples-to-apples same-harness
run; the existing Mooncake component throughput is not comparable to this P/D
latency experiment.

## Re-run

Inside an explicitly approved four-node allocation:

```bash
export TEMPO_PD_SAME_SERVER_APPROVED=YES
bash eval/sota_4node/run_tempo_pd_same_server_hybrid_phase_v185_rate56_in_allocation.sh \
  results/tempo_pd_heterogeneous_input_v115 \
  results/tempo_pd_same_server_hybrid_phase_v185_rate56_job_${SLURM_JOB_ID}
```

Then validate:

```bash
PYTHONPATH=. .vllm_venv/bin/python -m \
  eval.sota_4node.analyze_tempo_pd_same_epoch_phase_v186 \
  --root results/tempo_pd_same_server_hybrid_phase_v185_rate56_job_${SLURM_JOB_ID} \
  --allocation "${SLURM_JOB_ID}" \
  --output results/tempo_pd_same_server_hybrid_phase_v186_job_${SLURM_JOB_ID}.json
```
