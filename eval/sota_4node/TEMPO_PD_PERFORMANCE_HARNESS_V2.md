# TEMPO-PD actual-vLLM performance harness v2

This harness measures the admission controller on the actual vLLM P/D path.
It does not substitute a sidecar buffer for live vLLM KV state.

## Frozen comparison

All modes use the same Qwen2.5-7B model, 16-GPU budget, two replicas, TP4
prefill plus TP4 decode per replica, LMCacheConnectorV1, NIXL/UCX, and request
schedule. The five fresh server lifecycles are:

1. fixed-local calibration;
2. official-LMCache always-remote calibration;
3. fixed-local validation;
4. official-LMCache always-remote validation;
5. TEMPO automatic admission validation.

The calibration and validation prompts are distinct but have identical token
buckets. The screen-only policy manifest is written after lifecycle 2 and is
never modified during validation. Unknown keys and unavailable remote paths
fall back to local before remote work starts.

The frozen workload has prompt sizes 1,220, 3,652, and 7,300 tokens and emits
64 output tokens. Every prompt plus output fits the 8,192-token model limit.
Each bucket has three requests per mode. The client records TTFT, per-token
arrival times, TPOT, ITL, E2E, throughput, SLO goodput, exact output hashes,
router decisions, and routed KV-byte estimates.

## Run later inside one fresh four-node allocation

```bash
TEMPO_PD_PERF_APPROVED=YES \
  bash eval/sota_4node/run_tempo_pd_perf_v2_in_allocation.sh
```

The launcher performs exactly one bounded `srun`; it never submits or retries
an allocation. It refuses a pre-existing result directory. The final report is
`results/tempo_pd_perf_v2_job_$SLURM_JOB_ID/result.json`.

## Interpretation boundary

This is a screen-only comparison against official LMCache remote P/D and a
fixed-local baseline. A performance claim is suppressed unless the model,
workload schedule, and output hashes match across all three validation modes.
One allocation is enough to choose or reject the architecture, but not enough
for a publication-grade promotion claim; independent replication remains a
separate step. Mooncake is not included until an equally live, same-topology,
same-byte-path adapter exists.
