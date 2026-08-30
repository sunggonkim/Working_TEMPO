# TEMPO-PD final measured scheme (job 57012435)

## Frozen structure

TEMPO-PD uses workload-fingerprint calibration to choose between:

1. decoder-local recompute/cache; and
2. official `LMCacheConnectorV1` P/D using `PDBackendAsync`, NIXL/UCX, GPU
   connector V3, and a 256-token chunk.

The controller is fail-local. A remote route is admitted only when the frozen
same-topology calibration lower bound exceeds the configured margin. On this
Qwen2.5-7B workload all three lower bounds were negative, so the measured
TEMPO arm selected local for all nine validation requests.

## Stronger baseline first

Before the controller comparison, the official LMCache remote baseline was
strengthened by changing only the LMCache/proxy chunk size from 64 to 256.
The chunk256 arm preserved exact outputs and won 7/9 paired requests, with a
paired median E2E delta of -26.702485 ms, TTFT p50 improvement of 8.3648%, and
request-goodput improvement of 0.7851%. Therefore the final comparison uses
chunk256, not the weaker chunk64 baseline.

Artifact: `results/lmcache_chunk256_v7_job_57012435/result.json`.

## Final actual-vLLM result

Artifact: `results/tempo_pd_chunk256_v8_job_57012435/result.json`.

Frozen setup:

- Qwen2.5-7B-Instruct, BF16;
- four A100 nodes / 16 GPUs;
- two replicas, each TP4 prefill plus TP4 decode;
- identical three prompt buckets (1,220 / 3,652 / 7,300 observed tokens);
- three requests per bucket, 32 generated tokens, 2 requests/s, four workers;
- fresh engine lifecycle for every calibration and validation arm;
- exact output-text, model, workload, schedule, route, and stream validation.

TEMPO versus strengthened official LMCache always-remote:

- exact E2E wins: 9/9;
- paired median E2E delta: -223.050036 ms;
- E2E p50: 733.551372 vs 957.532235 ms (-23.3915%);
- E2E p99: 826.410698 vs 1,205.673185 ms (-31.4565%);
- TTFT p50: 100.478952 vs 112.424650 ms (-10.6255%);
- TPOT p50: 20.442382 vs 27.253770 ms (-24.9925%);
- request goodput: 1.864679 vs 1.742191 requests/s (+7.0307%).

TEMPO and fixed-local selected the same physical route. TEMPO's 9/9 win and
-10.227283 ms paired median against fixed-local are therefore run noise, not a
separate algorithmic claim.

## Falsified alternatives

- Exact-signature prepared-handle caching: 0 hits in 72 actual P/D transfers.
  It is not part of the final scheme.
- Aggressive completion polling (4,096 cooperative yields): paired median
  improved but request goodput regressed by 1.0333%; rejected.
- Balanced completion polling (16 yields): only 5/9 paired wins; rejected.
  The final scheme uses the stock LMCache polling path.
- NIXL LIBFABRIC/CXI: the cu12 plugin loaded, but actual compute-node backend
  creation failed on every node because `fi_getinfo` could not build a CXI
  PCIe topology (`NIXL_ERR_BACKEND`). It was stopped before performance use.
- Mooncake/CXI was not substituted with an incomparable proxy after the same
  topology/provider limitation became clear.

## Claim boundary

This is valid same-allocation evidence that profiled admission avoids harmful
remote P/D and beats a strengthened official LMCache always-remote component
screen under the frozen workload and GPU budget. It does not establish a new
transport, a generally faster remote path, an independent replication, or a
cross-platform state-of-the-art result. A publication claim still requires an
independent run and at least one workload/topology where the remote branch is
beneficial and selected.
