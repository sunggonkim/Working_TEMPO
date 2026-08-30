# TEMPO P/D final same-server evidence v4 (job 57029025)

This is the final report for allocation 57029025 and supersedes v1-v3.

## Frozen production scheme

- Observe pair-local arrivals and calibrate from the first three requests.
- Low and mid pressure: decoder-local.
- High pressure: decoder-local up to a live-work credit, then spill to official
  LMCache remote P/D.
- Output32: 58 ms high boundary, credit 8.
- Output64: 70 ms high boundary, credit 9.
- Unsupported output lengths fail closed.

The output-aware threshold and mid-local branch were selected by falsification,
not by the best isolated run. Credit 7, credit 6, credit 9, mid-remote, a
universal 70 ms threshold, and a universal 58 ms threshold all lost at least
one stronger same-server comparison.

## Fair comparison contract

Each campaign uses four A100 nodes, TP8 prefill plus TP8 decode, one live vLLM
server lifecycle, Qwen2.5-7B-Instruct, identical GPU budget/request schedule/KV
geometry, and the pinned official LMCache data path. Cold keys are disjoint but
token-count preserving. The six measured blocks are ordered
local/TEMPO/LMCache/LMCache/TEMPO/local. Exact normalized schedules, routes,
streamed output, and token equality are required.

## Final headline results

| Workload | TEMPO routes L/R | TEMPO goodput | Local goodput | LMCache goodput | TEMPO vs local | TEMPO vs LMCache |
|---|---:|---:|---:|---:|---:|---:|
| 512 prompt, output32 | 40 / 8 | 14.9607 | 14.8027 | 14.5346 | 27/48, -1.504 ms | 48/48, -60.100 ms |
| 1230 prompt, output32 | 40 / 8 | 13.9220 | 13.9079 | 13.5105 | 31/48, -8.234 ms | 47/48, -55.000 ms |
| 2048 prompt, output32 | 40 / 8 | 12.6685 | 12.5833 | 11.8797 | 32/48, -14.578 ms | 44/48, -161.675 ms |
| 1230 prompt, output64 | 36 / 12 | 7.7351 | 7.6970 | 7.4994 | 37/48, -18.242 ms | 45/48, -77.317 ms |

Paired columns report E2E win count and paired median delta. All four final
campaigns passed their correctness, routing, paired, tail, and goodput or
non-inferiority gates.

Artifacts:

- `results/tempo_pd_same_server_production_prompt512_v90_job_57029025/production_final.json`
- `results/tempo_pd_same_server_production_v86_job_57029025/production_final.json`
- `results/tempo_pd_same_server_production_prompt2048_v89_job_57029025/production_final.json`
- `results/tempo_pd_same_server_production_output64_v87_job_57029025/output64_final.json`

## Arrival-regime results

- 30 req/s with the old 70 ms threshold spilled early and lost to local
  (18/48, +6.804 ms). At 58 ms it routed local, won 39/48 by -8.661 ms, and
  beat LMCache 38/48 by -30.428 ms.
- 24 req/s mid-remote lost to local by +58.252 ms. Mid-local retained 98.94%
  local goodput, had a -1.513 ms paired median, and beat LMCache 40/48.
- 16 req/s low-local retained 99.27% local goodput, had a -1.807 ms paired
  median, and beat LMCache 18/18 by -72.015 ms.

## Verification and boundary

The focused controller/policy suite passes 11/11 and `git diff --check` passes.
The evidence supports a same-harness advantage over pinned official LMCache
remote P/D across the tested prompt lengths, output lengths, and arrival
regimes. It is not a universal SOTA claim. Mooncake remains outside the direct
table because this environment lacks its official same-lifecycle router path;
the available component benchmark has different topology and semantics.
