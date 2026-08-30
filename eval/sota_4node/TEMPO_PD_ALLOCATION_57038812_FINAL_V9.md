# TEMPO P/D production controller — allocation 57038812

## Frozen system

- Model: local `Qwen2.5-7B-Instruct`.
- Topology: four Perlmutter A100 nodes, four GPUs per node; one TP8 prefill
  server and one TP8 decode server.
- Baselines: fixed decoder-local execution and official pinned LMCache remote
  P/D under the same server lifecycle, prompts, output lengths, and GPU budget.
- Production policy after this allocation:
  - output 16, prompt <= 4096: direct decoder-local fast path;
  - output 32, prompt <= 2048: pair-arrival controller, 58 ms high-pressure
    threshold and eight local credits;
  - output 64, prompt <= 2048: prompt <= 512 force-local guard, otherwise 70 ms
    threshold and nine local credits;
  - output 128, prompt <= 4096: direct decoder-local fast path;
  - all other output lengths and prompts outside the bounds: fail closed.
- Arrival controllers are keyed by `(phase, output_tokens)`. Concurrent mixed
  output32/output64 requests no longer replace one another's live controller.

## Production output16

Artifact:
`results/tempo_pd_same_server_output16_production_v108_job_57038812/production_final.json`

- All gates passed; 48/48 TEMPO decisions used the production direct-local path.
- E2E p50/p99: TEMPO 462.494/579.277 ms, LMCache 529.988/694.185 ms.
- TPOT p99: TEMPO 33.758 ms, LMCache 40.722 ms.
- Paired TEMPO wins over LMCache by prompt bucket:
  - 512: 15/16, median -36.723 ms;
  - 1230: 16/16, median -77.021 ms;
  - 2048: 16/16, median -82.281 ms.

Independent saturated artifact:
`results/tempo_pd_same_server_output16_saturated_v104_job_57038812/saturated_final.json`

- All gates passed.
- TEMPO goodput 21.445 req/s versus LMCache 18.187 req/s (+17.9%).
- E2E p99: TEMPO 556.246 ms versus LMCache 706.319 ms.

## Production output128

Request-interleaved artifact:
`results/tempo_pd_same_server_output128_production_v111_job_57038812/production_final.json`

- All gates passed; 48/48 production decisions used output128 direct-local.
- E2E p50/p99: TEMPO 2894.307/3157.163 ms, LMCache
  3134.722/4121.901 ms.
- Paired wins and median deltas versus LMCache:
  - 512: 14/16, -37.358 ms;
  - 1230: 15/16, -276.121 ms;
  - 2048: 16/16, -312.408 ms.

Focused prompt512 artifact:
`results/tempo_pd_same_server_output128_prompt512_v110_job_57038812/output128_prompt512_final.json`

- All gates passed with 48 paired rows.
- TEMPO versus fixed local: 25/48 wins, median -1.141 ms.
- TEMPO versus LMCache: 40/48 wins, median -209.017 ms.

Saturated artifact:
`results/tempo_pd_same_server_output128_saturated_v112_job_57038812/saturated_final.json`

- All gates passed.
- TEMPO throughput 3.724 req/s versus LMCache 3.634 req/s (+2.47%).
- E2E p99 2888.420 ms versus 2913.771 ms.

## Mixed-output production bug and fix

The first heterogeneous run failed exactly six output64 requests while output32
local ownership was live. The production router had one mutable arrival
controller and rejected output-length epoch changes. The fix keeps a controller
per `(phase, output_tokens)` and a request-to-controller ownership map. A CPU
test admits overlapping output32/output64 requests and releases them in reverse
order with zero leaked credits.

After the fix, this artifact completed without request failures:
`results/tempo_pd_same_server_heterogeneous_multiepoch_v120_job_57038812/heterogeneous_final.json`

- TEMPO beat LMCache on all 48 paired heterogeneous requests, median -82.067 ms.
- E2E p50/p99: TEMPO 829.829/3128.908 ms, LMCache
  936.240/3207.406 ms.
- The conservative unified gate rejected only fixed-local order-noise criteria;
  TEMPO and fixed-local used the same local route, with median delta +8.100 ms.

At a 48 request/s offered load:
`results/tempo_pd_same_server_heterogeneous_rate48_v122_job_57038812/rate48_final.json`

- All predeclared gates passed.
- TEMPO throughput/goodput 6.446 req/s versus LMCache 6.256 req/s (+3.05%).
- E2E p50/p99: TEMPO 1028.970/3348.992 ms, LMCache
  1124.000/3396.451 ms.
- TPOT p99: TEMPO 26.503 ms versus LMCache 40.057 ms.
- Paired TEMPO versus LMCache: 35/48 wins, median -52.148 ms.
- TEMPO retained fixed-local behavior: throughput +1.38%, paired median
  +1.425 ms.

At 64 request/s, official LMCache failed during warmup: 2/24 requests timed out
after 600 seconds. No measured performance claim is made for that failed run:
`results/tempo_pd_same_server_heterogeneous_saturated_v121_job_57038812`.

## Production prompt4096 boundary

Artifact:
`results/tempo_pd_same_server_prompt4096_production_v125_job_57038812/production_final.json`

- All predeclared gates passed with exact 4094-token inputs; all 48 TEMPO
  decisions used the production direct-local path.
- Output16 versus LMCache: 24/24 paired wins, median -244.004 ms; versus fixed
  local: 17/24 wins, median -14.605 ms.
- Output128 versus LMCache: 24/24 paired wins, median -232.967 ms; versus fixed
  local: 16/24 wins, median -12.568 ms.
- Aggregate E2E p50/p99: TEMPO 629.932/3311.481 ms, LMCache
  840.249/3631.015 ms.
- TPOT p99: TEMPO 31.555 ms versus LMCache 46.631 ms.

High-load artifact (16 req/s offered load):
`results/tempo_pd_same_server_prompt4096_highload_v126_job_57038812/highload_final.json`

- All high-load gates passed. TEMPO throughput 2.392 req/s versus LMCache
  2.351 req/s (+1.72%); SLO goodput 2.392 versus 2.204 req/s (+8.50%).
- E2E p99: TEMPO 7092.432 ms versus LMCache 8710.088 ms; both output16 and
  output128 won all 24 paired requests against LMCache.
- This production verification extends the frozen direct-local prompt bound for
  output16 and output128 from 2048 to 4096. Larger prompts remain fail closed.

## Rejected boundary

Prompt6144-class direct-local was not promoted. The mixed diagnostic strongly
beat LMCache, but output16 failed fixed-local noninferiority (10/24 wins,
median +3.323 ms). A subsequent output128-only production run also failed its
fixed-local gate (22/48 wins, median +34.403 ms), despite beating LMCache 48/48
with median -457.756 ms. The policy was reverted to the 4096 prompt bound:
`results/tempo_pd_same_server_prompt6144_v127_job_57038812/prompt6144_final.json`
and
`results/tempo_pd_same_server_prompt6144_output128_production_v128_job_57038812/production_final.json`.

The output32/output64 arrival controller was also rejected at prompt4096. It
beat LMCache on 45/48 paired requests, but output32 missed fixed-local
noninferiority by a narrow margin (11/24 wins, median +25.334 ms), and aggregate
throughput was 3.425 versus LMCache 3.430 req/s. The controller is now
explicitly fail-closed above prompt2048 rather than extrapolating:
`results/tempo_pd_same_server_prompt4096_controller_v129_job_57038812/controller_final.json`.

Output256 direct-local was not promoted:
`results/tempo_pd_same_server_output256_interleaved_v113_job_57038812/output256_final.json`

- It beat LMCache in every prompt bucket, but failed the predeclared prompt2048
  fixed-local noninferiority gate (5/16 wins, median +21.986 ms) and its aggregate
  request throughput was about 1% below LMCache.
- Output256 therefore remains fail closed. The gate was not relaxed post hoc.

## Claim boundary

These results establish an actual-vLLM, live-KV P/D controller result against
the official pinned LMCache path on this TP8+TP8, four-node topology. The main
mechanism is workload-aware admission: avoid remote KV movement when its cost is
not amortized, and maintain independent state for concurrently active output
regimes. This is not a universal SOTA claim across models, clusters, transports,
or disaggregated serving configurations.

Mooncake is not reported as a direct baseline here. The installed environment
lacks a same-lifecycle Mooncake router/configuration with the identical
topology, requests, KV bytes, and GPU budget. The existing Mooncake component
benchmark is not apples-to-apples with these vLLM P/D results.
