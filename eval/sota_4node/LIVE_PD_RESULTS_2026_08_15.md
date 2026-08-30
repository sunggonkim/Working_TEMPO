# Live vLLM P/D admission results — 2026-08-15

## Outcome

The implemented mechanism is a measured P/D admission controller, not a new
transport.  For each frozen workload bucket it calibrates official remote P/D
against decoder-local execution, selects remote only when it is at least 5 ms
faster, freezes that choice, and then validates on an identical held-out
request.  In every valid screen below it rejected remote P/D.

Across eight explicit result artifacts and 24 paired validations, the
controller beat official LMCache always-remote P/D in 24/24 cases.  The median
E2E delta was -78.645 ms (range -414.190 to -5.138 ms).  This is a
single-allocation mechanism result, not an independent promotion result or a
claim of a faster LMCache/NIXL transport.

## Frozen environment

- Allocation: Slurm job `57002489`, four Perlmutter A100 nodes, 16 GPUs.
- Topology: two independent replicas, each one TP4 prefill server and one TP4
  decode server.  Both comparison modes reserve all 16 GPUs.
- vLLM: `0.26.0+cu129`; NIXL: `1.4.0` (`nixl_cu12`); backend: UCX.
- LMCache: clean pinned checkout `227d13f5c9fdb52ddb933641d34331f678de03a0`.
- Connector: official `LMCacheConnectorV1`, `PDBackendAsync`, official
  `examples/disagg_prefill/disagg_proxy_server.py`, GPU connector V3.
- Fair server settings: HMA disabled, prefix caching disabled, identical model,
  dtype, prompt, generation seed, output length, GPU budget, and background
  load within each paired comparison.
- Models: TinyLlama-1.1B and Qwen2.5-7B-Instruct from local immutable paths.

## Valid results

| Run | Foreground | Decoder load | KV bytes | Paired E2E deltas (Tempo − LMCache, ms) | Median |
|---|---:|---:|---:|---|---:|
| Tiny unloaded | 3 sizes, 32 output | none | 6–37 MB | -27.14, -47.80, -81.31 | -47.80 |
| Tiny loaded | 3 sizes, 32 output | 1×128 tokens | 6–37 MB | -5.14, -98.92, -75.98 | -75.98 |
| Qwen unloaded long | 774/3078/6150, 32 output | none | 44/177/353 MB | -51.12, -207.38, -414.19 | -207.38 |
| Qwen loaded short | 3 distinct 774-token prompts, 32 output | 1×128 | 44 MB | -82.27, -71.60, -66.55 | -71.60 |
| Qwen loaded heavy | same, 32 output | 3×128 | 44 MB | -58.00, -84.98, -61.87 | -61.87 |
| Qwen loaded saturated | same, 32 output | 7×128 | 44 MB | -122.61, -73.64, -105.31 | -105.31 |
| Qwen long TTFT, LMCache first | 774/3078/6150, 2 output | 7×128 | 44/177/353 MB | -58.67, -175.43, -353.38 | -175.43 |
| Qwen long TTFT, Tempo first | same, reversed lifecycle order | 7×128 | 44/177/353 MB | -54.48, -182.59, -338.36 | -182.59 |

The last two rows are the order check.  The sign, all three route choices, and
the effect-size scale remain stable when Tempo runs before LMCache instead of
after it.

The machine-readable aggregate is
[`results/live_pd_paper_evidence_all_job_57002489.json`](../../results/live_pd_paper_evidence_all_job_57002489.json).
It is generated only from explicitly named result paths by
[`analyze_live_pd_paper_evidence_v1.py`](analyze_live_pd_paper_evidence_v1.py).

## Mechanistic interpretation

For the 7-stream, long-context, two-token workload, remote and local TTFT were
often close, but official remote P/D inserted a very large first-to-second
token delay.  In the forward-order run the LMCache gap was 75/195/358 ms while
local execution was about 20–22 ms.  This made remote P/D progressively worse
as potential KV grew from 44 MB to 353 MB.  Reversing lifecycle order preserved
the result.

The supported contribution is therefore:

> Empirically calibrated, correctness-gated P/D admission can avoid harmful KV
> handoffs and substantially reduce E2E and decode-tail latency versus an
> always-remote LMCache policy on the measured Perlmutter workloads.

It is not evidence that Tempo transfers bytes faster than LMCache.  In observed
validation, the winning action was to omit `disagg_spec` and execute locally.

## Rejected and bounded cases

- With 8192 context but vLLM's default 2048-token batch limit, a 3079-token
  prefill was delivered to LMCache as cumulative 2048 then 3079 stores.  The
  receiver correctly rejected `declared total_chunks=49 but attempting 81`.
  The launch was fixed by setting `--max-num-batched-tokens 8192`, after which
  3079-token KV was stored/retrieved once and the full campaign passed.
- Qwen loaded 32-output-token runs diverged between remote and local output at
  the long bucket (6K in one run, 3K on the alternate replica in another).
  These runs were rejected rather than weakening exact-output correctness.
- A remote-selection crossover was not observed at 1, 3, or 7 concurrent
  decoder streams.  The remote branch therefore remains unvalidated.
- `promotion_valid` remains false because all evidence is from one allocation.
- Mooncake is not included in the same live-vLLM lifecycle.  The installed
  environment lacks the official router required for a fair connector-level
  comparison, and the existing component benchmark uses a different topology.

## Next paper-grade work

1. Move the calibrated decision into the production request router or vLLM
   connector scheduler, with online decoder queue/load features and a fallback
   to local execution before any remote allocation is issued.
2. Repeat on an independent allocation with a trace-driven arrival process and
   order-balanced lifecycles; report request throughput/goodput as well as
   TTFT/TPOT/E2E.
3. Find and validate a workload where the remote branch is selected and wins;
   otherwise simplify the policy to a measured remote-P/D rejection rule for
   this hardware/model regime.
4. Add Mooncake only through the same router, request set, KV bytes, topology,
   output checks, and GPU budget.  Do not compare against the existing raw
   component throughput number.
