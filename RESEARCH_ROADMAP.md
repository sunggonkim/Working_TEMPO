# TEMPO Elastic-PD research roadmap

Current decision: 2026-08-21

The canonical objective is to integrate TEMPO with the real Perlmutter
four-node vLLM P/D and official `LMCacheConnectorV1` paths, then demonstrate a
single frozen policy that clears the original correctness, strongest-fixed,
predictor, goodput, paired-win, tail, worst-regression, and route-
counterfactual gates. Those gates were not relaxed. The objective has reached
its explicit terminal negative path: three independent route/admission
mechanisms passed correctness but none jointly passed the median and tail
contract, and no diagnostic phase oracle passed the full contract.

The detailed v0--v544 evidence audit, prior-work map, claim boundary, workload
contract, and controller design are in
[`paper/TEMPO_ELASTIC_PD_CONTENTION_AUDIT.md`](paper/TEMPO_ELASTIC_PD_CONTENTION_AUDIT.md).

## Perlmutter execution rule

All vLLM, LMCache, CUDA, and GPU-backed experiment steps run only inside a
four-node, four-hour Perlmutter GPU interactive allocation. The login node is
limited to source edits, bounded artifact reads, schema checks, analyzers, and
CPU unit tests. The NERSC interactive QOS currently permits at most four GPU
nodes for four hours and requires an explicit GPU request. The canonical
allocation shape follows the official
[interactive-job guide](https://docs.nersc.gov/jobs/interactive/) and
[Perlmutter QOS limits](https://docs.nersc.gov/jobs/policy/):

```bash
salloc --nodes 4 --qos interactive --time 04:00:00 \
  --constraint gpu --gpus-per-node 4 --account <project>_g
```

Use the GPU project account ending in `_g`, as required by NERSC. Every GPU
step inside the allocation still uses explicit `srun` GPU flags; no server or
inference process falls back to the login node. Experiment wrappers make one
read-only Slurm admission check and reject anything other than a running
four-node, four-hour, 16-GPU `interactive` allocation. Consecutive discovery
steps reuse that live allocation. When less than 30 minutes remain for
teardown and artifact collection, GPU work stops and resumes only after a
fresh four-node/four-hour interactive allocation is acquired.

## Research decision

Stop tuning the v539--v544 scalar-pressure family and the completed A/B/C
request-routing family on frozen C4. Keep the actual P/D paths, request-start
one-way route commit, cache isolation, phase-correct credit release, separate
local/remote ownership, and the valid moving-contention workload as reusable
system evidence.

The completed paper question was:

> Can endpoint-feedback P/D admission outperform the strongest fixed path and
> a profile-only predictor when independent inference tenants alternately
> saturate decoder-local compute and the remote P/KV/receiver path?

Under the unchanged four-node data plane and original full gate, the answer is
no. This is not a conclusion that LMCache is universally bad or that
production/HPC cluster orchestration should be discarded.

## Milestone 0: lineage and claim boundary

Status: complete. Cost: zero node-hours.

- Audited the pre-reset TEMPO generation and its negative scheduler decision.
- Indexed v1--v450 source history and inspected v452--v544b discovery
  artifacts.
- Confirmed that remote is not intrinsically bad: it wins when decoder-local
  prefill is the limiting resource, and loses when P/KV/receiver pressure is
  dominant.
- Positioned against DistServe, P/D-Serve, Mooncake, Splitwise, TaiChi,
  Kairos, NetKV, NVIDIA Dynamo, EcoServe, ThunderServe, FlowKV/KVDirect,
  MRC, and Slingshot/Cassini documentation.
- Explicitly removed production scale as a novelty claim: P/D-Serve and
  Mooncake already provide much stronger deployment evidence. Kairos remains
  a queue/GPU-only ablation, not the headline system baseline.

## Milestone 1: endpoint evidence contract

Cost: zero node-hours for implementation and CPU tests.

Status: complete for the characterization contract. The current probe is
invoked only at the exact child-workload start boundary, each of six phase
midpoints, and each phase-end boundary (seven boundaries and six midpoints).
Boundary-to-boundary vLLM cumulative deltas and the two non-overlapping
start-to-midpoint/midpoint-to-end Cassini deltas remain endpoint-local. The
probe is not used by routing and cannot supply a performance claim. The
controller will consume asynchronously pushed state.

- Report P queue/service, D queue/active decode work, local-prefill work, KV
  bytes and semantic operations, and endpoint-local transfer/install duration.
- Sample all four Cassini NICs on every P/D node asynchronously.
- Keep TX pause, RX pause, host posted/non-posted backpressure, ECN,
  priority/overflow receive matches, NACK/retry/timeout counters separate.
- Never subtract clocks from different hosts; never map missing counters to
  zero; never synchronously fetch `/metrics` in the route decision.

Exit criterion: CPU tests prove schema, monotonic deltas, missing-counter
validity, endpoint role, bounded staleness, and request completion matching.

The v7 run produced 96 valid TC0--7 endpoint snapshots and all four node
completion markers. All ECN, pause, overflow, NACK, retry, and timeout deltas
were zero; decoder posted-blocked cycles/packet rose to about five under the
remote tenant. The result supports receiver/host pressure as an advisory
endpoint signal but does not prove switch-fabric congestion.

## Milestone 2: realistic contention workload

Cost target: one authorized four-node/four-hour interactive allocation.

Run independent actual inference tenants in five frozen states:

- C0 both paths cool;
- C1 decoder hot, remote path cool;
- C2 decoder cool, actual P + LMCache transfer/receiver path hot;
- C3 both paths hot;
- C4 cool/C1/C2/C3/recovery phase sequence.

The fixed-arm screen must show opposite crossovers: remote beats local by at
least 5% in C1 and local beats remote by at least 5% in C2. Foreground and
background goodput, TTFT, TPOT, exact outputs, transfer errors, and endpoint
signals are all recorded. Synthetic CXI load is an attribution ablation only.

Current status: complete for C1/C2 workload validation. Workload v1's
long-output tenant saturated decode work shared by both routes. v2 and v3 then
showed that provisional local/remote reference rates were not capacity
normalized; v3 also failed closed on one actual remote HTTP 502 at 8
requests/s. These negatives are retained.

Workload v4 freezes identical 4094-prompt/two-output unique-cold inference,
local/remote reference rates 32/6.8 requests/s, foreground 2 requests/s, 70%
background fractions, 15 s blocks, and two counterbalanced replicates. In job
`57335890`, remote beat local in C1 by 19.21% pooled/17.53% paired and local
beat remote in C2 by 73.02%/72.54%. All 1,868 actual requests passed exact
correctness. This authorizes controller calibration but remains explicitly
ineligible for a performance claim. The load ladder is closed.

The v7 TC0--7 repeat again passed C1 (+18.23% pooled/+16.11% paired) and C2
(+71.61%/+71.12%). It also showed why queue-only control is insufficient: C1
D queue time stayed around 0.007 ms/request while active prefill/inference
service roughly doubled. C2 exposed a large remote completion residual even
though P/D queue means were near zero. No Cassini fabric event appeared at
the cold C2 rate.

The P_ONLY attribution campaign is also complete. A 32-prompt 4094/2 bank was
seeded before every measured endpoint window and balanced 16/16 across the two
producer pairs. Every measured background request was an exact source hit.
Across the P_ONLY and C3 campaigns, all 2,336 full-hit remote completions
reported LMCache source residency of 4094/4094 tokens; all 224 explicit misses
reported zero. This LMCache residency count is intentionally distinct from
vLLM's producer execution metric: on local-foreground blocks, where P
endpoints served only this tenant, the scheduler loaded 4093/4094 external KV
tokens and recomputed exactly one final token to produce logits. The proxy
therefore reports source residency `P`, decoder/transfer prompt geometry
`P+1`, and the router checks both meanings separately.
Nevertheless the official LMCache remote path saturated near 9.7 requests/s:
at 12 offered requests/s the remote foreground median rose from 449.6 ms at
rate 4 to 1941.1 ms and the block drained for 11.47 s, while local foreground
remained 139.0 ms. vLLM queue means stayed near 0.01 ms and valid Cassini
samples still showed no ECN/pause/overflow/NACK/retry/timeout event. This
removes long P compute as the explanation and establishes an endpoint
retrieval/transfer/install/receiver completion bottleneck that queue gauges do
not expose; it still does not identify a saturated switch link.

The first coupled C3 pilot is complete. With the frozen 22.4/s decoder-hot
tenant, the fixed-route foreground medians were 528.9 ms local versus 448.4
ms remote at P_ONLY rate 0, 542.8 versus 589.1 ms at rate 4, 661.7 versus
655.3 ms at rate 8, and 674.9 versus 1832.0 ms at rate 12. Thus remote wins
when only the decoder-local path is hot, rate 8 is a near-tie shoulder, and
local wins once remote endpoint service inflates. All 1,944 requests were
valid. Because this pilot used one local-then-remote replicate, it establishes
the candidate crossover but not order-balanced reproducibility.

The frozen two-replicate ABBA campaign subsequently completed in interactive
job `57343718`. It retained the same source/profile hashes, 22.4/s decoder-hot
tenant, P_ONLY rates 0/4/8/12, and local/remote/remote/local block order. At
rate 0, remote beat local by 18.44% in the median and in both replicates
(19.12%, 17.76%). At rate 12, local beat remote by 66.14% in the median and in
both replicates (67.73%, 64.56%). All measured requests, source hits, pinned
routes, four-endpoint snapshots, and paired semantic schedules were valid.
The frozen gate therefore authorizes C4 workload characterization. It does
not authorize a policy performance claim or a physical-switch bottleneck
claim. Failed startup attempts from the same job remain retained separately
and are not evidence.

C4 is now frozen as a completion-backed six-phase trace:
C0/C1/cold-C2/KV-C2/C3/recovery, two replicates, fixed
local/remote/remote/local order, and per-request MISS/P_ONLY/D_ONLY/BOTH
states. The cold C2 state remains the realistic full remote-path tenant; the
P_ONLY tenant remains attribution and contention amplification, not a
replacement workload.

The cache-state protocol also closes a prior observability hole. Stock vLLM
OpenAI `cached_tokens` combines decoder-local APC hits and external KV
transfer, so the total alone cannot prove D residency. Decoder vLLM now
carries its existing V1 `PrefillStats` local/external split into the final SSE
usage object without changing scheduling, model execution, LMCache transfer,
or token chunks. The router and C4 client fail closed unless every completion
matches the exact route-specific local/external/total geometry. The official
P/D proxy consumes TEMPO's decoder-only APC control before producer prefill
and injects it only into the downstream decoder request.

The C4 client's parent no longer guesses workload start from `Popen`. The
child writes an atomic marker immediately after its exact `perf_counter_ns`
measurement clock starts; the parent captures the start snapshot before the
first scheduled arrival and then samples every midpoint/end against that same
clock. The gate requires all 36 phase-by-cache-geometry cells, at least four
paired local/remote samples per cell, all four ABBA blocks, and exact semantic
and output pairing. These rows characterize route crossover only; they do not
gate a performance direction.

The C4 manifest and four-node lifecycle were implemented, fail-closed, and
executed. The contracts bind the phase manifest, Git identities,
Python/vLLM/LMCache/Torch/Transformers versions, runtime environment, and
execution-critical file digests. Child raws are independently bound by path
and SHA-256; node 0 revalidates inventory, result-root containment, and
per-block completion. The pair-local semantic-load ledger remains an ingress
signal and never becomes a physical-network congestion claim.

The final Candidate C C4 campaign ran entirely inside four-node interactive
job `57362947`. All eight blocks completed, each with 1,283 valid requests;
all four node completion markers, official LMCache transfers, exact output and
stream proofs, cache-source geometry, fixed/TEMPO routes, queue drain, and
credit lifecycle checks passed. The campaign contains 360 paired foreground
requests and both counterbalanced replicates. No GPU, vLLM, LMCache, CUDA, or
inference step ran on the login node.

Exit criterion: a valid, reproducible crossover and enough endpoint evidence
to distinguish D compute, P queue, sender/network, and receiver/host pressure.
If no crossover exists, adjust only tenant load/rate under a preregistered
bounded calibration procedure; do not tune a controller.

## Milestone 3: controller and offline replay

Cost: zero node-hours.

Status: complete for controller implementation, router integration, C4
analysis, profile derivation, and paired replay. Actual completion-backed C4
profiles were generated and live-screened; no placeholder or old-profile
value was accepted.
Route-pinned independent tenants can push first-response service stretch into
the matching local/remote route without owning any TEMPO admission credit.
Their successes cannot clear `DENIED`/`PROBE`; only a bounded TEMPO probe can
recover a failed route. A success from a TEMPO request admitted before a newer
failure likewise releases only its own credit and cannot erase that failure.
Explicit measured-request markers keep warmup and preseed traffic from
contaminating online state.

- Keep the static latency profile as a prior.
- Add separate windows for local token-work, remote P token-work, KV bytes,
  and remote semantic operations.
- Update one set of route prices/windows from endpoint completion residuals.
- Use `GOOD/SKIP/DENIED/PROBE` route state and bounded probes for recovery.
- Remove priority exceptions from the core policy.
- Compare against a queue/GPU-only policy to isolate endpoint feedback.

The post-C4 path was preregistered as follows:

- `analyze_tempo_pd_c4_fixed_phase.py` revalidates the node result, manifest,
  75-file implementation contract, four child digests, exact cache-source
  decisions, ABBA gate, and phase timing. It emits exactly 96 endpoint-phase
  rows, 96 phase-by-tenant rows, and all paired foreground counterfactuals.
  Boundary-to-boundary vLLM deltas and midpoint/end Cassini half-windows are
  computed only within one endpoint. A boundary must finish before the next
  phase's first request or profile fitting is refused.
- `build_tempo_pd_c4_calibrated_profiles.py` uses only paired C0 samples for
  the six frozen geometry/state rows. E2E bounds are C0 maxima, TTFT priors are
  C0 medians, and route-gap uncertainty is the maximum absolute deviation
  from the paired median (minimum 1 ms). Admission concurrency is fixed by
  Little's-law occupancy plus one burst slot for each of the two physical P/D
  pairs, capped by the frozen 16-sequence endpoint limit. There is no
  parameter search. The previously missing 4094/256 D_ONLY row is mandatory;
  remote admission for D_ONLY/BOTH remains forbidden by the production router.
- `replay_tempo_pd_c4_calibrated_controller.py` replays every paired C4
  foreground arrival, releases each reservation only at its selected route's
  first response, and includes any admission-queue wait in E2E. It must drain
  every credit, exercise both routes, beat the calibration-only strongest
  fixed arm by 3% and the predictor by 2% in mean E2E, avoid goodput loss,
  limit p99 regression to 5%, and win at least 55% of pairs before a live
  adaptive screen is authorized. These are screen gates, not performance
  claims; independent validation retains the stricter original thresholds.
- The live and replay predictor now share one explicit cache-aware rule:
  D_ONLY/BOTH must remain local; MISS/P_ONLY use only the frozen C0 profile
  comparison. Background contention tenants retain their own fixed routes in
  every foreground block, so a local foreground arm cannot accidentally turn
  remote-hot background traffic into local work.
- `tempo_pd_c4_adaptive_implementation_contract_v1.json` first revalidates the
  complete 75-file fixed-C4 parent and then binds 11 post-C4 execution files.
  Its 33 exact runtime values explicitly pin `instant_score_v1`; the candidate
  no longer depends on the router's ambient default.
  Its SHA-256 is
  `15accb86ff433876e14ae8cffcd4b4537dde90147b2794bb47f3056b0bf8ca54`
  and its canonical fingerprint is
  `cdfd106e28a9284f5c1390ab4766f56f9c19215789fef8ef02a0a672bb0c6fa6`.

The `instant_score_v1` policy was Candidate A, not an assumed
winner. Real job 57352661 provides the motivation for a separate Candidate B.
Its valid eight-block official vLLM/LMCache observer measured the pair-local
request-start-to-HTTP-EOF decoder ledger. The foreground fraction arriving at
or above half of `max_num_seqs` was 0% in C0, 74.58% in decoder-hot C1, 1.67%
in remote-hot C2, 52.08% in KV-remote-hot C2, 100% in both-hot C3, and 26.25%
during recovery (whose median active count had already fallen to 3). Thus one
instantaneous scalar score does not identify a stable route regime, while a
high/low epoch can distinguish the observed moving pressure without consuming
the hidden phase label.

This conclusion is also consistent with the failed retry-7 live screen.
Always-remote was the strongest fixed arm; live TEMPO improved pooled median
E2E over the predictor by only 3.48% and regressed 2.92% against always-remote,
with paired wins of 54.2% and 68.9%, respectively. A diagnostic cross-fit
phase router showed 19.9% and 14.5% median gains, but still failed tail,
worst-regression, and paired gates. Those diagnostics motivate runtime epoch
state; they are not a performance result and phase labels are not policy
inputs.

Candidate B, `semantic_epoch_v1`, was frozen as a calibration-only live screen.
It combines all-tenant route-pinned endpoint credits through first response
with the pair-local full-stream decoder ledger, uses observed 1/2 and 1/4
decoder high/low watermarks, two confirmation requests, a 2.0 remote
service-stretch guard, and a 1.0 remote external-credit close fraction. These
values are no longer router constants: they are fingerprint-bound fields in
the v2 endpoint profile. The profile is a deterministic derivation that keeps
all 17 service rows, controller windows, deadline, Elastic binding, and
workload binding byte-identical to its v1 source. Its SHA-256 is
`d26cd1472d41557be7d234c4426211cac494f7392e4ab295c8a5b76d6269ba3f`
and its profile fingerprint is
`9b3db664595034c0bb50603f42c30068411321c62d654a097f2a28341d14a17c`.

The profile-bound Candidate B contract binds 15 direct implementation files
and recursively verifies the 75-file fixed-C4 parent. Its run-contract
SHA-256 is
`5bb3092444dd17f5dc69883dafa7d9619c95888de1a06d90c9b82af65e98f41c`
and its canonical fingerprint is
`750b9bbfe8118b9cef4028ad7e0c4dfd53fc4e8378811297f0e56528a9472c8f`.
The old Candidate B v2 artifact is superseded. Runtime refuses semantic mode
with a v1 profile and refuses a semantic v2 profile under Candidate A. Every
decision reports the exact profile fingerprint and policy fractions; the
analyzer requires exact ownership, both routes, epoch open and close, complete
external-credit release, and the original live screen gates. A successful
screen could carry the same routing-policy object through metadata-only profile
promotion, but could not change a field or skip the newly calibrated C4
integration screen before independent validation. Candidate B passed semantic
exercise but did not pass the original performance screen.

The post-C4 Candidate B integration implementation was separately frozen over
seven semantic-specific files on top of the 11-file adaptive parent. Its
implementation-contract SHA-256 is
`59098a2eff1588e02407ab3cd0f2b06ee702e79245c9ccc271ca42cc205f10d6`
and its canonical fingerprint is
`c7b925d3d6d4cd8fc2daea852761355877d7f5b999febe5f34016401894f303a`.
It requires the exploratory Candidate B result and the new fixed C4
calibration to carry the same Slurm job ID, deterministically derives a fresh
v2 profile from that C4 v1 profile, and runs the unchanged semantic policy in
the same four-arm integration harness. The integration analyzer—not the
exploratory result—would have authorized Candidate B for independent
validation. It did not do so.

Outcome: replay and live semantic safety/correctness passed, but no candidate
predicted or achieved the complete original performance contract without
changing frozen parameters.

## Milestone 4: bounded live screen

Status: complete with the preregistered negative stop. Candidate A used the
instant scalar score, Candidate B used a pair-local active-request watermark
epoch, and Candidate C replaced that signal with route-pinned local external
service credit. Candidate C's profile and run contract were frozen before its
live screen; no result-dependent threshold was changed.

| Candidate | Fixed median gain | Predictor median gain | Goodput gain | Paired wins | TPOT p99 regression | Worst regression |
|---|---:|---:|---:|---:|---:|---:|
| A | -2.92% | +3.48% | +10.17% | 68.89% | +44.53% | +2506.4 ms |
| B | +7.10% | +17.46% | +7.67% | 76.11% | +64.28% | +997.9 ms |
| C | +7.92% | +21.30% | +4.58% | 75.56% | +49.41% | +2278.7 ms |

All three candidates passed live correctness and kept the vLLM/LMCache data
plane unchanged. None passed both the 10% fixed-median gate and the tail
bundle. Candidate C exercised both routes with positive selected-route
counterfactual gains, but C1 paired wins were 33.33%, C3 paired wins were
58.33%, and C3 TPOT p99 rose from 189.0 to 283.6 ms.

All three hidden-phase cross-fit oracle diagnostics also failed the full gate.
Their fixed-median gains were 14.55%, 12.08%, and 10.68%, but paired wins were
71.11%, 72.50%, and 71.11%, TPOT p99 regressed by 47.31%, 52.78%, and 54.50%,
and worst regressions were multiple seconds. This closes threshold retuning:
the failure is not merely an online phase-classification error.

The SHA-bound terminal verdict is
`results/tempo_pd_c4_semantic_credit_epoch_candidate_v7_job_57362947/negative_conclusion_analysis_v2.json`
with SHA-256
`c8cb985aba33724b22c16d1501d9cdbd057d95ea5231b64de23a88d2572cd1f3`.
The report, workload table, and E2E/TTFT/TPOT/goodput plots are under
`negative_report_v3/`; its manifest SHA-256 is
`75ef151412e9059935bde302004c3ad8e184876ab77dd8d5d08e30aa5ce06eac`.
The Git-visible compact copy is `paper/tempo_pd_c4_negative_report_v1/` with
manifest SHA-256
`247a336b5c05d0039aa618c46ab28d435cf0bf74563acec726ad009f1818876f`.

Exit outcome: no candidate passed every original gate. The original
"median and tail cannot be achieved simultaneously" stop condition is met;
the gate is not weakened and no candidate is promoted.

## Milestone 5: independent frozen validation

Cost target had promotion occurred: one authorized four-node/four-hour
interactive allocation.

Status: not executed by design, and no validation allocation is requested.
The A/B/C screen reached the original negative stopping condition before any
candidate was eligible for promotion. Running held-out validation anyway would
violate the frozen protocol. The implemented preregistration is retained as
an unused artifact, not as pending work. It uses replicate IDs 2--5, a
disjoint prompt-token marker namespace,
12 s phases, and the same average foreground/background rates packed into the
first 250 ms of each one-second epoch. Four position-balanced arm orders yield
exactly 576 four-arm foreground pairs: 16 pairs in each of the 36
phase-by-geometry/cache-state groups. The validation Slurm job ID must differ
from the C4/adaptive calibration job ID, but all 16 validation blocks and the
authoritative analysis reuse one persistent four-node/four-hour allocation.

The original metric definitions are now executable contracts. Pooled E2E is
the median, not the mean. One pooled strongest fixed arm is selected once for
the headline, overall-pair, goodput, pooled-tail, and worst-regression gates.
Every group additionally compares against its own stronger fixed arm. Request
goodput is completed paired foreground requests divided by the summed
first-dispatch-to-last-stream-end windows; SLO success fraction is reported
but cannot replace that denominator. Local and remote route counterfactuals
use the paired opposite fixed arm and require at least 12 selected samples per
route. The stricter 100 ms worst-regression check is applied both against the
pooled strongest fixed arm and the per-request local/remote oracle.

Adaptive authorization promotes profiles by metadata only. Elastic identity,
rows, controller numbers, and endpoint rows/controller numbers must remain
byte-identical. A v2 semantic routing-policy object must also remain
byte-identical; only IDs, `screen_only` to `replicated`, `calibration_only` to
`frozen_validation`, profile fingerprints, and the exact held-out manifest
binding may change. Final execution requires the replicated profile and does
not accept the screen-profile opt-in. Every child also validates the frontend
pair-local active-request/decode-token ledger through HTTP EOF and reports it
by phase and pair without using it for post-hoc threshold selection.

The preregistration SHA-256 is
`c1de45c97025739e94b91b2da770a38b86b7a170d288c3f47c8cd4b774af7f86`.
The 12-file independent implementation-contract SHA-256 is
`2bf5b661e72c28a7b4abb8a1ee9d7dffa9e49f8cdca347555afe5f2470b3093a`
and its canonical fingerprint is
`32fe794b90a8ee5e465250e0b3d89dcbbf070a785a33e20be1f56e4af0b7ddf1`.
The same held-out machinery now accepts either authorized Candidate A or
Candidate B without changing the preregistration. Candidate B preserves the
v2 semantic policy and passive route-pinned endpoint credits, and final
correctness additionally requires both TEMPO routes, an advanced epoch
generation, remote-epoch open and close, and both external routes. Candidate A
continues to use `instant_score_v1` with passive feedback disabled.

No calibration or policy changes are allowed. The frozen candidate must
again pass:

- all correctness and lifecycle gates;
- at least 10% pooled E2E improvement over the strongest fixed arm;
- at least 5% pooled E2E improvement over the predictor;
- at least 5% request-goodput improvement over the strongest fixed arm;
- at least 75% paired wins overall and 60% per workload group;
- at most 5% p99/TPOT regression and at most 100 ms worst paired regression;
- at least 5% local and remote route-counterfactual gains.

If the independent run fails, follow the original stopping rule. Do not
weaken the goal, relabel a discovery run as validation, or count LMCache
failures as latency wins. A reproducible negative conclusion on a valid
contention workload is an acceptable terminal result.

After the adaptive analysis authorizes promotion, prepare the immutable
manifest/profiles/run contract with CPU-only work:

```bash
TEMPO_PD_INDEPENDENT_IMPLEMENTATION_SHA256=2bf5b661e72c28a7b4abb8a1ee9d7dffa9e49f8cdca347555afe5f2470b3093a \
  bash eval/sota_4node/prepare_tempo_pd_independent_validation.sh \
  results/<semantic-campaign>/semantic_integration_analysis.json \
  results/<semantic-campaign>/independent_preparation
```

Had a candidate been promoted, the next step would have been a fresh
four-node/four-hour interactive allocation:

```bash
salloc --nodes=4 --qos=interactive --time=04:00:00 \
  --constraint=gpu --gpus=16 --account=m1248_g --immediate=600

TEMPO_PD_INDEPENDENT_VALIDATION_APPROVED=YES \
TEMPO_PD_INDEPENDENT_IMPLEMENTATION_SHA256=2bf5b661e72c28a7b4abb8a1ee9d7dffa9e49f8cdca347555afe5f2470b3093a \
TEMPO_PD_INDEPENDENT_VALIDATION_RUN_CONTRACT_SHA256=<prepared-contract-sha256> \
  bash eval/sota_4node/run_tempo_pd_independent_validation_in_allocation.sh \
  results/<semantic-campaign>/independent_preparation/independent_run_contract.json \
  "results/tempo_pd_independent_validation_job_${SLURM_JOB_ID}"
```

## Milestone 6: conditional scale-out implementation

Status: outside the completed objective. The four-node gate did not pass, so
no scale-out allocation is requested under this protocol. The user's standing
authorization for repeated four-node/four-hour interactive allocations was
used for bounded work; it does not convert this conditional milestone into a
required experiment.

Any 8/16-node extension is a new research question, not a continuation that
may bypass the stop rule. It should jointly control decoder admission,
per-tenant fairness/SLO budgets, P/D pair dispatch, replica organization and
scaling, and endpoint recovery. It must measure scaling efficiency, aggregate
and per-tenant goodput, tail isolation, controller CPU, telemetry bandwidth,
dropped/stale evidence, and phase recovery, and compare against NVIDIA Dynamo,
P/D-Serve, Mooncake/Conductor, and a Kairos-like GPU/load policy.

This milestone may establish a scale-out implementation result. It cannot
retroactively turn a four-node mechanism result into a production-scale
claim.

## Claim boundary

The completed result is a reproducible negative for the tested family of
dynamic contention-aware request admission/routing on an unchanged four-node
vLLM/LMCache deployment. It preserves positive evidence that remote and local
are each useful in different states and that both selected routes can beat
their counterfactuals. It is not transport replacement, universal LMCache
inferiority, exact switch-level bottleneck localization, production readiness,
production-scale novelty, universal SOTA, or a proof that broader cluster
orchestration cannot work.
