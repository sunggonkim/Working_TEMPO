# TEMPO Elastic-PD contention audit and research decision

Audit date: 2026-08-21

This document is the terminal research decision for the current TEMPO
Elastic-PD objective. It supersedes the v449-only README interpretation, the
v539--v544 coefficient-tuning narrative, and the pre-screen Candidate B plan.
It does not delete, replace, or weaken the original success gates.

## 1. Decision

The original objective was kept unchanged. Its explicit negative-completion
path is now reached: no frozen request-level admission/routing candidate is
promoted to independent validation because three structurally different live
candidates and all three diagnostic phase-oracle policies fail to combine the
required median gain with the required tail guarantees.

TEMPO and production-scale orchestration are **not** being abandoned. What is
stopped for this objective is the following candidate scope:

> Per-request selection between decoder-local prefill and unchanged official
> LMCache remote prefill, on the frozen four-node C4 deployment, while leaving
> decoder scheduling, P/D pair count, replica allocation, and the data plane
> unchanged.

The earlier family below was stopped first:

> A router-local scalar called `fabric_pressure`, multiplied by a fixed
> prompt-token coefficient and combined with request-shape-specific priority
> exceptions.

The following implementation and mechanism evidence remains valid:

- actual vLLM decoder-local chunked prefill and official
  `LMCacheConnectorV1` remote prefill;
- one request-start route commit, without mid-request fallback;
- separate local-compute and remote-KV admission ownership;
- exact output, route, cache, stream, queue, and credit-lifecycle validation;
- the C1/C2/C3 workload demonstrates that local and remote paths fail under
  different contention states and that neither route is intrinsically bad;
- Candidates B and C selected both routes with positive route-specific
  counterfactual gains, despite failing the global performance contract.

The frozen research question was:

> Under phase-changing multi-tenant inference contention, can endpoint
> completion feedback and separate local/remote in-flight windows route each
> request better than both the strongest fixed path and a profile-only
> predictor, while preserving TTFT/TPOT tails and background-tenant goodput on
> the same vLLM and LMCache data planes?

The answer under the frozen contract is negative. This is a bounded
admission/routing result. It is not a new transport result, a claim that
LMCache is universally slower, a claim that a router-local Cassini counter
identifies the physical fabric bottleneck, or evidence that cluster-level
production orchestration has no value.

## 2. What "v0부터" means in this repository

There is no checked-in file with an `_v0` suffix. The pre-`_v1` history is the
git history before commit `e51affeb0`, when TEMPO was a training/checkpoint and
generic I/O-interference project. That history was included in this audit
rather than treating v535 as the beginning.

### 2.1 Pre-reset generation (conceptual v0)

| Generation | Mechanism proposed at the time | Evidence-bound conclusion now |
|---|---|---|
| Initial--v1 | Phase-gated KV/checkpoint I/O based on an assumed shared PCIe root | The broad phase-gating and first-interference claims collided with prior work; several early numbers mixed endpoints or measured proxies. |
| v2 | Communication/I/O co-scheduling, burst monitor, service-gain model | Useful motivation, but it did not establish which physical resource caused an end-to-end application effect. |
| v3 | Topology placement and Slingshot traffic-class control | Topology/QoS primitives were substantially prior art and the live causal path was not established. |
| v4 | Sparse transfer, peer cache, nano-overlap | A large composition whose individual primitives were not a defensible contribution. |
| dynamic/v6 | Look-ahead, per-rail routing, GPU-driven doorbells, NVLink multipath, libfabric/CXI controls | Complexity grew ahead of causal evidence and same-path baselines. |
| TEMPO-RD reset | Resource-domain accounting and a per-phase checkpoint controller | The matched optimized-open lane was strong. Two source-identical screens regressed tail by 26--40% and skew by 44--94%; a later pass did not reproduce. The scheduler was explicitly stopped. |

The reusable lesson from the pre-reset generation is not a mechanism. It is
the experimental rule in
[`TEMPO_RD_SCHEDULER_STOP_DECISION.md`](TEMPO_RD_SCHEDULER_STOP_DECISION.md):
do not tune a controller before a matched contention screen proves headroom,
and do not infer a physical bottleneck from placement or device-total counters.

### 2.2 Actual P/D generation, v1--v450 source history

The bounded `eval/sota_4node/` audit indexed 549 versioned TEMPO files,
covering 323 distinct numeric revisions from v1 through v450. Every milestone
evidence note was read and the raw result roots named by those notes were
checked. The lineage is:

| Revisions | What changed | What survived falsification |
|---|---|---|
| v1--v27 | First actual vLLM P/D, LMCache and NIXL screens, KV geometry and pressure probes | Admission is separate from transport. The first allocation selected local for 24/24 pairs, so it did not validate a remote branch. |
| v28--v60 | Exact-output fixes, queue crossover, offered-rate regimes, local credit | Remote is not always bad. At 32 req/s, a 16-local/8-remote mix beat both fixed paths because remote spill relieved decoder-local queueing. |
| v61--v95 | One live server epoch, balanced order, credit 6--9, shape thresholds | A mid-load remote rule was falsified (44/48 remote and slower); output-specific credits worked only in their calibration geometry. |
| v96--v129 | Request-level interleaving, short/long output and heterogeneous workloads | 30--60 ms temporal drift made sequential arm blocks invalid. State must be keyed by workload class, but proliferating fixed thresholds did not generalize. |
| v131--v186 | Cache catalog, immutable affinity, warm/cold hybrid phases | Cache residency is a first-class route constraint. Warm reuse produced small real gains, but not the final cold/contention claim. |
| v190--v245 | Saturation, tail-aware policies, 4K and output-256 composition | Aggregate gains were small and request-paired gates failed. A favorable aggregate alone is insufficient. |
| v248--v290 | Mixed-load frontier and externally frozen offered-rate policies | LMCache had real failures at rate 56, but availability evidence cannot be silently counted as a TEMPO performance win. The next missing signal was online congestion state. |
| v291--v321 | Online arrival regime, unique prompt chunks, pair-local fast path | Avoiding cache-key aliasing was necessary correctness work. The result still compared mainly against LMCache, not the strongest fixed path and predictor. |
| v322--v349 | Bursts, local credits and 25 ms microbursts | Local admission can absorb short bursts, but a fixed burst threshold is workload-specific. |
| v353--v430 | Phase change, prefix swap, adaptive/latched policies, cap-5/cap-6 | A cap-5 heavy-burst policy caused a +435.5 ms worst regression; cap-6 reduced it to +58.2 ms. Constants still came from the evaluated trace rather than an online service model. |
| v440--v450 | Native NIXL comparison and canonical four-arm Elastic-PD | One-way commit, phase-correct first-response credit release, cache isolation and weighted local/remote ownership are sound. v449 beat LMCache but only beat the predictor by 0.7%; a later canonical run routed 48/48 local and failed the final goal. |

### 2.3 v452--v544 raw-artifact history

The late revisions are mostly run-directory names rather than new versioned
source snapshots. Every canonical discovery run root from v452 through v544b
was inspected. The campaign varied profile construction, pair load, local and
remote credits, decoder chunking, cache state, request priorities, token-ID
forwarding, libfabric/CXI options, peer memory, and synthetic CXI background.

Only v492 passed all then-implemented candidate gates, and it did not
reproduce:

| Run | Best fixed | Local / remote / predictor / TEMPO median E2E (ms) | TEMPO vs fixed | vs predictor | goodput vs fixed | Routes L/R | Pass |
|---|---|---:|---:|---:|---:|---:|---|
| v492 | local | 1696.0 / 1716.8 / 1648.9 / 1487.3 | +12.31% | +9.80% | +9.17% | 28/20 | yes |
| v493 repeat | local | 1647.2 / 1841.2 / 1652.5 / 1511.4 | +8.24% | +8.54% | +1.13% | 28/20 | no |
| v502 policy repeat | local | 1684.7 / 1752.0 / 1659.9 / 1525.8 | +9.43% | +8.08% | -0.21% | 28/20 | no |
| v534, 25% CXI background | remote | 1750.8 / 1711.1 / 1545.6 / 1536.0 | +10.24% | +0.62% | +4.66% | 21/27 | no |
| v536, 100% background | local | 1847.2 / 1871.8 / 1627.0 / 1634.0 | +11.54% | -0.43% | -0.44% | 22/26 | no |
| v538, no background | remote | 1817.2 / 1690.9 / 1603.6 / 1619.7 | +4.21% | -1.00% | +4.75% | 20/28 | no |
| v540, adaptive no-priority | local | 1793.6 / 1847.3 / 1623.2 / 1594.9 | +11.08% | +1.75% | -2.85% | 26/22 | no |
| v543, no background | local | 1783.4 / 1783.6 / 1863.5 / 1595.2 | +10.55% | +14.40% | +1.99% | 21/27 | no |
| v544b, 100% background | remote | 1770.5 / 1660.2 / 1556.7 / 1587.6 | +4.37% | -1.99% | +1.59% | 26/22 | no |

The same-allocation v536/v538 comparison is informative. Relative to no
background, 100% synthetic CXI traffic increased local median E2E by about
30 ms and remote by about 181 ms. Remote therefore suffered about 151 ms more
degradation. This supports the user's contention hypothesis; it does not
identify whether the extra delay was a fabric link, sender, receiving NIC,
PCIe/host backpressure, or LMCache semantic-operation bottleneck.

v544b also proves that mixing routes can select individually useful requests:
the 26 local choices beat their remote counterfactual by 6.83% at the median,
and the 22 remote choices beat local by 6.90%. Nevertheless, full TEMPO lost
to the predictor, improved fixed-path goodput by only 1.59%, and had a
349.5 ms worst paired regression. Correctness passed; the scheduling claim
did not.

## 3. Why `remote` has often looked bad

There is no evidence that remote is intrinsically always bad.

- At low and moderate load, local avoids P queueing and KV transfer, so it
  should win unless cache residency or queue imbalance changes the result.
- At high decoder-local load, remote prefill can reduce decoder queueing and
  has won in the v47--v60 crossover and later mixed-load studies.
- Under synthetic CXI contention, remote was degraded much more than local in
  the matched v536/v538 comparison.
- Under other high-background runs, remote was still the best fixed path.
  Therefore `background=high` is not a sufficient system-state label.
- Most existing experiments ran one foreground campaign in an otherwise
  allocated four-node island. They did not keep both decoder compute and the
  actual P/KV path busy with independent inference tenants. They consequently
  under-sampled the state in which both paths are bad and the lesser bottleneck
  changes over time.

The correct motivation is a moving bottleneck, not remote inferiority:

```text
decoder-local path = D queue + mixed prefill/decode GPU externality
remote path        = P queue + KV production + sender endpoint/fabric
                   + receiver endpoint/host install + D queue
```

The two paths share the decoder after first-token handoff, so they are not
independent servers. A request decision also changes the latency of requests
already decoding. That externality is why a per-request static latency
predictor can be locally accurate yet globally suboptimal.

## 4. Current sensor is not a bottleneck classifier

`tempo/cassini_pressure.py` reads four local NICs but collapses six counters
into one scalar. The router then uses the larger of pause and host-blocked
normalizations as `fabric_pressure` and applies a fixed prompt-token penalty.

The official HPE counter semantics do not support that collapse:

- `HNI_TX_PAUSED`: this endpoint supplies faster than the network delivers;
- `HNI_RX_PAUSED`: the network supplies faster than this endpoint consumes;
- posted blocked cycles/packet above a few cycles: host backpressure, and this
  endpoint is likely a cause of congestion.

In v544b, valid samples had roughly 20% median RX pause, 0% median TX pause,
and 12.4 posted-blocked cycles/packet. Calling this simply "fabric congested"
conflates receiver/host pressure with network egress pressure. It is also
sampled only where the router runs, rather than at every P and D endpoint.

The replacement must preserve separate observations and add, when supported:

- per-traffic-class sent/received packets and ECN ratios;
- posted and non-posted host-blocked ratios;
- receive priority-match versus overflow-match ratio;
- resource-exhaustion NACKs, retries, and response timeouts;
- endpoint-local P queue, D queue, step/service times, and transfer
  enqueue/complete durations.

No cross-host monotonic timestamps will be subtracted. Each endpoint reports
its own duration and queue snapshot asynchronously. Missing counters stay
missing; they are not converted to zero pressure.

Primary platform references:

- [Perlmutter architecture](https://docs.nersc.gov/systems/perlmutter/architecture/)
- [HPE Cassini counter semantics](https://cpe.ext.hpe.com/docs/24.11/getting_started/HPE-Cassini-Performance-Counters.html)
- [Slingshot analysis](https://arxiv.org/abs/2008.08886)

## 5. Prior-work boundary

The basic idea of switching between local/aggregated and remote/disaggregated
prefill is occupied. TEMPO must not claim it.

| Work | What it already establishes | Remaining gap relevant to TEMPO |
|---|---|---|
| [DistServe](https://arxiv.org/html/2401.09670) | P/D interference, disaggregation, goodput-oriented placement with network affinity | Mostly deployment/profile planning, not request-scale response to endpoint contention. |
| [P/D-Serve](https://arxiv.org/abs/2408.08147) | Commercial P/D serving on tens of thousands of NPUs for more than eight months, dynamic P/D organization/ratios, reject-forward scheduling, and optimized D2D transfer | This already occupies production scale and end-to-end P/D deployment. Its control plane changes grouping, forwarding, and transfer implementation; it does not test live endpoint/receiver evidence for per-request local-prefill versus unchanged remote P/D admission. |
| [Mooncake](https://arxiv.org/abs/2407.00079) | Kimi production serving, a KV-centric disaggregated cache, and Conductor scheduling over cache and load under overload | It explicitly recognizes that transfer time changes with sender/network congestion, but its admission threshold is operator-adjusted and the architecture is always disaggregated. It does not retain decoder-local prefill as a per-request escape path. |
| [Splitwise](https://arxiv.org/abs/2311.18677) | Production-oriented phase splitting, independently provisioned prompt/token pools, hierarchical scheduling, and pending-token load balance | It assumes the provisioned back-plane transfer regime and schedules among pools; it does not close a request-scale loop around sender, network, receiver, and install completion. |
| [TaiChi](https://arxiv.org/html/2508.01989) | Unified aggregation/disaggregation and request-level latency shifting | Reconfiguration targets workload changes at minute/hour scale; it does not localize a live KV-path bottleneck. |
| [Kairos / load-aware prefill deflection](https://arxiv.org/html/2607.02043) | The closest queue/GPU-only algorithmic ablation: per-request decode-local prefill deflection with a TBT-safe chunk schedule on vLLM | Uses one-time GPU profiles and 100 ms GPU-state heartbeats; transfer is bytes/link-bandwidth, dynamic network/receiver contention and prefix residency are stated gaps. It is not the production-scale baseline. |
| [NetKV](https://arxiv.org/html/2606.03910) | Topology and dynamic-congestion terms for decode selection | Simulator-only; dynamic telemetry adds a small residual in its tests and assumes an operator-provided congestion map. It does not run the local-prefill alternative. |
| [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/latest/components/planner) | A production framework with vLLM disaggregation, KV/load-aware routing, engine FPMs, and short/long cadence P/D replica planning | The planner scales pools using queue tokens, decode KV utilization, traffic, and performance models at default 5 s/180 s cadences. The documented router does not expose Cassini sender/receiver pressure or choose decoder-local prefill as a per-request congestion escape on the same P/D pair. |
| [EcoServe](https://arxiv.org/html/2504.18154) | Full disaggregation can be network-limited; non-disaggregation has P/D interference; partial disaggregation avoids KV movement | Changes the serving architecture and temporal execution rather than choosing between the same live vLLM/LMCache paths. |
| [ThunderServe](https://arxiv.org/html/2502.09334) | Network/topology-aware heterogeneous deployment and lightweight phase-role rescheduling | Coarse deployment adaptation, not endpoint-feedback request admission. |
| [FlowKV](https://arxiv.org/abs/2504.03775) and [KVDirect](https://arxiv.org/abs/2501.14743) | KV-transfer/data-plane optimization and load-aware scheduling | Transport/data-plane improvements are outside the TEMPO claim and should remain unchanged in the main comparison. |
| [MRC transport](https://arxiv.org/html/2606.18170v1) and its [production evaluation](https://arxiv.org/html/2605.04333) | Per-path state, receiver-advertised packet and semantic-operation bounds, host backpressure, service-time compensation, probes, and victim-flow protection | It is a transport for Ethernet AI fabrics, not an LLM scheduler or Slingshot implementation. TEMPO can borrow the control structure only at request/operation granularity. |

Production scale, P/D deployment, KV-aware routing, and dynamic P/D ratios are
therefore not TEMPO novelty. The four-node Perlmutter experiment is a causal
mechanism test, not evidence of scale superiority. A scale claim requires a
separate post-validation campaign that preserves constant endpoint work and
demonstrates controller stability, telemetry overhead, and goodput at more
P/D pairs.

The defensible conditional contribution is therefore:

> A same-data-plane, request-start P/D admission controller that separates
> decoder GPU pressure, P queue, sender/network pressure, and receiver/host
> pressure; bounds local prefill token-work, remote KV bytes, and remote
> semantic operations independently; and uses endpoint completion feedback and
> explicit probes to adapt during sub-second contention phases without
> replacing vLLM or the official LMCache transport.

That sentence becomes a result only if the frozen evaluation gates pass.

## 6. Required workload before controller work

The headline workload uses independent actual inference tenants, not MPI or a
raw CXI bandwidth generator. Synthetic CXI remains a component attribution
ablation.

| State | Background tenant | Intended pressure | Required fixed-policy result |
|---|---|---|---|
| C0: both cool | no background beyond foreground | baseline | no forced winner |
| C1: decoder-local prefill hot, remote cool | long-prompt, two-token, unique-cold requests pinned to decoder-local prefill | decoder-local prefill queue/GPU externality | always-remote beats always-local by at least 5% on the preregistered foreground metric |
| C2: D cool, remote hot | long-prompt, two-token, unique-cold requests pinned through actual P + LMCache KV transfer | P queue, transfer, receiver/install path | always-local beats always-remote by at least 5% |
| C3: both hot | C1 and C2 tenants together | coupled overload | both background and foreground remain measurable; no transport failure is hidden as latency |
| C4: phase changing | cool -> C1 -> C2 -> C3 -> recovery, with frozen timing | moving bottleneck and recovery | both route directions must become useful in different phases |

Foreground geometry includes 512/2K/4K prompts, 16/128/256 outputs, and
MISS/P_ONLY/D_ONLY/BOTH cache states. Arrival patterns include stable,
microburst, and bounded overload. The background trace, seeds, cache
namespaces, and phase times are identical across arms. Arms are
counterbalanced within one live server epoch.

The workload is invalid if the C1/C2 crossover does not occur, if background
goodput differs materially before the policy acts, or if LMCache errors are
counted as successful slow requests. No controller coefficient is tuned on an
invalid workload.

### 6.1 Completed v1 negative characterization

The first workload revision used 512-prompt/256-output always-local requests
for C1. The complete preregistered UCX ladder ran in Slurm job `57335890` with
two replicates, exact cold completions, identical arm schedules, and no
synthetic network traffic:

| Fraction | C1 remote gain, pooled / paired | C1 pass | C2 local gain, pooled / paired | C2 pass |
|---:|---:|---:|---:|---:|
| 0.50 | -2.77% / -3.80% | no | +12.47% / +13.54% | yes |
| 0.70 | +1.39% / -1.29% | no | +28.99% / +30.92% | yes |
| 0.85 | -0.62% / -1.85% | no | +37.84% / +43.59% | yes |
| 1.00 | -0.50% / -2.30% | no | +34.90% / +43.54% | yes |

All three foreground geometries also failed the C1 direction at 0.85. This is
a useful negative, not permission to tune the controller: output-token decode
work is downstream of both routes because local and remote requests converge
on the same D instance. Saturating shared decode cannot by itself create a
resource that remote prefill escapes.

### 6.2 Capacity normalization failures in v2 and v3

Workload v2 changed C1 and C2 to the same 4094-prompt/two-output unique-cold
request geometry; only the pinned route differed. C1 occupied decoder-local
prefill while C2 occupied actual P prefill, official LMCache transfer, and D
receive/install. It could not be pooled with v1. Its provisional reference
rates, however, were reversed relative to measured capacity: local used 8
requests/s and remote used 16 requests/s.

| Fraction | v2 C1 remote gain, pooled / paired | v2 C2 local gain, pooled / paired |
|---:|---:|---:|
| 0.50 | -1.42% / -2.40% | +8.47% / +11.54% |
| 0.70 | -0.08% / -1.40% | +32.20% / +30.31% |
| 0.85 | +0.01% / -0.71% | +40.48% / +41.06% |
| 1.00 | -1.43% / -1.13% | +46.32% / +50.07% |

At full offered load, local-background median/p99 were 149.8/240.0 ms while
remote-background median/p99 were 4301.4/6215.1 ms. Even remote at 8
requests/s measured 366.3/707.7 ms. The v2 fractions therefore did not
represent equal fractions of route capacity.

Workload v3 used local/remote references 16/8 requests/s and restricted the
foreground to the same 4094/2 cold geometry. It exposed the opposite
normalization error: at local total load 18 requests/s, local background still
measured only 134.7/282.0 ms median/p99 and the 15.1 s client window did not
drain beyond the offered interval. C1 therefore strongly favored local rather
than demonstrating a local capacity knee. At remote reference 8, one actual
HTTP 502 occurred and the block failed closed. The highest error-free observed
remote offered rate was 6.8 requests/s. These are workload-calibration facts,
not controller failures.

### 6.3 Frozen v4 opposite crossover

Workload v4 fixed the route references at local 32 and remote 6.8 requests/s,
kept identical 4094/2 unique-cold inference for foreground and background,
and retained two counterbalanced replicates. The 0.50 screen still failed C1
(-70.67% pooled, -76.76% paired), so it was not selected. The first
preregistered fraction that passed both directions was 0.70:

| State | Background offered rate | Required winner | Pooled gain | Paired gain | Replicate directions |
|---|---:|---|---:|---:|---|
| C1 decoder-local hot | 22.4 requests/s | remote | +19.21% | +17.53% | 2/2 |
| C2 actual remote P/KV/D hot | 4.76 requests/s | local | +73.02% | +72.54% | 2/2 |

The eight blocks contain 1,868 actual requests: 1,464 local and 404 through
the official LMCache remote route. All returned HTTP 200; stream, route,
output-token proof, cold-cache completion, and background-completion checks
passed with zero errors or contract violations. Synthetic network traffic was
absent. The result deliberately sets `performance_claim_allowed=false`: it
validates the workload for controller calibration, not TEMPO performance.

The frozen evidence is Slurm job `57335890`, result SHA-256
`31388794cc979e7523c7aeecd246bbdda0f0b68ff252ed74a4c98b43dfbb5120`,
and aggregate raw SHA-256
`13f2baedd561400d6298a158be8673a74817c3dcdaba9ac858b4fbcb267396fc`.
No higher fraction is explored after this first valid crossover.

### 6.4 v6/v7 four-endpoint mechanism characterization

The frozen v4 matrix was repeated without changing its workload, first with
endpoint-local cumulative vLLM metrics (v6), then with all Cassini traffic
classes 0--7 included in every endpoint sample (v7). Both runs were
characterization-only and explicitly set `performance_claim_allowed=false`.
The final v7 run again passed both fixed-arm directions:

| State | Required winner | Pooled gain | Paired gain | Replicate directions |
|---|---|---:|---:|---|
| C1 decoder-local hot | remote | +18.23% | +16.11% | 2/2 |
| C2 actual remote P/KV/D hot | local | +71.61% | +71.12% | 2/2 |

All 1,868 requests completed with HTTP 200, exact two-token stream proofs,
zero request or contract errors, exact fixed routes, and no synthetic network
background. The route counts were 1,464 decoder-local and 404 official
LMCache remote requests. Every node produced its completion marker. The v7
result, aggregate raw artifact, and endpoint-characterization SHA-256 values
are respectively `f05dc7b6b3c966860a6eab742f0d388058f65b2cc9abd7f4a2b26d483cb2a24b`,
`271ab245f5169aaa34128b96268a116c4f03546e773f8d9fcc9fa6fe578b1914`, and
`2c477410706cfbb91c5a21d325cd9dd2b08181566bd83c621e78eaa2010b1946`.

The endpoint evidence changes the mechanism interpretation:

- In C1 local blocks, each D completed 183 requests and 749,202 prompt
  tokens. Mean queue time was only 0.007--0.008 ms/request, while mean
  prefill time was 192--213 ms and mean inference time was 378--418 ms.
  With the foreground routed remotely, D prefill fell to 101--104 ms and D
  inference to 188--193 ms. A zero waiting gauge therefore does not mean D
  has local-prefill headroom: continuous batching inflates active service
  time without building a visible FIFO queue.
- In C2 remote blocks, P mean prefill was 97--98 ms and D mean
  receive/recompute-side prefill was 29--34 ms, with endpoint queue means
  again only 0.008--0.011 ms. Nevertheless foreground median E2E was
  483--491 ms, versus 138--139 ms on the local arm. Endpoint durations from
  different clocks are not summed, but the observations are incompatible
  with a queue-count-only explanation; the remote completion path contains a
  material transfer/control/install residual outside the reported P/D queue
  gauges.
- Across 96 Cassini endpoint samples using the v2 TC0--7 inventory, ECN,
  RX/TX pause, receive overflow, resource NACK, retry, and response timeout
  deltas were all zero. Decoder endpoints reached roughly five posted
  blocked cycles/packet whenever the remote tenant was active, while P
  endpoints remained near zero. This is endpoint-total, advisory evidence
  consistent with receiver/host backpressure; it is not a route-specific or
  switch-fabric causal classifier.

Thus the valid C1/C2 crossover is a real inference-contention result, but it
does **not** establish a saturated Slingshot link. The next required step was
a separate P_ONLY attribution tenant that pre-seeded P cache outside the
measured window, removed long P compute from repeated transfers, and located
a bounded error-free KV-transfer/receiver knee. Section 6.5 records that
completed step. C3 then combines the KV-amplified remote tenant with the
frozen C1 local-prefill tenant. The original cold C2 remains in the final
matrix so attribution cannot replace a realistic end-to-end P/D workload with
a cache microbenchmark.

### 6.5 P_ONLY transfer/receiver attribution

The follow-up completed in the same Slurm allocation, job `57335890`, without
synthetic network traffic. Thirty-two distinct 4094/2 prompts were pre-seeded
before the first endpoint `before` snapshot and placed evenly across the two
producer pairs. The measured rate ladder was 4, 8, 12, 16, 24, and 32
requests/s, with fixed local and fixed remote foreground blocks at each rate.
All twelve blocks and all requests passed stream, route, output, transfer, and
source-hit validation.

The local-foreground blocks isolate each P endpoint to the P_ONLY tenant.
Their cumulative vLLM evidence is exact at every rate: each request contains
4094 prompt tokens, 4093 are external-prefix-cache hits, and only one residual
KV token is computed. Long producer prefill is therefore removed from the
measured tenant; zero producer work is not claimed.

| P_ONLY offered rate | Remote FG median | Local FG median | Remote background achieved rate | Remote client window |
|---:|---:|---:|---:|---:|
| 4/s | 449.6 ms | 141.9 ms | 3.86/s | 8.29 s |
| 8/s | 656.3 ms | 142.8 ms | 7.64/s | 8.37 s |
| 12/s | 1941.1 ms | 139.0 ms | 8.37/s | 11.47 s |
| 16/s | 3720.4 ms | 146.8 ms | 8.90/s | 14.39 s |
| 24/s | 5865.1 ms | 139.5 ms | 9.71/s | 19.77 s |
| 32/s | 5417.3 ms | 142.8 ms | 9.73/s | 26.32 s |

The first 2x remote-foreground inflation and first greater-than-10% drain both
occur at 12 requests/s. Achieved remote-background throughput plateaus around
9.7 requests/s even though all requests eventually complete. Across the rate
ladder, P and D mean queue times remain roughly 0.008--0.011 ms and midpoint
waiting gauges remain zero. Endpoint inference means account for only about
75--89 ms at the remote blocks, leaving a diagnostic client-visible residual
from 361 ms at rate 4 to multiple seconds above the knee. Endpoint-clock
durations are not treated as a formal additive decomposition.

Valid Cassini samples again contain zero ECN, RX/TX pause, receive overflow,
resource NACK, retry, and timeout events. Decoder-side posted blocked cycles
remain about five per packet. The defensible conclusion is therefore an
official-LMCache retrieval/transfer/control/install/receiver completion
bottleneck with an empirical service ceiling, not a proven Slingshot switch
bottleneck. This is exactly the signal gap the endpoint-completion controller
targets: static bytes/bandwidth and instantaneous queue gauges do not reveal
the multi-second service inflation.

The result, aggregate raw, and characterization SHA-256 values are
`b1420da5d8b4347a760999120d29c531c2c3b04782b944367d2a35889dd46833`,
`66e9cbecac2c09b0535a51c1c5b62b27e7c565feadac27a80ac84a5f8e895893`, and
`d87647f6f8f7a1134b2f70fe4c9d29a4ee5a8cf7d2377d2a4595ea9ac7214993`.
This campaign remains component attribution with
`performance_claim_allowed=false`.

### 6.6 Coupled C3 pilot

The next job combined the frozen C1 decoder-local tenant at 22.4 requests/s
with P_ONLY remote-path rates 0, 4, 8, and 12 requests/s. It retained the same
real four-node vLLM P/D deployment, official `LMCacheConnectorV1:UCX` path,
4094/2 foreground, pre-seeded 4094/2 P_ONLY pool, and exact route/cache/output
checks. All eight fixed-arm blocks and all 1,944 measured requests were valid.

| P_ONLY rate | Local FG median | Remote FG median | Winner |
|---:|---:|---:|---|
| 0/s | 528.9 ms | 448.4 ms | remote |
| 4/s | 542.8 ms | 589.1 ms | local |
| 8/s | 661.7 ms | 655.3 ms | near tie (remote by 1.0%) |
| 12/s | 674.9 ms | 1832.0 ms | local |

At rate 0, remote avoids the contended decoder-local prefill path. At rate 4,
the winner has already switched to local; at rate 8 the routes are within one
percent; and at rate 12 the remote completion residual rises to 1569.8 ms,
remote-background throughput reaches only 9.23 requests/s, and the client
window drains to 10.41 s. This is the realistic coupled motivation TEMPO
needs: neither route is globally superior, and queue-only state does not
identify the moving service bottleneck.

Unlike the isolated P_ONLY ladder, the coupled rate-12 blocks also contain the
first valid nonzero Cassini transport-fault deltas: one retry and one timeout
at the pair-0 decoder midpoint of the local-foreground block, then two retries
and three timeouts at the pair-1 prefill after-snapshot of the
remote-foreground block. ECN, pause, receive overflow, and resource-NACK
deltas remain zero, and all inference requests still complete correctly.
[HPE's Cassini counter guide](https://h41374.www4.it.hpe.com/docs/25.03/getting_started/HPE-Cassini-Performance-Counters.html)
defines these PCT counters as retries and response/close timeouts and explains
that packet loss or target-NIC resource exhaustion can invoke the retry
handler. These endpoint-total, low-count events are therefore packet-reissue
evidence concurrent with coupled load, not proof that this workload saturated
a particular switch link or caused every observed event. Their temporal and
role-local reproducibility is descriptive evidence to check in ABBA, not a
workload-validity gate.

This pilot had one `local -> remote` replicate per rate, so the exact crossover
locations were not yet order-balanced evidence. Its parent result and
characterization SHA-256 values are
`d021b3e19f0b52817fd27f3d167fab538417250b2c132fe2ddc3d207918f094c` and
`10aacb3c2deee29aa597b644acdf9d63b28a15e92c7c2cc6feb4c265fb271137`.

The frozen two-replicate ABBA confirmation subsequently completed in
interactive job `57343718` with within-rate order
`local, remote, remote, local`. At remote-tenant rate 0, remote beat local by
18.44% in the pooled median and in both replicates (19.12%, 17.76%). At rate
12, local beat remote by 66.14% and in both replicates (67.73%, 64.56%). Every
measured request, source hit, pinned route, four-endpoint snapshot, and paired
semantic schedule was valid. This authorized C4 workload characterization,
not a policy performance claim or a physical-switch bottleneck claim.

Two pre-measurement attempts in the same job remain retained separately: one
failed before `srun` because an approval variable was not exported, and one
timed out during unusually slow service startup. Neither produced a
result/raw artifact, so neither is counted as workload or policy evidence.

## 7. Controller structure to test

The profile-only predictor remains a prior, not the final decision rule.

Kairos already performs state-aware request-level local-prefill deflection.
Consequently, merely adding a live D queue or batch-size signal is not a TEMPO
contribution. The discriminating ablation is whether completion/service
residuals and independently owned remote byte/operation windows add value
when FIFO waiting remains near zero and the remote critical path is not
explained by a bytes/link-bandwidth model.

For request `r`, the conceptual costs are:

```text
C_local(r)  = static_local(r)
              + D_queue_price
              + local_prefill_externality(r)
              + uncertainty_local

C_remote(r) = static_remote(r)
              + P_queue_price
              + KV_sender_network_price(bytes_r)
              + D_receiver_install_price(ops_r, bytes_r)
              + uncertainty_remote
```

Admission uses separate bounded windows for:

1. decoder-local prefill token-milliseconds;
2. remote P prefill token-milliseconds;
3. remote KV bytes in flight; and
4. remote semantic transfers/install operations in flight.

The last two are intentionally separate, following MRC's distinction between
packet-fidelity and semantic-operation bounds. A small transfer can still be
operation-bound; a few long prompts can be byte-bound.

Each route/pair has `GOOD`, `SKIP`, `DENIED`, and `PROBE` state. Completion
feedback updates a bounded service residual or quantile. A denied route is
reopened only by a low-rate explicit probe, so recovery does not depend on a
stale EWMA crossing an arbitrary threshold. One controller owns price and
window updates; priority exceptions do not form a second control loop.

The live router also observes explicitly marked route-pinned background
tenants. Their first-response completion updates the same route-normalized
service stretch, which lets independent inference load expose decoder-local
or remote handoff inflation before a foreground TEMPO request finishes. This
is passive evidence only: it neither reserves nor releases local-token,
remote-prefill, KV-byte, or semantic-operation credit. A passive success is
ignored while a route has failure history or an active probe, so independent
traffic cannot bypass explicit recovery. The same rule applies to a TEMPO
request admitted before a newer concurrent failure: its late success releases
its reservation but does not recover the route. Warmup/preseed requests are
not marked and therefore cannot train the controller.

One-way route commit and phase-correct first-response credit release remain
unchanged. Mid-request migration is out of scope.

Cassini counters are advisory safety/attribution inputs. Zero pause/ECN does
not certify an uncongested path, and no scalar Cassini value may directly
select local or remote. The core online feedback is endpoint-owned service
completion: D local-prefill service inflation for the local window and
remote handoff completion residual for the byte and semantic-operation
windows. A queue/GPU-only policy retains the same static profile and D state
but omits the remote completion loop.

## 8. Baselines, gates, and stopping rule

Mandatory arms:

1. always decoder-local chunked prefill;
2. always official LMCache remote prefill;
3. current profile-only predictor;
4. endpoint-feedback TEMPO.

The v544 scalar-pressure policy is an ablation. A per-phase oracle is a
diagnostic upper bound, not a baseline claim. A queue/GPU-only policy is
included to separate Kairos-like load awareness from the network/receiver
feedback contribution.

The original final gates remain unchanged:

- all output, stream, route, cache, queue, transfer, fallback, and credit
  lifecycle correctness checks pass;
- pooled E2E at least 10% faster than the strongest fixed arm;
- pooled E2E at least 5% faster than the predictor;
- request goodput at least 5% above the strongest fixed arm;
- paired E2E wins at least 75% overall and 60% in every workload group;
- p99/TPOT regression at most 5%;
- worst paired E2E regression at most 100 ms;
- when TEMPO selects local, local beats the remote counterfactual by at least
  5%, and vice versa;
- a frozen, independent allocation reproduces the result.

The v539--v544 coefficient and priority variants count as one failed heuristic
family, not as independent proofs that orchestration is impossible. No more
members of that family were tuned. Three structurally different candidates
were then frozen and screened on the same valid C4 workload.

### 8.1 Terminal A/B/C live-screen result

All three candidates used the same Qwen2.5-7B four-node topology, official
`LMCacheConnectorV1:UCX` remote route, local route, predictor, workload rows,
paired request schedule, and original gates. All eight blocks in each screen
passed exact output, stream, route, cache, fallback, queue, credit, and child
artifact validation. Always-remote was the strongest pooled fixed arm.

| Candidate | Independent mechanism | Median gain vs fixed | Median gain vs predictor | Goodput gain | Paired wins | TPOT p99 regression | Worst paired regression |
|---|---|---:|---:|---:|---:|---:|---:|
| A | instant scalar score | -2.92% | +3.48% | +10.17% | 68.89% | +44.53% | +2506.4 ms |
| B | pair-local active-request watermark epoch | +7.10% | +17.46% | +7.67% | 76.11% | +64.28% | +997.9 ms |
| C | route-pinned local external-credit epoch | +7.92% | +21.30% | +4.58% | 75.56% | +49.41% | +2278.7 ms |

Candidate C changed the input signal, not a fitted threshold. Any positive
route-pinned local external service credit opened a remote epoch only after
two pair-local confirmations; two zero-credit confirmations closed it, and
remote unavailability closed it immediately. The frozen profile retained all
17 service rows, controller windows, deadline, workload binding, and data
plane from the source profile. Its profile SHA-256 is
`20caf9c155d927085c4f40535080c62e2309c317aff8d6e3954a6c67c8f9c8d1`
and its run-contract SHA-256 is
`ffc1b712b31bb674cfc87e7e73e31daac4bbc8239a36469e955a973562461fdf`.

Candidate C exercised both routes and both external route-pinned tenants. Its
stable route counts in each replicate were C0 60 local, C1 4 local/56 remote,
cold C2 56 local/4 remote, KV-hot C2 60 local, C3 60 local, and recovery 60
local. Requests selected local beat their remote counterfactual by 78.04% at
the median; requests selected remote beat local by 25.90%. This directly
refutes the interpretation that remote was never useful or that the policy
merely collapsed to always-local.

The global contract nevertheless failed. Candidate C missed the 10% fixed
median gate and the 5% goodput gate, C1 paired wins were 33.33%, C3 paired
wins were 58.33%, C3 TPOT p99 was 283.6 ms versus 189.0 ms for fixed remote,
and worst paired regression was 2278.7 ms. Candidate A also missed the
predictor gate. Candidates B and C cleared it, so the separate
two-predictor-failure condition was not invoked.

The robustness alternative also does not apply. None of A/B/C produced the
required 15% p99 or goodput improvement; their goodput gains were 10.17%,
7.67%, and 4.58%, respectively, while TPOT p99 regressed in every case.

### 8.2 Why this is a stop result rather than another threshold search

A cross-fit diagnostic router was allowed to consume the hidden phase label.
It is not deployable and supplies no performance claim; it asks whether online
phase classification alone left an easy route-only solution on the table.

| Candidate trace | Oracle median gain | Oracle goodput gain | Oracle paired wins | Oracle TPOT p99 regression | Oracle worst regression | Full gate |
|---|---:|---:|---:|---:|---:|---|
| A | +14.55% | +7.46% | 71.11% | +47.31% | +4168.8 ms | no |
| B | +12.08% | +5.67% | 72.50% | +52.78% | +3732.3 ms | no |
| C | +10.68% | +3.27% | 71.11% | +54.50% | +3940.8 ms | no |

Even this oracle failed the full gate on all three independent traces. The
failure is therefore not repaired by another watermark, prompt coefficient,
or phase classifier. Local and remote prefill both converge on the same
decoder, so route-only admission can improve median path choice while coupled
decode interference and run-level service variation still dominate TPOT and
worst-case tails.

The same-arm, same-semantic-cell delta between Candidate C's two internal
replicates gives diagnostic context: absolute E2E delta p99 was 2657.9 ms for
local, 3736.9 ms for remote, 2436.0 ms for predictor, and 2097.3 ms for TEMPO.
No cross-host clock subtraction was used. This does not relax or invalidate
the 100 ms gate; it explains why aggregate routing gains cannot be reported as
tail isolation on this coupled workload.

The preregistered condition "tail improvement and median improvement cannot be
achieved simultaneously" is satisfied by three independent mechanisms, zero
median-plus-tail joint passes, and zero full phase-oracle passes. The frozen
objective therefore terminates with a reproducible negative conclusion. No
candidate is promoted to held-out independent validation, because doing so
after a failed screen would violate the original protocol.

## 9. Artifacts, reproducibility, and next-project boundary

The authoritative negative analyzer binds every Candidate A/B/C analysis by
SHA-256, Candidate C's semantic verdict, its parent raw, and all eight child
artifacts. It refuses an overwrite, input drift, a correctness failure, a
changed transport, or a stop rule that is not satisfied.

The locally modified LMCache runtime is preserved without pushing an
unauthorized commit to the upstream LMCache repository. The parent repository
versions `lmcache_tempo_c4_runtime.patch` and a fail-closed apply script. The
script requires exact upstream commit
`227d13f5c9fdb52ddb933641d34331f678de03a0`, checks patch applicability, and
verifies all six runtime SHA-256 values from the frozen implementation
contract after application.

- Candidate A analysis SHA-256:
  `ca8823998192c7c7e68fe417cc7ba03e8aaa825894cd6fd3d6a799cbf1175f8c`
- Candidate B analysis SHA-256:
  `7f0119b4618fe1efeea60b644c5ceb54739c4b9f14af658566715f313b5ce3fd`
- Candidate C analysis SHA-256:
  `f5b9e8994d97974cddde1fc9529ee364f029278226f3fb3a5d1aa5d21374a773`
- Candidate C semantic-analysis SHA-256:
  `280fb6f59dafd18a94a08a7bacdbbc85bc505878b87f5b221006eb888be98d76`
- Terminal negative-analysis SHA-256:
  `c8cb985aba33724b22c16d1501d9cdbd057d95ea5231b64de23a88d2572cd1f3`
- Bound report manifest SHA-256:
  `75ef151412e9059935bde302004c3ad8e184876ab77dd8d5d08e30aa5ce06eac`
- Versioned compact report manifest SHA-256:
  `247a336b5c05d0039aa618c46ab28d435cf0bf74563acec726ad009f1818876f`

The report directory contains the A/B/C table, workload-group table, pooled
E2E/TTFT/TPOT/goodput plot, and phase-wise E2E/TTFT/TPOT/goodput plot:

`results/tempo_pd_c4_semantic_credit_epoch_candidate_v7_job_57362947/negative_report_v3/`

The same SHA-bound tables and authoritative SVGs are versioned under
`paper/tempo_pd_c4_negative_report_v1/`; large raw request artifacts remain
local by repository policy. The same directory includes the byte-identical
terminal aggregate JSON, and the frozen Candidate C run contract is versioned
as `eval/sota_4node/tempo_pd_c4_semantic_credit_epoch_candidate_v7_contract.json`.

The final focused regression suite ran inside compute allocation job
`57362947` and passed 89 tests plus 11 subtests. It covers endpoint
profile/service derivation, probes, C4 node/client contracts, both router
paths, semantic analysis, negative-stop evaluation, SHA tamper rejection, and
SVG report generation.

The exact analyzer invocation is:

```bash
.vllm_venv/bin/python \
  eval/sota_4node/analyze_tempo_pd_c4_negative_conclusion.py \
  --candidate-a results/tempo_pd_c4_phase_screen_v1_retry7_job_57352661/analysis_v2.json \
  --candidate-a-sha256 ca8823998192c7c7e68fe417cc7ba03e8aaa825894cd6fd3d6a799cbf1175f8c \
  --candidate-b results/tempo_pd_c4_semantic_epoch_candidate_v6_job_57362947/phase_screen_analysis.json \
  --candidate-b-sha256 7f0119b4618fe1efeea60b644c5ceb54739c4b9f14af658566715f313b5ce3fd \
  --candidate-c results/tempo_pd_c4_semantic_credit_epoch_candidate_v7_job_57362947/phase_screen_analysis.json \
  --candidate-c-sha256 f5b9e8994d97974cddde1fc9529ee364f029278226f3fb3a5d1aa5d21374a773 \
  --semantic-c results/tempo_pd_c4_semantic_credit_epoch_candidate_v7_job_57362947/semantic_epoch_analysis.json \
  --semantic-c-sha256 280fb6f59dafd18a94a08a7bacdbbc85bc505878b87f5b221006eb888be98d76 \
  --output <new-output.json>
```

This result does not justify deleting orchestration. A production/HPC-scale
follow-on would be a different system question: jointly control decoder
admission, fairness/SLO budgets, P/D pair selection, replica organization and
scaling, and endpoint congestion recovery across many simultaneous tenants.
It would compare against P/D-Serve, Mooncake/Conductor, Dynamo, and a
Kairos-like GPU/load policy, and measure per-tenant goodput and tail isolation
as pair count grows. That broader controller changes resources that the
current frozen objective intentionally held constant. It must begin with a
new preregistration and scale authorization; it cannot be smuggled in as
post-hoc tuning or used to turn this four-node negative into a production
claim.
