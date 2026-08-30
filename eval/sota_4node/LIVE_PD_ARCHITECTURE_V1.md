# TEMPO-PD architecture v1

## Decision

TEMPO-PD is a request-admission layer, not a transport.  It chooses between
decoder-local execution and a live remote prefill/KV handoff before the remote
prefill request or remote KV allocation is issued.  LMCache is the first
remote backend; Mooncake can be added behind the same interface without
changing the policy.

This is the structure to keep.  The `live_pd_controller_lmcache_v1` through
`v20` files remain experiment history, not the production architecture.

```text
request
  -> versioned workload classifier
  -> frozen TEMPO-PD policy
       -> LOCAL: decoder recompute/cache
       -> REMOTE: backend adapter -> prefill -> KV handoff -> decode
  -> immutable decision/route telemetry
```

The canonical machine-readable contract is
[`tempo_live_pd_architecture_v1.json`](tempo_live_pd_architecture_v1.json), and
the transport-independent implementation is
[`tempo/pd_admission.py`](../../tempo/pd_admission.py).

## Why this boundary

The live-vLLM experiments established a useful but narrower result than “a
faster KV transport”: across 24 paired validations, admission rejected remote
P/D and improved E2E in 24/24 cases versus official LMCache always-remote P/D.
All observed winners used decoder-local execution.  No validation selected and
proved the remote branch.  Therefore the validated contribution is preventing
harmful handoffs, while LMCache/NIXL remains the remote data plane.

Putting the decision inside LMCache would conflate admission with transport.
Putting it after prefill would be too late: the expensive work and allocation
would already have happened.  The request router is the earliest component
that owns both legal actions and can fall back locally without fabricating a
transfer success.

## Components and ownership

1. **Workload classifier** — maps a request and observed decoder state to an
   exact, versioned class: model/revision, topology, backend, prompt/output,
   decoder load, and potential KV-byte buckets.  Unknown classes never inherit
   a nearby profile implicitly.
2. **Calibration/profile builder** — runs outside the validation path.  It
   produces conservative local-latency lower and remote-latency upper bounds,
   sample counts, correctness evidence, transfer failures, and an epoch range.
3. **Frozen policy** — a pure deterministic lookup.  It selects remote only
   when every safety gate passes and the conservative remote advantage is at
   least 5 ms.  Missing, stale, screen-only, incorrect, failed, marginal, or
   deadline-infeasible profiles select local.
4. **Admission ledger** — commits the route before work starts.  A remote
   reservation may fall back to local only before remote prefill begins.  It
   rejects duplicate IDs and illegal phase transitions.
5. **Backend adapter** — executes remote P/D through official LMCache or a
   future official Mooncake path.  It owns transport-specific metadata, not
   route policy.
6. **Telemetry/evidence writer** — records class fingerprint, profile ID,
   epoch, reason, route, timing, bytes, output hash, and backend failures.  It
   cannot mutate the policy used by the same validation epoch.

## Frozen decision rule

For an exact workload profile,

```text
remote_benefit_lower_bound
  = local_latency_lower_bound - remote_latency_upper_bound
```

Remote P/D is admitted only if:

- the backend is ready;
- the profile is independently replicated for production use;
- both routes have at least three samples;
- remote and local outputs are exactly equivalent;
- the remote profile contains zero transfer failures;
- the policy epoch lies within the frozen profile range;
- the benefit lower bound is at least 5 ms; and
- the remote upper bound plus reserve fits the remaining request budget.

Otherwise the action is decoder-local.  The existing one-allocation profiles
can be replayed only by explicitly disabling the replicated-evidence and
three-sample production guards; they are not silently promoted.

## State and failure semantics

```text
local_selected  -> decode_started -> complete | failed
remote_selected -> remote_started -> decode_started -> complete | failed
remote_selected -> local_selected  (pre-start failure only)
```

Once remote prefill starts, the ledger does not silently relabel the request as
local.  A later recovery is a separately measured retry policy and must not be
mistaken for the original admission decision.

## What is frozen, and what remains replaceable

Frozen in v1:

- decision placement before remote work;
- two actions and fail-closed default;
- exact workload/profile identity;
- conservative benefit bound and 5 ms margin;
- correctness, failure, evidence, epoch, and deadline gates;
- immutable validation epoch and request state machine.

Replaceable without changing the architecture:

- bucket boundaries and online load features, through a new classifier
  version;
- bound estimator, through a new profile-builder version;
- LMCache versus Mooncake remote adapter;
- router implementation language or vLLM integration seam.

Any change to the frozen list requires architecture contract v2, not another
runtime monkeypatch in the experiment chain.

## Migration from the current harness

The next implementation step is mechanical, not a new algorithm:

1. preserve the existing official LMCache P/D request construction;
2. replace per-script `decisions[bucket]` dictionaries with a serialized
   `PDCalibrationProfile` registry;
3. call `PDAdmissionLedger.admit()` in the proxy before creating
   `disagg_spec`/remote allocation metadata;
4. issue local decode when the returned route is local, otherwise issue the
   unchanged official remote P/D sequence;
5. serialize the decision reason, workload fingerprint, profile ID, and phase
   transitions with every request;
6. keep calibration and validation processes separate so validation never
   tunes its own decision.

The older scripts remain reproducibility artifacts until this adapter matches
their outputs; they should then be deprecated rather than extended to v21.

## Required evidence after integration

No new GPU experiment should begin until the adapter passes CPU tests for
exact profile lookup, fail-closed behavior, duplicate request rejection,
pre-start fallback, and illegal late fallback.  The first GPU run after that
is an integration check, not a performance-tuning sweep.

Promotion later requires an independent allocation, a trace-driven arrival
process, order-balanced lifecycles, and the same model, requests, KV bytes,
topology, and GPU budget for local admission and official LMCache.  Report
route counts, throughput/SLO goodput, TTFT, TPOT, E2E, exact output equality,
and transport failures.  A remote-selected request must be observed and win
before claiming a generally adaptive crossover controller.  Until then the
supported paper statement is a correctness-gated harmful-handoff rejection
policy for the measured regime.

Mooncake belongs only in the same lifecycle through a real router and the same
request/KV/GPU contract.  A raw component-throughput number is not a substitute
for that comparison.
