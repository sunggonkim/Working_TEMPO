# TEMPO-RD prototype roadmap

## Research decision

Stop making the current per-phase v4 scheduler production-ready. Two
source-identical screens show that optimized-open is already a strong control,
and the later pass did not reproduce. More allocation on the same policy is
unlikely to answer a new question.

The near-term paper question is narrower:

> Which checkpoint stage creates collective-group tail, and can a minimal
> stage-aware admission rule beat the same requestized open data plane when
> that stage is actually contended?

Training checkpointing is primary. Inference KV remains an offline
API/generalization example until a training result exists.

## Milestone 0: clean, replayable baseline

Cost: zero node-hours.

- Freeze one source bundle and one workload fingerprint.
- Treat foreground-only and `v4_open` as mandatory controls.
- Build candidate C0 with only a work-conserving D2H inflight/rate cap. Do not
  predict phases, pause every collective, or create a full-event PFS lease.
- Replay jobs `56859316`, `56860098`, `56861820`, and `56861979`.
- Require no new phase boundaries, no future lease, zero capacity shortfall,
  exact completion, and feasibility under the slowest observed service rate.
- Fix parameters on a calibration subset and do not tune on the held-out runs.

Exit criterion: C0 predicts at least 15% improvement over optimized-open on
both tail and skew in a preregistered contention slice, with no deadline
failure. Otherwise stop C0 without an allocation.

## Milestone 1: one-node falsification

Cost target: one node for five minutes, only after explicit approval.

Run foreground-only, optimized-open, and frozen C0 as counterbalanced paired
blocks in one allocation. D2H-only and persistence-only are included only when
needed to verify the stage effect.

Use complete-group slowest-rank p99 and corrected arrival-skew p99 as primary
metrics. Exact bytes, deadline, fsync, commit, and fresh restore are hard
validity gates. GPU copy-engine bytes and stage timestamps are sufficient for
this stage claim; unavailable physical counters stay missing.

Stop after the controls if open is not at least 10% worse than foreground-only
on both metrics. A benign workload cannot demonstrate scheduler value.

Exit criterion: C0 improves both metrics by at least 10% over open and passes
all validity gates.

## Milestone 2: confirm before fabric

Repeat the identical one-node block once. Both blocks must pass. A pass/fail
pair ends C0 and returns the project to offline work.

Only after two passes may one two-node experiment isolate one concrete path,
such as P2P-enabled versus P2P-disabled. Do not launch a generic fabric matrix
to collect inventory.

## Milestone 3: final evaluation

Four nodes are reserved for a frozen candidate that passed both one-node
blocks and a targeted two-node path test. Run randomized matched blocks; do
not debug at four nodes. Broader 4-of-5 replication is considered only after
the first complete four-node win.

## Explicitly deferred

- production-grade concurrency and recovery engineering;
- exact attribution of all eight physical resource domains;
- native vLLM/SGLang/LMCache integration;
- large model/node scaling;
- universal topology-aware or SOTA claims; and
- HSN calibration retries without a new measurement mechanism.

## Publishable outcomes

Any of these is a valid result:

- a reproducible stage-aware win over optimized-open;
- evidence that optimized-open removes useful scheduling headroom;
- causal D2H/persistence stage decomposition with a negative scheduler result;
  or
- scoped P2P path sensitivity without claiming per-link byte attribution.

Optimize for reaching one honest result quickly, not for keeping every
possible claim alive.
