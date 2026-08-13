# TEMPO-RD status

Date: 2026-08-13

## Supported

- Checkpoint overlap can amplify synchronous collective-group tail. In job
  56500531, matched active FSDP p99 increased from 5.178 ms to 15.039 ms with
  DataStates.
- Exact-byte D2H/persistence accounting, one-second durability, commit, and
  fresh-restore gates are implemented in the prototype harness.
- The optimized-open requestized data plane is a strong matched control.
- CPU contracts exist for stage flows, admission, causal promotion, and the
  inference KV generalization scaffold.
- Disabling P2P in the matched two-node path experiment increased
  open-combined step p99 from 70.280 ms to 308.831 ms. This supports path
  sensitivity, not per-link byte attribution.

## Not supported

- The current per-phase v4 scheduler is not superior to optimized-open.
- No physical resource domain is controller-ready from live causal evidence.
- G1 job 56873704 is raw-complete but not promotion-ready: ten scoped
  mode/domain counter pairs remain missing.
- G2 raw observations do not provide rank/slice-bound fabric byte attribution.
- The Torch KV smoke is correctness-only and is not a native inference result.
- There is no topology-aware, universal, production, or SOTA claim.

## Decision

Stop the current phase/lease scheduler and do not repeat the same G1/G2
matrices. The next candidate, C0, is a minimal work-conserving D2H admission
cap with no phase prediction and no future persistence lease.

C0 receives an allocation only after historical replay shows:

1. optimized-open has a real foreground contention gap;
2. C0 predicts at least 15% improvement on both tail and skew;
3. no capacity shortfall or deadline infeasibility; and
4. parameters are frozen before held-out replay.

See `../RESEARCH_ROADMAP.md` for the bounded allocation sequence.

## Artifact policy

Raw job directories and Slurm logs are local research records and are not
versioned in Git. The compact evidence record lists job IDs, metrics, and claim
boundaries. Historical source snapshots remain locally archived by their
recorded digests; the fresh repository does not claim to be a self-contained
artifact-evaluation package.
