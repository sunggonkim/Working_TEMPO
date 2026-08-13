# TEMPO-RD research prototype

TEMPO-RD asks one question: can checkpoint state movement be admitted by the
resource stage it actually occupies, so synchronous training sees a lower
slowest-rank tail without missing the checkpoint deadline?

This repository is a research prototype. It is optimized for fast,
falsifiable experiments, not production deployment or a universal
checkpointing/SOTA claim.

## Current result

The motivation is real, but the current scheduler is not a win.

- In job `56500531`, DataStates increased matched active FSDP p99 from
  5.178 ms to 15.039 ms. Checkpoint overlap can amplify collective-group tail.
- TEMPO v3 missed every one-second checkpoint deadline in job `56493788`.
- The v4 phase scheduler regressed against optimized-open in two
  source-identical one-node screens (`56859316`, `56860098`).
- A later pass (`56861820`) was followed immediately by a capacity failure
  (`56861979`), so it is not reproducible promotion evidence.
- The latest G1 tier run (`56873704`) completed its raw matrix but remains
  `not_ready`: GPU-local D2H was observed, while scoped
  PCIe/NUMA/persistent evidence is missing.
- Disabling P2P increased open-combined step p99 from 70.280 ms to
  308.831 ms. This is a useful path candidate, not byte-attributed NVLink
  causality or a scheduler result.

The current per-phase scheduler is therefore stopped. The defensible artifact
today is a stage/resource-domain model, an optimized-open control,
exact-byte/deadline/correctness contracts, and negative results.

## Prototype scope

The active scope is deliberately narrow:

1. training checkpoint D2H and persistence stages;
2. complete collective-group tail and arrival skew;
3. optimized-open as the matched control;
4. exact bytes, one-second durability, commit, and fresh restore; and
5. promotion only when an intervention improves both tail and skew.

The eight-domain atlas, inference KV adapter, and multi-node fabric machinery
remain design scaffolding. They are not all required for the next experiment.
Native serving integration, topology-wide attribution, and SOTA replication
are deferred until the two-stage hypothesis survives cheap gates.

## Fast research loop

No Slurm job is the default.

1. **Offline replay:** require a historical contention point where
   optimized-open is materially worse than foreground-only and a candidate
   predicts at least 15% tail headroom.
2. **CPU/static gate:** run focused controller, flow, and validator tests.
3. **One-node falsification:** only with explicit approval, run one bounded
   allocation containing foreground-only, optimized-open, D2H-only,
   persistence-only, and one candidate.
4. **Repeat before scaling:** require two source-identical one-node passes on
   both tail and skew, with no deadline/correctness failure.
5. **Two nodes only for a path question:** G2 follows only after G1 identifies
   one concrete shared stage/path. Four nodes are final confirmation, not a
   debugging environment.

If optimized-open has no measurable headroom, stop scheduler work at that
workload and publish the measurement/data-plane result instead.

See [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) for concrete milestones and
[the scheduler stop decision](paper/TEMPO_RD_SCHEDULER_STOP_DECISION.md) for
the evidence behind the current stop.

## Repository layout

- `tempo/`: resource-stage models, admission prototype, causal gates, and
  focused CPU tests.
- `eval/sota_4node/`: workers, replay/analyzers, validators, and explicitly
  approved Slurm entrypoints.
- `paper/`: current status, compact evidence, and literature notes.
- `results/`: local experiment artifacts. Raw jobs and Slurm logs are
  ignored by Git; claims are summarized in the compact paper evidence.

## Local verification

The CPU suite requires no allocation:

```bash
python -m unittest discover -s tempo -p 'test_*.py'
python -m unittest discover -s eval/sota_4node -p 'test_*.py'
bash eval/sota_4node/test_run_g2_fabric_raw_static.sh
```

On Perlmutter, load the pinned PyTorch environment before tests that import
Torch. Do not submit, retry, or monitor Slurm jobs automatically.

## Evidence boundary

The authoritative status is in:

- `paper/STATUS.md`
- `paper/TEMPO_RD_COMPACT_EVIDENCE.json`
- `paper/TEMPO_RD_SCHEDULER_STOP_DECISION.md`

Current claims do not include scheduler superiority, topology causality,
native inference performance, or SOTA.

## NERSC safety

Read `NERSC_AGENT_SAFETY.md` before operating on Perlmutter. Keep filesystem
inspection bounded to explicit repository paths, and never submit or retry a
Slurm allocation without explicit user approval.
