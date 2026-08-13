# TEMPO-RD evaluation harness

This directory contains the current research-prototype workers and validators.
It is not a production deployment package.

## Active path

The next candidate is a minimal work-conserving D2H admission rule evaluated
against foreground-only and the requestized optimized-open data plane. The
current phase/lease scheduler is stopped; see
`../../paper/TEMPO_RD_SCHEDULER_STOP_DECISION.md`.

Use the code in this directory in the following order:

1. offline replay and focused CPU tests;
2. one bounded one-node screen only after explicit user approval;
3. an identical one-node confirmation;
4. one targeted two-node path intervention; and
5. four-node confirmation only for a frozen winner.

The same G1/G2 matrices must not be resubmitted merely to retry unsupported
hardware counters. Missing counters remain missing evidence.

## Current components

- `train.py`: common FSDP/DataStates prototype data path.
- `tier_attribution_runner.py`: stage-isolation worker.
- `build_g1_causal_readiness.py`, `compose_g1_result.py`, and
  `validate_g1_result.py`: fail-closed G1 analysis.
- `run_g2_fabric_raw.py` and `validate_g2_result.py`: targeted fabric
  observation path, gated by G1.
- `inference_kv_runner.py`: offline/generalization scaffold; it is not a
  native serving result.
- `test_*.py`: CPU/static checks.

Slurm entrypoints are experiment descriptions, not permission to submit.
Never submit, retry, or monitor an allocation automatically.

## Evidence boundary

The latest G1 raw result is structurally complete but not causally promoted.
The current scheduler has no reproducible matched-open win. Raw job trees are
local artifacts and are ignored by Git; compact claim status is maintained in
`../../paper/`.

See `../../RESEARCH_ROADMAP.md` for the next stop/go gates.
