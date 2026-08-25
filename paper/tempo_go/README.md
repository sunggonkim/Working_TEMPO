# TEMPO paper artifact

This directory contains the paper source and the compiled seven-page draft for
TEMPO, a cross-layer global orchestrator for disaggregated LLM inference on a
shared HPC fabric.

## Current evidence

The headline result is the source-frozen C9 run on a fresh four-node Perlmutter
`gpu_interactive` allocation.  It used four Qwen2.5-7B-Instruct vLLM 0.26.0
TP4 engines, the official LMCacheConnectorV1/NIXL-UCX/CXI path, matched offered
populations, and real background co-load.  TEMPO completed every foreground
request within the experiment SLO:

| Regime | SLO-good / offered | E2E p99 |
|---|---:|---:|
| normal | 60 / 60 | 3,149.88 ms |
| miss-hot | 120 / 120 | 3,336.68 ms |
| remote-favorable | 30 / 30 | 3,357.20 ms |

Under stressed regimes, TEMPO reduced p99 by 66.12--90.55% versus the strongest
fixed edge, 61.69--93.36% versus the predictor, and 58.48--93.54% versus the
queue-only policy.  It exercised both local edges and all four cross-pair remote
edges.  The background completion fraction was 0.85755, per-block/tenant minimum
0.76496, and Jain fairness 0.99787; therefore the foreground gain was not
obtained by silently eliminating background work.

C10 is an actual-system, same-carrier extension against two paper-derived
policies.  It is deliberately reported as **post-hoc**, not independent:

| Policy | normal SLO / p99 | miss-hot SLO / p99 | remote-favorable SLO / p99 |
|---|---:|---:|---:|
| TEMPO C9 | 60/60 / 3.150 s | 120/120 / 3.337 s | 30/30 / 3.357 s |
| Kairos `X={512}` | 30/60 / 3.223 s | 0/120 / undefined | 1/30 / 4.853 s |
| NetKV reproduction | 60/60 / 3.300 s | 73/120 / 13.780 s | 0/30 / 53.914 s |

`Kairos X={512}` is a restricted stock-vLLM subset reproduction, not the full
Kairos dynamic-chunk implementation.  NetKV is an Algorithm-1 policy
reproduction on the actual Slingshot/LMCache carrier.  An unchanged fresh-run
repeat is required before calling C10 an independent SOTA comparison.

The 2--1,024-pair experiment is a CPU control-plane scale receipt only.  At
1,024 logical pairs it reduced the global payload from 666,815 to 83,358 bytes
(87.499%), while bounded total p50/p99 was 85.33/158.24 ms.  It is not a claim
about native 1,024-pair GPU inference or production-scale goodput.

## Files and verification

- `main.tex`, `references.bib`: paper source.
- `main.pdf`: compiled paper.
- `artifact_manifest.json`: claim boundaries, headline metrics, authoritative
  paths, and SHA-256 digests.
- `../TEMPO_GO_UNIFIED_GOAL_STATE_AND_EXECUTION_PLAN.ko.md`: full Korean state,
  historical findings, execution gates, failures, and remaining work.

The Git artifact includes compact contracts, analyses, and aggregate result
JSON.  Full raw request/log trees remain on Perlmutter; their hashes and paths
are preserved in the analyses so omitted raw evidence cannot be silently
substituted.

Build with the Perlmutter TeX Live module:

```bash
module load texlive/2024
pdflatex -halt-on-error -interaction=nonstopmode main.tex
bibtex main
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
```

The checked PDF was built with zero LaTeX errors, unresolved citations,
unresolved references, or overfull boxes.

The final compute-node regression was intentionally split at the native process
boundary: the C9/current suite passed 294 tests plus 28 subtests (with two
historical C6 source-drift assertions deselected), the C10 process-entry suite
passed six tests, and all frozen/current Python and shell sources passed
`py_compile` and `bash -n`.  The two C6 checks assert that current source still
equals an old frozen contract; changing that contract would corrupt historical
provenance rather than fix current code.
