# Experiment artifacts

`results/` contains two different classes of data.

1. Large native raw trees, logs, snapshots, and partial attempts stay local on
   Perlmutter and are ignored by Git.
2. A small source-bound evidence set—contracts, terminal analyses, selected
   result JSON, failure receipts—is explicitly added to Git when it supports a
   documented claim or negative conclusion.

The current evidence index is
[`paper/tempo_go/current_evidence_manifest.json`](../paper/tempo_go/current_evidence_manifest.json).
It binds the latest Candidate M/N contracts and analyses by SHA-256 and records
that both performance gates are negative. Historical paper figures remain bound
by [`paper/tempo_go/artifact_manifest.json`](../paper/tempo_go/artifact_manifest.json)
and [`paper/tempo_go/figures/manifest.json`](../paper/tempo_go/figures/manifest.json).

## Artifact taxonomy

| artifact | meaning | performance sample? |
|---|---|---|
| source-bound contract/profile | exact source, workload, gate and topology identity | no |
| setup/preflight failure receipt | launch, port, dependency or scheduler boundary | no |
| native overload timeout after correctness | measured transport failure/containment outcome | yes, as failure outcome only |
| terminal `analysis.json` | offered-population aggregation with raw SHA references | yes |
| partial root without terminal receipt | debugging/provenance | no headline claim |
| post-hoc reanalysis | unchanged raw data with corrected analysis semantics | only with an explicit boundary |

Candidate N is the last case: all seven native arms reached terminal artifacts,
but the initial analyzer crashed because one offered regime had zero completed
victims. The corrected analyzer preserves its p50/p99 as JSON `null` and fails
the performance gate closed. No `completed_attempt.json` was fabricated.

Do not infer a promoted claim from a directory name or a high version number.
Use the root README, the current evidence manifest, and
`paper/TEMPO_GO_UNIFIED_GOAL_STATE_AND_EXECUTION_PLAN.ko.md` §74.49–§74.57.
