# TEMPO-RD scheduler decision at the current training point

This decision is based on the earlier source-identical one-node `v4_open`
versus `tempo_v4` work-conserving screens (`56859316`, `56860098`) and the
new source-identical pair (`56861820`, `56861979`).  It is a scope decision,
not a claim that resource
domain orchestration is impossible on every workload.

## Evidence

Both G1 screens completed checkpoint persistence, fsync/global commit, the
one-second deadline, and fresh restore.  Both failed the preregistered
matched-open tail gate:

| run | open active tail p99 | TEMPO active tail p99 | open active skew p99 | TEMPO active skew p99 |
|---|---:|---:|---:|---:|
| 56859316 | 2.096704 ms | 2.926528 ms (+39.58%) | 1.914185 ms | 3.706393 ms (+93.63%) |
| 56860098 | 2.511552 ms | 3.165984 ms (+26.06%) | 2.916252 ms | 4.194929 ms (+43.84%) |

The matched open lane is therefore already a strong requestized data-plane
control at this workload.  TEMPO's extra phase exposure and full-event PFS
future lease are reproducible design differences, not evidence of a missing
Lustre or Slingshot optimization.

The newer pair is informative but not a promotion: `56861820` passed the
one-node functional gate with active-tail/active-skew reductions of 13.46% and
26.86%, respectively, while `56861979` used the identical source bundle and
failed at the first checkpoint because D2H hard capacity was short by
16,785,416 bytes despite 633 ms of global slack.  The pair therefore shows
timing-sensitive credit realization rather than a reproducible scheduler win.

## Decision

Stop promoting the current per-phase scheduler for this training point.  Do
not spend a two-node or four-node allocation trying to rescue this exact
candidate.  The scheduler claim is not supported unless a new, offline-tested
candidate first clears two source-identical one-node runs on both tail and
skew against the same open lane, without an expiring predicted lease.

The paper scope remains viable in two forms:

1. **Measurement/domain evidence:** identify which HBM/copy, PCIe/NUMA,
   host, NIC/CXI/Slingshot, and persistent-endpoint domains are actually
   shared, with source-bound counters and interventions.  A domain is not
   called causal from placement or device-total counters alone.
2. **Data-plane/workload result:** preserve the bounded requestized open lane
   and report its correctness, durability, restore, and measured foreground
   effects.  A scheduler improvement is only an optional result for a future
   contention regime with preregistered open-vs-foreground headroom.

Inference KV movement remains a separate case study.  It must use the same
domain-footprint and SLO ledger, but it needs a runnable endpoint and current
same-endpoint baseline before any cross-workload claim.

## Required next evidence

- Offline replay must show no full-event lease and no phase exposure beyond
  the matched open lane before any new training allocation.
- One-node tier attribution must bind foreground and auxiliary bytes to
  GPU-local/copy, PCIe/NUMA/host, NIC/fabric, and persistent scopes; otherwise
  the result remains observational.
- A scheduler candidate may return only after a preregistered contention
  screen shows open is materially worse than foreground-only.  Otherwise the
  paper must report the negative scheduler result and retain the
  measurement/data-plane contribution.
