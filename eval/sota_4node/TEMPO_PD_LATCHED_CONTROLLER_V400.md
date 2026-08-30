# TEMPO P/D latched controller (v382)

## Frozen structure

The controller separates two decisions that earlier experiments incorrectly
coupled:

1. **Local-bypass opportunity is monotone within a session.** Four observed
   inter-arrival gaps are maintained. Once their median is at most 39 ms, the
   session latches `high_load_local_bypass`; later sparse intervals do not send
   requests back to the remote-prefill path.
2. **Backpressure is rolling, not latched.** A five-request local ownership cap
   is active only while the current four-gap median is at most 25 ms. It turns
   off after a burst, while the local-bypass opportunity remains latched.

The frozen implementation is
`tempo/pd_online_regime_latched_v380.py` plus
`eval/sota_4node/tempo_pd_same_server_latched_microburst25_v382.py`. The policy
identity is `tempo-pd-latched-bypass-rolling-credit5-382`.

## Evidence on allocation 57078464

All measurements use the same real-vLLM same-server paired harness, identical
prompt geometry, request-unique leading LMCache chunks, and the pinned LMCache
remote-prefill route as the comparator.

| Workload | Evaluated pairs | E2E wins | E2E median delta | TPOT wins | TPOT median delta | TPOT p90 delta |
|---|---:|---:|---:|---:|---:|---:|
| Sparse to burst | 16 | 13 | -119.814 ms | 14 | -3.542 ms | -1.957 ms |
| Stationary burst | 24 | 23 | -116.456 ms | 24 | -3.475 ms | -0.974 ms |
| Stationary steady | 24 | 23 | -171.565 ms | 24 | -5.055 ms | -3.498 ms |
| Burst to sparse | 24 | 24 | -105.998 ms | 23 | -3.452 ms | -1.367 ms |

In the burst-to-sparse trace, the final eight sparse pairs win E2E and TPOT
8/8. The raw rolling regime returns to affinity, the microburst credit turns
off, and the latched local route remains active. This directly validates the
state split rather than inferring it from stationary workloads.

## What is frozen and what is not

The state-machine structure is frozen for the next stage. The 39 ms and 25 ms
thresholds and local cap five are calibrated constants for this workload and
hardware; they are not universal constants. A production implementation must
derive them from an online service-time/deadline model or expose them as a
profile.

This evidence supports a same-server real-vLLM component-screen advantage over
the pinned LMCache remote-prefill route. It does **not** establish a Mooncake
win, an independent multi-allocation confidence interval, or a full external
1P1D system result. Mooncake was not runnable in the current environment with
an audited same-harness adapter, so no proxy number is reported.

## Next paper-grade step

Move the frozen admission state into the real vLLM 1P1D connector scheduler and
compare identical requests, KV bytes, TP split, GPU budget, cache policy, and
arrival traces against standalone `LMCacheConnectorV1` and vLLM's native NIXL
P/D path. Report raw per-request values and paired deltas; do not call a
`MultiConnector` run an LMCache transport baseline because it duplicates work
across children.
