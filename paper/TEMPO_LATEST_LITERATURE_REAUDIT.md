# TEMPO v4 latest-literature re-audit

Audit date: 2026-08-11 (supplemental inference-KV recheck)
Code inspected: `tempo/v4_controller.py`, `eval/sota_4node/train.py`, and
`paper/TEMPO_V4_POSITIONING.md` in the current workspace
Literature scope: primary papers, official proceedings pages, author PDFs, and
arXiv manuscripts from 2024--2026. CheckFreq and Gemini are retained only as
pre-2024 anchors because they are explicitly named in the comparison set.
Execution scope: literature and source inspection only; no Slurm allocation and
no GPU job were used.

## 결론

**넓은 TEMPO 아이디어는 이미 선행연구와 충돌한다. 그러나 논문 전체가 끝난
것은 아니다. 남은 것은 좁고 검증 의존적인 연구 질문이다.**

| 질문 | 현재 판정 | 이유 |
|---|---|---|
| 비동기 checkpoint I/O가 training을 방해한다는 문제가 새로운가? | 빨강 | DataStates가 interference를 목적 함수로 다루고, Checkmate는 GPU state copy가 training computation을 방해한다고 명시한다. |
| compute/communication phase를 보고 checkpoint chunk를 배치하는 것이 새로운가? | 빨강 | Gemini, ByteRobust, FFTrainer, ECCheck, Checkflow가 이미 이 영역을 점유한다. |
| deadline 또는 bandwidth budget에 맞춰 checkpoint를 나누는 것이 새로운가? | 빨강 | FastPersist, FlowCheck, TierCheck, CheckFreq가 각각 compute-window bandwidth, iteration deadline, paced budget, overhead bound를 제시한다. |
| D2H와 storage를 pipeline으로 묶는 것이 새로운가? | 빨강 | DataStates, FastPersist, PCcheck, MoC-System, GoCkpt가 이미 소유한다. |
| 실제 complete FSDP collective의 rank-max latency와 arrival spread를 다음 checkpoint admission 계획의 feedback으로 사용하는가? | 노랑-초록 | 검토한 corpus에서 같은 폐루프는 찾지 못했다. 단, "없음을 증명"한 것은 아니며 constituent idea는 모두 선행연구가 있다. |
| 현재 구현이 collective p99를 제한하고 rank-aligned credit을 보장하는가? | 빨강 | feedback은 risk scalar만 바꾸고, activation은 rank-local CUDA stream order이다. p99 제약, leader broadcast, consensus install, 공통 wall-clock gate가 없다. |
| 현재 TEMPO가 SOTA를 이겼는가? | 빨강 | checkpoint-active v4의 유효한 live 결과가 없다. endpoint가 다른 시스템과 raw number 비교도 공정하지 않다. |
| 수정된 v4 설계가 offline gate를 통과하는가? | 초록 | 최소 future-capacity reserve 수정과 mode별 producer coupling replay에서 다섯 event 모두 generation 9 `FINALIZE`에 도달한다. 이것은 credit-placement 증거이지 p99 측정이 아니다. |
| 다음 allocation을 지금 실행해도 되는가? | 노랑 | 최종 offline gate는 통과해 preallocation GO지만 실행은 사용자 명시 승인에만 조건부다. 승인 뒤에도 live 64 MiB/rank qD1/qP4/`O_DIRECT` group-min down-only calibration이 먼저 통과해야 하며, 현재 live 결과는 없다. |
| 논문으로 살릴 수 있는가? | 노랑 | 같은 DataStates/shared-PFS data plane에서 collective-tail/durability Pareto frontier를 실제로 개선해야 한다. 구현 설명만으로는 composition critique를 피하기 어렵다. |

가장 정확한 한 줄 평은 다음과 같다.

> **Broad phase-gating paper는 죽었다. 그러나 conventional shared-PFS
> checkpointing에서 actual FSDP group-tail feedback으로 admission을 조절하는
> 좁은 closed-loop systems paper는 아직 가능하다. 성능 결과가 나오기
> 전까지는 candidate contribution일 뿐이다.**

Current gate: **final offline preallocation GO; calibration/provenance
implemented and offline-validated; 441 Python tests run (440 pass, one hardware-dependent skip; 332 eval + 109 tempo; the focused analyzer/validator subset also passed) and
the final cross-file/delta audit (P0/P1=0) passed.**  No Slurm allocation or GPU
was used for this final validation.  A next allocation requires explicit user
approval; live calibration/performance and paper/SOTA remain unproven.

## Supplemental primary-source check (2026-08-11)

The current source check reinforces, rather than widens, the claim boundary.
The 2026 DataStates-LLM manuscript explicitly treats GPU/host/storage
heterogeneity, lazy asynchronous snapshots, coalescing, and storage overlap as
its system scope and reports evaluation up to 256 A100 GPUs.  TEMPO-RD must
therefore isolate a domain-specific causal intervention and cannot claim a
first asynchronous checkpoint data plane.  See the primary manuscript:
<https://arxiv.org/abs/2601.16956>.

For inference, the current KV-offload bottleneck analysis identifies PCIe
transfer as a dominant limiter and reports transfer-dominated serving.  This
supports the motivation for a resource-domain screen, but it does not provide
evidence for TEMPO-RD's controller or a shared-PFS endpoint.  See:
<https://arxiv.org/abs/2601.19910>.

The instrumentation plan is feasible in principle but must remain
path-specific. NVIDIA's GDS documentation describes GPU-to-NIC RDMA byte
counters, PCI-distance/affinity, dynamic routing, and Lustre-backed paths;
NCCL documentation exposes GDR path cutoffs (`LOC`/`PIX`/`PXB`/`PHB`/`SYS`) and
GDR-read controls. These are candidate evidence sources, not proof that a
particular training flow traversed every tier. See the primary documentation:
<https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html>
and <https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2251/user-guide/docs/env.html>.

The resulting experiment requirement is unchanged: record per-domain path
status and counter support, run a domain-specific intervention, and promote a
controller only when the intervention beats the matched open lane on the
pre-registered foreground metric without moving cost to another primary metric
or violating the correctness/deadline contract.

### Supplemental inference-KV primary-source recheck (2026-08-11)

Three recent primary sources make the inference half of TEMPO-RD narrower, not
broader.  *Understanding Bottlenecks for Efficiently Serving LLM Inference
With KV Offloading* reports that PCIe transfers dominate offloaded-KV serving;
this directly supports measuring the GPU--PCIe--host domain, but it blocks any
claim that TEMPO is the first interconnect-aware KV scheduler.  *CacheOPT*
already treats KV-cache competition as a TTFT/TBT-SLO scheduling problem with
admission, reservation, prefetch, and preemption decisions.  *Tutti* adds a
GPU-native SSD-backed KV path, GPU io_uring, and slack-aware I/O scheduling,
with a large TTFT improvement over its LMCache comparison.  Consequently the
TEMPO-RD inference study must use a current same-endpoint optimized-open lane,
separate PCIe/fabric/storage interventions, and treat these systems as direct
positioning comparators; it cannot claim first KV offload, first slack-aware
KV scheduling, or first GPU-native storage movement.

Sources: [KV-offload bottleneck analysis](https://arxiv.org/abs/2601.19910),
[CacheOPT](https://arxiv.org/abs/2503.13773), and
[Tutti](https://arxiv.org/abs/2605.03375).  None of these sources supplies a
live TEMPO-RD result or makes the current design's shared training/PFS
controller novel by itself.

## 1. What the current code actually implements

The implementation is narrower than the proposed sentence “the collective
group receives aligned D2H credits that bound collective p99 slowdown while
guaranteeing a durability deadline.” The following table is the code-level
contract that should govern paper wording.

| Mechanism | Current semantics | Claim boundary |
|---|---|---|
| Group observation | `TempoV4Backend._local_packet` and `_gather_packets` collect per-rank stage progress and the previous completed FSDP profile on the separate controller group. | It is a common gathered snapshot. It is not a leader decision or consensus protocol. |
| Collective feedback | `TempoV4Backend._update_tail_feedback` reduces each complete signature to slowest-rank exposed latency and corrected arrival spread, compared with warm-up signature-specific p99 references. | This is actual-group, one-step-delayed feedback. It does not causally attribute slowdown to D2H or PFS by itself. |
| Feedback action | `TailFeedback.observe` keeps bounded nonnegative debt; `apply_tail_feedback` adds it to D2H/PFS risk while leaving capacities unchanged. | The controller does **not** enforce a p99 SLO. Tail feedback changes risk order only. |
| Window construction | `TempoV4Backend._windows_from_packets` gives compute windows safe/hard capacity only when the previous profile has a positive corrected cross-rank intersection; collective windows have zero safe capacity and installable hard capacity. | Previous-step intersection-informed prediction, not a synchronized current-step safe window. |
| Risk ordering | `TempoV4Controller._allocate_by_risk` and `_allocate_rank` put compute windows before collective windows, then use stage risk and phase. | Feedback cannot move a collective window ahead of a compute window. The target is compute-first and risk-ordered, not optimal. |
| Deadline projection | `RankProgress.service_requirement_ns` uses `max(D2H_remaining/rate, PFS_remaining/rate) + pipeline_reserve`; `horizon_ns` subtracts finalization and deadline margins. | The selected rates are planning inputs, not certified envelopes. This is a two-stage overlap approximation, not an exact tandem model or deadline guarantee. |
| Receding target | `TempoV4Controller._stage_target` takes request-aligned fair share, last chance, and one-quantum liveness. The minimal fix removes pipeline reserve and the low-slack watermark only from future capacity; `_group_targets` raises ranks to the group maximum requested fraction. | A heuristic receding target, not a necessary/minimum or optimal schedule. |
| Two-stage feasibility | `_protect_pfs_limits` makes `PROTECT` use a group-fair snapshot-host-ready PFS cap; `BALANCED` retains `_allocate_rank`'s chronological same-plan lagged producer prefix. `validate_plan` checks both. | Deterministic greedy producer/consumer coupling. Producer lead is prior-art-adjacent implementation, not novelty. |
| Plan installation | Every rank independently runs `TempoV4Controller.plan`, validates its local result, then `_prepare_plan_transitions` and `_enqueue_phase_transition` transport cumulative prefixes whose deltas activate in local CUDA-stream order. | **Group-snapshot-informed, group-versioned, rank-local stream-ordered admission**; no wall-clock group alignment, leader broadcast, consensus install, or C++ digest install. |
| Stream-token lifetime | `close_step_credit_before_probe` repeats the final cumulative prefix as a zero-delta terminal CLOSE. The runtime emits $2N+2$ tokens for $N$ workload collectives, exactly 54 for the archived $N=26$ step. | The precursor 53-token callback batch ran live; the repaired 54-token checkpoint-active path has only offline fixtures. |
| Logical layout | `state_io_engine.get_last_checkpoint_layout()` is consumed by `_read_logical_layout_envelope` and `_validate_published_logical_layout` without querying the not-yet-created physical file. | Synchronous logical publication is implemented; later file allocation, `fsync`, manifests, global commit, and restore remain separate evidence. |
| Residual bound | `ControllerConfig` and the DataStates admission path use a 1 MiB D2H quantum, 4 MiB PFS request, and 16 MiB maximum PFS in flight per rank. | Bounds admitted nonpreemptible bytes per rank, not a PCIe-root, node, NIC, OST, or collective slowdown. |
| Failure mode | `TempoV4Controller.plan` and `TempoV4Backend._force_drain` latch irreversible `DRAIN`, opening remaining progress subject to data-plane queue caps. | Fail-open for liveness; the tail objective is abandoned and the deadline may still be missed. |
| Durability endpoint | The harness separately tracks data completion, file `fsync`, all-rank validation, and global commit. | Fixed reserves only project toward this endpoint; they do not prove a deadline. |

The frozen evidence ledger is:

| Claim | Evidence | Boundary |
|---|---|---|
| Planning-rate inputs are corrected | Valid job `56500531` DataStates evidence gives archive caps of 5.936536675 GB/s D2H (90% worst unpaced `wait(False)`) and 2.211348539 GB/s PFS (90% conservative full copy-persist pipeline proxy) | The old 0.745221781/0.878992034 GB/s pair is invalid; the new values are caps/proxies, not guarantees or isolated on-path v4 service |
| Mode-specific coupling reduces projected critical credit | Five-event exact geometry replay changes D2H collective credit 939.250 -> 189.250 MiB (-79.9%) and total collective credit 1083.196 -> 389.196 MiB (-64.1%); all five reach generation-9 `FINALIZE` | Replayed credit placement, not measured p99, realized service, or a deadline result |
| A matched rate gate is specified | One 64 MiB/rank qD1/qP4/four-request/16 MiB/`O_DIRECT` pass, selecting the all-rank group minimum only downward from archive caps and retaining provenance | Implemented and offline-validated with final analyzer/cross-file closure; there is no live calibration proof |
| v4 performance exists | None: job `56500531`'s old physical-stat adapter invalidated both candidates before an active plan | Motivation is valid; v4 outcome, first, and SOTA claims are unsupported |

Therefore, the current implementation should **not** be called a mechanism that
“limits collective p99 slowdown.” It observes tail excess and changes a risk
ordering. A limit would require a measured response model and an enforced
constraint such as

\[
\Pr[L_g > (1+\epsilon)L_g^0] \leq \alpha
\]

or an equivalent deterministic envelope. Neither exists in the current code.

## 2. Exact 2024--2026 collision map

“Not found” below means that the cited primary source does not describe the
element as part of its published mechanism. It is not a claim about unpublished
work or author intent.

### 2.1 Direct checkpoint-system comparisons

| System | What the primary source already owns | Exact collision with broad TEMPO wording | Narrow distinction still available to TEMPO |
|---|---|---|---|
| [DataStates-LLM, HPDC'24](https://arxiv.org/html/2406.10707) and [2026 State-Provider revision](https://arxiv.org/html/2601.16956) | Lazy nonblocking D2H during immutable forward/backward, pinned/coalesced host buffering, streaming D2H-to-persistent-storage, `liburing`, separate state providers, shared-Lustre evaluation, and global consistency. The 2026 revision explicitly measures indirect forward/backward interference and notes that checkpoint effective throughput is dictated by the slowest rank. | Blocks first async, phase-overlapped, two-stage, shared-PFS, slowest-rank, and interference-aware motivation claims. The 2024 paper even argues that separate PCIe/NVLink/RDMA paths should avoid interference; this is an assumption TEMPO may test, not ownership of a new data plane. | No published per-FSDP-collective max/arrival-spread feedback that changes byte admission toward an absolute `fsync`/global-commit deadline was found. DataStates is the substrate and direct baseline. |
| [FastPersist, 2024](https://arxiv.org/html/2406.13768) | Derives required checkpoint bandwidth as checkpoint size divided by the next forward/backward window; overlaps checkpoint work with the next iteration; stalls before update if necessary; uses page-locked buffers, double buffering, local NVMe, byte-balanced parallel writers, and socket-aware writer selection to avoid contention. | Blocks first bandwidth-from-window, overlap-feasibility, two-stage buffering, contention-aware writer, and per-iteration persistent checkpoint claims. | Static/profiled compute-window feasibility and local NVMe differ from delayed actual FSDP group-tail feedback and one shared-PFS commit deadline. |
| [PCcheck, ASPLOS'25](https://anakli.inf.ethz.ch/papers/PCcheck_asplos25.pdf) ([artifact](https://github.com/eth-easl/pccheck)) | Persistent concurrent checkpoints; chunked/pipelined GPU-to-CPU and CPU-to-SSD/PMEM paths; tuning of chunk size, storage threads, and checkpoints in flight. | Blocks first chunked tandem pipeline, bounded concurrency, concurrent-checkpoint, and tuning claims. | No actual FSDP group-tail feedback or absolute shared-PFS commit deadline was found. |
| [ByteCheckpoint, NSDI'25](https://www.usenix.org/system/files/nsdi25-wan-borui.pdf) | Parallelism-agnostic checkpoint representation/resharding, workload-balanced writer plans, fully asynchronous D2H-serialization-upload pipeline, ping-pong pinned buffers, storage parallelism, and per-rank checkpoint-stage/I/O heat maps and straggler alerts. | Blocks first full async pipeline, rank-aware load balance, per-rank checkpoint monitoring, resharding, and backend-neutral representation claims. | Its monitored “phases” are checkpoint planning/D2H/serialization/upload/atomic-barrier phases, not closed-loop measurements of actual training FSDP collective groups. |
| [ByteRobust, SOSP'25](https://i2.cs.hku.hk/~cwu/papers/brwan-sosp25.pdf) ([DOI](https://doi.org/10.1145/3731569.3764838)) | Schedules sharded model/optimizer backup in forward/backward communication-idle cycles, chunks and interleaves backup traffic with training communication, uses a separate CUDA stream, and places state in host memory/local SSD/peers. | Strong collision with first phase-aware, communication-gap-aware, chunk-interleaved, fine-grained, or production backup-scheduling claims. | No published complete-FSDP-group tail feedback or shared-PFS `fsync` deadline was found. |
| [FlowCheck, EuroSys'25](https://jhc.sjtu.edu.cn/~bjiang/papers/Huang_EuroSys2025_FlowCheck.pdf) ([DOI](https://doi.org/10.1145/3689031.3696088)) | Mirrors existing all-reduce gradient traffic at switches to CPU checkpoint nodes, pipelines packet dump/parse/update, and requires the shadow update to finish before the next iteration's communication. Periodic remote persistence is separate. | Blocks first collective-traffic-aware checkpoint and first checkpoint deadline/iteration deadline claims. | Specialized mirrored-network/CPU-shadow path; not GPU D2H through a shared PFS and not feedback from measured collective tails. |
| [Checkmate, NSDI'26](https://www.usenix.org/system/files/nsdi26-bhardwaj.pdf) ([official page](https://www.usenix.org/conference/nsdi26/presentation/bhardwaj)) | Programmable-switch multicast of gradients to CPU shadow nodes, per-iteration in-memory checkpoints, shadow-side optimizer reconstruction, and explicit identification that cloning GPU state in copy-persist systems interferes with training. Shadow nodes are provisioned to finish before the next GPU optimizer step. | Blocks first observation of copy-induced training interference, first per-iteration timing constraint, and broad zero-copy/decoupled checkpoint claims. | Different hardware/data path. Its comparison uses in-memory `nullfs` to remove persistent storage as a bottleneck. Critically, FSDP/ZeRO support is discussed but full adaptation/evaluation is explicitly left as future work; it is **not** an evaluated FSDP baseline. |
| [TierCheck, 2026 preprint](https://arxiv.org/html/2605.17821) | Three-tier local/peer/remote checkpointing, differential and base streams, paced microchunks under a bounded per-iteration bandwidth budget, payload-size exchange, adaptive chunk size, safety margin, cross-node ring placement, spill extension, fallback stall, and global consistency/watermarking. | Blocks first paced/adaptive/budgeted transfer, finish-before-next-checkpoint, topology-aware placement, multi-tier commit, and spill-handling claims. | No measured actual FSDP collective p99/arrival feedback or single shared-PFS D2H-plus-`fsync` receding controller was found. |
| [GoCkpt, 2025 preprint](https://arxiv.org/html/2511.07035) | Moves checkpoint pieces across multiple steps, uses low-precision gradients and CPU reconstruction of a consistent version, employs pinned memory/fine chunks, multi-threaded SSD persistence, and explicit NUMA/GPU/SSD affinity. | Blocks first multi-step fine-grained D2H or joint PCIe/SSD optimization claims. | No actual FSDP group-tail feedback tied to a shared-PFS commit deadline was found. |
| [MoC-System, ASPLOS'25](https://jyhuang91.github.io/papers/asplos2025-moc-system.pdf) ([DOI](https://doi.org/10.1145/3676641.3716006)) | Fully sharded rank-balanced MoE checkpointing, partial-expert state reduction, asynchronous GPU-to-CPU snapshot plus CPU-to-distributed-storage persistence, triple buffering, and adaptive snapshot/persist expert counts chosen so snapshot fits the next forward/backward interval. | Blocks first two-level snapshot/persist, adaptive overlap, rank balancing, and compute-window-fit claims. | Its central contribution changes MoE checkpoint contents and recovery semantics; it has no published actual-FSDP-tail admission loop. |
| [FFTrainer, 2025 preprint](https://arxiv.org/html/2512.03644) | Custom communication library with separate TRAIN/STATE queues; state transfer uses communication-idle capacity and D2H into RDMA-visible host memory for neighbor-memory failover. | Blocks first software-controlled idle-network, traffic-priority, or feasibility-aware checkpoint movement claims. | In-memory neighbor recovery rather than shared-PFS durability; no actual collective-tail feedback loop was found. |
| [AdaCheck, FAST'26](https://www.usenix.org/conference/fast26/presentation/liu-weijie) ([paper](https://www.usenix.org/system/files/fast26-liu-weijie.pdf)) | Detects and eliminates tensor redundancy across parallel layouts, architectures, and iterations using offline and online methods. | Blocks checkpoint-state redundancy/adaptive state-elision claims. | Orthogonal byte-volume optimization; TEMPO may consume the smaller state as input but does not own it. |
| [Universal Checkpointing, ATC'25](https://www.usenix.org/system/files/atc25-lian.pdf) ([official page](https://www.usenix.org/conference/atc25/presentation/lian)) | Decouples saved representation from runtime parallelism and supports pattern-based reconfiguration. | Blocks universal layout, flexible resharding, and parallelism-neutral representation claims. | Orthogonal representation layer, not phase/tail admission. |

### 2.2 Newly important collisions missing or understated in the current positioning

| Work | Why it matters for TEMPO |
|---|---|
| [Checkflow, IEEE CAL'25](https://shouxi.name/publications/cal25-checkflow.pdf) ([DOI](https://doi.org/10.1109/LCA.2025.3596616)) | This is distinct from EuroSys FlowCheck. Checkflow jointly schedules snapshot creation, preservation, and GPU-to-CPU offload at operator steps with an ILP. Its per-step transfer constraint is bytes no greater than PCIe bandwidth times operator duration, and it splits tensors until offload fits compute windows. This directly blocks “first phase-local byte capacity/credit” and “first bandwidth-derived compute-window schedule.” It is single-GPU and optimizes peak GPU memory; it assumes the persistent write can use the iteration and does not close a distributed group-tail/PFS-deadline loop. |
| [ECCheck, ICDCS'25](https://i.cs.hku.hk/~cwu/papers/gcqi-icdcs25.pdf) ([DOI](https://doi.org/10.1109/ICDCS63083.2025.00033)) | Uses in-memory erasure-coded checkpoints, serialization-free CPU coding, and communication-idle slots to pipeline coding/communication; it discusses data-, tensor-, pipeline-, and sharded training settings. It blocks broad first idle-slot or communication-aware checkpoint claims, but not TEMPO's shared-PFS/tail-feedback objective. |
| [Understanding LLM Checkpoint/Restore I/O Strategies and Patterns, arXiv Dec. 2025 / SCA/HPCAsiaWS'26](https://arxiv.org/html/2512.24511) | Shows that aggregation, alignment, batching, buffer reuse, and I/O coalescing materially determine PFS throughput. Its realistic benchmark reports up to 3.9x higher write throughput than DataStates-LLM. TEMPO therefore has **no defensible PFS/data-plane throughput novelty**. The current 4 MiB PFS requests should be described as a fine-grained controlled substrate with an unmeasured throughput tax, not an optimized I/O size. Any paper must report that tax and ideally compare against a coalesced substrate. |
| [LLMTailor, 2026 preprint](https://arxiv.org/html/2602.22158) | Selectively retains and merges layers, including optimizer state organization, to reduce checkpoint bytes/time while preserving model quality in its studied cases. This is orthogonal state-selection/content optimization. TEMPO may compose with it but cannot claim selective checkpointing. |
| [BitSnap, 2025 preprint](https://arxiv.org/abs/2511.12376) | Adaptive sparsification and quantization reduce checkpoint state. It reinforces that state volume/accuracy is a separate axis; TEMPO must keep equal-state semantics when evaluating admission. |
| [MoEvement, NSDI'26](https://www.usenix.org/conference/nsdi26/presentation/gandhi) | Sparse expert snapshots across iterations, sparse-to-dense reconstruction, upstream activation/gradient logging, and in-memory recovery. It is an important alternative recovery design, not a raw shared-PFS baseline. |
| [PHOENIX, 2026 preprint](https://arxiv.org/abs/2607.01646) | Off-critical-path in-memory checkpoints plus communicator hot swapping and spare-node recovery. It owns broad zero-overhead/hot-swap recovery territory but uses a different recovery endpoint. The current v2 title is PHOENIX; the earlier DeadPool label must not be used as the system name. |

### 2.3 Pre-2024 anchors that still constrain wording

| System | Claim already occupied |
|---|---|
| [CheckFreq, FAST'21](https://www.usenix.org/conference/fast21/presentation/mohan) | Online profiling, adaptive checkpoint frequency, user-selected overhead bound, and two-phase saving. TEMPO cannot claim first checkpoint feedback or bounded-overhead objective. |
| [Gemini, SOSP'23](https://zhuangwang93.github.io/docs/Gemini_SOSP23.pdf) ([official summary](https://www.amazon.science/publications/gemini-fast-failure-recovery-in-distributed-training-with-in-memory-checkpoints)) | Profiles training communication, chunks checkpoint transfers, schedules network work in communication-idle spans, and schedules local GPU-to-host copies. TEMPO cannot claim first phase/gap awareness, profiled scheduling, or chunked D2H. |

## 3. What is actually left as a candidate contribution

No exact mechanism collision was found for the following combination in the
reviewed corpus:

> On a conventional GPU-to-pinned-host-to-shared-PFS data path, TEMPO gathers
> per-rank checkpoint progress and complete FSDP-group observations; converts
> previous-step slowest-rank exposed latency and clock-corrected arrival spread
> into per-signature risk debt; approximates remaining D2H and PFS service
> against one absolute event deadline using selected capped planning rates and
> fixed reserves; and has every rank independently recompute a deterministic,
> versioned plan whose
> bounded phase-local byte deltas activate in local CUDA-stream order.

This is the **narrowest defensible implementation statement**. It deliberately
does not use “first,” “guarantee,” “bound p99,” “group-aligned activation,”
“optimal,” or “SOTA.”

The most defensible research question is narrower still:

> Does closing the loop on measured complete-FSDP-group tail behavior improve
> the collective-tail versus committed-checkpoint-latency/deadline Pareto
> frontier over the same DataStates copy-persist path, beyond what a fixed byte
> cap or profile-only phase schedule achieves?

The novelty is thus not any primitive, including producer lead.  The narrow
conditional contribution is the **choice of complete-FSDP-group tail signal
and admission objective for the complete FSDP state moving toward persistent
shared-PFS file `fsync` plus all-rank global commit under one absolute
deadline**, plus a demonstrated causal Pareto improvement if the experiment
succeeds.  “No exact collision found” is not “first” or SOTA.

### Why this is still high risk

The tuple is easy for a reviewer to describe as a composition:

1. slowest-rank synchronization is standard;
2. checkpoint interference is already documented by DataStates and Checkmate;
3. phase/gap scheduling is already in Gemini, ByteRobust, FFTrainer, ECCheck,
   and Checkflow;
4. deadline/budget logic is already in FastPersist, FlowCheck, CheckFreq, and
   TierCheck;
5. chunked two-stage paths are already in DataStates, PCcheck, MoC-System, and
   GoCkpt.

Only evidence that the complete-group feedback materially changes the Pareto
frontier can turn this composition into a systems result.

### 3.1 Does preflight or online stage-rate calibration create novelty?

**No, not in isolation.** A calibration-plus-slack controller is useful
engineering, but each broad ingredient already has close prior art.

| Proposed element | Closest collision | Assessment |
|---|---|---|
| Isolated preflight measurement of D2H/PFS or compute-window rates | FastPersist empirically measures forward/backward duration and checkpoint size to derive required bandwidth; Gemini profiles training communication windows; Gossman et al. characterize `liburing`, aggregation, alignment, and PFS concurrency. | Severe collision. Use calibration to make the evaluation valid, not as a headline contribution. |
| Online adaptation of chunk size, rate, concurrency, or checkpoint overhead | PCcheck tunes chunk size, storage threads, and checkpoints in flight; CheckFreq profiles execution/checkpoint overhead and adapts frequency; TierCheck adapts chunk size from exchanged volume and the remaining base-checkpoint interval. | Severe collision at the primitive level. Do not claim first online or adaptive controller. |
| Admission from remaining bytes divided by measured rate | FastPersist's required-bandwidth inequality and TierCheck's volume-over-available-iterations pacing already instantiate the same feasibility intuition. | A two-stage implementation is a refinement, not a standalone novelty. |
| Admission from an absolute persistent-PFS deadline and current slack | FlowCheck owns an iteration deadline, TierCheck owns finish-before-next-base-checkpoint pacing and fallback stall, and CheckFreq owns an overhead constraint. | Endpoint semantics are different, but broad deadline/slack scheduling is occupied. |
| Joint online service envelope plus complete-FSDP-group tail feedback under a shared-PFS `fsync`/global-commit objective | No exact collision was found in the reviewed corpus. | This is the strongest defensible distinction, provided the tail signal actually changes enforced admission and a valid experiment shows a better tail/durability frontier. |

The corrected rate contract uses archive caps of 5.936536675 GB/s D2H and
2.211348539 GB/s PFS, then permits only a down-only group-min selection from one
matched 64 MiB/rank qD1/qP4/`O_DIRECT` preflight.  That path and its provenance
checks are implemented and offline-validated; final analyzer/cross-file gate
closure is complete, but it has no live result.  It still is not an online envelope
or guarantee of later service; held-out
validation would strengthen rigor but not novelty.  The paper should present it as
calibration of a **group-tail-sensitive FSDP admission controller under a
persistent-PFS commit deadline**, which is the narrow remaining distinction.

## 4. Claim ledger

### Safe now

- “TEMPO v4 implements a prototype group-snapshot-informed admission controller
  over a modified DataStates data path.”
- “The prototype observes complete FSDP collective-group max latency and
  corrected rank-arrival spread and uses them as delayed risk-order feedback.”
- “The prototype couples per-rank D2H/PFS progress through `PROTECT`'s
  group-fair snapshot-host-ready producer lead and `BALANCED`'s chronological
  same-plan lagged prefix, using capped planning rates in a remaining-service
  projection.”
- “The minimal deadline-target fix reserves pipeline and low-slack time only
  from future capacity, and all five exact archived-geometry events reach
  generation-9 `FINALIZE`.”
- “The runtime publishes a logical layout synchronously and transports exactly
  $2N+2$ stream tokens, including a zero-delta terminal CLOSE.”
- “The controlled path has explicit per-rank request/residual geometry: up to
  1 MiB per D2H subcopy and 16 MiB of PFS work in flight under the configured
  path.”
- “No exact published mechanism collision was found in the reviewed primary
  corpus as of 2026-08-08.”

The replay and implementation statements above are not measured p99,
group-aligned activation, a deadline guarantee, an optimal/minimum-overlap
result, or live end-to-end evidence.

### Safe only after a valid live evaluation

Use an outcome statement, not a priority statement:

> At equal captured state, actual serialized bytes disclosed, and identical
> file-`fsync` plus global-commit semantics, TEMPO reduced checkpoint-induced
> complete-FSDP-group tail excess relative to the same DataStates path while
> meeting the evaluated deadline, with the measured cost of its fine-grained
> admission substrate.

That sentence is allowed only if all of its predicates are observed. If not all
deadlines are met, report the Pareto curve and miss rate rather than selecting a
favorable point.

### Not defensible for the current implementation

| Do not claim | Reason |
|---|---|
| “First phase-aware/collective-aware checkpoint scheduler.” | Gemini, ByteRobust, FFTrainer, ECCheck, and Checkflow. |
| “First deadline-aware or budgeted checkpointing.” | CheckFreq, FastPersist, FlowCheck, and TierCheck. |
| “First D2H/storage pipeline.” | DataStates, FastPersist, PCcheck, MoC-System, GoCkpt. |
| “First group coordination or straggler-aware checkpointing.” | ByteCheckpoint balances writers and diagnoses rank stragglers; TierCheck exchanges group sizes; the slowest-rank premise is standard. The only narrower difference is feedback from actual training collective groups. |
| “TEMPO bounds/limits collective p99.” | Risk changes order; no response model or p99 constraint is enforced. |
| “TEMPO guarantees the durability deadline.” | Archive caps, a one-pass down-only calibration, fixed reserves, and `DRAIN` cannot guarantee later service under PFS collapse. |
| “TEMPO allocates group-aligned credits.” | Plans derive from a common snapshot but activation is rank-local stream order; there is no common wall-clock gate or consensus install. |
| “TEMPO is topology-aware.” | No measured PCIe-root/NUMA/NIC/OST mapping changes the current plan. |
| “TEMPO's 4 MiB I/O is PFS-optimal or a throughput innovation.” | The Gossman et al. characterization shows strong benefits from aggregation/coalescing. Four MiB is currently a control granularity. |
| “TEMPO eliminates interference.” | Hard-capacity work, residuals, telemetry, and controller traffic remain, and DRAIN deliberately opens admission. |
| “TEMPO beats SOTA.” | There is no valid checkpoint-active v4 result, and many systems use incomparable NVMe, peer memory, CPU shadows, or programmable switches. |

## 5. Positioning corrections reflected in the current audit

The literature re-audit and `TEMPO_V4_POSITIONING.md` now share these
boundaries.

1. **Keep Checkmate's FSDP status precise.** The NSDI'26 paper discusses a
   possible extension and explicitly leaves full adaptation and evaluation to
   future work. The evaluated training path is DDP/DP/PP, not a full FSDP/ZeRO
   result.
2. **Add Checkflow and ECCheck.** They materially narrow phase-local capacity
   and communication-idle scheduling claims.
3. **Strengthen the TierCheck collision.** It does more than generic tiering:
   ranks exchange payload sizes, derive a bounded per-iteration paced schedule,
   adapt chunk size, reserve margin, extend the schedule, and stall on spill.
4. **Refine the DataStates distinction.** The 2026 revision already recognizes
   indirect training interference and slowest-rank checkpoint completion.
   TEMPO's remaining distinction is per-training-collective tail feedback, not
   first slowest-rank or first interference awareness.
5. **Replace “tandem projection” where it implies an exact queue model.** The
   current completion requirement is the maximum of two independent remaining
   service times plus a fixed pipeline reserve. “Two-stage overlap
   approximation with ready-prefix allocation” is exact.
6. **Replace “group-aligned” with “common-snapshot/group-informed and
   rank-local stream-ordered.”** Previous-step cross-rank intersections predict
   capacity; they do not create a current-step common safe gate.
7. **Say that feedback reorders risk, not that it limits p99.** Capacities are
   unchanged by `apply_tail_feedback`, and kind ordering always prefers compute
   windows.
8. **Clarify ByteCheckpoint monitoring.** Its phase heat maps are checkpoint
   pipeline phases, not actual FSDP training collective phases.
9. **Add the 2026 PFS characterization.** It removes any data-plane throughput
   novelty and turns 4 MiB request size into a measured design trade-off.
10. **Keep the no-priority posture.** “No exact tuple collision found” is an
    audit result, not evidence for “first.”
11. **Do not rehabilitate the old service floors.** The 0.745/0.879 GB/s values
    came from the wrong paths.  The 5.936536675/2.211348539 GB/s replacements
    are archive caps/proxies, and the matched down-only calibration is rigor,
    not novelty or a service guarantee.

## 6. Comparison strategy implied by the literature

TEMPO should not try to beat every named system on one throughput bar. The
hardware, failure model, captured state, and durability endpoint differ too
much. The direct experiment should isolate the controller on one data path.

| Role | Required comparison | Purpose |
|---|---|---|
| Same-path upper bound | No checkpoint | Establish the unperturbed complete-group tail. |
| Direct baseline | DataStates with TEMPO admission disabled, same patches needed only for measurement/correctness | Measure the cost of the existing asynchronous copy-persist engine. |
| Open-control control | Same fine-grained 1/4 MiB substrate with unlimited admission | Separate controller benefit from a changed chunk/request path. |
| Simple control | Fixed byte cap or fixed-rate throttle | Test whether group feedback is better than generic pacing. |
| Profile-only ablation | Previous-step windows, no live tail debt | Isolate the value of actual group-tail feedback. |
| Deadline-only ablation | Remaining-service projection, static risk | Isolate the value of deadline pressure. |
| Data-plane sensitivity | Coalesced/large-request DataStates-compatible path if feasible | Quantify the throughput tax of 4 MiB PFS control granularity highlighted by the 2026 I/O study. |

The primary plot should be a frontier, not one winner number:

- x-axis: committed checkpoint latency, deadline, or deadline-miss rate;
- y-axis: checkpoint-induced complete-FSDP-group p99 excess;
- annotations: DRAIN fraction, critical-window bytes, serialized bytes, PFS
  throughput, and controller overhead.

Specialized systems such as Checkmate, FlowCheck, TierCheck, Gemini, and
ByteRobust belong in the mechanism taxonomy unless their exact artifact can be
run with equivalent state and durability semantics. Their published raw
numbers are not comparable to Perlmutter shared-PFS `fsync`.

## 7. If a stronger new structure is desired

The current narrow claim may be publishable with strong evidence, but it is not
a strong architectural discontinuity. The clearest route to a more defensible
new structure is to implement properties that the present code only names.

1. **Make the collective-tail SLO an enforced constraint.** Learn or calibrate
   a monotone response envelope from resource-domain admitted bytes to group
   tail, and reduce capacity when the confidence bound violates the target.
   Risk reordering alone is insufficient.
2. **Allocate credits per shared resource domain.** Enforce one budget across
   GPUs sharing a PCIe root/NUMA path and one across writers sharing NIC/OST
   resources. The present per-rank bound can sum to 16 MiB D2H and 256 MiB PFS
   over 16 ranks without a group cap.
3. **Install one group decision.** Use a leader/broadcast or digest-checked
   consensus decision and a defined activation epoch/token. This would justify
   “group-coordinated credit” while still avoiding a blocking callback.
4. **Use an uncertainty-aware service model.** Replace archive-capped planning
   inputs and the `max` approximation with measured stage-active service
   envelopes, producer-ready queue state, and an explicit chance-constrained or
   robust commit-deadline test.
5. **Separate control granularity from storage granularity.** Preserve small
   admission quanta while coalescing admitted regions into larger PFS
   submissions. This directly addresses the 2026 aggregation result and avoids
   paying for control with obviously fragmented storage I/O.

That design would support a stronger future statement:

> TEMPO is a resource-domain, group-installed controller that enforces a
> measured collective-tail SLO while scheduling a coalesced two-stage
> checkpoint pipeline toward a probabilistic commit deadline.

The current implementation does **not** yet support that statement.

## 8. Primary-source index

- DataStates-LLM: [HPDC'24 paper](https://arxiv.org/html/2406.10707),
  [2026 State-Provider revision](https://arxiv.org/html/2601.16956),
  [artifact](https://github.com/DataStates/datastates-llm)
- FastPersist: [paper](https://arxiv.org/html/2406.13768)
- PCcheck: [author PDF](https://anakli.inf.ethz.ch/papers/PCcheck_asplos25.pdf),
  [artifact](https://github.com/eth-easl/pccheck)
- ByteCheckpoint: [official NSDI'25 PDF](https://www.usenix.org/system/files/nsdi25-wan-borui.pdf)
- ByteRobust: [author PDF](https://i2.cs.hku.hk/~cwu/papers/brwan-sosp25.pdf),
  [DOI](https://doi.org/10.1145/3731569.3764838)
- FlowCheck: [author PDF](https://jhc.sjtu.edu.cn/~bjiang/papers/Huang_EuroSys2025_FlowCheck.pdf),
  [DOI](https://doi.org/10.1145/3689031.3696088)
- Checkmate: [official NSDI'26 PDF](https://www.usenix.org/system/files/nsdi26-bhardwaj.pdf),
  [official page](https://www.usenix.org/conference/nsdi26/presentation/bhardwaj)
- TierCheck: [paper](https://arxiv.org/html/2605.17821)
- GoCkpt: [paper](https://arxiv.org/html/2511.07035)
- MoC-System: [author PDF](https://jyhuang91.github.io/papers/asplos2025-moc-system.pdf),
  [DOI](https://doi.org/10.1145/3676641.3716006)
- FFTrainer: [paper](https://arxiv.org/html/2512.03644)
- AdaCheck: [official FAST'26 PDF](https://www.usenix.org/system/files/fast26-liu-weijie.pdf)
- Universal Checkpointing: [official ATC'25 PDF](https://www.usenix.org/system/files/atc25-lian.pdf)
- Checkflow: [author PDF](https://shouxi.name/publications/cal25-checkflow.pdf),
  [DOI](https://doi.org/10.1109/LCA.2025.3596616)
- ECCheck: [author PDF](https://i.cs.hku.hk/~cwu/papers/gcqi-icdcs25.pdf),
  [DOI](https://doi.org/10.1109/ICDCS63083.2025.00033)
- Understanding LLM Checkpoint/Restore I/O Strategies and Patterns:
  [paper](https://arxiv.org/html/2512.24511)
- LLMTailor: [paper](https://arxiv.org/html/2602.22158)
- BitSnap: [paper](https://arxiv.org/abs/2511.12376)
- MoEvement: [official NSDI'26 page](https://www.usenix.org/conference/nsdi26/presentation/gandhi)
- PHOENIX: [paper](https://arxiv.org/abs/2607.01646)
- CheckFreq: [official FAST'21 page](https://www.usenix.org/conference/fast21/presentation/mohan)
- Gemini: [author PDF](https://zhuangwang93.github.io/docs/Gemini_SOSP23.pdf),
  [official summary](https://www.amazon.science/publications/gemini-fast-failure-recovery-in-distributed-training-with-in-memory-checkpoints)

## Audit caveat

This search covered the named systems, official 2025--2026 USENIX proceedings,
and targeted searches for LLM checkpointing with D2H, collective latency,
arrival skew, phase/window scheduling, pacing, tail feedback, and durability
deadlines. It cannot prove that no unpublished, proprietary, under-review, or
differently named collision exists. Refresh the search immediately before a
submission and avoid a priority claim even if the search remains negative.
