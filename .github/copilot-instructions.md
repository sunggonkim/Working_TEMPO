# TEMPO: AI Coding Agent Instructions

---

## ⛔ Research Integrity — ABSOLUTE RULES (No Exceptions)

Violating these rules = research misconduct. These apply to every file, README, paper draft, and comment.

1. **Never fabricate or estimate numbers.** Every metric in README.md, paper drafts, or comments MUST be computed directly from raw CSV files in `results/`. Before writing any number, run a verification script like:
   ```python
   import csv, statistics
   rows = list(csv.DictReader(open("results/pcie_contention/timeline_baseline.csv")))
   print(statistics.mean(float(r["allreduce_ms"]) for r in rows))
   ```
   If the CSV does not exist, write `[NOT YET MEASURED]` — never invent a plausible-sounding value.

2. **SLURM job IDs are mandatory provenance.** Every table or figure in the README must cite the exact SLURM job ID that produced the data (e.g., `job 52848625`). If the job ID is unknown, mark it `[job ID unknown — re-run required]`.

3. **Never adjust results to match a hypothesis.** If measured improvement is +3.4%, write +3.4%. Do not round up to +9.9% or reframe as something else. Negative results are publishable; fabricated results end careers.

4. **Before every README or paper number edit:** Open the source CSV, compute the statistic, confirm it matches. If there is a discrepancy, fix the README to match the data — never the reverse.

5. **Simulation / synthetic data must be clearly labelled.** Any number from `scripts/simulate_chunk_sweep.py` or any non-SLURM source must be tagged `[simulated]` or `[synthetic]` everywhere it appears.

6. **v6 Pillar implementations (gpu_driven.py, nvlink_router.py, libfabric_qos.py) are currently PoC/prototype.** Do not claim hardware-measured speedups for these until SLURM benchmarks exist. Use `[not yet benchmarked on Perlmutter]` as the placeholder.

---

## Research Vision & Positioning

**TEMPO** is evolving from a checkpoint-pacing system into an **async, topology-aware LLM orchestration middleware for HPC clusters** — targeting OSDI/SOSP publication.

### The Core Problem (Two Interference Paths)

In distributed LLM training on Perlmutter (NERSC), checkpoint I/O competes with NCCL all-reduce across two hardware bottlenecks:

1. **PCIe Root Complex** (within-node): GPU→NVMe DMA and NCCL gradient buffers share the same AMD EPYC PCIe I/O die → AllReduce latency +50%.
2. **Slingshot-11 Fabric** (across-node): Checkpoint flushes to Lustre share 200 Gbps optical global links with NCCL → bandwidth collapse to 11 GB/s.

### Research Gap vs. OSDI/SOSP State of the Art

Existing OSDI/SOSP work (DistServe 2024, Pie SOSP 2025, Aegaeon SOSP 2025, Teola OSDI 2024) targets cloud TCP/IP environments and assumes abstract network topology. **TEMPO's novelty**: software-level exploit of Perlmutter's specific HPC hardware (Multi-rail Slingshot-11, PCIe topology, Lustre) **without any hardware modification**.

Three research pillars (V6 implemented; V7 roadmap below):

| Pillar | V6 Status | Key File |
|--------|-----------|----------|
| **P1: GPU-Driven NIC Doorbell** | PoC implemented; `cudaMemcpyAsync` MMIO trigger; fallback to CPU `fi_send` | [tempo/gpu_driven.py](tempo/gpu_driven.py) |
| **P2: NVLink PCIe Multipath** | PoC implemented; Active-Standby failover at 80% NIC util; sysfs EMA polling | [tempo/nvlink_router.py](tempo/nvlink_router.py) |
| **P3: libfabric CXI TC Control** | PoC implemented; `fi_setopt` per-transfer TC mapping; 4 TC levels | [tempo/libfabric_qos.py](tempo/libfabric_qos.py) |

**V7 Roadmap (OSDI/SOSP hardening — not yet implemented):**

| Improvement | Target File | OSDI/SOSP Motivation |
|-------------|-------------|----------------------|
| **P1→ GICC Hybrid Progress**: separate lightweight NUMA-pinned host thread for `FI_PROGRESS_MANUAL` + CQ retirement + credit-based PCIe rate control | `tempo/gpu_driven.py` | Fixes CQ stall when GPU triggers but CPU never drains; prevents LLC overflow on high-rate RDMA ingest |
| **P2→ Slice Spraying (Active-Active multi-rail)**: split single KV chunk into 4 sub-chunks, relay via NVLink to 4 GPUs, egress simultaneously through 4 × hsn NICs | `tempo/nvlink_router.py` | Replaces Active-Standby with true 4-rail aggregation → ~4× node egress BW; vs FuseLink OSDI 2025 |
| **P3→ Endpoint Multiplexing**: pre-create 4 `fi_endpoint` objects at init (one per TC), route transfers to the correct endpoint at call time | `tempo/libfabric_qos.py` | Eliminates per-transfer `fi_setopt` lock overhead; runtime TC control cost → 0 ns |
| **New Pillar: Operation-level Nano-batching**: per-layer CUDA stream triggers GPU doorbell immediately after each layer's KV computation completes | `tempo/nano_overlap.py` | GPU SM never idles waiting for bulk transfer to start; vs NanoFlow OSDI 2025 |

### The Current Solution (Phase-Gate, proven)

**Phase-based gating**: `PhaseMonitor` detects NCCL vs. compute phases; `CheckpointManager` pauses Lustre flush during NCCL windows via threading `Event`. Result: AllReduce −50.1%, DMA −21.7% (phase7 data).

---

## Architecture: Component Map

### Core Components (`tempo/` module)

| Component | V | Purpose | Key Pattern |
|-----------|---|---------|-------------|
| `PhaseMonitor` | V1+ | Tracks training phase (COMPUTE / NCCL_COMM / CHECKPOINT) | Thread-safe `RLock` + `threading.Event` gating; zero overhead on training path |
| `CheckpointManager` | V1+ | O(1) local NVMe save + background Lustre flush | Chunk-based flush; `_flush_worker` polls `wait_for_io_allowed()` before each chunk |
| `TEMPOScheduler` | V1 | Orchestrator: monitor + manager | `compute_phase()` / `nccl_phase()` context managers; `checkpoint()` API |
| `TEMPOSchedulerV2` | V2 | Adds NetworkMonitor + ServiceGain + Interleaving | Co-schedules I/O and NCCL using bandwidth budget |
| `NetworkMonitor` | V2+ | Sysfs-based Slingshot-11 NIC polling | EMA + rolling max; blocks flush when util > 75% of 200 Gbps link |
| `ServiceGainScheduler` | V2+ | Priority heap for flush jobs | Score = α·learning_progress + β·recovery_value + γ·urgency; defers gain < 0.30 |
| `QoSMapper` | V2+ | Maps service-gain score → Slingshot TC via `socket.IP_TOS` (DSCP) | TC3=NCCL/high-gain, TC1=checkpoint, TC0=background bulk |
| `SparseTransferFilter` | V3+ | Attention-probe sparse KV selection (~8.5× reduction) | Hot ~12% tokens only; InfiniGen-style selection |
| `P2PCacheStore` | V4+ | DHT-style P2P DRAM/NVMe cache pool | Eliminates Lustre metadata latency for hot checkpoints |
| `NanoOverlapController` | V4+ | Per-layer CUDA stream pipelining | Overlaps next-layer compute with current-layer KV flush |
| `NexusCoordinator` | V5 | Nexus interoperability layer | Batch sizes + preemption (phase8) |

### Data Flow

```
Training Loop
  │
  ├─► PhaseMonitor (context manager signals NCCL vs COMPUTE)
  │     └─→ sets _io_allowed Event
  │
  ├─→ TEMPOScheduler.checkpoint(state_dict, step)
  │     └─→ CheckpointManager.save_async()
  │          ├─ Fast: Write to /tmp (local NVMe)
  │          └─ Background thread flushes to $PSCRATCH (Lustre)
  │             ├─ Wait for phase_monitor._io_allowed before each chunk
  │             └─ Respect NetworkMonitor bandwidth headroom (if available)
  │
  └─→ [FSDP / DDP all-reduce] — PhaseMonitor.fsdp_comm_hook auto-signals
```

---

## Three Research Pillars (OSDI/SOSP Novelty)

Unlike cloud systems (DistServe, Pie, Aegaeon, Teola) that target TCP/IP and abstract topology, TEMPO exploits Perlmutter's **specific HPC hardware** purely in software — no FPGA, no SmartNIC RTL changes.

### Pillar 1 — GPU-Driven NIC Doorbell (V6 PoC → V7 GICC Hybrid)

**V6 (implemented):** `GpuDrivenPool` in `tempo/gpu_driven.py` maps Cassini NIC MMIO page via `cudaHostRegister + cudaHostGetDevicePointer`. GPU kernel triggers RDMA send by writing 8-byte token via `cudaMemcpyAsync`. Falls back to CPU `fi_send` when `libfabric` CXI unavailable.

**V7 (roadmap):**
- Replace `cudaMemcpyAsync` with in-kernel MMIO write (`*doorbell_ptr = token`) for true zero-CPU-involvement.
- Add lightweight NUMA-pinned host monitor thread (isolated core, `pthread_setaffinity_np`) that calls `fi_progress(domain, FI_PROGRESS_MANUAL)` to drain Completion Queues and retire NIC resources — GPU cannot do this in OFI/CXI.
- Add credit-based ingress rate control in the monitor thread to prevent LLC overflow when RDMA data arrives faster than the CPU can process.

### Pillar 2 — NVLink PCIe Multipath Routing (V6 Active-Standby → V7 Slice Spraying)

**V6 (implemented):** `NVLinkRouter` in `tempo/nvlink_router.py` polls `/sys/class/net/hsn{i}/statistics/tx_bytes` at 5 ms (EMA α=0.3). When primary NIC > 80% utilization, `select_egress_gpu()` redirects to an idle NIC via NVLink P2P copy. O(1) decision, no ML.

**V7 (roadmap):**
- Replace Active-Standby with **Active-Active Slice Spraying**: split a KV-cache chunk into 4 equal sub-chunks, relay each via NVLink to GPU{0,1,2,3}, egress all 4 simultaneously through hsn{0,1,2,3} → aggregate ~4× single-NIC BW (~212 GB/s node egress).
- Replace sysfs 5 ms poll with memory-mapped Cassini hardware performance counters (nanosecond granularity) for sub-millisecond saturation detection.

### Pillar 3 — libfabric CXI TC Control (V6 per-call → V7 endpoint multiplexing)

**V6 (implemented):** `FabricQoSManager` in `tempo/libfabric_qos.py` calls `fi_setopt(ep, FI_OPT_ENDPOINT, FI_OPT_CXI_TRAFFIC_CLASS, &tc)` per transfer. Gain score → TC mapping: LOW_LATENCY=6 (≥0.70), BULK=4 (0.40–0.70), STORAGE=2 (0.15–0.40), BEST_EFFORT=1 (<0.15). `cxi_dry_run=True` safe on non-Perlmutter.

**V7 (roadmap):**
- Pre-create **4 `fi_endpoint` objects** at init time, one per TC level. Route each transfer to the pre-configured endpoint by TC — eliminates per-transfer `fi_setopt` call and associated libfabric lock overhead (runtime TC cost → 0 ns).
- Bind all endpoints to the same VNI to maintain application isolation while getting hardware-queue-level prioritization.

---

## Competitive Positioning (Head-to-Head vs. OSDI/SOSP Papers)

| System | Venue | Their Assumption | TEMPO Differentiator |
|--------|-------|-----------------|---------------------|
| DistServe | OSDI 2024 | Prefill/decode disaggregation on cloud VMs | TEMPO targets intra-step PCIe + Slingshot interference in HPC |
| Pie | SOSP 2025 | Wasm Inferlets, async I/O during generation | TEMPO maps directly to Slingshot hardware TC; Pie uses SW scheduler only |
| Aegaeon | SOSP 2025 | Token-level preemption across multiple cloud models | TEMPO is single-workload but exploits physical fabric QoS |
| Teola | OSDI 2024 | DAG decomposition over TCP/IP clusters | TEMPO uses `libfabric` RDMA + sysfs hardware counters |
| FuseLink | OSDI 2025 | Multi-rail aggregation for LLM comm | TEMPO achieves same via NVLink relay + Slingshot; no custom switches |
| NanoFlow | OSDI 2025 | Nano-batch compute/comm overlap | TEMPO targets checkpoint I/O overlap, not only intra-model comm |
| Blink / ShadowServe | ASPLOS/arXiv 2025 | CPU-bypass via SmartNIC/DPU | TEMPO achieves CPU-bypass in pure SW (libfabric CXI MMIO + CUDA) — no DPU required |

**TEMPO novelty claim**: pure software middleware that exploits Perlmutter's specific topology (Multi-rail Slingshot-11 · NVLink P2P · AMD EPYC PCIe I/O die) without any hardware modification, DPU, or kernel patch — and targets the HPC checkpoint I/O interference problem that all cloud-targeting papers ignore.

---

## Critical Patterns & Conventions

### 1. **Phase Annotation** (Required for Pacing)

Training loops **must** signal phase transitions. Three integration tiers:

**Option A: Manual context managers** (simplest, recommended for new code)
```python
from tempo import TEMPOScheduler

tempo = TEMPOScheduler(rank=rank, world_size=ws, mode="tempo")

for step in range(n_steps):
    tempo.on_step_begin(step)
    
    with tempo.compute_phase():  # ← signals PhaseMonitor: I/O allowed
        output = model(input_ids)
        loss = output.loss
        loss.backward()
    
    with tempo.nccl_phase():      # ← signals PhaseMonitor: I/O paused
        optimizer.step()  # (AllReduce inside DDP.step() or FSDP backward hook)
    
    if step % ckpt_every == 0:
        tempo.checkpoint(model.state_dict(), step)
```

**Option B: FSDP comm hook** (automatic, but opaque to backward pass)
```python
from tempo.phase_monitor import PhaseMonitor

monitor = PhaseMonitor(rank=rank)
model.register_comm_hook(monitor, PhaseMonitor.fsdp_comm_hook)
# → Automatically signals NCCL phase during gradient reduction
```

**Option C: DDP comm hook**
```python
monitor = PhaseMonitor(rank=rank)
model.register_comm_hook(None, monitor.make_ddp_comm_hook())
```

### 2. **Checkpoint Staging** (O(1) Latency Guarantee)

```python
# Do NOT write directly to Lustre in the training loop:
# ❌ torch.save(state_dict, "/pscratch/ckpt_step_100.pt")  # blocks ~seconds

# Instead:
tempo.checkpoint(state_dict, step)  # returns immediately (~10s ms to local NVMe)
# → Background daemon flushes to $PSCRATCH in configurable chunks
# → Respects NCCL phase gating
```

Key parameters:
- `flush_chunk_mb`: bytes per flush iteration (default 128 MB). Smaller → finer throttling; larger → higher throughput.
- `adaptive_chunk`: auto-tune chunk size based on rolling-avg NCCL duration. Useful when phase durations vary (e.g., gradient accumulation).

### 3. **Mode Selection** (Reproduce Paper Results)

```python
TEMPOScheduler(mode="baseline")  # Greedy flush, reproduces contention for comparison
TEMPOScheduler(mode="tempo")     # Paced flush, paper contribution
```

- **baseline**: Checkpoint written directly to Lustre → saturates NIC → NCCL bandwidth saws (low, high, low, high).
- **tempo**: Checkpoint staged to NVMe → flushed in phases → NCCL bandwidth stays flat.

### 4. **Rolling Window Metrics** (Adaptive Tuning)

PhaseMonitor maintains a circular buffer of recent NCCL phase durations:

```python
monitor._nccl_durations_ms  # deque of last 16 NCCL phase times (ms)
monitor.nccl_phase_duration_ms  # property: latest Gaussian-smoothed estimate
```

CheckpointManager uses this to adapt flush chunk size:
```python
if adaptive_chunk:
    # Aim to write ~50% of an NCCL window per chunk
    target_chunk = int(monitor.nccl_phase_duration_ms * 1e-3 * LUSTRE_BW * 0.5)
    chunk = clamp(target_chunk, MIN_CHUNK, MAX_CHUNK)
```

---

## Experimental Structure (phase0 → phase8)

Each phase isolates one system dimension. **Learn from existing phases before adding new experiments.**

| Phase | Focus | Key Script | Metric |
|-------|-------|-----------|--------|
| phase0 | PCIe timeline profiling; ITL CDF | `run_itl_cdf_eval.slurm` | AllReduce latency distribution |
| phase1 | Node scaling (2→4→8 node NCCL BW) | `run_phase1_*.slurm` | BW degradation curve |
| phase3 | "Killer Graph" (baseline vs tempo) | `run_evaluation.slurm` | BW over time with checkpoint spikes |
| phase4 | I/O + NCCL collisions (network sweep) | `run_io_nccl_sweep.slurm` | NCCL BW under varying flood rates |
| phase5 | Topology path layout (QoS mapping) | `run_phase5_eval.slurm` | Path diversity impact |
| phase7 | Full E2E timeline (PCIe + network) | `run_phase7_eval.slurm` | Integrated improvement breakdown |
| phase8 | Nexus interoperability | `run_nexus_eval.slurm` | Batch sizes, preemption latency |

**Workflow**: 
1. Review existing phase's SLURM script, training script, and plotting script.
2. CSV data → `results/{phase_name}/` directory.
3. Plot with `scripts/plot_readme_figures.py` or phase-specific plot script.

---

## Key Files & Integration Points

### Entry Points for New Features

| File | When to Edit | Pattern |
|------|---|---------|
| [tempo/__init__.py](tempo/__init__.py) | Adding new scheduler version (V6, etc.) | Extend `TEMPOScheduler` base class; update exports |
| [tempo/scheduler.py](tempo/scheduler.py#L48) | Core scheduling logic | Add new mode (e.g., "tempo_v6") in `__init__` + mode dispatch in `on_step_begin()` |
| [tempo/phase_monitor.py](tempo/phase_monitor.py#L48) | Phase detection (new DDP variant, etc.) | Add new `@contextmanager` or comm hook method |
| [tempo/checkpoint_manager.py](tempo/checkpoint_manager.py#L60) | Flush policy (new throttle strategy, e.g., token budget) | Extend `_flush_worker()` loop with new blocking condition |
| [phase3/train_with_tempo.py](phase3/train_with_tempo.py) | Baseline training model | Template for new evaluation; ensure FSDP wrapping matches |
| [src/c_api/](src/c_api/) | CUPTI-based profiling hooks | Rebuild CMake after edits; link via `ctypes.CDLL()` |

### Configuration & Environment

```bash
# Standard TEMPO environment variables:
export PSCRATCH="/pscratch/sd/s/$USER"     # Lustre destination
export CUDA_DEVICE_ORDER=PCI_BUS_ID        # Fixes GPU ordering on Perlmutter
export NCCL_P2P_DISABLE=1                  # Force AllReduce via NIC (not direct GPU P2P)
export NCCL_IB_QPS_PER_CONNECTION=4        # Slingshot fabric tuning
export OMP_NUM_THREADS=1                   # Avoid MKL contention
```

### SLURM Job Patterns (Perlmutter-specific)

```bash
# Standard TEMPO eval on 4 nodes, 4 GPUs/node:
srun --ntasks=16 --ntasks-per-node=4 --gpus-per-node=4 \
     python phase3/train_with_tempo.py --mode tempo --num-steps 400

# With profiling (CUPTI):
srun --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 \
     python phase7/pcie_timeline_profiler.py --output results/phase7/trace.json
```

---

## Debugging & Profiling Conventions

### Common Issues & Solutions

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `ModuleNotFoundError: tempo` | PYTHONPATH not set | Ensure repo root in `sys.path` or run from workspace root |
| Phase monitor hangs (checkpoint never unpaused) | Missing `tempo.on_step_begin()` call | Each step **must** call this to reset phase state |
| CSV shows zero NCCL bandwidth | PhaseMonitor not running; checkpoints not being saved | Add logging in phase transition; check temp directories |
| PCIe/network interference doesn't reproduce | Workload insufficient; network queueing not saturated | Increase `num_steps`, reduce `ckpt_every`, or run concurrent jobs |
| AllReduce latency spikes persist in tempo mode | Adaptive chunk did not converge; or phase signals too late | Lower `adaptive_chunk` or manually reduce chunk size by 50% |

### Logging Setup

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
)
# tempo.phase_monitor, tempo.checkpoint_manager, tempo.scheduler all log at INFO/DEBUG
```

---

## Development Principles

1. **O(1) Training Loop Overhead**: Phase annotation adds ~microseconds; zero if using comm hooks.
2. **Thread Safety**: All state in PhaseMonitor uses `RLock`; CheckpointManager uses `queue.Queue`.
3. **Hardware-Specific Fallbacks**: NetworkMonitor detects Slingshot sysfs; falls back to `/proc/net/dev` on other systems.
4. **Reproducibility**: All experiments log random seeds, model state, environment; CSV outputs are idempotent.
5. **Versioning**: Multiple scheduler versions (V1–V6) coexist; add features as new version, not branch.
6. **Research Integrity (non-negotiable)**:
   - Every number in README / paper drafts must be verified against raw CSV before being written.
   - V6 Pillar files (`gpu_driven.py`, `nvlink_router.py`, `libfabric_qos.py`) are PoC until SLURM benchmarks confirm hardware results. Do not claim measured speedups for them.
   - Simulated or synthetic data must be explicitly labelled `[simulated]` everywhere.
   - If a benchmark hasn't been run yet, write `[not yet measured]` — never estimate.
   - SLURM job IDs are mandatory provenance for all performance claims.

---

## Questions? Check These First

- **"How do I add new scheduler logic?"** → Extend `TEMPOScheduler` (see phase3/train_with_tempo.py).
- **"How do I measure the improvement?"** → Use phase3 or phase7 eval + CSV → plot with scripts/plot_readme_figures.py.
- **"Does this work on non-Perlmutter?"** → Phase annotation is portable; NetworkMonitor falls back to `/proc/net/dev`; checkpoint I/O pattern is general.
- **"Why so many scheduler versions?"** → Each version adds features (V3: sparse transfer, V4: P2P cache, V5: Nexus integration) without breaking prior interfaces.
