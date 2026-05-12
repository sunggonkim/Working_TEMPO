# TEMPO: AI Coding Agent Instructions

## Research Vision & Positioning

**TEMPO** is evolving from a checkpoint-pacing system into an **async, topology-aware LLM orchestration middleware for HPC clusters** — targeting OSDI/SOSP publication.

### The Core Problem (Two Interference Paths)

In distributed LLM training on Perlmutter (NERSC), checkpoint I/O competes with NCCL all-reduce across two hardware bottlenecks:

1. **PCIe Root Complex** (within-node): GPU→NVMe DMA and NCCL gradient buffers share the same AMD EPYC PCIe I/O die → AllReduce latency +50%.
2. **Slingshot-11 Fabric** (across-node): Checkpoint flushes to Lustre share 200 Gbps optical global links with NCCL → bandwidth collapse to 11 GB/s.

### Research Gap vs. OSDI/SOSP State of the Art

Existing OSDI/SOSP work (DistServe 2024, Pie SOSP 2025, Aegaeon SOSP 2025, Teola OSDI 2024) targets cloud TCP/IP environments and assumes abstract network topology. **TEMPO's novelty**: software-level exploit of Perlmutter's specific HPC hardware (Multi-rail Slingshot-11, PCIe topology, Lustre) **without any hardware modification**.

Three research pillars being developed:

| Pillar | Approach | Key File |
|--------|----------|----------|
| **Async Multi-rail RDMA** | `libfabric` CXI provider direct control; KV-cache transfer offloaded as background RDMA ops; Slingshot QoS TC mapping | [tempo/qos_mapper.py](tempo/qos_mapper.py) |
| **Topology-aware PCIe routing** | Dynamic multipath routing; NVLink for intra-node, RDMA via idle NIC for inter-node; PCIe bandwidth look-ahead scheduling | [tempo/network_monitor.py](tempo/network_monitor.py) |
| **Lightweight analytical prediction** | O(1) look-ahead: given batch size + token length → predict KV-cache bytes and PCIe pressure N ms ahead; no ML model | [tempo/service_gain.py](tempo/service_gain.py) |

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

### Pillar 1 — Async Multi-rail RDMA via `libfabric` CXI + Slingshot QoS

Perlmutter nodes have **4 GPUs × 4 Slingshot-11 NICs mapped 1:1**. Generic frameworks ignore this and stripe blindly, causing both PCIe and network bottlenecks.

- **Mechanism**: Use `libfabric` CXI provider API to post KV-cache transfers as non-blocking RDMA ops; GPU proceeds immediately.  
- **Traffic separation**: Mark I/O sockets with `socket.IP_TOS` (DSCP) to route into Slingshot hardware TC queues. NCCL AllReduce uses TC3 (Expedited Forwarding); checkpoint flush uses TC1/TC0 (Best-Effort). The switch ASIC enforces priority — zero CPU overhead.
- **Key file**: `tempo/qos_mapper.py` — `QoSMapper.apply(socket_fd, gain_score)` sets DSCP. TC thresholds: TC3 (gain ≥ 0.70), TC2 (0.40–0.70), TC1 (0.15–0.40), TC0 (< 0.15).

### Pillar 2 — Topology-Aware PCIe Multipath Routing

- **Mechanism**: Intra-node data (GPU→GPU) forced through NVLink; only inter-node traffic exits via NIC. If NIC 0's PCIe lanes are saturated, orchestrator reroutes through idle NIC 1/2.
- **Policy**: `NetworkMonitor` polls `/sys/class/net/hsn{0,1}/statistics/tx_bytes` at 5 ms intervals (EMA α=0.25). When util > 75% of 200 Gbps, `wait_for_bw_headroom()` blocks that flush job.
- **Key file**: `tempo/network_monitor.py` — `CassiniHWCounters` reads Slingshot ASIC performance counters for per-rail utilization.

### Pillar 3 — O(1) Lightweight Analytical Prediction (no ML)

LLM compute is deterministic: batch_size × seq_len → exact KV-cache bytes.

```python
# Look-ahead: given a pending request, predict PCIe pressure N ms from now
kv_bytes = num_layers * num_heads * head_dim * seq_len * 2 * dtype_bytes
flush_bw_needed = kv_bytes / target_flush_window_s
# If flush_bw_needed + current_nccl_bw > PCIe_ceiling (64 GB/s):
#   → micro-delay this job OR route to alternate NIC rail
```

- **Key file**: `tempo/service_gain.py` — `ServiceGainScheduler` computes per-job gain scores (α·learning_progress + β·recovery_value + γ·urgency) in O(1) and dispatches to a priority heap. Jobs with gain < 0.30 are deferred during network congestion.

---

## Competitive Positioning (Head-to-Head vs. OSDI/SOSP Papers)

| System | Venue | Their Assumption | TEMPO Differentiator |
|--------|-------|-----------------|---------------------|
| DistServe | OSDI 2024 | Prefill/decode disaggregation on cloud VMs | TEMPO targets intra-step PCIe + Slingshot interference in HPC |
| Pie | SOSP 2025 | Wasm Inferlets, async I/O during generation | TEMPO is hardware-topology-aware; directly maps to Slingshot TC |
| Aegaeon | SOSP 2025 | Token-level preemption across multiple cloud models | TEMPO is single-workload but exploits physical fabric QoS |
| Teola | OSDI 2024 | DAG decomposition over TCP/IP clusters | TEMPO uses `libfabric` RDMA + sysfs hardware counters |

**TEMPO novelty claim**: software middleware that exploits Perlmutter's Multi-rail Slingshot-11 + PCIe topology to achieve hardware-level interference isolation without any hardware modification — something no cloud-targeting system paper does.

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
5. **Versioning**: Multiple scheduler versions (V1–V5) coexist; add features as new version, not branch.

---

## Questions? Check These First

- **"How do I add new scheduler logic?"** → Extend `TEMPOScheduler` (see phase3/train_with_tempo.py).
- **"How do I measure the improvement?"** → Use phase3 or phase7 eval + CSV → plot with scripts/plot_readme_figures.py.
- **"Does this work on non-Perlmutter?"** → Phase annotation is portable; NetworkMonitor falls back to `/proc/net/dev`; checkpoint I/O pattern is general.
- **"Why so many scheduler versions?"** → Each version adds features (V3: sparse transfer, V4: P2P cache, V5: Nexus integration) without breaking prior interfaces.
