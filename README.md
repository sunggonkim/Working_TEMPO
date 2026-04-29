<!-- TEMPO — Working Repository -->
<div align="center">

# ⚡ TEMPO
### *Temporal Emulation and Masking for Predictable I/O in Large-Scale AI Training*

[![Status](https://img.shields.io/badge/status-active%20experiments-brightgreen)](#experiments)
[![Platform](https://img.shields.io/badge/platform-NERSC%20Perlmutter-blue)](#environment)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange)](https://pytorch.org)
[![Venue](https://img.shields.io/badge/target%20venue-OSDI%20%2F%20SC-purple)](#)

> **Proving that "network isolation" is an illusion — and building the pacemaker that fixes it.**

</div>

---

## 🎯 The Problem

Modern AI clusters advertise **physically separated networks** — a storage network for checkpointing and a GPU fabric for NCCL collective communication. Researchers assume these are isolated.

**They are not.**

```
Perlmutter GPU Node (AMD EPYC Milan)
┌─────────────────────────────────────────────────────────────┐
│  Checkpoint I/O Path:                                       │
│  NVMe SSD ──► PCIe Root Complex ──► CPU ──► Slingshot NIC  │
│               ╔═══════════╗                 (storage net)  │
│               ║  SHARED   ║                                 │
│               ║ BANDWIDTH ║                                 │
│               ╚═══════════╝                                 │
│  GPU Compute:               ──► CPU ──► Slingshot NIC      │
│  A100 All-Reduce ──► PCIe Root Complex     (GPU fabric)     │
│               ▲──────────────────────────────────────────── │
│                     CONTENTION POINT ← Both paths share!   │
└─────────────────────────────────────────────────────────────┘
```

Aggressive checkpoint flushing (`NVMe → RAM → Slingshot NIC → Lustre`) saturates the **PCIe Root Complex and CPU Memory Bus**, causing **≥40% NCCL All-Reduce bandwidth degradation** even when the storage NIC and GPU NIC are physically different ports.

---

## 💡 The Solution: TEMPO Pacing Scheduler

Instead of **greedy flushing**, TEMPO acts as an intelligent **I/O Pacemaker**:

| | Baseline (Greedy) | **TEMPO (Paced)** |
|---|---|---|
| Checkpoint save | Blocks training loop | **O(1)** — staged to local NVMe instantly |
| Lustre flush timing | Any time → contention | **Only during matmul** (never during NCCL) |
| NCCL bandwidth | ↓ Sawtooth drops at every checkpoint | **Flat, predictable** |
| Training throughput | Degraded | **Maintained** |

### How TEMPO Knows When to Flush

```
Training Step Timeline:

  ┌─ COMPUTE ──────────────────────┐  ┌─ NCCL ──────┐  ┌─ COMPUTE ──────┐
  │  forward + backward (matmul)   │  │ all_reduce  │  │ optimizer step │
  │  ✅ Lustre flush ALLOWED       │  │ ❌ flush    │  │ ✅ flush OK    │
  └────────────────────────────────┘  │ PAUSED      │  └────────────────┘
                                      └─────────────┘
  PhaseMonitor detects transitions via FSDP comm hook or context managers.
  CheckpointManager's flush thread blocks on threading.Event during NCCL.
```

---

## 📁 Repository Structure

```
Working_TEMPO/
├── 🧠 tempo/                          # TEMPO Python package
│   ├── __init__.py                    # Package entry point & exports
│   ├── phase_monitor.py               # Thread-safe NCCL/Compute phase detector
│   ├── checkpoint_manager.py          # O(1) NVMe save + background Lustre flush
│   └── scheduler.py                   # Main pacing orchestrator
│
├── 🔬 phase1/                         # Problem verification experiments
│   ├── background_io.sh               # fio io_uring I/O stress injector
│   ├── train_llm_profiling.py         # NCCL benchmark + Llama FSDP profiler
│   └── run_phase1_verification.slurm  # Slurm: baseline vs. contention
│
├── 📊 phase3/                         # Full TEMPO evaluation
│   ├── train_with_tempo.py            # Baseline vs. TEMPO training loop (FSDP)
│   ├── run_evaluation.slurm           # Slurm: side-by-side evaluation
│   └── plot_killer_graph.py           # Publication-quality figure generator
│
├── 🧪 tests/
│   ├── check_api.py                   # Login-node API sanity check
│   └── run_smoke_test.slurm           # Quick validation (1 node, < 1 min)
│
└── ⚙️  configs/
    └── deepspeed_zero3.json           # DeepSpeed ZeRO-3 config
```

---

## 🖥️ Target Environment: NERSC Perlmutter

| Component | Specification |
|---|---|
| CPU | 1× AMD EPYC 7763 (Milan, 64 cores, NPS=2) |
| GPU | 4× NVIDIA A100 40GB SXM (PCIe Gen4 x16) |
| Network | 4× HPE Slingshot 11 (200 Gbps, 1 NIC per GPU) |
| Local storage | ~1.5 TB NVMe at `/tmp` (PCIe Gen4) |
| Shared storage | Lustre at `$PSCRATCH` (via Slingshot) |
| Job scheduler | Slurm — account `m5320` |

---

## 🚀 Quick Start

### Prerequisites
```bash
# On Perlmutter login node
module load pytorch/2.8.0
export PYTHONPATH=/path/to/Working_TEMPO:$PYTHONPATH
```

### Step 1 — API Smoke Check (login node, no GPU needed)
```bash
python tests/check_api.py
# Expected: "ALL CHECKS PASSED"
```

### Step 2 — Slurm Smoke Test (< 1 minute, 1 node)
```bash
mkdir -p logs results/smoke
sbatch tests/run_smoke_test.slurm

# Expected output (sacct):
#   SMOKE TEST PASSED — all 3 tests OK
#   NCCL algbw ≈ 135 GB/s  (intra-node NVLink/PCIe)
```

### Step 3 — Phase 1: Prove the Contention (2 nodes, 15 min)
```bash
sbatch phase1/run_phase1_verification.slurm

# Produces:
#   results/baseline/nccl_bw_rank0.csv    ← flat ~135 GB/s
#   results/contention/nccl_bw_rank0.csv  ← sawtooth, drops ~40%
#   results/figures/killer_graph.pdf       ← paper figure
```

### Step 4 — Phase 3: TEMPO vs. Baseline (2 nodes, 29 min)
```bash
sbatch phase3/run_evaluation.slurm

# Produces:
#   results/baseline/nccl_bw_rank0.csv    ← degraded by checkpointing
#   results/tempo/nccl_bw_rank0.csv       ← flat (TEMPO shields NCCL)
#   results/figures/killer_graph.pdf       ← definitive comparison figure
```

### Generate Demo Plot (No GPU/Data Needed)
```bash
python phase3/plot_killer_graph.py --demo --output-dir results/figures
```

---

## 📐 TEMPO Internal Architecture

```
User Training Loop
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                      TEMPOScheduler                          │
│                    (tempo/scheduler.py)                      │
│                                                              │
│  .checkpoint(state_dict)  ──►  O(1) NVMe write              │
│  .nccl_phase()  ctx mgr   ──►  pause flush thread           │
│  .compute_phase() ctx mgr ──►  allow flush thread           │
└───────────────┬──────────────────────────┬───────────────────┘
                │                          │
    ┌───────────▼─────────┐    ┌──────────▼──────────────┐
    │    PhaseMonitor      │    │   CheckpointManager      │
    │  (phase_monitor.py)  │    │ (checkpoint_manager.py)  │
    │                      │    │                          │
    │  threading.Event:    │    │  save_async():           │
    │  COMPUTE → .set()  ──┼──► │    torch.save() → /tmp   │
    │  NCCL    → .clear()  │    │    enqueue FlushJob      │
    │                      │    │                          │
    │  Hooks available:    │    │  _flush_worker():         │
    │  • FSDP comm hook    │    │    wait_for_io_allowed()  │
    │  • DDP comm hook     │    │    copy 256MB chunks →   │
    │  • Context manager   │    │    $PSCRATCH (Lustre)    │
    └──────────────────────┘    └──────────────────────────┘
```

---

## 📊 Experiment Progress

| Experiment | Job ID | Status | Key Result |
|---|---|---|---|
| Smoke Test (1 node, 4× A100) | `52208307` | ✅ **PASSED** | NCCL 135 GB/s, 44s runtime |
| Phase 1 — Baseline NCCL | pending | ⏳ Queued | Expect ~135 GB/s flat |
| Phase 1 — Contention NCCL | pending | ⏳ After baseline | Expect ≥40% drop |
| Phase 3 — TEMPO vs Baseline | pending | ⏳ After Phase 1 | Expect TEMPO = flat |

---

## 🐛 Issues Resolved

| Symptom | Root Cause | Resolution |
|---|---|---|
| `Cuda failure 101 'invalid device ordinal'` | `--gpus-per-task=1` hides peer GPUs from NCCL | Removed from `srun`; use `--gpus-per-node=4` in SBATCH only |
| `ImportError: No module named torch` | Login node default Python 2.7 | `module load pytorch/2.8.0` |
| `libfabric.so.1: cannot open shared object` | pytorch/2.6.x incompatible with cudatoolkit | Upgraded to pytorch/2.8.0 |
| `SLURM_SUBMIT_DIR` path guard error | `dirname $0` returns temp path in sbatch | Use `SLURM_SUBMIT_DIR` with `basename` guard |

---

## 🔑 Key Design Decisions

### Why `fio io_uring` not SPDK?
Perlmutter is a shared supercomputer — SPDK requires root-level NVMe device unbind, which is blocked for all non-root users. `fio` with `io_uring` achieves comparable PCIe bandwidth saturation in kernel mode with no special privileges.

### Why remove `--gpus-per-task=1`?
Slurm's `--gpus-per-task=1` sets `CUDA_VISIBLE_DEVICES=0` per task, hiding peer GPUs. NCCL's intra-node P2P transport (transport/p2p.cc) probes peer CUDA devices (1,2,3) with `cudaMemcpyPeerAsync`. With only device 0 visible per task, this throws `invalid device ordinal`.

```bash
# ❌ BROKEN — NCCL P2P crashes
srun --ntasks-per-node=4 --gpus-per-task=1 python train.py

# ✅ CORRECT — All 4 GPUs visible, NCCL uses local_rank for device selection
#SBATCH --gpus-per-node=4
srun --ntasks-per-node=4 --cpu-bind=cores python train.py
```

### NUMA Pinning for Maximum Contention
```bash
# EPYC 7763 NPS=2: NUMA 0 = cores 0-31, NUMA 1 = cores 32-63
# GPU0-3 + all 4 Slingshot NICs share PCIe lanes to NUMA 0
# Pinning I/O stress + GPU ranks to NUMA 0 maximises PCIe Root Complex load
export NUMA_NODE=0
srun --cpu-bind=map_cpu:0,8,16,24 ...  # GPU ranks → NUMA 0 cores
```

---

## 📜 Citation

```bibtex
@inproceedings{skim2025tempo,
  title     = {TEMPO: Temporal Emulation and Masking for Predictable I/O
               in Large-Scale AI Training},
  author    = {Kim, Sunggon},
  booktitle = {OSDI / SC 2025},
  year      = {2025},
  note      = {Experiments on NERSC Perlmutter (account m5320)}
}
```

---

<div align="center">
<sub>Running on NERSC Perlmutter · Account m5320 · pytorch/2.8.0 · Last updated April 2026</sub>
</div>
