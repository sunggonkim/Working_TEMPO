<div align="center">

# TEMPO
### *Harmonious Burst Buffer for Jitter-Free LLM Systems*

> **"The fastest I/O is the I/O that does not interfere."**

[![Target](https://img.shields.io/badge/target-SC'26%20%2F%20OSDI'27-blueviolet)](#)
[![Platform](https://img.shields.io/badge/platform-NERSC%20Perlmutter-0075A2)](https://docs.nersc.gov/systems/perlmutter/)
[![Integration](https://img.shields.io/badge/integrates%20with-vLLM%20%7C%20SGLang%20%7C%20LMCache-orange)](#)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

</div>

---

## The Hardware Isolation Fallacy

Modern GPU servers physically separate compute (HBM), network (Slingshot NIC), and storage (NVMe). Naïve intuition: *background I/O in parallel with GPU compute is free — the paths are isolated.*

**This is wrong. The PCIe Root Complex is a shared interconnect.**

```
GPU Server Node (NERSC Perlmutter — AMD EPYC 7763, 4× A100 40GB)
┌──────────────────────────────────────────────────────────────────────┐
│  ┌──────────┐  HBM reads   ┌─────────────────┐   RDMA (NCCL/KV)     │
│  │  A100 #0 │ ────────── ▶ │                 │ ──────────────────▶  │
│  │  A100 #1 │ ────────── ▶ │  PCIe Root      │       Slingshot NIC  │
│  │  A100 #2 │ ────────── ▶ │  Complex        │ ◀──────────────────  │
│  │  A100 #3 │ ────────── ▶ │  (SHARED BUS)   │  KV eviction / ckpt  │
│  └──────────┘              └────────┬────────┘  ← COMPETES with GPU │
│  ┌──────────┐                       │                                │
│  │  NVMe    │ ── DMA (io_uring) ────┘                                │
│  │  /tmp    │   shares same PCIe lanes as GPU↔NIC path               │
│  └──────────┘                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### Observed Effects

| Scenario | Workload collision | Symptom |
|---|---|---|
| **Inference** | KV eviction (HBM→NVMe) during attention kernel (HBM reads) | P99 decode latency spikes 2–5× |
| **Training** | Checkpoint flush (NVMe→Lustre via Slingshot) during NCCL AllReduce (Slingshot) | NCCL BW drops ≥40% at every checkpoint |

Neither the inference nor the training community has a principled, phase-aware fix.

---

## TEMPO's Solution: Harmonious Pacing

TEMPO does **not** make I/O faster. It makes I/O *harmonious* with foreground workloads.

### Principle: Phase-Gated Absorption + Deferred Flush

```
INFERENCE — one LLM forward pass:
  ┌── FFN (GEMM) ──┐  ┌── Attention ─────────────────────┐  ┌── FFN ──┐
  │ compute-bound  │  │ HBM + PCIe bandwidth-bound        │  │ ✅ OK  │
  │ ✅ FLUSH KV    │  │ ❌ TEMPO PAUSES ALL KV FLUSH       │  │         │
  └────────────────┘  └───────────────────────────────────┘  └─────────┘
  Detection: CUPTI kernel name classifier (AttentionPhaseMonitor)

TRAINING — one gradient step:
  ┌── Forward + Backward ──┐  ┌── NCCL AllReduce ──┐  ┌── Optimizer ──┐
  │ compute-bound (matmul) │  │ NIC-bandwidth-bound │  │               │
  │ ✅ FLUSH CHECKPOINT    │  │ ❌ TEMPO PAUSES      │  │ ✅ FLUSH OK   │
  └────────────────────────┘  └────────────────────┘  └───────────────┘
  Detection: FSDP/DDP comm hook → PhaseMonitor.nccl_phase()
```

### Two Modes, One Principle

| Mode | Spike source | Foreground critical path | TEMPO action |
|---|---|---|---|
| **Inference** | KV cache eviction | Attention (PCIe+HBM-bound) | Absorb KV O(1) → flush during FFN |
| **Training** | Checkpoint flush | NCCL AllReduce (NIC-bound) | Stage ckpt O(1) → flush during forward |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   TEMPO C++ Library  (libtempo.so)                      │
│                                                                          │
│  AttentionPhaseMonitor          SpikeAbsorber         PacingDaemon      │
│  ───────────────────────        ─────────────────     ──────────────     │
│  CUPTI callback API             MPSC lock-free        io_uring           │
│  kernel name patterns:          ring buffer           IOSQE_ASYNC        │
│    flash_attn  → ATTN           O(1) wait-free        TokenBucket        │
│    cutlass_gemm → FFN           absorb()              Phase-gated        │
│  atomic<Phase>  ──────────────▶ eventfd notify  ────▶ Lustre flush       │
│  wait_for_ffn()                                                           │
│                                                                          │
│  C API:  tempo_create_engine()  tempo_absorb()  tempo_destroy_engine()  │
└──────────────────────────────────────────────────────────────────────────┘
        ▲ ctypes                               ▲ ctypes / pybind11
        │                                      │
┌───────┴─────────────────┐    ┌───────────────┴───────────────────────┐
│  tempo/vllm_hook.py      │    │  tempo/lmcache_connector.py            │
│  Patches vLLM v1         │    │  TEMPOStorageBackend                   │
│  LMCacheConnector        │    │  drop-in for any LMCache backend       │
│  .store_kv_cache()       │    │  (CPU / Disk / NIXL / Mooncake)        │
│  SGLang KVCache.evict()  │    │                                        │
└──────────────────────────┘    └───────────────────────────────────────┘
```

---

## Repository Structure

```
Working_TEMPO/
├── src/                              C++ core library
│   ├── attention_monitor/
│   │   ├── monitor.hpp               AttentionPhaseMonitor (CUPTI-based)
│   │   └── monitor.cu                Kernel classifier + callback impl
│   ├── spike_absorber/
│   │   ├── absorber.hpp              MPSC ring buffer interface
│   │   └── absorber.cpp              Vyukov MPSC — O(1) wait-free
│   ├── pacing_daemon/
│   │   ├── token_bucket.hpp          Non-blocking token bucket (header-only)
│   │   ├── pacing_daemon.hpp         PacingDaemon interface
│   │   └── pacing_daemon.cpp         io_uring harmonious flush loop
│   └── c_api/
│       └── tempo_c_api.cpp           Clean C API (ctypes-friendly)
│
├── tempo/                            Python integration layer
│   ├── __init__.py
│   ├── lmcache_connector.py          TEMPOStorageBackend (LMCache drop-in)
│   ├── vllm_hook.py                  vLLM v1 + SGLang monkey-patch
│   ├── phase_monitor.py              Training-mode phase monitor
│   ├── checkpoint_manager.py         Training checkpoint O(1) stage + flush
│   └── scheduler.py                  TEMPOScheduler (training orchestrator)
│
├── phase1/                           Training: quantify PCIe contention
│   ├── background_io.sh              fio io_uring stress injector
│   ├── train_llm_profiling.py        NCCL benchmark + Llama-3 FSDP profiler
│   └── run_phase1_verification.slurm
│
├── phase3/                           Training: TEMPO vs. Baseline
│   ├── train_with_tempo.py           Side-by-side evaluation
│   ├── run_evaluation.slurm
│   └── plot_killer_graph.py          IEEE-format figure generator
│
├── tests/
│   ├── check_api.py
│   └── run_smoke_test.slurm
│
├── configs/
│   └── deepspeed_zero3.json
│
└── CMakeLists.txt                    Build: libtempo.so + tempo_cpp.so
```

---

## Build

```bash
# On Perlmutter
module load pytorch/2.8.0

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;90"
make -j$(nproc)

# Produces:
#   build/libtempo.so      ← loaded by tempo/lmcache_connector.py
#   build/tempo_cpp*.so    ← pybind11 Python module
```

Dependencies: CUDA ≥ 12.0 (CUPTI included), `liburing-dev` ≥ 2.3, CMake ≥ 3.21, pybind11 (auto-fetched).

---

## Usage

### Inference: vLLM + LMCache (zero code change)

```bash
export TEMPO_RATE_GBPS=5
export TEMPO_LUSTRE_DIR=$PSCRATCH/kvcache
export LIBTEMPO_PATH=$(pwd)/build/libtempo.so

python - <<'EOF'
import tempo.vllm_hook
tempo.vllm_hook.install()   # one line — patches LMCacheConnector globally

from vllm import LLM
llm = LLM("meta-llama/Meta-Llama-3-8B", gpu_memory_utilization=0.85)
# KV evictions now routed through TEMPO — attention windows never see PCIe I/O
EOF
```

### Inference: LMCache API

```python
from lmcache.storage_backend.cpu_backend import CpuMemoryBackend
from tempo.lmcache_connector import TEMPOStorageBackend, TEMPOConfig

backend = TEMPOStorageBackend(
    backing = CpuMemoryBackend(lmcache_cfg),
    cfg     = TEMPOConfig(rate_gbps=5.0, strict_gate=True),
)
engine = LMCacheEngine(lmcache_cfg, backend)
```

### Training: FSDP + TEMPO

```python
from tempo import TEMPOScheduler

sched = TEMPOScheduler(rank=rank, world_size=world_size,
                        local_nvme_dir="/tmp/ckpt", lustre_dir=...,
                        mode="tempo")

for step in range(steps):
    with sched.compute_phase():
        loss = model(inputs); loss.backward()   # TEMPO flushes here
    optimizer.step()
    if step % 50 == 0:
        sched.checkpoint(model.state_dict(), step)  # O(1) — no training stall
```

---

## Expected Results: The Killer Graph

**Training** (2× 4-GPU Perlmutter, Llama-3-1B, FSDP, ckpt every 50 steps):

```
NCCL All-Reduce bandwidth (GB/s)
 135 ┤──────────────────────────────────────────  Baseline (no I/O)
 120 ┤·············TEMPO (paced)···················  ← shielded from flush
     │
  90 ┤
  80 ┤──╮  ────────╮  ────────╮  ────────╮  ──  Greedy flush baseline
     │  ↓ -40%     ↓           ↓           ↓
  70 ┤  └──        └──         └──         └──
     └────────────────────────────────────────────▶ training step
       0      50      100      150      200
```

**Inference** (A100 × 4, vLLM, Llama-3-8B, P99 decode latency ms):

```
P99 decode latency (ms)
 200 ┤             ●●●               ●●●
 160 ┤       ●●●●●    ●●●      ●●●●●    ●●
 120 ┤─────────────────────────────────────────────── vLLM+LMCache greedy
 100 ┤·····················································  TEMPO (paced)
     └────────────────────────────────────────────────────▶ request #
       0       100     200     300     400
     ↑ KV eviction events every ~50 requests
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TEMPO_RATE_GBPS` | `5.0` | Flush ceiling GB/s (keep ≤ 20% of NIC BW) |
| `TEMPO_BURST_MB` | `256` | Token bucket burst (MB) |
| `TEMPO_FFN_WAIT_US` | `200` | Max µs to wait for FFN window |
| `TEMPO_STRICT_GATE` | `1` | 1 = never flush during ATTENTION |
| `TEMPO_STAGE_DIR` | `/tmp/tempo_<pid>` | NVMe staging path |
| `TEMPO_LUSTRE_DIR` | `$PSCRATCH/tempo_kvcache` | Lustre target |
| `LIBTEMPO_PATH` | auto-detect | Path to `libtempo.so` |
| `TEMPO_VERBOSE` | `0` | Enable daemon debug logs |

---

## NERSC Perlmutter Experiments

```bash
# Smoke test (< 1 min)
sbatch tests/run_smoke_test.slurm

# Phase 1: Quantify PCIe contention (15 min, debug queue)
sbatch phase1/run_phase1_verification.slurm

# Phase 3: TEMPO vs. Baseline (29 min, chained)
PHASE1=$(sbatch --parsable phase1/run_phase1_verification.slurm)
sbatch --dependency=afterok:${PHASE1} phase3/run_evaluation.slurm
```

> **Critical SLURM rule**: Never use `--gpus-per-task=1` in `srun` on Perlmutter.
> Sets `CUDA_VISIBLE_DEVICES=0` per task → NCCL P2P probes peers (1,2,3) → `Cuda failure 101`.
> Fix: `--gpus-per-node=4` in `#SBATCH` + `export CUDA_VISIBLE_DEVICES=0,1,2,3`.

---

## Citation

```bibtex
@inproceedings{kim2026tempo,
  title     = {{TEMPO}: Harmonious Burst Buffering for Jitter-Free
               {LLM} Inference and Training at Scale},
  author    = {Kim, Sunggon},
  booktitle = {SC '26: The International Conference for High Performance
               Computing, Networking, Storage and Analysis},
  year      = {2026},
  note      = {Experiments on NERSC Perlmutter (account m5320)},
}
```

---

<div align="center">
<sub>NERSC Perlmutter · pytorch/2.8.0 · vLLM v1 · SGLang · LMCache v0.4+</sub>
</div>
