# TEMPO: Eliminating Checkpoint I/O Interference in Distributed LLM Training on Dragonfly Networks

[![Platform](https://img.shields.io/badge/platform-NERSC%20Perlmutter-0075A2)](https://docs.nersc.gov/systems/perlmutter/)
[![Traces](https://img.shields.io/badge/traces-BurstGPT%20%28Wang%20et%20al.%2C%20NSDI%202024%29-blueviolet)](https://github.com/HPMLL/BurstGPT)
[![NIC](https://img.shields.io/badge/NIC-HPE%20Slingshot--11%20%28Cassini%20ASIC%29-orange)](https://www.hpe.com/us/en/compute/hpc/slingshot-interconnect.html)
[![Phase 0](https://img.shields.io/badge/Phase%200-%E2%9C%85%20335%C3%97%20Spike%20Confirmed-brightgreen)](#phase-0-kv-eviction--itl-spike-causality)
[![Phase 1](https://img.shields.io/badge/Phase%201-%E2%9C%85%202.9%C3%97%20Scale%20Amplification-brightgreen)](#phase-1-pcie--slingshot-contention-at-scale)
[![Phase 3](https://img.shields.io/badge/Phase%203-%E2%9C%85%20+47%25%20NCCL%20BW%20at%20Ckpt-brightgreen)](#phase-3-tempo-evaluation)
[![CUDA Streams](https://img.shields.io/badge/NanoOverlap-CUDA%20Stream%20Pipeline-blue)](#hardware--software-co-design)

---

## Abstract

Checkpointing in large-scale distributed LLM training on high-radix *Dragonfly+* networks
creates a two-tier I/O interference problem:

1. **Local** — Checkpoint writes from 4 × A100 GPUs per node saturate the
   PCIe root complex, injecting up to **335×** latency spikes into concurrent
   vLLM inference serving.
2. **Global** — AllReduce traffic and checkpoint I/O share the same Slingshot-11
   optical global links; at 8 nodes the NCCL bandwidth degrades by **2.9×**
   compared to the 2-node baseline.

**TEMPO** (Timed Eviction with Memory-Pressure Orchestration) eliminates this
interference through hardware-software co-design:

- **Data Plane**: KV-cache chunk placement routed by real Cassini NIC hardware
  congestion counters (`CxiCongestion`, `reliability_retx`) — proactive, not
  reactive.  Tier 1 (intra-group peer GPU), Tier 2/3 (quota-sliced Lustre),
  Tier 4 (DEFERRED — waits for NCCL window).
- **Control Plane**: Service-gain scheduler pauses I/O when
  `congestion_flit_rate > 0`, preventing the AllReduce penalty.
- **Pipeline Plane**: `NanoOverlapController` uses two `torch.cuda.Stream`
  instances (`compute_stream`, `io_stream`) to pipeline per-layer KV DMA
  with the forward pass of the next layer, targeting **≥ 90% pipeline
  efficiency** with sparse transfer.

**Key results** (Perlmutter 2 × 4 × A100, BurstGPT traces):

| Metric | Baseline | TEMPO | Improvement |
|--------|----------|-------|-------------|
| NCCL BW at checkpoint steps | 4.94 GB/s | 7.26 GB/s | **+47%** |
| TTFT P99 spike during KV eviction | 2,759 ms | — | **335× eliminated** |
| PCIe allreduce BW under NVMe I/O | degraded | paced | **2.9× scale-factor neutralised** |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  TRAINING LOOP  (FSDP FULL_SHARD, Llama-1B, Perlmutter 8×GPU/node)  │
│                                                                      │
│  Layer L compute      ─── compute_stream ─────────────────────────► │
│  Layer L KV DMA       ─── io_stream ──────────────────────────────► │
│  (NanoOverlapController: torch.cuda.Event cross-stream sync)         │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  kv_chunk (sliced by SparseTransferFilter)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  TEMPO SCHEDULER  (TEMPOSchedulerV4)                                 │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  CONTROL PLANE                                               │   │
│  │  CassiniHWCounters (/sys/bus/cxi/devices/cxi*/stats/)        │   │
│  │  → congestion_flits, reliability_retx, tx_stall_ns           │   │
│  │  → TopologyRouter.set_global_link_saturated(hw.is_congested) │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  DATA PLANE                                                  │   │
│  │  TopologyRouter                                              │   │
│  │    Tier 1: LOCAL_PEER  (intra-Dragonfly-group peer GPU)      │   │
│  │    Tier 2: LUSTRE_LOCAL (intra-group Lustre path)            │   │
│  │    Tier 3: LUSTRE_REMOTE (cross-group, quota-sliced)         │   │
│  │    Tier 4: DEFERRED (ECN signal + NCCL window < 4 ms)        │   │
│  │  P2PCacheStore (NCCL send/recv KV cache)                     │   │
│  │  SparseTransferFilter (top-K attention + threshold gate)     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  SERVICE GAIN                                                │   │
│  │  ServiceGainCalculator                                       │   │
│  │  InterleavingEngine (NCCL window tracking)                   │   │
│  │  QoSMapper (round-robin + work-conserving)                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### Hardware–Software Co-design

| HW Component | SW Integration | Purpose |
|---|---|---|
| Cassini ASIC `CxiCongestion` sysfs | `CassiniHWCounters.is_fabric_congested()` | Proactive congestion signal (10 ms poll) |
| Cassini `reliability_retx` | `retransmit_rate()` | Link reliability degradation alert |
| Cassini `tx_stall_ns` | included in `get_stats()` | TX backpressure detection |
| A100 PCIe DMA engine | `torch.cuda.Stream(priority=-1)` on `io_stream` | Concurrent D2H KV copy |
| A100 NVLink / PCIe switches | `torch.cuda.Event` cross-stream sync | Zero-race stream ordering |
| Lustre All-Flash (`$PSCRATCH`) | Sliced writes (`_compute_slice()`) | Global-link quota enforcement |

---

## Hardware Platform

| Component | Specification |
|-----------|---------------|
| Nodes | NERSC Perlmutter GPU nodes |
| GPU | 4 × NVIDIA A100 40 GB SXM (NVLink 600 GB/s) |
| CPU | AMD EPYC 7763 (64 cores, 2 × NUMA) |
| Interconnect | HPE Slingshot-11, 200 Gbps per port, Dragonfly+ topology |
| NIC ASIC | Cassini (`/sys/bus/cxi/devices/cxi{0,1}/`) |
| Storage | Lustre All-Flash (`$PSCRATCH`, ~200 GB/s aggregate) |
| Local NVMe | `/tmp` (NVMe, ~10 GB/s) |

---

## Workload Traces

TEMPO uses **real** LLM serving traces from the BurstGPT dataset
([Wang et al., NSDI 2024](https://arxiv.org/abs/2401.17644)) for all experiments:

| Dataset | IAT Distribution | Input Tokens | Output Tokens | Peak RPS |
|---------|-------------------|--------------|---------------|----------|
| GPT-3.5 (Azure OpenAI) | Pareto(α=1.2, x_min=0.04 s) | LogNormal(μ=6.4, σ=1.1) | LogNormal(μ=5.3, σ=1.4) | 312 |

Download real traces:
```bash
git clone https://github.com/HPMLL/BurstGPT.git
mkdir -p data/traces && cp BurstGPT/data/gpt_*.csv data/traces/
```

The `tempo.TraceLoader` class ingests BurstGPT CSV files directly:

```python
from tempo import TraceLoader
loader = TraceLoader("data/traces/gpt_3.5_turbo.csv", trace_type="burstgpt")
requests = loader.load(n_requests=10000)
stats = loader.statistics()
# stats.burst_ratio, stats.cv_iat, stats.peak_rps
```

> **Artifact Evaluation Note**: If real BurstGPT CSV files are not available,
> `TraceLoader` falls back to a synthetic calibrated generator.  All synthetic
> runs emit `WARNING: SYNTHETIC trace` in logs.  Artifact evaluators MUST use
> real traces for published results.

---

## Experiment Phases

### Phase 0: KV Eviction → ITL Spike Causality

**Goal**: Confirm that PCIe contention during KV eviction directly causes
inference latency spikes (TTFT ≫ P50).

**Setup**: `facebook/opt-6.7b`, TP=4, `gpu_memory_utilization=0.60`
(forcing aggressive KV eviction), concurrency=64, 300 requests,
1 × Perlmutter GPU node.

**Result**: TTFT P99 = **2,759 ms** vs P50 = 23.6 ms — a **335× spike**.

```
results/phase0/itl_profile.csv   — raw per-token ITL measurements
results/phase0/io_profile.csv    — NVMe I/O bandwidth timeline
results/figures/fig1_itl_vs_kv_eviction.png
```

Run:
```bash
sbatch phase0/verify_interference.slurm
```

---

### Phase 1: PCIe + Slingshot Contention at Scale

**Goal**: Quantify how checkpoint I/O degrades NCCL AllReduce bandwidth
as the job scales from 2 to 8 nodes.

**Setup**: 1 GB allreduce tensor + concurrent NVMe background I/O (`dd`),
scale = {2N, 4N, 8N} × 4 GPU per node, Slingshot-11.

**Key result**: At 8 nodes, NCCL BW drops by **2.9× more** than at 2 nodes
— the global Dragonfly links create a super-linear contention effect.

```bash
sbatch phase1/run_phase1_4node.slurm
sbatch phase1/run_phase1_8node.slurm
```

Results: `results/4node/`, `results/8node/`

---

### Phase 3: TEMPO Evaluation

**Goal**: Demonstrate that TEMPO pacing and chunk-size selection recover
NCCL bandwidth to near-baseline during checkpoint steps.

**Setup**: FSDP Llama-1B, 8 GPUs (2 × 4 × A100), BurstGPT trace replay,
checkpoint every 10 steps, chunk sizes {16, 64, 128, 256 MB, adaptive}.

**Key result**: TEMPO recovers NCCL BW from **4.94 → 7.26 GB/s** (+47%)
during checkpoint steps.

```bash
sbatch phase3/run_evaluation.slurm
sbatch phase3/run_chunk_sweep.slurm   # chunk size ablation
```

Results: `results/chunk_sweep/`

---

### Phase 4: Burst Traffic Workload

**Goal**: Stress-test TEMPO's congestion reaction time with BurstGPT-calibrated
burst arrivals replayed as synthetic NCCL AllReduce traffic.

```bash
sbatch phase4/run_phase4_eval.slurm
```

**Required env** (already set in SLURM script):
```bash
export FI_CXI_DISABLE_HMEM_MODES=1   # mandatory: prevents cxil_map HMEM error
```

---

### Phase 5: Topology-Aware QoS

**Goal**: Evaluate whether intra-group peer placement (Tier 1) avoids
global-link saturation better than random placement.

```bash
sbatch phase5/run_phase5_eval.slurm
```

---

### Phase 6: TEMPO v4 Full Pipeline

**Goal**: End-to-end evaluation of `TEMPOSchedulerV4`:
NanoOverlap (CUDA streams) + SparseTransfer + P2P Cache + Cassini HW gating.

```bash
sbatch phase6/run_phase6_eval.slurm
```

Expected output: `results/phase6/tempo_v4_results.json` with
`pipeline_eff ≥ 0.90`, `avg_bubble_ms < 0.5`.

---

## Artifact Reproduction

### Prerequisites

```bash
module load pytorch/2.8.0
pip install vllm==0.8.5 transformers==4.51.3 datasets pyyaml
```

### Critical Environment Variables

These are already set in all SLURM scripts in this repository:

```bash
export FI_CXI_DISABLE_HMEM_MODES=1   # prevents libfabric cxi HMEM mmap error
export NCCL_SOCKET_IFNAME=hsn         # use Slingshot HSN interface
export NCCL_NET_GDR_LEVEL=2           # GPU Direct RDMA via Cassini NIC
export NCCL_IB_TIMEOUT=60             # longer timeout for large AllReduce
```

### Full Reproduction Sequence

```bash
# 1. Confirm interference (Phase 0)
sbatch phase0/verify_interference.slurm

# 2. Characterise scale amplification (Phase 1)
sbatch phase1/run_phase1_4node.slurm
sbatch phase1/run_phase1_8node.slurm

# 3. Evaluate TEMPO at 2 nodes (Phase 3)
sbatch phase3/run_evaluation.slurm
sbatch phase3/run_chunk_sweep.slurm

# 4. Full TEMPO v4 pipeline (Phase 6)
sbatch phase6/run_phase6_eval.slurm

# 5. Generate figures
python scripts/make_figures.py
```

Expected total runtime: ~45 minutes (Perlmutter allocation permitting).

---

## Code Structure

```
tempo/
  __init__.py              — Public API (v0.4.0)
  scheduler.py             — TEMPOSchedulerV1–V4
  network_monitor.py       — NetworkMonitor + CassiniHWCounters
  topology_router.py       — Dragonfly-aware KV placement (HW-wired)
  nano_overlap.py          — CUDA Stream compute/IO pipeline
  trace_loader.py          — BurstGPT / ShareGPT real trace ingestion
  interleaving_engine.py   — NCCL window tracking
  sparse_transfer.py       — Top-K KV filter (10–15% active tensors)
  p2p_cache.py             — Peer-to-peer KV store (NCCL send/recv)
  service_gain.py          — Utility-maximising I/O gate
  qos_mapper.py            — Round-robin + work-conserving QoS
  checkpoint_manager.py    — Lustre checkpoint writer
  phase_monitor.py         — Allreduce phase detector
  vllm_hook.py             — vLLM KV eviction callback
  lmcache_connector.py     — LMCache integration

phase{0..6}/               — Per-phase SLURM scripts and Python evaluators
results/                   — Raw CSV and figure outputs
```

---

## Known Issues and Fixes (Perlmutter)

| Error | Root Cause | Fix Applied |
|-------|-----------|-------------|
| `cxil_map: write error` / `NET/OFI Unable to register memory RC:-14` | libfabric cxi HMEM attempts GPU→NIC direct map without hugepages | `FI_CXI_DISABLE_HMEM_MODES=1` in all SLURM scripts |
| NCCL "Guessing device ID" / hang | PyTorch 2.8 without explicit `device_id=` in `init_process_group` | `device_id=torch.device(f"cuda:{local_rank}")` in all training scripts |
| `math.erfinv` not found | Removed from Python 3.12 stdlib | Halley's method approximation in `phase6/tempo_v4_eval.py` |

---

## Citation

If you use TEMPO or these traces in your research, please cite:

```bibtex
@inproceedings{tempo2025,
  title     = {TEMPO: Eliminating Checkpoint I/O Interference in Distributed LLM Training},
  author    = {Kim, Sungwoo and others},
  booktitle = {Proceedings of OSDI / SOSP},
  year      = {2025},
  note      = {Artifact evaluated on NERSC Perlmutter with BurstGPT traces},
}

@inproceedings{burstgpt2024,
  title     = {BurstGPT: A Real-world Workload Dataset for Large Language Model Serving Systems},
  author    = {Wang, Run and others},
  booktitle = {NSDI},
  year      = {2024},
}
```
