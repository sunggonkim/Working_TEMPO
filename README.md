# TEMPO: Timed Eviction with Memory-Pressure Orchestration

> **"먼저 문제가 존재하는지 증명하라. 그 다음 해결책을 설계하라."**

[![Phase](https://img.shields.io/badge/current%20phase-Phase%200%3A%20Verification-red)](#phase-0-the-proof)
[![Platform](https://img.shields.io/badge/platform-NERSC%20Perlmutter-0075A2)](https://docs.nersc.gov/systems/perlmutter/)
[![Target](https://img.shields.io/badge/target-OSDI%2726%20%2F%20ATC%2726-blueviolet)](#)

---

## ⚠️ PHASE 0: 문제가 실제로 존재하는가?

**솔루션보다 가설 검증이 먼저다.**

시스템 논문에서 가장 흔한 치명적 실수: 문제를 정량적으로 증명하지 않은 채
버퍼, 데몬 같은 솔루션부터 설계하는 것.
TEMPO는 아래 가설을 실험으로 먼저 증명한다.

---

## The Core Hypothesis

```
KV Cache Eviction  (HBM → CPU RAM → NVMe via PCIe DMA)
        │
        │  PCIe bus contention
        ▼
GPU Server Node (NERSC Perlmutter — AMD EPYC 7763, 4× A100 40GB)
┌──────────────────────────────────────────────────────────────────────┐
│  ┌──────────┐  HBM reads   ┌─────────────────┐                      │
│  │  A100 #0 │ ────────── ▶ │                 │                      │
│  │  A100 #1 │ ────────── ▶ │  PCIe Root      │  ← SHARED BUS        │
│  │  A100 #2 │ ────────── ▶ │  Complex        │                      │
│  │  A100 #3 │ ────────── ▶ │  (BOTTLENECK)   │                      │
│  └──────────┘              └────────┬────────┘                      │
│  ┌──────────┐                       │                                │
│  │  NVMe    │ ── KV eviction DMA ───┘  ← COMPETES with GPU HBM reads│
│  │  /tmp    │                                                        │
│  └──────────┘                                                        │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
  Decode phase stalls (memory-bandwidth bound)
        │
        ▼
  ITL spike:  50 ms → 500 ms+   (10× amplification?)
```

**직관**: vLLM/LMCache가 HBM 부족으로 KV cache를 NVMe에 내릴 때,
PCIe DMA가 GPU 메모리 접근을 방해하여 token 생성이 멈춘다.

**증명 목표**: NVMe write BW spike와 ITL spike가 동시에 발생함을 단일 그래프로 보인다.

---

## Phase 0: The Proof (지금 당장 실행)

### 실험 조건

| 항목 | 값 |
|---|---|
| 하드웨어 | 4× NVIDIA A100 40GB, AMD EPYC 7763, Local NVMe |
| vLLM `gpu_memory_utilization` | **0.60** (정상: 0.90 → 강제로 eviction 유발) |
| Concurrency | 64 simultaneous requests |
| Sequence length | 4096 tokens (large KV footprint) |
| 측정 해상도 | Per-token timestamp (microsecond) |
| NUMA pinning | Node 0 (CPU + GPU + NVMe 같은 PCIe root complex) |

### Quick Start (Perlmutter)

```bash
cd /pscratch/sd/s/sgkim/Skim-Tempo
sbatch phase0/verify_interference.slurm

# 실시간 로그
tail -f logs/phase0_<JOBID>.out

# 결과 그래프
ls results/figures/fig1_itl_vs_kv_eviction.pdf
```

### GPU 없이 데모 그래프 먼저 확인

```bash
# matplotlib만 있으면 실행 가능
python phase0/plot_inference_interference.py --demo
# → results/figures/fig1_itl_vs_kv_eviction_DEMO.png
```

### 단계별 수동 실행

```bash
# 터미널 1: 하드웨어 모니터 (100ms 간격)
bash phase0/hardware_monitor.sh results/phase0 100

# 터미널 2: vLLM 워크로드 주입 (gpu_util=0.6 → forced eviction)
python phase0/workload_injector.py \
    --model meta-llama/Llama-2-7b-hf \
    --tp 4 --gpu-util 0.60 \
    --concurrency 64 --num-requests 300

# 터미널 3: Killer Graph 생성
python phase0/plot_inference_interference.py \
    --itl results/phase0/itl_profile.csv \
    --io  results/phase0/io_profile.csv
```

---

## The Killer Graph (OSDI Figure 1)

아래 패턴이 나오면 가설 **CONFIRMED**:

```
ITL P99 (ms) ↑          NVMe Write BW (MB/s) ↑
             │  ▓▓                │  ▓▓
  500 ───────┤  ██  ← ITL spike   │  ██  ← KV eviction I/O
             │  ██                │  ██
  100 ───────┤                    │
   50 ───────┼────────────────    ┼────────────────
             └──────────────────  └──────────────────
                  Time (s)              Time (s)

  Overlaid: 두 spike가 동시에 발생 → PCIe contention 증명
```

**성공 기준**: P99 ITL 증폭 ≥ 3× + I/O burst와 시간적 동기화 확인

**실패 시 조치**:
1. `--gpu-util 0.5` 로 더 낮추기
2. `--concurrency 128` 로 더 높이기
3. vLLM 로그에서 `Swapping` 메시지 확인

---

## Phase 0 Files

```
phase0/
├── workload_injector.py
│       vLLM AsyncLLMEngine ITL profiler
│       gpu_util=0.6 → KV eviction 강제 유발
│       ShareGPT workload, per-token microsecond timestamp
│       출력: results/phase0/itl_profile.csv
│             (timestamp_ns, request_id, token_idx, itl_ms)
│
├── hardware_monitor.sh
│       100ms 간격 NVMe/GPU I/O 로거
│       iostat + nvidia-smi
│       출력: results/phase0/io_profile.csv
│             (timestamp_ns, nvme_write_mbps, gpu_hbm_used_mib)
│
├── plot_inference_interference.py
│       Dual-axis killer graph generator
│       ITL P50/P99 (좌축) + NVMe write BW (우축, 음영)
│       Eviction burst 자동 검출 및 spike amplification 계산
│       --demo 모드: GPU 없이 합성 데이터로 예상 결과 미리 확인
│
└── verify_interference.slurm
        Perlmutter SLURM job (-A m5320, debug queue, 30min)
        numactl --cpunodebind=0: PCIe contention 극대화
        4단계 자동 실행: monitor → inject → stop → plot
```

---

## Why This Matters: TEMPO의 당위성

Phase 0에서 가설이 증명되면, 이것이 TEMPO의 존재 이유가 된다:

| 문제 (Phase 0에서 증명) | 원인 | TEMPO 해결책 |
|---|---|---|
| Decode ITL spike | KV eviction DMA가 PCIe 점유 | **Pacing Scheduler**: FFN window에서만 I/O flush |
| P99 tail latency ↑↑ | 동기적 DMA가 GPU stall 유발 | **Spike Absorber**: lock-free ring buffer → 비동기화 |
| 예측 불가능한 지연 | Eviction 시점이 무작위 | **Phase Monitor**: CUPTI로 Attention/Decode 구분 |

**Phase 0 없이 솔루션 설계 = 존재하지 않는 문제의 해결책.**

---

## Research Timeline

```
Phase 0  [NOW]   → 문제 존재 증명 (killer graph) ← 현재 여기
Phase 1  [NEXT]  → NCCL baseline: PCIe contention 정량화
Phase 2  [TODO]  → TEMPO pacing scheduler 구현
Phase 3  [TODO]  → TEMPO vs baseline 평가
```

**Phase 0 결과(killer graph) 없이 Phase 2, 3은 시작하지 않는다.**

---

## Solution Architecture (Phase 0 이후 구현 예정)

> 아래는 Phase 0에서 가설이 증명된 후 정당화되는 솔루션이다.

### TEMPO's Principle: Phase-Gated Flush

```
INFERENCE — one LLM forward pass:
  ┌─── FFN (GEMM) ───┐  ┌── Attention ─────────────────────┐  ┌── FFN ──┐
  │  compute-bound   │  │  HBM + PCIe bandwidth-bound      │  │         │
  │  ✅ FLUSH KV I/O │  │  ❌ TEMPO BLOCKS ALL KV I/O       │  │ ✅ FLUSH│
  └──────────────────┘  └──────────────────────────────────┘  └─────────┘
  Detection: CUPTI kernel name classifier (flash_attn → ATTENTION, cutlass_gemm → FFN)
```

### C++ Core (libtempo.so — LD_PRELOAD)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   TEMPO C++ Library  (libtempo.so)                      │
│                                                                          │
│  AttentionPhaseMonitor     SpikeAbsorber          PacingDaemon          │
│  ─────────────────────     ─────────────          ────────────           │
│  CUPTI callback API        MPSC lock-free         io_uring               │
│  kernel → Phase enum       ring buffer            IOSQE_ASYNC            │
│  atomic<Phase>             O(1) wait-free         TokenBucket            │
│  wait_for_ffn() ─────────▶ absorb() ────────────▶ flush during FFN      │
│                                                                          │
│  C API: tempo_create_engine() / tempo_absorb() / tempo_destroy_engine() │
└─────────────────────────────────────────────────────────────────────────┘
         ▲ ctypes                              ▲ ctypes
         │                                     │
┌────────┴──────────────┐    ┌─────────────────┴───────────────────────┐
│  tempo/vllm_hook.py   │    │  tempo/lmcache_connector.py             │
│  vLLM v1 monkey-patch │    │  TEMPOStorageBackend (drop-in)          │
│  .store_kv_cache()    │    │  CPU / Disk / NIXL / Mooncake backends  │
└───────────────────────┘    └─────────────────────────────────────────┘
```

---

## Repository Structure

```
Working_TEMPO/
├── phase0/                           ← START HERE
│   ├── workload_injector.py          ITL profiler (forces KV eviction)
│   ├── hardware_monitor.sh           100ms NVMe/GPU I/O logger
│   ├── plot_inference_interference.py  Killer graph generator
│   └── verify_interference.slurm    Perlmutter job
│
├── src/                              C++ core (implement AFTER Phase 0)
│   ├── attention_monitor/
│   │   ├── monitor.hpp               Phase enum: ATTENTION, FFN, NCCL, IDLE
│   │   └── monitor.cu                CUPTI callback + kernel classifier
│   ├── spike_absorber/
│   │   ├── absorber.hpp              MPSC ring buffer interface
│   │   └── absorber.cpp              Vyukov MPSC — O(1) wait-free
│   ├── pacing_daemon/
│   │   ├── token_bucket.hpp          Non-blocking token bucket
│   │   ├── pacing_daemon.hpp
│   │   └── pacing_daemon.cpp         io_uring phase-gated flush loop
│   └── c_api/
│       └── tempo_c_api.cpp           C API (ctypes-friendly)
│
├── tempo/                            Python integration layer
│   ├── lmcache_connector.py          TEMPOStorageBackend (LMCache drop-in)
│   ├── vllm_hook.py                  vLLM v1 + SGLang monkey-patch
│   ├── phase_monitor.py
│   ├── checkpoint_manager.py
│   └── scheduler.py
│
├── phase1/                           Training: PCIe contention baseline
├── phase3/                           Training: TEMPO vs baseline evaluation
├── tests/
├── configs/
└── CMakeLists.txt                    Build: libtempo.so + tempo_cpp.so
```

---

## Build (Phase 0 이후)

```bash
module load pytorch/2.8.0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;90"
cmake --build build -j$(nproc)
# → build/libtempo.so   (LD_PRELOAD interceptor)
# → build/tempo_cpp.so  (Python ctypes bindings)
```

---

## Perlmutter Quick Reference

```bash
# Phase 0 제출 (지금 당장)
sbatch phase0/verify_interference.slurm

# 상태 확인
squeue -u $USER

# 실시간 로그
tail -f logs/phase0_<JOBID>.out

# SLURM 계정: -A m5320, -C gpu, pytorch/2.8.0
```

> **SLURM 주의**: `srun --gpus-per-task=1` 절대 사용 금지 →
> `CUDA_VISIBLE_DEVICES=0` 고정 → NCCL P2P peer 탐색 실패 (Cuda error 101).
> 올바른 방법: `#SBATCH --gpus-per-node=4` + `export CUDA_VISIBLE_DEVICES=0,1,2,3`

---

## Citation

```bibtex
@inproceedings{kim2026tempo,
  title     = {{TEMPO}: Phase-Aware KV Cache Eviction Pacing for
               Jitter-Free {LLM} Inference at Scale},
  author    = {Kim, Sunggon},
  booktitle = {OSDI '26},
  year      = {2026},
  note      = {NERSC Perlmutter, account m5320},
}
```

---

<div align="center">
<sub>NERSC Perlmutter · pytorch/2.8.0 · vLLM v1 · LMCache v0.4+ · Phase 0 First</sub>
</div>


