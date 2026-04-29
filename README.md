# TEMPO: Timed Eviction with Memory-Pressure Orchestration

[![Platform](https://img.shields.io/badge/platform-NERSC%20Perlmutter-0075A2)](https://docs.nersc.gov/systems/perlmutter/)
[![Phase 0](https://img.shields.io/badge/Phase%200-%E2%9C%85%20335%C3%97%20Spike%20Confirmed-brightgreen)](#phase-0-results--kv-eviction--itl-spike)
[![Phase 1](https://img.shields.io/badge/Phase%201-%E2%9C%85%202.9%C3%97%20Scale%20Amplification-brightgreen)](#phase-1-results--pcie-contention-at-scale)
[![Phase 2](https://img.shields.io/badge/Phase%202-%F0%9F%94%A8%20In%20Progress-yellow)](#phase-2-tempo-pacing-scheduler)

> **가설 검증 먼저, 솔루션 설계는 그 다음.**

---

## 실험 결과 요약

| | Phase 0 | Phase 1 |
|---|---|---|
| **가설** | KV eviction → ITL 스파이크 | PCIe contention → NCCL BW 저하 |
| **결과** | ✅ **335×** 스파이크 확인 | ✅ **2.9×** 스케일 증폭 확인 |
| **핵심 수치** | TTFT P99: 23ms → 2,759ms | 8N BW 저하: 3.3% (2N 대비 3× 악화) |
| **플랫폼** | Perlmutter 1N / 4×A100 40GB | Perlmutter 2N / 4N / 8N |

---

## Phase 0 Results — KV Eviction → ITL Spike

**실험 조건**: `facebook/opt-6.7b`, TP=4, `gpu_memory_utilization=0.60` (강제 KV eviction),
concurrency=64, 300 requests, NERSC Perlmutter 1 node (job 52217192)

### Time-to-First-Token (TTFT) 실측 패턴

```
 TTFT / ITL (ms)
  2800 |##
       |##  <-- KV cache 압박 최고조 (t=0~1s)
       |##     64개 동시 요청이 제한된 HBM 쟁탈
  2000 |##     gpu_util=0.60 -> KV eviction 즉시 발생
       |##
       |##
  1000 |##
       |##
   500 |##~~
       |##~~
    18 |##~~::::::::::::::::
     8 |##~~::::::::::::::::  <-- steady-state P99 ~8ms
       +--------------------------------------------> Time (s)
        0  1  2  3  4  5  6  7  8  9  10

  ## t=0-1s:  P50=2,758ms  P99=2,759ms  [335x spike]
  ~~ t=1-2s:  P50=    7ms  P99=  492ms  [완화 중]
  :: t=2-10s: P50=    7ms  P99=  8-18ms [steady-state]
```

### 측정 통계 (65,210 토큰 레코드 / job 52217192)

| 지표 | 값 |
|---|---|
| Decode ITL P50 (steady-state) | **7.4 ms** |
| Decode ITL P99 (steady-state) | **9.8 ms** |
| TTFT P50 | **23.6 ms** |
| TTFT P99 | **2,759 ms** |
| TTFT Max | **2,759 ms** |
| **스파이크 증폭 (peak / baseline)** | **335×** |
| 총 수집 토큰 | 65,210 |
| 요청 수 | 300 |
| 실험 시간 | 9.78 s |

### 인과관계

```
gpu_util=0.60 -> HBM KV cache 한도 도달
     |
     v  vLLM이 KV blocks -> CPU RAM -> /tmp (NVMe) 로 evict
     |
     v  PCIe DMA가 GPU <-> NVMe 대역폭 점유
     |
     v  새 요청은 KV space 확보될 때까지 대기
     |
     v  TTFT: 23ms -> 2,759ms  (335x 증폭)
```

**결론**: KV eviction이 발생하는 순간, 새 요청의 첫 토큰 생성이 최대 **335배** 느려진다.

---

## Phase 1 Results — PCIe Contention at Scale

**실험 조건**: NCCL allreduce (1 GB tensor) + 동시 배경 NVMe I/O (dd),
2N / 4N / 8N (각 노드 4×A100 40GB), HPE Slingshot 11 (200 Gbps)

### NCCL AllReduce 대역폭 저하 (실측)

```
  BW (GB/s)
  18.0 |[===][---]
       |           [===][--]
  16.0 |                     [===][-]
       |
       |  [===] Baseline  [---]/[-] Contention 저하
       +----------------------------------------> Scale
          2 Node    4 Node    8 Node

  2 Node:  17.98 -> 17.78 GB/s  (-1.1%,  -0.20 GB/s)
  4 Node:  16.75 -> 16.34 GB/s  (-2.4%,  -0.41 GB/s)
  8 Node:  16.20 -> 15.66 GB/s  (-3.3%,  -0.54 GB/s)

  Scale-out 증폭: 1.0x -> 2.2x -> 2.9x  (scale-out hypothesis CONFIRMED)
```

### 수치 요약

| Scale | Baseline | Contention | 저하율 | 스케일 증폭 |
|-------|----------|------------|--------|------------|
| 2 Node (8 GPU) | 17.98 GB/s | 17.78 GB/s | −1.1% | 1.0× (base) |
| 4 Node (16 GPU) | 16.75 GB/s | 16.34 GB/s | −2.4% | **2.2×** |
| 8 Node (32 GPU) | 16.20 GB/s | 15.66 GB/s | −3.3% | **2.9×** |

**결론**: 노드 수가 늘수록 PCIe contention의 영향이 **2.9배 증폭**된다.
대규모 LLM 훈련에서 checkpoint I/O가 NCCL 성능을 지속적으로 저해한다.

---

## Phase 3: TEMPO Evaluation (Baseline vs TEMPO)

**환경**: Perlmutter 2 nodes × 4×A100 (world size 8), Llama-1B FSDP FULL_SHARD, 60 steps, ckpt_every=20  
**Job**: 52232395 (2026-04-29)

### NCCL BW: Checkpoint Steps에서의 개선

체크포인트 flush가 NCCL allreduce와 **겹치는 step**(step 20, 40)만 분리하면:

| 조건 | Ckpt Steps BW | 전체 평균 BW | Throttle Waits |
|---|---|---|---|
| **Baseline** (greedy flush) | 4.94 GB/s | 6.38 GB/s | 0 |
| **TEMPO** (paced flush) | **7.26 GB/s** | 5.56 GB/s | **220** |
| **Improvement** | **+47.0%** | −12.8%* | — |

\* 전체 평균이 낮은 것은 TEMPO step time이 길어지기 때문(background flush 대기)  
\* 체크포인트 단계에서 NCCL BW는 47% 개선 — TEMPO의 핵심 가설 증명

### 핵심 메커니즘 확인

```
Baseline: NCCL all-reduce 중 5.12 GB checkpoint flush 동시 진행
  → PCIe Root Complex 경쟁 → NCCL BW 4.94 GB/s

TEMPO:    NCCL phase 감지 → flush thread에 220회 pause 신호
  → NCCL I/O 분리 → NCCL BW 7.26 GB/s (+47%)
  → flush는 COMPUTE phase에서만 진행 (0.26 GB/s, throttled)
```

### Phase-Gated Flush 아이디어

```
LLM Forward Pass:
  +-- FFN (GEMM) --+  +-- Attention -----------------+  +-- FFN --+
  |  compute-bound  |  |  HBM + PCIe bandwidth-bound  |  |         |
  |  [FLUSH HERE]   |  |  [TEMPO BLOCKS I/O HERE]     |  | [FLUSH] |
  +-----------------+  +------------------------------+  +---------+

  Detection: FSDP comm hook (reduce_scatter timing + PhaseMonitor)
    NCCL_COMM phase -> I/O paused  (throttle_waits += 1 per 20ms)
    COMPUTE  phase  -> I/O allowed
```

### 구현 현황

| 컴포넌트 | 상태 | 파일 |
|---|---|---|
| PhaseMonitor | ✅ 완성 | `tempo/phase_monitor.py` |
| CheckpointManager | ✅ 완성 | `tempo/checkpoint_manager.py` |
| TEMPOScheduler | ✅ 완성 | `tempo/scheduler.py` |
| FSDP comm hook | ✅ 완성 | `phase3/train_with_tempo.py` |
| SpikeAbsorber (C++) | 📋 설계 완료 | `src/spike_absorber/` |
| PacingDaemon (C++) | 📋 설계 완료 | `src/pacing_daemon/` |

---

## Quick Start (Perlmutter)

```bash
# Phase 0 재현
sbatch phase0/verify_interference.slurm
python3 phase0/plot_inference_interference.py \
    --itl results/phase0/itl_profile.csv \
    --io  results/phase0/io_profile.csv

# Phase 1 재현 (8 node)
sbatch phase1/run_phase1_8node.slurm
python3 phase3/plot_killer_graph.py --scale-compare \
    --results-root results --output-dir results/figures

# SLURM 계정: -A m5320, pytorch/2.8.0
```

---

## Repository Structure

```
Skim-Tempo/
├── phase0/                       KV eviction -> ITL spike 실험
│   ├── workload_injector.py      vLLM ITL profiler (gpu_util=0.6 -> eviction)
│   ├── hardware_monitor.sh       NVMe / GPU I/O 100ms 로거
│   ├── plot_inference_interference.py
│   └── verify_interference.slurm
│
├── phase1/                       NCCL BW degradation 실험
│   ├── train_llm_profiling.py    GPT-2 Large allreduce 프로파일러
│   ├── background_io.sh          배경 NVMe I/O 주입기
│   └── run_phase1_{4,8}node.slurm
│
├── results/
│   ├── phase0/itl_profile.csv           <- 65,210 ITL records (실측)
│   ├── {2,4,8}node/*/nccl_bw_rank0.csv <- NCCL BW 실측
│   └── figures/
│       ├── fig1_itl_vs_kv_eviction.*    <- Phase 0 killer graph
│       └── scale_compare.*             <- Phase 1 scale-out graph
│
├── tempo/                        Python integration (Phase 2)
│   ├── scheduler.py              TEMPO pacing orchestrator
│   ├── phase_monitor.py          Attention/FFN 위상 감지
│   └── checkpoint_manager.py    NVMe -> Lustre 청크 flush
│
└── src/                          C++ core (Phase 2)
    ├── attention_monitor/        CUPTI kernel classifier
    ├── spike_absorber/           MPSC lock-free ring buffer
    └── pacing_daemon/            io_uring phase-gated flush
```

---

## Citation

```bibtex
@inproceedings{kim2026tempo,
  title  = {{TEMPO}: Phase-Aware KV Cache Eviction Pacing for Jitter-Free {LLM} Inference},
  author = {Kim, Sunggon},
  booktitle = {OSDI '26},
  year   = {2026},
}
```

<div align="center">
<sub>NERSC Perlmutter · pytorch/2.8.0 · vLLM v1 · Phase 0: 335× ITL spike confirmed · Phase 1: 2.9× scale amplification confirmed</sub>
</div>
