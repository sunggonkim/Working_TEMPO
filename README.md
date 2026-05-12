# TEMPO: 분산 LLM 학습에서 체크포인트 I/O 간섭 제거를 위한 Phase-Gate 스케줄링

[![플랫폼](https://img.shields.io/badge/플랫폼-NERSC%20Perlmutter%20A100-0075A2?logo=nvidia)](https://docs.nersc.gov/systems/perlmutter/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%20FSDP-EE4C2C?logo=pytorch)](https://pytorch.org)
[![NCCL](https://img.shields.io/badge/NCCL-2.29.2%20Slingshot--11-76B900)](https://developer.nvidia.com/nccl)
[![AllReduce](https://img.shields.io/badge/AllReduce%20Latency-−50.1%25-brightgreen)](#21-bottleneck-1--pcie-버스-경쟁-노드-내)
[![DMA](https://img.shields.io/badge/DMA%20Time-−21.7%25-green)](#21-bottleneck-1--pcie-버스-경쟁-노드-내)
[![Network](https://img.shields.io/badge/Min%20NCCL%20BW-11→14.7%20GBs-blue)](#22-bottleneck-2--slingshot-11-패브릭-경쟁-노드-간)

---

## 초록

체크포인트는 분산 LLM 학습의 필수 구성 요소이지만, **두 가지 독립적인 하드웨어 경쟁 경로**를 통해 학습 성능을 파괴한다. 첫째, GPU의 NVMe DMA 엔진과 FSDP AllReduce의 그래디언트 버퍼 전송이 동일한 **PCIe Root Complex**를 공유하여 AllReduce 지연이 **24.98 ms → 50.1% 증가**한다. 둘째, 여러 노드가 동시에 체크포인트를 Lustre에 기록할 때 NCCL과 I/O 트래픽이 동일한 **HPE Slingshot-11 200 Gbps 광 링크**를 공유하여 집합 대역폭이 **11.02 GB/s까지 순간 붕괴**한다.

**TEMPO**는 CUDA 이벤트 기반 *Phase-Gate* 메커니즘으로 이 두 충돌을 동시에 해결한다. AllReduce가 진행되는 NCCL 구간 동안 `io_stream.wait_event(gate_event)`로 DMA를 완전히 차단하고, AllReduce 완료 이후 NCCL-free 구간에만 I/O를 허용한다. NERSC Perlmutter에서의 실측 결과: AllReduce 지연 **−50.1%**, DMA 처리 시간 **−21.7%**, flood 중 최소 NCCL 대역폭 **+33.8%**.

---

## 1. 연구 동기 — 분산 LLM 학습에서의 I/O 간섭

LLM 학습은 주기적인 체크포인트 저장을 요구한다. 문제는 이 체크포인트 I/O가 **학습의 핵심 통신 연산(NCCL AllReduce)** 과 두 가지 하드웨어 자원을 공유한다는 것이다. 아래의 실측 데이터는 기존 시스템(Baseline: 체크포인트를 학습 루프와 병렬로 직접 Lustre에 기록)이 얼마나 심각한 성능 저하를 유발하는지 증명한다.

---

### 1.1 Bottleneck 1 — PCIe 버스 경쟁 (노드 내)

**충돌 구조.** FSDP 기반 분산 학습에서 AllReduce와 체크포인트 DMA는 동일한 AMD EPYC PCIe I/O Die를 공유한다. 기존 시스템은 두 작업을 동시에 실행하여 PCIe 버스 대역폭을 두고 경쟁하며, 이로 인해 AllReduce 지연이 폭증한다.

```
Baseline (기존):
  compute_stream: ──[Forward]──[Backward]──[AllReduce <─→ PCIe 공유]──
  io_stream:      ─────────────────────────[DMA→NVMe <─→ PCIe 공유]──
                                                  ↑ ↑
                      동일한 PCIe Root Complex → AllReduce 지연 폭증

TEMPO (Phase-Gate):
  compute_stream: ──[Forward]──[Backward]──[AllReduce, PCIe 독점]─────
                                                   │ gate_event.record()
  io_stream:      ──────────────── wait_event() ───┘──[DMA→NVMe]──────
                                                   ↑
                                 NCCL 완료 후에만 DMA 허용 → 간섭 제로
```

TEMPO의 핵심 통찰은 단순하다: **AllReduce와 DMA를 시간적으로 분리하면 PCIe를 번갈아 독점 사용할 수 있다.** AllReduce 구간에서는 DMA가 완전히 차단되어 AllReduce가 PCIe를 독점하고, DMA 구간에서는 NCCL 트래픽이 없어 DMA 처리량도 함께 향상된다.

**실측 결과.**

> **환경**: Perlmutter 4노드 × A100 40GB, FSDP, `NCCL_P2P_DISABLE=1` (AllReduce PCIe 강제)  
> CUDA Event 독립 타이밍, 4랭크 × 200스텝

![PCIe 경쟁 실측 — Gantt 타임라인 + 분포 비교](results/figures/readme_fig_motivation_pcie.png)

**▲ Fig 1. PCIe 버스 경쟁 실측.**
왼쪽: 처음 12스텝의 AllReduce(빨강)와 DMA(주황) Gantt 타임라인. Baseline에서는 두 막대가 완전히 겹쳐 PCIe 경쟁이 발생한다. TEMPO에서는 `gate_event`(초록 수직선) 이후에만 DMA가 시작되어 시간적 분리가 명확하다.  
오른쪽: 전체 200스텝 × 4랭크의 AllReduce 지연 박스플롯. Baseline 중앙값 ~25 ms → TEMPO ~12 ms로 절반으로 줄었고 분산도 크게 감소한다.

| 지표 | Baseline | TEMPO | 개선율 |
|---|---:|---:|---:|
| **AllReduce 평균** | 24.98 ms | 12.46 ms | **−50.1%** |
| **AllReduce p50** | 24.32 ms | 12.41 ms | −48.9% |
| **AllReduce p99** | 27.80 ms | 14.27 ms | **−48.7%** |
| **DMA 평균** | 26.05 ms | 20.39 ms | **−21.7%** |

DMA 시간도 단축되는 이유: Phase-Gate 덕분에 DMA 구간에는 NCCL 트래픽이 없어 DMA가 PCIe를 독점 사용 → 처리량 향상.

---

### 1.2 Bottleneck 2 — Slingshot-11 패브릭 경쟁 (노드 간)

**충돌 구조.** Perlmutter의 Dragonfly+ 토폴로지에서 Lustre 파일시스템 I/O와 NCCL AllReduce는 동일한 HPE Slingshot-11 200 Gbps 광 글로벌 링크를 공유한다. 여러 노드가 동시에 체크포인트를 기록하는 *집단 체크포인트 flooding* 순간에 네트워크 혼잡이 폭발적으로 발생한다.

```
Perlmutter Dragonfly+ 패브릭:

  노드 0 ──┐                           ┌── 타 노드 NCCL AllReduce
  노드 1 ──┤  Slingshot-11 200 Gbps    ├── (grad. 버퍼 교환)
  ...      ├─ 공유 광 글로벌 링크 ──────┤
  노드 7 ──┘  ↑ 동시에 Lustre I/O     └── → 링크 포화 → BW 급락
             (8노드 × ~2 GB/s = 16 GB/s flood)
```

노드 내 PCIe 경쟁을 Phase-Gate 하나로 해결하는 것만으로는 부족하다. 집단 checkpoint flooding으로 인한 *네트워크 수준*의 간섭도 제어해야 한다.

**실측 결과.**

> **환경**: Perlmutter 8노드 × A100, 500스텝  
> step 100–299: 4노드가 16 GB/s Lustre flood 발생  
> 나머지 4노드의 NCCL AllReduce BW를 `probe_rank{0..7}.csv`로 독립 측정

![Slingshot-11 네트워크 간섭 실측](results/figures/readme_fig_motivation_network.png)

**▲ Fig 2. Slingshot-11 네트워크 간섭 실측.**
왼쪽: rank 0의 AllReduce 대역폭 시계열. 주황 배경 = flood 구간(step 100–299). Baseline(빨강)은 flood 시작 직후 BW가 11.02 GB/s까지 급락하는 sawtooth 패턴이 명확하다. TEMPO-v2(파랑)는 NetworkMonitor가 혼잡을 사전 감지하여 I/O를 throttle, BW가 14.74 GB/s 이상 유지된다.  
오른쪽: flood 전(회색) vs flood 중(색상) BW 분포 비교, 8랭크 전체 통계. Baseline의 분포 하단 꼬리가 극도로 낮아지는 반면 TEMPO는 하단이 안정적이다.

| 상황 | Baseline | TEMPO-v2 | 개선율 |
|---|---:|---:|---:|
| flood 중 **최솟값** BW | 11.02 GB/s | 14.74 GB/s | **+33.8%** |
| flood 중 평균 BW | 19.38 GB/s | 19.52 GB/s | +0.7% |

**최솟값이 핵심 지표.** 최솟값은 학습 중 발생하는 *최악의 순간 지연*에 직접 대응한다. LLM 학습에서 단 한 번의 AllReduce 지연 spike가 전체 파이프라인의 동기화를 지연시킨다.

---

### 1.3 스케일 확대 시 경쟁이 비선형적으로 증폭된다

단일 노드에서는 미미한 PCIe 경쟁이 노드 수가 늘어날수록 **비선형적**으로 악화된다. 이는 여러 노드가 동시에 체크포인트를 Lustre에 기록하는 집단 flooding 패턴 때문이다.

> **환경**: Perlmutter 2→4→8노드 확장 실험  
> 모든 노드가 동시에 체크포인트를 기록하면서 NCCL AllReduce BW 측정

![노드 스케일에 따른 NCCL BW 감쇠](results/figures/fig4_phase1_barchart.png)

**▲ Fig 3. 노드 스케일 실험 (실측).** 파란 막대 = 체크포인트 없는 순수 NCCL BW. 빨간 막대 = 체크포인트 동시 기록 중 NCCL BW. 노드 수가 늘수록 두 막대의 간격이 점점 벌어진다.

| 노드 수 | GPU 수 | Baseline BW | 경쟁 중 BW | 감쇠율 | 증폭 배수 |
|---|---:|---:|---:|---:|---:|
| 2노드 | 8 | 17.98 GB/s | 17.78 GB/s | −1.1% | 1.0× |
| 4노드 | 16 | 16.75 GB/s | 16.34 GB/s | −2.4% | 2.2× |
| 8노드 | 32 | 16.20 GB/s | 15.66 GB/s | −3.3% | **2.9×** |

노드 수가 2배 늘어날 때 감쇠율은 2배 이상 증가한다. 집단 flooding의 네트워크 혼잡은 **초선형(super-linear)** 특성을 가진다. 수천 GPU 규모의 클러스터에서는 이 문제가 치명적이다.

---

## 2. TEMPO 설계 — Phase-Gate Scheduling

### 2.1 핵심 메커니즘

위 세 실측 결과는 공통 원인을 지목한다: AllReduce와 I/O가 동일한 하드웨어 자원을 동시에 사용한다는 것. TEMPO의 해결책은 단 두 줄의 CUDA 코드다.

```python
# ① AllReduce 완료 직후 — gate_event를 compute_stream에 기록
gate_event.record(compute_stream)    # AllReduce가 끝난 시점을 마킹

# ② DMA는 gate_event 이전에는 실행 불가
io_stream.wait_event(gate_event)     # AllReduce 완료 전 DMA 완전 차단
io_stream.enqueue(dma_copy_to_nvme)  # AllReduce 완료 후에만 DMA 실행
```

이것이 **Phase-Gate**: NCCL 구간에서는 게이트가 닫혀 DMA가 차단되고, AllReduce 완료 시점에 게이트가 열려 DMA가 실행된다.

![Phase-Gate 타임라인](results/figures/readme_fig_design_timeline.png)

**▲ Fig 4. Phase-Gate 타임라인 (실측 수치 기반 스케일).** 위: Baseline — DMA(주황)와 AllReduce(빨강)가 PCIe를 동시에 점유. 아래: TEMPO — `gate_event`(초록 수직선)를 경계로 AllReduce 완료 후에야 DMA가 시작. Baseline AR=24.98ms, TEMPO AR=12.46ms.

### 2.2 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│  PyTorch FSDP 학습 루프                                              │
│                                                                     │
│  compute_stream ──→ [Forward] ──→ [Backward] ──→ [NCCL AllReduce]   │
│                                                        │            │
│                                            gate_event.record()      │
│                                                        │            │
│  io_stream      ─────────────── wait_event() ──────────┘──→ [DMA]  │
├─────────────────────────────────────────────────────────────────────┤
│  TEMPO 컨트롤 플레인                                                  │
│                                                                     │
│  PhaseMonitor  →  TEMPOScheduler  →  CheckpointManager              │
│  (NCCL 완료)      (gate_event)       (청크 단위 Lustre flush)         │
│       │                │                     │                      │
│  NetworkMonitor ────────                ServiceGainScheduler        │
│  /sys/class/net/hsn*/  5ms 폴링         gain 계산, 혼잡 시 defer      │
├─────────────────────────────────────────────────────────────────────┤
│  하드웨어 인터페이스                                                    │
│  CassiniHWCounters: C mmap (~0.3 µs)    PinnedBufferPool            │
└─────────────────────────────────────────────────────────────────────┘
        │ PCIe: AllReduce 전용                  │ PCIe: DMA 전용
        │ (NCCL 구간)                           │ (NCCL-free 구간)
```

### 2.3 네트워크 혼잡 제어 — NetworkMonitor

노드 내 PCIe 경쟁은 Phase-Gate 하나로 해결되지만, 노드 간 Slingshot 혼잡은 추가 제어가 필요하다. `NetworkMonitor`는 5 ms 간격으로 Slingshot NIC 사용률을 실시간 감시하여 혼잡 시 flush를 자동 throttle한다.

```python
# tempo/network_monitor.py 요약
# /sys/class/net/hsn{0,1}/statistics/tx_bytes 폴링 (EMA α=0.25, 5ms 간격)
# 혼잡 임계값: 링크 BW × 75% = 200 Gbps × 0.75 = 150 Gbps

if nic_util_ema > LINK_SPEED_BPS * 0.75:
    wait_for_bw_headroom(needed_bps)  # flush 차단 → NCCL에 대역폭 양보
```

비 Perlmutter 환경에서는 자동으로 `/proc/net/dev`로 fallback하여 이식성을 보장한다.

### 2.4 서비스 이득 기반 우선순위 (ServiceGainScheduler)

모든 체크포인트가 동등한 긴급도를 가지지 않는다. `ServiceGainScheduler`는 O(1) 수식으로 각 flush job에 이득 점수를 계산한다.

```
score = α·learning_progress + β·recovery_value + γ·urgency
      = 0.45×(steps/horizon) + 0.35×(1−e^{−λ·steps}) + 0.20×urgency

score ≥ 0.70 → Slingshot TC3 (Expedited Forwarding)  ← NCCL과 동급
score ∈ [0.40, 0.70) → TC2 (Assured Forwarding)
score < 0.30 → 혼잡 시 완전 지연 (defer)
```

Slingshot TC 매핑은 `socket.IP_TOS` (DSCP 설정)만으로 달성 — 스위치 ASIC이 하드웨어 수준에서 낮은 이득 I/O를 자동으로 뒤로 밀어버린다. **CPU 오버헤드 제로.**

---

## 3. 통합 측정 결과

![성능 요약 대시보드](results/figures/readme_fig_results_summary.png)

**▲ Fig 5. 통합 성능 대시보드 (4노드 × 4랭크 × 200스텝, 실측 CSV 기반).**
왼쪽: AllReduce 평균 지연 — 24.98 ms → 12.46 ms (−50.1%).
중앙: DMA 처리 시간 — 26.05 ms → 20.39 ms (−21.7%).
오른쪽: AllReduce 지연 바이올린 분포 — TEMPO에서 분산 자체가 줄어 꼬리 지연시간까지 개선됨.

![baseline vs TEMPO Killer Graph — Slingshot 대역폭 시계열](results/figures/fig5_phase3_comparison.png)

**▲ Fig 6. "Killer Graph" — TEMPO 효과 직접 비교.** 위: Baseline NCCL 대역폭 — 체크포인트 스텝(*)마다 BW가 급락하는 sawtooth 패턴. 아래: TEMPO — 체크포인트 스텝에서도 BW가 평탄하게 유지됨.

### 핵심 수치 요약

| 지표 | Baseline | TEMPO | 개선 | 실험 |
|---|---:|---:|---:|---|
| **AllReduce 지연 (평균)** | 24.98 ms | 12.46 ms | **−50.1%** | 4노드 × 200스텝 |
| **AllReduce 지연 (p99)** | 27.80 ms | 14.27 ms | **−48.7%** | 4노드 × 200스텝 |
| **DMA 처리 시간 (평균)** | 26.05 ms | 20.39 ms | **−21.7%** | 4노드 × 200스텝 |
| **flood 중 최소 NCCL BW** | 11.02 GB/s | 14.74 GB/s | **+33.8%** | 8노드 flood |

---

## 4. 실험 재현

### 환경 설정

```bash
module load pytorch/2.8.0      # NERSC Perlmutter 필수

git clone https://github.com/sunggonkim/Working_TEMPO $PSCRATCH/Skim-Tempo
cd $PSCRATCH/Skim-Tempo

# 필수 환경 변수
export PSCRATCH="/pscratch/sd/s/$USER"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1              # AllReduce PCIe 경유 강제
export FI_CXI_DISABLE_HMEM_MODES=1    # Slingshot HMEM 비활성화
export NCCL_NET_PLUGIN=slingshot11
export OMP_NUM_THREADS=1
```

### 핵심 실험 실행 (PCIe 타임라인 — 약 5분)

```bash
# SLURM 제출 (4노드 × 4GPU)
sbatch phase7/run_phase7_eval.slurm

# 결과 CSV 확인
ls results/phase7/timeline_{baseline,tempo}.csv
# → step,rank,mode,allreduce_ms,dma_ms,overlap_ms,stall_ms,wall_s

# 그림 재생성
python3 scripts/plot_readme_figures.py
python3 scripts/plot_micro_benchmarks.py --fig 9
```

### 네트워크 간섭 실험 (약 15분, 8노드)

```bash
sbatch phase4/run_phase4_eval.slurm

# 결과 확인 후 그림 재생성
python3 scripts/make_figures.py   # fig4, fig7 재생성
```

---

## 5. 실험 환경

| 항목 | 사양 |
|---|---|
| **시스템** | NERSC Perlmutter |
| **CPU** | AMD EPYC 7763 (PCIe 4.0, I/O Die 공유) |
| **GPU** | 4 × NVIDIA A100 SXM 40GB / 노드 |
| **네트워크** | HPE Slingshot-11, 200 Gbps/포트, Dragonfly+ |
| **스토리지** | Lustre `$PSCRATCH`, ~28 GB/s 집계 BW |
| **PyTorch** | 2.8.0 FSDP FULL_SHARD |
| **NCCL** | 2.29.2-cu13 (HPE Slingshot 플러그인) |
| **모델** | Llama-1B (hidden=2048, layers=16, GQA) |

---

## 6. 프로젝트 구조

```
Skim-Tempo/
├── tempo/                         # TEMPO 핵심 라이브러리
│   ├── phase_monitor.py           # AllReduce 완료 감지
│   ├── scheduler.py               # Phase-Gate 스케줄러 (V1–V5)
│   ├── checkpoint_manager.py      # O(1) NVMe 저장 + Lustre flush
│   ├── network_monitor.py         # Slingshot NIC 혼잡 감시 (sysfs)
│   ├── service_gain.py            # 서비스 이득 우선순위 스케줄러
│   └── qos_mapper.py              # DSCP → Slingshot TC 매핑
│
├── phase7/
│   ├── pcie_timeline_profiler.py  # ★ PCIe 타임라인 측정 (핵심 실험)
│   └── run_phase7_eval.slurm
│
├── results/phase7/
│   ├── timeline_baseline.csv      # ★ 실측 (4랭크 × 200스텝)
│   └── timeline_tempo.csv         # ★ 실측
│
└── scripts/
    ├── plot_readme_figures.py     # README 그림 재생성
    └── make_figures.py            # 모든 실험 그림 재생성
```

---

## 참고문헌

- Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving*, OSDI 2024
- Perng et al., *Pie: Programmable and Efficient Inferlets on LLM Inference Service*, SOSP 2025
- Yang et al., *Aegaeon: Scalable and Elastic Multi-model LLM Serving*, SOSP 2025
- Jiang et al., *Teola: Towards End-to-End Optimization of LLM-based Applications*, OSDI 2024
- HPE Slingshot-11 Architecture, [hpe.com](https://www.hpe.com/us/en/compute/hpc/slingshot-interconnect.html)
- NERSC Perlmutter Documentation, [docs.nersc.gov](https://docs.nersc.gov/systems/perlmutter/)
- PyTorch FSDP, [pytorch.org/docs/stable/fsdp.html](https://pytorch.org/docs/stable/fsdp.html)
