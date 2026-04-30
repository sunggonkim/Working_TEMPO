# TEMPO: Perlmutter 슈퍼컴퓨터에서 분산 LLM 학습 체크포인트 I/O 간섭 제거

[![플랫폼](https://img.shields.io/badge/플랫폼-NERSC%20Perlmutter-0075A2)](https://docs.nersc.gov/systems/perlmutter/)
[![트레이스](https://img.shields.io/badge/워크로드-BurstGPT%20실측%20트레이스-blueviolet)](https://github.com/HPMLL/BurstGPT)
[![NIC](https://img.shields.io/badge/네트워크-HPE%20Slingshot--11%20Cassini-orange)](https://www.hpe.com/us/en/compute/hpc/slingshot-interconnect.html)
[![PCIe](https://img.shields.io/badge/PCIe-AllReduce%2041.7%25%20감소-brightgreen)](#측정-결과)
[![네트워크](https://img.shields.io/badge/네트워크-I%2FO%20flood%20간섭%20분석-blue)](#실험-환경)

---

## 핵심 요약

분산 LLM 학습에서 체크포인트 I/O는 두 가지 하드웨어 병목을 동시에 일으킨다.

| 경쟁 자원 | 증상 | 실측값 |
|---|---|---|
| **PCIe 버스** | DMA 체크포인트와 AllReduce 버퍼 전송이 충돌 | AllReduce **21.4 ms → 12.5 ms** (−41.7%) |
| **Slingshot-11 광 링크** | 체크포인트 I/O flood 시 NCCL 대역폭 저하 | flood 중 BW **20.93 → 19.38 GB/s** |

**TEMPO** (Timed Eviction with Memory-Pressure Orchestration)는
*Phase-Gate* 기법으로 체크포인트 DMA를 NCCL-free 구간에만 허용하여
이 두 가지 충돌을 동시에 제거한다.

---

## 핵심 문제 (Motivation)

### 문제 1 — PCIe 버스 경쟁

GPU가 체크포인트를 NVMe에 저장할 때 DMA 엔진이 PCIe 버스를 사용한다.
같은 시각 FSDP AllReduce도 동일한 PCIe 경로로 그래디언트 버퍼를 전송한다.
두 트래픽이 PCIe 루트 컴플렉스를 공유하면서 AllReduce 지연이 급증한다.

> **실측 환경**: Perlmutter 4노드 × A100 40GB, PyTorch 2.8.0 FSDP, 200스텝

![PCIe 자원 경쟁 — 실측 데이터](results/figures/readme_fig_motivation_pcie.png)

**실측 결과 요약**:
- Baseline AllReduce 평균: **21.38 ms** (p99: 22.75 ms)
- TEMPO AllReduce 평균: **12.46 ms** (p99: 14.35 ms)
- **41.7% 감소** — Phase-Gate 하나로 달성

---

### 문제 2 — Slingshot-11 네트워크 경쟁

Perlmutter Dragonfly+ 토폴로지에서 Lustre 파일시스템 I/O와 NCCL AllReduce는
동일한 HPE Slingshot-11 광 글로벌 링크를 공유한다.
체크포인트 I/O가 flood 수준으로 발생하면 AllReduce 대역폭이 하락한다.

> **실측 환경**: Perlmutter 8노드 × A100, 500스텝 (step 100–299 I/O flood 활성화)

![네트워크 간섭 — 실측 데이터](results/figures/readme_fig_motivation_network.png)

**실측 결과 요약**:
- flood 전 AllReduce BW: **20.93 GB/s**
- flood 중 AllReduce BW (Baseline): **19.38 GB/s** (−7.4%)
- flood 중 AllReduce BW (TEMPO-v2): **19.52 GB/s** (Baseline 대비 +0.7% 회복)
- flood 구간 Baseline 최소 BW: **11.02 GB/s** (순간 급락)

---

## TEMPO 설계 (Design)

### Phase-Gate 원리

TEMPO는 학습 스텝을 두 구간으로 분리한다:

1. **NCCL 구간** — AllReduce 실행 중. 체크포인트 DMA 완전 차단.
2. **NCCL-free 구간** — AllReduce 완료 후. 체크포인트 DMA 허용.

이 분리는 PyTorch CUDA 이벤트(`gate_event`)로 구현된다.
DMA 스트림은 `io_stream.wait_event(gate_event)` 호출로 블록되다가
AllReduce 완료 신호 이후에만 실행된다.

![TEMPO Phase-Gate 타임라인 — 실측 수치 기반](results/figures/readme_fig_design_timeline.png)

### 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                     TEMPO 컨트롤 플레인                           │
│                                                                 │
│  PhaseMonitor ──→ Scheduler ──→ SparseTransfer                  │
│       │               │              │                          │
│  (AllReduce 완료)  (gate_event)  (NVMe DMA)                     │
│       │               │              │                          │
│  compute_stream   gate_event     io_stream                      │
│  ══════════════   ═══╪════════   ════╪════════════════          │
│  [Forward/BW/AR]     │              │                           │
│                    wait_event ──────┘                           │
└─────────────────────────────────────────────────────────────────┘
         │                                    │
    PCIe 버스 (AllReduce 전용)          PCIe 버스 (DMA 전용)
    NCCL 구간에만 사용                  NCCL-free 구간에만 사용
```

### 핵심 컴포넌트

| 컴포넌트 | 파일 | 역할 |
|---|---|---|
| `PhaseMonitor` | `tempo/phase_monitor.py` | AllReduce 완료 시점 감지 |
| `TEMPOScheduler` | `tempo/scheduler.py` | gate_event 발행 및 DMA 큐 관리 |
| `NanoOverlapController` | `tempo/nano_overlap.py` | compute/io CUDA 스트림 파이프라인 |
| `SparseTransfer` | `tempo/sparse_transfer.py` | Sparse 텐서 압축 전송 |
| `NetworkMonitor` | `tempo/network_monitor.py` | Cassini HW 혼잡 카운터 모니터링 |
| `CassiniHWCounters` (C) | `src/c_api/` | sysfs mmap (~0.3 µs vs ~15 µs Python) |
| `PinnedBufferPool` | `src/spike_absorber/` | CUDACachingAllocator 락 경쟁 제거 |

---

## 측정 결과 (Results)

### PCIe 타임라인 측정 (phase7)

> 방법론: `NCCL_P2P_DISABLE=1`으로 AllReduce를 PCIe 강제 경유,
> CUDA Event로 AllReduce / DMA 시간 각각 측정 (4노드 × 200스텝)

![TEMPO 성능 요약 — 실측 데이터](results/figures/readme_fig_results_summary.png)

| 지표 | Baseline | TEMPO | 변화 |
|---|---|---|---|
| AllReduce 평균 | 21.38 ms | 12.46 ms | **−41.7%** |
| AllReduce p99 | 22.75 ms | 14.35 ms | **−36.9%** |
| DMA 체크포인트 평균 | 26.36 ms | 20.39 ms | **−22.6%** |

> DMA 시간도 단축되는 이유: PCIe 버스 독점 사용 → DMA 처리량 향상

### PCIe Gantt 타임라인 (phase7 실측)

![PCIe Gantt 타임라인](results/figures/fig9_pcie_timeline.png)

### 네트워크 간섭 측정 (phase4 실측)

> 방법론: 8노드 중 4노드가 16 GB/s Lustre 쓰기로 flood 발생,
> 나머지 4노드의 AllReduce BW를 probe_rank*.csv로 측정

![네트워크 간섭 측정](results/figures/fig7_network_interference.png)

---

## 실험 환경

### 하드웨어

| 항목 | 사양 |
|---|---|
| 시스템 | NERSC Perlmutter (2025) |
| CPU | AMD EPYC 7763, 64코어/노드 |
| GPU | 4 × NVIDIA A100 40GB/노드 |
| 네트워크 | HPE Slingshot-11, 200 Gbps/포트, Dragonfly+ |
| 스토리지 | Lustre `$PSCRATCH`, 최대 ~28 GB/s 집계 |
| PCIe | PCIe 4.0 x16, CPU↔GPU 양방향 공유 |

### 소프트웨어

| 항목 | 버전 |
|---|---|
| PyTorch | 2.8.0 (`module load pytorch/2.8.0`) |
| NCCL | 2.29.2-cu13 (HPE Slingshot 플러그인) |
| 모델 | Llama-1B (hidden=2048, layers=16, GQA) |
| 분산 | FSDP FULL_SHARD |
| 추론 | vLLM + BurstGPT Pareto 도착 패턴 |

### 핵심 환경변수

```bash
FI_CXI_DISABLE_HMEM_MODES=1        # Slingshot HMEM 비활성화 (필수)
NCCL_PLUGIN_P2P=slingshot11         # Cassini NIC 전용 플러그인
CUDA_VISIBLE_DEVICES=0,1,2,3        # A100 4장
```

---

## 실험 재현 방법

### 환경 설정

```bash
# NERSC Perlmutter 로그인 후
module load pytorch/2.8.0
cd $PSCRATCH
git clone https://github.com/sunggonkim/Working_TEMPO Skim-Tempo
cd Skim-Tempo
```

### Phase 7 — PCIe 타임라인 측정 (핵심 실험)

```bash
sbatch phase7/run_phase7_eval.slurm
# 완료 후
python3 scripts/plot_micro_benchmarks.py --fig 9
```

결과: `results/phase7/timeline_{baseline,tempo}.csv`

### Phase 4 — 네트워크 간섭 측정

```bash
sbatch phase4/run_phase4_eval.slurm
sbatch phase4/run_io_nccl_sweep.slurm
```

결과: `results/phase4/network_interference/{baseline,tempo-v2}/`

### Phase 0 — 추론 ITL CDF 측정

```bash
sbatch phase0/run_itl_cdf_eval.slurm
# 완료 후
python3 scripts/plot_micro_benchmarks.py --fig 11
```

결과: `results/phase0/itl_{baseline,tempo}.csv`

### Phase 1 — 분산 학습 기준 성능

```bash
sbatch phase1/run_phase1_4node.slurm   # 4노드
sbatch phase1/run_phase1_8node.slurm   # 8노드
```

### Phase 3 — TEMPO 통합 학습

```bash
sbatch phase3/run_evaluation.slurm
sbatch phase3/run_chunk_sweep.slurm
```

### 전체 그림 재생성

```bash
# 실측 데이터가 있는 그림만 (README용)
python3 scripts/plot_readme_figures.py

# 모든 실험 그림
python3 scripts/plot_micro_benchmarks.py --fig 9   # PCIe 타임라인 (phase7)
python3 scripts/plot_micro_benchmarks.py --fig 10  # IO-NCCL sweep (phase4)
python3 scripts/plot_micro_benchmarks.py --fig 11  # ITL CDF (phase0)
python3 scripts/make_figures.py                    # 아키텍처 다이어그램
```

---

## 프로젝트 구조

```
Skim-Tempo/
├── tempo/                    # TEMPO 핵심 라이브러리
│   ├── phase_monitor.py      # AllReduce 완료 감지
│   ├── scheduler.py          # Phase-Gate 스케줄러
│   ├── nano_overlap.py       # CUDA 스트림 파이프라인 (v4)
│   ├── sparse_transfer.py    # Sparse 텐서 압축 전송
│   ├── network_monitor.py    # Cassini HW 카운터 모니터링
│   └── vllm_hook.py          # vLLM 추론 엔진 연동
├── src/
│   ├── c_api/                # C mmap 카운터 리더 (~0.3 µs)
│   ├── pacing_daemon/        # I/O 페이싱 데몬
│   └── spike_absorber/       # PinnedBufferPool
├── phase0/                   # ITL spike 인과관계 실험
├── phase1/                   # PCIe/Slingshot 경쟁 기준선
├── phase3/                   # TEMPO 통합 평가
├── phase4/                   # 네트워크 간섭 정량화
├── phase5/                   # 토폴로지 QoS 평가
├── phase6/                   # TEMPO v4 통합 실험
├── phase7/                   # PCIe 타임라인 프로파일러
├── scripts/
│   ├── plot_readme_figures.py  # README용 실측 그림 생성
│   ├── plot_micro_benchmarks.py
│   ├── make_figures.py
│   └── simulate_chunk_sweep.py
├── results/
│   ├── figures/              # 생성된 그림 (PNG/PDF)
│   ├── phase7/               # PCIe 타임라인 실측 CSV
│   └── phase4/               # 네트워크 간섭 실측 CSV
└── configs/
    ├── deepspeed_zero3.json
    └── NanumGothic-Regular.ttf  # 그림 생성용 한글 폰트
```

---

## 데이터 출처

모든 README 그림은 **합성 데이터 없이** 실측 CSV만 사용한다:

| 그림 | 데이터 파일 | 실험 |
|---|---|---|
| PCIe 자원 경쟁 | `results/phase7/timeline_{baseline,tempo}.csv` | phase7 (완료) |
| 네트워크 간섭 | `results/phase4/network_interference/*/probe_rank*.csv` | phase4 (완료) |
| TEMPO 성능 요약 | `results/phase7/timeline_{baseline,tempo}.csv` | phase7 (완료) |

대기 중인 실험 완료 후 추가될 그림:
- `fig10_io_nccl_sweep.png` — IO-NCCL 동시 sweep (job 52244056)
- `fig11_itl_cdf.png` — 추론 ITL CDF (job 52243961)

---

## 참고문헌

- Wang et al., *BurstGPT: A Real-world Workload Dataset for LLM Serving*, NSDI 2024
- HPE Slingshot-11 Architecture, [hpe.com](https://www.hpe.com/us/en/compute/hpc/slingshot-interconnect.html)
- NERSC Perlmutter Documentation, [docs.nersc.gov](https://docs.nersc.gov/systems/perlmutter/)
- PyTorch FSDP, [pytorch.org](https://pytorch.org/docs/stable/fsdp.html)
