# TEMPO-GO 논문·평가 Artifact

이 디렉터리는 TEMPO-GO 논문 소스, 빌드된 PDF, README 정량 그래프와
machine-readable artifact manifest를 제공한다. 연구 동기, 설계, 7-arm 결과,
관련 연구와 전체 실행 역사는 저장소의 [루트 README](../../README.md)와
[통합 연구 상태 문서](../TEMPO_GO_UNIFIED_GOAL_STATE_AND_EXECUTION_PLAN.ko.md)에
정리되어 있다.

## Claim status

| 평가 | 상태 | 해석 범위 |
|---|---|---|
| C9 fresh 4-node held-out | **independent positive** | whole-system headline claim |
| C10 NetKV/Kairos-X512 | **actual-system post-hoc positive** | independent SOTA claim 아님 |
| Kairos | **`X={512}` subset** | full dynamic-chunk implementation 아님 |
| 2–1,024 pair hierarchy | **CPU control-plane receipt** | native scale/goodput claim 아님 |

## 핵심 결과

| Policy | normal SLO / p99 | miss-hot SLO / p99 | remote-favorable SLO / p99 |
|---|---:|---:|---:|
| **TEMPO** | **60/60 / 3.150s** | **120/120 / 3.337s** | **30/30 / 3.357s** |
| strongest fixed | 60/60 / 3.156s | 97/120 / 9.849s | 0/30 / 35.508s |
| predictor | 60/60 / 3.104s | 100/120 / 8.709s | 13/30 / 50.557s |
| queue-GPU | 60/60 / 3.088s | 116/120 / 8.036s | 13/30 / 51.973s |
| Kairos `X={512}` | 30/60 / 3.223s | 0/120 / undefined | 1/30 / 4.853s |
| NetKV reproduction | 60/60 / 3.300s | 73/120 / 13.780s | 0/30 / 53.914s |

TEMPO의 stressed p99 감소율은 strongest fixed 대비 66.12–90.55%, predictor
대비 61.69–93.36%, queue-GPU 대비 58.48–93.54%다. C10 NetKV 대비
miss-hot/remote-favorable p99은 75.79%/93.77% 낮다.

## 정량 그래프

### C9 independent seven-arm evaluation

![C9 independent performance](figures/c9_independent_performance.svg)

### C9 P×D mesh actuation

![C9 mesh actuation](figures/c9_mesh_actuation.svg)

### Background fairness와 controller overhead

![C9 fairness telemetry](figures/c9_fairness_telemetry.svg)

### Actual NetKV/Kairos paper-policy extension

![C10 paper comparison](figures/c10_paper_policy_comparison.svg)

### 2–1,024 logical-pair hierarchy receipt

![Hierarchy control plane](figures/hierarchy_control_plane_scale.svg)

그래프는 [`render_readme_figures.py`](render_readme_figures.py)가 committed JSON을
읽어 생성한다. 숫자를 plotting source에 다시 입력하지 않는다.

```bash
.vllm_venv/bin/python paper/tempo_go/render_readme_figures.py
jq . paper/tempo_go/figures/manifest.json
```

[`figures/manifest.json`](figures/manifest.json)은 source JSON 12개와 SVG 5개의
SHA-256을 고정한다.

## Background와 telemetry gate

| 지표 | 관측값 | gate |
|---|---:|---:|
| background completion | 85.755% | ≥80% |
| minimum block/tenant completion | 76.496% | ≥70% |
| Jain fairness | 0.997873 | ≥0.99 |
| service-lane failure | 0.926% | ≤1% |
| telemetry collection p50/p99 | 28.62 / 132.42 ms | ≤50 / 250 ms |
| admission wait p50/p99 | 29.46 / 133.26 ms | ≤50 / 250 ms |

Cassini required endpoint signal은 29/30 decision, LMCache semantic/byte inflight는
30/30 decision에서 supported였다. `nccl_collective_p99_ms`,
`nccl_arrival_spread_ms`, `lmcache_transfer_p99_ms`는 explicit unsupported이며
0으로 대체하지 않았다.

## Paper

- [`main.tex`](main.tex): 2-column systems paper source
- [`references.bib`](references.bib): primary-source bibliography
- [`main.pdf`](main.pdf): compiled seven-page paper
- [`artifact_manifest.json`](artifact_manifest.json): headline claim과 SHA index

빌드:

```bash
cd paper/tempo_go
module load texlive/2024
pdflatex -halt-on-error -interaction=nonstopmode main.tex
bibtex main
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
```

현재 PDF SHA-256은
`3e35c65a92230ceef4576bff1ac5aa7ed42a33b82d76ab57a2d5cb2e3877f60f`다.
최종 build는 LaTeX error, unresolved citation/reference, overfull box 0으로
통과했다.

## Authoritative evidence

| 범위 | contract/analysis |
|---|---|
| C9 independent | `eval/sota_4node/tempo_go_c8_independent_validation_contract_v3.json` / `results/tempo_go_c8_independent_validation_job_57586612_v3/analysis.json` |
| C10 paper policy | `eval/sota_4node/tempo_go_c10_paper_sota_analysis_contract_v4.json` / `results/tempo_go_c10_paper_sota_job_57586612_v3/analysis.json` |
| hierarchy scale | `eval/sota_4node/tempo_go_hierarchy_scale_contract_20260825.json` / `results/tempo_go_hierarchy_scale_20260825_c9_c10_r15.json` |

Git artifact에는 aggregate result와 compact analysis를 포함한다. 전체 raw request/log
tree는 Perlmutter에 보존하며, compact analysis가 raw path와 SHA를 고정한다.

## Verification

- C9/current regression: 294 passed, 2 historical source-drift checks deselected,
  28 subtests passed
- C10 process-entry regression: 6 passed
- telemetry/endpoint regression: 93 passed
- bounded runtime import closure: 91 files `py_compile` passed
- canonical shell launchers: `bash -n` passed
- C9 frozen source inventory: 39/39 SHA match
- C10 v3→v4: performance source는 유지하고 analyzer만 명시적 analysis-only
  successor로 변경

## 남은 hard gate

1. 새 allocation에서 C10 source-unchanged repeat
2. full Kairos dynamic chunk candidate set 구현 또는 subset 표기 유지
3. NCCL/LMCache latency signal을 supported로 만든 causal ablation
4. 4노드보다 큰 native inference scale

이 gate 전에는 full Kairos 독립 우위, production-scale superiority, universal SOTA를
주장하지 않는다.
