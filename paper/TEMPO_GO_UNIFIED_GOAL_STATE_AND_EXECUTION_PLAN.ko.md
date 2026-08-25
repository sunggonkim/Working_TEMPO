# TEMPO-GO 단일 연구 목표·현재 결론·향후 실행계획

문서 버전: `unified-goal-v8-c9-independent-c10-sota`
기준일: 2026-08-25
대상 시스템: NERSC Perlmutter 전체 규모를 겨냥한 TEMPO-GO cross-layer global orchestrator
현재 native testbed: 4 nodes / 16 A100 GPUs / 실제 vLLM P/D / frozen official `LMCacheConnectorV1:UCX` 경로

최신 continuation 기준은 **§72의 C9 fresh held-out independent validation과
C10 actual paper-policy comparison**이다. allocation `57586612`의 4-node/16-A100
actual vLLM/official LMCache/NIXL-CXI run에서 TEMPO는 normal 60/60, miss-hot
120/120, remote-favorable 30/30 SLO-good을 달성했고 p99은 각각
3.150/3.337/3.357초였다. strongest fixed 대비 stressed p99은 66.12%/90.55%,
predictor 대비 61.69%/93.36%, queue-GPU 대비 58.48%/93.54% 감소했다.
fresh-allocation, one-shot, fairness, telemetry-overhead와 independent-positive gate가
모두 true이며 C9 independent performance claim은 허용된다. 같은 allocation의 C10
post-hoc extension에서 NetKV Algorithm-1 reproduction 대비 miss-hot/remote-favorable
p99을 75.79%/93.77% 줄였고 SLO-good을 73→120, 0→30으로 높였다. Kairos는
공개 코드 부재와 stock vLLM 경계 때문에 `X={512}` reproduction으로만 보고하며,
C10 SOTA-independent claim은 fresh unchanged rerun 전까지 금지한다. current-source
hierarchy benchmark는 1,024 logical pair에서 global payload를 87.499% 줄였지만 CPU
control-plane evidence일 뿐 native production-scale win은 아니다. v0–v107과 모든
failure receipt는 lineage/history로 보존한다. 최신 authoritative state는 문서 마지막
§72를 먼저 읽는다.

이 문서는 사용자의 요청에 따라 지금까지 흩어진 TEMPO 문서, 원본 목표, C4/C5 native receipt, 최신 Candidate G/I CPU audit와 향후 계획을 하나의 진입점으로 정리한 문서다. 과거 raw, profile, contract와 문서를 삭제하거나 덮어쓰지 않는다. 숫자나 결론이 충돌하면 다음 우선순위를 사용한다.

1. SHA-bound raw/result/failure receipt와 machine-check audit
2. 이 문서의 현재 상태·목표·stop/go 해석
3. `TEMPO_RESEARCH_HANDOFF_AND_IMPROVEMENT_GOAL.ko.md`의 Candidate I/strict audit 절과 canonical playbook의 마지막 addendum
4. canonical 본문, master, contention audit와 TEMPO-RD historical 문서

이 문서가 연구의 mission을 임의로 바꾸지는 않는다. 다만 이미 실패한 route-only/queue-threshold 후보를 성공 후보처럼 되살리지 않고, native result와 CPU replay를 분리하며, 앞으로 무엇을 해야 positive systems result가 될 수 있는지를 명시한다.

중요하게, 이 문서는 TEMPO의 가치를 기존 논문에 없는 작은 기능의 교집합으로 정의하지 않는다. **TEMPO의 연구 단위는 admission, routing, telemetry 같은 개별 부품이 아니라, Perlmutter급 shared HPC에서 vLLM service, LMCache/UCX transfer, NCCL collective, GPU/NVLink/PCIe, Cassini/Slingshot와 tenant business state를 하나의 관측·판단·실행 폐루프로 연결하는 global orchestration system 전체다.** 각 부품의 선행연구는 이 end-to-end systems contribution을 없애지 않는다. 오히려 이질적인 subsystem을 scale, correctness, stability와 실제 성능까지 닫는 것이 핵심 구현·연구 난제다.

---

## 0. 한 화면 결론

| 질문 | 현재 답 |
|---|---|
| 현실적인 동시 추론 부하에서 병목이 생기는가? | **그렇다.** local decoder service와 remote P/KV/receiver completion이 서로 다른 상태에서 무너지고 C3에서 승자가 뒤집혔다. |
| remote가 항상 나쁜가? | **아니다.** C1과 C3 rate 0에서는 remote가 local보다 유리했고, 최신 native epoch에서는 official remote가 가장 높은 request/output-token goodput을 냈다. |
| local이면 contention을 피하는가? | **아니다.** local prefill과 decode가 같은 decoder GPU를 공유해 active service와 TPOT을 악화시킨다. visible queue가 0이어도 service inflation이 존재했다. |
| global orchestrator가 필요한가? | **그렇다.** route, decoder, P/KV transfer, NCCL, Slingshot endpoint, tenant SLO와 pair capacity를 국소적으로 최적화하면 병목을 다른 layer로 밀어낸다. cross-layer global loop가 TEMPO의 핵심 가치다. |
| 현재 TEMPO-GO가 strongest fixed/predictor보다 빠른가? | **그렇다.** C9 fresh held-out 4-node run에서 strongest fixed 대비 miss-hot/remote-favorable p99 66.12%/90.55%, predictor 대비 61.69%/93.36% 감소했고 세 regime 모두 offered victim 100% SLO-good이다. |
| 현재 global candidate를 native로 더 돌려야 하는가? | **C9 tuning 반복은 STOP이다.** 다음 native 성능 실행은 unchanged C10 SOTA independent rerun, larger native rung 또는 새로운 failure/scale hypothesis에만 사용한다. |
| native 성능이 음성으로 증명됐는가? | **아니다.** v107과 여러 초기 family의 bounded negative는 보존되지만 C9 whole-system candidate는 fresh independent native positive다. 이것을 4-node보다 큰 production-scale superiority로 확대하지 않는다. |
| 연구를 폐기해야 하는가? | **아니다.** actual vLLM/LMCache/Slingshot/business 공동 제어의 4-node value가 검증됐다. 남은 일은 full Kairos/NetKV independent comparison, native scale와 paper artifact를 닫는 것이다. |
| 지금 4노드 GPU를 왜 쓰는가? | **새 threshold를 찾기 위해서가 아니다.** frozen whole-system result의 independent replication, SOTA policy reproduction, failure robustness와 larger-rung mechanism을 검증하는 데만 쓴다. |

### v26/v28 same-allocation discovery 판정

동일한 Slurm allocation `57415765`에서 4노드/16 GPU, official `LMCacheConnectorV1:UCX`, 8-rank NCCL+LMCache co-job을 유지한 seven-arm discovery를 닫았다. v26은 TEMPO가 130/276을 완료하고 146개를 reject했으며, `global_admission_queue_timeout`/`global_telemetry_refresh_timeout`이 주요 terminal reason이었다. 이 결과는 공동 부하에서 global admission이 실제로 발동한다는 integration evidence였지만 utility claim은 허용되지 않았다.

v26을 원인 분석한 뒤 v28에서 cross-layer actuation을 safe-envelope 초과분에만 적용하도록 수정하고, global queue wait cap을 8 s profile로 고정했다. v28은 co-job 10,000 blocks, 100% correctness, C5 측정 종료시점 coverage를 만족했다. 그러나 성능 gate는 아직 실패했다.

| v28 arm | 완료/제공 | E2E p50 / p99 (ms) | output-token goodput (/s) | 판정 |
|---|---:|---:|---:|---|
| local | 276/276 | 11,581 / 35,080 | 522.6 | clean fixed |
| remote | 276/276 | 8,970 / 36,022 | 570.7 | clean fixed |
| predictor | 276/276 | 11,776 / 37,597 | 502.0 | predictor baseline |
| queue_gpu | 276/276 | 12,709 / 33,673 | 616.8 | strongest fixed in this slice |
| network-request-only | 276/276 | 11,285 / 34,514 | 530.8 | ablation |
| app-global-only | 146/276 | 6,820 / 11,712 | 647.6 | reject-heavy global ablation |
| TEMPO v28 | 136/276 | 7,211 / 12,616 | 638.6 | app-only보다 1.4% 낮고 p99 약 7.7% 높음; gate 실패 |

따라서 v28은 TEMPO가 문제를 해결했다고 주장하는 결과가 아니다. 다만 v26 대비 TEMPO goodput은 557.5/s에서 638.6/s로 상승했고, cross-layer plan은 실제 NCCL p99·LMCache tail·Cassini vector를 provenance에 보존한 채 local/remote route와 joint limit을 바꿨다. business 관점에서는 starvation은 0이고 latency/interactive/batch/background가 각각 10/12, 11/12, 10/12, 105/240을 완료했지만, app-only의 10/12, 10/12, 11/12, 115/240보다 전체 서비스가 좋다고 말할 수 없다. 현재 병목은 signal 수집이 아니라, coupled signal을 과도한 reject 없이 tenant utility와 pair capacity로 변환하는 global actuation/calibration이다.

이 병목을 threshold 숫자 조정으로 덮지 않기 위해 v29에서 actuation mechanism을 바꿨다. `soft_shadow_price_v2`는 비임계 LMCache/NCCL pressure를 즉시 hard reject하지 않고 resource-specific overage penalty와 enforced lease로 global objective에 넣으며, Cassini retry/timeout·ECN·pause와 transport failure pressure는 `critical_guard`로 여전히 hard guard한다. 이 변경은 CPU에서 관련 테스트 83개를 통과했다. profile은 `results/tempo_go_c5_cross_layer_short_slice_v4_profiles/real_tempo_go_profile_short_slice_v3_soft_shadow_price.json`(SHA `bb7a31b2b7e9badf2a98d6620cbfd526242cbbf7e94ba158d22fe32b256bc877`)이고, source-bound v29b contract는 `results/tempo_go_c5_cross_layer_contract_v29b/native_run_contract.json`(SHA `af8ca46422fd3edf11eaf3a6d57c6de957b70311f7a8873513f87b8888aefb04`, fingerprint `1889e46442d9370a0d79ac907446a5fb606b723c8acce550658dd15606a899e6`)다.

v29f native attempt는 승인된 4-node/16-GPU allocation `57415765`에서 실제 vLLM TP4·LMCache/NIXL/UCX 초기화까지 도달했지만 첫 request/observer-bound C5 측정 전에 `exit 143`으로 종료됐다. `results/tempo_go_cross_layer_native_57415765_v29f/tempo/failure.json`(SHA `14d16889d5c14d0934ca26b910b81c2ebb3c6fcb6563d8c4f24780aa3a080351`)에 native-only, 4 nodes/16 GPUs, `LMCacheConnectorV1:UCX`, v29b contract identity를 고정했다. 따라서 v29f는 TEMPO 성능 음성도 양성도 아니며, controller tuning의 근거로 사용하지 않는다. v30 campaign contract는 resource-envelope spare-pair activation을 포함해 실행 전 `verify`를 통과했고, allocation `57423440`에서 native seven-arm/co-job coverage까지 닫혔다. 다만 현재 worktree에는 campaign 이후 별도 source 변경이 있어 v30 contract를 현재 소스 contract로 재사용하지 않는다. v31 shared-budget/stagger/pair-local-failure actuator와 CPU fan-in gate는 이제 구현·verify됐고, 다음 native 단계는 v31b contract를 승인된 4-node interactive allocation에서 matched discovery로 실행하는 것이다.

가장 정확한 현재 상태는 다음 한 문장이다.

> **TEMPO-GO의 문제와 native control-plane integration은 실재한다. 현재 application-only route/admission/reservation/circuit branch만 reproducible negative로 닫혔다. 다음 목표는 NCCL collective, Cassini/Slingshot, LMCache/UCX, vLLM scheduler와 business SLO를 같은 global state/actuation loop로 묶어 병목을 layer 사이에서 이동시키지 않고 전체 utility를 높이는 TEMPO-GO를 구현하는 것이다.**

---

## 1. 변하지 않는 mission과 현재 달성 상태

### 1.1 사용자가 처음 고정한 원본 목표

원본 attachment의 목표는 다음이다.

> **TEMPO Elastic-PD를 실제 vLLM P/D 경로에 통합하고, 단순 predictor와 가장 강한 고정 정책보다 유의미하게 빠른 하나의 최종 스킴으로 확정한다.**

원본 목표 attachment의 SHA-256은 `4f9650280307d6c352ada284b1fb7137e4f70c0189e6ad341104671cc1647a4a`다. 원본에는 positive win뿐 아니라 조건을 만족할 수 없다는 재현 가능한 negative conclusion도 완료 정의로 포함돼 있었다. 따라서 현재 negative 판정은 목표를 몰래 바꾼 것이 아니지만, positive mission이 달성됐다는 뜻도 아니다.

### 1.2 route selector가 아니라 global orchestrator로 구체화된 이유

원래의 request-start local/remote 선택은 최종 시스템의 한 decision일 뿐이다. C4 결과는 선택 자체가 유용한 request가 있어도 shared decoder TPOT과 worst tail을 제어하지 못한다는 것을 보였다. 최종 TEMPO-GO의 제어 범위는 다음 다섯 축이다.

1. decoder admission과 prefill/decode externality
2. tenant별 SLO, minimum service, fairness와 explicit defer/reject
3. P/D pair assignment와 prewarmed pair의 logical activation
4. local prefill과 official remote P/KV/receiver route commit
5. endpoint completion, failure quarantine, survivor capacity와 explicit probe recovery

연구 질문은 다음과 같다.

> Perlmutter급 shared HPC에서 TEMPO-GO가 vLLM queue/active service/KV state, LMCache/UCX transfer·receiver completion, NCCL collective progress/latency/health, GPU/NVLink/PCIe pressure, per-NIC Cassini/Slingshot counter vector, topology와 tenant business SLO를 일관된 multi-timescale global state로 결합하고, admission/defer/reject, P/D placement와 pair scaling, local/remote route, transfer/semantic concurrency, workload staggering과 failure recovery를 공동 제어하여 strongest fixed, predictor-only, queue/GPU-only와 network-aware request-local policy보다 offered-population SLO-goodput·tail·fairness·failure utility를 개선하는가?

이 질문의 novelty 단위는 센서 하나나 route rule 하나가 아니다. 다음을 동시에 만족하는 전체 시스템이다.

1. **cross-layer state plane**: application, communication library, GPU/node와 fabric endpoint 신호를 provenance·freshness·uncertainty와 함께 보존한다.
2. **global causal resource model**: 어느 신호가 높다는 이유로 즉시 route하지 않고, action별 completion-rate 변화와 병목 이동을 학습·검증한다.
3. **business-aware joint actuation**: request, pair, transfer와 failure decision을 tenant utility 아래 한 transaction으로 commit한다.
4. **hierarchical scale**: node agent → pair agent → sharded/global coordinator로 telemetry fan-in과 decision overhead를 bounded하게 만든다.
5. **actual-system evaluation**: real vLLM/LMCache/NCCL/Slingshot path에서 correctness, stability, overhead와 performance를 함께 증명한다.

### 1.3 상태를 계층별로 분리한다

| 층 | 상태 | 의미 |
|---|---|---|
| 문제/motivation | `SUPPORTED` | moving bottleneck, opposite crossover, remote service knee, LMCache/EngineCore failure와 business overload가 실제 native에서 관찰됐다. |
| 현재 구현/integration | `SUPPORTED WITH BOUNDARY` | global admission, reject, scheduler/endpoint receipt, pair activation, explicit failure/quarantine lifecycle가 구현·관찰됐다. production utility는 아니다. |
| positive performance mission | `NOT ACHIEVED` | independent native validation win이 없고 G/I는 CPU promotion gate에서 탈락했다. |
| application-only G/I branch | `COMPLETE: CPU NEGATIVE ONLY` | 두 구조적으로 다른 route/admission 후보의 frozen pre-native negative가 machine-check됐다. |
| cross-layer state plane | `IMPLEMENTED: LIVE OBSERVER + JOINT ACTUATION PATH; NATIVE UTILITY OPEN` | `PairTelemetry.cross_layer`와 `tempo-go-cross-layer-envelope-v1`가 vLLM/LMCache endpoint batch에 연결됐고, Cassini per-NIC/TC vector·support·topology/epoch·derived route externality가 immutable decision provenance에 들어간다. block별 atomic `tempo-nccl-observer-v1` snapshot이 이제 `JointActuationPlan`의 독립적인 local/remote prefill·KV·semantic-op limit와 bounded stagger를 만들고, global admission과 pair router가 같은 plan을 enforce한다. 4-node vLLM/co-job에서 coupled utility를 개선하는 것은 아직 검증 전이다. |
| 본래 TEMPO cross-layer campaign | `OPEN` | NCCL/Cassini/LMCache/vLLM/business를 결합한 hierarchical global candidate는 아직 만들어지지 않았으므로 G/I negative로 닫을 수 없다. |
| 다음 native action | `GO FOR C6 WORKLOAD/ACTION QUALIFICATION` | synthetic p07을 headline에서 내리고 actual LMCache/NCCL victim slowdown과 opposite crossover를 먼저 고정한다. 그 뒤 P×D mesh, receiver completion credit, business admission/staggering을 하나의 frozen candidate로 검증한다. |

---

## 2. 왜 지금까지 win이 안 나왔는가

현재 정체의 원인은 “부하를 못 만들었다” 하나도 아니고 “LMCache가 원래 느리다” 하나도 아니다. 확인된 원인은 다음과 같다.

### 2.1 route-only는 shared decoder를 제어하지 못한다

LOCAL과 REMOTE 모두 first-token handoff 이후 같은 decoder에서 decode한다. local/remote route가 foreground TTFT를 개선해도 이미 decoding 중인 request의 TPOT을 악화시킬 수 있다. C4 Candidate B/C와 hidden phase oracle이 median을 개선하면서도 TPOT p99와 worst paired tail을 실패한 핵심 이유다.

### 2.2 queue depth가 service pressure를 대표하지 않는다

C1 decoder-hot에서 visible D queue mean은 약 0.007–0.008 ms/request였지만 local prefill mean은 192–213 ms, inference mean은 378–418 ms였다. continuous batching은 FIFO queue를 만들지 않고도 active service를 팽창시킨다. remote P/D queue gauge도 거의 0인데 client-visible completion residual은 수초로 늘었다. 따라서 queue/GPU heartbeat만으로는 충분하지 않다.

### 2.3 현재 admission은 capacity를 만들지 않고 과도하게 shed했다

최신 source-rebound native TEMPO의 bounded coordinator queue p99는 0.506 ms로 작았지만 background admission wait p99는 5.072 s였다. 5초 budget과 tenant reservation을 넣고도 queue-timeout 822건, tenant-reservation reject 897건, telemetry-refresh timeout 11건으로 총 1,730건을 reject했다. completed-only E2E가 낮은 것은 빠른 subset만 완료한 결과이며 전체 offered population utility가 아니다.

Candidate E/G는 더 많은 request를 admit하거나 tenant queue slot을 보호했지만 service capacity를 늘리지 않아 tail과 background SLO-goodput을 악화시켰다. H의 proactive activation은 assignment를 바꿨지만 aggregate service를 바꾸지 못했다. I의 survivor reserve는 failure containment을 발동시켰지만 complete를 912개로 낮추고 p99를 늘렸다.

### 2.4 logical pair activation과 실제 incremental service는 다르다

pair1을 더 일찍 선택했다는 사실만으로 throughput이 늘지는 않는다. 다음 조건이 필요하다.

- pair1이 실제로 fresh endpoint capacity를 갖는가
- 필요한 cache residency와 P/D receiver state가 준비됐는가
- shared decoder 또는 remote endpoint의 dominant bottleneck을 정말 피하는가
- activation cost보다 완료된 SLO work가 더 늘어나는가

현재 CPU replay는 duration/profile 기반 control-plane replay라 이 native service 증가를 증명할 수 없다.

### 2.5 endpoint profile과 native state의 경계가 남아 있다

held-out frozen endpoint source는 17개 row가 모두 P_ONLY인 calibration evidence다. `FrozenServiceProxyPolicy`가 MISS remote를 열지 않고 provenance를 닫았지만, proxy는 exact MISS native service가 아니며 performance claim을 금지한다. CPU replay latency는 actual GPU/fabric/LMCache latency가 아니다.

### 2.6 official remote path는 run-to-run state와 failure에 민감하다

한 native epoch에서는 official remote가 2,712/2,712 완료하고 strongest fixed가 됐다. 다른 승인 allocation에서는 `CacheEngineKey ... not found in local data`, receiver allocation timeout, `EngineDeadError`, HTTP 500/502/503이 발생했다. 이 variability는 remote가 항상 나쁘다는 증거가 아니라 endpoint state/failure를 orchestration state로 다뤄야 한다는 증거다.

### 2.7 4노드는 TEMPO의 가치 상한이 아니라 첫 native scale rung이다

[Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/)는 1,792 GPU nodes, node당 4 A100과 4 Cassini NIC, 3-hop dragonfly Slingshot 11을 가진다. 현재 4-node allocation은 이 시스템의 최소 production-faithful slice다. 여기서 2개의 TP4 P/D pair, 또는 cross-node TP/NCCL topology를 사용해 actual data-plane coupling과 controller mechanism을 검증한다. 그 결과만으로 1,792-node superiority를 주장할 수는 없지만, 반대로 4-node limit가 cross-layer architecture의 연구 가치를 줄이지도 않는다.

scale evidence는 세 rung으로 만든다.

1. **native 4-node mechanism**: 실제 vLLM, LMCache/UCX, NCCL와 Cassini signal/actuator의 causal 효과
2. **many-pair control-plane scale**: trace/replay 또는 endpoint emulator에서 수십~수천 pair fan-in, sharding, stale/failure storm과 control overhead
3. **larger field validation**: 별도 allocation/운영 권한이 있을 때 topology group과 실제 concurrent jobs를 늘린 end-to-end validation

---

## 3. 이번 통합에서 읽고 대조한 자료

이번 통합은 shared filesystem을 넓게 순회하지 않고 사용자가 지정한 repository와 exact attachment/result path만 읽었다.

### 3.1 `paper/` Markdown 전체

| 문서 | 역할 | 현재 사용법 |
|---|---|---|
| `README.md`, `STATUS.md` | 초기 TEMPO-RD checkpoint 연구 상태 | negative 결과와 함께 FSDP/NCCL/phase/fabric instrumentation 자산의 출처로 사용 |
| `TEMPO_RD_SCHEDULER_STOP_DECISION.md` | matched optimized-open 대비 scheduler stop | 특정 checkpoint policy는 stop하되 cross-layer observation/control 자산은 보존 |
| `TEMPO_LATEST_LITERATURE_REAUDIT.md` | training/checkpoint literature audit | TEMPO가 inference와 co-running NCCL/I/O를 함께 다룰 때의 direct boundary와 workload 설계에 사용 |
| `TEMPO_ELASTIC_PD_CONTENTION_AUDIT.md` | v0–v544 및 C1–C4 terminal audit | route-only history와 contention evidence |
| `TEMPO_RESEARCH_MASTER_STATE_AND_NEXT_GOAL.ko.md` | C5 중간 master | held-out/native 이전 시점도 있어 최신 delta로 덮어 읽음 |
| `TEMPO_RESEARCH_HANDOFF_AND_IMPROVEMENT_GOAL.ko.md` | 최신 handoff-v10 | Candidate I와 strict audit가 포함된 직접 선행 문서 |
| `TEMPO_GLOBAL_ORCHESTRATOR_CANONICAL_PLAYBOOK.ko.md` | 상세 계보·설계·실행 log | 본문은 누적 history, 마지막 addendum를 최신 delta로 사용 |
| `tempo_pd_c4_negative_report_v1/negative_conclusion_report.md` | C4 SHA-bound compact negative | C4 route-only stop의 compact evidence |

세 attachment는 모두 224줄이며 같은 SHA `4f9650…a4a`의 원본 목표 사본이었다.

### 3.2 lineage를 해석하는 규칙

- conceptual v0는 `_v0` 파일 하나가 아니라 actual P/D 이전 TEMPO/TEMPO-RD 세대다.
- actual checked P/D source lineage는 v1–v450이다.
- v452–v544b는 주로 result root, launcher, analyzer와 campaign revision이다.
- 현재 bounded evidence에는 v545–v600의 연속된 canonical source/artifact가 확인되지 않았다.
- 없는 revision을 추정해서 채우지 않는다. 추가 자료가 있으면 exact path와 SHA로 이 문서에 붙인다.

이 문서는 이전 bounded audit가 이미 읽은 v1–v450 source와 v452–v544 raw lineage를 요약한다. 이번 문서 작업에서 수백 개 raw를 다시 unbounded traversal한 것은 아니다.

### 3.3 과거 세대에서 이미 만든 cross-layer 자산

이전 연구는 버릴 실패 묶음이 아니다. 현재 global orchestrator에 재사용할 수 있는 구현 자산이 이미 있다.

| 자산 | exact source | TEMPO-GO에서의 역할 |
|---|---|---|
| Cassini endpoint multi-signal sampler | `tempo/cassini_endpoint.py` | 4 NIC × 8 traffic-class의 explicit paths를 읽고 endpoint-level max/mean/count와 함께 NIC별·TC별 RX/TX pause vector, support/ambiguity를 출력 |
| early scalar sampler | `tempo/cassini_pressure.py` | scalar collapse가 왜 실패했는지 보여주는 historical ablation; 새 policy 입력으로 재사용하지 않음 |
| NCCL/CUDA collective observer | `eval/sota_4node/train.py::CudaCollectiveObserver` | collective type/bytes, CUDA elapsed time, enqueue/completion과 phase별 HSN byte delta를 수집하는 기존 instrumentation |
| P/D endpoint probe | `eval/sota_4node/tempo_pd_endpoint_probe.py` | actual P/D endpoint identity에 Cassini sampler를 붙인 native probe |
| LMCache×NCCL contention harness | `eval/sota_4node/run_lmcache_nixl_contention_2node.py` | official KV movement와 real NCCL collective의 동시 간섭을 재현하는 workload substrate |
| current application telemetry | `tempo/pd_global_telemetry.py` | vLLM scheduler, endpoint completion/failure와 controller ledger의 atomic all-pair batch |

현재 구현은 이 자산을 provenance-safe endpoint/pair state plane과 global route/pair score에 연결했고, live vector에서 독립 resource limit/stagger를 생성해 global admission과 endpoint controller가 공동 집행한다. 남은 핵심 gap은 **실제 native C5 co-job에서 이 plan이 물리 completion과 offered-population utility를 개선하는지, 그리고 node→pair→shard scale path를 닫는 것**이다. 다음 단계는 threshold variant가 아니라 이 joint action의 causal value를 검증하는 일이다.

---

## 4. v0부터 현재까지의 연구 계보

| 세대 | 실제로 시도한 것 | 남은 지식 | 현재 운명 |
|---|---|---|---|
| conceptual v0–v4 | phase-gated I/O, topology/QoS, sparse transfer, look-ahead | cross-layer control idea, collective observer와 fabric instrumentation; causal path와 matched baseline 필요 | 특정 policy는 폐기, instrumentation과 global thesis 보존 |
| TEMPO-RD reset | checkpoint resource-domain scheduler | optimized-open이 강했고 해당 scheduler는 회귀했지만 NCCL/collective/fabric timing을 actual workload에 심는 구현 확보 | scheduler만 stop, cross-workload sensor/actuator 자산 보존 |
| P/D v1–v27 | actual vLLM P/D, LMCache/NIXL, KV geometry | admission과 transport 분리, remote branch 실제 검증 필요 | substrate evidence |
| v28–v60 | offered-rate crossover, local credit | remote spill이 D-local queue를 완화해 mixed route가 두 fixed를 이긴 사례 | crossover evidence |
| v61–v129 | threshold, hysteresis, interleaving, heterogeneous geometry | trace-derived threshold와 sequential arm drift는 일반화되지 않음 | 실험 규칙으로 전환 |
| v131–v245 | cache catalog, warm/cold, saturation, tail | cache residency는 first-class constraint, aggregate median만으로 부족 | invariant/evidence |
| v248–v349 | overload, unique namespace, burst, credits | LMCache failure와 performance 분리, silent fallback 금지 | invariant/evidence |
| v353–v430 | phase change, adaptive caps | cap tuning은 worst-tail을 안정시키지 못함 | threshold family stop |
| v440–v450 | canonical Elastic-PD, native comparison | one-way commit, first-response release, exact route/cache ledger는 sound | TEMPO-GO component로 보존 |
| v452–v544b | profile/pair/credit/cache/CXI background discovery | v492 일회성 positive는 repeat에서 미재현, moving bottleneck 확인 | terminal discovery evidence |
| C1/C2/P_ONLY/C3 | actual independent inference contention | opposite crossover, remote service knee, queue-only sensor 한계 | final motivation |
| C4 | route-only Candidates A/B/C + phase oracle | route choice는 유용해도 shared decoder tail bundle 실패 | route-only terminal negative |
| C5 TEMPO-GO | application-visible global admission, tenant contract, pair activation, endpoint telemetry, failure receipt | native wiring과 failure containment 발동, utility win 없음 | application-only branch evidence |
| Candidate G/I audit | queue reservation vs endpoint telemetry circuit/survivor reserve | 구조적으로 달라도 frozen CPU promotion gate 실패 | 이 두 candidate만 pre-native negative completion |
| cross-layer TEMPO-GO | vLLM + LMCache/UCX + NCCL + GPU/topology + Cassini/Slingshot + business joint loop | 핵심 자산은 흩어져 있으나 global state/decision에는 미통합 | **현재의 본 연구 목표** |

---

## 5. 실험으로 확인된 problem evidence

### 5.1 C1/C2 opposite crossover

실제 native 4-node vLLM P/D, Qwen2.5-7B, official LMCache UCX, 4094/2 unique-cold workload의 first valid fraction 0.70에서 다음 방향이 재현됐다.

| state | background | required winner | pooled gain | paired gain |
|---|---:|---|---:|---:|
| C1 decoder-local hot | local-pinned 22.4 req/s | remote | +19.21% | +17.53% |
| C2 remote P/KV/D hot | remote-pinned 4.76 req/s | local | +73.02% | +72.54% |

1,868개 actual request가 HTTP 200, exact route, output과 cache contract를 통과했다. 이것은 controller win이 아니라 workload validity evidence다.

### 5.2 P_ONLY remote service knee

4094/2 P_ONLY prompts를 measurement 밖에서 preseed한 actual LMCache path의 결과다.

| offered rate | remote FG median | local FG median | achieved remote rate |
|---:|---:|---:|---:|
| 4/s | 449.6 ms | 141.9 ms | 3.86/s |
| 8/s | 656.3 ms | 142.8 ms | 7.64/s |
| 12/s | 1,941.1 ms | 139.0 ms | 8.37/s |
| 16/s | 3,720.4 ms | 146.8 ms | 8.90/s |
| 24/s | 5,865.1 ms | 139.5 ms | 9.71/s |
| 32/s | 5,417.3 ms | 142.8 ms | 9.73/s |

12/s에서 첫 2× inflation과 drain이 시작되고 achieved throughput은 약 9.7/s에서 포화됐다. 허용되는 표현은 “official LMCache retrieval/transfer/control/install/receiver completion service ceiling”이다. 특정 Slingshot switch/link가 병목이라고 단정하지 않는다.

### 5.3 C3 coupled crossover

C1 local-pinned 22.4/s와 P_ONLY remote tenant를 동시에 건 상태다.

| P_ONLY remote background | local FG median | remote FG median | winner |
|---:|---:|---:|---|
| 0/s | 528.9 ms | 448.4 ms | remote |
| 4/s | 542.8 ms | 589.1 ms | local |
| 8/s | 661.7 ms | 655.3 ms | near tie |
| 12/s | 674.9 ms | 1,832.0 ms | local |

ABBA confirmation에서 rate 0 remote win 18.44%, rate 12 local win 66.14%가 두 replicate 모두 재현됐다. 이것이 TEMPO-GO의 가장 강한 empirical motivation이다.

### 5.4 synthetic CXI는 attribution ablation일 뿐이다

v536/v538 same-allocation 비교에서 100% synthetic CXI background는 no-background 대비 local median을 약 30 ms, remote를 약 181 ms 악화시켰다. remote가 약 151 ms 더 민감했지만 sender, NIC, host/PCIe, switch, semantic operation 중 원인을 분해하지 못했다. 이것은 fabric 정보가 가치 없다는 뜻이 아니라 **device-total signal 하나를 scalar route penalty로 쓴 설계가 부족했다는 뜻**이다. 다음 headline은 synthetic CXI 하나가 아니라 actual inference + real NCCL collective + official LMCache transfer를 함께 실행하고 per-NIC vector, collective completion과 endpoint completion을 action-conditioned하게 대조한다.

### 5.5 C4 route-only terminal negative

| C4 candidate | mechanism | fixed median gain | predictor gain | goodput gain | TPOT p99 regression | worst regression |
|---|---|---:|---:|---:|---:|---:|
| A | instant scalar score | -2.92% | +3.48% | +10.17% | +44.53% | +2,506.4 ms |
| B | pair-local active watermark epoch | +7.10% | +17.46% | +7.67% | +64.28% | +997.9 ms |
| C | route-pinned local external credit | +7.92% | +21.30% | +4.58% | +49.41% | +2,278.7 ms |

세 candidate와 세 phase oracle 모두 full gate를 실패했다. C4 Candidate C의 local/remote 선택은 각각 counterfactual보다 유용했지만 tail bundle이 무너졌다. 따라서 scalar `fabric_pressure`, prompt coefficient, hidden phase classifier와 request-local threshold family는 terminal negative다.

---

## 6. 현재 실제로 구현된 TEMPO-GO

### 6.1 control flow

```text
independent tenant arrivals
        │
        ▼
request-triggered all-pair telemetry batch
  ├─ freshness / sequence / profile / endpoint identity
  ├─ decoder running/waiting/KV observations
  ├─ endpoint completion/failure observations
  └─ missing/partial/mixed generation => fail closed
        │
        ▼
global business/admission transaction
  ├─ tenant SLO, wait, weighted debt, minimum service
  ├─ pair×route candidate feasibility
  ├─ multi-resource capacity and logical pair activation
  ├─ route/pair health, quarantine and survivor reserve
  └─ ADMIT or bounded QUEUE or explicit REJECT
        │ immutable request-start pair×route commit
        ▼
LOCAL decoder prefill OR frozen official LMCache remote P/D
        │
        ├─ first response: endpoint/prefill/KV/semantic credit release
        ├─ HTTP EOF: decoder/active-sequence credit release
        └─ failure: terminal receipt + exact release + quarantine
```

### 6.2 주요 구현 파일

| component | file | 현재 역할 |
|---|---|---|
| global scheduler | `tempo/pd_global_orchestrator.py` | tenant queue, pair×route admission, resource credits, pair activation, failure circuit, lifecycle |
| global telemetry | `tempo/pd_global_telemetry.py` | all-pair batch identity/freshness와 scheduler/endpoint observation |
| request agent | `tempo/pd_global_agent.py` | bounded single-flight request-triggered refresh; background watcher 없음 |
| candidates | `tempo/pd_global_candidates.py` | exact cache/profile evidence 기반 pair×route candidates |
| profile | `tempo/pd_global_profile.py` | frozen identities, tenant policy, service proxy boundary |
| coordinator | `tempo/pd_global_coordinator.py` | async queue/admit/first-response/EOF/failure 연결 |
| native frontend/router | `eval/sota_4node/tempo_pd_elastic_frontend.py`, `tempo_pd_elastic_router.py` | actual vLLM/LMCache request path와 decision/receipt ledger |
| native contract | `eval/sota_4node/tempo_go_c5_run_contract.py` | workload/profile/source/launcher/analyzer/environment immutable binding |
| replay/analyzer | `eval/sota_4node/replay_tempo_go_c5_five_arm.py`, `analyze_tempo_go_c5_five_arm.py` | five-arm control-plane replay와 native receipt analysis |

### 6.3 multi-resource ownership

| resource | route/use | ownership lifetime |
|---|---|---|
| `active_sequences` | local/remote shared decoder | commit → EOF/failure |
| `decode_tokens` | local/remote shared decoder | commit → EOF/failure |
| `endpoint_requests` | local prefill or remote handoff | commit → first response/failure |
| `local_prefill_token_ms` | decoder-local prefill | commit → first response/failure |
| `remote_prefill_token_ms` | P endpoint | commit → first response/failure |
| `remote_kv_bytes` | LMCache transfer | commit → first response/failure |
| `remote_semantic_ops` | retrieve/install operation | commit → first response/failure |

controller-owned와 endpoint-observed 값을 무조건 더하지 않는다. resource contract에 따라 de-duplicate하고 현재 구현처럼 필요한 경우 `max(owned, observed)`를 사용한다.

### 6.4 지켜야 할 request/failure invariant

- pair×route는 upstream start 전에 one-way commit한다.
- prefill 시작 후 route/pair mutation, hidden recompute, silent local fallback은 0이어야 한다.
- 실패한 request를 같은 ID로 재시도하지 않는다.
- `UNKNOWN` cache는 hit가 아니며 fail closed한다.
- first response와 EOF release를 섞지 않는다.
- complete/abort/timeout/failure마다 held work를 exactly once 반환한다.
- failure를 latency sample이나 completion으로 바꾸지 않는다.
- `tempo-go-global-failure-v1` receipt에 request, tenant, pair, route, failure kind/scope, telemetry sequence, released work, `new_request_id_required`와 quarantine를 남긴다.
- recovery는 더 최신 telemetry sequence의 explicit `PROBE`일 때만 허용한다.

### 6.5 Candidate I에서 추가된 구조

- cumulative local/remote endpoint failure counter delta를 new admission 전에 관찰한다.
- default pair scope로 local/remote 두 route를 pre-admission quarantine한다.
- 한 pair가 완전히 격리되면 surviving pair capacity 25%를 low-weight burst가 즉시 소비하지 못하게 한다.
- weight ≥2.0, wait-budget 경계 또는 minimum-service deficit tenant는 reserve를 bypass할 수 있다.
- explicit request failure receipt와 telemetry-observed failure circuit은 분리돼 있다.

이 mechanism은 구현되고 CPU에서 발동했지만 utility win은 만들지 못했다.

### 6.6 현재 구현과 본래 cross-layer 목표 사이의 정확한 gap

현재 `tempo/pd_global_telemetry.py`는 physical-switch label과 future phase를 policy state에서 계속 제외하면서, vLLM scheduler/endpoint completion/failure와 opt-in cross-layer envelope를 atomic batch로 조립한다. Cassini vector는 support/epoch/topology와 함께 decision provenance에 보존되며, scalar Cassini classifier는 재사용하지 않는다. 이것은 최종 architecture의 state-plane 연결부이지, native performance 증거 자체는 아니다.

반면 `tempo/cassini_endpoint.py`는 scalar가 아닌 endpoint-scoped vector를 이미 보존하고, `CudaCollectiveObserver`는 actual NCCL collective의 type/bytes/CUDA completion과 local fabric byte delta를 수집한다. 아직 없는 것은 다음 연결부다.

1. existing NCCL/CUDA observer 또는 frozen co-job snapshot을 같은 pair/agent generation으로 native C5 decision에 공급하는 path
2. LMCache semantic completion·vLLM service·Cassini와 같은 bounded sampling window에서 action-conditioned service envelope를 갱신하는 agent
3. 이 envelope를 business utility, route, pair, concurrency와 동시에 푸는 joint actuator
4. 수십~수천 endpoint에서 중앙 poll bottleneck을 피하는 node→pair→shard aggregation
5. native causal intervention과 independent validation에서 위 경로의 utility를 증명하는 contract

따라서 현재 negative의 정확한 해석은 “global orchestration에 답이 없다”가 아니라 **global이라고 부른 현재 C5가 아직 application control plane 중심이었고, TEMPO가 원래 노린 HPC cross-layer state/actuation loop를 끝까지 구현하지 못했다**는 것이다.

### 6.7 이번 4-node component attribution receipt

승인된 allocation `57412204`의 2-node/8-GPU child step에서 official `third_party/lmcache` `NixlChannel`/NIXL UCX write와 NCCL `all_reduce`를 같은 block interval에 실행했다. 초기 0-byte receipt는 성능 실패가 아니라 두 API contract mismatch였다. (1) current NIXL binding은 index 배열을 요구하는데 Python list를 넘겼고, (2) LMCache wrapper는 GPU address를 descriptor index로 변환하지 않았다. wrapper를 page-aligned descriptor index로 fail-closed 변환하고 source-bound contract에 결박한 후 correctness가 닫혔다.

| matched run | background | NCCL token-tail p99 | background completion p99 | correctness |
|---|---|---:|---:|---|
| `nccl_only_control_v11.json` | 없음 | 8.107 ms | 0 ms | true |
| `nixl_nccl_contention_correct_v12.json` | 4 pairs, official NIXL/UCX, 2×32 MiB/block | 9.245 ms | 244.673 ms | true |

같은 run의 token-tail p99 차이는 `+1.138 ms`다. 이것은 LMCache가 항상 실패한다는 주장이나 TEMPO 성능 승리가 아니다. **shared NCCL/transfer externality가 native에서 측정 가능하고, correctness-preserving 상태로 global controller가 관측해야 하는 causal mechanism이 실재한다는 P2 evidence**다. 이 receipt는 아직 vLLM C5 route/pair/concurrency actuator를 그 신호로 조절한 결과가 아니므로 final cross-layer utility gate를 통과한 것으로 해석하지 않는다.

### 6.8 live observer path를 닫은 현재 delta

component receipt만으로는 C5 global decision이 NCCL/LMCache 상태를 보았다고 할 수 없다. 이를 분리하기 위해 `tempo/cross_layer_observer.py`에 strict `tempo-nccl-observer-v1` snapshot contract와 atomic publisher/reader를 추가했다. `run_lmcache_nixl_contention_2node.py --observer-output PATH`는 각 completed block에서 다음을 발행한다.

- producer `source_epoch`, monotonic `sequence`, producer wall-clock `sampled_unix_ns`, window와 communicator/topology identity
- rank-aggregated NCCL collective token-tail p99와 LMCache transfer p99
- clock-synchronized rank observer가 없으므로 `nccl_arrival_spread_ms`는 `null`로 남김
- `background_mode`, `producer_state`, exact byte/correctness receipt

router는 partial JSON, wrong epoch, stale snapshot, `producer_state=complete`, correctness failure를 모두 `not_collected`로 만들고, 값 0으로 대체하지 않는다. producer snapshot의 Unix timestamp는 freshness 검사에만 쓰며 cross-host monotonic duration을 빼지 않는다. 이 연결은 이제 **implemented mechanism/path**이며, allocation `57416103`의 retry6에서 actual vLLM P/D request lifecycle까지 닫혔다. 아직 다음은 열려 있다.

1. P1PAIR+COJOB의 supported observer state가 실제 route/pair/concurrency action을 바꾸는 범위를 held-out topology에서 재현
2. P2PAIR/TP8-CROSSNODE에서 pair scaling과 interconnect bottleneck migration을 확인
3. app-only/network-request-only/fixed/predictor와 same offered population에서 coupled utility와 fairness 비교

따라서 live observer와 joint actuation path의 구현 완료를 native performance win으로 표현하지 않는다. 현재는 mechanism/integration GO이며, native utility gate는 아직 열려 있다.

### 6.9.1 allocation 57416103에서 실제 vLLM P/D와 cross-layer global loop를 닫은 receipt

`P1PAIR+COJOB`를 같은 4-node/4-hour Perlmutter interactive allocation `57416103`에서 반복 검증했다. inference는 두 노드의 actual vLLM TP4 prefill/decode pair이고, 나머지 두 노드의 8 GPU는 official LMCache/NIXL/UCX transfer와 NCCL `all_reduce` co-job을 수행했다. 모든 실행은 다음 환경을 사용했다.

```text
NCCL_NET=Socket
NCCL_SOCKET_IFNAME=hsn
NCCL_IB_DISABLE=1
UCX_TLS=cuda_ipc,cuda_copy,tcp
UCX_NET_DEVICES=all
UCX_LOG_LEVEL=warn
```

중간 retry들의 의미는 분리한다. 최초 retry는 `MASTER_ADDR/MASTER_PORT` 누락, retry1은 empty warmup input, retry2는 실제 LMCache `meta.address` page-index와 NIXL physical pointer 불일치, retry3/4는 각각 launcher/route-commit receipt 문제였다. retry4에서 정확한 오류는 `TEMPO-GO joint actuation sequence differs`였고, global atomic `PairTelemetry` batch sequence와 cross-layer producer sequence를 joint commit의 같은 필드에 섞은 것이 원인이었다. `tempo/pd_global_orchestrator.py`는 joint commit에는 batch sequence를 사용하고 producer sequence는 provenance에 보존하도록 수정했다. 이 버그 receipt는 `results/tempo_go_p1pair_cojob_57416103_retry4/execution_failure.json`에 고정했다.

retry5는 2,712-row actual workload를 모두 terminal-close했고 2,669 complete/43 explicit global reject/invalid 0을 기록했지만, launcher가 observer path를 inference 환경에 export하지 않아 cross-layer signal integration receipt로 승격하지 않는다. wrapper에 `TEMPO_GO_NCCL_TELEMETRY_PATH=${RESULT_DIR}/nccl-observer.json` export를 추가한 뒤 retry6을 authoritative integration receipt로 삼았다.

retry6의 닫힌 사실은 다음과 같다.

- 2,712 requests 중 2,669 complete, 43은 `tempo_go_global_reject` terminal receipt, 0 invalid; `router_decisions_exact=true`, `terminal_contract_valid=true`, route failure 0이다.
- global coordinator는 1,414 refresh와 installed telemetry sequence 1,414를 기록했고, queue timeout 36, telemetry refresh timeout 7을 숨기지 않고 receipt에 남겼다.
- `nccl-observer.json`은 `tempo-nccl-observer-v1`, source epoch `slurm-57416103-p1pair-cojob`, final sequence 1,001, 1,000 active window와 terminal complete pointer, 8 ranks, correctness true를 기록한다. final observer p99는 NCCL collective `1.695149 ms`, LMCache transfer `26.691746 ms`이며 `nccl_arrival_spread_ms=null`은 미수집 상태로 유지된다.
- 249개의 global decision이 NCCL/LMCache observer signal을 supported 상태로 받았고, 그중 244 complete decision은 local route를 선택했으며 5개는 global reject였다. 실제 provenance에는 observer producer sequence와 atomic global batch sequence가 별도로 남고, 모든 joint commit에서 `plan.telemetry_sequence == global batch sequence`가 성립했다.
- co-job 자체는 1,000 block, 4 pair, 8 rank에서 correctness true이며 background completion p50/p99 `28.5669985/32.440656 ms`, global token-tail p50/p99 `1.63483/1.832928 ms`다.

이 receipt는 이제 **Perlmutter native vLLM/LMCache request path가 NCCL·LMCache observer와 Cassini/endpoint state를 global admission 및 joint route commit에 실제로 연결하고, stale/complete/timeout 상태를 fail-closed 처리한다**는 integration/mechanism gate를 닫는다. 이것은 TEMPO의 가치가 component들의 합에서 차감된다는 뜻이 아니다. 오히려 heterogeneous component signal을 같은 identity-bound state plane에서 묶어 business-aware admission, route, pair resource, transfer concurrency와 recovery에 연결한 것이 TEMPO의 연구 단위다. 단, retry6 하나에는 fixed/predictor/APP_GLOBAL_ONLY/NETWORK_REQUEST_ONLY matched arm이 없으므로 end-to-end utility superiority나 논문 headline win은 아직 주장하지 않는다. authoritative receipt는 `results/tempo_go_p1pair_cojob_57416103_retry6/native_integration_receipt.json`이다.

### 6.9 corrected native observer receipt와 현재 통합 상태

allocation `57415597`에서 Perlmutter의 실제 NIXL/UCX·NCCL co-job을 다음 native 환경으로 고정해 다시 실행했다.

```text
NCCL_NET=Socket
NCCL_SOCKET_IFNAME=hsn
NCCL_IB_DISABLE=1
UCX_TLS=cuda_ipc,cuda_copy,tcp
UCX_NET_DEVICES=all
UCX_LOG_LEVEL=warn
```

그 결과 `results/tempo_go_cross_layer_observer_diag_57415597/result.json`은 2 block 모두 `overall_correctness_met=true`를 기록했다. 4개의 source/receiver pair가 각 block에서 기대한 4,194,304 bytes를 완료·검증했고, global NCCL token-tail은 p50 `12.786 ms`, p99 `19.781 ms`, LMCache transfer completion은 p50 `23.751 ms`, p99 `29.410 ms`였다. immutable history에는 active sequence 1/2와 terminal `producer_state=complete` sequence 3이 각각 보존되어 있다. router는 마지막 complete pointer를 live signal로 소비하지 않고 fail-closed하며, active sequence만 freshness/epoch/correctness 검사를 거쳐 decision plane에 들어간다.

이 receipt는 Perlmutter에서 **NCCL collective progress와 official LMCache/NIXL transfer를 같은 native window에서 생성하고, 그 결과를 provenance-safe atomic observer path로 vLLM router가 읽을 수 있게 된 것**을 증명한다. 따라서 이제 cross-layer signal은 문서상의 schema나 synthetic zero가 아니라 native producer output이다. 다만 이것 자체는 component/observer integration evidence이며, 이 신호 때문에 TEMPO가 app-only/network-only/fixed/predictor보다 좋아졌다는 utility claim은 아직 허용하지 않는다. 특히 이전 component가 각각 prior work와 겹친다는 이유로 이 contribution을 차감하지 않는다. TEMPO의 연구 단위는 이 heterogeneous signal들을 하나의 identity-bound state plane에서 결합하고, business utility·fairness를 포함한 global admission, pair scaling, placement, route, transfer concurrency와 staggering을 함께 결정하는 end-to-end control loop다.

초기 allocation `57415291`의 official NIXL step은 `UCX_TLS`/`UCX_NET_DEVICES`를 고정하지 않았을 때 hang했고 NCCL-only control만 완료했다. 이는 LMCache가 본질적으로 실패했다는 결과가 아니라 Perlmutter runtime environment contract를 발견한 execution diagnosis다. 이후 corrected environment에서 `57415597`의 correctness receipt를 확보했으며, 이 exact environment를 향후 P1PAIR+COJOB launcher에 고정했다.

### 6.10 allocation 57415765의 첫 same-allocation C5 시도: 실행 실패로 격리

`results/tempo_go_cross_layer_native_57415765_retry1/execution_failure_receipt.json`은 v18 frozen contract로 수행한 첫 4-node/16-GPU same-allocation 시도의 execution-only receipt다. v16 시도는 source digest mismatch로 data plane 이전에 중단됐고, v18 retry는 실제 vLLM 초기화와 workload warmup까지 도달했다. 그러나 `local` arm에서 official LMCache/NIXL의 다음 오류가 반복됐다.

```text
NIXL MemoryObj must be one registered, page-aligned descriptor;
object address/size does not match the current channel buffer
```

local arm은 `rc=137`과 Slurm step cancellation으로 `result.json` 없이 끝났다. 이후 `remote`, `predictor`, `queue_gpu`, `tempo`의 `rc=143`은 독립적인 baseline/TEMPO 결과가 아니라 공통 runner teardown 중단이다. intended co-job root에는 `result.json`, `nccl_observer.json`, binding receipt가 모두 없었고, 따라서 이 run은 cross-layer provenance 또는 coupled utility를 증명하지 않는다. 이 사실은 “TEMPO가 졌다”가 아니라 **현재 native lifecycle/data-plane harness가 terminal comparison을 닫지 못했다**는 정확한 범위의 실패다.

이 시도에서 관찰된 NIXL 오류는 향후 contention에서 LMCache completion path가 취약할 수 있다는 중요한 failure hypothesis이지만, co-job identity와 offered-population ledger가 완성되지 않았으므로 LMCache의 contention-wide failure나 TEMPO의 성능 결론으로 승격하지 않는다. 다음 source-bound v19에는 Slurm TERM을 Python lifecycle exception으로 변환하는 node cleanup, 명시적 co-job step name, rank별 co-job logs와 nested-step cleanup을 포함했다. root/UDI/privileged NIC 변경은 하지 않았다.

### 6.11 seven-arm cross-layer ablation contract

v20에서는 native comparison을 다섯 arm에서 일곱 arm으로 확장했다. `NETWORK_REQUEST_ONLY`는 request geometry와 observed NCCL/LMCache/Cassini network/fabric signal만으로 route penalty를 계산하며 endpoint resource admission, tenant business debt, pair scaling과 global cross-layer actuation을 사용하지 않는다. `APP_GLOBAL_ONLY`는 TEMPO와 동일한 global admission/pair/endpoint business path를 사용하지만 frontend telemetry adapter에서 `cross_layer` envelope를 제거한다. 따라서 full TEMPO의 추가 입력과 joint actuator를 직접 ablate할 수 있다.

`results/tempo_go_c5_cross_layer_contract_v20/native_run_contract.json`은 이 seven-arm order와 same-allocation co-job, current source inventory, signal-safe cleanup을 모두 freeze했지만 아직 실행되지 않았다. 이 contract가 실제 native result와 independent validation gate를 갖기 전에는 ablation seam 구현을 성능 결론으로 표현하지 않는다.

v20 시도에서는 co-job root 사전 생성 bug가 발견되어 local arm을 의도적으로 중단했다. `results/tempo_go_cross_layer_native_57415765_v20/execution_failure_receipt.json`은 이 execution-only 사실을 보존하고, 수정 후 같은 allocation에서 `results/tempo_go_cross_layer_cojob_57415765_v20_probe`의 2-node/8-GPU correctness/observer receipt를 별도로 확보했다. 이것은 co-job launcher가 이제 실제로 동작한다는 증거이지 C5 utility 증거가 아니다.

v21은 수정된 launcher로 시작했지만 3-block co-job이 약 12초 만에 완료되어 C5의 측정 lifecycle과 겹치지 않았다. `results/tempo_go_cross_layer_native_57415765_v21/execution_failure_receipt.json`은 이를 `cojob_completed_before_measured_campaign` execution-only failure로 격리한다. co-job correctness 자체는 유효하지만 동시 contention window가 없으므로 v21에는 어떠한 성능·ablation claim도 허용하지 않는다. 이 실패는 TEMPO negative가 아니라 same-allocation offered-load binding이 충분히 길고 측정 전에 active observer를 보장해야 한다는 harness 조건을 확정한 것이다.

v22에서는 active-observer readiness gate 자체는 통과했고 local arm과 co-job이 실제로 겹쳤다. 그러나 600-block co-job은 native harness에서 151초 만에 끝났고 seven-arm C5가 끝나기 전에 terminal sequence 601을 기록했다. `results/tempo_go_cross_layer_native_57415765_v22/execution_failure_receipt.json`으로 이 실행을 partial-overlap execution-only failure로 격리한다. 즉 v22는 “동시 contention이 시작되는가”는 검증했지만 “모든 arm이 같은 offered load를 받는가”를 닫지 못했으며, 성능·ablation claim은 허용하지 않는다.

v23에서는 10,000-block을 요청했지만 co-job launcher에 남아 있던 `timeout 900s`와 `--time=00:20:00` 때문에 terminal observer/binding receipt를 만들기 전에 execution-only로 중단했다. `results/tempo_go_cross_layer_native_57415765_v23/execution_failure_receipt.json`은 local arm과 실제 contention overlap 및 후속 baseline failure receipts를 보존하되, seven-arm utility claim을 금지한다.

v24에서는 위 timeout을 고쳤지만 32 MiB/rank, 16 token-iteration, 4 MiB foreground의 지속 hot co-load가 local arm을 실질적으로 capacity 밖으로 밀어냈다. 약 5분 동안 HTTP completion은 27건에 그쳤고 observer LMCache transfer tail은 약 20초까지 상승했다. `results/tempo_go_cross_layer_native_57415765_v24/execution_failure_receipt.json`은 이를 `sustained_hot_contention_exceeds_campaign_capacity`로 보존한다. 이는 Perlmutter-style shared GPU/interconnect/LMCache contention이 실존한다는 headroom/bottleneck evidence이며, seven-arm utility result가 아니다.

v25에서는 이 sustained-moderate profile에서 local arm은 2,712-row를 닫았지만 remote arm은 같은 offered load에서 크게 느려져 7-arm 공통 campaign을 닫지 못했다. `results/tempo_go_cross_layer_native_57415765_v25/execution_failure_receipt.json`은 local full result, remote signal failure, active observer를 보존하되 utility claim을 금지한다. 이는 remote/KV path가 local보다 contention-sensitive하다는 headroom evidence다.

v27 `results/tempo_go_c5_cross_layer_contract_v27/native_run_contract.json`(SHA `e5966d95a5cc2f36173f004a96a0aa877ae05cadb875d81b76cd27391d68c367`, fingerprint `6d37eb6eb171b2bfd49100087787caa597608dae8bda77fffa24b7a6681ce49e`)과 v28 `results/tempo_go_c5_cross_layer_contract_v28/native_run_contract.json`(SHA `a745434411d74e569e734564c685047da7329fa1e702086d98ee12f5e587e483`, fingerprint `da86c83a23b72e2dd655406820bbbbe749e4e3528252097916c9814c4116cc2b`)은 source-bound discovery history로 보존한다. v28은 276-row short held-out seven-arm order, sustained-moderate co-load, 10,000-block ceiling, 3,600초 timeout, `01:00:00` step과 C5 coverage gate를 freeze했지만, 현재 router/orchestrator source와 digest가 달라 current-source contract가 아니다. v29 `results/tempo_go_c5_cross_layer_contract_v29/native_run_contract.json`(SHA `a56585dacff2c71f28882d9cc61a4c3b9ff8548d9ff505317282cd386710fda1`, fingerprint `341f5dbb1d72c5d0cf00b83ecf7003e67d07a90461266f9be91ad5676c30a8ac`)는 soft-shadow-price v2 history로 보존한다. resource-envelope spare-pair activation을 포함해 native campaign 전에 full verify를 통과한 v30 campaign contract는 `results/tempo_go_c5_cross_layer_contract_v30/native_run_contract.json`(SHA `eaab9c4d70731ae68c59158b9143da11bf602387b1b00cdd356523f35e24a830`, fingerprint `0f2f25f2f9ba58c46d82868ed553c2491b8369113cddc2bcf01849e9366cf9a6`)다.

v28의 실제 same-allocation execution은 `57415765`에서 닫혔다. local/remote/predictor/queue_gpu/network_request_only/app_global_only/TEMPO 모든 arm이 result를 만들었고, co-job binding은 `cojob_covered_c5_end=true`, observer terminal sequence `10001`, 10,000/10,000 correctness를 기록했다. 이는 v26/v28의 첫 유효한 seven-arm cross-layer discovery comparison이다. 다만 analyzer의 `performance_claim_allowed=false`와 §0의 reject/goodput/p99 수치가 적용되므로 independent validation이나 production superiority로 승격하지 않는다. v28은 v26보다 goodput을 개선했지만 app-only incremental gate와 strongest-fixed 5% gate를 통과하지 못했다. v29 soft-shadow-price와 v30 resource-envelope는 이 reject cost를 줄이기 위한 mechanism history이며, v29f가 측정 전에 종료됐으므로 현재 다음 source-bound native gate는 v30의 lifecycle-stable same-population validation이다.

### 6.12 v29 reject-aware actuation과 v29f 실행 경계

v28의 discovery에서 확인된 문제는 NCCL/Cassini/LMCache 신호가 없다는 것이 아니라, coupled pressure를 safety cap으로 번역하면서 valid work를 너무 많이 shed했다는 것이었다. v29는 이 지점을 다음처럼 분리한다.

- `hard_window_v1`의 resource target과 기존 provenance는 backward-compatible하게 보존한다.
- `soft_shadow_price_v2`에서는 비임계 overage를 `overage_fraction`과 `overage_penalty_ms`로 score에 반영하고, 현재 관측값을 포함하는 `enforced_*` lease를 atomic plan에 넣는다.
- Cassini pause/ECN/retry/timeout, transport failure와 같은 critical pressure는 `critical_guard=true`로 두고 기존 hard rejection을 유지한다.
- router는 v2 lease를 받아 local-prefill, remote-prefill, remote-KV bytes와 semantic-op credit을 실제 endpoint commit에 적용한다. 즉 단순 telemetry-only 변경이 아니다.

이 구조와 backward-compatible v1 parser는 cross-layer 관련 CPU test 83개, launcher `bash -n`, source `py_compile`을 통과했다. 그러나 v29f native attempt는 C5 첫 측정 전에 끝났으므로 v2가 v28보다 좋다거나 나쁘다는 성능 결론은 없다. allocation guard는 통과했고 actual TP4 vLLM/LMCache/NIXL 초기화 로그는 남았지만 request ledger, observer binding과 `result.json`은 생성되지 않았다. 이 실행 실패는 native launcher/process lifecycle을 고정해야 한다는 증거이며, root 권한·UDI·NCCL/Slingshot privileged reconfiguration 문제는 아니었다.

---

## 7. C5 native 및 global candidate 결과

### 7.1 최신 native held-out five-arm receipt

authoritative latest source-rebound root는 `results/tempo_go_c5_r8_16_20_20_native_job_57409956_v3`이고 analyzer SHA는 `b7e302ab1f893310602b491a8971138d3f4b3cd7fa906b4f7ce05848ac305f45`다. contract는 `results/tempo_go_c5_r8_16_20_20_contract_v3/native_run_contract.json`, SHA `002ee5424c9779b22d2cc622cb9143227f8370d03d6b22d0f3c9a560f153e481`, fingerprint `7691d005cad942c26a9a8792cf1487431ce5c4f7abe43ebb7b409a2fef5a854e`다.

| arm | terminal result | route | request goodput/s | output-token goodput/s | E2E p50/p99 |
|---|---|---|---:|---:|---:|
| ALWAYS_LOCAL | 2,712 complete | local 2,712 | 7.934 | 981.8 | 15,203 / 19,675 ms |
| ALWAYS_REMOTE | 2,712 complete | remote 2,712 | 9.581 | 1,185.7 | 11,090 / 20,560 ms |
| PREDICTOR_ONLY | 2,712 complete | local 2,592 / remote 120 | 7.928 | 981.1 | 15,530 / 19,525 ms |
| QUEUE_GPU_ONLY | measured raw 없음, exit 143 | — | — | — | execution failure |
| TEMPO_GO | 982 complete / 1,730 explicit reject / 0 fail | local 908 / remote 74 | 4.786 | 548.4 | 7,463 / 9,206 ms completed-only |

TEMPO completed-only latency는 같은 request population 비교가 아니므로 win이 아니다. global scheduler observation은 5,424건, invalid 0건, endpoint completion receipt는 982건, pair activation은 1건이다. background는 2,436건 중 769건, interactive는 96건 중 80건, latency는 96건 중 50건만 완료했다. 이 run이 증명한 것은 native global wiring과 explicit shedding이며 performance/fairness superiority가 아니다.

### 7.2 native failure-quarantine receipt

`results/tempo_go_c5_native_failure_quarantine_job_57404614_v1`에서 다음이 관찰됐다.

- raw 2,712 rows: semantic complete 1,633 / failed 9 / rejected 1,070
- global failure receipt 9: pair-scope transport 3 / route-scope HTTP 6
- `route_failure_quarantine` rejection 1,714
- official LMCache `CacheEngineKey ... not found in local data`, `EngineDeadError`
- proxy `ConnectError`, frontend error
- native step exit 143, `result.json` 없음
- `router_decisions_exact=false`, `terminal_contract_valid=false`, `performance_claim_allowed=false`

이것은 actual failure/quarantine wiring evidence다. failure를 완전히 흡수했거나 성능을 개선했다는 증거가 아니다.

### 7.3 C5 global candidate 계보

여기서 Candidate A–I는 C4 A/B/C와 다른 C5 global-candidate namespace다.

| C5 candidate | 구조 | 결과 | 판정 |
|---|---|---|---|
| guard/A | remote semantic-op 마지막 slot reserve | native terminal receipt를 닫았지만 explicit reject가 큼 | integration only |
| B / fairscale | queue/SLO-risk proactive pair activation | pair1 activation은 앞당겼으나 complete/reject와 SLO-goodput 개선 없음 | CPU negative |
| C | explicit request failure receipt + deny-until-probe quarantine | CPU와 native에서 failure receipt/quarantine 발동 | robustness mechanism only |
| D | C failure safety + B proactive scaling | pair assignment만 바뀌고 aggregate C와 동일 | CPU neutral/negative |
| E | global wait 2 s → 5 s | 1,433 complete/1,279 reject, p50/p99 6,623/8,446 ms | tail/SLO utility fail |
| F | queue capacity 128 → 2,048 diagnostic | 7분 이상 queue scan, artifact 없이 중단 | overhead/design negative |
| G | 5 s + tenant queue reservation 16 slots | 1,430/1,282, p50/p99 6,299/8,446 ms | primary/fairness utility fail |
| H | G reservation + 2 s + proactive pair trigger | 1,321/1,391, p50/p99 5,321/5,914 ms | fixed local과 exact neutral |
| I | telemetry failure pair circuit + 25% survivor service lane | normal은 fixed local과 neutral; failure 912/1,800, p99 7,550 ms | mechanism works, utility fail |

E–I latency는 held-out duration-based **CPU control-plane replay** 값이다. GPU, interconnect, LMCache native latency가 아니다.

### 7.4 strict G/I promotion negative

machine-check audit는 동일 manifest/workload/Elastic/endpoint/baseline identity와 같은 four fixed-arm receipt를 사용했다.

| item | strongest fixed CPU | predictor CPU | Candidate G | Candidate I |
|---|---:|---:|---:|---:|
| E2E p50 | QUEUE_GPU 5,297.971 ms | 5,344.029 ms | 6,298.891 ms | 5,321.023 ms |
| 10% fixed limit | 4,768.174 ms | — | fail | fail |
| 5% predictor limit | — | 5,076.827 ms | fail | fail |
| terminal | fixed fingerprint equal | fixed fingerprint equal | 1,430/1,282/0 | 1,321/1,391/0 |

I failure replay는 pair-0 quarantine을 발동했지만 912 complete/1,800 reject/0 fail, p99 7,549.511 ms로 robustness utility gate도 실패했다.

정확한 결론:

> 동일 frozen control-plane replay에서 구조적으로 다른 G와 I가 preregistered CPU promotion gate를 실패했으므로 native independent validation으로 승격하지 않는다.

정확하지 않은 결론:

> TEMPO-GO의 native 성능이 음성으로 증명됐다.

audit JSON은 이를 `completion_status=CPU_negative_only_native_validation_unproven`, `native_performance_negative_proven=false`, `performance_claim_allowed=false`로 고정한다.

### 7.5 v30 resource-envelope spare-pair native discovery

승인된 4-node/4-hour interactive allocation `57423440`에서 v30을 실제로
실행했다. `local`, `remote`, `predictor`, `queue_gpu`,
`network_request_only`, `app_global_only`, `tempo`의 7개 arm이 모두 같은
276-row workload에서 valid terminal receipt를 만들었고, 같은 allocation의
official LMCache/NIXL/UCX + 8-rank NCCL co-job은 10,000/10,000 block
correctness와 `cojob_covered_c5_end=true`를 닫았다. 따라서 이번 결과는
실행 실패가 아닌 coupled native discovery receipt다.

| arm | complete/reject/fail | output-token goodput/s | E2E p50/p99 ms |
|---|---:|---:|---:|
| local | 276/0/0 | 519.28 | 11,849 / 36,224 |
| remote | 276/0/0 | 514.07 | 10,249 / 36,952 |
| predictor | 276/0/0 | 496.34 | 11,556 / 37,243 |
| queue_gpu | 276/0/0 | 640.99 | 10,350 / 31,968 |
| network_request_only | 276/0/0 | 523.15 | 11,656 / 35,255 |
| app_global_only | 136/140/0 | 618.18 | 7,490 / 13,551 |
| TEMPO v30 | 134/142/0 | 533.01 | 7,172 / 14,791 |

이 표의 completed-only E2E는 reject가 있는 arm 사이의 승패를 뜻하지 않는다.
정식 gate는 false다. TEMPO는 strongest fixed queue-GPU보다 output-token
goodput이 약 16.9% 낮고, app-only보다 약 13.8% 낮았다. full TEMPO의
global decision은 141 queue timeout, 1 telemetry refresh timeout, 1 pair
activation을 기록했다. C3 both-hot에서 126개 중 113개가 reject되어, 새
pair activation이 shared-fabric externality를 해결하지 못했다. 모든 tenant는
starvation=false이고 terminal validity는 276/276이지만, 이것은 fairness
mechanism/correctness evidence이지 utility win이 아니다.

v30의 정확한 lesson은 “pair scaling이 불필요하다”가 아니다. co-job이 두
pair가 공유하는 NCCL/LMCache/fabric resource를 뜨겁게 만들면 per-pair
resource-envelope 초과만 보고 spare pair를 여는 것은 같은 병목을 새 pair로
복제한다. 다음 mechanism은 pair-local activation과 별도로 shared-fabric
resource budget/concurrency/stagger를 global layer에서 유지하고, shared
externality와 pair-local failure를 구분해야 한다. 이 v30 receipt는 그
설계 전환을 요구하는 native causal/headroom evidence이며 performance
negative 전체로 확대하지 않는다.

### 7.6 v31 shared-fabric global actuator: implementation gate

v31은 v30의 pair-local spare activation을 이름만 바꾼 threshold variant로
재실행하지 않고, 다음 state/action path를 현재 source에 구현했다.

- `CrossLayerTelemetry`의 `source_epoch`, topology fingerprint와
  `communicator_id`가 같은 pair만 하나의 compatible shared-fabric group으로
  묶는다. group이 한 pair뿐이면 shared externality를 추정하지 않고 기존
  pair-local v1/v2 actuator를 사용한다.
- group별 remote request slot, KV bytes, semantic-op의 세 budget을 별도로
  유지한다. NCCL p99/arrival spread, LMCache transfer p99, Cassini pause/ECN/
  retry/timeout와 remote in-flight vector는 resource별 contribution으로
  보존하고 missing/unsupported는 0으로 바꾸지 않는다.
- aggregate budget은 atomic admission 전에 global held remote usage와 합산해
  집행한다. 초과하면 `shared_remote_budget`으로 bounded queue/defer되고,
  selected v3 plan에는 limit, used-before, group identity, contribution과
  bounded dispatch stagger가 남는다.
- shared externality가 budget을 줄인 경우에는 spare pair activation을
  억제한다. 반대로 pair-local route health/circuit/capacity failure는 기존
  inactive spare candidate 경로를 통해 activation할 수 있다. 즉 “같은 shared
  병목을 복제하는 scale”과 “고장난 pair의 failure containment”을 분리한다.
- request decision receipt는 v3 shared group summary만 compact하게 보존하고,
  full raw endpoint provenance는 batch/snapshot artifact에 둔다. compatible
  group static budget은 telemetry epoch마다 한 번 계산하고 request path에는
  current held usage만 overlay한다.

현재 source-bound CPU evidence:

| check | result | interpretation |
|---|---:|---|
| v31 global/cross-layer/router related tests | 36 passed | v3 schema, shared budget, shared/no-scale, pair-local activation과 strict HTTP parser |
| all `tempo/test_pd_global*` + C5 node tests | 108 passed | 기존 lifecycle/fairness/queue/failure regression 보존 |
| py_compile: orchestrator/profile/router | passed | source syntax/static gate |
| logical pair fan-in 2→1024 | update 0.022→0.629 ms; request decision 0.585→1.347 ms | cached shared envelope와 compact decision provenance로 request path central scan 억제 |
| 1024-pair diagnostic snapshot | 173.353 ms | raw all-pair diagnostic serialization 비용; request decision과 별도 sampling/diagnostic path로 보고 |

이 수치는 native inference 성능이나 Perlmutter scale superiority가 아니다.
v31 profile `real_tempo_go_profile_short_slice_v4_shared_budget.json`은
`global_budget_v3`를 freeze했고 profile SHA는
`ed49fd3bac093e2bd74e1103e1474000c214fd959098774b7ae6ab84db6ac9dd`다.
현재 source에 맞춘 새 native contract는
`results/tempo_go_c5_cross_layer_contract_v31b/native_run_contract.json`
(file SHA `eca07e359e934e50b2b37d8f66ece23e5e9dee9099d5c55c0bb787f33fd3f732`,
fingerprint `9b3922220722a6218951edd049e20a683ec78085b3825d829a1b5725e662d80c`)
이며 full verify를 통과했다. 이 contract는 아직 native에서 실행되지 않았고,
따라서 performance claim은 여전히 false다.

v31b native launch는 현재 사용자 allocation 상태 때문에 execution-only로
격리한다. 신규 `salloc`은 `QOSMaxSubmitJobPerUserLimit`로 거부됐고, 기존
4-node allocation `57425033`에 붙인 step에서는 실제 allocation guard는
통과했지만 job-level request가 `cpu=4, CPUs/Task=1`이었다. frozen co-job의
`32 CPUs/task` step이 `Job step's --cpus-per-task value exceeds that of job`
및 `Error configuring interconnect`로 시작하지 못했고, `result.json`과
observer binding은 생성되지 않았다. 관련 stderr receipt는
`results/tempo_go_cross_layer_cojob_v31b_57425033/cojob.stderr.log`
(SHA `4f78aad5df6429613582b3c079fdd8cc767c15add5dfd755e42d6a76911dd893`)와
`results/tempo_go_cross_layer_cojob_v31b_retry_cpu32_57425033/cojob.stderr.log`
(SHA `4cc4fa58cbdb176a24b6c292a9ed655fe49671bb498c18500954eab4e1a135ce`)다.
이는 TEMPO나 LMCache contention의 negative가 아니며, 32-CPU/task를
요청한 새 4-node/4-hour interactive allocation이 확보될 때까지 native
utility gate를 열지 않는 환경 contract failure다. 기존 job은 취소하지 않는다.

### 7.7 v32 endpoint-queue-lease discovery와 downstream contract fix

기존 source-bound v32 endpoint-queue-lease profile로 같은 allocation
`57425033`에서 실제 4-node/16-GPU seven-arm campaign이 일부 진행됐다. 이
실행은 v31 shared-budget profile이 아니므로 v31 또는 TEMPO-GO의 최종
성능 결과로 사용하지 않는다. 그래도 whole-system failure boundary를
명확히 하는 useful evidence는 남겼다.

| arm | result | validity | observed failure |
|---|---:|---|---|
| local/remote/predictor/queue_gpu/network_request_only | 각각 276/276 complete | exact terminal/correctness valid | 없음 |
| app_global_only | 173 complete / 103 global reject | invalid: `router_decisions_exact=false`, 24 HTTP errors | shared-load ingress queue timeout + HTTP 502 |
| tempo | 164 complete / 112 global reject | invalid: `router_decisions_exact=false`, 18 HTTP errors | `resource_limits.local_token_ms` endpoint-window violation |

TEMPO의 18 HTTP 오류는 NCCL/LMCache가 성능상 졌다는 증거가 아니다. v32의
`endpoint_queue_lease`가 global capacity 초과 debt를 downstream
`EndpointFeedbackController`의 physical window보다 큰 `resource_limits`로
전달한 것이 직접 원인이었다. 따라서 global queue lease가 global ledger의
debt를 계속 소유하되, endpoint에는 its physical window를 넘지 않는 limit을
전달해 실제 endpoint queue가 대기하도록 router boundary를 수정했다.
이 수정은 `test_joint_commit_queue_lease_clamps_endpoint_window`로
over-capacity local commit과 downstream queue를 검증한다. v32 co-job은
10,000-block correctness/result binding을 만들기 전에 cancel되어 v32
campaign 전체는 performance claim이 아니다.

현재 source에 맞춘 v31c contract는
`results/tempo_go_c5_cross_layer_contract_v31c_endpoint_clamp/native_run_contract.json`
(file SHA `8b3705b09d0931428d89e44a379f321ae759365854d911ded545d5264892dd79`,
fingerprint `a84dfc2966d57c9c66d184705af98edca653479d4513da3d46e2a91d34f88942`)
로 build·verify됐다. v31c native launch는 잘못된 outer 4-node step이
co-job과 CPU를 중복 요청해 `More processors requested than permitted`에서
data plane 이전에 종료됐고, `results/tempo_go_cross_layer_cojob_v31c_57425033`
에는 observer/result가 없다. 이 또한 TEMPO/LMCache negative가 아니라
Slurm step topology failure다. 다음 native 실행은 allocation의 단일
interactive shell step에서 wrapper를 시작하고, wrapper가 co-job과 각
4-node arm step을 생성하는 형태로 고정하며 동시 campaign을 금지한다.

### 7.8 v34/v35 service-lane lease boundary receipt

v34는 endpoint physical-window clamp를 현재 source에 반영한 뒤 승인된
4-node/4-hour interactive allocation `57426273`에서 TEMPO arm을 실행했다.
이전 v32의 `resource_limits.local_token_ms` 초과 오류는 사라졌지만,
queue lease가 downstream fixed pair-router의 ingress queue로 들어간 뒤
16건이 `elastic ingress queue timeout`으로 502가 됐다. 이는 cross-layer
telemetry가 끊긴 것이 아니라, global ledger의 over-capacity debt와
endpoint service capacity 사이의 admission boundary가 닫히지 않았다는
증거다.

v35는 business deadline의 남은 budget을 service-lane queue wait로 전달하는
동적 lease를 source-bound contract
`results/tempo_go_c5_cross_layer_contract_v35_endpoint_queue_lease_service_lane/native_run_contract.json`
(file SHA `a38c959d11ba89c0163f89ebc1972ce3f149c806a3f88544f5f34c2c1626c8e1`,
fingerprint `39f949179df3c40352646495e7a30f7b04b65cd33f3280652d9acffb14727e57`)
로 고정하고 같은 allocation에서 TEMPO arm만 재검증했다. node source guard와
97개 targeted test는 통과했고, raw receipt
`results/tempo_go_cross_layer_native_v35_tempo_only_57426273/tempo/tempo_go_c5_discovery/raw.json`
는 276 requests 중 140 status-200, 116 global reject, 20 HTTP 502를
기록했다. queue lease 54건 중 34건만 완료됐고 20건은 모두 downstream
`bounded ingress queue timeout`이었다. coordinator는 observer sequence
102, NCCL/Cassini/LMCache provenance, route failure 0을 유지했다.

따라서 v35의 동적 대기는 성능 개선이나 lease 성공으로 승격하지 않는다.
현재의 정확한 설계 결론은 **global queue lease가 endpoint reservation/credit
receipt 없이 request를 downstream queue로 넘기는 것은 global orchestration이
아니며, completion utility를 오히려 502로 악화시킬 수 있다**는 것이다.
다음 implementation은 queue wait 상수 조정이 아니라
`global decision → endpoint service-lane admission/reservation → immutable
route commit`의 2-phase boundary를 구현해야 한다. endpoint가 실제로 받은
bounded service credit 또는 명시적 queue reservation을 반환하기 전에는
global lease를 accepted route로 간주하지 않으며, reservation 실패는
business-aware reject/defer receipt와 global debt release로 닫는다. 이것이
v35에서 검증된 next mechanism이며, v35 자체의 performance claim은 false다.

---

## 8. workload를 설정하는 정확한 방법

### 8.1 현재 held-out output128 compatibility contract

| field | frozen value |
|---|---|
| manifest | `results/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json` |
| manifest SHA | `6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c` |
| workload | `results/tempo_go_c5_heldout_output128_v1/workloads/validation.jsonl` |
| workload SHA | `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33` |
| validator SHA | `f00157c5f237c7a271197e499046e0e2a9884881cffeca46554accd015933fd0` |
| rows | 2,712, replicate `r02/r03` |
| prompt/output | foreground `(512,16)/(2048,256)/(4094,16)`; hot stream `4094/128` |
| output distribution | 16: 240, 128: 2,352, 256: 120 |
| cache | unique MISS 1,992 / P_ONLY 720 |
| arrival | explicit absolute offsets, 15 s phase + 2 s cooldown |
| policy exclusions | phase name, future arrival, oracle route, physical-switch label |

이 2,712-row trace는 application-level C5를 같은 조건에서 회귀 검증하는 immutable contract다. **NCCL/Slingshot 동시 부하가 없으므로 최종 cross-layer TEMPO의 headline workload나 유일한 promotion oracle로 쓰지 않는다.** G/I가 이 CPU/profile replay에서 실패했다는 사실은 보존하되, fabric-coupled candidate를 이 trace만으로 탈락시키지 않는다.

### 8.2 phase별 row와 intended load

| phase | rows/replicate | total rows | anchor input | 목적 |
|---|---:|---:|---|---|
| C0 cool | 30 | 60 | foreground/low load | normal regression/locality |
| C1 decoder-hot | 366 | 732 | local-pinned 22.4/s anchor | decoder prefill/decode externality |
| C2 remote-hot | 102 | 204 | remote-pinned 4.76/s anchor | P compute + transfer + receiver |
| C2 KV-remote-hot | 210 | 420 | P_ONLY 12/s knee anchor | retrieve/transfer/install ceiling |
| C3 both-hot | 618 | 1,236 | C1 + P_ONLY remote | coupled moving bottleneck |
| recovery | 30 | 60 | pressure removal | probe/hysteresis/scale-down |

output=2 C1/C2/C3 evidence와 output=128 held-out를 같은 성능 evidence로 섞지 않는다. held-out output128에서 phase-wise fixed-path direction이 새 geometry에서도 유지되는지 native fixed-arm characterization으로 다시 확인해야 한다.

### 8.3 tenant contract

| tenant | weight | minimum service | max wait | TTFT | TPOT | E2E |
|---|---:|---:|---:|---:|---:|---:|
| latency | 4.0 | 0.15 | 0.5 s | 1 s | 100 ms | 4 s |
| interactive | 2.0 | 0.15 | 1 s | 2 s | 150 ms | 8 s |
| batch | 1.0 | 0.10 | 2 s | 3 s | 250 ms | 16 s |
| background | 0.5 | 0.05 | 5 s | 5 s | 400 ms | 30 s |

held-out tenant totals은 background 2,436, batch 84, interactive 96, latency 96이다. label만 round-robin하는 것으로 multi-tenant라고 부르지 않는다. 각 tenant가 독립 arrival, geometry/cache distribution, SLO와 backlog를 가져야 한다.

### 8.4 realistic cross-layer headline workload가 반드시 포함할 상태

1. normal cool state
2. decoder-local prefill/decode hot
3. cold remote P/KV/receiver hot
4. P_ONLY transfer/semantic-operation hot
5. both hot coupled overload
6. asymmetric pair: pair0 hot/failing, pair1 cool 또는 다른 bottleneck
7. bounded burst와 sustained overload
8. actual NCCL collective hot: collective latency/arrival spread가 증가하지만 LMCache는 낮은 상태
9. NCCL + official LMCache/UCX + decoder가 동시에 뜨거운 shared-fabric state
10. per-NIC Cassini vector는 변하지만 application completion은 안정적인 false-positive state
11. application completion이 악화되지만 일부 Cassini counter는 0인 false-negative state
12. endpoint/NCCL peer failure receipt, stale/missing telemetry와 explicit PROBE recovery
13. pressure removal 후 scale-down/no-flap recovery

headline contention은 actual vLLM inference tenants, official LMCache transfer와 real NCCL collectives로 만든다. synthetic CXI는 별도 attribution ablation으로만 사용한다. phase label은 workload generator와 analyzer만 알고 policy에는 주지 않는다.

### 8.5 4-node에서 사용할 topology/workload matrix

4-node limit 안에서도 서로 다른 coupling을 분리할 수 있다.

| profile | node/GPU use | 무엇을 검증하는가 |
|---|---|---|
| `P2PAIR` | 4 nodes에 P0/D0/P1/D1, endpoint당 TP4 | 현재 two-pair scaling, local-vs-remote, pair asymmetry와 business admission |
| `P1PAIR+COJOB` | 2 nodes에 P/D TP4, 나머지 2 nodes/8 GPUs에 real NCCL+official NIXL/UCX co-job | independent HPC job의 collective와 KV traffic이 shared Slingshot에서 inference에 주는 externality 및 global staggering/admission 가치 |
| `TP8-CROSSNODE` | 2-node TP8 P + 2-node TP8 D | vLLM/NCCL TP communication과 remote KV movement가 모두 inter-node일 때의 direct coupling |
| `COMPONENT-ATTR` | exact `run_lmcache_nixl_contention_2node.py` 또는 bounded fixed-path intervention | NCCL collective tail과 LMCache completion 사이 causal direction 및 telemetry sensitivity |

모든 profile을 한 성능 표에 무조건 합치지 않는다. `P2PAIR`는 pair-control 결과, `P1PAIR+COJOB`은 multi-job orchestration 결과, `TP8-CROSSNODE`는 topology stress 결과로 각각 same-topology baseline과 비교한다. node placement/dragonfly locality는 run 전에 기록하고 arm 사이에 바꾸지 않는다.

### 8.6 workload validity gate

controller 결과를 보기 전에 다음을 확인한다.

- C1에서 remote가 local보다 preregistered margin 이상 유리한가
- C2/P_ONLY-hot에서 local이 remote보다 유리한가
- C3에서 winner가 state에 따라 실제로 이동하는가
- fixed arm의 offered/achieved background load와 cache preparation이 동일한가
- P_ONLY가 actual source hit/retrieve/install path를 타는가
- every request가 complete/reject/fail 중 하나의 exact terminal receipt를 갖는가
- output/stream/route/cache/namespace/contract SHA가 모두 valid한가
- failure를 successful slow request로 세지 않았는가
- NCCL collective type/bytes/count와 completion receipt가 arm 사이 동일한가
- Cassini sample이 NIC/traffic-class/support/freshness를 보존하고 missing을 0으로 바꾸지 않았는가
- LMCache bytes/semantic operations와 NCCL bytes가 실제로 겹친 interval이 존재하는가
- co-job이 단순 synthetic byte pump가 아니라 correct NCCL result와 exact LMCache byte verification을 남겼는가

방향 gate가 실패하면 controller coefficient를 조정하지 않고 workload, cache preparation 또는 native service state부터 고친다.

---

## 9. 다음 TEMPO-GO: Perlmutter cross-layer global orchestrator

현재 G/I를 섞거나 queue/reservation 숫자를 바꾼 Candidate J는 새 구조가 아니다. 다음 candidate의 핵심 mechanism은 **이질적인 telemetry를 더 많이 모으는 것 자체가 아니라, NCCL·Slingshot·LMCache·vLLM·business state를 resource-specific service envelope와 shadow price로 변환하고 전체 decision을 한 폐루프로 닫는 것**이다. 아래는 target architecture이며 아직 성능 result가 아니다.

### 9.1 architecture

```text
tenant/API demand + SLO/value/fairness contract
                       │
                       ▼
┌──────────────── TEMPO global coordinator ────────────────┐
│ offered-population utility, resource shadow prices,       │
│ placement/pair activation, admit/defer/reject, route,     │
│ concurrency/staggering, failure-domain recovery           │
└───────────────▲───────────────────────────┬───────────────┘
                │ aggregated envelopes      │ immutable plans
      ┌─────────┴─────────┐       ┌─────────┴─────────┐
      │ pair/node agent 0 │  ...  │ pair/node agent N │
      │ vLLM scheduler    │       │ vLLM scheduler    │
      │ LMCache/UCX ops   │       │ LMCache/UCX ops   │
      │ NCCL collectives  │       │ NCCL collectives  │
      │ GPU/NVLink/PCIe   │       │ GPU/NVLink/PCIe   │
      │ Cassini NIC vec   │       │ Cassini NIC vec   │
      └───────────────────┘       └───────────────────┘
```

central coordinator가 모든 raw counter를 request마다 직접 poll하지 않는다. node/pair agent가 local clock에서 raw signal을 검증하고 짧은 resource envelope, confidence와 failure delta로 축약한다. global layer는 이 envelope와 business demand를 받아 action을 결정한다.

deployment scope는 둘로 구분한다.

- **service scope (현재 4-node 구현)**: TEMPO가 자기 vLLM P/D fleet과 opt-in co-job agent를 제어한다. 다른 job의 load는 exogenous이며 Cassini/application completion으로 관찰할 뿐 그 process를 건드리지 않는다.
- **facility scope (최종 Perlmutter-scale 목표)**: NERSC/operator 권한 아래 job/QoS/placement와 opt-in NCCL agents를 shard coordinator에 연결해 service와 batch/training workload를 공동 배치·admit한다. 일반 사용자 권한의 4-node 실험에서 facility scheduler나 다른 사용자 job을 제어했다고 주장하지 않는다.

### 9.2 cross-layer signal plane

| layer | signal | 현재 자산/수집법 | policy 의미 |
|---|---|---|---|
| business | tenant value/weight, TTFT/TPOT/E2E SLO, min service, max wait/reject budget | current tenant profile/ledger | 어떤 work를 언제 보호·defer할지 |
| facility/workload | managed job ID/class/QoS, placement, declared phase/deadline, opt-in communication budget | current experiment manifest; production에서는 operator/Slurm integration이 있을 때만 | inference와 training/batch work의 global business priority와 placement |
| vLLM service | running/waiting, active sequences, KV usage, token completion, TTFT/TPOT stretch | current scheduler snapshot + frontend EOF | decoder/local-prefill capacity와 interference |
| LMCache/UCX | requested/completed KV bytes, P token-ms, semantic op, receiver/install residual, failure | endpoint controller/receipt | remote route의 실제 completion capacity |
| NCCL | collective kind/bytes/count, enqueue→CUDA completion, rank tail/arrival spread, async error/communicator health | existing `CudaCollectiveObserver`; installed version이 지원할 때만 NCCL RAS/profiler를 추가 | TP/DP collective pressure와 co-job interference |
| GPU/node | SM/memory activity, HBM, NVLink/PCIe traffic 또는 topology affinity | user-visible NVML/DCGM/topology source가 실제 allocation에서 지원되는 항목만 | compute·memory·node-I/O bottleneck 분리 |
| Slingshot endpoint | NIC별/TC별 pause, posted/non-posted blocked, packet, overflow, ECN, retry/timeout, support state | existing `CassiniEndpointSampler`의 explicit counter reader를 확장해 current endpoint aggregate와 새 per-NIC/TC vector를 함께 출력 | congestion/fault advisory, rail imbalance와 causal attribution |
| topology | node/GPU/NIC/pair/dragonfly locality, route identity | frozen placement receipt | demand가 어느 shared resource를 통과하는지 |

[NCCL RAS/profiler](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/troubleshooting/ras.html)는 version-dependent다. 현재 Perlmutter module의 NCCL version과 plugin availability를 native preflight에서 확인하고, 없으면 기존 CUDA-event observer를 쓴다. 최신 기능을 설치됐다고 가정하거나 container/root로 가져오지 않는다. [HPE Cassini counter guide](https://cpe.ext.hpe.com/docs/24.11/getting_started/HPE-Cassini-Performance-Counters.html)의 counter는 NIC 관점이며 application 영향과 일대일이 아니므로 per-NIC vector와 support state를 유지한다.

### 9.3 scalar predictor가 아닌 causal resource graph

sample 하나는 다음 identity를 갖는다.

```text
(agent_epoch, node, pair, endpoint, route, communicator,
 local_sequence, local_window, source, support, value, uncertainty)
```

cross-host monotonic timestamp를 직접 빼지 않는다. 각 agent가 자기 clock에서 duration/rate를 계산하고 global coordinator는 bounded collection interval과 sequence를 사용한다. missing/unsupported/stale signal은 0이 아니다.

`fabric_pressure = max(counter)` 같은 scalar를 만들지 않는다. 대신 action `a`가 resource `k`에 주는 demand `d[r,a,k]`, observed completion capacity `C[k,t]`, confidence `q[k,t]`와 action 이후 service delta를 유지한다. bottleneck attribution은 다음처럼 counterbalanced intervention으로 갱신한다.

- 같은 offered load에서 local↔remote를 바꿨을 때 decoder/NCCL/LMCache/Cassini completion이 어떻게 이동하는가
- pair/placement만 바꿨을 때 어느 NIC/communicator/endpoint tail이 이동하는가
- real NCCL co-job의 collective rate를 바꿨을 때 KV completion과 inference SLO가 함께 변하는가
- LMCache concurrency만 바꿨을 때 NCCL tail과 receiver residual이 어떻게 변하는가

관측 상관관계만으로 physical switch를 지목하지 않지만, signal을 버리지도 않는다. policy는 attribution confidence가 낮으면 conservative action을 선택하고, analyzer는 application bottleneck과 physical attribution claim을 분리한다.

### 9.4 global objective와 decision

request `r`의 action은 단순 LOCAL/REMOTE가 아니다.

```text
a_r = (admit | defer | reject,
       P/D pair and placement,
       local-prefill | remote-prefill,
       transfer/semantic concurrency class,
       start/stagger epoch,
       failure domain)
```

global coordinator는 offered population 전체에서 다음을 최적화한다.

```text
maximize  Σ_r value[r] × Pr(SLO_complete | state, a_r) × useful_output[r]
          - reject/defer/failure/tail cost
          - Σ_k shadow_price[k] × demand[r,a_r,k]
          - activation/cache/control/oscillation cost

subject to
  Σ admitted demand[r,a,k] <= conservative completion capacity[k]
  tenant minimum service / max wait / reject budget
  exact lifecycle, immutable commit and failure receipt
  normal-load regression and control-overhead bounds
```

`endpoint_queue_lease`는 위의 `admit`와 동일한 의미가 아니다. global
reservation window가 끝났을 때 endpoint queue로 넘기는 shortcut은
endpoint의 실제 service credit을 확인하지 못하면 global capacity debt를
downstream 502로 바꾼다. 따라서 production scheme은 다음 two-phase contract를
사용한다.

```text
global decision
  -> endpoint service-lane admission/reservation (bounded credit + epoch)
  -> immutable pair×route commit
  -> first-response/EOF release 또는 reservation-failure receipt
```

reservation은 active sequence, endpoint request slot, local prefill, remote KV/
semantic-op, vLLM waiting service와 business deadline을 함께 확인해야 한다.
reservation이 없거나 epoch/identity가 맞지 않으면 route를 commit하지 않고
tenant-aware defer/reject로 종료한다. endpoint queue에 무조건 밀어 넣는
`queue_wait_ms` 연장은 허용 actuator가 아니며, reservation failure는 global
debt를 exactly once 되돌리고 다음 decision의 telemetry/fairness ledger에
반영한다.

각 resource shadow price는 queue depth 하나가 아니라 completion deficit, SLO risk, telemetry confidence와 marginal intervention result로 갱신한다. 이 공통 가격이 decoder GPU를 아끼려고 fabric을 무너뜨리거나, fabric을 아끼려고 decoder TPOT을 무너뜨리는 국소 최적화를 막는다.

### 9.5 multi-timescale actuator

| timescale | decision | 허용 actuator |
|---|---|---|
| request/fast | feasibility와 immutable commit | admit/defer/reject, pair×route, transfer/semantic credits |
| epoch/mid | service envelope와 contention response | per-pair concurrency, bounded traffic staggering, work-conserving tenant lane, probe/quarantine |
| deployment/slow | capacity와 topology | prewarmed pair active set, P:D ratio/placement candidate, coordinator shard ownership, telemetry sampling budget |

초기 구현은 user-space workload actuator만 사용한다. Slingshot switch/QoS를 재설정하거나 NCCL internals를 privilege로 변경하지 않는다. NCCL/Slingshot은 중요한 observation source이고, TEMPO actuator는 application admission, placement, concurrency와 launch timing이다. installed API가 명시적으로 지원되고 별도 experiment contract가 있을 때만 NCCL dynamic control을 독립 ablation으로 다룬다.

`P1PAIR+COJOB`의 co-job은 primary experiment에서 exogenous offered schedule로 고정한다. full TEMPO가 그 schedule을 몰래 줄여 이기는 것을 금지하고 자기 inference action만 바꾼다. 별도의 facility-scope ablation에서만 managed co-job staggering을 actuator로 열고 양쪽 job utility를 함께 계산한다.

### 9.6 completion-driven work conservation과 pair activation

- decoder, P, remote bytes, semantic op, NCCL collective와 NIC endpoint마다 conservative online completion envelope를 둔다.
- safe envelope 안에서는 low-priority work도 idle capacity를 빌릴 수 있다.
- reserve는 failure/overload 때만 활성화하고 사용되지 않으면 bounded borrowing한다.
- global timeout 하나로 모두 자르지 않고 tenant deadline, predicted finish와 reject budget을 함께 본다.
- pair activation benefit은 `incremental SLO work - cache/activation/shared-fabric externality`로 계산한다.
- activation이 assignment만 바꾸고 completion capacity를 늘리지 않으면 scale win이 아니다.

v30에서 이 원칙의 첫 concrete actuator를 닫았다. soft-shadow-price v2가
request를 work-conserving하게 유지하더라도, 선택 후보가 resource-specific
cross-layer action target을 초과하면 `cross_layer_resource_envelope`를 현재-state
scale basis로 만든다. 그 순간에만 prewarmed spare pair 후보를 같은 atomic
decision에 열고, spare pair가 실제로 선택됐는지와 이후 completion/SLO work가
늘었는지를 receipt에서 분리한다. 이는 queue fraction이나 scalar
`fabric_pressure` threshold를 바꾼 것이 아니며, v30 native에서 utility를
증명하기 전까지는 mechanism/headroom evidence로만 취급한다.

v30 native가 요구하는 다음 v31 mechanism은 pair-local scaling의 확장이
아니다. 동일한 source epoch/topology/communicator를 공유하는 pair들의
NCCL·LMCache·Cassini envelope를 node→pair→global로 집계하고, 다음을 별도로
유지해야 한다.

1. shared-fabric remote KV/semantic-op concurrency budget과 pair-local decoder/endpoint budget
2. shared externality가 높을 때의 global remote dispatch staggering 및 tenant-value-aware defer/admit
3. shared bottleneck이면 spare pair를 열지 않고, pair-local bottleneck/failure일 때만 spare pair를 여는 activation rule
4. global budget이 실제 action을 바꾼 request 수, completion capacity, SLO-goodput와 fairness receipt

이 vector는 scalar `fabric_pressure`로 축약하지 않는다. v31은 shared
resource별 capacity/confidence/shadow price를 receipt에 남기고, app-only와
network-only가 같은 externality 아래 어떤 layer를 놓치는지 비교해야 한다.

### 9.7 failure와 recovery

Candidate I의 endpoint delta circuit과 Candidate C의 exact failure receipt는 재사용한다. 여기에 NCCL communicator/peer health, Cassini retry/timeout delta와 agent loss를 별도 failure source로 추가한다. 하나의 layer failure를 전체 pair failure로 자동 확대하지 않고 failure-domain graph로 scope한다. explicit newer-sequence PROBE만 회복을 허용하며, normal state에서 reserve가 idle capacity를 버리지 않아야 한다.

### 9.8 hierarchical scale

Perlmutter 전체에서 single Python lock/controller가 모든 pair를 poll하는 구조는 목표가 아니다.

1. node agent는 raw telemetry를 local bounded envelope로 축약한다.
2. pair agent는 P/D/NCCL/LMCache 상태와 failure domain을 조정한다.
3. shard coordinator는 topology group 또는 service pool 단위 admission budget과 shadow price를 유지한다.
4. global coordinator는 shard 간 business budget, placement와 capacity만 조정한다.
5. telemetry fan-in은 periodic full snapshot + event-driven delta를 혼합하고 overload 때 sampling traffic 자체를 bound한다.

scale metric은 pair 수에 따른 decision p50/p99, messages/bytes per second, stale fraction, coordinator CPU/memory, failover convergence, oscillation과 achieved utility다.

### 9.9 새 구조 후보로 세기 위한 최소 조건

다음 중 하나라도 없으면 G/I의 threshold variant로 취급한다.

1. NCCL 또는 Cassini vector와 application completion을 provenance-safe state plane에서 실제로 결합한다.
2. cross-layer signal을 하나의 scalar penalty가 아니라 resource별 capacity/confidence로 유지한다.
3. measured completion/service envelope가 admission, route, concurrency 또는 placement를 실제로 변경한다.
4. tenant reject/defer/failure cost와 minimum service가 global objective/gate에 포함된다.
5. action이 한 resource의 개선을 다른 resource의 tail 악화로 전가하는지 analyzer가 검증한다.
6. node→pair→shard/global hierarchy와 bounded telemetry overhead가 구현된다.
7. full TEMPO, app-only, network-only, predictor와 strongest fixed를 같은 offered population에서 비교한다.

---

## 10. prior work는 component map이지 TEMPO의 가치 상한이 아니다

2026-08-22 기준 primary paper/official documentation을 다시 확인했다. systems contribution은 “논문 A에 routing이 있고 B에 scaling이 있으니 남은 feature가 작다”는 집합 차감으로 평가하지 않는다. 중요한 질문은 **누가 Perlmutter급 shared HPC에서 이 layer들을 실제로 연결하고, coherent control contract와 scalable implementation을 만들고, moving contention/failure 아래 end-to-end utility를 증명했는가**다.

| system/source | 강한 local loop 또는 substrate | TEMPO가 받아야 할 baseline/lesson | TEMPO 전체-loop와의 차이 |
|---|---|---|---|
| [DistServe, OSDI'24](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | P/D 분리, TTFT/TPOT goodput-oriented placement/resource planning | P/D topology와 goodput planning baseline | runtime NCCL/Cassini/endpoint/business joint feedback loop는 평가 단위가 아님 |
| [Mooncake](https://arxiv.org/abs/2407.00079) | production KV-centric disaggregation, SLO scheduler, overload early rejection | KV locality, production trace와 SLO/reject baseline | local-prefill escape부터 NCCL/Slingshot co-job까지 한 HPC control loop는 아님 |
| [P/D-Serve](https://arxiv.org/abs/2408.08147) | 매우 큰 xPU 규모, dynamic P:D organization/ratio와 reject forwarding | scale architecture와 dynamic pool baseline | TEMPO는 cross-layer measured state와 business utility로 runtime actuation을 닫고 실제 HPC implementation을 보여야 함 |
| [Kairos](https://arxiv.org/html/2607.02043) | vLLM request-level decode-local prefill deflection, TBT-safe chunk schedule | 가장 강한 request-local compute-aware baseline | endpoint failure, NCCL/fabric state, pair/shard capacity와 tenant utility를 공동 제어하지 않음 |
| [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/dev/knowledge-base/modular-components/planner/overview) | KV/load-aware routing, runtime xPyD, TTFT/ITL SLA replica planning | production planner/scale baseline | Perlmutter Cassini/NCCL signal과 same-path local/remote actuation의 native evidence가 TEMPO의 추가 검증점 |
| [NetKV](https://arxiv.org/abs/2606.03910) | network cost oracle를 쓰는 decode-instance selection, 64-GPU simulator | network-aware request routing direct baseline | oracle/greedy routing을 넘어 measured endpoint+collective+business resource loop와 actual deployment가 필요 |
| [TopKV](https://arxiv.org/abs/2607.28633) | topology-aware transport selection, pipelined KV movement | topology/transport-aware baseline | projected/component 중심이며 TEMPO는 frozen transport 위 runtime contention orchestration과 native utility를 검증 |
| [MRC](https://arxiv.org/html/2606.18170v1) | packet multipath, receiver-advertised bounds, host backpressure, probe/failover | receiver credit, service compensation와 probe design | transport가 아니라 request/pair/job layer의 global utility controller이며 Slingshot을 privilege로 변경하지 않음 |
| [NCCL RAS/profiler](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/troubleshooting/ras.html) + [HPE Cassini counters](https://cpe.ext.hpe.com/docs/24.11/getting_started/HPE-Cassini-Performance-Counters.html) | communicator health/collective events와 NIC-level counters | TEMPO state plane의 primary sensor substrate | telemetry 자체는 admission/placement/business orchestration system이 아님 |

MRC의 privileged controller API나 Slingshot switch configuration은 사용하지 않는다. TEMPO는 application에서 허용되는 관측과 actuator로도 충분히 강한 system을 만드는 것이 목표다. [vLLM disaggregated-prefill documentation](https://docs.vllm.ai/en/latest/features/disagg_prefill/)이 disaggregation 자체는 throughput을 본질적으로 늘리지 않는다고 명시하는 점은 오히려 global orchestration의 필요성을 강화한다. 어느 data path를 보유했다는 사실과 전체 workload에서 그 path를 언제/얼마나 사용할지를 최적화하는 문제는 다르다.

### 10.1 TEMPO의 target systems contribution

final gate를 통과하면 contribution은 작은 route heuristic이 아니라 다음 다섯 묶음이다.

1. **Perlmutter-native cross-layer observability**: vLLM, LMCache/UCX, NCCL, GPU/topology, per-NIC Cassini와 business state를 provenance-safe하게 결합한 state plane
2. **causal bottleneck market**: heterogeneous signal을 resource capacity/confidence/shadow price로 변환하고 bottleneck migration을 joint decision에 반영하는 mechanism
3. **global business orchestration**: pair placement/scaling, local/remote route, admission, concurrency/staggering와 failure recovery를 offered-population utility 아래 atomic하게 제어
4. **hierarchical production implementation**: node/pair/shard/global 구조, bounded telemetry/control overhead, stale/failure storm 안정성
5. **actual-system evidence**: real vLLM P/D + official LMCache + real NCCL + Slingshot에서 fixed, predictor, Kairos/NetKV-like와 app-only ablation을 이기는 결과

각 항목의 primitive가 일부 존재해도 이 전체를 구현하고 검증하는 것은 독립적이고 큰 systems contribution이다. 다만 “first”라는 단어는 최종 related-work audit 뒤에만 사용하고, 현재 성능을 얻기 전에는 target contribution과 achieved contribution을 구분한다.

### 10.2 최종 headline candidate

> **TEMPO-GO is a hierarchical, business-aware cross-layer orchestrator for disaggregated LLM inference on shared HPC systems. It fuses vLLM service, LMCache/UCX completion, NCCL collective, GPU/topology, and Cassini/Slingshot endpoint state into causal resource envelopes and jointly controls admission, P/D placement/scaling, local/remote execution, communication concurrency, and failure recovery.**

이 문장은 research target이다. final result 문장에는 반드시 실제 scale, baseline, workload와 measured improvement 숫자를 붙인다.

---

## 11. falsifiable hypothesis ledger

| hypothesis | 현재 상태 | 필요한 다음 증거 |
|---|---|---|
| H1: local/remote winner는 contention state에 따라 이동한다 | `SUPPORTED` | software/model 변경 시 fixed-path reconfirmation |
| H2: queue/GPU-only state는 remote completion ceiling을 충분히 설명하지 못한다 | `SUPPORTED FOR CURRENT WORKLOAD` | endpoint service receipt와 Kairos-like direct ablation |
| H3: current application-only joint admission이 fixed/predictor보다 utility를 개선한다 | `NOT SUPPORTED` | G/I branch negative로 보존; 숫자 tuning 중단 |
| H4: failure circuit/survivor lane이 normal 손실 없이 high-priority utility를 보호한다 | `MECHANISM OBSERVED, UTILITY FAILED` | failure baseline 대비 SLO-goodput ≥15%, normal ≤3% regression |
| H5: proactive pair activation이 실제 incremental service를 만든다 | `NOT SUPPORTED` | asymmetric native phase에서 activation benefit과 no-flap |
| H6: NCCL/LMCache/Cassini/application signal을 결합하면 single-layer sensor보다 병목 이동을 더 정확히 예측한다 | `UNTESTED AS A JOINT LOOP` | real NCCL+KV intervention의 prediction/calibration, false-positive/negative 분석 |
| H7: cross-layer joint actuation이 app-only/network-only보다 utility를 개선한다 | `UNTESTED` | `P1PAIR+COJOB`와 `TP8-CROSSNODE` native ablation |
| H8: control plane이 many-pair scale에서 안정적이다 | `UNPROVEN` | hierarchical 2→32→hundreds/thousands pair stress + larger native campaign |
| H9: business-aware global policy가 aggregate gain을 tenant starvation으로 만들지 않는다 | `UNPROVEN` | per-tenant SLO-goodput/minimum-service/reject-budget gate |

---

## 12. baseline, metric과 성공 gate

### 12.1 mandatory arms

1. `ALWAYS_LOCAL`
2. `OFFICIAL_LMCACHE_ALWAYS_REMOTE`
3. `PREDICTOR_ONLY`
4. `QUEUE_GPU_ONLY` 또는 exact Kairos-like implementation
5. `NETWORK_REQUEST_ONLY`: topology/congestion oracle를 쓰되 global admission/NCCL-business actuation이 없는 NetKV-like baseline
6. `APP_GLOBAL_ONLY`: 동일 TEMPO controller에서 NCCL/Cassini input과 cross-job actuator만 제거한 ablation
7. `TEMPO_GO_CROSS_LAYER`: full state plane + global joint actuation

같은 topology/GPU budget/server lifecycle/request trace/cache namespace를 사용한다. queue-GPU가 native process failure면 latency sample을 만들지 않고 failure receipt로 남긴다. final performance paper에서 queue/GPU comparator가 계속 execution failure라면 이를 숨기지 않고 comparison limitation 또는 robustness result로 사전 정의한다.

### 12.2 correctness hard gate

- stream/output/token/text digest 100%
- exact terminal state 1/request
- route/commit/profile/workload provenance 100%
- hidden recompute, silent fallback, same-ID retry 0
- unreceipted timeout/failure 0
- terminal queue/inflight/owned resource residual 0
- credit underflow/leak/double release 0
- stale/partial/identity mismatch fail-closed
- failure receipt에 released work/quarantine/recovery identity 존재
- NCCL collective count/type/bytes/result와 communicator identity exact
- Cassini node/NIC/TC/support/sequence identity exact; missing-to-zero 0
- telemetry batch에 mixed epoch/profile/topology 0
- actuator 이후 resource ownership과 business ledger exactly once close

하나라도 실패하면 headline performance를 계산하지 않는다.

### 12.3 original primary gate

- strongest fixed 대비 pooled E2E median ≥10% 개선
- predictor-only 대비 E2E median ≥5% 개선
- request 또는 output-token goodput strongest fixed 대비 ≥5% 개선
- paired E2E win overall ≥75%, 각 workload group ≥60%
- 각 group E2E p99/TPOT p99 regression ≤5%
- worst paired E2E regression ≤100 ms
- selected local/remote가 반대 measured counterfactual보다 각각 median ≥5% 유리
- offered population과 tenant fairness gate 유지

### 12.4 robustness alternative

모두 만족해야 한다.

- normal-load median/goodput regression ≤3%
- overload p99 또는 goodput/SLO-goodput ≥15% 개선
- fatal failure, unclosed terminal과 queue-timeout을 제거하거나 유의미하게 감소
- 모든 tenant starvation 0
- tenant별 minimum service, max wait와 reject/defer budget 통과
- failure/recovery/correctness invariant 통과

### 12.5 fairness/scaling/overhead

- tenant별 offered/admitted/completed/rejected/failed 수
- tenant별 TTFT/TPOT/E2E SLO-goodput와 output-token goodput
- raw dominant service units와 weighted debt를 별도 보고
- minimum service fraction, maximum wait, starvation, Jain fairness
- pair activation/deactivation, residency, activation benefit, no-flap
- telemetry collection span, refresh timeout, admission CPU p50/p99
- stale/missing/mixed identity count
- NCCL collective latency/rank-tail, LMCache completion과 NIC-vector action-conditioned calibration error
- full TEMPO 대비 `APP_GLOBAL_ONLY`/`NETWORK_REQUEST_ONLY` incremental utility
- node/pair/shard별 telemetry messages/bytes, coordinator CPU/memory와 failure convergence

Candidate I의 CPU microbenchmark는 baseline control-plane p50/p99 170.748/236.081 µs, I 180.226/264.013 µs다. 이는 control overhead evidence일 뿐 native latency 대체값이 아니다.

### 12.6 cross-layer candidate는 CPU latency model로 죽이지 않는다

- CPU replay는 lifecycle, identity, fairness accounting, failure state machine, hierarchical scale와 control overhead를 검사한다.
- G/I의 frozen duration/profile 모델에서 performance gate를 실패한 사실은 G/I에만 적용한다.
- NCCL/Cassini/LMCache가 동시에 작동할 때 생기는 physical service coupling은 CPU replay에 없으므로, 새 cross-layer candidate를 CPU-predicted median으로 promotion/stop하지 않는다.
- native GO 조건은 unit/schema/correctness 통과, preregistered real-workload intervention, telemetry support receipt와 actuator headroom이다.
- telemetry-only normal overhead target은 ≤3%, request decision p99 target은 ≤5 ms, stale/mixed identity는 0으로 두고 exact sampling budget은 native preflight 뒤 freeze한다.
- coupled contention에서 full TEMPO는 `APP_GLOBAL_ONLY`와 `NETWORK_REQUEST_ONLY` 대비 SLO-goodput/goodput ≥5% 또는 p99 ≥10%의 incremental gain을 보여야 cross-layer mechanism이 유효하다.
- native discovery 뒤 code/profile/workload/analyzer를 바꾸면 새 validation contract와 새 allocation을 사용한다.
- final performance claim은 frozen independent native result만 사용한다.

---

## 13. 앞으로 해야 할 일

### Phase 0 — historical branch closure: 완료

- C4 route-only negative 보존
- held-out workload/contract identity 보존
- latest native integration/failure receipts 보존
- Candidate G/I current-source replay와 strict CPU negative audit 보존
- `native_performance_negative_proven=false` 경계 보존
- G/I native promotion STOP

이 단계에서 같은 profile을 다시 돌리거나 threshold를 바꾸는 작업은 없다. 단, 이것은 cross-layer TEMPO campaign의 STOP이 아니다.

### Phase 1 — 4-node native capability/state-plane preflight

승인된 4-node/4-hour interactive allocation에서 다음을 한 번에 확인한다. 이 단계는 성능 arm 비교가 아니라 사용할 수 있는 production signal의 contract를 만드는 단계다.

1. native PyTorch/CUDA/NCCL/vLLM/LMCache/UCX/libfabric/CXI module과 commit identity
2. NCCL version, RAS socket/monitor/profiler plugin 지원 여부; unsupported면 기존 CUDA observer fallback
3. node별 4 Cassini NIC와 `CassiniEndpointSampler` counter support/ambiguity/missing matrix
4. GPU↔NIC↔NUMA↔node↔P/D pair↔communicator topology receipt
5. vLLM scheduler, LMCache semantic completion, NCCL, Cassini sample rate/overhead/freshness
6. privilege 없이 읽을 수 없는 신호는 `unsupported`로 기록하고 우회하지 않음

산출물은 새 root의 immutable `cross_layer_capability_receipt.json`, support matrix, exact commands/log/SHA다. 현재 `pd_global_telemetry.py` schema를 몰래 확장하지 않고 새 schema/version을 만든다.

### Phase 2 — controller 전 native causal characterization

full controller를 만들기 전에 fixed action으로 signal과 actuator의 headroom을 확인한다.

1. `P2PAIR`: current C1/C2/P_ONLY/C3와 pair asymmetry를 fixed-path ABBA로 재확인
2. `COMPONENT-ATTR`: official LMCache NIXL/UCX writes와 real NCCL all-reduce를 같은 interval에 실행
3. `P1PAIR+COJOB`: actual inference와 independent NCCL+LMCache co-job을 동시에 실행
4. 가능하면 `TP8-CROSSNODE`: inter-node TP collective와 KV transfer direct coupling 측정
5. local/remote, LMCache concurrency, NCCL offered work, placement 중 한 actuator만 바꾸는 counterbalanced intervention
6. vLLM/LMCache/NCCL/Cassini raw vector와 completion delta를 같은 experiment identity로 저장

GO 조건은 (a) 적어도 두 workload state에서 strongest action이 달라지고, (b) app-only sensor가 놓치는 state를 cross-layer signal이 식별하며, (c) 허용 actuator가 utility를 바꿀 measurable headroom을 보이는 것이다. 특정 Cassini counter가 움직이지 않아도 application completion coupling이 있으면 sensor redesign으로 진행한다. 모든 action에 headroom이 없으면 controller를 만들기 전에 workload/topology 또는 research claim을 재검토한다.

현재 상태는 component headroom, observer production, actual one-pair vLLM/LMCache global lifecycle, v30 native discovery와 v31 shared-budget CPU implementation까지 GO다. `57416103` retry6은 request-triggered observer freshness, decision provenance, joint commit sequence identity, 실제 route/reject action을 닫았으며, 249개 decision에서 NCCL·LMCache observer state가 supported였다. 이후 v27/v28 seven-arm discovery에서는 5개 fixed/network arm이 276/276 valid complete였지만 app-global/TEMPO는 각각 stale source guard와 node-2 shared-memory startup stall로 utility window를 닫지 못했다. v29f는 execution-only로 격리됐고, v30은 lifecycle/correctness/coverage까지 닫혔지만 performance gate가 false였다. v30에서 134 complete/142 reject와 C3 both-hot 113 reject가 발생했고, pair activation 1건이 shared-fabric externality를 줄이지 못했다. 따라서 이것은 TEMPO 전체의 negative가 아니라 **pair-local scaling만으로 shared NCCL/LMCache/fabric bottleneck을 풀 수 없다는 native mechanism evidence**다. v31은 이 gap을 shared remote budget/concurrency/stagger와 pair-local failure separation으로 구현했고 CPU gate를 통과했다. v38/v39는 hierarchical global fan-in과 telemetry-aware frontier를 구현했으며, 현재는 v45 immutable snapshot native validation 단계다.

> 현재 source의 historical native 기준은 v46 immutable snapshot이었다. v45/v46
> 실행 영수증은
>
> 이 문서의 현재 continuation에서는 v47 Slingshot/OFI launch-fix snapshot이
> native 실행 기준이며, v45/v46은 historical execution receipt다.
> capability/source-boundary history로만 사용하며, v46 distributed frontier는
> raw candidate를 global로 보내지 않고 pair→shard에서 bounded frontier로
> 줄인 뒤에도 최종 business/fairness/cross-layer policy commit을 global에
> 남긴다.

### Phase 3 — hierarchical state plane와 TEMPO policy 구현

1. `tempo-go-cross-layer-telemetry-v1` node/pair batch와 support/freshness/provenance schema
2. vLLM, LMCache, NCCL observer/RAS와 Cassini adapter
3. node agent의 completion envelope/confidence estimator
4. pair agent의 resource/failure graph와 action-conditioned calibration
5. shard/global coordinator의 shadow price, business utility와 work-conserving admission
6. pair/route/concurrency/staggering actuator와 immutable plan receipt
7. full/app-only/network-only ablation profile을 같은 code path에서 feature flag로 freeze
8. tenant reject/defer budget, minimum service, normal regression, failure utility와 overhead gate preregistration

G/I 코드의 lifecycle/failure/fairness invariants는 재사용하되 decision mechanism은 별도 candidate ID, source SHA와 contract를 사용한다.

현재 구현에는 `tempo/cross_layer_observer.py`의 strict atomic observer, router의 stale/complete/wrong-epoch fail-closed 소비, `tempo/pd_global_profile.py`의 one-/two-pair topology, `vllm_lmcache_tempo_go_p1pair_node.py`의 actual one-pair vLLM/LMCache path, 그리고 같은 allocation에서 official NCCL/LMCache co-job과 두 TP4 P/D node를 분리 실행하는 `run_tempo_go_p1pair_cojob_in_allocation.sh`가 들어 있다. retry6은 observer path export까지 포함해 native integration/correctness를 닫았다. 이후 v29는 `soft_shadow_price_v2`와 critical hard guard를, v30은 resource-envelope 기반 spare-pair activation을 추가했고, v31은 shared remote budget/stagger, compatible-group cache와 compact decision provenance를 추가했다. 현재 source 기준 v31 관련 108 global/C5 node tests, cross-layer/router 36 tests와 py_compile이 통과했다. v30 native는 7/7 arm, 10,000-block co-job correctness와 C5 end coverage를 닫았지만 TEMPO utility/fairness/scale superiority는 증명하지 못했다. 다음 단계는 v31b native discovery다.

### Phase 4 — CPU correctness와 control-plane scale gate

1. profile/schema/source/run-contract identity test
2. exact lifecycle, reject, failure, quarantine, PROBE, release tests
3. mixed epoch, missing/unsupported counter와 NCCL peer failure tests
4. business fairness, work conservation과 reject/defer accounting
5. deterministic shadow-price/convergence/no-oscillation test
6. node/pair/shard failure와 coordinator failover
7. logical 2/4/8/16/32/64/128 pair, 이어서 256/512/1,024 agent emulator stress
8. decision CPU/memory, telemetry messages/bytes, collection span, stale rate와 fan-in scaling

CPU stage는 correctness/scale STOP gate다. lifecycle, identity, fairness 또는 overhead가 실패하면 native로 가지 않는다. CPU duration model이 native latency win을 예측하지 못한다는 이유만으로 cross-layer candidate를 죽이지 않는다.

### Phase 5 — 4-node native discovery

사용자가 허용한 4-node/16-GPU/최대 4시간 interactive allocation을 실험 세션으로 유지하고, login node가 아니라 그 allocation 안에서 다음을 실행한다.

1. preflight와 fixed-path workload validity
2. same topology/server lifecycle/cache/tenant trace에서 seven-arm balanced order
3. `P2PAIR`와 `P1PAIR+COJOB`을 primary profile로, `TP8-CROSSNODE`는 시간이 허용되면 topology stress로 실행
4. telemetry-only arm으로 normal overhead 측정
5. full TEMPO가 app-only/network-only 대비 병목을 다른 layer로 전가하지 않았는지 즉시 analyzer로 확인
6. correctness 실패 시 performance analysis 중단
7. 새 root에 raw request, NCCL, Cassini, LMCache, controller/failure receipt와 SHA 저장

discovery가 candidate headroom을 보이면 code/profile/workload/analyzer와 exact command를 freeze한다. 보이지 않으면 negative로 종료한다.

### Phase 6 — independent native validation

- 새 승인 allocation
- frozen contract 그대로 한 번 실행
- primary 또는 robustness/fairness gate를 사후 완화하지 않음
- queue-GPU failure와 TEMPO reject를 completion으로 치환하지 않음
- per-resource bottleneck migration, tenant utility와 cross-layer ablation을 함께 보고
- pass면 §10.1의 end-to-end systems contribution, fail이면 exact failed mechanism과 reduced claim

### Phase 7 — production/HPC scale evidence

hierarchical implementation과 1,024-agent stress는 Phase 4에서 먼저 만든다. 4-node positive mechanism이 재현된 뒤 더 큰 actual deployment는 별도 authorization으로 시작한다.

- pair/tenant/job 수를 독립적으로 늘리며 offered work를 통제
- topology group별 shard와 cross-shard placement/business budget
- telemetry fan-in, coordinator failover, stale/failure fan-out과 control overhead
- concurrent NCCL collective, KV movement와 inference diversity
- larger deployment에서 no central bottleneck/no oscillation, utility scaling

목표는 4-node heuristic이 아니라 Perlmutter-scale architecture다. 논문에는 native mechanism, many-agent scale와 larger field evidence의 범위를 각각 명시하고, 확보한 evidence보다 작은 claim으로 스스로 깎지도 크다고 과장하지도 않는다.

---

## 14. Perlmutter 실행·안전 계약

[Perlmutter architecture](https://docs.nersc.gov/systems/perlmutter/architecture/)상 GPU node는 4×A100과 4×Slingshot 11 Cassini NIC를 가진다. [NERSC interactive documentation](https://docs.nersc.gov/jobs/interactive/)의 interactive 최대 node 수는 4다. 이 연구의 native envelope는 사용자가 허용한 4 nodes / 16 GPUs / 최대 4시간이다.

### login node에서 허용

- bounded source/artifact inspection
- 문서/patch 작성
- 작은 unit/schema/static test
- 가벼운 SHA/analyzer 확인

substantial replay, vLLM/LMCache server, inference traffic와 GPU workload는 compute allocation 안에서 수행한다. [NERSC coding-agent guidance](https://docs.nersc.gov/development/coding-agents/)에 따라 broad traversal와 unsupervised resource use를 금지한다.

### interactive allocation 규칙

- concrete run 전에 frozen candidate/workload/contract와 사용자 승인을 확인한다.
- 한 allocation만 사용하고 그 안에서 server lifecycle를 유지한다.
- `srun`마다 GPU resource를 명시하고 CPU-only step이 allocation GRES를 잘못 상속하지 않게 한다.
- automatic submit/cancel/retry loop, background watcher, repeated login/SSH loop를 만들지 않는다.
- 새 candidate는 artifact 수집/teardown 30분을 제외한 시간이 충분할 때만 시작한다.
- 기존 result root를 overwrite하지 않는다.

### 절대 금지

- Shifter, Apptainer, Podman, Docker, `--image`, udiRoot
- `sudo`, `su`, root shell/ownership 변경
- `setcap`, `CAP_NET_ADMIN`, privileged MRC/NIC controller
- `/etc`, `/usr`, `/opt` write
- physical switch/NIC configuration 변경
- `/`, `/global`, `/pscratch` 등 shared top-level recursive traversal
- dirty worktree의 unrelated change 삭제/reset/stage

`udiRoot.conf must be owned by user root`가 나오면 ownership을 고치지 않는다. native-only launcher contract 위반으로 기록하고 중단한다. `exit 139`는 crash receipt로 기록하고 privilege 우회나 blind retry의 이유로 쓰지 않는다.

---

## 15. authoritative artifact index

| evidence | path | SHA/status |
|---|---|---|
| original goal | Codex attachment identical copies | `4f9650280307d6c352ada284b1fb7137e4f70c0189e6ad341104671cc1647a4a` |
| C4 terminal negative | `results/tempo_pd_c4_semantic_credit_epoch_candidate_v7_job_57362947/negative_conclusion_analysis_v2.json` | `c8cb985aba33724b22c16d1501d9cdbd057d95ea5231b64de23a88d2572cd1f3` |
| held-out manifest | `results/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json` | `6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c` |
| held-out workload | `results/tempo_go_c5_heldout_output128_v1/workloads/validation.jsonl` | `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33` |
| held-out validator | `results/tempo_go_c5_heldout_output128_v1/manifest_validation.json` | `f00157c5f237c7a271197e499046e0e2a9884881cffeca46554accd015933fd0` |
| latest source-rebound native contract | `results/tempo_go_c5_r8_16_20_20_contract_v3/native_run_contract.json` | file `002ee5424c9779b22d2cc622cb9143227f8370d03d6b22d0f3c9a560f153e481`; fingerprint `7691d005cad942c26a9a8792cf1487431ce5c4f7abe43ebb7b409a2fef5a854e` |
| latest source-rebound native analysis | `results/tempo_go_c5_r8_16_20_20_native_job_57409956_v3/native_five_arm_analysis.json` | `b7e302ab1f893310602b491a8971138d3f4b3cd7fa906b4f7ce05848ac305f45`; performance false |
| native C failure raw | `results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/tempo/tempo_go_c5_discovery/raw.json` | `c61626d6cef2b7353e0ec8a21609a9bc3b72ea6e4ed240ff5de2216cf9292124` |
| native C raw-backed analysis | `results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/native_c_analysis_raw_backed_v1.json` | `579f92d38140f0f7ccb31f18a19ce9c9670ea5b3371ba48e99cf7850dbd3a1ac` |
| Candidate G current-source contract | `results/tempo_go_c5_candidate_g_tenant_reservation_v1/native_run_contract_current_source_v1.json` | file `415d53329698870dd8b7c2d558a5f73d1a23311eac434d7dc835e04b55598478`; fingerprint `464068dcfbd47e574e2d3fe22a9c92dd5dd241583424297096b4e2959132f659` |
| Candidate G contract-bound replay | `results/tempo_go_c5_candidate_g_tenant_reservation_v1/heldout_cpu_replay_contract_bound_v1.json` | `7347c148c8102d44e976e3a82807f580639eaeccc7b1801615a26b5e57ca637f` |
| Candidate I contract | `results/tempo_go_c5_candidate_i_telemetry_survivor_v1/native_run_contract.json` | file `cab8942c74563552642278eb3c0f6aeb1fcbc7a72e3fa1a67461df230d538d5d`; fingerprint `ccb424d40e9fbd47060416599ee6f7351a68993c7c500b93e103f65439977826` |
| Candidate I normal replay | `results/tempo_go_c5_candidate_i_telemetry_survivor_v1/heldout_cpu_replay_contract_bound_v1.json` | `8e2819104d5bc02413e07c5c61245a7968990f534205259c2acc14d475c85f4e` |
| Candidate I failure replay | `results/tempo_go_c5_candidate_i_telemetry_survivor_v1/heldout_cpu_replay_telemetry_failure_contract_bound_v1.json` | `98d0580cefbc3f2c57573b6670876797719605c4721bb2d47c4424f8574df46b` |
| Candidate I overhead | `results/tempo_go_c5_candidate_i_telemetry_survivor_v1/control_plane_overhead_v1.json` | `b475e57710230ac77518c57eddfc77e947c773a1b67b2b9db120e1432aceeadf` |
| strict audit script | `eval/sota_4node/audit_tempo_go_cpu_negative.py` | `5ecfbe04b3f5c02c91c449149acdd36b70843a74ee888336dd582c1a33f59897` |
| strict audit result | `results/tempo_go_c5_candidate_i_telemetry_survivor_v1/cpu_negative_audit_v1.json` | `aed33cc340c27e2688de0dfb001009182f37236b291ebdb25399e4bd78358925` |
| cross-layer native capability | `results/tempo_go_cross_layer_capability_57412204/capability.ndjson` | `e4309218756c8d345af4aa7f2e7147ae32171245eae48e8c00bb311d8fb50a89`; 4 nodes/16 GPUs, 4 NICs, CUDA/NCCL available |
| minimal official NIXL+NCCL correctness | `results/tempo_go_cross_layer_component_57412204/nixl_debug_min_v11.json` | `d7622c295e628de28136e364cb60126f8c773ace91ade119295ce511012df9ae`; correctness true |
| matched NCCL-only control | `results/tempo_go_cross_layer_component_57412204/nccl_only_control_v11.json` | `c0283b9c987632f0e010aa8fd690f9de36a0ca20348b36612e42dae396ca6d7b`; correctness true |
| official NIXL+NCCL contention | `results/tempo_go_cross_layer_component_57412204/nixl_nccl_contention_correct_v12.json` | `292bd70317c6bcbfa63f735c09a10727cf516d8fc2c1f64987ebb6da1238f363`; correctness true, component attribution only |
| corrected live NCCL/LMCache observer result | `results/tempo_go_cross_layer_observer_diag_57415597/result.json` | `2a9a3593b3582fc956f2862009d48c41098a2b0d19bfedbb9badc7cf3254b053`; 2 blocks, 4 pairs, 4 MiB/block verified, native observer integration; utility claim false |
| corrected observer active history | `results/tempo_go_cross_layer_observer_diag_57415597/history/observer-seq-000001-active.json` and `observer-seq-000002-active.json` | `90772a47d7be94f39a398358a620d659dbcc3c3a4691d69142ec435e573aaf9b` and `1fa2934185664dcf8d30edbd04a8e44777879cfdbae51d086da7f45297879d93`; immutable live windows |
| corrected observer terminal history | `results/tempo_go_cross_layer_observer_diag_57415597/history/observer-seq-000003-complete.json` | `eb69f66767beee71eb476f7d280a04ca5150e25223d9f7792d1c6e920d8a14ff`; terminal pointer intentionally rejected by router |
| cross-layer source-bound contract | `results/tempo_go_c5_cross_layer_contract_v10/native_run_contract.json` | file `31fc6fccd1fb1f468253d0eacae52a864a615811ceaf24c4eb1c3bba1fd9a922`; fingerprint `a33c44d536f58a4485a86395d3315ae7ae594f8a0ad255aa67b879e39a906eb4`; performance claim false |
| live-observer source-bound contract | `results/tempo_go_c5_cross_layer_contract_v11/native_run_contract.json` | file `05e6c9030f932aacf1a1a7cf420a272250e226a8b06f4d537481c3767b3656bf`; fingerprint `241ba4fb044005a894a021827c9ff3a08506c9511b791873747efd2ba59ea398`; `tempo/cross_layer_observer.py` included; performance claim false |
| live-observer allocation launcher contract | `results/tempo_go_c5_cross_layer_contract_v13/native_run_contract.json` | file `7295d6b0f5736d533f065215a89b14b4cafcc1b53080671fea45c555ac7f4789`; fingerprint `20957a82a37b42e159588fafcd26195b4f9a9b9998860db54d54f029fb69d83b`; bounded in-allocation launcher included; performance claim false |
| live-observer bootstrap contract | `results/tempo_go_c5_cross_layer_contract_v14/native_run_contract.json` | file `848b4806452400c4ac9ad34c1de102ba526c66b2ee3e0ba720e1fd95ae43d517`; fingerprint `09a4ba90140b537316d4552dbb53da32ea272e822bbcf354cbd11e1eb18452a4`; missing Slurm rank bootstrap fails fast; performance claim false |
| joint-actuation source-bound contract | `results/tempo_go_c5_cross_layer_contract_v15/native_run_contract.json` | file `1b2fb3ec779edcd11527324f7d574ae33a1098f18968c440a35299844c2bbe45`; fingerprint `66f4b2c1336e8d5a0717a8c52d082c3fef579b0128a4c5cfefe6cb1b982a6956`; current global orchestrator/endpoint/frontend source inventory; performance claim false |
| same-allocation co-job source-bound contract | `results/tempo_go_c5_cross_layer_contract_v16/native_run_contract.json` | file `06f1344bc9c154e62a155add49c9ee6529fd91497fff7dca12359dd79681ff2d`; fingerprint `e070cf5a7743b15fd7bfc36ad6617ea2d4b27d38d96faf265bcd9e4e762a119f`; bounded NCCL/LMCache co-job wrapper with intentional Slurm step overlap; not executed, performance claim false |
| same-allocation co-job v17 contract | `results/tempo_go_c5_cross_layer_contract_v17/native_run_contract.json` | file `da70b0a43a42905995e6d55f4e4c90fa50d73d545acf44998630dff756c66309`; fingerprint `87e5107d402e8bd20f3a4807e609207051d3be7468fd17c047d30cad226541ee`; current-source contract with co-job root override; performance claim false |
| same-allocation co-job v18 contract | `results/tempo_go_c5_cross_layer_contract_v18/native_run_contract.json` | file `2c9175ecd82d3efaf901989e86f1930792e091fdd87b8c76c90c354ec504131a`; fingerprint `b80d1f154e0a08506d526f24be8ac6b2e6d4d466a83fede881f53aac5f9597b6`; retry contract used by allocation 57415765; performance claim false |
| v18 native execution-only failure receipt | `results/tempo_go_cross_layer_native_57415765_retry1/execution_failure_receipt.json` | file `4c8304ee3da0e6198898ef16019a900124ae427e356073e09661cec16cd162fd`; local rc137/NIXL descriptor failure, downstream rc143 teardown, no co-job/observer binding; performance claim false |
| signal-safe co-job v19 contract | `results/tempo_go_c5_cross_layer_contract_v19/native_run_contract.json` | file `70b665050c485d301b81fc4316e91b1fd9a0c87e5e5e1b9f74b071a35e6e5461`; fingerprint `a8f1a6668f16ee6b69a4a03231b859461acd8eba6f826186825bd100e921ea6c`; node/co-job cleanup hardened; not executed, performance claim false |
| seven-arm cross-layer ablation contract | `results/tempo_go_c5_cross_layer_contract_v20/native_run_contract.json` | file `f1e1813f59b2668035301d3976f3d47579920939931bc9771fde6c23cb37c969`; fingerprint `1e0d540a3f7ddfaffe7a342d04ac305f6898cff52f682097f08324292d75fb41`; fixed/local/remote/predictor/queue + NETWORK_REQUEST_ONLY + APP_GLOBAL_ONLY + TEMPO; not executed, performance claim false |
| v20 launcher-fix execution-only receipt | `results/tempo_go_cross_layer_native_57415765_v20/execution_failure_receipt.json` | file `5e7db516dced8c5e6ec9e291f0b5bca507d5f1bdc93025fd83a2ff97ab5c5400`; co-job root precreation contract bug, local INT safety stop, corrected probe separately correct; performance claim false |
| corrected co-job probe result | `results/tempo_go_cross_layer_cojob_57415765_v20_probe/result.json` + `nccl_observer.json` | result `535949608aea0ca9bfee443b584ba7b659736e462489db1daaef657b9c1a691c`; observer `27940abc5e21ba95fe129cf32529bc2fa7bc13119552d959ff0781a6905bfe1e`; 3 blocks, 8 ranks, correctness true; component/observer evidence only |
| seven-arm launcher-fix v21 contract | `results/tempo_go_c5_cross_layer_contract_v21/native_run_contract.json` | file `7945bce4687b4a669bb57766dc3dd855f3d4e3e22a43e2e59387e1e5081d7d37`; fingerprint `4de61f4d4d779ffc39a86634b064e01f6da8bfe3bc3cea058b6d286baa96f475`; current co-job result-dir contract; not executed, performance claim false |
| v21 same-allocation concurrency failure receipt | `results/tempo_go_cross_layer_native_57415765_v21/execution_failure_receipt.json` | 3-block co-job completed in about 12 s before C5 measured lifecycle; correctness probe only, no concurrent contention or performance claim |
| seven-arm co-job-duration v22 contract | `results/tempo_go_c5_cross_layer_contract_v22/native_run_contract.json` | file `f1d507e6f880b5b1c93e5bbbc4604178ac35a985597df3d87777e8662fb9bbf9`; fingerprint `e06aa24d701e93e5546bdecc33d0dd3ed00290ad0a590248333221d3c5d36453`; 600-block co-job + active-observer readiness gate; not executed, performance claim false |
| v22 partial-overlap execution-only receipt | `results/tempo_go_cross_layer_native_57415765_v22/execution_failure_receipt.json` | file `c28237f664fd47d384ae6e563882207003558a66a0796bf6dd03615092fdf2fd`; readiness passed and local arm overlapped, but 600 blocks ended after 151 s before seven-arm campaign completion; performance claim false |
| seven-arm co-job-coverage v23 contract | `results/tempo_go_c5_cross_layer_contract_v23/native_run_contract.json` | file `fb7a57b9eeb47df6e9746c0a81e3c9449ccaedf69d92176a7148f8f37bd90e5a`; fingerprint `25a6c781ddca5a18d51adac633f4576b9c5efe63ed587499428f59ef4434af36`; 10,000-block co-job + start/end coverage gate; not executed, performance claim false |
| v23 co-job timeout execution-only receipt | `results/tempo_go_cross_layer_native_57415765_v23/execution_failure_receipt.json` | file `f95d26f19bd304eea46c974311907a5d51b7334d4d7a8308c3d882619fd1c58`; old 900 s timeout/20-minute step cap interrupted terminal binding; local partial overlap only, performance claim false |
| seven-arm co-job-time-limit v24 contract | `results/tempo_go_c5_cross_layer_contract_v24/native_run_contract.json` | file `a19fc0a971af032a687c4a2650c7481704101d27db044ae9d88f7f99ad2dbcb3`; fingerprint `f75cdb684056c5585ffeaee39f0da17d0daaca1ec4a3cd7be97a7f17943a19a7`; 10,000 blocks + 3,600 s timeout + 01:00:00 step + start/end coverage gate; not executed, performance claim false |
| v24 hot-contention execution-only receipt | `results/tempo_go_cross_layer_native_57415765_v24/execution_failure_receipt.json` | file `bcbd1b1498544344a4e7a10b81e2c1349b675ea156aef49b12edf9841b676f66`; 32 MiB/rank sustained co-load, 27 local completions in about 5 min, LMCache tail about 20 s; bottleneck/headroom evidence, no utility claim |
| seven-arm sustained-moderate v25 contract | `results/tempo_go_c5_cross_layer_contract_v25/native_run_contract.json` | file `87d765f2155077afefe6249fa14153a1f6a051ccf477f8ad50f8f0e057894e3e`; fingerprint `e3c743f316b66dba178a5eaad71914c3e646286f73bdac11bde1cad3e2274037`; 4 MiB/rank + 8 iterations + 1 MiB foreground + 0.10 s duty-cycle delay, 10,000 blocks, start/end coverage gate; not executed, performance claim false |
| v25 sustained-moderate execution-only receipt | `results/tempo_go_cross_layer_native_57415765_v25/execution_failure_receipt.json` | file `0ed30a8f27965d6c5db3d7c6e3086a305534106fae100f677935b1b461cb8b5c`; local full arm closed, remote arm too slow for 2,712-row seven-arm campaign; active observer and partial native evidence, no utility claim |
| short-slice workload manifest | `results/tempo_go_c5_cross_layer_short_slice_v4/tempo_go_workload_manifest.json` | file `89fe5b25c8ef0cce9c6d86abcb7178027839087fdb7c8bd1c97d2cc6ab0b868d`; 276 rows, all four contention streams, replicates 6/7 |
| short-slice identity-only profile rebind | `results/tempo_go_c5_cross_layer_short_slice_v4_profiles/profile_rebind_provenance.json` | file `3c239add0ae5134fdacae1c3fbc5329a8801f5c0b3a1152e12625961f7ed9e49`; endpoint `8776221a0bc0d932205c97e1a943a130cd206e73ce6b863fb6cce67c33f1a4cf`, global `040b2f133db8e9b8b94b7d74637f0faad7e54a8acaff80c69b8a87b4743b2423`; numeric measurements unchanged |
| seven-arm short-slice v26 contract | `results/tempo_go_c5_cross_layer_contract_v26/native_run_contract.json` | file `c5a3740314f17539792ff24df8fa297f834f4fd2771f813f49dd00fcd80e705a`; fingerprint `6dd21047418e1029bc6af96f2336ed12d272ee80f68c986cf9ceb5b9cc7d2f48`; executed discovery baseline, performance claim false |
| seven-arm short-slice v27 current-source contract | `results/tempo_go_c5_cross_layer_contract_v27/native_run_contract.json` | file `e5966d95a5cc2f36173f004a96a0aa877ae05cadb875d81b76cd27391d68c367`; fingerprint `6d37eb6eb171b2bfd49100087787caa597608dae8bda77fffa24b7a6681ce49e`; current source inventory, observer export, batch-sequence fix, 276-row short held-out primary, sustained-moderate co-job + start/end coverage gate; not executed, performance claim false |
| v28 safe-envelope global profile | `results/tempo_go_c5_cross_layer_short_slice_v4_profiles/real_tempo_go_profile_short_slice_v2_safe_envelope.json` | `6dda4c02452e90c1f950b25b64c237bac0e4b18daf5296c1365e4e7021b19b0d`; queue wait cap 8 s, profile identity bound to the same 276-row manifest |
| v28 executed seven-arm contract | `results/tempo_go_c5_cross_layer_contract_v28/native_run_contract.json` | file `a745434411d74e569e734564c685047da7329fa1e702086d98ee12f5e587e483`; fingerprint `da86c83a23b72e2dd655406820bbbbe749e4e3528252097916c9814c4116cc2b`; safe-envelope actuation source and same-allocation co-job bound; performance claim false |
| v28 native seven-arm analysis | `results/tempo_go_cross_layer_native_57415765_v28/native_five_arm_analysis.json` | `943df59ee7bc44aea3e3a7e4d5594dd712c8bc78ebcb3f7a0e4ee6bb1f972ae9`; all seven arms present, same workload/request count, performance claim false |
| v28 co-job binding | `results/tempo_go_cross_layer_native_57415765_v28/cross_layer_cojob_binding.json` | `5b450bbc8e093f975bd2880ed9a11a1c5e4a9af230655033cf188540b18523a1`; same allocation, C5 end covered, 10,000 blocks, `cojob_covered_c5_end=true` |
| v28 official LMCache/NCCL co-job result | `results/tempo_go_cross_layer_cojob_57415765_v28/result.json` + `nccl_observer.json` | result `1608a655ec831c49ade5322157448f5ab2973fea5ed2c1eec3c976cfbfe67901`; observer `3ed6a9f2749902edb913f3b5f00fe4b1f26b36b7982af0337db4238fed6bb9ce`; 10,000/10,000 correct, terminal observer sequence 10,001 |
| v29 soft-shadow-price profile | `results/tempo_go_c5_cross_layer_short_slice_v4_profiles/real_tempo_go_profile_short_slice_v3_soft_shadow_price.json` | `bb7a31b2b7e9badf2a98d6620cbfd526242cbbf7e94ba158d22fe32b256bc877`; `soft_shadow_price_v2`, critical threshold 2.0, queue wait 8 s |
| v29 current-source history contract | `results/tempo_go_c5_cross_layer_contract_v29/native_run_contract.json` | file `a56585dacff2c71f28882d9cc61a4c3b9ff8548d9ff505317282cd386710fda1`; fingerprint `341f5dbb1d72c5d0cf00b83ecf7003e67d07a90461266f9be91ad5676c30a8ac`; full verify history, performance claim false |
| v29b soft-shadow-price source contract | `results/tempo_go_c5_cross_layer_contract_v29b/native_run_contract.json` | file `af8ca46422fd3edf11eaf3a6d57c6de957b70311f7a8873513f87b8888aefb04`; fingerprint `1889e46442d9370a0d79ac907446a5fb606b723c8acce550658dd15606a899e6`; launcher source fix, performance claim false |
| v29f native execution-only receipt | `results/tempo_go_cross_layer_native_57415765_v29f/tempo/failure.json` | file `14d16889d5c14d0934ca26b910b81c2ebb3c6fcb6563d8c4f24780aa3a080351`; rc143 before first C5 measurement, no result/observer binding, performance claim false |
| v30 resource-envelope spare-pair contract | `results/tempo_go_c5_cross_layer_contract_v30/native_run_contract.json` | file `eaab9c4d70731ae68c59158b9143da11bf602387b1b00cdd356523f35e24a830`; fingerprint `0f2f25f2f9ba58c46d82868ed553c2491b8369113cddc2bcf01849e9366cf9a6`; full verify passed, cross-layer soft overage can activate a prewarmed spare pair, performance claim false |
| v30 native seven-arm discovery analysis | `results/tempo_go_cross_layer_native_v30_57423440/native_five_arm_analysis.json` | `c5897e6151b452cccecf6e86f3de4d62d74afa45334bd39efc9c3ae1bed4842e`; 7/7 arms valid, same 276-row workload, TEMPO 134 complete/142 reject/0 fail, performance claim false |
| v30 same-allocation binding | `results/tempo_go_cross_layer_native_v30_57423440/cross_layer_cojob_binding.json` | `ddc209a8170df16a6ac835a488f542e5e1bc890c60d5aaf2608ff18edc2bd39c`; 10,000-block co-job, C5 end covered |
| v30 official LMCache/NCCL co-job result | `results/tempo_go_cross_layer_cojob_v30_57423440/result.json` + `nccl_observer.json` | result `639bf0ff4ee8918f39bd3ac9ee7bfe0ba2dd0bfdc91125e29a23977399c03eb3`, observer `72dd7ebc3014b91e30f5229393888e863c5923d2c15358963a0fdae19c5ba258`; 10,000/10,000 correct, terminal sequence 10,001 |
| v31 shared-budget profile | `results/tempo_go_c5_cross_layer_short_slice_v4_profiles/real_tempo_go_profile_short_slice_v4_shared_budget.json` | `ed49fd3bac093e2bd74e1103e1474000c214fd959098774b7ae6ab84db6ac9dd`; `global_budget_v3`, explicit aggregate remote request/KV/semantic capacities |
| v31b current-source native contract | `results/tempo_go_c5_cross_layer_contract_v31b/native_run_contract.json` | file `eca07e359e934e50b2b37d8f66ece23e5e9dee9099d5c55c0bb787f33fd3f732`; fingerprint `9b3922220722a6218951edd049e20a683ec78085b3825d829a1b5725e662d80c`; 7-arm same-allocation co-job contract, full verify passed, not executed |
| v31b native environment failures | `results/tempo_go_cross_layer_cojob_v31b_57425033/cojob.stderr.log` and `results/tempo_go_cross_layer_cojob_v31b_retry_cpu32_57425033/cojob.stderr.log` | `4f78aad5df6429613582b3c079fdd8cc767c15add5dfd755e42d6a76911dd893`, `4cc4fa58cbdb176a24b6c292a9ed655fe49671bb498c18500954eab4e1a135ce`; no result/observer, execution-only CPU/step/interconnect failure |
| v32 endpoint-queue-lease native discovery | `results/tempo_go_cross_layer_native_v32_57425033` | fixed five arms 276/276 valid; app-global 173/103 with 24 HTTP errors; TEMPO 164/112 with 18 invalid resource-limit HTTP errors; no performance claim |
| v32 cross-layer co-job termination | `results/tempo_go_cross_layer_cojob_v32_57425033` | observer active history existed, but no final `result.json`/binding; campaign teardown, performance claim false |
| v31c current-source endpoint-clamp contract | `results/tempo_go_c5_cross_layer_contract_v31c_endpoint_clamp/native_run_contract.json` | file `8b3705b09d0931428d89e44a379f321ae759365854d911ded545d5264892dd79`; fingerprint `a84dfc2966d57c9c66d184705af98edca653479d4513da3d46e2a91d34f88942`; router endpoint-window clamp, full verify passed, native not executed |
| v31c native step-shape failure | `results/tempo_go_cross_layer_cojob_v31c_57425033/cojob.stderr.log` | `More processors requested than permitted`; no observer/result, execution-only; performance claim false |
| v31 CPU implementation source | `tempo/pd_global_orchestrator.py`, `tempo/pd_global_profile.py`, `eval/sota_4node/tempo_pd_elastic_router.py` | current source SHAs `a1d9c1b64e7c50ab47ca65aaca6121f40a32f15610d20bc27ba2345f8c5381ed`, `4b59be49c4d1e6c38598e1cf72c8f3cff643560085ec77d3eee2d1f896e707b0`, `e6e39d4123db1e5d2e61087216be9301d09f593e6e3707ef914505f98bd5da58`; v3 shared budget/strict parser/cached fan-in/endpoint-window clamp |
| v31d current-source native contract | `results/tempo_go_c5_cross_layer_contract_v31d_current_source/native_run_contract.json` | file `4883e1e2ef6dd13b5622b18e2f2d9726f937419f3a9eac807c9665952a1b30a6`; fingerprint `880d358997a35b4a2a17dd3db094fdcf04bf160090908c731a65c93464564c9b`; full verify passed, native pending allocation/QOS |
| v35 endpoint-queue-lease service-lane native attempt | `results/tempo_go_cross_layer_native_v35_tempo_only_57426273/tempo/failure.json` + `results/tempo_go_cross_layer_native_v35_tempo_only_57426273/tempo/tempo_go_c5_discovery/raw.json` | contract `39f949179df3c40352646495e7a30f7b04b65cd33f3280652d9acffb14727e57`; 276 offered, 160 HTTP 200, 116 global rejects, 20 HTTP 502; LMCache p99 622.312 ms, NCCL p99 16.061 ms; invalid/ execution-only |
| v36b batch/background service-lane profile | `results/tempo_go_c5_cross_layer_short_slice_v4_profiles/real_tempo_go_profile_short_slice_v5_service_lane_batch_background.json` | profile `e086077017efd9a69adc274c60be9c035595e48ed9eb529624b4ffc285da58`; shared `global_budget_v3`, `endpoint_queue_lease`, only batch/background lease-enabled |
| v36b current-source native contract | `results/tempo_go_c5_cross_layer_contract_v36b_service_lane_batch_background_shared/native_run_contract.json` | file `a2ca8ae226a2e253b1b6de6e901de56857bb6ffd797bc109fcd9b50c9feb6b30`; fingerprint `c8a21819dfa58b4ff14aef9adbc1ca1fd311db77eff9191652729968f3113f2d`; full verify passed, native pending new allocation |
| v37 service-lane reservation source contract | `results/tempo_go_c5_cross_layer_contract_v37_service_lane_reservation/native_run_contract.json` | file `683b175d3be14168391a4074d395e504998de4e3d847be4af14ee9a00e55af55`; fingerprint `98dcf23c924609a1d1dc5664691a276e3b03b01b40f835b8c4918440f309e423`; reservation handshake/analyzer source, full verify passed, native blocked by `QOSMaxSubmitJobPerUserLimit` |
| v38 hierarchical fan-in source contract | `results/tempo_go_c5_cross_layer_contract_v38_hierarchical_fan_in/native_run_contract.json` | file `da327545cc09269384c62c2094f4fc5bb16ad89829f19ab2f838a62a68802511`; fingerprint `e52c310de04d11b5185b1657077404a4fe044e284504d559fbd68def90cb4474`; node→pair→shard→global bounded fan-in source, full verify passed, native pending allocation/QOS |
| v39 telemetry-aware frontier source contract | `results/tempo_go_c5_cross_layer_contract_v39_hierarchical_telemetry_frontier/native_run_contract.json` | file `eb060a5178f14977f3e3e48a622e735ddb814a78a3dd5d6d9acafc63be9c580b`; fingerprint `97f43a86bdbc079c0a6a1cd19ca4532ece2a1a3ba0d619e0db58c9c5bb1bf669`; live multiplier/health/observed pressure/cross-layer externality-aware frontier, full verify passed, native pending allocation/QOS |
| v37 stable-source native attempt | `results/tempo_go_cross_layer_native_v37_57426952` + `results/tempo_go_cross_layer_cojob_v37_57426952` | contract file `4e93ab2c36099c8f763ac3e76ad744601f8de8f12a0be5afae41df7b156c7051`; fingerprint `34b97506b4a440b9f2c5c23770f713aefa090ddee127c6c7d351e31873c906bc`; local 276/276 valid, source drift before remote, remaining arms execution-only failures, co-job canceled at observer sequence 1597; no performance claim |
| v27 five-arm native discovery receipts | `results/tempo_go_c5_native_cross_layer_57416103_v27/{local,remote,predictor,queue_gpu,network_request_only}/result.json` | result SHA-256s `244ace85370a2b7b6ddf7b88bff3fb425aed35a9109687c22d115a8225c7f285`, `c0a3fb796dff22a4cd47e213469fb02882ec9b8953657d6d26c3f662f5715763`, `e471cefcc7af21ff9ff0919fa0c9fad4887def8e03deed67a3f0103fa23819a0`, `eb7637c92e38c1ab0793680269a2966d6d0730f0ad983918fe4e905cbe87f28e`, `fc619b93bdd64e41c1edb8a753d408f65ffb4f334850aba09672ad189eea6835`; each 276/276 valid complete, discovery evidence only |
| v27 stale-source failures | `results/tempo_go_c5_native_cross_layer_57416103_v27/{app_global_only,tempo}/failure.json` | SHA-256s `ef72c06fd3080db21e22b459d21270b79f139668aafcdf55b6a44f37784c8840`, `16e7ebc6338f9d4698105c3ceccd2b652fe1d7378f4916fb720f6c02c50649a8`; both rejected before data plane by stale source digest guard; execution-only |
| v28 current-source startup failures | `results/tempo_go_c5_repair_57421132_v28/app_global_only/app_global_only/failure.json` + `tempo_long/tempo/failure.json` | SHA-256s `05cf4757eae27807ed72406d14989bc3e42db8f3be79858247b37ada4255a9ca`, `522391684456039d7c07bdd59734ef6ee573b1f16323145c2793ccf7f137a656`; node-2 EngineCore shared-memory broadcast stall under co-load, own step rc143; no utility claim |
| v28 long co-job receipt | `results/tempo_go_cross_layer_cojob_57421132_tempo_long_v28/result.json` + `nccl_observer.json` | result `cb977825b04203f84dbf00623f774c74d77208d8892c802bc3e558f46745e358`, observer `0e012754e6d5ed4385587a32013ee976286467324a85a03a33df3050f5b90ad1`; 10,000/10,000 correct, observer terminal sequence 10,001, background p50/p99 28.34/34.25 ms, final NCCL/LMCache p99 1.251/28.647 ms; component/headroom evidence only |
| P1PAIR one-pair profile | `results/tempo_go_p1pair_profile_c1/real_tempo_go_p1pair_profile.json` | fingerprint `d6585ced11732a6b7367ef94c08ea470953e9c913f4015b5daa25095c3fd6324`; 4-node topology retained, one active TP4 P/D pair |
| P1PAIR+COJOB source implementation | `eval/sota_4node/vllm_lmcache_tempo_go_p1pair_node.py` and `eval/sota_4node/run_tempo_go_p1pair_cojob_in_allocation.sh` | source implemented and static/preflight checked; native performance run pending an independent unoccupied 4-node allocation |
| P1PAIR launcher bootstrap failure | `results/tempo_go_p1pair_cojob_57416103/execution_failure.json` | co-job step failed before data plane because `MASTER_ADDR/MASTER_PORT` were omitted; no result/utility evidence; fix recorded, performance claim false |
| P1PAIR warmup input-contract failure | `results/tempo_go_p1pair_cojob_57416103_retry1/execution_failure.json` | co-job completed with 1,000 observer windows, but inference issued zero requests because warmup seed selection saw no frozen P_ONLY IDs; fix recorded, performance claim false |
| P1PAIR actual vLLM/LMCache descriptor failure | `results/tempo_go_p1pair_cojob_57416103_retry2/execution_failure.json` | co-job completed with 1,000 observer windows and actual vLLM requests reached LMCache, then PD transfer failed on page-index vs physical-pointer mismatch; fix recorded, performance claim false |
| P1PAIR route-commit sequence failure receipt | `results/tempo_go_p1pair_cojob_57416103_retry4/execution_failure.json` | actual workload reached global commit, but producer observer sequence was mixed with atomic global batch sequence; exact router body `TEMPO-GO joint actuation sequence differs`; fix recorded, performance claim false |
| P1PAIR native cross-layer integration receipt | `results/tempo_go_p1pair_cojob_57416103_retry6/native_integration_receipt.json` + `result.json` + `tempo/raw.json` | 4-node same-allocation actual vLLM/LMCache plus 8-rank NCCL/UCX co-job; 2,669 complete/43 receipted global rejects/0 invalid; 249 supported cross-layer decisions; observer path and joint commit sequence identity verified; utility claim false |
| native observer attempt failure receipt | `results/tempo_go_cross_layer_observer_component_57415034/native_attempt_failure.json` | corrected run had no result/observer output because allocation was externally revoked; execution-only receipt, not data-plane evidence |

모든 raw/profile/contract는 immutable evidence다. 새 분석과 실행은 새 path/SHA를 사용한다.

---

## 16. 논문/시스템 산출물

positive 또는 negative 어느 branch든 최종 산출물은 다음을 포함한다.

1. canonical node/pair/shard/global controller, frontend/router/runner/analyzer
2. exact model/runtime/vLLM/LMCache/UCX/NCCL/CXI identity와 topology receipt
3. cross-layer telemetry schema, support matrix, capability receipt와 sampling-overhead report
4. frozen workload/tenant/cache/arrival/NCCL-cojob manifest
5. frozen profile와 source-bound run contract
6. raw request, NCCL collective, LMCache completion, Cassini vector와 terminal/failure ledger
7. arm/phase/tenant/pair/route/cache/topology별 latency/goodput/SLO/fairness 표
8. action-conditioned bottleneck attribution, selected-action counterfactual과 pair activation benefit
9. full/app-only/network-only ablation과 telemetry/control scaling report
10. failed mechanism, stop-rule, claim boundary와 reproducibility instructions

현재 publishable contribution은 아직 positive controller performance가 아니다. 현재 가능한 결과는 다음 두 축이다.

- actual vLLM/LMCache contention, crossover와 failure/control-plane integration measurement
- route-only 및 frozen global candidate의 rigorous negative boundary

positive system paper를 주장하려면 §13의 새 mechanism과 independent native gate가 필요하다.

이는 TEMPO의 contribution이 작다는 뜻이 아니다. 현재 evidence와 target contribution을 정직하게 분리한 것이다. 목표대로 구현·검증되면 산출물은 하나의 scheduler knob가 아니라 Perlmutter-native cross-layer orchestration stack과 그 scale/evaluation 전체다.

---

## 17. 다시 하지 않을 것

- v535 하나만 보고 전체 목표를 재해석
- 없는 v545–v600을 추정
- prior work에 개별 component가 있다는 이유로 end-to-end TEMPO contribution을 feature 차감식으로 축소
- application-only G/I negative를 NCCL/Cassini/LMCache/vLLM/business cross-layer hypothesis의 negative로 확대
- scalar `fabric_pressure`와 prompt coefficient tuning
- hidden phase label/future arrival/oracle route 사용
- queue capacity, wait cap, reservation fraction만 반복 조정
- Candidate D/G/I를 새 이름으로 native 재실행
- pair1이 선택됐다는 이유로 scaling win 주장
- completed-only latency로 reject 비용 숨김
- queue-GPU/LMCache failure를 TEMPO latency win으로 계산
- LMCache failure를 silent local fallback/same-ID retry로 숨김
- application timing으로 physical Slingshot switch bottleneck 단정
- Cassini/NCCL 신호가 불완전하다는 이유로 버리거나 하나의 0/1 congestion label로 압축
- CPU replay를 native performance로 서술하거나 physical cross-layer candidate의 유일한 promotion gate로 사용
- discovery 결과를 보고 profile을 바꾼 뒤 independent validation이라고 부름
- current LMCache 문서/버전과 frozen experiment stack을 혼동
- root/container/udiRoot/CAP_NET_ADMIN 우회
- login node에서 substantial native workload 또는 blind retry

---

## 18. 다음 에이전트에게 그대로 줄 `/goal` 프롬프트

아래 블록을 그대로 목표 프롬프트로 사용할 수 있다.

최신 보정: 아래의 v39/v42/v45/v46/v47/v103/v104/v105 언급은 lineage 보존용
historical state다. §18의 mission, safety, same-population arm과 hard gate는
그대로 유지하지만 current native receipt, workload 판정과 다음 설계 gate는
§70이 우선한다. allocation `57490824`의 v107 immutable contract는
`results/tempo_go_c5_source_snapshot_v107_cxi_credit_refill/native_run_contract.json`
(SHA `de01e9907226c699b2a8a09d6bd6ec6d6d02fe7d2d4d3bf1c48c1e8d9ce28602`,
fingerprint `bb1134d4d6d811ae368d673a4b09947e5a1a0b77169b0e9a9bb1326de8411bba`)
이다. full TEMPO는 273 complete/3 reject/0 fail로 v106 terminal liveness를
개선했지만 strongest fixed remote 대비 mean/p50 4.06%/4.84% 개선,
p95/p99 5.75%/5.84% 악화, request goodput 3.18% 개선이라 gate를 실패했다.
v104–v107을 blind retry하거나 p07/queue/penalty 숫자만 조정하지 않는다.
short-slice+p07은 attribution history로 보존하고 §70의 actual-victim workload
gate와 receiver-credit P×D mesh를 먼저 구현·검증한다.

```text
/goal

System: NERSC Perlmutter, repository /pscratch/sd/s/sgkim/Skim-Tempo

먼저 다음 두 파일을 처음부터 끝까지 읽고, 목표·claim·safety boundary를 임의로 바꾸지 말라.

1. /pscratch/sd/s/sgkim/Skim-Tempo/NERSC_AGENT_SAFETY.md
2. /pscratch/sd/s/sgkim/Skim-Tempo/paper/TEMPO_GO_UNIFIED_GOAL_STATE_AND_EXECUTION_PLAN.ko.md

Current continuation은 §70이다. §18의 목표와 gate는 유지하되, historical
v104/v105 실행 지시와 v106/v107 p07 retry는 supersede됐다. short-slice의
1.5초 phase와 stationary one-decoder synthetic p07을 headline workload로
사용하지 말라. §70 Q0–Q3의 capacity-normalized actual LMCache/NCCL victim,
opposite action과 service-horizon gate를 먼저 닫고, fixed pair×route가 아닌
receiver-credit P×D mesh global mechanism을 설계·replay·freeze하라. 새 GPU
실행은 사용자가 승인한 foreground `gpu_interactive` allocation 안에서만 한다.

원래 mission을 유지하라. TEMPO Elastic-PD를 actual vLLM P/D 경로에 통합하고 strongest fixed와 predictor-only보다 유의미하게 빠르거나, normal-load 손실 없이 유의미한 overload robustness/fairness를 보이는 하나의 최종 TEMPO-GO global scheme으로 확정하는 것이 목표다.

TEMPO의 가치를 개별 prior-work component의 차집합으로 축소하지 말라. 연구 단위는 Perlmutter급 shared HPC에서 vLLM scheduler/service, official LMCache/UCX KV completion, NCCL collective progress/health, GPU/NVLink/PCIe/topology, per-NIC Cassini/Slingshot signal과 tenant business SLO를 하나의 provenance-safe hierarchical state plane으로 묶고, admission/defer/reject, P/D placement/pair scaling, local/remote execution, transfer/semantic concurrency, traffic staggering과 failure recovery를 공동 제어하는 end-to-end global orchestration system이다. data plane은 frozen official LMCacheConnectorV1:UCX로 유지하고 Slingshot/NCCL을 privileged하게 재설정하지 않는다.

현재 4-node user-level scope에서는 자기 P/D fleet과 opt-in experiment co-job만 제어하고 다른 job은 exogenous load로 취급하라. facility-scale joint scheduling은 NERSC/operator integration이 있는 target architecture로 분리하며 일반 사용자 권한으로 다른 job이나 Slurm policy를 제어하지 말라.

현재 사실을 정확히 보존하라.

- conceptual v0, P/D v1–v450 source, v452–v544b campaign, C1–C5를 전체 lineage로 취급한다. 없는 v545–v600을 추정하지 않는다.
- C1/C2/C3 actual-inference opposite crossover와 P_ONLY 12 req/s knee/약 9.7 req/s ceiling은 problem evidence다.
- C4 route-only Candidate A/B/C와 phase oracle은 median+tail full gate를 실패했다. scalar fabric_pressure, prompt coefficient, phase classifier와 route threshold family를 되살리지 않는다.
- v107 p07 seven-arm은 273/276 full TEMPO terminal을 확보했지만 strongest fixed remote 대비 §18 utility/tail gate와 full-vs-app-only incremental gate를 실패했다. 이 결과는 fixed-pair+p07 candidate negative이며 global orchestration hypothesis 전체의 negative가 아니다.
- p07 346.7 Gb/s와 146 MPICH timeout은 aggressor endpoint pressure receipt다. repeat-stable official LMCache victim collapse가 없으므로 synthetic p07은 headline이 아니라 attribution/false-positive ablation이다.
- 최신 source-rebound native analysis는 results/tempo_go_c5_r8_16_20_20_native_job_57409956_v3/native_five_arm_analysis.json, SHA b7e302ab1f893310602b491a8971138d3f4b3cd7fa906b4f7ce05848ac305f45다. local/remote/predictor는 2,712 complete, queue-GPU는 exit 143 execution failure, TEMPO는 982 complete/1,730 explicit reject/0 fail이다. TEMPO goodput 4.786/s는 local 7.934, remote 9.581, predictor 7.928보다 낮다. completed-only latency를 win으로 쓰지 않는다. frozen contract SHA는 002ee5424c9779b22d2cc622cb9143227f8370d03d6b22d0f3c9a560f153e481, fingerprint는 7691d005cad942c26a9a8792cf1487431ce5c4f7abe43ebb7b409a2fef5a854e다.
- v28 safe-envelope discovery는 같은 allocation `57415765`에서 7 arms, 276-row workload, 10,000-block official LMCache/NIXL+UCX/NCCL co-job과 C5 end coverage를 닫았다. TEMPO는 136/276 complete, 140 reject, output-token goodput 638.6/s; app-global-only는 146/276, 647.6/s; queue-GPU fixed는 616.8/s다. 따라서 TEMPO는 queue-GPU보다 약 3.5% 높지만 required 5% gate와 app-only incremental gate를 통과하지 못했고, app-only 대비 E2E p99가 12,616 ms 대 11,712 ms로 약 7.7% 높다. 이 결과는 cross-layer integration/headroom evidence이지 independent performance win이 아니다. v26보다 goodput은 557.5/s에서 638.6/s로 개선됐지만 reject cost가 남아 있다.
- native Candidate C root 57404614는 LMCache/EngineCore failure와 global failure receipt 9건/quarantine를 보였지만 step exit 143, invalid terminal/performance contract다. robustness wiring evidence일 뿐 성능 결과가 아니다.
- Candidate G와 I는 동일 frozen CPU promotion gate에서 구조적으로 다르지만 둘 다 fixed 10%/predictor 5% median gate를 실패했다. strict audit SHA는 aed33cc340c27e2688de0dfb001009182f37236b291ebdb25399e4bd78358925다. audit의 native_performance_negative_proven=false, performance_claim_allowed=false를 유지한다.
- G/I를 native에서 blind retry하거나 queue/wait/reservation/survivor 숫자만 바꾸지 않는다. application-only G/I branch만 CPU-negative-only로 닫혔다. audit은 native_performance_negative_proven=false이므로 이것을 cross-layer TEMPO 전체의 negative로 확대하지 않는다.
- `tempo/cassini_endpoint.py`에는 4 NIC×8 traffic-class explicit counter reader와 endpoint-level/per-NIC·TC multi-signal summary가, `eval/sota_4node/train.py::CudaCollectiveObserver`에는 actual NCCL collective type/bytes/CUDA completion observer가, `run_lmcache_nixl_contention_2node.py`에는 official LMCache NIXL/UCX와 real NCCL 동시 contention harness가 있다. `tempo-go-cross-layer-envelope-v1`와 `PairTelemetry.cross_layer`를 통해 Cassini vector와 `tempo-nccl-observer-v1` NCCL/LMCache snapshot은 현재 native `P1PAIR+COJOB` global decision state에 연결됐다. retry6에서 249개 decision이 observer signal을 supported 상태로 받았고 244개 complete decision이 local route를 선택했다. 남은 gap은 이 native cross-layer loop를 P2PAIR/TP8 topology와 matched full/ablation population에서 utility/fairness/scale로 검증하는 것이다.
- v18 contract의 첫 same-allocation attempt는 `results/tempo_go_cross_layer_native_57415765_retry1/execution_failure_receipt.json`(SHA `4c8304ee3da0e6198898ef16019a900124ae427e356073e09661cec16cd162fd`)으로 격리됐다. local arm은 반복 NIXL registered/page-aligned descriptor 오류 뒤 rc137, 후속 arms는 teardown rc143, intended co-job/observer binding은 없음이다. 이것을 TEMPO negative나 LMCache contention-wide conclusion으로 확대하지 말라. v19 `results/tempo_go_c5_cross_layer_contract_v19/native_run_contract.json`(SHA `70b665050c485d301b81fc4316e91b1fd9a0c87e5e5e1b9f74b071a35e6e5461`, fingerprint `a8f1a6668f16ee6b69a4a03231b859461acd8eba6f826186825bd100e921ea6c`)의 signal-safe lifecycle/explicit co-job-step cleanup을 검증한 뒤에만 새 native run을 시작하라.
- v20 `results/tempo_go_c5_cross_layer_contract_v20/native_run_contract.json`(SHA `f1e1813f59b2668035301d3976f3d47579920939931bc9771fde6c23cb37c969`, fingerprint `1e0d540a3f7ddfaffe7a342d04ac305f6898cff52f682097f08324292d75fb41`)이 seven-arm native comparison을 freeze한다: local, remote, predictor, queue_gpu, NETWORK_REQUEST_ONLY, APP_GLOBAL_ONLY, TEMPO. `NETWORK_REQUEST_ONLY`와 `APP_GLOBAL_ONLY`는 각각 network-only route ablation과 app-global-without-cross-layer ablation이다. full TEMPO의 incremental claim은 이 두 arm을 포함한 same offered population 결과가 없으면 허용하지 않는다.
- v20 native attempt는 co-job root precreation bug로 co-job이 data plane 이전에 종료되어 INT safety stop으로 격리됐다. 수정 후 `results/tempo_go_cross_layer_cojob_57415765_v20_probe`에서 3 blocks/8 ranks/overall correctness true와 active observer history를 확보했다. v21은 3-block co-job이 12초 만에 C5 measured window 이전에 끝났고, v22는 readiness 이후 local arm과 겹쳤지만 600 blocks가 151초에 끝나 seven-arm campaign 종료를 덮지 못했다. v23은 local arm과 실제 overlap까지 만들었지만 구 launcher의 900 s/20 min timeout으로 terminal binding 전에 중단됐다. v24는 timeout을 통과했지만 hot co-load가 약 5분에 local completion 27건/LMCache tail 약 20초로 campaign capacity를 넘어섰다. v25는 moderate profile에서도 local은 full 2,712-row를 닫았지만 remote가 너무 느려 long seven-arm을 닫지 못했다. 각각 v21–v25 execution receipt로 performance claim을 금지했다. v24/v25는 contention/headroom evidence로는 유효하지만 utility comparison은 아니다. v28은 discovery history, v29는 soft-shadow-price v2 history이며, v30 `results/tempo_go_c5_cross_layer_contract_v30/native_run_contract.json`이 resource-envelope spare-pair activation을 포함한 현재 source-bound primary contract다. 2,712-row long trace와 hot profile은 별도 robustness evidence로 유지한다.

다음 candidate는 §9의 cross-layer TEMPO-GO여야 한다. 다음 조건을 만족하지 않으면 G/I threshold variant로 판정하고 구현하지 말라.

1. NCCL 또는 Cassini vector를 vLLM/LMCache completion과 provenance-safe state plane에서 실제로 결합한다.
2. signal을 scalar fabric_pressure로 압축하지 않고 resource별 completion capacity, confidence와 shadow price로 유지한다.
3. measured state가 admission, pair/placement, local/remote, transfer concurrency 또는 staggering action을 실제로 바꾼다.
4. offered population 전체의 reject/defer/failure/tail cost, tenant SLO-goodput와 minimum service를 global objective에 포함한다.
5. pair activation은 queue fraction이 아니라 incremental SLO work minus cache/activation/cross-layer externality로 결정한다.
6. node→pair→shard/global hierarchy와 bounded telemetry/control overhead를 구현한다.
7. full TEMPO, app-only, network-request-only, queue/Kairos-like, predictor와 strongest fixed를 same topology/workload에서 비교한다.

기존 `results/tempo_go_c5_heldout_output128_v1/` 2,712-row r02/r03 trace는 immutable compatibility regression이다. manifest SHA는 6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c, workload SHA는 19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33다. 그러나 NCCL/Slingshot co-load가 없으므로 final cross-layer headline이나 CPU performance oracle이 아니다.

새 native workload는 actual inference + official LMCache transfer + real NCCL collective를 결합한다. `P2PAIR`(two TP4 P/D pairs), `P1PAIR+COJOB`(one TP4 P/D pair + independent 2-node/8-GPU NCCL/LMCache co-job), `TP8-CROSSNODE`와 component attribution profile을 분리한다. primary `P1PAIR+COJOB`에서 co-job offered schedule은 exogenous/frozen이며 TEMPO가 줄이지 못한다. managed co-job staggering은 facility-scope ablation에서만 양쪽 utility를 함께 계산할 때 허용한다. normal, decoder-hot, remote-KV-hot, NCCL-hot, combined-hot, pair asymmetry/failure와 recovery를 포함한다. phase/future arrival/oracle route/physical-switch label은 policy input이 아니다. exact MISS endpoint evidence가 없으면 FrozenServiceProxyPolicy boundary와 allowed_remote_cache_residencies=[prefill_only]를 보존하고 proxy를 native performance evidence로 부르지 않는다.

cross-layer sample은 agent epoch, node, pair, endpoint, communicator, local sequence/window, source/support/value/uncertainty를 가져야 한다. cross-host monotonic timestamp subtraction, missing-to-zero, mixed topology/profile epoch를 금지한다. installed NCCL RAS/profiler가 없으면 container/root 설치를 시도하지 말고 existing CudaCollectiveObserver fallback을 사용한다. Cassini는 NIC/TC별 vector와 support state를 유지하며 단일 congestion label로 바꾸지 않는다.

모든 request는 upstream 시작 전 pair×route immutable commit이어야 한다. prefill 후 route/pair migration, hidden recompute, silent local fallback, same-ID retry를 금지한다. first response에서 endpoint/prefill/KV/semantic credit을, HTTP EOF에서 decoder/active-sequence credit을 exactly once 반환한다. UNKNOWN cache는 hit가 아니다. failure는 tempo-go-global-failure-v1 receipt, telemetry sequence, released work, pair/route scope, new_request_id_required, quarantine와 explicit PROBE recovery로 기록한다.

tenant fairness는 weighted request count가 아니다. raw dominant service units와 weighted debt를 분리하고 tenant별 offered/admitted/completed/rejected/failed, TTFT/TPOT/E2E SLO-goodput, output-token goodput, max wait, minimum service, starvation, Jain fairness를 보고한다. reject는 valid terminal일 수 있지만 free performance improvement가 아니다.

실행 순서는 다음과 같다.

P0. bounded source/artifact/hash audit. 기존 dirty changes와 immutable artifacts를 보존한다.
P1. 승인된 4-node allocation에서 native stack/NCCL capability, Cassini support, GPU-NIC-pair topology와 sampling overhead를 `cross_layer_capability_receipt`로 고정한다.
P2. controller 없이 fixed action ABBA로 P2PAIR, real NCCL+LMCache component, P1PAIR+COJOB와 가능하면 TP8-CROSSNODE의 causal signal/actuator headroom을 측정한다.
P3. node/pair/shard/global telemetry, resource envelope/confidence/shadow price, business utility와 joint actuator를 새 candidate/source/profile/contract로 구현한다. 현재 source의 `tempo/pd_global_hierarchy.py`를 사용해 node→pair→shard→global bounded fan-in을 실제 호출 경계로 만들고, `tempo-go-node-envelope-v1`, `tempo-go-pair-envelope-v1`, `tempo-go-shard-envelope-v1`, `tempo-go-reduction-receipt-v1` identity/omission receipt를 보존한다. shard frontier는 후보를 줄이는 local ownership일 뿐이며 최종 business/fairness/cross-layer decision은 global orchestrator가 내린다.
P4. CPU에서는 lifecycle/identity/fairness/failure와 logical 2→1,024-agent scale/overhead만 gate한다. full-candidate global scan과 bounded-fan-in global evaluation을 같은 population에서 비교하고, raw reducer construction/serialization 비용, global decision p50/p99, forwarded candidate count/bytes, stale-failure convergence를 각각 분리 보고한다. CPU duration model로 physical performance candidate를 탈락시키지 않으며, bounded frontier가 full-candidate oracle과 달라질 수 있음을 명시하고 native utility gate로 검증한다.
P5 current-source correction: 아래 historical v39/v42/v45/v46/v47/v103
snapshot reference는 lineage 보존용이다. 실제 다음 native run은 v104
immutable snapshot contract
`results/tempo_go_c5_source_snapshot_v104_hierarchy_reducer_cache/native_run_contract.json`
(file SHA `4ddd901df87dc7107fac1ee4d7c76f8734d1651eb4502b8a12563d708b64ff6d`,
fingerprint `9378202301943156b1eaf500b72b974145b375948bd22fc686e611a7eea2a170`)
을 사용한다. v44/v45/v46/v47/v103 source-digest/interconnect failure는
performance result가 아니다. v104는 이전 launcher의 Perlmutter native
OFI/Libfabric launch boundary, allocation `Network=job_vni`, explicit
nested-step GPU/CPU shape, separate four-node capability receipt와
synchronized NIXL failure boundary를 유지하면서 global reducer의 repeated
rank/shard work를 cache한다. `--network=no_vni`는 capability probe에만
사용하고 NCCL/UCX data path에는 사용하지 않는다.
P5. 승인된 4-node/16-GPU/4-hour interactive allocation 안에서 seven-arm counterbalanced discovery를 수행한다. v28 discovery는 완료됐지만 performance gate는 false다. v29 `soft_shadow_price_v2` mechanism은 관련 CPU test 83개를 통과했으나 v29f native attempt는 actual initialization 뒤 첫 측정 전에 rc143으로 끝났고, 이를 성능 결과로 사용하지 않는다. v30 native discovery는 lifecycle/coverage까지 닫혔지만 utility gate가 false였고, v31 shared-budget implementation과 endpoint-window clamp는 CPU gate를 통과했다. v34/v35 native TEMPO-only receipt는 endpoint-window 오류를 제거했지만 queue lease 54건 중 20건이 downstream ingress timeout으로 실패했다. v37은 이 경계를 `global decision → endpoint submit/service-lane receipt → immutable forward 또는 debt release`로 구현했고 관련 CPU suite `194 passed, 22 subtests passed`를 통과했다. v38은 `tempo/pd_global_hierarchy.py`의 bounded node→pair→shard→global fan-in을 추가했고, v39는 live multiplier/health/observed-pressure/cross-layer-externality-aware frontier를 반영했다. v45/v46/v47/v103은 historical execution source로 보존한다. v104 contract의 `performance_claim_allowed=false`는 유지한다. v44–v103의 native attempts는 각각 execution, provenance, coverage 또는 failure-boundary evidence로만 해석하며 matched utility result가 없는 한 performance claim으로 승격하지 않는다. 따라서 historical source를 blind retry하거나 숫자만 바꾸지 말고, 기존 foreign allocation/step이 아닌 새로 승인된 4-node allocation에서 v104 snapshot으로 same-population arms를 실행한다. substantial workload를 login node에서 돌리지 않는다.
P6. correctness와 headroom이 닫히면 code/profile/workload/analyzer를 freeze한다.
P7. 새 승인 allocation에서 frozen independent validation을 한 번 수행한다. 결과를 본 뒤 tuning하지 않는다.
P8. native mechanism + cross-layer ablation + fairness/robustness + scale evidence를 모두 보고하고, 통과하면 end-to-end TEMPO systems contribution으로 작성한다. 실패하면 실패한 mechanism 범위의 reproducible negative로 종료한다.

primary gate는 strongest fixed 대비 E2E median 10%, predictor 대비 5%, goodput 5%, paired win 75% overall/60% per group, E2E/TPOT p99 regression 5% 이내, worst paired regression 100 ms 이내, selected-action counterfactual 5%다. robustness gate는 normal regression 3% 이내, overload p99 또는 goodput/SLO-goodput 15% 개선, failure/queue-timeout 감소, starvation 0과 correctness/fairness 통과를 모두 요구한다. full TEMPO는 coupled state에서 APP_GLOBAL_ONLY와 NETWORK_REQUEST_ONLY 대비 SLO-goodput/goodput 5% 또는 p99 10% incremental gain을 보여야 한다. telemetry-only normal overhead는 3% 이내, request decision p99는 5 ms 이내, mixed/stale identity는 0이어야 한다. gate를 사후 완화하지 않는다.

Perlmutter에서는 login node에서 bounded inspection, edit와 작은 test만 한다. substantial replay와 모든 vLLM/LMCache/GPU/traffic workload는 승인된 native interactive allocation 안에서만 실행한다. Slurm 자동 submit/cancel/retry, background watcher, broad shared-filesystem traversal을 만들지 않는다. container/Shifter/Apptainer/Podman/Docker/--image/udiRoot/sudo/su/root ownership/setcap/CAP_NET_ADMIN/system-file 변경은 절대 금지한다. udiRoot.conf must be owned by user root 또는 exit 139가 나오면 우회하지 말고 exact command/environment/node/log/exit receipt만 남기고 중단한다.

매 단계에서 현재 stage, 사용한 기존 evidence, cross-layer signal/actuator, 통과/실패 gate와 다음 STOP/GO를 짧게 보고하라. error fixing만 반복하지 말고 각 native run이 어느 hypothesis를 검증하고 다음 global policy decision을 어떻게 바꾸는지 명시하라. 통합 성공, remote/pair1 선택, 낮은 completed-only latency, LMCache baseline failure만으로 goal을 완료하지 말라. 완료는 frozen independent native win + scale evidence 또는 사전 정의된 정확한 범위의 reproducible negative다.
```

---

## 19. 최종 claim boundary

### 현재 허용

> Perlmutter native vLLM P/D에서 decoder-local과 frozen official LMCache remote 경로의 service state가 actual inference contention에 따라 교차하며, route-only policy는 shared decoder tail을 닫지 못하고, global admission/failure ledger가 actual path에서 발동한다.

> 동일 frozen CPU control-plane replay에서 구조적으로 다른 Candidate G와 I는 native promotion gate를 통과하지 못했다.

> repository에는 Cassini explicit counter reader/endpoint multi-signal summary, per-NIC/TC vector를 보존하는 cross-layer telemetry envelope, strict atomic `tempo-nccl-observer-v1` publisher/reader, actual NCCL/CUDA collective observer와 official LMCache×NCCL contention harness가 있다. 이 observer path는 current TEMPO-GO global decision에 연결되고, `JointActuationPlan`이 독립적인 prefill/KV/semantic-op limit와 bounded stagger로 global admission 및 pair endpoint에 전달·집행된다. 다만 4-node native C5 co-job에서 이 action이 coupled utility를 개선한다는 증거는 아직 없다. 따라서 G/I negative는 cross-layer TEMPO 전체의 negative가 아니다.

> 별도 matched component attribution에서는 official NIXL/UCX와 real NCCL `all_reduce`를 같은 2-node/8-GPU interval에 correctness-preserving으로 실행했고, NCCL token-tail p99가 NCCL-only control의 8.107 ms에서 9.245 ms로 증가했다. 이는 cross-layer externality mechanism evidence이며, 아직 vLLM C5 actuator utility win은 아니다.

### positive independent validation 성공 후에만 허용

> 동일 native topology와 frozen official LMCache data plane에서 TEMPO-GO의 hierarchical cross-layer orchestration이 vLLM, LMCache/UCX, NCCL, GPU/topology, Cassini/Slingshot와 business state를 causal resource envelope로 결합하고, fixed local/remote, predictor, Kairos/NetKV-like request-local, application-only global policy보다 offered-population latency/goodput, multi-tenant robustness와 failure utility를 개선했다.

### 금지

- LMCache transport 자체보다 빠르다.
- 특정 Slingshot switch/link가 병목이다.
- Kairos, Mooncake, P/D-Serve, Dynamo보다 보편적으로 우월하다.
- CPU replay가 native 성능 negative를 증명했다.
- 4노드 한 campaign이 production/HPC scale superiority를 증명했다.
- 모든 workload에서 항상 빠르다.

이 문서의 최종 목적은 에러를 더 잡았다는 기록을 남기는 것이 아니다. **Perlmutter의 compute, collective communication, KV data movement, fabric endpoint와 business demand를 전체로 orchestration하는 TEMPO를 실제로 만들고, 어떤 병목 이동을 막아 어떤 utility를 얻었는지를 scale과 native evidence로 증명하는 실행 가능한 연구 계약을 고정하는 것**이다.

## 20. historical source correction (v35 service-lane boundary)

최신 source-bound service-lane 실행 계약은
`results/tempo_go_c5_cross_layer_contract_v35_endpoint_queue_lease_service_lane/native_run_contract.json`이다.
file SHA는 `a38c959d11ba89c0163f89ebc1972ce3f149c806a3f88544f5f34c2c1626c8e1`,
fingerprint는 `39f949179df3c40352646495e7a30f7b04b65cd33f3280652d9acffb14727e57`이며
node source guard와 97개 targeted test/py_compile을 통과했다. v35 native
TEMPO-only raw는 276 offered, 140 HTTP 200, 116 global reject, 20 HTTP 502이고
performance claim은 false다. 현재 source SHA는 orchestrator
`a1d9c1b64e7c50ab47ca65aaca6121f40a32f15610d20bc27ba2345f8c5381ed`, router
`3ddf08dea04745ab41b9a039f9100d4f36bf8d01cc601c53c1eb0b85825d9b03`이다.
v36b batch/background-only profile와 contract는 artifact history로 보존하지만
reservation protocol이 없었으므로 native 실행을 다음 gate로 승격하지 않는다.

문서 안의 v28 discovery receipt, v29 soft-shadow-price history와 v30/v31/v35 서술은 서로 다른 source inventory 시점이다. native campaign 시점의 source inventory와 일치해 전체 `verify`를 통과한 v30 campaign contract는 `results/tempo_go_c5_cross_layer_contract_v30/native_run_contract.json`(file SHA `eaab9c4d70731ae68c59158b9143da11bf602387b1b00cdd356523f35e24a830`, fingerprint `0f2f25f2f9ba58c46d82868ed553c2491b8369113cddc2bcf01849e9366cf9a6`)다. v28은 `tempo_pd_elastic_router.py` source digest mismatch가 확인되어 current-source contract가 아니며, v29는 v2 soft-shadow-price source history로 보존한다. v28/v29 native receipts는 execution/robustness history로만 사용한다. v30 native discovery `57423440`는 7/7 arm, same-population, co-job end coverage와 correctness를 닫았지만 performance claim은 false였다. 현재 소스 기준 v31 shared-fabric resource budget/global remote concurrency·stagger/pair-local failure separation과 endpoint-window clamp는 구현·CPU gate까지 끝났고, v35는 endpoint queue lease의 downstream failure boundary를 native에서 확인했다. v37은 endpoint controller의 실제 `submit()` 결과를 `tempo-go-service-lane-reservation-v1` receipt로 노출하고, unavailable이면 route quarantine 없이 global ownership을 exactly once release한다. v31b/v31c/v31d/v36b는 이전 source-bound contract와 execution history로 보존한다. v37 source-bound matched native discovery는 contract verify까지 끝났지만, 새 4-node interactive 요청이 `QOSMaxSubmitJobPerUserLimit`로 거부되어 아직 실행되지 않았다.

## 21. historical continuation state: v35 failure와 v36b next candidate

기존 allocation `57426273`의 interactive shell에서는 v35 endpoint-queue-lease service-lane tempo-only 실행이 진행됐다. raw path는 actual vLLM P/D와 official `LMCacheConnectorV1:UCX`를 사용했고, co-job observer는 NCCL p99 `16.060774 ms`, LMCache transfer p99 `622.311661 ms`를 관측했다. 그러나 276 offered request 중 160개만 HTTP 200이었고 116개는 global admission reject, 20개는 ingress queue HTTP 502였다. `all_streams_valid=false`, `router_decisions_exact=false`, `terminal_contract_valid=false`이고 co-job final `result.json`도 생성되지 않았으므로 이 실행은 성능 결과가 아니라 execution/mechanism evidence로만 보존한다. failure receipt는 `results/tempo_go_cross_layer_native_v35_tempo_only_57426273/tempo/failure.json`이다.

이 결과는 현재 profile이 cross-layer budget을 갖고 있어도 queue-lease business action을 실제로 활성화하지 않으면 overload에서 reject cost가 커진다는 것을 보여준다. 따라서 새 v36b profile은 `global_budget_v3`를 유지하면서 `overload_action=endpoint_queue_lease`를 켜고, latency/interactive tenant는 보호하며 batch/background tenant만 명시적으로 lease한다. profile load, global profile fingerprint, current-source contract full verify와 CPU `160 passed, 11 subtests passed`가 닫혔다. v36b contract는 `results/tempo_go_c5_cross_layer_contract_v36b_service_lane_batch_background_shared/native_run_contract.json`이다. 이것은 아직 native performance win이 아니다.

이 continuation 중 기존 interactive shell에 `sattach`로 연결한 뒤 shell이 Ctrl-C를 받아 allocation `57426273`이 종료됐다. 다른 job이나 root/UDI/ownership은 건드리지 않았지만, 해당 allocation은 더 이상 재사용하지 않는다. 새 allocation이 승인되기 전에는 Slurm submit/retry를 하지 않고, v36b native matched discovery는 pending으로 유지한다.

## 22. latest continuation state: v37 reservation handshake implementation

v35의 downstream ingress timeout을 숫자 조정으로 덮지 않기 위해 다음 경계를
현재 source에 구현했다.

```text
global queue lease/admission decision
        │ provisional global ownership
        ▼
pair router EndpointFeedbackController.submit()
        ├─ local/remote physical window fits
        │    → tempo-go-service-lane-reservation-v1: accepted
        │    → immutable upstream route forward
        └─ bounded endpoint queue
             → receipt: unavailable
             → no endpoint queue wait / no upstream start
             → global service-lane failure receipt
             → exactly-once global debt release, no route quarantine
```

구현된 source contract는 다음과 같다.

| item | path/status |
|---|---|
| global release state machine | `tempo/pd_global_orchestrator.py`, service-lane failure receipt와 `fail_service_lane_reservation()` |
| async lifecycle bridge | `tempo/pd_global_coordinator.py`, reservation failure dispatch/status |
| endpoint receipt | `eval/sota_4node/tempo_pd_elastic_router.py`, accepted/unavailable reservation provenance |
| native HTTP boundary | `eval/sota_4node/tempo_pd_elastic_router_v448.py`, unavailable 즉시 503, queue wait 금지 |
| frontend ownership bridge | `eval/sota_4node/tempo_pd_elastic_frontend.py`, quarantine 없는 release와 ledger receipt |
| analyzer | `eval/sota_4node/analyze_tempo_go_c5_five_arm.py`, service-lane failure를 opaque 502와 분리 |
| CPU validation | `196 passed, 22 subtests passed` |
| v37 contract | `results/tempo_go_c5_cross_layer_contract_v37_service_lane_reservation/native_run_contract.json` |
| v37 contract SHA/fingerprint | `683b175d3be14168391a4074d395e504998de4e3d847be4af14ee9a00e55af55` / `98dcf23c924609a1d1dc5664691a276e3b03b01b40f835b8c4918440f309e423` |

CPU logical fan-in diagnostic은 2→1024 pair에서 one-shot candidate scan이
`0.174 ms → 15.086 ms`로 증가했다. 이것은 native inference latency가 아니며,
현재 2-pair implementation이 Perlmutter 전체 규모의 hierarchical scale을
아직 닫지 않았다는 control-plane evidence다. 따라서 v37 native 결과가
나오더라도 production-scale claim은 하지 않는다. 다음 scale work는
node→pair→shard/global candidate ownership과 bounded fan-in을 구현하고,
request decision p99/control bytes/stale-failure convergence를 별도 CPU
gate로 닫는 것이다.

v37 native launch는 source-bound contract verify 후 사용자가 허용한 exact
`/usr/bin/salloc -A m1248_g -C gpu -q interactive -t 04:00:00 -N 4
--gpus-per-node=4`를 한 번 요청했다. Perlmutter scheduler가
`QOSMaxSubmitJobPerUserLimit`로 거부했으며, 자동 재시도하지 않았다. 현재
좁은 `squeue -u sgkim` 확인에는 실행 중인 4-node allocation이 없고 사용자의
기존 16-node pending jobs만 보인다. 이것은 TEMPO/LMCache/NCCL native
negative가 아니라 scheduler admission blocker다. 새 allocation이 실제로
승인되면 v37 contract 그대로 seven-arm matched discovery를 시작한다.

## 23. latest native continuation: v37 source-drift invalidation (57426952)

사용자가 승인한 4-node/4-hour interactive allocation `57426952` 안에서
v37 reservation contract를 실제로 시작했다. co-job은 같은 allocation의
2-node/8-GPU NCCL+official LMCache/NIXL workload였고, observer는
`tempo-nccl-observer-v1`, `correctness_met=true`로 C5 전 구간에 겹쳐 있었다.
결과 root는 `results/tempo_go_cross_layer_native_v37_reservation_57426952/`,
co-job root는 `results/tempo_go_cross_layer_cojob_v37_reservation_57426952/`이다.

이 실행은 contention이 실제 coupled signal로 관측된다는 motivation evidence를
남겼다. co-job readiness 시점에는 LMCache transfer p99 `26.814915 ms`,
NCCL collective p99 `1.109056 ms`였으나 local arm의 measured interval 중에는
각각 `609.623996 ms`와 `13.899153 ms`까지 상승했다. 이후 active co-job의
회복 구간에서는 `28.203649 ms`와 `1.168452 ms`로 다시 낮아졌다. 이는
혼자 돌릴 때와 workload가 겹칠 때 KV movement와 collective tail이 함께
이동한다는 evidence다. 특정 Slingshot switch/link의 원인이라고 단정하지
않으며, 이 signal을 전체 admission/pair/route/transfer 제어에 쓰는 것이
TEMPO의 연구 가치다.

local arm은 actual vLLM P/D + `LMCacheConnectorV1:UCX` 경로에서 `276/276`,
`all_streams_valid=true`, `router_decisions_exact=true`,
`terminal_contract_valid=true`를 기록했다. 이것은 native correctness와
cross-layer observability가 닫혔다는 뜻이지 성능 승리가 아니다.

그러나 local arm 이후 concurrent source editor가
`tempo/pd_global_orchestrator.py`를 수정했다. v37 contract guard가 remote
arm에서 기대한 digest는 `f76191d9...`였고 guard 시점 파일은
`9c5da766...`로 달랐다. 이후 bounded inspection에서도 현재 파일이 다시
`d0833acf...`로 변했다. 따라서 remote, predictor, queue_gpu,
network_request_only, app_global_only, tempo arm은 inference 비교가 아니라
source guard에서 종료되었다(`native_arm_process_failed`). 이 campaign은
matched seven-arm performance result가 아니며 `performance_claim_allowed=false`
로 보존한다. local 단독 receipt를 나머지 arm과 비교하거나 TEMPO 전체의
negative로 확대하지 않는다.

이번 실행의 다음 gate는 고정한다. (1) concurrent source mutation 상태에서
native retry나 숫자 조정을 하지 않는다. (2) source owner를 하나로 고정하고
orchestrator·coordinator·router·frontend·analyzer·launcher 전체 hash를 새
contract에 재기록한다. (3) 같은 contract를 allocation 안팎에서 두 번 verify하고,
verify 후 source hash가 변하면 실행하지 않는다. (4) freeze 이후 동일 co-job
schedule과 offered population으로 seven-arm을 다시 수행해 E2E, SLO-goodput,
reject/defer/failure cost, tenant fairness, LMCache/NCCL/Cassini evidence를
한 표에서 분석한다.

co-job step `57426952.20`만 우리 실행의 정리 대상으로 종료했고 allocation
`57426952` 자체는 유지했다. 다른 job, root/UDI/container, ownership과 system
file은 건드리지 않았다. 현재 결론은 **contention 문제는 native Perlmutter
경로에서 실존하고 global cross-layer orchestrator가 필요하다는 동기 evidence는
확보했지만, TEMPO의 성능 우위는 아직 검증되지 않았다**이다.

## 24. latest continuation state: v38 hierarchical global fan-in

v37에서 endpoint service-lane reservation의 lifecycle 경계를 닫은 뒤, 현재
global control-plane의 실제 scale gap을 node→pair→shard→global ownership으로
확장했다. 이것은 TEMPO의 가치를 낮추는 기능 차감이 아니다. Perlmutter에서
동시에 몰리는 여러 inference tenant의 vLLM/LMCache/NCCL/Cassini 상태를
global business decision으로 가져오려면, raw pair 후보를 매 요청 중앙에서
전부 재스캔하는 구조 자체가 먼저 없어져야 한다.

```text
node agent
  raw vLLM scheduler / LMCache completion / NCCL / Cassini vector
       │ allocation epoch + topology/profile identity
       ▼
pair agent
  local/remote route frontier + pair resource envelope
       │ bounded pair summaries
       ▼
shard coordinator
  top pair frontiers, omitted-pair accounting, stale/mixed fail-closed
       │ bounded shard envelopes
       ▼
global TEMPO orchestrator
  tenant business SLO/fairness + shared fabric budget + capacity
  + joint local/remote/stagger/activation/failure lifecycle decision
```

현재 source 경계는 다음과 같다.

| item | implementation |
|---|---|
| node identity envelope | `tempo/pd_global_hierarchy.py::NodeTelemetryEnvelope`, `tempo-go-node-envelope-v1` |
| pair identity/cardinality envelope | `PairTelemetryEnvelope`, `tempo-go-pair-envelope-v1` |
| shard bounded frontier | `ShardCandidateEnvelope`, `tempo-go-shard-envelope-v1` |
| global omission/fan-in receipt | `HierarchicalReductionReceipt`, `tempo-go-reduction-receipt-v1` |
| global integration | `GlobalOrchestrator.telemetry_snapshot()` + `submit_hierarchical()`; 최종 submit은 기존 global lifecycle 사용 |
| identity policy | source epoch/profile/topology/sequence가 섞이면 `HierarchyIdentityError`; stale/partial telemetry도 fail-closed |
| native source contract | `results/tempo_go_c5_cross_layer_contract_v38_hierarchical_fan_in/native_run_contract.json` |
| v38 contract SHA/fingerprint | `da327545cc09269384c62c2094f4fc5bb16ad89829f19ab2f838a62a68802511` / `e52c310de04d11b5185b1657077404a4fe044e284504d559fbd68def90cb4474` |

reducer의 `max_routes_per_pair=2`, `max_pairs_per_shard=2`는 각 pair의
local/remote frontier를 보존하고 shard당 global로 전달하는 pair 수를 제한한다.
따라서 이는 route threshold나 phase classifier가 아니다. 후보가 omitted된
pair도 receipt에 남으며, global controller는 전달된 후보에 대해서도 live
telemetry, resource vector, shared remote budget, tenant SLO/fairness와
joint actuation을 다시 검사한다. bounded frontier가 full candidate oracle과
같은 선택을 한다는 가정은 하지 않고, native full/app-only/network-only와
별도 utility gate로 검증한다.

CPU control-plane evidence는 다음 one-shot run이다. `central_full`은 1024
pair×2 route를 기존 global controller가 모두 평가한 시간이고,
`global_reduced`는 64 shard가 shard당 2 pair frontier(총 256 forwarded
candidate)를 만든 뒤 global controller가 평가한 시간이다. reducer가 실제
분산 node/pair/shard agent에서 수행되는 비용과 global decision 비용을
섞지 않았다.

| pair | raw candidate | forwarded | central full global (ms) | reduced global (ms) |
|---:|---:|---:|---:|---:|
| 2 | 4 | 4 | 0.276 | 0.177 |
| 16 | 32 | 4 | 0.661 | 0.175 |
| 64 | 128 | 16 | 2.291 | 0.425 |
| 256 | 512 | 64 | 8.695 | 1.435 |
| 1024 | 2048 | 256 | 30.572 | 4.991 |

이는 native inference latency나 Perlmutter production-scale 성능 승리가
아니다. 반대로 raw reducer construction/serialization, cross-node telemetry
bytes, global p50/p99, omitted-frontier utility loss, stale/failure storm
convergence를 포함하는 P4 scale gate를 추가로 닫아야 한다. 현재 결과가
보이는 것은 중앙 global 후보 평가가 raw pair 수에 따라 커지는 control-plane
문제와, hierarchical ownership을 넣었을 때 global fan-in을 8배 줄일 수
있는 구현 가능성이다. 이후 live telemetry-aware frontier 보강으로 v38은
initial hierarchy contract history가 되었으며 native에서는 v39만 사용한다.

v39 native launch는 아직 수행하지 않았다. 이전과 동일하게 Perlmutter
4-node/4-hour interactive가 실제 승인되면 그 allocation 내부에서만
source-bound seven-arm matched run을 수행한다. `QOSMaxSubmitJobPerUserLimit`
거부는 scheduler admission blocker이지 TEMPO/LMCache/NCCL negative가 아니며,
자동 재시도하지 않는다. root, container, UDI ownership, `udiRoot.conf`,
login-node GPU workload는 건드리지 않는다.

## 25. latest continuation state: v39 telemetry-aware shard frontier

v38의 bounded fan-in이 static predicted cost만으로 shard 후보를 고르는
경계가 되지 않도록 v39에서 shard frontier 순서를 현재 pair telemetry로
보강했다. 각 candidate의 local/remote service multiplier, `GOOD/SKIP/DENIED`
path health, observed resource pressure와 supported cross-layer route
externality를 frontier rank에 반영한다. 따라서 hot/denied route가 단순
predicted latency 때문에 global 후보를 독점하지 않는다.

이 보강은 global authority를 shard로 분산시키지 않는다. shared remote
budget, tenant business SLO, weighted fairness/minimum service, survivor
reserve, joint actuation, queue/service-lane lifecycle과 failure receipt는
계속 `GlobalOrchestrator`가 최종 판정한다. shard는 bounded ownership과
candidate omission receipt만 책임진다. cross-layer identity가 섞이거나
telemetry가 stale/future이면 `HierarchyIdentityError`로 축약 submit을
차단한다.

v39 source-bound contract는 다음과 같이 verified됐다.

```text
path: results/tempo_go_c5_cross_layer_contract_v39_hierarchical_telemetry_frontier/native_run_contract.json
file sha256: eb060a5178f14977f3e3e48a622e735ddb814a78a3dd5d6d9acafc63be9c580b
fingerprint: 97f43a86bdbc079c0a6a1cd19ca4532ece2a1a3ba0d619e0db58c9c5bb1bf669
arms: local, remote, predictor, queue_gpu, network_request_only,
      app_global_only, tempo
performance_claim_allowed: false
```

v39 이후 native에서 사용할 source는 v39 contract를 verify한 뒤 freeze한다.
그 뒤 source hash가 바뀌면 run을 시작하지 않고 새 contract를 만든다. CPU
suite의 현재 hierarchy 추가 gate는 6개 hierarchy test를 포함하며, 기존
global/router/frontend/analyzer suite와 합쳐 `200 passed, 22 subtests passed`
로 통과했다. 별도의 historical candidate-I contract test가 현재 dirty
analyzer source drift를 감지하는 것은 source-bound safety failure이지
성능 결과가 아니며, 그 immutable contract를 덮어쓰지 않는다.

다음 실행 순서는 고정한다.

1. bounded CPU fan-in에서 full global scan 대비 reduced global decision p50/p99,
   raw reducer/serialization time, forwarded bytes와 omitted-pair utility를
   반복 측정한다.
2. 4-node interactive allocation이 승인되면 v39 verify 직후 same-allocation
   seven-arm + NCCL/LMCache co-job을 수행한다. normal/decoder-hot/
   remote-KV-hot/NCCL-hot/combined-hot/pair-asymmetry/failure-recovery를
   같은 offered population에서 유지한다.
3. full TEMPO, app-only, network-request-only, predictor, strongest fixed의
   E2E/SLO-goodput/reject·failure cost/fairness와 cross-layer signal tail을
   함께 분석한다. bounded frontier omission이 utility를 훼손하면 fan-in
   policy를 native 결과에 맞춰 몰래 튜닝하지 않고 failure mechanism으로
   기록한다.

QOS rejection은 계속 scheduler blocker로만 기록한다. allocation 승인 전에는
Slurm blind retry를 하지 않으며, login node에서 GPU/inference/co-job을
실행하지 않는다. root/UDI/container ownership과 `udiRoot.conf`는 절대
접근하지 않는다.

## 26. latest continuation state: immutable source snapshot and v41 execution boundary

v37/v38/v44에서 workspace의 Python 또는 co-job shell이 arm 사이에서 바뀌어
source guard가 실패한 반복을 끊기 위해, 단순히 contract를 다시 생성하는
방식 대신 immutable source boundary를 구현했다.

| item | evidence |
|---|---|
| snapshot builder | `eval/sota_4node/freeze_tempo_go_c5_source_snapshot.py` |
| snapshot contract | `results/tempo_go_c5_source_snapshot_v41_57426952/native_run_contract.json` |
| contract SHA/fingerprint | `e2531cd798e97aa893940750b796cb5af16ac702e193881c2648142a29a72f92` / `659af5f47c38bd10c510ac6e9cb7370c7331220d88e202eacd0def34866dc8aa` |
| immutable tree | `results/tempo_go_c5_source_snapshot_v41_57426952/source`, 1,799 Python + 1 co-job shell |
| tree SHA | `22d389dc0b62caf3aa392d4098a6d343e214d571a9e482719c6f3a9c8b5b16f9` |
| source import check | hierarchy, orchestrator, observer, co-job module 모두 snapshot path에서 import |
| bounded snapshot CPU suite | `197 passed, 22 subtests passed` |

builder는 copy 전후 source digest가 다르면 contract를 만들지 않는다. contract의
Python binding은 snapshot path를 가리키고, co-job shell도 snapshot으로
재바인딩한다. node는 snapshot root로 `cd`한 뒤 import하며, original repo root는
model/profile/workload와 결과 경로에만 사용한다. 따라서 이후 workspace 편집은
이미 생성된 native candidate의 controller code를 바꾸지 않는다. shell
launcher의 current hash와 source snapshot tree hash는 여전히 contract guard로
검사한다.

v41 native continuation은 snapshot contract verify까지는 통과했지만, 같은
allocation에서 co-job child가 실제 NCCL/LMCache observer를 publish하기 전에
Slurm `Error configuring interconnect`를 냈다. observer가 준비되지 않아 C5
inference arm은 시작하지 않았고, v41에는 offered request, latency, goodput
또는 TEMPO performance result가 없다. 이 execution failure를 fabric
bottleneck의 증거로 부르지 않는다. allocation `57426952` 안에는 현재 다른
same-allocation co-job step `57426952.53/.54`가 이미 16-GPU/8-GPU 자원을
사용하고 있어, 그 step은 건드리지 않고 v41 child step만 정리했다.

다음 GO gate는 다음과 같다.

1. 현재 allocation의 기존 step이 끝나고 GPU/node scope가 다시 비어야 한다.
2. 최신 telemetry-aware frontier source를 snapshot builder로 한 번 freeze하고,
   snapshot import + contract verify + bounded CPU suite를 다시 닫는다.
3. 같은 snapshot과 same-allocation co-job으로 seven-arm을 수행한다. observer
   readiness 실패나 interconnect setup failure가 다시 나오면 native 성능
   negative가 아니라 execution/capability receipt로 종료한다.
4. observer와 vLLM P/D가 모두 살아난 경우에만 offered-population E2E,
   SLO-goodput, reject/defer/failure cost, tenant fairness, LMCache/NCCL/Cassini
   signal을 함께 비교한다.

이 경계는 TEMPO의 목표를 축소하는 것이 아니다. 오히려 Perlmutter 규모의
global orchestrator가 실제로 어떤 controller code, fabric observer,
business policy와 data plane을 함께 실행했는지를 보존하기 위한 재현성
조건이다.

## 27. latest continuation state: v45 current-source freeze와 v44 실행 진단

### 27.1 현재 구현의 의미

현재 TEMPO-GO는 단순 route selector가 아니다. `tempo/pd_global_hierarchy.py`의
node→pair→shard→global reducer를 `GlobalOrchestrator.submit_hierarchical()`과
`GlobalAdmissionCoordinator`의 실제 frontend 경계에 연결했다. global profile을
사용하는 native frontend는 pair별 shard와 identity/capacity/freshness를 만들고,
admission 뒤 `tempo-go-reduction-receipt-v1`을 pair ledger와 reject ledger에
남긴다. analyzer는 raw/forwarded/omitted candidate count와 reduction receipt를
검증한다. 따라서 scale boundary가 설계 문서에만 있는 것이 아니라 decision
provenance의 일부가 됐다.

reducer는 local shard가 최종 business authority가 되도록 만들지 않는다. shard는
bounded frontier와 omission receipt만 만든다. tenant SLO/fairness, shared
remote budget, survivor capacity, service-lane ownership, joint local/remote/KV/
semantic actuation과 failure lifecycle은 계속 global orchestrator가 최종
commit한다. 이는 기존 component를 빼서 TEMPO를 축소한 것이 아니라, Perlmutter
전체에서 각 subsystem의 signal을 global business decision으로 안전하게 모으기
위한 ownership 구조다.

### 27.2 현재 CPU/contract gate

현재 변경 이후 집중 regression suite는 `174 passed, 22 subtests passed`다.
1024-pair fan-in의 최근 측정은 다음과 같다. 수치는 global decision stage만
측정하며 native inference 성능을 뜻하지 않는다.

| pair | raw candidate | forwarded | central full (ms) | reduced global (ms) |
|---:|---:|---:|---:|---:|
| 2 | 4 | 4 | 0.276 | 0.177 |
| 16 | 32 | 4 | 0.661 | 0.175 |
| 64 | 128 | 16 | 2.291 | 0.425 |
| 256 | 512 | 64 | 8.695 | 1.435 |
| 1024 | 2048 | 256 | 30.572 | 4.991 |

이 결과는 중앙 후보 평가의 fan-in 비용을 줄일 수 있다는 control-plane
evidence다. reducer construction/serialization, cross-node message bytes,
omitted-frontier utility loss와 native coupled utility는 별도 gate로 남는다.

v45 current-source base contract와 immutable snapshot은 모두 verify됐다.
native 실행에 사용할 snapshot은 다음에 고정한다.

```text
base contract:
  results/tempo_go_c5_cross_layer_contract_v45_current_source/native_run_contract.json
  sha256=192fea5a7969e54d4e35991087fd4aea638cb8d1f55f3aa45d0b7b3027a357ba
  fingerprint=a801655da8a6575f16ec961ae5d9232254d90ab7ea2f0eabe8bcbac99deade07

native snapshot contract:
  results/tempo_go_c5_source_snapshot_v45_current_immutable/native_run_contract.json
  sha256=b414538ba89a692e021a5400097b01f7afe42cb34a52a74bdaf413d7bdeeb826
  fingerprint=e2e3baabe5faa41f1a1ac0504a31c546c8d534922e1a51299177d93780593bdd
  tree_sha256=51db45b56cde6fd284f75bad6ba8a0077c3b6c246b966eecfeda5b585db70ecb
  python_files=1799
  performance_claim_allowed=false
```

### 27.3 v44 native attempt의 정확한 판정

현재 allocation `57426952`에서 v44 outer step은 4-node/16-GPU로 시작했고,
동일 allocation의 2-node/8-rank NCCL+LMCache/NIXL co-job은 실제 observer를
발행했다. latest observed receipt에는 `correctness_met=true`, `rank_count=8`,
`producer_state=active`, `sequence=2278`, LMCache/NIXL p99 약 `28.470 ms`,
NCCL collective p99 약 `1.313 ms`가 있었다. 이것은 실제 coupled observer와
headroom이 살아났다는 evidence이지 C5 성능 결과가 아니다.

v44 C5 arms는 contract가 만들어진 뒤 co-job allocation-aware guard를 수정한
상태에서 실행돼, `run_lmcache_nixl_contention_2node_in_allocation.sh`의 frozen
source digest가 달라졌다. remote/predictor 등 arm stderr가 이 mismatch를
명시했고, native failure receipt는 `native_arm_process_failed`로 기록됐다.
일부 nested step은 이어서 Slurm `Error configuring interconnect`를 냈지만,
그 메시지는 이미 source-bound arm validity가 깨진 실행에서 발생했다. 따라서
v44를 NCCL/Cassini bottleneck의 negative, LMCache의 negative, TEMPO의
performance negative 어느 것으로도 사용하지 않는다.

현재 user-owned allocation의 기존 parent/co-job step은 임의로 cancel하지 않고
그 natural teardown을 기다린다. 다음 native step은 그 GPU/node scope가 비고,
v103 snapshot contract를 allocation 안팎에서 verify한 뒤에만 시작한다. 새
실행의 첫 preflight는 inference가 아니라 다음 세 가지다.

1. snapshot root에서 Python import/source identity 확인
2. 4-node outer step과 nested GPU step의 `scontrol show step` resource shape 확인
3. co-job observer readiness와 C5 arm launch를 분리한 bounded execution receipt

이 preflight가 통과해야만 seven-arm offered-population 비교를 진행한다. 다시
step/interconnect setup failure가 나면 execution/capability receipt로만 닫고,
그 원인을 성능 결론으로 확대하지 않는다.

### 27.4 지금의 연구 결론과 다음 결정

현재까지의 강한 결론은 다음이다.

- shared native inference에서 local/remote/decoder/KV/fabric 상태가 서로 다른
  속도로 악화되며 bottleneck이 이동한다.
- 따라서 global orchestrator는 필요하다. local route, predictor, queue/GPU,
  LMCache-only 또는 application-only controller만으로는 이 coupling을 닫지
  못한다.
- TEMPO의 가치 단위는 NCCL reader, Cassini vector, LMCache completion,
  vLLM admission을 각각 발명했는지가 아니라, 이를 business-aware global
  action과 scale-bounded control loop로 묶어 실제 utility를 개선하는가다.
- 아직 strongest fixed/predictor 대비 TEMPO win은 없다. 현재 native
  performance claim은 계속 false다.

다음 GO는 application-only threshold variant가 아니라 v103 snapshot으로 같은
offered population의 full TEMPO/app-only/network-only/predictor/strongest-fixed
비교를 닫는 것이다. 여기서 full TEMPO가 normal-load 손실 없이 combined
contention에서 SLO-goodput, tail, fairness와 failure utility를 개선하면
end-to-end Perlmutter cross-layer contribution을 주장할 수 있다. 개선이
없으면 이번에 구현한 global mechanism의 정확한 실패 범위를 reproducible
negative로 남긴다. 목표 자체를 축소하거나 component별 feature subtraction으로
재정의하지 않는다.

## 28. superseded native candidate: v42 snapshot of telemetry-aware frontier

§25의 v39 telemetry-aware frontier contract를 base로 현재 source를 다시
freeze한 v42는 v45 snapshot 전에 사용했던 historical candidate다.

- contract: `results/tempo_go_c5_source_snapshot_v42_57426952/native_run_contract.json`
- contract SHA/fingerprint: `d57f10d8b7df2aedba9bc94fec1597e6ccafeb2ae3fcb2474d0302e04191a326` / `a893a6a31cbdf7bf7eca7b2f866ae50cd17f3d202a33d4a5c2434385b2b33586`
- snapshot tree: `22d389dc0b62caf3aa392d4098a6d343e214d571a9e482719c6f3a9c8b5b16f9`
- full contract verify: passed
- snapshot import/CPU gate: v41 snapshot에서 `197 passed, 22 subtests passed`

v42의 fixed profile, workload, seven-arm population과 cross-layer co-job 조건은
v39 contract를 그대로 보존한다. snapshot은 controller Python과 co-job shell을
고정하고, 현재 shell launcher는 별도 digest guard로 묶는다. 따라서 live
workspace에서 source가 다시 바뀌어도 v42 candidate 자체는 변하지 않는다.
현재 allocation `57426952.53/.54`가 이미 다른 same-allocation experiment로
GPU/node를 사용 중이므로, 그 step이 끝나기 전에는 v42를 시작하지 않는다.
v42는 v45 current-source snapshot으로 대체됐다. v42 실행 경계에서는
observer readiness와 실제 vLLM P/D startup을 모두 통과하기 전에는
성능·bottleneck·orchestrator superiority claim을 만들지 않는다는 원칙만
보존한다.

## 29. 연구 가치에 대한 정정과 MRC 개념의 위치

이 절을 이 문서의 최상위 해석 규칙으로 둔다. **TEMPO의 가치를 “이미 있는
component를 조금씩 붙인 것”의 차집합으로 평가하지 않는다.** NCCL reader,
LMCache completion, Cassini/Slingshot counter, vLLM admission과 같은 개별
mechanism에 선행연구가 있다는 사실은 TEMPO의 end-to-end contribution을
감소시키지 않는다. Perlmutter급 shared system에서 이 신호들을 동일한
identity/freshness/uncertainty 계약으로 수집하고, tenant business objective와
같은 global state에서 admission·placement·pair scaling·route·concurrency·
staggering·failure recovery를 공동 commit하는 것 자체가 TEMPO의 연구 단위다.

arXiv의 [Multipath Reliable Connection (MRC) Transport](https://arxiv.org/html/2606.18170v1)
는 이 방향을 transport 관점에서 강하게 뒷받침하는 관련 개념이다. MRC가
제시하는 multipath/path-health, bounded in-flight, receiver backpressure,
service-time compensation, endpoint probe와 fast failover는 “대규모 AI
시스템은 compute만 보고 조절할 수 없고 network/endpoint 상태를 제어 loop에
넣어야 한다”는 설계 교훈을 준다. 그러나 MRC는 transport specification이며,
TEMPO의 target은 그 위에서 실제 inference business를 조절하는 global
orchestrator다. 따라서 다음 경계를 지킨다.

1. 현재 Perlmutter 실험에서 MRC를 구현했다고 주장하지 않는다.
2. Slingshot/NCCL의 privileged configuration이나 root/controller 권한을
   만들지 않는다.
3. official `LMCacheConnectorV1:UCX` data plane과 native NCCL/Slingshot
   telemetry를 관측·actuation 대상으로 유지한다.
4. MRC의 bounded-flight/path-health/failure semantics를 설계 reference로
   사용하되, TEMPO의 승리는 transport microbenchmark가 아니라 실제 vLLM
   P/D offered-population utility, SLO-goodput, tail, fairness와 recovery로
   판정한다.

그러므로 현재의 정확한 negative는 “TEMPO는 가치가 없다”가 아니다. 지금까지
닫힌 것은 route-only, queue-threshold, application-only reservation/circuit
후보가 strongest fixed/predictor 대비 whole-system utility를 만들지 못했다는
것이다. **아직 닫히지 않은 본래의 목표는 NCCL·Slingshot/Cassini·LMCache·
vLLM P/D·tenant business를 함께 넣은 cross-layer global orchestrator의
causal native validation이다.** 이후 실험은 component별 feature score가 아니라
같은 offered workload에서 전체 시스템의 병목 이동, service capacity, SLO
goodput, fairness, failure cost를 비교해야 한다.

### 29.1 다음 에이전트가 반드시 유지할 목표 문장

> TEMPO Elastic-PD를 실제 vLLM P/D 경로에 통합하고, Perlmutter급 shared HPC의
> contention에서 vLLM service/decoder, LMCache/UCX KV transfer, NCCL collective,
> GPU/topology, Cassini/Slingshot telemetry와 tenant business state를 하나의
> hierarchical global state/actuation loop로 묶어, 단순 predictor와 가장 강한
> 고정 정책보다 whole-system SLO-goodput·tail·fairness·failure utility가
> 유의미하게 좋은 최종 TEMPO-GO scheme을 native matched validation으로
> 확정하라. 각 component에 선행연구가 있다는 이유로 이 global orchestration
> contribution을 축소하지 말고, 실제 contention workload와 scale/overhead를
> 함께 증명하라.

## 30. v45 native execution-boundary receipts

v45 snapshot native를 시작하기 전에 같은 allocation `57426952`에서 bounded
resource preflight를 수행했다. 4개 node 모두 `CPUs/task=128`, `GPUs/task=4`,
`step_id=65`, stderr empty와 exit 0을 기록했다. nested cojob shape도 별도로
검증해 2 nodes/8 tasks/32 CPUs per task/4 GPUs per node가 step `68`에서
hostname-only command로 exit 0을 만들었다. 따라서 allocation 자체와 nested
Slurm shape가 항상 불가능한 것은 아니다.

그 뒤 v45 snapshot contract로 시도한 execution-only receipts는 다음과 같다.

| step/root | observed boundary | 판정 |
|---|---|---|
| `57426952.66`, `...v45_snapshot_57426952` | outer step에 `--exact --gpus-per-task=4`를 넣은 상태에서 cojob이 `More processors requested than permitted`로 시작하지 못함 | wrapper resource-shape failure; no inference/performance |
| `57426952.69`, `...v45_snapshot_retry1_57426952` | outer `--wait=10` 때문에 no-op ranks 1–3이 먼저 끝난 뒤 long-running rank 0이 kill됨; cojob child `.70`은 orphan으로 남아 observer를 계속 발행 | outer lifecycle failure; no C5 arm |
| `57426952.71`, `...v45_snapshot_retry2_57426952` | outer wait를 제거한 뒤 cojob rank 0–3이 `Error configuring interconnect`; rank 4–7은 NCCL 초기화 후 rank0 주소 connection refused | node/rank interconnect capability failure; no C5 arm |

retry2의 rank stderr는 NCCL `2.28.9`, `ncclRemoteError`와
`socketPollConnect: connect ... returned Connection refused`를 보존한다. 이
것은 실제 cross-layer launch path에서 관찰된 Slingshot/NCCL execution failure라서
capability/headroom receipt로는 중요하지만, TEMPO 성능 negative나 LMCache
negative로 확대하지 않는다. retry1의 orphan child는 user-owned finite
10,000-block cojob이므로 안전 규칙에 따라 강제 cancel하지 않고 natural teardown을
기다린다. active observer가 correctness를 유지하는 동안 v45 native를 중첩하지
않는다.

현재 다음 실행의 고정 조건은 다음과 같다.

1. historical v47/v103 immutable snapshot contract와 SHA는 lineage 보존을
   위해 변경하지 않는다. 현재 실행 contract는 v104 immutable snapshot이다.
2. outer `--exact`, outer `--gpus-per-task`, outer `--wait=10`을 사용하지 않는다.
   no-op ranks는 outer shell에서 장시간 유지하거나 별도 완료 경계를 둔다.
3. cojob observer readiness, nested step success, C5 arm launch를 서로 분리한
   receipt로 기록한다.
4. rank0 interconnect launch failure가 다시 발생하면 그 node/step capability
   receipt에서 멈추고, 다른 profile/threshold나 completed-only performance를
   계산하지 않는다.

따라서 현재 v45 native utility gate는 아직 `OPEN/UNPROVEN`이다. 실제 vLLM
P/D arms와 full/app-only/network-only/predictor/fixed same-population result가
생성되기 전에는 strongest fixed 대비 TEMPO 개선을 주장하지 않는다.

## 30. v45 immutable snapshot의 P4 fan-in 재측정

v45 immutable source snapshot을 import root로 두고, 같은 `GlobalRequest`/
`PairTelemetry` population에 대해 bounded reducer를 100회씩 재측정했다. 이
측정은 login-node CPU control-plane benchmark이며 native inference latency가
아니다. `full`은 shard당 모든 pair를 global reducer가 평가하는 경로이고,
`reduced`는 shard당 최대 2 pair/2 route만 global envelope로 전달하는 경로다.

| pairs | raw candidates | forwarded | omitted pairs | full reduce p50/p99 (ms) | reduced reduce p50/p99 (ms) | full/reduced envelope bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 4 | 0 | 0.233 / 0.340 | 0.234 / 0.307 | 3,413 / 3,413 |
| 16 | 32 | 32 | 0 | 1.403 / 2.330 | 1.415 / 1.709 | 21,624 / 21,623 |
| 64 | 128 | 128 | 0 | 5.802 / 10.995 | 5.795 / 7.687 | 84,266 / 84,265 |
| 256 | 512 | 256 | 128 | 18.241 / 22.475 | 15.205 / 18.497 | 221,868 / 130,600 |
| 1024 | 2,048 | 256 | 896 | 119.626 / 230.940 | 53.925 / 164.360 | 774,548 / 134,658 |

1024-pair에서 reduced envelope는 full의 17.4%이며 raw candidate population은
725,332 bytes였다. 이것은 global network/control payload를 줄이는 강한
근거다. 그러나 현재 `HierarchicalCandidateReducer.reduce()`는 한 Python
process 안에서 raw candidate를 먼저 모두 group/sort하므로 reducer p50 자체는
119.6→53.9 ms일 뿐이다. 즉 이 결과는 “Perlmutter 전체에서 계산이 이미
분산됐다”는 증거가 아니다. 실제 scale contribution을 닫으려면 pair/node
agent가 local frontier와 `PairTelemetryEnvelope`를 먼저 만들고, shard/global
coordinator에는 bounded frontier·omission receipt만 보내는 호출 경계를 실제
runtime path에서 사용해야 한다. 이 gap은 다음 native utility run과 별도로
P4 implementation gate로 남긴다.

## 31. 현재 continuation: v46 distributed frontier 구현과 native GO 경계

§30의 gap을 단순한 문서상의 future work로 남기지 않기 위해 현재 source에
실제 호출 경계를 추가했다. `tempo/pd_global_hierarchy.py`의
`HierarchicalRequestHeader`, `PairCandidateFrontier`,
`build_pair_frontier()`, `reduce_frontiers()`,
`reduce_shard_frontiers()`가 다음 ownership을 강제한다.

1. pair agent가 자기 pair의 local/remote route 후보를 telemetry-aware rank로
   줄이고 bounded `PairCandidateFrontier`를 만든다.
2. shard agent가 pair frontier를 다시 shard budget으로 줄이며 raw pair 수,
   raw candidate 수, forwarded pair 수, omitted pair 수와 shard identity를
   receipt로 보존한다.
3. global coordinator는 raw candidate population을 다시 받지 않고 bounded
   frontier와 omission receipt만 받아 최종 business/fairness/cross-layer
   decision을 내린다.
4. source epoch, topology/profile fingerprint, sequence, node identity와
   cross-layer support가 다르면 fail-closed한다. 즉 fan-in을 줄이는 대신
   global decision authority를 pair/shard agent로 넘기지 않는다.

이 경계는 component feature의 합이 아니다. local GPU/decoder capacity,
remote KV/LMCache transfer, NCCL/Cassini fabric pressure, tenant deadline와
fairness를 global action에 넣기 위해 관측과 후보 population의 ownership을
분산하고, 최종 policy commit은 global에 남기는 TEMPO의 scale mechanism이다.

현재 source CPU gate는 기존 suite와 함께 다음을 통과했다.

- `tempo/test_pd_global_hierarchy.py`: 9 passed
- v46 snapshot import root의 동일 hierarchy test: 9 passed
- current source `py_compile`: passed
- v46 contract verify: passed; candidate
  `tempo-go-cross-layer-seven-arm-v16-distributed-frontier`, contract SHA
  `0875ac3c5f05bd1904a3b4cf8341ed59b22dd3d4223afb6707207b9b4f7737c8`
- v46 snapshot root:
  `results/tempo_go_c5_source_snapshot_v46_distributed_frontier_current_immutable_57426952`

같은 population을 raw global reducer와 shard-frontier global reducer로 다시
50회씩 재측정한 CPU control-plane 결과는 다음과 같다. 64/256/1024 pair에서
각각 raw reducer p50이 `6.356/15.509/55.875 ms`, shard-frontier global stage
p50이 `12.969/17.622/18.300 ms`였다. 1024 pair에서 global로 전달된 pair
envelope는 256개, 직렬화 payload는 166,984 bytes였고, raw candidate를 global이
직접 보던 670,037-byte population을 global 네트워크 경계 밖으로 밀어냈다. 이 결과는
분산 ownership이 필요한 이유와 control-plane fan-in 감소를 보여주지만,
native inference latency나 TEMPO speedup을 뜻하지 않는다. 특히 pair-agent의
local frontier build, serialization, network transfer와 omission에 따른
utility loss는 native matched workload에서 함께 측정해야 한다.

따라서 현재 연구 상태는 명확하다. TEMPO global mechanism의 구현 방향은
확정되었고, “component를 조금씩 붙인 것뿐”이라는 해석은 폐기한다. 그러나
최종 논문 성능 결론은 아직 열려 있다. 기존 allocation의 `.53`과 `.54`는
이제 종료됐지만 native C5 결과를 만들지 못했다. 다음 allocation에서 v104
immutable contract를 verify하고, 같은 offered population의 seven-arm native
matched run을 수행한다. 그때
비교할 것은 단일 route latency가 아니라 combined contention에서의
SLO-goodput, p99 tail, tenant fairness, pair scaling, fabric/LMCache failure
recovery와 orchestrator overhead다. 이 native utility가 strongest fixed와
simple predictor를 이겨야만 TEMPO의 최종 speedup claim을 열 수 있다.

## 32. `57426952` allocation 종료 영수증과 목표 유지

위 실행은 더 이상 진행 중인 작업이 아니다. allocation `57426952`는
`2026-08-22T12:58:08Z`에 시작해 `14:25:52Z`에 종료됐고, parent job은
`FAILED 130:0`으로 기록됐다. 이 allocation에서 보존할 수 있는 결과는
다음처럼 경계를 나눠야 한다.

- step `.65`: 4 nodes/16 GPUs에서 node별 `CPUs/task=128`, `GPUs/task=4`와
  empty stderr를 확인한 bounded preflight 성공
- step `.68`: 2-node/8-task/32-CPU-per-task/4-GPU-per-node nested shape의
  hostname-only preflight 성공
- step `.66`: `--exact --gpus-per-task=4` wrapper가
  `More processors requested than permitted`로 co-job을 시작하지 못함
- step `.69`: outer `--wait=10` lifecycle 때문에 no-op rank가 먼저 끝나고
  long-running rank가 종료됨
- step `.71`: wrapper wait를 제거한 실제 co-job에서 `nid001037`의 ranks
  0–3이 `Error configuring interconnect`로 launch 실패; ranks 4–7은
  NCCL 2.28.9 초기화 중 rank0 주소에 `Connection refused`
- step `.73`: 살아 있는 observer를 재사용한 C5-only 시도도 첫 native inner
  step에서 같은 interconnect launch boundary에 걸려 `143`으로 종료됨.
  metadata에는 C5 start/end만 있고 arm result는 없다.

따라서 이 allocation은 TEMPO의 성능 negative가 아니다. 실제
NCCL/LMCache/vLLM cross-layer launch path와 Perlmutter execution boundary를
드러낸 capability receipt이지만, vLLM P/D seven-arm의 matched
SLO-goodput·tail·fairness 비교가 아니기 때문이다. `udiRoot.conf`나 root
권한, container/UDI 설정은 건드리지 않았고, system bypass도 하지 않았다.

연구 목표는 여기서 바뀌지 않는다. **TEMPO는 local decoder, remote KV/LMCache,
NCCL collective, Cassini/Slingshot fabric, GPU/topology와 tenant business
state를 하나의 global closed loop에서 조절하는 시스템이다.** admission,
routing, telemetry, pair scaling을 각각 이미 알려진 부품이라고 세어
TEMPO의 기여를 깎지 않는다. TEMPO의 핵심은 contention이 심해질수록
병목이 어느 layer로 이동했는지 모르는 shared HPC에서, 이 heterogeneous
state를 같은 identity/freshness/uncertainty 아래 결합하고, business-aware
global action을 안정적으로 commit하는 scale/correctness/performance다.

다음 native 단계는 새 4-node/4-hour interactive allocation에서 v103 immutable
snapshot을 verify한 뒤, 동일 offered population으로 full TEMPO, app-only,
network-only, simple predictor와 strongest fixed를 동시에 실행하는 것이다.
새로운 threshold나 component-only experiment를 추가하지 않는다. 먼저
observer readiness, nested co-job, C5 arm launch를 각각 별도 receipt로
닫고, 모두 성공한 경우에만 combined-contention 성능 gate를 계산한다.

## 33. v47 Perlmutter Slingshot/OFI launch correction과 성능수치 수집 계획

### 33.1 `.71/.73`의 원인과 정확한 범위

이 문제는 TEMPO policy가 느려서 생긴 것이 아니라, C5 request가 시작되기
전에 native step이 성립하지 않은 실행 경계 문제다. 관측 순서는 다음과 같다.

1. `.71`에서는 `nid001037`의 co-job ranks 0–3이 Python/NCCL collective
   측정 전에 Slurm `Error configuring interconnect`로 launch 실패했다.
2. 같은 step에서 살아남은 ranks 4–7은 NCCL bootstrap 중 rank0 endpoint에
   `Connection refused`를 냈다. 따라서 이 refusal은 “Slingshot path의
   throughput이 낮다”는 측정값이 아니라, peer rank0이 존재하지 않는
   상태에서 생긴 2차 bootstrap failure다.
3. `.73`의 C5-only retry도 arm result를 만들기 전에 같은 Slurm interconnect
   경계에서 종료했다. 따라서 이 allocation에는 C5 E2E median, p99,
   SLO-goodput, fairness를 계산할 complete population이 없다.

Perlmutter의 [NERSC DDT 실행 문서](https://docs.nersc.gov/tools/debug/ddt/)
는 실제 `srun`에 CPU/GPU 자원 플래그를 전달하지 않으면 같은
`Error configuring interconnect`가 발생할 수 있음을 명시한다. [NERSC
Perlmutter job 문서](https://docs.nersc.gov/systems/perlmutter/running-jobs/)
의 원칙대로 GPU/task와 CPU/task를 실제 nested step에 명시한다. 또한 HPE
[Slingshot CCL/NCCL 문서](https://github.com/HewlettPackard/shs-ccl-docs)는
multi-node 실행에서 Slingshot 환경을 로드하고 `--network=disable_rdzv_get`
를 `srun`에 전달하는 실행 형태를 사용한다.

기존 co-job launcher에는 두 가지 치명적인 계약 위반이 있었다.

- `NCCL_NET=Socket`이 Perlmutter NERSC module이 제공하는 `NCCL_NET="AWS
  Libfabric"`를 덮어썼다. `NCCL_IB_DISABLE=1`도 함께 설정돼 native OFI/
  Libfabric 경로를 의도적으로 배제했다. [NVIDIA NCCL environment
  문서](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
  에서 `NCCL_NET`은 transport 선택을 강제하는 변수이며, `Socket`은
  external fabric plugin을 쓰는 설정이 아니다.
- co-job과 C5 inner `srun`에 `--network=disable_rdzv_get`가 빠져 있었다.
  nested step resource shape가 hostname-only preflight에서 통과해도, 실제
  interconnect-enabled launch에서 별도의 실패가 날 수 있다.

v47은 이 원인을 고쳤다. `pytorch/2.8.0`이 로드한 native NERSC stack의
`NCCL_NET="AWS Libfabric"`를 보존하고 `NCCL_IB_DISABLE`를 제거하며, 실제
co-job/C5 nested `srun`에 `--network=disable_rdzv_get`를 넣었다. launcher는
transport가 다시 Socket으로 오염되면 즉시 fail-closed하고
`native_transport_receipt.json`에 NCCL/UCX/FI 환경, node/step shape,
`slingshot_path=nersc-nccl-ofi-libfabric`와 production-transport 검증 여부를
기록한다. 이 수정은 root, UDI, container, CAP_NET_ADMIN, system file을
건드리지 않는다.

### 33.2 v47 native GO 순서

새로 승인된 4-node/4-hour interactive allocation에서 아래 순서를 지킨다.
기존 workdir가 다른 user job은 사용하거나 종료하지 않는다.

1. allocation/step의 node, GPU, CPU, GPU UUID/PCI bus, loaded module,
   `NCCL_NET`, `NCCL_SOCKET_IFNAME`, `FI_*`, `UCX_*`와 immutable v47 contract
   hash를 capability receipt로 고정한다.
2. 같은 allocation 안에서 2-node/8-rank co-job의 작은 correctness slice를
   먼저 실행한다. `NCCL_DEBUG=INFO`는 이 단계에서만 사용하고, rank별
   `NCCL_DEBUG_FILE`과 launcher stderr를 보존한다. `INIT,BOOTSTRAP,NET,GRAPH,
   TUNING` 로그가 native Libfabric plugin과 expected rank topology를
   가리키는지 확인한다.
3. co-job correctness와 observer readiness가 닫힌 뒤에만 실제 C5 inner
   step을 시작한다. outer `--exact`, outer `--gpus-per-task`, 짧은 outer
   `--wait`로 child rank를 죽이지 않는다. nested `srun`의 resource shape와
   `--network=disable_rdzv_get`를 receipt에서 확인한다.
4. 첫 slice가 complete terminal receipt를 만들면 같은 frozen offered
   population으로 seven-arm counterbalanced run을 수행한다. 결과를 본 뒤
   profile, threshold, arm order를 바꾸지 않는다. 실패하면 execution receipt로
   닫고 performance number를 계산하지 않는다.

### 33.3 성능수치의 계층과 source of truth

성능수치는 하나의 도구가 아니라 같은 `request_id × pair_id × node_id ×
endpoint_id × sequence × topology/profile epoch`를 공유하는 계층형 receipt로
수집한다. 모든 timestamp는 monotonic clock과 wall-clock correlation을 함께
기록하고, missing signal을 zero로 치환하지 않는다.

| 계층 | 반드시 수집할 값 | 용도 |
|---|---|---|
| business/request | offered/admitted/deferred/rejected/failed/completed, TTFT, TPOT, E2E, deadline miss, output-token goodput, SLO-goodput, max wait, tenant debt/fairness | 최종 TEMPO utility와 reject cost |
| vLLM P/D | prefill/decoder queue, active sequence, first-response/EOF credit, pair/route commit, service-lane wait, admission action | actuator가 실제 inference path를 바꿨는지 확인 |
| LMCache/UCX | official connector completion, KV bytes, transfer start/end, transfer p50/p95/p99, semantic op timeout/failure, local/remote residency | remote-KV 경로의 실제 완료와 tail |
| NCCL | actual collective type, bytes, sequence, CUDA completion duration, init/bootstrap result, communicator identity | NCCL collective externality와 rank straggler |
| Cassini/Slingshot | existing `tempo/cassini_endpoint.py`의 NIC×traffic-class vector, support/ambiguous/unsupported, window delta, pause/ECN/retry/timeout 계열 counter | 어느 NIC/TC에서 fabric signal이 변했는지 |
| GPU/topology | GPU UUID, PCI bus, node, GPU-NIC pairing, utilization, memory, power, PCIe/NVLink counters when supported | GPU bottleneck과 fabric bottleneck 분리 |
| Slurm/transport | job/step/node/task/GPU shape, module/version, NCCL/UCX/FI env, transport receipt, rank launch stderr | native launch validity와 provenance |

기존 `CudaCollectiveObserver`, official LMCache/NIXL completion observer와
Cassini endpoint sampler를 primary low-overhead source로 사용한다. NCCL log는
매번의 headline latency source가 아니라 transport/init 진단과 communicator
identity 검증용이다. production-like run에서는 INFO logging을 끄고 structured
observer만 남긴다. `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET,
GRAPH,TUNING`, `NCCL_DEBUG_FILE` 설정은 launch/capability slice 또는 별도
diagnostic slice에 한정한다. [NVIDIA NCCL logging 문서](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/user-guide/docs/troubleshooting/logging.html)
의 방식으로 init/NET/GRAPH를 확인하되 log overhead를 final arm에 섞지 않는다.

GPU metric은 먼저 Perlmutter에서 허용된 DCGM/nvidia-smi capability를 read-only
probe한다. DCGM hostengine/field가 지원될 때만 GPU UUID를 durable identity로
사용해 utilization, power, memory, PCIe/NVLink counter를 붙인다. [NERSC NVIDIA
profiling 문서](https://docs.nersc.gov/tools/performance/nvidiaproftools/)
와 [NVIDIA DCGM field 문서](https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html)를 기준으로 하며, 지원되지 않는 field를
추정하거나 numeric GPU index를 영구 identity로 저장하지 않는다. DCGM이
권한/hostengine 문제로 unavailable이면 그 사실을 receipt에 남기고 existing
CUDA/NCCL/Cassini signal로 계속 진행한다.

### 33.4 Nsight와 eBPF의 위치

`nsys`는 모든 arm의 primary metric collector가 아니다. overhead를 포함한
대표 interval 한 개의 causal attribution에만 사용한다. native allocation의
Scratch에 rank별 report를 저장하고, `nsys profile -t nvtx,cuda,nccl
--stats=true`와 Perlmutter 문서의 Slurm-safe filename pattern을 사용한다.
[NERSC readiness 문서](https://docs.nersc.gov/performance/readiness/)와
[Nsight Systems NCCL tracing 문서](https://docs.nvidia.com/nsight-systems/UserGuide/)를 따른다. 한 normal interval과 한 combined-hot interval에서
NCCL wait, CUDA kernel gap, NVTX request phase, LMCache completion과 GPU
overlap을 확인하되, Nsight trace time을 seven-arm headline number에 넣지
않는다. Nsight Compute는 overhead가 커서 이 목표의 first-line tool이 아니다.

eBPF는 이 문제의 주 metric source가 아니다. GPU kernel, NCCL collective,
Cassini NIC/traffic-class counter를 unprivileged eBPF가 완전하게 대체하지
못하며, Perlmutter compute-node 정책/권한에 따라 attach가 거부될 수 있다.
따라서 `bpftrace`/`perf trace`는 launch 실패가 반복될 때 connect/exec/socket
sequence를 확인하는 bounded diagnostic으로만 선택한다. eBPF output으로
“Slingshot link가 병목”이라고 단정하지 않고, NCCL/Cassini/DCGM/observer
receipt를 대체하지 않는다. background watcher나 system-wide probe를 만들지
않고, 승인된 step 안에서 짧은 diagnostic interval만 기록한다.

`MPICH_OFI_CXI_COUNTER_REPORT`는 NERSC가 MPI/SHS timeout 분석용으로 제공하는
경로이므로 NCCL OFI 성능의 직접 대체 source로 쓰지 않는다. [NERSC network
문서](https://docs.nersc.gov/performance/network/)의 CXI report는 MPI
counter timeout이 있을 때 보조 receipt로 남기고, NCCL에는 existing Cassini
sysfs reader와 NCCL observer를 사용한다.

### 33.5 bottleneck attribution과 TEMPO claim boundary

각 window에서 다음 causal tuple을 만든다.

```text
(action_commit,
  request/pair/tenant identity,
  vLLM queue + decoder service time,
  LMCache completion tail,
  NCCL collective tail/bytes,
  Cassini NIC×TC delta + support,
  GPU utilization/topology,
  SLO-goodput/fairness outcome)
```

이 tuple로 “어디가 병목인가”를 다음처럼 구분한다.

- NCCL p99와 rank straggler가 오르고 Cassini NIC/TC vector도 같은 window에서
  support 상태로 악화되며 GPU가 collective wait에 있으면 NCCL/fabric-coupled
  externality evidence다.
- LMCache completion p99만 오르고 NCCL/Cassini가 안정적이면 remote KV/
  endpoint/UCX path evidence다.
- GPU utilization이 포화되고 queue/service time이 늘지만 collective와
  Cassini가 안정적이면 compute/decoder saturation evidence다.
- 서로 다른 layer가 다른 시간에 악화되면 단일 bottleneck label로 합치지
  말고 uncertainty와 source epoch를 보존한다. physical switch/link claim은
  per-link evidence가 있을 때만 별도 주장한다.

TEMPO의 최종 비교는 이 signal 자체의 예쁜 그래프가 아니라, 같은 offered
population에서 global action이 위 tuple의 coupled externality를 줄였는지다.
full TEMPO, APP_GLOBAL_ONLY, NETWORK_REQUEST_ONLY, predictor, strongest fixed와
같은 topology/workload를 ABBA/counterbalanced로 실행하고, 모든 arm에서
reject/defer/failure를 포함한 E2E/SLO-goodput/p99/fairness를 계산한다. native
transport receipt가 없거나 observer identity가 섞이면 그 arm은 성능 비교에서
제외하고 execution failure로 닫는다.

현재 이 절의 STOP/GO는 다음과 같다.

- GO: v47 source/contract verify, focused hierarchy tests `9 passed`, shell
  syntax/static tests `8 passed`, Perlmutter docs에 맞춘 OFI/Slingshot launch
  correction과 transport receipt 구현.
- STOP: v47 native matched C5 performance number는 아직 없음. 새 승인
  allocation에서 transport/correctness/C5 launch receipt가 순서대로 닫히기
  전에는 TEMPO positive도 negative도 주장하지 않는다.
- NEXT: v47 capability slice → small NCCL/LMCache correctness → C5 readiness
  → frozen seven-arm performance → targeted Nsight attribution → independent
  validation. 성능 결과를 본 뒤 launcher/profile을 고치지 않는다.

## 34. v47 lifecycle-safe tmux native receipt: contention은 실재하지만 TEMPO utility는 아직 미닫힘

§33의 launch-correction 계획과 별도로, `v47_lifecycle_safe` immutable
snapshot으로 실제 4-node/16-GPU native campaign을 수행한 영수증을 보존한다.
이 절은 Slingshot/OFI capability와 inference utility를 섞지 않기 위한
실행 후 보정이다. 사용한 job은 `57430779`, node는
`nid[001013,001016,001057,001065]`, output root는
`results/tempo_go_cross_layer_native_v47_lifecycle_safe_tmux`이고,
immutable contract는
`results/tempo_go_c5_source_snapshot_v47_lifecycle_safe_current_immutable_57430054/native_run_contract.json`이다.
contract file SHA는
`44f7455f83cb7c83a946078d51cb5975c79c3cfdf2200ece0b1312054e34e814`,
fingerprint는
`9ef946f63976a65f952d38166481de779e2564e1ad0a29d15ee5995b18ded32b`이다.
모든 native arm은 실제 vLLM P/D endpoint를 기동했고 workload는 276
requests, `max_workers=128`, seed `20260815`였다.

### 34.1 무엇이 닫혔는가

co-job의 `results/tempo_go_cross_layer_cojob_v47_lifecycle_safe_tmux/nccl_observer.json`에는
8-rank producer correctness, `producer_state=complete`, sequence `10001`,
NCCL collective p99 `0.429873 ms`, LMCache transfer p99 `29.101359 ms`,
topology/source epoch가 기록됐다. 즉 이 run은 local-only toy path가 아니라
실제 inference와 별도로 살아 있는 NCCL/LMCache cross-layer pressure source를
기동했다. TEMPO raw receipt에도 hierarchical fan-in이 실제 호출되어
reductions `259`, telemetry refresh `105`, cache hit `170`, queued `234`,
admitted `160`이 남았다. pair→shard→global ownership과 telemetry-aware
decision path가 코드상 mock-only가 아니라 native request lifecycle에 연결된
것은 확인됐다.

이것은 TEMPO의 가치를 구성요소 feature 차감으로 평가할 수 없다는 강한
mechanism evidence다. local decoder, remote KV/LMCache, NCCL/Slingshot
externality, GPU capacity, tenant request admission은 한 request의 동일한
상태/identity/sequence 아래에서 서로 영향을 준다. component 각각에 선행
연구가 있다는 사실은 이 global coupling을 없애지 않는다.

### 34.2 같은 offered population에서 실제로 보인 contention

완료된 fixed/predictor/network arm은 각각 276/276 terminal-valid였고,
각 arm의 client-side stream receipt에서 계산한 값은 다음과 같다. 단위는
ms이며, 여기에는 이 workload의 scheduled dispatch와 queue wait가 포함된다.

| arm | p50 E2E | p95 E2E | p50 TTFT | route population |
|---|---:|---:|---:|---|
| local | 28,057.9 | 67,163.7 | 22,182.7 | local 276 |
| remote | 26,120.5 | 54,930.6 | 18,944.4 | remote 276 |
| predictor | 25,884.9 | 57,816.6 | 19,984.3 | local 264 / remote 12 |
| queue_gpu | 22,523.7 | 41,805.0 | 17,594.4 | local 66 / remote 210 |
| network_request_only | 24,558.9 | 55,225.9 | 21,055.9 | local 274 / remote 2 |

따라서 현재 workload에서는 local/remote 선택과 GPU queue admission이
E2E tail을 수 초 단위로 바꾼다. 특히 queue/GPU arm은 이 slice에서 가장
낮은 p50/p95를 보였지만, 이것만으로 strongest-fixed 전체의 최종 승리나
TEMPO의 패배를 선언하지 않는다. workload/profile이 frozen counterbalanced
seven-arm contract와 완전히 닫혔는지, fairness와 failure cost까지 포함한
independent validation이 아직 필요하다. 다만 “contention이 없어서
orchestrator가 불필요하다”는 가설은 이 결과와 양립하지 않는다.

### 34.3 현재 TEMPO가 무너진 정확한 지점

TEMPO arm의 raw receipt는
`results/tempo_go_cross_layer_native_v47_lifecycle_safe_tmux/tempo/tempo_go_c5_discovery/raw.json`에
있다.

- request `276`건 중 `160`건 admitted/completed path, `116`건 global reject;
  `performance_claim_allowed=false`, `terminal_contract_valid=false`다.
- terminal errors는 `global_admission_queue_timeout=115`,
  `global_telemetry_refresh_timeout=1`, HTTP `503=23`이다.
- coordinator는 `queued=234`, `queue_leases=53`, `delivered_from_queue=82`,
  `service_lane_reservation_failures=23`, `route_failures=0`을 기록했다.
- hierarchical fan-in은 `reductions=259`까지 진행됐지만
  `identity_rejections=16`이 발생했다. telemetry는 105 refresh 중 1 timeout,
  170 cache hit를 기록했다.

즉 현재 failure는 “global orchestration의 가치가 없다”는 결과가 아니다.
오히려 global controller가 실제 overload를 보고 admission/queue/telemetry/
service-lane을 함께 건드리는 경계까지 도달했으며, 그 capacity policy가
offered population을 보호하지 못해 queue timeout과 503으로 변환한 것이다.
다음 수정의 목표는 timeout 숫자를 숨기는 것이 아니라, 같은 offered
population에서 다음을 보장하는 것이다.

1. service-lane capacity를 초과하기 전 degradation-aware local/remote/
   defer decision을 commit하고, bounded queue가 유효한 business deadline 안에서
   drain되지 않으면 명시적인 reject/debt/fairness receipt로 끝낸다.
2. telemetry freshness/identity mismatch는 안전한 bounded recovery 또는
   uncertainty 상승으로 처리하되, 정상적으로 설치된 같은-allocation state를
   16회 identity rejection으로 곧바로 usable decision에서 제거하지 않는다.
3. queue lease, endpoint reservation, first-response credit, HTTP EOF credit을
   하나의 admission budget으로 결합하고, 503을 controller success로 세지 않는다.
4. full TEMPO가 overload를 피하기 위해 무조건 local로 숨거나 offered load를
   버리지 않도록 app-only/network-only/predictor/strongest-fixed와 동일한
   reject, defer, tail, fairness denominator를 유지한다.

APP_GLOBAL_ONLY도 `exit_code=1`로 종료되어 complete performance arm이 되지
못했다. 따라서 app-only 대비 incremental gain은 계산하지 않는다. co-job의
NCCL/LMCache observer는 correctness를 닫았지만, TEMPO가 C5 end까지 가지 못해
`cross_layer_binding.json`은 생성되지 않았다. 이는 co-job source가 없었다는
뜻이 아니라 TEMPO native campaign의 terminal binding이 닫히지 않았다는
뜻이다.

### 34.4 결론과 다음 GO/STOP

현재까지의 연구 결론은 세 줄로 고정한다.

1. **문제는 실재한다.** 4-node Perlmutter-like shared inference에서 local
   decoder, remote KV/LMCache, queue/GPU, NCCL/LMCache co-load가 서로 다른
   latency/tail과 admission pressure를 만든다. 따라서 global orchestrator는
   필요하다.
2. **TEMPO mechanism은 실제 경로에 들어갔다.** pair/shard/global frontier,
   cross-layer telemetry와 business admission authority가 native vLLM
   lifecycle에 연결됐다. TEMPO를 “기존 component 몇 개를 붙인 것”으로
   평가하지 않는다.
3. **최종 성능 승리는 아직 없다.** v47 arm은 TEMPO/app-global의 terminal
   contract가 닫히지 않았으므로 speedup, goodput win, fairness win을 주장하지
   않는다. 현재의 정확한 negative는 “이 global capacity/admission loop가
   overload에서 queue timeout/503을 내며 실패했다”는 mechanism-scoped
   execution/utility failure다.

다음 native GO는 새로운 component-only 실험이나 threshold 숫자 바꾸기가
아니다. v47 launch-correction contract와 immutable source를 새 승인
4-node/4-hour interactive allocation에서 capability→NCCL/LMCache
correctness→C5 readiness 순서로 닫고, 같은 offered population의 frozen
seven-arm run을 다시 수행한다. 그때 primary gate는 full TEMPO가 strongest
fixed와 predictor보다 normal-load 손실 없이 combined contention의
SLO-goodput/E2E p99/fairness/failure recovery를 개선하는지다. 이번 v47의
숫자는 그 gate를 통과한 결과가 아니라, 왜 이 gate와 global control이
필요한지를 보여주는 native receipt로만 사용한다.

## 35. v48--v51 Perlmutter native continuation: global control is connected, but the overload loop is not closed

이 절은 v47 이후의 결과를 구성요소별 성능 차감으로 해석하지 않고, TEMPO가
Perlmutter의 shared GPU/NCCL/LMCache/endpoint/business state를 하나의
allocation-scoped decision으로 묶었을 때 실제로 어디에서 무너지는지를
기록한다. 모든 run은 같은 승인 allocation `57433031`의 4-node/16-A100
interactive environment에서 수행했고, vLLM P/D 요청과 2-node NCCL/LMCache
co-job을 같은 allocation 안에서 겹쳤다. login node에서 substantial workload를
실행하지 않았고, UDI/container/root/ownership 경계를 건드리지 않았다.

### 35.1 source/profile identity와 실행 순서

| candidate | global profile | immutable contract | native root | 핵심 변화 |
|---|---|---|---|---|
| v48 | `tempo-go-qwen25-perlmutter-short-slice-v48-global-bounded-lease`, profile SHA `cd2bdd8f845b8943f8c61b626bfb235d66d0b7072f8cee5b4a9319fa979d72d9`, fingerprint `7f283398a0614cb66aa368136a3ea08177ca01875c18039c52897a1d2baa914b` | `results/tempo_go_c5_source_snapshot_v48_global_bounded_lease_immutable/native_run_contract.json`, SHA `0533859592d970317044bf88663cca2ef972b835161a6455699a5aa4c8053701`, fingerprint `542db96dd55d11659a51cc5ff0bf4642f933d0fb9a4c0a9959e29bdeca2c95fe` | `results/tempo_go_c5_cross_layer_native_v48_global_bounded_lease_57433031` | bounded endpoint queue lease, 100 ms telemetry freshness |
| v49 | profile SHA `52cbe927858d07ed9559ef7c5780ffe2f36b907fd5fe56d49e7bad2eb7266131`, fingerprint `1af69103c30b8977cb73d402826864c937881092183124255f1e65ec4619be8d` | `results/tempo_go_c5_source_snapshot_v49_deadline_freshness_immutable/native_run_contract.json`, SHA `8812935558d86dbea186ba0ec9c01278d9667492ed88ff4987688ca6d62009a2`, fingerprint `6bd645bb1c21498506f23b8bf9278aea49a7439ddb1ffb601e03a5f3e8144b16` | `results/tempo_go_c5_cross_layer_native_v49_deadline_freshness_57433031` | freshness 250 ms; frontend가 TEMPO decision 이후 남은 deadline을 vLLM endpoint에 전달 |
| v50 | v49 profile 유지 | `results/tempo_go_c5_source_snapshot_v50_global_endpoint_queue_immutable/native_run_contract.json`, SHA `16ee81c98f35b74154d4067918630e9b01cdc653c299340ce4ba50afcd1ab389`, fingerprint `c147d40215abebe255eb49e40cc034c3e07edd04012ae74707572358a333ae9e` | `results/tempo_go_c5_cross_layer_native_v50_global_endpoint_queue_57433031` | global route가 이미 request를 소유하면 endpoint queue도 bounded retry |
| v51 | `tempo-go-qwen25-perlmutter-short-slice-v51-telemetry-budget`, profile SHA `c30b3b8a936cf368d071842fced3d27c0be35613d69e5b1e470f651a1aee0ed8`, fingerprint `205b6d577d12c5b87735758088bf429fbbcb6f292e61c291ea112fd3192bb767` | `results/tempo_go_c5_source_snapshot_v51_telemetry_budget_immutable/native_run_contract.json`, SHA `82f5c647df84a4aad46f45b71aa54741d65a629c505f345527bffcc64ce076d5`, fingerprint `0cd6168980f8ae39b6f0943592c65ae51558d4b345caa22c6fccce517d2690fb` | `results/tempo_go_c5_cross_layer_native_v51_telemetry_budget_57433031` | freshness 250 ms, refresh timeout/collection span 100 ms |

모든 contract의 arm order는 `local, remote, predictor, queue_gpu,
network_request_only, app_global_only, tempo`이며, v49 fixed arms는 각각
276/276 stream-valid이었다. `app_global_only`는 v48/v49 모두 native
baseline execution failure로 닫히지 않았으므로 app-only 대비 성능 숫자는
계산하지 않는다. v48의 full campaign analyzer는 TEMPO가 닫히기 전에
실패해 partial raw를 남겼고, v49 analyzer는 native 실행 후 live shell
digest가 contract 시점과 달라 독립 analyzer gate를 통과시키지 않았다. v50과
v51 single-arm analyzer는 immutable Python snapshot과 contract를 검증했지만,
둘 다 `performance_claim_allowed=false`인 execution-failure receipt다.

### 35.2 native contention에서 확인된 전체 경로의 병목

v48 TEMPO raw의 validation은 `all_streams_valid=false`, completed `154`,
global reject `122`였다. coordinator snapshot은 `queue_leases=55`,
`queue_timeouts=112`, `service_lane_reservation_failures=3`,
`hierarchical_fan_in.reductions=266`, `identity_rejections=10`,
`route_failures=0`을 기록했다. terminal error는 global queue timeout
`122`, HTTP 503 `3`, HTTP error `15`였고, explicit queue lease 중에는
endpoint bounded lease receipt가 실제로 accepted 되었다. 즉 lease handshake와
hierarchical global path는 native에 연결되었지만, overload에서 admission debt가
endpoint와 같은 business deadline으로 닫히지 않았다.

v49는 v48의 두 boundary를 분리했다.

- 100 ms freshness에서는 contention 중 telemetry sampled state가 실제로
  stale해졌다. freshness를 250 ms로 늘리자 hierarchy stale reject는
  `10 -> 4`로 줄었고, 별도로 refresh timeout `3`건이 관찰됐다.
- v48의 frontend는 global coordinator에서 이미 사용하는 16 s endpoint
  deadline을 backend에 전달하지 않아, downstream v448 queue가 기본 1 s로
  잘렸다. v49는 global request가 endpoint에 도착할 때의 실제 남은 deadline을
  전달했고, lease의 global queue wait는 약 10--11 s 범위로 관측됐다.
- 그런데 global local/remote commit 뒤 endpoint controller가 순간적으로
  `QUEUE`를 반환하면, 기존 router는 이를 곧바로
  `endpoint_service_lane_capacity_unavailable`로 바꿨다. v49에서는 이
  immediate service-lane failure가 `38`건까지 나타났다. 이것은 GPU/NIC가
  부족하지 않다는 뜻이 아니라, global ownership과 endpoint physical queue의
  credit boundary가 분리돼 있었다는 뜻이다.

v50은 global-owned route의 endpoint `QUEUE`를 bounded global-route wait로
  수용하도록 바꿨다. 그 결과 service-lane immediate failure는 `38 -> 0`이
  되었고 모든 endpoint queue receipt가 `accepted`였다. 그러나 해당 요청은
  endpoint queue timeout으로 끝났고, validation은 completed `171`, global
  reject `105`, global queue timeout `99`, telemetry refresh timeout `6`,
  HTTP 503 `44`였다. 따라서 이 수정은 ownership/receipt semantics는
  올바르게 만들었지만 throughput이나 goodput을 개선한 것이 아니다. failure를
  다른 label로 옮긴 것과 실제 service capacity를 확보한 것은 구분한다.

v51은 telemetry collection 자체가 shared contention에서 병목인지 확인하기
  위해 freshness 250 ms는 유지하고 refresh timeout과 maximum collection span을
  50 ms에서 100 ms로 함께 freeze했다. validation은 completed `186`, global
  reject `90`, global queue timeout `87`, telemetry refresh timeout `3`,
  HTTP 503 `47`였고 coordinator는 `queue_leases=119`,
  `service_lane_reservation_failures=0`, stale reject `4`, refresh timeout `3`을
  기록했다. v50 대비 telemetry timeout은 줄었지만 endpoint bounded timeout은
  `47`로 남았고 full stream-valid gate는 여전히 false다. v51 single-arm
  descriptive metrics는 completed 139, request goodput 4.206/s,
  TTFT p50 5,478 ms, E2E p50 9,551 ms였으나 partial population이므로
  fixed/predictor 대비 speedup으로 보고하지 않는다.

### 35.3 이 결과가 TEMPO의 가치와 목표에 대해 의미하는 것

이 native sequence는 “각 component를 조금씩 붙였으니 TEMPO의 가치는 작다”는
해석을 지지하지 않는다. 실제로 하나의 request가 다음 상태를 동시에 필요로
했다.

`Perlmutter shared NCCL/Slingshot pressure → LMCache/UCX remote KV state →
local decoder queue/GPU credit → endpoint service-lane feedback → tenant
deadline/fairness → global pair activation and route debt`

v48--v51에서 이 상태들을 같은 global telemetry identity, pair/shard/global
frontier, admission ledger, endpoint receipt, first-response/EOF lifecycle로
연결한 것은 확인됐다. 특히 co-job이 살아 있는 동안 `path_skip`, global queue
timeout, stale/refresh failure, endpoint bounded timeout이 함께 변했고,
`route_failures=0`인 상태에서도 request-level utility가 무너질 수 있었다.
이는 isolated local-vs-remote predictor로는 관측하거나 제어할 수 없는
cross-layer failure surface다.

동시에 현재 TEMPO 구현은 아직 목표를 달성하지 않았다. strongest fixed인
`queue_gpu` 및 predictor와 동일 offered population에서 full TEMPO의
SLO-goodput/E2E-tail/fairness를 닫지 못했고, partial valid request의 낮은
latency를 성능 승리로 사용할 수 없다. 현재의 정확한 결론은 다음이다.

1. **contention 문제는 실존한다.** local decoder, remote LMCache/UCX,
   NCCL/LMCache co-load, endpoint queue, telemetry fetch가 동시에 나빠지고
   bottleneck이 이동한다. 따라서 Perlmutter-scale global orchestrator는
   필요하다.
2. **TEMPO mechanism은 실제 vLLM P/D 경로에 들어갔다.** global controller가
   business admission과 pair/fabric/endpoint state를 함께 소유하는 가치가
   native receipt로 확인됐다.
3. **현재 control loop는 overload에서 아직 실용적인 성능 승리를 못 냈다.**
   queue lease와 endpoint ownership을 고쳤지만 global queue timeout과
   endpoint bounded timeout이 남아 있어, 이 상태를 paper speedup으로 쓰면
   안 된다.

### 35.4 다음 실행 목표: component ablation이 아니라 global overload closure

다음 목표는 또 다른 local/remote/predictor component를 만드는 것이 아니다.
다음 native candidate는 다음 네 가지를 하나의 frozen controller로 묶어야
한다.

1. **Admission/scale coupling:** queue age/tenant reservation, active pair
   utilization, endpoint service-lane credit, remote semantic/fabric budget을
   동시에 보고 spare pair activation 또는 bounded defer를 결정한다. queue
   timeout 직전의 `path_skip`, capacity, deadline, survivor-reserve 사유를
   immutable receipt에 남겨 “왜 lease가 불가능했는지”를 관측 가능하게 한다.
2. **Telemetry resilience:** 250 ms freshness와 100 ms collection budget을
   baseline으로 두되, stale state를 조용히 재사용하지 않는다. refresh가
   deadline을 먹을 때는 uncertainty/route restriction/tenant-aware defer를
   명시적으로 선택하고, refresh failure가 모든 request를 opaque 503으로
   바꾸지 않게 한다.
3. **Endpoint/global credit unification:** global-owned local/remote commit,
   endpoint `QUEUE`, queue lease, first-response credit, EOF release를 하나의
   debt ledger로 연결한다. immediate reservation failure와 bounded timeout을
   분리하고, endpoint queue가 실제로 drain되지 않으면 tenant/fairness 비용을
   계산한 뒤 다음 request에 credit를 남긴다.
4. **Primary evaluation:** 새 contract는 fixed local/remote, predictor,
   queue_gpu, network-only, app-global-only, full TEMPO를 같은 276-row 이상
   offered population과 같은 same-allocation co-load에서 닫아야 한다. full
   TEMPO가 all-stream-valid가 아니면 speedup 결론은 금지한다. valid closure
   후에만 strongest fixed와 predictor 대비 SLO-goodput, E2E p95/p99,
   tenant fairness, rejection/debt recovery를 독립적으로 비교한다.

따라서 현재 연구 상태는 **STOP이 아니라 mechanism-connected / utility-not-closed**다.
문제의 실존성과 global orchestration 필요성은 native evidence로 고정됐고,
다음 작업의 성공 기준은 component 숫자가 아니라 같은 shared-contention
allocation에서 global TEMPO가 full terminal contract를 닫고 strongest fixed와
predictor를 business-aware utility로 이기는 것이다.

## 36. v53 queue-lease 원인 보존 시도: measured utility로 승격하지 않음

v51에서 고정 arm을 이미 닫은 뒤, `GlobalOrchestrator`가 endpoint queue lease를
시도했지만 후보를 모두 거부하는 경우에도 그 원인을 terminal
`global_admission_queue_timeout` decision에 보존하도록 수정했다. 대상은
`tempo/pd_global_orchestrator.py`의 bounded queue-lease handoff이며, 단일
component 성능 실험이 아니라 다음 global overload control을 선택하기 위한
관측 closure다. CPU 관련 회귀는 `9 passed`, 전체 관련 suite는 기존 dirty
policy test 2개를 제외하고 `292 passed, 36 subtests passed`였다.

새 immutable source snapshot과 contract는 다음이다.

| item | value |
|---|---|
| snapshot | `results/tempo_go_c5_source_snapshot_v53_queue_lease_causes_immutable` |
| contract SHA | `41916d143374e69b7b03c1030c2507bd6a4b5c2ae80f58795cc962b3900117bb` |
| contract fingerprint | `da4a21f318437f32275eff441a26c4ef2e7f504bdbc77d4c88ac354decaa3880` |
| native root | `results/tempo_go_c5_cross_layer_native_v53_queue_lease_causes_57433031` |
| allocation | `57433031`, same 4-node/16-GPU co-load |

v53은 warmup 24/24 stream-valid receipt까지 만들었으나 measured phase에서
node-1 vLLM EngineCore가
`LMCache CacheEngineKey(...chunk_hash=...) not found in local data`로 종료했고,
다른 node가 이를 따라 종료되며 Slurm step은 `rc=137`이 됐다. 이 결과에는
measured `raw.json`과 full terminal population이 없으므로 queue-lease 원인
분포, goodput, tail latency, strongest-fixed 대비 성능 어느 것도 계산하지
않는다. 이는 이번 receipt 보존 패치의 정책 효과를 검증한 run이 아니다.

그러나 이 실패는 연구 질문과도 일관된 경고를 준다. 같은 shared allocation의
LMCache/vLLM 상태 자체가 request phase 중 cache key ownership 불일치로
무너질 수 있으며, 이 상태를 TEMPO speedup으로 둔갑시키거나 단순 HTTP 503으로
숨기면 안 된다. 다음 native run 전에는 이 execution failure와 queue-lease
cause instrumentation을 분리한다. 즉, `CacheEngineKey` failure는
**shared-cache execution failure receipt**이고, queue lease 후보별 원인은
**global admission receipt**여야 한다. 둘을 하나의 timeout 숫자로 합치지 않는다.

따라서 v53의 결론은 “TEMPO가 느리다”도 “TEMPO가 이겼다”도 아니다.
정확한 상태는 **instrumentation은 CPU에서 연결됐고, native measured window는
LMCache cache-state failure로 닫히지 않았다**이다. 다음 global control 후보는
이 원인 분리를 전제로 admission/scale coupling, endpoint credit, telemetry
freshness, LMCache/NCCL shared-fabric pressure를 하나의 business-aware debt
ledger에서 조절해야 한다.

## 37. v54--v56 global overload loop closure 결과

v53에서 얻은 cache-state failure와 queue-lease 원인 보존을 바탕으로, 다음
세 단계의 global control을 같은 승인 allocation `57433031`에서 이어서
검증했다. 이 절의 숫자는 모두 discovery/diagnostic receipt이며, fixed arm과
동일한 독립 full-valid population을 닫지 못했으므로 성능 승리 주장이 아니다.

### 37.1 v54: cache-failure circuit profile

v54 profile은 `telemetry_failure_quarantine_mode=deny_until_probe`,
`telemetry_failure_quarantine_scope=pair`, survivor capacity reserve `0.25`,
survivor bypass weight `2.0`을 하나의 business-aware failure policy로
freeze했다. profile file SHA는
`53f1f558baaaf367430a995a95cc149f69006d624502e37680e4c52793df3dd6`이고,
immutable contract는
`results/tempo_go_c5_source_snapshot_v54_cache_failure_circuit_immutable/native_run_contract.json`
(SHA `28086d47ac5413cfe21ec664f08d3640246509a8f4ccf9f02fb86fef89401a2d`,
fingerprint `d3e75bfb759cba0c75bdcbc1bb0d748322086435bd9c1bfbe1142d41409e7cac`)다.

v54는 v53처럼 EngineCore가 즉시 죽지는 않았고 raw 276-row를 보존했다.
그러나 `all_streams_valid=false`, raw completed `199`, global reject `77`,
global queue timeout `77`, HTTP 503 `51`이었다. queue-lease terminal
영수증의 후보별 원인은 `telemetry_missing_or_stale=272`, `deadline=6`,
`path_skip=2`였다. 즉 원인 보존이 실제로 작동했으며, global queue가 오래
기다린 뒤 250 ms freshness를 넘긴 telemetry로 모든 pair×route를 보수적으로
닫고 있었다. service-valid subset은 completed `148`, request goodput
`4.379/s`, output-token goodput `514.6/s`, E2E p50 `8.228 s`, TTFT p50
`3.904 s`였지만 partial subset이므로 v51 fixed와 직접 비교하지 않는다.

### 37.2 v55: queue-boundary telemetry refresh

v54의 원인을 닫기 위해 `GlobalAdmissionCoordinator._timeout()`에서 waiter를
제거하기 전에 한 번의 bounded allocation-wide telemetry refresh와 dispatch를
수행하도록 했다. refresh가 성공하면 최신 endpoint scheduler/LMCache/NCCL
state로 queue lease를 다시 평가하고, refresh 자체가 실패하면
`global_telemetry_refresh_timeout` 등 별도의 terminal receipt를 남긴다.
v55 snapshot contract는
`results/tempo_go_c5_source_snapshot_v55_queue_boundary_refresh_immutable/native_run_contract.json`
(SHA `c94502c4b953d3f1249481ce27033e6870fdd8d46ac0772827f001fb00e9bbfb`,
fingerprint `c22ae4455c02fd2d1cccde8fc911a596ea20398701b08f2694e951bcf414245a`)다.

이 변화는 stale bottleneck을 실제로 줄였다. `telemetry_missing_or_stale`
후보 원인은 `272 -> 8`, global queue timeout은 `70 -> 18`로 줄었고 raw
completed는 `199 -> 248`, global reject는 `77 -> 28`로 늘었다. 그러나
최신 telemetry로 queue lease를 더 많이 허용하면서 endpoint physical queue
drain을 따라가지 못했고, queue-lease commit은 `113 -> 165`,
`endpoint_bounded_queue_lease_timeout`은 `51 -> 111`로 폭증했다. service
valid subset은 completed `137`, request goodput `3.388/s`, output-token
goodput `400.1/s`로 오히려 낮아졌다. 이는 **telemetry refresh만으로는
global credit와 endpoint credit이 통합되지 않는다**는 직접적인 native
evidence다. stale reject를 줄인 것을 utility 개선으로 보고하지 않는다.

### 37.3 v56: endpoint queue-credit cooldown과 반복되는 LMCache failure

v55의 endpoint timeout receipt를 다음 lease 결정에 연결하기 위해, global
orchestrator가 `endpoint_bounded_queue_lease_timeout`을 받은 pair의 **queue
lease만** cooldown하도록 했다. 이후 더 높은 telemetry sequence에서
`scheduler_waiting_requests=0`이 관측될 때만 cooldown을 해제한다. 일반 local /
remote admission과 route health quarantine은 이 cooldown으로 막지 않는다.
CPU coordinator/orchestrator 회귀는 `10 passed`였고, immutable contract는
`results/tempo_go_c5_source_snapshot_v56_queue_credit_cooldown_immutable/native_run_contract.json`
(SHA `1954f6db0015898b2d0e875b4e929624656bd6fd5c6aa4da659564494f73933f`,
fingerprint `53fa69c50e81b2ab170e332bfaf4e2fc29d76ebbabd1615fcb80b59b46be3e70`)다.

하지만 v56 measured phase에서는 node-1의 네 TP worker가 다시 같은
LMCache assertion을 냈다.

`Key CacheEngineKey(... world_size=4, worker_id=..., chunk_hash=...) not found in local data`

EngineCore가 종료되고 native step은 `rc=137`이 됐다. warmup와 log는
보존됐지만 measured `raw.json`의 full terminal validity가 없으므로 cooldown의
성능 효과는 계산하지 않는다. v53과 v56에서 같은 종류의 cache-key ownership
failure가 반복됐다는 점은 중요하다. 이는 “LMCache가 항상 실패한다”는
일반화가 아니라, 이 shared-contention workload에서 cache state inconsistency가
실제로 global inference service를 죽일 수 있다는 execution evidence다.

### 37.4 현재의 연구 결론과 다음 gate

v54--v56은 TEMPO의 가치를 절하하는 component subtraction 결과가 아니다.
오히려 다음 global loop가 필요한 이유를 한 run sequence 안에서 보여준다.

`fresh telemetry at business timeout → global admission/lease → endpoint physical queue → first-response/EOF credit → LMCache/NCCL failure feedback → pair/fleet protection`

현재 결론은 다음과 같다.

1. **queue-boundary refresh는 필요하다.** 없으면 queue wait가 telemetry
   freshness보다 긴 현실적인 contention에서 모든 후보가 stale로 닫힌다.
2. **refresh만으로는 부족하다.** stale rejection을 줄이면 endpoint queue
   lease가 늘고, endpoint drain보다 앞서면 503 timeout cascade가 커진다.
3. **endpoint lease cooldown과 cache-failure circuit이 필요하다.** 다만
   v56처럼 EngineCore가 먼저 죽으면 telemetry circuit이 관측할 기회 자체가
   사라질 수 있으므로, LMCache/cache-state health를 endpoint feedback과
   동일한 allocation-scoped failure plane에 넣어야 한다.
4. **아직 performance claim은 금지다.** v54/v55는 fixed/predictor와
   independent full-valid matched comparison이 아니고, v56은 execution
   failure다. 다음 primary run은 이 세 control을 하나의 frozen profile로
   묶고, same offered population의 fixed local/remote/predictor/queue-GPU/
   network-only/app-global-only/full TEMPO를 모두 terminal-valid로 닫아야
   한다.

따라서 TEMPO의 현재 상태는 **Perlmutter-scale cross-layer global mechanism은
실재하고 계속 확장 중이지만, LMCache/NCCL/fabric contention 아래에서
end-to-end utility closure는 아직 미완료**다. 다음 연구 작업은 또 다른
component를 덧붙이는 것이 아니라, queue credit·cache failure·fabric
telemetry를 하나의 business-aware global debt/circuit ledger로 freeze한 뒤
full-valid native primary를 닫는 것이다.

## 38. v57 global-fabric/cache-circuit 통합 gate

v54--v56에서 확인된 문제는 pair-local 제어만 더하는 것으로 닫히지 않는다.
두 P/D pair가 같은 Perlmutter allocation의 NCCL/Slingshot/LMCache remote
fabric을 공유하므로, `remote requests`, `remote KV bytes`,
`remote semantic operations`를 하나의 compatible communicator/epoch group에서
합산해야 한다. 최신 통합 profile은 v54의 queue-boundary refresh, endpoint
queue-credit cooldown, telemetry/cache-failure circuit을 유지하면서 v31의
`global_budget_v3`를 다시 결합했다.

### 38.1 frozen artifact

| item | artifact / digest |
|---|---|
| v57 profile | `results/tempo_go_c5_cross_layer_short_slice_v4_profiles_v57_global_fabric_cache_circuit/real_tempo_go_profile_short_slice_v57_global_fabric_cache_circuit.json`; file SHA `9b68809be83735107608d56226e27cf5cbd218bc06c36cdbec0f2c657a12be70`; profile fingerprint `f3e7428e5a9c90ba966a9688cfe248050b055d1bf7c50a24ff4e7a608315d1a7` |
| v57b source-bound contract | `results/tempo_go_c5_source_snapshot_v57b_global_fabric_cache_circuit_immutable/native_run_contract.json`; contract SHA `854c3a8c1954959afda8ba20cd8e916cde6f8b627ecf6641dab361ede636e27c`; fingerprint `8c0f70863f826186819303aaf0952ca712ab624aa5b2913b339a91e231e93147` |
| source snapshot | tree SHA `948d37914e8d02a0dce73b4e1fe1d6562fed532bf0f67ce049608c08af596a29`; 1,800 Python files; v57b contract full verify passed |

profile fingerprint와 file SHA는 위 artifact에 고정했으며, immutable
contract가 참조하는 profile binding을 primary의 source of truth로 삼는다.

v57 controller의 핵심 값은 shared remote request capacity `32`, remote KV
capacity `1,878,130,688` bytes, semantic-op capacity `8`, shared floor
`0.25`, bounded stagger `2,000 us`다. `deny_until_probe` pair circuit,
survivor reserve `0.25`, queue-boundary refresh와 endpoint queue-credit
cooldown도 동시에 유지된다. 따라서 이것은 local/remote/predictor 중 하나를
고르는 predictor 후보가 아니라, business deadline·endpoint scheduler
credit·LMCache transfer·NCCL/Cassini signal을 같은 global held/debt ledger에
넣는 통합 policy candidate다.

### 38.2 cross-layer control-plane replay

replay에 실제 fabric 성능을 가장했다고 쓰지 않기 위해, 별도의 명시적인
offline fixture만 추가했다. 두 pair에 동일한 communicator/epoch/topology
identity를 주고 NCCL p99 `25 ms`, LMCache transfer p99 `80 ms`, remote KV
in-flight `768 MiB`, Cassini RX pause `0.25`를 공급했다. v57 TEMPO arm의
snapshot은 다음을 보였다.

```text
shared remote requests: 32 -> 18
shared remote semantic ops: 8 -> 5
shared remote KV bytes: 1,878,130,688 -> 1,032,971,879
dispatch stagger: 2,000 us
shared pressure 아래 spare-pair activation: suppressed
```

동일 276-row trace의 five-arm replay는 `all_arms_have_same_request_count`,
`all_arms_have_same_trace_sha`, phase/physical-switch policy-input exclusion,
terminal/leak-free, failure receipt, telemetry-failure receipt와 frozen
contract gate를 모두 통과했다. v57b replay artifact는
`results/tempo_go_c5_replay_v57b_global_fabric_cache_cross_layer_fixture.json`
(SHA `825fc56745f8c4464fb93285dcb5bc3c9aedbb9c034af8ba4343dff42fbf458b`)이다.
이는 **global budget이 실제 decision path에 연결된 control-plane evidence**이지
GPU throughput, LMCache throughput, NCCL goodput 또는 end-to-end performance
claim이 아니다.

### 38.3 native gate와 해석

v57b native에서만 다음을 허용한다.

1. 같은 immutable contract와 같은 276-row offered population으로
   `local`, `remote`, `predictor`, `queue_gpu`, `network_request_only`,
   `app_global_only`, `tempo`를 모두 닫는다.
2. 같은 4-node allocation co-job이 NCCL/Cassini/LMCache observer window를
   제공하고, pair identity가 compatible할 때만 shared budget을 집행한다.
3. LMCache `CacheEngineKey ... not found in local data`가 다시 나오면 이를
   latency sample로 변환하지 않고, upstream stream failure → pair/fleet
   circuit → survivor fairness/debt release의 terminal receipt로 남긴다.
4. 모든 arm이 terminal-valid가 되기 전에는 v57을 v54/v55의 partial subset과
   수치 비교하지 않는다. full-valid matched primary가 닫힐 때에만 strongest
   fixed/predictor 대비 성능 gate를 평가한다.

현재 allocation `57433031`에서는 v53/v56와 같은 LMCache EngineCore fatal을
반복하지 않기 위해 v57 native를 재시도하지 않았다. 따라서 v57의 현재 상태는
**global-fabric/cache-failure control은 CPU replay와 immutable contract까지
연결됐고, native end-to-end utility는 아직 미측정**이다. 다음 4-node/4-hour
interactive allocation의 첫 primary는 v57b snapshot을 사용한다.

## 39. v58 explicit route-failure circuit 보정과 최종 native 후보

v57 replay를 검토한 결과, cumulative telemetry failure는 pair circuit으로
들어갔지만 explicit upstream stream failure는 profile의
`route_failure_quarantine_mode=disabled` 때문에 request release만 수행했다.
LMCache EngineCore fatal이 실제 stream/transport failure로 frontend에
도달하는 경우까지 같은 global failure plane에서 보호하려면 두 경로를
동시에 켜야 한다. 따라서 v58은 v57의 shared-fabric budget, queue-boundary
refresh, endpoint queue-credit cooldown, telemetry pair circuit을 그대로
유지하면서 explicit route failure도 `deny_until_probe`로 freeze한 최종
native 후보다. v57/v57b artifact는 변경하지 않고 history로 보존한다.

v58 profile file SHA는
`ced84a33f38923868bebb54d8704eda733b003d0ff46abe714c7fcc360ebd466`, profile
fingerprint는
`d0aac6cde622ec588d1d647abb6ea086f440c573b64afc122157b5b17300b40f`다.
현재 source-bound immutable contract는
`results/tempo_go_c5_source_snapshot_v58_global_failure_fabric_circuit_immutable/native_run_contract.json`
(SHA `ea9291e47e4af3c35edfa3d05e97627b633a0f69ed0cfb49f32bac13028d9dd3`,
fingerprint `a02ef954f51d26393c29acefa96ab1ef3a373b0ceab742c256c791e3839def14`)
이며 source tree SHA는 v57b와 같은
`948d37914e8d02a0dce73b4e1fe1d6562fed532bf0f67ce049608c08af596a29`다.

v58 cross-layer/failure replay는
`results/tempo_go_c5_replay_v58_global_failure_fabric_cross_layer_fixture.json`
(SHA `b4c8dec5aa8f4b1c36ea321c4896fcd6b3dbbed1352d085ce74be862959dd92e`)으로
닫았다. 동일 276-row trace에서 다음 gate가 모두 true였다.

- all arms same request count/trace SHA
- phase name·physical switch label을 policy input으로 사용하지 않음
- all arms terminal/leak-free
- frozen v58 contract valid
- explicit route-failure receipt schema valid
- cumulative LMCache failure telemetry receipt valid
- shared communicator pressure 아래 global budget/stagger receipt valid

replay의 TEMPO arm은 explicit remote route failure를
`global_route_failure_quarantine` receipt로 남겼고, 이후 cumulative
`observed_lmcache_engine_failure`가 pair scope에서 local/remote 두 route를
함께 quarantine했다. terminal state는 276건으로 정리됐으며, 이 결과는
**failure containment/control-plane gate**다. replay에는 실제 GPU/NCCL/LMCache
latency가 없으므로 성능 또는 stability superiority 주장은 금지한다.

이제 native 후보는 v57b가 아니라 v58이다. next allocation에서 v58의
seven-arm primary가 full-valid로 닫히면 그때에만 fixed/predictor 대비
E2E/goodput/fairness gate를 계산한다. 다시 EngineCore cache-key assertion이
발생하면 실패한 request와 pair를 정확한 receipt로 기록하고, 다른 arm의
partial rows와 합쳐 성능 숫자를 만들지 않는다.

## 40. v59 LMCache cache-key pair-failure binding 보정

v58 candidate를 native로 재사용하기 전에, v53/v56에서 반복된
`CacheEngineKey ... not found in local data` 경계가 현재 frontend에서 어떻게
global failure plane으로 들어가는지 점검했다. 기존 분류기는 backend HTTP 500을
일반 route failure로만 분류했기 때문에, decoder/cache ownership을 공유하는
pair에서 한 semantic route만 quarantine할 수 있었다. 이것은 cache fallback이나
same-ID retry로 숨길 문제가 아니라 pair-scoped failure receipt로 닫아야 하는
cross-layer lifecycle 문제다.

`eval/sota_4node/tempo_pd_elastic_frontend.py`에 다음 보정을 넣었다.

- bounded upstream body에서 `CacheEngineKey`/`not found`/local-data marker를
  명시적으로 식별한다.
- failure kind를 `lmcache_cache_key_ownership_failure`로 보존한다.
- 해당 failure scope를 `pair`로 바꿔 local/remote 두 route를 함께
  `deny_until_probe` quarantine한다.
- unrelated HTTP 500은 기존처럼 route-scoped로 유지한다.
- request를 같은 ID로 재시도하거나 silent local fallback하지 않는다.

집중 회귀는 `87 passed`, frontend `py_compile`과 `git diff --check`를 통과했다.
이 source 변경으로 v58 contract는 더 이상 native source-of-truth가 아니므로
v58 artifact를 덮지 않고 v59로 재동결했다.

| item | v59 artifact |
|---|---|
| base contract | `results/tempo_go_c5_cross_layer_contract_v59_global_failure_fabric_circuit/native_run_contract.json` |
| base contract SHA / fingerprint | `e42a7609208db32771a7236d0493945ab9d9c870d90b5bcf606ee2f612b2af4c` / `0b6145b538611b50cc8f7643d676e1bdb44cac740b1d310e19e7532871421699` |
| immutable snapshot contract | `results/tempo_go_c5_source_snapshot_v59_global_failure_fabric_circuit_immutable/native_run_contract.json` |
| snapshot contract SHA / fingerprint | `83d5e3236345de8d09b372cf8c7ba8c697d9d1a84adc6ab5076279f52f487b77` / `43cde254188997d5e8cd50e1e2e11d63ffc57300ce91d166cc43a4f8c2599e0b` |
| snapshot tree SHA / Python count | `d4ca292b6210523a2d600445ea59ca105365950b1eba6767196bf01f949d50be` / `1,800` |
| contract status | full verify passed; `performance_claim_allowed=false` |

v59 contract-bound cross-layer replay는
`results/tempo_go_c5_replay_v59_global_failure_fabric_cross_layer_fixture.json`
에 저장했다. 동일 276-row trace, cross-layer pressure fixture, explicit remote
route failure와 cumulative LMCache failure injection을 사용했고, request count/
trace SHA/phase·physical-label exclusion/terminal-leak-free/failure receipt/
telemetry receipt/frozen contract gate가 모두 true다. replay는 control-plane
failure containment 증거일 뿐 native throughput·latency·stability superiority
증거가 아니다. TEMPO arm은 injected failure 1건과 subsequent pair quarantine을
정확히 기록했으며, 다른 arm의 partial rows와 성능 비교하지 않는다.

현재 최종 native 후보는 **v59 immutable snapshot**이다. 다음 승인된
4-node/4-hour allocation에서 capability → co-job readiness → C5 launch →
동일 offered-population seven-arm primary 순서로만 실행한다. 현재 allocation의
반복된 EngineCore cache-key fatal 때문에 그 allocation 안에서 v59 native를
blind retry하지 않는다. full-valid primary가 닫히기 전까지 TEMPO speedup,
goodput win 또는 robustness superiority를 주장하지 않는다.

## 41. v60 global cache-residency invalidation closure

v59의 pair-scoped LMCache failure 분류만으로는 다음 request의 frontend
affinity ledger가 stale P/D cache residency를 계속 신뢰할 수 있었다. 실제
v56 log에서는 `PDBackendAsync.get_blocking()`이
`CacheEngineKey ... not found in local data`를 냈고, 기존 completed warm-probe
affinity는 그 receiver object의 현재 생존을 보장하지 않았다.

따라서 `PairLoadLedger`의 global failure lifecycle을 다음처럼 보정했다.

- pair-scoped global failure가 terminal receipt로 들어오면 실패 pair의
  prefill affinity owner와 decode affinity owner를 제거한다.
- 다른 pair의 replica evidence는 유지한다.
- owner가 하나도 남지 않으면 해당 prompt는 `UNKNOWN`/`confirmed_miss`
  경계로 내려가며, 새 completed probe 없이 remote warm affinity를 재사용하지
  않는다.
- invalidation 여부와 원인을 request ledger에 남겨 cache/business/failure
  provenance가 끊기지 않게 한다.
- route failure 이후 same-ID retry나 silent local fallback은 여전히 금지한다.

이 보정은 global orchestrator가 NCCL/LMCache/vLLM endpoint failure를 단순
HTTP error로 세는 것이 아니라, 이후 admission/placement의 cache state와
survivor service까지 바꾸는 전체 loop를 닫는다. 집중 회귀는 `88 passed`,
`py_compile`, `git diff --check`를 통과했다. v59 contract를 덮지 않고 v60으로
source snapshot을 새로 만들었다.

| item | v60 artifact |
|---|---|
| base contract | `results/tempo_go_c5_cross_layer_contract_v60_global_cache_failure_closure/native_run_contract.json` |
| base contract SHA / fingerprint | `45c4d8588600c83e39c06206e311095d614e6583ae071d6da355e6df0ff7ca63` / `99e682c944872ea54986e6a7714378bc4a5bb66819a212966a004642b82654ba` |
| immutable snapshot contract | `results/tempo_go_c5_source_snapshot_v60_global_cache_failure_closure_immutable/native_run_contract.json` |
| snapshot contract SHA / fingerprint | `9911a6213a80603e7f7f79f3fad9ccafb779d642395b9f0895b0838bff012df5` / `90710e270f8f5cfb6ffc31bcd67d34b155543508370ef7c79810abc11428a689` |
| snapshot tree SHA / Python count | `01508ddf6395d790f9643608e9773b44b9da26d55ad99239c9f50b0812c5cff4` / `1,800` |
| contract status | full verify passed; `performance_claim_allowed=false` |

v60 contract-bound replay는
`results/tempo_go_c5_replay_v60_global_cache_failure_cross_layer_fixture.json`
에 저장했다. 동일 276-row trace와 cross-layer pressure/failure injection을
사용했고, matched request/trace, phase·physical-label exclusion,
terminal/leak-free, failure/telemetry receipt, frozen contract gate가 모두
true다. 이는 failure containment/control-plane evidence이며 native
throughput·latency·stability superiority 결과가 아니다.

다음 native primary는 이제 **v60 immutable snapshot**을 사용한다. 승인된
4-node/4-hour allocation에서 co-job readiness와 actual LMCache/NCCL observer를
먼저 닫고, 같은 offered population의 seven-arm full-valid result가 생성될
때까지 성능 숫자를 계산하지 않는다.

## 42. v61 global duplicate cache-chunk serialization

v60에서 pair-scoped failure가 stale affinity를 제거했지만, 같은 shared-prefix
request가 동시에 동일 pair의 remote LMCache transfer를 시작하는 race 자체는
남아 있었다. 실제 LMCache `PDBackendAsync` receiver는 push-based ownership을
전제로 하므로, duplicate transfer가 first response 전에 겹치면 두 번째
request가 `CacheEngineKey ... not found in local data`를 내며 EngineCore를
죽일 수 있다. 이것은 endpoint 내부 retry로 감출 문제가 아니라, pair-local
cache ownership을 global admission이 알고 serialize해야 하는 contention
문제다.

v61은 native tokenizer가 계산한 완전한 256-token rolling-prefix identity를
`GlobalRequest.cache_group_key`로 전달하고, global orchestrator에
`(pair_index, cache_group_key) -> request_id` hold를 둔다.

- remote transfer가 route committed 된 뒤 first response를 받을 때까지 같은
  pair/cache group의 중복 remote candidate를
  `cache_chunk_transfer_serialization`으로 거절/queue한다.
- local route 또는 다른 pair는 계속 후보로 남겨 cache serialization이
  전체 시스템을 불필요하게 멈추지 않게 한다.
- first response, EOF, ordinary failure, service-lane failure, explicit
  LMCache route failure에서 hold를 exactly once 해제한다.
- decision receipt와 orchestrator snapshot에 cache group/hold owner를 남겨
  business admission, pair load, LMCache failure provenance를 연결한다.
- LMCache third-party data plane이나 root/UDI/container 설정은 수정하지
  않았다. 이는 Perlmutter shared resource를 TEMPO global loop에서 제어하는
  안전한 admission-plane 변경이다.

집중 global 회귀는 `89 passed`, frontend 회귀는 `23 passed`와 11 subtests였고,
`py_compile` 및 `git diff --check`를 통과했다. v60 contract를 덮지 않고 v61
source snapshot을 새로 만들었다.

| item | v61 artifact |
|---|---|
| base contract | `results/tempo_go_c5_cross_layer_contract_v61_cache_chunk_serialization/native_run_contract.json` |
| base contract SHA / fingerprint | `27f7265d6f80dfcb9359afb71300de50d1c7f6185abd95f590b0a6ead5272d41` / `e989cf55ed677288e5d7454319269a5db297763cfd2186103f1370814d71ba69` |
| immutable snapshot contract | `results/tempo_go_c5_source_snapshot_v61_cache_chunk_serialization_immutable/native_run_contract.json` |
| snapshot contract SHA / fingerprint | `78240fcb608e384b2705552e42c0ab61415bebf8b03708bd8ca047dbbd53c0e0` / `2289902fe0d9700abe5027ab469c6f9692dbc8355f2faea8ce16648cc9599997` |
| snapshot tree SHA / Python count | `346f82d7cf80e6133cdd82a6e91968a9aff387473ce874787061a6b7237fb356` / `1,800` |
| contract status | full verify passed; `performance_claim_allowed=false` |

v61 contract-bound replay는
`results/tempo_go_c5_replay_v61_cache_chunk_serialization_cross_layer_fixture.json`
(SHA `3b9349213cfce94908b9980c649d2397d5e4add962619322925bb7910c06ee32`)에
저장했다. 동일 276-row trace에서 request/trace binding, phase·physical-label
exclusion, terminal/leak-free, explicit failure receipt, telemetry failure
injection, frozen contract gate가 모두 true다. cross-layer pressure fixture는
offline control-plane 전용이며 `performance_claim_allowed=false`다. replay의
arm별 rejection/failed count를 native throughput 또는 latency 결과로 해석하지
않는다.

다음 native primary는 이제 **v61 immutable snapshot**을 사용한다. 승인된
4-node/4-hour interactive allocation에서 capability → co-job readiness →
actual NCCL/Cassini/LMCache observer 확인 → 동일 offered-population seven-arm
full-valid run 순서로만 진행한다. 그 native 결과가 fixed/predictor보다
유의미하게 빠른지, 그리고 contention/failure 상황에서 goodput·SLO·fairness가
개선되는지를 처음으로 판단한다.

## 43. v61 native execution evidence: shared co-job lifecycle failure

v61 immutable snapshot으로 승인된 clean 4-node/16-GPU interactive allocation
`57439545`에서 same-allocation NCCL/LMCache co-job과 C5를 실제로 겹쳐
실행했다. native guard, contract verify, Slingshot/NCCL `AWS Libfabric`,
official `LMCacheConnectorV1:UCX`, 4-node vLLM startup는 모두 통과했다.

local arm은 `276/276` request complete, `0` global reject, `0` failure,
`all_streams_valid=true`, `router_decisions_exact=true`,
`terminal_contract_valid=true`로 닫혔다. 다만 이것은 local 단일 arm receipt라서
cross-arm performance 비교가 아니다.

같은 allocation의 co-job은 observer가 `producer_state=active`와 sequence
`2086`까지 실제로 publish한 뒤 약 6분 53초에 Slurm `SIGKILL`을 받아
`result.json` 없이 종료했다. node logs에는 NIXL initialization/memory
registration 뒤 외부 step kill이 남았고, co-job이 C5 종료까지 coverage하지
못했으므로 same-contention primary 조건이 깨졌다. local arm 종료 후 runner의
remote arm은 이 오염된 상태에서 계속 측정하지 않고 명시적으로 중단했다.

| item | native v61 execution evidence |
|---|---|
| allocation | `57439545`, nodes `nid[001233,001236-001237,001240]`, 4-node/16-GPU |
| immutable contract | `results/tempo_go_c5_source_snapshot_v61_cache_chunk_serialization_immutable/native_run_contract.json` (SHA `78240fcb608e384b2705552e42c0ab61415bebf8b03708bd8ca047dbbd53c0e0`) |
| local result | `results/tempo_go_c5_cross_layer_native_v61_cache_chunk_serialization_57439545/local/result.json` (SHA `133e8dd6ed7e3f4b3c2e7f952a358cb8820bcab209ca6204335eb1a74a8968c8`) |
| local raw | `results/tempo_go_c5_cross_layer_native_v61_cache_chunk_serialization_57439545/local/tempo_go_c5_discovery/raw.json` (SHA `55b207542ce03249f9aecf9dfa1be32cf4aa4cf275cae486947f4557c9c2564b`) |
| co-job observer at kill | `results/tempo_go_c5_cross_layer_cojob_v61_57439545/nccl_observer.json` (SHA `f93306b15896d549c6c4c343eb814502e7d824179f387c0c50340c338c8d4e7b`), active/sequence 2086, no terminal result |
| co-job stderr | `results/tempo_go_c5_cross_layer_cojob_v61_57439545/cojob.stderr.log` (SHA `03a5e97c70c881fe3c3eb636caf7cc37088d7891debd2235d5bd1b2a5fd9c14a`) |
| interrupted remote receipt | `results/tempo_go_c5_cross_layer_native_v61_cache_chunk_serialization_57439545/remote/failure.json` (SHA `f48c27387552adc6d24c9f2bc16c1c75336a80c9bfec971ea52edb408ddb9208`) |

이 결과는 **TEMPO가 필요 없다는 증거가 아니다.** 오히려 shared GPU/fabric
co-job의 lifecycle·health·termination을 global telemetry/failure plane이
받지 못하면, local/remote/predictor를 같은 contention window에서 비교할 수
없다는 것을 보여준다. 다음 native gate는 co-job을 단순 background process로
간주하지 않고, observer heartbeat/step state/termination receipt를 TEMPO
control loop의 external business/resource signal로 bind하는 것이다. 그 뒤에야
v61 cache-chunk serialization이 실제 duplicate LMCache failure를 줄이는지와
fixed/predictor 대비 E2E/goodput/fairness를 판정할 수 있다.

## 44. v62 campaign-level global lifecycle fail-closed

v61 native attempt는 co-job이 observer를 실제로 publish하다가 먼저 종료했는데도,
foreground C5 runner가 그 사실을 모른 채 다음 arm으로 진행할 수 있는 lifecycle
hole을 드러냈다. 이것은 LMCache/NCCL 컴포넌트 하나의 문제가 아니라, shared
Perlmutter resource/business signal을 묶은 global orchestrator campaign의
정합성 문제다. v62에서는 co-job launcher와 seven-arm C5 runner를 같은 parent
wrapper가 소유하도록 바꿨다.

- co-job step이 readiness 전이나 C5 측정 중 종료되면 parent가 즉시
  `cojob_failure.json`을 기록하고 C5를 중단한다.
- receipt에는 co-job step/process 상태, observer의 마지막 producer state와
  sequence, C5 result root, Slurm allocation, `native_only=true`를 남긴다.
- co-job component 자체도 non-zero step exit을
  `tempo-go-cross-layer-cojob-failure-v1` receipt로 기록한다. 따라서 Slurm
  step failure가 빈 디렉터리나 뒤늦은 timeout으로 흡수되지 않는다.
- C5와 co-job이 모두 끝나고 나서만 campaign-level result binding을 수행한다.
  co-job이 없는 arm 일부의 latency/goodput을 matched cross-arm 성능으로
  계산하지 않는다.
- 이 변경은 root/UDI/container ownership, NCCL transport, LMCache data plane을
  건드리지 않는다. 관찰된 공동 실행의 lifecycle을 global control/failure
  plane에 연결하는 admission·measurement integrity 변경이다.

집중 회귀는 observer/LMCache contention 관련 `12 passed`였고, 관련 Python
`py_compile`, shell `bash -n`, `git diff --check`를 통과했다. v61 contract를
덮지 않고 새 v62 base contract와 immutable source snapshot을 만들었다.

| item | v62 artifact |
|---|---|
| base contract | `results/tempo_go_c5_cross_layer_contract_v62_cojob_lifecycle_fail_closed/native_run_contract.json` |
| base contract SHA / fingerprint | `135c09c6e6ad45f45ea9cb0c6412481ee044fc61799adec2b6f5e17f074aedc9` / `719ca6b23108cd92a38c774e392aaff62d2f40b8bc48d91918892a21d1ed3bcf782` |
| immutable snapshot contract | `results/tempo_go_c5_source_snapshot_v62_cojob_lifecycle_fail_closed_immutable/native_run_contract.json` |
| snapshot contract SHA / fingerprint | `e9243b398f3596b36295c2f950a85da3600819297dcd91918892f03c26f9f23c` / `40473f1549b14f03c8e87e0fd3001a6cb036ec942ff4fd3dc7bba6819a87dff0` |
| snapshot tree SHA / Python count | `6a6a3ee7882c9ae485fcf04445e715ba469c56e321a39d6b462d074a99ebcc8c` / `1,799` |
| current runner SHA | `2bd16562151c552ad718011e04457af72fae11d99de7bac3e00bc8096dfad8b4` |
| current co-job component SHA | `94fec62e874856a0a7eadc1019f376da8ea4878f40627c32fd62c39b5fc31211` |
| contract status | full verify passed; `performance_claim_allowed=false` |

v62 contract-bound replay는
`results/tempo_go_c5_replay_v62_cojob_lifecycle_fail_closed_cross_layer_fixture.json`
(SHA `95f18b1e63f448f95e87d32ee68c7ba1e258f33c7dc84baff6e8901bdc29e7dc`)에
저장했다. 동일 276-row trace와 v61의 cross-layer pressure 및 telemetry
failure injection을 사용했고, request/trace binding, phase·physical-label
exclusion, terminal/leak-free, failure receipt, frozen-contract gate가 모두
true다. 이것은 lifecycle 정책의 offline control-plane 회귀이며 GPU,
LMCache, NCCL, Slingshot, latency, goodput 또는 production 성능 증거가 아니다.

다음 native primary는 source-owned immutable runner와 v63 계열 control-plane
snapshot으로 승인된 새 4-node/4-hour interactive allocation에서 실행한다.
순서는 (1) Perlmutter native-step
preflight, (2) co-job NCCL/LMCache observer active, (3) parent wrapper가
co-job 생존을 감시하는 동안 seven-arm C5 전체 수행, (4) co-job terminal
receipt와 모든 arm의 동일 offered population 검증이다. co-job lifecycle이
다시 끊기면 자동으로 campaign을 폐기하고, full-valid matched result가
생길 때까지 fixed/predictor/TEMPO 성능 우열을 주장하지 않는다. lifecycle이
유지된 native 결과에서만 TEMPO의 전체 cross-layer orchestration이
contention하의 goodput·SLO·fairness를 개선하는지 판정한다.

## 45. v62 native readiness stop and v63 correction

v62 immutable contract로 새 clean allocation `57440162`에서 native launcher를
실행했다. Perlmutter 4-node/16-GPU native-step preflight는 통과했고, co-job
Slurm step `57440162.1`도 2-node/8-GPU로 기동되어 NIXL UCX initialization과
LMCache process-group barrier까지 도달했다. 그러나 observer가 active snapshot을
쓰기 전에 v62 parent의 60초 readiness ceiling에 걸렸다. parent는 C5를 시작하지
않고 co-job을 정리했으며, allocation은 `FAILED/exit 1`, co-job step은
`CANCELLED/exit 143`으로 끝났다. 따라서 이 attempt에는 local/remote/predictor
어떤 C5 measured arm도 없고 성능 숫자도 없다.

v62의 실패 receipt는 component가 먼저 만든
`tempo-go-cross-layer-cojob-failure-v1`이어서 parent campaign receipt를
가렸다. 이 관찰을 반영해 v63에서 readiness ceiling을 300초 bounded wait로
늘리고, parent의 `tempo-go-cross-layer-campaign-failure-v1` receipt를
`campaign_failure.json`으로 component의 `cojob_failure.json`과 분리했다.
readiness timeout도 이제 명시적 failure kind로 기록되며, C5 root가 아직
생기지 않은 시점에도 campaign provenance가 보존된다.

| item | v62 execution / v63 correction |
|---|---|
| v62 allocation | `57440162`, nodes `nid[001164-001165,001168-001169]`, 4-node/16-GPU |
| v62 preflight | `results/tempo_go_c5_cross_layer_cojob_v62_57440162/perlmutter-native-step-preflight/receipt.json` (passed) |
| v62 component failure | `results/tempo_go_c5_cross_layer_cojob_v62_57440162/cojob_failure.json` (SHA `92b9f2461eeeb9fefa0b91078f1a8f1d77149c7626a6c6de4c0f1b2be5645700`), no observer state/sequence, exit `143` |
| v62 C5 result | absent; no arm started; no performance claim |
| v63 base contract | `results/tempo_go_c5_cross_layer_contract_v63_cojob_readiness_receipt/native_run_contract.json` (SHA `675260c9d67e985fbde9cb35862fc0fad0619ebd2a61105b1dd1fea30228cc59`, fingerprint `16ec323daac3001ee5e70346e4b11d46c7baac88d5fc4cd4b1762e4455b35203`) |
| v63 immutable snapshot | `results/tempo_go_c5_source_snapshot_v63_cojob_readiness_receipt_immutable/native_run_contract.json` (SHA `9f26de5939df9b89091f24b333fdee786e424f0a2b5617548794a4c45064892c`, fingerprint `edac785d893ced928deeb2e269b16c707d3d3d941493670f5a0f810d11646d65`) |
| v63 tree SHA / Python count | `bb736fe3d6bfe4fdd9b6db18e4040e025b460bcd5d879f00ca45795743a8b9a0` / `1,799` |
| v63 replay | `results/tempo_go_c5_replay_v63_cojob_readiness_receipt_cross_layer_fixture.json` (SHA `3f19304ba63a0611e99c7e969aadcbc16c6f04bd302ed31ef16a4840cdfcf553`) |
| v63 replay status | all request/trace, exclusion, terminal/leak-free, failure receipt and frozen-contract gates true; `performance_claim_allowed=false` |

이 stop은 “contention이 없다”는 결과가 아니다. 실제 shared co-job은
Perlmutter native launch 뒤 초기화·동기화 lifecycle을 가지며, 그 lifecycle을
기다리는 global campaign boundary가 필요하다는 실행 증거다. 다음 source-stable
native attempt는 readiness 300초를 사용하되, timeout 또는 co-job termination이면
`campaign_failure.json`을 확인하고 즉시 해당 campaign을 폐기한다. observer가
active가 된 뒤에만 동일 offered population의 seven-arm C5를 시작한다.

## 46. v63 native source-integrity stop

v63 allocation `57440554`에서는 observer가 active/sequence `140`까지 올라가
co-job readiness gate 자체는 통과했다. 그러나 C5 측의 frozen contract verify가
runner source
`eval/sota_4node/run_tempo_go_cross_layer_with_cojob_in_allocation.sh`의 live
SHA가 contract-bound SHA와 다르다고 거부했다. 이후 생성된 local/remote/
predictor/queue/network/app ablation 디렉터리의 `rc=143` receipt는 parent
interrupt/cleanup에 의한 invalid attempt이며 valid arm 결과가 아니다. TEMPO
arm까지 진행하지 않았고 이 allocation은 명시적으로 중단·해제했다.

| item | v63 native source-integrity evidence |
|---|---|
| allocation | `57440554`, nodes `nid[001192-001193,001196-001197]`, 4-node/16-GPU |
| observer at stop | `results/tempo_go_c5_cross_layer_cojob_v63_57440554/nccl_observer.json`, active/sequence `140` (SHA `204f859dd9b6ebe3abd501f261f2aa618ee51314258f3da4c52d8a01dbdce563`) |
| immutable contract | v63 snapshot contract SHA `9f26de5939df9b89091f24b333fdee786e424f0a2b5617548794a4c45064892c`; bound runner SHA `4e66eaf2fee45c9a42e2d9ff38f1845282059be23c367e453d64615a0f2a71a0` |
| live runner observed during verify | SHA `b5f8073e9bb09ee551ca1eac83140f14c7d8b3229ae0a7f000cce94fbbeee852`; contract rejected it |
| native arm receipts | local/remote/predictor/queue/network/app `native_arm_process_failed`, `rc=143`; no full-valid arm |
| claim status | no performance claim; allocation released |

이것은 TEMPO의 orchestration hypothesis가 실패한 것이 아니라, 여러 lifecycle
실행이 공유하는 mutable launcher를 immutable contract와 함께 쓰면 안 된다는
measurement-integrity 결과다. 이후 live runner는 다시 다른 내용으로 바뀌어
현재 확인 SHA도 `4f30adbe84a31b73ec581c4b7f2fc675b2021561f6b3abc9949055dd56071b97`
가 되었고, 현재 파일에는 이 goal turn에서 작성하지 않은 pre-verify/mkdir 및
preflight command 변경이 포함되어 있다. 파일이 untracked인 상태에서 이를
임의로 덮어쓰거나 합치지 않는다. source ownership이 하나로 정리되고
runner·component·contract가 같은 immutable source boundary를 가리킨 뒤에만
다음 4-node native allocation을 허용한다.

## 47. v66 lifecycle evidence, v67 producer lifetime correction, v68 NIC-aware global loop

v66 immutable source에서 승인된 4-node/16-GPU allocation `57441308`으로
cross-layer seven-arm campaign을 시도했다. C5의 모든 raw arm 디렉터리는
생성됐지만, co-job이 고정된 10,000 block을 먼저 소진해 `producer_state=complete`
가 되었고 TEMPO arm이 그 뒤에 시작됐다. 따라서 observer는 C5 측정 구간을
덮지 못했고, endpoint fallback topology identity가 pair별로 달라져
hierarchical fan-in이 fail-closed 되었다. 이는 global hierarchy가 잘못된
데이터를 합치지 않았다는 correctness evidence이지 TEMPO 성능 negative가
아니다.

| item | v66 evidence |
|---|---|
| allocation | `57441308`, 4-node/16-GPU/4-hour interactive |
| campaign failure | `results/tempo_go_c5_cross_layer_cojob_v66_57441308/campaign_failure.json` (SHA `d7f84ae48506df5178b1658c029d5d5b73de211333858576ca84fb2b7668863f`), `cojob_ended_before_c5_end` |
| co-job result | SHA `ebf246d263e4b5dd214508a40c9ac9435fc8d8ab1742880aecd0ac3837309f3f`, producer sequence `10001` |
| posthoc analyzer | `results/tempo_go_c5_cross_layer_native_v66_parent_log_outside_c5_57441308/native_five_arm_analysis_posthoc.json` |
| claim gate | TEMPO endpoint/global observer coverage false; `performance_claim_allowed=false` |

v67에서는 이것을 component timeout 숫자로 덮지 않고, co-job과 C5를 하나의
global campaign lifecycle로 묶었다. co-job은 충분한 bounded block ceiling을
가지고 계속 active 상태로 유지되며, C5가 끝난 뒤 parent가 stop file을
발행한다. producer는 synchronized block을 정상 종료하고 C5 종료 이후의
`producer_state=complete` snapshot을 publish한 뒤에만 parent가 co-job result와
binding receipt를 검증한다. CPU/contract/replay 회귀는 `124 passed, 11
subtests passed`였고, v67 offline replay의 failure/telemetry receipt와
terminal/leak-free gate는 true였지만 성능 claim은 금지했다.

v68에서는 TEMPO가 raw `cassini_by_nic`를 receipt에만 보존하던 경계를
global decision에 연결했다. `CrossLayerTelemetry`가 4 NIC×traffic-class
pause vector에서 NIC별 최대값을 derive하고, 이를
`route_externality`, v2/v3 joint actuation, shared remote budget과 pair
activation에 동일한 atomic batch provenance로 전달한다. shared communicator
pressure만 있고 NIC identity가 없으면 기존 fail-closed scaling을 유지한다.
반대로 한 pair의 NIC만 hot하고 다른 prewarmed pair가 cool하다는 실제 관측
가능한 상태에서는, global controller가 hot pair를 무조건 막는 대신 cool
pair를 같은 decision에서 activate하고 route를 commit한다. 이 구조는 NCCL
collective/LMCache transfer latency와 Cassini per-NIC/TC pressure를 vLLM
decoder admission·pair scaling·tenant service decision에 연결하는 전체
orchestration 경로이며, 단일 Cassini component 개선으로 해석하지 않는다.

| item | v68 artifact |
|---|---|
| immutable source snapshot | `results/tempo_go_c5_source_snapshot_v68_nic_aware_global_immutable` |
| contract SHA / fingerprint | `4b086516bcdd7bf467dffa9cd702afeb991ec899b808b7fb50c66243691e257d` / `44ff374e1558cfbe1f0f29c8ba7f45991a8a051f1201ebb07a83992897312671` |
| source tree SHA / Python count | `fe0e98b4bc3321bdb776cb37245c086cd69188b7a36401ced2cf4083c38704a6` / `1,806` |
| focused global/Cassini regression | `107 passed` |
| contract-bound replay | `results/tempo_go_replay_v68_nic_aware_contract_fixture.json` (SHA `696225570eb8285c449baa929f4ed3fa366938df507924dfe4e9d9c63727c9e6`) |
| replay evidence | all request/trace, policy-input exclusion, terminal/leak-free, failure receipt and frozen-contract gates true; NIC-imbalance fixture activates cool pair 1; `performance_claim_allowed=false` |

v68 replay에서 TEMPO는 NIC-imbalance fixture에서 `223/276`을 complete하고
pair 0/1에 `109/114`건을 배치했다. 이전 동일 control-plane fixture의
`222/276` 및 pair 0/1 `116/106` 대비 pair distribution이 hot NIC에서 cool
NIC 쪽으로 이동했다. strongest queue/GPU-only의 `231/276`보다 낮으므로 이
replay는 성능 승리가 아니라 **NIC-aware global actuation이 실제 decision을
변경한다는 mechanism evidence**다. GPU, LMCache, Slingshot fabric,
end-to-end latency/goodput 또는 production utility 결론은 native matched
campaign 뒤에만 허용한다.

현재 native blocker는 source나 root 환경이 아니다. v68 새 4-node interactive
allocation 요청은 Perlmutter의 정확한 형식
`/usr/bin/salloc -A m1248_g -C gpu -q interactive -t 04:00:00 -N 4 --gpus-per-node=4`
로 실행했지만 `QOSMaxSubmitJobPerUserLimit`로 거절됐다. 기존 user allocation
`57443205`와 `57443240`은 다른 workflow일 가능성이 있어 attach/cancel하지
않았다. 새 allocation 또는 사용자가 지정한 기존 interactive shell이 확보되면
v68 immutable runner로 (1) native preflight, (2) active co-job, (3) seven-arm
C5 전체, (4) co-job/C5 동일-population binding, (5) analyzer gate 순서로
즉시 진행한다. 그 전까지 v68은 implementation/control-plane readiness이며
native performance result가 아니다.

## 48. whole-system orchestration 경계 재확인

이 연구의 평가 단위를 Cassini, NCCL, LMCache, vLLM scheduler, endpoint
feedback controller 중 하나로 축소하지 않는다. 현재 canonical native 경로는
다음 하나의 request lifetime으로 연결돼 있다.

`vLLM frontend request`
→ `tokenizer/cache-group/cache-residency evidence`
→ `GlobalAdmissionCoordinator`
→ `GlobalOrchestrator (pair × route × shared fabric budget × tenant/SLO)`
→ `X-Tempo-GO joint commit`
→ `pair router service-lane reservation`
→ `vLLM/LMCache upstream`
→ `first response/HTTP EOF release`

각 pair의 `/tempo/runtime_telemetry`는 vLLM scheduler gauge, endpoint resource
ownership/health/failure, Cassini endpoint delta, NCCL/LMCache observer envelope를
하나의 freshness-bound all-pair batch로 반환한다. 따라서 TEMPO의 hypothesis는
“각 component를 조금씩 개선하면 빨라진다”가 아니라, contention으로 병목이
GPU에서 endpoint queue, LMCache/UCX, NCCL/Slingshot, 특정 NIC, 또는 business
tenant queue로 이동할 때 그 이동을 관측하고 route·pair·remote budget·fairness를
같은 admission transaction에서 조절하면 전체 offered workload의 utility가
개선된다는 것이다.

이 경계가 실제 코드에 연결돼 있는지 확인하는 focused native-path 회귀는
`eval/sota_4node/test_tempo_pd_elastic_frontend.py`,
`eval/sota_4node/test_tempo_pd_elastic_router.py`,
`tempo/test_pd_global_orchestrator.py`,
`tempo/test_pd_global_coordinator.py`,
`tempo/test_pd_global_telemetry.py`를 함께 실행해 `116 passed, 22 subtests
passed`로 통과했다. 이는 performance result가 아니라 whole-system commit과
telemetry contract의 readiness evidence다.

따라서 다음 native campaign에서 고정 정책과 비교할 단위도 component latency가
아니다. 동일한 offered population과 동일한 concurrent co-job 아래에서
`request goodput`, `output-token goodput`, `TTFT/E2E p50/p95/p99`, tenant별
SLO violation, Jain/service-share fairness, global reject/queue-lease rate,
pair/NIC migration, NCCL·LMCache overhead를 함께 비교한다. TEMPO의 성능 승리는
이 joint end-to-end gate를 통과할 때만 주장한다.

## 49. queue lease의 whole-system 수정: admission capacity와 native waiting queue의 분리

v70 replay의 request-level rejection receipt를 다시 열어 본 결과, TEMPO의
53건 rejection은 NCCL·Cassini·LMCache failure가 아니라 모든 후보 pair에서
`endpoint_queue_lease_capacity_guard(active_sequences)`가 발생한 것이었다.
이는 `endpoint_queue_lease`라는 business policy의 의미와 구현이 충돌한
상태였다. decoder의 `active_sequences`/`endpoint_requests` service window가
가득 찬 순간에도 vLLM은 native waiting queue를 가질 수 있는데, global
orchestrator가 두 자원을 downstream queue의 여유와 동일시해 expired waiter를
ingress reject로 바꾸고 있었다. 이 때문에 TEMPO가 queue-GPU-only보다 덜
work-conserving해지는 구조적 결함이 있었다.

수정된 canonical semantics는 다음과 같다.

`GlobalAdmissionCoordinator queue timeout`
→ `GlobalOrchestrator queue-lease candidate evaluation`
→ `active_sequences/endpoint_requests` overage를 global ownership ledger와
   immutable decision receipt에 보존
→ `X-Tempo-GO queue lease`로 pair router에 전달
→ downstream vLLM/endpoint bounded waiting queue가 실제 수용 여부 결정
→ 수용 실패면 service-lane failure receipt로 정확히 global debt release

따라서 queue lease는 resource guard 우회가 아니다. tenant queue-lease opt-in,
deadline/SLO, telemetry freshness, path health, cache-group, survivor reserve,
shared remote budget, critical Cassini transport guard는 계속 fail-closed로
남는다. 단지 global reservation capacity와 downstream decoder waiting
capacity를 분리하고, 실제 endpoint handshake가 queue 수용 여부의 최종
authority가 된다. overage는 `binding_resources`로 남아 이후 route/pair/fabric
결정의 cost에 반영되며 first-response/EOF에서 exactly-once release된다.

이 수정의 focused regression은 `59 passed`였다. 같은 276-request
cross-layer replay에서 v71 TEMPO는 `276/276 complete, 0 reject`가 되었고,
queue lease 92건이 explicit debt receipt를 남겼다. 비교 arm은 always-local
`276/276`, queue-GPU-only `276/276`이었다. TEMPO의 replay p50 E2E는
`7404.00 ms`로 always-local `7574.31 ms`보다 낮았지만 queue-GPU-only
`7310.71 ms`보다 높고, p99는 `8646.22 ms`로 queue-GPU-only `8447.54 ms`보다
높았다. 이는 policy가 더 이상 contention에서 일을 버리지 않는다는
mechanism/readiness evidence일 뿐이며 replay는 downstream vLLM·NCCL·Slingshot
성능을 측정하지 않으므로 `performance_claim_allowed=false`를 유지한다.

이 queue-debt 수정본의 native source boundary는
`results/tempo_go_c5_source_snapshot_v71_queue_debt_immutable`로 고정했다.
contract SHA는
`cf48bcc68f5c4107fc8b76b8d2099e44b559f2128f47f3023e983d269429ce52`,
fingerprint는
`42e60b7892ecd47a0f983bd207cae2bc5af3df45630243cfe223d64503fb46cc`,
source tree SHA는
`3b59feb4b08e947d89a1423ed82af74eaf8dc16532220e38f3e63009fdd63073`이다.
이 contract에 묶은 replay artifact는
`results/tempo_go_replay_v72_queue_debt_contract_bound.json`이며
`frozen_run_contract_valid=true`, 동일 request/trace, terminal/leak-free,
failure/telemetry receipt gate가 모두 true였다.
현재 Perlmutter에는 기존 4-node interactive allocation이 보이지만 다른
workflow일 가능성이 있는 allocation에 임의 attach하지 않는다. 사용자가
지정한 interactive shell에서만 이 v71 contract로 native preflight와 matched
campaign을 실행한다.

현재 지정 범위로 확인된 allocation `57443205`는 `WorkDir`가 이 repository인
4-node/16-GPU/4-hour interactive job이다. v71 native-step preflight
(`srun --jobid=57443205 --overlap --exact --nodes=4 --ntasks=4
--ntasks-per-node=1 --gpus-per-task=4 --cpus-per-task=128
--network=disable_rdzv_get ... nvidia-smi -L`)에서 `nid001148`와
`nid001149`의 task 0/1은 `Error configuring interconnect`로 실패했고,
`nid001152`와 `nid001153`의 task 2/3은 각 4개 A100을 확인했다. 따라서
allocation 전체 preflight는 fail-closed이며 C5 arm은 시작하지 않았다. v71
parent runner를 같은 allocation에서 공식 receipt로 남기려는 시도도 shared
`.tempo_go_native_campaign.lock` 보유로 중단됐다. lock을 지우거나 우회하지
않았고, 이 attempt에는 native 성능 수치나 partial arm 비교를 만들지 않는다.

다음 native gate에서는 이 수정이 실제로 queue-GPU-only의 단순 waiting보다
나은지 확인해야 한다. 특히 (a) active sequence/full endpoint queue,
(b) LMCache remote KV·semantic-op pressure, (c) shared NCCL/Cassini hot NIC와
cool spare pair, (d) tenant burst와 SLO mix를 동일 offered population으로
재현하고, TEMPO가 queue lease를 많이 쓰는 것 자체가 아니라 output-token
goodput·tail SLO·fairness의 joint utility를 개선하는지 확인한다. native
matched win이 없으면 추가 component tuning을 하지 말고 route score,
queue-debt price, fabric shared budget, pair activation의 global control
loop를 함께 재설계한다.

## 50. queue lease의 live service-feedback 보정

v71/v72 replay의 tail을 request 단위로 대조한 결과, ordinary admission은
`PairTelemetry.remote_service_multiplier`를 route score에 반영하지만
`lease_queued_to_endpoint()`의 expired-waiter 경로는 nominal predicted E2E만
비교하고 있었다. 그 결과 C3 both-hot에서 remote service stretch가 이미
관측돼도 queue lease가 remote를 선택해 queue-GPU-only보다 tail이 나빠졌다.
이는 remote component의 단순 tuning 문제가 아니라, 같은 global state를
ordinary admission과 queue-debt admission이 서로 다르게 해석한 orchestration
consistency bug다.

수정 후 queue lease score는 ordinary admission과 동일하게

`predicted_e2e + predicted_ttft × (live_service_multiplier − 1)
 + uncertainty + scheduler/completion pressure + cross-layer externality
 + queue-debt overage penalty`

를 사용한다. queue lease가 global reservation window를 넘길 수 있다는
뜻이지, 측정된 route service degradation을 무시해 느린 path로 보내도 된다는
뜻은 아니다. 이 score는 local/remote 선택, shared budget, endpoint queue
debt와 tenant deadline을 같은 decision에서 연결하며, 선택된 route와 탈락
후보의 사유는 immutable receipt에 남는다.

focused regression은 `131 passed, 22 subtests passed`였다. v74 immutable
source boundary는
`results/tempo_go_c5_source_snapshot_v74_queue_lease_live_feedback_immutable`이고,
contract SHA/fingerprint/tree SHA는 각각
`d68fca6e93337327d1dc1f4e849da076e071b5a29252aa17e92ba7c5bec3cb9b`,
`33fc035e8402679dc70987f2259cde06b76dd40600f2dfe805d59314f98f28ea`,
`b9424e556fcc83c1deaeb034dfe5e718aa0fe28fa25a4da3bb8a184069d3a759`이다.
동일 276-row cross-layer replay
`results/tempo_go_replay_v75_queue_lease_live_feedback_contract_bound.json`은
동일 request/trace, failure/telemetry receipt, terminal/leak-free와
`frozen_run_contract_valid=true`를 통과했다.

| arm | complete/reject | E2E p50/p95/p99 (ms) | route local/remote |
|---|---:|---:|---:|
| queue-GPU-only | 276/0 | 7310.71 / 8447.54 / 8447.54 | 224/52 |
| TEMPO v75 | 276/0 | 7404.00 / 8447.54 / 8463.82 | 248/28 |

v75는 v72 대비 p99를 약 2.1% 낮췄지만 queue-GPU-only보다 p50 약 1.3%,
p99 약 0.2% 높다. 따라서 이것은 global decision consistency와 tail
recovery의 mechanism evidence이지 performance claim이 아니다. native에서는
같은 live endpoint feedback이 실제 vLLM scheduler/LMCache/NCCL/Cassini
contention에서 유지되는지, route 감소가 output-token goodput·tenant
SLO/fairness·failure cost를 함께 개선하는지 확인해야 한다. v75 이후에는
이 원인에 대해 추가 queue-score 숫자 tuning을 하지 않고 native matched gate로
넘긴다.
p50 차이는 reject 때문이 아니라 route mix에서 발생했다. v75에서 TEMPO는
remote를 28건, queue-GPU-only는 52건 선택했고, TEMPO의 phase별 remote 선택은
C1 `13/74`, C2 KV-hot `4/42`, C3 `7/126`이었다. 이 replay의 cross-layer
pressure는 명시적 offline fixture이고 실제 remote completion tail을 측정하지
않으므로, 이를 근거로 remote 비율을 다시 올리는 coefficient tuning을 하지
않는다. native observer가 같은 window의 LMCache/NCCL/Cassini externality와
실제 output-token/SLO utility를 보여줄 때에만 이 보수적 route mix가 이득인지
판정한다.

## 51. whole-system utility를 native에서 판정하기 위한 v76/v77 경계

현재 cross-layer global loop가 개별 컴포넌트의 숫자 조정이 아니라 전체
decision path를 바꾸는지 확인하기 위해, 동일한 276-row offered population을
현재 source로 다시 replay했다. v76은 cross-layer fixture를 끈 mutable
control-plane replay이고, v77은 동일한 request/endpoint prior에 NCCL,
LMCache, Cassini vector와 shared remote budget/stagger를 넣은 mutable
control-plane replay다. 둘 다 GPU/NCCL/LMCache/Slingshot의 실제 completion을
측정하지 않으며 performance claim을 허용하지 않는다.

| arm | replay | complete/reject | E2E p50/p95/p99 (ms) | TTFT p50 (ms) | local/remote |
|---|---|---:|---:|---:|---:|
| queue-GPU-only | v76/v77 | 276/0 | 7310.71 / 8447.54 / 8447.54 | 4516.14 | 224/52 |
| TEMPO | v76 no cross-layer | 276/0 | 7361.03 / 8447.54 / 9244.11 | 4595.46 | 238/38 |
| TEMPO | v77 cross-layer | 276/0 | 7404.00 / 8447.54 / 8463.82 | 4635.57 | 248/28 |

v77은 v76보다 p99를 약 8.4% 낮췄지만, queue-GPU-only보다 p50은 약
1.3%, p99는 약 0.2% 높다. cross-layer state가 route mix를 실제로 바꿨고
tail recovery mechanism이 작동했다는 점은 확인되지만, primary 5% goodput/
median gate나 robustness 15% gate는 통과하지 않았다. background tenant의
replay SLO-goodput도 queue-GPU-only 122건에 비해 TEMPO 113건으로 낮았다.
이 숫자는 control-plane service model의 accounting일 뿐 native service
capacity나 physical fabric 성능이 아니다.

이 결과의 전체 시스템 해석은 명확하다. 현재 policy가 remote를 덜 선택한
것은 remote가 본질적으로 나빠서가 아니라, offline fixture의 shared
cross-layer externality price가 route prior의 remote benefit보다 크게
계산됐기 때문이다. 따라서 이 fixture를 이용해 remote coefficient나
queue-score 숫자를 다시 튜닝하지 않는다. native에서 같은 window의
LMCache/UCX completion, NCCL collective progress, per-NIC Cassini/Slingshot
externality와 tenant별 output-token/SLO utility를 함께 관찰해야 한다. 그
결과 remote benefit이 shared externality보다 큰 business class에서는
현재의 보수적 route mix가 잘못된 것이고, global score를 business-aware
marginal utility allocation으로 바꿔야 한다. 반대로 native에서도
externality가 실제 service capacity를 낮추면 현재 throttle은 robustness
actuator로 보존하고, queue-GPU 대비 aggregate goodput과 tail이 개선되는지
검증한다. 어느 경우도 component-only negative로 TEMPO 전체를 판정하지 않는다.

현재 승인 allocation `57443205`는 여전히 4-node/16-GPU interactive로
RUNNING이지만, bounded P1 preflight에서 `nid001148`과 `nid001149`의
`Error configuring interconnect`가 발생했고 `nid001152`와 `nid001153`만
GPU visibility를 통과했다. native campaign lock도 held 상태다. 이 때문에
lock이나 allocation을 우회하지 않고 C5 matched performance run을 만들지
않았다. 이 stop은 TEMPO utility negative가 아니라 allocation/interconnect
readiness failure다. lock이 정상 해제되고 allocation-wide capability가
통과한 뒤에만 P2/P5의 same-population native campaign을 재개한다.

v76 output SHA는
`ab7f5102122c9ba962efa183ce4c3d30452886e4fcb3b7e93ff6bec9623d5fa9`,
v77 output SHA는
`0cd33f70d6fd5de1f4b7651f22786a973ffd5cf129f3851d83074cdd81876b8d`다.

## 52. native readiness 원인 분리와 launcher 수정

같은 승인 allocation `57443205`에서 campaign launcher가 사용하던
`--network=disable_rdzv_get`를 제외한 bounded four-node GPU preflight를
직접 실행했다. `nid001148`, `nid001149`, `nid001152`, `nid001153` 모두
호스트와 4개의 A100 visibility를 통과했다. 앞서 동일 allocation에서
network option을 포함한 preflight는 `nid001148`과 `nid001149`에서만
`Error configuring interconnect`를 냈다. 따라서 이번 stop의 원인은 GPU
allocation 자체 또는 root/UDI 문제가 아니라, 특정 Slurm network flag를
모든 allocation에 강제한 launcher readiness contract였다.

`run_tempo_go_cross_layer_with_cojob_in_allocation.sh`와
`run_lmcache_nixl_contention_2node_in_allocation.sh`를 수정했다. 기본
preflight와 co-job step은 native Slurm network 설정을 그대로 사용하고,
`TEMPO_GO_SRUN_NETWORK_MODE=disable_rdzv_get`가 명시적으로 제공될 때만
opt-in한다. 자동 network fallback, repeated retry, Slingshot/VNI
reconfiguration은 추가하지 않았다. preflight receipt의 network field도
실제 mode(`default` 또는 opt-in 값)를 보존한다. 이 수정은 native
readiness를 닫기 위한 execution-boundary correction이지 TEMPO 성능
결과가 아니다.

wrapper source가 바뀌었으므로 historical v74 contract/snapshot을
재사용하지 않는다. lock이 정상 해제된 뒤 현재 source에서 새 immutable
source snapshot과 run contract를 만들고, 새 contract verify → 기본 network
mode P1 preflight → NCCL/LMCache co-job readiness → same-population matched
campaign 순서로 진행한다. 현재 allocation과 campaign lock을 우회하지
않았으며, 이 수정 전후에 native request/utility 수치를 만들지 않았다.

v78 immutable snapshot contract는
`results/tempo_go_c5_source_snapshot_v78_network_mode_default_immutable/native_run_contract.json`이며
contract SHA는
`0fba037a8ac1e1e2affa69ff99f39eab03180e5633dfc424fe577466e2ee660a`,
fingerprint는
`41a56895fdb9eea618674ca77cc5ede320b28e333dbf90a04b2d9f5e8c06bb45`,
source tree SHA는
`4d6b9a5ff453241f11675420c02845c00fa7f76f9bfa1291c1fd128ec6d27d6d`다.
새 contract verify는 `performance_claim_allowed=false`로 통과했다.

v78 P1 capability receipt는
`results/tempo_go_c5_source_snapshot_v78_network_mode_default_immutable/p1_capability_receipt/`
에 있으며, 네 노드 모두 `cuda_available`, `nccl_available`, A100×4,
Cassini core/optional counter와 NCCL/CUDACollective·official LMCache/NIXL
harness source gate를 통과했다. 네 rank의 Cassini topology fingerprint는
`f6e6adfd453414cfdf1d379ae14e6d0eb1007ad382948e76dc14d755c259f445`로
일치한다. 따라서 P1 capability는 GO이고, 다음 unresolved gate는
campaign lock이 해제된 뒤의 P2 co-job/vLLM causal overlap과 P5 matched
end-to-end utility다.

## 53. v78 native launch stop

v78 contract verify 직후 승인 allocation `57443205` 내부에서 1-task
launcher step을 실행했다. contract receipt는 정상 출력됐지만 runner가
workload/co-job 이전에 `TEMPO native campaign lock is held; refusing a
duplicate launcher`로 exit 3했다. login node에서 lock이 잠시 free로
관찰된 뒤 compute step에서는 held였고, 현재 holder는 user systemd 경계로
확인된다. lock file을 삭제하거나 다른 경로로 launch를 우회하지 않았다.

따라서 v78 attempt에는 native request, NCCL/LMCache overlap, vLLM arm,
성능/utility 수치가 없다. stdout SHA는
`b81c0a09638db046997be139cdfcc08207e92e3afba35047eee27e54d7d14f30`,
stderr SHA는
`6d36cec36c90927ad06ff14befa8145c2ad5dbe5fc41011da15460bf8e976147`다.
P1 capability GO와 launcher/lock execution stop을 분리 보존하며, 같은
allocation에서 duplicate launch를 반복하지 않는다. lock holder가
정상적으로 종료된 뒤에만 이미 verify된 v78 contract와 default-network
wrapper로 P2/P5를 재개한다.

native launch stop 이후에도 v78 source/contract boundary가 policy replay를
깨뜨리지 않았는지 확인하기 위해 contract-bound v78 replay를 실행했다.
`results/tempo_go_replay_v78_network_mode_default_contract_bound.json`은
`frozen_run_contract_valid=true`, same request/trace, no phase/physical-switch
input, terminal/leak-free와 failure/telemetry receipt gate를 모두 통과했다.
TEMPO는 276/276 complete, 0 reject, E2E p50/p95/p99 `7404.004/8447.538/
8463.818 ms`, route local/remote `248/28`이었다. queue-GPU-only는
276/276, E2E `7310.711/8447.538/8447.538 ms`, local/remote `224/52`였다.
이는 v75/v77과 일치하는 control-plane mechanism replay이며 native
performance claim은 아니다. output SHA는
`d9c0c8bfb9008bd1dc4813374ceb5ac15eb98d30ab9cc40ffdfdacbb12a99b4c`다.

## 54. P4 hierarchical scale evidence

native lock 때문에 P2/P5를 반복하지 않는 동안 §18 P4의 CPU hierarchy
gate를 실행했다. `tempo/test_pd_global_hierarchy.py`와 coordinator
integration test는 `20 passed`였다. 동일한 identity/telemetry contract로
pair population을 2, 8, 32, 128, 1024로 늘린 bounded reducer 결과는 다음과
같다.

| pairs | raw candidates | forwarded candidates | omitted pairs | shards | reduce p50 |
|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 4 | 0 | 1 | 0.624 ms |
| 8 | 16 | 4 | 6 | 1 | 0.816 ms |
| 32 | 64 | 8 | 28 | 2 | 2.360 ms |
| 128 | 256 | 32 | 112 | 8 | 7.525 ms |
| 1024 | 2048 | 256 | 896 | 64 | 69.986 ms |

이 결과는 node→pair→shard fan-in이 global로 전달하는 후보 수를 bounded
하게 유지하고 omission receipt를 보존한다는 P4 control-plane evidence다.
1024-pair에서도 forwarded candidate는 256개로 제한됐지만, 70ms는 native
request latency나 Perlmutter-scale production overhead가 아니다. 실제
P2/P5에서는 telemetry collection, serialization, cross-node transport,
global decision과 tenant utility를 별도로 측정해야 하며, 이 CPU reducer
결과만으로 performance gate를 열지 않는다.

## 55. v80 native-network arm과 공유 allocation fabric-admission stop

v79 source snapshot을 그대로 덮지 않고, C5의 measured arm에도 v78/v79
wrapper와 같은 network 계약을 적용한 v80 immutable snapshot을 만들었다.
C5 arm은 기본적으로 Slurm/Perlmutter의 native network 설정을 사용하고,
`TEMPO_GO_SRUN_NETWORK_MODE=disable_rdzv_get`가 명시된 경우에만 해당
옵션을 전달한다. v80 contract는
`results/tempo_go_c5_source_snapshot_v80_native_network_arm_immutable/native_run_contract.json`이며,
contract SHA는
`778a04f0ceababb54e78bc439d71897670f896bb15231191a805ab9b7bd36ee7`,
fingerprint는
`cee261bcf8b1f86a1f547ff23d4b9bd364eb59108ff74fd8165ae7a77f3e1a31`,
source tree SHA는
`56f2e6428b26f4639208877797b0196a173202295302f39adc942f726eda8836`다.
verify는 `performance_claim_allowed=false`로 통과했다.

v80을 승인 allocation `57443205`의 올바른 4-node 부모 step에서 시작하려고
했지만, 부모 step `57443205.36`이 task 0/1 launch 단계에서
`Error configuring interconnect`로 실패했다. task 2/3가 남아 orphan된
상태는 정확한 step `57443205.36`만 종료해 정리했으며 allocation 자체,
`57443240`, 기존 실험 step은 건드리지 않았다. native launch receipt는
`results/tempo_go_native_v80b_outer_57443205.receipt.json`이다.

그 직후 같은 allocation에서 application을 전혀 시작하지 않는 bounded
4-node default-network preflight도 네 rank 모두 같은 interconnect 설정
실패를 냈다. receipt와 명령은
`results/tempo_go_direct_preflight_v80_57443205/receipt.json`에 보존했다.
이 시점의 Slurm 상태에는 기존 `57443205.27`의 2-node
`tempo-go-cross-layer-cojob-57443205`와 `57443205.35`의 4-node queue-GPU
arm이 RUNNING으로 남아 있었다. 따라서 active steps가 이번 VNI/fabric
admission 실패의 직접 원인이라고 단정하지는 않지만, **공유 allocation에서
이미 co-job·P/D arm이 자원을 사용 중일 때 신규 global step이 interconnect
구성 단계에서 거부될 수 있음**은 관찰됐다. 이는 실제 contention 경로에서
Slingshot/Cassini 상태와 scheduler/step admission을 함께 제어해야 한다는
TEMPO의 whole-system motivation을 강화한다.

중요하게도 v80은 preflight 이전에 멈췄다. vLLM P/D request, LMCache/NIXL
completion, NCCL collective, Cassini counter window, output-token goodput,
SLO/fairness를 하나도 측정하지 않았으므로 native performance claim이나
TEMPO negative로 해석하지 않는다. 다음 실행은 기존 running step이 정상
종료되고 allocation-wide native step admission이 다시 GO가 된 뒤에만
재개한다. 기존 step을 강제로 정리하거나 `--network` 옵션으로 우회하지
않으며, 그 전에는 새 duplicate campaign을 시작하지 않는다.

## 56. v69 shared-contention seven-arm 결과: global path가 드러낸 실제 병목

기존 allocation `57443205`에서 실행 중이던 v69 seven-arm campaign과 같은
allocation의 official LMCache/NIXL+NCCL co-job이 모두 종료됐다. co-job은
3,881개 measured block에서 `130,224,750,592` bytes를 source/receiver 모두
검증했고, `overall_correctness_met=true`를 기록했다. co-job summary는
background completion p50/p99 `27.318/2109.602 ms`, global token-tail
p50/p99 `0.298/11.633 ms`였고, final observer sequence는 3,882였다.

native analyzer는
`results/tempo_go_c5_cross_layer_native_v69_tenant_reserved_admission_57443205/native_five_arm_analysis_v69.json`
에 생성됐다. seven-arm identity, same workload SHA, frozen v69 contract,
4-node/16-GPU/UCX와 endpoint completion receipt gate는 보존됐지만
`performance_claim_allowed=false`다.

| arm | request/complete | reject | service failure | output tokens | descriptive output-token goodput (/s) | E2E p50/p99 (completed only, ms) |
|---|---:|---:|---:|---:|---:|---:|
| local | 276/276 | 0 | 0 | 34,176 | 563.499 | 11,766.8 / 29,727.2 |
| remote | 276/276 | 0 | 0 | 34,176 | 645.407 | 10,265.2 / 28,859.2 |
| predictor | 276/276 | 0 | 0 | 34,176 | 607.282 | 11,855.4 / 28,489.2 |
| queue-GPU-only | 276/276 | 0 | 0 | 34,176 | 659.605 | 10,783.9 / 27,445.8 |
| network-request-only | 276/276 | 0 | 0 | 34,176 | 557.318 | 11,809.7 / 31,195.6 |
| app-global-only | 276/0 | 0 | 1 execution failure | — | — | — |
| TEMPO | 276/173 | 57 | 46 endpoint/HTTP failures | 20,464 | 606.353* | 10,065.5 / 15,027.9* |

`*` TEMPO 값은 arm이 중간에 실패한 뒤 남은 completed subset에 대한
descriptive 수치이며, queue-GPU/predictor와의 performance 비교에 사용하지
않는다. TEMPO의 46 failures는 모두
`endpoint_bounded_queue_lease_timeout`이었고, global decision reason은
`global_endpoint_queue_lease_route_committed` 132건,
`global_min_cost_fair_route_committed` 86건,
`global_admission_queue_timeout` 47건,
`global_telemetry_stale` 8건,
`global_telemetry_refresh_timeout` 2건이었다. app-global-only는 warmup
뒤 실제 요청에서 `502 Bad Gateway`, `httpx ReadError/RemoteProtocolError`로
exit 143했다. 즉 local/remote route selector만의 문제가 아니라, global
decision이 endpoint service capacity와 live telemetry를 소유하는 순간의
failure boundary가 전체 service utility를 결정한다.

v69 TEMPO arm의 cross-layer observer는 219 valid provenance observation을
남겼고, NCCL/LMCache co-job correctness는 유지됐다. 동시에 scheduler
observation invalid count 333, cross-layer payload invalid count 276이
발생해 scheduler/fabric signal freshness와 endpoint queue lease를 함께
닫아야 함을 보여줬다. 이것은 TEMPO의 가치가 없다는 결과가 아니라,
**global orchestrator가 실제로 제어권을 행사할 때 어떤 공동 상태를
actuation에 넣어야 하는지 보여준 whole-system causal failure**다.

따라서 v69에서 다음을 하지 않는다.

- completed-only TEMPO p50을 이용해 queue-GPU보다 빠르다고 주장하지 않는다.
- endpoint lease timeout 숫자나 route coefficient를 offline에서 다시 맞추지 않는다.
- app-global-only 또는 v69 source를 TEMPO 전체의 negative로 일반화하지 않는다.

현재 source의 live queue-lease feedback/native endpoint-debt 경계를 freeze한
v80 contract를 이제 동일 population과 co-job에서 실행해, v69의 46 lease
failure와 telemetry invalidity가 줄면서 전체 output-token goodput, SLO,
fairness와 tail이 동시에 좋아지는지를 검증한다.

business-class 기준으로 보면 v69 TEMPO는 latency `12/12`, batch `11/12`,
interactive `7/12`, background `143/240`을 완료했고, queue-GPU-only는 네
class 모두 `100%`를 완료했다. TEMPO의 latency class 보호 자체는 global
fairness actuator의 작동 증거지만, background/interactive의 lease timeout과
failure를 합치면 aggregate business utility는 개선되지 않았다. 다음
version의 성공 조건은 latency tenant를 보호하는 것과 동시에 background
work를 무기한 희생하지 않는 bounded business utility/fairness이며, 이를
endpoint lease·telemetry freshness·pair capacity를 하나의 decision으로
계산해 검증한다.

## 57. v81 launcher lifecycle correction

v69 종료 후 allocation `57443205`에는 active experiment step이 없고
`squeue --steps`에는 `extern`만 남았지만, v80 allocation lock
`results/.tempo_go_native_campaign_57443205.lock`은 user-systemd PID
`1953729`가 계속 보유한 상태로 관찰됐다. lock file을 지우거나 `flock`을
우회하지 않았다. 이 상태에서 v80을 재실행하면 duplicate guard가 정상적으로
거부하므로, 현재 source에 launcher lifecycle correction을 추가했다.

`run_tempo_go_cross_layer_with_cojob_in_allocation.sh`는 lock 획득 직후
preflight/early-exit에도 적용되는 `release_campaign_lock` EXIT cleanup을
등록하고, co-job lifecycle EXIT trap에서도 `stop_cojob` 뒤 명시적으로
unlock한다. lock file 삭제나 다른 holder 강제해제는 하지 않는다. 이 변경은
controller coefficient나 policy를 바꾸지 않는 execution-safety correction이다.

그 source를 freeze한 v81 immutable contract는
`results/tempo_go_c5_source_snapshot_v81_lock_cleanup_immutable/native_run_contract.json`이며,
contract SHA는
`8588b4911cfda8663933afff6912e497b16e8aee1300220891cd4b81ccb0e929`,
fingerprint는
`afe6cee5b58dd6ff16a1004f4e4203d71e4985ecf23c8b4430c0b7093a4ce1e4`,
source tree SHA는
`8099bf42c3f71b565833fd1fa53e4b0c8aa33f6e44653aeff8eb852093b96d8c`다.
full contract verify는 `performance_claim_allowed=false`로 통과했다.

v69 native analysis SHA는
`221a4c35119a243784ea774e28cdccdd834ee10ecbb7387a696990c4d8b61681`다.
다음 native 시도는 기존 lock을 우회하지 않고, user-systemd holder가
정상적으로 release된 뒤 v81 contract로만 수행한다. 그때 v69의 same
population/co-job을 유지한 채 endpoint lease failure, invalid telemetry,
business-class completion과 전체 output-token goodput을 재비교한다.

## 58. Perlmutter `Error configuring interconnect`의 실행 경계 판정

2026-08-23 현재 이 오류를 TEMPO controller의 실패로 분류하지 않는다.
과거 receipt를 다시 대조한 결과, 실패 순서는 다음과 같다.

1. Slurm이 한 노드의 일부 task에 `Error configuring interconnect`를 내고
   native Python/NCCL 프로세스를 시작하지 못한다.
2. 다른 노드의 task는 먼저 시작되어 NCCL rendezvous 주소를 기다린다.
3. 실패한 노드의 rendezvous listener가 존재하지 않으므로 나머지 task에
   `socketPollConnect ... Connection refused`가 연쇄적으로 발생한다.

따라서 `Connection refused`는 1번의 원인이 아니라 1번의 2차 증상이다.
실제 증거는 [v45 retry co-job stderr](../results/tempo_go_cross_layer_cojob_v45_snapshot_retry2_57426952/cojob.stderr.log)의
Slurm launch failure와 rank 4--7 NCCL log, 그리고 [v80 direct preflight receipt](../results/tempo_go_direct_preflight_v80_57443205/receipt.json)의
`application_started=false`다. v47 receipt도 이를
`allocation-node-interconnect-capability-failure`로 분류하며 Python/NCCL이
시작되지 않았음을 기록한다.

원인은 Perlmutter의 job-step 자원 모델을 위반한 실행 topology다. NERSC
문서는 같은 노드에서 병렬 `srun`들이 CPU/memory뿐 아니라 Slingshot
network resource도 동시에 over-allocate할 수 없고, 현재 Slingshot
configuration은 노드당 동시에 세 application까지만 지원한다고 명시한다.
`--overlap`은 자원 공유를 허용할 뿐 이 network 한도를 없애지 않는다.
따라서 outer `srun`/`srun --pty` + parent interactive step + C5 + co-job 또는
이전 orphan step이 남아 있으면 일부 노드에서 네 번째 network user가 되어
opaque한 interconnect error가 된다. `srun --network=no_vni`는 network
resource를 사용하지 않는 진단/쉘 step에만 사용해야 하며 NCCL/UCX co-job의
대체 transport로 쓰면 안 된다.

현재 올바른 native topology는 다음 하나다.

```text
plain 4-node salloc --network=job_vni
  └─ optional allocation interactive parent (1)
      ├─ official 2-node LMCache/NIXL + NCCL co-job (1)
      └─ 4-node vLLM P/D C5 step (1)
```

현재 launcher는 이 경계를 반영한다. allocation 자체의 `Network=job_vni`,
4 nodes/16 GPUs, `CPUs/Task>=128`을 검사하고, `squeue --steps`에서
extern/interactive/current launcher 이외의 live step이 있으면 시작하지
않는다. 그 다음 four-node GPU smoke step을 먼저 실행한다. smoke가 실패하면
co-job과 C5를 시작하지 않는다. 이는 `Error configuring interconnect`를
NCCL 성능 결과로 오염시키지 않는 P0/P1 실행 gate다.

현재 확인된 native facts:

- v97 very-light run은 `job_vni`와 같은 allocation co-job에서 official
  `NCCL_NET=AWS Libfabric`, NIXL UCX, NCCL correctness를 통과했다. 그러므로
  Perlmutter에서 native path 자체가 불가능한 것은 아니다.
- v97 stress directory의 모든 arm failure는 interconnect가 아니라 immutable
  v97 source snapshot과 live `tempo/pd_global_orchestrator.py` digest mismatch
  (`source digest differs`)였다. v98은 새 immutable source snapshot과 contract
  로 이를 고쳤다.
- v98 policy의 bounded offline replay는 v97 headroom profile 대비 TEMPO
  rejection을 0에서 24/276으로 예측한다. 이는 실행 가능성/decision 변화의
  control-plane evidence일 뿐 성능 결과가 아니다. 실제 native 성능 주장은
  새 4-node allocation에서 다시 측정할 때까지 금지한다.

## 59. Native bottleneck telemetry protocol

`fabric_pressure` 하나로 결론을 내리지 않는다. 한 allocation의 각 C5 arm과
동일 co-job window에 대해 다음 네 계층을 같은 monotonic/unix timestamp와
`slurm_job_id`, step id, hostname, NIC id, GPU UUID로 저장한다.

| 계층 | 수집값 | 도구/출처 |
|---|---|---|
| GPU | SM/HBM utilization, power/thermal violation, PCIe RX/TX, NVLink RX/TX, active processes | `nvidia-smi dmon`, `nvidia-smi pmon`, `nvidia-smi topo -m`, NCCL INFO/RAS |
| Slingshot/Cassini | NIC link/rate, pause PCP 0--7, corrected/uncorrected rate, AER, per-NIC identity | `cxi_stat --list`, `cxi_stat --rates --pause=1`, `cxi_stat --aer`, `fi_info -p cxi -l` |
| transport | NCCL collective tail/arrival spread, NIXL transfer p50/p99, timeout/error, provider/env | existing `tempo-nccl-observer-v1`, NCCL INFO, native transport receipt, UCX/NIXL logs |
| service/business | vLLM queue wait, active seq/decode tokens, TTFT/TPOT/E2E p50/p95/p99, output tokens, rejection/failure/fairness | vLLM observe-only snapshot + C5 request ledger + global decision receipts |

진단용 `nvidia-smi`/`cxi_stat` step은 NERSC가 허용한
`--network=no_vni`로 실행해 network-step budget을 소비하지 않는다. 반대로
NCCL/NIXL co-job과 C5는 반드시 allocation-time `job_vni` 안에서 native
Slingshot path를 사용한다. eBPF/bpftrace는 CPU syscall/런타임 scheduling
지연을 보조하는 도구이지 Cassini congestion counter의 대체가 아니므로,
권한/오버헤드가 확인된 경우에만 별도 보조 trace로 둔다.

다음 native run의 중단/성공 판정은 고정한다.

- allocation `Network=job_vni`, exact shape, no orphan sibling step;
- four-node smoke pass;
- co-job `overall_correctness_met=true`, observer가 C5 window 전체를 덮음;
- 일곱 arm 동일 population 및 동일 co-job;
- `Error configuring interconnect`, source digest mismatch, NCCL rendezvous
  cascade가 하나라도 있으면 해당 run은 performance-invalid receipt;
- 그 뒤에만 aggregate output-token goodput, class별 completion/fairness,
  TTFT/E2E/p99/SLO, Cassini/NCCL/NIXL 공동 telemetry를 비교한다.

이 경계는 TEMPO의 가치를 축소하는 것이 아니라, TEMPO가 해결하려는 바로
그 shared-resource contention을 측정할 때 launcher 자체의 Slurm network
contention을 실험 workload와 혼동하지 않게 하는 필수 조건이다.

## 60. v99 queue-debt controller와 최신 native contention receipt

2026-08-23 현재 primary source-bound contract는
`results/tempo_go_c5_source_snapshot_v99_queue_debt_capacity/native_run_contract.json`이다.
이 contract의 file SHA는
`93d44da1ebd782e5849dc8b88f8f54d4acc88858c6123aa49d421e9980e31f54`,
fingerprint는
`dbbbc868568b6fc2ec364e505e4cb2b274746d3e7b93e6eae1073d25cc21ffe3`이며,
source tree SHA는
`eb16b4c71982c536baf30cd6fd5654df85b37922abc4f67ee058cb08fa3d01a4`다.
v99의 controller 변경은 `endpoint_queue_capacity`를 active endpoint
reservation(`capacity.endpoint_requests`)과 합치지 않고 vLLM waiting queue
debt로 별도 계산하는 것이다. 따라서 active reservation이 가득 차 있어도
실제 scheduler waiting queue에 headroom이 있으면 queue lease를 정확히
판정할 수 있다. controller parameter tuning이나 data-plane 변경은 없다.

v99 current-source CPU/control-plane verification은 다음을 통과했다.

```text
tempo/test_pd_global_orchestrator.py
tempo/test_pd_global_coordinator.py
tempo/test_pd_global_cross_layer.py
eval/sota_4node/test_run_tempo_go_cross_layer_allocation_static.py
eval/sota_4node/test_analyze_tempo_go_c5_five_arm.py
82 passed
```

### v99 native stress receipt

allocation `57473916`은 `Network=job_vni`, 4 nodes/16 GPUs,
`CPUs/Task=128`의 four-node native preflight를 통과했다. co-job의
`native_transport_receipt.json`은 `NCCL_NET=AWS Libfabric`, CXI provider,
GPU Direct RDMA, `FI_CXI_RX_MATCH_MODE=hybrid`, official LMCache/NIXL UCX를
기록한다. 이는 NERSC가 설명하는 GPU-node 구성(4 A100 + 4 Slingshot NIC)과
일치한다. [NERSC network documentation](https://docs.nersc.gov/performance/network/)

hot contention profile에서는 observer sequence 904까지
`lmcache_transfer_p99_ms=14829.458686`이 관측되었고,
`nccl_collective_p99_ms=5.381873`이었다. 이후 8-rank co-job의 모든 rank가
NCCL `ALLREDUCE SeqNum=15384`에서 약 60초 정지하여 step exit 143이
발생했다. 이 receipt는 allocation/VNI/NCCL initialization failure가
아니라 실제 LMCache/NIXL--NCCL shared-fabric overload/stall evidence다.
그러나 co-job이 TEMPO arm 전에 종료되어 7-arm utility comparison은
성립하지 않는다. 따라서 이 run의 `performance_claim_allowed`는 false로
유지한다.

완료된 partial fixed arms의 수치는 아래처럼 descriptive evidence로만
보존한다.

| arm | request/complete | output-token goodput (/s) | E2E p50/p99 (ms) |
|---|---:|---:|---:|
| local | 276/276 | 518.304 | 11234.6 / 35835.9 |
| remote | 276/276 | 615.439 | 10644.5 / 30453.0 |
| predictor | 276/276 | 492.665 | 11415.3 / 36531.6 |
| queue-GPU-only | 276/276 | 594.972 | 8167.2 / 36640.8 |
| network-request-only | 276/276 | 838.935 | 10338.4 / 16849.8 |
| app-global-only | not started | — | — |
| TEMPO | not started | — | — |

`network-request-only`의 partial 수치가 높아 보여도 arm order가 sequential이고
global/app/TEMPO arm이 없으므로 causal superiority로 해석하지 않는다.

### 다음 native gate

현재 두 개의 `gpu_interactive` allocation이 실행 중일 때 Perlmutter QOS
submit limit 2 때문에 새 `salloc`은 `QOSMaxSubmitJobPerUserLimit`로
거부된다. 이것은 performance failure가 아니며 기존 allocation을
`sattach`, `scancel`, step hijack으로 재사용하지 않는다. [NERSC QOS limit table](https://docs.nersc.gov/jobs/policy/)
에 따라 슬롯이 확보된 뒤에만 다음 exact allocation을 요청한다.

```text
-A m1248_g -C gpu -q interactive -t 04:00:00 -N 4
--ntasks-per-node=1 --cpus-per-task=128 --gpus-per-node=4
--network=job_vni
```

새 allocation에서는 v100 contract와 같은 7-arm order, 같은 validation
workload, official LMCache Connector V1/UCX data plane을 사용하고,
co-job은 hot stress가 아니라 4 MiB/rank, 8 iterations, 1 MiB foreground,
0.25 s block cadence의 sustained-moderate profile로 C5 measured window를
끝까지 덮는다. 성공 판정은 TEMPO가 strongest fixed(`local`, `remote`,
`queue_gpu`)와 predictor를 aggregate output-token goodput/SLO-goodput,
E2E/TTFT/TPOT p99, business-class fairness에서 동시에 gate 통과하는지다.
실패하면 해당 mechanism 범위의 reproducible negative로 기록하고,
single-component result를 TEMPO claim으로 승격하지 않는다.

## 61. v100 NCCL RAS socket isolation

v99 native receipt 이후 co-job과 C5가 같은 두 GPU node에서 독립 NCCL
communicator를 동시에 만드는 경계를 다시 점검했다. NCCL RAS는 main
NCCL/Slingshot data path와 별도의 localhost socket을 사용하며, NVIDIA
문서는 같은 node에서 독립 job이 겹칠 때 `NCCL_RAS_ADDR`를 job별로
분리하도록 요구한다. 따라서 co-job launcher에만
`NCCL_RAS_ADDR=127.0.0.1:<job-local-port>`를 추가했다. 이 변경은
controller parameter, route policy, LMCache/NIXL backend, NCCL transport,
GPU/CPU allocation을 바꾸지 않는 execution isolation이다.

이 수정은 v100 immutable snapshot에 rebinding했다.

| 항목 | 값 |
|---|---|
| contract | `results/tempo_go_c5_source_snapshot_v100_ras_isolated/native_run_contract.json` |
| candidate | `tempo-go-cross-layer-v100-ras-isolated` |
| revision | `v100-job-local-nccl-ras-address-only` |
| contract SHA | `722b50a2964248df004c89e4c5889b6051d39a72087e2ae6d95fb278b1c3804e` |
| fingerprint | `3c1c4c3cc8aea0c55e10eafb14b79ed7a56b74289d2b6fdc0592f9ee6ce8a767` |
| source tree SHA | `3b27f0354622f9386ae1fe66a8436fb79f5ca616bbc486dba20a2e9f4e90fe64` |
| contract gate | `performance_claim_allowed=false` |

`bash -n`, Python compile, static launcher tests와 cross-layer tests는
`17 passed`이고, v100 contract verify도 동일 validation workload와
7-arm order에 대해 통과했다. 다음 allocation에서 사용할 contract는
v99가 아니라 이 v100 snapshot으로 고정한다. RAS isolation이 성능을
개선했다고 주장하지 않으며, 단지 NCCL 초기화/진단 socket 충돌을 성능
실험의 원인으로 섞지 않기 위한 실행 gate다.

v99 hot observer snapshot을 현재 controller의
`CrossLayerTelemetry.route_externality()`에 replay한 결과도 보존한다.
동일한 `lmcache_transfer_p99_ms=14829.458686`과
`nccl_collective_p99_ms=5.381873`을 넣었을 때 route cost는 다음과 같다.

```text
local  = 0.822 ms      (confidence 0.875)
remote = 11123.185 ms  (confidence 0.727)
```

따라서 hot LMCache/NIXL tail은 global decision path에서 remote route의
공동 externality로 반영되고, local/fallback 방향의 actuation 근거가 된다.
이 replay는 telemetry-to-actuation control-plane evidence일 뿐이며,
native E2E goodput/p99 superiority 또는 physical switch bottleneck claim으로
승격하지 않는다. 그 판단은 v100 7-arm same-allocation run과 independent
validation gate 뒤에만 한다.

## 62. Perlmutter 문서 대조 후 native launch boundary 수정

Perlmutter 공식 문서는 GPU `srun`마다 GPU 자원을 명시해야 하며, `srun`의
resource shape가 빠지면 `Error configuring interconnect`가 날 수 있다고
설명한다. 또한 동시 application은 node당 Slingshot network resource를
최대 3개까지만 사용할 수 있고, `--network=no_vni`는 interconnect를 쓰지
않는 작업에만 허용된다. 따라서 TEMPO cross-layer 실험의 NCCL/UCX 경로는
allocation 단계에서 `--network=job_vni`를 받아야 하며, step 단계에서
뒤늦게 VNI를 만들거나 `no_vni`로 대체해서는 안 된다.

기존 preflight는 `Network=(null)`만 거부하고 `Network=no_vni`도 통과시킬
수 있었다. 이 경우 Slurm resource preflight는 통과하지만 NCCL/UCX가
유효한 Slingshot VNI를 얻지 못해 뒤늦게 `Connection refused` 또는 NCCL
초기화/collective failure로 보일 수 있다. 이를 다음처럼 수정했다.

1. `run_tempo_go_cross_layer_with_cojob_in_allocation.sh`는
   `scontrol show job`의 `Network`가 정확히 `job_vni`일 때만 co-job/C5를
   시작한다.
2. `(null)`은 `allocation_missing_job_vni`, 그 외 값(특히 `no_vni`)은
   `allocation_network_not_job_vni`로 별도 preflight receipt를 남긴다.
3. 자동 network-mode retry, UDI/Shifter/root 경로, 기존 allocation step
   hijack은 하지 않는다.

문서 대조 후 shell syntax와 관련 static tests는 `6 passed`다. 현재 확인된
정상 후보 allocation `57474485`는 `NumNodes=4`, `NumCPUs=512`,
`CPUs/Task=128`, `gres/gpu=16`, `Network=job_vni`로 이 gate를 만족한다.
반대로 별도 사용자 allocation `57474645`는 `Network=(null)`이고
`CPUs/Task=1`이므로 TEMPO가 절대로 재사용하거나 step을 걸어서는 안 된다.
다음 실행은 QOS slot이 비면 exact `gpu_interactive` allocation을 새로
받은 뒤, 그 foreground allocation shell에서만 v100 native campaign을
시작한다.

## 63. v101 provenance rebinding과 §18 P4 hierarchy scale receipt

§18의 historical v47 문구는 lineage 보존용으로 유지하고, 현재 실행 후보는
v100 이후의 exact `job_vni` preflight gate를 포함한 v101로 rebinding했다.
v100 source snapshot은 현재 outer launcher의
`Network=job_vni` exact gate를 포함하지 않으므로 v100을 그대로 재사용하지
않았다.

| 항목 | v101 값 |
|---|---|
| snapshot root | `results/tempo_go_c5_source_snapshot_v101_job_vni_preflight/source` |
| candidate | `tempo-go-cross-layer-v101-job-vni-preflight` |
| revision | `v101-exact-job-vni-launch-gate` |
| contract SHA | `3d676574d96692bbc611420200c1a0d04e8a1eb23653d1dfd088ac9f9c619019` |
| fingerprint | `2296eea9cbbf5e832916e22b78b0566e10c88927eef61d065280c0dc5d0fdd54` |
| source tree SHA | `e3e7c594fdb51a1054d28234c5865a269ba76614fec07a3a417a72b0bf40c330` |
| launcher runner SHA | `6ea7452cc640e834c8f755b120bb9f381e2a7333f1252662f8db75d8d7b7eaa4` |
| claim gate | `performance_claim_allowed=false` |

v101 contract verify, shell syntax와 cross-layer/controller/static suite는
`85 passed`다. 이 숫자는 native 성능 결과가 아니다. contract는 여전히
post-validation tuning과 performance claim을 금지한다.

§18 P4에 맞춰 같은 request/candidate/telemetry population에서 full global
candidate scan과 node→pair→shard bounded fan-in을 비교한
`results/tempo_go_hierarchy_scale_20260823.json`을 생성했다. 1,024 pair에서
raw 2,048 candidates가 bounded global layer에는 256개만 전달되고 896 pair가
omitted receipt에 기록됐으며 serialized global payload는 666,815 bytes에서
83,358 bytes로 87.499% 줄었다. bounded global reduction p50은 63.932 ms,
full reduction p50은 121.896 ms였다. 그러나 현재 단일 Python process에서
pair-agent frontier construction까지 합친 bounded total p50은 131.229 ms로
full scan보다 빠르지 않았다. 이것은 hierarchy가 이미 scale을 해결했다는
주장이 아니라, **global fan-in/payload는 줄지만 local frontier construction과
serialization이 아직 end-to-end control overhead**라는 중요한 P4 결과다.
따라서 native scale claim은 금지하고, 다음 구현/분석에서는 distributed agent
wall-clock, wire bytes, reducer CPU와 global decision latency를 분리 보고한다.

2026-08-23에 사용자 승인 exact allocation 요청을 한 번 실행했지만
`QOSMaxSubmitJobPerUserLimit`로 거부됐다. 새 Slurm job은 생성되지 않았고,
기존 `57474485` allocation에는 step을 걸지 않았다. 이 상태는 external
scheduler admission blocker이며 native performance failure가 아니다. 다음
확인에서 별도 `cs_hold_int` allocation `57475141`가 시작되어 현재
`gpu_interactive` 두 slot이 모두 사용 중임을 확인했다. 이 allocation 역시
재사용하거나 step을 걸지 않는다. 다음 시도는 반복 polling/retry 없이 QOS
slot이 확보된 뒤 v104 contract로 한 번만 수행한다.

## 64. v99 co-job timeout root-cause isolation과 v102 rebinding

v99의 `Error configuring interconnect` 계열과 최신 hot receipt를 같은
실패로 묶지 않는다. `results/tempo_go_cross_layer_cojob_57473916`의 native
preflight는 다음을 모두 통과했다.

- allocation `Network=job_vni`;
- 4 nodes/16 GPUs, `CPUs/Task=128`;
- `NCCL_NET=AWS Libfabric`, CXI/GDRDMA;
- `FI_CXI_RX_MATCH_MODE=hybrid`와 `MPICH_OFI_CXI_COUNTER_REPORT=2`;
- official LMCache/NIXL UCX channel initialization.

따라서 v99는 Slingshot VNI 생성 실패가 아니다. hot co-job은 observer
sequence 904까지 살아 있었고 `lmcache_transfer_p99_ms=14829.458686`을
기록했다. 그 뒤 한 source rank의 official NIXL `batched_write`가
configured 30초 drain window를 넘어서 `_run_block()`에서 예외를 냈다.
다른 rank들은 unconditional NCCL barrier에 진입했고, 그 결과
`ALLREDUCE SeqNum=15384`, 60초 watchdog timeout, step exit 143가 뒤따랐다.
rank stderr의 stack trace가 `run_lmcache_nixl_contention_2node.py:580`
barrier를 가리키므로, 이 run의 NCCL timeout은 primary fabric-init failure가
아니라 NIXL-overload 뒤의 rank-divergence cascade다. co-job이 C5의 모든
arm을 덮지 못했으므로 performance claim은 여전히 금지한다.

이 경계는 Perlmutter 문서의 concurrent `srun --overlap` 규칙에 맞춘다.
동시 application은 node당 Slingshot network resource를 최대 3개까지 쓸 수
있고, `--network=no_vni`는 interconnect를 사용하지 않는 작업에만 해당한다.
또한 `FI_CXI_RX_MATCH_MODE=hybrid`는 hardware message queue가 찼을 때
software matching으로 전환하며, `MPICH_OFI_CXI_COUNTER_REPORT=1..5`는
network timeout/counter report를 남긴다. 이 실험은 `--overlap`를 유지하되
`job_vni`, explicit GPU/CPU shape, hybrid matching, CXI report를 모두 receipt에
고정한다. [NERSC simultaneous-job guidance](https://docs.nersc.gov/jobs/examples/)
및 [NERSC Slingshot network guidance](https://docs.nersc.gov/performance/network/)

재현성 있는 다음 source boundary를 위해 co-job runner의 실패 전파를
수정했다. source-side NIXL timeout에서 한 rank만 즉시 raise하지 않고,
8 rank가 이미 초기화된 NCCL communicator의 bounded `MAX` status reduction에
참여한다. 하나라도 실패하면 전 rank가
`synchronized co-job block failure`로 함께 종료한다. retry, silent fallback,
NCCL/Slingshot privileged reconfiguration은 없다. 이 수정은 workload,
controller parameter, LMCache backend와 transport를 바꾸지 않고, primary
failure를 후속 unconditional barrier timeout으로 오염시키지 않는 lifecycle
수정이다.

| 항목 | v102 값 |
|---|---|
| snapshot root | `results/tempo_go_c5_source_snapshot_v102_sync_fail_boundary/source` |
| candidate | `tempo-go-cross-layer-v102-sync-fail-boundary` |
| revision | `v102-nixl-timeout-synchronized-before-nccl-barrier` |
| contract SHA | `9de2d50ff816c8468a1a278795fe3f5f0de245998f942e0f32d7d77c0a855f22` |
| fingerprint | `68fc8dfe6c6d095a409843eba7518d272ed7dc6ba510f579c576236e077aa251` |
| source tree SHA | `9203f2cec992b06e392ba068c97aa899c336aa0d3be803d86a5512385517c870` |
| claim gate | `performance_claim_allowed=false` |

v102의 LMCache/NIXL static/contract tests는 `12 passed`, shell syntax와
contract verify도 통과했다. 새 native 실행은 기존 allocation을 재사용하지
않으며, 새로 승인된 exact 4-node/4-hour `job_vni` interactive shell에서만
`eval/sota_4node/run_tempo_go_v104_moderate_in_allocation.sh`를 한 번 실행한다.
moderate profile은 v99 hot stress와 구분해 4 MiB/rank, 8 iterations,
1 MiB foreground, 0.25 s cadence를 사용한다. 이 run이 C5 전체 seven-arm을
완주한 뒤에만 aggregate utility/fairness/p99를 분석한다.

## 65. v103 P1 capability receipt와 다음 native 실행 boundary

v102까지는 `job_vni` allocation과 native NCCL/UCX smoke를 확인했지만,
co-job을 시작하기 전에 4개 node가 실제로 TEMPO가 읽을 수 있는
cross-layer capability를 모두 제공하는지에 대한 machine-readable receipt가
없었다. 이를 실행 경계에 추가했다. capability probe는 각 node에서 다음을
bounded하게 수집한다.

- `/sys/class/cxi` 아래의 node-local Cassini device/traffic-class 관측 가능성;
- `nvidia-smi -L`로 보이는 GPU 수와 GPU topology capability;
- PyTorch CUDA/NCCL runtime capability;
- optional counter가 없는 경우에도 `unsupported`/`unavailable` 상태를 보존하는
  support-state vector.

이 probe는 NERSC가 interconnect를 사용하지 않는 작업에 허용하는
`--network=no_vni`를 사용하고, 4개 node 각각에 별도 `srun` task를 둔다.
따라서 NCCL/UCX co-job의 `job_vni` step과 network resource를 공유하지 않으며,
receipt 생성이 끝난 뒤 종료된다. `no_vni`를 cross-layer data path의
fallback으로 사용하는 것이 아니다. 4개의 receipt가 정확히 병합되지 않으면
C5를 시작하지 않고 preflight failure로 끝낸다.

이 boundary는 capability가 곧 congestion 증거라는 뜻도 아니다. 실제
contention 판단에는 계속해서 Cassini per-NIC/traffic-class counter,
NCCL observer epoch/sequence/freshness/correctness, LMCache/NIXL in-flight와
vLLM scheduler snapshot을 함께 사용한다. P1 receipt의 목적은 “측정할 수
있는 자원과 못 측정하는 자원”을 숨기지 않고 global orchestrator 입력의
지원 상태를 고정하는 것이다.

v103은 v102의 synchronized co-job failure propagation 위에 이 capability
receipt만 추가한 immutable source boundary다. workload, seven-arm order,
controller parameter, LMCache/NIXL backend, NCCL transport와 GPU/CPU shape는
변경하지 않았다.

| 항목 | v103 값 |
|---|---|
| snapshot root | `results/tempo_go_c5_source_snapshot_v103_capability_receipt/source` |
| candidate | `tempo-go-cross-layer-v103-capability-receipt` |
| revision | `v103-p1-four-node-capability-receipt-before-cojob` |
| contract SHA | `0d420caf3e016f04201849e57f90d86f092e39bd1b450e1701180918236b3d8d` |
| fingerprint | `deb8ee45251c9783ab5d8f37ce97f3404218b5f087421d949e5b0cf8ed32bc0e` |
| source tree SHA | `d9cc0d2be1aa78d5c12f0b76a1ac630d718bb7846aaeeec114f97b873f860f42` |
| claim gate | `performance_claim_allowed=false` |

v103 contract verify, shell syntax, capability static test와 synchronized
co-job test는 `13 passed`다. 다음 실행 wrapper는
`eval/sota_4node/run_tempo_go_v103_moderate_in_allocation.sh`이며, 정확히
4-node `gpu_interactive`/`interactive`, 16 GPU, `CPUs/Task=128`,
`Network=job_vni`인 foreground allocation에서만 실행한다. 현재 두
`gpu_interactive` slot이 사용 중이므로 새 allocation이 실제로 확보되기
전에는 재시도하지 않는다.

## 66. P4 hierarchy benchmark의 artificial partition 비용 제거

기존 `results/tempo_go_hierarchy_scale_20260823.json`은 pair-agent frontier
시간을 측정할 때 각 pair마다 전체 request population을 다시 훑어 local
candidate를 추출했다. 실제 distributed deployment에서는 pair agent가
candidate ownership을 가진 상태로 시작하므로, 이 구현은 frontier ranking에
없는 인공적인 `O(pair_count * candidate_count)` 비용을 bounded path에
부과했다. 기존 receipt의 payload/omission/fingerprint evidence는 보존하되,
그 bounded-total timing은 current scale gate의 근거로 사용하지 않는다.

benchmark를 pre-partitioned pair ownership으로 수정하고 같은 population,
같은 64-shard/2-pair-per-shard bound, 같은 15회 반복으로 다시 측정했다.
새 receipt는
`results/tempo_go_hierarchy_scale_20260823_prepartitioned_r15.json`이다.
1,024 pair에서 다음을 얻었다.

| path | p50 | p99 | scope |
|---|---:|---:|---|
| full candidate global scan | 136.033 ms | 258.430 ms | 모든 2,048 candidate가 한 reducer로 전달 |
| pair-agent local frontier | 31.388 ms | 32.778 ms | ownership partition 이후 local ranking/receipt |
| bounded global fan-in | 71.083 ms | 165.554 ms | 256 forwarded candidate |
| bounded end-to-end control path | 102.286 ms | 197.658 ms | local frontier + global fan-in |

bounded path는 global payload를 666,815 bytes에서 83,358 bytes로
87.499% 줄이고, 1,024 pair 중 896 pair를 omission receipt에 기록한다.
pre-partitioned 측정에서는 bounded total p50이 full scan p50보다 약
24.8% 낮다. 이는 **P4 CPU control-plane scale evidence**로 승격할 수
있지만, p99가 full p50보다 낮다는 뜻은 아니며 GPU/NCCL/LMCache/Slingshot
native performance나 vLLM goodput claim도 아니다. native에서는 distributed
agent wall-clock, wire bytes, reducer CPU, global decision p50/p99를 이와
동일한 분해로 수집한다.

수정 후 hierarchy/coordinator test는 `23 passed`다. 이 결과는 TEMPO의
전체 global orchestrator가 Perlmutter-scale fan-in에서 실제로 가져야 할
control overhead를 줄이는 방향의 구현 개선이며, component-only benchmark가
아니다. 다음 native gate는 v104 seven-arm coupled run에서 이 control path
receipt와 aggregate business utility를 함께 bind하는 것이다.

## 67. v104 global reducer optimization rebinding

§66의 pre-partitioned benchmark가 distributed ownership 경계를 올바르게
측정하도록 고친 뒤, 실제 `HierarchicalCandidateReducer.reduce()`에도
동일한 global path 최적화를 반영했다. 하나의 reduction 안에서 candidate
rank를 cache하고, pair를 shard별로 한 번 bucket하며, forwarded count를
forwarded list 재검색 없이 직접 누적한다. 이 변경은 route score, telemetry
identity, fairness, admission/action semantics를 바꾸지 않고 repeated
control-plane work만 줄인다.

v104 source snapshot은 v103의 native execution boundary와 이 reducer
optimization을 함께 고정한다.

| 항목 | v104 값 |
|---|---|
| snapshot root | `results/tempo_go_c5_source_snapshot_v104_hierarchy_reducer_cache/source` |
| candidate | `tempo-go-cross-layer-v104-hierarchy-reducer-cache` |
| revision | `v104-cached-rank-and-shard-bucket-global-reducer` |
| contract SHA | `4ddd901df87dc7107fac1ee4d7c76f8734d1651eb4502b8a12563d708b64ff6d` |
| fingerprint | `9378202301943156b1eaf500b72b974145b375948bd22fc686e611a7eea2a170` |
| source tree SHA | `eb2ebcd3eda9d8dbcd3e0a339b9e006aa14656626c2ff542f7767b2245fd0d86` |
| claim gate | `performance_claim_allowed=false` |

현재 reducer source의 1,024-pair/15-repeat control-plane receipt는
`results/tempo_go_hierarchy_scale_20260823_reducer_cached_r15.json`이다.
full scan p50 `56.260 ms`, bounded global fan-in p50 `56.833 ms`, bounded
end-to-end control path p50 `87.271 ms`, bounded control-path p99 `96.655 ms`를
기록했다. global payload는 여전히 87.499% 감소하고 896 pair omission receipt를
보존한다. full scan과 bounded path의 local CPU p50만으로 native 우위를
주장하지 않으며, 이 결과는 v104 native run에서 wire bytes와 actual global
decision latency를 함께 수집하기 위한 implementation/measurement boundary다.

v104 contract verify, shell syntax, capability/co-job static tests와 hierarchy
reducer tests는 `36 passed`다. 다음 native 실행은
`eval/sota_4node/run_tempo_go_v104_moderate_in_allocation.sh`에서만 수행하고,
그 결과가 나오기 전에는 v104 reducer 최적화를 performance win으로
승격하지 않는다.

## 68. 2026-08-23 current-state audit: 왜 아직 성능 gate를 못 넘었고 현재 실행 상태는 무엇인가

이 절이 현재 상태의 최우선 continuation이다. 목표는 바뀌지 않았다.
**TEMPO Elastic-PD를 actual vLLM P/D와 frozen official
`LMCacheConnectorV1:UCX` 경로에 통합하고, NCCL/Slingshot/Cassini,
LMCache/UCX, vLLM/GPU service와 tenant business state를 하나의
node→pair→shard→global 폐루프로 공동 제어해 strongest fixed 및
predictor보다 유의미하게 나은 whole-system utility를 만드는 것**이다.
component별 기능 개수나 isolated microbenchmark로 이 목표를 축소하지 않는다.

### 68.1 현재 판정

| 항목 | 현재 판정 | 근거/의미 |
|---|---|---|
| contention problem | `SUPPORTED` | C1/C2/C3 winner crossover, P_ONLY remote service knee, v24/v25/v37/v69/v99의 decoder·KV·NCCL/LMCache 공동 tail |
| global orchestrator 필요성 | `SUPPORTED` | local도 decoder GPU에서 무너지고 remote도 KV/receiver/fabric에서 무너지며 pair가 shared fabric을 공유하면 병목이 layer 사이로 이동함 |
| cross-layer mechanism integration | `SUPPORTED WITH BOUNDARY` | actual vLLM P/D request가 LMCache/NIXL, NCCL observer, Cassini vector, business admission과 hierarchy receipt를 통과함 |
| strongest fixed/predictor 대비 native win | **`NOT ACHIEVED`** | independent full-valid win이 없고 현재 performance claim은 false |
| current candidate | `v104 FROZEN; NATIVE CAMPAIGN NOT RUNNING` | immutable contract SHA/fingerprint verify 재통과, controller parameter와 seven-arm order는 frozen; current-source seven-arm result는 없음 |
| P4 scale | `CPU CONTROL-PLANE EVIDENCE ONLY` | 1,024 pair에서 global payload 87.499% 감소; native request goodput이나 fabric scale win은 아님 |

한 줄 결론은 다음과 같다.

> **문제와 global orchestration의 필요성에는 답이 있다. 아직 답이 없는 것은
> “현재 TEMPO policy가 그 필요성을 실제 offered-population speedup으로
> 바꾸는가”이며, 지금까지의 valid native 결과는 gate 미달 또는 실패다.**

### 68.2 valid 성능 영수증이 실제로 말하는 것

| run | 무엇이 valid했는가 | TEMPO와 strongest comparator | 정확한 판정 |
|---|---|---|---|
| v28 / allocation `57415765` | same-allocation seven arms, 276 rows, 10,000-block co-job, C5 end coverage | TEMPO `638.6` output tok/s vs queue-GPU `616.8` (`+3.53%`); TEMPO 136 complete/140 reject; app-global `647.6`; TEMPO p99 `12,616 ms` vs app-global `11,712 ms` | required fixed-goodput `+5%`와 app-only incremental gate 실패 |
| v30 / allocation `57423440` | 7/7 terminal-valid arms, same population, co-job correctness/coverage | TEMPO `533.01` vs queue-GPU `640.99` (`-16.85%`), app-global `618.18` (`-13.78%`); TEMPO 134 complete/142 reject | pair activation이 shared NCCL/LMCache/fabric capacity를 만들지 못함 |
| v69 / allocation `57443205` | five fixed/network arms와 co-job은 valid | queue-GPU `659.605` tok/s; TEMPO 173 complete/57 reject/46 endpoint failures, completed-subset `606.353` | global arm terminal-invalid; completed-only 비교 금지 |
| v97 very-light | full seven-arm descriptive slice | remote와 app-global이 일부 지표에서 강했고 TEMPO는 reject 50 | load regime에 따라 remote 가치가 바뀜; frozen validation claim 아님 |
| v99 hot / allocation `57473916` | native `job_vni`, AWS Libfabric/CXI/GDRDMA, fixed five arms, observer sequence 904 | remote `615.439`, network-only `838.935`; app-global/TEMPO 미시작 | NIXL source timeout 뒤 rank-divergence/NCCL barrier cascade; cross-arm performance-invalid |
| v104 | contract와 P4 reducer만 검증 | native seven-arm result 없음 | utility 판단 전 |

따라서 “remote는 항상 나쁘다”가 결론이 아니다. 실제로 remote는 여러
allocation/state에서 local보다 빨랐다. 문제는 local, remote, queue-GPU와
network-only의 승자가 load와 shared externality에 따라 바뀌며, 현재 TEMPO가
그 이동을 reject 없이 안정적으로 따라가지 못했다는 것이다.

### 68.3 아직 개선을 못 만든 직접 원인

1. **admission이 physical service capacity를 만들지 못했다.** 이전 global
   policy는 overload를 queue/reject로 분류했지만 decoder completion,
   LMCache receiver drain과 shared NCCL/fabric capacity를 늘리지 못했다.
   completed subset의 낮은 latency는 offered population의 개선이 아니다.
2. **global debt를 endpoint queue로 넘겼다.** queue lease와 service-lane
   handshake를 여러 번 보정했지만, endpoint가 deadline 안에 실제로 drain할
   completion credit 없이 request를 넘기면 queue timeout/503으로 병목 이름만
   바뀌었다.
3. **logical pair scaling이 physical independence를 보장하지 않았다.** 두
   pair가 같은 communicator epoch, LMCache transfer path 또는 Slingshot
   externality를 공유하면 spare pair activation은 같은 병목을 복제한다.
   v30이 이 실패를 terminal-valid native result로 보여줬다.
4. **cross-layer signal은 연결됐지만 marginal utility calibration이 부족하다.**
   NCCL/LMCache/Cassini state가 action을 바꾸는 mechanism은 구현됐으나,
   `action → completion capacity → tenant SLO work`의 실제 native delta를
   global price/admission에 충분히 반영하지 못했다.
5. **current-source full comparison을 끝까지 닫지 못했다.** source drift,
   co-job coverage/readiness, Slurm step/VNI shape, LMCache cache ownership과
   v99 NIXL timeout cascade 때문에 후반 app-global/TEMPO arm이 시작되지 않거나
   terminal-invalid가 됐다. 이 execution work는 재현성을 높였지만 성능을
   자동으로 개선하지는 않는다.
6. **v104는 service policy 개선판이 아니다.** v104의 controller parameter는
   v103과 동일하고 reducer의 repeated CPU work만 줄였다. 이는 Perlmutter-scale
   control-plane에 필요하지만, v30의 reject/capacity 문제를 그 자체로
   뒤집을 수 없다. v104 native run의 목적은 현재 전체 scheme의 물리적
   utility를 처음부터 끝까지 측정하는 것이다.

즉 지금까지 시간이 오래 걸린 이유는 “병목이 없어서”가 아니라, execution
failure와 policy utility failure를 같은 숫자로 섞지 않으면서 actual
vLLM/LMCache/NCCL/Slingshot 경로를 재현 가능하게 닫는 작업과, shared capacity를
만들지 못한 global policy의 실패가 함께 누적됐기 때문이다. 이제 launcher
오류를 더 수집하는 것이 목표가 아니며, v104 complete population이 없으면
다음 policy 판단도 하지 않는다.

### 68.4 현재 Slurm/native 상태

2026-08-23에 요청 범위를 실행 승인으로 잘못 해석해 TEMPO 전용 allocation을
다음 exact shape로 한 번 요청했다. job `57486244`는
`m1248_g/gpu_interactive`, 4 nodes, 4-hour limit으로 시작됐다.

```text
/usr/bin/salloc -A m1248_g -C gpu -q interactive -t 04:00:00 -N 4
  --ntasks-per-node=1 --cpus-per-task=128 --gpus-per-node=4
  --network=job_vni -J tempo_v104_int
```

이 allocation에서 allocation preflight와 네 node capability probe는
`passed`였다. `Network=job_vni`, 4 nodes/512 CPUs/16 A100 GPUs, node당 네
Cassini NIC telemetry root, CUDA/NCCL support와 동일 topology fingerprint를
확인했고, co-job도 AWS Libfabric/CXI native transport에서 8 rank
`correctness_met=true`, active observer sequence 796까지 도달했다. 그러나
**C5 seven-arm의 어느 arm도 시작하지 않았고 request/result도 생성되지
않았다.** 따라서 이것은 capability/readiness 영수증일 뿐 성능 영수증이
아니다.

사용자가 설명만 요청한 것이라고 범위를 정정한 즉시 co-job을 중단하고
allocation을 해제했다. `sacct`의 최종 상태는 job `57486244`
`CANCELLED by 76516`, elapsed `00:23:08`, end
`2026-08-23T15:03:34`다. **현재 이 문서에 따른 TEMPO native campaign이나
TEMPO GPU allocation은 실행 중이지 않다.**

계정 전체 queue에 보이는 `57484098 cs_hold_int`와 2026-08-23 14:31 이후
생긴 `ol_*`/`mx_*` pending jobs는 account `m5320_g`, workdir
`/pscratch/sd/s/sgkim/kcj/Cascade-kcj`의 별도 workflow다. 이 v104 요청이
만든 job이 아니며 TEMPO evidence에 섞지 않는다. TEMPO는 `57486244`만
사용했으며 다른 allocation에 attach하거나 step을 걸지 않았다.

### 68.5 향후 명시적 실행 요청이 있을 때의 순서와 hard gate

현재 실행 승인은 없다. 사용자가 이후 native campaign을 명시적으로 요청할
때에만 다음 순서를 적용하며, 이를 건너뛰거나 component microbenchmark로
대체하지 않는다.

1. **P0 — 완료:** safety/goal 문서 전체, v104 source snapshot, contract SHA,
   workload와 seven-arm order를 bounded audit했다. v104 contract verify는
   현재 workspace에서 다시 통과했다.
2. **P1 — allocation capability:** 새로 명시 승인된 foreground allocation
   안에서만 exact `Network=job_vni`, 4 nodes/512 CPUs/16 GPUs/128 CPUs-task, native
   NCCL AWS Libfabric/CXI, official LMCache/NIXL/UCX, four-node Cassini/GPU
   support receipt와 clean live-step topology를 확인한다. 하나라도 실패하면
   performance run을 시작하지 않는다.
3. **P2 — causal co-load readiness:** sustained-moderate co-job을 4 MiB/rank,
   8 token iterations, 1 MiB foreground, 0.25 s cadence로 실행해 8-rank
   correctness, synchronized failure propagation, active observer와 C5 end
   coverage를 확인한다. hot v99 stress로 profile을 바꾸지 않는다.
4. **P5 discovery — 전체를 한 번에 닫음:** 같은 frozen server/cache/co-load와
   276-row offered population에서 `local → remote → predictor → queue_gpu →
   network_request_only → app_global_only → tempo` 일곱 arm을 실행한다.
   source/profile/workload/analyzer를 중간에 바꾸지 않는다.
5. **correctness gate:** 모든 request가 complete/reject/fail exact terminal,
   stream/output/route/cache provenance 100%, hidden fallback/same-ID retry 0,
   resource leak 0, co-job correctness/coverage true, mixed/stale identity 0이어야
   성능을 계산한다.
6. **utility gate:** reject/failure를 denominator에서 빼지 않고 strongest fixed
   대비 median `10%`, predictor 대비 `5%`, output-token/request goodput `5%`,
   paired win `75%` overall/`60%` per group, p99 regression `≤5%`를 적용한다.
   full TEMPO는 app-global/network-only 대비 goodput/SLO-goodput `5%` 또는
   p99 `10%` incremental gain도 보여야 한다.
7. **P6/P7:** discovery가 gate를 통과하면 code/profile/workload/analyzer를
   그대로 freeze하고 새 승인 allocation에서 independent validation을 한 번
   수행한다. 결과를 본 뒤 tuning하지 않는다.

### 68.6 v104가 valid하게 졌을 때의 다음 policy 결정

v104가 full terminal-valid인데도 gate를 실패하면 또 다른 queue timeout,
route coefficient나 reservation fraction 조정으로 가지 않는다. native receipt를
사용해 다음 하나의 global mechanism을 설계하거나 해당 hypothesis를
reproducible negative로 닫는다.

- decoder token service, local prefill, remote KV/semantic completion,
  NCCL/fabric shared group의 **measured completion capacity**를 epoch별로 분리한다.
- request action마다 필요한 multi-resource demand와 tenant value/deadline을
  사용해 **marginal completed SLO work**를 계산한다.
- pair activation은 새 pair가 independent completion capacity를 실제로
  추가할 때만 허용하고, shared bottleneck이면 concurrency/stagger/defer로
  제어한다.
- predicted completion이 deadline 안에 없으면 queue lease를 accepted work로
  만들지 않고 business-aware explicit defer/reject로 닫는다.
- low-value work를 무조건 버리지 않는 work-conserving borrowing과 tenant
  minimum service를 유지하며 aggregate output-token goodput과 p99를 함께
  최적화한다.

이 후속 mechanism은 v104 native causal receipt가 요구할 때만 새 candidate로
freeze한다. 현재 우선순위는 새 component 추가가 아니라 **v104 전체 7-arm
native population을 성공 또는 정확한 failure receipt로 닫는 것**이다.

### 68.7 안전·운영 경계

- 현재 TEMPO GPU workload와 allocation은 없다. 향후 substantial
  GPU/vLLM/LMCache/NCCL workload는 사용자가 명시 승인한 exact
  `gpu_interactive` allocation 안에서만 실행한다.
- non-interactive batch job을 TEMPO가 submit하지 않는다.
- 다른 workflow의 allocation/job/step을 attach, reuse, cancel하거나 hijack하지
  않는다.
- root, UDI, container, `udiRoot.conf`, Slingshot privileged configuration과
  system ownership을 건드리지 않는다.
- source snapshot을 실행 중 수정하지 않고 result root를 overwrite하지 않는다.
- launch/capability 실패를 performance negative로 쓰지 않고, valid complete
  population이 있을 때만 utility를 판정한다.

## 69. v105 completion-liveness native seven-arm 결과와 다음 global mechanism

이 절은 2026-08-23 최신 current continuation이며 §68의 “v104 미실행” 상태를
supersede한다. §18의 원래 mission, safety boundary, same-population seven-arm
순서와 hard utility gate는 바꾸지 않는다.

### 69.1 실행 provenance와 Perlmutter 경계

사용자가 명시 승인한 4-node/4-hour `gpu_interactive` allocation 하나만
사용했다.

```text
job                 57488718 (tempo_v105_int)
account / QOS        m1248_g / gpu_interactive
nodes / GPUs         4 / 16 A100
CPUs                 512, CPUs/Task=128
allocation network   Network=job_vni
elapsed / final      00:47:29 / COMPLETED 0:0
nodes                nid[001157,001160-001161,001164]
```

NERSC 문서대로 `salloc` foreground shell 안에서만 `srun` step을 시작했고,
각 GPU step에 GPU/CPU shape와 core binding을 명시했다. NERSC는 interactive
GPU allocation에 GPU project/constraint와 `srun`의 explicit GPU request를
요구한다([Interactive documentation](https://docs.nersc.gov/jobs/interactive/),
[Perlmutter running jobs](https://docs.nersc.gov/systems/perlmutter/running-jobs/)).
batch/non-interactive job을 submit하지 않았고 다른 allocation/job/step에
attach하거나 취소하지 않았다. root, UDI, container, `udiRoot.conf`, privileged
NIC/Slingshot 설정은 사용하지 않았다.

첫 capability attempt는 성능 arm이나 co-job 전에 중단됐다. 네 node 중 세
node receipt는 정상 생성됐지만 `nid001164`의 PyTorch/NCCL cold import가 다른
task보다 5초 늦었고, probe의 `srun --wait=5`가 그 task를 SIGKILL해 step
`57488718.1`이 `exit 137`, `CANCELLED by 0`이 됐다. 여기서 `by 0`은 Slurm
daemon이 남은 task를 정리한 accounting이며 사용자가 root 명령을 실행한
것이 아니다. Slurm의 `--wait`는 첫 task 종료 뒤 남은 task를 종료하기까지의
시간이다([official `srun` documentation](https://slurm.schedmd.com/srun.html)).
따라서 capability import에만 60초를 주고 첫 failure root는 삭제하지 않은 채
`attempt2`로 분리했다. frozen source snapshot/controller/profile은 수정하지
않았다.

- 첫 false-negative receipt:
  `results/tempo_go_cross_layer_cojob_57488718/perlmutter-native-step-preflight/cross-layer-capability/receipt.json`
- valid result root:
  `results/tempo_go_cross_layer_native_v105_completion_liveness_57488718_attempt2`
- valid co-job root:
  `results/tempo_go_cross_layer_cojob_57488718_attempt2`
- frozen contract:
  `results/tempo_go_c5_source_snapshot_v105_completion_liveness/native_run_contract.json`
- contract SHA:
  `6606e3da1b21074f4cb8fc6bb9b6663cfde6963c255b7174a62f1f4bf6c8d165`
- contract fingerprint:
  `c4c146200d796268d92613f15cd9e776b8ad86f803ccca9eb5e178884dbed895`
- source tree SHA:
  `d83d4319e03d9c514e2603629fa422397c07d3aa0f458a5262d8744e9fd41d27`
- final analysis SHA:
  `a9595c6ea3dfb6c559787914ee778580ba00171896f69059639d3dc740f3590a`
- full TEMPO raw SHA:
  `e37e3047fdfd8690511f0faffb12a72b6f67d8a4a1f6858f792daaafc406df67`
- co-job result SHA:
  `60c3ac066a341e2ab1d3d50f580664db51710f956b6207082b6dec143bee6fd0`

네 node capability receipt는 각각 A100 4개, CUDA 12.9, PyTorch
`2.11.0+cu129`, NCCL availability, node당 네 Cassini NIC와 core/optional
telemetry-counter support를 확인했다. 이후 local, remote, predictor,
queue-GPU, network-request-only, app-global-only, full TEMPO의 일곱 arm은 모두
실제 vLLM/LMCache endpoint를 시작해 같은 276-row population을 처리했다.

### 69.2 seven-arm native 수치와 exact 판정

아래 E2E/TTFT는 **완료 request subset**의 지표다. reject/fail은 별도 terminal
열에 그대로 남으며 latency denominator에서 사라졌다는 이유로 win으로
해석하지 않는다.

| arm | complete / fail / reject | E2E p50 / p95 (ms) | TTFT p50 / p95 (ms) | request goodput (/s) | output tok/s | route local / remote |
|---|---:|---:|---:|---:|---:|---:|
| local | 276 / 0 / 0 | 11,668.62 / 34,354.27 | 7,760.02 / 28,402.64 | 4.292 | 531.40 | 276 / 0 |
| remote | 276 / 0 / 0 | 9,461.44 / 27,436.75 | 3,077.26 / 23,668.55 | 5.205 | 644.49 | 0 / 276 |
| predictor | 276 / 0 / 0 | 11,679.31 / 32,795.80 | 7,756.36 / 26,217.90 | 4.317 | 534.59 | 264 / 12 |
| queue-GPU | 276 / 0 / 0 | 11,001.81 / 27,352.01 | 4,087.46 / 20,988.87 | 5.456 | 675.65 | 69 / 207 |
| network-request-only | 276 / 0 / 0 | 11,576.91 / 30,059.08 | 7,840.26 / 24,343.03 | 4.596 | 569.12 | 275 / 1 |
| app-global-only | 267 / 1 / 8 | 10,268.10 / 16,298.04 | 3,651.61 / 9,098.34 | 7.192 | 872.36 | 71 / 196 |
| full TEMPO v105 | 252 / 5 / 19 | 9,729.86 / 15,841.71 | 3,660.38 / 9,630.28 | 6.928 | 842.80 | 87 / 165 |

강한 fixed policy는 이 allocation에서 **remote**다. full TEMPO는 remote 대비
completed-subset E2E p50이 2.84% 느리고 p95는 42.26% 낮으며, request/output
goodput은 각각 33.11%/30.77% 높다. predictor 대비 p50/p95는
16.69%/51.70% 낮고 request goodput은 60.47% 높다. 그러나 full TEMPO는
24/276, 즉 8.70%를 완료하지 못했으므로 이 completed-subset/shorter-window
수치는 positive utility claim이 아니다. app-global-only보다 p50/p95는
5.24%/2.80% 낮지만 complete가 15개 적고 request goodput도 3.68% 낮다.
따라서 full cross-layer incremental gate도 실패다.

공식 machine analysis의 structural gate는 seven arms present, same request
count/workload SHA, frozen contract valid, native 4-node/16-GPU UCX,
endpoint-completion receipt와 scheduler observation을 확인했다. 그러나
`performance_claim_allowed=false`를 유지한다.

### 69.3 실제 contention과 병목 위치

동일 allocation의 opt-in co-job은 2 nodes/8 ranks에서 official LMCache
`NixlChannel`/UCX background transfer와 real NCCL CUDA collective를 동시에
실행했다. NCCL transport receipt는 `AWS Libfabric`, CXI/OFI, HSN과 GDR level
`PHB`를 확인했다. C5 시작 전에 ready였고 C5 종료 뒤까지 살아 있었으며,
실제 4,582 blocks에서 full bytes completion/verification와
`overall_correctness_met=true`를 만족했다.

- LMCache/NIXL background completion: p50 `27.809 ms`, p99 `2,219.128 ms`
- NCCL token tail: p50 `0.313 ms`, p99 `11.659 ms`
- full TEMPO decision의 LMCache transfer pressure: 276개 중 273개가 `1.0`
  saturation
- NCCL collective pressure: p50 `0.694`, p95/max `0.932`
- Cassini pause pressure: 275개 중 8개만 미세 양수, 최대
  `9.864e-8`; ECN/retry/timeout과 TX pause는 0

따라서 이번 allocation에서 관측된 주 병목은 물리 Slingshot link pause/ECN이
아니라 LMCache transfer completion, NCCL/GPU scheduling tail, endpoint
completion과 global debt accounting이다. 이것은 Slingshot telemetry가
불필요하다는 뜻이 아니다. Cassini vector가 실제로 “link-level saturation은
아님”을 구분해 줬고, LMCache/NCCL signal이 상위 shared-resource 병목을
보였다. 다른 placement/외부 co-tenant에서는 Cassini 병목으로 이동할 수
있으므로 signal을 제거하거나 scalar 하나로 축소하지 않는다.

remote도 항상 나쁘지 않다. 이번 slice에서 all-remote가 local/predictor보다
빠른 strongest fixed였고 276개를 모두 완료했다. LMCache co-job도 오류 없이
끝났지만 p99가 2.2초까지 튀었다. 즉 LMCache는 “항상 실패”가 아니라
contention에서 long-tail shared service가 되며, local decoder와 remote
transfer 사이 winner가 state에 따라 바뀐다는 원래 motivation이 다시
확인됐다.

### 69.4 v105가 v104 liveness를 개선했지만 gate를 못 닫은 이유

v104 single-arm과 같은 276-row workload를 비교하면 v105의 mechanism 효과는
분명하다.

| 항목 | v104 | v105 | 변화 |
|---|---:|---:|---:|
| complete | 165 | 252 | +87 |
| global reject | 76 | 19 | -57 |
| endpoint queue timeout/fail | 35 | 5 | -30 |
| request goodput (/s) | 5.352 | 6.928 | +29.5% |

v104의 failure-free stale-SKIP deadlock을 겨냥한 completion-liveness probe는
실제로 두 request를 local recovery probe로 commit했고 둘 다 first response를
완료했다. 따라서 mechanism은 dead code가 아니다. 그러나 다음 병목으로
이동했다.

1. full TEMPO에서 liveness probe commit은 2건뿐인데
   `completion_liveness_probe_inflight`가 29번 다른 candidate를 막았다.
2. global reject 19건은 admission queue timeout 13, telemetry stale 4,
   telemetry refresh timeout 2다. telemetry freshness가 일시적으로 어긋난
   여섯 request는 work-conserving safe fallback 없이 바로 503이 됐다.
3. endpoint fail 5건은 모두 global queue lease를 accept한 뒤
   `endpoint_bounded_queue_lease_timeout`이 된 request다. 두 remote request는
   C1 decoder-hot, 세 local request는 C3 both-hot였다.
4. `soft_shadow_price_v2`는 shared request count만 hard cap으로 유지하고
   KV/semantic target을 soft price로 만든다. v2는 per-pair remote semantic
   guard까지 해제한다. 그 결과 decision-time `remote_kv_bytes`와
   `remote_semantic_ops` debt는 각각 configured limit의 최대 4배였고,
   276개 중 170개 decision에서 limit를 넘었다. 일반 queue lease score에는
   liveness probe와 달리 전체 queue-wave completion delay가 들어가지 않아
   8초 lease 안에 drain할 수 없는 work도 commit됐다.
5. full TEMPO terminal loss는 C0 `1`, C1 `4`, C2-KV `2`, C2-remote `1`,
   C3 `11`, recovery `5`다. 특히 recovery는 6개 중 1개만 complete했다.
6. tenant별 complete는 background `227/240`, batch `10/12`, latency
   `10/12`, interactive `5/12`다. weighted dominant-service bookkeeping이
   있어도 interactive minimum service를 보장하지 못했으므로 business
   fairness gate도 실패다.

즉 v105는 기존 deadlock을 크게 줄였지만 **completion capacity로 보증되지
않은 soft debt와 단일-probe 직렬화, telemetry hard dependency, tenant-unaware
recovery**가 결합했다. 문제는 signal 부족이 아니라 signal을 offered
population completion으로 바꾸는 global control law다.

### 69.5 STOP/GO와 다음 global orchestrator scheme

현재 판정은 다음과 같다.

- `GO`: realistic contention problem, moving bottleneck, cross-layer state
  plane, actual joint decision path와 global orchestrator 필요성
- `NEGATIVE`: v105 completion-liveness endpoint queue v2 candidate의 native
  offered-population utility
- `STOP`: v104/v105 blind retry, queue timeout/pressure coefficient/credit
  숫자만 사후 조정, completed-only latency claim
- `OPEN`: Perlmutter-scale completion-rate-backed business/fabric global
  orchestrator와 independent native validation

다음 candidate는 component를 더 붙이는 버전이 아니라 하나의
**completion-rate-backed multi-resource global admission/lease controller**로
설계한다.

1. endpoint/pair/shared-group별로 decoder first-response/output-token drain,
   local prefill, remote KV bytes/semantic completion과 NCCL/fabric service의
   measured completion rate·residual·confidence를 같은 epoch ledger에 둔다.
2. 모든 ordinary queue lease와 recovery probe에 대해 `work ahead / measured
   drain rate + service prior + uncertainty`로 finish time을 계산한다.
   liveness probe만 queue-wave delay를 보는 현재 비대칭을 없앤다.
3. soft shadow price는 hard capacity를 대체하지 않는다. debt ceiling은 고정
   배수가 아니라 deadline 안에 실제 drain 가능한 completion credit으로
   정하며, route/pair/shared group에 exactly-once reserve/release한다.
4. telemetry stale/refresh timeout은 transport failure와 구분한다. identity가
   맞고 최근 health/completion receipt가 있으면 bounded last-known-safe
   service 또는 tenant minimum-service lane으로 work-conserving하게 진행하고,
   uncertainty가 실제 deadline을 넘을 때만 explicit defer/reject한다.
5. recovery probe는 pair×route 하나의 background request가 독점하지 않게
   global EDF/WDRF와 tenant minimum-service 아래 scheduling한다. latency와
   interactive tenant가 12개 중 5개만 complete하는 결과를 허용하지 않는다.
6. pair activation은 logical queue가 아니라 새 pair/NIC/path가 incremental
   completion capacity를 추가한다는 Cassini/NCCL/LMCache evidence가 있을 때만
   한다. shared bottleneck이면 concurrency/debt/stagger를 공동 제어한다.
7. v104/v105 raw replay에서 terminal conservation, no leak, no stale-identity,
   276 offered request에 대한 bounded decision과 minimum service를 먼저
   machine-check한다. CPU replay는 control-plane promotion gate일 뿐 성능
   claim으로 사용하지 않는다.
8. 이 mechanism과 profile을 결과 보기 전에 새 immutable candidate로 freeze한
   뒤에만, 사용자가 다시 명시 승인한 4-node `gpu_interactive` allocation에서
   동일 seven-arm discovery를 한 번 실행한다. gate를 통과하면 별도 승인
   allocation에서 independent validation을 수행한다.

allocation `57488718`은 분석용 compute 작업까지 끝낸 뒤 foreground shell을
정상 종료해 `COMPLETED 0:0`으로 해제했다. 현재 TEMPO GPU allocation이나
실행 중인 TEMPO step은 없다.

## 70. v106/v107 p07 결과, workload 판정 오류와 C6 P×D mesh reset

이 절은 2026-08-23의 최신 continuation이며 §69의 “completion-rate-backed
fixed pair×route controller가 다음 candidate”라는 방향을 더 구체화한다. §18의
mission, safety boundary, same-population comparison과 final hard gate는 바꾸지
않는다. 다만 v106/v107을 분석한 결과, 현재 276-row short-slice+p07 campaign을
더 강하게 반복하는 것은 original mission을 검증하지 못하므로 중단한다.

### 70.1 v107 provenance와 exact 결과

사용자가 명시 승인한 하나의 4-node/4-hour `gpu_interactive` allocation만
사용했다.

```text
job                 57490824 (tempo_v106_int)
nodes / GPUs         4 / 16 A100
CPUs                 512, CPUs/Task=128
allocation network   Network=job_vni
nodes                nid[001101,001104-001105,001108]
```

batch/non-interactive job, root, UDI/container, `udiRoot.conf`, privileged
Slingshot/NIC 설정은 사용하지 않았다. v107 immutable identities는 다음과
같다.

- result root:
  `results/tempo_go_cxi_native_v107_credit_refill_57490824`
- p07 co-job root:
  `results/tempo_go_cxi_background_v107_57490824`
- frozen contract:
  `results/tempo_go_c5_source_snapshot_v107_cxi_credit_refill/native_run_contract.json`
- contract SHA:
  `de01e9907226c699b2a8a09d6bd6ec6d6d02fe7d2d4d3bf1c48c1e8d9ce28602`
- contract fingerprint:
  `bb1134d4d6d811ae368d673a4b09947e5a1a0b77169b0e9a9bb1326de8411bba`
- analysis SHA:
  `585c2f2a24c4dae76014ae2c3bdf74854816d4da72ffb61b179a810de814517d`
- full TEMPO raw SHA:
  `0a8752bbbc0dee3c4bed45eee1874e6a754218efd3f898077af3dc79469696be`
- p07 binding SHA:
  `2557f5dc51b60382344bcbdfa06159e7c9d4af4f941d294b20319f3c45dcaaa0`

아래 latency는 completed subset이다. reject는 별도 terminal로 남기고 성능
이득으로 계산하지 않는다.

| arm | complete / reject / fail | E2E mean / p50 / p99 (ms) | request goodput (/s) | route local / remote |
|---|---:|---:|---:|---:|
| local | 276 / 0 / 0 | 9,620.32 / 10,147.35 / 16,562.71 | 6.967 | 276 / 0 |
| remote | 276 / 0 / 0 | 7,941.99 / 8,523.70 / 12,962.36 | 7.939 | 0 / 276 |
| predictor | 276 / 0 / 0 | 9,695.07 / 10,117.51 / 16,756.62 | 6.892 | 264 / 12 |
| queue-GPU | 276 / 0 / 0 | 7,971.39 / 7,756.37 / 15,332.79 | 7.807 | 65 / 211 |
| network-request-only | 276 / 0 / 0 | 9,703.12 / 10,143.33 / 16,721.00 | 6.993 | 264 / 12 |
| app-global-only | 259 / 17 / 0 | 7,407.54 / 8,182.82 / 12,602.50 | 8.310 | 115 / 144 |
| full TEMPO v107 | 273 / 3 / 0 | 7,619.84 / 8,111.34 / 13,719.20 | 8.191 | 118 / 155 |

strongest all-complete fixed는 remote다. full TEMPO는 remote 대비 completed
E2E mean/p50을 4.06%/4.84% 줄였지만 p95/p99는 5.75%/5.84% 늘렸고 request
goodput은 3.18%, output-token goodput은 1.97% 늘리는 데 그쳤다. 3개 request도
reject했으므로 §18의 fixed 10%, goodput 5%, tail non-regression gate를
실패한다. app-global-only보다 14개를 더 완료한 것은 개선이지만, completed
mean과 raw goodput은 오히려 app-only가 높아 full cross-layer incremental
gate도 통과하지 못한다.

v106의 249 complete/27 reject에서 v107의 273 complete/3 reject로 바뀐 것은
`completion_credit_endpoint_queue_v3`에 endpoint remote semantic guard를 두
번 적용하던 버그를 제거한 결과다. 이 causal fix는 유효하다. 그러나 이것은
controller의 큰 utility win이 아니라 terminal conservation/liveness 수정이다.

### 70.2 p07이 강한 counter를 만들었지만 큰 application opportunity를 만들지 못한 이유

p07은 16 MiB, inflight 8, 16 MPI ranks, 3P→1D(node3) incast를 full campaign
동안 유지했다. 1,460.75초 동안 node3 ingress 346.70 Gb/s, correctness true를
기록했고 MPICH summary에는 146 network timeout, TC1 RX pause와 최대 약 38
posted-blocked cycles/packet이 남았다. 이것은 **p07 MPI receiver/Cassini
endpoint가 심하게 압박됐다**는 유효한 mechanism receipt다.

그러나 independent LMCache victim의 큰 slowdown은 입증하지 못했다.

1. p07은 node3, 즉 pair1 decoder 하나만 계속 때렸다. pair0 remote edge는
   같은 강도의 victim이 아니며 fixed remote frontend는 138/138로 두 pair를
   분산했다.
2. v107 all-remote에서 pair1 mean E2E는 pair0보다 7.34% 느렸지만, TTFT는
   오히려 pair1이 낮았다. v106에서는 pair1 E2E가 pair0보다 약간 낮았다.
   즉 현재 차이는 repeat-stable victim degradation이 아니다.
3. v536/v538 same-allocation causal 비교에서도 synthetic CXI는 no-background
   대비 local median을 약 30 ms, remote를 약 181 ms만 악화시켰다. remote가
   더 민감한 것은 맞지만 50%급 service collapse는 아니었다.
4. p07 MPI endpoint와 official LMCache NIXL/UCX transfer는 같은 node/NIC의
   일부 자원을 공유해도 endpoint, queue, semantic completion과 burst timing이
   같지 않다. aggressor 자체 timeout/counter를 victim slowdown으로 대체할 수
   없다.
5. MRC companion evaluation처럼 congestion 실험은 aggressor rate가 아니라
   동시에 실행한 독립 victim의 throughput/tail degradation으로
   qualification해야 한다. p07은 현재 그 gate를 닫지 못했다.

따라서 p07은 `fabric calibration / false-positive attribution ablation`으로
보존하고 headline workload와 controller promotion oracle에서는 제외한다.
346.7 Gb/s라는 숫자나 Cassini pause counter만으로 physical Slingshot fabric
bottleneck 또는 TEMPO opportunity를 주장하지 않는다.

### 70.3 short-slice 자체가 global controller의 기회를 없앤 구조적 문제

현재 276-row workload는 모든 request를 약 20.25초 안에 제공하고 각
`C0→C1→C2→C2_KV→C3→recovery` phase는 1.5초, cooldown은 0.25초다. 반면
v107 request E2E는 대략 8–16초이고 global admission wait도 수초다. 따라서
policy가 한 state를 관찰하고 completion feedback을 받아 다음 state에
대응하기 전에 여러 phase가 같은 endpoint queue에 섞인다. analyzer에는
phase별 표가 남지만 controller 관점에서는 거의 하나의 large burst다.

이는 `tempo/pd_contention_workload.py`가 원래 정의한 30초 phase와도 다르다.
원래 C1/C2/C3 workload는 actual route-pinned inference tenant로 opposite
crossover를 qualification한 뒤 controller를 실행하도록 설계됐다. campaign
시간과 startup failure를 줄이려고 만든 short-slice를 최종 workload처럼
사용하면서 다음 세 가지가 발생했다.

- stationary p07과 phase-mixed application burst 때문에 strongest fixed
  remote 하나가 대부분의 window에서 강해졌다.
- current `pair_index×{LOCAL,REMOTE}` action은 `P0→D0`, `P1→D1` remote edge를
  미리 고정한다. controller는 어느 P가 어느 D로 보낼지, receiver incast를
  어떻게 stagger할지 선택할 수 없다.
- mean E2E over completed requests를 중심으로 보면 overload admission,
  tenant minimum service와 recovery의 큰 차이가 희석되고 reject-heavy policy는
  오히려 짧은 measurement window로 raw goodput이 높아 보일 수 있다.

즉 v107의 1–5%는 “문제가 작다”는 결론이 아니라 **workload horizon, victim
coupling과 action space를 잘못 맞춘 결과**다. §5.4와 §8.4가 이미 synthetic
CXI는 attribution only이고 headline은 actual inference+official LMCache+real
NCCL이어야 한다고 명시했는데, v106/v107은 visible Cassini signal을 얻는 데
집중하면서 이 경계를 사실상 거슬렀다. 이 방향을 더 반복하지 않는다.

### 70.4 기존 연구와의 재대조: 큰 gain은 어느 regime에서 나오는가

- [MRC transport](https://arxiv.org/html/2606.18170v1)는 receiver가 packet
  in-flight와 semantic WriteIMM in-flight를 별도로 제한한다. TEMPO가 가져올
  핵심은 transport 복제가 아니라 receiver completion capacity를 상위
  scheduler credit으로 노출하는 원리다.
- [MRC/SRv6 companion](https://arxiv.org/html/2605.04333v1)은 7:1 incast와
  별도 victim flow를 동시에 실행해 victim이 평균 25%, 1초 구간 최대 75%
  떨어지는지를 측정한다. counter가 높은 것만으로 contention을 선언하지
  않는다.
- [Mooncake](https://arxiv.org/abs/2407.00079)는 23,000-request real trace를
  2× replay해 overload를 만들고 SLO-goodput/early rejection을 평가한다.
  16K–128K long-context와 real workload에서 50–525% throughput 또는 75%
  request-capacity 개선을 보고한다. 이는 짧은 prompt의 moderate mean E2E가
  아니라 long-context, prefix reuse, overload와 SLO를 결합한 regime다.
- [DistServe](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)는
  request rate를 sweep해 90% SLO attainment를 유지하는 최대 rate를 비교하고,
  KV communication pressure를 충분히 주기 위해 MHA 모델을 선택했다.
- [P/D-Serve](https://arxiv.org/abs/2408.08147)는 production-scale에서 P/D
  ratio, on-demand forwarding와 D2D transfer를 함께 조정해 E2E throughput,
  TTFT SLO와 D2D transfer에서 각각 60%, 42%, 46% 개선을 보고한다.

따라서 50%급 개선을 기대하는 것은 비현실적이지 않다. 다만 그것은
`strongest fixed`가 실제 overload state 일부에서 queue/SLO collapse를 겪고,
global policy가 다른 healthy completion path나 admission/stagger action으로
유효 work를 보존할 때의 **offered-population SLO-goodput, tail 또는 recovery**
지표에서 요구해야 한다. 그런 state가 없는 moderate workload에서 pooled mean
E2E 50%를 억지로 만들면 baseline을 고의로 망가뜨린 실험이 된다.

### 70.5 C6 scheme: business-aware receiver-credit P×D mesh orchestrator

다음 candidate는 `pair×route` threshold variant가 아니라 P와 D를 독립된
resource pool로 모델링하는 하나의 end-to-end scheme이다. 임시 연구명은
**TEMPO-GO C6 receiver-credit P×D mesh**다.

현재 candidate key:

```text
(pair_index, LOCAL | REMOTE)
REMOTE pair0 = P0→D0
REMOTE pair1 = P1→D1
```

C6 candidate key:

```text
LOCAL(D0), LOCAL(D1)
REMOTE(P0→D0), REMOTE(P0→D1), REMOTE(P1→D0), REMOTE(P1→D1)
QUEUE / DEFER / REJECT
```

구현은 다음을 하나의 transaction으로 묶는다.

1. `RouteCandidate`에 `prefill_index`, `decoder_index`, `edge_id`를 분리한다.
   local은 `D_i`의 local prefill이고 remote는 `P_i→D_j` immutable edge
   commit이다.
2. source P에는 prefill token-ms/completion-rate credit, destination D에는
   decoder active-sequence/output-token credit와 receiver KV-byte/semantic-op
   credit를 둔다. edge에는 measured transfer completion rate, residual,
   uncertainty와 Cassini/NCCL externality를 둔다.
3. LMCache receiver가 first-response/completion receipt로 credit을 반환한다.
   MRC의 receiver-advertised semantic in-flight 원리를 application-level
   admission에 적용하되 transport나 Slingshot을 privileged하게 변경하지
   않는다.
4. 한 request씩 greedy route만 고르지 않고 bounded admission epoch의 remote
   candidates를 bipartite matching한다. 같은 D로 KV burst가 몰리면 receiver
   credit 안에서 stagger하고, 다른 D/edge가 실제 fresh capacity를 더할 때만
   이동한다.
5. cache affinity가 P 선택을, decoder SLO/decode debt가 D 선택을, edge
   completion/Cassini/NCCL state가 transfer 시점과 concurrency를 결정한다.
   어느 하나의 scalar `fabric_pressure`로 합치지 않는다.
6. EDF/WDRF와 tenant value/minimum service를 matching objective에 넣는다.
   deadline 안에 complete할 수 없는 request만 prefill 전에 defer/reject하고,
   reject를 free speedup으로 계산하지 않는다.
7. failure/quarantine와 recovery는 P, D와 edge scope로 분리한다. 한 edge가
   실패해도 healthy edge로 새 request ID를 사용해 재결정하며 in-flight
   request의 hidden migration/recompute는 금지한다.

이 scheme의 novelty는 network-aware decoder selection 하나가 아니다.
vLLM decoder service, LMCache source/cache/receiver completion, NCCL/GPU shared
progress, per-NIC Cassini path state와 business utility를 P×D matching과
receiver credit이라는 동일한 global transaction으로 닫는 것이 연구 단위다.

### 70.6 C6 workload qualification gate

controller를 실행하기 전에 workload가 material opportunity를 갖는지 다음을
machine-check한다. 이 gate를 실패하면 threshold를 튜닝하지 않고 workload
또는 action scope를 수정한다.

#### Q0. capacity-normalized offered load

absolute req/s를 임의로 고르지 않는다. fixed local/remote와 실제 LMCache
receiver의 sustainable completion knee를 먼저 측정하고 각 state를
`0.8× normal`, `1.0× knee`, `1.2× overload`, `1.5× severe overload`로
정의한다. 기존 C1 22.4/s, remote 4.76/s, P_ONLY 9.7/s ceiling은 시작 prior일
뿐이며 새 geometry에서는 다시 qualification한다.

#### Q1. actual victim–aggressor matrix

다음 victim을 no-aggressor와 같은 request population으로 비교한다.

- official LMCache `P_i→D_j` transfer/TTFT victim
- decoder TPOT/output-token completion victim
- real NCCL collective completion victim

aggressor는 actual route-pinned vLLM inference, official LMCache NIXL/UCX와
real NCCL이어야 한다. synthetic p07은 보조 counter calibration만 담당한다.
다음 중 하나도 없으면 “network/fabric-hot” state로 승격하지 않는다.

- victim p50 completion/TTFT가 25% 이상 악화
- victim p99가 2× 이상 악화
- SLO attainment가 20 percentage points 이상 하락

Cassini pause/blocked counter만 상승하고 victim completion이 안정적이면
false-positive state로 보존한다.

#### Q2. opposite action opportunity

최소 두 phase에서 strongest fixed winner가 반대이고 각 winner margin이 15%
이상이어야 한다. 그중 하나의 overload phase는 margin 30% 이상이어야 한다.
또한 alternate D/edge 또는 local route가 실제 completion capacity를 20% 이상
회복해야 한다. 모든 edge가 같은 shared bottleneck이면 routing win을 기대하지
않고 global admission/stagger/fairness 문제로 명시한다.

#### Q3. phase와 geometry

각 phase는 `max(30 s, 3× measured p95 first-response horizon)` 이상 유지하고
pressure removal 뒤 최소 5초 recovery를 둔다. request phase label, future
arrival와 oracle winner는 policy input이 아니다.

primary sequence는 다음과 같다.

1. clean/normal
2. decoder compute hot, network clean: remote가 이겨야 함
3. one LMCache receiver/edge hot: local 또는 alternate D가 이겨야 함
4. asymmetric P/D/cache hotspot: P×D matching이 fixed pair보다 유리해야 함
5. both decoders + LMCache + real NCCL hot: admission/stagger/fairness가 필요
6. pressure removal/recovery

prompt/output은 먼저 이미 검증된 4094/6144와 16/128/256 tokens를 사용한다.
8K/16K는 model max length, KV memory와 2 GiB LMCache buffer capability를
preflight한 뒤 별도 long-context tier로 추가한다. prefix reuse는 session-like
temporal locality와 hot-key skew를 포함하고 arm별 cache namespace를 분리한다.

#### Q4. performance target

§18의 final fixed 10%, predictor 5%, goodput 5%, tail/correctness gate는 그대로
적용한다. 그 위에 C6 overload headline은 다음 중 하나를 목표로 한다.

- strongest fixed 대비 offered-population SLO-goodput **1.5× 이상**, reject/fail
  증가 없음, p99 non-regression
- 또는 p99 **30% 이상 감소**, recovery time **50% 이상 감소**, SLO-goodput
  **10% 이상 증가**를 동시에 달성

normal 0.8× state의 E2E/goodput 회귀는 3% 이내여야 한다. full C6는
app-global-only와 network-request-only보다도 SLO-goodput 10% 또는 p99 20%
incremental gain을 보여야 한다. 모든 지표의 denominator는 offered population
전체이며 reject/fail을 제외하지 않는다.

### 70.7 실행 순서와 STOP/GO

1. **STOP:** v108 route coefficient, queue cap, Cassini penalty 또는 p07 intensity
   조정. v107 seven-arm blind retry도 하지 않는다.
2. **P0:** v0–v544b, C1–C5, v536/v538, v105–v107의 raw-backed capacity와
   crossover를 C6 schema로 재정리한다. 이미 있는 valid experiment를 다시
   실행하지 않는다.
3. **P1:** `RouteCandidate`, resource ledger, telemetry와 native header/official
   proxy seam을 P×D edge identity로 확장하고 terminal conservation, immutable
   commit, no hidden fallback unit/integration test를 통과한다.
4. **P2:** controller 없이 fixed local/remote/fixed-pair/edge victim matrix로
   Q0–Q3 workload gate를 닫는다. 이 단계의 목적은 component score가 아니라
   전체 dynamic workload가 material orchestration opportunity를 갖는지
   판정하는 것이다.
5. **P3:** C6 controller/profile/workload/aggressor schedule/analyzer를 함께
   freeze한다. CPU replay는 conservation와 bounded decision만 검증하며 native
   성능 oracle로 사용하지 않는다.
6. **P4:** 사용자가 승인한 하나의 4-node `gpu_interactive` allocation에서
   local, remote, predictor, queue-GPU, network-request-only, app-global-only,
   full C6를 same offered population과 같은 exogenous NCCL/LMCache schedule로
   실행한다.
7. **P5:** discovery가 Q4와 §18 gate를 통과한 경우에만 새 승인 allocation에서
   frozen independent validation을 한 번 수행한다.

Q1/Q2가 실패하면 “TEMPO가 5%밖에 못 한다”고 결론내리지 않는다. 해당
4-node topology/workload에는 material actuator headroom이 없다고 기록하고
long-context, P×D mesh 또는 scale tier로 이동한다. 반대로 Q1/Q2를 통과한 뒤
full C6가 app-only/fixed를 못 이기면 그때 C6 control law를 native negative로
닫는다.

### 70.8 현재 allocation 상태와 실행 경계

v107 seven-arm p07 campaign은 정상 완료됐다. 이후 계획했던 no-background
remote 중복 arm은 GPU child step/data plane 전에 Slurm active-step query가
지연됐고, v536/v538에 이미 같은 causal 비교가 있음을 확인해 사용자가 승인한
allocation을 낭비하지 않도록 foreground에서 중단했다.
`results/tempo_go_cxi_v107_remote_noload_57490824`의 빈 pre-data-plane 경로는
성능 evidence가 아니다. 이 절 작성 시점에는 allocation `57490824`의
foreground interactive shell만 유지되고 TEMPO GPU child step은 실행 중이지
않다. 이후에도 batch job을 submit하지 않고, 실행이 필요하면 이 승인된
interactive allocation 안의 bounded foreground step만 사용한다.

---

## 71. C6 current-source native discovery positive와 다음 freeze/validation gate

이 절은 2026-08-24의 최신 continuation이며 §70.8의 allocation 상태와
performance 상태를 supersede한다. §18의 mission, original final gate와 safety
boundary는 바꾸지 않는다. 특히 아래 결과를 independent validation이나
Perlmutter-scale final cross-layer superiority로 확대하지 않는다.

### 71.1 실행 provenance와 고정 비교 경계

사용자가 승인한 기존 4-node/4-hour `gpu_interactive` allocation만 사용했다.
Slurm 표시 시각은 NERSC local PDT이고 shell `date -u`는 UTC이므로, 종료된
것처럼 보였던 job `57505530`은 실제로 계속 RUNNING이었다. 확인 중 시작한
추가 `salloc`은 job ID가 생성되기 전에 중단됐고
`QOSMaxSubmitJobPerUserLimit`로 abort됐다. 새 allocation이나 batch job은
생성하지 않았다. 다른 사용자 job은 조회만 했고 cancel/modify하지 않았다.

```text
job / qos            57505530 / gpu_interactive
nodes / GPUs         4 / 16 A100
nodes                nid001093,nid001096,nid001097,nid001100
full child step      bounded 4-node foreground srun in the existing allocation
container/root/UDI   none
```

비교에 사용한 frozen identity는 다음과 같다.

- qualification contract:
  `eval/sota_4node/tempo_go_c6_performance_contract_v1.json`
- contract SHA:
  `93419d37870ec9a2b5bf3dc8e0e3cfb5429901e5e73d9061522e6cc751eb4472`
- strongest fixed source:
  `results/tempo_go_c6_performance_job_57498728_v2/fixed_p1d0/result.json`
- current full result:
  `results/tempo_go_c6_performance_job_57505530_hierarchy_fix7_prefetch/full_c6/result.json`
- current full result SHA:
  `7911d4a438716ee54fec97fd73b67a665b2aff85fa89ed12aacaf0a99f781598`
- current raw aggregate SHA:
  `ba857c064a03a0444b969883df2e66f16b487ea3a6731ec55bfebf7baf7000cd`
- fixed comparison analysis:
  `results/tempo_go_c6_performance_job_57505530_hierarchy_fix7_prefetch/analysis_with_fixed_57498728.json`
- analysis SHA:
  `8f862c9dfce8963f08412a0f8c7a453819978b585ee2a32767bd507bf81fc662`

fixed와 full은 exact same offered-population contract를 사용하지만 서로 다른
allocation의 receipt다. 따라서 discovery comparison으로는 유효하되 최종
same-allocation counterbalanced 또는 independent validation을 대신하지 않는다.

### 71.2 중단됐던 native 경로의 원인과 전체-system 수정

초기 C6 full은 global scheme이 불필요해서 실패한 것이 아니었다. strongest
fixed `fixed_p1d0`은 normal p50 `3,264.34 ms`에서는 강했지만 hotD0 p50/p99
`23,161.41/24,772.26 ms`, hotD1 `5,124.44/5,907.51 ms`로 무너졌고 overload
SLO-good victim은 `126/240`뿐이었다. material bottleneck과 alternate decoder
headroom은 실제로 존재했다.

current source까지의 causal correction은 다음과 같다.

1. all-pair telemetry의 equal-deadline outer timeout race를 없애고 endpoint별
   bounded fetch/quarantine를 적용했다. 한 slow endpoint가 전체 global refresh를
   timeout으로 만들지 않는다.
2. 완전히 quarantined된 pair의 missing cross-layer envelope는 identity receipt에
   남기되 candidate frontier에서 제외하도록 hierarchy를 수정했다.
3. endpoint service feedback age만으로 healthy decoder를 영구 `path_skip`하던
   문제를 고쳤다. mesh mode에서 explicit failure 0, fresh scheduler와 completion
   evidence가 모두 있을 때만 failure-free stale-feedback fallback을 허용한다.
   `DENIED`, failure count, stale scheduler/completion은 계속 fail closed다.
4. collection-span validation failure batch는 설치하지 않되 foreground admission
   안에서 정확히 한 번만 validation-only refresh를 허용했다. background retry와
   unbounded loop는 없다.
5. 정상 critical path에서 HTTP tokenizer와 request-triggered all-pair telemetry를
   직렬로 수행하던 약 10 ms를 request-scoped concurrency로 겹쳤다. 준비 batch가
   다른 refresh에 추월되면 더 오래된 sequence를 재설치하지 않는다. freshness,
   hierarchy와 identity 검사는 admission 시점에 그대로 적용된다.

마지막 변경 전 `fix6`도 overload SLO-goodput `1.8889×`, p99 `85.88%` 감소,
reject/fail 0을 달성했지만 normal p50 회귀 `3.461%`로 3% guard를 한 항목
실패했다. `fix7_prefetch`는 profile, workload, route coefficient와 gate를
바꾸지 않고 admission p50을 normal에서 `37.22→27.14 ms`로 줄여 이 경계를
닫았다. 관련 global/frontend/router suite는 `166 passed, 22 subtests passed`,
pycompile도 통과했다.

### 71.3 current authoritative 성능 결과

| phase | full complete/reject/fail | full E2E p50 / p99 (ms) | SLO-good | selected edge |
|---|---:|---:|---:|---|
| normal | 120/0/0 | 3,343.61 / 3,401.70 | 120/120 | `local:d0` 120 |
| hotD0 | 120/0/0 | 3,159.42 / 3,367.70 | 119/120 | `local:d1` 119, `local:d0` 1 |
| hotD1 | 120/0/0 | 3,124.12 / 3,389.05 | 119/120 | `local:d0` 119, `local:d1` 1 |

strongest fixed `fixed_p1d0`과 비교한 offered-population 효과는 다음과 같다.

- overload SLO-goodput ratio: `1.8888888889×` (`238/240` 대 `126/240`)
- overload worst-p99 reduction: `86.3192%`
- all-phase SLO-goodput ratio: `1.4552845528×` (`358/360` 대 `246/360`)
- normal E2E p50 regression: `2.4283%`
- reject/fail delta: `0`

machine-check gate는 모두 true다.

```text
same_population_all_phases                    true
overload_slo_goodput_at_least_1_5x           true
overload_p99_reduction_at_least_30pct         true
normal_e2e_regression_at_most_3pct            true
reject_or_fail_does_not_increase              true
all_phase_incremental_goodput_at_least_1_1x   true
c6_performance_gate_pass                      true
```

모든 phase에서 stream/output/router decision과 terminal contract가 valid였고
360/360 admission, queue timeout 0, telemetry rejection 0, route failure 0,
hierarchy identity/stale rejection 0이다. request-scoped telemetry preparation은
360회 모두 사용됐고 superseded batch 재설치는 0이었다. hot phase의 collection
validation retry는 누적 6회 발생했지만 모두 두 번째 bounded collection에서
회복됐으며 terminal reject로 변환된 request는 0이다. persistent/background
polling은 false다.

### 71.4 정확한 연구 해석

이 결과는 “문제가 작아서 1–5%밖에 개선할 수 없다”는 이전 우려를 뒤집는다.
fixed placement가 contention state를 잘못 만나면 SLO attainment가 52.5%로
무너지고, fresh scheduler/completion/health를 결합한 global decoder assignment가
healthy D로 work를 보냄으로써 overload SLO-good work를 88.9% 늘리고 p99를
86.3% 줄였다. 즉 realistic contention에서 global orchestration의 material
opportunity와 actuator headroom이 모두 실재한다.

다만 이 run에서 full C6의 selected route는 360건 모두 LOCAL이었다. 따라서
현재 positive의 직접 원인은 P×D mesh 전체 중 **business-safe global D placement,
completion/health hierarchy와 failure-free recovery**이고, remote P→D edge,
official LMCache receiver credit, NCCL/Cassini route/stagger action의 incremental
utility는 이 receipt 하나로 증명되지 않았다. TEMPO를 component 차감식으로
축소할 이유는 없지만, 아직 발동하지 않은 component의 성능 가치를 주장해서도
안 된다.

현재 허용되는 주장은 다음이다.

> 동일 frozen C6 offered population에서 current TEMPO-GO discovery candidate는
> strongest fixed cross-edge policy보다 decoder-asymmetric overload의
> SLO-goodput을 1.889×로 높이고 worst p99를 86.3% 줄였으며, normal p50 손실을
> 2.43%와 reject/fail 0으로 제한했다.

`independent_validation_claim_allowed`는 여전히 false다. predictor-only,
queue-GPU, APP_GLOBAL_ONLY, NETWORK_REQUEST_ONLY와의 incremental superiority,
remote route counterfactual, actual LMCache/NCCL victim, larger-scale overhead와
독립 재현 전에는 final paper headline 또는 production-ready 주장을 금지한다.

### 71.5 다음 실행 순서와 STOP/GO

1. **STOP:** C6 route coefficient, normal threshold, 3% gate 또는 workload를
   사후 조정하지 않는다. `fix7_prefetch` current source와 profile/contract를
   immutable snapshot으로 freeze한다.
2. **GO — same-allocation ablation expansion:** fixed local/cross-edge,
   predictor, queue-GPU, NETWORK_REQUEST_ONLY, APP_GLOBAL_ONLY와 full C6를 같은
   offered population, exogenous schedule와 server lifecycle에서 counterbalance한다.
   full의 app-only/network-only incremental gate를 닫는다.
3. **GO — remote/cross-layer activation tier:** §70 Q1–Q3에 따라 actual official
   LMCache receiver victim과 real NCCL collective victim을 넣고, remote P0/P1→D0/D1
   edge와 receiver stagger/credit이 실제로 선택되는 phase를 qualification한다.
   synthetic p07 counter만으로 network-hot을 선언하지 않는다.
4. **GO — freeze audit:** source/profile/workload/aggressor/analyzer SHA, route
   counterfactual, tenant SLO/fairness와 no-fallback invariant를 machine-check한다.
5. **P5 independent validation:** 위 두 discovery gate를 통과한 frozen candidate만
   새 승인 4-node `gpu_interactive` allocation에서 한 번 검증한다. 결과를 본 뒤
   tuning하지 않는다.

이 절 작성 시점에 `57505530` allocation은 유지되지만 TEMPO GPU child step은
종료됐고 result/analysis가 닫혔다. 남은 allocation은 사용자가 승인한 범위의
다음 same-allocation ablation에만 사용한다. batch submit, container/UDI/root,
privileged Slingshot/NIC 변경과 자동 retry는 계속 금지한다.

### 71.6 same-allocation dynamic ablation 완료와 cross-layer activation 판정

§71.5의 dynamic ablation 항목은 2026-08-24 allocation `57505530`에서
완료됐다. fixed `fixed_p1d0`만 기존 allocation `57498728`의 exact-population
receipt이고 predictor, queue-GPU, NETWORK_REQUEST_ONLY, APP_GLOBAL_ONLY와
full은 모두 `57505530`에서 실행됐다. 여섯 arm의 semantic schedule SHA는
normal/hotD0/hotD1 모두 동일하고 profile SHA는
`bb8b31c55a2cc8642209c7eb2426007a02b94c25b37b808cd45e6e49b2442f53`다.

새 ablation contract와 authoritative campaign analysis는 다음과 같다.

- ablation contract:
  `eval/sota_4node/tempo_go_c6_ablation_contract_v1.json`
- contract SHA:
  `023dd5a176615ed2fce4872d4b9bb8e7e57e2398bc4a64a7fb0ee6b293dee636`
- campaign analysis:
  `results/tempo_go_c6_ablation_job_57505530_campaign/analysis.json`
- campaign analysis SHA:
  `49729cfd44457da84813a5307f84b6fa66182f5a51adc1ef41672a6ee555d5b6`
- receipt-only analyzer SHA:
  `f2d78a3d0a3e8462413bf7954ec3c0b3ba1f5cd0e0eab03a7925ed2959e35df4`

각 source-bound result SHA는 다음과 같다.

| arm | job | result SHA-256 |
|---|---:|---|
| fixed `fixed_p1d0` | 57498728 | 기존 §71.1 receipt |
| predictor | 57505530 | `5f93c3801e6d41107d30734b894abd2cdd7056b63ba6cba19a4e40d29bb0a9e1` |
| queue-GPU | 57505530 | `6fb722fb1cede425f15f1e3f44947467116a54e2866f54fdf96f8a9a73a61380` |
| NETWORK_REQUEST_ONLY | 57505530 | `4b381cbd7836ff37af1b0c0371e0085a8a8994b48c486254286264c4ad2ddf5b` |
| APP_GLOBAL_ONLY | 57505530 | `b0ff0e5078934a97445549e2d1f7429cb021f8168aa2f5d95cfcbe6770cf9e04` |
| full C6 | 57505530 | `7911d4a438716ee54fec97fd73b67a665b2aff85fa89ed12aacaf0a99f781598` |

모든 arm은 offered/completed `360/360`, global reject 0, failure 0,
terminal/stream/output/router exact를 만족했다. 성능은 다음과 같다.

| arm | normal p50 (ms) | hotD0 p99 (ms) | hotD1 p99 (ms) | overload SLO-good | all SLO-good | route |
|---|---:|---:|---:|---:|---:|---|
| fixed `p1d0` | 3,264.34 | 24,772.26 | 5,907.51 | 126/240 | 246/360 | REMOTE 360 |
| predictor | 3,037.53 | 35,931.67 | 36,560.71 | 122/240 | 242/360 | LOCAL 360 |
| queue-GPU | 3,054.13 | 5,169.60 | 5,159.45 | 240/240 | 360/360 | LOCAL 360 |
| NETWORK_REQUEST_ONLY | 3,001.80 | 35,988.62 | 36,378.54 | 122/240 | 242/360 | LOCAL 360 |
| APP_GLOBAL_ONLY | 3,348.22 | 3,491.09 | 3,418.03 | 238/240 | 358/360 | LOCAL 360 |
| full C6 | 3,343.61 | 3,367.70 | 3,389.05 | 238/240 | 358/360 | LOCAL 360 |

full C6 대비 효과는 다음과 같다. normal 값은 양수일수록 full의 회귀다.

| baseline | overload SLO-good ratio | worst-hot p99 감소 | normal p50 변화 | reject/fail delta |
|---|---:|---:|---:|---:|
| fixed `p1d0` | 1.8889× | 86.319% | +2.428% | 0 |
| predictor | 1.9508× | 90.730% | +10.077% | 0 |
| NETWORK_REQUEST_ONLY | 1.9508× | 90.684% | +11.387% | 0 |
| queue-GPU | 0.9917× | 34.443% | +9.478% | 0 |
| APP_GLOBAL_ONLY | 1.0000× | 2.923% | -0.138% | 0 |

machine-readable 판정은 다음과 같다.

```text
decoder_global_discovery_positive                  true
full_vs_fixed_material_overload                   true
full_vs_predictor_material_overload               true
full_vs_queue_gpu_tail_reduction_at_least_30pct   true
strict_full_superiority_over_queue_gpu             false
cross_layer_incremental_superiority_over_app       false
remote_cross_layer_activation_in_full              false
independent_validation_claim_allowed               false
```

연구 결론은 component 차감식이 아니라 다음의 causal decomposition이다.

1. predictor와 NETWORK_REQUEST_ONLY는 hot decoder를 보지 못해 각 decoder에
   60/60을 보내고 overload SLO-good이 122/240으로 무너졌다. full의 global
   decoder placement는 이 실패를 실질적으로 해결했다.
2. queue-GPU도 hot decoder를 정확히 피해 SLO-good 240/240을 달성했다. 하지만
   request-start queue/KV gauge만 사용하는 이 baseline의 worst hot p99는
   5.170초였고 full은 3.389초로 34.44% 낮았다. 반대로 full은 normal p50가
   9.48% 느리고 SLO-good이 2건 적다. 따라서 strongest dynamic baseline에 대한
   모든 지표의 strict superiority는 아직 아니다.
3. APP_GLOBAL_ONLY와 full은 같은 119/1 healthy-decoder assignment와 같은
   238/240 overload SLO-good을 보였다. full의 cross-layer envelope 추가 효과는
   worst p99 2.92%뿐으로 사전 5% incremental gate를 통과하지 못했다.
4. fixed arm만 official LMCache REMOTE 360건을 실행했고 나머지 dynamic arm은
   모두 LOCAL이었다. 따라서 현재 workload는 NCCL/Slingshot/LMCache actuator를
   켜는 실험이 아니며, 그 component의 무가치를 뜻하지 않는다. **현재 증명된
   것은 decoder-global orchestration opportunity이고, fabric/remote 공동제어는
   아직 activation 전이다.**

실행 중 APP_GLOBAL_ONLY의 첫 시도는 node source verifier의 `_sha256` 누락,
두 번째는 기존 C5 stream wrapper의 C6 arm-ID 재작성 때문에 measurement 전에
실패했다. 두 결함을 source inventory와 전용 C6 stream seam으로 고친 뒤
`fix3`가 성공했다. 중간의 한 `srun`은 child step 전 RPC에서 정지해 정확한
로컬 launcher PID만 종료했고 allocation은 건드리지 않았다. 마지막 network
arm의 step 생성도 약 2분 지연됐지만 중복 호출 없이 자체 회복했다. batch,
container/UDI, root, privileged NIC 설정은 사용하지 않았다. 최종 관련 suite는
`222 passed, 28 subtests passed`다.

다음 실행 순서는 이제 다음으로 좁힌다.

1. **STOP:** 현재 decoder-asymmetric C6에서 queue-GPU나 APP_GLOBAL_ONLY를
   이기기 위해 coefficient/SLO/workload를 사후 조정하지 않는다. 이 receipt는
   decoder-global positive와 cross-layer-not-activated 결과로 freeze한다.
2. **GO — C7 remote/fabric activation qualification:** actual official LMCache
   receiver traffic과 real NCCL collective co-job을 같은 phase에 넣고, LOCAL과
   REMOTE가 모두 합리적인 crossover를 갖도록 한다. 최소 한 phase에서 full이
   remote P0/P1→D0/D1와 receiver credit/stagger를 실제 선택해야 한다.
3. **GO — joint-control comparison:** 동일 frozen population에서 queue-GPU,
   APP_GLOBAL_ONLY, NETWORK_REQUEST_ONLY와 full을 비교한다. full은 fabric victim
   보호, inference SLO-goodput, tail, fairness 중 사전 고정된 compound gate를
   통과해야 한다. 단순히 network counter가 올랐다는 것으로 통과시키지 않는다.
4. **GO — business/scale tier:** interactive/batch deadline, admission fairness,
   pair scale-out/in과 node/fabric failure quarantine를 함께 넣어 4-node에서 먼저
   correctness와 utility를 닫은 뒤 더 큰 Perlmutter allocation으로 확장한다.
5. **P5 independent validation:** C7 discovery gate를 통과한 frozen source만 새
   승인 allocation에서 한 번 재현한다. 결과를 본 뒤 tuning하지 않는다.

이 업데이트 시점에는 모든 GPU child step과 foreground launcher가 종료됐다.
allocation `57505530`은 취소하지 않았고 새 Slurm job도 만들지 않았다.

### 71.7 C7 combined-hot actual-vLLM campaign 최종 상태: bottleneck은 실존하지만 full gate는 미통과

새 4-node `gpu_interactive` allocation `57517126`에서 §71.5의 C7 joint-control
workload를 실제 vLLM P/D 경로로 다시 실행했다. 모든 arm은 동일한
combined-hot population을 사용했다. 각 hot phase에는 두 P에서 공식 LMCache
remote prefill이 같은 hot decoder로 fan-in되고, paired P의 local
chunked-prefill aggressor가 동시에 실행된다. interactive victim은 `4094/128
MISS`, 1 req/s이고 hot decoder는 D0/D1로 phase-switch한다. transport는
official LMCacheConnectorV1 NIXL/UCX와 vLLM TP4 NCCL/Perlmutter 경로이며
synthetic fabric traffic, root, UDI/container, privileged NIC 설정은 사용하지
않았다.

authoritative contract/campaign receipt:

- contract: `eval/sota_4node/tempo_go_c7_joint_control_contract_v1.json`
- contract SHA-256: `b1c8c195548fdb88cefa8db826746de0b754d2e441bcfd9d5cd26b136a974904`
- campaign analysis: `results/tempo_go_c7_joint_control_job_57517126_combined_v4_campaign/analysis.json`
- campaign analysis SHA-256: `0f9d0f480178f62bf6fb724f12bd1f58f7f39c03326921cfa456863699c905da`

| arm | normal p50 (ms) | hot p99 (ms) | hot SLO-good | global reject | observed edge |
|---|---:|---:|---:|---:|---|
| fixed local D0 | 3,048 | 9,547 | 37/60 | 0 | local:D0 |
| fixed local D1 | 2,982 | 9,619 | 37/60 | 0 | local:D1 |
| fixed remote P0→D1 | 3,144 | 18,784 | 30/60 | 0 | remote:P0→D1 |
| fixed remote P1→D0 | 3,135 | 17,351 | 28/60 | 0 | remote:P1→D0 |
| predictor | 2,886 | 9,349 | 38/60 | 0 | local D0/D1 |
| queue-GPU | 2,906 | 8,132 | 54/60 | 0 | local D0/D1 |
| NETWORK_REQUEST_ONLY | 2,919 | 8,378 | 49/60 | 0 | local D0/D1 |
| APP_GLOBAL_ONLY | 3,063 | 4,030 | 27/60 | 33 | local D0/D1 |
| full C7 | 3,060 | 4,009 | 30/60 | 30 | local D0/D1 |

이 결과가 답하는 연구 질문은 다음과 같다.

1. **Contention/bottleneck은 실존한다.** fixed remote는 같은 workload에서 hot p99가
   `17.35–18.78 s`로 상승했다. LMCache가 문제를 만들지 못한 것이 아니라, 실제
   P0/P1→D receiver fan-in과 local decoder work가 겹칠 때 remote data plane과
   decoder service window가 함께 무너지는 regime이 존재한다.
2. **Global decoder orchestration은 필요하다.** full C7은 hot D0에서 D1로,
   hot D1에서 D0로 victim을 우회했고 `full_switches_away_from_hot_receiver`
   gate는 true다. telemetry attribution bug도 고쳐 P outbound prefill credit와
   D receiver KV/semantic credit를 분리했다.
3. **하지만 현재 TEMPO full은 아직 해답이 아니다.** full은 hot p99만 보면 fixed
   local보다 58.0% 낮지만, 120 offered victim 중 30건을 global queue timeout으로
   거절했다. predictor와 queue-GPU 대비 compound robustness gate는 false이고,
   C7 discovery positive도 false다. `full_uses_both_local_and_remote=false`라서
   cross-layer remote actuator가 실제 선택된 positive phase도 아직 없다.
4. **이것은 TEMPO negative라기보다 admission objective의 미완성이다.** 현재 full은
   cross-layer/decoder safety를 우선해 healthy spare decoder로 이동한 뒤 global queue
   timeout에서 보수적으로 reject한다. queue-GPU는 native queue를 더 오래 활용해
   `54/60` SLO-good을 얻었다. completion-conditioned p99만으로 full을 평가하면
   reject로 tail을 숨기는 오류가 생긴다.

C7에서 수정되어 검증된 구현 경계:

- `tempo/pd_global_telemetry.py`: source-P remote prefill과 destination-D remote
  KV/semantic usage를 decoder target 기준으로 집계한다.
- `eval/sota_4node/tempo_pd_elastic_router.py`: `mesh_remote_by_decoder`로 실제
  P→D 목적지를 보존한다.
- `tempo/pd_global_hierarchy.py`: 모든 pair가 quarantine되어 빈 frontier가 되는
  경우를 HTTP 400으로 누출하지 않고 admission-level rejection으로 처리한다.
- `tempo/pd_global_coordinator.py`: 이를 `global_hierarchy_no_candidate`의 정상
  business rejection으로 기록한다.
- 관련 targeted suite는 `85 passed`; C7 raw block은 400/internal error 없이
  terminal contract를 만족했다. 별도의 기존 C6 ablation test 두 건은 C6 contract의
  예전 router hash drift를 검출한 것이며 C7 변경 실패가 아니다.

목표 자체는 유지하되, 다음처럼 성공 조건을 명확히 한다.

> Perlmutter-scale NCCL/Slingshot/LMCache/vLLM/business cross-layer global
> orchestrator를 만들어 shared-fabric/receiver/decoder contention에서 route 선택,
> admission, fairness, queueing을 공동 제어하고, 단순 predictor와 strongest fixed
> policy보다 **offered-work 기준 SLO-goodput**을 높인다.

다음 구현/실험 gate는 고정한다.

1. **Goodput-aware admission:** completion만의 p99를 최적화하지 않는다. offered
   request 기준 SLO-goodput, reject ratio, deadline miss를 함께 utility로 두고,
   transport-critical 상태에서만 hard reject한다. healthy alternate decoder가 있으면
   global queue가 first-response release를 기다리도록 tenant business wait budget과
   reserved interactive slots를 명시적으로 사용한다.
2. **Remote activation matrix:** remote source/edge가 cool하고 receiver만 hot인
   phase와 source/fabric까지 hot인 phase를 분리해 causal workload로 freeze한다.
   전자에서는 full이 최소 한 phase에서 remote P→D를 실제 선택해야 하고, 후자에서는
   remote를 닫고 local survivor로 이동해야 한다. 현재처럼 모든 remote 후보가 source
   credit에서 닫힌 campaign만으로 cross-layer value를 판정하지 않는다.
3. **Compound comparison:** fixed local, fixed remote, predictor, queue-GPU,
   APP_GLOBAL_ONLY, NETWORK_REQUEST_ONLY, full을 같은 offered population으로 다시
   비교한다. gate는 offered SLO-goodput 우위, p99, normal regression, fairness,
   reject/failure, 실제 edge actuation을 동시에 요구한다.
4. **Business/scale layer:** interactive/batch weighted fairness, deadline class,
   pair scale-out/in, node/fabric failure quarantine를 4-node에서 먼저 닫고,
   그 뒤에만 더 큰 Perlmutter allocation으로 확장한다.

이번 C7 campaign은 allocation `57517126`의 foreground interactive 실행으로
완료됐으며, batch submit이나 다른 Slurm job은 만들지 않았다. 현재 결론은
**“문제 실존 및 global orchestration 필요성은 입증; TEMPO full의
offered-goodput 우위는 아직 미입증”**으로 freeze한다.

### 71.8 C7 goodput-aware admission probe: queue lease는 작동했지만 remote activation은 아직 미증명

§71.7에서 확인한 global timeout을 바로 remote routing 문제로 해석하지 않기 위해,
같은 4-node interactive allocation `57517126`에서 full C7 arm만 diagnostic probe로
재실행했다. 이 probe는 새로운 성능 claim이나 9-arm comparison이 아니며, 목적은
`offered request`를 queue timeout으로 버리는 현상을 admission controller가
completion-conditioned bounded lease로 완화할 수 있는지 확인하는 것이었다.

추가 구현은 다음 경계를 사용한다.

- `tempo/pd_global_orchestrator.py`에
  `completion_credit_mesh_endpoint_queue_v1`를 추가했다. completion credit이
  실제 first-response completion에서만 생기고, lease commit 때 차감된다.
- mesh lease도 source prefill, edge, receiver credit 및 destination work를 그대로
  보존한다. 즉 queue debt를 숨겨 remote/fabric capacity를 초과시키는 bypass가
  아니다.
- interactive tenant만 bounded lease를 허용하고 batch/background는 허용하지 않는다.
  v2 profile은 `maximum_queue_wait_ns=2s`, `headroom_first_v1`, endpoint queue
  capacity 32, `receiver_credit_pxd_v1`를 사용했다.
- `tempo/test_pd_global_orchestrator.py`에 destination/edge ownership과 credit
  차감을 검증하는 mesh lease unit test를 추가했고, 관련 orchestrator/coordinator
  suite는 `75 passed`였다.

receipt와 source identity는 다음과 같다.

- profile:
  `results/tempo_go_c7_goodput_mesh_lease_profile_v2/real_tempo_go_c7_goodput_mesh_lease_profile_v2.json`
- profile SHA-256: `d8f08bedf828650f731591c93cf47fd3fc1976370eaf7a4e9881cc9655a6efe5`
- profile fingerprint: `e88bba0ce29a003b483a05896ef3a0e31ea93e05904a974c67684fa4b7163d2c`
- contract:
  `eval/sota_4node/tempo_go_c7_goodput_mesh_lease_contract_v2.json`
- contract SHA-256: `433c395af880a2fa624be9c8cb714f7f055b4a24ba1d118264e53a8e3c306df9`
- probe receipt:
  `results/tempo_go_c7_goodput_mesh_lease_job_57517126_fullprobe_v2/full_c7/result.json`
- controller source SHA-256:
  `7598afdc60388c30c2a2776089fcf173ba0d9ff734be7e41318f1ae18dfa28f5`

v1은 `maximum_queue_wait_ns=6s`를 사용했는데, hot phase에서 victim이 약 6초를
기다린 뒤 local candidate의 deadline headroom이 사라져 global timeout으로 거절됐다.
v1 full probe는 120 offered 중 90 complete, 30 reject, hot SLO-good 90/120였고
queue lease는 0건이었다. 이것은 lease 구현의 실패라기보다 긴 대기 자체가
offered-goodput을 파괴한 receipt다.

v2는 같은 C7 phase와 같은 120 offered victim에서 다음을 기록했다.

| metric | §71.7 full C7 | v2 goodput probe | 변화 해석 |
|---|---:|---:|---|
| offered victims | 120 | 120 | 동일 offered population |
| completed | 90 | 120 | admission reject 제거 |
| global rejects | 30 | 0 | queue-timeout 제거 |
| offered SLO-good | 90/120 (75.0%) | 120/120 (100.0%) | diagnostic goodput 개선 |
| all-victim p99 | 4,008.95 ms | 3,247.23 ms | completion tail도 감소 |
| hot-block max p99 | 4,008.95 ms | 3,365.45 ms | hot tail 감소 |
| observed route | local only | local only | remote actuator 미활성 |
| completion-credit queue leases | 0 | 32 | bounded lease는 실제 사용 |

v2의 block별 hot 결과도 두 hot decoder에서 모두 `30/30` SLO-good이었다.
`01_hot_d0` p99는 `3,365.45 ms`, `02_hot_d1` p99는 `3,188.63 ms`였으며,
모든 victim route는 `decoder_local_chunked_prefill`였다. 따라서 이 probe가
증명한 것은 **business-aware offered-goodput admission과 bounded local queue
lease가 C7의 reject collapse를 완화한다**는 것까지다. `full_uses_both_local_and_remote`
는 여전히 false이며, remote P→D edge를 고른 victim은 0건이다. 그러므로 이 결과를
NCCL/Slingshot/LMCache cross-layer remote orchestration의 최종 성능 결과로 보고하지
않는다. result receipt의 `performance_claim_allowed`도 false로 유지되어 있다.

이번 probe 뒤의 연구 판단과 다음 gate는 다음과 같이 고정한다.

1. C7의 문제는 admission과 remote activation이 결합된 두 개의 failure mode다.
   v2로 첫 번째(global queue timeout)는 개선됐지만, 두 번째(remote source/edge/
   receiver가 실제로 선택되는 crossover)는 아직 workload가 닫지 못했다.
2. 다음에는 coefficient를 더 조정하지 않는다. remote source/fabric이 cool하고
   decoder/receiver만 hot인 phase와 source/fabric까지 hot인 phase를 분리한
   remote activation matrix를 새 frozen contract로 만든다. cool phase에서는
   remote P→D가 실제 선택되어야 하고, fabric-hot phase에서는 같은 controller가
   remote를 차단하고 local survivor/queue lease로 이동해야 한다.
3. 그 matrix가 성립한 뒤에만 fixed local, fixed remote, predictor, queue-GPU,
   APP_GLOBAL_ONLY, NETWORK_REQUEST_ONLY, full을 동일 offered population으로
   재실행한다. 최종 gate는 offered SLO-goodput, p99, normal regression,
   fairness/reject, 실제 edge actuation을 동시에 본다.

따라서 현재 상태는 **“goodput-aware global admission은 첫 native positive;
remote/fabric 공동제어의 positive는 아직 없음; 다음 필수 작업은 remote activation
matrix”**다. 이번 probe도 기존 allocation `57517126` 안에서만 실행했으며 batch
submit, allocation cancel, root, UDI/container, privileged NIC 설정은 사용하지
않았다.

### 71.9 C7 activation matrix 원인 규명: 후보 손실이 아니라 inactive-pair 정책이 remote-cool edge를 버림

§71.8의 다음 gate를 수행하기 위해 `remote_cool_hot_d0/d1`와
`combined_hot_d0/d1`를 분리한 activation matrix를 실제 4-node path에서 실행했다.
현재 definitive receipt는 다음과 같다.

- matrix contract v1: `eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v1.json`
- matrix full v1 receipt: `results/tempo_go_c7_remote_activation_matrix_job_57517126_full`
- matrix fixed remote receipt: `results/tempo_go_c7_remote_activation_matrix_job_57517126_fixed_remote_p0d1`
- matrix full v2 receipt: `results/tempo_go_c7_remote_activation_matrix_job_57517126_full_v2`
- 새 route-benefit contract: `eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v2.json`
- 새 profile fingerprint: `4a03620a21ccb06b554630b566bcd4305dc6f5eed3780ce4b0e3ae4b3ba70b1e`

실험은 controller에 phase label을 주지 않고, 실제 request-triggered vLLM scheduler
telemetry와 endpoint/Cassini envelope만 사용했다. 결과는 “remote path가 존재하지
않는다”가 아니라 **remote-cool edge가 실제로 유리하지만 full controller가 inactive
decoder를 후보 선택 단계에서 닫고 있었다**는 것이다.

| block | full v1/v2 route | full v2 p99 | fixed P0→D1 p99 | 해석 |
|---|---|---:|---:|---|
| remote-cool hot D0 | local:D0 | 7,868.36 ms | 3,315.18 ms | D1 receiver가 cool일 때 remote가 실제 유리 |
| combined-hot D0 | local:D1 after one reject | 5,047.72 ms | 9,667.18 ms | remote를 무조건 쓰면 안 됨 |
| remote-cool hot D1 | local:D1 | 7,834.19 ms | 6,720.51 ms | hot receiver 회피가 필요 |
| combined-hot D1 | local:D0 | 3,237.60 ms | 19,648.15 ms | source/receiver가 함께 hot이면 remote 악화 |

full v2 전체는 `179/180` completed, `179/180` SLO-good, hot은 `119/120`,
remote route는 `0`건이었다. 반면 fixed P0→D1은 remote-cool D0 block에서
`30/30` SLO-good, p99 `3,315.18 ms`를 기록했지만 combined-hot block에서는
p99가 `9,667.18` 및 `19,648.15 ms`까지 악화됐다. 따라서 TEMPO가 배워야 할 것은
“remote를 켜라”가 아니라 **현재 decoder/receiver와 source/edge/fabric의 공동 상태로
remote edge를 조건부 활성화하라**이다.

raw hierarchy receipt는 각 victim에 `raw_candidate_count=6`,
`forwarded_candidate_count=6`을 기록했다. 즉 P0/P1→D0/D1 후보가 hierarchy에서
사라진 것은 아니다. 그러나 `GlobalOrchestrator._options()`는 active pair의
local candidate 하나가 admissible하면 inactive pair 후보를 rejected receipt에도
남기지 않고 폐기했다. 당시 D0가 active이고 D1이 prewarmed spare였으므로
`remote:p0->d1`와 `remote:p1->d1`는 global score 계산에 들어가지 않았다. 이는
§71.8의 scheduler queue-wave penalty만 추가해도 route가 바뀌지 않았던 이유다.
queue penalty는 D0 score를 올렸지만, inactive D1 edge 자체를 selection frontier에
넣지 않았기 때문이다.

이번 원인에 대한 구현 수정은 다음으로 고정했다.

- `GlobalOrchestratorConfig`에
  `proactive_scale_up_route_benefit_margin_ms`를 추가했다. C7 v3 profile은 값을
  `0.0`으로 고정한다.
- `_options()`가 active 후보와 inactive 후보를 모두 fully-priced 한 뒤,
  scheduler queue-wave, endpoint/receiver, source/edge, cross-layer externality,
  activation cost를 포함한 inactive 최저 score가 active 최저 score보다 margin만큼
  낮을 때만 spare pair를 atomic activation한다.
- queue occupancy/SLO-risk가 이미 scale 이유인 경우 기존 business reason을
  유지한다. phase label, future arrival, oracle route는 입력하지 않는다.
- route-benefit activation을 검증하는 unit test를 추가했고 관련
  orchestrator/coordinator/profile/hierarchy suite는 `111 passed`다.

새 frozen artifact는 다음과 같다.

- profile:
  `results/tempo_go_c7_goodput_mesh_lease_profile_v3/real_tempo_go_c7_goodput_mesh_lease_profile_v3.json`
- profile file SHA-256: `399b41551b01f63c5b3463731b4da46cfbde2023eb95190512aaf6a80c6e369e`
- profile fingerprint: `4a03620a21ccb06b554630b566bcd4305dc6f5eed3780ce4b0e3ae4b3ba70b1e`
- contract file SHA-256: `d4c8b113d1ad3103f7c7e4b0e1fb02d3bea6e9b9f2a129f8dc28657686f7db72`
- current orchestrator source SHA-256: `25efbd894d7c5b62337742934f59003fb6fc942b83318088094a72ce08390f12`
- `_qualification()`는 source inventory 19개와 함께 통과했다.

allocation `57517126`은 이 수정 후 native rerun을 시작하기 전에 4시간 time
limit으로 종료됐다. 그러므로 route-benefit 수정은 아직 성능 결과가 아니라
**원인에 맞춘 검증 가능한 다음 controller version**이며, 이 문서에서 positive
performance claim으로 세지 않는다.

다음 실행 순서는 고정한다.

1. 새 4-node `gpu_interactive`에서 contract v2의 full arm을 먼저 실행해
   `remote-cool D0/D1`에서 `remote:p0/p1→cool D`가 실제 선택되는지 확인한다.
2. 같은 allocation에서 fixed local, fixed remote P0→D1, full을 같은 matrix로
   재실행한다. full이 remote-cool에서는 remote를 쓰고 combined-hot에서는 remote를
   닫지 못하면 구현을 통과시키지 않는다.
3. 그 뒤에만 §71.8의 goodput probe와 동일한 offered-work compound comparison으로
   돌아가 queue-GPU/predictor/APP_GLOBAL_ONLY/NETWORK_REQUEST_ONLY를 포함한다.

현재 연구 결론은 다음으로 업데이트한다.

> contention과 remote-cool crossover는 실제이며 global orchestrator의 필요성은
> 입증됐다. 첫 full 실패의 직접 원인은 LMCache가 문제를 만들지 못해서가 아니라,
> inactive prewarmed decoder의 cross-destination edge를 global selection이
> 조기에 버린 것이다. route-benefit activation을 추가했지만 native 성능 positive는
> 아직 미실행이므로, 다음 gate는 이 edge가 실제 선택되는지의 증명이다.

### 71.10 4-node 32/s overload boundary: 문제는 실존하지만 현재 global control-plane도 함께 무너짐

§71.9의 route-benefit 수정 뒤 allocation `57529507`에서 4-node `gpu_interactive`
내 native workload를 더 강하게 만들었다. 기존 7.8/s는 decoder/receiver pressure는
만들었지만 remote fabric counter가 거의 변하지 않았으므로, 이번 contract는
reference rate 32/s를 selective/combined block에 명시적으로 고정했다. 각 hot block은
30초 동안 remote aggressor 960건(4094 prompt token, request당 KV
234,766,336 bytes, 총 약 225.4 GB)과 30건 interactive victim을 같은 population으로
제공한다. controller는 phase label을 받지 않는다.

현재 artifact:

- contract: `eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v9.json`
- contract SHA-256: `710bfaf2d7e4ec1dadbebf04a709d5960767ea527486aecbd50bcef1f0c3d353`
- selective receipt: `results/tempo_go_c7_remote_activation_matrix_job_57529507_full_v11/full_c7/tempo_go_c7_joint_control/c7_joint_full_c7_measured/01_remote_cool_hot_d0.raw.json`
- combined partial receipt: `results/tempo_go_c7_remote_activation_matrix_job_57529507_full_v9/full_c7/tempo_go_c7_joint_control/c7_joint_full_c7_measured/02_combined_hot_d0.raw.json`
- current client SHA-256: `4744da967b1cec4248dde169cd8040500dca88b915c59a8f61f3a2d29734f0a0`
- current orchestrator SHA-256: `2e9cbd84236cab3b0f86bbabdd79cb0dd182fa0db2e1034747351421bf2720de`

#### Native overload receipt

selective `01_remote_cool_hot_d0`는 990 total requests(960 background + 30 victim)를
끝까지 terminalize했다.

| metric | receipt |
|---|---:|
| completed total | 962 |
| interactive victim HTTP 200 | 2/30 |
| interactive victim global reject | 28/30 |
| global reject breakdown | 27 `global_telemetry_validation_failed`, 1 `global_admission_queue_timeout` |
| terminal contract | true |
| background route | 960 `official_lmcache_remote_prefill` |

combined `02_combined_hot_d0` partial receipt는 1950 total requests에서 1921
completed, 29 global rejects를 남겼다. 이 block은 두 remote source와 local hot
aggressor를 동시에 넣은 뒤 native stream child가 exit code 2로 종료되어 campaign
전체 performance result로 승격하지 않는다. 그러나 이것은 조용히 사라진 실패가
아니라 `endpoint_bounded_queue_lease_timeout`, `global_admission_queue_timeout`,
`global_hierarchy_no_candidate`, `global_telemetry_validation_failed`가 분리된
terminal receipt다.

#### Bottleneck attribution

이 실험은 “부하가 없다”는 결론을 반박한다. endpoint evidence midpoint에서
combined block의 hot D0는 `running_requests=5`, `waiting_requests=34`, KV usage
약 `0.0194`를 보였다. 반면 같은 시점 Cassini sampled counters는 pause fraction이
RX/TX 모두 `0`이었다. selective block도 Cassini RX/TX pause는 `0`이었고,
OXE channel active fraction 최대치는 약 `8.34e-5`였다. 즉 이번 rate에서 먼저
무너진 것은 GPU/vLLM scheduler, endpoint bounded queue, LMCache semantic/remote
credit 및 telemetry refresh path이며, sampled Cassini pause counter가 saturated
fabric을 증명하지는 않았다.

이 결과의 의미는 interconnect가 불필요하다는 것이 아니다. 현재 4-node workload와
실제 NIXL/UCX path에서는 약 225 GB의 remote KV가 전송되는 동안에도 link-level
pause가 발생하지 않았고, 대신 endpoint receiver/GPU service window가 먼저
병목이 됐다. 따라서 다음 TEMPO controller는 다음 세 resource를 분리해 동시에
보아야 한다.

1. GPU scheduler/decoder queue와 endpoint completion window;
2. P-side remote prefill credit, D-side receiver KV/semantic credit, edge inflight;
3. Cassini/NCCL/Slingshot sampled fabric evidence와 그 freshness/causal span.

#### 새로 드러난 TEMPO failure mode

32/s에서 첫 두 victim만 local:D1로 commit된 뒤, 나머지는 후보 score를 비교하기
전에 `global_telemetry_validation_failed`로 거절됐다. 마지막 유효 telemetry
sequence는 32였고, 이후 sequence 34 refresh가 overloaded frontend에서 causal
collection span을 지키지 못했다. controller의 stale grace는 1.0초로 제한되어
있는데, client worker pool과 native queue가 120초 이상 밀린 victim은 그 grace를
넘어 stale snapshot도 사용할 수 없었다.

이것은 단순한 LMCache 실패가 아니라 global orchestrator의 control-plane 문제다.
현재 global coordinator가 가진 stale-snapshot fallback 자체는 capacity/health
guard를 우회하지 않지만, freshness window를 넘으면 interactive request를
비즈니스 우선순위에 따라 빠르게 alternate route로 보내지 못하고 telemetry
validation rejection으로 끝난다. 동시에 32/s combined phase는 endpoint service
lane이 bounded queue lease timeout을 내어도 기존 C7 augmentation validator가
`bounded_ingress_queue`를 invalid route로 오인하는 parser bug도 드러냈다. 이
parser bug는 수정했고 관련 suite는 `85 passed, 11 subtests passed`다. service-lane
failure는 global commit 이후에도 발생할 수 있으므로, commit 여부와 terminal
service failure를 별도 필드로 보존한다.

따라서 현재 연구 결론은 다음과 같이 갱신한다.

> 32/s native 4-node workload에서 contention은 확실히 실존한다. 다만 이 실행에서
> 첫 bottleneck은 Cassini pause가 아니라 vLLM/endpoint/LMCache receiver와
> telemetry control-plane이었다. 이것은 TEMPO의 가치가 줄어드는 결과가 아니라,
> global orchestrator가 GPU·LMCache·fabric·business admission을 공동 제어해야
> 하는 더 강한 근거다. 아직 TEMPO가 strongest fixed/predictor보다 빠르다는
> performance claim은 하지 않는다.

다음 실행 gate는 rate coefficient를 더 만지는 것이 아니다.

1. **Telemetry continuity gate:** overload 중에도 request-triggered telemetry의
   causal span, freshness, stale fallback, validation failure를 별도 측정한다.
   interactive tenant는 120초 뒤 reject되는 대신 business wait budget 안에서
   bounded alternate admission 또는 명시적 fast reject를 받아야 한다.
2. **Ingress realism gate:** `max_workers=256`이 background request에 묶여 victim이
   controller에 도달하지 못하는 client-side artifact인지 분리한다. open-loop
   arrival, bounded frontend concurrency, tenant-priority ingress를 각각 같은
   offered population으로 비교한다.
3. **Fabric gate:** Cassini pause/active fraction만으로 “fabric hot”을 선언하지
   않는다. NIXL remote bytes, packet/byte rate, endpoint receiver queue,
   scheduler waiting, NCCL/Slingshot evidence의 같은 interval delta를 하나의
   cross-layer envelope로 묶고, missing/invalid signal도 성능 결과에 표시한다.
4. **Actuation gate:** remote-cool phase에서 `remote:p0->cool-D`가 실제 선택되고,
   combined-hot phase에서는 remote를 닫고 local survivor/queue lease로 이동해야
   한다. 두 phase 중 하나만 성공하면 TEMPO positive가 아니다.
5. **Comparison gate:** 위 control-plane 보완 후에만 fixed local, fixed remote,
   predictor, queue-GPU, APP_GLOBAL_ONLY, NETWORK_REQUEST_ONLY, full을 다시
   동일 offered-work로 비교한다. reject를 숨긴 completion p99는 사용하지 않고
   offered SLO-goodput, deadline miss, fairness, edge actuation, normal regression을
   동시에 보고한다.

### 71.11 v13 ingress-priority native receipt: client starvation은 제거됐지만 global control-plane이 실제 overload 경계에 도달함

§71.10에서 지적한 `max_workers=256` client artifact를 분리하기 위해, 같은
4-node native 경로에서 interactive victim에 16개 worker를 예약하는
`interactive_reserved` ingress policy를 적용했다. controller에는 ingress lane을
전달하지 않고, background 240개와 interactive 16개의 executor만 client 쪽에서
분리했다. 따라서 offered population과 open-loop arrival clock은 유지되고,
TEMPO가 business lane을 직접 입력으로 받아 부당하게 유리해지는 일은 없다.

이번 실행의 frozen artifact는 다음과 같다.

- allocation: `57529507`, 4-node `gpu_interactive`; 실행 종료 후 allocation은
  만료됐으며 새 Slurm 작업은 제출하지 않았다.
- contract:
  `eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v13.json`
  (SHA-256 `184c0cd4a8517914a84cc43d2828613a834556e09c55f511bfec54b6b68c0fd1`,
  source inventory 23개)
- ingress profile: `interactive_reserved`, reserved 16, background 240,
  client `max_workers=256`
- profile:
  `results/tempo_go_c7_goodput_mesh_lease_profile_v5/real_tempo_go_c7_goodput_mesh_lease_profile_v5.json`
  (SHA-256 `b8721542466022ee1d92854b6d216520b030aa37a1618e30b1758394e744a4c7`,
  fingerprint `281787022702da89cde485db04ca3d57892ee0e0028ed8a39082bff9026b3b87`)
- measured root:
  `results/tempo_go_c7_remote_activation_matrix_job_57529507_full_v13/full_c7/tempo_go_c7_joint_control/c7_joint_full_c7_measured/`

#### Block별 native terminal receipt

| block | offered | completed | global reject | reject breakdown | stream/terminal | performance claim |
|---|---:|---:|---:|---|---|---|
| `00_control_a` | 30 | 30 | 0 | — | valid / valid | allowed for control only |
| `01_remote_cool_hot_d0` | 990 | 962 | 28 | global: telemetry 27, queue 1; 별도 terminal service-lane timeout 1 | valid / valid | false |
| `02_combined_hot_d0` | 1,950 | 1,920 | 30 | telemetry 26, queue 4 | valid / valid | false |
| `03_remote_cool_hot_d1` | 990 | 960 | 30 | telemetry 27, queue 3 | valid / valid | false |
| `04_combined_hot_d1` | 1,950 | 1,920 | 30 | telemetry 20, queue 10 | invalid / invalid | false |

`04_combined_hot_d1`의 마지막 remote aggressor 10개는 HTTP status가 200까지
갔지만 `done_event_missing`, `final_usage_missing`,
`finish_reason_not_exactly_length`, `requested_completion_tokens_mismatch`를
남겼다. 이것은 reject를 완료로 세어 성능을 부풀릴 수 있는 상황이므로 stream
contract를 실패로 유지하고, 해당 child의 exit code 2와 나머지 node의 timeout을
campaign performance result로 승격하지 않았다.

#### Ingress artifact와 실제 병목의 분리

reserved lane이 실제로 작동했는지는 victim dispatch offset으로 확인했다. selective
block의 victim은 예정된 0.5초, 1.5초, 2.5초, 3.5초, … 시점에 각각 약
`+0.07 ms`, `+0.11 ms`, `+0.04 ms`, `+0.68 ms` 지연으로 dispatch됐다. 즉
§71.10의 “background executor가 victim을 120초 밀었다”는 client-side 설명은
v13에서 제거됐다. 그럼에도 `01_remote_cool_hot_d0`에서 interactive는 30개 중
HTTP 200이 2개뿐이고 28개가 global reject였다. 병목은 client ingress만의 문제가
아니다.

v13 combined D0 endpoint evidence의 최대 관측값은 decoder 기준
`running_requests=8`, `waiting_requests=29`, KV usage `0.03097`였고,
Cassini sampled `rx_pause_fraction_max=0`, `tx_pause_fraction_max=0`, OXE active
fraction 최대 `9.70e-5`였다. 이 값은 endpoint/GPU scheduler와 telemetry/queue
경로가 실제로 압력을 받았다는 증거지만, 현재 sampled Cassini counter만으로
Slingshot/NCCL fabric saturation을 주장할 수 있다는 뜻은 아니다. 특히 225 GB
규모의 remote KV offered load가 있어도 link pause가 0일 수 있으므로, 다음
fabric gate에서는 byte/packet delta, NIXL/UCX transfer, receiver queue,
scheduler waiting, NCCL/Slingshot signal을 같은 시간 창에서 묶어야 한다.

이번 receipt가 확정한 것은 다음 두 가지다.

1. realistic shared contention은 실존한다. vLLM endpoint bounded service lane,
   receiver/LMCache path, global telemetry refresh/admission이 함께 overload
   경계에 도달했고, interactive business request도 이를 피해가지 못했다.
2. 아직 TEMPO가 fixed local, fixed remote, predictor보다 빠르다는 결론은 없다.
   v13은 ingress 원인을 제거한 뒤 control-plane rejection과 native termination을
   보인 진단/경계 실험이며, 성능 비교용 positive receipt가 아니다.

#### 구현 및 unit gate 상태

v13을 위해 interactive tenant별 `telemetry_stale_grace_ns`를 profile에 둘 수
있게 했고, v5 profile은 interactive에 5초 grace를 부여한다. coordinator는
tenant business policy를 사용해 stale snapshot을 재사용할 수 있으며, batch와
interactive의 freshness policy를 분리한다. 또한 ingress receipt 검증, endpoint
bounded service-lane failure와 global commit의 분리, v13 source inventory를
추가했다. native rerun 없이 다음 control-plane receipt도 추가했다.

- `GlobalAdmissionPreparation`은 request-scoped telemetry collection의 시작/종료,
  elapsed time, attempt 수, validation retry 여부, batch sequence 또는 refresh
  failure reason을 `tempo-go-admission-preparation-v1`로 보존한다.
- frontend ledger는 이를 accepted/rejected global decision 양쪽에 붙인다. 따라서
  다음 native run에서는 “telemetry가 늦었는지”, “validation retry가 발생했는지”,
  “stale fallback이 아니라 fail-closed reject였는지”를 HTTP 503 하나로 뭉개지
  않고 request별로 계산할 수 있다.
- post-v13 source SHA-256은 coordinator
  `b574ebda4465ddfdb0926c0ce32d66bbc9e086e380fe2b6cf0f80d68d4738fef`, frontend
  `f7173e4a1a32e3d4d2f70bf54276d0305eb0b01d05c7b2988118d721bb3f5e22`이다. 이
  instrumentation은 v13 native receipt에 소급 적용하지 않으며 다음 allocation의
  measurement prerequisite다.

관련 suite는 다음과 같이 현재 통과한다.

```text
137 passed, 11 subtests passed in 6.94s
```

단, unit pass는 native overload에서 telemetry continuity가 확보됐다는 뜻이
아니다. v13 native receipt에는 여전히 `global_telemetry_validation_failed`가
20–27건씩 발생했으므로, 5초 stale grace가 이 workload에서 business SLO를
지켰다고 보고하지 않는다.

#### 다음 실행 gate

allocation이 끝났으므로 자동 재시도하지 않는다. 다음 4-node
`gpu_interactive`에서만 아래 순서로 재개한다.

1. v13과 같은 offered population으로 telemetry refresh latency, stale age,
   validation failure, queue wait, endpoint service-lane failure를 request별로
   수집한다. stale fallback이 된 request와 fast reject된 request를 분리한다.
2. 같은 workload를 `shared_pool`과 `interactive_reserved`로 실행해 ingress
   priority의 효과를 측정한다. 이것은 TEMPO positive가 아니라 workload validity
   gate다.
3. remote-cool block에서 실제 `remote:p0/p1→cool-D` actuation이 일어나고,
   combined-hot block에서는 remote를 닫고 local survivor 또는 bounded queue
   lease로 이동하는지 확인한다.
4. 그 gate가 통과한 뒤에만 fixed local, fixed remote, simple predictor,
   queue-GPU, `APP_GLOBAL_ONLY`, `NETWORK_REQUEST_ONLY`, TEMPO full을 동일
   offered-work로 비교한다. 보고 지표는 completion-only p99가 아니라 offered
   SLO-goodput, deadline miss, interactive fairness, reject reason, route
   actuation, cross-layer attribution이다.

따라서 현재 연구 상태는 **“부하 문제와 global orchestrator 필요성은 native
경로에서 확인됐다. v13은 client starvation을 제거했지만 global telemetry/
endpoint control-plane의 overload 경계를 드러냈고, TEMPO 성능 우위는 아직
미검증이다.”**이다. 목표는 변경하지 않는다. 다음 목표는 이 control-plane과
fabric attribution을 보강한 뒤 동일 population의 fixed/predictor/full 비교를
성립시키는 것이다.

### 71.12 v14 native receipt와 v15 managed-background gate: 문제는 실제였고, 기존 full_c7의 제어 범위가 부족했다

#### v14 실행 결과

v13의 request-scoped telemetry preparation instrumentation을 포함한 현재 source로
4-node `gpu_interactive` allocation `57548970`에서 `full_c7`를 다시 실행했다.
allocation은 마지막 `05_control_b` 단계가 exit 1로 끝나 종료됐고, Slurm receipt는
`57548970|FAILED|143:0|00:19:09`, step `57548970.0|FAILED|1:0|00:18:51`이다.
따라서 아래 block 00–04의 raw/endpoint receipt만 보존하며, 전체 campaign
`analysis.json`이나 성능 positive 결론은 만들지 않는다.

- contract:
  `eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v14.json`
  (SHA-256 `b1b0443af2bcd18c7742fc193c877aacbc5b2af044dc9c43815f30be2fb3af95`)
- measured root:
  `results/tempo_go_c7_remote_activation_matrix_job_57548970_full_v14/full_c7/tempo_go_c7_joint_control/`
- same ingress policy: `interactive_reserved`, interactive 16 workers,
  background 240 workers, `max_workers=256`
- profile SHA/fingerprint: v5 profile의
  `b8721542466022ee1d92854b6d216520b030aa37a1618e30b1758394e744a4c7` /
  `281787022702da89cde485db04ca3d57892ee0e0028ed8a39082bff9026b3b87`

| block | offered | completed | global reject | terminal validity |
|---|---:|---:|---:|---|
| `00_control_a` | 30 | 30 | 0 | valid / valid |
| `01_remote_cool_hot_d0` | 990 | 962 | 28 | valid / valid |
| `02_combined_hot_d0` | 1,950 | 1,920 | 30 | valid / valid |
| `03_remote_cool_hot_d1` | 990 | 960 | 30 | valid / valid |
| `04_combined_hot_d1` | 1,950 | 1,920 | 30 | valid / valid |

v14의 global reject는 request별 preparation receipt와 함께 보인다. block별로
`prep_count/batch_count/refresh_failed/retry`는 각각

- `00`: `30/30/0/0`
- `01`: `30/4/26/26`
- `02`: `30/5/25/25`
- `03`: `30/4/26/26`
- `04`: `30/9/21/21`

이다. 즉 telemetry failure는 추상적인 “503”이 아니라 native overload 중
request-scoped refresh가 실패하고 retry한 control-plane 사건으로 영수증화됐다.

동시에 이것은 실제 vLLM/LMCache 부하였다. node-0/node-2 vLLM log에는
LMCache의 `Failed to allocate memory block ... because no memory is available`가
반복됐고, remote proxy의 최종 관측값은 node-0 `Num requests=954`, prefill TTFT
average/median/p99 `25939.39/29930.50/35248.25 ms`, node-2 `Num requests=2871`,
`18805.18/16286.84/35487.97 ms`였다. frontend에는 다수 HTTP 503과 502가 남았고,
router 종료 구간에는 runtime telemetry에 대한 `All connection attempts failed`가
남았다. 이것은 부하가 작아서 interconnect 문제가 안 보인 것이 아니라, 현재
workload에서 먼저 endpoint/LMCache memory와 telemetry control-plane이 무너진
것이다. 다만 sampled Cassini pause가 0인 상태이므로 이 receipt만으로 Slingshot
link saturation을 주장하지 않는다.

#### 핵심 구조적 결론

현재 C7 `full_c7`에서 background aggressor request ID는
`epd-remote-background...` 또는 `epd-local-background...`이고, router arm이
`remote`/`local`이다. 따라서 960/1,920개의 remote/local aggressor는
`tempo_go_request`가 아니며 global coordinator의 admission, tenant fairness,
route actuation, queue lease를 받지 않는다. TEMPO가 직접 제어한 것은 30개의
interactive victim뿐이다. `interactive_reserved`는 client executor starvation은
막았지만, controller 바깥의 exogenous background가 native endpoint/LMCache를
압박하는 구조 자체는 바꾸지 않았다.

이것은 workload를 약하게 만들어야 한다는 뜻이 아니다. 오히려 Perlmutter-scale
business 상황에서는 background/batch/interactive가 같은 global business budget과
resource envelope에 들어오고, background는 낮은 SLO/weight로 admission delay,
reject, route change를 받아야 한다. 그렇지 않으면 “victim만 orchestration”하고
실제 production load를 방치하게 된다. 따라서 v14는 contention 존재와
control-plane failure를 확정하지만 TEMPO 성능 우위의 비교 결과는 아니다.

#### v15에서 새로 고정한 managed-background arm

현재 source에 `full_c7_managed_background` arm을 추가하고 contract를 생성했다.

- contract:
  `eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v15.json`
  (SHA-256 `a6173a81c9b33e6514c765917ef6e730e69239524abf302ea48f68827a823e2c`)
- remote/local background request 모두 `epd-tempo-background-...` ID를 사용한다.
- client는 ID에서 business tenant를 복원해 `X-Tempo-Tenant-Id: background`를
  보낸다. 이는 phase/route/future arrival을 보내는 것이 아니라 profile의
  business identity만 전달한다.
- remote와 local background가 동일한 `background` tenant budget을 공유하므로,
  global coordinator가 두 경로를 비교해 route를 commit하거나 queue/reject할 수
  있다. 기존 `full_c7`의 exogenous fixed semantics는 변경하지 않는다.
- managed arm의 모든 admitted request는 global commit receipt를 가져야 하며,
  global reject는 victim에만 허용했던 v14와 달리 background에도 terminal reject로
  기록된다. 따라서 background throttling/rejection이 숨겨진 completion 누락으로
  변하지 않는다.

v15 qualification과 local schedule qualification은 통과했고 native allocation도
실행했다. 관련 unit gate는 `26 passed` (stream metrics, C7 analyzer, global
coordinator targeted suite)이며, py_compile과 `git diff --check`도 통과했다.

#### 다음 native gate — 성능 비교 전에 반드시 통과할 것

다음 승인된 4-node `gpu_interactive`에서 `full_c7_managed_background`만 먼저
실행한다. 목적은 속도 숫자를 만드는 것이 아니라, 같은 offered population을
global business admission으로 실제 제어할 수 있는지 확인하는 것이다.

1. v14와 동일한 32/s remote, 32/s local combined-hot load와 interactive 1/s를
   유지한다. request arrival, geometry, model, LMCache, NCCL/Slingshot transport,
   endpoint probe cadence를 바꾸지 않는다.
2. background의 admitted/rejected/queued 수, tenant virtual service, interactive
   SLO-goodput, route edge, local/remote actuation, telemetry preparation receipt를
   함께 수집한다. background reject가 증가하면서 interactive가 살아나는지가
   business control의 핵심이다.
3. LMCache allocation warning, endpoint waiting/running, NIXL/UCX bytes/ops,
   NCCL/Slingshot/Cassini supported counters를 같은 interval envelope에 넣는다.
   Cassini pause 0은 fabric cool의 증명이 아니며, unsupported signal은 0으로
   바꾸지 않고 missing으로 남긴다.
4. managed arm이 정확한 terminal contract를 만들면 같은 allocation budget에서
   fixed strongest, predictor, 기존 `full_c7`와 offered-population comparison을
   수행한다. 비교 지표는 completion-only p99가 아니라 interactive/business
   goodput, deadline miss, fairness, background service fraction, route actuation,
   normal regression, cross-layer attribution이다.

현재 목표는 그대로다. **실제 Perlmutter-scale contention에서 NCCL/Slingshot/
LMCache/vLLM/business 상태를 공동으로 관측하고, 모든 business class를 global
orchestrator가 admission·fairness·pair scaling·fabric-aware route로 조절해
고정 정책과 predictor보다 유의미한 goodput/SLO 개선을 보이는 것**이다. v14는
그 필요성을 더 강하게 입증했지만, managed-background가 검증되기 전에는
TEMPO positive 성능 결론을 내리지 않는다.

### 71.13 v15 managed-background native receipt: global control이 실제로 작동했지만 edge validator가 잘못 중단함

v15 managed arm을 allocation `57550515`에서 실행했다. allocation은 4-node
`gpu_interactive` guard를 통과했고, `00_control_a`와
`01_remote_cool_hot_d0`의 native raw/endpoint receipt를 만들었다. Slurm receipt는
`57550515|FAILED|143:0|00:06:49`, step `57550515.0|FAILED|1:0|00:06:30`이다.
이는 v14처럼 endpoint가 먼저 무너진 것이 아니라, block 01 augmentation validator가
첫 managed remote route를 잘못 해석해 exit 1로 끝난 것이다.

block 01은 다음과 같이 **global background admission이 실제 적용된 것**을
확정한다.

- offered 990, completed 665, explicit global reject 325
- reject reason 325건 모두 `global_admission_queue_timeout`
- completed 665건 모두 `tempo_go_global_commit_applied=true`
- route: local 479건, official remote 186건
- global decision은 remote request에서
  `frontend_pair_index`(assignment/endpoint field)와
  `tempo_go_global_commit_prefill_index`(canonical P source)가 다를 수 있음을
  보여줬다.

마지막 차이가 validator bug의 원인이다. managed branch가
`frontend_pair_index`를 global prefill source로 강제했기 때문에, 예를 들어 실제
commit `remote:p1->d0`를 `frontend_pair_index=0`으로 오인해 block을 중단했다.
이것은 workload의 rejection이나 LMCache failure가 아니다. v16에서 validator는
global commit receipt의 `tempo_go_global_commit_prefill_index`와
`tempo_go_global_commit_decoder_index`를 canonical edge source로 사용하도록
수정했고, old fixed/full arm의 의미는 바꾸지 않았다.

v15 partial root는 다음이다.

`results/tempo_go_c7_remote_activation_matrix_job_57550515_managed_background_v15/full_c7_managed_background/tempo_go_c7_joint_control/`

v15에서 확인된 설계적 의미는 명확하다. background를 global coordinator 밖에
두면 v14처럼 native endpoint/LMCache pressure와 telemetry reject가 victim에게
전가된다. background를 `background` tenant로 global admission에 넣으면, 같은
offered population에서 background queue/reject를 business policy로 흡수하면서
interactive를 보호할 수 있는 제어면이 실제로 생긴다. 다만 이번 partial run은
validator가 중단했으므로 managed arm의 full-block SLO-goodput 성능 결론은 아직
없다.

다음 native gate는 수정된 contract
`eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v16.json`
(SHA-256 `a98c4c1354eee682e03da7d232bc4fd321cfba1f169bcc6b4f0d037996585d78`)로
동일한 `full_c7_managed_background` arm을 재실행한다. v16 qualification은 source
inventory 23개와 managed arm을 확인했고 targeted unit suite `26 passed`다.
allocation이 끝난 뒤에는 partial raw를 performance result로 승격하지 않는다.

### 71.14 v16 managed-background full native receipt: global admission은 입증됐지만 interactive 보호 fairness가 아직 부족함

validator를 수정한 v16을 allocation `57552065`에서 같은 4-node
`gpu_interactive` native path로 실행했고, 전체 block 00–05와 arm analyzer가
정상 완료됐다. Slurm receipt는 allocation/step 모두 `COMPLETED|0:0`이며,
managed arm 결과는 다음에 있다.

`results/tempo_go_c7_remote_activation_matrix_job_57552065_managed_background_v16/full_c7_managed_background/result.json`

contract SHA는
`a98c4c1354eee682e03da7d232bc4fd321cfba1f169bcc6b4f0d037996585d78`이다.

#### v16 native 결과

| block | victim offered | victim complete | victim global reject | background complete | background global reject | terminal contract |
|---|---:|---:|---:|---:|---:|---|
| `00_control_a` | 30 | 30 | 0 | — | — | valid |
| `01_remote_cool_hot_d0` | 30 | 18 | 12 | 666 | 294 | valid |
| `02_combined_hot_d0` | 30 | 5 | 25 | 523 | 1,364 (+ service 33) | valid |
| `03_remote_cool_hot_d1` | 30 | 18 | 12 | 626 | 334 | valid |
| `04_combined_hot_d1` | 30 | 8 | 22 | 544 | 1,372 (+ service 4) | valid |
| `05_control_b` | 30 | 30 | 0 | — | — | valid |

전체 hot victim은 120개 중 49개만 completion, 71개 global reject이고, offered
SLO-goodput은 `37/120 = 0.3083`이다. managed arm hot victim p99 E2E는
`10103.94 ms`, normal p50 E2E는 `2923.61 ms`, 모든 block의 terminal
contract는 valid이며 stream failure는 0이다. 따라서 이 receipt의
`performance_claim_allowed`는 false다. 이 수치를 TEMPO 성능 우위로 포장하지
않는다.

대신 global business control 자체는 명확히 관측됐다.

- remote-cool D0에서 global admitted requests는 local 499, official remote 185로
  실제 route choice가 발생했다.
- combined-hot D0에서 local 375, remote 153, bounded service-lane failure 33이
  기록됐다.
- remote-cool D1은 local 473/remote 171, combined-hot D1은
  local 388/remote 164와 service-lane failure 4였다.
- background가 global coordinator에 들어간 뒤 대부분의 reject가
  `global_hierarchy_no_candidate` 또는 `global_admission_queue_timeout`으로
  명시화됐다. 이는 background를 endpoint로 무제한 밀어 넣지 않고 global
  admission에서 줄였다는 뜻이다.

#### v16이 보여준 현재 TEMPO의 부족한 점

현재 `minimum_service_fraction`과 `queue_reservation_slots`는 queue ordering과
queue slot 보호에 가깝고, 이미 admitted된 low-priority background가 pair/GPU/
endpoint capacity를 점유한 뒤 도착한 interactive를 위한 hard capacity reserve는
아니다. 그래서 interactive와 background 모두 global queue에 들어왔지만,
hierarchy frontier가 no-candidate가 되면 interactive도 함께 reject됐다. 이것이
v16의 핵심 negative다. 단순히 stale grace를 늘리거나 reject를 숨기는 방식으로
고치면 안 된다.

다음 scheme은 다음의 명시적 business/resource contract가 필요하다.

1. `interactive`/`latency`에 tenant-scoped protected capacity budget을 둔다.
   background는 이 reserve를 침범하지 않고, reserve가 idle일 때만 work-conserving
   하게 사용한다. 즉 queue reservation과 GPU/endpoint admission reserve를
   분리한다.
2. background가 no-candidate를 만나면 background만 defer/reject하고,
   interactive가 feasible candidate를 기다릴 수 있는 bounded protected lane을
   유지한다. 이때 interactive도 no-candidate라면 거짓 positive가 아니라
   명시적 capacity failure로 기록한다.
3. pair scaling은 “queue가 찼다”만 보지 않고 tenant protected demand,
   decoder active/waiting, LMCache KV memory, remote semantic-op, endpoint
   first-response credit을 함께 보고 spare pair activation을 판단한다.
4. analyzer는 victim-only p99가 아니라 tenant별 offered goodput, reject share,
   service fraction, protected-lane violation, background utility와 native
   endpoint/fabric envelope를 함께 gate한다.

#### v16 cross-layer evidence 경계

v16에서도 LMCache node-0/node-2 log에
`Failed to allocate memory block ... because no memory is available`가 반복됐다.
반면 endpoint evidence의 sampled maximum은 block별로 대략
`running=0–7`, `waiting=0`, KV usage `0.0–0.0273`, Cassini RX/TX pause `0`,
OXE active fraction 최대 약 `8.05e-4`였다. 따라서 managed global admission이
native endpoint waiting을 줄인 것은 보이지만, Cassini pause 0을 Slingshot/NCCL
fabric이 cool하다는 증명으로 해석하지 않는다. NCCL/Slingshot/NIXL byte·packet
및 supported NIC vector를 같은 interval에 추가하는 fabric gate는 여전히 남아
있다.

현재 결론은 다음처럼 정밀하게 고정한다.

> 현실적인 32/s remote + 32/s local + interactive 부하에서 contention과
> LMCache/vLLM memory pressure는 실존한다. background를 global business tenant로
> 편입하면 TEMPO가 실제로 route/admission/reject를 공동 제어한다. 그러나 v16은
> 현재 fairness/capacity reserve가 interactive를 보호하기에 부족해 성능 positive가
> 아니다. 다음 연구 기여는 protected business capacity와 cross-layer pair scaling을
> 추가해 background를 희생시키면서 interactive goodput을 보존하는 global
> orchestrator를 만드는 것이다.

따라서 다음 구현 gate는 `tenant-protected-capacity` unit/replay gate를 먼저 만들고,
그 뒤에만 새 native allocation을 사용한다. 새 allocation에서 다시 같은 managed
run을 반복하는 것은 아직 필요 없다. 현재 native evidence가 부족한 것이 아니라,
다음 정책 변경을 검증할 비교군과 reserve semantics가 부족한 상태다.

### 71.15 protected-business capacity 구현 및 v17 native gate 준비

v16에서 드러난 fairness 결손을 다음처럼 코드에 반영했다.

- `TenantPolicy`에 `admission_priority`와
  `protected_capacity_fraction`을 추가했다.
- lower-priority request의 immediate admission과 endpoint queue lease 모두
  higher-priority tenant의 reserve를 침범하면
  `tenant_protected_capacity_reserve`로 거절한다.
- 기존 profile과 기존 arm은 두 필드의 default `0`을 사용하므로 semantics가
  바뀌지 않는다. queue reservation은 queue slot 보호로, protected capacity는
  pair/GPU/endpoint resource 보호로 분리된다.
- snapshot/decision boundary에는 priority와 reserve를 남겨 business policy가
  실제 admission guard로 사용됐는지 확인할 수 있다.

새 profile과 contract는 준비됐다.

- profile:
  `results/tempo_go_c7_protected_business_profile_v6/real_tempo_go_c7_protected_business_profile_v6.json`
  (file SHA `e4c63169748626952a42e97264ed9c7e3cc6686f8a65a049c7efff09da278829`,
  fingerprint `b5860151ec2384ed6753e204a49ad0af2b851f737a93a43ee59b80928e7ca9e3`)
- policy: latency priority 1000, interactive 800, batch 400, background 0;
  latency/interactive protected capacity fraction 0.20, batch/background 0
- contract:
  `eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v17.json`
  (SHA `c6bec3de13b3079a65e761c0d97e6930ada3cd5d3f41660ba9c9ac64334ec45b`)
- v17 qualification: source inventory 23개와 managed arm round-trip 통과
- unit gate: global orchestrator/profile/coordinator `100 passed`, C7/stream
  targeted suite `26 passed`

v17 native에서는 v16과 동일한 offered population을 유지한다. 바뀌는 것은
business policy뿐이다. 성공 조건은 background reject를 줄이는 것이 아니라,
background가 reserve 밖에서만 admitted되면서 interactive hot offered
SLO-goodput과 reject share가 v16보다 유의미하게 개선되는 것이다. 동시에
background service fraction, local/remote route, LMCache memory warning,
endpoint queue, telemetry preparation, Cassini/NIC vector를 함께 비교한다.
reserve가 interactive를 보호하지 못하면 native positive가 아니며, reserve 때문에
capacity가 영구적으로 낭비되는지도 control/no-load block에서 확인한다.

### 71.16 v17 protected-business native receipt: reserve는 실제로 작동했지만 아직 충분한 보호가 아님

v17을 allocation `57553227`의 4-node `gpu_interactive` native path에서 실행했다.
5개 workload block의 raw/endpoint receipt는 모두 생성됐고, 실행 중 endpoint가
먼저 죽거나 allocation이 외부에서 끊긴 것은 아니다. 다만 마지막 analyzer가
`service_lane_failure` row를 `valid=true` completion으로 잘못 해석해
`completed victim lacks token arrivals`로 step을 exit 1시켰다. 이 row는
`HTTP 503`, `terminal_kind=service_lane_failure`,
`terminal_error_kind=endpoint_bounded_global_route_timeout`인 명시적 service
failure였다. analyzer를 고쳐 해당 row를 failure로 세고, 이미 생성된 native raw를
재분석해 arm receipt를 복원했다. 이 수정으로 raw evidence를 삭제하거나 성공으로
위장하지 않았다.

v17 arm receipt는 다음과 같다.

`results/tempo_go_c7_remote_activation_matrix_job_57553227_protected_business_v17/full_c7_managed_background/result.json`

실행에 사용한 contract는
`eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v17.json`
(SHA `c6bec3de13b3079a65e761c0d97e6930ada3cd5d3f41660ba9c9ac64334ec45b`)이고,
validator 수정 후 다음 실행용 contract v18도 새 source hash로 freeze했다.
v18 SHA는 `2227ad78c754a0526f10fc9d1264f336a15c4bec28331ca3d2b83a57ad354b2b`이다.

#### v17과 v16의 동일 offered population 비교

| metric | v16 managed background | v17 protected business | 해석 |
|---|---:|---:|---|
| hot victim offered | 120 | 120 | 동일 |
| hot victim completed | 49 | 49 | reserve만으로 capacity frontier는 아직 회복되지 않음 |
| hot victim global reject | 71 | 70 | reject 수 자체의 개선은 거의 없음 |
| hot victim service failure | 0 | 1 | bounded service lane까지 보호하지 못함 |
| hot SLO-good | 37/120 (0.3083) | 44/120 (0.3667) | 7건 개선, 약 18.9% relative 개선 |
| hot E2E p99 | 10,103.94 ms | 8,591.24 ms | 약 15.0% 감소 |
| normal victim p50 | 2,923.61 ms | 2,969.74 ms | 약 1.6% regression |

block별 victim 결과는 `00_control_a` 30/30, `01_remote_cool_hot_d0`
20 complete/10 reject, `02_combined_hot_d0` 4 complete/25 reject/1
service failure, `03_remote_cool_hot_d1` 19 complete/11 reject,
`04_combined_hot_d1` 6 complete/24 reject, `05_control_b` 30/30이다.
따라서 v17은 v16보다 지연 tail과 SLO-goodput이 개선되는 신호는 보였지만,
reject와 service failure가 남아 있고 same-population fixed/predictor 비교가
없으므로 TEMPO performance positive나 50%급 개선 결론은 아직 허용하지 않는다.

#### reserve가 실제 global orchestrator에 들어갔다는 증거

v17의 interactive victim reject에는
`global_hierarchy_no_candidate`, `global_admission_queue_timeout`이 남았고,
background reject에는 `tenant_protected_capacity_reserve`가 전부를 차지하는
형태가 아니라 hierarchy/queue/service failure가 함께 나타났다. 즉 현재 구현은
protected capacity를 candidate 평가와 queue lease에 전달했지만, background가
이미 endpoint bounded queue에 들어간 뒤의 service-lane credit과 pair-level
capacity를 완전히 분리하지 못했다.

실제 admitted victim route는 local 104, official LMCache remote 5였고, D0/D1
hot block 모두 local/remote edge가 살아 있었다. endpoint envelope의 최대치는
block별 running 2–4, waiting 0, KV usage 약 0.0079–0.0157, Cassini pause
0이었다. v16의 hot block 최대 running 3–7, KV usage 약 0.0118–0.0273보다
낮아진 것은 reserve가 pressure를 앞단에서 줄였다는 일관된 신호지만, Cassini
pause 0은 fabric이 한산했다는 증거가 아니다. 현재 endpoint evidence에서는
KV transfer inflight와 일부 NIC vector가 `not_collected`이므로, NCCL/Slingshot/
Cassini attribution gate는 아직 미완료다.

현재 연구 판단은 다음으로 갱신한다.

> tenant protected capacity는 필요한 방향이며 v17에서 hot SLO-goodput과 tail을
> 개선하는 첫 native 신호를 만들었다. 그러나 static reserve만으로는 부족하다.
> TEMPO의 다음 구현은 business priority를 endpoint service-lane credit,
> decoder pair scaling, remote KV/semantic-op budget, 그리고 supported fabric
> telemetry freshness와 함께 제어해야 한다. 그래야 background를 단순히
> reject하는 정책이 아니라, 같은 offered population에서 interactive goodput을
> 보존하면서 background service fraction도 정의하는 global orchestrator가 된다.

따라서 다음 순서는 (1) service-lane credit을 protected capacity와 동일한
tenant/resource ledger에 연결하고, (2) pair scaling과 telemetry freshness를
같은 admission transaction에 포함하고, (3) 같은 v18 contract로 managed arm을
재검증한 뒤, (4) 동일한 offered population에서 strongest fixed/predictor/
`full_c7`와 비교하는 것이다. 그 비교 전에는 수치 개선을 논문 결론으로 승격하지
않는다.

### 71.17 v19 service-lane preflight: global ownership과 endpoint physical credit의 순서를 닫음

v17 raw receipt의 핵심 단절은 static protected capacity가 아니었다. global
coordinator가 `endpoint_requests`를 보유한 뒤 frontend가 pair router로
forward했고, router의 실제 `EndpointFeedbackController`가 그 request를
`bounded_ingress_queue`로만 판단하는 순간까지도 global route commit이 이미
immutable하게 기록되어 있었다. 그 뒤 queue가 business deadline 안에 drain되지
않으면 `endpoint_bounded_global_route_timeout`이 HTTP 503으로 나타났다. 따라서
“global admission 성공”과 “endpoint가 실행 가능한 service-lane credit을
획득함”을 같은 사건으로 세면 안 된다.

이를 v19에서 다음의 request-scoped two-phase boundary로 수정했다.

```text
GlobalAdmissionCoordinator
  -> selected P×D edge / tenant / telemetry epoch
  -> pair router service_lane_preflight
       -> endpoint feedback score + physical resource fit
       -> first-response service-lane reservation
  -> immutable global pair×route commit
  -> /v1/completions
  -> first response / EOF / failure exactly-once release
```

- `ElasticPDRouterCore.preflight_global_service_lane()`은 provisional global
  header를 endpoint feedback 입력으로만 사용하고, 직접 route가 endpoint
  queue로 떨어지면 upstream을 시작하지 않고
  `tempo-go-service-lane-preflight-v1 / unavailable` receipt를 반환한다.
- endpoint가 실제 local/remote service credit을 잡은 뒤에만
  `_request_global_commits`에 immutable commit을 기록한다. 명시적인 global
  queue lease만 queue route를 계속 허용한다.
- frontend는 tokenized geometry와 remaining business deadline을 함께 보내고,
  preflight failure를 global service-lane failure/debt release로 연결한다.
  이미 예약된 endpoint가 upstream abort를 만나면 bounded abort RPC가
  endpoint credit을 해제한다. retry loop나 hidden local fallback은 없다.
- `/v1/completions`의 같은 commit header는 preflight가 만든 immutable receipt와
  일치할 때만 idempotently 재확인된다. 다른 route/header로 바꾸면 fail-closed한다.

이것은 router component의 latency 개선이 아니라 v17에서 끊긴
`global admission → endpoint physical credit → pair/fabric ownership`의
cross-layer transaction을 닫는 correctness/control mechanism이다. pair scaling과
telemetry freshness는 기존 global candidate/reducer의 same-epoch, all-pair,
stale/quarantine guard를 그대로 통과해야 하며, preflight 자체가 stale state를
새 capacity로 만들지 않는다.

v19 source-bound artifact와 CPU gate는 다음과 같다.

| item | value |
|---|---|
| contract | `eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v19.json` |
| contract source inventory | 23 files; current router/frontend/orchestrator SHA verified |
| profile | v6 protected-business profile, unchanged |
| performance claim | `false` until native full-valid comparison |
| focused CPU/control suite | `80 passed` |
| syntax/integrity | Python 3.12 module `py_compile`, `git diff --check` passed |

기본 login Python 3.6은 `from __future__ import annotations`를 지원하지 않아
사용하지 않았고, system Python 3.11은 `torch/httpx`가 없어 router test
collection을 완주하지 못했다. Perlmutter 공식 `pytorch/2.8.0` Python 3.12
환경에서 global/profile/telemetry/hierarchy/cross-layer/stream/analyzer/static
gate를 실행했다. `httpx`가 포함된 vLLM router runtime이 없는 login 환경에서
임의 package 설치를 하지 않았다. 이 환경 경계는 native allocation의
capability receipt에 별도로 기록한다.

현재 native GO gate는 v19에서 고정한다.

1. 승인된 한 개의 4-node `gpu_interactive` allocation 안에서 P1 capability와
   `Network=job_vni`, 4 GPU/node, 128 CPU/task, official NCCL AWS Libfabric와
   LMCache/NIXL/UCX를 확인한다.
2. 같은 allocation에서 moderate co-job observer가 C7의 모든 managed-background
   block을 덮는지 확인한다. observer가 먼저 끝나거나 step/interconnect가
   실패하면 성능 숫자를 만들지 않는다.
3. v19 `full_c7_managed_background`를 동일 120 hot-victim/동일 background
   offered population으로 실행한다. service-lane preflight가 direct queue
   timeout을 줄이는지, background reserve·interactive SLO·route edge·pair
   scaling·telemetry freshness를 함께 확인한다.
4. managed arm이 full terminal-valid이면 같은 offered population에서 strongest
   fixed/predictor/queue-GPU/APP_GLOBAL_ONLY/NETWORK_REQUEST_ONLY와 비교한다.
   completion-only p99나 reject를 버린 subset은 사용하지 않고 offered
   SLO-goodput, output-token goodput, p99, fairness와 failure cost를 계산한다.
5. v19에서 service-lane failure가 줄어도 fixed/predictor 대비 사전 gate를
   통과하지 못하면 coefficient를 다시 맞추지 않는다. 그 경우 service-lane
   preflight의 mechanism positive와 whole-system utility negative를 분리해
   기록하고, 다음 단계는 measured pair/service completion capacity의
   global scaling 여부를 판정한다.

### 71.18 v23--v27 correctness closure와 동일 population matrix

v19 이후 native 실패를 단순 재실행하거나 coefficient tuning으로 처리하지 않고,
request identity부터 endpoint credit까지의 cross-layer transaction을 순서대로
닫았다.

- global request identity가 HTTP header를 거치며 integer/string으로 달라져
  immutable commit이 HTTP 409를 내던 문제를 canonical string identity로
  통일했다.
- service-lane preflight가 endpoint ownership을 잡은 뒤 실패 경로에서 credit을
  남기던 leak을 exactly-once release로 닫았다.
- endpoint queue를 허용할 때 global coordinator와 pair router가 서로 다른 queue
  사실을 기록하던 경계를 queue negotiation, provisional lease, promotion receipt의
  two-phase transaction으로 바꿨다.
- official LMCache namespace를 prompt text가 아니라 tokenizer가 만든 prompt token
  hash에 고정해 prefill/decoder가 같은 KV object를 가리키도록 했다.
- zero-completion block과 `service_lane_failure`를 analyzer가 정상 terminal로
  오인하던 문제를 수정했다. reject/failure를 completion latency에서 숨기지 않고
  offered-population SLO-goodput에 그대로 포함한다.

관련 focused suite는 87개, 기존 vLLM runtime을 사용한 broad suite는
`228 passed, 22 subtests passed`였다. 별도 package 설치나 root/UDI 설정 변경은
없었다.

allocation `57556218`의 4-node `gpu_interactive`
(`nid[001176-001177,001180,008368]`)에서 v27 native managed run은 victim
180/180을 실제 vLLM/official LMCache 경로로 끝냈다. 그러나 hot victim은
118/120 SLO-good, E2E p99 9,116.70 ms였고 normal p50은 2,931.24 ms였다.
같은 allocation과 같은 180-victim population의 matrix는 다음 구조적 사실을
보였다.

| arm | hot SLO-good / 120 | hot E2E p99 | normal E2E p50 |
|---|---:|---:|---:|
| fixed local D0 | 97 | 9,610.41 ms | 2,980.00 ms |
| fixed local D1 | 97 | 8,904.58 ms | 3,120.18 ms |
| fixed remote P0->D1 | 89 | 15,599.06 ms | 3,210.00 ms |
| fixed remote P1->D0 | 44 | 21,234.76 ms | 3,095.41 ms |
| predictor | 106 | 9,298.37 ms | 2,867.44 ms |
| queue-GPU | 119 | 7,690.14 ms | 2,918.39 ms |
| network-request-only | 101 | 8,789.06 ms | 2,912.04 ms |
| app-global-only | 56 | 4,035.21 ms (completion subset) | 2,942.11 ms |
| v27 full C7 unmanaged background | 47 | 4,091.08 ms (completion subset) | 3,026.12 ms |

`app-global-only`와 unmanaged full의 짧은 completion p99는 다수 reject 뒤 남은
subset의 값이므로 성능 positive가 아니다. v27 matrix wrapper는 unmanaged full의
zero-completion block을 분석하다 멈췄지만 raw는 보존됐고, analyzer 수정 후
hot offered 120, complete/SLO-good 47, global reject 73으로 복원했다.

### 71.19 왜 v27 managed가 queue-GPU보다 나빴는가: 부하가 아니라 orchestration target 문제

v27의 실패 원인은 "Perlmutter에서 contention이 생기지 않는다"거나
"LMCache가 문제를 만들지 못한다"가 아니었다. 실제 문제는 managed background를
global coordinator에 넣고도 두 decoder pair에 분산시켜, 보호해야 할 cool pair까지
background가 점유하게 만든 것이었다.

- combined-hot D0에서 victim은 D0 20건, D1 10건으로 갈라졌고 p99가
  9.117초였다.
- combined-hot D1에서는 victim이 D0/D1 15건씩 갈라졌고 p99가 9.621초였다.
- 반면 phase마다 반대편 decoder를 고정한 local arm에서는 같은 종류의 request가
  약 3.0--3.3초에 끝났다.

즉 capacity는 있었지만, 기존 global policy가 low-priority tenant를 모든 pair로
spread한 뒤 request-level minimum score만 고르는 바람에 clean capacity를 스스로
오염시켰다. queue-GPU가 더 좋았던 것도 queue 한 숫자가 충분해서가 아니라,
일부 victim을 hot receiver 밖으로 보낸 우연한 isolation 효과였다. 필요한 제어는
static coefficient 조정이 아니라 business priority에 따른 pair packing과 clean-pair
isolation이었다.

### 71.20 v28 tenant-aware pair packing/isolation 구현

global orchestrator에 다음 semantics를 추가했다.

1. `TenantPolicy.pair_spread_limit`으로 low-priority tenant가 한 busy epoch에
   사용할 pair 수를 제한한다. v28의 `background`는 1개 pair에 packing한다.
2. assignment는 route commit, first response, EOF/failure의 request lifecycle 동안
   유지하며, 해당 tenant가 idle인 상태가 `scale_down_idle_ns`를 넘을 때만
   만료한다.
3. higher-priority request에 clean pair가 있으면 lower-priority tenant가 점유한
   candidate를 `higher_priority_clean_pair_available` /
   `tenant_pair_isolation`으로 제외한다.
4. 필요하면 inactive clean pair도
   `tenant_protected_pair_activated_and_route_committed`로 활성화한다.
5. ordinary admission과 endpoint queue lease가 같은 assignment/isolation ledger를
   사용하고, snapshot에 tenant-to-pair assignment를 남긴다. 따라서 frontend
   route만 바꾼 것이 아니라 business policy, global admission, pair scaling,
   endpoint service credit을 한 transaction에서 제어한다.

source-bound profile은
`results/tempo_go_c7_tenant_pair_packing_profile_v7/real_tempo_go_c7_tenant_pair_packing_profile_v7.json`
(fingerprint `12c6fef39c7f86291abeaf48345382c8061dfb6522c279707467dc1c8eec582e`,
file SHA `b67a180611a45a15bcea43fca23337b9f21181e82d6189535eb54f59bfb6ff44`)이고,
v28 contract는
`eval/sota_4node/tempo_go_c7_remote_activation_matrix_contract_v28.json`
(SHA `cbde14c68e4f23fd08ada66b320ee6ac842e81880a3f19b29390b8907d7c71ff`)이다.
contract의 source inventory와 profile SHA gate는 native 실행 전에 통과했다.

### 71.21 v28 실제 4-node native 결과: 첫 50%급 robustness positive

v28 `full_c7_managed_background`를 allocation `57556218`에서 실제 vLLM과
official LMCache native path로 실행했다. 결과는 다음 파일에 고정돼 있다.

`results/tempo_go_c7_joint_control_job_57556218_v28_pair_packing/full_c7_managed_background/result.json`

모든 block의 terminal contract가 valid였고 결과는 다음과 같다.

| population | offered | complete | global reject | failure | SLO-good | E2E p50 | E2E p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| normal victim | 60 | 60 | 0 | 0 | 60 | 2,966.24 ms | 3,061.40 ms |
| hot victim | 120 | 119 | 1 | 0 | 119 | 3,034.71 ms | 3,230.71 ms |
| all victim | 180 | 179 | 1 | 0 | 179 | 3,007.77 ms | 3,230.71 ms |

hot TTFT p99는 318.72 ms였다. 네 hot block의 completion은 각각 30, 30, 30,
29건이고 block p99도 3,175.92, 3,198.29, 3,230.71, 3,320.66 ms로
안정됐다. admitted victim은 local D0 37건, local D1 142건이었다. raw decision
receipt에는 contaminated D0 candidate가
`higher_priority_clean_pair_available`로 제외되고 D1 clean pair가 선택된 사실이
남아 있다.

background를 전부 버려 victim만 좋게 만든 결과도 아니다. background offered
1,404건 중 1,223건(87.1%)이 complete했고, global reject 152건, service-lane
failure 29건이었다. 다만 background utility/fairness 자체는 후속 multi-tenant
frontier에서 별도 최적화해야 한다.

v28 뒤 같은 allocation에서 predictor를 다시 실행해 시간 순서 drift를 줄였다.
adjacent predictor는 hot SLO-good 103/120, hot p99 8,593.77 ms, normal p50
2,913.17 ms였다. 따라서 v28은 predictor 대비:

- hot p99 62.4% 감소,
- hot offered SLO-goodput 103 -> 119, 즉 15.5% 증가,
- normal p50 regression 1.82%였다.

사전 robustness 조건인 "hot p99 15% 이상 감소 또는 SLO-good ratio 1.15 이상,
normal p50 regression 3% 이하"를 통과한다. 같은-allocation comparison analyzer의
추가 결과는 다음과 같다.

| comparator | hot p99 reduction | hot SLO-good ratio | normal p50 regression |
|---|---:|---:|---:|
| strongest fixed: local D1 | 63.7% | 1.227 | -4.93% |
| adjacent predictor | 62.4% | 1.155 | 1.82% |
| queue-GPU | 58.0% | 1.000 | 1.64% |

따라서 correctness, cross-layer incremental, strongest-fixed robustness,
predictor robustness, queue-GPU robustness gate는 모두 `true`다. 이는 작은 1--5%
tuning signal이 아니라, realistic co-load에서 business-aware pair isolation이 tail을
절반 이상 줄인 whole-system robustness positive다. pooled median을 개선했다는
주장은 하지 않는다. 이 실험의 가치는 정상 시 median이 아니라 contention 시
tail/SLO collapse를 막은 데 있다.

comparison receipt는
`results/tempo_go_c7_joint_control_job_57556218_v28_pair_packing/same_allocation_comparison_analysis.json`
이며 source result의 절대 경로와 SHA를 모두 포함한다.

### 71.22 현재 claim boundary와 다음 GO gate

v28은 TEMPO 전체 목표를 완료한 최종 논문 claim은 아니다. analyzer가
`performance_claim_allowed=false`로 남긴 이유는 성능 비교 실패가 아니라 다음 두
route/mechanism gate가 아직 false이기 때문이다.

- `full_uses_both_local_and_remote=false`: 이번 C7 cold-miss/decoder-contention
  regime에서 victim 179건은 모두 local이었다. remote fixed arm이 15.6--21.2초로
  나빴으므로 remote를 피한 판단은 합리적이지만, remote가 유리한 regime에서
  official LMCache remote path를 선택·성공시키는 능력은 아직 검증하지 않았다.
- `full_switches_away_from_hot_receiver=false`: 기존 analyzer는 workload가 붙인
  exogenous `hot_decoder_index`만 본다. v28은 background 자체를 한 pair로 이동해
  실제 contaminated pair가 이 label과 달라질 수 있으므로 이 old gate는 새
  business-isolation receipt를 표현하지 못한다. 그렇다고 gate를 사후 삭제하거나
  true로 바꾸지 않는다.

또한 v27/v28 비교는 같은 allocation의 discovery evidence다. 다른 allocation/
seed에서 독립 재현하지 않았으므로 `independent_validation_claim_allowed=false`다.
현재 정확한 결론은 다음과 같다.

> Perlmutter 4-node 실제 vLLM/LMCache contention에서 문제는 재현됐고,
> request-level routing만으로는 clean capacity를 보존하지 못했다. TEMPO의
> business-aware global pair packing/isolation은 같은 offered population에서
> strongest fixed, adjacent predictor, queue-GPU보다 hot tail을 58--64% 줄이고
> SLO-goodput을 보존했다. 그러나 이번 regime은 remote avoidance 증거이며,
> NCCL/Slingshot/LMCache remote activation까지 포함한 최종 cross-layer claim은
> 아직 열려 있다.

다음 실행은 v28을 반복하거나 coefficient를 다시 맞추는 것이 아니다.

1. 현재 C7을 `compute/KV pressure + pair isolation` regime으로 freeze한다.
2. 동일 4-node topology에 long shared prefix를 사전 warm하고 local decoder를
   독립적으로 압박해, remote prefill의 transfer cost보다 local queue/compute
   cost가 커지는 `remote-favorable` regime을 추가한다. official LMCache hit와
   transferred-token receipt가 없으면 remote 성공으로 세지 않는다.
3. supported NCCL/libfabric/Cassini/Slingshot counter의 수집 가능성·freshness를
   capability receipt로 먼저 고정하고, 미수집 값을 0으로 대체하지 않는다.
4. 두 regime를 한 campaign에 넣어 같은 global policy가 C7에서는 clean local
   pair를 보호하고 remote-favorable block에서는 remote edge를 활성화하는지
   검증한다.
5. frozen contract로 새 allocation/seed에서 strongest fixed, adjacent predictor,
   queue-GPU, full TEMPO를 다시 실행한다. 그때 route diversity, actual
   contaminated-pair avoidance, background utility, normal-load regression까지
   모두 통과해야 최종 `performance_claim_allowed`와 independent validation을
   true로 승격한다.

그러므로 현재 우선순위는 "orchestration을 버릴지"가 아니다. v28에서 global
orchestration의 큰 효과는 이미 확인됐다. 남은 연구 문제는 그 control law를
local protection과 remote activation 두 regime 모두에서 검증하고, supported
fabric telemetry를 의사결정 원인으로 귀속시키는 것이다.

### 71.23 C8가 오래 막힌 실제 이유: remote transfer가 아니라 running decode service 제어 부재

v28 뒤의 C8 시도는 "Perlmutter interconnect contention 문제가 없다"는 negative가
아니었다. remote path를 활성화한 뒤에도 이미 decoder에서 실행 중인 low-priority
local prefill/decode가 non-preemptive하게 GPU service를 점유했는데, 당시 global
orchestrator는 route와 admission만 결정하고 이 running set 앞의 service 순서를
보호하지 못했다.

- v32 managed는 두 decoder의 running 8, waiting 68--69 상태에서 global reserve가
  모든 C8 victim을 reject해 remote completion이 0/30이었다.
- v33 `fixed_remote_p0d1`은 30/30을 실제 remote로 보냈지만 C8 SLO-good은 0/30,
  p99는 37.16초였다. 따라서 LMCache/NIXL transfer 성공만으로는 decoder tail을
  해결하지 못한다.
- v37은 priority lane으로 6건의 exact LMCache full hit를 만들었지만 나머지 24건을
  reserve에서 막았다.
- v38 managed는 C7 miss-hot 120/120, p99 3.318초를 유지했지만 C8은 6/30만
  complete했고 completion-subset p99도 53.850초였다. vLLM priority가 waiting/TTFT는
  앞당겨도 이미 running인 background를 선점하지 못해 128-token decode TPOT가
  약 402 ms로 무너졌다.

동시에 global telemetry agent가 모든 endpoint fetch timeout을 pair별 장애와 동일하게
취급해 all-pair quarantine snapshot을 설치하는 control-plane bug가 있었다. data-plane
saturation 중 telemetry batch 하나가 실패하면 정상적인 last-complete snapshot까지
버려 orchestration이 스스로 모든 경로를 닫았다. 그러므로 필요한 수정은 단순 score
coefficient가 아니라 `business admission -> decoder running-set bound -> priority service
lane -> exact remote affinity -> telemetry stale grace`의 한 control transaction이었다.

### 71.24 v39 whole-system 수정: business decoder gate와 remote priority lane의 결합

v39에는 다음 제어를 실제 frontend/global coordinator/router/vLLM 경로에 결합했다.

1. 모든 endpoint fetch가 실패하면 새 all-pair quarantine을 만들지 않고 마지막
   complete telemetry batch를 보존한다. partial failure만 해당 pair를 quarantine하며,
   이후 tenant-scoped stale grace가 적용된다.
2. `vllm_priority_remote_cache_v1` global priority lane은 high-priority, opt-in,
   exact P_ONLY cache-affinity remote request에만 허용한다. decoder당 capacity는 8,
   실제 vLLM priority는 -2이며 ordinary completion/waiting reserve만 우회한다.
   physical capacity, fabric health, deadline, exact source/destination/edge gate는 그대로다.
3. `priority_drain_v1` decoder business admission gate는 decoder의 active-sequence 16에서
   protected lane 8을 뺀 8개만 background upstream으로 허용한다. interactive/latency
   tenant는 protected로 즉시 admit하고, protected request가 active인 동안 새 background는
   frontend에서 기다린다. background는 drop하지 않으며 60초 starvation escape를 가진다.
   lease는 HTTP EOF/failure까지 유지된다.
4. frontend가 business tenant를 pair router까지 전달하고, route decision에 protected/
   background admission과 held/released receipt를 남긴다. global queue lease는 replicated
   prefill source identity, destination decoder, canonical edge를 끝까지 보존한다.
5. C8 verifier는 frontend destination과 LMCache가 보고한 actual source를 분리해 검증한다.
   remote completion은 official LMCache full-source hit, 4,094 cached tokens, committed
   destination/decoder/edge가 모두 맞아야 성공이다.

C8 workload는 모든 arm에서 동일하다. C7의 네 30초 miss-hot block과 두 30-request
control block에 더해, remote-favorable block에서 decoder별 22.4 req/s, 합계 44.8 req/s의
4,094-token local miss aggressor를 넣고, 4,094-token P_ONLY prefix + 128 output-token victim
30건을 제공한다. owner별 8개, 총 16개 P_ONLY pool은 measurement 밖에서 physical
preseed한다. ingress는 16 interactive-reserved worker와 240 background worker를 사용한다.

### 71.25 v39 첫 dual-regime breakthrough

allocation `57570808`의 4-node `gpu_interactive`
(`nid[001176-001177,001180-001181]`)에서 v39 managed는 다음 결과를 냈다.

- C7 miss-hot: offered/complete/SLO-good 120/120/120, E2E p99 3,278.48 ms.
- C8 remote-favorable: offered/complete/SLO-good 30/30/30, E2E p50
  3,102.05 ms, p99 3,179.09 ms, TTFT p99 310.32 ms, TPOT p99 22.69 ms.
- C8 route: official LMCache remote 25, local 5. remote 25/25는 exact full-source
  hit와 vLLM priority -2/priority-lane receipt를 가졌다.
- 같은 allocation의 predictor는 C7 99/120, p99 8,951.61 ms, C8 13/30,
  p99 50,839.18 ms였다. 따라서 managed는 predictor 대비 C7 p99 63.4%,
  C8 p99 93.7%를 줄였다.
- v38 대비 C8 completion은 6 -> 30, SLO-good은 0 -> 30, p99는
  53.850초 -> 3.179초로 약 94.1% 감소했다.

v39 managed receipt는
`results/tempo_go_c8_dual_regime_job_57570808_v39_managed/full_c7_managed_background/result.json`
이다. 이는 route-only tuning이 아니라 business decoder gate가 running-set saturation을
막고 global remote lane이 그 보호된 capacity를 실제 LMCache path에 연결한 결과다.

### 71.26 queue-GPU preseed bug와 v40 source-bound 수정

same-allocation matrix를 완성하던 중 queue-GPU arm은 measured C7을 끝낸 뒤
`C8 P_ONLY probe lacks exact replicated source-hit evidence`로 fail-closed했다. 16개 probe
모두 4,094-token LMCache full hit였지만 queue-GPU가 measurement 밖의 physical preseed
probe까지 순간 queue가 짧은 D1으로 보내 owner-0 request의 physical owner receipt를
깨뜨렸다. 이는 LMCache failure가 아니라 harness topology-setup bug였다.

v40은 C8 measured queue policy를 바꾸지 않고 physical warm/seed request만 다음처럼
수정했다.

- queue-GPU physical seed ID도 owner-pin 대상으로 분류한다.
- `pair_pin_preferred`와 queue-GPU selection은 상호배타적이다.
- receipt에 `physical-p-only-seed-owner-pin-v1`을 기록한다.

실제 4-node 재실행에서 16/16 probe가 owner 0 -> pair 0, owner 1 -> pair 1로 갔고,
모두 physical pin true와 4,094/4,094 LMCache full hit를 남겼다. v40 contract는
`eval/sota_4node/tempo_go_c8_dual_regime_contract_v40.json`
(SHA `2552097345519ec0d61b5484cf3cea26c1c1c9fe6da90ebf8e70e826d2dc94c7`,
source inventory 32/32)이다. global profile은
`results/tempo_go_c8_priority_service_lane_profile_v10/real_tempo_go_c8_priority_service_lane_profile_v10.json`
(file SHA `353056c568bec92c2ca0cb06bb4b7497990f1bccd8781c739792dfe36a720d01`,
fingerprint `e6e160daf66773cbc2a3038b65b1b40c49a55ef736cc0efcebc508f066fe5deb`)이다.

### 71.27 v40 같은-allocation 7-arm native matrix

allocation `57576758`의 4-node `gpu_interactive`
(`nid[001156-001157,001160-001161]`)에서 v40의 일곱 arm을 같은 contract와 offered
population으로 실행했다. root/container/UDI나 batch/debug job은 사용하지 않았다.

| arm | C7 miss-hot SLO-good / 120 | C7 p99 | C8 SLO-good / 30 | C8 p99 | C8 remote / local |
|---|---:|---:|---:|---:|---:|
| fixed local D0 | 97 | 10,249.15 ms | 0 | 36,734.17 ms | 0 / 30 |
| fixed local D1 | 97 | 9,853.98 ms | 0 | 36,155.50 ms | 0 / 30 |
| fixed remote P0->D1 | 79 | 18,308.26 ms | 0 | 37,713.24 ms | 30 / 0 |
| fixed remote P1->D0 | 44 | 20,120.32 ms | 0 | 38,009.22 ms | 30 / 0 |
| predictor | 97 | 9,040.40 ms | 13 | 50,646.13 ms | 0 / 30 |
| queue-GPU | 118 | 8,006.84 ms | 14 | 50,516.09 ms | 18 / 12 |
| v40 full managed | **120** | **3,260.40 ms** | **30** | **3,385.06 ms** | **24 / 6** |

managed의 normal control p50은 2,893.10 ms이고 normal 60/60도 모두 SLO-good이다.
analyzer가 고른 strongest fixed는 두 regime 모두 fixed local D1이다. managed 효과는:

- strongest fixed 대비 C7 p99 66.9% 감소, SLO-good ratio 1.237;
- strongest fixed 대비 C8 p99 90.6% 감소, 0/30 -> 30/30;
- predictor 대비 C7 p99 63.9% 감소, SLO-good ratio 1.237;
- predictor 대비 C8 p99 93.3% 감소, SLO-good 13 -> 30;
- queue-GPU 대비 C7 p99 59.3% 감소, C8 p99 93.3% 감소,
  C8 SLO-good 14 -> 30.

managed C8에서는 background 1,344/1,344가 complete했고 reject/drop/HTTP error는 0이다.
victim 30/30은 protected decoder admission receipt를 가졌으며 최대 gate wait는
2,034 ns였다. remote 24/24는 priority service lane과 exact 4,094-token LMCache
full-source hit를 가졌다. 즉 foreground positive는 background 삭제나 completion subset
선별로 만든 것이 아니다. C7 background도 1,404 offered 중 1,199 complete(85.4%),
195 global reject, 10 service-lane failure였으므로 background utility 비용까지 receipt에
남아 있다. 이 fairness/utility frontier는 후속 독립 검증에서 함께 최적화한다.

headline result는
`results/tempo_go_c8_dual_regime_job_57576758_v40_managed_retry1/full_c7_managed_background/result.json`
(SHA `055d97e1f2da76fc82cb532291f4c57f2a46483413a9c83e11994d34170e5b22`)이고,
campaign analysis는
`results/tempo_go_c8_dual_regime_job_57576758_v40_campaign_analysis.json`
(SHA `9d0bc958844eb865d47c6957aef0b11534df99c6be47a04fbd3a9e9d000c81a1`)이다.
현재 v40 32-source identity 검사와 historical C5 fail-closed 검사를 포함한 관련
global/C8/frontend/router/C5 regression은 `258 passed, 22 subtests passed`다.

한 managed 시도는 498-request combined-hot block의 단일 frontend->router 연결이 빈
HTTP 502로 끝나 terminal contract가 false였다. vLLM/NCCL/LMCache error는 없었고
497/498 decision이 존재했다. source를 바꾸지 않은 retry가 해당 block과 전체 campaign을
완주했으므로 이 시도는 숨기지 않고 transient execution receipt로 남긴다.

### 71.28 v40 gate 판정과 현재 claim boundary

frozen analyzer의 gate는 다음과 같다.

- correctness/native terminal contract: true
- C7 strongest-fixed robustness: true
- C7 predictor robustness: true
- C8 predictor robustness: true
- C8 best-fixed non-inferiority: true
- C8 remote activation fraction: 24/30 = 0.80, true
- exact official LMCache full-hit: true
- global priority service lane: true
- decoder business admission: true
- full uses both local and remote: true
- cross-pair remote: 0/24, required 10%, **false**

따라서 `c8_dual_regime_discovery_positive=false`,
`performance_claim_allowed=false`, `independent_validation_claim_allowed=false`를 그대로
유지한다. 성능 positive를 축소할 이유는 없지만, 모든 gate를 통과한 최종 논문 claim으로
과장해서도 안 된다. 현재 정확한 결론은 다음과 같다.

> Perlmutter 4-node 실제 vLLM/official-LMCache co-load에서 local-only, remote-only,
> predictor, queue-GPU는 decoder service contention 때문에 C8 p99 36--51초와
> 0--14/30 SLO-good으로 붕괴했다. TEMPO의 business pair isolation, decoder admission,
> remote priority lane, exact cache affinity, stale telemetry 제어를 결합하면 같은
> background를 모두 완료하면서 C7/C8를 모두 3.3초 안팎, 100% SLO로 보호한다.
> 다만 이번 최적 경로는 remote source P0에 집중돼 cross-pair fabric diversity를 아직
> 증명하지 않았다.

### 71.29 다음 GO gate: 강제 percentage가 아닌 fabric-aware remote-source 분산

다음 단계에서 10% 수치를 맞추기 위해 request ordinal로 cross edge를 강제하거나 gate를
사후 낮추지 않는다. 현재 receipt에서 `remote:p1->d0`은 feasible했지만
`higher_global_score`로 탈락했고, 24건은 모두 `remote:p0->d0`이었다. 따라서 다음
global-orchestrator 작업은 replicated cache owner가 여러 개일 때의 source/edge 선택을
명시적으로 만드는 것이다.

1. rejected remote candidate에도 score decomposition, source-prefill owned work,
   edge inflight, telemetry sequence/freshness를 남겨 P0/P1 차이가 real fabric signal인지
   deterministic tie인지 증명한다.
2. score가 uncertainty/near-tie 범위 안에 있을 때만 source-prefill credit과 edge load를
   tie-break로 사용한다. 더 나쁜 cross edge를 quota 때문에 강제하지 않는다.
3. Cassini/Slingshot/libfabric/NCCL capability receipt가 있는 경우 source/edge pressure를
   tie-break 원인으로 사용하고, unavailable counter를 0으로 대체하지 않는다.
4. 새 frozen contract에서 managed가 C7/C8 100% SLO와 3초대 tail을 유지하면서 실제
   cross-pair edge를 선택하는지 먼저 검증한다. 그 뒤 일곱 arm 전체를 새 allocation/seed로
   반복해 independent validation을 수행한다.
5. background utility, tenant별 goodput/Jain fairness, pair scale-up/down, telemetry overhead,
   4-node보다 큰 hierarchical control-plane replay를 함께 보고 production-HPC claim으로
   확장한다.

현재 우선순위는 orchestration을 폐기하는 것도, LMCache만 더 세게 부하시키는 것도 아니다.
v40은 문제와 50%를 훨씬 넘는 whole-system 개선을 이미 재현했다. 남은 핵심은 이 성능을
유지한 채 fabric-aware source diversity와 independent allocation validation을 통과하는
것이다.

### 71.30 C9 구현: telemetry uncertainty 안에서만 source/edge virtual service

v40의 마지막 false gate는 remote transfer가 없어서가 아니라, 동일 prefix replica를 가진
여러 prefill source가 score상 사실상 동률인데도 정적 tie가 P0 하나로 굳는 문제였다. C9는
이를 request ordinal이나 cross-edge quota로 강제하지 않고 global orchestrator의 실제
제어 상태로 해결한다.

1. `telemetry_uncertainty_virtual_service_v1`은 두 후보가 모두 remote이고, 같은 decoder,
   같은 exact cache affinity, 같은 priority-lane/completion/debt 상태, 같은 work vector를
   가지며 score 차이가 두 후보 uncertainty의 최솟값에 frozen fraction을 곱한 범위 안일
   때만 source-balance peer로 인정한다.
2. 기존 semantic/service/score 정렬의 첫 후보를 static anchor로 유지한다. near-tie 집합
   안에서만 controller-owned source-prefill virtual finish와 P_i->D_j edge virtual finish가
   작은 후보를 고른다. 선택 결과가 다음 request의 상태가 되므로 request ID나 사후 비율은
   사용하지 않는다.
3. remote reserve 시 source/edge virtual service를 charge하고 first response 뒤 physical
   credit을 release해도 virtual service ledger는 유지한다. 따라서 동시 요청과 순차 요청
   모두 실제 누적 service를 보며, 더 나쁜 route를 quota 때문에 선택할 수 없다.
4. decision에는 score window/delta, source/edge service-before와
   `mesh_telemetry_uncertainty_source_virtual_service` binding을 남긴다. 탈락 후보에도
   evaluated score, signed delta, uncertainty, near-tie eligibility를 남겨 business/priority
   의미가 숫자 score보다 먼저 적용된 경우까지 설명한다.

global profile v11은
`results/tempo_go_c8_priority_service_lane_profile_v11/real_tempo_go_c8_priority_service_lane_profile_v11.json`
(file SHA `35dc9b916ddc862c17dd054c90225b01bfc535c27eabd2e3106350a417710eaf`,
fingerprint `cc07338db2b124b585132e622effa77e2afbc607bd1baaaca0c7b9888d90c994`)이다.
최종 source-bound contract는
`eval/sota_4node/tempo_go_c8_dual_regime_contract_v45.json`
(SHA `1521d855b8dbddde58afff0a92050969123c0218004288b33b756498f88ca260`,
source inventory 33/33)이다. contract는 near-tie가 route quota가 아님을 명시하고,
cross-pair completion마다 causal source-balance receipt를 요구한다.

### 71.31 실행 경계 수정: reattach guard, fail-closed wrapper, no-shell allocation

이번 실행에서 성능과 무관하지만 반복 실험을 깨뜨리던 세 경계를 닫았다.

- reattached overlap step에서는 `SLURM_JOB_NUM_NODES`와 `SLURM_JOB_NODES`가 둘 다
  비어 있을 수 있다. C8 wrapper의 낡은 문자열 비교를 제거하고, 한 번의 bounded
  `scontrol show job` 영수증으로 RUNNING, gpu_interactive, 4:00:00, NumNodes=4,
  gres/gpu=16을 확인하는 mandatory native guard를 source했다. guard 자체도 contract
  source inventory에 포함했다.
- guard가 `$(set +o)`로 caller option을 저장하면 command substitution 안에서
  `errexit`가 꺼진 상태를 캡처해 wrapper의 `set -e`를 무효화했다. `$-`와 pipefail을
  직접 보존하도록 수정했고, wrapper는 srun 실패, result 부재, jq 실패를 각각 명시적
  nonzero로 반환한다. 따라서 실패 arm 뒤에 성공 receipt를 출력하지 않는다.
- ordinary `salloc`은 login-side command가 끝나면 allocation을 반환하고 SIGHUP도
  allocation을 해제한다. 실제 job `57576758`은 3:20:45, `57582286`은 0:57:53에
  launcher 종료와 함께 끝나 실행 중 step이 SIGTERM을 받았다. 최종 job `57583281`은
  다음처럼 launcher process가 없는 supported Slurm allocation으로 받았다.

```bash
/usr/bin/salloc --no-shell --nodes=4 --qos=interactive --time=04:00:00 \
  --constraint=gpu --gpus=16 --account=m5320_g
```

`scontrol` 영수증은 `JobName=no-shell`, `Command=(null)`, QOS=`gpu_interactive`,
NumNodes=4, GPU=16, EndTime=`2026-08-25T08:19:43-07:00`을 보였다. 각 experiment는
login node에서 계산하지 않고 `srun --jobid=57583281`로 compute node에 붙였다.
NERSC의 [interactive job 문서](https://docs.nersc.gov/jobs/interactive/)가 요구하는
GPU/account/constraint 경계를 지켰고, Slurm의
[`salloc --no-shell` 의미](https://slurm.schedmd.com/salloc.html)에 따라 제어 연결이
끝나도 allocation은 time limit까지 유지된다. background watcher, root, UDI/container,
batch/debug job은 사용하지 않았다.

두 fail-closed receipt도 숨기지 않는다.

- v44 managed는 C8 초반 victim 5건이 `score_delta_ms must be finite and >= 0.0` HTTP
  400으로 끝났다. queue lease가 business/priority 의미를 먼저 적용하면 탈락 후보의
  score delta가 signed negative일 수 있는데 이를 금지한 validation bug였다. delta를
  finite signed explanation으로 수정하고 해당 semantic-override 단위테스트를 추가해 v45로
  다시 동결했다.
- 최종 matrix의 첫 managed 시도는 C8 background aggressor 1/1,374가 빈 HTTP 502로
  끝나 terminal contract가 false였다. victim/global decision/LMCache/NCCL 오류는 아니며,
  source를 바꾸지 않은 retry1이 전부 완주했다. 실패 raw와 log는
  `results/tempo_go_c8_dual_regime_job_57583281_v45_matrix_full_managed/`에 보존한다.

관련 C9/global/C8/frontend/router/C5 source-bound regression은
`306 passed, 22 subtests passed`다. 더 넓은 실행에서 C9와 무관한 historical
`CompositionAffinityV9Test` 하나가 실제 affinity-8 module을 import하면서 policy ID
`...-9`를 기대해 별도로 실패했다. 이를 숨기거나 C8 source에 섞어 고치지 않았다.

### 71.32 v45 no-shell 같은-allocation 7-arm matrix

allocation `57583281`의 동일 네 노드
`nid[001012-001013,001016-001017]`에서 v45 일곱 arm을 같은 contract, profile,
offered population으로 실행했다. P1->D0 arm은 ordinary allocation 종료로 끊긴 시도를
버리고 no-shell job에서 다시 완주한 result만 사용했다. managed의 single-502 시도도
제외하고 source-identical retry1만 headline으로 사용했다.

| arm | C7 miss-hot SLO-good / 120 | C7 p99 | C8 SLO-good / 30 | C8 p99 | C8 remote / local |
|---|---:|---:|---:|---:|---:|
| fixed local D0 | 97 | 9,888.32 ms | 0 | 37,722.01 ms | 0 / 30 |
| fixed local D1 | 96 | 10,467.82 ms | 0 | 37,441.89 ms | 0 / 30 |
| fixed remote P0->D1 | 86 | 18,893.68 ms | 0 | 38,155.08 ms | 30 / 0 |
| fixed remote P1->D0 | 41 | 21,935.08 ms | 0 | 37,461.62 ms | 30 / 0 |
| predictor | 107 | 8,739.80 ms | 13 | 50,985.99 ms | 0 / 30 |
| queue-GPU | 112 | 8,387.60 ms | 13 | 52,710.82 ms | 18 / 12 |
| v45 full managed | **120** | **3,355.59 ms** | **30** | **3,331.21 ms** | **30 / 0** |

analyzer가 고른 strongest fixed는 C7에서 fixed local D0, C8에서 fixed local D1이다.
managed 효과는 다음과 같다.

- strongest fixed 대비 C7 p99 66.1% 감소, SLO-good 97 -> 120;
- strongest fixed 대비 C8 p99 91.1% 감소, SLO-good 0 -> 30;
- predictor 대비 C7 p99 61.6% 감소, C8 p99 93.5% 감소,
  SLO-good 107/13 -> 120/30;
- queue-GPU 대비 C7 p99 60.0% 감소, C8 p99 93.7% 감소,
  SLO-good 112/13 -> 120/30.

managed normal control도 60/60 SLO-good, p50 2,925.16 ms였다. C8는 background
1,344/1,344와 victim 30/30이 모두 complete했고 reject/drop/HTTP error는 0이다.
C7 네 hot block의 background는 1,404 offered 중 1,193 complete, 202 global reject,
9 service-lane failure였다. 따라서 foreground positive는 C8 background를 삭제하거나
completion subset만 골라 만든 결과가 아니며, C7 fairness/utility 비용도 계속 공개한다.

C8 30건은 모두 official LMCache 4,094-token exact full-source hit, vLLM priority lane,
protected decoder business admission, source-balance receipt를 가졌다. 실제 edge는
`P0->D0=7`, `P0->D1=7`, `P1->D0=7`, `P1->D1=9`였고, cross-pair는 14/30 = 46.7%다.
cross-pair 14/14 모두 telemetry-uncertainty source-balance receipt를 가졌다. 이 결과는
remote를 한 source/decoder에 고정한 것이 아니라 두 source와 두 decoder, 네 fabric edge를
실제 요청 경로에서 사용하면서 3.3초 tail을 유지했음을 증명한다.

headline result는
`results/tempo_go_c8_dual_regime_job_57583281_v45_matrix_full_managed_retry1/full_c7_managed_background/result.json`
(SHA `f8afc75b30861222d69e9cc78cd0c85bdd2ef2e3e158fdc1c7dcbf20e28acce1`)이고,
campaign analysis는
`results/tempo_go_c8_dual_regime_job_57583281_v45_campaign_analysis.json`
(SHA `d835f9e3f5c4f7861da3852936adf4f6e6772ac550fe9b133781ac12f4f5b975`)이다.

### 71.33 v45 gate 판정과 정확한 현재 결론

frozen analyzer의 결과는 다음과 같다.

- correctness/native terminal contract: true
- C7 strongest-fixed robustness: true
- C7 predictor robustness: true
- C8 predictor robustness: true
- C8 best-fixed non-inferiority: true
- C8 remote activation: 30/30 = 1.0, true
- exact official LMCache full-hit: true
- global priority service lane: true
- decoder business admission: true
- full uses both local and remote: true
- cross-pair remote: 14/30 = 46.7%, true
- causal source-balance: cross-pair 14/14, true
- `c8_dual_regime_discovery_positive=true`
- `performance_claim_allowed=true`

따라서 v40에서 유일하게 남았던 cross-pair gate가 이제 성능 손실 없이 닫혔다. 현재
증거가 지지하는 결론은 다음과 같다.

> Perlmutter 4-node actual-vLLM/official-LMCache co-load에서 local-only, remote-only,
> predictor, queue-GPU는 decoder running-set과 P/D resource contention 아래 C8 p99
> 37--53초, SLO-good 0--13/30으로 붕괴한다. TEMPO는 business admission, protected
> vLLM priority service, exact cache affinity, telemetry freshness, source/edge virtual
> service를 하나의 global transaction으로 결합해 C7/C8를 모두 약 3.3초와 100% SLO로
> 보호하고, 두 prefill source·두 decoder·네 remote edge를 실제로 사용한다.

contract의 `independent_validation_claim_allowed=false`는 여전히 그대로다. 서로 다른 두
allocation에서 clean managed 결과와 4-edge 분산을 반복했지만, v45는 C9 개발에 사용한
동일 workload/seed/profile이며 blind held-out validation contract가 아니다. 따라서
performance claim은 열렸지만 최종 independent-validation/paper-final claim은 아직 열지
않는다.

### 71.34 다음 실행 gate

1. v45 source를 더 바꾸지 않고 별도 allocation, held-out seed와 counterbalanced arm order를
   가진 independent contract를 동결한다. headline, strongest fixed, predictor, queue를
   최소 필수 arm으로 하고 필요하면 full seven-arm을 반복한다.
2. C8 foreground와 함께 C7/C8 background goodput, tenant별 SLO-goodput, Jain fairness,
   reject/service-lane failure를 formal gate로 승격한다. background를 희생해 foreground를
   보호하는 policy와 TEMPO를 구분해야 한다.
3. source/edge virtual service가 Cassini/Slingshot/libfabric/NCCL supported telemetry를
   얼마나 자주 실제 tie-break 원인으로 사용했는지, unavailable/stale 비율과 telemetry
   overhead를 report한다. unavailable counter를 0 pressure로 간주하지 않는다.
4. pair scale-up/down과 decoder capacity 변화가 함께 있는 trace replay, 4노드보다 큰
   hierarchical control-plane simulation 또는 가능한 native scale에서 coordinator fan-in과
   control overhead를 검증한다.
5. single empty-502를 재시도에만 의존하지 않도록 frontend->router hop의 failure origin을
   별도 reliability work item으로 추적하되, 현재 성능 contract나 workload를 사후 완화하지
   않는다.

즉 다음 목표는 orchestration을 다시 줄이거나 workload를 새로 꾸며 positive를 찾는 것이
아니다. C9에서 이미 whole-system positive와 fabric diversity를 동시에 달성했다. 다음은 이
동결된 결과를 held-out validation, fairness, telemetry overhead, larger-scale hierarchy로
확장해 production-HPC global orchestrator claim을 완성하는 것이다.

## 72. C9 independent validation, C10 paper-policy comparison과 현재 최종 상태

§71.34의 실행순서를 그대로 수행했다. 새 정책 threshold를 찾거나 C9 source를 결과에
맞춰 바꾸지 않았다. held-out contract를 먼저 동결하고 새 no-shell
`gpu_interactive` allocation `57586612`의 네 노드
`nid001144,nid001145,nid001148,nid001149`에서 one-shot matrix를 실행했다. 모든 계산은
`srun --jobid=57586612` compute step에서 수행했고 login node 계산, root, UDI/container,
batch/debug submission과 background watcher는 사용하지 않았다.

### 72.1 C9 fresh held-out 4-node result

authoritative contract는
`eval/sota_4node/tempo_go_c8_independent_validation_contract_v3.json`
(SHA `e2d07e8c50316620cee29a82ae06bbb4e3efd5e8c18c07347a34a4f532f07a76`)이고,
analysis는
`results/tempo_go_c8_independent_validation_job_57586612_v3/analysis.json`
(SHA `844d7b317c0c4839f62feb9a748785a328f9b86b05b52949d81486de09719c47`)이다.
headline TEMPO result SHA는
`0e2069313073001dac277726cfe01dc607defd2eea69e48d3f414d0a7d68de64`다.

| regime | offered / complete / SLO-good | E2E p50 | E2E p99 | route/edge 요약 |
|---|---:|---:|---:|---|
| normal | 60 / 60 / 60 | 2,909.84 ms | 3,149.88 ms | local D0=35, D1=25 |
| miss-hot | 120 / 120 / 120 | 3,058.49 ms | 3,336.68 ms | local D1=120 |
| remote-favorable P_ONLY | 30 / 30 / 30 | 3,067.22 ms | 3,357.20 ms | remote 29, local 1; 네 P→D remote edge 모두 사용 |

fresh allocation, source/contract integrity, one-shot execution, base performance,
background utility/fairness, telemetry overhead와 independent-positive gate가 모두 true다.
따라서 C9 independent performance claim은 허용된다. 효과는 다음과 같다.

- strongest fixed 대비 miss-hot/remote-favorable p99 `66.12%/90.55%` 감소;
- predictor 대비 `61.69%/93.36%` 감소;
- queue-GPU 대비 `58.48%/93.54%` 감소;
- 세 regime 모두 foreground SLO-good fraction `1.0`;
- 실제 topology 사용은 local D0/D1과 remote P0→D0, P0→D1, P1→D0, P1→D1의
  여섯 physical edge다.

foreground만 살리고 background를 버린 결과도 아니다. C7 background는
1,404 offered 중 1,204 complete(`85.755%`), block×tenant minimum completion
`76.496%`, tenant Jain fairness `0.99787`, service-lane failure fraction `0.926%`다.
C8 background는 1,344/1,344 complete다. request-triggered cross-layer telemetry
collection p50/p99은 `28.62/132.42 ms`, admission p50/p99은 `29.46/133.26 ms`다.
Cassini 계열 signal은 30 decision 모두 explicit support 상태를 남겼고 29/30에서
supported였으며, LMCache inflight signal은 30/30 supported다. 이 run에서
`nccl_collective_p99_ms`, `nccl_arrival_spread_ms`, `lmcache_transfer_p99_ms`는
supported가 아니므로 0으로 꾸미지 않고 unavailable로 남겼다.

### 72.2 C10 actual Kairos/NetKV policy comparison

C10은 동일 Qwen2.5-7B-Instruct, vLLM 0.26.0, TP4×4 engine, official
LMCacheConnectorV1/NIXL-UCX/CXI, 동일 seed·block order·arrival jitter·background와
victim population을 사용했다. baseline에는 TEMPO business priority/fairness reserve,
pair scaling, shared-fabric admission과 dispatch stagger가 없다. raw artifact에서
compatibility admission receipt가 `wait_ns=0`, `policy_effect=none`,
`evidence_only_no_throttle`인지 모두 검증했다. Kairos 2,779건, NetKV 2,926건의
held/released receipt와 일곱 block snapshot이 이 gate를 통과했다.

Kairos는 paper의 alpha `1.3`, TBT safety `0.9`를 사용하되 공개 code URL이 freeze 시점에
없고 stock vLLM이 per-request dynamic chunk schedule을 제공하지 않아 실제 decoder
`max_num_batched_tokens=512`, 즉 명시적 `X={512}` subset reproduction이다. 따라서
Kairos 저자의 full implementation 결과라고 주장하지 않는다. NetKV는 Algorithm 1의
remote candidate, KV bytes, decoder queue/first-step와 Perlmutter의 NIC당 25 GB/s,
Cassini congestion, LMCache self-inflight를 사용한 reproduction이다.

| policy | normal SLO / p99 | miss-hot SLO / p99 | remote-favorable SLO / p99 |
|---|---:|---:|---:|
| TEMPO C9 | 60/60 / 3,149.88 ms | 120/120 / 3,336.68 ms | 30/30 / 3,357.20 ms |
| Kairos `X={512}` | 30/60 / 3,223.21 ms | 0/120 / undefined | 1/30 / 4,852.70 ms |
| NetKV reproduction | 60/60 / 3,299.62 ms | 73/120 / 13,779.74 ms | 0/30 / 53,914.27 ms |

TEMPO는 NetKV 대비 normal p99 `4.54%`, miss-hot p99 `75.79%`, remote-favorable
p99 `93.77%` 감소했고 stressed SLO-good을 `73→120`, `0→30`으로 높였다. Kairos
대비 normal/remote-favorable p99은 `2.28%/30.82%` 감소하고 SLO-good을
`30→60`, `0→120`, `1→30`으로 높였다. Kairos miss-hot은 complete가 0이므로
존재하지 않는 latency quantile을 무한 개선율로 만들지 않고 completion/SLO dominance로
보고한다.

authoritative analysis-only contract는
`eval/sota_4node/tempo_go_c10_paper_sota_analysis_contract_v4.json`
(SHA `60d4958f73774d932b44a9b95f5b5247952ba5fffdc6fae84c0969619a2b0525`)이고,
analysis는
`results/tempo_go_c10_paper_sota_job_57586612_v3/analysis.json`
(SHA `bdf8604b96715cd31163101cdb4c684feb8fc26aa0d9af0bcf285c558e9a5d0f`)이다.
actual SOTA extension gate는 true다. 그러나 C10 adapter는 parent allocation 시작 뒤
동결됐고 NetKV evidence-validator fix도 같은 allocation에서 이루어졌으므로
`independent_validation_claim_allowed=false`다. 논문에서는 C9를 independent headline,
C10을 actual-system post-hoc SOTA extension으로 분리한다. unchanged C10 fresh-allocation
rerun 전에는 “independently beats NetKV/full Kairos”라고 쓰지 않는다.

### 72.3 실패 receipt와 수정 범위

C10 v1은 Kairos measured workload에서 decoder admission receipt를 반환하지 않아 frozen
validator가 중단됐다. v2 adapter는 policy에는 개입하지 않고 held/released evidence만
`wait=0`, `policy_effect=none`으로 남겼다. v2 NetKV는 유효한 `remote:p0→d1` request를
실행했지만 legacy C7 validator가 destination `frontend_pair_index`를 prefill source로
오인해 중단됐다. v3은 workload/NetKV score를 바꾸지 않고 commit의
`prefill_index`, `decoder_index`, `edge_id`를 별도로 검증했다. 성공한 Kairos v2 result는
SHA 고정해 재사용했고 NetKV만 v3 source에서 one-shot으로 실행했다. 모든 실패 raw/log는
삭제하지 않는다.

### 72.4 current-source hierarchy scale receipt

current C9/C10 hierarchy source를
`eval/sota_4node/tempo_go_hierarchy_scale_contract_20260825.json`
(SHA `7f7753a4bdb0de4fe4f5effdd750b5ba189fb6678060f1e3904772d9880772a2`)에
동결하고 compute node에서 pair 2, 8, 32, 128, 512, 1,024, 각 15회 같은-population
full scan 대 bounded node→pair→shard/global path를 측정했다. result는
`results/tempo_go_hierarchy_scale_20260825_c9_c10_r15.json`
(SHA `90f4e2ab3b3645e7c4ee62c3c95a11f1debb0cb2eb400bb189ac85b36e047088`)이다.

1,024 pair에서 raw candidate 2,048개 중 global로 256개를 전달하고 896 pair omission
receipt를 남겼다. payload는 666,815 B→83,358 B, `87.499%` 감소했다. full-scan
p50/p99은 `49.74/152.84 ms`, prepartitioned pair-agent p50은 `29.65 ms`, bounded
global p50은 `55.43 ms`, bounded total p50/p99은 `85.33/158.24 ms`다. 즉 payload와
global fan-in은 bounded하지만 이 Python single-process CPU timing이 full scan보다 빠르다는
결론은 아니다. GPU/NCCL/LMCache/Slingshot/native request goodput 또는 production-scale
성능 claim으로 사용하지 않는다.

### 72.5 현재 결론과 남은 hard gate

현재 답은 “문제가 없어서 orchestration이 필요 없다”도 “remote를 버려라”도 아니다.
actual co-load에서 fixed local/remote, predictor, queue-only, Kairos subset과 NetKV는
각각 decoder compute, KV/source reuse, receiver running set, fabric/self-contention 또는
admission feasibility 중 일부만 최적화해 workload regime가 바뀌면 completion과 tail이
붕괴한다. C9 TEMPO는 business admission, local/remote candidate, decoder capacity,
LMCache semantics, Cassini state, topology와 source/edge service를 한 global transaction으로
묶어 그 bottleneck migration을 막았다.

완료된 gate는 fresh 4-node independent whole-system win, fairness/background utility,
telemetry overhead, six-edge actuation, actual NetKV/Kairos-subset comparison과 1,024-pair
CPU fan-in receipt다. 남은 gate는 다음과 같다.

1. C10 source를 더 바꾸지 않은 fresh allocation에서 Kairos/NetKV/TEMPO를 다시 실행해
   independent SOTA extension을 닫는다.
2. 가능하면 Kairos의 full dynamic chunk candidate set을 vLLM scheduler에 구현하거나,
   현재 결과를 계속 `X={512}` subset으로만 표기한다.
3. C9 held-out run에서 unavailable였던 NCCL/LMCache latency signal을 opt-in co-job 또는
   larger native rung에서 supported로 만들고 joint-control causal ablation을 반복한다.
4. 4-node보다 큰 native allocation이 승인될 때 node→pair→shard/global wall-clock,
   wire bytes, failure convergence와 inference utility를 함께 측정한다. 현재 CPU scale
   receipt만으로 Perlmutter production-scale superiority를 주장하지 않는다.
5. paper artifact는 C9 independent claim과 C10 post-hoc claim을 분리하고 raw SHA,
   workload, failure receipt, baseline limitation을 그대로 공개한 뒤 scoped Git commit으로
   게시한다.

### 72.6 paper artifact와 최종 release regression

논문 source는 `paper/tempo_go/main.tex`, bibliography는
`paper/tempo_go/references.bib`, 빌드된 7-page PDF는
`paper/tempo_go/main.pdf`이다. PDF SHA는
`3e35c65a92230ceef4576bff1ac5aa7ed42a33b82d76ab57a2d5cb2e3877f60f`이며,
TeX Live 2024에서 LaTeX error, unresolved citation/reference, overfull box가 모두 0인
상태로 세 번의 `pdflatex`와 `bibtex`를 통과했다. 전 페이지 raster와 text extraction을
확인해 표/본문의 잘림이나 겹침이 없고 bibliography의 UTF-8 표기도 교정했다.

기계 판독 가능한 release index는
`paper/tempo_go/artifact_manifest.json`이다. 여기에 C9 independent, C10 post-hoc,
Kairos `X={512}` subset, NetKV actual-carrier reproduction, 1,024-pair CPU-only scale의
서로 다른 claim boundary, authoritative contract/analysis/result path와 SHA를 넣었다.
Git에는 거대한 raw/log tree 대신 aggregate result와 compact analysis를 포함한다. raw는
Perlmutter에 보존하고 analysis가 path와 SHA를 고정하므로 다른 파일로 조용히 대체할 수
없다.

최종 regression은 native process 경계와 같은 방식으로 분리했다.

- C9/current global/frontend/router/C5--C9 suite:
  `294 passed, 2 deselected, 28 subtests passed`.
- C10 paper-baseline process-entry suite: `6 passed`.
- C9 source inventory 39/39 SHA 일치, Python `py_compile`, canonical launcher
  `bash -n`: passed.

두 deselect는 현재 C9 source가 과거 C6 frozen source hash와 같아야 한다고 요구하는
역사 계약 검사다. 그 contract를 새 hash로 덮으면 과거 실행 증거를 위조하므로 실행하지
않았다. 최초 한-process collection은 `299 passed` 뒤 이 두 historical drift test와 C8
admission test 하나가 실패했다. C8 실패는 C10 전용 frontend entrypoint가 import 때
같은 프로세스의 canonical frontend binding을 baseline/no-op class로 바꾸기 때문이었다.
C8 단독 11/11, C9 independent/C10 focused 27/27이 통과했고, 실제 native arm은 서로 다른
프로세스이므로 정책 혼입은 없었다. 최종 gate는 이 실제 배포 경계를 따라 C9와 C10을
분리했으며 최초 실패 receipt도 삭제하지 않는다.
