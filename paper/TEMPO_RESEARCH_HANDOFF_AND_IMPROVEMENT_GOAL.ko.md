# TEMPO 연구 통합 기준 문서와 다음 개선 목표

문서 버전: `handoff-v11`, 2026-08-22
실행 대상: Perlmutter native 4노드 / 16 A100 / 실제 vLLM P/D / official `LMCacheConnectorV1:UCX`

이 문서는 지금까지의 TEMPO 연구를 한 번에 이어받기 위한 단일 통합 문서다. 원래 목표를 축소하거나 request-local predictor 문제로 바꾸지 않는다. 기존 master/playbook/audit와 모든 raw artifact는 증거 보존용으로 유지하고, 다음 작업은 이 문서를 첫 진입점으로 사용한다. 이 문서에 없는 수치나 revision은 추정하지 않고 exact path/SHA가 있는 자료만 사용한다.

## 1. 결론부터

현실적인 동시 부하를 주면 병목은 생긴다. 혼자 돌린 interactive 실험에서 interconnect가 여유로운 것은 contention이 없는 조건일 뿐이며, 그것이 global orchestration의 필요성을 반박하지 않는다. 실제 4노드 vLLM P/D에서는 다음이 확인됐다.

- local prefill이 항상 이기지 않는다. decoder-local path가 뜨거우면 remote가 이기는 구간이 있다.
- remote가 항상 이기지도 않는다. P/KV transfer, receiver/install, endpoint 또는 shared decoder가 뜨거워지면 local이 크게 이긴다.
- C3에서 local/remote의 승자가 부하 단계에 따라 뒤집혔다.
- native C5에서는 business-level admission reject와 official LMCache의 `CacheEngineKey ... not found in local data`, `EngineDeadError`, HTTP 503/502가 관찰됐다.
- held-out native v3에서는 같은 4노드/16-GPU 조건에서도 run-to-run 상태 의존성이 확인됐다. 이번 epoch의 official remote arm은 2,712/2,712 완료했지만, 직전 승인 allocation에서는 LMCache memory-block exhaustion과 HTTP 500/502로 실패했다.
- 따라서 orchestration은 필요하다.

다만 현재까지의 결론은 “TEMPO가 이미 빠르다”가 아니다. 현재 결론은 다음처럼 정확히 나뉜다.

1. **문제 실존**: contention, local/remote crossover, queue/SLO 압박, LMCache failure는 실험적으로 존재한다.
2. **route-only의 한계**: request-local route threshold, scalar pressure, phase oracle까지 C4 full gate를 통과하지 못했다. 이 방향의 미세조정은 종료한다.
3. **global control-plane 통합**: decoder admission, tenant SLO/fairness, pair activation, route, endpoint failure/quarantine의 control plane은 CPU/native receipt 수준으로 상당 부분 구현됐다.
4. **현재 native에서 확인된 trade-off**: 최신 source-rebound v3 native run은 전역 admission cap을 5 s로 늘리고 tenant queue reservation을 넣었어도 2,712건 중 982건만 완료하고 1,730건을 명시적으로 reject했다. background는 2,436건 중 769건, interactive는 96건 중 80건, latency는 96건 중 50건만 완료했다. TEMPO output-token goodput은 548.4/s로 local 981.8/s·remote 1,185.7/s·predictor 981.1/s보다 낮았다. 따라서 zero-failure와 낮은 completed-only latency만으로 fairness나 production utility를 주장할 수 없다.
5. **아직 없는 것**: frozen code/profile/manifest/run-contract를 사용한 독립 native validation에서 strongest fixed 및 predictor-only보다 빠르거나 robust하다는 최종 증거. 최신 v3도 native descriptive discovery일 뿐이고 queue-GPU-only execution failure와 TEMPO의 admission-feasibility 실패 때문에 performance gate를 닫지 못했다.

## 2. 원래 목표와 현재의 정확한 재정의

원래 목표는 다음이다.

> TEMPO Elastic-PD를 실제 vLLM P/D 경로에 통합하고, 단순 predictor와 가장 강한 고정 정책보다 유의미하게 빠른 하나의 최종 스킴으로 확정한다.

이 목표는 유지한다. 단, v0~최신 실험을 모두 읽은 결과, 최종 스킴의 대상은 단순 route selector가 아니라 **TEMPO-GO global orchestrator**로 구체화한다.

연구 질문:

> 동일한 native 4-node/16-A100 vLLM P/D topology, GPU budget, request trace, cache namespace와 official LMCache data plane에서 TEMPO-GO가 moving multi-tenant contention을 관찰하고 decoder admission, tenant SLO/fairness, P/D pair assignment 및 logical scaling, local/remote route, endpoint congestion/failure recovery를 공동 제어하여 strongest fixed policy와 predictor-only보다 latency/goodput 또는 overload robustness를 개선하는가?

허용되는 최종 claim은 다음뿐이다.

> 동일한 실제 vLLM P/D topology와 official LMCache data plane에서 TEMPO-GO의 request/global admission policy가 fixed local/remote 및 predictor-only policy보다 낮은 latency, 높은 goodput, 또는 더 강한 multi-tenant overload robustness를 보였다.

LMCache transport 자체보다 빠르다, Mooncake/Kairos/Dynamo보다 보편적으로 우월하다, 특정 switch/link가 병목이라고 확정했다, 모든 workload에서 항상 빠르다, 단일 allocation으로 production-ready라는 주장은 금지한다.

### 원본 목표에서 보존하는 구현 계약

원래 목표 문서의 다음 설계 계약은 폐기하지 않는다. 이것들은 단순 predictor의 설명이
아니라 TEMPO-GO가 실제 vLLM P/D 경로에서 지켜야 하는 request/control-plane
불변식이다.

1. 요청 시작 전에 `LOCAL`, `REMOTE`, `QUEUE/REJECT` 중 하나를 one-way commit하고,
   prefill 이후 route 변경·hidden recompute·silent fallback·same-ID retry를 금지한다.
2. remote는 confirmed cache evidence와 측정된 이득이 동시에 있을 때만 연다. `UNKNOWN`은
   hit가 아니며, remote 예상 이득이 유의미하지 않으면 local 또는 명시적 admission으로
   닫는다. route 비율을 미리 강제하지 않는다.
3. pair-local hysteresis와 명시적 recovery probe로 arrival burst에 정책이 매 요청마다
   흔들리지 않게 한다. cross-host clock subtraction과 phase/future oracle은 사용하지
   않는다.
4. local compute, remote P service/KV bytes, semantic operation, endpoint/decoder를
   독립 credit으로 bound하고 bounded queue와 explicit reject만 허용한다. request count
   하나로 모든 자원을 대표하지 않는다.
5. prefill/endpoint credit은 first response에서, decoder credit은 EOF/complete에서
   반환하며 complete·abort·timeout·failure마다 exactly once를 보장한다.
6. data plane은 official `LMCacheConnectorV1:UCX`로 고정한다. TEMPO-GO의 기여는
   transport 교체가 아니라 admission, business fairness, pair scaling, route와
   congestion/failure recovery의 공동 제어다. NIXL 교체, token-level decode hook,
   global fence, busy polling, sidecar microburst는 최종 scheme의 필수 구성으로
   승격하지 않는다.

## 3. v0부터 현재까지의 계보와 배운 것

현재 bounded workspace에서 확인 가능한 source revision은 `eval/sota_4node/`의 v1~v450 계열이며, v452~v544는 late raw-artifact lineage다. workspace에 없는 v545~v600을 추정해 만들지 않는다.

| 시기 | 실제로 한 것 | 남긴 교훈 | 현재 운명 |
|---|---|---|---|
| conceptual v0~pre-reset | phase-gated I/O, topology/QoS, sparse transfer, TEMPO-RD | broad motivation만으로는 contribution이 닫히지 않음 | 역사적 motivation |
| actual v1~v27 | vLLM P/D, LMCache/NIXL, KV geometry, pressure probe | admission과 transport를 분리해야 함; 초기는 remote branch가 충분히 검증되지 않음 | evidence 보존 |
| v28~v60 | queue crossover, offered rate, local credit | remote는 항상 나쁘지 않으며 mixed route crossover가 있음 | crossover evidence |
| v61~v129 | threshold, interleaving, heterogeneous prompt/output | trace-derived constant와 순차 arm block은 일반화되지 않음; paired/counterbalanced가 필요 | 실험 규칙 |
| v131~v245 | cache catalog, warm/cold, saturation, tail | cache residency는 route constraint이며 aggregate median만으로 부족 | cache/tail evidence |
| v248~v349 | overload, unique cache key, burst, local credits | silent fallback과 aliasing은 correctness를 깨뜨림; bounded admission이 필요 | invariant |
| v353~v430 | phase change, prefix swap, adaptive cap | fixed threshold와 phase classifier를 늘려도 tail이 안정되지 않음 | route-only 종료 근거 |
| v440~v450 | native NIXL comparison, Elastic-PD component | one-way commit, first-response release, cache isolation, weighted ownership은 sound | global scheme의 component |
| v452~v544b | profile/pair/credit/chunk/cache/CXI background | favorable single number는 재현되지 않으며 bottleneck은 이동함 | negative evidence |
| C0~C5 TEMPO-GO | global admission, tenant contract, telemetry, pair activation, failure receipt | global control plane은 native에 들어갔고 overload/failure도 발동 | 현재 검증 대상 |

핵심 교훈:

- `fabric_pressure` 하나로 병목 위치를 판정하지 않는다.
- queue depth는 service pressure와 같지 않다.
- local prefill, remote P service, KV bytes, semantic operation, receiver/install residual, shared decoder를 분리한다.
- cache `UNKNOWN`을 hit로 취급하지 않는다.
- route는 request 시작 시 한 번 commit하며, prefill 후 route 변경/hidden recompute/silent fallback을 금지한다.
- first response에서 endpoint/prefill credit을 반환하고, EOF/complete에서 decoder credit을 반환한다.
- 실패는 latency sample로 바꾸지 않고 explicit failure receipt와 quarantine로 남긴다.
- discovery 결과를 보고 profile/threshold/code를 바꾸면 independent validation이 아니다.

### 관련 transport 컨셉: MRC와 TEMPO-GO의 경계

MRC(Multipath Reliable Connection)는 multipath/multi-plane packet spraying,
receiver-driven bounded in-flight transmission, endpoint backpressure와 fast failover를
transport 계층의 primitive로 제안한다. 이는 TEMPO-GO의 pair×route credit bound,
endpoint health/probe, path quarantine, service-time 분리와 직접적으로 맞닿아 있다.
그러나 MRC는 packet transport를 개선하는 설계이고, TEMPO-GO의 연구 target은 그 위의
inference business/control plane이다. 즉 tenant SLO/fairness, decoder admission,
P/D logical pair scaling, official LMCache endpoint completion과 request-level route
commit을 transport 교체 없이 공동 제어한다. MRC를 근거로 LMCache transport를
교체하거나 TEMPO의 inference orchestration 기여를 transport 성능 기여로 바꾸지 않는다.
참고: [The Multipath Reliable Connection (MRC) Transport](https://arxiv.org/html/2606.18170v1).

## 4. contention과 negative/positive evidence

### 4.1 C3 opposite crossover

실제 4노드 vLLM P/D, official `LMCacheConnectorV1:UCX`, 4094-token/2-output unique-cold foreground에서 remote background rate를 바꿨다.

| remote background | local median | remote median | 해석 |
|---:|---:|---:|---|
| 0/s | 528.9 ms | 448.4 ms | decoder-local path가 뜨거우면 remote가 이김 |
| 4/s | 542.8 ms | 589.1 ms | 승자가 local로 이동 |
| 8/s | 661.7 ms | 655.3 ms | near tie |
| 12/s | 674.9 ms | 1832.0 ms | remote P/KV/receiver path가 무너짐 |

rate 12에서 remote completion residual은 1569.8 ms까지 늘고 achieved remote background throughput은 약 9.23 req/s였다. 이 결과는 “remote가 나쁘다”가 아니라 서비스 상태가 이동한다는 근거다.

### 4.2 C4 route-only terminal negative

scalar score, pair-local active watermark, route-pinned local external-credit, phase oracle를 포함한 구조적으로 다른 후보들이 strongest fixed/predictor/tail/goodput/paired gate를 동시에 통과하지 못했다. 따라서 prompt coefficient, scalar pressure, phase classifier, request-local threshold를 더 튜닝하지 않는다.

이 negative는 orchestration 전체의 실패가 아니다. local과 remote 모두 shared decoder와 endpoint 외부성을 갖기 때문에 **route만 고르는 제어로는 coupled decode tail과 business fairness를 제어할 수 없다**는 뜻이다.

### 4.3 C5 native evidence

`57402376` guarded five-arm discovery:

- arm order: `tempo → queue_gpu → predictor → remote → local`
- local/predictor/remote: 각 2,712/2,712 valid
- queue-GPU-only: LMCache receiver allocation/EngineCore failure receipt; latency baseline으로 대체하지 않음
- TEMPO: 2,712 terminal-valid, 1,865 complete, 847 explicit global reject, local 1,686 / remote 179, pair activation 1
- TEMPO output-token goodput 136.9/s, always-local 190.2/s
- 판정: native integration/discovery evidence; performance claim 및 independent validation 아님

`57404614` Candidate C failure-quarantine:

- raw 2,712 rows: complete 1,633, failed 9, rejected 1,070
- global failure receipt 9건: pair-scope transport 3, route-scope HTTP 6
- quarantine rejection 1,714건, same request ID retry 없음
- node log에 official LMCache key assertion/EngineDeadError, proxy ConnectError
- step exit 143, `result.json` 없음, `router_decisions_exact=false`
- 판정: native failure/quarantine robustness evidence; 성능 증거 아님

기존 `57400890` guarded single-arm은 2,712/2,712 exact terminal receipt, 1,904 complete, 808 explicit reject, local 1,623/remote 281을 만들었다. 이것은 guard와 receipt closure 증거이지 성능 승리가 아니다.

### 4.4 held-out native v3 five-arm discovery: 통합은 닫혔지만 policy는 아직 과보수적

승인된 Perlmutter interactive allocation `57407705`에서 현재 v3 run contract와 같은 held-out
2,712-row workload를 사용해 `local → remote → predictor → queue_gpu → tempo`를
순차 실행했다.

| arm | terminal 상태 | route | request goodput/s | E2E p50/p99 (ms) | 판정 |
|---|---|---|---:|---:|---|
| ALWAYS_LOCAL | 2,712 complete, 0 reject, 0 fail | local 2,712 | 7.876 | 15,542 / 19,685 | clean fixed receipt |
| ALWAYS_REMOTE | 2,712 complete, 0 reject, 0 fail | remote 2,712 | 9.695 | 10,631 / 20,037 | clean fixed receipt; 이 epoch에서는 성공 |
| PREDICTOR_ONLY | 2,712 complete, 0 reject, 0 fail | local 2,592 / remote 120 | 7.801 | 15,055 / 20,962 | clean predictor receipt |
| QUEUE_GPU_ONLY | measured raw 없음, process exit 143 | 없음 | 없음 | 없음 | execution failure; latency comparator에서 제외 |
| TEMPO_GO | 900 complete, 1,812 explicit global reject, 0 fail | local 852 / remote 48 | 4.382 | 5,364 / 8,539 (completed only) | global lifecycle는 닫혔으나 performance/fairness 승리 아님 |

TEMPO의 native receipt에는 endpoint completion 900건, global reject 1,812건,
global scheduler observation payload 2,712건과 observation 5,424건, endpoint feedback
900건, pair activation 1건이 있다. global decision reason은 queue timeout 1,808,
fair-route commit 899, proactive queue scale 1, telemetry-refresh timeout 4였다.
모든 4개 tenant는 starvation=false였지만, background는 2,436건 중 688건만 완료하고
1,748건을 reject했다. 이것은 “reject했으므로 fairness가 좋다”가 아니라 weighted
service debt/minimum service fraction과 admission budget을 재설계해야 한다는 신호다.

이번 epoch의 fixed arm은 모두 terminal-clean이었고 queue-GPU-only만 rc=143으로
실패했다. 따라서 analyzer의 `performance_claim_allowed=false`는 올바르다. TEMPO의
completed-only latency가 낮아 보이는 것은 동일 request population을 완료한 비교가
아니며, 4.382 request/s는 local 7.876, remote 9.695, predictor 7.801보다 낮다.
이 숫자는 descriptive discovery evidence일 뿐 performance claim이 아니다. 반대로
queue-GPU-only의 failure receipt와 TEMPO의 0-failure/explicit-reject receipt는
global admission 및 failure containment가 실제 vLLM/LMCache 경로에서 발동했다는
robustness/integration evidence다.

중요한 해석은 세 가지다. (1) remote가 이번 epoch에서 성공했다고 해서 항상 안전한
것은 아니다. 직전 native attempt `57407330`에서는 같은 계열 부하에서 LMCache block
allocation exhaustion, proxy ConnectError, HTTP 500/502가 발생했다. (2) TEMPO가
실패를 숨기지는 않았지만 현재 policy는 overload에서 background service를 크게
희생한다. (3) 이는 fabric bottleneck 위치를 증명한 결과가 아니다. 실제 queue wait는
p99 0.482 ms였지만 global admission decision wait는 p99 2,115 ms였고, profile의
controller `maximum_queue_wait_ns=2,000,000,000`과 coordinator timeout이 모든 tenant의
대기 상한으로 작동했다. 다음 연구 단계는 route threshold를 재튜닝하는 것이 아니라,
tenant-aware admission budget, minimum service fraction, reject/defer policy, pair
scale capacity와 endpoint failure recovery를 함께 고쳐야 한다.

### 4.5 최신 source-rebound native v3: reservation은 queue를 보호했지만 service feasibility를 보장하지 못함

이전 `57407705` receipt보다 최신인 source-rebound 결과를 현재 native 기준으로
사용한다. 승인된 Perlmutter interactive allocation `57409956`에서 새 contract와
`r8_16_20_20` tenant-reservation profile을 고정하고, 같은 held-out 2,712-row
workload를 `local → remote → predictor → queue_gpu → tempo` 순서로 실행했다.
이 결과는 기존 root를 덮어쓰지 않은 별도 artifact이며, 어떠한 root/udiRoot/container
수정도 하지 않았다.

- contract: `results/tempo_go_c5_r8_16_20_20_contract_v3/native_run_contract.json`
- contract SHA / fingerprint: `002ee5424c9779b22d2cc622cb9143227f8370d03d6b22d0f3c9a560f153e481` /
  `7691d005cad942c26a9a8792cf1487431ce5c4f7abe43ebb7b409a2fef5a854e`
- result root: `results/tempo_go_c5_r8_16_20_20_native_job_57409956_v3`
- analyzer SHA: `b7e302ab1f893310602b491a8971138d3f4b3cd7fa906b4f7ce05848ac305f45`
- topology: native Perlmutter 4 nodes / 16 A100 / official `LMCacheConnectorV1:UCX`
- profile: `results/tempo_go_c5_reservation_sweep_profiles/r8_16_20_20.json`
  (global maximum queue wait 5 s; reservations latency 20, interactive 20, batch 16,
  background 8; queue capacity 128)

| arm | terminal 상태 | request goodput/s | output-token goodput/s | E2E p50/p99 (ms) |
|---|---|---:|---:|---:|
| ALWAYS_LOCAL | 2,712 complete / 0 reject / 0 fail | 7.934 | 981.8 | 15,202.5 / 19,675.3 |
| ALWAYS_REMOTE | 2,712 complete / 0 reject / 0 fail | 9.581 | 1,185.7 | 11,090.4 / 20,559.8 |
| PREDICTOR_ONLY | 2,712 complete / 0 reject / 0 fail | 7.928 | 981.1 | 15,530.2 / 19,524.7 |
| QUEUE_GPU_ONLY | measured raw 없음, exit 143 | — | — | — |
| TEMPO_GO | 982 complete / 1,730 reject / 0 fail | 4.786 | 548.4 | 7,463.4 / 9,206.1* |

`*` TEMPO latency는 완료된 982건만의 값이므로 fixed arm과 같은 request
population의 latency 승리가 아니다. TEMPO tenant 결과는 background
`769/2,436`, batch `83/84`, interactive `80/96`, latency `50/96` complete/request
였고, rejection reason은 queue timeout 822, tenant reservation 897, fair-route
commit 981, telemetry timeout 11, proactive scale 1이었다. 즉 reservation slot은
ingress queue occupancy만 보호할 뿐, 실제 decoder/P/remote/endpoint service
capacity와 SLO deadline을 보장하지 못했다.

TEMPO는 endpoint completion receipt 982건, valid scheduler observation 5,424건
(2,712 payload × 2 pair), pair activation 1건, local route 908건/remote route
74건을 기록했다. measured bounded queue wait p99는 0.51 ms였지만 global admission
wait p99는 5,070.85 ms였다. 따라서 이 결과는 “fabric의 특정 link/NIC가 병목”이라고
말할 수 있는 증거가 아니다. 확정된 것은 실제 native contention에서 global
admission이 business reject로 발동했고, 현재 policy가 service-feasibility를
잘못 예측했다는 점이다.

`QUEUE_GPU_ONLY`의 exit 143 원인은 `run_tempo_go_c5_stream_client.py`의
`http.client.IncompleteRead: 0 bytes read` 뒤 `CalledProcessError`였으며, latency
sample로 대체하지 않는다. 최신 run에는 `udiRoot.conf` 오류, root 권한 변경,
container 실행 또는 exit 139가 없었다. 이 arm은 robustness execution-failure
receipt로만 보존한다.

이 결과에서 다음 causal target은 queue reservation 숫자나 route threshold를
조금씩 바꾸는 것이 아니다. (1) tenant SLO deadline 안에 완료 가능한지 pair×route별
service residual로 계산하고, (2) queue slot이 아니라 decoder/P/remote/endpoint
capacity lease를 tenant class별로 보호하며, (3) lease가 infeasible할 때 우선순위
tenant은 surviving pair/spare pair로 재평가하고 background는 명시적 defer/reject하며,
(4) pair activation이 실제 feasibility를 개선했는지 receipt로 검증하는 **admission
feasibility controller**가 다음 후보의 핵심이어야 한다. 이 candidate가 CPU
fairness/SLO/overhead gate를 통과하기 전에는 native retry를 하지 않는다.

### 4.6 최근 correctness와 analyzer closure

- `GlobalOrchestrator._effective_deadline_ns()`가 외부/default deadline을 tenant의 frozen E2E SLO로 cap한다. candidate feasibility와 fair queue ordering이 같은 business deadline을 사용한다.
- 이는 route threshold, phase input, future arrival을 추가한 것이 아니라 tenant-aware global admission correctness fix다.
- raw-backed failure analyzer가 `failure.json` 옆의 raw ledger를 읽어 semantic terminal phase와 failure receipt를 복원한다. `router_decisions_exact=false` execution failure는 여전히 성능 성공으로 승격하지 않는다.
- 이전 broader focused suite 기록은 `128 passed, 11 subtests passed`였고, policy/replay
  변경 후 현재 `.vllm_venv` bounded suite는 `130 passed, 11 subtests passed`다.
- 이 CPU/test 결과는 기존 native 결과를 재검증하여 성능 승격한 것이 아니다. 다음 native result root에서 code revision을 freeze해야 한다.

### 4.7 C5 contract와 offline lifecycle closure

가장 최근에 완료된 native discovery는 held-out output=128 artifact에 결박된
`results/tempo_go_c5_heldout_frozen_proxy_v3/native_run_contract.json`이다.
v1 contract는 node-entry source inventory가 stale했던 historical 실패 기준이고,
현재 v3 contract file SHA는
`c280a889e148069b2678c53dc3cdb738219e6c6a64f80b9594b220c7d2f4f3f4`, fingerprint는
`1fd9ff9f894b916a855c9aa93adb66a4a1bc4e1d05107cb09e690f300d857b73`이며, 4-node/16-GPU,
official `LMCacheConnectorV1:UCX`, workload/manifest/model/profile SHA, arm order와
launcher/node/analyzer/source inventory를 묶고 verifier를 통과했다. 이것은 native
실행 전 identity freeze이지 native 성능 결과가 아니다.

이 v3 contract와 allocation `57407705`는 완료된 historical discovery receipt로
보존한다. 현재 소스 수정 이후 v3 contract는 재사용 가능한 최신 실행 contract가
아니며, split replay로 만든 Candidate G contract가 최신 CPU candidate identity다.
Candidate G는 correctness는 통과했지만 tail/SLO-goodput gate를 통과하지 못했으므로
아직 native allocation을 허가하지 않는다.

정상 CPU replay는 2,712개 동일 trace를 다섯 arm에 흘려 모든 request terminal,
owned-resource leak 0, queued/inflight 0, phase/future-arrival/physical-switch
policy input false를 확인했다. TEMPO-GO는 1,629 complete, 1,083 explicit reject,
실패 0이었다. `performance_claim_allowed=false`와 `native_gpu_run_allowed=false`를
유지하므로 이 수치는 GPU/fabric latency나 성능 승리가 아니다.

failure 경로는 `frozen_failure_global_profile.json`의 quarantine-enabled profile에서만
검증했다. held-out normal replay에서 실제 remote로 admit된 index 591에
`injected_remote_route_failure`를 주입해 `tempo-go-global-failure-v1` receipt 1건,
route-scope quarantine, released work, `new_request_id_required`, terminal/leak-free를
확인했다. TEMPO-GO는 1 failed, 1,630 complete, 1,081 reject였고 tenant starvation은
없었다. 이 결과도 failure lifecycle robustness의 CPU evidence이며 performance claim은
금지한다. 기존 v10/v11은 이 closure 이전의 immutable historical guard/failure replay로
보존하지만 현재 held-out native contract로 재사용하지 않는다.

failure 주입을 quarantine-disabled profile에서 실행하면 replay를 부분 진행하지
않고 즉시 fail-closed한다. 이 경계를 지키지 않아 음수 telemetry counter가
발생했던 bug를 수정했고, receipt 생성이 성공한 뒤에만 replay-side credit을
반환하도록 했다.

### 4.8 global admission candidate의 CPU gate 결과와 native stop/go

v3 native의 원인이 route threshold가 아니라 ingress/admission임을 확인하기 위해,
동일한 2,712-row held-out trace에서 fixed arm은 기존 frozen profile을 사용하고
TEMPO-GO만 candidate profile을 사용하는 split replay를 추가했다. 이전 replay가
모든 arm에 candidate profile을 적용하던 비교 오염도 이때 수정했다. 수정 후 focused
suite는 `131 passed, 11 subtests passed`이고, 기존 v3 contract는 replay source SHA가
달라져 fail-closed했으므로 v3 contract를 덮어쓰지 않고 Candidate G용 새 contract를
생성했다.

| candidate | business mechanism | TEMPO result | strongest fixed local | gate 판정 |
|---|---|---:|---:|---|
| E | global wait cap 2 s → 5 s; reservation/route/pair logic unchanged | 1,433 complete, 1,279 reject; E2E p50/p99 `6,623/8,446 ms`; background SLO-goodput 636 | 1,321 complete; `5,321/5,914 ms`; background SLO-goodput 1,087 | admission throughput은 늘지만 p50 약 +24.5%, background SLO-goodput 악화; native 금지 |
| G | 5 s budget + tenant별 bounded queue reservation 16 slots | 1,430 complete, 1,282 reject; `6,299/8,446 ms`; background SLO-goodput 720; batch/interactive 84/96 | 1,321 complete; `5,321/5,914 ms`; background 1,087 | priority queue protection은 발동했지만 tail과 SLO-goodput gate 실패; native 금지 |
| H | G의 reservation + 2 s budget + queue/wait 25% proactive pair trigger | 1,321 complete, 1,391 reject; `5,321/5,914 ms` | 동일 | 결과가 baseline과 중립; pair activation benefit 없음; native 금지 |

Candidate E/G의 split replay는 각각 다음 artifact에 고정되어 있다.

- E profile SHA `b1c0257762d1ea9fee0377edc4fffb99902c1abdcb156312faed7d38e1c15630`, replay SHA `8d3d67d65b04e0488e5c4a1f1601b139b51b7f85ac42f3c89a6cc47483efa7b6`
- G profile SHA `28a1b9f4fc1033f01d340f37e59ca04eeb26233afaf4a587ddfb02d33270dcba`, replay SHA `099281de85332879ad9d22e87f237b3c0e207830e0363d5bae45f399f95c5635`
- G native run contract SHA `701b2e8bea75471a597b21f43554a63f386afd2621e413c0265f6898e67b2cf4`, fingerprint `8c68dbe41ffa1126a9a364e459b85b3130df92cfa80fec8824ff6515a73069d8`; performance claim은 false

별도 ingress-window diagnostic F는 queue capacity를 128에서 2,048로 키웠을 때
replay가 7분 이상 queue scan에 머물러 중단됐다. output artifact를 성능 결과로
만들지 않았으며, bounded queue를 무작정 키우는 방식은 controller overhead gate를
통과하지 못한다는 negative evidence다.

현재 stop/go 결론은 다음이다. contention과 global orchestration 문제는 실존하고,
TEMPO-GO의 tenant/fairness/pair/telemetry control plane도 실제 vLLM/LMCache 경로에
통합됐다. 그러나 현재 admission candidate들은 fixed 대비 성능 또는 robustness
gate를 통과하지 못했다. 따라서 이 결과만으로 native 4-node allocation을 소비해
profile을 튜닝하지 않는다. 다음 후보가 필요하면 queue reservation을 더 세밀하게
조정하는 threshold search가 아니라, bounded admission과 pair service-rate를 함께
늘리는 causal mechanism을 새 profile/contract로 등록하고 CPU gate부터 다시 통과시켜야
한다. 그렇지 않으면 “global control plane은 필요하지만 현재 TEMPO policy는 utility
승리를 만들지 못했다”는 재현 가능한 negative conclusion으로 닫는다.

## 5. TEMPO-GO가 실제로 만들어야 하는 scheme

```text
tenant ingress
  -> telemetry freshness/identity validation
  -> tenant SLO + weighted fairness ledger
  -> pair×route candidate builder
  -> multi-resource admission / bounded queue / explicit reject
  -> logical active-pair assignment
  -> immutable local or official remote commit
  -> scheduler + endpoint completion receipt
  -> complete/fail/quarantine/probe recovery
```

제어 대상은 한 transaction에서 연결하되, 시간축은 분리한다.

| 시간축 | 제어 | 허용 입력 | 금지 입력 |
|---|---|---|---|
| request | pair×route commit, credit, queue/reject | fresh telemetry, confirmed cache, request contract | future arrival, phase label, hidden fallback |
| epoch | service residual, fairness debt, route health | completion/first-response/EOF, endpoint snapshot | stale EWMA 단독 회복 |
| slow global | logical active pair, reservation, probe | sustained queue/SLO/endpoint pressure | physical migration, switch privilege |

pair×route resource vector:

- decoder: running/waiting sequences, decode tokens, KV usage
- local prefill: token-ms/count/service residual
- remote P service: request/queue/service duration
- transfer: KV bytes in flight, actual bytes, transfer completion
- semantic: remote operations in flight, receiver/install residual
- endpoint: first-response residual, request completion, stale/partial identity
- fabric advisory: pause/ECN/retry/blocked/timeout if available; attribution/safety용
- business: tenant debt, SLO remaining, queue wait, minimum service fraction

Pair activation은 prewarmed pair의 logical active set만 바꾼다. physical GPU migration이나 switch reconfiguration으로 과장하지 않으며 이미 commit된 request를 migrate하지 않는다.

### Business/fairness contract

tenant별로 weight, TTFT/TPOT/E2E SLO, maximum queue wait, minimum service fraction, request class를 고정한다. weighted request count가 아니라 weighted service debt와 raw service units를 분리한다. 반드시 tenant별 SLO-goodput, starvation, max wait, queue wait, Jain fairness, rejection reason을 보고한다.

### 다음 후보: admission-feasibility controller

최신 native 결과가 보여준 실패는 “queue가 가득 찼다”가 아니라 “reservation slot이
있어도 해당 요청을 SLO 안에 끝낼 service capacity가 없다”는 것이다. 따라서 다음
후보는 `queue_reservation_slots`만 조정하는 Candidate J가 아니다. request admission
때 다음 값을 pair×route별로 계산하는 별도 controller를 설계한다.

```text
feasible_finish = now
  + tenant_admission_wait
  + decoder_service_residual
  + prefill_or_remote_service_residual
  + transfer/semantic/endpoint residual

admit iff feasible_finish <= tenant E2E deadline
       and protected service-lane lease remains available
```

- queue slot과 service-lane capacity lease를 분리한다. latency/interactive는
  contract에 고정한 minimum-service lease를 가지며, background는 그 lease를
  선점하지 못한다.
- candidate가 deadline infeasible이면 단순 global timeout까지 기다리지 말고
  `global_tenant_slo_infeasible` 또는 `global_service_lane_unavailable`로 명시적
  defer/reject한다. 우선순위 tenant은 surviving pair와 inactive spare pair를
  같은 transaction에서 재평가한다.
- pair activation은 queue depth가 아니라 forecasted service infeasibility를
  줄일 때만 수행한다. activation 전후의 pair별 service residual, tenant SLO-goodput,
  reservation consumption을 receipt로 비교한다.
- 현재 one-way route commit, official `LMCacheConnectorV1:UCX`, first-response/
  EOF credit release, failure quarantine와 same-ID retry 금지는 그대로 유지한다.

CPU gate에서 반드시 숫자로 고정할 항목은 tenant별 minimum service fraction,
maximum reject/defer budget, protected lease units, service-residual estimator의
calibration source, pair activation trigger, control-plane overhead 상한이다. 같은
held-out trace에서 priority SLO-goodput이 실제로 증가하고 background starvation이
없으며, normal regression 3% 이내와 overhead gate를 통과하기 전에는 native
allocation을 사용하지 않는다. 이 후보가 실패하면 추가 reservation/timeout sweep을
중단하고, Candidate G/I와 함께 “global control은 필요하지만 이 workload/contract에서
promotion utility를 만들지 못했다”는 negative conclusion을 작성한다.

### Failure semantics

failure receipt에는 request ID, route/pair scope, telemetry sequence, failure kind, released resource, quarantine interval, `new_request_id_required`, probe/recovery event를 기록한다. 같은 request ID 재시도와 silent local fallback은 금지한다. pair-scope transport failure는 pair의 local/remote를 함께 quarantine할 수 있고, route-scope HTTP failure는 해당 semantic route만 격리한다.

## 6. Workload 설정법

### 6.1 공통 계약

- native topology: 4 nodes / 16 A100 / 실제 vLLM P/D / official LMCacheConnectorV1:UCX
- 같은 server lifecycle, topology, model, request trace, cache namespace, counterbalanced arm order
- output 2는 C1/C2 mechanism screen용 `screen_only`로만 취급한다. 현재 frozen held-out output128 workload는 foreground `512/16`, `2048/256`, `4094/16`, pressure stream `4094/128`을 명시적으로 사용한다. 새 output geometry를 만들면 manifest/profile/contract에 먼저 고정하고 기존 artifact와 섞지 않는다.
- prompt geometry: tokenizer로 검증한 512/2048/4094 tokens
- tenant: `latency`, `interactive`, `batch`, `background`
- cache state: `MISS`, `P_ONLY`, `D_ONLY`, `BOTH`, `UNKNOWN`; `UNKNOWN`은 hit가 아님
- phase metadata는 workload generator/analyzer에는 있어도 policy input에는 전달하지 않음
- synthetic CXI/network background는 headline workload가 아니라 별도 attribution ablation

### 6.2 C5 phase matrix

권장 흐름은 `C0 cool → C1 decoder-hot → C2 remote-hot → C2 P_ONLY/KV-hot → C3 both-hot → recovery`다. 각 phase는 15초, cooldown 2초, replicate 2회, explicit absolute `arrival_offset_ms`를 사용한다. 현재 v3 anchor는 총 2,712 rows이며 MISS 1,992 unique / P_ONLY 720이다. anchor rate는 C1 22.4/s, C2 remote-hot 4.76/s, KV/remote-hot 12/s이며 final universal constant가 아니라 characterization input이다.

held-out validation은 같은 trace 복사가 아니라 source pool과 arrival schedule을 새로 만들고, manifest/workload/profile/code/analyzer SHA를 모두 새 contract에 묶는다. 기존 v3 결과의 threshold/profile을 discovery 후 바꾸어 재사용하지 않는다.

### 6.3 현재 held-out output=128 artifact

현재 생성·검증된 held-out artifact는 v3 anchor를 덮어쓰지 않고 별도 경로에 둔다.

- manifest: `results/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json`
- manifest SHA-256: `6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c`
- workload: `results/tempo_go_c5_heldout_output128_v1/workloads/validation.jsonl`
- workload SHA-256: `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33`
- validator report SHA-256: `f00157c5f237c7a271197e499046e0e2a9884881cffeca46554accd015933fd0`
- 2,712 rows, replicate `r02/r03`, hot output `128`
- actual foreground geometry: `(512,16)`, `(2048,256)`, `(4094,16)`
- pressure streams use prompt geometry `4094` and `max_tokens=128`; 전체 workload의 `max_tokens` 분포는 `16:240`, `128:2,352`, `256:120`이다. 따라서 `background_output_tokens=128`과 foreground의 `512/16`, `2048/256`, `4094/16`을 혼동하지 않는다.
- cache contract: unique `MISS` 1,992 rows, `P_ONLY` 720 rows
- phase name, future arrival, oracle route, physical switch label은 workload metadata에만 있고 policy 입력에서 제외
- manifest execution contract: explicit absolute arrivals, warmup outside measurement, synthetic network background 금지, performance claim 금지

이 artifact는 workload validity가 닫힌 상태이지 native performance evidence가 아니다. source
endpoint/Elastic evidence의 geometry·residency·transport 대조와 frozen contract binding은
offline에서 닫혔고, 남은 단계는 승인된 interactive에서 native discovery를 실행하는 것이다.

현재 endpoint profile `results/tempo_go_c5_anchor_priors_c12_v3_retry1/real_tempo_pd_endpoint_service_profile_c12_anchor_output2_calibration_v3.json`은 SHA-256 `62181a17df4aaa66f12d77d3546bb22188a42ac4cf409c9579383a05b23eebaf`, fingerprint `f5e8a4d234638344f85c7db5970679b57710fa977d7f72856345055a52fe0f3`이며 17개 row가 모두 `P_ONLY`인 `calibration_only` profile이다. 따라서 이것만으로 held-out `MISS` remote path를 independent frozen profile로 승격하면 안 된다.

정확한 MISS receipt가 없을 때만 `FrozenServiceProxyPolicy`를 사용할 수 있다. 이 정책은 endpoint profile ID/fingerprint, calibration receipt SHA, 허용 geometry/residency/lookup mode를 명시하고 `proxy_is_not_exact=true`, `numeric_rows_unchanged=true`, `performance_claim_allowed=false`를 강제한다. 이번 승격에서는 `allowed_cache_residencies=[confirmed_miss,prefill_only]`와 `allowed_remote_cache_residencies=[prefill_only]`를 분리했다. 따라서 MISS는 P_ONLY 측정 row를 local service ceiling proxy로만 사용할 수 있고, MISS remote candidate는 fail-closed로 금지된다. P_ONLY remote는 exact endpoint row가 있을 때만 허용된다. 정책은 `GlobalOrchestratorConfig`의 capacity나 성능 입력이 아니며, exact MISS evidence를 가장하지 않는다.

strict source receipt와 새 frozen profile은 offline에서 닫혔다. 생성 도구는 `eval/sota_4node/build_tempo_go_heldout_frozen_proxy.py`이고, C4 manifest에 SHA로 결박된 네 raw stream(`00_local_r0`, `01_remote_r0`, `06_remote_r1`, `07_local_r1`)에서 네 held-out geometry `(512,16)`, `(2048,256)`, `(4094,16)`, `(4094,128)` 각각 local/remote 2개 이상, P_ONLY, cross-route output hash 동등성을 확인했다. 이는 MISS를 실측했다는 뜻이 아니라, P_ONLY calibration row를 명시적으로 제한된 proxy로 사용하는 provenance closure다.

- source receipt: `results/tempo_go_c5_heldout_frozen_proxy_v1/heldout_proxy_source_receipt.json`
- receipt SHA-256: `c7374e9af49dc0ac833e6f857bd64f4677c08d6d57529bab33cc9723f2ed800f`
- frozen endpoint: `results/tempo_go_c5_heldout_frozen_proxy_v1/frozen_endpoint_service_profile.json`, fingerprint `79a379a8d75aa4f00a678b6417620ae83fad9f2008da3662a894fa4285467057`
- frozen global: `results/tempo_go_c5_heldout_frozen_proxy_v1/frozen_global_profile.json`, fingerprint `7e67b52d2af30335b591e8145ee8f97ffe61958dc48c0bf831f783e9984d052`

이 artifact들은 native 성능 증거가 아니다. policy 또는 exact source receipt가 없으면 frozen native run contract를 만들지 않는다는 원칙은 유지한다.

quarantine robustness는 primary profile과 분리한 `frozen_failure_global_profile.json`에서만
검증했다. 정상 replay에서 실제 TEMPO remote 선택이 확인된 held-out index 591을
`injected_remote_route_failure` 대상으로 사용했고, 1 failure receipt, route-scope
quarantine, released work, `new_request_id_required`, terminal/leak-free를 확인했다.
첫 시도에서 remote가 아닌 index 499를 주입했을 때는 receipt 없이 fail-closed했으며,
그 결과를 robustness 성공으로 집계하지 않았다.

### 6.4 재현용 held-out manifest 생성 예시

```bash
# historical v3 anchor builder (background output flag는 metadata-only)
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=. \
  .vllm_venv/bin/python -m eval.sota_4node.build_tempo_go_c5_manifest \
  --source-512 <POOL512> --source-2048 <POOL2048> --source-4094 <POOL4094> \
  --model /pscratch/sd/s/sgkim/Skim-Tempo/models/Qwen2.5-7B-Instruct \
  --output-dir <NEW_RESULT_ROOT>/tempo_go_c5_phased \
  --replicates 2 --duration-ms 15000 --cooldown-ms 2000 \
  --foreground-rate 2 --decoder-hot-rate 22.4 \
  --remote-hot-rate 4.76 --kv-remote-hot-rate 12 \
  --anchor-output-tokens 2 --background-output-tokens 128

# final held-out output=128 artifact는 위 historical builder가 아니라 다음
# immutable builder를 사용한다. <NEW_RESULT_ROOT>가 비어 있지 않으면 실행하지 않는다.
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=. \
  .vllm_venv/bin/python -m eval.sota_4node.build_tempo_go_c5_heldout_manifest \
  --source-512 <POOL512> --source-2048 <POOL2048> --source-4094 <POOL4094> \
  --model /pscratch/sd/s/sgkim/Skim-Tempo/models/Qwen2.5-7B-Instruct \
  --parent-manifest <OLD_V3_MANIFEST> \
  --output-dir <NEW_RESULT_ROOT>/tempo_go_c5_heldout_output128_v1 \
  --duration-ms 15000 --cooldown-ms 2000 \
  --foreground-rate 2 --decoder-hot-rate 22.4 \
  --remote-hot-rate 4.76 --kv-remote-hot-rate 12 \
  --hot-output-tokens 128 --replicate-start 2 --miss-marker-base 200000

PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=. \
  .vllm_venv/bin/python -m eval.sota_4node.validate_tempo_go_manifest \
  --manifest <NEW_RESULT_ROOT>/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json \
  --workload <NEW_RESULT_ROOT>/tempo_go_c5_heldout_output128_v1/workloads/validation.jsonl \
  --output <NEW_RESULT_ROOT>/tempo_go_c5_heldout_output128_v1/manifest_validation.json
```

이 명령은 workload artifact만 만든다. profile binding과 endpoint prior가 새로 고정되기 전에는 native launcher에 전달하지 않는다.

### 6.5 비교 arm

동일 trace/topology/server lifecycle에서 다음을 비교한다.

1. `ALWAYS_LOCAL`
2. official LMCache `ALWAYS_REMOTE`
3. `PREDICTOR_ONLY`
4. `QUEUE_GPU_ONLY` 또는 Kairos-like fixed queue policy
5. `TEMPO_GO`

queue-GPU-only가 EngineCore/LMCache failure로 종료되면 latency 결과로 대체하지 않는다. clean fixed arm만 performance comparator로 쓰고, baseline failure는 별도 robustness evidence로 보고한다.

## 7. Telemetry와 analyzer 계약

요청별 ledger에는 arrival, classify, route commit, pair, credit acquire/release, upstream start, prefill start/end, KV transfer start/end/actual bytes, semantic operation, first token/first response, EOF/last token, TTFT/TPOT/E2E, cache evidence, queue depth/credit, decision reason, retry/fallback/timeout/failure, output token/text digest를 기록한다.

global snapshot에는 pair별 decoder scheduler running/waiting/KV, P/remote endpoint queue/service, receiver/install residual, route health, telemetry sequence/age/identity, tenant debt/SLO state를 포함한다. cross-host clock subtraction은 하지 않고 endpoint-owned duration을 사용한다. stale/missing/partial identity는 zero나 healthy로 대체하지 말고 fail-closed한다.

분석은 pooled aggregate만이 아니라 workload group, tenant, phase, pair, route, cache state별로 E2E/TTFT/TPOT p50/p95/p99, request/output-token goodput, SLO-goodput, queue wait, starvation, fairness, reject/failure, pair activation, telemetry overhead를 낸다. 선택된 route의 counterfactual은 actual selected-route prior로만 사용하고, route oracle이나 future input으로 정책을 오염시키지 않는다.

### 7.1 현재 CPU 검증 명령

다음은 login node에서 실행 가능한 bounded 검증이다. native GPU workload나 Slurm submit을 포함하지 않는다.

```bash
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=. \
  .vllm_venv/bin/python -m pytest -q \
  tempo/test_pd_global_*.py tempo/test_pd_endpoint_profile.py \
  eval/sota_4node/test_tempo_pd_elastic_frontend.py \
  eval/sota_4node/test_tempo_go_c5_run_contract.py \
  eval/sota_4node/test_tempo_go_c5_node.py
# 현재 기록: 130 passed, 11 subtests passed.
```

held-out manifest를 native launcher에 넘기기 전에는 다음 validator가 통과해야 한다.

```bash
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=. \
  .vllm_venv/bin/python -m eval.sota_4node.validate_tempo_go_manifest \
  --manifest results/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json \
  --workload results/tempo_go_c5_heldout_output128_v1/workloads/validation.jsonl \
  --output results/tempo_go_c5_heldout_output128_v1/manifest_validation_v2.json
```

validator가 workload geometry/cache contract를 닫아도 endpoint source receipt와 frozen policy가 자동으로 닫히는 것은 아니다. 그 둘과 새 run contract가 없으면 native 실행으로 넘어가지 않는다.

현재 생성된 frozen profile/contract를 다시 검증하는 bounded 명령은 다음과 같다.

```bash
module load pytorch/2.8.0
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=. \
  python -m eval.sota_4node.tempo_go_c5_run_contract verify \
  --repo-root /pscratch/sd/s/sgkim/Skim-Tempo \
  --contract results/tempo_go_c5_heldout_frozen_proxy_v3/native_run_contract.json \
  --sha256 c280a889e148069b2678c53dc3cdb738219e6c6a64f80b9594b220c7d2f4f3f4 \
  --workload-input results/tempo_go_c5_heldout_output128_v1/workloads/validation.jsonl

PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=. \
  python -m eval.sota_4node.replay_tempo_go_c5_five_arm \
  --manifest results/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json \
  --workload results/tempo_go_c5_heldout_output128_v1/workloads/validation.jsonl \
  --model models/Qwen2.5-7B-Instruct \
  --global-profile results/tempo_go_c5_heldout_frozen_proxy_v1/frozen_global_profile.json \
  --elastic-profile results/tempo_go_c5_anchor_priors_c12_v3_retry1/real_tempo_pd_elastic_profile_c12_anchor_output2_screen_v3.json \
  --endpoint-profile results/tempo_go_c5_heldout_frozen_proxy_v1/frozen_endpoint_service_profile.json \
  --output results/tempo_go_c5_heldout_frozen_proxy_v1/heldout_cpu_replay.json
```

이 replay는 five-arm control-plane/invariant 검증이다. 5개 arm 모두 2,712 terminal,
error/queue/inflight/resource leak 0, phase/physical-switch policy input false를 통과했으며,
performance claim은 여전히 금지된다. Replay harness는 frozen TEMPO policy와 fixed
baseline 후보 집합을 분리한다. 그렇지 않으면 MISS remote를 금지하는 TEMPO policy가
official-always-remote baseline까지 막아 비교 자체를 오염시키기 때문이다.

## 8. Correctness·성능·robustness gate

### 8.1 Correctness 선행 gate

- stream/output/token digest 100% 일치
- 모든 request가 정확히 한 terminal state
- route provenance/decision ledger 100% coverage
- hidden recompute/silent fallback/same-ID retry 0
- transfer error, unreceipted timeout, terminal queue residual, credit leak/double release 0
- 모든 workload group별로 gate 통과
- native EngineCore failure는 성공/latency sample이 아닌 execution failure

### 8.2 Primary performance gate

모든 조건을 함께 만족해야 한다.

- strongest fixed 대비 pooled E2E median 10% 이상 개선
- predictor-only 대비 E2E median 5% 이상 개선
- strongest fixed 대비 request 또는 output-token goodput 5% 이상 개선
- paired E2E win overall 75% 이상, 각 workload group 60% 이상
- 각 group의 E2E p99/TPOT p99 악화 5% 이내
- worst paired E2E regression 100 ms 이내
- selected local/remote 각각 counterfactual 대비 median 5% 이상 이득
- tenant starvation 없이 SLO-goodput/fairness를 별도 통과

### 8.3 Robustness 대체 gate

median gate가 부족해도 정상 부하 regression 3% 이내, overload p99 또는 goodput 15% 이상 개선, fatal failure/queue-timeout 억제, fairness/correctness 통과가 모두 있으면 performance가 아닌 robustness claim만 허용한다.

### 8.4 중단 규칙

- 두 구조적으로 다른 global candidate가 predictor 대비 5% 개선에 실패하면 threshold 미세조정을 반복하지 않는다.
- remote가 실제로 유리한 workload에서도 local 대비 5% 이득을 못 내면 remote branch와 workload validity를 재검토하고, 단순 polling/threshold 조정은 하지 않는다.
- tail과 median을 함께 달성할 수 없으면 simpler global admission/local-first/predictor-only로 축소한다.
- correctness/failure/fairness invariant를 안정적으로 못 닫으면 성능 실험을 중단한다.

## 9. Perlmutter 운영 규칙

### 반드시 지킬 것

- login node: 코드 수정, bounded file inspection, unit/replay/analyzer/static check만 수행
- GPU/vLLM/LMCache/traffic: 사용자가 승인한 native interactive 4-node/4-hour allocation 안에서만 실행
- 현재 interactive allocation이 없으면 임의 submit하지 않고 사용자 승인 대기
- 기존 allocation을 재사용할 수 있어도 자동 submit/cancel/retry loop를 만들지 않음
- 결과는 새 result root에 저장하고 이전 raw/result를 overwrite하지 않음

### 절대 금지

- container, Shifter, Apptainer, Podman, Docker, `--image`, udiRoot
- sudo/su/root ownership/setcap/CAP_NET_ADMIN/system file 변경
- `/etc`, `/usr`, `/opt` 수정
- shared filesystem의 unbounded recursive traversal
- root 권한을 고치려는 우회

`udiRoot.conf must be owned by user root` 또는 `exit 139`가 나오면 우회하지 않는다. command, environment, node, step state, stderr/stdout, exit/signal, artifact SHA만 보존하고 해당 arm을 execution failure로 판정한다.

## 10. 현재 상태와 바로 다음 작업

### 닫힌 것

- v0~v450 source와 v452~v544 raw lineage를 조사하고 route-only negative를 정리함
- C1/C2/C3 opposite crossover와 P_ONLY knee를 확인함
- global orchestrator의 tenant contract, bounded admission, pair activation, telemetry, endpoint completion, failure/quarantine lifecycle 구현
- CPU replay/failure injection/credit invariant/analyzer raw-backed closure
- native C5에서 business reject, scheduler/completion provenance, LMCache failure receipt 확인
- held-out output=128 workload builder와 manifest validator closure
- `FrozenServiceProxyPolicy`의 explicit identity/allowlist/fail-closed schema와 discovery-scope CPU replay closure
- tenant SLO deadline ordering fix와 current source revision 검증
- immutable C5 contract v10/v11의 historical guard/failure replay 및 terminal/leak gate
- held-out P_ONLY-only source receipt와 frozen endpoint/global profile 승격
- MISS remote 차단과 fixed official-remote baseline을 분리한 replay harness 수정
- candidate/fixed profile을 분리하는 held-out split replay harness 수정 및 source-contract 재동결
- Candidate E tenant wait-budget, Candidate G tenant queue reservation, Candidate H reservation+proactive-scale CPU gate를 동일 trace에서 닫음; E/G는 tail·SLO utility gate 실패, H는 baseline 중립
- held-out output=128 five-arm CPU replay 및 새 immutable contract/source-inventory verify
- held-out quarantine-enabled failure profile/replay: remote-selected target의 failure receipt, route quarantine, released work, terminal/leak gate

### 아직 안 된 것

- v1~v2 C5 contract는 이후 source revision/profile 변경으로 stale하며 덮어쓰지 않고 보존함. `57407705`는 historical held-out discovery로 남기고, 현재 native 기준은 source-rebound `57409956`/v3 contract다. 최신 native도 performance claim을 허용하지 않는다. 현재 소스의 최신 CPU candidate와 native result 모두 admission-feasibility gate를 통과하지 못했으므로 추가 native retry는 stop 상태다.
- endpoint profile `calibration_only`를 넘는 held-out independent validation
- TEMPO_GO의 strongest fixed/predictor 대비 native performance 또는 preregistered robustness 승리
- tenant SLO-goodput/fairness와 pair activation/scaling의 native benefit; v3에서는 starvation=false였지만 background rejection이 과도해 fairness/utility gate를 아직 닫지 못함
- queue-GPU-only failure와 TEMPO business reject를 분리한 v3 native analyzer report는 생성됐지만, TEMPO route/failure recovery의 independent validation은 아직 없음

### 다음 순서

1. 현재 source/test/evidence를 bounded audit한다.
2. held-out manifest/workload/validator SHA와 actual geometry를 재검증한다.
3. strict source receipt, frozen endpoint/global profile, policy route boundary를 확인한다.
4. 현재 held-out immutable native run contract의 source inventory·runner·node·analyzer SHA와 profile/workload binding을 verify한다. mismatch면 새 result root에서만 contract를 만들고 기존 root는 덮어쓰지 않는다.
5. frozen primary profile은 quarantine disabled, failure profile은 quarantine enabled라는 경계를 유지하며 analyzer/static launcher check를 닫는다.
6. E/G/H CPU gate가 이미 native stop 조건을 충족하지 못했으므로, route threshold를 미세조정하거나 같은 후보를 native에서 반복하지 않는다.
7. 연구를 계속할 경우에만 queue reservation 숫자가 아닌 admission-feasibility/service-lane lease를 바꾸는 구조적으로 다른 Candidate J를 새 immutable contract/result root로 설계하고 CPU replay/fairness/failure/overhead gate부터 통과시킨다. v3/E/G/H/I root와 contract는 덮어쓰지 않는다.
8. 새 candidate가 correctness/fairness와 사전 등록 performance/robustness gate를 모두 통과할 때만 사용자가 승인한 native 4-node/16-GPU/4-hour interactive independent validation을 수행한다. `57409956`에서 이미 실제 native admission failure를 확인했으므로, 같은 profile/trace의 blind retry는 하지 않는다.
9. independent validation에서 strongest fixed/predictor 대비 성능 또는 사전 등록 robustness gate를 통과한 경우에만 최종 scheme claim을 작성한다.

held-out source/profile/contract freeze는 닫혔고, historical allocation `57407196`에서
native 실행을 시도했다. 첫 v1 contract는 compute-side source inventory mismatch로
차단됐으며, 그 receipt는 stale-contract execution failure로만 보존한다. 현재 node
source를 포함한 v2 contract로 다시 시작한 실행은 vLLM/LMCache/UCX와 frontend/proxy
health까지 올라갔고, warmup artifact 2,712 rows를 생성했지만 measured `raw.json` 전에
수동 `INT`로 종료됐다. 따라서 v2도 incomplete execution receipt이며 native validation,
performance, fairness 또는 robustness result가 아니다. 새 output=128 workload가
존재한다는 사실만으로 성능을 주장하지 않는다. 현재 frozen profile은 exact MISS profile이
아니라 P_ONLY-only evidence-bound proxy이며, MISS remote는 정책상 금지된다. native
실측 결과가 correctness/fairness/failure gate를 먼저 통과하기 전에는 성능 claim을 하지
않는다.

현재 native 기준은 allocation `57409956`의 source-rebound v3이다. contract는
`results/tempo_go_c5_r8_16_20_20_contract_v3/native_run_contract.json` (SHA
`002ee5424c9779b22d2cc622cb9143227f8370d03d6b22d0f3c9a560f153e481`, fingerprint
`7691d005cad942c26a9a8792cf1487431ce5c4f7abe43ebb7b409a2fef5a854e`), analyzer는
`results/tempo_go_c5_r8_16_20_20_native_job_57409956_v3/native_five_arm_analysis.json`
(SHA `b7e302ab1f893310602b491a8971138d3f4b3cd7fa906b4f7ce05848ac305f45`)이다.
local/remote/predictor는 각각 2,712/2,712 complete였고 request goodput은
7.934/9.581/7.928 s⁻¹, output-token goodput은 981.8/1,185.7/981.1 s⁻¹였다.
queue-GPU-only는 exit 143의 `IncompleteRead` execution failure로 raw latency가
없다. TEMPO는 2,712 terminal-valid 중 982 complete, 1,730 explicit global reject,
0 failure였고, background 2,436건 중 769건만 완료했다. endpoint completion 982,
valid scheduler observation 5,424, pair activation 1을 기록했지만 performance 및
fairness/utility gate는 실패했다. 따라서 이 run은 실제 global admission/telemetry
통합과 현재 admission-feasibility 정책의 실패를 증명하며, 성능 승리나
production-ready 결론이 아니다.

## 11. Authoritative artifact index

- 통합 기준 문서: 이 파일
- 상세 계보/playbook: `paper/TEMPO_GLOBAL_ORCHESTRATOR_CANONICAL_PLAYBOOK.ko.md`
- 이전 master: `paper/TEMPO_RESEARCH_MASTER_STATE_AND_NEXT_GOAL.ko.md`
- contention audit: `paper/TEMPO_ELASTIC_PD_CONTENTION_AUDIT.md`
- goal file: `eval/sota_4node/TEMPO_GO_GLOBAL_ORCHESTRATOR_GOAL_V1.ko.md`
- C5 v3 manifest: `results/tempo_go_c5_cpu_gate_20260821_anchor_v3_retry2/tempo_go_workload_manifest.json`
- C5 manifest SHA: `849bb5cf284c60215d12165e409ac426adc6e5bba3427cda8932c7379fb819fd`
- validation workload SHA: `38224ae6e421a0950080951a963ff7d82af480edfa15220c9a45c5c2064ad2f5`
- guarded five-arm root: `results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1`
- guarded five-arm analysis SHA: `921ec4ad74dc28604bc65a65a734e8638817cf4d1b51d745a416064820cd350d`
- C5 contract verifier: `eval/sota_4node/tempo_go_c5_run_contract.py`
- C5 frozen contract v10 (normal guard discovery): `results/tempo_go_c5_frozen_contract_v10/native_run_contract.json`
- v10 contract file SHA/fingerprint: `63d33edf83c5825ba9d1981e68f0ece761e739d6d1b977e610be6f947d3c065c` / `d37b8330734a7479f48c8bd844cccbe91403f96d368918119197ddacb598a737`
- v10 normal offline replay: `results/tempo_go_c5_frozen_contract_v10/offline_replay.json`, SHA `9010c46bc0949419518b7dcf15ab2a8ef5b1d0d2a46f47d3188d1b52942c5496`
- C5 frozen contract v11 (quarantine failure replay): `results/tempo_go_c5_frozen_contract_v11/native_run_contract.json`
- v11 contract file SHA/fingerprint: `7713f6414c34c6a6ef52f485e546b11086620bb83b87f1ccc2ccacc9facb6699` / `76c6651ab8b673f78bc1173a08e66d885414050015236bd11b942572abc31728`
- v11 failure offline replay: `results/tempo_go_c5_frozen_contract_v11/offline_failure_replay.json`, SHA `2b38d895d77ee56da7112ef168a97082da2c40fb002acd44e9bb7bfccfbaf5b0`
- failure-injection global profile: `results/tempo_go_c5_failure_profile_v1/real_tempo_go_profile_c12_failure_injection_v1.json`, SHA `33b4feebc47ef7bb8686d986082ca826f5e4bafe343efb7cf858eab8ee3b0327`, fingerprint `d5db711c984a06572eb594ed9c7ab175e4aba5e2002f5b3138a5cc7614baa906`
- old v2 contract SHA/fingerprint (stale, immutable evidence only): `b34a4b52d81b45957a1ef1d5c8bb3f3a1a54c8dabdd39d12bd23dc14d80197af` / `df8d85610f70d72d62dc0a36962a09b7190e28b05526dd051364653733adf248`
- failure-quarantine root: `results/tempo_go_c5_native_failure_quarantine_job_57404614_v1`
- failure raw SHA: `c61626d6cef2b7353e0ec8a21609a9bc3b72ea6e4ed240ff5de2216cf9292124`
- raw-backed failure analysis SHA: `579f92d38140f0f7ccb31f18a19ce9c9670ea5b3371ba48e99cf7850dbd3a1ac`
- held-out manifest: `results/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json`
- held-out manifest/workload/validator SHA: `6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c` / `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33` / `f00157c5f237c7a271197e499046e0e2a9884881cffeca46554accd015933fd0`
- current endpoint calibration source SHA/fingerprint: `62181a17df4aaa66f12d77d3546bb22188a42ac4cf409c9579383a05b23eebaf` / `f5e8a4d234638344f85c7db5970679b57710fa977d7f72856345055a52fe0f3`
- current endpoint source scope: `calibration_only`, 17 rows, all `P_ONLY`; not sufficient as exact held-out MISS evidence
- held-out source receipt: `results/tempo_go_c5_heldout_frozen_proxy_v1/heldout_proxy_source_receipt.json`, SHA `c7374e9af49dc0ac833e6f857bd64f4677c08d6d57529bab33cc9723f2ed800f`
- held-out frozen endpoint: `results/tempo_go_c5_heldout_frozen_proxy_v1/frozen_endpoint_service_profile.json`, SHA `f4a4939cfcffe08a9a7b21d732f684787fc0dbe06bc32d65c3d477f005ca363e`, fingerprint `79a379a8d75aa4f00a678b6417620ae83fad9f2008da3662a894fa4285467057`
- held-out frozen global: `results/tempo_go_c5_heldout_frozen_proxy_v1/frozen_global_profile.json`, SHA `51b297c104aa5f5b0a7dff16499ce94b1f93267842999d494a83dbf8f6a81b2a`, fingerprint `7e67b52d2af30335b591e8145ee8f97ffe61958dc48c0bf831f783e9984d052`
- held-out CPU replay: `results/tempo_go_c5_heldout_frozen_proxy_v1/heldout_cpu_replay.json`, SHA `d66e3dfd56d9ae1460d4355c896ef250a9b72a11f58f751909e9105c6dc68d3e`; performance claim false, all five arms terminal/leak-free
- historical held-out v1 native run contract: `results/tempo_go_c5_heldout_frozen_proxy_v1/native_run_contract.json`, SHA `0ce783401d58f5606211d9664184bc49fdd450f0e8dd5fc3b4cdb92c15f18cb7`, fingerprint `1cfe4e6c66503bc5fc67b756a0ac60c6cb6f0f8f8acd8792e472cbf433a4f3e1`
- held-out v1 native attempt: `results/tempo_go_c5_native_heldout_frozen_proxy_v1_job_57407196/local/failure.json`, SHA `9e85ae8c677cb79451c52a2a7b1c52dc1d758d302a26407c207e68b9d6c8db39`; stale v1 source-contract mismatch, no performance evidence
- held-out v2 native run contract: `results/tempo_go_c5_heldout_frozen_proxy_v2/native_run_contract.json`, SHA `b9e3a16d05ae2dcf420f00bda7b8bc6912cd4b13ed5dbc3f6c3275db7ec47aba`, fingerprint `4ad18da2c062dc4b0ad132d1c4be2501bdbe4c0bd1ef492a6eb611444eaf10b5`; source-integrity corrected, performance claim false
- held-out v2 native partial attempt: `results/tempo_go_c5_native_heldout_frozen_proxy_v2_job_57407196/local/failure.json`, SHA `af223ad9a2fb5e8e922e3a12ac37aa2e42395b1b90720c482162784fb77e7c0d`; `native_arm_step_signal`, local arm only, warmup-only, no measured raw/result
- current held-out v3 native run contract: `results/tempo_go_c5_heldout_frozen_proxy_v3/native_run_contract.json`, SHA `c280a889e148069b2678c53dc3cdb738219e6c6a64f80b9594b220c7d2f4f3f4`, fingerprint `1fd9ff9f894b916a855c9aa93adb66a4a1bc4e1d05107cb09e690f300d857b73`; source revision includes fixed-baseline-failure continuation
- current native v3 five-arm root: `results/tempo_go_c5_native_heldout_output128_job_57407705_v3`
- current native v3 analyzer: `results/tempo_go_c5_native_heldout_output128_job_57407705_v3/native_five_arm_analysis.json`, SHA `982f3bb6df93986bbfcd491e733819590efb341341cbecd879119cda53939332`; `performance_claim_allowed=false`
- latest source-rebound native v3: `results/tempo_go_c5_r8_16_20_20_native_job_57409956_v3`, analyzer SHA `b7e302ab1f893310602b491a8971138d3f4b3cd7fa906b4f7ce05848ac305f45`; contract SHA `002ee5424c9779b22d2cc622cb9143227f8370d03d6b22d0f3c9a560f153e481`, fingerprint `7691d005cad942c26a9a8792cf1487431ce5c4f7abe43ebb7b409a2fef5a854e`
- current v3 raw/failure SHAs: local `d8d3630889664d6000fbd0907c3a0c69590490f63a9f41c9a4395b27d29cee85`, remote `b4ecfa889eefb9985da463c62240352c99139f8972e288eccb2cd96b9f20dfdf`, predictor `d932973adaec6d0a110bf80d19d4f356ec8320dee34960665e3f85d8d94f5996`, queue-GPU failure `e9ae3a272d543fd365f3bd20e3a0112209c071855d25a36ca360d7d44c5b97cf`, TEMPO `799ed43939ec6a751daa5ab6e4f0854d290392c6c4d7e3e48dde730b498c48ab`
- current v3 native interpretation: local/remote/predictor `2,712/2,712` complete; queue-GPU-only `exit 143` execution failure; TEMPO `900 complete / 1,812 explicit global reject / 0 failed`; performance claim forbidden. Global scheduler observations are `5,424` with invalid `0`, and endpoint completion receipts are `900`.
- latest source-rebound interpretation supersedes the preceding historical line: local/remote/predictor `2,712/2,712` complete with request goodput `7.934/9.581/7.928 s⁻¹`; queue-GPU-only exit 143 `IncompleteRead`; TEMPO `982 complete / 1,730 reject / 0 failed`, background `769/2,436`, interactive `80/96`, latency `50/96`, endpoint receipts `982`, scheduler observations `5,424`, pair activation `1`; performance/fairness claim forbidden.
- latest corrected CPU sweep best profile: `results/tempo_go_c5_reservation_sweep_profiles/r8_16_20_20.json`; replay SHA `8c4db9ff6e5ccf770f409bff4670eef52ac586cb7e4f7a73f168d6d177c1c8dd`; TEMPO `1,430/1,282` complete/reject, total SLO-good `990`, weighted SLO-good `884.0`; CPU-only evidence.
- split replay harness: `eval/sota_4node/replay_tempo_go_c5_five_arm.py`, SHA `3d5396f4c1988fe848c45f4f234c0cf6f0ae5dcd02d0f567f39b4b40fc7ca72f`; fixed arms and TEMPO candidate now have separate frozen global profiles with shared workload/Elastic/endpoint identities
- Candidate E global profile/replay: `results/tempo_go_c5_candidate_e_tenant_budget_frozen_proxy_v1/frozen_global_profile.json` / `heldout_cpu_replay_split.json`, SHA `b1c0257762d1ea9fee0377edc4fffb99902c1abdcb156312faed7d38e1c15630` / `8d3d67d65b04e0488e5c4a1f1601b139b51b7f85ac42f3c89a6cc47483efa7b6`; 5 s global wait candidate, performance claim forbidden
- Candidate G profile/replay/contract: `results/tempo_go_c5_candidate_g_tenant_reservation_v1/`, profile/replay/contract SHA `28a1b9f4fc1033f01d340f37e59ca04eeb26233afaf4a587ddfb02d33270dcba` / `099281de85332879ad9d22e87f237b3c0e207830e0363d5bae45f399f95c5635` / `701b2e8bea75471a597b21f43554a63f386afd2621e413c0265f6898e67b2cf4`; contract fingerprint `8c68dbe41ffa1126a9a364e459b85b3130df92cfa80fec8824ff6515a73069d8`; performance claim forbidden
- Candidate H profile/replay: `results/tempo_go_c5_candidate_h_reserve_scale_v1/`, SHA `75b7ad821a1831195ada9dd68e8248190f54baedec7339b7cb339e66024a1144` / `86945f039e41a6ba7d3faa53033515ac807b50448e589a1a0b328d4efd4e7401`; reservation+proactive-scale candidate, performance claim forbidden
- held-out v2 warmup SHA: `755e62d5fd3b8f649b84ee89cc4ce045a50067cf9adc77f50d11a2ee11d66a42`; 2,712 generated warmup rows, not measured evidence
- held-out v2 node-0 vLLM log SHA: `d215bbe2ae1d37aad24a335b7fbd1ddc66a83d6f49657b8e642b16eb7df4aaf5`; EngineCore/LMCache/UCX initialized and API health returned before the manual signal
- held-out promotion utility: `eval/sota_4node/build_tempo_go_heldout_frozen_proxy.py`
- held-out failure profile utility: `eval/sota_4node/build_tempo_go_frozen_failure_profile.py`
- held-out failure profile: `results/tempo_go_c5_heldout_frozen_proxy_v1/frozen_failure_global_profile.json`, SHA `2e9e17e8ba4cfd049e77a975ea1ddc1f5d7d98d8093bb01e4c9b6dcf31011966`, fingerprint `3eca3db05f6d2081d0d03a446189342b3ada9563cfda6bc6c15f234250fb3554`
- held-out failure CPU replay: `results/tempo_go_c5_heldout_frozen_proxy_v1/heldout_failure_cpu_replay_v2.json`, SHA `2ee369b24450bd89c3488cb5a1e2aa2d613f02c964671165f779b847802f1fed`; failure receipt/quarantine/terminal-leak gate pass, performance claim false
- post-policy discovery CPU replay: `results/tempo_go_c5_quarantine_replay_v3_after_proxy_contract.json`, SHA `46af0a06cdc8a043caf31bd5852240040a54810ce3365cbfd94467b2ce332c64`
- old v2 contract SHA/fingerprint (stale, immutable evidence only): `b34a4b52d81b45957a1ef1d5c8bb3f3a1a54c8dabdd39d12bd23dc14d80197af` / `df8d85610f70d72d62dc0a36962a09b7190e28b05526dd051364653733adf248`
- guard profile SHA: `8082f4190d56016d7bac6abacbf659017a4fb20a50d1b474223cf9157c1fd3ec`
- guard profile fingerprint: `f8163ff115a2478614afccf57b02a1c535c7dd4e2b3e54f47beda83d1ae3c2a0`
- C5 failure profile SHA: `33b4feebc47ef7bb8686d986082ca826f5e4bafe343efb7cf858eab8ee3b0327`

기존 artifact는 immutable evidence다. 새 실행은 새 directory, 새 profile ID, 새 contract SHA를 사용한다.

## 12. 다음 에이전트에게 주는 개선 목표 프롬프트

아래 블록을 그대로 새 목표 prompt로 사용한다.

> 이 저장소의 연구 목표를 임의로 축소하거나 request-local predictor 연구로 바꾸지 말라. 먼저 `paper/TEMPO_RESEARCH_HANDOFF_AND_IMPROVEMENT_GOAL.ko.md`를 처음부터 끝까지 읽고, 이어서 `paper/TEMPO_GLOBAL_ORCHESTRATOR_CANONICAL_PLAYBOOK.ko.md`, master, contention audit, goal 문서를 대조하라. v535 하나만 보지 말고 conceptual v0, actual P/D v1~v450 source, v452~v544 raw-artifact lineage, C0~C5 receipt를 계보로 취급하라. workspace에 없는 v545~v600을 추정하지 말고 exact path/SHA가 있는 자료만 사용하라.
>
> 원래 목표를 유지하라: “TEMPO Elastic-PD를 실제 vLLM P/D 경로에 통합하고, 단순 predictor와 가장 강한 고정 정책보다 유의미하게 빠른 하나의 최종 스킴으로 확정한다.” 현재는 TEMPO-GO global orchestrator를 만든다. Perlmutter native 4-node/16-A100, 실제 vLLM P/D, official `LMCacheConnectorV1:UCX` data plane을 고정하고, moving multi-tenant contention에서 decoder admission, tenant SLO/fairness, pair assignment 및 prewarmed logical scaling, local/remote route, endpoint congestion/failure quarantine/recovery를 하나의 control plane으로 공동 제어하라. LMCache transport를 교체하거나 연구 목표를 route-only predictor로 축소하지 말라.

> 현재 CPU/native stop/go도 반영하라. Candidate E(global wait 5 s), G(tenant queue reservation), H(reservation+proactive scale), I(telemetry-failure pair circuit)는 CPU correctness는 통과했지만 primary 또는 robustness utility gate를 통과하지 못했다. 최신 r8 reservation sweep은 CPU weighted SLO-goodput 884로 가장 좋았지만 native `57409956`에서 982 complete/1,730 reject와 high-priority rejection을 보였다. E/G/H/I를 native에서 맹목 반복하거나 queue capacity/timeout/reservation 숫자만 sweep하지 말라. 새 후보는 queue slot이 아니라 pair×route service-feasibility와 tenant service-lane lease를 바꾸는 구조적으로 다른 mechanism이어야 하며, 새 profile/contract SHA와 CPU overhead/fairness/failure gate를 만든 뒤에만 native allocation을 고려하라.
>
> 먼저 현재 결론을 정확히 보존하라. C3에서 local/remote crossover와 contention은 실존한다. C4의 scalar pressure/phase/request-local route 후보들은 strongest fixed, predictor, tail, goodput gate를 통과하지 못했으므로 route threshold 미세조정은 종료한다. Native `57402376`과 historical `57407705`는 integration/discovery receipt이지 performance 승리가 아니다. 최신 source-rebound native `57409956`에서 local/remote/predictor는 2,712 complete였지만 TEMPO는 982 complete/1,730 explicit reject, request goodput 4.786/s로 local 7.934/s·remote 9.581/s·predictor 7.928/s보다 낮았다. queue-GPU-only는 exit 143 `IncompleteRead` execution failure였다. `57404614`는 LMCache/EngineCore failure와 global failure receipt/quarantine을 보여준 robustness evidence이지 성능 결과가 아니다. Candidate B/D/G/I의 CPU neutral/negative 결과도 성능 승리로 재사용하지 말라.
>
> held-out workload는 이미 별도 immutable artifact로 생성됐다: manifest `results/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json` (SHA `6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c`), workload (SHA `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33`), validator report SHA `f00157c5f237c7a271197e499046e0e2a9884881cffeca46554accd015933fd0`. 실제 2,712 rows, r02/r03, hot output=128, foreground geometry `(512,16)/(2048,256)/(4094,16)`, unique MISS 1,992/P_ONLY 720이다. historical v3의 output=2 anchor와 섞지 말고, phase/future arrival/oracle route/physical switch label은 policy input에서 제외하라.
>
> strict promotion은 이미 offline에서 실행됐지만 exact MISS evidence는 여전히 없다. `allowed_remote_cache_residencies=[prefill_only]`를 반드시 보존하여 MISS remote candidate를 열지 말라. `heldout_proxy_source_receipt.json`, frozen endpoint/global profile, contract SHA를 서로 대조하고 하나라도 mismatch면 native에서 fail-closed하라.
>
> old v1~v9 `native_run_contract.json`은 이후 source/profile revision 때문에 stale하다. 덮어쓰거나 재사용하지 말라. v10/v11은 historical guard normal/failure replay 전용이다. `results/tempo_go_c5_heldout_frozen_proxy_v3/native_run_contract.json` (SHA `c280a889e148069b2678c53dc3cdb738219e6c6a64f80b9594b220c7d2f4f3f4`, fingerprint `1fd9ff9f894b916a855c9aa93adb66a4a1bc4e1d05107cb09e690f300d857b73`)은 최신 완료 native discovery의 immutable receipt로만 사용한다. 현재 소스의 최신 CPU candidate contract는 `results/tempo_go_c5_candidate_g_tenant_reservation_v1/native_run_contract.json` (SHA `701b2e8bea75471a597b21f43554a63f386afd2621e413c0265f6898e67b2cf4`, fingerprint `8c68dbe41ffa1126a9a364e459b85b3130df92cfa80fec8824ff6515a73069d8`)이며, CPU utility gate 실패로 native 실행은 stop이다. v1/v2 contract와 `57407196`/`57407330` attempt는 historical evidence로만 보존한다. contract에는 candidate ID/revision, 4-node/16-GPU topology, exact `LMCacheConnectorV1:UCX`, arm order, manifest/workload/model SHA, global/Elastic/endpoint profile path/SHA/fingerprint/scope, service-proxy policy receipt, controller/frontend/node-entry/runner/analyzer source inventory SHA, runtime/environment, output geometry, tenant SLO contract, gates를 포함하라. runner/node는 contract/profile/manifest/code/telemetry identity mismatch에서 fail-closed해야 한다.

> 위 문단의 “최신” 표현은 historical artifact 설명이다. 현재 source-bound native contract는 `results/tempo_go_c5_r8_16_20_20_contract_v3/native_run_contract.json`이며, candidate `tempo-go-c12-r8-16-20-20`, revision `tenant-reservation-admission-v2-per-request-timeout-replay-v2-source-rebound-v3`, allocation `57409956`에 결박된다. 이 contract도 `performance_claim_allowed=false`이므로 independent validation contract로 재사용하지 말라.

> 최신 native discovery도 정확히 보존하라. allocation `57407705`의 analyzer는 `results/tempo_go_c5_native_heldout_output128_job_57407705_v3/native_five_arm_analysis.json` (SHA `982f3bb6df93986bbfcd491e733819590efb341341cbecd879119cda53939332`)이다. local/remote/predictor는 2,712 complete였고 queue-GPU-only는 raw 없이 exit 143 execution failure였다. TEMPO는 900 complete, 1,812 explicit global reject, 0 failure였으며 background는 2,436개 중 1,748개를 reject했다. TEMPO request goodput 4.382/s는 local 7.876/s, remote 9.695/s, predictor 7.801/s보다 낮으므로 performance claim을 하지 말라. global scheduler observation은 5,424건(유효 5,424), endpoint completion receipt는 900건이다. 이 결과의 valid conclusion은 global telemetry/admission/failure containment가 native에서 발동했다는 것과, 현재 `maximum_queue_wait_ns=2 s` admission budget이 과보수적이라는 것이다. 실제 fabric bottleneck 위치는 이 receipt만으로 단정하지 말라.

> **최신 source-rebound native correction (이 문단이 위 historical paragraph보다 우선한다).** allocation `57409956`의 contract는 `results/tempo_go_c5_r8_16_20_20_contract_v3/native_run_contract.json` (SHA `002ee5424c9779b22d2cc622cb9143227f8370d03d6b22d0f3c9a560f153e481`, fingerprint `7691d005cad942c26a9a8792cf1487431ce5c4f7abe43ebb7b409a2fef5a854e`)이고 analyzer는 `results/tempo_go_c5_r8_16_20_20_native_job_57409956_v3/native_five_arm_analysis.json` (SHA `b7e302ab1f893310602b491a8971138d3f4b3cd7fa906b4f7ce05848ac305f45`)이다. local/remote/predictor는 2,712 complete, request goodput 7.934/9.581/7.928/s, output-token goodput 981.8/1,185.7/981.1/s였다. queue-GPU-only는 exit 143 `IncompleteRead` execution failure로 latency를 만들지 못했다. TEMPO는 982 complete/1,730 explicit global reject/0 failure, background 769/2,436, interactive 80/96, latency 50/96 complete였다. endpoint completion 982, valid scheduler observation 5,424, pair activation 1이 닫혔지만 performance/fairness/utility gate는 실패했다. `maximum_queue_wait_ns=5 s`와 queue reservation은 queue occupancy만 보호하고 실제 service feasibility를 보장하지 못했다. 따라서 실제 fabric link/NIC 병목을 단정하지 말고 다음 후보를 admission-feasibility/service-lane lease mechanism으로 설계하라.

> 현재 CPU gate 결과도 함께 보존하라. v10 normal replay/v11 failure replay와 held-out output=128 normal/failure replay 모두 `performance_claim_allowed=false`이며 native GPU 실행 허가가 아니다. held-out frozen primary profile은 quarantine disabled로 유지하고, failure injection은 별도 quarantine-enabled profile/contract에서만 실행하라. disabled profile에서 failure replay를 억지로 계속하지 말고 fail-closed하라. 실제 remote가 선택되지 않은 request를 failure target으로 지정했을 때도 receipt를 만들지 말고 오류로 남겨야 한다.
>
> workload는 `C0 cool → C1 decoder-hot → C2 remote-hot → C2 P_ONLY/KV-hot → C3 both-hot → recovery`를 사용하고 stable/burst/overload/recovery를 포함하라. prompt 512/2048/4094를 tokenizer로 검증하고, 현재 frozen held-out output128 artifact의 `max_tokens=16/128/256` 분포를 그대로 사용하라. `latency`/`interactive`/`batch`/`background` tenant와 `MISS`/`P_ONLY`/`D_ONLY`/`BOTH`/`UNKNOWN` cache contract를 분리하라. `UNKNOWN`은 hit가 아니다. 같은 server lifecycle/topology/cache namespace/request trace를 사용하고 arm 순서는 counterbalanced로 고정하라. phase label, future arrival, oracle route, physical switch label, synthetic network background를 headline policy input으로 넣지 말라. output=2는 mechanism screen일 뿐 final performance evidence가 아니다.
>
> controller 입력은 scalar `fabric_pressure` 하나가 아니다. pair별 decoder scheduler running/waiting/KV, local prefill token-ms, remote P service, remote KV bytes, semantic operation, endpoint first-response/completion residual, telemetry sequence/age/identity, tenant queue/SLO/debt를 분리하라. cross-host clock subtraction을 하지 말고 endpoint-owned duration을 사용하라. local/remote/semantic credits를 독립적으로 bound하고 bounded queue와 explicit reject만 허용하라. request마다 pair×route immutable commit을 하고 prefill 후 route 변경, hidden recompute, silent fallback, same-ID retry를 금지하라. first response/EOF/complete/abort/timeout/failure에서 resource를 exactly once 반환하라.
>
> tenant fairness는 weighted request count가 아니다. weighted service debt와 raw service units를 분리하고 tenant별 TTFT/TPOT/E2E SLO-goodput, queue wait/max wait, starvation, Jain fairness, minimum service fraction, rejection reason, pair activation을 보고하라. pair activation은 prewarmed logical active set만 바꾸고 physical migration이나 이미 commit된 request migration을 하지 말라. stale/partial telemetry는 healthy/zero로 대체하지 말고 fail-closed한다.

> v3 native에서 모든 tenant가 starvation=false였어도 background 2,436건 중 1,748건을 reject했으므로 이를 fairness 성공으로 해석하지 말라. 다음 candidate는 run 전에 tenant별 minimum service fraction, maximum reject/defer budget, priority/SLO rule을 contract에 숫자로 고정하고, 같은 workload에서 admitted service와 SLO-goodput을 높이면서 high-priority tail을 악화시키지 않는지 검증해야 한다. completed-only latency를 전체 goodput 개선으로 보고하지 말고, reject와 failure를 separate terminal class로 유지하라.

> 최신 correction: source-rebound `57409956`에서는 queue reservation을 넣었어도 background 2,436건 중 769건, interactive 96건 중 80건, latency 96건 중 50건만 완료했다. 그러므로 reservation이 priority fairness를 보장한다고 쓰지 말고, service-lane feasibility와 실제 admitted SLO-goodput을 검증하라.

> 다음 후보는 Candidate J `admission-feasibility/service-lane lease`다. queue reservation만 늘리지 말고, pair×route별 decoder/P/remote/transfer/semantic/endpoint service residual을 사용해 `feasible_finish <= tenant E2E deadline`을 계산하라. latency/interactive의 protected service lease, batch/background의 elastic lane, surviving-pair/spare-pair 재평가, `global_tenant_slo_infeasible`와 `global_service_lane_unavailable` explicit terminal reason을 contract에 넣어라. activation 전후 service residual/SLO-goodput/lease consumption을 receipt로 남겨라. CPU에서 priority SLO-goodput, background starvation, normal regression 3%, control-plane overhead와 failure quarantine gate를 모두 통과하기 전에는 native를 실행하지 말라.
>
> 비교 arm은 `ALWAYS_LOCAL`, official LMCache `ALWAYS_REMOTE`, `PREDICTOR_ONLY`, `QUEUE_GPU_ONLY`/Kairos-like fixed queue policy, `TEMPO_GO`다. 같은 topology/GPU budget/server lifecycle/request/cache namespace에서 실행하라. queue-GPU-only가 EngineCore/LMCache failure면 latency sample로 대체하지 말고 execution failure로 분리하라. LMCache failure는 explicit `tempo-go-global-failure-v1` receipt, released work, route/pair quarantine scope, telemetry sequence, probe/recovery, new-request-ID rule로 기록하라.
>
> 다음 실행 순서는 최신 `57409956` native receipt와 CPU sweep 분석 → queue slot이 아닌 pair×route service-feasibility/service-lane lease를 바꾸는 Candidate J 설계 → tenant minimum service/reject/defer budget, pair-scale, endpoint-recovery와 estimator calibration을 함께 고정 → 새 code/profile/contract SHA freeze → CPU replay/fairness/failure/overhead/analyzer/static gate → 새 승인 native 4-node/16-GPU/4-hour independent validation이다. 기존 native root를 덮어쓰지 말고 단순 route threshold만 튜닝하지 말라. correctness gate를 먼저 통과한 뒤 strongest fixed 대비 E2E median 10%, predictor 대비 5%, goodput 5%, paired/tail/fairness gate를 모두 검사하라. median gate가 부족해도 정상부하 regression 3% 이내에서 overload p99/goodput 15% 개선과 failure/fairness/correctness를 통과한 경우에만 robustness claim을 하라. queue-GPU-only failure를 성능 sample로 대체하지 말고, TEMPO reject를 completion으로 위장하지 말라.
>
> 두 구조적으로 다른 global candidate가 predictor 대비 개선하지 못하면 threshold 미세조정을 반복하지 말고 simpler global admission/local-first/predictor-only로 축소하여 재현 가능한 negative conclusion을 작성하라. 선택 route의 counterfactual은 실제 measured prior로만 계산하고 oracle/future input으로 정책을 오염시키지 말라.
>
> Perlmutter에서는 login node에서 코드 수정, bounded inspection, CPU test/replay/analyzer만 수행하고 GPU/vLLM/LMCache workload는 사용자가 승인한 native interactive 4-node/4-hour allocation 안에서만 실행하라. Slurm 자동 submit/cancel/retry loop를 만들지 말라. container/Shifter/Apptainer/Podman/Docker/`--image`/udiRoot/sudo/su/root ownership/setcap/CAP_NET_ADMIN/system-file 변경을 절대 하지 말라. `udiRoot.conf must be owned by user root` 또는 `exit 139`가 나오면 우회하지 말고 command/environment/node/log/exit evidence만 보존하고 해당 arm을 execution failure로 판정하라. dirty worktree의 unrelated 변경을 지우거나 stage하지 말라. 매 단계마다 현재 단계, 사용한 기존 evidence, 새로 바꾼 causal mechanism, 실패한 gate와 다음 stop/go 판단을 짧게 계속 보고하라.

## 12. Candidate I: telemetry failure delta circuit + surviving-pair service lane

Candidate E/G/H 이후에는 같은 queue reservation이나 proactive scale의 숫자만
바꾸지 않고, 구조적으로 다른 global mechanism을 한 번만 CPU에서 검증했다. Candidate I는
endpoint telemetry의 누적 failure counter delta를 새 request admission보다 먼저 관찰해
pair circuit을 닫고, 한 pair가 격리되면 살아남은 pair capacity의 일부를 low-priority
burst가 즉시 소비하지 못하게 하며 urgent/minimum-service tenant에 남기는 정책이다.

정확한 계약은 다음과 같다.

- `telemetry_failure_quarantine_mode=deny_until_probe`
- `telemetry_failure_quarantine_scope=pair`
- 누적 local/remote failure counter 증가가 보이면 현재 telemetry sequence와
  failure kind을 receipt provenance에 남기고 pair의 두 route를 pre-admission quarantine한다.
- recovery는 더 최신 sequence의 명시적 `PROBE` telemetry가 올 때만 허용한다.
- `survivor_capacity_reserve_fraction=0.25`, `survivor_reserve_bypass_min_weight=2.0`.
  즉 latency/interactive 같은 높은 weight 또는 wait/minimum-service 경계의 tenant는
  reserve를 bypass할 수 있지만 background burst는 살아남은 pair의 보호 용량을 선점하지
  못한다.
- 기존 explicit request failure receipt, one-way route commit, first-response endpoint
  release, EOF decoder release, same-ID retry 금지와 official `LMCacheConnectorV1:UCX`는
  그대로 유지한다.

Candidate I artifact와 immutable identity:

- profile: `results/tempo_go_c5_candidate_i_telemetry_survivor_v1/frozen_global_profile.json`
  SHA `9fd212df642124c28982888cffd4506a1680d8a4ac70ea9944b18663a74ee10c`
- native run contract(아직 native 실행 허가가 아님):
  `results/tempo_go_c5_candidate_i_telemetry_survivor_v1/native_run_contract.json`,
  file SHA `cab8942c74563552642278eb3c0f6aeb1fcbc7a72e3fa1a67461df230d538d5d`,
  fingerprint `ccb424d40e9fbd47060416599ee6f7351a68993c7c500b93e103f65439977826`
- normal CPU replay:
  `.../heldout_cpu_replay_normal_v2.json`, SHA
  `3db5e131fa07fa5da723200852d407b2ea4c71d094908d772f981ccccd36e18a`
- telemetry-failure CPU replay:
  `.../heldout_cpu_replay_telemetry_failure_index800.json`, SHA
  `b95fae11c24ae55a6d6219864cfa3db1efc7bbb86557b4b26c78332a4759db1c`
- control-plane overhead:
  `.../control_plane_overhead_v1.json`, SHA
  `b475e57710230ac77518c57eddfc77e947c773a1b67b2b9db120e1432aceeadf`

Normal held-out replay는 2,712개 동일 trace에서 ALWAYS_LOCAL `1321 complete /
1391 reject`, TEMPO-GO도 정확히 `1321/1391`이고 TEMPO route는 local 1,315,
remote 6이었다. E2E p50/p99도 TEMPO `5321.02/5913.90 ms`로 ALWAYS_LOCAL과
동일했다. 그러므로 정상 성능 개선은 없다.

Telemetry-failure replay는 index 800에서 pair 0의 cumulative remote failure를
`observed_lmcache_engine_failure`로 주입했다. Candidate I는 pair 0의 두 route를
`telemetry_failure_delta` trigger로 격리하고 pair 1만 사용했다. 결과는 TEMPO
`912 complete / 1800 reject / 0 failed`, E2E p50/p99 `5237.69/7549.51 ms`였다.
background는 `673 complete / 1763 reject`, latency `59 complete / 37 reject`,
interactive와 batch는 각각 96/96, 84/84 complete였다. 따라서 우선순위 보호와
pre-admission quarantine은 발동했지만 overload/robustness utility gate를 통과한
것이 아니며, 성능 claim도 금지한다.

별도 CPU microbenchmark(200 warmup + 2,000 sample, synthetic lifecycle only)는
baseline 대비 Candidate I의 control-plane p50/p99가 `170.748/236.081 us`에서
`180.226/264.013 us`로, telemetry refresh p50/p99가 `39.634/59.772 us`에서
`43.692/67.205 us`로 증가했다. 이는 구현 overhead가 측정 가능하지만 작은 편이라는
자료일 뿐 GPU/network/LMCache latency나 native goodput의 대체 측정이 아니다.

따라서 Candidate I의 현재 stop/go는 `CPU correctness=GO`, `CPU overhead=GO`
(별도 기록 완료), `normal performance=FAIL/neutral`, `failure robustness=mechanism
observed but utility gate=FAIL`, `native=STOP`이다. Candidate I를 native에서
맹목 반복하지 않는다. 다음 native 전 필수 작업은 independent validation용으로
동일 contract의 primary/robustness/fairness analyzer가 위 수치를 재현하고, pair
격리 후 survivor reserve가 high-priority SLO-goodput을 실제로 개선하는지, normal
load regression이 3% 이내인지, overhead를 포함해 모든 correctness gate를 통과하는지
검증하는 것이다. 그 전에는 기존 4-node interactive allocation을 점유하거나 새
allocation을 자동 요청하지 않는다.

Candidate I source SHA:

- `tempo/pd_global_orchestrator.py`: `229d7dc085ef6ab93ed7cd1b4277d55df65332591a240fe0580ff0a7a1f9572b`
- `tempo/pd_global_profile.py`: `df6a71f89f3b51d6cbb9f278ed0ddcbe593c28a1bd8f938655b92a93a75867b7`
- `eval/sota_4node/replay_tempo_go_c5_five_arm.py`: `e6c8969fbd0c4d4a2091a809cb7eac92839ae942faab42af93de5cc9ea7ff7d0`
- `eval/sota_4node/measure_tempo_go_admission_overhead.py`: `922f693e7d4549316760bc2cb8e5b1cc21f3dd397a7ea0fcb751807bcfe978fe`
- focused regression: `133 passed, 11 subtests passed in 6.96s`

## 13. Strict completion audit: reproducible pre-native negative

CPU replay가 native 성능 측정이 아니라는 boundary를 유지한 채, canonical의
완료 대안인 “두 구조적으로 다른 global candidate가 같은 preregistered gate를
실패하는 재현 가능한 negative”를 별도 audit로 닫았다.

Audit artifact:

- script: `eval/sota_4node/audit_tempo_go_cpu_negative.py`, SHA
  `5ecfbe04b3f5c02c91c449149acdd36b70843a74ee888336dd582c1a33f59897`
- result: `results/tempo_go_c5_candidate_i_telemetry_survivor_v1/cpu_negative_audit_v1.json`,
  SHA `aed33cc340c27e2688de0dfb001009182f37236b291ebdb25399e4bd78358925`

Audit가 machine-check한 사항:

- Candidate G와 I 모두 동일 held-out manifest
  `6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c`, workload
  `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33`, Elastic
  profile, endpoint profile, baseline global profile을 사용했다.
- fixed four-arm receipt fingerprint가 두 replay에서 동일하고, 2,712 request,
  phase/oracle/physical-switch input 제외, manifest valid, terminal/leak-free,
  현재 source-bound run contract 검증이 모두 true다.
- G는 tenant별 queue reservation 16 slots, I는 reservation 없이 telemetry
  failure pair quarantine과 survivor reserve를 사용하므로 mechanism이 구조적으로
  다르다.
- 동일 primary median gate에서 strongest fixed는 `QUEUE_GPU_ONLY` p50
  `5297.970741 ms`였으므로 TEMPO p50은 `4768.173667 ms` 이하여야 하고,
  predictor p50 `5344.028590 ms` 기준 `5076.827160 ms` 이하여야 했다.
  G는 `6298.891169 ms`, I는 `5321.022663 ms`로 둘 다 실패했다.
- I telemetry-failure replay도 contract-bound로 재현됐고 pair-0 quarantine이
  발동했지만 `912 complete / 1800 reject / 0 failed`, p99 `7549.510599 ms`로
  utility robustness gate를 통과하지 못했다.

이 결과의 정확한 범위는 다음과 같다.

> 동일 held-out duration-based control-plane replay와 frozen service-proxy
> boundary에서 Candidate G와 Candidate I는 strongest fixed/predictor primary
> median gate를 통과하지 못했다. 따라서 이 두 후보는 native independent
> validation으로 승격하지 않는다.

이것은 native 성능이 음성이라고 주장하는 결과가 아니다. `native_performance_negative_proven=false`,
`performance_claim_allowed=false`, `native=STOP`을 명시적으로 보존한다. 즉
현재 연구의 완료는 native win이 아니라, 두 구조적으로 다른 global candidate가
사전 등록 CPU promotion gate에서 재현 가능하게 탈락하여 추가 threshold search와
blind native retry를 중단하는 negative conclusion 경로로 닫힌다. 기존 native
integration/failure receipts는 실제 vLLM/UCX contention과 control-plane wiring의
증거로 유지하지만, 이 audit에 의해 성능 결과로 승격하지 않는다.
