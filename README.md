# TEMPO-GO

## Shared HPC Fabric에서의 Disaggregated LLM Inference를 위한 Cross-Layer Global Orchestration

TEMPO-GO는 local/remote 중 하나를 고르는 단순 router가 아니다. 실제 vLLM
Prefill/Decode(P/D), LMCache KV 이동, decoder scheduler, Cassini/Slingshot 상태,
tenant priority와 fairness, P×D mesh capacity를 하나의 admission·dispatch·release
transaction으로 제어하는 Perlmutter 지향 global orchestrator다.

**현재 상태(2026-08-30):** 실제 contention 문제와 global orchestration의
필요성은 확인됐지만, 현재 policy의 성능 우위는 확인되지 않았다. 새
4-node/16-A100 `gpu_interactive` allocation `57736076`에서 Candidate O를 actual
vLLM P/D, official LMCache/NIXL-UCX, 양쪽 physical pair의 NCCL/Slingshot
co-job으로 7-arm one-campaign 실행했다. O는 remote-favorable 30/30과 p99
3.170초를 냈다. 별도 allocation의 M과 비교하면 foreground HTTP-200 completion은
203→207, background는 1,478→2,004였지만 이는 비인과 context다. O가 바꾼
route-scoped telemetry quarantine은 1,614개 global decision에서 한 번도
발동하지 않았으므로 이 차이를 해당 mechanism의 효과로 귀속할 수 없다. 또한
miss-hot p99 35.578초/SLO 65/120은 strongest
fixed local-d0의 10.846초/81/120보다 낮고, normal p99도 26.365초로 가장 강한
fixed의 10.862초보다 2.43배 길다. observer coverage는 37/210(17.62%)뿐이다.
따라서 Candidate O도 `causal_discovery_positive=false`이며 성능 claim은 닫혀 있다.

이 결과는 “remote가 항상 나쁘다”거나 “interconnect 병목이 없다”는 뜻이 아니다.
같은 carrier에서 fixed route에 따라 miss-hot p99가 9.8–79.0초로 벌어졌고,
official LMCache/NIXL transfer는 correctness 뒤 60초 timeout 경계에 도달했다.
문제는 scalar route score와 O의 현재 policy bundle만으로 decoder admission,
transfer concurrency, pair capacity와 foreground/background utility를 원자적으로
소유할 수 없다는 것이다. 현재 source-of-truth는
[통합 계획 §74.49–§74.58](paper/TEMPO_GO_UNIFIED_GOAL_STATE_AND_EXECUTION_PLAN.ko.md)이다.

[논문 PDF (historical positive draft)](paper/tempo_go/main.pdf) ·
[논문 소스 (current rewrite 필요)](paper/tempo_go/main.tex) ·
[Artifact manifest](paper/tempo_go/artifact_manifest.json) ·
[Current evidence manifest](paper/tempo_go/current_evidence_manifest.json) ·
[전체 연구 상태·실행계획](paper/TEMPO_GO_UNIFIED_GOAL_STATE_AND_EXECUTION_PLAN.ko.md)

> [!IMPORTANT]
> historical C8 positive와 C10 NetKV/Kairos 결과는 보존하지만 current independent
> claim이 아니다. C10은 post-hoc이고 Kairos는 `X={512}` subset이며, 1,024-pair
> 결과는 CPU control-plane receipt다. setup failure, native overload outcome,
> policy negative를 서로 섞지 않는다. Candidate N의 일곱 native arm은 terminal이지만
> 최초 analyzer가 빈 completed population에서 중단되어 top-level
> `completed_attempt.json`은 없다. 수정된 `analysis.json`은 raw를 바꾸지 않은
> post-hoc reanalysis다. Candidate O의 native `analysis.json`과
> `completed_attempt.json`은 그대로 보존한다. 별도 terminal-semantics v2 분석은
> `valid=true`인 HTTP 503 failure receipt를 completion으로 세던 분석 오류만
> raw SHA를 검증해 교정했다. O의 정확한 business 결과는 foreground 207 complete,
> 3 service-lane failure; background 2,004 complete, 40 failure, 704 queue reject다.

## 최근 2주 연구 지도

1,600개가 넘는 historical evaluator/launcher 버전을 한 줄씩 나열하지 않고,
연구 질문과 증거가 바뀐 지점으로 묶는다. 세부 버전·allocation·SHA·버그 수정은
통합 계획에 전부 보존한다.

| 단계 | 핵심 질문 | actual-system 증거 | 결론/다음 변화 |
|---|---|---|---|
| C0–C4 | local/remote prediction만으로 충분한가 | cache/queue crossover는 재현, shared decoder tail은 미해결 | route-only 접근 기각 |
| C5–C7 | LMCache/NCCL/Cassini 관측을 붙이면 충분한가 | signal freshness와 actuation receipt 구현 | telemetry-only 접근 기각, lease 도입 |
| C8 v48 | service lane과 mesh를 공동 제어하면 개선되는가 | 210/210, p99 3.243초 discovery | 강한 positive discovery, independent 아님 |
| C9 v15–v27 | receiver incast/hard guard가 causal한가 | 6–60초 LMCache tail, NCCL 최대 38.7ms, guard actuation | overload 실존; hard guard는 병목 이동 |
| J/K/L | feasibility·protected reserve로 victim을 지킬 수 있는가 | native transport/correctness 통과, SLO/normal gate 실패 | reject/floor만으로 부족 |
| v13 | dual-route business lane을 7-arm으로 비교하면 어떤가 | 210/210 완료, p99 23.917초, SLO 32.9% | route 다양성은 얻었지만 utility 실패 |
| M | pressure-triggered pair spill이 두 regime을 함께 살리는가 | remote 30/30, miss-hot 30/120, actual background success 53.78% | pair spill 단독 기각 |
| N | receiver-tail price가 global frontier를 만드는가 | normal 0/60, miss-hot 0/120, remote 0/30 | scalar price 기각 |
| O | failure quarantine를 pair→route로 줄이면 surviving route가 사는가 | remote 30/30, miss-hot 65/120, background 72.93%; quarantine event 0 | mechanism 비활성, strongest-fixed/observer gate 실패 |

이제 target은 `route = local or remote`가 아니라 다음 joint transaction이다.

```text
atomic observation
  → business admission/fairness
  → pair activation + decoder/source/receiver lease
  → route + transfer-concurrency commit
  → completion/failure release and debt update
```

다음 변경은 O의 threshold를 다시 조절하는 것이 아니다. 모든 decision에 fresh
observer epoch를 붙이고, decoder lane·pair/edge·receiver·fabric token을 하나의
atomic lease로 예약하며, foreground SLO와 background minimum service를 같은
utility frontier에서 판정해야 한다. 그 transaction이 구현된 뒤 같은 offered
population을 capacity-normalized 1/2/4-node rung과 NetKV, full Kairos-compatible,
queue/predictor/fixed baseline에 적용한다.

---

## 초록

Disaggregated LLM serving은 prefill과 decode의 간섭을 분리하지만, 병목을 없애는
것은 아니다. 실제 HPC 환경에서는 decoder compute, local prefill, remote KV 이동,
LMCache inflight, receiver queue, cache ownership, Slingshot congestion과 tenant SLO가
동시에 변한다. 따라서 항상 local, 항상 remote, queue-only, request-level predictor,
network-only selector 같은 부분 정책은 한 상태에서 맞더라도 병목이 이동하면
completion과 tail latency를 함께 잃을 수 있다.

TEMPO-GO는 각 요청에 대해 모든 local/remote P×D 후보를 구성하고, atomic
cross-layer telemetry batch와 controller-owned reservation을 바탕으로 후보의
feasibility와 externality를 평가한다. 선택은 business admission, decoder service
lane, source/edge virtual service, bounded hierarchy와 하나의 lifecycle로 결합된다.
실행이 시작된 뒤에는 경로를 바꾸지 않으며, first response·completion·failure·timeout
각 경계에서 정확히 한 번 lease를 반환한다.

Qwen2.5-7B-Instruct, vLLM 0.26.0, TP4×4 engine, official
LMCacheConnectorV1/NIXL-UCX/CXI를 사용한 Perlmutter 4-node/16-A100 campaign은
두 가지를 확인했다. 첫째, fixed local/remote의 regime별 tail이 수십 초 차이 나고
LMCache transfer가 60초 timeout에 도달하므로 cross-layer contention 문제는
실재한다. 둘째, 현재 구현된 receiver guard, protected reserve, pressure spill과
scalar receiver price는 병목을 공동 제어하지 못했고, O의 route-scope 변경은
native run에서 비활성이어서 효과 자체가 아직 미증명이다.
최신 O는 remote-favorable 30/30과 background 72.93%를 회복했지만 miss-hot에서
strongest fixed보다 p99가 3.28배 길고, normal p99도 2.43배 길었다. foreground
service-lane failure 3건과 background failure/reject 744건, observer coverage
17.62%도 남았다. 과거 C8에서 210/210과 약 3.2초 p99의 positive discovery가
있었지만 fresh independent repeat가 아니므로 현재 성능 claim으로 승격하지 않는다.
이 증거는 다음 설계가 decoder admission/fairness, pair scaling, transfer
concurrency와 business utility를 하나의 global lease로 결합해야 함을 보여준다.

---

## 1. 한눈에 보는 결론

| 구분 | 상태 | 핵심 결과 | 허용되는 주장 |
|---|---|---|---|
| 문제 재현 | **닫힘** | fixed miss-hot p99 9.8–79.0초, NIXL 60초 timeout | actual shared-path overload 실존 |
| C8 v48 | **positive discovery** | 210/210, p99 3.243초 | independent claim 아님 |
| v13 dual-route | **negative** | p99 23.917초, SLO 32.9%, observer 46.7% | route 다양성/actuation만 확인 |
| Candidate M | **negative** | remote 30/30, miss-hot 30/120, actual background success 53.78% | pair spill 단독으로 부족 |
| Candidate N | **negative** | SLO 0/60, 0/120, 0/30, observer 10.48% | receiver-price factor 기각 |
| Candidate O | **negative, current** | remote 30/30, miss-hot 65/120, background 72.93%, observer 17.62%; changed mechanism 0회 | O bundle 결과만 유효, route-scope 인과효과는 미증명 |
| NetKV reproduction | **historical post-hoc** | actual carrier에서 실행 | fresh independent 비교 아님 |
| Kairos `X={512}` | **historical subset** | stock-vLLM 호환 subset | full Kairos 아님 |
| 1/2/4-node inference scale | **OPEN** | topology contract만 5 tests 통과 | native scale claim 없음 |
| 1,024 logical pairs | **CPU receipt** | payload 87.499% 감소 | native GPU inference 아님 |

![Latest Candidate O native matrix](paper/tempo_go/figures/current_candidate_o_native_matrix.svg)

![Candidate M N O business progression](paper/tempo_go/figures/current_candidate_business_progression.svg)

핵심 메시지는 “remote가 항상 나쁘다”도 “LMCache가 문제를 만들지 못한다”도 아니다.
실제 문제는 **어떤 병목이 현재 지배적인지 계속 바뀌는데 부분 정책은 그 이동을 함께
보지 못한다는 것**이다. 지금까지의 positive와 negative를 함께 보면 다음 TEMPO의
가치는 작은 threshold tuning이 아니라 admission, cache semantics,
source/receiver service, fabric state와 business objective를 하나의 global
decision으로 묶고 실제 scale에서 검증하는 데 있다.

---

## 2. historical positive와 current negative를 함께 읽는 법

이전 루트 README는 2026-08-21 C4 route-only 연구에서 멈춰 있었다. 그 결과 자체는
유효하다. C4는 같은 decoder로 합류하는 두 경로 중 local/remote를 request 단위로
선택했으며, route 선택 정확도는 높았지만 shared decoder admission과 tenant fairness,
P×D mesh source balance를 제어하지 못했다. 따라서 median 이득과 TPOT/worst-tail
isolation을 동시에 달성하지 못했다.

TEMPO-GO는 C4 negative나 C8 positive를 삭제하거나 뒤집지 않는다. 각 결과가
답한 질문과 source/allocation/claim gate가 다르다. 이후 held-out 성격의 campaign이
같은 이득을 재현하지 못했으므로 현재 claim은 negative 쪽에 맞춘다.

| 단계 | 질문 | 확인된 사실 | 다음 단계에 반영된 변화 |
|---|---|---|---|
| C1–C3 | local과 remote 중 어느 쪽이 빠른가 | decoder-hot에서는 remote, remote-path-hot에서는 local | 양쪽 path 모두 필요 |
| C4 | request-level predictor와 dual credit이면 충분한가 | 경로 선택은 맞아도 shared decoder tail은 보호 못함 | decoder admission 필요 |
| C5–C6 | fabric/LMCache/NCCL 관측을 붙이면 충분한가 | telemetry만 보고 action이 없으면 utility 개선 없음 | joint actuation과 lease 필요 |
| C7–C8 | business lane과 mesh source balance가 필요한가 | 한 discovery에서 foreground·background·remote activation 동시 개선 | global transaction 후보 확보 |
| C9 current-source | fresh workload에서 재현되는가 | v13/J/K/L/M/N/O의 utility gate 반복 실패 | positive claim 보류, atomic admission frontier 재설계 |
| C10 historical | paper-derived partial policy와 어떤 차이가 있는가 | NetKV reproduction/Kairos subset의 post-hoc 비교 | fresh unchanged repeat와 full baseline 구현 필요 |

즉 C4의 결론은 “TEMPO가 가치 없다”가 아니라 **route-only로 TEMPO를 축소하면
안 된다**는 motivation이고, 최신 negative의 결론은 global이라는 이름만 붙인
scalar policy도 충분하지 않다는 것이다.

---

## 3. 연구 동기: bottleneck은 하나가 아니라 이동한다

### 3.1 local과 remote는 모두 overload될 수 있다

- local path는 KV transfer를 피하지만 decoder-local prefill이 decode와 GPU를 공유한다.
- remote path는 prefill compute를 분리하지만 KV bytes, source service, receiver queue와
  LMCache/CXI path를 소비한다.
- cache hit는 binary label이 아니다. `MISS`, `P_ONLY`, `D_ONLY`, `BOTH`, exact
  source ownership에 따라 가능한 후보와 transfer cost가 달라진다.
- decoder가 가득 차면 빠른 network도 요청을 살리지 못한다.
- receiver가 여유로워도 source·edge·LMCache inflight가 포화되면 remote tail이
  폭증한다.
- foreground만 보호하고 background를 사실상 제거하면 business objective를 달성한
  것이 아니다.

이 결합 때문에 “interconnect가 항상 headline bottleneck이어야 한다”는 가정도
부정확하다. 실제 global controller의 역할은 network를 무조건 쓰거나 피하는 것이
아니라, **decoder·source·fabric·cache·business 병목이 이동할 때 전체 utility가
붕괴하지 않도록 action을 함께 바꾸는 것**이다.

### 3.2 기존 partial policy가 놓치는 축

| 정책 계열 | 보는 것 | 놓치는 대표 상태 |
|---|---|---|
| strongest fixed | 한 edge의 평균 강점 | phase change, 다른 decoder/source의 spare capacity |
| request predictor | 예상 local/remote latency | global reservations, tenant debt, simultaneous arrivals |
| queue-GPU | decoder running/waiting | KV ownership, fabric/self-contention, source fairness |
| NetKV-style network-aware selection | KV transfer + decoder free slot | business reserve, pair scaling, lifecycle debt, joint admission |
| Kairos-style prefill deflection | decoder TBT와 chunk schedule | fabric/cache/source global state, tenant fairness |
| transport-only multipath | packet/path utilization | 어떤 LLM request를 어느 P/D pair에 admit할지 |

각 컴포넌트를 누군가 조금씩 다뤘다는 사실은 global orchestration의 가치를 없애지
않는다. 실제 Perlmutter-scale system에서는 이 신호와 actuator를 **같은 request
lifecycle과 같은 objective 아래 묶어 구현하고 검증하는 것**이 핵심이다.

### 3.3 연구 가설

요청 `r`의 후보 `e=(p,d,ρ)`는 prefill source `p`, decoder `d`, route
`ρ∈{local,remote}`를 나타낸다. TEMPO는 다음 개념적 objective를 사용한다.

```text
e* = argmin feasible(e, state)
       predicted_service(e)
     + fabric_externality(e)
     + decoder/source_service_debt(e)
     + business_debt(r, e)
```

중요한 것은 score 하나가 아니다. `feasible`이 cache identity, capacity, freshness,
deadline, health, admission과 lease를 먼저 강제하며, unsupported telemetry를 0으로
바꾸지 않는다.

---

## 4. TEMPO-GO 설계

```mermaid
flowchart TB
    subgraph Signal[Cross-layer state plane]
        V[vLLM scheduler<br/>running · waiting · KV occupancy]
        L[LMCache/NIXL<br/>semantic ops · KV bytes inflight]
        C[Cassini/Slingshot<br/>pause · ECN · packet rate · OXE activity]
        B[Business state<br/>tenant priority · SLO · debt · fairness]
        H[Health/lifecycle<br/>epoch · failure · quarantine · lease]
    end

    V --> T[Atomic telemetry batch]
    L --> T
    C --> T
    B --> T
    H --> T

    subgraph Global[TEMPO global control transaction]
        T --> G[Node → pair → shard → global reducer]
        G --> K[P×D local/remote candidate builder]
        K --> F{Feasibility + externality + debt}
        F -->|admit| R[Atomic reserve/commit]
        F -->|bounded wait| Q[Business-aware queue lease]
        F -->|infeasible| X[Explicit reject receipt]
        Q --> G
    end

    R --> DL[Decoder-local chunked prefill]
    R --> LR[Official LMCache remote prefill]
    R --> SL[Priority service lane / dispatch stagger]
    DL --> E[First response · completion · failure]
    LR --> E
    SL --> E
    E --> P[Exactly-once release + causal receipt]
    P --> T
```

### 4.1 Atomic state plane

한 decision은 서로 다른 시각에 수집된 scalar를 임의로 섞지 않는다. frontend,
각 endpoint, LMCache와 Cassini 신호를 complete batch로 묶고 epoch/freshness를
검증한다. fetch가 실패하면 last complete batch와 tenant별 stale grace를 사용하거나
fail closed한다.

### 4.2 P×D candidate와 자원 벡터

후보는 “pair index 하나”가 아니라 source, destination, route를 분리한다. 각 후보는
다음 자원을 가진다.

```text
(decode tokens, active sequences, endpoint requests,
 local prefill token-ms, remote prefill token-ms,
 remote KV bytes, remote semantic operations)
```

따라서 `remote:p0→d1`의 source credit과 D1 receiver credit을 각각 charge할 수 있고,
cache owner와 destination을 혼동하지 않는다.

### 4.3 Business-aware admission과 fairness

interactive/latency tenant는 priority service lane과 reservation을 받는다. background는
무조건 drop하지 않고 decoder별 bounded concurrency와 starvation escape를 가진다.
controller는 foreground SLO와 함께 background completion, block/tenant minimum,
Jain fairness와 service-lane failure를 gate로 검사한다.

### 4.4 Source/edge virtual service

동일 decoder·cache affinity·priority 상태에서 telemetry uncertainty 안의 near-tie
remote 후보만 source-balance 대상으로 인정한다. static score anchor를 유지하면서
source-prefill과 `P_i→D_j` edge virtual finish가 작은 후보를 선택한다. quota를 맞추기
위해 더 나쁜 route를 강제로 선택하지 않는다.

### 4.5 Exactly-once lifecycle

```text
global decision
  → endpoint admission/service-lane receipt
  → immutable upstream route
  → first response / completion / failure / timeout
  → physical credit와 business lease의 exactly-once release
```

부분 실행 뒤 다른 경로로 fallback하지 않으며, duplicate release·underflow·orphan
reservation은 unit/integration gate에서 실패한다.

### 4.6 Hierarchical fan-in

모든 pair의 모든 후보를 한 Python/global process로 보내지 않는다. pair agent가 local
frontier와 omission receipt를 만들고, bounded shard reducer가 global frontier만 전달한다.
이 구조는 4-node actual path에서도 같은 reducer를 사용하며, logical scale에서 payload와
fan-in bound를 검증한다.

---

## 5. 구현 범위

| 계층 | canonical 구현 | 역할 |
|---|---|---|
| global state | `tempo/pd_global_telemetry.py`, `pd_global_agent.py` | atomic batch, support/freshness/epoch |
| candidate model | `tempo/pd_global_candidates.py` | cache-aware P×D local/remote candidate |
| orchestrator | `tempo/pd_global_orchestrator.py` | feasibility, score, reservations, business debt |
| transaction | `tempo/pd_global_coordinator.py` | prepare/commit/release/failure lifecycle |
| hierarchy | `tempo/pd_global_hierarchy.py` | pair→shard→global bounded reduction |
| profile | `tempo/pd_global_profile.py` | immutable capacities, tenants, gates, fingerprints |
| fabric observation | `tempo/cassini_endpoint.py`, `cross_layer_observer.py` | application-visible Cassini/LMCache/NCCL schema |
| actual ingress | `eval/sota_4node/tempo_pd_elastic_frontend.py` | business admission과 global commit |
| actual P/D router | `eval/sota_4node/tempo_pd_elastic_router.py` | vLLM/LMCache route execution과 receipts |
| LMCache integration | `eval/sota_4node/lmcache_tempo_current.patch` | pinned upstream 위 proxy/PD/NIXL hotpath 변경 |
| paper baselines | `tempo/pd_paper_baselines.py` | NetKV/Kairos-X512 reproduction |

이 구현은 root 권한, UDI/container, switch configuration, `CAP_NET_ADMIN`, system file
변경을 사용하지 않는다. application/user scope에서 자신의 allocation과 endpoint만
관측·제어한다.

---

## 6. 평가 방법

### 6.1 시스템

| 항목 | 설정 |
|---|---|
| machine | NERSC Perlmutter GPU partition |
| allocation | fresh 4 nodes / 16 NVIDIA A100 GPUs / 4 hours |
| topology | P0/D0/P1/D1, four actual TP4 engines |
| model | Qwen2.5-7B-Instruct |
| serving | vLLM 0.26.0 |
| KV path | official LMCacheConnectorV1, NIXL-UCX/CXI |
| network | HPE Slingshot 11, four Cassini NICs per GPU node |
| execution | `gpu_interactive` no-shell allocation + foreground `srun` only |

Perlmutter GPU node는 네 개의 Slingshot 11 NIC를 가지며 NIC당 25 GB/s injection
bandwidth를 제공한다. TEMPO는 이 hardware fact를 capacity prior로 사용하되, 지원되지
않는 switch/global counter를 만들어내지 않는다.

### 6.2 workload regimes

| regime | offered foreground | cache/pressure | 목적 |
|---|---:|---|---|
| normal | 60 | control A/B, MISS | 정상 구간 regression 확인 |
| miss-hot | 120 | MISS + decoder/local/remote co-load | bottleneck migration과 tail isolation |
| remote-favorable | 30 | exact P_ONLY + dual decoder-local pressure | 실제 cross-pair LMCache activation |

각 arm은 같은 request seed, arrival jitter, prompt/output geometry, background population,
block population을 사용한다. C9 validation arm/block order는 discovery의 역순이며,
fresh allocation에서 one-shot으로 실행됐다. controller는 workload phase label, future
arrival, seed를 입력받지 않는다.

### 6.3 비교 정책

1. fixed local D0
2. fixed local D1
3. fixed remote P0→D1
4. fixed remote P1→D0
5. request predictor
6. queue-GPU
7. TEMPO full cross-layer global control

C10은 같은 carrier와 held-out population 위에서 NetKV reproduction과 Kairos
`X={512}` subset을 추가한다. baseline frontend의 compatibility receipt는
`wait_ns=0`, `policy_effect=none`, `evidence_only_no_throttle`인지 analyzer가 전부
검증한다. 따라서 baseline이 TEMPO admission을 몰래 사용하지 않는다.

### 6.4 성공 gate

- foreground offered-population SLO와 E2E p50/p99
- strongest fixed, predictor, queue-only 대비 성능
- background completion/fairness/service-lane failure
- actual local/remote/cross-edge activation
- complete telemetry batch, supported/unsupported signal classification
- collection/admission overhead
- source-frozen contract와 raw/result SHA
- fresh allocation, one-shot execution, no hidden retry

---

## 7. Historical C9 positive artifact (현재 claim 아님)

이 절의 표와 그래프는 mechanism discovery와 artifact pipeline을 보존하기 위한
historical 결과다. 최신 v13/M/N/O에서 독립 재현되지 않았으므로 현재 headline이나
성능 우위로 사용하지 않는다.

![C9 independent seven-arm performance](paper/tempo_go/figures/c9_independent_performance.svg)

### 7.1 전체 7-arm matrix

| Policy | normal SLO / p99 | miss-hot SLO / p99 | remote-favorable SLO / p99 |
|---|---:|---:|---:|
| fixed local D0 | 60/60 / 3.232s | 97/120 / 10.051s | 0/30 / 35.944s |
| fixed local D1 | 60/60 / 3.156s | 97/120 / 9.849s | 0/30 / 39.559s |
| fixed remote P0→D1 | 60/60 / 3.530s | 87/120 / 22.128s | 0/30 / 36.148s |
| fixed remote P1→D0 | 60/60 / 3.258s | 38/120 / 24.030s | 0/30 / 35.508s |
| predictor | 60/60 / 3.104s | 100/120 / 8.709s | 13/30 / 50.557s |
| queue-GPU | 60/60 / 3.088s | 116/120 / 8.036s | 13/30 / 51.973s |
| **TEMPO** | **60/60 / 3.150s** | **120/120 / 3.337s** | **30/30 / 3.357s** |

normal에서는 queue/predictor가 TEMPO보다 p99 기준 약 46–62 ms 빠르다. TEMPO가 모든
구간에서 무조건 가장 빠르다고 주장하지 않는다. 중요한 결과는 정상 구간을 약 3.15초로
유지하면서 두 stressed regime의 completion과 tail collapse를 동시에 제거했다는 것이다.

### 7.2 p99 감소율

| 비교 대상 | miss-hot | remote-favorable |
|---|---:|---:|
| strongest fixed | **66.12%** | **90.55%** |
| predictor | **61.69%** | **93.36%** |
| queue-GPU | **58.48%** | **93.54%** |

이 결과가 기존 1–5% 수준의 차이와 달라진 이유는 workload 숫자를 과장했기 때문이
아니다. 이전 workload는 한 경로만 독립적으로 자극하거나 동일 decoder bottleneck을
모든 정책에 공통으로 남겨 orchestration의 action space를 닫았다. C9은 실제
decoder-local pressure, remote source/fabric pressure, P_ONLY ownership과 background
business traffic을 조합해 **병목 이동 자체**를 측정하고, TEMPO에는 그 상태를 바꿀
실제 admission/dispatch actuator를 제공했다.

---

## 8. Historical C9 경로 actuation

![TEMPO mesh actuation](paper/tempo_go/figures/c9_mesh_actuation.svg)

| edge | completed foreground |
|---|---:|
| local D0 | 35 |
| local D1 | 146 |
| remote P0→D0 | 9 |
| remote P0→D1 | 6 |
| remote P1→D0 | 8 |
| remote P1→D1 | 6 |

전체 210개 foreground 중 local 181개, official LMCache remote 29개가 완료됐다.
remote-favorable에서는 30개 중 29개를 remote로 보냈고 exact official LMCache full
source hit, decoder business admission, priority lane과 source-balance receipt를
검증했다. 반대로 miss-hot에서는 decoder/source/fabric externality를 보고 local D1을
사용했다. 이것이 static local/remote policy와 TEMPO의 차이다.

---

## 9. Historical C9 fairness와 telemetry overhead

![C9 fairness and telemetry](paper/tempo_go/figures/c9_fairness_telemetry.svg)

### 9.1 background utility

| 지표 | 관측값 | preregistered gate | 결과 |
|---|---:|---:|---|
| C7 background completion | 1,204/1,404 = **85.755%** | ≥80% | pass |
| minimum block/tenant completion | **76.496%** | ≥70% | pass |
| tenant Jain fairness | **0.997873** | ≥0.99 | pass |
| service-lane failure | 13/1,404 = **0.926%** | ≤1% | pass |
| C8 local background completion | 1,344/1,344 = **100%** | ≥99% | pass |

foreground SLO를 높이기 위해 background를 측정에서 제거하지 않았다. 각 block의
terminal counts에는 complete, global reject, service-lane failure가 모두 남는다.

### 9.2 controller overhead

| 경로 | p50 | p99 | gate |
|---|---:|---:|---:|
| telemetry collection | 28.62 ms | 132.42 ms | 50 / 250 ms |
| admission wait | 29.46 ms | 133.26 ms | 50 / 250 ms |

remote-favorable decision 30건 모두 complete batch로 분류됐다. Cassini required signal은
29/30 decision에서 supported였고 LMCache semantic/byte inflight는 30/30 supported였다.
다음 세 신호는 explicit `unsupported`이며 0 pressure로 꾸미지 않았다.

- `nccl_collective_p99_ms`
- `nccl_arrival_spread_ms`
- `lmcache_transfer_p99_ms`

따라서 headline gain을 “NCCL latency signal이 직접 유도했다”고 주장하지 않는다.
현재 증거는 vLLM, LMCache inflight, Cassini endpoint state, topology, business admission과
service ledger를 결합한 whole-system 결과다.

---

## 10. Historical C10 NetKV/Kairos post-hoc comparison

![C10 paper policy comparison](paper/tempo_go/figures/c10_paper_policy_comparison.svg)

| Policy | normal SLO / p99 | miss-hot SLO / p99 | remote-favorable SLO / p99 |
|---|---:|---:|---:|
| **TEMPO C9** | **60/60 / 3.150s** | **120/120 / 3.337s** | **30/30 / 3.357s** |
| Kairos `X={512}` | 30/60 / 3.223s | 0/120 / undefined | 1/30 / 4.853s |
| NetKV reproduction | 60/60 / 3.300s | 73/120 / 13.780s | 0/30 / 53.914s |

### 10.1 NetKV reproduction

NetKV-style policy는 remote candidate, estimated KV bytes, decoder queue/first-step,
Perlmutter NIC당 25 GB/s prior, Cassini congestion과 LMCache self-inflight를 사용한다.
정상 구간에서는 TEMPO와 가까우나 miss-hot에서 3 reject와 44 completed-but-SLO-miss가
발생하고, remote-favorable에서는 29 reject와 1개의 53.914초 completion을 남긴다.

TEMPO p99 감소율:

- normal: 4.54%
- miss-hot: **75.79%**, SLO-good 73→120
- remote-favorable: **93.77%**, SLO-good 0→30

해석은 network-aware selection이 틀렸다는 것이 아니다. 실제 overload에서는 network
cost와 decoder free slot만으로 business admission, receiver running set, source/edge
service debt와 pair-level joint actuation을 대체할 수 없다는 것이다.

### 10.2 Kairos `X={512}` subset

Kairos paper의 `α=1.3`, TBT safety `0.9`를 사용했지만 stock vLLM 0.26.0은
per-request dynamic chunk candidate schedule을 제공하지 않는다. 따라서 실제 decoder
`max_num_batched_tokens=512`, 즉 `X={512}` 하나만 사용한 제한적 reproduction이다.

- normal: TEMPO p99 2.28% 감소, SLO-good 30→60
- miss-hot: Kairos complete 0이므로 존재하지 않는 latency quantile의 개선율을 만들지
  않고 completion/SLO 0→120으로 보고
- remote-favorable: p99 30.82% 감소, SLO-good 1→30

### 10.3 C10 claim boundary

C10 adapter는 parent allocation 시작 뒤 동결됐고 NetKV evidence validator fix도 같은
allocation에서 이루어졌다. 성공한 Kairos result는 SHA로 고정해 재사용했고 NetKV만
workload/score 변경 없이 validator-corrected source에서 다시 실행했다. 따라서 실제
vLLM/LMCache 성능 측정이지만 `independent_validation_claim_allowed=false`다. 다음
논문용 hard gate는 source를 더 바꾸지 않은 fresh allocation에서 TEMPO, NetKV,
Kairos subset을 다시 실행하는 것이다.

---

## 11. Historical 계층형 control-plane scale

![Hierarchy control-plane scale](paper/tempo_go/figures/hierarchy_control_plane_scale.svg)

| logical pairs | raw candidates | global forwarded | full payload | bounded payload | payload 감소 | bounded total p50/p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 4 | 1,296 B | 1,296 B | 0% | 1.80 / 2.05 ms |
| 8 | 16 | 16 | 5,181 B | 5,181 B | 0% | 2.37 / 2.39 ms |
| 32 | 64 | 64 | 20,760 B | 20,760 B | 0% | 4.53 / 4.62 ms |
| 128 | 256 | 256 | 83,153 B | 83,153 B | 0% | 13.14 / 13.26 ms |
| 512 | 1,024 | 256 | 333,274 B | 83,322 B | 74.999% | 43.96 / 47.83 ms |
| 1,024 | 2,048 | 256 | 666,815 B | 83,358 B | **87.499%** | 85.33 / 158.24 ms |

1,024 pair에서 896개의 pair omission receipt를 남기고 global frontier를 256개로
제한했다. 다만 현재 구현의 Python single-process bounded total p50 85.33 ms는 full scan
p50 49.74 ms보다 빠르지 않다. 현재 결론은 **payload와 global fan-in이 bounded**라는
것이지 production-scale wall-clock superiority가 아니다. 4노드보다 큰 native
allocation에서 distributed agent wall-clock, wire bytes, failure convergence와 inference
utility를 함께 측정해야 한다.

---

## 12. 실패를 숨기지 않는 실행 discipline

### 12.1 C9 preflight failures

두 preflight attempt는 performance result가 아니며 별도 failure receipt로 남아 있다.
최종 C9은 새 no-shell allocation에서 source-frozen one-shot으로 실행됐다.

### 12.2 C10 v1/v2

- v1 Kairos measured workload는 decoder admission evidence receipt가 없어 frozen
  validator가 중단했다. 성능 실패로 해석하지 않았다.
- v2 NetKV는 유효한 `remote:p0→d1` request를 실행했으나 legacy validator가 frontend
  destination을 prefill source로 오인했다.
- v3은 commit의 `prefill_index`, `decoder_index`, `edge_id`를 분리 검증했다.
  workload와 policy score는 바꾸지 않았다.

### 12.3 테스트 process boundary

C10 baseline frontend는 전용 process에서 canonical frontend class를 baseline/no-op
class로 binding한다. C8과 C10 test를 한 pytest collection에 넣으면 import-time
process binding이 C8 test에 섞인다. 실제 native arm은 서로 다른 process이므로 최종
regression도 그 경계를 따라 분리했다.

- C9/current: 294 passed, 2 historical source-drift checks deselected, 28 subtests passed
- C10 process-entry: 6 passed
- telemetry/endpoint closure: 93 passed
- bounded import closure: 91 Python files `py_compile` passed
- canonical launchers: `bash -n` passed

두 deselect는 현재 C9 source가 과거 C6 frozen source hash와 같아야 한다는 historical
assertion이다. 과거 contract를 새 hash로 덮는 것은 fix가 아니라 provenance 훼손이므로
실행하지 않았다.

---

## 13. Current native status (2026-08-30)

M/N은 allocation `57732862`, O는 새 allocation `57736076`에서 실행됐다. 세
campaign은 모두 210-victim/2,748-background offered population과 actual
vLLM/LMCache/NCCL carrier를 사용했다. O의 r3는 canonical outer
`--gpus=0 --gres=none --network=no_vni`와 실제 GPU/VNI child step을 분리해 일곱
arm의 `block_execution_receipt`, `analysis.json`, `completed_attempt.json`을 모두
생성했다.

| candidate | normal SLO / p99 | miss-hot SLO / p99 | remote-favorable SLO / p99 | observer |
|---|---:|---:|---:|---:|
| M pressure spill | 30/60 / 10.246s | 30/120 / 31.785s | 30/30 / 3.195s | 95/210 |
| N receiver price | 0/60 / 31.344s | 0/120 / 18.187s | 0/30 / undefined | 22/210 |
| **O route liveness** | **35/60 / 26.365s** | **65/120 / 35.578s** | **30/30 / 3.170s** | **37/210** |

O campaign은 exact terminal 기준 foreground completion 207/210, background
2,004/2,748, background queue reject 704, service-lane failure 40을 기록했다.
별도 allocation의 M(203/210, 1,478/2,748, reject 1,197, failure 73)보다 좋아
보이지만 paired causal comparison은 아니다. 더 중요하게, 모든 raw decision의
`telemetry_provenance.route_failures`가 0이었고 `route_failure_quarantine` reject도
0건이라 O의 유일한 변경 mechanism은 비활성이었다. 따라서 이 차이는 co-job
pressure/observer 생존의 run-to-run 변화와 jointly active control을 포함한 O bundle의
관측값일 뿐 route-scope 효과가 아니다. O의 miss-hot p99는
strongest fixed local-d0보다 228.0% 길고 SLO는 65/120 대 81/120이다. normal p99도
best fixed보다 142.7% 길며, observer-supported decision은 M보다 58건 줄었다.

이번에 분석 의미 버그도 닫았다. stream client의 `valid=true`는 “terminal receipt가
router ledger와 일치한다”는 뜻이며, HTTP 503 service-lane failure도 valid할 수
있다. 기존 C9 business 집계는 이를 completion으로 오해했다. terminal-semantics
v2는 completion을 `HTTP 200 ∧ done_seen ∧ exact output tokens`로 정의하고, 원본
native `analysis.json`과 raw를 수정하지 않은 새 post-hoc artifact를 만든다.
`correctness=true`는 계속 receipt/sidecar integrity를 뜻하며, business success는
foreground 207/210과 background 2,004/2,748로 별도 판정한다.

이제 J/K/L/M/N/O의 숫자를 다시 조절하지 않는다. 다음 구현 단위는
`fresh atomic observation → business admission/fairness → pair/decoder/receiver/
fabric lease → route+concurrency commit → completion/failure release/debt` 전체다.
observer coverage와 terminal success, background progress를 latency gate와 함께
통과하기 전에는 새 performance claim을 열지 않는다.

## 14. Claim boundary

### 현재 주장할 수 있는 것

> Fresh 4-node Perlmutter allocation의 actual vLLM/official LMCache/NIXL-CXI
> workload에서 cross-layer receiver overload와 service-path contention을
> 재현했고, fixed/predictor/queue 정책의 regime별 failure와 함께 현재 TEMPO의
> receiver guard, reserve, spill과 scalar price가 overload를 제거하지 못하고 다른
> 경로로 이동시키는 failure mode를 관측했다. 최신 O bundle은 높은 completion을
> 기록했지만 strongest-fixed tail과 observer/business gate를 동시에 통과하지
> 못했고, route-scope 변경은 발동하지 않아 인과효과를 주장할 수 없다.

따라서 현재까지는 global orchestration의 필요성과 측정 가능한 failure boundary는
지지되지만, matched fixed/predictor 대비 성능 우위는 주장하지 않는다.  아래의
과거 positive 문구와 NetKV/Kairos subset 비교는 historical discovery 범위로만
보존한다.

actual carrier에서 NetKV reproduction과 Kairos `X={512}` subset보다 service
dominance를 보인 과거 post-hoc evidence는 별도 artifact로 보존하지만 current
independent comparison으로 사용하지 않는다.

### 아직 주장하지 않는 것

- full Kairos 저자 구현보다 독립적으로 우수하다는 주장
- C10의 independent SOTA claim
- 1,024-pair native GPU inference/goodput superiority
- unsupported NCCL/LMCache latency signal이 headline gain의 직접 원인이라는 주장
- Slingshot switch-level global congestion을 완전 관측한다는 주장
- 다른 모델, context, topology, cache tier로의 보편적 일반화
- facility-wide scheduler 또는 다른 사용자의 job을 제어한다는 주장
- production readiness
- current M/N/O가 strongest fixed 또는 predictor보다 빠르다는 주장
- 1/2/4-node capacity-normalized native scaling이 끝났다는 주장

---

## 15. 재현과 artifact 검증

### 15.1 README 그래프 재생성

그래프는 수치를 소스에 복사하지 않고 committed result JSON에서 생성한다.

```bash
.vllm_venv/bin/python paper/tempo_go/render_readme_figures.py
jq . paper/tempo_go/figures/manifest.json
```

figure manifest는 현재 17개 source JSON과 7개 SVG의 SHA-256을 고정한다.

### 15.2 논문 빌드

```bash
cd paper/tempo_go
module load texlive/2024
pdflatex -halt-on-error -interaction=nonstopmode main.tex
bibtex main
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
```

현재 PDF는 7 pages, SHA
`3e35c65a92230ceef4576bff1ac5aa7ed42a33b82d76ab57a2d5cb2e3877f60f`이며
LaTeX error, unresolved citation/reference, overfull box가 모두 0이다.

### 15.3 current authoritative artifacts

| artifact | path | SHA-256 |
|---|---|---|
| M population contract | `results/tempo_go_c9_candidate_m_pressure_spill_v1/tempo_go_c9_candidate_m_pressure_spill_population_contract.json` | `ef876c73...26721c` |
| M native analysis | `results/tempo_go_c9_causal_burst_job_57732862/analysis.json` | `18fdad31...e5623` |
| N population contract | `results/tempo_go_c9_candidate_n_global_frontier_v2/tempo_go_c9_candidate_n_global_frontier_population_contract.json` | `28d8bd6f...6b637` |
| N native reanalysis | `results/tempo_go_c9_global_frontier_job_57732862/analysis.json` | `dd58c228...cb8d1` |
| O population contract | `results/tempo_go_c9_candidate_o_route_liveness_v1/tempo_go_c9_candidate_o_route_liveness_population_contract.json` | `9936a4a9...cc5` |
| O native analysis | `results/tempo_go_c9_route_liveness_job_57736076_r3_canonical_outer/analysis.json` | `1d5f9c5a...b9a5` |
| M fail-closed business analysis | `results/tempo_go_c9_causal_burst_job_57732862/analysis_failclosed_business_v3.json` | `d7b95ed1...d5fe` |
| N fail-closed business analysis | `results/tempo_go_c9_global_frontier_job_57732862/analysis_failclosed_business_v3.json` | `577fb39f...57af` |
| O fail-closed business analysis | `.../analysis_failclosed_business_v2.json` | `850c2858...3590` |
| O post-hoc receipt | `.../posthoc_business_reanalysis_receipt.json` | `f63c5869...c607` |
| O mechanism diagnosis | `.../candidate_o_diagnosis.json` | `1c8aa9c7...33bd` |
| O completion receipt | `.../completed_attempt.json` | `a56da801...a9` |
| C9 analyzer | `eval/sota_4node/analyze_tempo_go_c9_causal_burst_discovery.py` | current manifest에 full SHA 고정 |

M과 O는 terminal native campaign receipt가 있고 N은 일곱 arm terminal artifact의
post-hoc reanalysis다. N에 `completed_attempt.json`이 없다는 경계를 숨기지 않는다.
O의 fail-closed business 분석은 원본 native receipt를 대체하지 않고 business
completion 의미만 교정한다. diagnosis는 route-scope mechanism이 0회 발동했음을
raw SHA를 다시 검증해 고정한다.

### 15.4 historical paper artifacts

| artifact | path | SHA-256 |
|---|---|---|
| C9 contract | `eval/sota_4node/tempo_go_c8_independent_validation_contract_v3.json` | `e2d07e8c...07a76` |
| C9 analysis | `results/tempo_go_c8_independent_validation_job_57586612_v3/analysis.json` | `844d7b31...19c47` |
| C9 TEMPO result | `.../full_c7_managed_background/result.json` | `0e206931...de64` |
| C10 analysis contract | `eval/sota_4node/tempo_go_c10_paper_sota_analysis_contract_v4.json` | `60d4958f...0525` |
| C10 analysis | `results/tempo_go_c10_paper_sota_job_57586612_v3/analysis.json` | `bdf8604b...5d0f` |
| Kairos-X512 result | `results/tempo_go_c10_paper_sota_job_57586612_v2/kairos_x512/result.json` | `49c5f7ec...c3c8` |
| NetKV result | `results/tempo_go_c10_paper_sota_job_57586612_v3/netkv/result.json` | `2cbeffd8...0759` |
| scale contract | `eval/sota_4node/tempo_go_hierarchy_scale_contract_20260825.json` | `7f7753a4...72a2` |
| scale result | `results/tempo_go_hierarchy_scale_20260825_c9_c10_r15.json` | `90f4e2ab...7088` |

전체 SHA와 source inventory는
[`paper/tempo_go/artifact_manifest.json`](paper/tempo_go/artifact_manifest.json)과
[`paper/tempo_go/figures/manifest.json`](paper/tempo_go/figures/manifest.json)에서
기계 판독 가능하다. 대용량 raw/log tree는 Perlmutter에 보존되며 compact analysis가
각 raw path와 SHA를 고정한다.

### 15.5 SC AD/AE contribution mapping

SC artifact review가 요구하는 contribution과 supporting object의 관계를 다음처럼
고정한다. 구조는 [SC26 Technical Papers](https://sc26.supercomputing.org/program/papers/)
및 [SC26 AD/AE 지침](https://sc26.supercomputing.org/program/papers/reproducibility-appendices-badges/)의
mandatory artifact description, contribution↔artifact mapping, setup/execution/analysis
분리 원칙을 따른다.

| contribution | supporting artifact | 재현되는 결과 |
|---|---|---|
| C1: realistic cross-layer overload | native co-job receipts + M/N/O analysis | NIXL timeout, fixed-policy crossover, terminal correctness |
| C2: identity-bound global mechanism | `tempo/` orchestrator/telemetry/profile + focused tests | admission/lease/actuation invariant |
| C3: current policy failure boundary | v13/M/N/O contracts and analyses | offered SLO, p50/p99, background reject, observer coverage |
| C4: reproducible claim discipline | analyzer, SHA-bound contracts, README/manifest | null population, setup/overload/policy-negative 분리 |

workflow는 `setup → contract freeze → native execution → terminal receipt → analysis
→ table/figure`다. setup과 CPU tests는 일반 login environment에서 가능하지만,
native execution은 Perlmutter 4-node/16-A100와 Slingshot/LMCache/NIXL이 필요하다.
일곱 arm은 약 150분 이상이고, analyzer와 README table verification은 수분 이내다.

### 15.6 Perlmutter 실행 원칙

실제 vLLM/LMCache/GPU/traffic workload는 승인된 4-node `gpu_interactive`
allocation 안에서만 실행한다. 지원되는 no-shell 패턴은 다음과 같다.

```bash
salloc --no-shell --nodes=4 --qos=interactive --time=04:00:00 \
  --constraint=gpu --gpus=16 --account=<approved-account>
```

이후 exact job receipt를 확인하고 foreground `srun --jobid=<jobid>`로 attach한다.
README 명령은 job 제출 승인을 대신하지 않는다. 자동 submit/cancel/retry, background
watcher, login-node GPU workload는 금지한다.

---

## 16. 기존 연구와의 관계

- [vLLM](https://arxiv.org/abs/2309.06180)은 PagedAttention과 serving engine의
  기반을 제공한다.
- [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)는
  prefill/decode 분리와 goodput 최적화를 제시한다.
- [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin)는
  KVCache-centric disaggregation과 storage/compute tradeoff를 다룬다.
- [LMCache](https://arxiv.org/abs/2510.09665)는 KV cache 계층과 실제 remote reuse
  data path를 제공한다.
- [NetKV](https://arxiv.org/abs/2606.03910)는 network-aware decode instance
  selection을 formalize한다.
- [Kairos](https://arxiv.org/abs/2607.02043)는 decoder TBT를 고려한 load-aware
  prefill deflection을 제시한다.
- [MRC](https://arxiv.org/abs/2606.18170)는 multipath transport에서 path utilization을
  개선한다.
- [Perlmutter architecture](https://docs.nersc.gov/systems/perlmutter/architecture/)는
  실제 A100/Cassini/Slingshot deployment boundary를 정의한다.

### 16.1 비교대상 구현·평가 matrix

| 비교대상 | 원 논문의 주 제어축 | 현재 저장소 상태 | TEMPO와 공정하게 비교할 항목 |
|---|---|---|---|
| strongest fixed 4개 | 고정 P/D·local/remote edge | actual native | regime별 SLO/p99와 route crossover |
| predictor | request-level local/remote latency | actual native | same offered population, global state 없음 |
| queue-GPU | decoder running/waiting | actual native | queue-only 대비 cache/fabric/business 효과 |
| [NetKV](https://arxiv.org/html/2606.03910) | cache locality + decode queue + network cost oracle | Algorithm-1-style actual-carrier reproduction; 저자 simulator 전체는 아님 | TTFT/SLO, oracle freshness, prefix-sharing, topology/load sweep |
| [Kairos](https://arxiv.org/html/2607.02043) | decode TBT-safe dynamic chunk schedule로 prefill deflection | `X={512}` subset만 있음 | full chunk-sequence sweep, TTFT/TBT/output goodput |
| [MRC](https://arxiv.org/html/2606.18170v1) | Ethernet/RoCE per-packet multipath, bounded inflight, recovery | Perlmutter Slingshot transport에 직접 구현되지 않음 | transport mechanism은 component ablation; global request utility와 분리 |
| [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin) | KVCache-centric disaggregation과 production trace | official trace/workload intake + LMCache carrier; Mooncake engine 전체 아님 | cache hit/miss, long context, burst와 storage/compute tradeoff |
| [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | P/D goodput와 pool sizing | topology/workload reference | capacity-normalized goodput와 SLO, initial placement 대비 per-request control |

NetKV는 64-GPU simulator와 Mooncake trace에서 network oracle을 평가했고, Kairos는
2P2D A100에서 per-request dynamic chunk schedule을 구현했다. 따라서 현재의
NetKV reproduction과 Kairos `X={512}` 결과를 각각 저자 시스템 전체와 같다고
표기하지 않는다. MRC는 best-effort Ethernet/RoCE transport이므로 Slingshot에서
그 수치를 그대로 재사용하지 않고, multipath/bounded-inflight/recovery 개념의
component ablation으로만 연결한다.

TEMPO의 초점은 이 컴포넌트를 차감해 남는 작은 routing heuristic이 아니다. 실제
HPC deployment에서 application, cache, communication, topology와 business state를
하나의 global lifecycle로 연결하고, partial-policy failure와 전체 utility를 같은
carrier에서 검증하는 것이다.

---

## 17. 남은 연구 gate

Preregistered Candidate P는 O policy를 바꾸지 않고 bounded-resident dual-pair
co-load로 observer lifetime confound만 분리하는 진단이다. 아직 native result가
없으며, realistic overload나 아래 durable state-plane/joint-policy gate를 대체하지
않는다.

1. **Observer closure:** 모든 global decision에서 vLLM scheduler, LMCache/NIXL,
   NCCL/Cassini epoch와 support/freshness를 atomic receipt로 남긴다.
2. **Joint policy:** decoder admission/fairness, pair scaling, receiver/source lease,
   transfer concurrency와 business debt를 하나의 commit/release transaction으로
   구현한다.
3. **Fresh four-node validation:** 동일 210-victim/2,748-background population에서
   strongest fixed, predictor, queue-GPU 대비 normal regression, stressed p99/SLO,
   background minimum service gate를 모두 통과한다.
4. **Native 1/2/4-node scale:** capacity-normalized offered load와 workload identity를
   고정해 actual vLLM/LMCache lifecycle, utility와 fabric telemetry를 함께 비교한다.
5. **SOTA/workload matrix:** NetKV, full 또는 명확히 제한된 Kairos-compatible policy,
   MRC-compatible transport baseline과 code/chat/long-context/real-trace burst를 같은
   carrier에서 평가한다.
6. **Artifact freeze:** claim↔artifact mapping, setup/execution/analysis 시간, compact
   raw/result, figure regeneration을 SC AD/AE workflow로 고정한다.

---

## 18. 저장소 구성

```text
tempo/                         global state/candidate/coordinator/orchestrator
eval/sota_4node/               actual vLLM/LMCache nodes, launchers, clients, analyzers
results/                       compact committed evidence + local raw artifact trees
paper/tempo_go/                paper source, PDF, README figures, artifact manifests
paper/TEMPO_GO_...PLAN.ko.md    v0~C10 통합 연구 상태와 다음 gate
NERSC_AGENT_SAFETY.md          mandatory Perlmutter operating rules
```

과거 `v0~v600` 계열 파일은 성공/실패 provenance와 mechanism discovery를 보존한다.
새 실행은 historical version을 blind retry하지 않고 current canonical contract와 source
inventory에서 시작한다.

---

## 최신 4-node native 검증: M, N, O가 확정한 것

allocation `57732862`의 M/N과 새 allocation `57736076`의 O 결과는 단순
threshold tuning의 한계와 mechanism activation 확인의 필요성을 분리해 보였다.

- M: remote-favorable를 30/30, p99 3.195초로 회복했지만 miss-hot은
  30/120이고 background 1,197건을 global reject했다.
- N: receiver-tail price를 추가했지만 normal/miss-hot/remote-favorable SLO가
  각각 0/60, 0/120, 0/30이 됐다.
- O: remote 30/30과 background 72.93%를 냈지만 changed route-quarantine은
  0/1,614 decisions에서 발동하지 않았다. miss-hot p99/SLO도 35.578초/65로
  strongest fixed의 10.846초/81보다 낮았다.
- fixed routes: O campaign에서도 miss-hot p99가 10.846–80.159초로 벌어졌다.
- predictor/queue: O campaign의 remote-favorable p99는 50.888초와 53.074초지만,
  O는 3.170초였다. 반대로 normal에서는 predictor 11.035초보다 O가 26.365초였다.
- transport: native rendezvous/correctness 뒤 official LMCache/NIXL transfer
  timeout과 UCX endpoint timeout이 재현됐다.

따라서 workload가 orchestration 차이를 만들지 못한 것이 아니다. local GPU,
remote KV/receiver와 shared fabric 중 어느 곳이 먼저 포화되는지에 따라 정책
순위가 크게 바뀌었다. 현재 실패 원인은 controller가 그 상태를 충분히 관측하지
못한 채 route와 admission을 결정하고, decoder/fabric/background budget을 같은
lease로 예약하지 않는 데 있다.

정확한 결과는 다음 artifact를 기준으로 한다.

- M: `results/tempo_go_c9_causal_burst_job_57732862/analysis.json`
- N: `results/tempo_go_c9_global_frontier_job_57732862/analysis.json`
- O native: `results/tempo_go_c9_route_liveness_job_57736076_r3_canonical_outer/analysis.json`
- O corrected terminal semantics: 같은 root의 `analysis_failclosed_business_v2.json`
- O candidate-specific diagnosis: 같은 root의 `candidate_o_diagnosis.json`
- M contract: `results/tempo_go_c9_candidate_m_pressure_spill_v1/`
- N contract: `results/tempo_go_c9_candidate_n_global_frontier_v2/`
- 전체 버전·실패·수정 provenance: 통합 계획 §74.33–§74.58

다음 acceptance gate는 정상 p50 regression ≤3%, 두 stressed regime 모두에서
strongest fixed와 predictor 대비 p99 ≥15% 개선, offered SLO non-regression,
background minimum service와 observer coverage 충족이다. reject나 미완료
population을 latency 개선으로 계산하지 않는다.

---

## 19. NERSC 안전

Perlmutter에서 작업하기 전에
[`NERSC_AGENT_SAFETY.md`](NERSC_AGENT_SAFETY.md)를 읽어야 한다.

- `/`, `/global`, `/pscratch`, `/usr` 등 shared top-level을 재귀 탐색하지 않는다.
- login node에서 substantial replay, vLLM, GPU, network traffic을 실행하지 않는다.
- 명시적 승인 없이 Slurm submit/cancel/retry를 하지 않는다.
- root, `sudo`, `su`, UDI/container, system configuration, capability 변경을 사용하지
  않는다.
- 모든 실패는 exact command, job/node, log, exit receipt로 남기고 우회하지 않는다.

---

## 인용

현재 논문 초안의 BibTeX는
[`paper/tempo_go/references.bib`](paper/tempo_go/references.bib)에 있다. TEMPO-GO를
인용할 공개 citation metadata는 independent C10 repeat와 저자/venue 확정 뒤 추가한다.
