# TEMPO 전체 연구 상태·증거·글로벌 오케스트레이터 다음 목표

문서 버전: `master-v5`, 2026-08-22
대상 환경: Perlmutter native 4노드 / 16 A100 / 실제 vLLM P/D / official `LMCacheConnectorV1:UCX`

이 문서는 지금까지의 TEMPO 연구를 다음 에이전트에게 넘기기 위한 하나의 기준 문서다. 원래 목표를 축소하거나 다른 목표로 바꾸지 않는다. 과거 버전의 코드는 삭제하지 않고 증거로 보존하며, 이 문서에는 각 세대에서 무엇을 배웠는지, 무엇이 실패했는지, 현재 결론이 무엇인지, 앞으로 어떤 global orchestrator를 어떤 workload와 gate로 검증해야 하는지를 적는다.

## 0. 먼저 읽어야 할 결론

### 0.1 질문에 대한 현재의 직접 답

**현실적인 동시 부하를 주면 병목은 생긴다.** 다만 하나의 `fabric_pressure` 숫자나 하나의 queue depth로 병목의 위치를 확정할 수 없다. local decoder prefill 경로와 official LMCache remote prefill 경로는 서로 다른 단계에서 느려지지만 최종적으로 같은 decoder와 일부 endpoint 자원을 공유한다. 그래서 다음이 모두 관찰됐다.

- 낮은 부하에서는 local이 더 빠른 경우가 많다.
- decoder-local prefill이 뜨거우면 remote가 local을 이기는 구간이 있다.
- remote P/KV/receiver/install 경로가 뜨거워지면 local이 remote를 크게 이긴다.
- 두 경로가 동시에 뜨거우면 어느 경로가 이기는지가 phase와 tenant mix에 따라 이동한다.
- 실제 native C5 부하에서는 official LMCache의 `CacheEngineKey ... not found in local data` assertion과 `EngineDeadError`, frontend 503이 관찰됐다.

따라서 **orchestration은 필요하다.** 그러나 기존 request-local route selector를 조금 더 복잡하게 만드는 것이 목표가 아니다. 목표는 다음 네 자원을 함께 보는 global orchestrator다.

1. decoder admission과 decode interference
2. tenant별 SLO·fairness·starvation
3. P/D pair assignment와 prewarmed pair의 logical activation/scaling
4. local/remote route, endpoint congestion, failure quarantine/recovery

### 0.2 원래 목표는 유지한다

첨부된 원래 목표는 다음이었다.

> TEMPO Elastic-PD를 실제 vLLM P/D 경로에 통합하고, 단순 predictor와 가장 강한 고정 정책보다 유의미하게 빠른 하나의 최종 스킴으로 확정한다.

이 목표를 임의로 “route-only 연구를 중단한다” 또는 “LMCache가 항상 나쁘다는 결론”으로 바꾸지 않는다. 정확한 해석은 다음과 같다.

- **기존 고정 4노드에서 request-level local/remote route만 바꾸는 후보군**은 full performance gate를 통과하지 못해 종결됐다.
- **global orchestration 자체**는 폐기되지 않았다. 오히려 route-only 후보가 실패한 이유가 global control의 필요성을 보여준다.
- 새 연구 질문은 route decision을 global admission·fairness·pair scaling·failure recovery에 포함한 TEMPO-GO를 실제 native contention에서 검증하는 것이다.

### 0.3 현재 증거의 정확한 경계

현재까지는 다음을 주장할 수 있다.

- contention과 local/remote opposite crossover는 실제 vLLM P/D에서 재현됐다.
- remote가 항상 나쁜 것은 아니다.
- synthetic CXI background는 remote를 local보다 더 악화시킬 수 있지만, 어느 물리 자원이 원인인지 단일 counter로 확정할 수 없다.
- C4 route-only global gate는 세 구조적으로 다른 후보와 phase oracle까지 실패했다.
- TEMPO-GO의 control-plane lifecycle, explicit reject, scheduler/completion provenance, CPU failure receipt/quarantine wiring은 구현·replay 수준에서 상당 부분 닫혔다.
- native C5에서 global admission rejection과 official LMCache data-plane failure가 실제로 발동했다.
- Candidate C의 native failure-quarantine receipt가 확보됐지만 step exit `143`으로
  끝났고, Candidate D는 C의 failure safety와 B의 proactive scaling을 결합한 CPU
  neutral/negative 후보로 고정됐다. 둘 다 성능 승리나 independent validation으로
  세지 않는다.

아직 주장할 수 없는 것:

- TEMPO-GO가 native independent validation에서 strongest fixed보다 빠르다.
- TEMPO-GO가 predictor-only보다 유의미하게 빠르다.
- TEMPO-GO가 native LMCache failure를 pair quarantine/reassignment로 흡수한다.
- 4노드 결과만으로 production-ready 또는 HPC-scale superiority다.

---

## 1. 조사 범위: v0부터 최신까지

### 1.1 현재 저장소에서 확인된 lineage

“v535 하나만 보지 말고 v0부터 보라”는 요구를 반영해 conceptual v0, actual P/D source history, late raw-artifact history를 분리해서 읽는다.

| 범위 | 저장소에서 확인된 내용 | 현재 해석 |
|---|---|---|
| conceptual v0 / pre-reset | TEMPO-RD, phase-gated I/O, topology/QoS, sparse transfer, dynamic routing 세대와 git history | mechanism을 그대로 재사용하지 않는다. matched baseline과 causal evidence를 먼저 요구하는 실험 규칙만 남긴다. |
| actual P/D v1–v450 | `eval/sota_4node/`의 549 versioned TEMPO files, 323 distinct numeric revisions | actual vLLM P/D·LMCache·NIXL·workload·admission·cache·tail 실험의 본 계보다. |
| late raw artifacts v452–v544b | source filename이 아닌 run root/profile/analyzer revision으로 남은 discovery campaign | v492의 일회성 positive와 v493/v502/v534/v536/v538/v540/v543/v544b의 비재현·tail/goodput negative를 함께 봐야 한다. |
| current TEMPO-GO C5 | C0–C5 workload, five/six-arm native harness, global telemetry, tenant contract, failure receipt | route-only가 아니라 global orchestration의 integration/robustness를 검증하는 현재 단계다. |

현재 bounded workspace에서 versioned source filename의 최고 revision은 v450이고, v452–v544는 late raw-artifact lineage로 문서화돼 있다. checked-in source나 canonical artifact가 v545–v600까지 연속으로 존재한다고 가정하지 않는다. 외부에서 “v600”을 가리키는 자료가 있으면 exact path와 SHA를 추가해 이 문서의 artifact index에 명시해야 한다. 없는 버전을 있는 것처럼 합쳐서 서술하지 않는다.

### 1.2 세대별로 남은 것과 버린 것

| 세대 | 시도 | 배운 것 | 최종 운명 |
|---|---|---|---|
| conceptual v0–v4 | phase gate, I/O co-scheduling, topology/QoS, sparse/nano overlap | broad motivation은 있었으나 실제 end-to-end causal path와 novelty가 닫히지 않았다 | 역사적 motivation만 보존 |
| dynamic/v6·TEMPO-RD | look-ahead, per-rail/NVLink/libfabric/CXI control, resource-domain scheduler | complexity가 causal evidence보다 빠르게 커졌다. matched contention과 reproducibility가 먼저다 | 최종 P/D scheme에서 제외 |
| v1–v27 | actual vLLM P/D, LMCache/NIXL, KV geometry, pressure probe | admission과 transport는 분리해야 한다. 첫 화면은 local 24/24로 remote branch를 검증하지 못했다 | 실험 기반으로 보존 |
| v28–v60 | exact output, queue crossover, offered rate, local credit | remote는 항상 나쁘지 않다. 약 32 req/s 부근에서 mixed route가 두 fixed path를 이긴 사례가 있다 | crossover evidence |
| v61–v95 | one live epoch, balanced order, credit/shape threshold | mid-load remote rule이 44/48 remote로 선택돼 느려졌다. calibrated constant는 일반 법칙이 아니다 | threshold family 중단 |
| v96–v129 | interleaving, heterogeneous prompt/output | sequential arm block은 30–60 ms drift에 오염된다. 같은 epoch·counterbalance가 필요하다 | workload discipline으로 승격 |
| v131–v186 | cache catalog, affinity, warm/cold hybrid | cache residency는 route constraint다. warm reuse의 작은 이득만으로 cold contention 결론을 만들 수 없다 | cache evidence로 보존 |
| v190–v245 | saturation, tail-aware, 4K/output-256 composition | aggregate gain만으로는 부족하고 paired tail gate가 필요하다 | negative evidence |
| v248–v290 | mixed frontier, rate overload, fixed policy | LMCache failure는 availability evidence이지 TEMPO performance win이 아니다 | failure와 performance 분리 |
| v291–v321 | arrival regime, unique prompt chunks, pair-local fast path | cache-key aliasing 제거는 correctness 필수지만 predictor보다 강하다는 증거는 아니다 | correctness invariant |
| v322–v349 | burst, local credits, 25 ms microburst | local admission이 burst를 흡수할 수 있으나 fixed burst threshold는 trace-specific이다 | admission primitive |
| v353–v430 | phase change, prefix swap, adaptive/latched cap | cap-5 worst regression +435.5 ms, cap-6 +58.2 ms. trace-derived constants가 일반화되지 않았다 | route-only tuning 종료 |
| v440–v450 | native NIXL comparison, canonical Elastic-PD | one-way commit, first-response credit release, cache isolation, weighted ownership은 sound하다. predictor 대비 추가 이득은 작다 | TEMPO-GO의 component base |
| v452–v544b | profile, pair load, credits, chunking, cache, CXI background, priority | 한 번의 좋은 숫자는 재현되지 않았다. bottleneck은 moving하고 route-only tail control은 불충분하다 | terminal negative evidence |
| C5 TEMPO-GO | global admission, SLO/fairness, pair activation, endpoint completion, failure receipt | global control plane은 native에 들어갔고 overload/failure가 실제 발동했다. superiority는 아직 미검증 | 현재 연구 대상 |

---

## 2. 실제로 확인된 contention과 remote failure

### 2.1 C3 coupled crossover

실제 4노드 vLLM P/D, official `LMCacheConnectorV1:UCX`, 4094-token/2-output unique-cold foreground를 유지한 coupled C3에서 remote-tenant rate를 바꿨다.

| remote background rate | local foreground median | remote foreground median | 해석 |
|---:|---:|---:|---|
| 0/s | 528.9 ms | 448.4 ms | decoder-local path가 뜨거우면 remote가 이긴다 |
| 4/s | 542.8 ms | 589.1 ms | 승자가 local로 이동 |
| 8/s | 661.7 ms | 655.3 ms | near tie |
| 12/s | 674.9 ms | 1832.0 ms | remote P/KV/receiver path가 무너져 local이 크게 이긴다 |

rate 12에서 remote completion residual은 1569.8 ms까지 늘었고 remote background throughput은 9.23 req/s, client window drain은 10.41 s였다. 이 결과는 “remote가 나쁘다”가 아니라 “같은 topology에서 route의 서비스 상태가 이동한다”는 증거다.

### 2.2 동일 allocation background 비교

v536/v538 same-allocation 비교에서는 100% synthetic CXI background가 no-background 대비 local median을 약 30 ms, remote median을 약 181 ms 늘렸다. remote가 약 151 ms 더 악화됐지만, 이 숫자만으로 switch link·sender NIC·receiver NIC·PCIe/host backpressure·LMCache semantic operation 중 하나를 원인으로 확정할 수 없다. 따라서 physical fabric claim은 하지 않고 endpoint telemetry와 service completion residual을 수집한다.

### 2.3 C4 route-only terminal negative

세 구조적으로 다른 live 후보를 동일 workload에 적용했다.

| 후보 | mechanism | fixed 대비 median | predictor 대비 median | goodput 대비 fixed | 문제 |
|---|---|---:|---:|---:|---|
| A | instant scalar score | -2.92% | +3.48% | +10.17% | predictor gate와 tail gate 실패 |
| B | pair-local active watermark epoch | +7.10% | +17.46% | +7.67% | TPOT p99 +64.28%, worst +997.9 ms |
| C | route-pinned local external-credit epoch | +7.92% | +21.30% | +4.58% | goodput gate, C1/C3 paired, TPOT tail, worst +2278.7 ms 실패 |

phase label을 아는 oracle도 full gate를 통과하지 못했다. 따라서 더 많은 prompt coefficient, scalar pressure, phase classifier, threshold micro-tuning으로 route-only 후보를 계속 늘리지 않는다. local과 remote 모두 결국 shared decoder의 service를 외부화하므로, route-only 선택만으로 coupled decode tail을 안정적으로 제어할 수 없다는 결론이다.

### 2.4 C5 native에서 실제로 본 LMCache failure

현재 승인된 native interactive allocation `57404614`의 TEMPO failure-quarantine run에서 measured phase 도중 node-3 vLLM log에 다음 failure shape가 관찰됐다.

```text
AssertionError: Key CacheEngineKey(..., chunk_hash=...) not found in local data.
RuntimeError: Worker failed with error 'Key CacheEngineKey ... not found in local data.'
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
frontend: HTTP 503 Service Unavailable
```

이는 realistic contention 중 official LMCache data plane/receiver state가 닫히지 않은 실제 native evidence다. run 종료 후 확인한 결과는 다음과 같다.

| artifact | 관측 |
|---|---|
| native failure schema | `tempo-go-c5-native-arm-failure-v1` |
| arm / Slurm | `tempo`, job `57404614` |
| runner exit | 143 (`native_arm_process_failed`) |
| measured raw | 2,712 rows; semantic terminal summary 1,633 complete + 9 failed + 1,070 rejected |
| client validation counters | response-completed 1,642; queue-timeout 1,063; telemetry-refresh-timeout 7; HTTP error 6 |
| validity | `router_decisions_exact=false`, `terminal_contract_valid=false`, `all_streams_valid=false` |
| claim | `performance_claim_allowed=false` |
| failure receipt | native `tempo-go-global-failure-v1` 9건; pair-scope transport 3건 + route-scope HTTP 6건 |
| quarantine | `route_failure_quarantine` rejection 1,714건; failed request same-ID retry 없음 |

따라서 이 실행은 “contention 중 LMCache/EngineCore failure와 business rejection이 실제 발생했고, global failure receipt·route/pair quarantine ledger가 native에서 관측됐다”는 robustness/integration evidence다. 그러나 step exit 143, `router_decisions_exact=false`, `terminal_contract_valid=false`이므로 이를 성능 승리나 independent validation으로 세지 않는다. 에러를 latency로 치환하지 않는다.

---

## 3. prior work와 MRC 계열에서 배울 것

TEMPO의 novelty는 “P/D를 쓰자”, “KV-aware routing을 하자”, “remote/local을 선택하자”가 아니다. DistServe, P/D-Serve, Mooncake/Conductor, Splitwise, Kairos, Dynamo, EcoServe, NetKV, FlowKV/KVDirect 등과 겹치는 넓은 주장은 하지 않는다.

사용자가 제시한 [MRC 개념](https://arxiv.org/html/2606.18170v1)에서 가져올 수 있는 것은 transport 자체가 아니라 **제어 구조**다.

- path별 상태를 분리한다.
- receiver가 광고한 packet/semantic-operation bound를 별도로 둔다.
- host backpressure와 network egress/receiver pressure를 같은 scalar로 합치지 않는다.
- service-time compensation과 completion residual을 사용한다.
- stale state에서 무리하게 보내지 않고 bounded probe로 회복한다.
- victim flow/tenant의 business SLO를 보호한다.

MRC는 Ethernet AI fabric transport이고 Perlmutter Slingshot의 vLLM scheduler가 아니다. TEMPO는 MRC의 transport 성능이나 topology superiority를 주장하지 않는다. 다만 다음과 같이 request/operation granularity의 global control에 적용한다.

이 경계는 Perlmutter 구조상 특히 중요하다. NERSC architecture 문서에 따르면 GPU
node는 4개의 HPE Slingshot 11 Cassini NIC을 사용하고 GPU cabinet은 3-hop dragonfly
topology다. 따라서 4-node 실험에서 관찰하는 NCCL/Libfabric tail은 여러 endpoint와
경로가 합쳐진 end-to-end evidence이지, 특정 switch/link의 utilization을 직접 읽은
증거가 아니다. [Perlmutter architecture](https://docs.nersc.gov/systems/perlmutter/architecture/)
와 [NERSC network documentation](https://docs.nersc.gov/performance/network/)을
근거로, TEMPO는 `remote_completion_pressure`와 `shared_fabric_observed_tail`을
분리해 기록하고 physical-link claim은 하지 않는다.

MRC에서 TEMPO가 차용할 수 있는 것은 다음의 **cross-layer control analogy**다.

| MRC transport primitive | TEMPO application/global analogue |
|---|---|
| responder-advertised MPR / WriteIMM bound | decoder·receiver semantic-op·KV-byte credit |
| host backpressure / service-time compensation | LMCache install residual·endpoint service time·first-response residual |
| EV/path state와 bounded probe | pair/route health, stale telemetry quarantine, bounded recovery probe |
| sender congestion control | 새 request의 route/admission lease와 background drain |

MRC 자체는 privileged controller API와 새 wire transport를 요구하므로 현재 Perlmutter
native 실험에서 교체할 수 없다. TEMPO의 contribution은 MRC를 재구현하는 것이 아니라,
기존 NCCL AWS Libfabric과 official LMCache/NIXL data plane 위에서 transport가 노출하는
endpoint/backpressure/completion evidence와 vLLM decoder business state를 하나의
bounded global admission transaction으로 연결하는 것이다. 이는 [MRC specification
paper](https://arxiv.org/html/2606.18170v1)의 receiver-driven bounded flight,
service-time reporting, congestion/load-balancing integration을 application-layer
orchestration 문제로 옮긴 해석이며, MRC 성능을 TEMPO의 baseline으로 허위 표기하지 않는다.

```text
TEMPO state = decoder service + P service + KV bytes + semantic ops
             + receiver/install residual + endpoint advisory + tenant debt
decision     = admission + fairness + pair + route + failure state
transport    = unchanged official LMCacheConnectorV1:UCX
```

---

## 4. 새 연구 질문과 허용되는 contribution

### 4.1 연구 질문

> 동일한 native 4-node/16-GPU vLLM P/D topology와 official LMCache data plane에서, TEMPO-GO가 moving multi-tenant contention을 관찰하고 decoder admission, tenant SLO/fairness, pair assignment/proactive logical scaling, local/remote route, endpoint failure recovery를 하나의 global transaction으로 공동 제어하여 strongest fixed policy와 predictor-only보다 latency/goodput 또는 overload robustness를 개선하는가?

### 4.2 정확한 contribution 후보

다음 조건을 모두 충족할 때만 contribution으로 쓴다.

1. **Global causal state**: pair별 decoder, endpoint, transfer, tenant state를 같은 epoch/sequence로 모으되 cross-host clock subtraction을 하지 않는다.
2. **Multi-resource admission**: local token-work, remote P token-work, remote KV bytes, semantic operation count를 독립적으로 bound한다.
3. **Business-aware fairness**: tenant weight, SLO deadline, minimum service fraction, queue-wait budget을 controller의 admission order에 직접 반영한다.
4. **Pair assignment/scaling**: prewarmed pair를 logical active set에 포함/제외하고, 이미 commit된 request를 migration하지 않는다.
5. **Endpoint failure recovery**: failure를 explicit receipt로 닫고, released work를 정확히 반환하며, pair×route를 quarantine하고 new request ID만 surviving route/probe로 보낸다.
6. **Fixed data plane**: LMCache transport를 교체하지 않고 동일 baseline과 동일 route data plane을 사용한다.

반대로 다음은 contribution이 아니다.

- scalar `fabric_pressure`
- fixed prompt-token coefficient
- phase label classifier
- future arrival 사용
- per-request route threshold만 추가
- native NIXL/Mooncake transport 교체
- synthetic CXI background를 production traffic으로 둔갑

---

## 5. TEMPO-GO global scheme

### 5.1 system topology

```text
Perlmutter 4 nodes / 16 A100
  pair-0: node-0 P + node-1 D
  pair-1: node-2 P + node-3 D
  official LMCacheConnectorV1:UCX

tenant ingress
      │
      ▼
global coordinator
  ├─ telemetry collector / freshness validator
  ├─ tenant SLO + weighted fairness ledger
  ├─ pair×route candidate builder
  ├─ multi-resource admission / bounded queue
  ├─ active-pair set / logical scaling controller
  └─ failure receipt / quarantine / probe recovery
      │ immutable request-start commit
      ▼
pair router → local decoder prefill OR official remote prefill
      │
      └─ endpoint completion / first response / EOF / failure receipt
```

process를 매 요청마다 재시작하지 않는다. pair activation은 logical admission capacity의 변화이며 physical GPU migration이나 switch reconfiguration으로 과장하지 않는다.

### 5.2 세 개의 제어 시간축

| 시간축 | 제어 | 입력 | 금지 |
|---|---|---|---|
| fast request | pair×route immutable commit, credit acquire, reject/queue | 현재 fresh telemetry, confirmed cache evidence, request contract | future arrival, phase label, hidden fallback |
| epoch | service residual, route health, bounded windows, fairness debt | completion/first-response/EOF, endpoint snapshot | stale EWMA 단독 회복, scalar fabric classifier |
| slow global | active pair set, tenant reservation/fairness, prewarmed capacity | sustained queue/SLO/endpoint pressure | process migration, physical NIC privilege, post-hoc profile tuning |

### 5.3 resource vector

각 pair×route는 다음 resource vector를 유지한다.

| 영역 | 관측/소유 단위 | 이유 |
|---|---|---|
| decoder | `active_sequences`, `decode_tokens`, scheduler running/waiting, KV usage | 모든 route가 최종적으로 공유하는 외부성 |
| local prefill | `local_prefill_token_ms`, local prefill count/service | decoder-local admission bound |
| remote prefill | `remote_prefill_token_ms`, P endpoint request/service | P endpoint queue bound |
| transfer | `remote_kv_bytes`, bytes in flight | byte-bound path |
| semantic | `remote_semantic_ops`, operations in flight | 작은 transfer라도 op-bound일 수 있음 |
| endpoint | first-response residual, transfer enqueue/complete, endpoint request count | queue depth만으로 못 보는 service inflation |
| fabric advisory | per-endpoint pause/ECN/blocked/retry/timeout if available | attribution/safety only; route classifier가 아님 |
| business | tenant debt, queue wait, SLO remaining, minimum service | starvation 및 overload 방지 |

controller-owned와 endpoint-observed 값은 합산해 double-count하지 않는다. 같은 resource의 owned/observed snapshot은 `max(owned, observed)` 또는 명시된 reconciliation rule로 합친다. stale/partial/identity mismatch는 zero pressure가 아니라 fail-closed queue/deny다.

### 5.4 request admission

1. ingress에서 request를 tokenized/cache evidence와 tenant contract에 매핑한다.
2. `UNKNOWN` cache를 hit로 취급하지 않는다.
3. 각 active pair의 `LOCAL`, `REMOTE` 후보를 만든다.
4. 후보마다 resource vector, cache affinity, predicted service, uncertainty, tenant business cost를 계산한다.
5. global capacity와 fairness order를 동시에 만족하는 한 후보만 commit한다.
6. capacity가 없으면 bounded queue 또는 explicit `REJECT`를 반환한다.
7. prefill 시작 후 route 변경·silent local fallback·same-request retry를 금지한다.
8. first response에서 prefill/endpoint/transfer credit을 반환하고, EOF에서 decode credit을 반환한다. complete/abort/timeout/failure마다 정확히 한 번 반환한다.

### 5.5 tenant fairness와 business contract

request count가 아니라 tenant별 business unit을 기록한다.

- `weight`: priority가 아니라 expected service share 계산의 입력
- `minimum_service_fraction`: prolonged load에서 starvation 방지
- `maximum_queue_wait`: admission deadline
- `ttft_slo_ms`, `tpot_slo_ms`, `e2e_slo_ms`: SLO-goodput 계산
- `weighted_service_debt`: fair ordering용
- `raw_service_units`: 실제 받은 resource/service량; weighted debt와 혼동 금지

현재 C5 manifest contract는 다음을 사용한다.

| tenant | weight | min service | max wait | TTFT SLO | TPOT SLO | E2E SLO |
|---|---:|---:|---:|---:|---:|---:|
| latency | 4.0 | 0.15 | 0.5 s | 1 s | 100 ms | 4 s |
| interactive | 2.0 | 0.15 | 1 s | 2 s | 150 ms | 8 s |
| batch | 1.0 | 0.10 | 2 s | 3 s | 250 ms | 16 s |
| background | 0.5 | 0.05 | 5 s | 5 s | 400 ms | 30 s |

최종 validation에서는 tenant별 SLO-goodput, starvation count, maximum wait, weighted service ratio, Jain fairness를 모두 낸다. aggregate complete 수가 늘었다고 latency tenant가 굶으면 실패다.

### 5.6 pair assignment와 logical scaling

- low load에서는 minimum active pair를 유지해 cache affinity와 locality를 보존한다.
- active pair의 dominant utilization·queue·SLO pressure가 sustained threshold를 넘으면 fresh telemetry가 있는 spare prewarmed pair를 activate한다.
- pair activation은 신규 request admission에만 적용한다. 이미 commit된 request는 원래 pair에서 EOF/failure까지 끝낸다.
- deactivate는 held resource가 0이고 hysteresis idle window가 지난 뒤에만 한다.
- stale telemetry면 activation을 추측하지 않는다.
- pair scale이 local/remote route보다 먼저 결정되는 경우와 route가 pair selection에 영향을 주는 경우를 trace에 명시한다. 두 제어기가 서로 다른 owner로 drift하지 않게 한다.

### 5.7 failure receipt와 quarantine

failure는 다음 receipt를 남겨야 한다.

```text
schema: tempo-go-global-failure-v1
request_id / new_request_id_required
tenant / pair / route
failure_phase / telemetry_sequence / profile_fingerprint
released_work:
  active_sequences
  decode_tokens
  endpoint_requests
  local_prefill_token_ms
  remote_prefill_token_ms
  remote_kv_bytes
  remote_semantic_ops
quarantine_scope: pair×route
reassignment_policy: new_request_id_required
recovery: PROBE before GOOD
```

실패한 request 자체를 같은 ID로 이동하지 않는다. pending request는 surviving pair×route로 새 ID를 받을 수 있지만, 그 사실을 retry latency에 섞지 않고 별도 business event로 기록한다. pair×route는 explicit low-rate probe가 성공하기 전까지 `QUARANTINED`다. native EngineCore가 죽어 receipt를 쓸 수 없다면 orchestrator success가 아니라 **unreceipted execution failure**로 분류하고 runner가 signal/process-failure artifact를 남겨야 한다.

---

## 6. workload를 설정하는 정확한 방법

### 6.1 공통 실행 계약

- native 4 nodes / 16 GPUs / same server lifecycle
- pair-0 = node-0/1, pair-1 = node-2/3
- model = Qwen2.5-7B-Instruct
- P/D transport = `LMCacheConnectorV1:UCX`
- warmup은 measurement 밖
- client request rate flag는 생략하고 manifest의 absolute arrival offset을 사용
- synthetic CXI/network background는 headline C5에서 금지; 별도 attribution ablation으로만 유지
- 같은 request set, same cache namespace, same counterbalanced arm order
- phase name은 분석 label이지 controller input이 아님
- `future_arrivals`, oracle route, physical switch label, phase label을 policy input으로 넣지 않음

### 6.2 다섯 arm

최종 비교는 같은 trace·topology·server lifecycle에서 다음 다섯 arm을 사용한다.

1. `ALWAYS_LOCAL`
2. `OFFICIAL_LMCACHE_ALWAYS_REMOTE`
3. `PREDICTOR_ONLY`
4. `QUEUE_GPU_ONLY` (Kairos-like GPU/queue-only ablation)
5. `TEMPO_GO`

manifest가 historical compatibility 때문에 `kairos_like`를 별도 arm으로 갖는 경우에는 “다섯 정책”인지 “여섯 label”인지 analyzer에서 명시한다. arm 이름을 바꿔 성능 비교 수를 늘리지 않는다.

### 6.3 C5 phase matrix

현재 anchor manifest의 정확한 구조는 다음과 같다.

| phase | duration | request 수/replicate | intended pressure |
|---|---:|---:|---|
| `c0_cool` | 15 s + 2 s cooldown | 30 | baseline |
| `c1_decoder_hot` | 15 s + 2 s cooldown | 366 | decoder-local prefill hot |
| `c2_remote_hot` | 15 s + 2 s cooldown | 102 | remote P/transfer hot |
| `c2_kv_remote_hot` | 15 s + 2 s cooldown | 210 | P-only/KV remote path |
| `c3_both_hot` | 15 s + 2 s cooldown | 618 | both paths coupled |
| `recovery` | 15 s + 2 s cooldown | 30 | pressure release/recovery |

2 replicates로 총 2,712 request다. anchor rates는 C1 decoder-hot 22.4/s, C2 remote-hot 4.76/s 또는 12/s, C3 both-hot이며, 이 rate는 final universal constant가 아니라 현재 C1/C2/C3 characterization을 재현하는 frozen workload input이다.

### 6.4 tenant와 prompt/cache matrix

tenant stream은 `latency`, `interactive`, `batch`, `background`를 섞는다. prompt source pool은 최소 512, 2048, 4094 tokens를 포함한다. C1/C2 anchor에서 사용한 output=2는 fixed-path mechanism characterization용 `screen_only` prior다. output=2 anchor를 final business-performance evidence로 재사용하지 않는다.

단, 위 문장의 output 설명은 현재 v3 artifact와 일치하지 않아 정정한다. 실제
`build_tempo_go_c5_manifest.py`와 v3 `validation.jsonl`을 다시 세면 foreground는
`(512,16)`, `(2048,256)`, `(4094,16)`이고, `decoder-hot`/`remote-hot`/
`kv-remote-hot` stream은 모두 `max_tokens=2`다. v3 row count는 foreground
`16=240`, foreground `256=120`, hot stream `2=2,352`이며 `max_tokens=128`
요청은 없다. CLI의 `--background-output-tokens 128`은 현재
`build_contention_workload()`에서 row 생성에 사용되지 않고 manifest field에만
기록된다. 따라서 v3는 **output=2 anchor-hot discovery workload**이지
background output=128의 business workload가 아니다. 이 사실을 숨기고 v3를
최종 multi-tenant performance workload라고 부르지 않는다.

다음 held-out workload를 만들 때는 이 문제를 먼저 해결하고 contract에 실제
row geometry를 기록한다. C1/C2/C3 rate를 그대로 재사용할지, hot stream을
`4094/128`로 바꿀지는 fixed-path direction gate와 endpoint/Elastic evidence를
먼저 통과시켜 frozen한다. CLI parameter 이름만 바꾸고 실제 JSONL geometry가
바뀌지 않는 상태로 native를 실행하지 않는다.

cache 상태는 다음처럼 분리한다.

| state | 의미 | policy 처리 |
|---|---|---|
| `MISS` | measured cache miss | local/remote 둘 다 후보가 될 수 있으나 resource/tenant state로 결정 |
| `P_ONLY` | prefill-side cache evidence가 실제 완료 event로 확인됨 | remote 후보 허용 조건 중 하나 |
| `D_ONLY` | decoder-side reuse evidence | local/cache affinity 후보 |
| `BOTH` | 양측 evidence | affinity와 queue를 함께 봄 |
| `UNKNOWN` | catalog/receipt 불완전 | hit로 취급 금지, fail-closed |

arm·prompt·tenant별 cache key namespace를 분리하고, baseline의 warmup이 다음 arm의 measured hit로 누출되지 않게 한다. request ID에 cache state label을 넣더라도 실제 source hit receipt가 없으면 evidence가 아니다.

### 6.5 workload validity gate

성능분석 전에 fixed-path characterization을 먼저 수행한다.

- C1에서 always-remote가 always-local을 이기는 방향이 재현되는가?
- C2에서 always-local이 always-remote를 이기는 방향이 재현되는가?
- C3에서 두 경로가 동시에 바빠질 때 queue/service residual이 실제로 움직이는가?
- P_ONLY branch가 실제 LMCache retrieve/receiver/install을 타는가?
- 모든 output/stream/route/cache provenance가 valid인가?
- 각 arm의 reject와 failure가 terminal receipt로 닫히는가?

이 방향 gate가 실패하면 controller를 튜닝하지 말고 workload/cache preparation부터 수정한다. phase label로 결과를 맞추지 않는다.

---

## 7. telemetry와 분석 데이터 계약

### 7.1 per-request ledger

필수 event:

```text
arrival
telemetry_snapshot(sequence, freshness, pair identity)
classified(cache state, tenant contract)
admission / explicit reject / queue
pair_assigned
route_committed(LOCAL|REMOTE)
credit_acquired
upstream_started
prefill_start / prefill_end
kv_transfer_start / kv_transfer_end(bytes, semantic_ops)
first_token / first_response
eof / abort / timeout / route_failure
credit_released(exactly once)
terminal
```

각 row에 request id, tenant, prompt/output geometry, profile/manifest SHA, scheduler observation source, endpoint receipt, pair, route, reason, error, TTFT/TPOT/E2E, output token/text digest를 남긴다.

### 7.2 global/endpoint snapshot

- scheduler running/waiting, active sequences, KV usage
- P endpoint request/queue/service
- D endpoint request/queue/service
- local/remote prefill token-ms
- KV bytes와 semantic operation count
- first-response completion residual
- route health: GOOD/SKIP/DENIED/PROBE/QUARANTINED
- tenant queue wait, debt, service units, SLO remaining
- stale/partial/identity mismatch count
- telemetry collection span, admission CPU time, coordinator overhead

Cassini/NIC counter는 지원되는 범위에서 per-endpoint advisory로 기록한다. TX pause, RX pause, host-blocked, ECN, retry, timeout, overflow를 같은 scalar로 합치지 않는다. counter가 없으면 missing으로 남기며 0으로 바꾸지 않는다. cross-host monotonic timestamp를 빼지 않는다.

### 7.3 반드시 분리해서 보고할 metric

| 차원 | metric |
|---|---|
| latency | TTFT/TPOT/E2E p50/p95/p99, queue wait |
| throughput | completed request goodput, output-token goodput |
| business | tenant별 SLO-goodput, minimum service, starvation, maximum wait |
| fairness | weighted service ratio, raw service units, Jain fairness |
| route | local/remote 선택 수, route-specific counterfactual, selected-route correctness |
| scaling | pair activation/deactivation, active pair residency, activation latency |
| failure | receipt count, released work, quarantine duration, probe recovery, unreceipted failure |
| overhead | telemetry span, coordinator CPU, admission latency, dropped/stale snapshot |

failed request를 p99 latency의 한 row로 억지로 넣지 않는다. failure rate와 business reject는 별도 metric이고, performance claim에 포함할 때는 사전 정의된 robustness analysis를 사용한다.

---

## 8. 성공 gate와 축소 규칙

### 8.1 correctness gate

다음 중 하나라도 실패하면 성능 수치를 headline으로 쓰지 않는다.

- stream/output/token digest 100%
- route provenance 100%
- hidden recompute·silent fallback·same-ID retry 0
- transfer error/timeout/unclosed terminal queue 0
- credit leak/underflow/double release 0
- request마다 terminal receipt 1개
- tenant starvation 0, explicit queue/reject semantics
- failure receipt에 released work/quarantine scope/profile identity 존재

### 8.2 primary performance gate

frozen profile과 **독립 allocation**에서만 평가한다.

- TEMPO-GO vs strongest fixed: pooled E2E median ≥10% 개선
- TEMPO-GO vs predictor-only: E2E median ≥5% 개선
- output-token 또는 request goodput: strongest fixed 대비 ≥5% 개선
- paired E2E win rate 전체 ≥75%, 각 workload group ≥60%
- group별 E2E p99와 TPOT p99 regression ≤5%
- worst paired E2E regression ≤100 ms
- 선택된 local은 remote counterfactual보다, 선택된 remote는 local counterfactual보다 각각 median ≥5% 유리

### 8.3 robustness alternative

median gate를 통과하지 못해도 다음이 모두 성립하면 robustness contribution으로만 쓸 수 있다.

- overload에서 fatal failure/queue timeout을 제거하거나 명확히 감소
- p99 또는 goodput ≥15% 개선
- normal-load performance regression ≤3%
- tenant fairness와 correctness gate 유지

### 8.4 중단·축소 규칙

- 독립적으로 구조가 다른 후보 두 개가 predictor 대비 5% 개선하지 못하면 threshold micro-tuning 중단
- remote가 실제로 유리한 fixed-path workload를 재현했는데도 5% 이득이 없으면 route predictor를 더 복잡하게 만들지 않음
- tail과 median을 동시에 달성하지 못하면 route-only 후보를 버리고 global admission/fairness/pair/failure로 범위를 유지하거나, 그래도 안 되면 simpler local-first/predictor-only로 축소
- native failure가 receipt를 닫지 못하면 성능 run을 반복하지 말고 failure closure부터 수정
- allocation에 teardown/artifact 수집을 제외하고 30분 미만 남으면 새 candidate를 시작하지 않음

---

## 9. Perlmutter 실행·안전 계약

### 9.1 어디서 무엇을 실행하는가

login node:

- bounded `rg`, source audit, schema/hash 확인
- unit/CPU replay/py_compile
- launcher 정적 검사
- raw artifact analyzer

approved interactive allocation:

- vLLM P/D server
- LMCache/UCX native path
- multi-tenant workload client
- endpoint telemetry와 global orchestrator
- substantial replay/native run

최근 사용한 native allocation `57402376`과 `57404614`는 모두 종료됐다. 전자는
guard profile 기반 counterbalanced five-arm discovery, 후자는 Candidate C
failure-quarantine arm이었다. 현재 활성 allocation이 있다고 가정하지 말고, 새
GPU 실행은 frozen candidate/workload/run contract와 사용자의 승인 뒤에만 한다.
`57404614`를 재사용하거나 같은 native arm을 맹목적으로 retry하지 않는다.

### 9.2 절대 금지

- container, Shifter/Apptainer/Podman/Docker, `--image`
- udiRoot 생성/수정/ownership 변경
- `sudo`, `su`, root shell, `/etc`, `/usr`, `/opt` system write
- privileged NIC control, `CAP_NET_ADMIN`
- broad shared-filesystem traversal
- 자동 Slurm submit/cancel/retry loop
- dirty worktree의 unrelated change 삭제·reset·stage

`udiRoot.conf must be owned by user root` 또는 `exit 139`가 나오면 launcher/environment 계약 위반 또는 native crash로 분류한다. ownership을 고치거나 root 권한을 얻어 우회하지 않는다. 정확한 command, allocation, node, log, signal/exit code만 보존하고 실행을 멈춘다.

### 9.3 실행 순서

```text
G0  native identity / UCX / vLLM / LMCache capability
G1  telemetry completeness and profile/manifest SHA binding
G2  fixed-path C1/C2/C3 workload validity
G3  CPU replay, fairness, failure injection, credit invariant
G4  one approved interactive discovery allocation
G5  freeze controller/profile after discovery
G6  new independent allocation: primary or robustness/fairness validation
G7  analyzer, artifact SHA, claim-boundary report
```

G4 결과를 보고 profile을 다시 바꾸면 G6가 아니다. failure closure를 위해 코드를 바꾼 경우에는 새 candidate revision으로 이름을 올리고, 이전 raw를 덮어쓰지 않는다.

---

## 10. 현재 구현·실험 상태

### 10.1 닫힌 항목

- `tempo/pd_global_orchestrator.py`에 tenant SLO/fairness, dominant-resource service, bounded queue/reject, pair activation 후보가 있다.
- `tempo/pd_global_telemetry.py`와 pair router/frontend가 scheduler observation과 endpoint completion receipt를 연결한다.
- C5 manifest와 global profile이 workload/profile fingerprint에 bound된다.
- C1/C2 anchor prior를 이용한 deterministic five-arm CPU replay는 모든 row를 complete 또는 explicit reject로 terminal 처리하고 resource leak/inflight/queue residual 0을 확인했다.
- Candidate C failure-injected CPU replay는 `tempo-go-global-failure-v1` receipt 1개, released work, pair-0 remote quarantine, 이후 같은 pair remote admission 0을 확인했다.
- Candidate D는 C의 `deny_until_probe`/semantic reserve와 B의 queue/SLO-risk
  proactive scaling을 결합했지만 동일 trace aggregate가 C와 같아 native로 승격하지
  않았다.
- native guarded five-arm discovery는 local/predictor/remote의 2,712/2,712 raw와
  queue-GPU-only process-failure receipt를 확보했고, TEMPO arm은
  scheduler-observe-only snapshot 5,424건과 endpoint completion receipt를 닫았다.
  최신 bounded regression suite는 `131 passed`다.

### 10.2 아직 gap인 항목

- endpoint profile scope가 `calibration_only`인 상태를 넘어서는 frozen validation
  profile promotion과 새 held-out workload/run contract
- 현재 C5 anchor endpoint profile의 17개 row는 모두 `cache_residency=
  prefill_only(P_ONLY)`다. `MISS`/`UNKNOWN` 요청에 대한 exact service row가
  없고, frozen frontend는 discovery 때만 허용하던 external geometry-ceiling
  proxy를 자동으로 허용하지 않는다. 따라서 기존 profile을 scope와 manifest
  hash만 바꿔 frozen으로 승격하는 것은 유효하지 않다.
- 이 gap은 두 방법 중 하나로만 닫는다. (a) held-out geometry와 실제 cache
  residency별 endpoint completion calibration을 먼저 수집해 exact rows를
  만들거나, (b) numeric row를 몰래 확장하지 않고, 사전에 allowlist한
  evidence-bound frozen proxy contract를 schema/code/receipt/test에 명시한다.
  단순히 `allow_service_proxy=True`로 flag를 뒤집거나 MISS를 P_ONLY로
  재분류하는 것은 금지한다.
- strongest fixed/predictor 대비 TEMPO-GO native independent performance 또는
  preregistered overload robustness
- tenant SLO-goodput/fairness 우위와 pair activation/scaling의 native benefit
- queue-GPU-only failure를 latency arm으로 대체하지 않으면서도, 동일 frozen
  validation에서 baseline failure와 TEMPO business reject/failure를 공정하게
  분리하는 analyzer closure
- 두 번째로 구조적으로 다른 global candidate의 독립 validation negative를 통해
  broader global scope를 종료할지 여부

이번 구현은 위 gap을 자동으로 닫았다고 해석하지 않는다. held-out workload와
proxy policy schema/test는 준비 단계만 닫았고, endpoint profile의
`calibration_only` scope와 exact MISS evidence, native independent validation은
여전히 미완료다.

이번 continuation에서 추가한 구현과 검증:

- 기존 v3 builder/artifact를 덮어쓰지 않고
  `eval/sota_4node/build_tempo_go_c5_heldout_manifest.py`로 held-out
  `results/tempo_go_c5_heldout_output128_v1/`를 만들었다. manifest SHA는
  `6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c`,
  validation workload SHA는
  `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33`다.
  실제 geometry는 foreground `(512,16)/(2048,256)/(4094,16)`, 모든 hot
  stream은 output `128`이고, `r02/r03`, MISS 1,992 unique, P_ONLY 720이다.
  validator report SHA는
  `f00157c5f237c7a271197e499046e0e2a9884881cffeca46554accd015933fd0`다.
- frozen 경로에서는 기존 `allow_service_proxy=True` flag를 넓게 재사용하지
  않는다. `FrozenServiceProxyPolicy`가 endpoint profile identity, calibration
  receipt SHA, 허용 geometry/residency/lookup mode를 profile fingerprint에
  묶고, proxy non-exact·numeric row 불변·performance claim 금지를 강제한다.
  frozen global profile에 policy가 없으면 로드 단계에서 fail-closed하며,
  policy는 GlobalOrchestrator capacity/config 인자로 전달되지 않는다.
- frontend와 CPU replay가 이 policy를 candidate builder에 전달하도록 고쳤다.
  global/profile/frontend/replay/run-contract bounded suite는 현재
  `128 passed, 11 subtests passed`다. source revision 때문에 기존
  `results/tempo_go_c5_frozen_contract_v2/native_run_contract.json`은 stale로
  판정되며 기존 contract를 덮어쓰지 않는다. 새 held-out frozen contract는
  exact endpoint calibration 또는 receipt-bound proxy source가 닫힌 뒤 별도
  경로에서 생성해야 한다.

### 10.3 현재 실행된 native run의 최종 판정

`results/tempo_go_c5_native_failure_quarantine_job_57404614_v1`는 native TEMPO failure-quarantine run으로 종료됐다. measured `raw.json`과 top-level `tempo/failure.json`은 존재하지만 `result.json`은 없고, failure receipt/quarantine event는 raw router ledger 안에서 확인됐다. 이 run의 최종 판정은 다음과 같다.

- `raw.json`: 2712 requests; semantic terminal summary 1633 complete, 9 failed, 1070 global reject
- raw client validation의 `completed_count=1642`는 failed HTTP 200/terminal rows를 포함하는 response counter이므로 semantic complete와 동일시하지 않는다.
- `router_decisions_exact=false`
- `terminal_contract_valid=false`
- `all_streams_valid=false`
- `performance_claim_allowed=false`
- errors: queue timeout 1063, telemetry refresh timeout 7, HTTP error 6; native global failure receipt 9, quarantine rejection 1714
- top-level schema: `tempo-go-c5-native-arm-failure-v1`
- top-level exit code: 143

job이 최종 종료한 뒤 검사한 순서는 다음과 같다.

1. Slurm job state와 step exit/signal
2. raw/result/failure 존재 여부와 schema
3. node별 EngineDeadError/LMCache failure context
4. frontend/router의 route_failure/failure_receipt/quarantine event
5. request ledger terminal coverage와 released resource
6. receipt SHA와 profile/manifest SHA

이번 결과는 “native LMCache/EngineCore failure observed, global failure/quarantine receipt observed, but terminal-valid independent performance artifact missing”으로 기록한다. receipt가 있더라도 이는 robustness evidence이며, 성능 승리로 세지 않는다.

이 failure raw를 분석하는 과정에서 증거 파이프라인의 별도 결함도 발견해 닫았다. 기존
analyzer는 `failure.json`만 남은 arm을 zero-request execution failure로 요약했기
때문에, 옆에 보존된 `tempo_go_c5_discovery/raw.json`의 실제 terminal ledger와
failure receipt를 읽지 못했다. 이제 raw-backed native failure에서는 top-level
failure receipt를 immutable binding으로 유지하면서 raw request/decision coverage,
semantic terminal phase, failure kind/scope, tenant/phase service metrics, pair
activation과 telemetry를 분석한다. `router_decisions_exact=false`인 execution
failure를 성공으로 승격하지 않으며, 분석 결과도 계속
`performance_claim_allowed=false`다.

재분석 artifact는
`results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/native_c_analysis_raw_backed_v1.json`
(SHA `579f92d38140f0f7ccb31f18a19ce9c9670ea5b3371ba48e99cf7850dbd3a1ac`)이다.
이 artifact에서 semantic terminal phase는 complete `1,633`, failed `9`, rejected
`1,070`, global failure receipt `9`(pair `3`, route `6`)로 복원됐고, service-metric
기준 completed는 `1,623`, failed `9`, rejected `1,070`이다. 이는 native 실행을
재시도한 결과가 아니라 기존 raw/failure receipt에 대한 CPU-side analyzer closure다.

### 10.3.1 최신 guarded five-arm discovery와 Candidate D 판정

`results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1`은 같은 v3 trace와
guard profile을 사용한 native discovery다. arm order는
`tempo → queue_gpu → predictor → remote → local`이다. local/predictor/remote는
각각 2,712/2,712 valid였고, queue-GPU-only는 LMCache receiver allocation
timeout/EngineCore 종료를 `tempo-go-c5-native-arm-failure-v1` receipt로 닫았지만
latency 결과는 만들지 못했다. TEMPO는 1,865 complete와 847 explicit global
reject, E2E p50/p99 `983.9/14,485.0 ms`, output-token goodput `136.9/s`를
기록했고 always-local `190.2/s`보다 낮았다. 따라서 이 run은 global wiring,
business reject, scheduler/completion provenance와 baseline failure isolation의
discovery evidence이지 성능 승리나 independent validation이 아니다.

이 native discovery에서 TEMPO global provenance는 pair당 2,712건, 총 5,424건의
`router_local_vllm_prometheus_observe_only` scheduler snapshot을 담았고 invalid
snapshot은 0이었다. Candidate C native run `57404614`의 9개 failure receipt와
Candidate D CPU replay는 각각 failure robustness와 mechanism-neutral evidence로
보존한다. D는 native에 올리지 않는다.

추가 audit에서는 native frontend가 모든 tenant에 endpoint 기본 deadline을
전달하여 짧은 tenant SLO가 queued-request ordering에서 약화되는 문제도 확인했다.
`GlobalOrchestrator._effective_deadline_ns()`가 외부 remaining deadline을 tenant의
frozen E2E SLO로 cap하도록 수정했고, candidate feasibility와 fair queue ordering이
동일한 business deadline을 사용하게 했다. 이는 route threshold나 phase input이
아닌 tenant-aware global admission correctness fix다. 관련 focused suite는 이
수정 후 `78 passed`이며, native 성능 결과나 기존 artifact의 재검증 승격은 아니다.

### 10.4 authoritative artifacts

- 전체 계보/실행 playbook: [`TEMPO_GLOBAL_ORCHESTRATOR_CANONICAL_PLAYBOOK.ko.md`](TEMPO_GLOBAL_ORCHESTRATOR_CANONICAL_PLAYBOOK.ko.md)
- 이전 contention audit: [`TEMPO_ELASTIC_PD_CONTENTION_AUDIT.md`](TEMPO_ELASTIC_PD_CONTENTION_AUDIT.md)
- 현재 goal/실행 프롬프트: [`../eval/sota_4node/TEMPO_GO_GLOBAL_ORCHESTRATOR_GOAL_V1.ko.md`](../eval/sota_4node/TEMPO_GO_GLOBAL_ORCHESTRATOR_GOAL_V1.ko.md)
- CPU failure replay: `results/tempo_go_c5_quarantine_replay_v2/five_arm_replay_remote_failure_index0_v1.json`
- CPU failure replay SHA: `2edd9f616fc94f4ee6e55e88e6a647b1cd55974f05ea130d5203f2f428270f21`
- failure receipt SHA: `3f4b0b773e63bb341b7d87a7dc71a6c743d677822f2e27af60020b7645fd0410`
- C5 manifest: `results/tempo_go_c5_cpu_gate_20260821_anchor_v3_retry2/tempo_go_workload_manifest.json`
- manifest SHA: `849bb5cf284c60215d12165e409ac426adc6e5bba3427cda8932c7379fb819fd`
- validation workload SHA: `38224ae6e421a0950080951a963ff7d82af480edfa15220c9a45c5c2064ad2f5`
- held-out output128 manifest:
  `results/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json`
- held-out output128 manifest SHA:
  `6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c`
- held-out validation workload:
  `results/tempo_go_c5_heldout_output128_v1/workloads/validation.jsonl`
- held-out validation workload SHA:
  `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33`
- held-out manifest validation SHA:
  `f00157c5f237c7a271197e499046e0e2a9884881cffeca46554accd015933fd0`
- held-out geometry: foreground `(512,16)/(2048,256)/(4094,16)`, hot output
  `128`, `r02/r03`, `MISS 1,992 unique / P_ONLY 720`; performance claim forbidden
- post-policy v3 CPU replay:
  `results/tempo_go_c5_quarantine_replay_v3_after_proxy_contract.json`
- post-policy v3 replay SHA:
  `46af0a06cdc8a043caf31bd5852240040a54810ce3365cbfd94467b2ce332c64`
  (all five arms terminal/leak-free; performance claim forbidden)
- current native root: `results/tempo_go_c5_native_failure_quarantine_job_57404614_v1`
- native failure artifact SHA: `cc05a04378e05e0a07dae2386ec081e1df78f9f2e3dd807a592dfa2b639ccd6e`
- native measured raw SHA: `c61626d6cef2b7353e0ec8a21609a9bc3b72ea6e4ed240ff5de2216cf9292124`
- raw-backed native analysis: `results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/native_c_analysis_raw_backed_v1.json`
- raw-backed analysis SHA: `579f92d38140f0f7ccb31f18a19ce9c9670ea5b3371ba48e99cf7850dbd3a1ac`
- guarded five-arm analysis:
  `results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1/native_five_arm_analysis_v3.json`
  (SHA `921ec4ad74dc28604bc65a65a734e8638817cf4d1b51d745a416064820cd350d`)
- guarded TEMPO raw SHA:
  `7ae7552a39e132c3e00a670e310fe04421fd44b5ebebc4b17dc3f880caeea87e`
- queue-GPU-only native failure receipt SHA:
  `8b56e53bb7e6b8ff975742c185810acfcd8b55f9f445cf11ab04fa4cda5e4c38`
- Candidate D profile SHA/fingerprint:
  `d8bb3e893fa3279e004e020c2dcf1e34bf7af46dd0ff1d4527863a49816f566d` /
  `75bc2b6f76bded31f1582aac46e2d3594afdf4c79714b80535afa6987848ab18`
- Candidate D same-trace replay SHA:
  `b9567186c224a41a74bedf8744e0a797ba4a0c7838908574c5e0dee9f97777b9`
- tenant-SLO deadline ordering fix: `tempo/pd_global_orchestrator.py`의
  `_effective_deadline_ns()`와 관련 CPU suite `78 passed`; native validation 전
  새 code revision으로 freeze 필요

raw evidence is immutable by path and SHA. 새 결과를 만들 때 기존 artifact를 overwrite하지 않는다.

---

## 11. 다음 에이전트가 바로 실행할 작업

1. 이 master 문서와 canonical playbook/audit/goal 파일을 읽고, v0 conceptual →
   v1–v450 source → v452–v544 raw → C0–C5 순서로 evidence map을 다시 확인한다.
2. 종료된 `57404614`와 `57402376` raw/failure receipt를 canonical analyzer로
   재검증하되, 같은 native runner를 재시작하지 않는다.
3. Candidate D를 native 후보로 부활시키지 않는다. v3 workload의 실제 geometry
   bug(`--background-output-tokens`가 row에 적용되지 않음)는 별도 held-out
   builder로 고정했지만, held-out manifest만으로 independent validation이
   되지는 않는다. 현재 endpoint profile이 P_ONLY row만 가진다는 사실을
   고려해 exact residency calibration 또는 명시적 evidence-bound frozen proxy
   source receipt를 먼저 닫고, 그 뒤 새 manifest/profile/run-contract를 함께
   freeze한다. 이번에 구현한 `FrozenServiceProxyPolicy`가 있다고 해서 현재
   calibration-only endpoint profile을 자동 promotion하지 않는다. 기존 v3와
   held-out trace 모두 native performance evidence라고 부르지 않는다.
4. fixed-path C1/C2/C3 direction, tenant business contract, manifest/profile
   binding을 통과시키고, global candidate는 global resource vector·fairness·pair
   active set·failure state를 한 transaction으로 유지한다. request-local threshold
   family는 다시 탐색하지 않는다.
5. CPU replay, failure injection, fairness/telemetry regression과 analyzer gate를
   먼저 실행한다. 새 GPU는 frozen candidate와 held-out manifest가 고정되고,
   사용자가 승인한 native 4-node/4-hour interactive allocation이 있을 때만 한다.
6. 독립 validation 결과를 보고 profile/code/manifest를 수정하지 않는다. primary
   또는 robustness/fairness gate가 실패하면 negative/reduced claim으로 기록한다.
7. 두 구조적으로 다른 global candidate가 같은 preregistered gate를 실패하면
   global scope를 더 넓히지 말고 재현 가능한 negative conclusion으로 종료한다.

---

## 12. 다음 에이전트에게 그대로 전달할 개선 목표 프롬프트

아래 블록을 새 에이전트의 목표 프롬프트로 그대로 사용한다.

> 이 저장소의 현재 연구 목표를 임의로 바꾸거나 request-local predictor 연구로 축소하지 말라. 먼저 `paper/TEMPO_RESEARCH_MASTER_STATE_AND_NEXT_GOAL.ko.md`, `paper/TEMPO_GLOBAL_ORCHESTRATOR_CANONICAL_PLAYBOOK.ko.md`, `paper/TEMPO_ELASTIC_PD_CONTENTION_AUDIT.md`, `eval/sota_4node/TEMPO_GO_GLOBAL_ORCHESTRATOR_GOAL_V1.ko.md`를 처음부터 끝까지 읽어라. v535만 보지 말고 conceptual v0, actual P/D v1–v450 source, v452–v544 raw artifact, C0–C5 evidence를 모두 계보로 취급하라. 현재 workspace에 없는 v545–v600을 추정해서 만들지 말고, 새 자료가 있으면 exact path/SHA로 추가하라.
>
> 원래 목표는 “TEMPO Elastic-PD를 실제 vLLM P/D 경로에 통합하고, 단순 predictor와 가장 강한 고정 정책보다 유의미하게 빠른 최종 스킴으로 확정”하는 것이다. 현재 결론은 route-only 후보를 버리고 orchestration을 버리는 것이 아니다. 실제 contention은 존재하고, C3에서 local/remote 승자가 뒤집히며, C5 native에서 official LMCache `CacheEngineKey ... not found in local data`/EngineDeadError/503가 관찰됐다. 따라서 새 목표는 Perlmutter native 4-node/16-A100, 실제 vLLM P/D, official `LMCacheConnectorV1:UCX`를 유지하면서 decoder admission, tenant SLO/fairness, pair assignment/proactive logical scaling, local/remote route, endpoint failure quarantine/recovery를 하나의 TEMPO-GO global orchestrator로 공동 제어하는 것이다.
>
> 최신 상태를 정확히 반영하라. `57402376` guarded five-arm discovery는
> local/predictor/remote의 clean raw, queue-GPU-only process-failure receipt,
> TEMPO global scheduler/completion provenance를 확보했지만 TEMPO goodput은
> always-local보다 낮고 847건 explicit reject가 있어 performance evidence가
> 아니다. `57404614` Candidate C native arm은 LMCache/EngineCore failure와
> `tempo-go-global-failure-v1` receipt 9건을 만들었지만 step exit `143`이라
> 독립 성능 결과가 없다. Candidate D는 C의 quarantine safety와 B의 proactive
> scaling을 결합한 CPU neutral candidate라 native에 올리지 않는다. 따라서 다음
> 단계는 같은 v3 trace를 반복하는 것이 아니라 `calibration_only` endpoint
> profile을 넘는 held-out workload와 frozen independent validation contract를
> 만드는 것이다.
>
> 고정 baseline은 `ALWAYS_LOCAL`, `OFFICIAL_LMCACHE_ALWAYS_REMOTE`, `PREDICTOR_ONLY`, `QUEUE_GPU_ONLY`(Kairos-like), `TEMPO_GO`다. same server lifecycle, same topology, same request trace, same cache namespace, counterbalanced order를 사용하라. 현재 v3 trace의 실제 geometry는 foreground `(512,16)/(2048,256)/(4094,16)`와 hot-stream output `2`이고, `--background-output-tokens 128`은 아직 row에 적용되지 않았음을 명시하라. 별도 held-out artifact `results/tempo_go_c5_heldout_output128_v1/`는 이미 만들었지만 실제 JSONL geometry와 SHA를 재검증하라. held-out은 `r02/r03`, hot output `128`, MISS 1,992 unique/P_ONLY 720이며 performance claim은 금지다. workload는 2 replicate의 C0 cool → C1 decoder-hot → C2 remote-hot → C2 P_ONLY/KV-hot → C3 both-hot → recovery, latency/interactive/batch/background tenant, explicit absolute arrival offset, cold MISS/P_ONLY evidence를 포함하라. phase label, future arrival, oracle route, physical switch label, synthetic network background를 policy input으로 넣지 말라. output=2 anchor와 output=128 held-out 모두 exact endpoint/frozen contract가 닫히기 전에는 final performance evidence가 아니다.

> frozen endpoint profile에 service row가 부족할 때 기존 `allow_service_proxy=True`를 임의로 켜지 말라. `FrozenServiceProxyPolicy`의 endpoint identity, calibration receipt SHA, allowlisted geometry/residency/lookup mode를 검증하고, proxy non-exact·numeric-unchanged·performance-forbidden flags를 유지하라. policy가 없는 frozen global profile은 fail-closed해야 하며, proxy policy는 GlobalOrchestrator capacity/config 입력이 아니다. exact MISS calibration 또는 policy source receipt가 없으면 native run contract를 만들지 말라.
>
> controller 입력은 scalar `fabric_pressure`가 아니다. pair별 decoder running/waiting/active sequences/KV usage, local prefill token-ms, remote P token-ms, remote KV bytes, remote semantic operations, endpoint request/queue/service, first-response completion residual, stale/partial identity, tenant queue/SLO/debt를 분리해서 보라. cross-host clock subtraction을 하지 말고 endpoint-owned duration을 사용하라. missing telemetry는 zero로 바꾸지 말고 fail-closed하라. local/remote/semantic resource credit은 독립적으로 bounded admission하라.
>
> 모든 request는 pair×route immutable request-start commit이어야 한다. prefill 시작 후 route 변경, hidden recompute, silent local fallback, same request ID retry를 금지하라. first response/EOF/complete/abort/timeout/failure에서 credit을 정확히 한 번 반환하라. failure는 latency로 치환하지 말고 `tempo-go-global-failure-v1` receipt, telemetry sequence, released work, quarantine scope, `new_request_id_required`, probe recovery를 기록하라. native EngineCore가 죽어 receipt가 없으면 success가 아니라 unreceipted execution failure로 남겨라.
>
> tenant fairness는 weighted request count가 아니다. weighted service debt와 raw service units를 분리하고 tenant별 TTFT/TPOT/E2E SLO, maximum queue wait, minimum service fraction, SLO-goodput, starvation, maximum wait, Jain fairness를 측정하라. aggregate completed 수만 보고 tenant starvation을 숨기지 말라. pair activation은 prewarmed pair의 logical active set만 바꾸며 이미 commit된 request를 migrate하지 말라.
>
> 순서는 (1) bounded source/test/evidence audit, (2) profile/manifest/fingerprint binding, (3) CPU replay와 failure injection, (4) native receipt closure, (5) fixed-path C1/C2/C3 validity, (6) approved interactive discovery, (7) frozen independent primary/robustness/fairness validation이다. discovery 결과를 보고 profile을 바꾸면 independent validation이 아니다. TEMPO-GO가 strongest fixed 대비 E2E 10% 또는 predictor 대비 5%, goodput 5%, paired/tail/fairness/correctness gate를 충족할 때만 median performance claim을 하라. median이 부족해도 overload p99/goodput 15% 개선과 정상부하 regression 3% 이내, failure/fairness/correctness 통과가 있을 때만 robustness claim을 하라.
>
> 두 구조적으로 다른 후보가 predictor 대비 개선하지 못하거나, remote가 실제로 유리한 workload에서도 5% 이득을 못 내면 threshold 미세조정을 반복하지 말고 simpler global admission/local-first/predictor-only로 축소하고 negative conclusion을 작성하라. LMCache transport 자체보다 빠르다, Mooncake보다 빠르다, 보편적 SOTA다, 모든 workload에서 항상 빠르다, 단일 allocation으로 production-ready라는 주장은 금지한다.
>
> Perlmutter에서는 login node에서 코드와 가벼운 검증만 수행하고, GPU workload는 승인된 native 4-node/16-GPU interactive allocation에서만 실행하라. 기존 4시간 allocation을 재사용하고 자동 Slurm submit/cancel/retry loop를 만들지 말라. container/root/udiRoot/sudo/system-file/ownership 변경은 절대 금지한다. `udiRoot.conf must be owned by user root` 또는 `exit 139`가 나오면 우회하지 말고 command/environment/node/log/exit evidence만 보존하라. dirty worktree의 unrelated 변경을 지우거나 stage하지 말고, 새 artifact와 SHA를 새 경로에 남겨라.

---

## 13. 최종 claim boundary

### 성공 시 허용

> 동일한 native 4-node 실제 vLLM P/D topology와 official LMCache data plane에서 TEMPO-GO의 causal global admission/orchestration이 fixed local/remote 및 predictor-only policy보다 낮은 latency, 높은 goodput, 또는 더 강한 multi-tenant overload robustness를 보였다.

### 현재 이미 허용

> Perlmutter native vLLM P/D에서 local과 official LMCache remote 경로의 서비스 상태가 coupled contention에 따라 교차하며, decoder·endpoint·transfer·tenant state를 함께 관찰해야 한다는 workload/mechanism evidence를 확보했다.

### 금지

- LMCache transport 자체보다 빠르다.
- Mooncake/P/D-Serve/Dynamo/Kairos보다 보편적으로 우월하다.
- interconnect의 특정 switch/link가 병목이라고 확정했다.
- 모든 workload에서 항상 빠르다.
- 실패한 native EngineCore를 receipt 없이 orchestrator가 복구했다고 말했다.
- 한 allocation 또는 한 favorable aggregate만으로 production-ready라고 말했다.

이 문서의 완료 조건은 코드 실행 성공이 아니다. frozen global controller가 독립 validation에서 correctness·fairness·failure·tail·goodput gate를 모두 통과하거나, 사전 정의된 negative/robustness 결론을 재현 가능하게 남길 때 비로소 해당 목표를 종료한다.

---

## 19. 최신 4-node realistic overload 결과와 다음 실행의 정확한 경계

2026-08-26 allocation `57631504`에서 phase-gated v2 campaign을 수행했다. 4-node/16
A100, actual vLLM P/D, official LMCache/NIXL-UCX, NCCL AWS Libfabric, 4-source→1-
receiver 512 MiB incast, 4,096-token NCCL burst를 사용했고, burst는 inference control
marker 뒤에 시작했다. NetKV/Dynamo/TEMPO 각각의 counterbalanced A/B arm과 동일한
210-request offered population을 사용했다.

| arm | completed / SLO-good | reject | failure | all-population e2e p99 |
|---|---:|---:|---:|---:|
| NetKV A | 29 / 29 | 176 | 5 | 6,095.6 ms |
| Dynamo A | 0 / 0 | 210 | 0 | null |
| TEMPO remote-only A | 181 / 178 | 29 | 0 | 8,159.7 ms |
| TEMPO remote-only B | 181 / 176 | 29 | 0 | 8,516.9 ms |
| Dynamo B | 0 / 0 | 210 | 0 | null |
| NetKV B | 3 / 1 | 207 | 0 | 54,674.8 ms |

각 arm의 official co-job은 correctness를 만족했고, background completion p99은
51.7--93.9초, NCCL token-tail p99은 baseline 약 5.3 ms와 TEMPO 약 36.7 ms로
관측됐다. 따라서 contention은 steady-state bandwidth가 아닌 receiver incast,
LMCache service time, decoder admission이 겹치는 microburst 형태로도 충분히 실재한다.
정책별 makespan과 노출된 burst block 수가 다르므로 co-job p99 자체를 정책 승리의
headline으로 사용하지 않으며, 이 run은 discovery evidence다.

중요한 actuator 결론은 기존 remote-only TEMPO가 normal/miss 요청은 176--178 SLO-good으로
보호했지만 remote-favorable 요청은 각 arm에서 1/30만 완료했다는 점이다. 이는
interconnect telemetry의 부재가 아니라 remote deadline infeasible 뒤 local reserve와
global queue timeout으로 병목이 이동한 것이다. 따라서 목표는 global orchestration을
버리는 것이 아니라 local/remote route-symmetric business lane으로 확장하는 것이다.

후속 `vllm_priority_business_dual_route_v2` ABBA는 같은 allocation에서 첫
`app_global_only A`만 완료했다(210/210 completed, 204 SLO-good, hot 148/150,
remote-favorable 30/30, all p99 31.4초). 다음 arm 전 quiescence probe의
`--cpu-bind=cores`가 parent attach step CPU cpuset과 충돌해 exit 192가 되었고, full
dual-route TEMPO arm은 아직 측정되지 않았다. 이 결과는 setup/performance claim이
아니며, fixed launcher는 `--cpu-bind=none` 및 allocation-level node-list 파싱으로
수정하고 새 source-bound v7/v4 contract와
C9/C8 18-test regression을 통과시켰다.

v2의 완전 reject arm을 verifier가 `null p99` 때문에 버리지 않도록
`eval/sota_4node/analyze_scale_paper_causal_sota_v2_posthoc.py`를 추가했고,
`results/tempo_scale_paper_causal_sota_v2_job_57631504/posthoc_analysis.json`을
생성했다. raw/result/phase receipt는 변경하지 않았다.

현재 완료된 것과 남은 것은 분리한다.

- 완료: actual carrier에서 severe contention/microburst 재현, phase-gated co-job
  correctness, NetKV/Dynamo/TEMPO offered-population collapse 비교, remote-only
  actuator dead-end 진단, launcher bug 수정 및 post-hoc 분석기.
- 미완료: dual-route TEMPO의 native ABBA, real-trace native preflight, 1/2/4-node
  capacity-normalized comparison, strongest fixed/simple predictor/NetKV/Dynamo/Kairos
  independent validation, fairness/goodput/tail headline, 논문 최종화.
- 다음 GO: 정상 filesystem/quota와 새 승인 `gpu_interactive` allocation에서 v7/v4
  dual-route ABBA를 한 번 실행하고, 결과를 본 뒤 source/profile/threshold를 바꾸지
  않은 independent validation으로 넘어간다. 이때에도 4-arm 전체가 완결되기 전에는
  dual-route 성능 주장을 하지 않는다.
