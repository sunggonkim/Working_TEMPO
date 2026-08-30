# TEMPO 글로벌 P/D 오케스트레이터: 전체 연구 계보, 증거, 설계 및 실행 플레이북

문서 상태: 이 저장소의 단일 기준 문서(single source of truth), canonical research handoff 및 새 프로젝트 preregistration 초안
기준일: 2026-08-22 (C5 v2 native-invalid evidence, v3 workload/profile/replay,
receipt-closure retry6, post-patch retry7 startup failure 및 retry9 LMCache
data-plane failure, Candidate C failure-injected CPU replay, native C
failure-quarantine receipt, Candidate D combined CPU neutral replay, C5 v10/v11
immutable contract와 normal/failure offline replay 반영)
대상 환경: NERSC Perlmutter, native 4 nodes / 16 A100 GPUs
대상 시스템: 실제 vLLM P/D + official `LMCacheConnectorV1:UCX`

## 0. 이 문서의 역할

이 문서는 TEMPO의 conceptual v0, 실제 P/D v1--v450, v452--v544 실행
계보, C0--C4 contention 연구, canonical Elastic-PD 음성 결과, 그리고 현재
TEMPO-GO 구현을 하나의 기준 문서로 합친다. 다음 작업자는 특정 최신 버전만
보고 목표를 다시 해석하지 말고 이 문서를 처음부터 끝까지 읽은 뒤 작업해야
한다. 아래에 링크된 다른 문서는 historical evidence와 상세 raw receipt이며,
목표·gate·안전계약이 충돌하면 이 문서가 우선한다.

이 문서가 고정하는 핵심 판단은 다음과 같다.

1. 실제 inference contention과 moving bottleneck은 이미 존재가 확인됐다.
2. local과 remote 중 어느 경로도 항상 우월하지 않다.
3. 기존 request-level route-only Elastic-PD는 정확한 선택을 일부 했지만,
   shared decoder tail과 tenant isolation을 제어하지 못해 최종 gate를
   통과하지 못했다.
4. 다음 연구는 route threshold를 더 조정하는 작업이 아니다. decoder
   admission, tenant SLO/fairness, pair assignment/activation, local/remote
   route와 endpoint recovery를 공동 제어하는 글로벌 오케스트레이터다.
5. 현재 TEMPO-GO 코드는 control-plane wiring, terminal overload semantics와
   lifecycle invariant를 구현했지만, production-style workload의 native 실행,
   실제 endpoint service sensing, proactive pair scaling, multi-resource
   fairness의 성능 검증은 아직 완료되지 않았다.

이 문서는 다음 기존 자료를 폐기하지 않고 요약·연결한다.

- 원래 Elastic-PD 목표: 사용자 첨부 문서 SHA-256
  `4f9650280307d6c352ada284b1fb7137e4f70c0189e6ad341104671cc1647a4a`
- 전체 contention 감사:
  [`TEMPO_ELASTIC_PD_CONTENTION_AUDIT.md`](TEMPO_ELASTIC_PD_CONTENTION_AUDIT.md)
- route-only terminal negative:
  [`tempo_pd_c4_negative_report_v1/`](tempo_pd_c4_negative_report_v1/)
- 현재 글로벌 목표 초안:
  [`../eval/sota_4node/TEMPO_GO_GLOBAL_ORCHESTRATOR_GOAL_V1.ko.md`](../eval/sota_4node/TEMPO_GO_GLOBAL_ORCHESTRATOR_GOAL_V1.ko.md)
- canonical Elastic-PD 결과:
  [`../eval/sota_4node/TEMPO_ELASTIC_PD_CANONICAL_README.ko.md`](../eval/sota_4node/TEMPO_ELASTIC_PD_CANONICAL_README.ko.md)

충돌 시 raw artifact와 SHA-bound analyzer를 사실의 기준으로 삼고, 이 문서는
그 사실에서 도출한 연구 방향의 기준으로 삼는다.

## 1. 최종 연구 목표

### 1.1 원래 목표와 그 결론

원래 질문은 고정된 4-node P/D deployment 안에서 request마다 decoder-local
prefill과 official LMCache remote prefill을 선택하여 strongest fixed path와
predictor-only보다 의미 있게 빠를 수 있는가였다.

그 질문은 C4 terminal screen에서 재현 가능한 음성 결론에 도달했다. 이는
TEMPO 전체 또는 orchestration 전체의 실패가 아니다. 다음의 좁은 범위가
실패한 것이다.

> decoder scheduler, pair 수, replica 배치와 data plane을 고정하고,
> request-start에서 local prefill과 unchanged remote prefill 중 하나만
> 고르는 route-only admission controller

Candidate B와 C는 양쪽 route를 실제로 사용했고 predictor보다 빨랐지만,
shared decoder의 TPOT와 worst-tail을 제어하지 못했다. phase oracle도 최종
tail gate를 통과하지 못했으므로 추가 threshold 또는 phase classifier는
같은 연구 질문의 해법이 아니다.

사용자가 처음 붙여넣은 목표 snapshot에는 다음 headline도 포함되어 있었다.

| Historical snapshot arm | E2E median |
|---|---:|
| Full TEMPO | 1629.016 ms |
| Official LMCache always-remote | 1835.751 ms |
| Always-local | 1737.351 ms |
| Predictor-only | 1641.057 ms |

그 snapshot은 TEMPO가 LMCache 대비 약 11.3% 개선했지만 predictor 대비 약
0.7%밖에 개선하지 못했고 48개 중 42개를 local로 보냈다는 점을 이미
명시했다. 이 수치는 원래 목표와 문제의식을 보존하기 위한 historical
context이지, 최종 성능 근거가 아니다. 이후 SHA-bound raw/canonical run12와
C4 terminal screen이 더 엄격한 paired/group/tail gate를 적용했으므로, 현재
문서의 사실 판정은 후속 artifact를 우선한다. 즉 원래 목표를 임의로 삭제한
것이 아니라, route-only 질문의 한계를 확인한 뒤 같은 목표를 global
orchestration으로 확장한 것이다.

### 1.2 새 글로벌 연구 질문

새 질문은 다음과 같다.

> 동일한 native 4-node 실제 vLLM P/D topology와 official
> `LMCacheConnectorV1:UCX` data plane에서, TEMPO-GO가 phase-changing
> multi-tenant contention을 causal하게 관찰하고 decoder admission,
> tenant SLO/fairness, P/D pair assignment/activation, local/remote route와
> endpoint recovery를 공동 제어하여 strongest fixed policy,
> profile-only predictor와 queue/GPU-only policy보다 latency, goodput,
> tail isolation 또는 overload robustness를 개선하는가?

이 연구의 novelty 후보는 “local과 remote를 바꿔 고른다”가 아니다. 그
아이디어와 production P/D orchestration 자체는 prior art에 이미 존재한다.
검증할 차이는 다음 조합이다.

- 동일한 두 실제 data path를 유지한다.
- decoder, P endpoint, remote KV bytes와 semantic operations를 서로 다른
  resource로 소유한다.
- endpoint completion residual과 receiver/install pressure를 queue/GPU
  signal과 분리한다.
- per-request admission과 slower pair activation을 하나의 causal global
  state 위에서 연결한다.
- tenant별 SLO goodput과 starvation을 성능과 함께 측정한다.

## 2. 절대 변경하지 않을 실행·안전 계약

### 2.1 Perlmutter native-only

- 모든 GPU/vLLM/LMCache/traffic 실험은 4-node, 16-GPU, 최대 4시간의
  interactive allocation 안에서만 실행한다.
- login node에서는 코드 편집, bounded static check, 작은 CPU test와
  artifact 분석만 수행한다.
- 이미 유효한 allocation이 있으면 새 allocation을 만들지 않고 재사용한다.
- 한 번에 allocation 하나만 사용하고 자동 submit/retry loop를 만들지
  않는다.
- 실행 전 반드시
  [`../NERSC_AGENT_SAFETY.md`](../NERSC_AGENT_SAFETY.md)와 repository
  `AGENTS.md`를 읽고 따른다.
- shared top-level filesystem을 recursive traversal하지 않는다. 모든 검색은
  workspace 또는 명시된 result root로 제한한다.

### 2.2 root, container와 udiRoot 금지

- `sudo`, `su`, root ownership 변경, `setcap`, `CAP_NET_ADMIN`을 사용하지
  않는다.
- `/etc`, `/usr`, `/opt`를 수정하지 않는다.
- Shifter, Apptainer, Podman, Docker, `--image`, UDI launcher를 사용하지
  않는다.
- `udiRoot.conf must be owned by user root`가 나타나면 native contract를
  위반한 launcher가 호출된 것이다. 즉시 중단하고 command, environment,
  stdout/stderr만 보존한다. ownership을 고치거나 우회하지 않는다.
- `exit 139`는 해당 child/launcher의 SIGSEGV이지 privilege 변경 허가가
  아니다.

### 2.3 data plane과 topology

- 모델은 frozen experiment에서 Qwen2.5-7B-Instruct를 사용한다.
- 4개 node를 두 개의 prewarmed TP4 P/D pair로 구성한다.
  - pair 0: node 0 prefill, node 1 decode
  - pair 1: node 2 prefill, node 3 decode
- transport는 official `LMCacheConnectorV1:UCX`로 고정한다.
- main comparison에서 transport, chunk protocol 또는 KV format을 바꾸지
  않는다.
- Mooncake, custom NIXL transport, token-level decode hook, global fence,
  busy polling과 synthetic CXI load는 headline path에 섞지 않는다.
- synthetic CXI는 필요하면 mechanism attribution ablation으로만 사용하고
  actual inference tenant 결과와 분리한다.

## 3. 용어와 resource model

### 3.1 경로

- `LOCAL`: 요청의 prefill을 해당 decoder에서 chunked prefill로 수행하고
  같은 decoder에서 decode한다.
- `REMOTE`: paired prefill endpoint에서 prefill한 뒤 official LMCache를
  통해 KV를 paired decoder로 전달하고 decode한다.
- 두 경로는 first-token 이후 같은 decoder resource를 공유한다. 독립된
  두 server로 모델링하면 안 된다.

### 3.2 cache residency

- `MISS`: P와 D 어느 쪽에도 재사용 가능한 cache가 확인되지 않음
- `P_ONLY`: P에만 source hit가 완료 이벤트로 확인됨
- `D_ONLY`: D에서 local reuse가 확인됨
- `BOTH`: 양쪽에 완료 이벤트 기반 residency가 확인됨
- `UNKNOWN`: 확인되지 않음. hit로 추정하지 않고 fail closed한다.

cache state는 profile이나 prompt 길이로 추정하지 않는다. seed completion,
source cached-token evidence, decoder prefix evidence와 EOF lifecycle로
확인한다.

### 3.3 글로벌 resource vector

각 pair와 각 candidate는 최소 다음 resource를 갖는다.

| Resource | 의미 | 권장 소유 기간 |
|---|---|---|
| `decode_tokens` | 요청이 decoder에 남기는 예상 decode work | route commit부터 HTTP EOF |
| `active_sequences` | decoder scheduler/KV slot 점유 | route commit부터 HTTP EOF |
| `endpoint_requests` | local prefill 또는 remote handoff endpoint slot | route commit부터 first response |
| `local_prefill_token_ms` | decoder-local prefill service work | route commit부터 first response |
| `remote_prefill_token_ms` | P endpoint prefill service work | route commit부터 first response |
| `remote_kv_bytes` | remote KV transfer byte work | route commit부터 first response |
| `remote_semantic_ops` | transfer/install operation work | route commit부터 first response |

`decode_tokens`는 requested maximum을 보수적으로 예약하고 EOF에서 실제
completion tokens로 회계 기록을 보정할 수 있다. admission capacity는 미래
출력 길이를 oracle로 알았다고 가정하지 않는다.

## 4. 전체 연구 계보에서 확인된 것

### 4.1 conceptual v0와 TEMPO-RD reset

checked-in `_v0` 파일은 없다. conceptual v0는 actual P/D 이전의 training,
checkpoint와 generic I/O interference 연구를 뜻한다.

| 세대 | 당시 아이디어 | 현재 남은 교훈 |
|---|---|---|
| initial--v1 | shared PCIe root를 가정한 phase-gated KV/checkpoint I/O | endpoint가 다른 숫자와 proxy를 섞으면 causal claim이 되지 않는다. |
| v2 | communication/I/O co-scheduling, burst monitor, service gain | end-to-end 변화만으로 physical resource 원인을 확정할 수 없다. |
| v3 | topology placement와 Slingshot traffic class control | topology/QoS 자체는 novelty가 아니며 live causal path가 필요하다. |
| v4/dynamic/v6 | sparse transfer, peer cache, look-ahead, per-rail routing, GPU doorbell | causal headroom 전에 복잡도를 늘리면 연구가 controller tuning으로 붕괴한다. |
| TEMPO-RD reset | resource-domain accounting과 checkpoint phase controller | matched optimized-open이 강했고 scheduler tail이 26--40% 회귀하여 중단됐다. |

가장 중요한 실험 원칙은 다음이다.

> matched fixed-path screen이 실제 crossover와 headroom을 먼저 증명하기
> 전에는 controller를 튜닝하지 않는다.

### 4.2 actual P/D v1--v450

bounded workspace inventory에는 약 1,149개의 경로와 1,171개의 numeric
version reference가 있고, code filename 기준 revision은 v1--v450까지다. 이후
v452--v544는 주로 result/launcher/campaign lineage로 기록되어 있다. bounded
repository search에서 v545--v600에 해당하는 별도 canonical controller 또는
실험 세대는 확인되지 않았다. 따라서 사용자가 말하는 v0--v600 계보를 읽을
때도 숫자 하나를 완성된 orchestrator 하나로 세지 않고, v0 conceptual history,
v1--v450 implementation, v452--v544 campaign lineage, v545 이후의 부재를
구분한다. node, launcher, analyzer, fixture, test와 result가 각각 revision을
소비하며, 동일한 아이디어의 반복과 correctness-only 수정이 많이 포함된다.

| Revision | 핵심 변화 | 살아남은 결론 |
|---|---|---|
| v1--v27 | 첫 실제 vLLM P/D, LMCache/NIXL screen, KV geometry | admission과 transport를 분리해야 한다. 첫 결과는 24/24 local이라 remote branch 증거가 아니었다. |
| v28--v60 | exact output, queue crossover, offered-rate regime, local credit | 32 req/s에서 16 local/8 remote가 두 fixed path를 모두 이겼다. remote spill이 D-local queue를 완화할 수 있다. |
| v61--v95 | one live epoch, balanced order, credit 6--9, shape threshold | mid-load remote rule은 44/48 remote로 가며 느려졌다. geometry-specific threshold는 일반화되지 않았다. |
| v96--v129 | request-interleaving, short/long output, heterogeneous workload | 30--60 ms drift 때문에 sequential arm block이 무효였다. state를 workload class별로 나눠도 threshold 증식은 해결책이 아니다. |
| v131--v186 | cache catalog, immutable affinity, warm/cold hybrid phase | cache residency는 first-class route constraint다. warm reuse gain은 실재하지만 좁다. |
| v190--v245 | saturation, tail-aware, 4K/output256 composition | aggregate gain은 있었지만 paired/group tail gate가 실패했다. pooled headline으로 실패를 숨기면 안 된다. |
| v248--v290 | mixed frontier, externally frozen offered-rate policy | LMCache rate56 failure는 availability evidence이며 성능 win으로 세면 안 된다. online state가 필요했다. |
| v291--v321 | online pair-local arrival regime, unique chunks | cache-key aliasing 수정은 correctness였다. strongest fixed/predictor 비교는 아직 약했다. |
| v322--v349 | burst, local credit, 25 ms microburst | short burst admission은 유효하지만 fixed threshold는 trace-specific이다. |
| v353--v430 | phase change, adaptive/latched, cap5/cap6 | cap5 worst +435.5 ms, cap6 +58.2 ms. 구조는 유용하지만 상수는 online service model이 아니었다. |
| v440--v450 | native NIXL, canonical four-arm Elastic-PD | one-way commit, exact cache isolation과 phase-correct release는 재사용한다. route-only 성능 목표는 통과하지 못했다. |

### 4.3 v449 positive component evidence와 canonical negative

v449에서는 42 local/6 remote로 official LMCache always-remote 대비 E2E
45/48을 이기고 paired median을 209.356 ms 줄였다. first-response credit
release와 bounded queue invariant도 통과했다.

하지만 predictor 대비 추가 이득은 약 0.7%뿐이었다. 이후 canonical
run12는 48/48 local, remote 0이었고 다음 결과를 냈다.

| Arm | E2E median | Route |
|---|---:|---|
| always local | 1667.684 ms | 48 local |
| always remote | 1816.462 ms | 48 remote |
| predictor | 1610.236 ms | 40 local/8 remote |
| full TEMPO | 1648.479 ms | 48 local |

full TEMPO는 best fixed보다 1.152% 빨랐지만 predictor보다 2.375% 느렸고
goodput도 best fixed보다 1.327% 낮았다. 따라서 v449의 LMCache 대비 win을
complex controller의 일반적 win으로 해석하지 않는다.

### 4.4 v452--v544 late campaign

v452--v544는 대부분 source revision보다 run-directory revision이다. profile,
pair load, local/remote credit, chunking, cache state, request priority,
token-ID forwarding, libfabric/CXI option, peer memory와 synthetic background를
바꿨다.

v492만 당시 gate를 한 번 통과했으나 v493 repeat와 v502 policy repeat에서
재현되지 않았다. v534--v544에서도 TEMPO는 fixed path 대비 median을 개선한
경우가 있었지만 predictor/goodput/tail을 동시에 만족하지 못했다.

v536 100% synthetic CXI background와 v538 no-background를 같은 allocation에서
비교하면 local median은 약 30 ms, remote median은 약 181 ms 악화됐다.
remote가 약 151 ms 더 크게 영향받았다는 contention 증거지만, sender,
switch link, receiving NIC, PCIe/host 또는 LMCache semantic bottleneck 중
어디가 원인인지는 확정하지 않는다.

### 4.5 C1/C2 opposite crossover

controller 작업 전에 actual inference tenant로 fixed-path crossover를
검증했다.

frozen v4 workload는 foreground/background 모두 4094 prompt, output 2의
unique-cold actual request를 사용했다. local capacity reference는 32/s,
remote reference는 6.8/s였고, 첫 유효 fraction 0.70에서 다음을 얻었다.

| State | Background | Required winner | Pooled gain | Paired gain |
|---|---:|---|---:|---:|
| C1 decoder-local hot | local pinned 22.4/s | remote | +19.21% | +17.53% |
| C2 remote P/KV/D hot | remote pinned 4.76/s | local | +73.02% | +72.54% |

두 replicate 8개 block에서 실제 request 1,868개가 모두 HTTP 200, exact route,
two-token output와 cache contract를 통과했다. 이 결과는 controller win이
아니라 workload validity 증거다.

### 4.6 endpoint mechanism과 P_ONLY service knee

C1에서 D endpoint의 visible queue mean은 0.007--0.008 ms였지만 local prefill
mean은 192--213 ms, inference mean은 378--418 ms였다. remote foreground를
보내면 D prefill은 101--104 ms, inference는 188--193 ms로 줄었다. waiting
gauge가 0이어도 continuous batching service inflation이 존재한다.

C2에서도 P/D queue mean은 약 0.01 ms였지만 remote foreground E2E는
483--491 ms, local은 138--139 ms였다. queue count만으로 remote critical
path를 설명할 수 없다.

P_ONLY attribution은 32개의 distinct 4094/2 prompt를 두 P pair에 16개씩
preseed하고 rate 4, 8, 12, 16, 24, 32/s를 측정했다.

| P_ONLY rate | Remote FG median | Local FG median | Achieved remote rate |
|---:|---:|---:|---:|
| 4/s | 449.6 ms | 141.9 ms | 3.86/s |
| 8/s | 656.3 ms | 142.8 ms | 7.64/s |
| 12/s | 1941.1 ms | 139.0 ms | 8.37/s |
| 16/s | 3720.4 ms | 146.8 ms | 8.90/s |
| 24/s | 5865.1 ms | 139.5 ms | 9.71/s |
| 32/s | 5417.3 ms | 142.8 ms | 9.73/s |

첫 2x inflation과 10% 이상 drain이 12/s에서 나타났고 achieved throughput은
약 9.7/s에 포화됐다. queue와 vLLM inference time 밖에 multi-second residual이
남는다. 허용되는 결론은 official LMCache retrieval/transfer/control/install/
receiver completion service ceiling이다. Slingshot switch bottleneck이라고
부르지 않는다.

### 4.7 C3 coupled ABBA

C3는 local-pinned decoder-hot tenant 22.4/s와 P_ONLY remote tenant
0/4/8/12/s를 동시에 실행했다. foreground는 4094/2 MISS, 2/s였다.

| P_ONLY rate | Local FG median | Remote FG median | Winner |
|---:|---:|---:|---|
| 0/s | 528.9 ms | 448.4 ms | remote |
| 4/s | 542.8 ms | 589.1 ms | local |
| 8/s | 661.7 ms | 655.3 ms | near tie |
| 12/s | 674.9 ms | 1832.0 ms | local |

1,944개 request가 모두 valid했다. 이후 within-rate order
`local, remote, remote, local` ABBA에서 rate0 remote win 18.44%, rate12
local win 66.14%가 두 replicate 모두 재현됐다.

이 결과가 최종 motivation이다.

> neither route is globally superior이고, instantaneous queue-only state는
> moving service bottleneck을 식별하지 못한다.

### 4.8 C4 route-only terminal screen

C4는 `cool -> C1 -> cold C2 -> P_ONLY C2 -> C3 -> recovery`의 frozen phase를
사용하고 다음 네 arm을 비교했다.

1. always local
2. always official LMCache remote
3. profile-only predictor
4. TEMPO candidate

세 구조적으로 다른 candidate가 모두 exact correctness를 통과했지만 최종
gate를 통과하지 못했다.

| Candidate | Mechanism | Median gain vs fixed | vs predictor | Goodput | Paired wins | TPOT p99 regression | Worst regression |
|---|---|---:|---:|---:|---:|---:|---:|
| A | instant scalar score | -2.92% | +3.48% | +10.17% | 68.89% | +44.53% | +2506.4 ms |
| B | pair-local active watermark epoch | +7.10% | +17.46% | +7.67% | 76.11% | +64.28% | +997.9 ms |
| C | local external service-credit epoch | +7.92% | +21.30% | +4.58% | 75.56% | +49.41% | +2278.7 ms |

Candidate C의 local 선택은 remote counterfactual보다 median 78.04%, remote
선택은 local보다 25.90% 빨랐다. route selection은 실제로 유용했다. 그러나
C1/C3 group과 TPOT/worst-tail이 실패했다.

hidden phase label을 쓰는 diagnostic oracle도 세 trace 모두 full gate를
통과하지 못했다. route classifier만 더 튜닝하는 연구는 여기서 중단한다.

## 5. 현재 TEMPO-GO 구현 상태

### 5.1 구현된 control plane

현재 workspace에는 다음 구현이 있다.

- `tempo/pd_global_orchestrator.py`
  - bounded global queue
  - pair×route candidate admission
  - resource capacity check
  - tenant virtual service
  - immutable route commit
  - first-response와 EOF의 분리된 release
  - pair active set과 idle scale-down
- `tempo/pd_global_telemetry.py`
  - all-pair atomic telemetry batch
  - frontend/router/profile/generation identity validation
  - stale, partial, mixed batch fail-closed
- `tempo/pd_global_agent.py`
  - request-triggered bounded single-flight refresh
  - background polling 없음
- `tempo/pd_global_candidates.py`
  - exact profile와 cache evidence에서 pair×route candidate 생성
  - `UNKNOWN` fail-closed
- `tempo/pd_global_coordinator.py`
  - queue/admit/first-response/complete/fail lifecycle 연결
- `eval/sota_4node/tempo_pd_elastic_frontend.py`
  - global commit, frontend ledger와 pair reservation
- `eval/sota_4node/tempo_pd_elastic_router.py`
  - commit 검증, endpoint state와 lifecycle telemetry

### 5.2 G0 native 증거

job `57384994`의 G0 manifest는 다음을 검증했다.

- native 4 nodes / 16 GPUs
- A100 SXM4 40 GB
- non-root UID
- vLLM `0.26.0+cu129`
- LMCache `0.1.dev1`
- torch `2.11.0+cu129`
- transport `LMCacheConnectorV1:UCX`
- privileged NIC control 없음

authoritative artifact는
`eval/sota_4node/results/tempo_go_g0/job-57384994-native-v1/manifest.json`이다.

### 5.3 현재 C5 smoke evidence

C5 discovery v9와 output128 run은 실제 vLLM/LMCache path에서 각각 16/16
valid request, 16/16 global commit, zero client error를 기록했다. 네 tenant
label이 각각 4개 request를 가졌고 cache evidence는 P_ONLY였다.

하지만 measured phase는 모두 pair0의 `LOCAL`이었다.

| Run | Output | Requests | Pair | Route | Max active seq | Max decode tokens |
|---|---:|---:|---|---|---:|---:|
| v9 | 16 | 16 | pair0 16/16 | local 16/16 | 7 | 112 |
| o128_v1 | 128 | 16 | pair0 16/16 | local 16/16 | 15 | 1920 |

pair capacity는 active sequences 16, decode tokens 4096이므로 pair activation이
발생하지 않았다. 24-row output128 workload는 생성됐지만 measured run은 아직
없다.

따라서 이 두 결과의 허용되는 주장은 다음뿐이다.

- native global wiring과 request lifecycle이 동작한다.
- all-pair telemetry sequence와 immutable commit이 실제 request에 연결됐다.
- performance, fairness, pair scaling 또는 remote branch superiority는 아직
  검증하지 않았다.

이후 CPU-only stop gate는 기존 v1 manifest를 덮어쓰지 않고 v2와 v3로 분리했다.
v2 manifest는 2,712 rows, 2 replicates이며 measured cache contract는 `MISS`
1,992건과 `P_ONLY` 720건이었다. 그러나 v2 source pool의 24개 prompt를 MISS
stream이 반복하여 explicit MISS namespace 계약을 위반했다. native retry3에서
router가 이를 정확히 거부했고, 2,712건 중 768건만 valid, 1,944건은 HTTP 502
(`explicit MISS request namespace was previously observed`)였다. 따라서 v2는
성능 결과가 아니라 workload-validity negative evidence다.

이 native 시도에서 발생한 앞선 실패도 원인별로 보존한다. 첫 runner는
`module reset` 뒤 `set -e`가 풀려 C4 Python overlay prepare artifact 없이 arm을
시작했다. retry1은 CPU-only overlay `srun`에 `--gpus=0`가 없어 allocation GRES를
잘못 상속했다. retry2는 generated warmup ID만 보고 P_ONLY row를 필터링하여
empty warmup을 만들었다. 이 세 문제는 root/container/udiRoot 문제가 아니라
launcher lifecycle 계약 문제였고, 각각 fail-fast, CPU-only GRES 명시, immutable
source workload override로 고쳤다. 그 뒤 retry3에서 처음으로 실제 native
workload validity 문제가 노출되었다. 즉 에러를 숨기지 않고 “launcher bug”와
“실험 workload bug”를 분리한 것이 v3로 넘어간 이유다.

이 문제를 숨기거나 prompt를 임의로 합성하지 않고, 기존 C4에서 검증된
token-preserving marker 방법으로 v3를 만들었다. source pool은 여전히 semantic
template/geometry의 근거이고, `MISS` row에만 first chunk token sequence를
보존하는 unique marker를 앞에 삽입하여 각 measured MISS prompt namespace를
유일하게 만들었다. `P_ONLY` row는 source prompt를 재사용하고 measurement 밖의
warm seed만 deduplicate한다.

v3 workload는 다음 gate를 통과했다.

- manifest:
  `results/tempo_go_c5_cpu_gate_20260821_anchor_v3_retry2/tempo_go_workload_manifest.json`
- manifest SHA:
  `849bb5cf284c60215d12165e409ac426adc6e5bba3427cda8932c7379fb819fd`
- validation workload SHA:
  `38224ae6e421a0950080951a963ff7d82af480edfa15220c9a45c5c2064ad2f5`
- 2,712 rows = MISS 1,992 (unique prompt 1,992) + P_ONLY 720
- explicit arrival monotonicity, six-phase/two-replicate count, tenant contract와
  native-client field gate 통과
- v3 anchor profile/provenance와 global profile이 manifest SHA 및
  elastic/endpoint fingerprint에 재결합됨
- CPU five-arm replay v3는 동일 trace, no phase/oracle input,
  terminal/no-error/no-queue/no-owned-resource-leak gate 통과

v3 profile/replay artifact는 다음이다.

- `results/tempo_go_c5_anchor_priors_c12_v3_retry1/real_tempo_go_profile_c12_anchor_v3.json`
  (global fingerprint `e30744d097ddf66095387e7478a48be88e89da6582c55ef84bfaa864a1f6f012`)
- `results/tempo_go_c5_anchor_priors_c12_v3_retry1/real_tempo_pd_endpoint_service_profile_c12_anchor_output2_calibration_v3.json`
  (endpoint fingerprint `f5e8a4d234638344f85c7db5970679b57710fa977d7f72856345055a52fe0f3`)
- `results/tempo_go_c5_anchor_priors_c12_v3_retry1/five_arm_replay_v3.json`
  (SHA `1bf2977119153e502c07cb0ad56b8a570876e7b8388b2431f6b16fb9a5f08378`)

v3 replay 역시 물리 GPU/fabric latency를 모델링하지 않으므로 성능 claim은
여전히 금지한다. 기존 allocation에서 v3 native discovery를 부분 수행했지만
세 arm(local/official-remote/predictor)만 유효 receipt를 만들었고, queue-GPU-only는
LMCache assertion으로 실패했다. TEMPO-GO는 retry4에서 exact endpoint lookup
bug로 중단됐고, retry5에서 admitted request의 실제 scheduler/endpoint receipt를
만들었지만 reject receipt gap으로 전체 coverage를 닫지 못했다. retry6에서는
TEMPO 부하 중 official LMCache의 `CacheEngineKey ... not found in local data`
assertion으로 vLLM EngineCore가 죽었다. 따라서 five-arm 비교의 다음 gate는
reject까지 포함한 유효 TEMPO-GO receipt와 동일 epoch의 scheduler/endpoint
telemetry를 확보하고, LMCache data-plane failure를 global pair-health/admission이
격리할 수 있는지 검증하는 것이다.

### 5.3.1 C5 v3 native partial discovery receipt

기존 allocation `57395883`에서 v3 workload의 native arm을 일부 실행했다.
이 결과는 아직 TEMPO-GO 완료 결과가 아니다. 같은 v3 manifest와 Qwen/vLLM/
official UCX path를 사용했으며, 세 arm은 2,712/2,712 valid, exact router
decision, output hash 일치였다.

| Native arm | Valid | Route | E2E p50 / p95 / p99 (ms) | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---:|---|---:|---:|---:|
| `ALWAYS_LOCAL` | 2712/2712 | local 2712 | 1192.6 / 5843.5 / 16997.1 | 335.4 | 169.8 |
| `OFFICIAL_LMCACHE_ALWAYS_REMOTE` | 2712/2712 | remote 2712 | 10934.6 / 13416.8 / 16357.4 | 10767.8 | 27.4 |
| `PREDICTOR_ONLY` | 2712/2712 | local 240, remote 2472 | 10588.0 / 12394.1 / 12942.8 | 10537.3 | 26.6 |

모든 세 arm에서 request별 output text SHA-256이 index 대응 기준 2,712/2,712
일치했다. raw SHA는 각각 `a4a43442ed0a2697c50cc503aa53f89c3b60e709f0f10e6be93466eedc1dd8e9`,
`6a45ae723efde74499e3adca59d77e4c8a1bd2bb3527b9d2a9ed6426a509e555`,
`bf9010b29d958e974cf067027baa3c587679bdfd05da7bdd778bcc69b4d89a51`이다.
각 arm의 result receipt는 동일 v3 manifest SHA, profile SHA와 native 4-node /
16-GPU / `LMCacheConnectorV1:UCX` identity를 기록한다.

tenant SLO는 아직 aggregate 승리로 해석하지 않지만 business signal은 이미
명확하다. frozen SLO contract 기준 SLO-good request 수는 local
`2053/2712` (75.7%), remote `115/2712` (4.2%), predictor `271/2712` (10.0%)였다.
Local arm도 `latency` tenant 34/96, `interactive` 52/96만 SLO-good이므로
decoder contention tail이 실제 business failure로 나타난다. 이는 aggregate
median 하나로 숨길 수 없는 이유다.

반면 `QUEUE_GPU_ONLY` arm은 성능값으로 기록할 raw를 만들지 못했다. node1
receiver에서 LMCache가 다음 assertion으로 vLLM EngineCore를 종료했다.

```text
현재 상태 우선순위: 아래의 historical retry 설명보다 다음 v3 native receipt를
우선한다. 현재 contract는
`results/tempo_go_c5_heldout_frozen_proxy_v3/native_run_contract.json` (file SHA
`c280a889e148069b2678c53dc3cdb738219e6c6a64f80b9594b220c7d2f4f3f4`, fingerprint
`1fd9ff9f894b916a855c9aa93adb66a4a1bc4e1d05107cb09e690f300d857b73`)이고, native
result root는 `results/tempo_go_c5_native_heldout_frozen_proxy_v3_job_57407675`다.
v3는 local/remote/predictor 각 2,712 complete, queue-GPU-only exit 143 failure,
TEMPO 904 complete/1,808 explicit global reject/0 failed를 기록했다. TEMPO의
request goodput 4.401/s와 output-token goodput 497.2/s는 local 7.912/979.2,
remote 9.628/1,191.4, predictor 7.879/975.0보다 낮다. 따라서 이 receipt로
performance claim을 하지 말라. valid conclusion은 native global control-plane과
telemetry/completion receipt가 실제로 발동했다는 것, 그리고 전역 2초 admission
timeout이 background를 과도하게 거절한 negative라는 것이다. 다음 작업은 route
threshold가 아니라 tenant-aware admission/defer/fairness budget candidate를
CPU에서 먼저 검증하는 것이다.

AssertionError: Key CacheEngineKey(... worker_id=1 ... chunk_hash=...) not found in local data.
vllm.v1.engine.exceptions.EngineDeadError
```

그 결과 frontend는 incomplete chunk/HTTP 500을 냈고 `raw.json`은 생성되지
않았다. node1 vLLM log SHA는
`e0473e91f7db5824286998a916f69fb3966af5aabaa87728efa17dba9771054d`다. 이것은
“queue/GPU-only fixed policy가 contention에서 안정적이다”라고 가정할 수 없다는
native failure evidence지만, TEMPO-GO의 승리로 세지 않는다. `TEMPO_GO` retry4는
`(4094,2,MISS)` exact endpoint-row lookup bug로 raw를 만들지 못했고, retry5는
2,712개 중 1,029 admitted/complete와 1,683 rejected-or-timeout attempts를
기록했지만 reject receipt gap으로 invalid했다. 둘 다 성능 근거가 아니다.

따라서 이 partial discovery에서 허용되는 결론은 다음까지다.

- v3 workload validity fix는 실제 native local/remote/predictor에서 통과했다.
- fixed local과 official remote는 같은 topology에서도 전혀 다른 TTFT/E2E
  regime를 만든다. contention과 moving bottleneck은 실존한다.
- predictor-only는 두 route를 실제로 사용했지만, 이 campaign에서는 global
  admission/fairness/pair scaling 우위를 증명하지 못한다.
- queue/GPU-only는 LMCache cache-state failure로 독립적으로 무너졌다.
- TEMPO-GO의 native 성능·robustness·fairness 우위는 아직 미결이며, 이 결과만으로
  완료 또는 production claim을 하지 않는다.

### 5.3.2 C5 v3 TEMPO retry6: LMCache data-plane failure receipt

현재 allocation `57400890`에서 counterbalanced five-arm runner가 TEMPO arm을
먼저 시작했다. warmup은 끝났지만 measured TEMPO arm은 2,712개 요청을 모두
terminal coverage로 닫지 못하고 vLLM EngineCore failure로 종료됐다. raw는
보존하며 성능표에 넣지 않는다.

| 항목 | 관측값 |
|---|---:|
| attempted requests | 2,712 |
| valid stream records | 833 |
| router decisions | 919 (complete 845, rejected 73, failed 1) |
| structured global reject body | 73 (`global_admission_queue_timeout`) |
| telemetry refresh failed | 1,783 |
| telemetry refresh timed out | 7 |
| upstream 502 | 4 |
| raw SHA-256 | `c6b01f5aa1f5c5df248ef4e42493d269e586cb5fb4302131fabc022f66801c8b` |

첫 fatal error는 node-3 vLLM log에서 2026-08-21 21:23:04에 발생했다.
공식 LMCache `pd_backend_async.py:get_blocking()`이 다음 key를 local data에서
찾지 못해 assertion을 냈고, vLLM EngineCore가 `EngineDeadError`로 종료됐다.

```text
AssertionError: Key CacheEngineKey(... world_size=4,
worker_id={0,1,2,3}, chunk_hash=7002223948977228521,
dtype=torch.bfloat16, ...) not found in local data.
RuntimeError: Worker failed with error 'Key CacheEngineKey ... not found in local data.'
vllm.v1.engine.exceptions.EngineDeadError
```

이것은 root 권한, container, `udiRoot.conf` 문제도 아니고 단순 HTTP latency도
아니다. P/D LMCache retrieve path의 per-worker key consistency 또는 transfer/
install ordering이 moving contention에서 깨져 data-plane process가 죽은 것이다.
현재 로그만으로 switch link가 원인이라고 단정하지 않으며, 확인된 표현은
“LMCache cache-key/data-plane failure under contention”이다. node-2에서는 같은
시각의 다른 P_ONLY request가 4,094 tokens를 성공적으로 retrieve/store했으므로
모든 remote request가 항상 실패한다고 일반화하지 않는다.

이 receipt가 global orchestrator에 주는 설계 요구는 명확하다. (1) pair별 API
health와 LMCache retrieve/install failure를 endpoint-health state로 승격하고,
(2) `/metrics` 또는 completion heartbeat가 끊긴 pair를 즉시 admission 대상에서
제외하며, (3) 남은 pair로의 logical reassignment 또는 명시적 tenant-aware
reject를 수행하고, (4) local recompute로 조용히 바꾸지 말고 failure provenance를
기록해야 한다. 다음 native retry는 동일 workload를 무작정 반복하는 것이 아니라
이 pair quarantine/health receipt가 실제로 닫히는지를 먼저 검증한다.

### 5.3.3 C5 v3 TEMPO retry9: 동일 data-plane failure의 재현

retry6 이후 pair quarantine patch를 반영한 새 result root에서 TEMPO arm만
다시 실행했다. overlay extraction과 native vLLM readiness는 모두 통과했고,
따라서 이번에는 실제 vLLM/LMCache data plane에 진입했다. warmup은 24/24
terminal이었지만 measured phase의 `raw.json`과 `result.json`은 생성되지 않았고,
runner는 native child failure를 감지해 종료했다. 이 결과도 성능 evidence가
아니다.

| 항목 | 관측값 |
|---|---:|
| result root | `results/tempo_go_c5_native_tempo_only_job_57400890_v3_retry9_quarantine` |
| warmup | 24/24 valid, `official_lmcache_remote_prefill` |
| measured raw/result | 없음; runner return code 143 |
| 첫 fatal error | node-3, 2026-08-21 21:46:39 |
| 요청 직전 상태 | P_ONLY request `...kv-remote-hot-001262`, LMCache hit 3840/4094, load 3840 |
| 실패 key | `chunk_hash=2816912063036934730`, `world_size=4`, worker 0/1/2/3 |
| node-3 vLLM log SHA-256 | `b09c0fe17ad1e4ddd0148b99c7ae30abae7c7bb8910de9558eeb4dc8e708473c` |
| warmup raw SHA-256 | `a0d36b21d113c4ea7f3b281377350b60490755ee6b89fcccfe1f0179d590c6bf2` |

retry9에서도 공식 LMCache `pd_backend_async.py:get_blocking()`의
`CacheEngineKey ... not found in local data` assertion이 네 TP worker에서 같은
chunk에 대해 발생했고, 이후 node-2 router의 telemetry 요청은 죽은 EngineCore에
연결하지 못해 500이 됐다. retry6과 chunk hash는 다르지만 failure shape는 같다.
따라서 retry6을 우연한 단일 key 오류라고 취급할 수 없으며, “realistic contention
중 official LMCache retrieve/install data-plane의 cache-key consistency failure가
발생할 수 있다”는 robustness 문제는 재현됐다. 다만 두 실행 모두 같은 native
code/workload/profile의 invalid run이므로 이것만으로 switch/NIC의 정확한 원인이나
TEMPO의 성능 우위를 주장하지 않는다.

이 증거는 현재 구현의 한계를 더 정확히 나눈다. pair quarantine은 data-plane이
죽은 뒤 새 admission을 막는 사후 안전장치이며, 이미 admission되어 vLLM 내부에서
remote retrieve 중인 요청을 살리지 못한다. 다음 구현은 (a) pair-level health,
quarantine, reassignment/reject ledger를 유지하면서 (b) remote semantic operation
및 cache-consistency risk를 사전에 제한하는 보수적 admission/guard를 추가해야
한다. 임의의 remote threshold를 낮추거나 local fallback으로 실패를 숨기지 말고,
remote operation lifecycle, per-worker key/install evidence, failure 후 surviving
pair의 실제 terminal receipt를 먼저 계측한다.

이 guard의 control-plane 구현은 이제 들어갔다. `GlobalOrchestratorConfig`와
fingerprinted global profile은 선택적
`remote_semantic_ops_safety_reserve`를 받으며, guarded discovery에서는 endpoint
profile의 semantic-operation window가 4일 때 1 slot을 보존하여 TEMPO-GO remote
admission limit을 3으로 둔다. 이 값은 route latency threshold나 fabric scalar가
아니라 data-plane operation의 마지막 slot을 무조건 사용하지 않기 위한 보수적
pre-admission safety reserve다. 예약량을 초과하는 remote candidate는 local로
몰래 바꾸지 않고 `remote_semantic_ops_admission_guard`와 binding resource를
decision/reject provenance에 남긴다. legacy profile의 기본값은 0으로 보존하며,
새 native run은 반드시 새 profile SHA를 사용해야 한다. 현재 이 동작은 global
orchestrator/profile CPU contract에서 검증됐지만, native LMCache failure를
quarantine 후 surviving pair/reject receipt까지 흡수했다는 증거는 아직 없다.

### 5.3.4 Guarded TEMPO single-arm receipt closure

retry6/retry9 이후 현재 승인 allocation `57400890`에서 동일 v3 workload를
guarded profile로 한 번 실행했다. 이것은 five-arm 성능 실험이 아니라 remote
data-plane pre-admission guard와 terminal ledger를 닫는 integration receipt다.

| 항목 | 관측값 |
|---|---:|
| result root | `results/tempo_go_c5_native_tempo_guard1_job_57400890_v1` |
| global profile | `f8163ff115a2478614afccf57b02a1c535c7dd4e2b3e54f47beda83d1ae3c2a0` |
| raw SHA-256 | `3e95bf0fd6bc1317079e1d3ca58dbf646de18cdffd0438c90ecf7fd6d8485364` |
| workload / manifest SHA | `38224ae6e421a0950080951a963ff7d82af480edfa15220c9a45c5c2064ad2f5` / `849bb5cf284c60215d12165e409ac426adc6e5bba3427cda8932c7379fb819fd` |
| requests / valid | 2,712 / 2,712 |
| complete / explicit global reject | 1,904 / 808 |
| route counts among completed | local 1,623 / remote 281 |
| pair activation | 1 |
| semantic-op guard candidate rejections | 328 |
| reject reasons | queue timeout 799 / telemetry refresh timeout 9 |
| exact terminal / performance claim | true / false |

이번 run에서는 node vLLM logs에 retry6/retry9의
`CacheEngineKey ... not found in local data`, `EngineDeadError`, SIGBUS 또는
SIGSEGV가 나타나지 않았다. 이는 guarded admission이 remote operation pressure를
실제로 제한하고 explicit reject/complete receipt를 닫았다는 integration evidence다.
하지만 guard를 한 번 적용한 single-arm 결과일 뿐이고, unguarded counterfactual이나
독립 allocation이 없으므로 LMCache failure의 원인을 해결했다고, 또는 TEMPO가
빠르다고 주장할 수 없다. pair quarantine 후 surviving-pair reassignment는
이번 run에서 fatal failure가 발생하지 않아 아직 native failure-triggered path로
검증되지 않았다. single-arm 결과는
`tempo-go-c5-native-single-arm-analysis-v1` analyzer output으로 보존한다.
receipt 생성 뒤 allocation `57400890` 자체는 제가 cancel하지 않았지만 Slurm
상태가 `FAILED`, `ExitCode=143`으로 종료됐다. 이 allocation 상태는 별도 실행
수명 문제로 기록하며, 이미 완전히 닫힌 guarded raw/analysis receipt의 validity와
혼동하지 않는다. 새 native campaign에는 새로 승인된 interactive allocation이
필요하다.

### 5.3.5 Guarded counterbalanced five-arm native discovery

single-arm receipt closure 뒤 사용자가 승인한 새 Perlmutter interactive
allocation `57402376`에서 guard profile을 주입한 counterbalanced five-arm
discovery를 끝냈다. 이것은 독립 validation이 아니라, 동일 v3 trace에서 global
admission/fairness/pair/route/telemetry bookkeeping이 실제 vLLM P/D와 official
`LMCacheConnectorV1:UCX` 경로에 동시에 연결되는지 보는 descriptive run이다.

| 항목 | 관측값 |
|---|---:|
| result root / allocation | `results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1` / `57402376` |
| allocation shape | native Perlmutter, 4 nodes, 16 A100, QOS `interactive`, 4 h |
| arm order / order SHA | `tempo → queue_gpu → predictor → remote → local` / `aee580c27e32f7491b2e2e7f4f41900eff073aae898199c2e88ce5c30326c37f` |
| profile / global fingerprint | guard profile file SHA `8082f4190d56016d7bac6abacbf659017a4fb20a50d1b474223cf9157c1fd3ec` / `f8163ff115a2478614afccf57b02a1c535c7dd4e2b3e54f47beda83d1ae3c2a0` |
| workload / manifest SHA | `38224ae6e421a0950080951a963ff7d82af480edfa15220c9a45c5c2064ad2f5` / `849bb5cf284c60215d12165e409ac426adc6e5bba3427cda8932c7379fb819fd` |
| analysis SHA | v2 `dbefc699ef7448e2f03a43d4a1e5f779ffbfa8cc47f68330d01a58d69289fb18`; scheduler-provenance 재분석 v3 `921ec4ad74dc28604bc65a65a734e8638817cf4d1b51d745a416064820cd350d` |
| analyzer gates | all five present, same request/workload, native 4-node/16-GPU/UCX true; TEMPO endpoint completion and global scheduler provenance closure true; `performance_claim_allowed=false` |

네 arm은 2,712/2,712 raw가 `all_streams_valid=true`,
`router_decisions_exact=true`, `terminal_contract_valid=true`로 닫혔다. raw SHA는
TEMPO `7ae7552a39e132c3e00a670e310fe04421fd44b5ebebc4b17dc3f880caeea87e`,
ALWAYS_LOCAL `05dfdf837ffe8c91113efff2766b8b7f1011f05a073ee57ada852f0b41c3e6aa`,
PREDICTOR_ONLY `323140b6041aecd9a983b7210540afa4b80258011bd9f8dac7fc25166c48f8b4`,
OFFICIAL_LMCACHE_ALWAYS_REMOTE
`781d1396dc058edde92a8d7d8dbefa1a2e48e234da5b38ae5fa6203860f629cd`다.
`QUEUE_GPU_ONLY`는 measured raw를 만들지 못하고
`results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1/queue_gpu/failure.json`에
`tempo-go-c5-native-arm-failure-v1`, `exit_code=143`, native 4-node/16-GPU,
UCX와 manifest/workload SHA를 남겼다. failure receipt SHA는
`8b56e53bb7e6b8ff975742c185810acfcd8b55f9f445cf11ab04fa4cda5e4c38`이며,
analyzer는 `queue_gpu_failure_receipted=true`지만
`queue_gpu_has_scheduler_observation=false`로 판정했다. 그러므로 이 arm의
실패를 latency 승리나 성능 숫자로 대체하지 않는다.

TEMPO의 request-start 필드 `vllm_load_decision_mode=disabled`는 adaptive
endpoint feedback이 동기식 request-start `/metrics`를 금지하기 때문에 의도된
값이다. 실제 vLLM scheduler observation은 global decision provenance의
allocation-scoped telemetry에 있으며, scheduler-provenance 재분석 v3에서 두
pair 모두 `router_local_vllm_prometheus_observe_only` snapshot을 2,712건씩,
총 5,424건 수집했고 invalid snapshot은 0건이었다. 따라서 TEMPO arm의 scheduler
telemetry closure는 true지만, performance claim은 여전히 금지된다.

이 failure는 단순 client 502가 아니다. queue arm의 node-3 vLLM log에는
LMCache `pd_backend_async.py:get_blocking()`의
`CacheEngineKey(... chunk_hash=2816912063036934730) not found in local data`
assertion과 `EngineDeadError`가 남았고, 뒤이어 receiver의 key allocation이
약 30초 timeout으로 반복됐다. 이는 기존 retry6/retry9와 같은 official
LMCache data-plane consistency/failure shape가 moving C3 contention에서
다시 나타난 것이다. 이 arm은 TEMPO가 이를 흡수했다는 증거가 아니라,
global orchestrator가 이 상태를 latency로 숨기지 않고 baseline failure로
분리한 증거다.

TEMPO arm 자체는 global profile SHA를 decision provenance에 남기고 2,712개를
모두 terminal 처리했다. 1,865 complete와 847 explicit global reject였으며,
완료 route는 local 1,686 / remote 179, pair activation은 1이었다. global
decision reason은 fair route commit 1,864, pair activation 1, queue timeout 839,
telemetry timeout 8이었다. rejected-candidate provenance에는
`remote_semantic_ops_admission_guard` 282건, capacity 434건, deadline 1건이
포함됐다. 이는 guard가 remote semantic window의 마지막 자리를 보수적으로
차단하고 rejection ledger를 닫았다는 증거지만, “부하를 잘 처리해 더 빠르다”는
증거는 아니다.

TEMPO의 tenant receipt는 background `1,626 complete / 810 reject / 1,579
SLO-good`, batch `84 / 0 / 81`, interactive `95 / 1 / 69`, latency `60 / 36 /
44`이고 starvation은 모두 false였다. global completed 기준 request goodput은
9.092 req/s, SLO-goodput은 8.644 req/s, output-token goodput은 136.865 tok/s,
E2E p50/p99는 983.9/14,485.0 ms였다. 비교 arm의 descriptive global request
goodput은 ALWAYS_LOCAL 13.137, PREDICTOR_ONLY 11.367,
OFFICIAL_LMCACHE_ALWAYS_REMOTE 10.734 req/s였지만, queue arm이 clean하지 않고
TEMPO가 847건을 명시적으로 거절했으므로 이 표는 paired performance claim이
아니다. single-arm과 마찬가지로 “guarded control-plane/data-plane integration
및 business reject가 실제로 발동했다”까지만 결론으로 삼는다.

따라서 이번 run은 다음을 닫았다. (a) v3 manifest의 동일 2,712-request trace,
(b) native P/D/UCX 경로의 four clean receipts, (c) tenant별 SLO/reject/
starvation accounting, (d) pair activation과 selected local/remote route, (e)
guard/queue/telemetry rejection provenance, (f) baseline failure를 성능으로
숨기지 않는 analyzer gate. 다음 stop/go는 queue-GPU-only를 clean하게 복구하거나
failure를 구조적으로 교체할 수 있는 별도 global candidate를 만들어, 두 candidate
모두에서 frozen independent validation을 수행하는 것이다. 현재 결과만으로
TEMPO-GO promotion, LMCache data-plane 원인 해결, 또는 production/HPC-scale
성능 우위를 선언하지 않는다.

### 5.3.6 Candidate B: queue/SLO-risk proactive pair scaling

guard profile의 `remote_semantic_ops_safety_reserve=1`은 data-plane safety
boundary로 유지하되, reject 비용과 pair activation timing을 개선할 구조적으로
다른 global mechanism을 CPU에서 먼저 만들었다. Candidate B는
`proactive_scale_up_queue_fraction`, `proactive_scale_up_wait_fraction`으로
현재 ingress queue occupancy와 tenant별 maximum queue-wait의 현재 경과비율을
읽고, trigger가 살아 있는 동안 active pair의 score에
`proactive_scale_up_active_pair_penalty_ms`를 적용한다. 이로써 fresh telemetry가
있는 prewarmed spare pair가 단순히 후보로만 남지 않고, observed queue/SLO risk가
있을 때 실제 pair×route assignment에서 선택될 수 있다. future arrival, workload
phase, oracle route, physical fabric label은 입력이 아니다.

Candidate B의 immutable discovery profile은
`results/tempo_go_c5_queue_scale_profile_v1/real_tempo_go_profile_c12_queue_scale_v3.json`이며,
file SHA는
`b705c4688a6061d3025a0e63a56a4edaa32e9384dee4edfbcb281fada5195b33`, global
fingerprint는 `507d415764ef2dde8661d3516c08fb51aca6643bdca4ce3e8a29e82559eb55f3`다.
설정은 semantic-op reserve `1`, queue fraction `0.25`, wait fraction `0.25`,
active-pair penalty `25 ms`이며, 같은 v3 manifest/elastic/endpoint prior에
bound된다. controller 구현과 profile round-trip을 포함한 bounded suite는
`123 passed`다.

동일 v3 2,712-request trace의 Candidate B CPU replay는
`results/tempo_go_c5_queue_scale_replay_20260821_v2.json`이며 file SHA는
`f0c74068a7aa5e187e49b44460a404e0a2c33a6623a02b4d4ada2fbd0f2b9ed2`다. five-arm
동일 trace, terminal/leak-free, no phase/physical-switch input gate는 모두
통과했고 performance claim은 금지된다. Candidate A guarded replay와 비교하면
B는 pair1을 index 5부터 사용하고 1,165건을 pair1에 배정했으며, A는 index 35부터
1,154건을 배정했다. 그러나 aggregate complete/reject는 B도 `2,433/279`로
동일하고, 따라서 이 replay는 “queue/SLO-risk가 pair activation을 앞당긴다”는
mechanism evidence이지 goodput/latency 개선 증거가 아니다. 이 결과만으로 B를
native frozen validation candidate로 승격하거나 allocation을 반복하지 않는다.

### 5.3.7 Candidate B-fairscale와 fairness accounting 교정

Candidate B의 mechanism을 더 일찍 발동시키는 `0.5/0.5` queue/SLO-risk
variant도 별도 immutable profile로 만들었다. profile은
`results/tempo_go_c5_fairscale_profile_v1/real_tempo_go_profile_c12_fairscale_v1.json`이며
file SHA는
`f0caa6d73a77b35235035ba3247f79dc6e66ade2a71295713f3fe75fc7f9ca95`, global
fingerprint는
`175f31f9db13a9ce5bd45aaf95574f38483a5ca36db651d6c4b050301df27e8e`다. 같은
v3 trace에서 pair1 activation은 baseline보다 앞당겨졌고, activation reason도
`global_proactive_queue_scale_tenant_queue_slo_risk_and_route_committed`로
receipt에 남았다. 그러나 baseline TEMPO-GO의 `2,461 complete/251 reject`에
비해 fairscale은 `2,433/279`였고, aggregate SLO-goodput도 `2,238`에서
`2,202`로 감소했다. 따라서 Candidate B-fairscale은 mechanism/invariant
evidence일 뿐 performance 또는 robustness 승리가 아니며 native로 승격하지
않는다.

이 과정에서 두 개의 control-plane accounting 결함도 교정했다.
`_tenant_virtual_service`는 이미 dominant-resource service를 tenant weight로
나눈 weighted debt인데 `_tenant_key`에서 weight로 한 번 더 나누고 있었고,
minimum-service fraction도 weighted debt를 raw work fraction처럼 사용하고
있었다. 이제 weighted debt와 raw dominant-resource service units를 분리해
weighted fair ordering과 minimum-service contract를 각각 올바른 단위로
계산한다. 또한 async coordinator는 전역 queue timeout만 사용하지 않고
`min(global maximum_queue_wait, tenant maximum_queue_wait)`를 admission wait
budget으로 사용한다. 이는 throughput knob가 아니라 tenant business contract를
실제 admission boundary에 연결하는 correctness fix다.

이 수정 후 baseline/fairscale five-arm replay receipt는 각각
`results/tempo_go_c5_fairness_fix_v1/five_arm_replay_fairness_fix_baseline_v1.json`
(SHA `3af04cb0962c617ab65f3daa39cdeab90faeb1d9c901d9d8ebda970fb6e6d366`)과
`results/tempo_go_c5_fairness_fix_v1/five_arm_replay_fairness_fix_fairscale_v1.json`
(SHA `9213563e5c2a5817b8a94e92e3a8e12f575741bf9fc12f01f7fd00488068a087`)이다.
두 receipt 모두 2,712 request 동일 trace, terminal/leak-free, no phase/physical
switch input gate를 통과했고 `performance_claim_allowed=false`다. 이 수정은
baseline 선택량을 과장해서 개선하지 않았으며, fairscale의 negative 결과도
재현됐다. 관련 bounded regression은 최신 broad run에서 `104 passed, 11
subtests passed`다.

### 5.3.8 Candidate C failure-injected CPU replay

Candidate C의 핵심 경로가 단순 unit test에만 머물지 않는지 확인하기 위해,
동일한 C5 v3 2,712-request trace를 다섯 arm에 재생하면서 TEMPO-GO arm의
workload index `0` request에만 명시적인 remote route failure를 주입했다. 주입
대상은 실제 replay admission에서 pair `0`의
`official_lmcache_remote_prefill`로 commit된 request이며, route가 local 또는
reject로 바뀌었으면 replay gate가 실패하도록 만들었다. 이 이벤트는 정상
first-response 직전에 발생하고, 같은 request의 normal first-response/EOF는
무효화된다. 따라서 silent local fallback이나 같은 request ID migration으로
failure를 숨기지 않는다.

| 항목 | 관측값 |
|---|---:|
| replay artifact / SHA | `results/tempo_go_c5_quarantine_replay_v2/five_arm_replay_remote_failure_index0_v1.json` / `2edd9f616fc94f4ee6e55e88e6a647b1cd55974f05ea130d5203f2f428270f21` |
| manifest / workload SHA | `849bb5cf284c60215d12165e409ac426adc6e5bba3427cda8932c7379fb819fd` / `38224ae6e421a0950080951a963ff7d82af480edfa15220c9a45c5c2064ad2f5` |
| C global profile SHA / fingerprint | `1e3c861593a5bd802fa73aa5199657a6ec410d28ee9ebf8aad6970538e127cb6` / `6657de67fa75bef8241ffe4148126fe5977fe73da0312b801d7a1843a38849eb` |
| injected target | index `0`, tenant `latency`, pair `0`, remote, phase `route_committed` |
| TEMPO-GO terminal result | `2,343 complete / 1 failed / 368 rejected` |
| failure receipt | 정확히 `1`, schema `tempo-go-global-failure-v1`, receipt SHA `3f4b0b773e63bb341b7d87a7dc71a6c743d677822f2e27af60020b7645fd0410` |
| quarantine | pair `0`의 remote route만 quarantine; failure 이후 같은 pair remote admission `0` |
| all-arm control gates | 동일 request 수, terminal/leak-free, no errors, no in-flight/queue, performance claim 금지 |

receipt는 `released_work`에 active sequence, decode token, endpoint request,
remote KV bytes와 semantic operation을 모두 남겼고,
`reassignment_policy=new_request_id_required`를 고정했다. 이미 queue된 다른
request는 surviving pair×route로 dispatch될 수 있지만, 실패한 request 자체는
재시도하지 않는다. 이 replay는 route quarantine과 release/ledger wiring을
실제 trace event 순서에서 확인한다. PROBE 이후 recovery와 pair-scope
quarantine은 기존 orchestrator focused test에서 별도로 확인한다.

이 결과는 CPU control-plane robustness evidence일 뿐이다. GPU, vLLM, LMCache,
interconnect 또는 latency/goodput을 측정하지 않았고, failure를 주입한 arm과
주입하지 않은 baseline의 complete 수를 성능 비교에 사용하지 않는다. native
Candidate C arm에서도 별도의 실제 failure receipt가 확보됐지만, 그 run은
step exit `143`으로 끝나 `result.json`이 없었다. 따라서 native C receipt는
failure robustness evidence이지 performance/production claim의 근거가 아니다.

### 5.3.9 Candidate D: failure safety와 proactive pair scaling 결합

Candidate D는 C의 `route_failure_quarantine_mode=deny_until_probe`와
`remote_semantic_ops_safety_reserve=1`에 B의 queue/SLO-risk proactive scaling
(`0.25/0.25`, active-pair penalty `25 ms`)을 결합한 별도 frozen profile이다.
목적은 pair health failure를 숨기지 않으면서 spare pair를 더 일찍 admission
대상으로 만들어 reject와 tenant wait를 줄이는지 확인하는 것이다. profile은
다음 immutable identity를 가진다.

| 항목 | 값 |
|---|---|
| profile | `results/tempo_go_c5_candidate_d_profile_v1/real_tempo_go_profile_c12_candidate_d_v1.json` |
| file SHA / fingerprint | `d8bb3e893fa3279e004e020c2dcf1e34bf7af46dd0ff1d4527863a49816f566d` / `75bc2b6f76bded31f1582aac46e2d3594afdf4c79714b80535afa6987848ab18` |
| no-failure replay | `five_arm_replay_candidate_d_v1.json`, SHA `b9567186c224a41a74bedf8744e0a797ba4a0c7838908574bb5c4e0dee9f97777b9` |
| failure-injected replay | `five_arm_replay_candidate_d_failure_index0_v1.json`, SHA `fa959472271982f9d6f6f48ab282922c4d799bf24a6236a13f5867e521c70b4e` |

동일한 2,712-request v3 trace에서 D는 모든 arm terminal/leak-free를 유지했다.
TEMPO-GO는 `2,433 complete / 279 reject`, pair activation `1`, local/remote
`1,958/475`, SLO-goodput과 E2E/TTFT/TPOT를 Candidate C와 동일하게 냈다.
차이는 pair assignment뿐이었다(C `1,275/1,158`, D `1,268/1,165`). 따라서
proactive pair activation 시점을 앞당기는 것만으로 aggregate business metric을
개선하지 못했다.

failure-injected D replay도 정확히 한 `tempo-go-global-failure-v1` receipt를
생성했고, 대상 remote route를 quarantine한 뒤 `2,343 complete / 1 failed /
368 reject`로 terminal/leak-free를 유지했다. 이 결과는 C의 failure semantics와
B의 scaling mechanism이 서로 충돌하지 않음을 보여주지만, D도 성능 승리나
independent validation 후보로 승격할 근거가 없다. Candidate D는 native로
실행하지 않고 CPU neutral/negative evidence로 고정한다.

### 5.4 현재 구현의 중요한 gap

이 절은 구현 전 상태가 아니라, 현재 partial implementation 이후에도 남아 있는
실제 stop gate를 적는다. 이미 닫힌 항목을 다시 gap으로 세지 않는다.

1. `pd_global_workload.py`와 C5 builder는 이제 C0/C1/C2/C3/recovery phase,
   explicit arrival, exact 512/2048/4094 source pool, 네 tenant label과 다섯
   policy arm metadata를 생성한다. C1/C2 actual raw를 이용한 output=2 anchor
   prior와 C5 anchor manifest는 CPU validation을 통과했다. v3 global profile은
   v3 manifest SHA에 rebind됐지만 endpoint profile scope는 아직
   `calibration_only`이므로 native independent validation용 final profile로
   승격되지 않았다. v2는 workload-validity 실패 때문에 native 결과로
   재사용하지 않는다.
2. tenant weight, TTFT/TPOT/E2E SLO, maximum queue wait, minimum service와
   `overload_action=reject_new_request`가 schema, decision provenance와
   terminal accounting에 들어갔다. queue capacity 초과는 예외가 아니라
   명시적 terminal reject가 되고, 남은 queued request도 end-of-trace에서
   queue-timeout reject로 정리된다. 이것은 CPU lifecycle closure이지 native
   business-SLO 성능 증거는 아니다.
3. global adapter는 `scheduler_observation_required=true`일 때 실제 vLLM
   scheduler running/waiting/KV usage snapshot과 endpoint completion
   first-response/residual을 누락 시 fail-closed한다. `57402376` TEMPO arm은
   allocation-scoped global provenance로 pair당 2,712건, 총 5,424건의
   scheduler observation과 invalid 0을 닫았고 endpoint completion receipt도
   닫았다. 남은 gap은 이 telemetry를 사용한 independent performance/fairness/
   scaling 승리이지 sensor provenance 자체가 아니다.
4. dominant-resource fairness와 inactive-pair proactive admission은 CPU
   invariant로 구현됐고, weighted debt/raw service-unit 분리와 tenant-aware
   queue budget도 교정됐다. 그러나 native pair activation이 SLO/queue/endpoint
   externality를 줄이는지는 아직 negative/independent gate 상태다. logical
   activation을 physical process migration이나 switch reconfiguration으로
   과장하지 않는다.
5. v3 native-discovery profile의 workload binding은 C5 v3 manifest
   `849bb5cf284c60215d12165e409ac426adc6e5bba3427cda8932c7379fb819fd`에
   묶여 있고 global identity/fingerprint, elastic fingerprint와 endpoint
   fingerprint가 서로 검증된다. 다만 endpoint profile scope는 여전히
   `calibration_only`다. 따라서 이 profile은 discovery용 calibration prior이지
   independent validation용 final service profile이 아니다. native endpoint
   completion receipt를 얻은 뒤에만 final profile freeze를 검토한다.
6. C1/C2 anchor prior로 다섯 arm deterministic replay v3를 실행했다. 5개 arm이
   동일한 2,712-request trace를 처리했고, 모든 request가 complete 또는
   explicit reject로 terminal 처리되었으며 error, inflight, owned-resource
   leak, residual queue가 0이었다. 그러나 TEMPO-GO는 2,461 complete/251
   reject, `QUEUE_GPU_ONLY`는 2,594 complete/118 reject였고, 이 결과는
   performance claim을 허용하지 않는다. overload rejection은 구현 gate를
   닫았지만 controller superiority를 증명하지 않는다.
7. 동일 native epoch의 `ALWAYS_LOCAL`, `OFFICIAL_LMCACHE_ALWAYS_REMOTE`,
   `PREDICTOR_ONLY`, `QUEUE_GPU_ONLY/Kairos-like`, `TEMPO_GO`를 새 phased
   workload에서 실제 실행하는
   `eval/sota_4node/run_tempo_go_c5_five_arm_in_allocation.sh`와 receipt-only
   analyzer가 구현됐다. 기존 C4 replay와 v2 retry3 native step은 workload
   contract 위반으로 1,944/2,712가 거부되어 무효 처리했다. v3 native partial
   discovery에서는 local/remote/predictor가 각각 2,712/2,712 valid였지만
   queue-GPU-only가 LMCache cache-key assertion으로 EngineCore를 잃었다.
   TEMPO-GO retry5에는 admitted request의 실제 scheduler/completion receipt가
   있었지만 reject receipt gap 때문에 전체 coverage가 닫히지 않았다. retry6은
   structured reject body를 client가 분류하지 못한 문제와 별개로, 실제
   LMCache `CacheEngineKey ... not found in local data` assertion으로 node-3
   EngineCore가 죽어 measured TEMPO arm 자체가 중단됐다. reject-receipt patch와
   route failure provenance, `remote_pre_admission_guard`, semantic-op reserve
   guard는 CPU tests를 통과했고, Candidate C native arm에서도 9건의 global
   failure receipt와 pair/route quarantine provenance를 실제로 만들었다. 다만
   그 arm은 step exit `143`으로 끝나 `result.json`이 없으므로 actual
   scheduler/completion receipt가 일부 존재해도 TEMPO-GO native
   performance/fairness/scaling claim은 금지한다. Candidate D의 combined
   profile은 same-trace replay와 failure-injected replay를 terminal/leak-free로
   통과했지만 aggregate metric이 C와 같아 native로 승격하지 않았다. 기존 C4
   replay와 v3 CPU replay 역시 native performance evidence가 아니다.
8. 따라서 현재 단계의 결론은 “global control-plane, terminal overload
   semantics, sensor contract, five-arm comparison path와 offline lifecycle
   gate를 구현했고, native retry5에서 overload가 실제 발동했으며 retry6에서
   official LMCache 경로의 cache-key/EngineCore failure가 contention 중
   발생함을 관측했고, 이를 막기 위한 guard/ledger 경로를 CPU와 native C
   failure receipt에서 확인했다. Candidate D는 C의 failure safety와 B의
   proactive scaling을 결합했지만 CPU aggregate neutral로 고정됐다.”까지다.
   contention 존재, opposite crossover와 route-only failure는 이미 증명됐지만
   TEMPO-GO의 native 성능·fairness·scaling 우위와 frozen independent validation은
   아직 증명되지 않았다.

9. native identity gap은 별도 C5 contract로 닫기 시작했다. `eval/sota_4node/tempo_go_c5_run_contract.py`가 workload/manifest, global/Elastic/endpoint profile, model config, controller/frontend/node-entry/runner/analyzer와 직접 import되는 lifecycle source SHA, arm order, node parameter, fixed environment를 하나의 immutable JSON으로 묶는다. five-arm runner와 node entry는 contract path/SHA를 필수로 받고 profile/manifest/code mismatch에서 fail-closed하며, native result/failure receipt와 analyzer도 contract fingerprint를 보존·검증한다. 현재 normal discovery contract는 `results/tempo_go_c5_frozen_contract_v10/native_run_contract.json` (file SHA `63d33edf83c5825ba9d1981e68f0ece761e739d6d1b977e610be6f947d3c065c`, fingerprint `d37b8330734a7479f48c8bd844cccbe91403f96d368918119197ddacb598a737`)이고, failure-injection replay는 quarantine-enabled profile을 묶은 v11 contract `results/tempo_go_c5_frozen_contract_v11/native_run_contract.json` (file SHA `7713f6414c34c6a6ef52f485e546b11086620bb83b87f1ccc2ccacc9facb6699`, fingerprint `76c6651ab8b673f78bc1173a08e66d885414050015236bd11b942572abc31728`)이다. v10 normal replay는 2,712-row five-arm에서 TEMPO-GO `2,433/279` complete/reject, all-arm terminal/leak-free를 통과했고, v11 failure replay는 remote failure receipt 1건과 pair-0 remote quarantine, all-arm terminal/leak-free를 통과했다. 두 contract 모두 guard/discovery/calibration scope라 independent validation 또는 performance claim을 허용하지 않는다. v1~v9는 이후 source/profile revision 때문에 stale이며 overwrite하지 않았다. failure injection은 `route_failure_quarantine_mode=deny_until_probe` profile에서만 허용하고, disabled profile에서는 replay를 즉시 fail-closed한다.

## 6. 지금까지 배운 설계 원칙

### 6.1 remote가 항상 나쁜 것이 아니다

low load에서는 transfer가 없는 local이 자연스럽게 유리하다. D-local
prefill이 뜨거우면 remote가 D queue externality를 줄일 수 있다. remote
completion path가 뜨거우면 local이 크게 유리하다. 따라서 route prior는
필요하지만 system state 없이 geometry만 보는 predictor는 moving crossover를
놓친다.

### 6.2 queue depth는 service pressure와 다르다

continuous batching은 visible FIFO waiting 없이 active service time을 늘린다.
remote는 P/D queue gauge가 0에 가까워도 transfer/control/install residual이
수초까지 늘어날 수 있다. `num_waiting == 0`을 idle 또는 healthy로 해석하지
않는다.

### 6.3 fabric이라는 단어를 정확히 사용한다

현재 증거는 endpoint-total Cassini counter, application completion time,
LMCache transfer completion과 vLLM metrics다. 이것으로 remote completion
path pressure는 확인할 수 있지만 switch link saturation을 확정할 수 없다.

정책에서는 다음을 분리한다.

- sender/P endpoint service
- remote KV byte work
- semantic transfer/install operation
- receiver/D endpoint service
- advisory endpoint/NIC fault counter

물리적 원인을 모르면 `remote_completion_pressure`로 기록하고 `fabric
congestion`이라고 단정하지 않는다.

### 6.4 shared decoder가 global control의 중심이다

LOCAL과 REMOTE 모두 decode 단계에서 같은 D에 합류한다. route decision은
새 요청의 TTFT만이 아니라 이미 decoding 중인 request의 TPOT을 바꾼다.
route-only oracle도 tail에 실패한 이유다.

따라서 decoder admission과 active-sequence/KV headroom을 route 선택과 같은
transaction에서 처리해야 한다.

### 6.5 correctness fix와 performance mechanism을 분리한다

request ID, seed, cache namespace, first-token geometry, schema, credit release와
shutdown fix는 필요하지만 논문 mechanism이 아니다. correctness를 먼저
통과시킨 뒤 동일한 code/profile로 performance screen을 해야 한다.

### 6.6 LMCache 실패를 숨기지 않는다

concurrent repeated prompt chunk에서 `CacheEngineKey ... not found`와 HTTP
failure가 관찰됐다. unique namespace로 correctness aliasing을 제거한 결과와
실제 production-like repeated-prefix 결과를 구분한다. transport failure를
느린 latency로 바꾸거나 silent local fallback으로 성공 처리하지 않는다.

LMCache를 폐기하는 것이 목표가 아니다. unchanged official data plane의
failure, throughput ceiling과 recovery를 global admission이 관리할 수 있는지
검증한다.

## 7. 개선된 글로벌 scheme

### 7.1 세 개의 제어 시간축

하나의 giant heuristic 대신 서로 다른 시간축을 분리한다.

1. per-request admission
   - tenant 선택
   - pair×route feasibility와 marginal cost 평가
   - immutable commit
2. sub-second endpoint health
   - first-response completion residual
   - failure history, `GOOD/SKIP/DENIED/PROBE`
   - bounded explicit recovery probe
3. seconds-scale logical pair activation
   - 두 pair는 계속 prewarmed한다.
   - controller가 admission 대상으로 사용할 active set만 바꾼다.
   - process를 request마다 시작/종료하지 않는다.

### 7.2 business/SLO contract

tenant label만 붙이지 말고 frozen profile에 다음을 선언한다.

| Field | 의미 |
|---|---|
| `tenant_id` | stable tenant identity |
| `weight` | overload 시 relative service entitlement |
| `ttft_slo_ms` | first-token SLO |
| `tpot_slo_ms` | decode isolation SLO |
| `e2e_slo_ms` | request completion SLO |
| `max_queue_wait_ms` | global queue에서 허용되는 최대 대기 |
| `minimum_service_fraction` | sustained backlog에서 보장할 최소 share |
| `overload_action` | queue, reject 또는 deadline-aware shed 중 frozen action |

권장 tenant 역할은 다음과 같다.

- `interactive`: 짧은 prompt/output, bursty arrival, 짧은 TTFT deadline
- `latency`: 긴 prompt와 P_ONLY cache, remote utility를 검증하는 deadline
- `batch`: 긴 output, 높은 offered load, output-token goodput 중심
- `background`: 낮은 weight, 지속 backlog, starvation 검증

현재 weight `4/2/1/0.5`는 discovery seed일 뿐이다. validation weight와 SLO는
fixed-arm characterization 후, controller 결과를 보기 전에 freeze한다.

### 7.3 causal global state

frontend의 request-triggered bounded agent가 같은 interval 안에서 모든 pair를
수집한다. 각 field는 다음 provenance 중 하나를 반드시 갖는다.

- `measured`: endpoint/application metric에서 직접 관찰
- `owned`: global controller가 commit 후 보유한 reservation
- `derived`: prompt/output geometry로 계산
- `profile_prior`: frozen calibration prior
- `missing`: 관찰 불가. 0으로 대체하지 않음

`observed + owned`를 무조건 합치지 않는다. 동일 request가 양쪽에 포함될 수
있으므로 resource별 ownership contract에 따라 de-duplicate하고 현재 구조처럼
필요한 경우 `max(observed, owned)`를 사용한다.

batch는 sequence, sampled/collected interval, agent epoch, profile fingerprint,
controller generation과 pair identity가 모두 일치해야 한다. stale, partial,
out-of-order 또는 mixed-generation이면 전체 decision을 queue/deny한다.

노드 간 monotonic timestamp를 빼지 않는다. 각 endpoint가 자기 clock으로
측정한 queue/service/transfer duration을 보고하고 global controller는 최신
snapshot과 completion residual을 사용한다.

### 7.4 실제로 수집할 상태

#### Decoder endpoint

- running/waiting request 수
- scheduled/active tokens와 current batch size
- KV cache usage
- request start-to-first-token local service stretch
- decode TPOT/step stretch의 bounded quantile
- frontend-owned active sequences와 decode tokens

vLLM metric을 synchronous per-request critical path에서 매번 읽지 않는다.
request-triggered single-flight snapshot과 completion event를 사용하고 collection
overhead를 별도로 측정한다.

#### Prefill/LMCache endpoint

- P running/waiting request와 prefill service duration
- source cached tokens와 actual source-hit validation
- remote transfer requested/completed bytes
- transfer enqueue-to-complete duration
- semantic operations in flight
- first-response service stretch
- retry, response timeout, transfer failure와 recovery-probe 결과

#### NIC/endpoint advisory

- 지원되는 경우 traffic class별 sent/received, ECN, RX/TX pause
- posted/non-posted blocked ratio
- receive priority/overflow match
- resource NACK, retry, response timeout

이 값은 route를 직접 선택하는 scalar로 collapse하지 않는다. unsupported
counter는 missing으로 기록한다. root 또는 privileged access를 요구하지 않는다.

### 7.5 global admission과 fairness

decision 순서는 다음처럼 분리한다.

1. backlog tenant 중 deadline urgency, age와 weighted dominant service를
   이용해 다음 tenant를 선택한다.
2. 해당 request의 모든 pair×route candidate를 capacity와 health로 거른다.
3. feasible candidate의 predicted own latency뿐 아니라 shared decoder와
   endpoint에 추가하는 marginal externality를 계산한다.
4. 가장 낮은 global cost candidate를 immutable commit한다.

tenant fairness는 decode token virtual service 하나만 쓰지 않는다. tenant가
소비한 resource vector의 dominant normalized share를 추적한다. 다만 deadline을
지키기 위한 temporary borrowing은 허용하고, completion 후 debt를 bounded하게
상환한다. starvation emergency는 명시적이고 기록 가능해야 한다.

개념적 candidate cost는 다음과 같다.

```text
C(pair, local, r) = static_local_prior(r)
                  + D_service_price(pair)
                  + local_prefill_externality(pair, r)
                  + tenant_debt(r)
                  + uncertainty

C(pair, remote, r) = static_remote_prior(r)
                   + P_service_price(pair)
                   + remote_byte_price(pair, bytes_r)
                   + remote_semantic_price(pair, ops_r)
                   + D_receive/decode_price(pair)
                   + tenant_debt(r)
                   + uncertainty
```

이 식은 additive physical latency decomposition 주장이 아니다. causal
admission score이며 각 price는 endpoint-owned service completion과 resource
utilization으로 bounded update한다.

### 7.6 pair assignment와 activation

현재처럼 active pair가 hard capacity를 넘을 때까지 spare를 무시하지 않는다.
inactive pair를 고려하는 조건은 profile에 freeze한다.

- active pair dominant utilization이 threshold를 넘을 것으로 예측됨
- oldest feasible queue wait가 tenant deadline budget을 침범함
- route-specific endpoint가 `SKIP/DENIED`이며 spare의 해당 경로가 healthy함
- active pair TPOT/service stretch가 frozen threshold를 넘음

activation은 새 request admission에만 적용한다. 기존 request의 pair/route는
변하지 않는다. scale down은 해당 pair의 owned/observed resource가 모두 0이고
idle hysteresis가 지난 뒤에만 한다. activation/deactivation flap count와
activation benefit을 결과에 보고한다.

### 7.7 lifecycle와 failure semantics

request는 다음 상태를 정확히 한 번씩 지난다.

```text
ARRIVED -> QUEUED -> ADMITTED/ROUTE_COMMITTED
        -> FIRST_RESPONSE -> HTTP_EOF -> COMPLETE
```

실패는 어느 단계에서든 `FAILED` terminal로 간다.

- route commit 전에만 pair/route를 변경할 수 있다.
- first response에서 endpoint/prefill/remote credit을 반환한다.
- HTTP EOF에서 active-sequence/decode credit을 반환한다.
- timeout/abort/failure는 보유 credit을 정확히 한 번 반환한다.
- mid-request route migration, hidden recompute와 silent fallback은 금지한다.
- failure 후 재시도는 새 request ID와 새 admission event로 명시한다.

## 8. workload를 설정하는 방법

### 8.1 원칙: fixed-path characterization이 먼저다

새 model/topology/profile에서는 controller를 먼저 돌리지 않는다.

1. always-local과 always-remote의 개별 capacity ladder를 측정한다.
2. 동일 geometry, duration과 request namespace를 사용한다.
3. local pressure와 remote pressure를 독립된 actual inference tenant로 만든다.
4. C1에서 remote win, C2에서 local win이 preregistered margin 이상인지
   확인한다.
5. crossover가 없으면 workload를 invalid로 판정하고 controller threshold를
   조정하지 않는다.
6. 첫 valid crossover fraction을 freeze하고 더 높은 fraction을 사후 탐색하지
   않는다.

### 8.2 재사용할 C4 anchor

새 C5는 다음 proven anchor를 버리지 않는다.

- C1 local-pinned background: 4094/2 unique-cold, 22.4 req/s
- C2 cold remote-pinned background: 4094/2, 4.76 req/s
- P_ONLY remote ladder: 4, 8, 12, 16, 24, 32 req/s
- frozen remote knee: 12 req/s
- C3: local 22.4 req/s + P_ONLY remote 0/4/8/12 req/s
- foreground control: 4094/2 MISS, 2 req/s
- P_ONLY pool: 32 prompts, pair당 16개, measurement 밖에서 preseed
- ABBA order: `local, remote, remote, local`

이 anchor는 Qwen2.5-7B, 해당 vLLM/LMCache build와 topology에만 유효하다.
software/model이 바뀌면 짧은 fixed-path confirmation을 먼저 수행한다.

### 8.3 production-style C5 phase matrix

headline validation은 최소 다음 phase를 하나의 frozen manifest에 포함한다.
phase label은 analyzer에만 전달하고 policy input으로 쓰지 않는다.

| Phase | Independent tenant load | 검증 목적 |
|---|---|---|
| C0 cool | foreground + low background | normal-load regression과 locality |
| C1 D-local hot | local-pinned long-prompt background | remote escape와 D service sensing |
| C2 cold remote hot | remote-pinned MISS background | P compute + transfer + receiver pressure |
| C2 P_ONLY hot | preseeded remote tenant near 12/s knee | transfer/control/install service ceiling |
| C3 both hot | C1 + P_ONLY remote tenant | moving lesser bottleneck과 shared decoder tail |
| asymmetric pair | pair0에 한 pressure, pair1 cool 또는 다른 pressure | global pair assignment/activation |
| recovery | background를 제거하되 foreground 지속 | denied path probe, hysteresis와 scale-down |

phase duration은 최소한 steady sample과 drain/recovery를 모두 관측할 수 있어야
한다. 기존 C4는 15초 phase, 2초 cooldown, 2 repetitions, foreground 2/s와
max workers 128을 사용했다. 새 duration을 바꾸면 fixed-arm characterization에서
먼저 freeze한다.

### 8.4 foreground geometry와 cache matrix

최소 다음 geometry를 교차한다.

| Class | Prompt | Output | Cache | 주 목적 |
|---|---:|---:|---|---|
| interactive-short | 512 | 16 또는 64 | MISS/D_ONLY | short deadline, decoder burst |
| latency-warm | 2048 또는 4094 | 16 또는 64 | P_ONLY | remote utility와 transfer pressure |
| batch-decode | 512 또는 2048 | 128 또는 256 | MISS | shared decoder TPOT/goodput |
| long-cold | 4094 | 128 | MISS | local prefill externality와 admission |
| reuse-control | 512/2048 | 64/128 | D_ONLY/BOTH | local/cache affinity correctness |

C4 manifest의 foreground anchor `(512,16)`, `(2048,256)`, `(4094,16)`도
counterfactual continuity를 위해 유지할 수 있다. exact profile row가 없는
geometry는 discovery proxy로 final decision하지 않고 fail closed한다.

### 8.5 tenant arrival와 business semantics

각 tenant는 서로 다른 process 또는 명시적으로 독립된 stream으로 arrival을
만든다.

- `interactive`: bounded microburst와 idle gap 반복
- `latency`: stable low-rate P_ONLY long prompts
- `batch`: sustained backlog 또는 bounded overload
- `background`: 전체 phase 동안 지속되는 낮은-weight backlog

단순히 request ID를 round-robin으로 바꾸는 것은 multi-tenant workload가
아니다. 각 tenant가 독립된 arrival schedule, deadline, geometry/cache
distribution과 service objective를 가져야 한다.

stable -> burst -> overload -> recovery 순서는 frozen arrival offsets로
manifest에 기록한다. policy에는 미래 offset이나 phase 이름을 전달하지 않는다.

### 8.6 prompt와 cache namespace

- arm, replicate, tenant, phase와 item을 포함한 globally unique request ID를
  사용한다.
- source pool의 semantic/template 의미와 exact tokenizer geometry는 유지한다.
  MISS stream에서 accidental LMCache chunk aliasing을 막기 위한 unique first
  chunk가 필요하면 C4에서 검증된 token-preserving marker만 사용하고,
  production repeated-prefix variant는 별도 workload로 보존한다.
- baseline arm끼리 cache를 오염시키지 않도록 namespace를 분리한다.
- P_ONLY는 source hit를 먼저 만들고 seed 완료를 measurement 밖에서 검증한다.
- seed와 measured request는 exact prompt geometry와 지원되는 output contract를
  사용한다.
- official proxy first-token protocol의 `N` 대 `N+1` geometry를 analyzer가
  명시적으로 검증한다.
- D_ONLY/BOTH는 실제 decoder completion/prefix evidence 없이는 선언하지 않는다.
- warmup, seed, reset과 measured phase를 artifact에서 분리한다.
- native five-arm warmup은 P_ONLY source-hit row만 사용하고, measured MISS row는
  warmup에서 제외한다. 각 arm은 `epd-{arm}-...` request namespace를 사용하며
  measured workload의 cache contract marker를 보존한다.
- 현재 C5 v3 builder는 각 measured MISS row에 대해 source prompt의 뒤쪽 token
  geometry를 보존하면서 검증된 marker를 first chunk에 삽입한다. 따라서 source
  pool count가 24개라는 사실과 measured MISS namespace가 1,992개라는 사실은
  모순되지 않는다. validator는 MISS prompt 문자열의 uniqueness와 manifest의
  `miss_unique_prompt_count`를 모두 확인한다. 이 방법은 P_ONLY source-hit
  reuse를 바꾸지 않으며, 임의 문장을 붙여 token 수만 맞추는 방식이 아니다.
- `TEMPO_GO_C5_ARM`은 runner와 node entry에서 동일하게 검증한다. fixed arm은
  endpoint feedback를 끄고, `QUEUE_GPU_ONLY`만 vLLM scheduler
  `observe_only` snapshot을 사용할 수 있으며, TEMPO-GO만 adaptive endpoint
  feedback와 semantic completion telemetry를 사용한다. 이 차이를 receipt에
  기록하지 못하면 arm 비교는 무효다.

### 8.7 pair-scaling workload

pair scaling은 request 수만 17개로 늘리는 시험으로 끝내지 않는다.

1. pair0만 active 대상으로 시작하되 pair1은 prewarmed한다.
2. phase마다 pair0의 decode work, local prefill 또는 remote endpoint 중 하나를
   threshold 위로 올린다.
3. pair1에 동일 cache evidence를 준비한 경우와 준비하지 않은 경우를 분리한다.
4. activation 전후 queue wait, p95/p99, throughput과 cache-locality cost를
   측정한다.
5. load가 내려간 recovery에서 idle hysteresis와 no-flap을 검증한다.
6. already committed request가 pair 이동하지 않았는지 검사한다.

### 8.8 fairness workload

fairness screen은 sustained backlog가 있어야 한다.

- background와 batch는 전체 phase 동안 backlog를 유지한다.
- interactive와 latency는 burst/deadline traffic을 추가한다.
- 각 tenant의 admitted/completed request, output tokens, queue wait와 SLO
  goodput을 기록한다.
- high-weight tenant의 우선권과 low-weight tenant의 no-starvation을 동시에
  검사한다.
- aggregate Jain index 하나로 deadline violation을 숨기지 않는다.

### 8.9 arm과 order

필수 deployable arm은 다음 다섯 개다.

1. `ALWAYS_LOCAL`
2. `OFFICIAL_LMCACHE_ALWAYS_REMOTE`
3. `PREDICTOR_ONLY`
4. `QUEUE_GPU_ONLY` 또는 Kairos-like load-aware policy
5. `TEMPO_GO`

기존 pair-local TEMPO를 ablation으로 추가할 수 있다. phase oracle은 diagnostic
upper bound일 뿐 baseline performance claim이 아니다.

한 live server epoch 안에서 counterbalanced order를 사용한다. 기존 C4 order
`[local, remote, predictor, tempo]`와 reverse replicate를 확장하여 새 다섯
arm의 Latin-square 또는 preregistered balanced order를 사용한다. arm별
background request set, phase duration, cache preparation과 topology는 동일해야
한다.

strongest fixed는 validation 결과를 보고 phase별로 사후 선택하지 않는다.
discovery fixed-arm evidence로 하나의 fixed policy를 freeze하거나, local과
remote 두 fixed arm을 모두 headline table에 유지한다.

### 8.10 현재 builder를 사용할 때의 정확한 설정

현재 구현된 builder의 기본값은 다음과 같다. 이 값들은 임의의 “부하를 세게
주기 위한” 숫자가 아니라 C1/C2/C3의 기존 actual-inference anchor를 다시
만들기 위한 시작점이다.

| 항목 | 현재 기본값 | 의미와 주의점 |
|---|---:|---|
| phase duration | 15,000 ms | steady state와 first-response/EOF를 함께 관측하기 위한 값 |
| cooldown | 2,000 ms | phase 사이 drain과 epoch 경계; policy input이 아님 |
| foreground | 2 req/s | 512/16, 2048/256, 4094/16 cycle |
| decoder-hot | 22.4 req/s | C1 local decoder pressure anchor |
| remote-hot | 4.76 req/s | cold remote P/KV/D pressure anchor |
| P_ONLY-hot | 12 req/s | 기존 remote service knee 근처의 별도 stream |
| hot anchor output | 2 | C1/C2 proven rate와 직접 비교하는 고정값 |
| production output | 128 | 별도 lower-rate production workload에서만 사용 |
| workers | 128 | explicit arrival workload에서는 client `--request-rate`를 제거 |

중요한 재현 조건은 다음과 같다.

1. `--source-512`, `--source-2048`, `--source-4094`는 기존 local JSONL
   source pool이어야 하며 builder가 local tokenizer로 exact token count를
   다시 확인한다. semantic source를 새로 합성해 길이만 맞추지 않는다. 다만
   measured MISS namespace에 한해서는 위의 proven token-preserving marker를
   builder가 적용한다.
2. historical v3 builder는 foreground `(512,16)`, `(2048,256)`, `(4094,16)`을
   cycle하고 hot stream은 `4094/anchor_output_tokens`를 사용한다. 기본 anchor
   값은 2이며 v3 hot row에는 output=128이 없다. output=128을 22.4/4.76/12
   req/s에 그대로 붙이지 않는다. 그 조합은 C1/C2 evidence가 아니며 queue
   overflow만 재현할 수 있다. 현재 held-out builder는 같은 phase structure를
   별도 `r02/r03` artifact로 만들고 hot row의 실제 `max_tokens`를 128로
   rewrite한다. 생성 후 JSONL counter와 SHA를 다시 읽어 확인한다.
3. C2 `P_ONLY`는 prompt source hit와 seed completion을 measurement 밖에서
   확인해야 한다. 단순히 phase 이름을 붙이거나 prompt가 길다는 이유로
   `P_ONLY`라고 선언하지 않는다.
4. manifest에는 phase/rate label이 남지만 controller에는 phase name, future
   arrival, oracle route를 전달하지 않는다. phase metadata가 policy input으로
   흘러가면 workload validity gate를 실패시킨다.
5. validator가 measured MISS prompt uniqueness를 통과하고, `validation.jsonl`의
   sidecar SHA가 global profile identity의
   `workload_manifest_sha256`와 일치하지 않으면 native launcher는 실행하지
   않는다. 새 manifest를 만든 뒤 profile을 먼저 재생성하고, 이전 C4 prior를
   자동 재사용하지 않는다.

## 9. telemetry와 데이터 계약

### 9.1 per-request ledger

각 request는 최소 다음 field를 남긴다.

- request/tenant/workload/replicate identity
- prompt/output geometry와 cache contract
- arrival, queue enter/leave와 deadline
- telemetry sequence, interval과 provenance
- tenant service/debt before/after
- candidate별 pair, route, predicted cost, uncertainty와 rejection reason
- selected pair/route와 immutable commit SHA
- resource acquire/release event와 ownership scope
- upstream/prefill/transfer start/end
- actual transfer bytes와 semantic operations
- first token, last token, HTTP EOF
- TTFT, TPOT, ITL, E2E와 queue wait
- vLLM running/waiting/KV snapshot
- endpoint completion stretch와 health transition
- output token proof, text/token digest
- retry, timeout, transfer failure, fallback와 terminal reason

### 9.2 aggregate

다음 dimension별 p50/p95/p99와 goodput을 낸다.

- arm
- phase/workload group
- tenant
- pair
- route
- cache residency
- prompt/output geometry

추가로 다음을 필수 보고한다.

- per-tenant SLO goodput와 maximum wait
- starvation count와 weighted service share
- pair activation/deactivation count, latency와 benefit
- resource별 dominant utilization
- endpoint health/probe/denial/recovery count
- stale/partial/mixed telemetry count
- telemetry collection span, timeout과 admission CPU p50/p99
- selected-route counterfactual
- correctness/failure ledger

### 9.3 physical claim boundary

application endpoint timing과 Cassini endpoint-total counter를 additive physical
decomposition으로 합치지 않는다. 다음 세 수준을 구분한다.

1. `observed end-to-end service inflation`: 직접 주장 가능
2. `endpoint/receiver completion pressure`: endpoint evidence가 일관될 때 가능
3. `physical switch/fabric link bottleneck`: route-specific causal intervention과
   counter가 없으면 주장 금지

## 10. 평가 gate

### 10.1 correctness

모든 performance 분석 전에 다음이 100% 통과해야 한다.

- stream/output/token digest 일치
- route provenance와 commit SHA 일치
- hidden recompute/silent fallback 0
- cache-state contract 위반 0
- transfer error/timeout을 성공 latency로 처리한 건수 0
- terminal queue/request 0
- credit underflow/leak/double release 0
- pair/route mid-request mutation 0
- tenant identity 또는 workload schedule mismatch 0

### 10.2 workload validity

- C1에서 remote, C2에서 local이 frozen fixed-path margin 이상 이긴다.
- C3에서 두 route가 다른 state에서 실제로 유용하다.
- fixed arm 간 background achieved load가 policy 실행 전에 동등하다.
- overload에서도 foreground와 background가 측정 가능하다.
- failure는 latency가 아니라 failure로 집계한다.
- phase/order/cache artifact가 모든 arm에서 동일하다.

### 10.3 primary performance

기존 원래 gate를 유지한다.

- strongest fixed 대비 pooled E2E median 10% 이상 개선
- predictor 대비 E2E median 5% 이상 개선
- strongest fixed 대비 request 또는 output-token goodput 5% 이상 개선
- paired E2E win 전체 75% 이상
- 각 workload group paired win 60% 이상
- 각 group E2E p99와 TPOT p99 regression 5% 이내
- worst paired E2E regression 100 ms 이내
- selected local/remote가 반대 counterfactual보다 각각 median 5% 이상 유리

### 10.4 robustness alternative

median gate를 못 넘더라도 다음을 모두 만족하면 overload robustness claim을
검토한다.

- overload fatal failure/terminal queue를 제거
- overload p99 또는 goodput을 15% 이상 개선
- normal-load median regression 3% 이내
- 모든 tenant starvation 0
- per-tenant SLO goodput과 tail isolation이 fixed/predictor보다 개선

### 10.5 fairness, scaling과 overhead

정확한 numeric budget은 discovery 전에 profile에 freeze해야 한다. 최소 gate는
다음이다.

- starvation 0
- tenant queue wait가 각 frozen `max_queue_wait_ms` 안에 있음
- continuously backlogged tenant가 frozen minimum service fraction을 받음
- pair activation이 preregistered overload phase에서 발생하고 recovery에서
  bounded time 안에 안정화됨
- activation/deactivation flap 0 또는 frozen allowance 이내
- telemetry/admission overhead가 별도 보고되고 normal-load 3% regression
  budget 안에 포함됨

## 11. 실험 실행 순서

### G0. Native capability와 identity

- 4 nodes / 16 GPUs / non-root 확인
- model, vLLM, LMCache, torch와 transport SHA/version 기록
- no container/udiRoot/privileged NIC 확인
- topology와 pair identity receipt 생성

### G1. Sensor closure

- 각 resource field를 measured/owned/derived/prior/missing으로 분류
- all-pair atomic state와 freshness/failure test
- actual vLLM scheduler와 endpoint completion signal 연결
- overhead 측정
- physical switch claim 금지선을 analyzer에 encode

### G2. Fixed-path workload confirmation

- C1/C2 anchor를 동일 build에서 짧게 확인
- P_ONLY knee와 C3 rate0/rate12 opposite winner 확인
- pair-asymmetric load와 tenant backlog를 추가
- workload validity가 실패하면 controller를 실행하지 않음

### G3. Offline replay와 CPU invariants

- 기존 C4 raw와 새 fixed-arm trace를 replay
- predictor, queue/GPU-only와 TEMPO-GO를 동일 trace에서 비교
- fairness, pair activation, stale telemetry, failure/probe invariant test
- 구조적으로 두 candidate만 freeze하고 threshold family를 증식하지 않음

### G4. One-allocation discovery

- 한 번의 native 4-node/4-hour allocation
- 같은 server epoch와 counterbalanced arm
- 모든 raw artifact와 SHA 저장
- correctness/workload gate가 실패하면 performance 해석 중단
- 원인 수정은 다음 새 result root에서만 수행

### G5. Freeze

- controller code SHA
- global profile와 tenant/SLO profile SHA
- workload/arrival/cache manifest SHA
- analyzer와 gate SHA
- exact native launcher command

### G6. Independent validation

- 새 allocation에서 frozen code/profile/manifest를 수정하지 않고 한 번 실행
- 결과를 본 후 tuning/재컴파일 금지
- 실패하면 negative 또는 reduced claim으로 종료

### G7. Report

- raw per-request JSONL, aggregate JSON와 SHA manifest
- arm/phase/tenant/pair/route별 표와 plot
- counterfactual, fairness, scaling과 overhead
- failed ablation과 negative evidence
- 허용/금지 claim

## 12. 중단 및 축소 규칙

- route-only threshold, prompt coefficient 또는 phase classifier는 C4 negative로
  이미 중단됐다. 다시 시작하지 않는다.
- 실제 endpoint service와 decoder admission을 포함하지 않은 candidate는 새
  global candidate로 세지 않는다.
- 두 구조적으로 다른 global candidate가 predictor/queue-GPU-only보다 primary
  또는 robustness gate를 모두 실패하면 threshold search를 중단한다.
- pair scaling이 activated pair의 cache/locality cost보다 이득을 내지 못하면
  fixed two-pair admission으로 축소한다.
- endpoint completion feedback이 queue/GPU-only ablation보다 의미 있는
  incremental gain을 못 내면 NIC/fabric-aware claim을 철회한다.
- fairness가 aggregate latency를 개선하면서도 한 tenant를 굶기면 실패다.
- independent validation 실패를 discovery result로 대체하지 않는다.
- allocation 잔여 시간이 teardown/artifact 수집 30분보다 적으면 새 GPU
  candidate를 시작하지 않는다.

## 13. 재사용할 것과 폐기할 것

### 그대로 재사용

- actual vLLM P/D와 official LMCache proxy lifecycle
- one-way route commit
- fail-closed cache residency
- pair-local immutable affinity
- first-response endpoint credit release와 EOF decoder release
- bounded queue와 exact terminal ownership
- request-triggered all-pair telemetry identity/freshness contract
- C1/C2/C3 workload anchor와 ABBA evidence
- raw stream/output/route/cache validation

### 수정 후 재사용

- static endpoint profile: prior로만 사용하고 online completion으로 보정
- global resource vector: actual endpoint provenance 추가
- tenant virtual service: multi-resource dominant share와 deadline debt로 확장
- active pair set: proactive threshold와 SLO pressure로 확장
- C5 wrapper: `build_tempo_go_c5_manifest.py`와 explicit-arrival validator를
  phased path로 사용한다. 기존 `build_tempo_go_c5_workload.py`는 16-request
  smoke용으로만 남긴다.

### 폐기 또는 diagnostic으로만 유지

- router-local scalar `fabric_pressure`
- fixed prompt-token penalty와 request priority exception family
- hidden phase-label classifier
- sequential arm block의 작은 latency 차이
- LMCache failure를 TEMPO success로 세는 분석
- unsupported NIC counter를 0 pressure로 해석
- 16-request/8-rps single-geometry C5를 performance workload로 해석

## 14. 현재 즉시 해야 할 구현 작업

1. v2 C5 anchor manifest는 historical invalid evidence로 보존하고, v3 C5
   anchor manifest와 validator/replay receipt를 입력 SHA까지 고정한다. source
   pool/marker contract가 바뀔 때만 새 manifest를 만든다.
2. 기존 C4 anchor를 v3 manifest의 C1/C2/C3에 직접 binding하고 SHA를 남긴다.
   v3 profile은 manifest/elastic/endpoint identity를 검증하도록 생성됐다.
3. C1/C2 anchor prior와 C5 manifest의 provenance를 보존하고, native run에서
   endpoint service profile을 `calibration_only`로 명시적으로 유지한다. native
   completion receipt 없이는 final profile로 승격하지 않는다.
4. `overload_action=reject_new_request`의 native admission/reject/fairness
   behavior를 측정한다. CPU replay에서 닫힌 terminal lifecycle을 다시
   implementation task로 되돌리지 않는다.
   - native retry4는 endpoint service profile에 없는 `(prompt=4094,
     output=2, MISS)` exact row를 직접 조회해 `ValueError`로 중단했다. 이는
     profile builder가 의도한 bounded P_ONLY geometry proxy를 router가
     사용하지 않은 integration bug였고, `external_credit_proxy()`와 lookup
     provenance를 canonical router에 반영했다.
   - retry5는 그 수정 뒤 실제 4-node vLLM/LMCache 경로에서 2,712 requests를
     보냈다. 1,029 admitted request에는 global decision, scheduler
     observe-only snapshot, endpoint completion receipt가 남았지만, 1,677은
     `global admission queue timed out`, 6은 telemetry refresh timeout으로
     503이 됐다. 당시 client/frontend가 reject를 terminal decision ledger에
     남기지 않아 `router_decisions_exact=false`가 됐다.
   - 따라서 queue timeout을 예외로 버리지 않고 `GlobalDecisionKind.REJECT`
     로 반환하고, frontend `/tempo/decisions`에 `phase=rejected`, reason,
     decision SHA를 기록하며, client는 명시적 reject receipt와 request ID가
     매칭될 때만 503을 정상 terminal event로 인정하도록 수정했다. 이 수정은
     CPU 회귀 테스트로 검증했다. 이후 별도 receipt-closure retry6 native run을
     수행했지만 telemetry-timeout receipt가 닫히지 않았고, post-patch retry7은
     workload 진입 전 vLLM child startup SIGBUS로 종료됐다. 따라서 이 native
     runs 어느 것도 성능 evidence가 아니다.
   - 별도 receipt-closure run
     `results/tempo_go_native_receipt_closure_57400890_retry6`은 같은 v3
     workload로 2,712 requests를 시도해 2,049 complete와 653 queue reject를
     만들었다. 그러나 telemetry refresh timeout 10건은 당시 coordinator가
     `submit` 전에 예외로만 반환해 decision ledger에서 빠졌고,
     `router_decisions_exact=false`인 raw가 남았다. 이 raw는
     `09add8b7ac40920c6ea938f7e8d173c7ddadc9047704362bf6379e00af61b8c9`로
     보존하며 성능 evidence로 해석하지 않는다.
   - 이 gap을 닫기 위해 telemetry refresh timeout/failed/validation failure를
     `GlobalOrchestrator.reject_unadmitted()`의 명시적 `REJECT`로 변환하고,
     기존 frontend rejection recorder가 decision SHA와 request ID를 남기도록
     수정했다. CPU global/orchestrator/receipt suite가 통과했지만 native에서
     이 수정의 exact coverage는 아직 재확인하지 않았다.
   - 추가로 arm wrapper가 request-ID namespace를 바꾼 client workload SHA와
     canonical manifest workload SHA를 다르게 기록하는 receipt identity gap을
     발견했다. node receipt는 두 SHA를 모두 기록하고 analyzer는 source와
     arm-rewritten workload를 각각 검증하도록 수정했다. 이 또한 성능 결과가
     아니라 artifact closure이며, 기존 raw를 overwrite하지 않는다.
   - runner는 이제 queue-GPU-only뿐 아니라 모든 native arm의 startup/process
     failure에 `failure.json`을 남기고, analyzer가 이를 zero-request execution
     failure로 판정할 수 있게 한다. retry7 TEMPO SIGBUS receipt가 이 경로를
     수동으로 보강한 첫 artifact이며, 다음 run부터는 runner가 직접 생성한다.
5. decoder telemetry를 frontend reservation과 actual vLLM scheduler snapshot으로
   분리한다.
6. remote bytes/ops credit와 observed completion residual의 provenance를 분리한다.
7. tenant fairness를 decode token 하나가 아닌 dominant resource service로
   확장한다.
8. inactive pair를 predicted pressure/SLO miss 전에 고려하도록 pair activation을
   고친다.
9. 구현된
   `run_tempo_go_c5_five_arm_in_allocation.sh`와
   `analyze_tempo_go_c5_five_arm.py`를 사용해 fixed/predictor/queue-GPU/TEMPO-GO
   다섯 arm을 같은 epoch, cache preparation과 counterbalanced order로 고정한다.
   analyzer가 native 4-node/16-GPU, official UCX, raw SHA, scheduler observation,
   TEMPO endpoint completion receipt와 explicit reject terminal receipt를 모두
   확인하기 전에는 성능표를 만들지 않는다. 명시적 reject와 scheduler
   starvation을 같은 failure로 합치지 않고, global admission wait도 별도
   provenance로 집계한다.
10. fairness, pair activation, overhead와 failure를 primary report에 포함한다.
11. v3 CPU replay와 static runner/analyzer checks가 통과하기 전에는 새 4-node
   performance run을 시작하지 않는다. 이 gate는 통과했고 v2 native-invalid
   retry와 retry4/retry5/retry6/retry9 invalid TEMPO receipts는 보존한다. retry6과
   retry9에서 확인된 LMCache cache-key failure를 pair-health/quarantine receipt와
   CPU-verified remote pre-admission guard로 닫았다. 이제 native에서 guard 전후
   route failure provenance, surviving-pair assignment 또는 tenant-aware reject와
   structured reject가 request ledger와 매칭되는 것을 확인하기 전에는 performance
   run으로 승격하지 않는다. 현재 v3 validator/replay는 2,712 request,
   MISS 1,992/P_ONLY 720, all-arm terminal·leak-free, no phase/oracle policy
   input과 profile binding 일치를 재확인했으며, performance claim은 여전히
   false다.

12. retry6/retry9의 repeated LMCache failure를 반영한 guarded discovery profile은
    기존 `real_tempo_go_profile_c12_anchor_v3.json`을 덮어쓰지 않고 별도 result
    root에 생성한다. builder의
    `--remote-semantic-ops-safety-reserve 1`을 사용하고, 생성 직후 profile
    fingerprint, endpoint fingerprint, manifest SHA와 controller limit을
    출력·검증한다. guard rejection은 native receipt에서 request ID, pair,
    route, observed remote semantic ops, configured limit, reason과 함께 닫혀야
    하며, remote failure 뒤에는 pair quarantine/reassignment 또는 명시적
    tenant-aware reject 중 하나가 terminal ledger에 남아야 한다. 이 receipt
    closure 전에는 native performance arm을 재시도하지 않는다. five-arm
    launcher는 `TEMPO_GO_C5_RUN_CONTRACT`와 SHA를 먼저 검증하고,
    `TEMPO_GO_GLOBAL_PROFILE`, `TEMPO_GO_ELASTIC_PROFILE_PATH`,
    `TEMPO_GO_ENDPOINT_PROFILE_PATH`의 inherited override를 거부한 뒤 contract
    binding에서만 값을 읽는다. guarded profile을 legacy default로 덮어쓰면 안 된다.

24-row output128 workload를 단순 실행하는 것은 integration 확인에는 쓸 수
있지만 위 작업을 대신하지 않는다.

## 15. 주장 경계

### 성공 시 허용 가능한 주장

> 동일한 native 4-node 실제 vLLM P/D topology와 official LMCache data
> plane에서, TEMPO-GO의 causal global admission/orchestration이 moving
> multi-tenant contention 아래 decoder admission, tenant service, pair
> assignment와 local/remote endpoint recovery를 공동 제어하여 fixed,
> predictor-only와 queue/GPU-only policy보다 낮은 latency, 높은 goodput,
> 더 강한 tail isolation 또는 overload robustness를 보였다.

### 현재 이미 허용되는 주장

- local과 remote는 서로 다른 contention state에서 각각 유리하다.
- official remote completion path에는 queue gauge만으로 보이지 않는 empirical
  service ceiling이 있다.
- route-only controller는 median route choice를 개선해도 shared-decoder tail을
  동시에 제어하지 못했다.
- TEMPO-GO의 native global commit/lifecycle wiring이 실제 vLLM path에서
  동작했다.

### 금지되는 주장

- LMCache transport 자체보다 빠르다.
- physical Slingshot switch bottleneck을 식별했다.
- Mooncake/P-D Serve/Dynamo/Kairos보다 빠르다.
- production scale에서 검증됐다.
- 모든 workload에서 항상 빠르다.
- 한 allocation만으로 production-ready다.
- C5 smoke run이 global performance를 증명했다.

## 16. 완료 정의

다음 중 하나가 충족될 때만 새 목표를 완료한다.

1. frozen TEMPO-GO가 independent validation에서 correctness, workload,
   performance 또는 robustness, fairness, scaling과 overhead gate를 통과한다.
2. 두 구조적으로 다른 global candidate와 diagnostic upper bound가 같은
   preregistered gate를 실패하여 broader global scope의 재현 가능한 음성
   결론을 낸다.

코드가 실행되거나 pair1/remote가 한 번 선택되는 것은 완료가 아니다.

## 17. authoritative artifact index

다음 compact artifact는 각 단계의 authoritative entry point다. large raw는
각 result root 아래에 보존되어 있다.

| Evidence | Path | SHA-256 또는 상태 |
|---|---|---|
| v449 strengthened analysis | `results/tempo_elastic_pd_v449_job_57086357/elastic_pd_final_v450.json` | `c5cd52e4968209235be33d2aa62019591c6c30f56988e6f17fdccb3d13b4f242` |
| canonical run12 negative | `results/tempo_elastic_pd_canonical_discovery_57133688/run12/elastic_pd_final.json` | `253a91c90ae77014f812a6ec384945827a5b9b40d23e45e6499e2e1e2676a3` |
| C1/C2 four-endpoint v7 | `results/tempo_pd_contention_fixed_v7_job_57335890_f70_endpoint_tc8_finalchar01/result.json` | `f05dc7b6b3c966860a6eab742f0d388058f65b2cc9abd7f4a2b26d483cb2a24b` |
| P_ONLY attribution | `results/tempo_pd_kv_only_attr_v2_job_57335890/result.json` | `b1420da5d8b4347a760999120d29c531c2c3b04782b944367d2a35889dd46833` |
| C3 ABBA node result | `results/tempo_pd_coupled_abba_v5_job_57343718/result.json` | `e4bf0ec88653d4e0eb25174fc2091bcf940a448a68558cbb5e443082568194a9` |
| C3 gate | `results/tempo_pd_coupled_abba_v5_job_57343718/c3_abba_gate_v1.json` | `560233e9e227377565a794ca875f8b51e9f0080bcb510eade348c82ff5ab2460` |
| C4 Candidate A | `results/tempo_pd_c4_phase_screen_v1_retry7_job_57352661/analysis_v2.json` | `ca8823998192c7c7e68fe417cc7ba03e8aaa825894cd6fd3d6a799cbf1175f8c` |
| C4 Candidate B | `results/tempo_pd_c4_semantic_epoch_candidate_v6_job_57362947/phase_screen_analysis.json` | `7f0119b4618fe1efeea60b644c5ceb54739c4b9f14af658566715f313b5ce3fd` |
| C4 Candidate C | `results/tempo_pd_c4_semantic_credit_epoch_candidate_v7_job_57362947/phase_screen_analysis.json` | `f5b9e8994d97974cddde1fc9529ee364f029278226f3fb3a5d1aa5d21374a773` |
| C4 terminal negative | `results/tempo_pd_c4_semantic_credit_epoch_candidate_v7_job_57362947/negative_conclusion_analysis_v2.json` | `c8cb985aba33724b22c16d1501d9cdbd057d95ea5231b64de23a88d2572cd1f3` |
| G0 native capability | `eval/sota_4node/results/tempo_go_g0/job-57384994-native-v1/manifest.json` | `3fea4508dfd45d306906ec0a495b68e24862e4502b85ebf13027bc4e77edfe52` |
| TEMPO-GO output16 smoke | `results/tempo_go_c5_discovery_57384994_v9/result.json` | `c330fd198d33843f9b512dc58d83d88dd81df017ea67e9660ea3ec50d1cf4150` |
| TEMPO-GO output128 smoke | `results/tempo_go_c5_discovery_57384994_o128_v1/result.json` | `295f9bff9910b400e04186218f280cd0933d88d9f88b4ea13be4ba3573a82bf4` |
| Current global discovery profile | `eval/sota_4node/real_tempo_go_discovery_profile_v1.json` | `383503d03ee5c5a45ff9540c6a8bf635a0807b5fc6bf10b3c4d2381baef31f0f` |
| Phased C1/C2/C3 builder | `eval/sota_4node/build_tempo_go_c5_manifest.py` | implementation; not yet GPU evidence |
| CPU manifest stop gate | `eval/sota_4node/validate_tempo_go_manifest.py` | implementation; 107 tests pass |
| C5 anchor manifest | `results/tempo_go_c5_cpu_gate_20260821_anchor_v1/tempo_go_workload_manifest.json` | `4193da5808aefac5c7214198b07da5a7077ba626c51a24cbe107b1b89d218ccb` |
| C1/C2 output=2 elastic anchor prior | `results/tempo_go_c5_anchor_priors_c12_v1/real_tempo_pd_elastic_profile_c12_anchor_output2_screen_v1.json` | `bb8b31c55a2cc8642209c7eb2426007a02b94c25b37b808cd45e6e49b2442f53`; `screen_only` |
| C1/C2 output=2 endpoint prior | `results/tempo_go_c5_anchor_priors_c12_v1/real_tempo_pd_endpoint_service_profile_c12_anchor_output2_calibration_v1.json` | `9e5ae4d72ee4dd31f285a46c7b48eecd8fd5953a534c663f572ef0fbade7c356`; `calibration_only` |
| C1/C2 anchor global profile | `results/tempo_go_c5_anchor_priors_c12_v1/real_tempo_go_profile_c12_anchor_v1.json` | `a57540fb864ffc675a215c2046a0b57327a51a2f6a0378c53b54a1034cbb498f` |
| C1/C2 anchor provenance | `results/tempo_go_c5_anchor_priors_c12_v1/anchor_profile_provenance.json` | `f9f731c42c59ca6f217b406b90e92c330ddf777c5d3a6f8df183a69eddbc149f` |
| CPU five-arm replay v2 | `results/tempo_go_c5_anchor_priors_c12_v1/five_arm_replay_v2.json` | `ee5331e3a60de20da1e0d62189a3cadceef25994aaeb3160981bd749f6f274de`; terminal/leak-free, performance claim forbidden |
| C5 anchor manifest v2 (invalid workload) | `results/tempo_go_c5_cpu_gate_20260821_anchor_v2/tempo_go_workload_manifest.json` | `3298cbe86e0684e5810f5f5981acb1dd2d20a471943a3f9d0766e911643992d8`; MISS namespace reused; native retry3 invalid |
| C1/C2 output=2 elastic profile v2 | `results/tempo_go_c5_anchor_priors_c12_v2/real_tempo_pd_elastic_profile_c12_anchor_output2_screen_v2.json` | file SHA `bb8b31c55a2cc8642209c7eb2426007a02b94c25b37b808cd45e6e49b2442f53`; fingerprint `fff40706fc2a8ded7226a38efeff62f367a0fc9597d5e24a45424714f3c608ff` |
| C1/C2 output=2 endpoint profile v2 | `results/tempo_go_c5_anchor_priors_c12_v2/real_tempo_pd_endpoint_service_profile_c12_anchor_output2_calibration_v2.json` | file SHA `96f06e8e7fde0388c83054a4e67457af89f6b036b91e0bb697d58a9638cd0a41`; `calibration_only` |
| C1/C2 global profile v2 | `results/tempo_go_c5_anchor_priors_c12_v2/real_tempo_go_profile_c12_anchor_v2.json` | file SHA `05e6c77dee8282b51d0ffbbdab59e5ecf0913b52c853245da2e5885cfe62b934`; fingerprint `1e729131b35615e0376422d6fdb69161794e37234c73cca5378dccbbb441a754`; bound to v2 manifest |
| C1/C2 anchor provenance v2 | `results/tempo_go_c5_anchor_priors_c12_v2/anchor_profile_provenance_v2.json` | `7a1da0bcca0fc85d54619f933ef1fd0e4a02c083b68e2b913da5106d37b776c7` |
| CPU five-arm replay v2 | `results/tempo_go_c5_anchor_priors_c12_v2/five_arm_replay_v2.json` | `c1f9d58abd2438dc4f6e869f976a3d6f06d27d43158b1bcb585ad19a9c55b1fe`; terminal/leak-free, performance claim forbidden |
| v2 native-invalid receipt | `results/tempo_go_c5_native_five_arm_job_57395883_v2_retry3/local/tempo_go_c5_discovery/raw.json` | 2,712 attempted; 768 valid, 1,944 HTTP 502 from repeated explicit MISS namespace; not performance evidence |
| C5 anchor manifest v3 | `results/tempo_go_c5_cpu_gate_20260821_anchor_v3_retry2/tempo_go_workload_manifest.json` | `849bb5cf284c60215d12165e409ac426adc6e5bba3427cda8932c7379fb819fd`; MISS 1,992 unique / P_ONLY 720 |
| C5 manifest validation v3 | `results/tempo_go_c5_cpu_gate_20260821_anchor_v3_retry2/manifest_validation.json` | `d934e9807fcd66b8e5ae5b314878b096101b1fa9f70d42982144ab6f8948d165`; workload SHA `38224ae6e421a0950080951a963ff7d82af480edfa15220c9a45c5c2064ad2f5` |
| C1/C2 output=2 elastic profile v3 | `results/tempo_go_c5_anchor_priors_c12_v3_retry1/real_tempo_pd_elastic_profile_c12_anchor_output2_screen_v3.json` | `bb8b31c55a2cc8642209c7eb2426007a02b94c25b37b808cd45e6e49b2442f53`; fingerprint `fff40706fc2a8ded7226a38efeff62f367a0fc9597d5e24a45424714f3c608ff`; `screen_only` |
| C1/C2 output=2 endpoint profile v3 | `results/tempo_go_c5_anchor_priors_c12_v3_retry1/real_tempo_pd_endpoint_service_profile_c12_anchor_output2_calibration_v3.json` | `62181a17df4aaa66f12d77d3546bb22188a42ac4cf409c9579383a05b23eebaf`; fingerprint `f5e8a4d234638344f85c7db5970679b57710fa977d7f72856345055a52fe0f3`; `calibration_only` |
| C1/C2 global profile v3 | `results/tempo_go_c5_anchor_priors_c12_v3_retry1/real_tempo_go_profile_c12_anchor_v3.json` | `492adcfe015fdb9f8f26011af8c6606143a277853306b96f6a0992e83ec05f1e`; fingerprint `e30744d097ddf66095387e7478a48be88e89da6582c55ef84bfaa864a1f6f012`; bound to v3 manifest |
| C5 remote-guard profile v3 | `results/tempo_go_c5_guard_profile_v3_retry1/real_tempo_go_profile_c12_guard_v3.json` | `8082f4190d56016d7bac6abacbf659017a4fb20a50d1b474223cf9157c1fd3ec`; fingerprint `f8163ff115a2478614afccf57b02a1c535c7dd4e2b3e54f47beda83d1ae3c2a0`; semantic-op reserve `1`; discovery/calibration only, no native evidence |
| C1/C2 anchor provenance v3 | `results/tempo_go_c5_anchor_priors_c12_v3_retry1/anchor_profile_provenance_v3.json` | `63ae865f513c360e7a3886db4293ff7c6d822148d100e9b1cac685973591af28` |
| CPU five-arm replay v3 | `results/tempo_go_c5_anchor_priors_c12_v3_retry1/five_arm_replay_v3.json` | `1bf2977119153e502c07cb0ad56b8a570876e7b8388b2431f6b16fb9a5f08378`; terminal/leak-free, performance claim forbidden |
| Guarded CPU five-arm replay v3 | `results/tempo_go_c5_guard1_anchor_replay_20260821` | file SHA `f8762d0fb8cd24bba633c83bd887d235a7f673b1b137f336d515cf1b4a82b0dd`; schema `tempo-go-global-five-arm-replay-v1`; all five arms 2,712 requests, terminal/leak-free, same trace/no phase-oracle input, performance claim forbidden; guard global fingerprint `f8163ff115a2478614afccf57b02a1c535c7dd4e2b3e54f47beda83d1ae3c2a0` |
| Candidate B queue/SLO-risk profile | `results/tempo_go_c5_queue_scale_profile_v1/real_tempo_go_profile_c12_queue_scale_v3.json` | file SHA `b705c4688a6061d3025a0e63a56a4edaa32e9384dee4edfbcb281fada5195b33`; fingerprint `507d415764ef2dde8661d3516c08fb51aca6643bdca4ce3e8a29e82559eb55f3`; reserve `1`, queue/wait fractions `0.25/0.25`, active-pair penalty `25 ms`; discovery/calibration only |
| Candidate B queue/SLO-risk CPU replay | `results/tempo_go_c5_queue_scale_replay_20260821_v2.json` | `f0c74068a7aa5e187e49b44460a404e0a2c33a6623a02b4d4ada2fbd0f2b9ed2`; same v3 trace, five-arm terminal/leak-free, performance claim forbidden; B pair1 first index `5` vs guard A `35`, but both TEMPO replay `2,433/279` complete/reject |
| Candidate B-fairscale profile | `results/tempo_go_c5_fairscale_profile_v1/real_tempo_go_profile_c12_fairscale_v1.json` | file SHA `f0caa6d73a77b35235035ba3247f79dc6e66ade2a71295713f3fe75fc7f9ca95`; fingerprint `175f31f9db13a9ce5bd45aaf95574f38483a5ca36db651d6c4b050301df27e8e`; reserve `1`, queue/wait fractions `0.5/0.5`; negative candidate |
| Fairness-fix baseline replay | `results/tempo_go_c5_fairness_fix_v1/five_arm_replay_fairness_fix_baseline_v1.json` | `3af04cb0962c617ab65f3daa39cdeab90faeb1d9c901d9d8ebda970fb6e6d366`; same 2,712 trace, terminal/leak-free, performance claim forbidden |
| Fairness-fix fairscale replay | `results/tempo_go_c5_fairness_fix_v1/five_arm_replay_fairness_fix_fairscale_v1.json` | `9213563e5c2a5817b8a94e92e3a8e12f575741bf9fc12f01f7fd00488068a087`; TEMPO `2,433/279` complete/reject, SLO-goodput `2,202`; performance claim forbidden |
| Candidate C quarantine profile (superseded) | `results/tempo_go_c5_quarantine_profile_v1/real_tempo_go_profile_c12_quarantine_v1.json` | file SHA `45fa3d250fa2672ba149cfd696ef5ba32713d949d9b4f3ef0292f7f7f2aa7158`; fingerprint `35fddcbd15eff36608a37cfc7fc86017438b6857e1668996da5f8dd972fcd14e`; retained historical artifact, native validation not run |
| Candidate C failure-quarantine profile (current) | `results/tempo_go_c5_failure_quarantine_profile_v1/real_tempo_go_profile_c12_failure_quarantine_v1.json` | file SHA `1e3c861593a5bd802fa73aa5199657a6ec410d28ee9ebf8aad6970538e127cb6`; fingerprint `6657de67fa75bef8241ffe4148126fe5977fe73da0312b801d7a1843a38849eb`; reserve `1`, `route_failure_quarantine_mode=deny_until_probe`; native attempt did not yield a performance result |
| Candidate C same-trace CPU replay | `results/tempo_go_c5_failure_quarantine_replay_20260821.json` | file SHA `97743f63474ed2b6e5a81f37aad07383138f014e29dda62127db5f3c031e1dc1`; same 2,712-request trace, all five arms terminal/leak-free, TEMPO `2,433/279` complete/reject; no failure injected, performance claim forbidden |
| Candidate C failure-injected CPU replay | `results/tempo_go_c5_quarantine_replay_v2/five_arm_replay_remote_failure_index0_v1.json` | file SHA `2edd9f616fc94f4ee6e55e88e6a647b1cd55974f05ea130d5203f2f428270f21`; one explicit TEMPO remote failure, `tempo-go-global-failure-v1` receipt, pair-0 remote quarantine, all-arm terminal/leak-free; performance claim forbidden |
| Candidate D combined profile | `results/tempo_go_c5_candidate_d_profile_v1/real_tempo_go_profile_c12_candidate_d_v1.json` | file SHA `d8bb3e893fa3279e004e020c2dcf1e34bf7af46dd0ff1d4527863a49816f566d`; fingerprint `75bc2b6f76bded31f1582aac46e2d3594afdf4c79714b80535afa6987848ab18`; C semantic-op reserve/quarantine + B queue/wait `0.25/0.25`, active-pair penalty `25 ms`; endpoint scope `calibration_only` |
| Candidate D same-trace CPU replay | `results/tempo_go_c5_candidate_d_profile_v1/five_arm_replay_candidate_d_v1.json` | SHA `b9567186c224a41a74bedf8744e0a797ba4a0c7838908574bb5c4e0dee9f97777b9`; all five arms terminal/leak-free; TEMPO `2,433/279`, SLO-goodput and E2E equal Candidate C; only pair distribution changed; performance claim forbidden |
| Candidate D failure-injected CPU replay | `results/tempo_go_c5_candidate_d_profile_v1/five_arm_replay_candidate_d_failure_index0_v1.json` | SHA `fa959472271982f9d6f6f48ab282922c4d799bf24a6236a13f5867e521c70b4e`; one remote failure, one `tempo-go-global-failure-v1` receipt, route quarantine, all-arm terminal/leak-free; performance claim forbidden |
| Native C failure-quarantine execution receipt | `results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/tempo/failure.json` | SHA `cc05a04378e05e0a07dae2386ec081e1df78f9f2e3dd807a592dfa2b639ccd6e`; C arm only, 4-node/16-GPU native step exit `143`; execution failure, not a performance result |
| Native C failure-quarantine raw | `results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/tempo/tempo_go_c5_discovery/raw.json` | SHA `c61626d6cef2b7353e0ec8a21609a9bc3b72ea6e4ed240ff5de2216cf9292124`; 2,712 rows, terminal phases complete `1,633`, failed `9`, rejected `1,070`; global failure receipts `9`, quarantine rejections `1,714`; no valid five-arm comparison |
| Native C raw-backed CPU analysis | `results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/native_c_analysis_raw_backed_v1.json` | SHA `579f92d38140f0f7ccb31f18a19ce9c9670ea5b3371ba48e99cf7850dbd3a1ac`; raw-backed execution-failure summary, performance claim forbidden |
| Native C vLLM/LMCache failure log | `results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/tempo/tempo_go_c5_discovery/node-3-vllm.log` | SHA `8856cf969e7deacc75d45d06c00dd3451eb3acf3e7b6bf8228454f892ddcd063`; official LMCache `CacheEngineKey ... not found in local data` followed by `EngineCore`/`EngineDeadError` |
| Native C proxy failure log | `results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/tempo/tempo_go_c5_discovery/node-2-proxy.log` | SHA `42ecf002a501b6604fc4e351432f1466597c80df30bc24fd85fb9a4dd0873e2b`; `httpx.ConnectError: All connection attempts failed` |
| Native v3 ALWAYS_LOCAL | `results/tempo_go_c5_native_five_arm_job_57395883_v3_retry3/local/tempo_go_c5_discovery/raw.json` | `a4a43442ed0a2697c50cc503aa53f89c3b60e709f0f10e6be93466eedc1dd8e9`; 2712/2712 valid |
| Native v3 official remote | `results/tempo_go_c5_native_five_arm_job_57395883_v3_retry3/remote/tempo_go_c5_discovery/raw.json` | `6a45ae723efde74499e3adca59d77e4c8a1bd2bb3527b9d2a9ed6426a509e555`; 2712/2712 valid |
| Native v3 predictor-only | `results/tempo_go_c5_native_five_arm_job_57395883_v3_retry3/predictor/tempo_go_c5_discovery/raw.json` | `bf9010b29d958e974cf067027baa3c587679bdfd05da7bdd778bcc69b4d89a51`; 2712/2712 valid |
| Native v3 queue-GPU-only failure | `results/tempo_go_c5_native_five_arm_job_57395883_v3_retry3/queue_gpu/tempo_go_c5_discovery/node-1-vllm.log` | `e0473e91f7db5824286998a916f69fb3966af5aabaa87728efa17dba9771054d`; LMCache cache-key assertion / EngineDeadError; no raw performance result |
| Native TEMPO retry4 integration failure | `results/tempo_go_c5_native_tempo_only_job_57395883_v3_retry4/tempo/tempo_go_c5_discovery/frontend.log` | no raw receipt; exact endpoint row lookup failure for `4094/2/MISS`; fixed by bounded endpoint proxy lookup |
| Native TEMPO retry5 pre-fix receipt | `results/tempo_go_c5_native_tempo_only_job_57395883_v3_retry5/tempo/tempo_go_c5_discovery/raw.json` | `33fda0e11dfcb439471027a7cad0c99460bf07b84721c7bfaa6f6bc2b36a2523`; 2,712 attempted, 1,029 admitted/complete, 1,677 queue-timeout 503, 6 telemetry-timeout 503; `router_decisions_exact=false`, not performance evidence |
| Native TEMPO retry5 warmup receipt | `results/tempo_go_c5_native_tempo_only_job_57395883_v3_retry5/tempo/tempo_go_c5_discovery/warmup.raw.json` | `4e2a8801544c3eaddbcd01747b0d56c0c10464e133b267551dd33bf3c5d821d8`; warmup only |
| Native TEMPO receipt-closure retry6 raw | `results/tempo_go_native_receipt_closure_57400890_retry6/tempo/tempo_go_c5_discovery/raw.json` | `09add8b7ac40920c6ea938f7e8d173c7ddadc9047704362bf6379e00af61b8c9`; 2,712 attempted, 2,049 complete, 653 explicit queue rejects, 10 telemetry-timeout requests without decisions; `router_decisions_exact=false`, not performance evidence |
| Native TEMPO receipt-closure retry6 warmup | `results/tempo_go_native_receipt_closure_57400890_retry6/tempo/tempo_go_c5_discovery/warmup.raw.json` | `25ff3b395b99bc8700b131fe1a7c307bb3188a6e6ef791d0bdf10dc77ac57f8f`; warmup only |
| Native TEMPO receipt-closure retry7 startup failure | `results/tempo_go_native_receipt_closure_57400890_retry7/tempo/failure.json` | `49e78165cd98fec60bad80fa0f0c82da8f3c690140cf4047458b271c87680214`; node 1 vLLM child exited `-7` before readiness; stderr SHA `6bf46cfcc0c6cfbe578b6cca442c9c5a5a7f5502b3c681fa081e14013964c5f1`; no workload/raw; no retry |
| Native TEMPO retry6 invalid receipt | `results/tempo_go_c5_native_five_arm_job_57400890_v3_retry6/tempo/tempo_go_c5_discovery/raw.json` | `c6b01f5aa1f5c5df248ef4e42493d269e586cb5fb4302131fabc022f66801c8b`; 2,712 attempted, 833 valid, 1,783 telemetry-refresh-failed, 73 structured queue rejects, 7 telemetry timeouts, 4 upstream 502; LMCache `CacheEngineKey ... not found` killed EngineCore; node-3 log SHA `7f0b7fdbcdacaeff5f8484659d23f90da02a49f527981585b3855828c9835d65`; not performance evidence |
| Native TEMPO retry9 repeated data-plane failure | `results/tempo_go_c5_native_tempo_only_job_57400890_v3_retry9_quarantine/tempo/tempo_go_c5_discovery/` | measured raw/result 없음; warmup SHA `a0d36b21d113c4ea7f3b281377350b60490755ee6b89fcccfe1f0179d590c6bf2`; node-3 log SHA `b09c0fe17ad1e4ddd0148b99c7ae30abae7c7bb8910de9558eeb4dc8e708473c`; same LMCache `CacheEngineKey ... not found` shape on chunk `2816912063036934730`; not performance evidence |
| Native guarded TEMPO single-arm raw | `results/tempo_go_c5_native_tempo_guard1_job_57400890_v1/tempo/tempo_go_c5_discovery/raw.json` | `3e95bf0fd6bc1317079e1d3ca58dbf646de18cdffd0438c90ecf7fd6d8485364`; 2,712/2,712 terminal-valid, 1,904 complete + 808 explicit reject, semantic-op guard candidate rejection 328; no EngineCore/LMCache assertion; receipt closure only |
| Native guarded TEMPO single-arm analysis | `results/tempo_go_c5_native_tempo_guard1_job_57400890_v1/tempo_go_guard_analysis.json` | `13a46b1b8a7d12b182724add975a1b605dc6888e94a2a48b932bda4c2944b4d8`; `router_decisions_exact=true`, `terminal_contract_valid=true`, `performance_claim_allowed=false` |
| Native guarded TEMPO follow-up v7 (unreceipted) | `results/tempo_go_c5_native_tempo_guard1_job_57400890_v7/tempo/tempo_go_c5_discovery/` | four vLLM/LMCache services reached health and 2,712-row warmup input was created; warmup SHA `b2cfa8f5d5c1a950a3cfce9a611beb89d9c6488809d8ecbd6dfae376df42476d`; inner step failed on `nid001168`; measured/warmup raw and `failure.json` absent; scheduler-step failure only, no performance evidence |
| Native guarded C5 five-arm discovery v1 | `results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1/native_five_arm_analysis.json` | `dbefc699ef7448e2f03a43d4a1e5f779ffbfa8cc47f68330d01a58d69289fb18`; order `tempo,queue_gpu,predictor,remote,local`; all five present, same request/workload identity, queue-GPU failure receipted, performance claim forbidden |
| Native guarded C5 scheduler-provenance reanalysis v3 | `results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1/native_five_arm_analysis_v3.json` | `921ec4ad74dc28604bc65a65a734e8638817cf4d1b51d745a416064820cd350d`; TEMPO global scheduler observations `5,424 = 2,712 × 2`, invalid `0`, `tempo_has_global_scheduler_observation=true`; request-start load mode remains intentionally `disabled`; performance claim forbidden |
| Native guarded C5 TEMPO arm v1 | `results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1/tempo/tempo_go_c5_discovery/raw.json` | `7ae7552a39e132c3e00a670e310fe04421fd44b5ebebc4b17dc3f880caeea87e`; 2,712/2,712 terminal-valid, 1,865 complete, 847 explicit reject, local 1,686 / remote 179, endpoint completion receipt 1,865 |
| Native guarded C5 queue-GPU failure v1 | `results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1/queue_gpu/failure.json` | `8b56e53bb7e6b8ff975742c185810acfcd8b55f9f445cf11ab04fa4cda5e4c38`; 4-node/16-GPU UCX process failure after LMCache receiver allocation timeout / EngineCore exit; no latency substitution |
| Native guarded C5 fixed arms v1 | `results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1/{local,predictor,remote}/` | local raw `05dfdf837ffe8c91113efff2766b8b7f1011f05a073ee57ada852f0b41c3e6aa`, predictor raw `323140b6041aecd9a983b7210540afa4b80258011bd9f8dac7fc25166c48f8b4`, remote raw `781d1396dc058edde92a8d7d8dbefa1a2e48e234da5b38ae5fa6203860f629cd`; each 2,712/2,712 valid |
| Native five-arm runner | `eval/sota_4node/run_tempo_go_c5_five_arm_in_allocation.sh` | implementation; requires existing interactive 4-node allocation |
| Native five-arm analyzer | `eval/sota_4node/analyze_tempo_go_c5_five_arm.py` | receipt/gate checker; no performance claim without native evidence |
| C5 native run-contract module | `eval/sota_4node/tempo_go_c5_run_contract.py` | schema `tempo-go-c5-native-run-contract-v1`; strict artifact/source/environment binding |
| C5 frozen contract v2 | `results/tempo_go_c5_frozen_contract_v2/native_run_contract.json` | file SHA `b34a4b52d81b45957a1ef1d5c8bb3f3a1a54c8dabdd39d12bd23dc14d80197af`; fingerprint `df8d85610f70d72d62dc0a36962a09b7190e28b05526dd051364653733adf248`; discovery only, performance claim forbidden |
| C5 frozen contract v6 | `results/tempo_go_c5_frozen_contract_v6/native_run_contract.json` | file SHA `aa30e6d263fb90f925a7101f86afa8e4b8ec22813439199cd75cc58748bf151f`; fingerprint `a3b3d8f08945a0578489ff911bcfa41cf8430b0a028aa4fde74820581d2ecacb`; guard normal discovery only, performance claim forbidden |
| C5 v6 normal offline replay | `results/tempo_go_c5_frozen_contract_v6/offline_replay.json` | SHA `7d5ea84429cce69936ed44531fe42a365da650d7250c5b3ef0237f5dba960623`; 2,712-row five-arm, terminal/leak-free, performance claim forbidden |
| C5 failure-injection profile | `results/tempo_go_c5_failure_profile_v1/real_tempo_go_profile_c12_failure_injection_v1.json` | file SHA `33b4feebc47ef7bb8686d986082ca826f5e4bafe343efb7cf858eab8ee3b0327`; fingerprint `d5db711c984a06572eb594ed9c7ab175e4aba5e2002f5b3138a5cc7614baa906`; `deny_until_probe`, discovery only |
| C5 frozen contract v7 | `results/tempo_go_c5_frozen_contract_v7/native_run_contract.json` | file SHA `9ece4a15902a57259365f53593dc6e06940659a99262ac66af81b16e468e7bb6`; fingerprint `d8aca7f9e5dacdfd87fd34f978d9ea3d1d15e35aae5a7159079b6a022081cc07`; failure replay only, performance claim forbidden |
| C5 v7 failure offline replay | `results/tempo_go_c5_frozen_contract_v7/offline_failure_replay.json` | SHA `fb8d0f24e42ea76142cb6b38ca3c904b482cdeb5c27e7f746826193879f32ff9`; one remote failure receipt, quarantine, terminal/leak-free, performance claim forbidden |
| C5 frozen contract v8 | `results/tempo_go_c5_frozen_contract_v8/native_run_contract.json` | file SHA `bed1edfed9c4e829fd712c34b3b7c631bbe6a4332c57d5d0ded91c02bb7728cf`; fingerprint `b5f1fea71b2606323a43baeaa66756fb7e48ea7460f44fa0cf3eefe708c0cd9b`; guard normal discovery only, performance claim forbidden |
| C5 v8 normal offline replay | `results/tempo_go_c5_frozen_contract_v8/offline_replay.json` | SHA `11f67c5d861cb32a4bc02d83afc6370a6f4ad8cec64d61ab370df944fde5facd`; 2,712-row five-arm, terminal/leak-free, performance claim forbidden |
| C5 frozen contract v9 | `results/tempo_go_c5_frozen_contract_v9/native_run_contract.json` | file SHA `fd99c3804430ec3a95cc52772c0715222262ba7f8a2b02046f60a8d338308aae`; fingerprint `c43d4019c3a2628d193b34bce7885bd5dd8a513db1513fc5792812e07b0a3cd2`; failure replay only, performance claim forbidden |
| C5 v9 failure offline replay | `results/tempo_go_c5_frozen_contract_v9/offline_failure_replay.json` | SHA `f772b9f9190b03f8b8f5cbdd4c8ad59f92a659f67bb55c77c95d2c894fc071b2`; one remote failure receipt, quarantine, terminal/leak-free, performance claim forbidden |
| C5 frozen contract v10 | `results/tempo_go_c5_frozen_contract_v10/native_run_contract.json` | file SHA `63d33edf83c5825ba9d1981e68f0ece761e739d6d1b977e610be6f947d3c065c`; fingerprint `d37b8330734a7479f48c8bd844cccbe91403f96d368918119197ddacb598a737`; guard normal discovery only, performance claim forbidden |
| C5 v10 normal offline replay | `results/tempo_go_c5_frozen_contract_v10/offline_replay.json` | SHA `9010c46bc0949419518b7dcf15ab2a8ef5b1d0d2a46f47d3188d1b52942c5496`; 2,712-row five-arm, terminal/leak-free, performance claim forbidden |
| C5 frozen contract v11 | `results/tempo_go_c5_frozen_contract_v11/native_run_contract.json` | file SHA `7713f6414c34c6a6ef52f485e546b11086620bb83b87f1ccc2ccacc9facb6699`; fingerprint `76c6651ab8b673f78bc1173a08e66d885414050015236bd11b942572abc31728`; failure replay only, performance claim forbidden |
| C5 v11 failure offline replay | `results/tempo_go_c5_frozen_contract_v11/offline_failure_replay.json` | SHA `2b38d895d77ee56da7112ef168a97082da2c40fb002acd44e9bb7bfccfbaf5b0`; one remote failure receipt, quarantine, terminal/leak-free, performance claim forbidden |

재분석할 때 원본을 overwrite하지 않는다. analyzer output은 새 경로에 만들고
input SHA, code/profile/workload manifest SHA와 parent raw SHA를 함께 기록한다.

## 18. 현재 구현 상태와 다음 stop/go 판단 (2026-08-22)

이 문서의 목표를 임의로 축소하지 않고, 위 원칙을 코드에 반영한 현재 상태는
다음과 같다.

- `tempo/pd_global_orchestrator.py`는 tenant별 TTFT/TPOT/E2E SLO, queue wait,
  minimum service를 보존하고, weighted decode-token service가 아니라
  pair-capacity 기준 dominant-resource service를 사용한다. weighted service
  debt와 raw service units를 분리해 fair ordering과 minimum-service fraction의
  단위를 맞췄다. coordinator도 tenant별 queue SLO budget을 admission timeout에
  적용한다.
- active pair의 실제 pressure가 `scale_up_utilization`을 넘으면, 아직 active가
  아니지만 telemetry가 fresh한 spare pair×route도 같은 admission transaction에서
  비교한다. 단순히 active pair capacity가 완전히 실패한 뒤에만 pair1을 쓰지 않는다.
- `tempo/pd_global_telemetry.py`는 optional discovery contract를 넘어,
  `scheduler_observation_required=true`인 profile에서 실제 vLLM scheduler
  snapshot과 endpoint first-response completion/residual을 누락 시 fail-closed한다.
  scheduler source는 `router_local_vllm_prometheus_observe_only`로 고정하며,
  physical NIC counter나 phase label은 허용하지 않는다.
- pair router의 `/tempo/runtime_telemetry`와 frontend request-triggered
  all-pair fetch가 연결됐다. 이 endpoint는 local vLLM `/metrics`의 running,
  waiting, KV usage와 endpoint-controller snapshot을 함께 묶는다.
- `tempo/pd_global_workload.py`와
  `eval/sota_4node/build_tempo_go_c5_manifest.py`는 C1/C2/C3 anchor rate
  (22.4/s, 4.76/s, 12/s), 15초 phase, 2초 cooldown, 명시적
  `arrival_offset_ms`, 4 tenant stream, 512/2048/4094 geometry를 가진
  native-client JSONL/sidecar manifest를 생성한다. phase metadata는
  controller 입력으로 전달되지 않는다.
- base `real_tempo_go_discovery_profile_v1.json` fingerprint는
  `383503d03ee5c5a45ff9540c6a8bf635a0807b5fc6bf10b3c4d2381baef31f0f`이며,
  C5 v3 bound global profile
  `results/tempo_go_c5_anchor_priors_c12_v3_retry1/real_tempo_go_profile_c12_anchor_v3.json`
  의 fingerprint는
  `e30744d097ddf66095387e7478a48be88e89da6582c55ef84bfaa864a1f6f012`다. 둘 다
  scheduler/completion telemetry와 tenant business contract를 요구한다.
- retry6/retry9의 remote data-plane failure 뒤에는 기존 v3 profile을 덮어쓰지
  않고 `results/tempo_go_c5_guard_profile_v3_retry1/`에 semantic-op
  safety reserve `1`을 가진 guard profile을 별도 생성했다. 파일 SHA는
  `8082f4190d56016d7bac6abacbf659017a4fb20a50d1b474223cf9157c1fd3ec`,
  global fingerprint는
  `f8163ff115a2478614afccf57b02a1c535c7dd4e2b3e54f47beda83d1ae3c2a0`다.
  이는 native 성능 결과가 아니라 다음 receipt-closure run의 frozen input이다.
- `overload_action=reject_new_request`는 이제 구현됐다. ingress queue가 가득
  차면 예외를 던지지 않고 명시적 `REJECT` terminal decision을 기록하며,
  replay 종료 시 남은 queued request도 `global_admission_queue_timeout`으로
  reject한다. 이 변경은 overload를 숨기지 않고 관측 가능한 business action으로
  만든 것이며, throughput 우위를 의미하지 않는다. retry5 native run에서 실제
  이 경로가 발동해 queue-timeout 503이 1,677건 관측됐다.
- C1/C2 actual raw의 output=2 anchor prior를 만들었다. C1 local과 C2 remote의
  동일 semantic output hash를 확인했지만 raw request-id pairing은 아니므로
  `screen_only` prior다. C4 mixed-tail에서 만든 더 보수적인 prior는 C3 tail을
  잘못 합산하므로 native profile의 근거로 사용하지 않는다.
- historical CPU contract receipt는 `107 passed, 11 subtests passed`였고,
  reject-receipt patch와 retry6 structured-error classifier를 포함한 기존
  global focused suite는 `66 passed, 11 subtests passed`다. 여기에 이번
  pair-quarantine 변경의 agent/telemetry/orchestrator 회귀를 합친 현재 bounded
  suite는 현재 fairness/accounting 교정까지 포함해 `104 passed, 11 subtests
  passed`로 다시 통과했다.
  C1/C2 anchor five-arm replay v3도 모든 arm에서
  2,712개 request를 terminal 처리하고 error/inflight/resource leak/queue
  residual 0을 통과했다. 그러나 TEMPO-GO의 complete/reject는
  `2,461/251`, queue-GPU-only는 `2,594/118`이므로 replay의
  `performance_claim_allowed`는 false다.
- endpoint failure provenance, `remote_pre_admission_guard`, semantic-op reserve와
  surviving-pair 선택을 포함한 현재 focused CPU suite는
  `104 passed, 11 subtests passed`다. 이 결과는 native pair-health receipt를
  대체하지 않는다.
- 새 phased workload의 native discovery는 부분 실행됐다. local/official-remote/
  predictor 세 arm은 2,712/2,712 valid receipt와 실제 E2E/TTFT/TPOT를 만들었지만,
  queue-GPU-only는 LMCache key assertion으로 종료됐다. TEMPO retry4는 exact
  endpoint lookup bug로 raw를 만들지 못했고 retry5는 1,029 admitted/complete와
  1,683 rejected-or-timeout attempts를 기록했지만 reject receipt가 닫히지 않아
  `router_decisions_exact=false`다. 같은 allocation의 retry6 TEMPO arm은
  structured queue reject를 73건 관측했지만, 1,783건의 telemetry refresh
  failure 뒤 official LMCache cache-key assertion으로 EngineCore가 죽어 833/2712
  valid에 그쳤다. 이후 같은 allocation의 retry9는 native readiness와 overlay를
  통과한 뒤 P_ONLY remote retrieve 중 다른 chunk에서 같은 LMCache assertion을
  재현했고 measured receipt를 만들지 못했다. 따라서 native contention이
  business SLO를 깨뜨리고 global admission rejection을 필요로 한다는 증거와
  official LMCache data-plane failure가 단일 우연이 아니라는 robustness 증거는
  확보했지만, data-plane failure를 pair quarantine/reassignment로 흡수했다는
  증거와 TEMPO-GO의 performance, fairness, pair scaling 승리는 아직 주장하지
  않는다.

- 이후 guarded single-arm `results/tempo_go_c5_native_tempo_guard1_job_57400890_v1`
  은 같은 v3 workload에서 2,712/2,712 exact terminal receipt를 만들었다.
  1,904 complete와 808 explicit global reject가 모두 decision ledger에 매칭됐고,
  local 1,623 / remote 281 completed route, pair activation 1, semantic-op guard
  candidate rejection 328이 기록됐다. retry6/retry9의 LMCache assertion은 이
  guarded arm에서 재현되지 않았다. 이것은 guard가 실제 native data path에서
  작동하고 receipt를 닫았다는 증거이지, failure 원인의 단독 인과 증명이나
  성능 승리가 아니다. quarantine 후 reassignment path는 fatal failure가 없어
  아직 직접 검증되지 않았다.

- 그 뒤 같은 guard profile로 시도한 `results/tempo_go_c5_native_tempo_guard1_job_57400890_v7`
  은 네 node의 vLLM health와 LMCacheConnectorV1:UCX 초기화까지 도달했고
  2,712-row warmup 입력도 생성했지만, `warmup.raw.json`과 measured `raw.json`
  이 생기기 전에 내부 native Slurm step이 `nid001168`에서 실패했다. `failure.json`
  도 step kill 때문에 없으므로 이 run은 unreceipted scheduler-step failure로
  분류한다. v1의 닫힌 receipt를 무효화하지 않으며, v7을 LMCache data-plane
  regression이나 성능 결과로 해석하지 않는다. allocation 안에서 추가 blind
  retry하지 않고, 다음 실행은 receipt-producing five-arm runner의 step-level
  failure receipt를 먼저 보강해야 한다.

- v7에서 확인한 signal 종료 gap은
  `eval/sota_4node/run_tempo_go_c5_five_arm_in_allocation.sh`에 현재 arm·signal·
  workload/manifest SHA를 남기는 `tempo-go-c5-native-arm-signal-failure-v1`
  trap으로 보강했고, shell syntax/diff 검사를 통과했다. 이 patch는 v7의
  누락된 receipt를 사후 생성하지 않으며, 다음 native run에서만 적용된다.
  analyzer도 process-failure와 signal-failure 두 schema를 모두 zero-request
  execution failure로 소비하고 signal provenance를 보존한다. bounded CPU
  analyzer test 3건과 전체 관련 suite 96건(+11 subtests)이 통과했지만,
  실제 Slurm signal이 이 receipt를 생성하는 native 증거는 아직 없다.

- 이후 현재 승인된 `57402376` native interactive allocation에서 guard profile과
  v3 manifest를 고정한 counterbalanced five-arm discovery가 완료됐다. arm order는
  `tempo -> queue_gpu -> predictor -> remote -> local`이며, v3 workload SHA와
  manifest SHA는 모든 receipt에서 일치했다. local/predictor/remote는 각각
  2,712/2,712 valid였고, queue-GPU-only는 `failure.json`으로 닫혔다. TEMPO는
  2,712/2,712 terminal-valid, 1,865 complete, 847 explicit global reject,
  local 1,686 / remote 179, endpoint completion receipt 1,865와 pair activation
  1건을 기록했다.

- 이 discovery의 descriptive 수치는 TEMPO E2E p50/p99 `983.9/14,485.0 ms`,
  local `1,197.1/17,392.2 ms`, predictor `9,026.6/17,308.9 ms`, official
  remote `10,921.8/17,441.0 ms`였다. 그러나 TEMPO output-token goodput은
  `136.9/s`로 always-local `190.2/s`보다 낮고, 847건 reject를 포함하므로
  primary performance gate를 통과하지 않는다. queue-GPU-only는 LMCache
  receiver allocation timeout 뒤 decoder EngineCore가 종료되고 frontend 502가
  이어져 latency arm으로 대체하지 않았다. 따라서 이 run은 global admission,
  explicit business rejection, endpoint completion telemetry와 baseline failure
  isolation을 native에서 닫은 integration/robustness evidence이지, TEMPO의
  performance 승리나 independent validation이 아니다. 다음 stop/go는 reject
  비율·tenant fairness·pair scaling을 먼저 개선한 frozen candidate의 독립
  validation이며, 이번 discovery만으로 완료하지 않는다.

- 위 native receipt의 request-start `vllm_load_decision_mode=disabled`는 adaptive
  endpoint feedback에서 synchronous `/metrics`를 호출하지 않기 위한 의도적
  ablation이다. 이것과 별개로 TEMPO global decision provenance에는
  `router_local_vllm_prometheus_observe_only` scheduler snapshot이 두 pair에
  대해 각각 2,712건, 총 5,424건 들어 있고 invalid snapshot은 0이다. 따라서
  TEMPO arm의 actual scheduler/completion provenance는 닫혔지만, 이는
  performance/fairness/scaling 승리를 뜻하지 않는다.

- 별도의 receipt-closure retry6 run은 LMCache EngineCore crash 없이 workload의
  2,712 request를 모두 수집했지만, 2,049 complete, 653 queue reject, 10
  telemetry-refresh 503로 `router_decisions_exact=false`였다. 즉 queue overload
  rejection은 native ledger에 닫혔고, telemetry failure rejection만 누락됐다.
  이 결과는 contention과 business reject가 실제로 발동했다는 integration
  evidence이지만 성능·fairness·scaling 결과가 아니다.

- 수정 후 receipt-closure retry7은 workload 진입 전 node 1 vLLM child가 exit
  code `-7`로 readiness에 실패했다. raw/request receipt가 생성되지 않았고,
  이는 telemetry rejection patch의 결과가 아니라 native startup failure다.
  같은 allocation에서 추가 retry하지 않고 stderr SHA와 startup log만 보존한다.

retry5와 receipt-closure retry6의 queue-timeout과 telemetry-refresh timeout은
같은 실패가 아니다. 전자는 frozen overload policy의 business reject이고 후자는
sensor/control-plane robustness failure다. coordinator는 이제 둘을 모두 별도
명시적 terminal `REJECT`로 기록하도록 수정됐으며, 다음 native run에서
`router_decisions_exact=true`와 모든 request ID coverage를 확인해야 한다. 503
status만으로 성공 처리하지 않는다.

fixed-arm analyzer를 기존 v3 raw에 적용하는 과정에서는 arm별 rewritten
workload SHA와 receipt의 canonical source SHA를 하나로 비교하던 identity gap도
확인했다. canonical source SHA는 manifest binding용으로 유지하고, raw client
workload SHA는 arm namespace 검증용으로 별도 기록하도록 node receipt와
analyzer를 수정했다. 이 수정 후 local/remote/predictor의 offline descriptive
metrics가 실제로 산출되지만, 이는 native run을 재실행한 증거가 아니다.

이후 analyzer의 service-metrics 중간 구현이 정리 과정에서 잘린 것을 발견해
복구했다. 현재 analyzer는 tenant/phase별 SLO-goodput, output-token goodput,
queue wait와 global admission wait, pair activation, request-start ablation과
global-provenance scheduler observation,
endpoint feedback, selected-route prior counterfactual을 모두 계산한다. reject
request는 `terminal_kind=global_reject`와 matching decision receipt가 있을 때만
정상 terminal로 인정하며, rejection-only tenant를 starvation으로 표시하지
않는다. 해당 복구와 native node/receipt 회귀 검증은 최신 focused suite에서
`96 passed, 11 subtests passed`로 통과했고, scheduler-provenance analyzer
회귀 test를 포함한 별도 analyzer test도 `4 passed`다. 이 CPU 결과 역시 native
performance evidence가 아니다.

추가 audit에서 native frontend가 모든 tenant에 endpoint 기본 deadline을 전달해
`latency`/`interactive`의 더 짧은 SLO가 queued-request ordering에서 약화되는
gap을 확인했다. `GlobalOrchestrator._effective_deadline_ns()`는 이제 외부
remaining deadline을 tenant의 frozen E2E SLO로 cap하고, candidate feasibility와
fair queue ordering이 같은 business deadline을 사용한다. route, phase, future
arrival을 추가 입력으로 넣지 않으며, 이는 request-local threshold가 아니라
tenant-aware global admission correctness fix다. global orchestrator/coordinator/
profile/telemetry와 C5 node/analyzer focused suite는 이 수정 후 `78 passed`였다.
이후 strict C5 run-contract builder/runner/node/analyzer receipt binding과
regression test를 추가했고, 현재 bounded focused suite는 `86 passed`다. 이는
기존 native receipt를 재분석하거나 성능 승격한 결과가 아니며, 다음 native result
root에서 해당 code revision을 freeze해 검증해야 한다.

현재 v10 normal contract는 contract file SHA와 internal fingerprint, v3
manifest/workload SHA, guard global profile fingerprint, Elastic/endpoint
fingerprint, model config SHA, source inventory, exact five-arm order와 node
parameters가 모두 일치한다. v11는 동일 workload/source identity에
`deny_until_probe` failure profile만 바인딩한 별도 contract다. runner가
contract 없이 시작되거나 inherited profile/parameter override를 받으면 중단한다.
v10 normal replay는 all-arm terminal/leak-free를 통과했고 v11 failure replay는
정확히 한 remote failure receipt와 pair-0 remote quarantine을 통과했다. 두 replay
모두 `performance_claim_allowed=false`, `native_gpu_run_allowed=false`이며,
endpoint profile promotion과 held-out independent validation은 아직 남아 있다.
quarantine-disabled profile에서 failure injection을 시도할 때는 replay를 부분
진행하지 않고 fail-closed하며, receipt 승인 뒤에만 replay-side credit을 반환한다.

기존 4-node/16-GPU interactive allocation `57395883`은 종료됐다. 승인된 새
allocation `57400890`에서는 `tempo_go_c5_native_five_arm_job_57400890_v3_retry6`
의 data-plane failure receipt와 별도로
`tempo_go_native_receipt_closure_57400890_retry6`의 receipt-closure raw를
보존했다. coordinator 수정 후 native exact coverage를 확인할 때도 새 result
root를 사용하며, 기존 retry4/retry5/retry6/retry9 raw와 로그를 overwrite하지
않는다. retry6/retry9 두 번의 native LMCache failure가 확인됐으므로, 새 guard
profile과 route failure provenance를 먼저 freeze했고, 다음 native run에서는
failure 후 surviving-pair ledger 또는 명시적 tenant-aware reject가 닫히는지
확인한다.

v3 profile은 의도적으로 independent validation 승격을 막는 scope boundary를
가진다. `identity.workload_manifest_sha256`는 이제 C5 v3 manifest에 bound되고
elastic/endpoint fingerprint와 provenance도 일치하지만,
`identity.endpoint_profile_deployment_scope`는 `calibration_only`다. 이것은
버그를 숨기는 값이 아니라 native endpoint completion receipt가 없는 상태에서
final C5 claim을 막는 안전장치다. 새 phased manifest를 만들 경우에도 hash만
바꾸지 말고 manifest, endpoint prior receipt와 global profile을 함께 재생성한다.

phased workload 생성/검증은 다음처럼 한다. `<POOL512>`, `<POOL2048>`,
`<POOL4094>`는 tokenizer로 각각 exact geometry를 확인할 수 있는 기존 local
JSONL source pool이어야 하며, 임의 prompt를 만들어 길이를 맞추면 안 된다.

```bash
PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=. \
  .vllm_venv/bin/python -m eval.sota_4node.build_tempo_go_c5_manifest \
  --source-512 <POOL512> --source-2048 <POOL2048> --source-4094 <POOL4094> \
  --model /pscratch/sd/s/sgkim/Skim-Tempo/models/Qwen2.5-7B-Instruct \
  --output-dir <NEW_RESULT_ROOT>/tempo_go_c5_phased \
  --replicates 2 --duration-ms 15000 --cooldown-ms 2000 \
  --foreground-rate 2 --decoder-hot-rate 22.4 \
  --remote-hot-rate 4.76 --kv-remote-hot-rate 12 \
  --anchor-output-tokens 2 \
  --background-output-tokens 128

PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 PYTHONPATH=. \
  .vllm_venv/bin/python -m eval.sota_4node.validate_tempo_go_manifest \
  --manifest <NEW_RESULT_ROOT>/tempo_go_c5_phased/tempo_go_workload_manifest.json \
  --workload <NEW_RESULT_ROOT>/tempo_go_c5_phased/workloads/validation.jsonl \
  --output <NEW_RESULT_ROOT>/tempo_go_c5_phased/manifest_validation.json
```

이 명령은 workload artifact만 만든다. profile의 workload SHA와 endpoint prior
binding을 새로 만들기 전에는 `c5_tempo_go_node_entry.sh`에 전달하지 않는다.

남은 stop/go 순서는 명확하다. v3 manifest의 JSONL loader/offline replay,
phase count, arrival monotonicity, tenant fairness, proactive pair activation,
stale/missing telemetry fail-closed와 static runner/analyzer check는 이미
통과했다. 여기서 기존 C4 replay artifact나 v3 CPU replay를 native global
performance evidence로 오인하지 않는다. 사용자가 허용한
4-node/4-hour interactive allocation에서 retry6/retry9의 LMCache failure를
재현 가능한 pair-health event로 닫고, remote pre-admission guard와
reject-receipt patch가 반영된 TEMPO-GO 한 arm의 request ledger를 확인해야 한다.
retry4/retry5/retry6/retry9 invalid receipt를 성능으로 세지 않는다. v2 retry3은
workload-invalid으로 폐기했으므로
v3 profile/manifest로만 실행한다. runner는 arm별 profile mode와 counterbalanced order를
고정하고, launcher는 sidecar manifest hash가 profile과 다르면 실행을 거부한다.
새 run은 old smoke result를 overwrite하지 않고 profile/workload/code SHA와
actual scheduler/completion receipt를 모두 기록한다. native receipt가 닫힌
뒤에만 endpoint profile의 calibration-only scope를 final validation용으로
재검토한다.

현재 Candidate C는 CPU와 native failure receipt에서 failure-driven path를 닫은
별도 구조 후보이다.
`route_failure_quarantine_mode=deny_until_probe` profile은 명시적 endpoint/upstream
failure를 `tempo-go-global-failure-v1` terminal receipt로 기록하고, 현재 request
ID를 유지한 채 재시도하지 않으며, 실패한 route를 다음 PROBE telemetry까지
admission에서 제외한다. 이미 queue된 다른 request만 surviving pair×route로
재평가한다. pair-level transport/runtime failure는 local과 remote를 함께
quarantine하고, HTTP status failure는 해당 semantic route만 quarantine한다.
focused test와 bounded suite(`104 passed, 11 subtests passed`)는 quarantine receipt, exactly-once
resource release, surviving-pair assignment, quarantine rejection, pair-scope
quarantine, PROBE recovery를 모두 확인했다. 동일 trace replay에서는 C profile이
terminal/leak-free였고, 별도의 failure-injected replay에서는 TEMPO-GO가 정확히
한 remote failure receipt를 생성해 pair-0 remote를 quarantine한 뒤 같은 pair의
remote 재사용을 막았다. 주입된 arm의 complete 수를 baseline과 비교하지 않으며
performance claim은 여전히 금지된다. 승인된 interactive allocation
`57404614`에서 Candidate C arm을 실제 native path로 실행했고,
`frontend_tempo_go_failure` receipt 9건을 확보했다. raw는 2,712건의 terminal
ledger를 보존하지만 native step이 exit `143`으로 종료되어 `result.json`과
독립 성능 비교는 없다. pair-scope transport failure 3건, route-scope HTTP
failure 6건, `route_failure_quarantine` rejection 1,714건과 LMCache
`EngineDeadError`/proxy `ConnectError` log를 artifact index에 SHA와 함께
등록했다. 따라서 C는 native failure-robustness evidence가 있는 validation
후보이지 성능 승리가 아니다. allocation `57404614`는 artifact 수집 후
종료했고, 추가 Slurm submit/retry는 하지 않는다.

Candidate D는 Candidate C의 safety boundary와 Candidate B의 proactive
queue/SLO scaling을 결합한 별도 frozen CPU profile이다. profile fingerprint는
`75bc2b6f76bded31f1582aac46e2d3594afdf4c79714b80535afa6987848ab18`이고, 동일
2,712-row trace에서 TEMPO-GO `2,433/279` complete/reject, local/remote
`1,958/475`, pair activation `1`을 기록해 C와 aggregate metric이 동일했다.
failure-injected replay도 정확히 한 failure receipt와 quarantine을 유지했다.
따라서 D는 pair assignment mechanism이 바뀌었지만 business metric 개선이 없는
neutral/negative candidate로 고정하며 native allocation에 올리지 않는다.

현재 Candidate B는 이 stop gate를 통과한 상태가 아니다. queue/SLO-risk trigger와
active-pair externality penalty는 CPU contract와 replay에서 닫혔지만, 같은
anchor trace의 aggregate reject/goodput이 Candidate A와 같았다. 따라서 현재
다음 선택지는 (1) failure-driven pair-health quarantine와 surviving-pair
reassignment를 실제 terminal event/telemetry contract에 연결해 Candidate B와
구조적으로 다른 후보를 만들거나, (2) B의 mechanism이 실제 native endpoint
completion pressure에서 aggregate business metric을 바꾸는지에 대한 새 frozen
independent validation contract를 먼저 만드는 것이다. 어느 경우에도 queue-GPU
failure arm을 latency 결과로 대체하거나, 기존 guard profile을 덮어쓰거나,
같은 native discovery를 맹목 반복하지 않는다.

### 18.1 held-out output geometry와 frozen proxy contract의 현재 closure

v3 builder의 `--background-output-tokens 128` metadata-only 결함을 고친
별도 builder `eval/sota_4node/build_tempo_go_c5_heldout_manifest.py`를
추가했다. 기존 v3 manifest/workload는 immutable하게 보존한다. 현재 held-out
artifact는 다음과 같다.

- manifest:
  `results/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json`
- manifest SHA:
  `6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c`
- validation workload:
  `results/tempo_go_c5_heldout_output128_v1/workloads/validation.jsonl`
- workload SHA:
  `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33`
- validator report SHA:
  `f00157c5f237c7a271197e499046e0e2a9884881cffeca46554accd015933fd0`

실제 JSONL row는 2,712개이며 foreground는 `(512,16)`, `(2048,256)`,
`(4094,16)`, 모든 decoder/remote/KV-hot stream은 output `128`이다. replicate는
`r02/r03`, cache contract는 MISS 1,992개(각각 token-preserving unique first
chunk)와 P_ONLY 720개다. phase는 C0 cool → C1 decoder-hot → C2 remote-hot →
C2 P_ONLY/KV-hot → C3 both-hot → recovery이고, 모든 decision policy input에서
phase/future/oracle/physical-switch label은 제외한다. 이 artifact는 workload
geometry/manifest validation closure이며 performance claim은 허용하지 않는다.

endpoint profile의 모든 17개 row가 P_ONLY인 현재 경계에서 frozen 경로가
`allow_service_proxy=True`를 discovery처럼 넓게 사용하는 것은 금지한다.
`tempo/pd_global_profile.py`의 `FrozenServiceProxyPolicy`는 다음을 immutable
profile fingerprint에 묶는다.

- endpoint profile ID/fingerprint와 calibration receipt SHA
- 허용된 `(prompt_tokens, output_tokens)` geometry
- 허용된 cache residency와 endpoint lookup mode
- `proxy_is_not_exact=true`, numeric row 불변, `performance_claim_allowed=false`

frozen global profile은 policy가 없으면 로드 단계에서 fail-closed한다. policy는
`GlobalOrchestratorConfig`로 전달되지 않으며, candidate builder가 geometry,
residency, lookup mode를 모두 allowlist 밖이면 중단한다. frontend와 CPU replay는
이 policy를 실제로 전달한다. 이 구현은 endpoint scope를 calibration-only에서
frozen-validation으로 자동 승격하지 않으며, exact MISS calibration 또는 policy
source receipt가 닫히기 전에는 native run contract/independent validation을
만들 수 없다. 이 source revision으로 기존
v1~v9 C5 contract는 stale이며 기존 contract를 overwrite하지 않는다. 현재 v10
normal discovery contract와 v11 quarantine-failure contract는 artifact index에
별도로 기록돼 있고 둘 다 performance claim을 금지한다. 관련 current bounded
suite는 `86 passed`이며, 이전 broader suite 기록은 `128 passed, 11 subtests
passed`다.

### 18.2 held-out v3 native five-arm discovery의 현재 해석

현재 native 기준은 v3 contract와 새 allocation `57407675`이다. v2와 이전
allocation은 historical/incomplete receipt로 보존하며 현재 결과로 섞지 않는다.

- contract: `results/tempo_go_c5_heldout_frozen_proxy_v3/native_run_contract.json`
- contract file SHA / fingerprint: `c280a889e148069b2678c53dc3cdb738219e6c6a64f80b9594b220c7d2f4f3f4` /
  `1fd9ff9f894b916a855c9aa93adb66a4a1bc4e1d05107cb09e690f300d857b73`
- result root: `results/tempo_go_c5_native_heldout_frozen_proxy_v3_job_57407675`
- analyzer SHA: `7bc286b5f1149c30aa54765665f40b70839abc779ea0968455b70f23ec31a8e8`
- topology: native Perlmutter 4 nodes / 16 A100 / official `LMCacheConnectorV1:UCX`
- arm order: `local → remote → predictor → queue_gpu → tempo`

| arm | terminal result | request goodput/s | E2E p50 / p99 (ms) | output-token goodput/s |
|---|---|---:|---:|---:|
| ALWAYS_LOCAL | 2,712 complete / 0 reject / 0 fail | 7.912 | 15,627 / 19,686 | 979.2 |
| ALWAYS_REMOTE | 2,712 complete / 0 reject / 0 fail | 9.628 | 12,104 / 16,545 | 1,191.4 |
| PREDICTOR_ONLY | 2,712 complete / 0 reject / 0 fail | 7.879 | 14,783 / 20,342 | 975.0 |
| QUEUE_GPU_ONLY | measured raw 없음, exit 143 | — | — | — |
| TEMPO_GO | 904 complete / 1,808 reject / 0 fail | 4.401 | 5,417 / 8,544* | 497.2 |

`*` TEMPO latency is completed-only and therefore not comparable as a same-population
latency win. TEMPO의 endpoint completion receipt는 904건, global scheduler
observation은 5,424건(2,712 payload × 2 pair), invalid snapshot은 0건, pair
activation은 1건이다. `QUEUE_GPU_ONLY`는 latency arm으로 대체하지 않고 execution
failure로 분리한다. raw SHA는 local `7ec171d8ac6c1e921b44adda524991a6c297b5ba3826e7f89967e942166821eb`,
remote `1dc08d8911c3e43eab6f2e0129cac2b75587c98bb589438098f4600775068e0`, predictor
`716034e0ad7bb8ae9f702be3b26d8ab332f25f9398bc994e77a3fd40b32f0bb5`, queue failure
`71b62581208b6ea3cb6a5a5bdd71aac3c52c923d92a33beb5c8d4c7ef5062357`, TEMPO
`fd29da960e68eadfe83f7ec24797c8b51461134b0021d7d0d0d9177c3ff835e0`이다.

이번 결과로 닫힌 것은 실제 vLLM P/D 경로에서 global admission, tenant ledger,
all-pair scheduler telemetry, endpoint completion receipt, logical pair activation과
명시적 reject가 발동한다는 사실이다. 닫히지 않은 것은 성능 승리, fairness 승리,
failure recovery 승리다. 특히 TEMPO의 reject reason은
`global_admission_queue_timeout` 1,801, fair-route commit 903, proactive queue
scale 1, telemetry refresh timeout 7이다. profile controller의 전역
`maximum_queue_wait_ns=2,000,000,000`과 coordinator의 동일 admission wait 상한이
모든 tenant의 대기를 자르므로, background 2,436건 중 1,746건이 reject됐다.
반면 실제 bounded queue wait p99는 0.482 ms이고 global admission decision wait
p99는 2,115 ms였다. 따라서 이번 receipt는 interconnect의 특정 link/NIC 병목을
증명하지 않으며, 먼저 admission policy의 business trade-off를 고쳐야 한다.

다음 candidate의 causal target은 route threshold가 아니다. (1) tenant별 SLO와
minimum service를 보존하면서 reject/defer budget을 숫자로 고정하고, (2) global
queue timeout을 하나의 전역 2초 값으로 적용하지 않고 tenant/business state와
pair capacity에 맞게 분리하며, (3) pair scale이 실제 queue/SLO risk를 줄이는지,
(4) endpoint failure/quarantine 후 surviving pair/reject receipt가 닫히는지를
CPU replay에서 먼저 검증한다. 그 뒤 새 profile/code/contract SHA를 freeze하고,
새 승인 native 4-node allocation에서 independent validation만 수행한다. 이 v3
discovery 자체는 `performance_claim_allowed=false`다.

### 18.3 held-out v2 native attempt의 정확한 해석

held-out output128 v1 contract는 현재 node entry source SHA와 달라 native node
계층에서 fail-closed했다. 이 실행은 다음 receipt로 보존하며 성능 evidence로 세지
않는다.

- v1 failure receipt:
  `results/tempo_go_c5_native_heldout_frozen_proxy_v1_job_57407196/local/failure.json`
  (SHA `9e85ae8c677cb79451c52a2a7b1c52dc1d758d302a26407c207e68b9d6c8db39`)
- failure: stale contract source inventory mismatch for
  `vllm_lmcache_tempo_go_c5_node.py`
- no measured request, no performance/fairness/robustness result

현재 node source를 포함하도록 별도 생성된 v2 contract는 immutable하게 유지한다.

- contract:
  `results/tempo_go_c5_heldout_frozen_proxy_v2/native_run_contract.json`
- contract file SHA:
  `b9e3a16d05ae2dcf420f00bda7b8bc6912cd4b13ed5dbc3f6c3275db7ec47aba`
- contract fingerprint:
  `4ad18da2c062dc4b0ad132d1c4be2501bdbe4c0bd1ef492a6eb611444eaf10b5`
- candidate revision: `frozen-proxy-v2-native-scope-fix`
- arm order: `local, remote, predictor, queue_gpu, tempo`
- output tokens: `128`; topology: 4 nodes/16 GPUs; transport:
  `LMCacheConnectorV1:UCX`

같은 allocation `57407196`에서 v2 local arm은 실제로 다음 단계까지 도달했다.

1. 네 node의 Python overlay/LMCache/vLLM process가 올라왔다.
2. official UCX backend와 LMCacheConnectorV1이 초기화됐다.
3. frontend/proxy `/health`가 응답했다.
4. canonical lifecycle이 warmup workload를 생성했다.

그 뒤 measured `raw.json`을 만들기 전에 수동 `INT`로 중단했다. 따라서 이를
“EngineCore가 영구적으로 startup hang했다”거나 LMCache data-plane failure라고
부르지 않는다. node log에서 02:20:29에 API server/LMCache connector가 초기화됐고,
frontend health는 200을 반환했다. `No available shared memory broadcast block`은
초기화 중간의 vLLM informational wait였으며, 이 receipt에서 원인이나 fabric
bottleneck을 단정하지 않는다.

v2 receipt와 raw evidence는 다음과 같다.

- failure receipt:
  `results/tempo_go_c5_native_heldout_frozen_proxy_v2_job_57407196/local/failure.json`
  (SHA `af223ad9a2fb5e8e922e3a12ac37aa2e42395b1b90720c482162784fb77e7c0d`)
- failure schema: `tempo-go-c5-native-arm-signal-failure-v1`
- arm: `local`; signal: `INT`; measured raw/result: 없음
- generated warmup JSONL: 2,712 rows, SHA
  `755e62d5fd3b8f649b84ee89cc4ce045a50067cf9adc77f50d11a2ee11d66a42`
- node-0 vLLM log SHA:
  `d215bbe2ae1d37aad24a335b7fbd1ddc66a83d6f49657b8e642b16eb7df4aaf5`
- frontend log SHA:
  `013ac99fe10f633c966f872b75751e1abedcfdbda4ca1d25551315a584b6dafb`

이 receipt는 campaign incomplete execution으로 분류한다. 기존 one-allocation
규칙에 따라 같은 allocation에서 blind retry하지 않았고, 별도 실행 중인 사용자
allocation을 이 작업이 임의로 점유하지 않았다. 다음 native 실행 전에는 warmup
단계가 실제로 P_ONLY seed subset만 전송하는지, warmup 완료 후 measured phase로
넘어가는 시간과 signal/timeout receipt를 별도 기록한 뒤 새 allocation에서 frozen
contract를 다시 검증해야 한다. 이 결과도 performance claim을 허용하지 않는다.

### 18.4 최신 source-rebound native v3 결과: global admission은 발동했지만 TEMPO-GO는 아직 실패

앞의 `57407705` held-out v3는 historical discovery로 보존한다. 현재 native 기준은
승인된 Perlmutter interactive allocation `57409956`에서 source inventory를 다시
결박한 v3 contract로 실행한 다음 결과다.

- contract: `results/tempo_go_c5_r8_16_20_20_contract_v3/native_run_contract.json`
- contract SHA / fingerprint: `002ee5424c9779b22d2cc622cb9143227f8370d03d6b22d0f3c9a560f153e481` /
  `7691d005cad942c26a9a8792cf1487431ce5c4f7abe43ebb7b409a2fef5a854e`
- result root: `results/tempo_go_c5_r8_16_20_20_native_job_57409956_v3`
- analyzer SHA: `b7e302ab1f893310602b491a8971138d3f4b3cd7fa906b4f7ce05848ac305f45`
- topology: native Perlmutter 4 nodes / 16 A100 / official `LMCacheConnectorV1:UCX`
- arm order: `local → remote → predictor → queue_gpu → tempo`
- profile: `results/tempo_go_c5_reservation_sweep_profiles/r8_16_20_20.json`
  (maximum queue wait 5 s; queue capacity 128; reservations latency 20,
  interactive 20, batch 16, background 8)

| arm | terminal result | request goodput/s | output-token goodput/s | E2E p50/p99 (ms) |
|---|---|---:|---:|---:|
| ALWAYS_LOCAL | 2,712 complete / 0 reject / 0 fail | 7.934 | 981.8 | 15,202.5 / 19,675.3 |
| ALWAYS_REMOTE | 2,712 complete / 0 reject / 0 fail | 9.581 | 1,185.7 | 11,090.4 / 20,559.8 |
| PREDICTOR_ONLY | 2,712 complete / 0 reject / 0 fail | 7.928 | 981.1 | 15,530.2 / 19,524.7 |
| QUEUE_GPU_ONLY | measured raw 없음, exit 143 | — | — | — |
| TEMPO_GO | 982 complete / 1,730 reject / 0 fail | 4.786 | 548.4 | 7,463.4 / 9,206.1* |

`*` TEMPO latency는 완료된 요청만의 completed-only 수치이므로 fixed arm과 같은
population의 latency 승리가 아니다. tenant별 complete/request는 background
`769/2,436`, batch `83/84`, interactive `80/96`, latency `50/96`이다. global
decision reason은 queue timeout 822, tenant reservation 897, fair-route commit
981, telemetry refresh timeout 11, proactive queue scale 1이다. endpoint completion
receipt 982건, valid scheduler observation 5,424건(2,712×2 pair), pair activation
1건, route local 908/remote 74가 닫혔다. bounded queue wait p99는 0.51 ms였지만
global admission wait p99는 5,070.85 ms였다.

이 결과가 증명하는 것은 native contention에서 global admission, tenant ledger,
scheduler/completion telemetry가 실제로 작동했다는 것과, 현재 queue reservation이
service feasibility를 보장하지 못했다는 점이다. reservation slot은 ingress queue
occupancy를 보호했지만 decoder/P/remote/endpoint capacity와 tenant SLO를 보장하지
못했다. 따라서 이를 fairness 또는 orchestration 성능 승리로 해석하지 않는다. 이
receipt만으로 interconnect의 특정 NIC/link가 병목이라고 주장하지 않는다.

`QUEUE_GPU_ONLY`는 client의 `http.client.IncompleteRead: 0 bytes read` 뒤 exit 143
execution failure로 닫혔으며 latency arm으로 대체하지 않는다. 최신 run에는
`udiRoot.conf` 오류, root ownership 변경, container 실행, exit 139가 없었다.

다음 causal candidate는 route threshold나 queue reservation sweep이 아니라
**admission-feasibility/service-lane lease**다. pair×route별 decoder/P/remote/
transfer/semantic/endpoint residual로 tenant deadline 안의 feasible finish를
계산하고, latency/interactive 보호 lane, batch/background elastic lane,
surviving-pair/spare-pair 재평가와 explicit `global_tenant_slo_infeasible`/
`global_service_lane_unavailable` terminal reason을 고정한다. 이 candidate가 CPU
fairness/SLO/overhead/failure gate를 통과하기 전에는 새 native allocation을
사용하지 않는다.

## 19. 다음 작업에 사용할 목표 프롬프트

아래 블록은 새 세션에서 그대로 붙여넣을 수 있는 canonical 목표
프롬프트다. 경로는 이 repository 기준 절대 경로로 유지한다.

```text
/goal

먼저 다음 두 파일을 전체 읽고, 그 내용과 claim/stop boundary를 변경하지
말고 작업하라.

1. /pscratch/sd/s/sgkim/Skim-Tempo/NERSC_AGENT_SAFETY.md
2. /pscratch/sd/s/sgkim/Skim-Tempo/paper/TEMPO_GLOBAL_ORCHESTRATOR_CANONICAL_PLAYBOOK.ko.md

현재 partial implementation이 이미 있다. `tempo/pd_global_orchestrator.py`,
`tempo/pd_global_telemetry.py`, `tempo/pd_global_workload.py`,
`eval/sota_4node/build_tempo_go_c5_manifest.py`,
`eval/sota_4node/validate_tempo_go_manifest.py`와 실제 router/frontend의
`/tempo/runtime_telemetry` 경로를 먼저 읽고, 이미 통과한 CPU contract를
되풀이해서 다시 만들지 말고 missing integration과 evidence만 닫아라. base
discovery profile fingerprint는
`383503d03ee5c5a45ff9540c6a8bf635a0807b5fc6bf10b3c4d2381baef31f0f`이고, 현재
C5 v3 bound global profile
`results/tempo_go_c5_anchor_priors_c12_v3_retry1/real_tempo_go_profile_c12_anchor_v3.json`
fingerprint는
`e30744d097ddf66095387e7478a48be88e89da6582c55ef84bfaa864a1f6f012`이다.

**최신 native correction:** 이 prompt 안의 이전 `57407705`/2초 admission 수치는
historical discovery로만 취급하라. 현재 source-bound native receipt는 allocation
`57409956`, contract
`results/tempo_go_c5_r8_16_20_20_contract_v3/native_run_contract.json` (SHA
`002ee5424c9779b22d2cc622cb9143227f8370d03d6b22d0f3c9a560f153e481`, fingerprint
`7691d005cad942c26a9a8792cf1487431ce5c4f7abe43ebb7b409a2fef5a854e`)와 result root
`results/tempo_go_c5_r8_16_20_20_native_job_57409956_v3`다. local/remote/predictor는
2,712 complete, request goodput 7.934/9.581/7.928 s⁻¹였고 queue-GPU-only는 exit
143 `IncompleteRead` execution failure였다. TEMPO는 982 complete/1,730 explicit
reject/0 failure, background 769/2,436, interactive 80/96, latency 50/96 complete로
`performance_claim_allowed=false`다. 5초 global cap과 queue reservation은 queue
occupancy만 보호했으므로, 다음 목표는 queue/reservation 숫자 sweep이 아니라
pair×route service-feasibility와 tenant service-lane lease다. 이 correction은 아래
historical paragraphs보다 우선한다.

retry6/retry9 이후 native receipt-closure에는 기존 v3 profile을 사용하지
않고 다음 immutable guard profile을 사용했다.
`results/tempo_go_c5_guard_profile_v3_retry1/real_tempo_go_profile_c12_guard_v3.json`
의 file SHA는
`8082f4190d56016d7bac6abacbf659017a4fb20a50d1b474223cf9157c1fd3ec`, global
fingerprint는
`f8163ff115a2478614afccf57b02a1c535c7dd4e2b3e54f47beda83d1ae3c2a0`다. 이
profile은 `remote_semantic_ops_safety_reserve=1`인 discovery/calibration
input이며 native 성능 evidence가 아니다. 이 profile은 먼저
`TEMPO_GO_C5_ARM_ONLY=tempo` 단일 arm에서 receipt closure를 통과한 뒤,
환경변수 `TEMPO_GO_GLOBAL_PROFILE`과 log/receipt fingerprint를 고정한
counterbalanced five-arm discovery에 사용됐다.

역사적 native partial receipt도 보존한다. `results/tempo_go_c5_native_five_arm_job_57395883_v3_retry3`
에서 `ALWAYS_LOCAL`, official always-remote, predictor-only는 각각 2712/2712
valid이고, queue-GPU-only는 LMCache cache-key assertion으로 EngineCore가
죽어 raw가 없다. TEMPO retry4는
`results/tempo_go_c5_native_tempo_only_job_57395883_v3_retry4`에서 exact
endpoint lookup bug로 중단됐고, retry5 raw는
`results/tempo_go_c5_native_tempo_only_job_57395883_v3_retry5/tempo/tempo_go_c5_discovery/raw.json`
에 있지만 1,029/2,712 decision만 있어 invalid하다. retry5에서 관측된
1,677 queue-timeout과 6 telemetry-timeout을 분리하고, 이 partial result는
재실행하거나 성능으로 해석하지 말고 SHA-bound raw를 먼저 분석하라. 현재
allocation `57400890`의 retry6 raw
`results/tempo_go_c5_native_five_arm_job_57400890_v3_retry6/tempo/tempo_go_c5_discovery/raw.json`
도 보존돼 있다. 2,712 attempted 중 833 valid이며, 1,783 telemetry refresh
failure 뒤 node-3 vLLM에서 official LMCache `CacheEngineKey ... not found in
local data` assertion/EngineCore death가 발생했다. structured queue reject 73건,
telemetry timeout 7건, upstream 502 4건도 별도 provenance로 남아 있다. retry6은
performance evidence가 아니며, retry9에서도 같은 failure shape가 다른 chunk에서
재현됐다. retry9 result root는
`results/tempo_go_c5_native_tempo_only_job_57400890_v3_retry9_quarantine`이고,
warmup만 24/24 valid이며 measured raw/result는 없다. node-3 log SHA는
`b09c0fe17ad1e4ddd0148b99c7ae30abae7c7bb8910de9558eeb4dc8e708473c`이다. 따라서
동일 workload를 단순 반복하지 말고 pair-health quarantine, surviving-pair
reassignment 또는 명시적 tenant-aware reject와 함께 remote semantic-operation/
cache-consistency pre-admission guard를 먼저 구현·검증하라. local fallback으로
실패를 숨기지 말고 guard 전후의 per-worker key/install evidence를 남겨라.

현재 Candidate C failure-quarantine profile은
`results/tempo_go_c5_failure_quarantine_profile_v1/real_tempo_go_profile_c12_failure_quarantine_v1.json`
(file SHA `1e3c861593a5bd802fa73aa5199657a6ec410d28ee9ebf8aad6970538e127cb6`,
fingerprint `6657de67fa75bef8241ffe4148126fe5977fe73da0312b801d7a1843a38849eb`)이다.
allocation `57404614`의 native C raw
`results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/tempo/tempo_go_c5_discovery/raw.json`
(SHA `c61626d6cef2b7353e0ec8a21609a9bc3b72ea6e4ed240ff5de2216cf9292124`)에는
2,712 rows, complete `1,633`, failed `9`, rejected `1,070`, global failure
receipt `9`, quarantine rejection `1,714`가 있다. pair-scope transport failure
3건과 route-scope HTTP failure 6건이 실제로 기록됐고, node-3 LMCache
`CacheEngineKey ... not found in local data`/`EngineDeadError`와 node-2
proxy `ConnectError`가 동반됐다. 그러나 native step은 exit `143`으로 종료되어
성능 `result.json`이 없으므로 이 artifact는 failure robustness evidence로만
사용한다. 같은 native run을 재시도하거나 이 raw에서 성능 승리를 추출하지 말고,
다음에는 이 profile을 기준으로 CPU contract와 independent validation contract를
먼저 고정하라.

guarded single-arm receipt
`results/tempo_go_c5_native_tempo_guard1_job_57400890_v1`은 이미 닫혔다:
2,712/2,712 terminal-valid, 1,904 complete, 808 explicit global reject,
`router_decisions_exact=true`, `terminal_contract_valid=true`, remote completed
281건, semantic-op guard candidate rejection 328건이며, retry6/retry9의 LMCache
assertion은 재현되지 않았다. 이 결과는 single-arm integration evidence일 뿐이다.
그 뒤 guard profile을 사용하는 counterbalanced five-arm discovery도
`results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1`에서 완료됐다.
이 single-arm 결과로 성능 승리를 선언하지 마라. `analyze_tempo_go_c5_five_arm.py`의 single-arm output schema는
`tempo-go-c5-native-single-arm-analysis-v1`이다.

현재 guard profile을 사용한 discovery도 이미 완료됐다:
`results/tempo_go_c5_native_five_arm_guard1_job_57402376_v1`이다. order는
`tempo,queue_gpu,predictor,remote,local`; local/predictor/remote는 각각
2,712/2,712 valid, queue-GPU-only는 `exit_code=143`의
`tempo-go-c5-native-arm-failure-v1` receipt이며 measured raw와 scheduler
observation이 없다. (TEMPO arm 자체의 global scheduler observation은 위의
provenance receipt로 닫혔다.) TEMPO는 1,865 complete/847 explicit reject로 p50 E2E
983.9 ms를 기록했지만 output-token goodput 136.9/s로 always-local 190.2/s보다
낮았다. 그러므로 이 run도 native descriptive discovery와 failure-isolation
evidence일 뿐이며, 다음 작업은 같은 workload를 맹목 반복하는 것이 아니라
reject/tenant fairness/pair scaling mechanism을 개선한 frozen candidate의
independent validation 또는 두 candidate negative 판정이다.

동일 allocation에서 후속 v7 시도는 네 node service readiness 뒤 내부 step이
`nid001168`에서 종료되어 raw/terminal ledger를 남기지 못했다. 이것은 v1의
guard receipt와 별개의 unreceipted harness failure이며, native performance나
LMCache failure 원인으로 분류하지 마라. 최신 runner의 signal-failure receipt
contract와 analyzer test는 이후 five-arm에서 실제로 사용됐고 queue-GPU-only
failure receipt가 생성됐다. 이번 discovery의 performance 수치를 재사용해
validation claim을 만들지 말고, 다음 native run은 frozen candidate와 새
independent allocation에서만 수행하라.

이번 guard revision의 구체적 계약은 다음과 같다. 기존 profile을 수정하지 말고
`build_tempo_go_discovery_profile.py --remote-semantic-ops-safety-reserve 1`로
새 immutable profile을 만든다. endpoint semantic-op window가 4이면 global
remote limit은 3이다. limit을 넘는 remote candidate는
`remote_semantic_ops_admission_guard`로 거절되거나 surviving pair의 local/remote
candidate로 재선정되어야 하며, silent recompute/fallback은 금지한다. 먼저
`tempo/test_pd_global_orchestrator.py`와 `tempo/test_pd_global_profile.py`를
포함한 bounded CPU suite를 실행하고, 그 다음에만 현재 승인 allocation 안에서
guarded native TEMPO arm을 실행한다. native run은 guard decision, pair health,
per-worker cache-key/install failure, telemetry failure, surviving-pair assignment,
terminal reject를 request ID 단위로 수집해야 한다. native receipt가 닫히기 전에는
성능 수치나 orchestration 승리를 주장하지 마라.

CPU에서 구현한 Candidate B도 후보 상태를 정확히 유지한다. profile은
`results/tempo_go_c5_queue_scale_profile_v1/real_tempo_go_profile_c12_queue_scale_v3.json`
(file SHA
`b705c4688a6061d3025a0e63a56a4edaa32e9384dee4edfbcb281fada5195b33`, global
fingerprint
`507d415764ef2dde8661d3516c08fb51aca6643bdca4ce3e8a29e82559eb55f3`)이고,
queue/wait fraction `0.25/0.25`, active-pair penalty `25 ms`인 queue/SLO-risk
proactive scaling candidate다. bounded suite는 123 passed이며 CPU replay
`results/tempo_go_c5_queue_scale_replay_20260821_v2.json` SHA는
`f0c74068a7aa5e187e49b44460a404e0a2c33a6623a02b4d4ada2fbd0f2b9ed2`다. B는
guard A보다 pair1 activation을 앞당겼지만 aggregate complete/reject는
`2,433/279`로 같았고, 이 결과는 performance win이 아니다. B를 native에
맹목 실행하지 말고, 먼저 failure-driven quarantine/reassignment 후보 또는
실제 endpoint completion pressure에서 B가 business metric을 바꾼다는 frozen
contract를 만들어라.

최신 CPU 상태도 반영하라. `results/tempo_go_c5_fairscale_profile_v1/`
`real_tempo_go_profile_c12_fairscale_v1.json`은 queue/wait fraction `0.5/0.5`의
immutable B-fairscale variant이며 file SHA는
`f0caa6d73a77b35235035ba3247f79dc6e66ade2a71295713f3fe75fc7f9ca95`, global
fingerprint는
`175f31f9db13a9ce5bd45aaf95574f38483a5ca36db651d6c4b050301df27e8e`다. fairness
accounting 교정 후에도 동일 v3 trace에서 baseline TEMPO-GO가 `2,461/251`
(complete/reject), SLO-goodput `2,238`이고 fairscale이 `2,433/279`, `2,202`로
악화되어 B-fairscale은 negative candidate로 고정됐다. 관련 receipts는
`results/tempo_go_c5_fairness_fix_v1/` 아래에 있으며 두 replay 모두
`performance_claim_allowed=false`다. `_tenant_virtual_service`의 weighted
debt와 raw service units를 섞지 말고, coordinator admission wait는 반드시
`min(global maximum_queue_wait, tenant maximum_queue_wait)`를 사용하라. 이
correctness fix를 성능 승리로 보고하지 말고, 다음 candidate는 failure-driven
pair-health quarantine/surviving-pair reassignment처럼 B와 causal mechanism이
다른 방향으로 설계하라.

현재 Candidate C의 frozen CPU profile은
`results/tempo_go_c5_failure_quarantine_profile_v1/real_tempo_go_profile_c12_failure_quarantine_v1.json`
이며 file SHA `1e3c861593a5bd802fa73aa5199657a6ec410d28ee9ebf8aad6970538e127cb6`,
global fingerprint `6657de67fa75bef8241ffe4148126fe5977fe73da0312b801d7a1843a38849eb`다.
이 profile은 `route_failure_quarantine_mode=deny_until_probe`와 semantic-op
reserve `1`을 사용한다. `report_route_failure`/`fail_route` 경로는 failure
receipt, released work, surviving-pair dispatch, pair-scope quarantine와 PROBE
recovery를 CPU에서 검증했고, failure-injected C5 replay에서도 실제 trace event
순서의 remote failure receipt와 pair-0 route quarantine을 확인했다. 다만 native
failure receipt는 성능 결과가 아니라는 조건부로 확보됐다. allocation
`57404614`의 native C arm은 2,712-row raw를 만들었고, terminal phase는
complete `1,633`, failed `9`, rejected `1,070`이었다. 그중
`frontend_tempo_go_failure` receipt는 9건(transport pair-scope 3건, HTTP
route-scope 6건)이며 `route_failure_quarantine` rejection은 1,714건이다.
pair-scope failure는 해당 pair의 local+remote route를 함께 격리했고, route-scope
failure는 해당 route만 격리했으며, 실패한 request ID를 같은 ID로 재시도하지
않는 receipt와 released work가 raw에 남았다. node-3 vLLM log에는 official
LMCache `CacheEngineKey ... not found in local data` 뒤 `EngineCore`/
`EngineDeadError`가, node-2 proxy log에는 `httpx.ConnectError`가 남았다.
따라서 native C는 “실패가 발생하면 global failure/quarantine ledger가 실제
경로에서 관측된다”는 robustness evidence지만, step exit `143`으로 종료되어
`result.json`과 독립적인 성능 비교를 만들지 못했다. 이 raw를 TEMPO 승리나
pair scaling 승리로 해석하지 말라. native에서는 실제 upstream failure가
발생할 때만 이 path를 사용하고, local fallback으로 failure를 숨기거나 같은
request ID로 재시도하지 말라. route failure가 없는 run은 C mechanism의
negative/neutral observation으로 기록한다.

이 C artifact를 사후 분석하는 과정에서 analyzer의 raw-backed execution-failure
경로도 보강했다. 이전 analyzer는 `failure.json`만 있는 arm을 zero-request
summary로 반환해, 같은 arm directory에 보존된
`tempo_go_c5_discovery/raw.json`의 terminal ledger와 global failure receipt를
분석하지 못했다. 현재 `eval/sota_4node/analyze_tempo_go_c5_five_arm.py`는
top-level native failure receipt와 raw SHA/workload identity를 함께 검증하고,
`router_decisions_exact=false`인 경우에도 execution failure로 명시된 때에만
request/decision coverage, semantic terminal phase, failure kind/scope,
tenant/phase service metrics, pair activation과 telemetry를 계산한다. 이
경로는 성공 gate를 완화하지 않으며 `performance_claim_allowed=false`를
유지한다.

기존 `57404614` raw에 대한 새 CPU-side analysis artifact는
`results/tempo_go_c5_native_failure_quarantine_job_57404614_v1/native_c_analysis_raw_backed_v1.json`
(SHA `579f92d38140f0f7ccb31f18a19ce9c9670ea5b3371ba48e99cf7850dbd3a1ac`)이다.
semantic phase `1,633/9/1,070`, global failure receipt `9`(pair `3`, route
`6`), service-metric completed `1,623`/failed `9`/rejected `1,070`을 raw에서
복원했지만, 이는 native rerun이나 performance validation이 아니다.

Candidate D는 이 frozen C profile에 B의 queue/SLO proactive scaling을 결합한
CPU-only negative/neutral 후보이다. profile file SHA는
`d8bb3e893fa3279e004e020c2dcf1e34bf7af46dd0ff1d4527863a49816f566d`, global
fingerprint는 `75bc2b6f76bded31f1582aac46e2d3594afdf4c79714b80535afa6987848ab18`다.
same-trace replay SHA는
`b9567186c224a41a74bedf8744e0a797ba4a0c7838908574bb5c4e0dee9f97777b9`이고,
failure-injected replay SHA는
`fa959472271982f9d6f6f48ab282922c4d799bf24a6236a13f5867e521c70b4e`다. D는
C와 동일한 `2,433/279` complete/reject와 SLO/E2E를 내고 pair distribution만
바꿨으므로 native에 올리지 않는다. 다음 작업자는 D를 성능 후보로 부활시키지
말고, 이미 확보한 C native failure receipt와 D CPU neutral result를 사용해
새로운 causal mechanism 또는 두 후보의 독립 negative gate를 설계하라.

목표는 기존 route-only Elastic-PD threshold를 더 튜닝하는 것이 아니다.
Perlmutter native 4-node/16-A100의 실제 vLLM P/D와 official
LMCacheConnectorV1:UCX data plane을 그대로 유지하면서, moving multi-tenant
contention 아래 decoder admission, tenant SLO/fairness, pair assignment와
logical active-pair scaling, local/remote route, endpoint congestion recovery를
공동 제어하는 TEMPO-GO global orchestrator를 구현하고 검증하라.

이미 확정된 사실을 다시 처음부터 탐색하지 마라.

- C1/C2/C3 actual-inference workload에서 local/remote opposite crossover는
  이미 검증됐다.
- P_ONLY remote path는 약 12 req/s에서 drain knee가 시작되고 achieved rate가
  약 9.7 req/s에서 포화된 기존 evidence가 있다.
- C4 route-only Candidates A/B/C와 phase oracle은 median과 TPOT/worst-tail을
  동시에 만족하지 못해 중단됐다.
- 따라서 prompt coefficient, scalar fabric_pressure, phase classifier와
  request-local route threshold family를 되살리지 마라.
- 현재 TEMPO-GO v9/output128 결과는 global wiring smoke evidence일 뿐
  performance evidence가 아니다. held-out output128 validation artifact는
  `results/tempo_go_c5_heldout_output128_v1/`에 고정되어 있고, 현재 native
  기준 contract는
  `results/tempo_go_c5_heldout_frozen_proxy_v3/native_run_contract.json`
  (file SHA
  `c280a889e148069b2678c53dc3cdb738219e6c6a64f80b9594b220c7d2f4f3f4`,
  fingerprint
  `1fd9ff9f894b916a855c9aa93adb66a4a1bc4e1d05107cb09e690f300d857b73`)다.
  v1 contract는 node-entry source inventory mismatch로 fail-closed된
  historical attempt이며 재사용하지 않는다. v2는 실제 vLLM/LMCache/UCX와
  frontend/proxy health 및 warmup 생성까지 도달했지만 measured raw 전에 INT로
  중단됐으므로 아직 independent validation이 아니다.

우선순위는 다음과 같다.

1. v2 manifest
   `results/tempo_go_c5_cpu_gate_20260821_anchor_v2/tempo_go_workload_manifest.json`
   (SHA
   `3298cbe86e0684e5810f5f5981acb1dd2d20a471943a3f9d0766e911643992d8`)와 v3
   anchor manifest는 historical evidence로만 보존한다. v2 native retry3의
   repeated-MISS 502 raw와 기존 v3 raw를 성능 수치로 분석하지 않는다. 현재
   independent-validation 실행 입력은 held-out manifest
   `results/tempo_go_c5_heldout_output128_v1/tempo_go_workload_manifest.json`
   (SHA
   `6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c`)와
   validation workload
   `results/tempo_go_c5_heldout_output128_v1/workloads/validation.jsonl`
   (SHA
   `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33`)다.
   실제 2,712 rows, r02/r03, output=128, foreground geometry
   `(512,16)/(2048,256)/(4094,16)`, MISS 1,992/P_ONLY 720을 사용한다.
   held-out v3 global/endpoint/profile/contract binding과 sidecar SHA가 모두
   일치하기 전에는 새 C5 candidate를 실행하지 않는다. v2는 historical
   incomplete attempt로만 보존한다. 기존 C4 replay를 새 global
   five-arm replay로 간주하지 말고, C1/C2 anchor output=2와 production
   output=128을 별도 evidence로 유지한다.
   canonical source workload SHA와 arm wrapper가 생성한 rewritten client
   workload SHA를 receipt에서 혼동하지 마라. 둘을 모두 기록·검증하되,
   manifest/profile binding에는 canonical source SHA만 사용한다.
2. 구현된 tenant별 weight, TTFT/TPOT/E2E SLO, max queue wait, minimum service
   contract와 `overload_action=reject_new_request`가 workload/profile/decision
   provenance에 일치하는지 검증한다. CPU replay에서 이미 닫힌 reject/queue
   terminal accounting을 되돌리지 말고, native admission/fairness 결과를
   측정한다.
   기존 v3 native local/remote/predictor raw의 output hash, p50/p95/p99,
   tenant SLO-goodput과 queue/failure provenance를 먼저 집계한다. queue-GPU-only
   LMCache assertion은 failure로 집계하고 latency로 대체하지 않는다.
3. 구현된 frontend reservation과 actual vLLM scheduler state 분리를 native
   endpoint에서 확인하고, P service,
   actual transfer completion/bytes, semantic operation, receiver/install service
   residual을 provenance와 함께 all-pair causal telemetry로 수집한다.
   scheduler/completion receipt가 없는 native run은 성능 결과로 해석하지
   않는다.
4. 구현된 tenant 선택의 deadline/age와 weighted dominant-resource service를
   replay에서 검증하고,
   pair×route 선택은 shared decoder externality와 endpoint completion pressure를
   같은 admission transaction에서 평가한다.
5. 구현된 pair1 proactive activation이 predicted dominant pressure,
   queue/SLO risk 또는 route health를 근거로 proactive하게 logical activate한다.
6. one-way route commit, UNKNOWN fail-closed, first-response endpoint credit
   release, EOF decoder release, bounded queue와 exact terminal ownership은
   보존한다.
7. ALWAYS_LOCAL, OFFICIAL_LMCACHE_ALWAYS_REMOTE, PREDICTOR_ONLY,
   QUEUE_GPU_ONLY/Kairos-like, TEMPO_GO를 같은 server epoch와 counterbalanced
   workload에서 비교한다. 구현된
   `eval/sota_4node/run_tempo_go_c5_five_arm_in_allocation.sh`와
   `eval/sota_4node/analyze_tempo_go_c5_five_arm.py`를 사용하고, 기존 route-only
   analyzer 결과를 재활용해 성능 승리를 선언하지 않는다.
8. correctness, workload validity, per-tenant SLO/fairness, pair scaling,
   telemetry overhead, E2E/TTFT/TPOT/goodput와 selected-route counterfactual을
   모두 analyzer에 넣는다.

실행 규칙:

- GPU/vLLM/LMCache/traffic은 login node가 아니라 native interactive allocation
  안에서만 실행한다.
- 4 nodes, 16 GPUs, 최대 4시간 allocation 하나를 재사용하고 자동 submit/retry
  loop를 만들지 않는다.
- container, Shifter, Apptainer, Podman, Docker, --image, udiRoot, sudo, su,
  root ownership 변경, setcap, CAP_NET_ADMIN, /etc·/usr·/opt 수정은 금지한다.
- `udiRoot.conf must be owned by user root` 또는 container launcher 흔적이
  나오면 즉시 중단하고 command/environment/log만 보존한다.
- privileged physical-NIC control 없이 application-visible endpoint telemetry만
  사용한다. physical switch bottleneck을 추측하지 않는다.
- shared filesystem을 unbounded recursive search하지 않는다.
- dirty worktree의 사용자 변경을 보존하고 관련 파일만 수정한다.
- commit/push/PR은 사용자가 그 turn에서 명시적으로 요청한 경우에만 한다.

실험 순서는 G0 native identity -> G1 sensor closure -> G2 fixed-path workload
confirmation -> G3 offline replay/CPU invariants -> G4 one-allocation discovery ->
G5 freeze -> G6 independent validation -> G7 SHA-bound report다. 현재 held-out
CPU replay와 static runner/analyzer gate는 통과했고, v1 native-invalid retry와
TEMPO retry4/retry5/retry6/retry9 invalid receipts는 폐기하지 말고 보존했다. G4
discovery의 기존 five-arm run은 네 arm의 valid raw와 queue-GPU-only failure
receipt를 확보했지만 queue baseline이 clean하지 않아 G5 freeze와 G6 independent
validation은 아직 시작하지 않았다. 추가로 allocation `57404614`에서 Candidate C
failure-quarantine arm을 실제 native 경로에 연결해 9건의 global failure receipt를
확보했으나, step exit `143`으로 끝나 성능 `result.json`은 만들지 못했다. 이
실행은 failure robustness evidence로만 보존한다. receipt-closure retry6에서는
10건의 telemetry timeout이 decision ledger에서 빠졌고, post-patch retry7은
workload 진입 전 node-1 vLLM child SIGBUS로 종료됐다. 같은 allocation의 retry9는
native readiness를 통과했지만 measured phase에서 LMCache key assertion으로
다시 EngineCore가 죽었다. allocation `57402376`과 `57404614`는 모두 종료됐고
login node에서는 code/CPU/replay/receipt analysis만 수행한다. held-out v2 native
attempt는 allocation `57407196`에서 contract verification, four-node
vLLM/LMCache/UCX initialization, frontend/proxy health, warmup generation까지
도달했으나 measured raw 전에 manual INT가 들어간 incomplete execution이다.
따라서 startup hang, LMCache data-plane failure 또는 fabric bottleneck으로
분류하지 말고, warmup 완료를 기다릴 수 있는 새 user-approved 4-node/4-hour
interactive allocation에서 v2 contract를 그대로 재검증한다. 이미 실행 중인
allocation은 소유자와 목적을 확인하지 않고 점유하거나 재사용하지 않는다.
queue-GPU baseline과 TEMPO reject/fairness 비용을 개선한 frozen candidate를
먼저 만들고, 그 contract가 독립적으로 고정된 뒤에만 independent validation을
수행한다. workload crossover와 telemetry closure가 되기 전에 controller
threshold를 튜닝하거나 유효 arm을 반복 실행하지 마라.

기존 primary gate를 유지하라: strongest fixed 대비 E2E median 10%, predictor
대비 5%, goodput 5%, paired wins 75% overall/60% each group, p99/TPOT regression
5% 이내, worst paired regression 100 ms 이내, selected route counterfactual
gain 5% 이상. 또는 normal-load regression 3% 이내에서 overload p99/goodput
15% 이상과 starvation 0을 만족하는 robustness gate를 통과해야 한다.

완료는 frozen independent validation win 또는 두 구조적으로 다른 global
candidate의 재현 가능한 negative conclusion이다. 단순 integration 성공,
remote/pair1 한 번 선택, LMCache 대비 win만으로 완료하지 마라.

작업 중에는 현재 단계, 사용한 기존 evidence, 새로 바꾼 causal mechanism,
실패한 gate와 다음 stop/go 판단을 짧게 계속 보고하라.
```

## Addendum: Candidate I CPU gate와 native stop/go (2026-08-22)

Candidate E/G/H 이후의 다음 구조적 후보는 telemetry failure delta 기반 pair
pre-admission circuit과 surviving-pair service lane이다. `deny_until_probe`와
pair scope에서 누적 local/remote failure counter 증가를 관찰하면 pair의 두
route를 `telemetry_failure_delta` trigger로 quarantine하고, 살아남은 pair 용량의
25%를 weight 2.0 미만 tenant의 burst가 소비하지 못하게 한다. recovery는 더 최신
sequence의 명시적 PROBE에서만 허용한다. one-way route, 독립 endpoint/decoder
credit, first-response/EOF 반환, official `LMCacheConnectorV1:UCX` 계약은 변경하지
않았다.

Candidate I CPU artifacts:

- profile SHA: `9fd212df642124c28982888cffd4506a1680d8a4ac70ea9944b18663a74ee10c`
- normal replay SHA: `3db5e131fa07fa5da723200852d407b2ea4c71d094908d772f981ccccd36e18a`
- telemetry-failure replay SHA: `b95fae11c24ae55a6d6219864cfa3db1efc7bbb86557b4b26c78332a4759db1c`
- control-plane overhead SHA: `b475e57710230ac77518c57eddfc77e947c773a1b67b2b9db120e1432aceeadf`
- Candidate I run-contract SHA/fingerprint:
  `cab8942c74563552642278eb3c0f6aeb1fcbc7a72e3fa1a67461df230d538d5d` /
  `ccb424d40e9fbd47060416599ee6f7351a68993c7c500b93e103f65439977826`

동일 held-out 2,712-request replay에서 normal TEMPO-GO는 ALWAYS_LOCAL과
`1321 complete / 1391 reject`, E2E p50/p99 `5321.02/5913.90 ms`로 정확히
중립이었다. index 800에서 pair 0의 remote failure delta를 주입한 replay에서는
pair 0 quarantine과 pair 1 survivor가 실제 발동했지만 TEMPO는 `912 complete /
1800 reject / 0 failed`, background `673/1763`, latency `59/37` complete/reject로
utility robustness gate를 통과하지 못했다. 이는 failure containment mechanism의
작동 증거이지 performance/fairness 승리가 아니다.

CPU-only overhead도 별도 측정했다(200 warmup, 2,000 samples). baseline 대비
Candidate I control-plane total p50/p99는 `170.748/236.081 us`에서
`180.226/264.013 us`, telemetry refresh p50/p99는 `39.634/59.772 us`에서
`43.692/67.205 us`였다. overhead gate 자료로만 사용하고 GPU/network/LMCache
latency로 해석하지 않는다. source revision을 반영한 Candidate I contract와
회귀 결과는 `133 passed, 11 subtests passed`이다.

판정: Candidate I는 CPU correctness/overhead 기록은 통과했지만 normal performance는
neutral이고 failure robustness utility gate는 실패했다. 그러므로 native 4-node
validation은 STOP이다. 다음 후보를 만들더라도 같은 trace의 reject 비용, tenant별
SLO-goodput/minimum service, survivor reserve의 실제 이득, normal regression 3%,
control-plane overhead를 하나의 frozen contract에 숫자로 고정하고 CPU gate부터
다시 통과해야 한다. 기존 G/native receipt를 수정하거나 Candidate I를 native에서
맹목 반복하지 않는다.

## Addendum 2: strict CPU negative completion audit (2026-08-22)

Candidate G와 Candidate I를 현재 source-bound contract로 다시 replay하고, 두
candidate가 같은 preregistered primary gate를 실패하는지 machine-check했다.
Audit script는
`eval/sota_4node/audit_tempo_go_cpu_negative.py` (SHA
`5ecfbe04b3f5c02c91c449149acdd36b70843a74ee888336dd582c1a33f59897`)이며 결과는
`results/tempo_go_c5_candidate_i_telemetry_survivor_v1/cpu_negative_audit_v1.json`
(SHA `aed33cc340c27e2688de0dfb001009182f37236b291ebdb25399e4bd78358925`)이다.

두 replay는 동일 held-out manifest/workload/Elastic/endpoint/baseline identity,
동일 fixed-arm receipt fingerprint, 2,712 rows, no phase/oracle/physical-switch
input, terminal/leak-free와 current run-contract verification을 모두 통과했다.
G는 tenant queue reservation 16 slots이고 I는 telemetry failure pair circuit과
survivor reserve이므로 causal mechanism이 다르다. strongest fixed인
QUEUE_GPU_ONLY의 p50 `5297.970741 ms`에 primary 10% gate를 적용한 limit은
`4768.173667 ms`, predictor p50 `5344.028590 ms`의 5% gate limit은
`5076.827160 ms`다. G p50은 `6298.891169 ms`, I p50은 `5321.022663 ms`로
둘 다 같은 primary median gate를 실패했다. I의 contract-bound telemetry failure
replay도 pair quarantine을 실제 발동했지만 `912/1800/0` complete/reject/fail,
p99 `7549.510599 ms`로 robustness utility gate를 실패했다.

이 negative conclusion의 범위는 CPU control-plane promotion gate다. 이것으로
native latency/goodput이 음성이라고 주장하지 않는다. 결과 JSON의
`native_performance_negative_proven=false`, `performance_claim_allowed=false`,
`completion_status=CPU_negative_only_native_validation_unproven`을 보존한다.
따라서 canonical 완료 대안은 “두 구조적으로 다른 global candidate의 frozen
pre-native gate negative를 재현했으므로 추가 threshold search와 blind native
retry를 중단한다”로 닫히며, native independent validation win은 발생하지
않았다. native integration/failure receipt는 실제 vLLM/UCX contention과
control-plane wiring 증거로만 유지한다.
