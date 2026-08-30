# TEMPO-GO: 4노드 글로벌 contention orchestrator 연구 목표

> 이 파일은 실행 목표의 호환용 사본이다. 현재 연구 상태·계보·workload·증거
> 경계·Perlmutter 안전 규칙·다음 agent prompt의 authoritative single source는
> [`paper/TEMPO_RESEARCH_HANDOFF_AND_IMPROVEMENT_GOAL.ko.md`](../../paper/TEMPO_RESEARCH_HANDOFF_AND_IMPROVEMENT_GOAL.ko.md)다.
> 두 문서가 다르면 handoff 문서를 따른다.

## 목표

기존의 request-local Elastic-PD 선택기를 최종 목표로 삼지 않는다. 목표는
Perlmutter native 4노드/16-A100에서 실제 vLLM P/D와
`LMCacheConnectorV1:UCX` data plane을 유지한 채, 여러 tenant가 동시에
요청을 보내 local decoder, remote prefill endpoint, pair별 GPU queue와
interconnect를 함께 압박하는 상황에서 하나의 글로벌 orchestrator를
검증하는 것이다.

정확한 연구 질문은 다음과 같다.

> 동일한 4노드 P/D 배치와 official LMCache data plane에서, TEMPO-GO가
> moving multi-tenant contention을 관찰하고 decoder admission, tenant
> fairness, pair assignment/scaling, local/remote route를 공동 제어하여
> strongest fixed policy와 predictor-only보다 latency/goodput과 overload
> robustness를 동시에 개선하는가?

단일 client의 여유 있는 interconnect에서 local/remote를 고르는 실험은
주장 근거가 아니다. 실제로는 모든 pair의 decoder와 endpoint가 동시에
바쁘고, 어느 queue 또는 fabric path가 병목인지 사전에 알 수 없다는
상황을 기본 workload로 삼는다.

## 왜 필요한가

기존 실험에서 TEMPO의 개선 대부분은 remote가 불필요한 요청을 local로
보낸 효과였고, predictor-only와의 차이는 작았다. 이 결과는 predictor를
조금 더 조정할 문제가 아니라, 다음 자원이 서로 독립적이지 않다는
증거로 해석한다.

| 동시에 경쟁하는 자원 | local 경로 | official remote 경로 |
| --- | --- | --- |
| decoder work | decode tokens, active sequences | decode tokens, active sequences |
| endpoint work | local prefill token-time, endpoint slot | remote prefill token-time, endpoint slot |
| transfer | 없음 | KV bytes, semantic transfer operations |
| shared effects | GPU scheduler/queue | GPU scheduler/queue + UCX/NIXL fabric |

LMCache는 data plane 자체가 잘못되어서가 아니라, remote 요청이 decoder
admission이나 endpoint queue와 분리되어 몰리면 tail과 queue stability에서
불리해질 수 있다. 따라서 LMCache를 교체하는 연구가 아니라, 그 위에서
어떤 요청을 언제 어느 pair에 admission할지를 globally 조절하는 연구로
범위를 고정한다.

현재까지의 결론 경계도 고정한다. C1/C2/C3에서 local/remote의 crossover와
moving contention은 확인됐고, P_ONLY에서는 remote가 유효한 regime도 있다.
반대로 C4 route-only controller는 shared decoder tail과 tenant business
metric을 함께 제어하지 못해 최종 후보가 되지 못했다. Native guarded C5
discovery는 global admission/reject/endpoint completion provenance를
연결했지만 TEMPO output-token goodput이 always-local보다 낮아 성능 승리가
아니다. `QUEUE_GPU_ONLY`는 realistic contention 중 official LMCache
cache-key/EngineCore failure로 latency baseline이 되지 못했다. Candidate B의
proactive queue/SLO pair scaling도 CPU mechanism은 보였지만 aggregate
complete/reject와 SLO-goodput을 개선하지 못했다.

Candidate C는 명시적인 endpoint/upstream failure를
`tempo-go-global-failure-v1` receipt로 닫고 route/pair를 PROBE 전까지
quarantine하는 구조 후보이다. C5 동일 trace의 failure-injected CPU replay는
TEMPO-GO에서 remote failure receipt 1건, pair-0 remote quarantine, 이후 같은
pair remote admission 0건, 모든 resource/terminal invariant 통과를 확인했다.
이것은 control-plane robustness evidence이며 native failure receipt나 성능
증거가 아니다. 따라서 목표는 여전히 “global orchestrator를 native
contention에서 검증”하는 것이고, route threshold를 계속 미세조정하는 것이
아니다.

## 고정된 실험 범위와 금지선

- Perlmutter native allocation만 사용한다: 4 nodes, 16 GPUs, 4시간
  interactive allocation 안에서 모든 GPU/vLLM/LMCache/traffic workload를
  실행한다. login node에서는 코드 정리와 가벼운 검증만 한다.
- 2개의 prewarmed P/D pair를 사용한다. pair 0은 node 0/1, pair 1은
  node 2/3의 실제 vLLM P/D 경로로 구성한다. pair switching은 HTTP
  request-start 전에 commit하며 prefill 이후 route를 바꾸지 않는다.
- P/D transport는 official `LMCacheConnectorV1:UCX`로 고정한다. native
  NIXL 대체, Mooncake data plane, token-level decode hook, global fence,
  busy polling, synthetic sidecar traffic은 TEMPO-GO 최종 경로에서
  사용하지 않는다.
- container, Shifter/Apptainer/Podman/Docker, `--image`, udiRoot,
  `sudo`/`su`, root ownership 변경, `/etc`/`/usr`/`/opt` 수정,
  `CAP_NET_ADMIN`과 privileged NIC control은 절대 사용하지 않는다.
  `udiRoot.conf must be owned by user root`가 나오면 launcher가 native
  계약을 위반한 것이므로 즉시 중단하고 command/environment/log만
  보존한다. ownership을 바꾸거나 우회하지 않는다.
- interconnect는 애플리케이션에서 읽을 수 있는 endpoint/queue/transfer
  telemetry로만 추론한다. physical switch label, privileged NIC counter,
  미래 arrival, benchmark phase label, oracle state를 정책 입력으로
  넣지 않는다.

## 제안하는 글로벌 scheme

### 1. Causal global state

frontend가 request-triggered, bounded, single-flight poll을 수행하여
모든 pair의 state를 하나의 sequence로 묶는다. frontend ledger와 각
pair router의 endpoint controller snapshot이 같은 collection interval에
있지 않거나 profile/generation이 다르면 전체 batch를 폐기한다.

각 pair에서 유지하는 semantic resource는 다음과 같다.

- decoder: `decode_tokens`, `active_sequences`
- endpoint: `endpoint_requests`
- local prefill: `local_prefill_token_ms`
- remote prefill: `remote_prefill_token_ms`
- fabric/transfer: `remote_kv_bytes`, `remote_semantic_ops`

controller가 이미 소유한 값과 endpoint가 관찰한 값은 합산하지 않고
resource별 `max(owned, observed)`를 사용한다. 이 규칙이 없으면 같은
remote request를 두 번 세어 global admission이 실제보다 보수적이거나
불안정해진다. frontend의 monotonic clock interval start만 사용하고
노드 간 monotonic timestamp를 빼지 않는다. telemetry가 stale, partial,
out-of-order, identity mismatch이면 fail-closed queue/deny한다.

### 2. Request-level immutable route commit

각 request는 tokenization/cache evidence와 frozen profile을 이용해
`pair × {LOCAL, REMOTE}` 후보를 만든다. 후보에는 route별 predicted
TTFT/E2E, uncertainty, cache affinity와 위 resource vector가 포함된다.

- `UNKNOWN` cache는 hit로 취급하지 않는다.
- remote는 exact evidence row가 있고 `MISS` 또는 confirmed `P_ONLY`일
  때만 허용한다. 긴 prompt라고 remote를 강제하지 않는다.
- candidate가 global admission을 통과하면 pair와 route를 HTTP provenance
  header와 request ledger에 commit한다.
- prefill 시작 후 local/remote 전환, hidden recompute, silent fallback,
  retry를 통한 route 변경은 금지한다.
- first response에서 endpoint/prefill/transfer credit을 반환하고,
  HTTP EOF에서 decoder credit을 반환한다. abort/timeout/failure는
  모든 held credit을 정확히 한 번 반환한다.

### 3. Decoder admission과 tenant fairness의 공동 제어

단순 request-count cap을 사용하지 않는다. global queue는 weighted
deficit/virtual-service 순서로 tenant를 선택하고 dominant resource
pressure를 함께 본다. tenant weight가 큰 tenant가 더 많은 service를
받되, 한 tenant가 decoder 또는 remote bytes를 독점하지 못하도록
admission은 모든 resource capacity를 동시에 만족해야 한다.

다음 invariant를 frozen validation에서 직접 검사한다.

- bounded queue 외의 unbounded buffering 없음
- tenant별 maximum wait/deadline와 weighted service 추적
- 어떤 tenant도 다른 tenant의 지속적인 load 때문에 starvation하지 않음
- terminal request마다 route decision, first-response/EOF 또는 failure가
  하나씩 존재
- credit underflow, leak, double release가 0

### 4. Pair assignment와 active-pair scaling

4노드 topology에서는 process를 매 요청마다 재시작하지 않는다. 두 pair는
미리 준비하고 global controller가 traffic admission 관점에서 active
pair set을 조절한다.

- low load에서는 minimum active pair만 사용하여 cache affinity와 local
  queue locality를 보존한다.
- active pair의 dominant utilization 또는 queue pressure가 threshold를
  넘으면 prewarmed spare pair를 activate한다.
- route-specific bottleneck 때문에 pair를 activate할 수 있지만, global
  telemetry가 stale이면 activation을 추측하지 않는다.
- idle pair는 모든 held resource가 0이고 hysteresis idle window가 지난
  뒤에만 비활성화한다.
- pair activation/deactivation은 request route를 바꾸지 않는다. 이미
  commit된 request는 EOF까지 원래 pair에서 끝난다.

이것은 Karios식 고정/국소 predictor를 흉내 내는 것이 아니라, decoder와
endpoint/fabric pressure를 같은 admission transaction에서 보면서 pair
assignment를 결정하는 차이이다.

## 현실적인 C5 workload

단일 foreground stream이 아닌 tenant가 섞인 moving contention을 사용한다.
현재 보존된 C5 v3 trace의 실제 geometry는 foreground
`(512,16)/(2048,256)/(4094,16)`이고 decoder-hot/remote-hot/kv-remote-hot은
모두 output `2`다. `--background-output-tokens 128`은 현재 builder에서
manifest metadata에만 기록되고 JSONL row에는 적용되지 않는다. 따라서 v3는
output=2 anchor-hot discovery trace이며 output=128 business workload라고
부르면 안 된다. 다음 held-out workload에서는 actual JSONL geometry를 먼저
검증하고, evidence가 있는 `(4094,128)` background stream을 사용할지 fixed-path
direction gate 뒤에 freeze한다. 현재 별도 held-out artifact
`results/tempo_go_c5_heldout_output128_v1/`는 이미 생성되었지만, manifest SHA
`6a143841df6c11768e6dedfc1492c8a6aa1395b4ec80e94166573bd5a40fc62c`, workload
SHA `19ec105d678f51d4145af58173fe63e9973fb0b4a0aabd08681ade14af353f33`,
validator SHA `f00157c5f237c7a271197e499046e0e2a9884881cffeca46554accd015933fd0`를
contract에 묶기 전에는 native evidence가 아니다. 실제 row는 hot output `128`,
replicate `r02/r03`, MISS `1,992` unique/P_ONLY `720`이다.

각 workload는 prompt 512/2048/4094와 명시된 output geometry를 고정하고,
request id와 cache namespace를 완전히 분리한다.

1. `interactive`: 짧은 deadline, bursty arrival, local decoder pressure
2. `latency`: 긴 prompt, P-only warm evidence, remote branch가 실제로
   이길 가능성이 있는 traffic
3. `batch`: 높은 offered load, cold miss와 remote KV 경쟁
4. `background`: 낮은 weight, long-lived queue pressure, starvation 검증

부하는 stable → burst → overload → recovery 순서로 움직인다. 이 phase
이름은 분석용 label일 뿐 controller input이 아니다. paired counterbalanced
arm은 동일 request set, topology, warmup/reset, duration을 사용하며
다음 다섯 arm을 같은 trace·topology·server lifecycle에서 비교한다.

- `ALWAYS_LOCAL`
- `OFFICIAL_LMCACHE_ALWAYS_REMOTE`
- `PREDICTOR_ONLY`
- `QUEUE_GPU_ONLY` (Kairos-like queue/GPU-only fixed policy)
- `TEMPO_GO`

가장 강한 fixed policy는 workload별로 사후 선택하지 않고, discovery에서
profile을 freeze한 뒤 validation에서는 하나의 고정 정책으로 유지한다.

## 관측해야 할 증거

per-request ledger에는 arrival, telemetry sequence, classified, admission,
pair, route commit, credit acquire/release, upstream start, prefill start/end,
transfer bytes/ops, first token, EOF, error/timeout, output token/text digest,
TTFT/TPOT/E2E를 기록한다. aggregate는 workload/tenant/pair/route별로
E2E·TTFT·TPOT p50/p95/p99, request/output-token goodput, queue wait,
remote win rate와 counterfactual을 분리한다.

추가로 다음 fairness/overhead 지표를 필수로 낸다.

- tenant weighted service ratio와 Jain fairness
- maximum tenant wait와 starvation count
- pair별 dominant utilization, queue depth, activation time
- telemetry collection span, refresh timeout, admission CPU time
- stale/partial batch count, route commit mismatch, credit invariant count

## 성공/축소 판정

모든 correctness gate를 먼저 통과해야 한다. stream/output digest 100%,
route provenance 100%, hidden fallback 0, transfer/timeout/terminal queue
0, credit leak/double release 0, tenant starvation 0이어야 한다.

그 다음 pooled aggregate만으로 숨기지 않고 각 workload group에서 다음을
검사한다.

- `TEMPO_GO`가 strongest fixed 대비 E2E p50 10% 이상 개선하거나,
  overload p99/goodput robustness를 15% 이상 개선
- predictor-only 대비 E2E p50 5% 이상 개선하거나, 정상부하 회귀 3%
  이내에서 overload failure/tail을 15% 이상 제거
- output-token goodput이 strongest fixed 대비 5% 이상 개선
- 전체 paired 승률 75% 이상, 각 workload group 60% 이상
- strongest fixed 대비 각 group의 p99/TPOT p99 악화 5% 이내
- 선택된 remote가 local counterfactual보다, 선택된 local이 remote
  counterfactual보다 각각 중앙값 5% 이상 유리
- fairness ratio가 frozen tenant contract를 만족하고 overhead가 별도
  CPU/latency budget 안에 있음

두 독립 candidate revision이 predictor 대비 개선하지 못하거나 remote가
실제로 유리한 workload에서도 5% 이득을 보이지 않으면 threshold를 계속
미세조정하지 않는다. 그 경우 TEMPO-GO의 복잡도를 철회하고
predictor/local-first로 축소한다. 반대로 correctness와 robustness만
개선되는 경우에는 median 성능 주장이 아니라 overload robustness 주장만
허용한다.

## 완료 산출물과 주장 범위

완료에는 canonical global controller/router/frontend/runner/analyzer,
frozen profile와 C5 manifest, native 실행 명령, raw JSONL, workload별
paired/counterfactual 표, fairness/overhead 표, 실패 ablation, 한국어
claim-boundary 문서가 포함되어야 한다. 관련 파일만 명시적으로 commit하며
기존 사용자 변경은 보존한다.

허용되는 최종 주장은 다음으로 제한한다.

> 동일한 native 4-node 실제 vLLM P/D topology와 official LMCache data
> plane에서 TEMPO-GO의 causal global admission/orchestration이 fixed
> local/remote 및 predictor-only policy보다 낮은 latency, 높은 goodput,
> 또는 더 강한 multi-tenant overload robustness를 보였다.

LMCache transport 자체보다 빠르다, Mooncake보다 빠르다, 보편적 SOTA다,
모든 workload에서 항상 빠르다, 단일 allocation만으로 production-ready라는
주장은 하지 않는다.

## 다음 agent에게 그대로 전달할 개선 프롬프트

다음 지시를 임의로 축소하거나 다른 목표로 바꾸지 말고 그대로 수행한다.

> 먼저 저장소의
> `paper/TEMPO_GLOBAL_ORCHESTRATOR_CANONICAL_PLAYBOOK.ko.md`를 처음부터 끝까지
> 읽고, 이 goal 파일과 대조하라. v0 conceptual work부터 v1--v450,
> v452--v544, C0--C5의 raw receipt와 negative evidence를 최신 v535 하나로
> 대체하지 말라. 현재 연구 질문은 Perlmutter native 4-node/16-A100의 실제
> vLLM P/D + official `LMCacheConnectorV1:UCX`에서 moving multi-tenant
> contention을 관찰하고 decoder admission, tenant SLO/fairness, pair
> assignment/proactive scaling, local/remote route, endpoint failure recovery를
> 하나의 global orchestrator로 공동 제어할 수 있는지 검증하는 것이다.
>
> 현재 frozen boundary를 보존하라: C4 route-only negative, C1/C2/C3
> crossover, P_ONLY remote regime, native queue-GPU LMCache failure, native
> TEMPO descriptive result의 goodput negative, Candidate B aggregate negative,
> Candidate C failure-injected CPU replay의 receipt/quarantine evidence를
> 성능 승리로 과장하지 말라. Candidate C native run은 LMCache/EngineCore
> failure와 global failure receipt 9건, pair/route quarantine을 관찰했지만
> step exit 143, incomplete terminal contract, `router_decisions_exact=false`라
> 성능 결과가 아니다. 이를 failure robustness evidence로만 보존하고, 다음
> native는 같은 workload를 맹목적으로 retry하지 말고 frozen contract가 닫힌
> 뒤 새 result root에서 수행하라.
>
> C5는 다섯 arm
> (`ALWAYS_LOCAL`, `OFFICIAL_LMCACHE_ALWAYS_REMOTE`, `PREDICTOR_ONLY`,
> `QUEUE_GPU_ONLY`, `TEMPO_GO`)을 동일 workload와 counterbalanced order로
> 실행한다. workload는 latency/interactive/batch/background tenant를 섞고,
> 512/2048/4094 prompt, cold MISS/P_ONLY evidence, stable→burst→overload→recovery,
> queue wait, scheduler running/waiting, endpoint completion residual, pair
> activation, remote semantic-op reserve, failure/reject receipt를 기록해야
> 한다. phase label이나 future arrival을 policy input으로 넣지 말라.
>
> 모든 admission은 pair×route immutable commit이어야 하고 hidden recompute,
> silent local fallback, same request ID retry, credit leak/double release를
> 허용하지 말라. failure는 latency로 치환하지 말고
> `tempo-go-global-failure-v1` receipt, released work, quarantine scope,
> telemetry sequence, new-request-ID retry policy를 남겨라. tenant별 weighted
> service debt와 raw service units를 혼동하지 말고 minimum service fraction과
> tenant queue-wait budget을 각각 검증하라.
>
> 작업 순서는 (1) bounded source/test/evidence audit, (2) frozen profile과
> manifest binding, (3) 현재 v3 builder의 output geometry bug를 고정하고 CPU
> replay 및 failure injection, (4) endpoint `calibration_only` profile에 대해
> held-out geometry/cache-residency exact calibration 또는 schema에 명시된
> evidence-bound frozen proxy contract를 만들고 검증, (5) native readiness와
> exact terminal receipt, (6) 독립 primary/robustness/fairness validation이다.
> 기존 profile의 scope/hash만 바꾸어 frozen promotion하지 말라. 현재 17개
> endpoint row가 모두 P_ONLY라 MISS/UNKNOWN exact lookup이 닫혀 있지 않으며,
> frozen frontend는 discovery용 external proxy를 자동 허용하지 않는다. 단순히
> `allow_service_proxy=True`로 바꾸거나 MISS를 P_ONLY로 속이는 것은 금지한다.
> 현재 구현한 `FrozenServiceProxyPolicy`를 사용할 경우에도 endpoint identity,
> calibration receipt SHA, allowlisted geometry/residency/lookup mode와 non-exact/
> numeric-unchanged/performance-forbidden flags를 검증해야 하며, policy는 global
> orchestrator capacity/config 입력이 아니다. policy 또는 exact MISS receipt가
> 없는 상태에서는 frozen run contract/native allocation을 만들지 않는다.
> `TEMPO_GO`가 predictor 대비 5% 미만이면 threshold 미세조정을 반복하지
> 말고 controller를 단순화하거나 negative conclusion을 작성하라. 성능
> claim은 correctness, fairness, failure, tail, goodput gate를 모두 통과할
> 때만 허용한다.
>
> Perlmutter에서는 login node에서 코드/가벼운 검증만 하고 GPU workload는
> 승인된 native 4-node/16-GPU interactive allocation에서만 수행하라. 최대
> 4시간 allocation을 재사용하고, Slurm job을 자동 submit/cancel/retry하지
> 말라. container/root/udiRoot/sudo/system-file/ownership 변경은 금지한다.
> `udiRoot.conf must be owned by user root`가 보이면 중단하고 로그만 보존하라.
> 관련 없는 dirty-worktree 변경을 지우거나 stage하지 말고, 결과 artifact와
> SHA를 새 경로에 immutable하게 남겨라. native allocation이 정책상 거절되면
> 더 반복하지 말고 blocker와 CPU evidence를 문서화하라.
