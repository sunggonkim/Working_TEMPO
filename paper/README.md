# TEMPO-GO research and paper index

이 디렉터리는 TEMPO-GO의 논문 초안, 전체 연구 계보, Perlmutter 실행 규칙과
machine-readable evidence를 연결한다. 현재 headline은 과거 C8 positive가 아니라
allocation `57736076`의 Candidate O 7-arm native negative다.

- [통합 목표·상태·실행계획](TEMPO_GO_UNIFIED_GOAL_STATE_AND_EXECUTION_PLAN.ko.md):
  v0~v600, C0~C10, §74.58 Candidate O까지의 authoritative chronology
- [현재 연구 master state](TEMPO_RESEARCH_MASTER_STATE_AND_NEXT_GOAL.ko.md):
  연구 질문, contribution, 다음 global transaction
- [논문·artifact README](tempo_go/README.md): 최신 표·그래프, historical paper,
  재현 명령과 claim boundary
- [논문 소스](tempo_go/main.tex)와 [historical PDF](tempo_go/main.pdf)
- [current evidence manifest](tempo_go/current_evidence_manifest.json): M/N/O의
  contract, immutable native analysis, fail-closed business analysis와 SHA

현재 결론은 세 부분으로 분리한다.

1. actual vLLM P/D + official LMCache/NIXL + NCCL/Slingshot co-job에서
   receiver/data-plane overload와 pair asymmetry는 재현됐다.
2. Candidate O bundle은 remote-favorable 30/30과 높은 business completion을
   기록했지만, M과는 allocation이 달라 비인과 context이고 O가 바꾼 route-scope
   quarantine은 1,614개 decision에서 한 번도 발동하지 않았다.
3. O는 strongest fixed의 normal/miss-hot tail, observer coverage와 background
   utility gate를 함께 통과하지 못했다. 따라서
   `performance_claim_allowed=false`다.

SC26 artifact 구조에 맞춰 contribution↔artifact mapping, setup/execution/analysis
분리, immutable receipt와 one-command figure regeneration을 유지한다. Native
4-node 결과를 downscaled CPU hierarchy 결과나 historical post-hoc SOTA 비교로
대체하지 않는다.
