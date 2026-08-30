# TEMPO Elastic-PD canonical actual-vLLM 통합

이 문서는 사용자가 붙여넣은 원래 목표를 변경하지 않고, 그 목표를 검증하기 위한 canonical 경로와 현재 증거 경계를 기록한다.

## 현재 결론: 재현 가능한 negative conclusion

목표는 변경하지 않았다. canonical controller, router, child client, profile guard, analyzer를 실제 vLLM P/D 경로에 연결하고, `ALWAYS_LOCAL`, `OFFICIAL_LMCACHE_ALWAYS_REMOTE`, `PREDICTOR_ONLY`, `FULL_TEMPO`를 동일 workload에서 비교했다.

최종 Perlmutter 실행은 `results/tempo_elastic_pd_canonical_discovery_57133688/run12`에 있다. canonical analyzer 결과는 다음과 같다.

- `schema`: `tempo-elastic-pd-analysis-canonical`
- `verdict`: `negative_conclusion_or_simplify`
- correctness/lifecycle: `true`
- TEMPO route: 48/48 local (`decoder_local_chunked_prefill`), remote 0/48
- TEMPO cache evidence: 48/48 `confirmed_miss`
- TEMPO 대비 best fixed E2E: `+1.152%` 개선 (10% gate 미달)
- TEMPO 대비 predictor E2E: `-2.375%` (predictor보다 느림)
- TEMPO 대비 best fixed goodput: `-1.327%`
- paired win fraction, p99/TPOT, remote-branch counterfactual, workload-group gate: 실패

run12의 48 paired samples arm median은 다음과 같다 (ms).

| arm | TTFT | E2E | TPOT | route |
|---|---:|---:|---:|---|
| ALWAYS_LOCAL | 91.303 | 1667.684 | 21.440 | local 48/48 |
| ALWAYS_REMOTE | 349.431 | 1816.462 | 22.113 | remote 48/48 |
| PREDICTOR_ONLY | 82.186 | 1610.236 | 21.531 | remote 8/48 |
| FULL_TEMPO | 81.245 | 1648.479 | 21.445 | local 48/48 |

따라서 이 실행은 “predictor와 strongest fixed policy보다 유의미하게 빠르다”는 성능 승리를 입증하지 않는다. 오히려 현재 disjoint-cache balanced workload가 실제 P-only warm state를 만들지 못해 TEMPO의 remote 선택 branch를 검증하지 못했다는 재현 가능한 음성 결과다. `legacy_screen`의 더 낙관적인 판정은 최종 근거로 사용하지 않는다.

## canonical 실행 경로

`elastic_pd_node_entry.sh` → `vllm_lmcache_elastic_pd_node.py` → 실제 vLLM/LMCACHE lifecycle → `tempo_pd_elastic_router.py` → `run_tempo_pd_elastic.py` → spawned `run_tempo_pd_elastic_stream_metrics.py` 순서다.

canonical 경로의 중요한 불변식은 다음과 같다.

- router가 모르는 cache residency는 `UNKNOWN`으로 fail-closed 처리한다.
- `D_ONLY`와 `BOTH`는 remote admission을 허용하지 않는다.
- remote는 frozen profile의 보수적 benefit margin이 5 ms 이상일 때만 허용한다.
- child Python process까지 `tempo-elastic-pd-router-canonical` wire schema를 사용한다.
- `/health`, `/tempo/decisions`, streaming response header가 같은 canonical schema를 반환한다.
- screen profile은 discovery에만 허용하고, final profile은 `--require-replicated-profile`로 검증한다.

## Perlmutter 실행

Perlmutter interactive GPU job 규칙에 맞춰 bounded allocation만 사용한다. 공식 문서: [Interactive jobs](https://docs.nersc.gov/jobs/interactive/), [Running jobs](https://docs.nersc.gov/systems/perlmutter/running-jobs/), [Perlmutter architecture](https://docs.nersc.gov/systems/perlmutter/architecture/), [Perlmutter scratch](https://docs.nersc.gov/filesystems/perlmutter-scratch/).

```bash
salloc --nodes=4 --qos=interactive --time=04:00:00 \
  --constraint=gpu --gpus=16 --account=m1248_g --immediate=600
```

allocation 안에서만 다음 launcher를 실행한다. launcher는 4개 node, GPU 4개/node, `pytorch/2.8.0`, 하나의 `srun` step을 강제한다.

```bash
TEMPO_ELASTIC_PD_APPROVED=YES \
bash eval/sota_4node/run_tempo_pd_elastic_in_allocation.sh \
  results/<workload>.jsonl results/<new-result-dir>
```

현재 `screen_only` discovery를 재현하려면 기본 profile을 사용한다. replicated final profile을 사용할 때는 profile과 scope를 모두 명시한다.

```bash
TEMPO_ELASTIC_PD_PROFILE=/absolute/path/to/replicated-profile.json \
TEMPO_ELASTIC_PD_PROFILE_SCOPE=replicated \
TEMPO_ELASTIC_PD_APPROVED=YES \
bash eval/sota_4node/run_tempo_pd_elastic_in_allocation.sh \
  results/<workload>.jsonl results/<new-result-dir>
```

replicated profile은 각 route에 최소 3개 sample, exact output equivalence, transfer failure 0을 만족해야 한다. 현재 `real_tempo_pd_elastic_profile_v447.json`은 `screen_only`이며 local/remote sample 수가 부족하므로 final profile로 승격하지 않는다.

최종 검증은 위 allocation에서 다음처럼 실행했다.

```bash
TEMPO_ELASTIC_PD_APPROVED=YES \
bash eval/sota_4node/run_tempo_pd_elastic_in_allocation.sh \
  results/tempo_elastic_pd_v449_job_57086357/tempo_elastic_pd_v445/warmup.jsonl \
  results/tempo_elastic_pd_canonical_discovery_57133688/run12
```

이 launcher는 공식 LMCache P/D proxy의 실제 transfer 완료 신호(`X-Tempo-LMCache-PD-Transfer: complete`)와 request id/prompt token/KV byte telemetry를 사용한다. 공식 proxy의 first-token protocol 때문에 실제 transfer geometry는 normalized prompt `N`에 대해 `N+1` token이며, analyzer/router가 이를 명시적으로 검증한다. proxy shutdown 경로도 clean exit를 확인했다.

## 보존된 실행 증거

- `57128085/run1`, `run2`: nonce 없는 generated workload가 inherited client 계약을 위반.
- `57128085/run3`: child stream client가 old router schema를 기대.
- `57128758/run4`: parent schema patch 후 child subprocess가 old stream module을 계속 spawn.
- `57129063/run5`: child wrapper 후 `/tempo/decisions`가 runtime old schema를 반환.

run5 이후 app-local schema middleware, 실제 proxy telemetry, strict transfer geometry, queue-wait/max-batch guard를 추가했다. `run12`에서는 8개 arm artifact 모두 `all_streams_valid=true`, `router_decisions_exact=true`, HTTP/stream error 및 timeout이 없었고, canonical analyzer가 위 음성 판정을 재생성했다.

## 로컬 검증

```bash
PYTHONPATH=. .vllm_venv/bin/python -m unittest \
  tempo.test_pd_elastic_controller \
  tempo.test_pd_elastic_profile_v444 \
  tempo.test_pd_elastic_cache_residency_v450 \
  eval.sota_4node.test_tempo_pd_elastic_router \
  eval.sota_4node.test_tempo_pd_elastic_router_v445 \
  eval.sota_4node.test_tempo_pd_elastic_router_v449 \
  eval.sota_4node.test_analyze_tempo_pd_elastic_balanced_v450
```

최종 run12 raw stage를 다시 확인할 때는 다음을 사용한다.

```bash
PYTHONPATH=. .vllm_venv/bin/python \
  eval/sota_4node/analyze_tempo_pd_elastic.py \
  --stage-root results/tempo_elastic_pd_canonical_discovery_57133688/run12/tempo_elastic_pd_v445 \
  --output /tmp/tempo-elastic-analysis-run12.json
```

분석 결과 JSON은 `run12/elastic_pd_final.json`, paired rows·counterfactual·group summaries는 그 파일에 포함된다. 원시 요청/stream/router telemetry는 `run12/tempo_elastic_pd_v445/elastic_balanced_measured/*.raw.json` 및 `raw.json`에 보존된다.
추가 파생 artifact는 `run12/tempo_elastic_pd_v445/aggregate.csv`, `paired_rows.jsonl`, `counterfactuals.json`, `failed_gates.json`, `e2e_median.svg`, `artifact_manifest.json`이다. 이 파일들은 run12 분석 JSON에서만 생성했으며 새로운 측정이나 route evidence를 포함하지 않는다.

## claim boundary

최종 보고서에서 허용되는 주장은 동일 Perlmutter A100 4-node/16-GPU topology, 동일 vLLM P/D lifecycle, 동일 official LMCacheConnectorV1 data plane에서의 통합·telemetry·음성 결과다. 이 결과로 transport speed, Mooncake, SOTA, universal workload, production readiness 또는 TEMPO 성능 우월성을 주장하지 않는다. 특히 TEMPO가 remote route를 선택한 측정 sample이 0개이므로 remote-selected branch의 성능 승리도 주장하지 않는다.
