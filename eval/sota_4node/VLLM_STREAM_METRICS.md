# Native vLLM streaming metrics contract

This is a bounded research-prototype client for a separately launched vLLM
server. It does not start, discover, retry, monitor, or stop the server and it
does not interact with Slurm.

## CLI

The model argument is an absolute, already downloaded local directory. The
client appends the fixed public endpoint `/v1/completions` to `--base-url`.

```bash
.vllm_venv/bin/python -m eval.sota_4node.run_vllm_stream_metrics \
  --base-url http://127.0.0.1:8000 \
  --model /pscratch/sd/s/sgkim/Skim-Tempo/models/TinyLlama-1.1B-Chat-v1.0 \
  --workload /explicit/path/workload.jsonl \
  --output /explicit/path/fg_only.raw.json \
  --mode fg_only \
  --max-workers 4 \
  --default-max-tokens 64
```

Each nonblank workload line has exactly this closed schema:

```json
{"request_id":"request-0","prompt":"A deterministic prompt","max_tokens":64,"arrival_offset_ms":0.0}
```

`max_tokens` and `arrival_offset_ms` are optional. Unknown fields, duplicate
IDs, a token count below two, or a mixture of explicit and implicit arrival
times are rejected. `--request-rate` provides open-loop offsets when explicit
offsets are absent. With neither, the requests form a bounded burst; the
worker count bounds concurrency.

The request fixes `temperature=0`, `n=1`, `echo=false`, `ignore_eos=true`,
`stream=true`, `stream_options.include_usage=true`, and `logprobs=1`. There
are no automatic retries. `ignore_eos` plus strict final token-count and usage
checks force exactly `max_tokens` output tokens.

## Importable sidecar API

An integrated rank-zero controller can call the same strict parser without
creating a workload or output file:

```python
from eval.sota_4node.vllm_stream_metrics_api import run_workload

artifact = run_workload(
    "http://127.0.0.1:8000",
    "/absolute/local/model",
    [
        {"request_id": f"r{i}", "prompt": "prompt", "max_tokens": 64}
        for i in range(4)
    ],
    mode="fg_only",
    max_workers=4,
)
```

The return value is the same `tempo-vllm-stream-metrics-raw-1` object written
by the CLI. The call performs only the bounded HTTP request block. The caller
owns server readiness and lifecycle.

## Exactness and failure behavior

The raw artifact stores dispatch, every output-token SSE arrival, last stream
event, returned token strings, returned text, hashes, final usage, response
identity, and request contract. A timestamp is accepted only if
`choices[0].logprobs.tokens` identifies exactly one model token in that SSE
event. Multiple tokens in one event, text without token identity, missing
usage, missing `[DONE]`, a finish reason other than `length`, token-count
mismatch, changed response identity, malformed SSE, timeout, or HTTP error
makes the request invalid.

The client still writes a diagnostic raw artifact on a request failure, then
returns exit code 2. It does not retry. The analyzer emits no performance
aggregate at all for a run containing an invalid request. This prevents a
partial-success result from becoming a benchmark claim.

These are client-observed times. They include server queueing, transport, and
client parsing; they are not GPU kernel timestamps.

## Analyzer and metric definitions

```bash
.vllm_venv/bin/python -m eval.sota_4node.analyze_vllm_stream_metrics \
  --run fg=fg_only.raw.json \
  --run greedy=lmcache_greedy.raw.json \
  --run tempo=tempo.raw.json \
  --output comparison.json \
  --ttft-slo-ms 500 \
  --tpot-slo-ms 50 \
  --itl-slo-ms 100
```

- TTFT = first output-token event arrival minus request dispatch.
- Per-token ITL = each consecutive pair of output-token event arrivals.
- TPOT = last minus first token arrival, divided by output token count minus
  one. Thus TPOT is the mean of the request's recorded ITLs and excludes
  TTFT.
- E2E = last output-token arrival minus request dispatch.
- The throughput window is earliest dispatch through latest last-token
  arrival. Output throughput is every completed output token divided by this
  full window.
- A request satisfies the SLO only when all configured TTFT, TPOT, maximum
  ITL, and optional E2E thresholds pass. Request goodput and output-token
  goodput divide only SLO-passing work by the same full window. Failed work is
  never removed from the denominator.

For two or more runs, the analyzer also requires one model-config digest, the
same request IDs/prompts/token counts/schedules, and byte-exact returned token
and text equality before setting `comparison_claim_allowed=true`.

## CPU verification

```bash
PYTHONDONTWRITEBYTECODE=1 .vllm_venv/bin/python -m unittest -v \
  eval.sota_4node.test_run_vllm_stream_metrics \
  eval.sota_4node.test_analyze_vllm_stream_metrics
```

## Bounded development command log (2026-08-14 UTC)

All reads were scoped to the repository paths shown. No GPU, network request,
Slurm action, filesystem discovery, or background process was used.

```text
sed -n '1,260p' NERSC_AGENT_SAFETY.md
ls -la .
ls -la scripts
ls -la tests
ls -la tempo
ls -la eval
sed -n '1,240p' README.md
ls -la eval/sota_4node
rg -n --glob '*.py' 'stream_options|/v1/completions|urllib\.request|requests\.|httpx|ttft|time_to_first|tokens_per_second|goodput' eval/sota_4node
sed -n '1,240p' eval/sota_4node/run_inference_kv_live_smoke.py
sed -n '1,220p' eval/sota_4node/analyze_active_pulse_campaign.py
sed -n '1,220p' eval/sota_4node/test_analyze_active_pulse_campaign.py
env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 40s .vllm_venv/bin/python -c <public-field import check; failed closed because the old module path is absent>
env PYTHONDONTWRITEBYTECODE=1 timeout 30s .vllm_venv/bin/python -m py_compile eval/sota_4node/run_vllm_stream_metrics.py eval/sota_4node/vllm_stream_metrics_api.py eval/sota_4node/analyze_vllm_stream_metrics.py
env PYTHONDONTWRITEBYTECODE=1 timeout 60s .vllm_venv/bin/python -m unittest -v eval.sota_4node.test_run_vllm_stream_metrics eval.sota_4node.test_analyze_vllm_stream_metrics
```
