#!/usr/bin/env python3
"""One-GPU live KV-flow smoke for the shared TEMPO-RD admission contract.

This is not a vLLM/LMCache benchmark and never claims inference SOTA.  It
exercises exact KV version identity, host-pinned -> GPU movement, the shared
DomainAdmissionController, and deterministic output equivalence.  The result
is a backend-independent feasibility artifact for the later native backend
adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from tempo.domain_admission import DomainAdmissionController, DomainBudget
from tempo.kv_flow import KVFlowLedger, KVOperation, KVTransferRequest, KVVersion
from tempo.resource_domain import ResourceDomain


SCHEMA = "tempo-rd-inference-kv-live-smoke-1"
ROUTE = (ResourceDomain.HOST_NUMA, ResourceDomain.PCIE_HOST, ResourceDomain.GPU_LOCAL)


def _token(logit: torch.Tensor) -> int:
    return int(abs(float(logit.item())) * 1_000_000.0) % 50_257


def _attention_tokens(kv: torch.Tensor, query: torch.Tensor, tokens: int) -> list[int]:
    # Keep the compute deterministic and small while touching the complete KV
    # payload.  The returned token sequence is the correctness oracle.
    outputs: list[int] = []
    for index in range(tokens):
        q = query + (index + 1) * 1e-4
        outputs.append(_token(torch.sum(kv * q)))
    return outputs


def _controller() -> DomainAdmissionController:
    return DomainAdmissionController(
        {domain: DomainBudget(domain, 5_936_536_675, 16 * 1024 * 1024) for domain in ROUTE},
        catch_up_slack_ns=25_000_000,
    )


def _run_mode(
    mode: str,
    host_kv: torch.Tensor,
    query: torch.Tensor,
    *,
    requests: int,
    tokens: int,
    chunk_bytes: int,
    deadline_ns: int,
) -> dict[str, object]:
    elements = host_kv.numel()
    total_bytes = elements * host_kv.element_size()
    gpu_kv = torch.empty(elements, dtype=host_kv.dtype, device="cuda")
    baseline_controller = _controller() if mode == "tempo_controlled" else None
    ttft_ms: list[float] = []
    itl_ms: list[float] = []
    outputs: list[list[int]] = []
    admitted = completed = 0
    rejected = 0
    for request_index in range(requests):
        version = KVVersion.from_bytes(f"smoke-{mode}", request_index, host_kv.cpu().numpy().tobytes())
        ledger = KVFlowLedger()
        ledger.publish(version)
        request_begin = time.perf_counter()
        if mode == "none":
            # GPU-resident foreground-only baseline.
            if request_index == 0:
                gpu_kv.copy_(host_kv, non_blocking=False)
                torch.cuda.synchronize()
        elif mode == "kv_open":
            gpu_kv.copy_(host_kv, non_blocking=False)
            torch.cuda.synchronize()
        elif mode == "tempo_controlled":
            assert baseline_controller is not None
            for offset in range(0, total_bytes, chunk_bytes):
                size = min(chunk_bytes, total_bytes - offset)
                req = KVTransferRequest(
                    request_id=f"kv-{request_index}-{offset}",
                    version=version,
                    operation=KVOperation.PREFETCH,
                    bytes=size,
                    source=ResourceDomain.HOST_NUMA,
                    destination=ResourceDomain.GPU_LOCAL,
                    route=ROUTE,
                    deadline_ns=time.monotonic_ns() + deadline_ns,
                    max_residual_bytes=min(size, chunk_bytes),
                )
                decision = ledger.admit_via_domain_controller(
                    req, baseline_controller, now_ns=time.monotonic_ns()
                )
                if decision.status != "admitted":
                    rejected += 1
                    raise RuntimeError(f"KV admission rejected: {decision.reason}")
                admitted += decision.admitted_bytes
                first = offset // host_kv.element_size()
                last = first + size // host_kv.element_size()
                gpu_kv[first:last].copy_(host_kv[first:last], non_blocking=False)
                torch.cuda.synchronize()
                ledger.complete(req.request_id, decision.admitted_bytes)
                completed += decision.admitted_bytes
        else:
            raise ValueError(mode)
        first_token_begin = time.perf_counter()
        result = _attention_tokens(gpu_kv, query, tokens)
        torch.cuda.synchronize()
        first_token_ms = (time.perf_counter() - first_token_begin) * 1000
        total_ms = (time.perf_counter() - request_begin) * 1000
        ttft_ms.append(total_ms)
        itl_ms.append(first_token_ms / max(tokens, 1))
        outputs.append(result)
    return {
        "mode": mode,
        "samples": requests,
        "ttft_p50_ms": statistics.median(ttft_ms),
        "ttft_p99_ms": sorted(ttft_ms)[max(0, int(len(ttft_ms) * 0.99) - 1)],
        "itl_p50_ms": statistics.median(itl_ms),
        "itl_p99_ms": sorted(itl_ms)[max(0, int(len(itl_ms) * 0.99) - 1)],
        "outputs": outputs,
        "admitted_bytes": admitted,
        "completed_bytes": completed,
        "rejected_requests": rejected,
        "total_bytes_per_request": total_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--kv-mib", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the live KV smoke")
    if args.requests < 4 or args.tokens < 1 or args.kv_mib < 1 or args.chunk_mib < 1:
        raise SystemExit("requests/tokens/kv-mib/chunk-mib are invalid")
    torch.manual_seed(20260813)
    elements = args.kv_mib * 1024 * 1024 // 4
    host_kv = torch.arange(elements, dtype=torch.float32, pin_memory=True)
    query = torch.linspace(0.001, 0.009, elements, dtype=torch.float32, device="cuda")
    host_digest = hashlib.sha256(host_kv.numpy().tobytes()).hexdigest()
    modes = []
    for mode in ("none", "kv_open", "tempo_controlled"):
        modes.append(_run_mode(
            mode, host_kv, query, requests=args.requests, tokens=args.tokens,
            chunk_bytes=args.chunk_mib * 1024 * 1024, deadline_ns=250_000_000,
        ))
    baseline = modes[0]["outputs"]
    for item in modes:
        item["correctness_met"] = item["outputs"] == baseline
        item.pop("outputs")
        if not item["correctness_met"]:
            raise RuntimeError(f"output equivalence failed for {item['mode']}")
    payload = {
        "schema_version": SCHEMA,
        "evidence_state": "live_smoke",
        "backend": {"name": "torch_kv_microbench", "version": torch.__version__},
        "device": torch.cuda.get_device_name(0),
        "world_size": 1,
        "nodes": 1,
        "kv_bytes_per_request": args.kv_mib * 1024 * 1024,
        "chunk_bytes": args.chunk_mib * 1024 * 1024,
        "deadline_ns": 250_000_000,
        "host_kv_sha256": host_digest,
        "modes": modes,
        "causal_claim_allowed": False,
        "limitations": [
            "not a native vLLM/SGLang/LMCache backend",
            "single-GPU local H2D route; no NIC/Slingshot/PFS stage",
            "TTFT/ITL are microbenchmark diagnostics only",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "correctness": True, "modes": len(modes)}, sort_keys=True))


if __name__ == "__main__":
    main()
