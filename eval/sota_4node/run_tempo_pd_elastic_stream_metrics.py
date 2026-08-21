#!/usr/bin/env python3
"""Canonical stream metrics child for the actual-vLLM Elastic-PD path."""

from dataclasses import replace

from eval.sota_4node import run_tempo_pd_elastic_stream_metrics_v445 as _prior


ROUTER_SCHEMA = "tempo-elastic-pd-router-canonical"


def main() -> int:
    old_schema = _prior.ROUTER_SCHEMA
    old_execute = _prior.forced.execute_request

    def execute_with_p_only_seed(item, *args, **kwargs):
        if "-warm-" not in item.request_id:
            return old_execute(item, *args, **kwargs)
        seed_item = replace(
            item,
            request_id=item.request_id.replace(
                "-warm-", f"-warm-seed-o{item.max_tokens}-", 1),
            max_tokens=2,
        )
        seed = old_execute(seed_item, *args, **kwargs)
        if not seed.get("valid"):
            raise RuntimeError(
                "P-only cache seed failed: "
                f"{seed_item.request_id}: {seed.get('error')} "
                f"{seed.get('contract_violations')}"
            )
        result = old_execute(item, *args, **kwargs)
        result["p_only_cache_seed"] = {
            "request_id": seed_item.request_id,
            "valid": True,
            "route": seed["router"]["route"],
            "reason": seed["router"]["reason"],
        }
        return result

    _prior.ROUTER_SCHEMA = ROUTER_SCHEMA
    _prior.forced.execute_request = execute_with_p_only_seed
    try:
        return _prior.main()
    finally:
        _prior.forced.execute_request = old_execute
        _prior.ROUTER_SCHEMA = old_schema


if __name__ == "__main__":
    raise SystemExit(main())
