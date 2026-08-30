"""CPU-fixture compatibility shim for the add-only hybrid v6 tests."""

from eval.sota_4node.test_vllm_lmcache_tp16_hybrid_boost_audit_v5 import (
    _args,
    _valid_records,
)

__all__ = ["_args", "_valid_records"]
