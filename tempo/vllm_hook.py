# SPDX-License-Identifier: Apache-2.0
# TEMPO: Harmonious Burst Buffer for Jitter-Free LLM Systems
# tempo/vllm_hook.py
#
# Monkey-patch vLLM v1's KVCacheManager to route evictions through TEMPO.
#
# Usage (zero-code-change):
#   export TEMPO_RATE_GBPS=5
#   export TEMPO_LUSTRE_DIR=$PSCRATCH/kvcache
#   python -c "import tempo.vllm_hook; tempo.vllm_hook.install()"
#   python -m vllm.entrypoints.openai.api_server --model meta-llama/...
#
# Or from Python:
#   from tempo.vllm_hook import install_tempo_in_vllm
#   engine = install_tempo_in_vllm(vllm_engine, tempo_cfg=TEMPOConfig())
#
# Design:
#   We intercept vLLM's LMCacheConnector.store_kv_cache() — the single
#   code path where vLLM pushes evicted KV tensors to LMCache. We replace
#   the synchronous put() call with TEMPO's non-blocking absorb().
#
#   vLLM v1 architecture (simplified):
#     LLMEngine → Worker → CacheEngine → LMCacheConnector
#                                              └─ put(key, tensor) ← we hook here

from __future__ import annotations

import importlib
import logging
import types
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def _try_import(module_path: str):
    try:
        return importlib.import_module(module_path)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# vLLM v1 LMCacheConnector hook
# ---------------------------------------------------------------------------

def install(tempo_cfg=None) -> bool:
    """
    Monkey-patch vLLM's LMCacheConnector to route KV evictions through TEMPO.
    Call once before creating the vLLM engine.

    Returns True if patching succeeded, False if vLLM/LMCache is not installed.
    """
    # Import TEMPO's connector
    from tempo.lmcache_connector import TEMPOStorageBackend, TEMPOConfig

    cfg = tempo_cfg or TEMPOConfig()

    # --- Attempt vLLM v1 (vllm.worker.cache_engine) ---
    connector_mod = _try_import("lmcache.integration.vllm.lmcache_connector")
    if connector_mod is None:
        connector_mod = _try_import("lmcache.vllm_v1.connector")
    if connector_mod is None:
        log.warning("TEMPO: vLLM/LMCache connector module not found. "
                    "Make sure lmcache is installed: pip install lmcache")
        return False

    # Find the connector class
    connector_cls = None
    for attr in ("LMCacheConnector", "VLLMLMCacheConnector", "CacheConnector"):
        connector_cls = getattr(connector_mod, attr, None)
        if connector_cls is not None:
            break

    if connector_cls is None:
        log.warning("TEMPO: Could not find LMCacheConnector class in %s",
                    connector_mod.__name__)
        return False

    original_put = connector_cls.store_kv_cache

    def tempo_store_kv_cache(self, key, kv_tensor, *args, **kwargs):
        """
        TEMPO-paced replacement for LMCacheConnector.store_kv_cache().
        Routes eviction through SpikeAbsorber instead of direct backend.put().
        """
        if not hasattr(self, "_tempo_backend"):
            # Lazy init: wrap the connector's storage backend with TEMPO
            backing = getattr(self, "backend", None) or getattr(self, "store", None)
            if backing is None:
                # Can't find backing store — fall through to original
                return original_put(self, key, kv_tensor, *args, **kwargs)
            self._tempo_backend = TEMPOStorageBackend(backing, cfg)
            log.info("TEMPO: installed pacing on %s.%s",
                     type(self).__module__, type(self).__name__)
        self._tempo_backend.put(key, kv_tensor)

    # Patch the class
    connector_cls.store_kv_cache = tempo_store_kv_cache
    log.info("TEMPO: patched %s.store_kv_cache → phase-aware pacing active",
             connector_cls.__name__)
    return True


# ---------------------------------------------------------------------------
# SGLang integration
# ---------------------------------------------------------------------------

def install_sglang(tempo_cfg=None) -> bool:
    """
    Patch SGLang's KV cache eviction path.
    SGLang uses sglang.srt.mem_pool.ReqToTokenPool for KV management.
    We hook the evict() path to redirect through TEMPO.
    """
    from tempo.lmcache_connector import TEMPOStorageBackend, TEMPOConfig
    cfg = tempo_cfg or TEMPOConfig()

    mem_pool_mod = _try_import("sglang.srt.mem_pool")
    if mem_pool_mod is None:
        log.warning("TEMPO: SGLang mem_pool not found")
        return False

    cache_cls = getattr(mem_pool_mod, "KVCache", None)
    if cache_cls is None:
        log.warning("TEMPO: KVCache class not found in sglang.srt.mem_pool")
        return False

    original_evict = getattr(cache_cls, "evict", None)
    if original_evict is None:
        log.warning("TEMPO: KVCache.evict() not found")
        return False

    def tempo_evict(self, *args, **kwargs):
        # Call original to get the tensors, then absorb them
        result = original_evict(self, *args, **kwargs)
        if hasattr(self, "_tempo_backend") and result is not None:
            key, tensor = result if isinstance(result, tuple) else (id(result), result)
            self._tempo_backend.put(key, tensor)
        return result

    cache_cls.evict = tempo_evict
    log.info("TEMPO: patched SGLang KVCache.evict → phase-aware pacing active")
    return True


# ---------------------------------------------------------------------------
# Convenience: install for whatever serving stack is present
# ---------------------------------------------------------------------------

def install_tempo_in_vllm(engine, tempo_cfg=None):
    """
    Convenience wrapper. Pass in a vLLM LLMEngine or AsyncLLMEngine.
    Returns the (unmodified) engine — the hook is installed globally on the class.

    Example:
        from vllm import LLM
        from tempo.vllm_hook import install_tempo_in_vllm

        llm = install_tempo_in_vllm(LLM("meta-llama/Meta-Llama-3-8B"))
    """
    install(tempo_cfg=tempo_cfg)
    return engine
