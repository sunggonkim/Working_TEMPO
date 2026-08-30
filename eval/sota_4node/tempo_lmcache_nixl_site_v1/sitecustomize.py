"""Install TEMPO's LMCache NIXL patch after LMCache naturally imports it.

This avoids importing torch/LMCache during Python startup, which could poison
vLLM's multiprocessing initialization. The import hook removes itself as soon
as the exact target module has completed loading.
"""

from __future__ import annotations

import builtins
import importlib
import os
import sys
import threading


TARGET = "lmcache.v1.transfer_channel.nixl_channel"
_original_import = builtins.__import__
_lock = threading.Lock()
_installed = False


def _maybe_install() -> None:
    global _installed
    if _installed:
        return
    module = sys.modules.get(TARGET)
    if module is None or not hasattr(module, "NixlChannel"):
        return
    with _lock:
        if _installed:
            return
        builtins.__import__ = _original_import
        hotpath = importlib.import_module(
            "lmcache.v1.transfer_channel.tempo_nixl_hotpath"
        )
        hotpath.install(module.NixlChannel)
        _installed = True
        os.environ["TEMPO_LMCACHE_NIXL_HOTPATH_INSTALLED"] = "1"


def _tempo_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = _original_import(name, globals, locals, fromlist, level)
    if TARGET in sys.modules:
        _maybe_install()
    return result


if os.environ.get("TEMPO_LMCACHE_NIXL_HOTPATH") == "1":
    builtins.__import__ = _tempo_import
