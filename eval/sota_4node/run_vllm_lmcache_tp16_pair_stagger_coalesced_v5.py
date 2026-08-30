#!/usr/bin/env python3
"""Runtime compatibility entrypoint for the true single-descriptor v4 screen.

NIXL 1.4 exposes descriptor-list cardinality as ``descCount()`` rather than
Python ``len()``.  The v4 implementation intentionally checks the physical
descriptor count at channel construction; this entrypoint maps that supported
binding method to ``__len__`` in-process, then runs the otherwise unchanged
v4 experiment and contract.
"""

from __future__ import annotations

import os
import sys

from eval.sota_4node import (
    run_vllm_lmcache_tp16_pair_stagger_coalesced_v4 as v4,
)


def install_nixl_descriptor_count_compatibility() -> type:
    from nixl import _api

    descriptor_type = _api.nixlBind.nixlXferDList
    desc_count = getattr(descriptor_type, "descCount", None)
    if not callable(desc_count):
        raise RuntimeError("NIXL descriptor list does not expose descCount()")
    if not hasattr(descriptor_type, "__len__"):
        descriptor_type.__len__ = lambda self: int(self.descCount())
    return descriptor_type


def main() -> None:
    descriptor_type = install_nixl_descriptor_count_compatibility()
    if not hasattr(descriptor_type, "__len__"):
        raise RuntimeError("NIXL descriptor count compatibility was not installed")
    v4.main()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
