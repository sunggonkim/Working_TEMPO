#!/usr/bin/env python3
"""Stable entrypoint for the add-only hybrid v6 implementation."""

from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v5 as old
from eval.sota_4node import run_vllm_lmcache_tp16_hybrid_boost_v6 as fixed


# apply_patch cannot update a newly added file in this workspace.  Bind the
# original v5 functions explicitly so v6 wrappers cannot recurse after v6
# installs itself into v5's main.
fixed._ORIGINAL_VALIDATE_TRACE = old._validate_trace
fixed._ORIGINAL_AGGREGATE = old._aggregate


if __name__ == "__main__":
    fixed.main()
