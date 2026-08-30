#!/usr/bin/env python3
"""Launch-safe M20 entry retaining the audited v6 worker signature."""
from eval.sota_4node import run_vllm_lmcache_tp16_predecode_phase_m20_entry as m20
from eval.sota_4node import run_vllm_lmcache_tp16_predecode_phase_m20_v2_entry as v2

def main() -> None:
    audited_worker = m20.fixed._transfer_worker
    m20.old._transfer_worker = audited_worker
    m20._run_block = v2._run_block
    m20.main()

if __name__ == "__main__":
    main()
