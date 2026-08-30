#!/usr/bin/env python3
"""Wire the quiescence scout v2 runner into the audited node-v1 lifecycle."""

from eval.sota_4node import vllm_lmcache_tp16_quiescence_scout_node_v1 as impl


def main() -> None:
    impl.RUNNER_MODULE = (
        "eval.sota_4node.run_vllm_lmcache_tp16_quiescence_scout_v2"
    )
    impl.main()


if __name__ == "__main__":
    main()
