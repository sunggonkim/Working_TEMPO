#!/usr/bin/env python3
"""Front-loaded 16 MiB two-wave TP16 campaign wiring."""

from eval.sota_4node import vllm_lmcache_tp16_campaign_node_v1 as impl


def main() -> None:
    impl.PLAN_RELATIVE = impl.Path(
        "eval/sota_4node/real_tp16_frontload16mib_v7.json"
    )
    impl.RUNNER_MODULE = (
        "eval.sota_4node.run_vllm_lmcache_tp16_frontload16mib_v7"
    )
    impl.main()


if __name__ == "__main__":
    main()
