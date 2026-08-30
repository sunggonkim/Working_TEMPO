#!/usr/bin/env python3
"""512 KiB token-microburst TP16 campaign wiring."""

from eval.sota_4node import vllm_lmcache_tp16_campaign_node_v1 as impl


def main() -> None:
    impl.PLAN_RELATIVE = impl.Path(
        "eval/sota_4node/real_tp16_token_microburst512kib_v8.json"
    )
    impl.RUNNER_MODULE = (
        "eval.sota_4node.run_vllm_lmcache_tp16_token_microburst512kib_v8"
    )
    impl.main()


if __name__ == "__main__":
    main()
