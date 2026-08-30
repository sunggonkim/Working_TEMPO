#!/usr/bin/env python3
"""High-load policy11: bypass every warm remote fetch at 52 requests/s."""

from eval.sota_4node import tempo_pd_same_server_policy10_router_v274 as policy10


POLICY_ID = "qwen25-7b-tp4x2-warm-highload-local-11"


def main(argv=None) -> int:
    old_buckets = policy10.REMOTE_BUCKETS
    old_id = policy10.POLICY_ID
    policy10.REMOTE_BUCKETS = frozenset()
    policy10.POLICY_ID = POLICY_ID
    try:
        return policy10.main(argv)
    finally:
        policy10.REMOTE_BUCKETS = old_buckets
        policy10.POLICY_ID = old_id


if __name__ == "__main__":
    raise SystemExit(main())
