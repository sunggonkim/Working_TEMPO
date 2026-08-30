#!/usr/bin/env python3
"""Policy10: suppress only the repeatedly losing (512, 32) remote bucket."""

from eval.sota_4node import tempo_pd_same_server_hybrid_phase_router_v181 as phase
from tempo import pd_cache_affinity as affinity
from tempo import pd_hybrid_controller as hybrid


POLICY_ID = "qwen25-7b-tp4x2-warm-affinity-10"
REMOTE_BUCKETS = frozenset({
    (512, 64), (512, 128), (2048, 64), (2048, 256),
})


def main(argv=None) -> int:
    old_buckets = affinity.REMOTE_BUCKETS
    old_affinity_id = affinity.POLICY_ID
    old_hybrid_id = hybrid.AFFINITY_POLICY_ID
    affinity.REMOTE_BUCKETS = REMOTE_BUCKETS
    affinity.POLICY_ID = POLICY_ID
    hybrid.AFFINITY_POLICY_ID = POLICY_ID
    try:
        return phase.main(argv)
    finally:
        affinity.REMOTE_BUCKETS = old_buckets
        affinity.POLICY_ID = old_affinity_id
        hybrid.AFFINITY_POLICY_ID = old_hybrid_id


if __name__ == "__main__":
    raise SystemExit(main())
