"""TP8 smoke v3: keep v2 semantics with a measured multi-node startup bound."""

from __future__ import annotations

import vllm_tp8_mp_smoke_node_v2 as base


base.READINESS_SECONDS = 300.0
base.REMOTE_FINISH_SECONDS = 480.0


if __name__ == "__main__":
    raise SystemExit(base.main())
