#!/usr/bin/env python3
"""Final evidence-gated candidate using the deterministic stream payload."""

from eval.sota_4node import vllm_lmcache_evidence_fail_local_node_v28 as v28


_ORIGINAL_CLIENT_COMMAND = v28.base.stream_v3._client_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    index = command.index("eval.sota_4node.run_tempo_pd_stream_metrics_v3")
    command[index] = "eval.sota_4node.run_tempo_pd_stream_metrics_forced_v32"
    return command


def main() -> int:
    v28.base.stream_v3._client_command = _client_command
    return v28.main()


if __name__ == "__main__":
    raise SystemExit(main())
