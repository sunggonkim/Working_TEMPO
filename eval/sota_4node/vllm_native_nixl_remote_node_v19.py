#!/usr/bin/env python3
"""v19 native-Nixl node using its exact single-decoder stream contract."""

from __future__ import annotations

from eval.sota_4node import vllm_native_nixl_remote_node_v17 as v17


_ORIGINAL_CLIENT_COMMAND = v17.base.stream_v3._client_command


def _client_command(*args, **kwargs):
    command = _ORIGINAL_CLIENT_COMMAND(*args, **kwargs)
    index = command.index("eval.sota_4node.run_tempo_pd_stream_metrics_v3")
    command[index] = "eval.sota_4node.run_tempo_pd_stream_metrics_native_v18"
    return command


def main() -> int:
    v17.base.stream_v3._client_command = _client_command
    return v17.main()


if __name__ == "__main__":
    raise SystemExit(main())
