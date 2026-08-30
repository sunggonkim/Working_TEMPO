#!/usr/bin/env python3
"""v17 node wrapper selecting the internal-ID native proxy."""

from __future__ import annotations

from pathlib import Path

from eval.sota_4node import vllm_native_nixl_remote_node_v15 as base


def _proxy_command(python: Path, _proxy_script: Path, _model: Path, *,
                   prefill_host: str, decode_host: str, ports: dict[str, int]):
    return [
        str(python), "-m", "eval.sota_4node.native_nixl_pd_proxy_v17",
        "--host", "0.0.0.0", "--port", str(ports["proxy_http"]),
        "--prefill-url", f"http://{prefill_host}:{ports['prefill_api']}",
        "--decode-url", f"http://{decode_host}:{ports['decode_api']}",
        "--served-model", base.base.SERVED_MODEL,
    ]


def main() -> int:
    base._proxy_command = _proxy_command
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
