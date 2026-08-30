"""Run the pinned LMCache proxy while preserving upstream HTTP error bodies."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import httpx


_ORIGINAL_RAISE = httpx.Response.raise_for_status


def _raise_with_body(response: httpx.Response) -> httpx.Response:
    try:
        return _ORIGINAL_RAISE(response)
    except httpx.HTTPStatusError:
        print(
            "LMCache proxy upstream HTTP error body: " + response.text,
            file=sys.stderr,
            flush=True,
        )
        raise


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    upstream = repo / "third_party/lmcache/examples/disagg_prefill/disagg_proxy_server.py"
    if not upstream.is_file():
        raise FileNotFoundError(upstream)
    httpx.Response.raise_for_status = _raise_with_body
    runpy.run_path(str(upstream), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
