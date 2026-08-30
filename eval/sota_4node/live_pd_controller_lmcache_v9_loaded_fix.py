"""Callback-normalized entry for the loaded live-P/D experiment."""

from __future__ import annotations

from typing import Any, Callable

from eval.sota_4node import live_pd_controller_lmcache_v8_loaded as loaded
from eval.sota_4node import live_pd_controller_v1 as base


_ORIGINAL_WITH_BACKGROUND = loaded._with_background


def _with_background(*args, foreground: Callable[[], Any] | None = None, **kwargs):
    if foreground is None:
        foreground = args[-1]
        args = args[:-1]

    def normalized() -> dict[str, Any]:
        result = foreground()
        if callable(result):
            result = result()
        base._require(isinstance(result, dict), "foreground callback must return an object")
        return result

    return _ORIGINAL_WITH_BACKGROUND(*args, foreground=normalized, **kwargs)


def main() -> int:
    old = loaded._with_background
    loaded._with_background = _with_background
    try:
        return loaded.main()
    finally:
        loaded._with_background = old


if __name__ == "__main__":
    raise SystemExit(main())
