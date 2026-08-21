#!/usr/bin/env python3
"""Punctuation-preserving revision of the unique-chunk mixed client."""

from eval.sota_4node import run_tempo_pd_same_server_mixed_only_client_unique_chunks_v305 as base


_BASE_MARKER = base._marker


def _marker(marker_id: int) -> str:
    return _BASE_MARKER(marker_id) + "."


def _rows(source, phase):
    original = base._marker
    base._marker = _marker
    try:
        return base._rows(source, phase)
    finally:
        base._marker = original


def main() -> int:
    original = base._marker
    base._marker = _marker
    try:
        return base.main()
    finally:
        base._marker = original


if __name__ == "__main__":
    raise SystemExit(main())
