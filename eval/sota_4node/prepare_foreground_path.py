#!/usr/bin/env python3
"""Validate and canonically publish a measured foreground-path record.

This helper is deliberately a *boundary*, not a counter collector.  A GPU
instrumentation process or site-specific capture tool must provide the input
JSON.  The helper refuses topology-only labels, host-wide totals, missing
domains, non-monotonic series, and zero-traffic samples, then writes the
canonical record consumed by ``build_g1_causal_readiness.py`` and the live
result validators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

try:
    from tempo.foreground_path import validate_foreground_path
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tempo.foreground_path import validate_foreground_path


def prepare_foreground_path(
    input_path: Path,
    output_path: Path,
    *,
    intervention_id: str = "fg_only",
) -> dict[str, Any]:
    """Validate ``input_path`` and publish deterministic JSON to ``output_path``."""

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    normalized = validate_foreground_path(raw, intervention_id=intervention_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")) + "\n"
    output_path.write_text(encoded, encoding="utf-8")
    # Keep a separate digest so the raw input and published record can be
    # bound in a source manifest without weakening the exact JSON schema.
    digest_path = output_path.with_suffix(output_path.suffix + ".sha256")
    digest_path.write_text(hashlib.sha256(encoded.encode("utf-8")).hexdigest() + "\n", encoding="utf-8")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--intervention-id", default="fg_only")
    args = parser.parse_args()
    prepare_foreground_path(args.input, args.output, intervention_id=args.intervention_id)


if __name__ == "__main__":
    main()
