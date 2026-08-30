#!/usr/bin/env python3
"""Build or verify a bounded Mooncake FAST'25 token-ID population."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "tempo/mooncake_fast25_workload.py"
_SPEC = importlib.util.spec_from_file_location(
    "_tempo_mooncake_fast25_workload_cli", _MODULE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load Mooncake workload module: {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
CONTEXT_POLICIES = _MODULE.CONTEXT_POLICIES
DEFAULT_MAPPING_SEED = _MODULE.DEFAULT_MAPPING_SEED
MaterializationSpec = _MODULE.MaterializationSpec
TraceContractError = _MODULE.TraceContractError
build_population = _MODULE.build_population
load_and_verify_population = _MODULE.load_and_verify_population
load_source_manifest = _MODULE.load_source_manifest
load_trace = _MODULE.load_trace


DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT
    / "eval/sota_4node/data/mooncake_fast25/source_manifest_v1.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser(
        "verify-source", help="verify one pinned upstream trace",
    )
    source.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST,
    )
    source.add_argument("--trace", required=True)

    build = subparsers.add_parser(
        "build", help="materialize one contiguous source window",
    )
    build.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST,
    )
    build.add_argument("--trace", required=True)
    build.add_argument("--start-index", type=int, required=True)
    build.add_argument("--request-count", type=int, required=True)
    build.add_argument("--arrival-load-multiplier", type=float, default=1.0)
    build.add_argument("--max-model-len", type=int, default=32_768)
    build.add_argument("--min-output-tokens", type=int, default=2)
    build.add_argument("--max-output-tokens", type=int, default=512)
    build.add_argument(
        "--context-policy", choices=sorted(CONTEXT_POLICIES),
        default="prefix_clip",
    )
    build.add_argument("--token-id-min", type=int, default=1_000)
    build.add_argument("--token-id-max-exclusive", type=int, default=120_000)
    build.add_argument("--mapping-seed", default=DEFAULT_MAPPING_SEED)
    build.add_argument("--max-materialized-tokens", type=int, default=20_000_000)
    build.add_argument("--output-workload", type=Path, required=True)
    build.add_argument("--output-manifest", type=Path, required=True)

    verify = subparsers.add_parser(
        "verify", help="verify a materialized workload against its sidecar",
    )
    verify.add_argument("--workload", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def _refuse_overwrite(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise TraceContractError(f"refusing to overwrite: {existing}")


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise TraceContractError(f"temporary output already exists: {temporary}")
    temporary.write_bytes(value)
    temporary.replace(path)


def _build(args: argparse.Namespace) -> dict[str, object]:
    _refuse_overwrite((args.output_workload, args.output_manifest))
    rows, source_receipt = load_trace(args.source_manifest, args.trace)
    spec = MaterializationSpec(
        trace_name=args.trace,
        start_index=args.start_index,
        request_count=args.request_count,
        arrival_load_multiplier=args.arrival_load_multiplier,
        max_model_len=args.max_model_len,
        min_output_tokens=args.min_output_tokens,
        max_output_tokens=args.max_output_tokens,
        context_policy=args.context_policy,
        token_id_min=args.token_id_min,
        token_id_max_exclusive=args.token_id_max_exclusive,
        mapping_seed=args.mapping_seed,
        max_materialized_tokens=args.max_materialized_tokens,
    )
    _workload, manifest, workload_bytes = build_population(
        rows, source_receipt, spec,
    )
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _atomic_write(args.output_workload, workload_bytes)
    try:
        _atomic_write(args.output_manifest, manifest_bytes)
    except BaseException:
        # The workload is new and this command refused any pre-existing target,
        # so cleanup restores the all-or-nothing output contract.
        args.output_workload.unlink(missing_ok=True)
        raise
    verification = load_and_verify_population(
        args.output_workload, args.output_manifest,
    )
    return {
        "command": "build",
        "trace": args.trace,
        "workload": str(args.output_workload.resolve()),
        "manifest": str(args.output_manifest.resolve()),
        "request_count": verification["request_count"],
        "workload_sha256": verification["workload_sha256"],
        "performance_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-source":
            manifest = load_source_manifest(args.source_manifest)
            rows, receipt = load_trace(args.source_manifest, args.trace)
            result = {
                "command": "verify-source",
                "schema": manifest["schema"],
                "trace": args.trace,
                "requests": len(rows),
                "source_sha256": receipt["source_sha256"],
                "source_git_blob_sha1": receipt["source_git_blob_sha1"],
                "valid": True,
            }
        elif args.command == "build":
            result = _build(args)
        else:
            result = {
                "command": "verify",
                **load_and_verify_population(args.workload, args.manifest),
            }
    except (OSError, TraceContractError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
