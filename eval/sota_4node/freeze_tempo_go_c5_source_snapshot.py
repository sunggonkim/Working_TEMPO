#!/usr/bin/env python3
"""Create an immutable C5 source tree and rebind a C5 contract to it.

The native C5 launcher runs inside a shared worktree. A concurrent edit in
that worktree must not invalidate an arm halfway through a campaign or make
different arms execute different controller code. This helper copies the
bounded TEMPO/evaluation Python trees and the complete shell launcher closure,
verifies that every copied input was unchanged during the copy, and writes a
contract whose source bindings point at the resulting snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


CONTRACT_SCHEMA = "tempo-go-c5-native-run-contract-v1"
SNAPSHOT_SCHEMA = "tempo-go-source-snapshot-v1"
PYTHON_ROOTS = (
  Path("tempo"),
  Path("eval/sota_4node"),
  Path("third_party/lmcache/lmcache"),
)
SNAPSHOT_EXTRA_FILES = (
    Path("eval/sota_4node/run_lmcache_nixl_contention_2node_in_allocation.sh"),
    Path("eval/sota_4node/run_tempo_go_cross_layer_with_cojob_in_allocation.sh"),
    Path("eval/sota_4node/run_tempo_go_cxi_background_with_c5_in_allocation.sh"),
    Path("eval/sota_4node/cxi_background_traffic.c"),
    Path("eval/sota_4node/prepare_c4_python_overlay.sh"),
    Path("eval/sota_4node/stage_c4_python_overlay.sh"),
    Path("eval/sota_4node/require_perlmutter_4node_4h_interactive.sh"),
    Path("eval/sota_4node/c5_tempo_go_node_entry.sh"),
    Path("eval/sota_4node/run_tempo_go_c5_five_arm_in_allocation.sh"),
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def contract_fingerprint(value: dict[str, object]) -> str:
    payload = dict(value)
    payload.pop("fingerprint_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bounded_python_files(repo_root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative_root in PYTHON_ROOTS:
        root = (repo_root / relative_root).resolve()
        require(root.is_dir(), f"snapshot source root is missing: {root}")
        for directory, dirnames, names in os.walk(root):
            dirnames[:] = sorted(
                name for name in dirnames
                if name != "__pycache__" and not name.startswith(".")
            )
            for name in sorted(names):
                if name.endswith(".py"):
                    files.add((Path(directory) / name).resolve())
    for relative in SNAPSHOT_EXTRA_FILES:
        path = (repo_root / relative).resolve()
        require(path.is_file(), f"snapshot source file is missing: {path}")
        files.add(path)
    return sorted(files)


def copy_python_files(
    repo_root: Path, snapshot_root: Path, files: Iterable[Path],
) -> list[tuple[Path, Path]]:
    copied: list[tuple[Path, Path]] = []
    for source in files:
        relative = source.relative_to(repo_root)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append((source, destination))
    return copied


def snapshot_contract(
    base: dict[str, object],
    *,
    repo_root: Path,
    snapshot_root: Path,
    copied: list[tuple[Path, Path]],
    candidate_id: str | None = None,
    candidate_revision: str | None = None,
) -> dict[str, object]:
    value = json.loads(json.dumps(base))
    require(value.get("schema") == CONTRACT_SCHEMA, "base contract schema differs")
    previous_snapshot = value.get("source_snapshot")
    previous_snapshot_root: Path | None = None
    if isinstance(previous_snapshot, dict) and previous_snapshot.get("root"):
        previous_snapshot_root = Path(str(previous_snapshot["root"])).resolve()

    def current_source_path(raw_path: object) -> Path:
        path = Path(str(raw_path)).resolve()
        if previous_snapshot_root is not None:
            try:
                relative = path.relative_to(previous_snapshot_root)
            except ValueError:
                pass
            else:
                return (repo_root / relative).resolve()
        return path

    sources = value.get("source_inventory")
    require(isinstance(sources, dict), "base contract source inventory is missing")

    copied_by_source = {
        source.resolve(): destination for source, destination in copied
    }
    for name, binding in sources.items():
        require(isinstance(binding, dict), f"invalid source binding: {name}")
        source = current_source_path(binding["path"])
        destination = copied_by_source.get(source)
        if destination is not None:
            require(destination is not None,
                    f"Python source was not snapshotted: {name}")
            binding["path"] = str(destination)
            binding["sha256"] = digest(destination)
        else:
            binding["path"] = str(source)
            binding["sha256"] = digest(source)

    launcher = value.get("launcher")
    require(isinstance(launcher, dict), "base contract launcher is missing")
    analyzer = launcher.get("analyzer")
    require(isinstance(analyzer, dict),
            "base contract analyzer binding is missing")
    analyzer_source = current_source_path(analyzer["path"])
    analyzer_snapshot = copied_by_source.get(analyzer_source)
    require(analyzer_snapshot is not None, "analyzer was not snapshotted")
    analyzer["path"] = str(analyzer_snapshot)
    analyzer["sha256"] = digest(analyzer_snapshot)
    for launcher_name in ("runner", "node_entry"):
        launcher_binding = launcher.get(launcher_name)
        require(isinstance(launcher_binding, dict),
                f"base contract launcher lacks {launcher_name}")
        launcher_path = current_source_path(launcher_binding["path"])
        launcher_snapshot = copied_by_source.get(launcher_path)
        if launcher_snapshot is not None:
            launcher_binding["path"] = str(launcher_snapshot)
            launcher_binding["sha256"] = digest(launcher_snapshot)
        else:
            launcher_binding["path"] = str(launcher_path)
            launcher_binding["sha256"] = digest(launcher_path)

    candidate = value.get("candidate")
    require(isinstance(candidate, dict), "base contract candidate is missing")
    # Preserve the base candidate identity.  The snapshot is a source
    # boundary, not a new policy candidate; hard-coding the historical v9/v39
    # label here made a current snapshot look like an older experiment.
    candidate["id"] = str(
        candidate_id
        if candidate_id is not None
        else candidate.get("id", "tempo-go-cross-layer-seven-arm-source-snapshot")
    )
    candidate["revision"] = str(
        candidate_revision
        if candidate_revision is not None
        else candidate.get("revision", "immutable-python-source-snapshot")
    )
    candidate["controller_parameters_unchanged"] = True
    candidate["post_validation_tuning_allowed"] = False

    relative_digests = [
        f"{destination.relative_to(snapshot_root)}:{digest(destination)}"
        for _, destination in copied
    ]
    snapshot_digest = hashlib.sha256(
        "\n".join(relative_digests).encode("utf-8"),
    ).hexdigest()
    value["source_snapshot"] = {
        "schema": SNAPSHOT_SCHEMA,
        "root": str(snapshot_root),
        "python_file_count": sum(
            destination.suffix == ".py" for _, destination in copied),
        "shell_file_count": sum(
            destination.suffix == ".sh" for _, destination in copied),
        "source_file_count": len(copied),
        "tree_sha256": snapshot_digest,
    }
    value["fingerprint_sha256"] = contract_fingerprint(value)
    return value


def build(args: argparse.Namespace) -> dict[str, object]:
    repo_root = Path(args.repo_root).resolve()
    base_path = Path(args.base_contract).resolve()
    output_root = Path(args.output_root).resolve()
    require(repo_root.is_dir(), f"repository is missing: {repo_root}")
    require(repo_root in base_path.parents,
            "base contract must be below repository")
    require(base_path.is_file(), f"base contract is missing: {base_path}")
    require(repo_root / "results" in output_root.parents,
            "snapshot output must be below repository/results")
    require(not output_root.exists(),
            f"snapshot output already exists: {output_root}")

    output_root.mkdir(parents=True)
    if args.reuse_source_snapshot is not None:
        snapshot_root = Path(args.reuse_source_snapshot).resolve()
        require(snapshot_root.is_dir(),
                f"reused source snapshot is missing: {snapshot_root}")
        base = json.loads(base_path.read_text(encoding="utf-8"))
        declared_snapshot = base.get("source_snapshot")
        require(isinstance(declared_snapshot, dict),
                "base contract source snapshot is missing")
        require(Path(str(declared_snapshot.get("root"))).resolve()
                == snapshot_root,
                "reused source snapshot does not match base contract")
        sources = base.get("source_inventory")
        require(isinstance(sources, dict),
                "base contract source inventory is missing")
        for name, binding in sources.items():
            require(isinstance(binding, dict),
                    f"invalid source binding: {name}")
            path = Path(str(binding["path"])).resolve()
            try:
                path.relative_to(snapshot_root)
            except ValueError:
                require(path.is_file(), f"reused source binding is missing: {name}")
                binding["sha256"] = digest(path)
            else:
                require(path.is_file() and digest(path) == binding.get("sha256"),
                        f"reused source binding is stale: {name}")
        value = json.loads(json.dumps(base))
        launcher = value.get("launcher")
        require(isinstance(launcher, dict), "base contract launcher is missing")
        for launcher_name in ("runner", "node_entry"):
            binding = launcher.get(launcher_name)
            require(isinstance(binding, dict),
                    f"base contract launcher lacks {launcher_name}")
            path = (repo_root / Path(str(binding["path"])).name).resolve()
            if launcher_name == "runner":
                path = (repo_root / "eval/sota_4node/run_tempo_go_cross_layer_with_cojob_in_allocation.sh").resolve()
            else:
                path = (repo_root / "eval/sota_4node/c5_tempo_go_node_entry.sh").resolve()
            require(path.is_file(), f"launcher is missing: {path}")
            binding["path"] = str(path)
            binding["sha256"] = digest(path)
        candidate = value.get("candidate")
        require(isinstance(candidate, dict), "base contract candidate is missing")
        candidate["id"] = str(
            args.candidate_id
            if args.candidate_id is not None
            else candidate.get("id", "tempo-go-cross-layer-seven-arm-source-snapshot")
        )
        candidate["revision"] = str(
            args.candidate_revision
            if args.candidate_revision is not None
            else candidate.get("revision", "immutable-python-source-snapshot")
        )
        candidate["controller_parameters_unchanged"] = True
        candidate["post_validation_tuning_allowed"] = False
        value["fingerprint_sha256"] = contract_fingerprint(value)
        contract_path = output_root / "native_run_contract.json"
        contract_path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        return {
            "schema": SNAPSHOT_SCHEMA,
            "source_root": str(snapshot_root),
            "python_file_count": value["source_snapshot"]["python_file_count"],
            "tree_sha256": value["source_snapshot"]["tree_sha256"],
            "contract": str(contract_path),
            "contract_sha256": digest(contract_path),
            "fingerprint_sha256": value["fingerprint_sha256"],
        }
    snapshot_root = output_root / "source"
    snapshot_root.mkdir()
    files = bounded_python_files(repo_root)
    before = {path: digest(path) for path in files}
    copied = copy_python_files(repo_root, snapshot_root, files)
    after = {path: digest(path) for path in files}
    require(before == after,
            "source changed while snapshot was being copied; discard this attempt")

    base = json.loads(base_path.read_text(encoding="utf-8"))
    value = snapshot_contract(
        base,
        repo_root=repo_root,
        snapshot_root=snapshot_root,
        copied=copied,
        candidate_id=args.candidate_id,
        candidate_revision=args.candidate_revision,
    )
    contract_path = output_root / "native_run_contract.json"
    contract_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return {
        "schema": SNAPSHOT_SCHEMA,
        "source_root": str(snapshot_root),
        "python_file_count": sum(
            destination.suffix == ".py" for _, destination in copied),
        "shell_file_count": sum(
            destination.suffix == ".sh" for _, destination in copied),
        "source_file_count": len(copied),
        "tree_sha256": value["source_snapshot"]["tree_sha256"],
        "contract": str(contract_path),
        "contract_sha256": digest(contract_path),
        "fingerprint_sha256": value["fingerprint_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--base-contract", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--candidate-revision")
    parser.add_argument("--reuse-source-snapshot", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
