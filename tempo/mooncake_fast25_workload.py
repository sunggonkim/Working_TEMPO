"""Strict Mooncake FAST'25 trace ingestion and token-ID materialization.

Mooncake intentionally releases timing, token lengths, and remapped 512-token
prefix-block hashes rather than private prompts.  This module turns a bounded,
contiguous trace window into deterministic token IDs while preserving the
released prefix-sharing relation.  Every transformation that changes the
offered population is counted in the returned manifest.

The module does not launch a server, submit Slurm work, or infer a load level.
Capacity normalization and business/SLO assignment belong to the experiment
contract that consumes the materialized population.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCE_MANIFEST_SCHEMA = "tempo-go-mooncake-fast25-source-manifest-v1"
POPULATION_MANIFEST_SCHEMA = "tempo-go-mooncake-fast25-population-manifest-v1"
TOKEN_WORKLOAD_SCHEMA = "tempo-go-token-id-stream-workload-jsonl-v1"
TRACE_FIELDS = frozenset({
    "timestamp", "input_length", "output_length", "hash_ids",
})
WORKLOAD_FIELDS = frozenset({
    "request_id", "prompt", "max_tokens", "arrival_offset_ms",
})
BLOCK_SIZE_TOKENS = 512
CONTEXT_POLICIES = frozenset({"reject", "prefix_clip"})
DEFAULT_MAPPING_SEED = "tempo-go-mooncake-fast25-token-map-v1"


class TraceContractError(ValueError):
    """The source or generated population violates its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceContractError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _git_blob_sha1(value: bytes) -> str:
    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    _require(type(value) is int and value > 0, f"{field} must be a positive int")
    return int(value)


def _nonnegative_int(value: Any, field: str) -> int:
    _require(
        type(value) is int and value >= 0,
        f"{field} must be a non-negative int",
    )
    return int(value)


def _positive_float(value: Any, field: str) -> float:
    _require(
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0,
        f"{field} must be finite and positive",
    )
    return float(value)


@dataclass(frozen=True)
class TraceRow:
    """One validated upstream trace row."""

    source_index: int
    timestamp_ms: int
    input_tokens: int
    output_tokens: int
    hash_ids: tuple[int, ...]


@dataclass(frozen=True)
class MaterializationSpec:
    """A bounded and fully explicit trace-to-token transformation."""

    trace_name: str
    start_index: int
    request_count: int
    arrival_load_multiplier: float = 1.0
    max_model_len: int = 32_768
    min_output_tokens: int = 2
    max_output_tokens: int = 512
    context_policy: str = "prefix_clip"
    token_id_min: int = 1_000
    token_id_max_exclusive: int = 120_000
    mapping_seed: str = DEFAULT_MAPPING_SEED
    max_materialized_tokens: int = 20_000_000

    def __post_init__(self) -> None:
        _require(
            isinstance(self.trace_name, str) and bool(self.trace_name.strip()),
            "trace_name must be nonempty",
        )
        _nonnegative_int(self.start_index, "start_index")
        _positive_int(self.request_count, "request_count")
        _positive_float(self.arrival_load_multiplier, "arrival_load_multiplier")
        _positive_int(self.max_model_len, "max_model_len")
        _positive_int(self.min_output_tokens, "min_output_tokens")
        _positive_int(self.max_output_tokens, "max_output_tokens")
        _require(
            self.min_output_tokens <= self.max_output_tokens,
            "min_output_tokens exceeds max_output_tokens",
        )
        _require(
            self.max_output_tokens < self.max_model_len,
            "max_output_tokens must leave room for a prompt token",
        )
        _require(
            self.context_policy in CONTEXT_POLICIES,
            f"unsupported context_policy: {self.context_policy}",
        )
        _nonnegative_int(self.token_id_min, "token_id_min")
        _positive_int(self.token_id_max_exclusive, "token_id_max_exclusive")
        _require(
            self.token_id_max_exclusive - self.token_id_min >= 1024,
            "token-ID interval must contain at least 1024 IDs",
        )
        _require(
            isinstance(self.mapping_seed, str) and bool(self.mapping_seed),
            "mapping_seed must be nonempty",
        )
        _positive_int(self.max_materialized_tokens, "max_materialized_tokens")


def load_source_manifest(path: Path) -> dict[str, Any]:
    """Load one explicitly named source manifest without file discovery."""

    _require(path.is_file(), f"source manifest is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TraceContractError(f"cannot read source manifest: {path}") from exc
    _require(isinstance(value, dict), "source manifest must be an object")
    _require(
        value.get("schema") == SOURCE_MANIFEST_SCHEMA,
        "source manifest schema differs",
    )
    trace_contract = value.get("trace_contract")
    _require(isinstance(trace_contract, dict), "trace_contract is missing")
    _require(
        trace_contract.get("block_size_tokens") == BLOCK_SIZE_TOKENS,
        "source block size differs",
    )
    _require(
        trace_contract.get("fields") == [
            "timestamp", "input_length", "output_length", "hash_ids",
        ],
        "source field order/contract differs",
    )
    upstream = value.get("upstream")
    _require(isinstance(upstream, dict), "upstream identity is missing")
    commit = upstream.get("commit")
    _require(
        isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit),
        "upstream commit is not a full lowercase Git SHA",
    )
    files = value.get("files")
    _require(isinstance(files, dict) and bool(files), "source files are missing")
    return value


def _resolve_source_path(manifest_path: Path, local_path: str) -> Path:
    _require(
        isinstance(local_path, str) and bool(local_path),
        "source local_path must be nonempty",
    )
    relative = Path(local_path)
    _require(not relative.is_absolute(), "source local_path must be relative")
    root = manifest_path.resolve().parent
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TraceContractError("source local_path escapes the data root") from exc
    return resolved


def load_trace(
    source_manifest_path: Path,
    trace_name: str,
) -> tuple[tuple[TraceRow, ...], dict[str, Any]]:
    """Verify and parse one exact Mooncake source file."""

    manifest = load_source_manifest(source_manifest_path)
    files = manifest["files"]
    _require(trace_name in files, f"unknown trace: {trace_name}")
    entry = files[trace_name]
    _require(isinstance(entry, dict), f"trace entry is invalid: {trace_name}")
    source_path = _resolve_source_path(
        source_manifest_path, entry.get("local_path"),
    )
    _require(source_path.is_file(), f"trace file is missing: {source_path}")
    raw = source_path.read_bytes()
    _require(len(raw) == entry.get("bytes"), "trace byte count differs")
    _require(_sha256_bytes(raw) == entry.get("sha256"), "trace SHA-256 differs")
    _require(
        _git_blob_sha1(raw) == entry.get("git_blob_sha1"),
        "trace Git blob identity differs",
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TraceContractError("trace is not UTF-8") from exc

    rows: list[TraceRow] = []
    previous_timestamp = -1
    for line_number, line in enumerate(text.splitlines(), start=1):
        _require(bool(line.strip()), f"blank trace line: {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceContractError(
                f"invalid trace JSON at line {line_number}: {exc}",
            ) from exc
        _require(isinstance(value, dict), f"trace line {line_number} is not an object")
        _require(
            set(value) == TRACE_FIELDS,
            f"trace line {line_number} fields differ: {sorted(value)}",
        )
        timestamp = _nonnegative_int(
            value["timestamp"], f"trace line {line_number} timestamp",
        )
        _require(
            timestamp >= previous_timestamp,
            f"trace timestamps decrease at line {line_number}",
        )
        input_tokens = _positive_int(
            value["input_length"], f"trace line {line_number} input_length",
        )
        output_tokens = _positive_int(
            value["output_length"], f"trace line {line_number} output_length",
        )
        raw_hash_ids = value["hash_ids"]
        _require(
            isinstance(raw_hash_ids, list)
            and bool(raw_hash_ids)
            and all(type(item) is int and item >= 0 for item in raw_hash_ids),
            f"trace line {line_number} hash_ids are invalid",
        )
        expected_blocks = math.ceil(input_tokens / BLOCK_SIZE_TOKENS)
        _require(
            len(raw_hash_ids) == expected_blocks,
            f"trace line {line_number} hash count differs from input length",
        )
        rows.append(TraceRow(
            source_index=line_number - 1,
            timestamp_ms=timestamp,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            hash_ids=tuple(raw_hash_ids),
        ))
        previous_timestamp = timestamp
    _require(len(rows) == entry.get("requests"), "trace request count differs")
    receipt = {
        "trace_name": trace_name,
        "source_path": str(source_path),
        "source_sha256": entry["sha256"],
        "source_git_blob_sha1": entry["git_blob_sha1"],
        "source_bytes": entry["bytes"],
        "source_requests": entry["requests"],
        "upstream_commit": manifest["upstream"]["commit"],
        "upstream_url": entry["url"],
        "source_manifest_path": str(source_manifest_path.resolve()),
        "source_manifest_sha256": _sha256_bytes(source_manifest_path.read_bytes()),
    }
    return tuple(rows), receipt


def _token_block(
    *,
    block_id: int,
    namespace: str,
    token_id_min: int,
    token_id_max_exclusive: int,
) -> tuple[int, ...]:
    """Return one deterministic synthetic block for one remapped hash ID."""

    width = token_id_max_exclusive - token_id_min
    identity = _canonical_json_bytes({
        "schema": "tempo-go-mooncake-token-block-v1",
        "namespace": namespace,
        "block_id": block_id,
    })
    tokens: list[int] = []
    counter = 0
    while len(tokens) < BLOCK_SIZE_TOKENS:
        digest = hashlib.sha256(
            identity + counter.to_bytes(4, byteorder="big", signed=False),
        ).digest()
        for offset in range(0, len(digest), 4):
            number = int.from_bytes(
                digest[offset:offset + 4], byteorder="big", signed=False,
            )
            tokens.append(token_id_min + number % width)
            if len(tokens) == BLOCK_SIZE_TOKENS:
                break
        counter += 1
    return tuple(tokens)


def _quantiles(values: Sequence[int | float]) -> dict[str, float]:
    _require(bool(values), "cannot summarize an empty sequence")
    ordered = sorted(float(value) for value in values)

    def pick(probability: float) -> float:
        index = int(math.ceil(probability * len(ordered))) - 1
        return ordered[max(0, min(index, len(ordered) - 1))]

    return {
        "min": ordered[0],
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _prior_reusable_prefix_tokens(
    block_ids: Sequence[int],
    *,
    complete_blocks: int,
    prefix_trie: dict[int, Any],
) -> int:
    node = prefix_trie
    reusable_blocks = 0
    for block_id in block_ids[:complete_blocks]:
        child = node.get(block_id)
        if child is None:
            break
        reusable_blocks += 1
        node = child
    node = prefix_trie
    for block_id in block_ids[:complete_blocks]:
        node = node.setdefault(block_id, {})
    return reusable_blocks * BLOCK_SIZE_TOKENS


def build_population(
    rows: Sequence[TraceRow],
    source_receipt: Mapping[str, Any],
    spec: MaterializationSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    """Build one contiguous, deterministic token-ID population."""

    _require(bool(rows), "source trace is empty")
    _require(
        source_receipt.get("trace_name") == spec.trace_name,
        "materialization trace differs from source receipt",
    )
    stop_index = spec.start_index + spec.request_count
    _require(stop_index <= len(rows), "selected window exceeds the source trace")
    selected = tuple(rows[spec.start_index:stop_index])
    first_timestamp = selected[0].timestamp_ms
    namespace = (
        f"{spec.mapping_seed}|{spec.trace_name}|"
        f"{source_receipt['source_sha256']}"
    )
    block_cache: dict[int, tuple[int, ...]] = {}
    block_digests: dict[int, str] = {}
    materialized_block_digest_set: set[str] = set()
    prefix_trie: dict[int, Any] = {}
    workload: list[dict[str, Any]] = []
    request_index: dict[str, dict[str, Any]] = {}
    semantic_rows: list[dict[str, Any]] = []
    materialized_tokens = 0
    context_clipped_requests = 0
    context_clipped_tokens = 0
    output_floor_requests = 0
    output_floor_added_tokens = 0
    output_cap_requests = 0
    output_cap_removed_tokens = 0
    full_block_occurrences = 0
    block_occurrences = 0
    unique_block_ids: set[int] = set()
    reusable_prefix_tokens: list[int] = []

    for row in selected:
        effective_output = row.output_tokens
        if effective_output < spec.min_output_tokens:
            output_floor_requests += 1
            output_floor_added_tokens += spec.min_output_tokens - effective_output
            effective_output = spec.min_output_tokens
        if effective_output > spec.max_output_tokens:
            output_cap_requests += 1
            output_cap_removed_tokens += effective_output - spec.max_output_tokens
            effective_output = spec.max_output_tokens
        maximum_prompt = spec.max_model_len - effective_output
        _require(maximum_prompt > 0, "output leaves no model context for prompt")
        effective_input = row.input_tokens
        if effective_input > maximum_prompt:
            if spec.context_policy == "reject":
                raise TraceContractError(
                    "selected request exceeds max_model_len under reject policy: "
                    f"source_index={row.source_index} input={row.input_tokens} "
                    f"output={effective_output} max_model_len={spec.max_model_len}",
                )
            context_clipped_requests += 1
            context_clipped_tokens += effective_input - maximum_prompt
            effective_input = maximum_prompt
        materialized_tokens += effective_input
        _require(
            materialized_tokens <= spec.max_materialized_tokens,
            "selected population exceeds max_materialized_tokens",
        )
        used_blocks = math.ceil(effective_input / BLOCK_SIZE_TOKENS)
        effective_hash_ids = row.hash_ids[:used_blocks]
        prompt: list[int] = []
        for block_id in effective_hash_ids:
            block = block_cache.get(block_id)
            if block is None:
                block = _token_block(
                    block_id=block_id,
                    namespace=namespace,
                    token_id_min=spec.token_id_min,
                    token_id_max_exclusive=spec.token_id_max_exclusive,
                )
                digest = _sha256_bytes(_canonical_json_bytes(block))
                _require(
                    digest not in materialized_block_digest_set,
                    "distinct hash IDs materialized to the same token block",
                )
                block_cache[block_id] = block
                block_digests[block_id] = digest
                materialized_block_digest_set.add(digest)
            prompt.extend(block)
        prompt = prompt[:effective_input]
        _require(len(prompt) == effective_input, "materialized prompt length differs")
        complete_blocks = effective_input // BLOCK_SIZE_TOKENS
        prior_reuse = _prior_reusable_prefix_tokens(
            effective_hash_ids,
            complete_blocks=complete_blocks,
            prefix_trie=prefix_trie,
        )
        reusable_prefix_tokens.append(prior_reuse)
        full_block_occurrences += complete_blocks
        block_occurrences += used_blocks
        unique_block_ids.update(effective_hash_ids)
        arrival_offset_ms = round(
            (row.timestamp_ms - first_timestamp)
            / spec.arrival_load_multiplier,
            6,
        )
        request_id = f"mooncake-{spec.trace_name}-{row.source_index:06d}"
        workload.append({
            "request_id": request_id,
            "prompt": prompt,
            "max_tokens": effective_output,
            "arrival_offset_ms": arrival_offset_ms,
        })
        metadata = {
            "source_index": row.source_index,
            "source_timestamp_ms": row.timestamp_ms,
            "original_input_tokens": row.input_tokens,
            "effective_input_tokens": effective_input,
            "original_output_tokens": row.output_tokens,
            "effective_output_tokens": effective_output,
            "used_hash_blocks": used_blocks,
            "complete_hash_blocks": complete_blocks,
            "hash_ids_sha256": _sha256_bytes(
                _canonical_json_bytes(effective_hash_ids),
            ),
            "prior_reusable_prefix_tokens_upper_bound": prior_reuse,
            "context_clipped_tokens": row.input_tokens - effective_input,
            "output_floor_added_tokens": max(0, spec.min_output_tokens - row.output_tokens),
            "output_cap_removed_tokens": max(0, row.output_tokens - spec.max_output_tokens),
        }
        request_index[request_id] = metadata
        semantic_rows.append({"request_id": request_id, **metadata})

    _require(
        all(
            float(left["arrival_offset_ms"]) <= float(right["arrival_offset_ms"])
            for left, right in zip(workload, workload[1:])
        ),
        "materialized arrivals are not monotonic",
    )
    workload_bytes = b"".join(
        _canonical_json_bytes(row) + b"\n" for row in workload
    )
    duration_ms = float(workload[-1]["arrival_offset_ms"])
    offered_rate = (
        (len(workload) - 1) * 1000.0 / duration_ms
        if len(workload) > 1 and duration_ms > 0.0
        else None
    )
    repeated_occurrences = block_occurrences - len(unique_block_ids)
    manifest: dict[str, Any] = {
        "schema": POPULATION_MANIFEST_SCHEMA,
        "workload_schema": TOKEN_WORKLOAD_SCHEMA,
        "source": dict(source_receipt),
        "selection": {
            "contiguous_source_window": True,
            "start_index": spec.start_index,
            "stop_index_exclusive": stop_index,
            "request_count": spec.request_count,
            "first_source_timestamp_ms": selected[0].timestamp_ms,
            "last_source_timestamp_ms": selected[-1].timestamp_ms,
        },
        "arrival": {
            "semantics": "explicit_offset_ms_from_first_selected_request",
            "source_order_preserved": True,
            "load_multiplier": spec.arrival_load_multiplier,
            "duration_ms": duration_ms,
            "effective_offered_rate_per_s": offered_rate,
        },
        "context": {
            "max_model_len": spec.max_model_len,
            "policy": spec.context_policy,
            "clipped_requests": context_clipped_requests,
            "clipped_input_tokens": context_clipped_tokens,
            "silent_drop_count": 0,
        },
        "output": {
            "min_output_tokens": spec.min_output_tokens,
            "max_output_tokens": spec.max_output_tokens,
            "floor_adjusted_requests": output_floor_requests,
            "floor_added_tokens": output_floor_added_tokens,
            "cap_adjusted_requests": output_cap_requests,
            "cap_removed_tokens": output_cap_removed_tokens,
        },
        "token_materialization": {
            "schema": "tempo-go-mooncake-hash-block-token-map-v1",
            "block_size_tokens": BLOCK_SIZE_TOKENS,
            "mapping_seed": spec.mapping_seed,
            "namespace_sha256": _sha256_bytes(namespace.encode("utf-8")),
            "token_id_min_inclusive": spec.token_id_min,
            "token_id_max_exclusive": spec.token_id_max_exclusive,
            "tokenizer_runtime_validation_required": True,
            "materialized_prompt_tokens": materialized_tokens,
            "max_materialized_tokens_gate": spec.max_materialized_tokens,
            "unique_hash_blocks": len(unique_block_ids),
            "block_token_digest_collision_count": 0,
        },
        "reuse": {
            "semantics": (
                "upper bound from a prior request sharing complete leading "
                "512-token hash blocks; eviction and placement are not assumed"
            ),
            "block_occurrences_including_partial_tail": block_occurrences,
            "complete_block_occurrences": full_block_occurrences,
            "repeated_block_occurrences_including_nonprefix": repeated_occurrences,
            "requests_with_prior_reusable_prefix": sum(
                value > 0 for value in reusable_prefix_tokens
            ),
            "prior_reusable_prefix_tokens": _quantiles(reusable_prefix_tokens),
        },
        "distribution": {
            "original_input_tokens": _quantiles(
                [row.input_tokens for row in selected],
            ),
            "effective_input_tokens": _quantiles(
                [entry["effective_input_tokens"] for entry in request_index.values()],
            ),
            "original_output_tokens": _quantiles(
                [row.output_tokens for row in selected],
            ),
            "effective_output_tokens": _quantiles(
                [entry["effective_output_tokens"] for entry in request_index.values()],
            ),
        },
        "population_semantic_sha256": _sha256_bytes(
            _canonical_json_bytes(semantic_rows),
        ),
        "workload_sha256": _sha256_bytes(workload_bytes),
        "request_index": request_index,
        "policy_inputs_excluded": [
            "future_arrivals", "oracle_route", "future_cache_evictions",
            "physical_switch_label",
        ],
        "performance_claim_allowed": False,
    }
    return workload, manifest, workload_bytes


def verify_population(
    workload_bytes: bytes,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on a materialized workload/sidecar mismatch."""

    _require(
        manifest.get("schema") == POPULATION_MANIFEST_SCHEMA,
        "population manifest schema differs",
    )
    _require(
        manifest.get("workload_schema") == TOKEN_WORKLOAD_SCHEMA,
        "token workload schema differs",
    )
    _require(
        _sha256_bytes(workload_bytes) == manifest.get("workload_sha256"),
        "materialized workload SHA-256 differs",
    )
    context = manifest.get("context")
    token_contract = manifest.get("token_materialization")
    selection = manifest.get("selection")
    request_index = manifest.get("request_index")
    _require(isinstance(context, dict), "context receipt is missing")
    _require(isinstance(token_contract, dict), "token receipt is missing")
    _require(isinstance(selection, dict), "selection receipt is missing")
    _require(isinstance(request_index, dict), "request index is missing")
    max_model_len = _positive_int(context.get("max_model_len"), "max_model_len")
    token_min = _nonnegative_int(
        token_contract.get("token_id_min_inclusive"), "token_id_min",
    )
    token_max = _positive_int(
        token_contract.get("token_id_max_exclusive"), "token_id_max",
    )
    rows: list[dict[str, Any]] = []
    try:
        text = workload_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TraceContractError("materialized workload is not UTF-8") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        _require(bool(line.strip()), f"blank workload line: {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TraceContractError(
                f"invalid workload JSON at line {line_number}: {exc}",
            ) from exc
        _require(isinstance(value, dict), f"workload line {line_number} is not an object")
        _require(
            set(value) == WORKLOAD_FIELDS,
            f"workload line {line_number} fields differ",
        )
        request_id = value["request_id"]
        _require(
            isinstance(request_id, str) and bool(request_id),
            f"workload line {line_number} request_id is invalid",
        )
        prompt = value["prompt"]
        _require(
            isinstance(prompt, list)
            and bool(prompt)
            and all(type(token) is int and token_min <= token < token_max for token in prompt),
            f"workload line {line_number} prompt token IDs are invalid",
        )
        output_tokens = _positive_int(
            value["max_tokens"], f"workload line {line_number} max_tokens",
        )
        _require(
            len(prompt) + output_tokens <= max_model_len,
            f"workload line {line_number} exceeds max_model_len",
        )
        _require(
            type(value["arrival_offset_ms"]) in (int, float)
            and not isinstance(value["arrival_offset_ms"], bool)
            and math.isfinite(float(value["arrival_offset_ms"]))
            and float(value["arrival_offset_ms"]) >= 0.0,
            f"workload line {line_number} arrival is invalid",
        )
        metadata = request_index.get(request_id)
        _require(isinstance(metadata, dict), f"request index missing: {request_id}")
        _require(
            metadata.get("effective_input_tokens") == len(prompt)
            and metadata.get("effective_output_tokens") == output_tokens,
            f"request index geometry differs: {request_id}",
        )
        rows.append(value)
    _require(bool(rows), "materialized workload is empty")
    _require(
        len(rows) == selection.get("request_count") == len(request_index),
        "materialized request count differs",
    )
    identifiers = [row["request_id"] for row in rows]
    _require(len(identifiers) == len(set(identifiers)), "request IDs are not unique")
    _require(
        all(
            float(left["arrival_offset_ms"]) <= float(right["arrival_offset_ms"])
            for left, right in zip(rows, rows[1:])
        ),
        "materialized arrivals decrease",
    )
    _require(float(rows[0]["arrival_offset_ms"]) == 0.0, "first arrival is not zero")
    return {
        "schema": "tempo-go-mooncake-fast25-population-verification-v1",
        "request_count": len(rows),
        "workload_sha256": _sha256_bytes(workload_bytes),
        "max_model_len": max_model_len,
        "all_rows_valid": True,
        "performance_claim_allowed": False,
    }


def load_and_verify_population(
    workload_path: Path,
    population_manifest_path: Path,
) -> dict[str, Any]:
    _require(workload_path.is_file(), f"workload is not a file: {workload_path}")
    _require(
        population_manifest_path.is_file(),
        f"population manifest is not a file: {population_manifest_path}",
    )
    try:
        manifest = json.loads(population_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TraceContractError("cannot read population manifest") from exc
    _require(isinstance(manifest, dict), "population manifest must be an object")
    return verify_population(workload_path.read_bytes(), manifest)


__all__ = [
    "BLOCK_SIZE_TOKENS",
    "CONTEXT_POLICIES",
    "DEFAULT_MAPPING_SEED",
    "MaterializationSpec",
    "POPULATION_MANIFEST_SCHEMA",
    "SOURCE_MANIFEST_SCHEMA",
    "TOKEN_WORKLOAD_SCHEMA",
    "TraceContractError",
    "TraceRow",
    "build_population",
    "load_and_verify_population",
    "load_source_manifest",
    "load_trace",
    "verify_population",
]
