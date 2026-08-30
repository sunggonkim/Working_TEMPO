"""Explicit Perlmutter topology contract for the 1/2/4-node scale rungs.

The existing C9 campaign is intentionally fixed to P0/D0/P1/D1 on four
nodes.  Scale experiments must not obtain a different topology by changing a
launcher argument in an ad-hoc way, so this module is the single source for
the initial capacity-normalized rungs:

* 1 node: one local P/D pair, TP2 per role, two GPUs per role;
* 2 nodes: one inter-node P/D pair, TP4 per role;
* 4 nodes: two inter-node P/D pairs, TP4 per role.

These are topology contracts, not performance results.  A native runner must
bind its capability receipt to the returned object before starting vLLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Role = Literal["prefill", "decode"]
SUPPORTED_NODE_COUNTS = (1, 2, 4)
GPUS_PER_PERLMUTTER_NODE = 4


@dataclass(frozen=True)
class EndpointPlacement:
    """One vLLM endpoint's node and exclusive local GPU partition."""

    pair_index: int
    role: Role
    node_index: int
    gpu_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.pair_index) is not int or self.pair_index < 0:
            raise ValueError("pair_index must be a non-negative int")
        if self.role not in ("prefill", "decode"):
            raise ValueError("role must be prefill or decode")
        if type(self.node_index) is not int or self.node_index < 0:
            raise ValueError("node_index must be a non-negative int")
        if not self.gpu_indices or len(set(self.gpu_indices)) != len(self.gpu_indices):
            raise ValueError("gpu_indices must be non-empty and unique")
        if any(type(index) is not int or index < 0 for index in self.gpu_indices):
            raise ValueError("gpu_indices must contain non-negative ints")


@dataclass(frozen=True)
class ScaleTopology:
    """Immutable placement and TP contract for one scale rung."""

    node_count: int
    gpu_count: int
    tensor_parallel_size: int
    placements: tuple[EndpointPlacement, ...]

    def __post_init__(self) -> None:
        if self.node_count not in SUPPORTED_NODE_COUNTS:
            raise ValueError("node_count must be one of 1, 2, or 4")
        if self.gpu_count != self.node_count * GPUS_PER_PERLMUTTER_NODE:
            raise ValueError("gpu_count must equal four GPUs per node")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if not self.placements:
            raise ValueError("placements must not be empty")

        expected_gpu_ids = set(range(GPUS_PER_PERLMUTTER_NODE))
        by_node: dict[int, list[EndpointPlacement]] = {}
        for placement in self.placements:
            if placement.node_index >= self.node_count:
                raise ValueError("placement references a node outside the rung")
            if len(placement.gpu_indices) != self.tensor_parallel_size:
                raise ValueError("endpoint GPU count must equal TP size")
            if not set(placement.gpu_indices) <= expected_gpu_ids:
                raise ValueError("GPU index exceeds the Perlmutter node partition")
            by_node.setdefault(placement.node_index, []).append(placement)

        pair_roles = {
            (placement.pair_index, placement.role)
            for placement in self.placements
        }
        pair_indices = {placement.pair_index for placement in self.placements}
        if pair_roles != {
            (pair, role)
            for pair in pair_indices
            for role in ("prefill", "decode")
        }:
            raise ValueError("each pair must have exactly one P and one D")
        for node_index, placements in by_node.items():
            used: set[int] = set()
            for placement in placements:
                if used.intersection(placement.gpu_indices):
                    raise ValueError("endpoint GPU partitions overlap on a node")
                used.update(placement.gpu_indices)

    @property
    def pair_count(self) -> int:
        return len({placement.pair_index for placement in self.placements})

    def placement(self, pair_index: int, role: Role) -> EndpointPlacement:
        matches = [
            item for item in self.placements
            if item.pair_index == pair_index and item.role == role
        ]
        if len(matches) != 1:
            raise KeyError((pair_index, role))
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_count": self.node_count,
            "gpu_count": self.gpu_count,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pair_count": self.pair_count,
            "placements": [
                {
                    "pair_index": item.pair_index,
                    "role": item.role,
                    "node_index": item.node_index,
                    "gpu_indices": list(item.gpu_indices),
                }
                for item in self.placements
            ],
        }


def scale_topology(node_count: int) -> ScaleTopology:
    """Return the frozen initial topology for a 1/2/4-node experiment."""

    if node_count == 1:
        # One node is a local P/D control rung.  The two endpoints use
        # disjoint GPU partitions; this is not presented as a Slingshot rung.
        return ScaleTopology(
            node_count=1,
            gpu_count=4,
            tensor_parallel_size=2,
            placements=(
                EndpointPlacement(0, "prefill", 0, (0, 1)),
                EndpointPlacement(0, "decode", 0, (2, 3)),
            ),
        )
    if node_count == 2:
        return ScaleTopology(
            node_count=2,
            gpu_count=8,
            tensor_parallel_size=4,
            placements=(
                EndpointPlacement(0, "prefill", 0, (0, 1, 2, 3)),
                EndpointPlacement(0, "decode", 1, (0, 1, 2, 3)),
            ),
        )
    if node_count == 4:
        return ScaleTopology(
            node_count=4,
            gpu_count=16,
            tensor_parallel_size=4,
            placements=(
                EndpointPlacement(0, "prefill", 0, (0, 1, 2, 3)),
                EndpointPlacement(0, "decode", 1, (0, 1, 2, 3)),
                EndpointPlacement(1, "prefill", 2, (0, 1, 2, 3)),
                EndpointPlacement(1, "decode", 3, (0, 1, 2, 3)),
            ),
        )
    raise ValueError("node_count must be one of 1, 2, or 4")
