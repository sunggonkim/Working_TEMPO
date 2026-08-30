from __future__ import annotations

import pytest

from eval.sota_4node.tempo_go_scale_topology import scale_topology


@pytest.mark.parametrize(
    ("node_count", "gpu_count", "tp", "pair_count"),
    ((1, 4, 2, 1), (2, 8, 4, 1), (4, 16, 4, 2)),
)
def test_scale_rungs_are_explicit_and_gpu_disjoint(
    node_count: int, gpu_count: int, tp: int, pair_count: int,
) -> None:
    topology = scale_topology(node_count)
    assert topology.gpu_count == gpu_count
    assert topology.tensor_parallel_size == tp
    assert topology.pair_count == pair_count
    assert len(topology.placements) == pair_count * 2
    for pair in range(pair_count):
        prefill = topology.placement(pair, "prefill")
        decode = topology.placement(pair, "decode")
        assert len(prefill.gpu_indices) == tp
        assert len(decode.gpu_indices) == tp
        assert set(prefill.gpu_indices).isdisjoint(decode.gpu_indices) \
            or prefill.node_index != decode.node_index


def test_one_node_is_not_advertised_as_fabric_contention() -> None:
    topology = scale_topology(1)
    assert topology.placement(0, "prefill").node_index == 0
    assert topology.placement(0, "decode").node_index == 0
    assert topology.to_dict()["pair_count"] == 1


def test_invalid_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="one of 1, 2, or 4"):
        scale_topology(3)
