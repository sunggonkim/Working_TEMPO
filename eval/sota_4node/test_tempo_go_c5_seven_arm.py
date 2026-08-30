from eval.sota_4node.tempo_pd_elastic_router import ElasticPDRouterCore
from eval.sota_4node.tempo_pd_elastic_router_v444 import ElasticExperimentArm


def test_cross_layer_ablation_arms_are_explicit_wire_values():
    assert ElasticExperimentArm.NETWORK_REQUEST_ONLY.value == (
        "network_request_only")
    assert ElasticExperimentArm.APP_GLOBAL_ONLY.value == "app_global_only"


def test_network_request_only_uses_only_network_signal_names():
    penalty, evidence = ElasticPDRouterCore._network_request_penalty({
        "sequence": 7,
        "communicator_id": "c5",
        "topology_fingerprint_sha256": "a" * 64,
        "signals": [
            {"name": "nccl_collective_p99_ms", "value": 8.0,
             "support": "supported"},
            {"name": "lmcache_transfer_p99_ms", "value": 12.0,
             "support": "supported"},
            {"name": "vllm_num_requests_waiting", "value": 999,
             "support": "supported"},
        ],
    })
    assert penalty == 8.0 * 0.25 + 12.0 * 0.50
    assert set(evidence["observed_signals"]) == {
        "nccl_collective_p99_ms", "lmcache_transfer_p99_ms",
    }
