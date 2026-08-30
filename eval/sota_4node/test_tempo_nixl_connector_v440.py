import inspect
import unittest

from eval.sota_4node.tempo_nixl_connector_v440 import (
    TempoNixlConnector,
    TempoNixlPullScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)


class TempoNixlConnectorContractTest(unittest.TestCase):
    def test_external_connector_has_current_three_argument_signature(self):
        parameters = tuple(inspect.signature(TempoNixlConnector).parameters)
        self.assertEqual(parameters, ("vllm_config", "role", "kv_cache_config"))

    def test_upstream_worker_and_wire_contract_are_inherited(self):
        self.assertTrue(issubclass(TempoNixlConnector, NixlPullConnector))
        self.assertTrue(issubclass(TempoNixlPullScheduler, NixlPullConnectorScheduler))
        self.assertIs(
            TempoNixlPullScheduler.update_state_after_alloc,
            NixlPullConnectorScheduler.update_state_after_alloc,
        )


if __name__ == "__main__":
    unittest.main()
