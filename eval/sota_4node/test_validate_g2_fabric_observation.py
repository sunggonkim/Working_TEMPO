import json
import unittest
from pathlib import Path

try:
    from .validate_g2_fabric_observation import validate_observation
except ImportError:
    from validate_g2_fabric_observation import validate_observation


class G2FabricObservationValidatorTests(unittest.TestCase):
    def test_current_raw_artifact_is_explicitly_noncausal(self):
        path = Path("results/sota_4node/staged_g2_job_56685044/fabric_observation_tempo_v4.json")
        if not path.is_file():
            self.skipTest("historical staged artifact is not present")
        result = validate_observation(json.loads(path.read_text()))
        self.assertFalse(result["promotion_eligible"])
        self.assertFalse(result["causal_claim_allowed"])
        self.assertGreater(result["groups"], 0)

    def test_route_witness_cannot_be_promoted(self):
        path = Path("results/sota_4node/staged_g2_job_56685044/fabric_observation_tempo_v4.json")
        if not path.is_file():
            self.skipTest("historical staged artifact is not present")
        raw = json.loads(path.read_text())
        raw["route_witness"]["bound_to_rank_bytes"] = True
        with self.assertRaises(ValueError):
            validate_observation(raw)


if __name__ == "__main__":
    unittest.main()
