import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from rl.browser_bc_server import BrowserBcModel
    from rl.policy_model import TacticalPolicyValueNetwork
    from rl.test_policy_model import PolicyModelTest
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed; install requirements-rl.txt")
class BrowserBcServerTest(unittest.TestCase):
    def test_loads_best_checkpoint_and_returns_one_supplied_action_key(self):
        helper = PolicyModelTest()
        feature_spec = helper.feature_spec()
        model = TacticalPolicyValueNetwork(feature_spec)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bc-best.pt"
            torch.save({
                "schemaVersion": 1,
                "featureSpec": feature_spec,
                "modelStateDict": model.state_dict(),
                "metadata": {"source": "fixture"},
            }, checkpoint)
            loaded = BrowserBcModel(str(checkpoint))
            selected = loaded.infer({
                "featureSpec": feature_spec,
                "observation": helper.observation(),
                "actions": [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
                "actionKeys": ["first", "second"],
            })
            self.assertIn(selected, ("first", "second"))

    def test_rejects_feature_spec_mismatch(self):
        helper = PolicyModelTest()
        feature_spec = helper.feature_spec()
        model = TacticalPolicyValueNetwork(feature_spec)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "bc-best.pt"
            torch.save({"schemaVersion": 1, "featureSpec": feature_spec, "modelStateDict": model.state_dict()}, checkpoint)
            loaded = BrowserBcModel(str(checkpoint))
            with self.assertRaisesRegex(ValueError, "featureSpec"):
                loaded.infer({"featureSpec": {**feature_spec, "globalWidth": 99}, "observation": helper.observation(), "actions": [[1] * 6], "actionKeys": ["only"]})


if __name__ == "__main__":
    unittest.main()
