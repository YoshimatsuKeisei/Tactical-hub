import tempfile
import unittest
from pathlib import Path

import torch

from rl.bc_trainer import BehavioralCloningTrainer
from rl.test_policy_model import PolicyModelTest


class BehavioralCloningTrainerTest(unittest.TestCase):
    def test_backward_changes_weights_and_checkpoint_reloads(self):
        helper = PolicyModelTest()
        spec = helper.feature_spec()
        observation = helper.observation()
        samples = [{
            "observation": observation,
            "actions": [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
            "targetIndex": 1,
        }]
        trainer = BehavioralCloningTrainer(spec, learning_rate=1e-3, seed=17)
        before = trainer.parameter_hash()
        metrics = trainer.process_batch(samples, train=True)
        after = trainer.parameter_hash()
        self.assertNotEqual(before, after)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["lossSum"])))
        self.assertEqual(metrics["count"], 1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = str(Path(directory) / "model.pt")
            trainer.save(checkpoint, {"test": True})
            restored = BehavioralCloningTrainer(spec, learning_rate=1e-3, seed=99)
            restored.load(checkpoint)
            self.assertEqual(restored.parameter_hash(), after)


if __name__ == "__main__":
    unittest.main()
