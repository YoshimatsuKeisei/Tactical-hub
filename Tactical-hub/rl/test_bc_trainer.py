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

    def test_vectorized_batch_matches_single_sample_loss_and_ignores_padding(self):
        helper = PolicyModelTest()
        spec = helper.feature_spec()
        first_observation = helper.observation()
        second_observation = helper.observation()
        second_observation["strategicState"]["siegeStates"] = [[1, 2, 3]]
        samples = [
            {
                "observation": first_observation,
                "actions": [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
                "targetIndex": 1,
            },
            {
                "observation": second_observation,
                "actions": [[0, 0, 1, 0, 0, 0]],
                "targetIndex": 0,
            },
        ]
        trainer = BehavioralCloningTrainer(spec, learning_rate=1e-3, seed=23)
        trainer.model.eval()
        with torch.no_grad():
            expected_losses = []
            for sample in samples:
                logits, _, _, _ = trainer.model(sample["observation"], sample["actions"])
                expected_losses.append(torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([sample["targetIndex"]], dtype=torch.long),
                ))
        metrics = trainer.process_batch(samples, train=False)
        self.assertAlmostEqual(
            metrics["lossSum"],
            float(torch.stack(expected_losses).sum().item()),
            places=5,
        )
        self.assertEqual(metrics["count"], 2)


if __name__ == "__main__":
    unittest.main()
