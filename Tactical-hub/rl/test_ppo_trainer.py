import os
import tempfile
import unittest

try:
    import torch
    from rl.policy_model import TacticalPolicyValueNetwork, padded_rows
    from rl.ppo_trainer import PpoTrainer
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed; install requirements-rl.txt")
class PpoTrainerTest(unittest.TestCase):
    def feature_spec(self):
        table_names = (
            "siegeStates", "kingCampaignStates", "rewardPlacementRequests",
            "strategistCooldowns", "teleportCooldowns", "productionIntents",
            "movementIntents", "attackIntents", "strategistActionIntents", "teleportIntents",
        )
        return {
            "schemaVersion": 2, "observationSchemaVersion": 2, "actionSchemaVersion": 2,
            "globalWidth": 2, "teamWidth": 2, "unitWidth": 2, "mapTileWidth": 2,
            "baseWidth": 2, "constructionWidth": 2, "strategicGlobalWidth": 2,
            "strategicTableRowWidths": {name: 2 for name in table_names},
            "actionFeatureWidth": 2,
        }

    def observation(self):
        strategic = {"global": [0, 1]}
        strategic.update({name: [] for name in self.feature_spec()["strategicTableRowWidths"]})
        return {
            "schemaVersion": 2, "global": [1, 0], "teams": [[1, 0]], "teamMask": [1],
            "units": [[1, 0]], "unitMask": [1], "map": [[[1, 0]]],
            "bases": [[1, 0]], "baseMask": [1], "constructions": [[0, 0]],
            "constructionMask": [0], "strategicState": strategic,
        }

    def hyperparameters(self):
        return {
            "learningRate": 0.001, "gamma": 0.99, "gaeLambda": 0.95,
            "clipEpsilon": 0.2, "valueCoefficient": 0.5,
            "entropyCoefficient": 0.01, "maxGradientNorm": 0.5,
        }

    def samples(self, trainer):
        actions = [[1, 0], [0, 1]]
        samples = []
        for selected, advantage, expected_return in ((0, 1.0, 1.0), (1, -0.25, -1.0)):
            with torch.no_grad():
                logits, _, _, _ = trainer.model(self.observation(), actions)
                old_log_probability = torch.log_softmax(logits, dim=0)[selected].item()
            samples.append({
                "observation": self.observation(), "actions": actions,
                "selectedActionIndex": selected, "oldLogProbability": old_log_probability,
                "advantage": advantage, "return": expected_return,
            })
        return samples

    def make_initial_checkpoint(self, directory):
        spec = self.feature_spec()
        torch.manual_seed(3)
        model = TacticalPolicyValueNetwork(spec)
        path = os.path.join(directory, "bc-v2-init.pt")
        torch.save({"schemaVersion": 2, "featureSpec": spec, "modelStateDict": model.state_dict()}, path)
        return path

    def test_update_is_finite_changes_policy_and_value_and_resume_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = self.make_initial_checkpoint(directory)
            trainer = PpoTrainer(self.feature_spec(), self.hyperparameters(), 17, "cpu")
            trainer.load_initial_model(initial)
            result = trainer.update(self.samples(trainer))
            trainer.episode_count = 1
            self.assertEqual(result["sampleCount"], 2)
            self.assertTrue(all(torch.isfinite(torch.tensor(result[key])) for key in ("loss", "policyLoss", "valueLoss", "entropy", "gradientNorm")))
            self.assertTrue(result["policyParametersChanged"])
            self.assertTrue(result["valueParametersChanged"])

            checkpoint = os.path.join(directory, "ppo.pt")
            trainer.save(checkpoint, {"test": True})
            self.assertFalse(any(name.startswith(".ppo-checkpoint-") for name in os.listdir(directory)))
            model_hash = trainer.parameter_hash()
            optimizer_hash = trainer.optimizer_hash()

            resumed = PpoTrainer(self.feature_spec(), self.hyperparameters(), 17, "cpu")
            state = resumed.resume(checkpoint)
            self.assertEqual(state, {"updateCount": 1, "episodeCount": 1})
            self.assertEqual(resumed.parameter_hash(), model_hash)
            self.assertEqual(resumed.optimizer_hash(), optimizer_hash)
            continued = resumed.update(self.samples(resumed))
            self.assertEqual(continued["updateCount"], 2)
            self.assertTrue(torch.isfinite(torch.tensor(continued["loss"])))

    def test_rejects_schema_and_feature_spec_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = self.make_initial_checkpoint(directory)
            trainer = PpoTrainer(self.feature_spec(), self.hyperparameters(), 17, "cpu")
            trainer.load_initial_model(initial)
            trainer.save(os.path.join(directory, "ppo.pt"))
            different = self.feature_spec()
            different["actionFeatureWidth"] = 3
            with self.assertRaisesRegex(ValueError, "Feature Spec mismatch"):
                PpoTrainer(different, self.hyperparameters(), 17, "cpu").resume(os.path.join(directory, "ppo.pt"))
            with self.assertRaisesRegex(ValueError, "schemaVersion 2"):
                PpoTrainer({**self.feature_spec(), "schemaVersion": 1}, self.hyperparameters(), 17, "cpu")

    def test_chunked_gradient_accumulation_matches_single_full_batch_update(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = self.make_initial_checkpoint(directory)
            full = PpoTrainer(self.feature_spec(), self.hyperparameters(), 29, "cpu")
            chunked = PpoTrainer(self.feature_spec(), self.hyperparameters(), 29, "cpu")
            full.load_initial_model(initial)
            chunked.load_initial_model(initial)
            samples = self.samples(full)
            full_result = full.update(samples)

            chunked.begin_accumulated_update(len(samples))
            for sample in samples:
                prepared = chunked.model.prepare_observation_batch([sample["observation"]])
                actions, action_mask = padded_rows(
                    [[*sample["actions"]]], chunked.feature_spec["actionFeatureWidth"], chunked.device
                )
                chunked.accumulate_prepared_chunk(
                    prepared, actions, action_mask,
                    torch.tensor([sample["selectedActionIndex"]], dtype=torch.long),
                    torch.tensor([sample["oldLogProbability"]], dtype=torch.float32),
                    torch.tensor([sample["advantage"]], dtype=torch.float32),
                    torch.tensor([sample["return"]], dtype=torch.float32),
                )
            chunked_result = chunked.finish_accumulated_update()
            for name, parameter in full.model.state_dict().items():
                self.assertTrue(torch.allclose(parameter, chunked.model.state_dict()[name], atol=1e-7, rtol=1e-6), name)
            for key in ("loss", "policyLoss", "valueLoss", "entropy"):
                self.assertAlmostEqual(full_result[key], chunked_result[key], places=6)
            self.assertEqual(chunked_result["sampleCount"], len(samples))
            self.assertEqual(chunked_result["updateCount"], 1)


if __name__ == "__main__":
    unittest.main()
