import tempfile
import unittest
import copy
from pathlib import Path

import torch
import numpy as np

from rl.bc_packed import decode_packed_views, prepare_packed_tensors
from rl.bc_trainer import BehavioralCloningTrainer
from rl.policy_model import padded_rows
from rl.test_policy_model import PolicyModelTest


class BehavioralCloningTrainerTest(unittest.TestCase):
    def test_packed_views_match_json_tensor_path_logits_targets_loss_and_correct(self):
        helper = PolicyModelTest()
        spec = helper.feature_spec()
        samples = [{
            "observation": helper.observation(),
            "actions": [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
            "targetIndex": 1,
        }]
        trainer = BehavioralCloningTrainer(spec, learning_rate=1e-3, seed=53)
        old_observation = trainer.model.prepare_observation_batch([samples[0]["observation"]])
        old_actions, old_action_mask = padded_rows([samples[0]["actions"]], spec["actionFeatureWidth"], trainer.device)
        old_targets = torch.tensor([1], dtype=torch.long)

        arrays = {
            "global": old_observation["global"],
            "strategicGlobal": old_observation["strategicGlobal"],
            "map": old_observation["map"][0],
            "mapMask": old_observation["map"][1],
            "actions": old_actions,
            "actionMask": old_action_mask,
            "targets": old_targets.to(torch.int32),
        }
        for name, (values, mask) in old_observation["masked"].items():
            arrays[name] = values
            arrays[{"teams": "teamMask", "units": "unitMask", "bases": "baseMask", "constructions": "constructionMask"}[name]] = mask
        for name, (values, mask) in old_observation["strategic"].items():
            arrays[f"strategic.{name}"] = values
            arrays[f"strategicMask.{name}"] = mask

        payload = bytearray()
        descriptors = []
        for name, tensor in arrays.items():
            array = tensor.detach().cpu().contiguous().numpy()
            if array.dtype == np.bool_:
                array = array.astype(np.uint8)
            elif array.dtype == np.int32:
                pass
            else:
                array = array.astype(np.float32)
            raw = array.tobytes()
            descriptors.append({
                "name": name,
                "dtype": "uint8" if array.dtype == np.uint8 else "int32" if array.dtype == np.int32 else "float32",
                "shape": list(array.shape),
                "byteOffset": len(payload),
                "byteLength": len(raw),
            })
            payload.extend(raw)
        views = decode_packed_views({"tensors": descriptors}, payload)
        new_observation, new_actions, new_action_mask, new_targets = prepare_packed_tensors(views, trainer.device)
        trainer.model.eval()
        with torch.no_grad():
            old_logits = trainer.model.forward_prepared_batch(old_observation, old_actions, old_action_mask)[0]
            new_logits = trainer.model.forward_prepared_batch(new_observation, new_actions, new_action_mask)[0]
        self.assertEqual(tuple(new_logits.shape), (1, 2))
        self.assertTrue(torch.equal(new_targets, old_targets))
        self.assertTrue(torch.allclose(new_logits, old_logits, atol=1e-7, rtol=1e-7))
        old_loss = torch.nn.functional.cross_entropy(old_logits, old_targets)
        new_loss = torch.nn.functional.cross_entropy(new_logits, new_targets)
        self.assertTrue(torch.allclose(new_loss, old_loss, atol=1e-7, rtol=1e-7))
        self.assertEqual(
            int((torch.argmax(new_logits, dim=1) == new_targets).sum()),
            int((torch.argmax(old_logits, dim=1) == old_targets).sum()),
        )

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

            if torch.cuda.is_available():
                gpu = BehavioralCloningTrainer(spec, learning_rate=1e-3, seed=99, device="cuda")
                gpu.load(checkpoint)
                self.assertEqual(gpu.parameter_hash(), after)
                gpu_checkpoint = str(Path(directory) / "model-gpu.pt")
                gpu.save(gpu_checkpoint, {"test": True})
                cpu = BehavioralCloningTrainer(spec, learning_rate=1e-3, seed=99, device="cpu")
                cpu.load(gpu_checkpoint)
                self.assertEqual(cpu.parameter_hash(), after)

    def test_complete_training_checkpoint_restores_optimizer_and_matches_continuous_training(self):
        helper = PolicyModelTest()
        spec = helper.feature_spec()
        samples = [{
            "observation": helper.observation(),
            "actions": [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
            "targetIndex": 1,
        }]
        continuous = BehavioralCloningTrainer(spec, learning_rate=1e-3, seed=71)
        continuous.process_batch(samples, train=True)
        continuous.process_batch(samples, train=True)

        interrupted = BehavioralCloningTrainer(spec, learning_rate=1e-3, seed=71)
        interrupted.process_batch(samples, train=True)
        checkpoint_state = {
            "schemaVersion": 3,
            "checkpointKind": "behavioral_cloning_training",
            "currentEpoch": 1,
            "completedEpoch": 0,
            "phase": "train",
            "nextEpisodeNumber": 3,
            "completedTrainEpisodes": [1, 2],
            "completedValidationEpisodes": [],
            "trainAccumulator": {"lossSum": 1.0, "correct": 1, "count": 2},
            "validationAccumulator": {"lossSum": 0.0, "correct": 0, "count": 0},
            "bestEpoch": None,
            "bestValidationAccuracy": None,
            "seed": 71,
            "learningRate": 1e-3,
            "batchSize": 4,
            "trainRange": {"from": 1, "to": 4},
            "validationRange": {"from": 5, "to": 5},
            "testRange": {"from": 6, "to": 6},
            "metadata": {"test": True},
        }
        expected = {key: checkpoint_state[key] for key in ("seed", "learningRate", "batchSize", "trainRange", "validationRange", "testRange")}
        with tempfile.TemporaryDirectory() as directory:
            latest = str(Path(directory) / "latest.pt")
            interrupted.save_training_checkpoint(latest, checkpoint_state)
            resumed = BehavioralCloningTrainer(spec, learning_rate=1e-3, seed=999)
            state = resumed.resume_training_checkpoint(latest, expected)
            self.assertEqual(state["nextEpisodeNumber"], 3)
            self.assertEqual(state["completedTrainEpisodes"], [1, 2])
            resumed.process_batch(samples, train=True)
            self.assertEqual(resumed.parameter_hash(), continuous.parameter_hash())
            for left, right in zip(resumed.optimizer.state_dict()["state"].values(), continuous.optimizer.state_dict()["state"].values()):
                for key in left:
                    if torch.is_tensor(left[key]):
                        self.assertTrue(torch.equal(left[key], right[key]))
                    else:
                        self.assertEqual(left[key], right[key])
            self.assertFalse(any(path.name.startswith(".bc-checkpoint-") for path in Path(directory).iterdir()))

    def test_resume_rejects_model_only_and_feature_mismatched_checkpoints(self):
        helper = PolicyModelTest()
        spec = helper.feature_spec()
        trainer = BehavioralCloningTrainer(spec, learning_rate=1e-3, seed=73)
        with tempfile.TemporaryDirectory() as directory:
            old_checkpoint = str(Path(directory) / "old.pt")
            trainer.save(old_checkpoint, {"epoch": 1})
            with self.assertRaisesRegex(ValueError, "does not contain optimizer state"):
                trainer.resume_training_checkpoint(old_checkpoint, {"seed": 73, "learningRate": 1e-3})

            latest = str(Path(directory) / "latest.pt")
            state = {
                "schemaVersion": 3, "checkpointKind": "behavioral_cloning_training", "currentEpoch": 1, "completedEpoch": 0,
                "phase": "validation", "nextEpisodeNumber": 5, "completedTrainEpisodes": [1, 2, 3, 4],
                "completedValidationEpisodes": [],
                "trainAccumulator": {"lossSum": 1.0, "correct": 1, "count": 2},
                "validationAccumulator": {"lossSum": 0.0, "correct": 0, "count": 0},
                "bestEpoch": None, "bestValidationAccuracy": None, "seed": 73, "learningRate": 1e-3, "batchSize": 4,
                "trainRange": {"from": 1, "to": 4}, "validationRange": {"from": 5, "to": 5}, "testRange": {"from": 6, "to": 6},
                "metadata": {},
            }
            trainer.save_training_checkpoint(latest, state)
            incompatible_spec = copy.deepcopy(spec)
            incompatible_spec["globalWidth"] += 1
            incompatible = BehavioralCloningTrainer(incompatible_spec, learning_rate=1e-3, seed=73)
            with self.assertRaisesRegex(ValueError, "featureSpec does not match"):
                incompatible.resume_training_checkpoint(latest, {key: state[key] for key in ("seed", "learningRate", "batchSize", "trainRange", "validationRange", "testRange")})

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

    def test_profile_batch_reports_finite_stage_timings(self):
        helper = PolicyModelTest()
        trainer = BehavioralCloningTrainer(helper.feature_spec(), learning_rate=1e-3, seed=41, device="cpu")
        result = trainer.process_profile_batch([{
            "observation": helper.observation(),
            "actions": [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
            "targetIndex": 1,
        }])
        self.assertEqual(result["count"], 1)
        self.assertEqual(set(result["timings"]), {
            "tensorPreparationMs", "forwardMs", "lossMs", "backwardMs", "optimizerStepMs",
        })
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) and value >= 0 for value in result["timings"].values()))


if __name__ == "__main__":
    unittest.main()
