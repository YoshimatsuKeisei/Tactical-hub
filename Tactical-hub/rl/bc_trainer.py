from __future__ import annotations

from typing import Any
import hashlib
import math
import os
import tempfile
import time

import torch
from torch.nn import functional as F

from rl.policy_model import TacticalPolicyValueNetwork, padded_rows


class BehavioralCloningTrainer:
    def __init__(self, feature_spec: dict[str, Any], learning_rate: float, seed: int, device: torch.device | str = "cpu"):
        torch.manual_seed(seed)
        self.feature_spec = feature_spec
        self.device = torch.device(device)
        self.model = TacticalPolicyValueNetwork(feature_spec).to(self.device)
        self.optimizer = torch.optim.Adam(
            [parameter for name, parameter in self.model.named_parameters() if not name.startswith("value_head.")],
            lr=learning_rate,
        )

    def process_batch(self, samples: list[dict[str, Any]], train: bool) -> dict[str, float | int]:
        if not samples:
            raise ValueError("Behavioral cloning batch cannot be empty")
        self.model.train(train)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for sample in samples:
                actions = sample["actions"]
                target = int(sample["targetIndex"])
                if target < 0 or target >= len(actions):
                    raise ValueError(f"targetIndex {target} is outside {len(actions)} legal actions")
            targets = torch.tensor([int(sample["targetIndex"]) for sample in samples], dtype=torch.long, device=self.device)
            logits, _, _, _, action_mask = self.model.forward_batch(
                [sample["observation"] for sample in samples],
                [sample["actions"] for sample in samples],
            )
            if not torch.isfinite(logits[action_mask]).all() or torch.isnan(logits).any():
                raise FloatingPointError("Policy logits contain NaN or Inf")
            mean_loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(mean_loss):
                raise FloatingPointError("Behavioral cloning loss contains NaN or Inf")
            correct = int((torch.argmax(logits, dim=1) == targets).sum().item())
            if train:
                mean_loss.backward()
                for parameter in self.model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Policy gradient contains NaN or Inf")
                self.optimizer.step()
        return {
            "lossSum": float(mean_loss.detach().item()) * len(samples),
            "correct": correct,
            "count": len(samples),
        }

    def process_packed_batch(
        self,
        prepared_observations: dict[str, Any],
        prepared_actions: torch.Tensor,
        action_mask: torch.Tensor,
        targets: torch.Tensor,
        train: bool,
    ) -> dict[str, float | int]:
        batch_size = int(targets.shape[0])
        if batch_size == 0:
            raise ValueError("Behavioral cloning batch cannot be empty")
        if torch.any(targets < 0) or torch.any(targets >= action_mask.shape[1]):
            raise ValueError("targetIndex is outside legal action tensor")
        if not torch.all(action_mask.gather(1, targets.unsqueeze(1))):
            raise ValueError("targetIndex points to a padded action")
        self.model.train(train)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            logits, _, _, _, returned_mask = self.model.forward_prepared_batch(
                prepared_observations, prepared_actions, action_mask
            )
            if not torch.isfinite(logits[returned_mask]).all() or torch.isnan(logits).any():
                raise FloatingPointError("Policy logits contain NaN or Inf")
            mean_loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(mean_loss):
                raise FloatingPointError("Behavioral cloning loss contains NaN or Inf")
            correct = int((torch.argmax(logits, dim=1) == targets).sum().item())
            if train:
                mean_loss.backward()
                for parameter in self.model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Policy gradient contains NaN or Inf")
                self.optimizer.step()
        return {"lossSum": float(mean_loss.detach().item()) * batch_size, "correct": correct, "count": batch_size}

    def process_profile_packed_batch(
        self,
        prepared_observations: dict[str, Any],
        prepared_actions: torch.Tensor,
        action_mask: torch.Tensor,
        targets: torch.Tensor,
        tensor_preparation_ms: float,
    ) -> dict[str, Any]:
        if int(targets.shape[0]) == 0:
            raise ValueError("Behavioral cloning profile batch cannot be empty")
        self.model.train(True)
        timings = {
            "tensorPreparationMs": tensor_preparation_ms,
            "forwardMs": 0.0,
            "lossMs": 0.0,
            "backwardMs": 0.0,
            "optimizerStepMs": 0.0,
        }

        def synchronize() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        if torch.any(targets < 0) or torch.any(targets >= action_mask.shape[1]):
            raise ValueError("targetIndex is outside legal action tensor")
        if not torch.all(action_mask.gather(1, targets.unsqueeze(1))):
            raise ValueError("targetIndex points to a padded action")
        self.optimizer.zero_grad(set_to_none=True)
        synchronize()
        started = time.perf_counter()
        logits, _, _, _, returned_mask = self.model.forward_prepared_batch(
            prepared_observations, prepared_actions, action_mask
        )
        synchronize()
        timings["forwardMs"] = (time.perf_counter() - started) * 1000
        if not torch.isfinite(logits[returned_mask]).all() or torch.isnan(logits).any():
            raise FloatingPointError("Policy logits contain NaN or Inf")
        started = time.perf_counter()
        loss = F.cross_entropy(logits, targets)
        synchronize()
        timings["lossMs"] = (time.perf_counter() - started) * 1000
        if not torch.isfinite(loss):
            raise FloatingPointError("Behavioral cloning loss contains NaN or Inf")
        started = time.perf_counter()
        loss.backward()
        synchronize()
        timings["backwardMs"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        self.optimizer.step()
        synchronize()
        timings["optimizerStepMs"] = (time.perf_counter() - started) * 1000
        batch_size = int(targets.shape[0])
        return {
            "timings": timings,
            "lossSum": float(loss.detach().item()) * batch_size,
            "correct": int((torch.argmax(logits, dim=1) == targets).sum().item()),
            "count": batch_size,
        }

    def process_profile_batch(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Behavioral cloning profile batch cannot be empty")
        self.model.train(True)
        timings = {
            "tensorPreparationMs": 0.0,
            "forwardMs": 0.0,
            "lossMs": 0.0,
            "backwardMs": 0.0,
            "optimizerStepMs": 0.0,
        }

        def synchronize() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        for sample in samples:
            target = int(sample["targetIndex"])
            if target < 0 or target >= len(sample["actions"]):
                raise ValueError(f"targetIndex {target} is outside {len(sample['actions'])} legal actions")
        self.optimizer.zero_grad(set_to_none=True)
        synchronize()
        started = time.perf_counter()
        prepared_observations = self.model.prepare_observation_batch([sample["observation"] for sample in samples])
        prepared_actions, action_mask = padded_rows(
            [sample["actions"] for sample in samples],
            self.feature_spec["actionFeatureWidth"],
            self.device,
        )
        targets = torch.tensor([int(sample["targetIndex"]) for sample in samples], dtype=torch.long, device=self.device)
        synchronize()
        timings["tensorPreparationMs"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        logits, _, _, _, returned_mask = self.model.forward_prepared_batch(prepared_observations, prepared_actions, action_mask)
        synchronize()
        timings["forwardMs"] = (time.perf_counter() - started) * 1000
        if not torch.isfinite(logits[returned_mask]).all() or torch.isnan(logits).any():
            raise FloatingPointError("Policy logits contain NaN or Inf")

        started = time.perf_counter()
        loss = F.cross_entropy(logits, targets)
        synchronize()
        timings["lossMs"] = (time.perf_counter() - started) * 1000
        if not torch.isfinite(loss):
            raise FloatingPointError("Behavioral cloning loss contains NaN or Inf")

        started = time.perf_counter()
        loss.backward()
        synchronize()
        timings["backwardMs"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        self.optimizer.step()
        synchronize()
        timings["optimizerStepMs"] = (time.perf_counter() - started) * 1000
        return {
            "timings": timings,
            "lossSum": float(loss.detach().item()) * len(samples),
            "correct": int((torch.argmax(logits, dim=1) == targets).sum().item()),
            "count": len(samples),
        }

    @staticmethod
    def _atomic_torch_save(checkpoint: dict[str, Any], path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(prefix=".bc-checkpoint-", suffix=".tmp", dir=directory)
        os.close(file_descriptor)
        try:
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def save(self, path: str, metadata: dict[str, Any]) -> None:
        self._atomic_torch_save({
            "schemaVersion": 1,
            "featureSpec": self.feature_spec,
            "modelStateDict": self.model.state_dict(),
            "metadata": metadata,
        }, path)

    def save_training_checkpoint(
        self,
        path: str,
        completed_epoch: int,
        best_epoch: int,
        best_validation_accuracy: float,
        seed: int,
        learning_rate: float,
        metadata: dict[str, Any],
    ) -> None:
        if not isinstance(completed_epoch, int) or isinstance(completed_epoch, bool) or completed_epoch < 0:
            raise ValueError("completedEpoch must be a non-negative integer")
        if not isinstance(best_epoch, int) or isinstance(best_epoch, bool) or best_epoch < 0 or best_epoch > completed_epoch:
            raise ValueError("bestEpoch must be an integer between 0 and completedEpoch")
        if not math.isfinite(best_validation_accuracy):
            raise ValueError("bestValidationAccuracy must be finite")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learningRate must be finite and positive")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        self._atomic_torch_save({
            "schemaVersion": 2,
            "checkpointKind": "behavioral_cloning_training",
            "featureSpec": self.feature_spec,
            "modelStateDict": self.model.state_dict(),
            "optimizerStateDict": self.optimizer.state_dict(),
            "completedEpoch": completed_epoch,
            "bestEpoch": best_epoch,
            "bestValidationAccuracy": best_validation_accuracy,
            "seed": seed,
            "learningRate": learning_rate,
            "metadata": metadata,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["featureSpec"] != self.feature_spec:
            raise ValueError("Checkpoint feature spec does not match")
        self.model.load_state_dict(checkpoint["modelStateDict"])

    def resume_training_checkpoint(self, path: str, expected_seed: int, expected_learning_rate: float) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Resume checkpoint is not an object")
        if checkpoint.get("schemaVersion") != 2 or checkpoint.get("checkpointKind") != "behavioral_cloning_training":
            if "optimizerStateDict" not in checkpoint:
                raise ValueError("Checkpoint does not contain optimizer state and cannot be used for complete resume")
            raise ValueError(f"Unsupported resume checkpoint schemaVersion: {checkpoint.get('schemaVersion')}")
        if checkpoint.get("featureSpec") != self.feature_spec:
            raise ValueError("Resume checkpoint featureSpec does not match current replay data")
        if "modelStateDict" not in checkpoint:
            raise ValueError("Resume checkpoint is missing modelStateDict")
        if "optimizerStateDict" not in checkpoint:
            raise ValueError("Resume checkpoint is missing optimizerStateDict")
        completed_epoch = checkpoint.get("completedEpoch")
        best_epoch = checkpoint.get("bestEpoch")
        best_accuracy = checkpoint.get("bestValidationAccuracy")
        seed = checkpoint.get("seed")
        learning_rate = checkpoint.get("learningRate")
        metadata = checkpoint.get("metadata")
        if not isinstance(completed_epoch, int) or isinstance(completed_epoch, bool) or completed_epoch < 0:
            raise ValueError("Resume checkpoint completedEpoch must be a non-negative integer")
        if not isinstance(best_epoch, int) or isinstance(best_epoch, bool) or best_epoch < 0 or best_epoch > completed_epoch:
            raise ValueError("Resume checkpoint bestEpoch is invalid")
        if not isinstance(best_accuracy, (int, float)) or isinstance(best_accuracy, bool) or not math.isfinite(best_accuracy):
            raise ValueError("Resume checkpoint bestValidationAccuracy must be finite")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("Resume checkpoint seed must be an integer")
        if seed != expected_seed:
            raise ValueError(f"Resume checkpoint seed mismatch: checkpoint={seed}, requested={expected_seed}")
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Resume checkpoint learningRate must be finite and positive")
        if not math.isclose(float(learning_rate), expected_learning_rate, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                f"Resume checkpoint learningRate mismatch: checkpoint={learning_rate}, requested={expected_learning_rate}"
            )
        if not isinstance(metadata, dict):
            raise ValueError("Resume checkpoint metadata must be an object")
        self.model.load_state_dict(checkpoint["modelStateDict"])
        self.optimizer.load_state_dict(checkpoint["optimizerStateDict"])
        return {
            "completedEpoch": completed_epoch,
            "bestEpoch": best_epoch,
            "bestValidationAccuracy": float(best_accuracy),
            "seed": seed,
            "learningRate": float(learning_rate),
            "metadata": metadata,
        }

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for parameter in self.model.parameters():
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
