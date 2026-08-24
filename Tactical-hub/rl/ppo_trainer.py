from __future__ import annotations

from typing import Any
import hashlib
import os
import tempfile

import torch
from torch.nn import functional as F

from rl.policy_model import TacticalPolicyValueNetwork


PPO_CHECKPOINT_SCHEMA_VERSION = 1


class PpoTrainer:
    def __init__(self, feature_spec: dict[str, Any], hyperparameters: dict[str, Any], seed: int, device: torch.device | str):
        if feature_spec.get("schemaVersion") != 2:
            raise ValueError("PPO requires Feature Spec schemaVersion 2")
        self.feature_spec = feature_spec
        self.hyperparameters = dict(hyperparameters)
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        self.model = TacticalPolicyValueNetwork(feature_spec).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(hyperparameters["learningRate"]))
        self.update_count = 0
        self.episode_count = 0
        self._accumulation: dict[str, Any] | None = None

    def load_initial_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("schemaVersion") != 2:
            raise ValueError("PPO initial checkpoint must use schemaVersion 2")
        if checkpoint.get("featureSpec") != self.feature_spec:
            raise ValueError("PPO initial checkpoint Feature Spec mismatch")
        self.model.load_state_dict(checkpoint.get("modelStateDict"), strict=True)

    def act(self, observation: dict[str, Any], actions: list[list[float]]) -> dict[str, float | int]:
        if not actions:
            raise ValueError("PPO cannot act without legal actions")
        self.model.eval()
        with torch.no_grad():
            logits, value, _, _ = self.model(observation, actions)
            distribution = torch.distributions.Categorical(logits=logits)
            selected = distribution.sample()
            log_probability = distribution.log_prob(selected)
        return {
            "actionIndex": int(selected.item()),
            "logProbability": float(log_probability.item()),
            "value": float(value.item()),
        }

    def update(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("PPO update requires at least one sample")
        self.model.train(True)
        actions = [sample["actions"] for sample in samples]
        selected = torch.tensor([int(sample["selectedActionIndex"]) for sample in samples], dtype=torch.long, device=self.device)
        old_log_probabilities = torch.tensor([float(sample["oldLogProbability"]) for sample in samples], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([float(sample["advantage"]) for sample in samples], dtype=torch.float32, device=self.device)
        returns = torch.tensor([float(sample["return"]) for sample in samples], dtype=torch.float32, device=self.device)
        if not all(torch.isfinite(value).all() for value in (old_log_probabilities, advantages, returns)):
            raise FloatingPointError("PPO input contains NaN or Inf")
        logits, values, _, _, action_mask = self.model.forward_batch([sample["observation"] for sample in samples], actions)
        if torch.any(selected < 0) or torch.any(selected >= action_mask.shape[1]) or not torch.all(action_mask.gather(1, selected.unsqueeze(1))):
            raise ValueError("PPO selected action is outside the legal action mask")
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(selected)
        entropy = distribution.entropy().mean()
        ratio = torch.exp(log_probabilities - old_log_probabilities)
        clip_epsilon = float(self.hyperparameters["clipEpsilon"])
        policy_loss = -torch.minimum(ratio * advantages, torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + float(self.hyperparameters["valueCoefficient"]) * value_loss - float(self.hyperparameters["entropyCoefficient"]) * entropy
        if not all(torch.isfinite(value).all() for value in (logits[action_mask], values, log_probabilities, entropy, policy_loss, value_loss, loss)):
            raise FloatingPointError("PPO calculation contains NaN or Inf")
        policy_before = self.parameter_hash(exclude_prefix="value_head.")
        value_before = self.parameter_hash(include_prefix="value_head.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("PPO gradient contains NaN or Inf")
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.hyperparameters["maxGradientNorm"]))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("PPO gradient norm is NaN or Inf")
        self.optimizer.step()
        self.update_count += 1
        policy_after = self.parameter_hash(exclude_prefix="value_head.")
        value_after = self.parameter_hash(include_prefix="value_head.")
        return {
            "sampleCount": len(samples), "loss": float(loss.detach().item()),
            "policyLoss": float(policy_loss.detach().item()), "valueLoss": float(value_loss.detach().item()),
            "entropy": float(entropy.detach().item()), "gradientNorm": float(gradient_norm.detach().item()),
            "policyParametersChanged": policy_before != policy_after,
            "valueParametersChanged": value_before != value_after,
            "updateCount": self.update_count,
        }

    def act_prepared(
        self,
        prepared_observations: dict[str, Any],
        prepared_actions: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> dict[str, float | int]:
        if prepared_actions.shape[0] != 1 or not bool(action_mask[0].any()):
            raise ValueError("Packed PPO act requires exactly one sample with legal actions")
        self.model.eval()
        with torch.no_grad():
            logits, values, _, _, returned_mask = self.model.forward_prepared_batch(
                prepared_observations, prepared_actions, action_mask
            )
            if not torch.isfinite(logits[returned_mask]).all() or not torch.isfinite(values).all():
                raise FloatingPointError("Packed PPO action calculation contains NaN or Inf")
            distribution = torch.distributions.Categorical(logits=logits[0])
            selected = distribution.sample()
            log_probability = distribution.log_prob(selected)
        return {
            "actionIndex": int(selected.item()),
            "logProbability": float(log_probability.item()),
            "value": float(values[0].item()),
        }

    def begin_accumulated_update(self, total_samples: int) -> dict[str, int]:
        if self._accumulation is not None:
            raise RuntimeError("A PPO accumulated update is already active")
        if not isinstance(total_samples, int) or isinstance(total_samples, bool) or total_samples <= 0:
            raise ValueError("PPO total sample count must be a positive integer")
        self.model.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        self._accumulation = {
            "totalSamples": total_samples,
            "processedSamples": 0,
            "policyLossSum": 0.0,
            "valueLossSum": 0.0,
            "entropySum": 0.0,
            "policyBefore": self.parameter_hash(exclude_prefix="value_head."),
            "valueBefore": self.parameter_hash(include_prefix="value_head."),
        }
        return {"totalSamples": total_samples}

    def accumulate_prepared_chunk(
        self,
        prepared_observations: dict[str, Any],
        prepared_actions: torch.Tensor,
        action_mask: torch.Tensor,
        selected: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, int]:
        state = self._accumulation
        if state is None:
            raise RuntimeError("No PPO accumulated update is active")
        chunk_size = int(selected.shape[0])
        if chunk_size <= 0 or state["processedSamples"] + chunk_size > state["totalSamples"]:
            raise ValueError("PPO accumulated chunk exceeds declared sample count")
        if not all(torch.isfinite(value).all() for value in (old_log_probabilities, advantages, returns)):
            raise FloatingPointError("PPO packed input contains NaN or Inf")
        logits, values, _, _, returned_mask = self.model.forward_prepared_batch(
            prepared_observations, prepared_actions, action_mask
        )
        if torch.any(selected < 0) or torch.any(selected >= returned_mask.shape[1]) or not torch.all(returned_mask.gather(1, selected.unsqueeze(1))):
            raise ValueError("PPO selected action is outside the legal action mask")
        distribution = torch.distributions.Categorical(logits=logits)
        log_probabilities = distribution.log_prob(selected)
        entropies = distribution.entropy()
        ratio = torch.exp(log_probabilities - old_log_probabilities)
        clip_epsilon = float(self.hyperparameters["clipEpsilon"])
        clipped_objective = torch.minimum(
            ratio * advantages,
            torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages,
        )
        squared_errors = F.mse_loss(values, returns, reduction="none")
        if not all(torch.isfinite(value).all() for value in (logits[returned_mask], values, log_probabilities, entropies, clipped_objective, squared_errors)):
            raise FloatingPointError("PPO packed calculation contains NaN or Inf")
        total_samples = state["totalSamples"]
        chunk_loss = (
            -clipped_objective.sum()
            + float(self.hyperparameters["valueCoefficient"]) * squared_errors.sum()
            - float(self.hyperparameters["entropyCoefficient"]) * entropies.sum()
        ) / total_samples
        if not torch.isfinite(chunk_loss):
            raise FloatingPointError("PPO packed loss contains NaN or Inf")
        chunk_loss.backward()
        state["processedSamples"] += chunk_size
        state["policyLossSum"] += float((-clipped_objective.sum()).detach().item())
        state["valueLossSum"] += float(squared_errors.sum().detach().item())
        state["entropySum"] += float(entropies.sum().detach().item())
        return {"acceptedSamples": chunk_size, "accumulatedSamples": state["processedSamples"]}

    def finish_accumulated_update(self) -> dict[str, Any]:
        state = self._accumulation
        if state is None:
            raise RuntimeError("No PPO accumulated update is active")
        if state["processedSamples"] != state["totalSamples"]:
            raise ValueError(
                f"PPO accumulated sample count mismatch: {state['processedSamples']} != {state['totalSamples']}"
            )
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("PPO gradient contains NaN or Inf")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), float(self.hyperparameters["maxGradientNorm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("PPO gradient norm is NaN or Inf")
        self.optimizer.step()
        self.update_count += 1
        total = state["totalSamples"]
        policy_loss = state["policyLossSum"] / total
        value_loss = state["valueLossSum"] / total
        entropy = state["entropySum"] / total
        loss = policy_loss + float(self.hyperparameters["valueCoefficient"]) * value_loss - float(self.hyperparameters["entropyCoefficient"]) * entropy
        result = {
            "sampleCount": total,
            "loss": loss,
            "policyLoss": policy_loss,
            "valueLoss": value_loss,
            "entropy": entropy,
            "gradientNorm": float(gradient_norm.detach().item()),
            "policyParametersChanged": state["policyBefore"] != self.parameter_hash(exclude_prefix="value_head."),
            "valueParametersChanged": state["valueBefore"] != self.parameter_hash(include_prefix="value_head."),
            "updateCount": self.update_count,
        }
        self._accumulation = None
        return result

    def parameter_hash(self, include_prefix: str | None = None, exclude_prefix: str | None = None) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters()):
            if include_prefix is not None and not name.startswith(include_prefix):
                continue
            if exclude_prefix is not None and name.startswith(exclude_prefix):
                continue
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def optimizer_hash(self) -> str:
        """Read-only, deterministic hash used by resume regression tests."""
        digest = hashlib.sha256()
        state = self.optimizer.state_dict()
        for group_index, group in enumerate(state["param_groups"]):
            digest.update(f"group:{group_index}".encode())
            for key in sorted(key for key in group if key != "params"):
                digest.update(key.encode())
                digest.update(repr(group[key]).encode())
            digest.update(repr(tuple(group["params"])).encode())
        for parameter_id in sorted(state["state"]):
            digest.update(f"parameter:{parameter_id}".encode())
            for key, value in sorted(state["state"][parameter_id].items()):
                digest.update(key.encode())
                if torch.is_tensor(value):
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                else:
                    digest.update(repr(value).encode())
        return digest.hexdigest()

    @staticmethod
    def _atomic_save(payload: dict[str, Any], path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(prefix=".ppo-checkpoint-", suffix=".tmp", dir=directory)
        os.close(descriptor)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def save(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        self._atomic_save({
            "schemaVersion": PPO_CHECKPOINT_SCHEMA_VERSION,
            "checkpointKind": "ppo_self_play",
            "featureSpec": self.feature_spec,
            "modelStateDict": self.model.state_dict(),
            "optimizerStateDict": self.optimizer.state_dict(),
            "updateCount": self.update_count,
            "episodeCount": self.episode_count,
            "hyperparameters": self.hyperparameters,
            "seed": self.seed,
            "torchRngState": torch.get_rng_state(),
            "cudaRngStateAll": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "metadata": metadata or {},
        }, path)

    def resume(self, path: str) -> dict[str, int]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or checkpoint.get("schemaVersion") != PPO_CHECKPOINT_SCHEMA_VERSION or checkpoint.get("checkpointKind") != "ppo_self_play":
            raise ValueError("Unsupported PPO checkpoint")
        if checkpoint.get("featureSpec") != self.feature_spec:
            raise ValueError("PPO resume Feature Spec mismatch")
        if checkpoint.get("hyperparameters") != self.hyperparameters or checkpoint.get("seed") != self.seed:
            raise ValueError("PPO resume configuration mismatch")
        for field in ("modelStateDict", "optimizerStateDict", "torchRngState"):
            if field not in checkpoint:
                raise ValueError(f"PPO checkpoint is missing {field}")
        self.model.load_state_dict(checkpoint["modelStateDict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizerStateDict"])
        self.update_count = int(checkpoint.get("updateCount", -1))
        self.episode_count = int(checkpoint.get("episodeCount", -1))
        if self.update_count < 0 or self.episode_count < 0:
            raise ValueError("PPO checkpoint counters are invalid")
        torch.set_rng_state(checkpoint["torchRngState"].cpu())
        cuda_states = checkpoint.get("cudaRngStateAll")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)
        return {"updateCount": self.update_count, "episodeCount": self.episode_count}
