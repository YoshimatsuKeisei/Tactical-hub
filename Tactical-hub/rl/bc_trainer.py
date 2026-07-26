from __future__ import annotations

from typing import Any
import hashlib

import torch
from torch.nn import functional as F

from rl.policy_model import TacticalPolicyValueNetwork


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

    def save(self, path: str, metadata: dict[str, Any]) -> None:
        torch.save({
            "schemaVersion": 1,
            "featureSpec": self.feature_spec,
            "modelStateDict": self.model.state_dict(),
            "metadata": metadata,
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["featureSpec"] != self.feature_spec:
            raise ValueError("Checkpoint feature spec does not match")
        self.model.load_state_dict(checkpoint["modelStateDict"])

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for parameter in self.model.parameters():
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
