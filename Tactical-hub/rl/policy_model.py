from __future__ import annotations

from typing import Any

import torch
from torch import nn


STRATEGIC_TABLES = (
    "siegeStates",
    "kingCampaignStates",
    "rewardPlacementRequests",
    "strategistCooldowns",
    "teleportCooldowns",
    "productionIntents",
    "movementIntents",
    "attackIntents",
    "strategistActionIntents",
    "teleportIntents",
)


def mlp(input_width: int, hidden_width: int, output_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_width, hidden_width),
        nn.ReLU(),
        nn.Linear(hidden_width, output_width),
        nn.ReLU(),
    )


def rows(value: list[list[float]], width: int) -> torch.Tensor:
    if not value:
        return torch.empty((0, width), dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)


def mean_pool(encoded: torch.Tensor) -> torch.Tensor:
    if encoded.shape[0] == 0:
        return torch.zeros(encoded.shape[-1], dtype=encoded.dtype, device=encoded.device)
    return encoded.mean(dim=0)


def masked_mean_pool(encoded: torch.Tensor, mask_value: list[float]) -> torch.Tensor:
    if encoded.shape[0] == 0:
        return torch.zeros(encoded.shape[-1], dtype=encoded.dtype, device=encoded.device)
    mask = torch.tensor(mask_value, dtype=encoded.dtype, device=encoded.device).reshape(-1, 1)
    denominator = mask.sum().clamp(min=1.0)
    return (encoded * mask).sum(dim=0) / denominator


class TacticalPolicyValueNetwork(nn.Module):
    def __init__(self, feature_spec: dict[str, Any]):
        super().__init__()
        self.feature_spec = feature_spec
        self.global_encoder = mlp(feature_spec["globalWidth"], 64, 64)
        self.team_encoder = mlp(feature_spec["teamWidth"], 64, 64)
        self.unit_encoder = mlp(feature_spec["unitWidth"], 64, 64)
        self.map_encoder = mlp(feature_spec["mapTileWidth"], 64, 64)
        self.base_encoder = mlp(feature_spec["baseWidth"], 64, 64)
        self.construction_encoder = mlp(feature_spec["constructionWidth"], 64, 64)
        self.strategic_global_encoder = mlp(feature_spec["strategicGlobalWidth"], 64, 64)
        self.strategic_encoders = nn.ModuleDict({
            name: mlp(feature_spec["strategicTableRowWidths"][name], 64, 64)
            for name in STRATEGIC_TABLES
        })
        embedding_count = 7 + len(STRATEGIC_TABLES)
        self.state_encoder = nn.Sequential(
            nn.Linear(embedding_count * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(feature_spec["actionFeatureWidth"], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def encode_state(self, observation: dict[str, Any]) -> torch.Tensor:
        strategic = observation["strategicState"]
        embeddings = [
            self.global_encoder(torch.tensor(observation["global"], dtype=torch.float32)),
            masked_mean_pool(self.team_encoder(rows(observation["teams"], self.feature_spec["teamWidth"])), observation["teamMask"]),
            masked_mean_pool(self.unit_encoder(rows(observation["units"], self.feature_spec["unitWidth"])), observation["unitMask"]),
            mean_pool(self.map_encoder(rows([tile for row in observation["map"] for tile in row], self.feature_spec["mapTileWidth"]))),
            masked_mean_pool(self.base_encoder(rows(observation["bases"], self.feature_spec["baseWidth"])), observation["baseMask"]),
            masked_mean_pool(self.construction_encoder(rows(observation["constructions"], self.feature_spec["constructionWidth"])), observation["constructionMask"]),
            self.strategic_global_encoder(torch.tensor(strategic["global"], dtype=torch.float32)),
        ]
        embeddings.extend(
            mean_pool(self.strategic_encoders[name](rows(strategic[name], self.feature_spec["strategicTableRowWidths"][name])))
            for name in STRATEGIC_TABLES
        )
        return self.state_encoder(torch.cat(embeddings, dim=0))

    def encode_actions(self, action_rows: list[list[float]]) -> torch.Tensor:
        return self.action_encoder(rows(action_rows, self.feature_spec["actionFeatureWidth"]))

    def forward(self, observation: dict[str, Any], action_rows: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_embedding = self.encode_state(observation)
        action_embeddings = self.encode_actions(action_rows)
        repeated_state = state_embedding.unsqueeze(0).expand(action_embeddings.shape[0], -1)
        logits = self.score_head(torch.cat((repeated_state, action_embeddings), dim=1)).squeeze(-1)
        value = self.value_head(state_embedding).squeeze(-1)
        return logits, value, state_embedding, action_embeddings

    def act(self, observation: dict[str, Any], action_rows: list[list[float]]) -> tuple[int, float]:
        if not action_rows:
            raise ValueError("Cannot act without legal actions")
        with torch.no_grad():
            logits, value, _, _ = self.forward(observation, action_rows)
            action_index = int(torch.distributions.Categorical(logits=logits).sample().item())
        return action_index, float(value.item())
