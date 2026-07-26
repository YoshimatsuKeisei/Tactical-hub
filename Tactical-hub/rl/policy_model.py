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


def rows(value: list[list[float]], width: int, device: torch.device) -> torch.Tensor:
    if not value:
        return torch.empty((0, width), dtype=torch.float32, device=device)
    return torch.tensor(value, dtype=torch.float32, device=device)


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


def padded_rows(
    values: list[list[list[float]]],
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(values)
    max_rows = max((len(value) for value in values), default=0)
    tensor = torch.zeros((batch_size, max_rows, width), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, max_rows), dtype=torch.bool, device=device)
    for batch_index, value in enumerate(values):
        if value:
            row_tensor = torch.tensor(value, dtype=torch.float32, device=device)
            tensor[batch_index, :len(value)] = row_tensor
            mask[batch_index, :len(value)] = True
    return tensor, mask


def padded_masks(values: list[list[float]], max_rows: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(values), max_rows), dtype=torch.bool, device=device)
    for batch_index, value in enumerate(values):
        if value:
            mask[batch_index, :len(value)] = torch.tensor(value, dtype=torch.bool, device=device)
    return mask


def batched_masked_mean_pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if encoded.shape[1] == 0:
        return torch.zeros(
            (encoded.shape[0], encoded.shape[-1]),
            dtype=encoded.dtype,
            device=encoded.device,
        )
    numeric_mask = mask.to(dtype=encoded.dtype, device=encoded.device).unsqueeze(-1)
    denominator = numeric_mask.sum(dim=1).clamp(min=1.0)
    return (encoded * numeric_mask).sum(dim=1) / denominator


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

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_state(self, observation: dict[str, Any]) -> torch.Tensor:
        strategic = observation["strategicState"]
        embeddings = [
            self.global_encoder(torch.tensor(observation["global"], dtype=torch.float32, device=self.device)),
            masked_mean_pool(self.team_encoder(rows(observation["teams"], self.feature_spec["teamWidth"], self.device)), observation["teamMask"]),
            masked_mean_pool(self.unit_encoder(rows(observation["units"], self.feature_spec["unitWidth"], self.device)), observation["unitMask"]),
            mean_pool(self.map_encoder(rows([tile for row in observation["map"] for tile in row], self.feature_spec["mapTileWidth"], self.device))),
            masked_mean_pool(self.base_encoder(rows(observation["bases"], self.feature_spec["baseWidth"], self.device)), observation["baseMask"]),
            masked_mean_pool(self.construction_encoder(rows(observation["constructions"], self.feature_spec["constructionWidth"], self.device)), observation["constructionMask"]),
            self.strategic_global_encoder(torch.tensor(strategic["global"], dtype=torch.float32, device=self.device)),
        ]
        embeddings.extend(
            mean_pool(self.strategic_encoders[name](rows(strategic[name], self.feature_spec["strategicTableRowWidths"][name], self.device)))
            for name in STRATEGIC_TABLES
        )
        return self.state_encoder(torch.cat(embeddings, dim=0))

    def encode_actions(self, action_rows: list[list[float]]) -> torch.Tensor:
        return self.action_encoder(rows(action_rows, self.feature_spec["actionFeatureWidth"], self.device))

    def encode_state_batch(self, observations: list[dict[str, Any]]) -> torch.Tensor:
        if not observations:
            raise ValueError("Cannot encode an empty observation batch")

        def encode_masked_table(
            key: str,
            mask_key: str,
            width: int,
            encoder: nn.Module,
        ) -> torch.Tensor:
            table, presence_mask = padded_rows([observation[key] for observation in observations], width, self.device)
            explicit_mask = padded_masks([observation[mask_key] for observation in observations], table.shape[1], self.device)
            return batched_masked_mean_pool(encoder(table), presence_mask & explicit_mask)

        team_embedding = encode_masked_table("teams", "teamMask", self.feature_spec["teamWidth"], self.team_encoder)
        unit_embedding = encode_masked_table("units", "unitMask", self.feature_spec["unitWidth"], self.unit_encoder)
        base_embedding = encode_masked_table("bases", "baseMask", self.feature_spec["baseWidth"], self.base_encoder)
        construction_embedding = encode_masked_table(
            "constructions",
            "constructionMask",
            self.feature_spec["constructionWidth"],
            self.construction_encoder,
        )
        map_rows = [[tile for row in observation["map"] for tile in row] for observation in observations]
        map_table, map_mask = padded_rows(map_rows, self.feature_spec["mapTileWidth"], self.device)
        embeddings = [
            self.global_encoder(torch.tensor([observation["global"] for observation in observations], dtype=torch.float32, device=self.device)),
            team_embedding,
            unit_embedding,
            batched_masked_mean_pool(self.map_encoder(map_table), map_mask),
            base_embedding,
            construction_embedding,
            self.strategic_global_encoder(torch.tensor(
                [observation["strategicState"]["global"] for observation in observations],
                dtype=torch.float32,
                device=self.device,
            )),
        ]
        for name in STRATEGIC_TABLES:
            table, mask = padded_rows(
                [observation["strategicState"][name] for observation in observations],
                self.feature_spec["strategicTableRowWidths"][name],
                self.device,
            )
            embeddings.append(batched_masked_mean_pool(self.strategic_encoders[name](table), mask))
        return self.state_encoder(torch.cat(embeddings, dim=1))

    def encode_actions_batch(
        self,
        action_rows_batch: list[list[list[float]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_rows, action_mask = padded_rows(action_rows_batch, self.feature_spec["actionFeatureWidth"], self.device)
        return self.action_encoder(action_rows), action_mask

    def forward_batch(
        self,
        observations: list[dict[str, Any]],
        action_rows_batch: list[list[list[float]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(observations) != len(action_rows_batch):
            raise ValueError("Observation and action batch sizes must match")
        if any(not action_rows for action_rows in action_rows_batch):
            raise ValueError("Cannot forward a sample without legal actions")
        state_embeddings = self.encode_state_batch(observations)
        action_embeddings, action_mask = self.encode_actions_batch(action_rows_batch)
        repeated_states = state_embeddings.unsqueeze(1).expand(-1, action_embeddings.shape[1], -1)
        logits = self.score_head(torch.cat((repeated_states, action_embeddings), dim=2)).squeeze(-1)
        logits = logits.masked_fill(~action_mask, float("-inf"))
        values = self.value_head(state_embeddings).squeeze(-1)
        return logits, values, state_embeddings, action_embeddings, action_mask

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
