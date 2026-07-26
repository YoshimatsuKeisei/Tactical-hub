from __future__ import annotations

from typing import Any

import numpy as np
import torch


_DTYPES = {
    "float32": np.dtype("<f4"),
    "int32": np.dtype("<i4"),
    "uint8": np.dtype("u1"),
}


def decode_packed_views(header: dict[str, Any], payload: bytearray) -> dict[str, np.ndarray]:
    views: dict[str, np.ndarray] = {}
    for descriptor in header["tensors"]:
        dtype = _DTYPES.get(descriptor["dtype"])
        if dtype is None:
            raise ValueError(f"Unsupported packed dtype: {descriptor['dtype']}")
        shape = tuple(int(value) for value in descriptor["shape"])
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected != int(descriptor["byteLength"]):
            raise ValueError(f"Packed tensor byte length mismatch: {descriptor['name']}")
        offset = int(descriptor["byteOffset"])
        end = offset + expected
        if offset < 0 or end > len(payload):
            raise ValueError(f"Packed tensor is outside payload: {descriptor['name']}")
        views[descriptor["name"]] = np.frombuffer(
            payload, dtype=dtype, count=expected // dtype.itemsize, offset=offset
        ).reshape(shape)
    return views


def prepare_packed_tensors(
    views: dict[str, np.ndarray], device: torch.device
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    def floating(name: str) -> torch.Tensor:
        return torch.from_numpy(views[name]).to(device=device, dtype=torch.float32)

    def mask(name: str) -> torch.Tensor:
        return torch.from_numpy(views[name]).to(device=device, dtype=torch.bool)

    strategic_names = (
        "siegeStates", "kingCampaignStates", "rewardPlacementRequests",
        "strategistCooldowns", "teleportCooldowns", "productionIntents",
        "movementIntents", "attackIntents", "strategistActionIntents",
        "teleportIntents",
    )
    prepared = {
        "global": floating("global"),
        "strategicGlobal": floating("strategicGlobal"),
        "masked": {
            name: (floating(name), mask(mask_name))
            for name, mask_name in (
                ("teams", "teamMask"),
                ("units", "unitMask"),
                ("bases", "baseMask"),
                ("constructions", "constructionMask"),
            )
        },
        "map": (floating("map"), mask("mapMask")),
        "strategic": {
            name: (floating(f"strategic.{name}"), mask(f"strategicMask.{name}"))
            for name in strategic_names
        },
    }
    return (
        prepared,
        floating("actions"),
        mask("actionMask"),
        torch.from_numpy(views["targets"]).to(device=device, dtype=torch.long),
    )
