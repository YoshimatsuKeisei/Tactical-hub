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


_VARIABLE_ROW_TENSORS = {
    "teams", "units", "bases", "constructions", "map", "actions",
    "strategic.siegeStates", "strategic.kingCampaignStates",
    "strategic.rewardPlacementRequests", "strategic.strategistCooldowns",
    "strategic.teleportCooldowns", "strategic.productionIntents",
    "strategic.movementIntents", "strategic.attackIntents",
    "strategic.strategistActionIntents", "strategic.teleportIntents",
}

_VARIABLE_ROW_MASKS = {
    "teamMask", "unitMask", "baseMask", "constructionMask", "mapMask", "actionMask",
    "strategicMask.siegeStates", "strategicMask.kingCampaignStates",
    "strategicMask.rewardPlacementRequests", "strategicMask.strategistCooldowns",
    "strategicMask.teleportCooldowns", "strategicMask.productionIntents",
    "strategicMask.movementIntents", "strategicMask.attackIntents",
    "strategicMask.strategistActionIntents", "strategicMask.teleportIntents",
}


def combine_single_sample_packed_views(
    records: list[dict[str, np.ndarray]],
    selected_action_indices: list[int],
) -> dict[str, np.ndarray]:
    if not records or len(records) != len(selected_action_indices):
        raise ValueError("Retained PPO records and selected-action indices must be non-empty and aligned")
    names = list(records[0].keys())
    if any(list(record.keys()) != names for record in records):
        raise ValueError("Retained PPO tensor names/order mismatch")
    batch_size = len(records)
    combined: dict[str, np.ndarray] = {}
    for name in names:
        if name == "targets":
            continue
        arrays = [record[name] for record in records]
        if any(array.shape[0] != 1 for array in arrays):
            raise ValueError(f"Retained PPO tensor is not a single-sample batch: {name}")
        if name in _VARIABLE_ROW_TENSORS:
            if any(array.ndim != 3 for array in arrays):
                raise ValueError(f"Retained PPO row tensor rank mismatch: {name}")
            width = arrays[0].shape[2]
            if any(array.shape[2] != width for array in arrays):
                raise ValueError(f"Retained PPO row tensor width mismatch: {name}")
            max_rows = max(array.shape[1] for array in arrays)
            output = np.zeros((batch_size, max_rows, width), dtype=arrays[0].dtype)
            for index, array in enumerate(arrays):
                output[index, : array.shape[1], :] = array[0]
            combined[name] = output
        elif name in _VARIABLE_ROW_MASKS:
            if any(array.ndim != 2 for array in arrays):
                raise ValueError(f"Retained PPO mask tensor rank mismatch: {name}")
            max_rows = max(array.shape[1] for array in arrays)
            output = np.zeros((batch_size, max_rows), dtype=arrays[0].dtype)
            for index, array in enumerate(arrays):
                output[index, : array.shape[1]] = array[0]
            combined[name] = output
        else:
            shapes = [array.shape[1:] for array in arrays]
            if any(shape != shapes[0] for shape in shapes):
                raise ValueError(f"Retained PPO fixed tensor shape mismatch: {name}")
            combined[name] = np.concatenate(arrays, axis=0)
    combined["targets"] = np.asarray(selected_action_indices, dtype=np.dtype("<i4"))
    return combined


def packed_views_audit(
    views: dict[str, np.ndarray],
    template_descriptors: list[dict[str, Any]],
) -> dict[str, Any]:
    import hashlib
    import json

    digest = hashlib.sha256()
    descriptors: list[dict[str, Any]] = []
    raw_bytes = 0
    for template in template_descriptors:
        name = template["name"]
        array = np.ascontiguousarray(views[name])
        payload = array.tobytes(order="C")
        digest.update(payload)
        descriptors.append({
            "name": name,
            "dtype": template["dtype"],
            "shape": list(array.shape),
            "byteLength": len(payload),
        })
        raw_bytes += len(payload)
    signature = hashlib.sha256(
        json.dumps(descriptors, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "batchSize": int(views["targets"].shape[0]),
        "rawBytes": raw_bytes,
        "rawSha256": digest.hexdigest(),
        "tensorSignature": signature,
    }


def prepare_packed_tensors_grouped_h2d(
    header: dict[str, Any],
    payload: bytearray,
    device: torch.device,
    *,
    include_targets: bool = True,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor | None]:
    descriptors = header["tensors"]
    by_name = {descriptor["name"]: descriptor for descriptor in descriptors}

    def grouped(dtype_name: str, torch_dtype: torch.dtype) -> tuple[torch.Tensor, int, int]:
        selected = [descriptor for descriptor in descriptors if descriptor["dtype"] == dtype_name]
        if not selected:
            return torch.empty(0, dtype=torch_dtype, device=device), 0, 1
        itemsize = _DTYPES[dtype_name].itemsize
        start = int(selected[0]["byteOffset"])
        end = int(selected[-1]["byteOffset"]) + int(selected[-1]["byteLength"])
        cursor = start
        for descriptor in selected:
            if int(descriptor["byteOffset"]) != cursor:
                raise ValueError(f"Packed {dtype_name} tensors are not contiguous")
            cursor += int(descriptor["byteLength"])
        cpu = np.frombuffer(
            payload,
            dtype=_DTYPES[dtype_name],
            count=(end - start) // itemsize,
            offset=start,
        )
        return torch.from_numpy(cpu).to(device=device, dtype=torch_dtype), start, itemsize

    floats, float_start, float_itemsize = grouped("float32", torch.float32)
    masks, mask_start, mask_itemsize = grouped("uint8", torch.bool)
    targets_flat: torch.Tensor | None = None
    int_start = 0
    int_itemsize = _DTYPES["int32"].itemsize
    if include_targets:
        targets_flat, int_start, int_itemsize = grouped("int32", torch.long)

    def view(name: str, source: torch.Tensor, group_start: int, itemsize: int) -> torch.Tensor:
        descriptor = by_name[name]
        offset_bytes = int(descriptor["byteOffset"]) - group_start
        if offset_bytes < 0 or offset_bytes % itemsize:
            raise ValueError(f"Packed tensor alignment mismatch: {name}")
        shape = tuple(int(value) for value in descriptor["shape"])
        count = int(np.prod(shape, dtype=np.int64))
        return source.narrow(0, offset_bytes // itemsize, count).view(shape)

    def floating(name: str) -> torch.Tensor:
        return view(name, floats, float_start, float_itemsize)

    def mask(name: str) -> torch.Tensor:
        return view(name, masks, mask_start, mask_itemsize)

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
    targets = (
        view("targets", targets_flat, int_start, int_itemsize)
        if include_targets and targets_flat is not None
        else None
    )
    return prepared, floating("actions"), mask("actionMask"), targets


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
