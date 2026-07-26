from __future__ import annotations

import sys
from typing import Literal

import torch

RequestedDevice = Literal["auto", "cpu", "cuda"]


def resolve_torch_device(requested: str) -> torch.device:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device("cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu")


def report_torch_device(requested: str, selected: torch.device) -> None:
    available = torch.cuda.is_available()
    details = [
        f"requested device={requested}",
        f"selected device={selected.type}",
        f"PyTorch={torch.__version__}",
        f"cuda.is_available={available}",
    ]
    if selected.type == "cuda":
        details.extend([
            f"GPU={torch.cuda.get_device_name(selected)}",
            f"PyTorch CUDA={torch.version.cuda}",
        ])
    print("[RL Device] " + " | ".join(details), file=sys.stderr, flush=True)
