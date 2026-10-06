from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

FRAME_SIZE = 128

TEAM_COLORS: dict[str, str] = {
    "red": "#d94a4a",
    "blue": "#3e7bd8",
    "green": "#36a166",
    "yellow": "#c58a2b",
}


@dataclass(frozen=True)
class GridSpec:
    columns: int
    rows: int
    frame_size: int = FRAME_SIZE


@dataclass(frozen=True)
class PromptPoint:
    x: int
    y: int
    label: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must be 0 (negative) or 1 (positive)")
        if not (0 <= self.x < FRAME_SIZE and 0 <= self.y < FRAME_SIZE):
            raise ValueError("prompt must be inside the 128x128 frame")


def load_rgba_png(path: str | Path) -> Image.Image:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".png":
        raise ValueError("input must be a PNG sprite sheet")
    image = Image.open(path)
    image.load()
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


def infer_grid(image: Image.Image, frame_size: int = FRAME_SIZE) -> GridSpec:
    if image.width % frame_size or image.height % frame_size:
        raise ValueError(
            f"sprite sheet size must be a multiple of {frame_size}; "
            f"got {image.width}x{image.height}"
        )
    columns = image.width // frame_size
    rows = image.height // frame_size
    if columns < 1 or rows < 1:
        raise ValueError("sprite sheet contains no frames")
    return GridSpec(columns=columns, rows=rows, frame_size=frame_size)


def extract_frame(
    sheet: Image.Image,
    row: int,
    column: int,
    frame_size: int = FRAME_SIZE,
) -> Image.Image:
    grid = infer_grid(sheet, frame_size)
    if not (0 <= row < grid.rows):
        raise IndexError("row out of range")
    if not (0 <= column < grid.columns):
        raise IndexError("column out of range")
    left = column * frame_size
    top = row * frame_size
    return sheet.crop((left, top, left + frame_size, top + frame_size)).copy()


def extract_row_frames(
    sheet: Image.Image,
    row: int,
    frame_size: int = FRAME_SIZE,
) -> list[Image.Image]:
    grid = infer_grid(sheet, frame_size)
    return [extract_frame(sheet, row, column, frame_size) for column in range(grid.columns)]


def clip_mask_to_alpha(frame: Image.Image, mask: np.ndarray) -> np.ndarray:
    if mask.shape != (frame.height, frame.width):
        raise ValueError(
            f"mask shape {mask.shape} does not match frame {(frame.height, frame.width)}"
        )
    alpha = np.asarray(frame.getchannel("A")) > 0
    return np.asarray(mask, dtype=bool) & alpha


def _hex_to_rgb01(value: str) -> np.ndarray:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected #RRGGBB, got {value!r}")
    try:
        channels = [int(text[index : index + 2], 16) for index in (0, 2, 4)]
    except ValueError as error:
        raise ValueError(f"expected #RRGGBB, got {value!r}") from error
    return np.asarray(channels, dtype=np.float32) / 255.0


def recolor_rgba(
    image: Image.Image,
    mask: np.ndarray,
    target_hex: str,
    *,
    strength: float = 1.0,
    preserve_outline: bool = True,
    outline_luma_threshold: float = 0.055,
) -> Image.Image:
    """Recolor only mask pixels while preserving original light/dark shading.

    The mapping is luminance anchored: black -> black, the target color appears
    around the target color's own luminance, and highlights move toward white.
    This produces dark/mid/light variants of one team hue without flattening
    the original sprite shading.
    """

    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    selected = clip_mask_to_alpha(image.convert("RGBA"), mask)
    if not selected.any():
        return Image.fromarray(rgba, mode="RGBA")

    original = rgba[..., :3].astype(np.float32) / 255.0
    target = _hex_to_rgb01(target_hex)

    # Rec.709 luma is used only as a stable brightness carrier.
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma = np.sum(original * weights, axis=2)
    target_luma = float(np.dot(target, weights))
    target_luma = min(max(target_luma, 1e-4), 1.0 - 1e-4)

    colored = np.empty_like(original)

    darker = luma <= target_luma
    dark_scale = (luma / target_luma)[..., None]
    colored[darker] = (target * dark_scale[darker]).reshape(-1, 3)

    lighter = ~darker
    light_ratio = ((luma - target_luma) / (1.0 - target_luma))[..., None]
    light_ratio = np.clip(light_ratio, 0.0, 1.0)
    colored[lighter] = (
        target + (1.0 - target) * light_ratio[lighter]
    ).reshape(-1, 3)

    colored = np.clip(colored, 0.0, 1.0)
    mixed = original * (1.0 - strength) + colored * strength

    active = selected.copy()
    if preserve_outline:
        active &= luma >= outline_luma_threshold

    rgba[active, :3] = np.rint(mixed[active] * 255.0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def apply_row_masks(
    sheet: Image.Image,
    row: int,
    masks: Iterable[np.ndarray],
    target_hex: str,
    *,
    strength: float = 1.0,
    preserve_outline: bool = True,
    frame_size: int = FRAME_SIZE,
) -> Image.Image:
    grid = infer_grid(sheet, frame_size)
    mask_list = list(masks)
    if len(mask_list) != grid.columns:
        raise ValueError(
            f"expected {grid.columns} masks for row, got {len(mask_list)}"
        )

    output = sheet.convert("RGBA").copy()
    for column, mask in enumerate(mask_list):
        frame = extract_frame(output, row, column, frame_size)
        recolored = recolor_rgba(
            frame,
            mask,
            target_hex,
            strength=strength,
            preserve_outline=preserve_outline,
        )
        output.paste(recolored, (column * frame_size, row * frame_size))
    return output


def apply_all_tracked_rows(
    sheet: Image.Image,
    row_masks: dict[int, list[np.ndarray]],
    target_hex: str,
    *,
    strength: float = 1.0,
    preserve_outline: bool = True,
) -> Image.Image:
    output = sheet.convert("RGBA").copy()
    for row in sorted(row_masks):
        output = apply_row_masks(
            output,
            row,
            row_masks[row],
            target_hex,
            strength=strength,
            preserve_outline=preserve_outline,
        )
    return output


def save_team_variants(
    sheet: Image.Image,
    row_masks: dict[int, list[np.ndarray]],
    output_dir: str | Path,
    source_stem: str,
    *,
    strength: float = 1.0,
    preserve_outline: bool = True,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for team_name, color in TEAM_COLORS.items():
        variant = apply_all_tracked_rows(
            sheet,
            row_masks,
            color,
            strength=strength,
            preserve_outline=preserve_outline,
        )
        destination = output_dir / f"{source_stem}_{team_name}.png"
        variant.save(destination)
        written[team_name] = destination
    return written
