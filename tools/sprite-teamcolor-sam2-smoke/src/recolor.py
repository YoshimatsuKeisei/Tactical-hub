#!/usr/bin/env python3
"""Recolor existing SAM 2 mask regions while preserving sprite shading."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


FRAME_NAME = re.compile(r"frame_(\d+)\.png")
DEFAULT_TARGETS = {
    "red": (210, 48, 48),
    "blue": (48, 92, 210),
    "green": (48, 170, 78),
    "yellow": (220, 180, 40),
}


def parse_target_color(value: str) -> tuple[int, int, int]:
    """Parse a named preset, #RRGGBB, or R,G,B target color."""
    normalized = value.strip().lower()
    if normalized in DEFAULT_TARGETS:
        return DEFAULT_TARGETS[normalized]
    if normalized.startswith("#") and len(normalized) == 7:
        try:
            return tuple(int(normalized[index : index + 2], 16) for index in (1, 3, 5))
        except ValueError as error:
            raise argparse.ArgumentTypeError("target must be a preset, #RRGGBB, or R,G,B") from error
    try:
        channels = tuple(int(channel) for channel in normalized.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("target must be a preset, #RRGGBB, or R,G,B") from error
    if len(channels) != 3 or any(channel < 0 or channel > 255 for channel in channels):
        raise argparse.ArgumentTypeError("target channels must be integers from 0 to 255")
    return channels


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if not 0 <= low < high <= 1:
        raise ValueError("shadow protection points must satisfy 0 <= low < high <= 1")
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _target_hsv_rgb(values: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    target_float = tuple(channel / 255.0 for channel in target)
    hue, saturation, _ = colorsys.rgb_to_hsv(*target_float)
    hue_sector = hue * 6.0
    sector = int(math.floor(hue_sector)) % 6
    chroma = values * saturation
    second = chroma * (1.0 - abs((hue_sector % 2.0) - 1.0))
    zero = np.zeros_like(values)
    sector_channels = (
        (chroma, second, zero),
        (second, chroma, zero),
        (zero, chroma, second),
        (zero, second, chroma),
        (second, zero, chroma),
        (chroma, zero, second),
    )
    base = np.stack(sector_channels[sector], axis=-1)
    return base + (values - chroma)[..., None]


def recolor_masked(
    image: Image.Image,
    mask: Image.Image,
    target: tuple[int, int, int] = DEFAULT_TARGETS["red"],
    strength: float = 0.85,
    shadow_protect: tuple[float, float] = (0.05, 0.25),
) -> Image.Image:
    """Shift masked RGB pixels toward a target hue without changing alpha.

    The original HSV value (maximum RGB channel) is restored after hue blending,
    so highlights and shadows retain their ordering. Very dark pixels receive a
    progressively weaker blend to avoid coloring near-black outlines too heavily.
    A grayscale mask is also accepted, allowing a lightly feathered mask edge.
    """
    if image.mode not in ("RGB", "RGBA"):
        raise ValueError(f"Expected RGB or RGBA input, got {image.mode}")
    if mask.size != image.size:
        raise ValueError(f"Mask size {mask.size} does not match image size {image.size}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between 0 and 1")
    if len(target) != 3 or any(channel < 0 or channel > 255 for channel in target):
        raise ValueError("target must contain three channels from 0 to 255")

    source_array = np.asarray(image, dtype=np.uint8)
    source_rgb = source_array[:, :, :3].astype(np.float32) / 255.0
    value = source_rgb.max(axis=2)
    tinted = _target_hsv_rgb(value, target)

    outline_weight = _smoothstep(value, *shadow_protect)
    color_weight = (strength * outline_weight)[..., None]
    recolored = source_rgb * (1.0 - color_weight) + tinted * color_weight

    # Blending two hues can lower the maximum channel. Normalize it back to the
    # original HSV value, which preserves each pixel's light/dark level.
    recolored_value = recolored.max(axis=2)
    scale = np.divide(value, recolored_value, out=np.ones_like(value), where=recolored_value > 0)
    recolored = np.clip(recolored * scale[..., None], 0.0, 1.0)

    mask_weight = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    result_rgb = source_rgb * (1.0 - mask_weight[..., None]) + recolored * mask_weight[..., None]
    result_rgb_u8 = np.rint(result_rgb * 255.0).astype(np.uint8)

    if image.mode == "RGBA":
        # Reuse the original bytes rather than recomputing alpha.
        result = np.concatenate((result_rgb_u8, source_array[:, :, 3:4]), axis=2)
        return Image.fromarray(result)
    return Image.fromarray(result_rgb_u8)


def discover_masks(mask_dir: Path, frame_count: int) -> list[tuple[int, Path]]:
    if frame_count <= 0:
        raise ValueError("frame count must be positive")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
    indexed: list[tuple[int, Path]] = []
    for path in mask_dir.iterdir():
        match = FRAME_NAME.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort(key=lambda item: item[0])
    if len(indexed) < frame_count:
        raise ValueError(f"Expected at least {frame_count} frame_*.png masks, found {len(indexed)}")
    return indexed[:frame_count]


def load_source_frames(
    input_path: Path,
    masks: Sequence[tuple[int, Path]],
    row: int,
    frame_size: tuple[int, int],
) -> list[Image.Image]:
    """Load matching frame PNGs or crop indexed frames from a sprite sheet."""
    if row < 0:
        raise ValueError("row must be non-negative")
    if input_path.is_dir():
        frames: list[Image.Image] = []
        for _, mask_path in masks:
            frame_path = input_path / mask_path.name
            if not frame_path.is_file():
                raise FileNotFoundError(f"Source frame not found: {frame_path}")
            with Image.open(frame_path) as frame:
                frame.load()
                frames.append(frame.copy())
        return frames
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image or frame directory not found: {input_path}")

    frame_width, frame_height = frame_size
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame width and height must be positive")
    with Image.open(input_path) as sprite:
        sprite.load()
        max_index = max(index for index, _ in masks)
        required_width = (max_index + 1) * frame_width
        required_height = (row + 1) * frame_height
        if sprite.width < required_width or sprite.height < required_height:
            raise ValueError(
                f"Sprite sheet {sprite.size} is too small for row {row}, frame {max_index}, "
                f"and frame size {frame_size}"
            )
        top = row * frame_height
        return [
            sprite.crop((index * frame_width, top, (index + 1) * frame_width, top + frame_height)).copy()
            for index, _ in masks
        ]


def _checkerboard(size: tuple[int, int], cell: int = 8) -> Image.Image:
    output = Image.new("RGB", size, (50, 50, 50))
    draw = ImageDraw.Draw(output)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle(
                    (x, y, min(x + cell - 1, size[0] - 1), min(y + cell - 1, size[1] - 1)),
                    fill=(72, 72, 72),
                )
    return output


def make_contact_sheet(
    frames: Sequence[Image.Image],
    labels: Sequence[str] | None = None,
    columns: int = 5,
) -> Image.Image:
    """Build the same labeled checkerboard contact sheet used by the CLI and GUI."""
    if not frames:
        raise ValueError("At least one frame is required for a contact sheet")
    if columns <= 0:
        raise ValueError("columns must be positive")
    if labels is None:
        labels = [f"frame_{index:03}" for index in range(len(frames))]
    if len(labels) != len(frames):
        raise ValueError("labels must match the frame count")

    sizes = [frame.size for frame in frames]
    cell_width = max(width for width, _ in sizes)
    cell_height = max(height for _, height in sizes)
    actual_columns = min(columns, len(frames))
    rows = math.ceil(len(frames) / actual_columns)
    label_height = 18
    sheet_size = (actual_columns * cell_width, rows * (cell_height + label_height))
    sheet = Image.new("RGB", sheet_size, (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for position, (source, label) in enumerate(zip(frames, labels, strict=True)):
        x = (position % actual_columns) * cell_width
        y = (position // actual_columns) * (cell_height + label_height)
        frame = source.convert("RGBA")
        background = _checkerboard((cell_width, cell_height))
        background.paste(frame, (0, 0), frame.getchannel("A"))
        sheet.paste(background, (x, y))
        draw.text((x + 4, y + cell_height + 2), label, fill=(240, 240, 240))

    return sheet


def save_contact_sheet(frame_paths: Iterable[Path], destination: Path, columns: int = 5) -> tuple[int, int]:
    paths = list(frame_paths)
    frames: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            source.load()
            frames.append(source.copy())
    sheet = make_contact_sheet(frames, [path.stem for path in paths], columns=columns)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return sheet.size


def run(args: argparse.Namespace) -> dict[str, object]:
    input_path = args.input.resolve()
    mask_dir = args.masks.resolve()
    output_dir = args.output.resolve()
    masks = discover_masks(mask_dir, args.frame_count)
    frames = load_source_frames(input_path, masks, args.row, (args.frame_width, args.frame_height))
    frames_dir = output_dir / "recolored_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for frame, (_, mask_path) in zip(frames, masks, strict=True):
        with Image.open(mask_path) as mask_source:
            mask_source.load()
            mask = mask_source.copy()
        recolored = recolor_masked(
            frame,
            mask,
            target=args.target,
            strength=args.strength,
            shadow_protect=(args.shadow_protect_low, args.shadow_protect_high),
        )
        destination = frames_dir / mask_path.name
        recolored.save(destination)
        output_paths.append(destination)

    contact_sheet = output_dir / "recolored_contact_sheet.png"
    contact_size = save_contact_sheet(output_paths, contact_sheet, columns=args.columns)
    return {
        "input": str(input_path),
        "masks": str(mask_dir),
        "target_rgb": list(args.target),
        "frame_count": len(output_paths),
        "recolored_frames": str(frames_dir),
        "contact_sheet": str(contact_sheet),
        "contact_sheet_size": list(contact_size),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="sprite sheet PNG or frame_*.png directory")
    parser.add_argument("--masks", required=True, type=Path, help="directory containing frame_*.png masks")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", type=parse_target_color, default=DEFAULT_TARGETS["red"])
    parser.add_argument("--frame-count", type=int, default=15)
    parser.add_argument("--row", type=int, default=0, help="sprite-sheet row to crop")
    parser.add_argument("--frame-width", type=int, default=128)
    parser.add_argument("--frame-height", type=int, default=128)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--shadow-protect-low", type=float, default=0.05)
    parser.add_argument("--shadow-protect-high", type=float, default=0.25)
    return parser


def main() -> int:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
