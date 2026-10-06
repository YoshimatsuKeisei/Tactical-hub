"""GUI-independent state and image operations for the sprite team-color MVP."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from recolor import DEFAULT_TARGETS, make_contact_sheet, parse_target_color, recolor_masked


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SheetLayout:
    frame_width: int = 128
    frame_height: int = 128
    columns: int = 15
    rows: int = 8
    target_row: int = 0
    frame_count: int = 15

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.frame_width, self.frame_height

    def validate(self, image_size: tuple[int, int]) -> None:
        integer_values = (
            self.frame_width,
            self.frame_height,
            self.columns,
            self.rows,
            self.frame_count,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("Frame dimensions, columns, rows, and frame count must be positive")
        if not 0 <= self.target_row < self.rows:
            raise ValueError(f"Target row must be between 0 and {self.rows - 1}")
        if self.frame_count > self.columns:
            raise ValueError("Frame count cannot exceed the number of columns")
        expected = (self.frame_width * self.columns, self.frame_height * self.rows)
        if image_size != expected:
            raise ValueError(
                f"Spritesheet size {image_size[0]}x{image_size[1]} does not match "
                f"{self.columns}x{self.rows} frames of {self.frame_width}x{self.frame_height} "
                f"(expected {expected[0]}x{expected[1]})"
            )


@dataclass(frozen=True)
class PromptClick:
    x: int
    y: int
    label: int


class PromptState:
    def __init__(self, frame_size: tuple[int, int]) -> None:
        self.frame_size = frame_size
        self._clicks: list[PromptClick] = []

    @property
    def clicks(self) -> tuple[PromptClick, ...]:
        return tuple(self._clicks)

    @property
    def has_positive(self) -> bool:
        return any(click.label == 1 for click in self._clicks)

    def add(self, x: int, y: int, label: int) -> PromptClick:
        width, height = self.frame_size
        if label not in (0, 1):
            raise ValueError("Prompt label must be 0 (negative) or 1 (positive)")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Prompt point is outside the frame")
        click = PromptClick(x, y, label)
        self._clicks.append(click)
        return click

    def undo(self) -> PromptClick | None:
        return self._clicks.pop() if self._clicks else None

    def clear(self) -> None:
        self._clicks.clear()


def load_sprite_png(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Spritesheet not found: {path}")
    try:
        with Image.open(path) as source:
            if source.format != "PNG":
                raise ValueError(f"Expected a PNG spritesheet, got {source.format or 'unknown format'}")
            source.load()
            if source.mode not in ("RGB", "RGBA"):
                raise ValueError(f"Expected RGB or RGBA spritesheet, got {source.mode}")
            return source.copy()
    except OSError as error:
        raise ValueError(f"Could not read PNG spritesheet: {error}") from error


def extract_row_frames(sprite: Image.Image, layout: SheetLayout) -> list[Image.Image]:
    layout.validate(sprite.size)
    top = layout.target_row * layout.frame_height
    return [
        sprite.crop(
            (
                index * layout.frame_width,
                top,
                (index + 1) * layout.frame_width,
                top + layout.frame_height,
            )
        ).copy()
        for index in range(layout.frame_count)
    ]


def display_to_frame(
    display_x: float,
    display_y: float,
    display_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> tuple[int, int]:
    display_width, display_height = display_size
    frame_width, frame_height = frame_size
    if display_width <= 0 or display_height <= 0:
        raise ValueError("Display dimensions must be positive")
    if not (0 <= display_x < display_width and 0 <= display_y < display_height):
        raise ValueError("Display coordinate is outside the image")
    x = min(int(display_x * frame_width / display_width), frame_width - 1)
    y = min(int(display_y * frame_height / display_height), frame_height - 1)
    return x, y


def choose_display_scale(
    frame_size: tuple[int, int],
    available_size: tuple[int, int],
    preferred_scale: int = 4,
) -> int:
    """Choose the largest whole-number scale that fits, capped at the preference."""
    frame_width, frame_height = frame_size
    available_width, available_height = available_size
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive")
    if available_width <= 0 or available_height <= 0:
        raise ValueError("Available display dimensions must be positive")
    if preferred_scale <= 0:
        raise ValueError("Preferred display scale must be positive")
    fitting_scale = min(available_width // frame_width, available_height // frame_height)
    return max(1, min(preferred_scale, fitting_scale))


def frame_output_names(frame_count: int) -> list[str]:
    if frame_count <= 0:
        raise ValueError("Frame count must be positive")
    return [f"frame_{index:03}.png" for index in range(frame_count)]


def parse_team_color(preset: str, custom: str = "") -> tuple[int, int, int]:
    value = custom.strip() if preset.strip().lower() == "custom" else preset
    if not value:
        raise ValueError("Enter a custom color as #RRGGBB or R,G,B")
    return parse_target_color(value)


def clip_mask_to_frame(mask: np.ndarray, frame: Image.Image) -> np.ndarray:
    boolean_mask = np.asarray(mask, dtype=bool)
    expected_shape = (frame.height, frame.width)
    if boolean_mask.shape != expected_shape:
        raise ValueError(f"Mask shape {boolean_mask.shape} does not match frame {expected_shape}")
    if frame.mode == "RGBA":
        alpha_support = np.asarray(frame.getchannel("A")) > 0
        return np.logical_and(boolean_mask, alpha_support)
    return boolean_mask.copy()


def make_mask_overlay(
    frame: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 170, 255),
    opacity: float = 0.55,
) -> Image.Image:
    if not 0 <= opacity <= 1:
        raise ValueError("Overlay opacity must be between 0 and 1")
    rgba = np.asarray(frame.convert("RGBA"), dtype=np.uint8).copy()
    selected = clip_mask_to_frame(mask, frame)
    original = rgba[selected, :3].astype(np.float32)
    overlay = np.asarray(color, dtype=np.float32)
    rgba[selected, :3] = np.rint(original * (1.0 - opacity) + overlay * opacity).astype(np.uint8)
    return Image.fromarray(rgba)


def recolor_frame_sequence(
    frames: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    target: tuple[int, int, int],
) -> list[Image.Image]:
    if len(frames) != len(masks):
        raise ValueError("Frame and mask counts do not match")
    output: list[Image.Image] = []
    for frame, mask in zip(frames, masks, strict=True):
        clipped = clip_mask_to_frame(mask, frame)
        mask_image = Image.fromarray(clipped.astype(np.uint8) * 255)
        output.append(recolor_masked(frame, mask_image, target=target))
    return output


def save_results(
    destination: Path,
    frames: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    recolored_frames: Sequence[Image.Image],
    columns: int = 5,
) -> dict[str, Path]:
    if not (len(frames) == len(masks) == len(recolored_frames)):
        raise ValueError("Frame, mask, and recolored frame counts do not match")
    names = frame_output_names(len(frames))
    masks_dir = destination / "masks"
    recolored_dir = destination / "recolored_frames"
    masks_dir.mkdir(parents=True, exist_ok=True)
    recolored_dir.mkdir(parents=True, exist_ok=True)

    overlays: list[Image.Image] = []
    for name, frame, mask, recolored in zip(names, frames, masks, recolored_frames, strict=True):
        clipped = clip_mask_to_frame(mask, frame)
        Image.fromarray(clipped.astype(np.uint8) * 255).save(masks_dir / name)
        recolored.save(recolored_dir / name)
        overlays.append(make_mask_overlay(frame, clipped))

    labels = [Path(name).stem for name in names]
    recolored_contact = make_contact_sheet(list(recolored_frames), labels, columns=columns)
    overlay_contact = make_contact_sheet(overlays, labels, columns=columns)
    recolored_contact_path = destination / "recolored_contact_sheet.png"
    overlay_contact_path = destination / "overlay_contact_sheet.png"
    recolored_contact.save(recolored_contact_path)
    overlay_contact.save(overlay_contact_path)
    return {
        "masks": masks_dir,
        "recolored_frames": recolored_dir,
        "recolored_contact_sheet": recolored_contact_path,
        "overlay_contact_sheet": overlay_contact_path,
    }


class Sam2GuiSession:
    """Lazily owns the official SAM 2 predictor for one extracted row."""

    def __init__(
        self,
        frames: Sequence[Image.Image],
        checkpoint_path: Path,
        device_name: str = "auto",
        work_size: int = 1024,
        model_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
    ) -> None:
        if not frames:
            raise ValueError("At least one frame is required")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM 2.1 checkpoint not found: {checkpoint_path}")
        self.frames = [frame.copy() for frame in frames]
        self.checkpoint_path = checkpoint_path
        self.device_name = device_name
        self.work_size = work_size
        self.model_config = model_config
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sprite-teamcolor-sam2-"))
        self.predictor: object | None = None
        self.inference_state: object | None = None
        self.device: object | None = None
        self.frame0_mask: np.ndarray | None = None
        self.masks: list[np.ndarray] | None = None

    def _ensure_ready(self, progress: ProgressCallback) -> None:
        if self.predictor is not None:
            return
        try:
            from smoke import (
                create_sam2_predictor,
                initialize_sam2_state,
                prepare_frame_sequence,
                resolve_device,
            )
        except ImportError as error:
            raise RuntimeError("Could not import the SAM 2 smoke integration") from error

        progress("Preparing SAM 2 frames...")
        sam_frames = prepare_frame_sequence(self.frames, self.temp_dir, self.work_size)
        progress("Loading SAM 2 model...")
        self.device = resolve_device(self.device_name)
        self.predictor = create_sam2_predictor(self.model_config, self.checkpoint_path, self.device)
        progress("Initializing SAM 2 video state...")
        self.inference_state = initialize_sam2_state(self.predictor, sam_frames, self.device)

    def generate_frame0_mask(
        self,
        clicks: Sequence[PromptClick],
        progress: ProgressCallback = lambda _message: None,
    ) -> np.ndarray:
        if not any(click.label == 1 for click in clicks):
            raise ValueError("Add at least one positive click before generating a mask")
        self._ensure_ready(progress)
        from smoke import PromptPoint, add_sam2_prompts

        progress("Generating frame 0 mask...")
        points = [PromptPoint(click.x, click.y, click.label) for click in clicks]
        raw_mask = add_sam2_prompts(
            self.predictor,
            self.inference_state,
            points,
            self.frames[0].size,
            self.work_size,
        )
        self.frame0_mask = clip_mask_to_frame(raw_mask, self.frames[0])
        self.masks = None
        return self.frame0_mask.copy()

    def track_across_frames(
        self,
        progress: ProgressCallback = lambda _message: None,
    ) -> list[np.ndarray]:
        if self.frame0_mask is None or self.predictor is None or self.inference_state is None:
            raise RuntimeError("Generate and review the frame 0 mask before tracking")
        from smoke import propagate_sam2_masks

        progress("Propagating mask across frames...")
        propagated = propagate_sam2_masks(self.predictor, self.inference_state, self.frames[0].size)
        expected = list(range(len(self.frames)))
        if sorted(propagated) != expected:
            raise RuntimeError(f"Expected propagated frames {expected}, got {sorted(propagated)}")
        self.masks = [clip_mask_to_frame(propagated[index], frame) for index, frame in enumerate(self.frames)]
        self.masks[0] = self.frame0_mask.copy()
        return [mask.copy() for mask in self.masks]

    def close(self) -> None:
        self.predictor = None
        self.inference_state = None
        self.device = None
        shutil.rmtree(self.temp_dir, ignore_errors=True)


TEAM_COLOR_NAMES = tuple(name.title() for name in DEFAULT_TARGETS) + ("Custom",)
