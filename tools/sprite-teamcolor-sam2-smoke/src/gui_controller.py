"""GUI-independent state and image operations for the sprite team-color MVP."""

from __future__ import annotations

from collections import deque
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from recolor import DEFAULT_TARGETS, make_contact_sheet, parse_target_color, recolor_masked


ProgressCallback = Callable[[str], None]
MIN_ZOOM = 1
MAX_ZOOM = 16
MIN_MASK_THRESHOLD = -2.0
MAX_MASK_THRESHOLD = 2.0
DEFAULT_MASK_THRESHOLD = 0.0


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


def polygon_to_mask(
    points: Sequence[tuple[int, int]],
    frame_size: tuple[int, int],
) -> np.ndarray:
    """Rasterize a closed freehand polygon into a frame-sized boolean mask."""
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    if len(points) < 3:
        raise ValueError("A lasso requires at least three points")
    normalized = [(int(x), int(y)) for x, y in points]
    if any(not (0 <= x < width and 0 <= y < height) for x, y in normalized):
        raise ValueError("Lasso point is outside the frame")
    image = Image.new("1", frame_size, 0)
    ImageDraw.Draw(image).polygon(normalized, fill=1)
    return np.asarray(image, dtype=bool).copy()


def shape_bounds(
    start: tuple[int, int],
    end: tuple[int, int],
    frame_size: tuple[int, int],
    constrain_square: bool = False,
) -> tuple[int, int, int, int]:
    """Return an inclusive, normalized drag box inside the source frame."""
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    if any(not (0 <= x < width and 0 <= y < height) for x, y in (start, end)):
        raise ValueError("Shape point is outside the frame")
    start_x, start_y = (int(start[0]), int(start[1]))
    end_x, end_y = (int(end[0]), int(end[1]))
    if constrain_square:
        direction_x = 1 if end_x >= start_x else -1
        direction_y = 1 if end_y >= start_y else -1
        available_x = width - 1 - start_x if direction_x > 0 else start_x
        available_y = height - 1 - start_y if direction_y > 0 else start_y
        side = min(
            max(abs(end_x - start_x), abs(end_y - start_y)),
            available_x,
            available_y,
        )
        end_x = start_x + side * direction_x
        end_y = start_y + side * direction_y
    return min(start_x, end_x), min(start_y, end_y), max(start_x, end_x), max(start_y, end_y)


def rectangle_to_mask(
    start: tuple[int, int],
    end: tuple[int, int],
    frame_size: tuple[int, int],
    constrain_square: bool = False,
) -> np.ndarray:
    """Rasterize an inclusive drag rectangle into a boolean mask."""
    bounds = shape_bounds(start, end, frame_size, constrain_square)
    image = Image.new("1", frame_size, 0)
    ImageDraw.Draw(image).rectangle(bounds, fill=1)
    return np.asarray(image, dtype=bool).copy()


def ellipse_to_mask(
    start: tuple[int, int],
    end: tuple[int, int],
    frame_size: tuple[int, int],
    constrain_circle: bool = False,
) -> np.ndarray:
    """Rasterize an ellipse within an inclusive drag box into a boolean mask."""
    bounds = shape_bounds(start, end, frame_size, constrain_circle)
    image = Image.new("1", frame_size, 0)
    ImageDraw.Draw(image).ellipse(bounds, fill=1)
    return np.asarray(image, dtype=bool).copy()


def select_all_mask(frame: Image.Image) -> np.ndarray:
    """Select visible RGBA pixels, or the complete frame for RGB input."""
    if frame.mode == "RGBA":
        return (np.asarray(frame.getchannel("A")) > 0).copy()
    if frame.mode == "RGB":
        return np.ones((frame.height, frame.width), dtype=bool)
    raise ValueError(f"Select All requires an RGB or RGBA frame, got {frame.mode}")


class LassoSelectionState:
    """Boolean selection with a lightweight multi-level undo history."""

    def __init__(self, frame_size: tuple[int, int]) -> None:
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        self.frame_size = frame_size
        self._mask = np.zeros((height, width), dtype=bool)
        self._history: list[np.ndarray] = []

    @property
    def mask(self) -> np.ndarray:
        return self._mask.copy()

    @property
    def is_empty(self) -> bool:
        return not bool(self._mask.any())

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def apply(self, points: Sequence[tuple[int, int]], operation: str) -> np.ndarray:
        polygon = polygon_to_mask(points, self.frame_size)
        return self.apply_mask(polygon, operation)

    def apply_mask(self, mask: np.ndarray, operation: str) -> np.ndarray:
        """Apply any frame-sized shape through the shared Add/Subtract history."""
        shape_mask = np.asarray(mask, dtype=bool)
        expected_shape = (self.frame_size[1], self.frame_size[0])
        if shape_mask.shape != expected_shape:
            raise ValueError(f"Selection mask shape {shape_mask.shape} does not match {expected_shape}")
        normalized_operation = operation.strip().lower()
        if normalized_operation not in ("add", "subtract"):
            raise ValueError("Selection operation must be add or subtract")
        self._history.append(self._mask.copy())
        if normalized_operation == "add":
            self._mask = np.logical_or(self._mask, shape_mask)
        else:
            self._mask = np.logical_and(self._mask, ~shape_mask)
        return self.mask

    def undo(self) -> np.ndarray | None:
        if not self._history:
            return None
        self._mask = self._history.pop()
        return self.mask

    def clear(self) -> np.ndarray:
        if self._mask.any():
            self._history.append(self._mask.copy())
        self._mask.fill(False)
        return self.mask


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


def clamp_zoom(zoom: int, minimum: int = MIN_ZOOM, maximum: int = MAX_ZOOM) -> int:
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Zoom limits must satisfy 0 < minimum <= maximum")
    return max(minimum, min(maximum, int(zoom)))


def percentage_to_unit(value: float) -> float:
    """Convert one GUI percentage in the inclusive 0..100 range to 0..1."""
    normalized = float(value)
    if not 0.0 <= normalized <= 100.0:
        raise ValueError("percentage must be between 0 and 100")
    return normalized / 100.0


def validate_mask_threshold(value: float) -> float:
    """Validate the diagnostic SAM logit threshold used by the GUI."""
    threshold = float(value)
    if not MIN_MASK_THRESHOLD <= threshold <= MAX_MASK_THRESHOLD:
        raise ValueError(
            f"Mask threshold must be between {MIN_MASK_THRESHOLD:.1f} and {MAX_MASK_THRESHOLD:.1f}"
        )
    return threshold


def viewport_to_frame(
    viewport_x: float,
    viewport_y: float,
    zoom: int,
    scroll_offset: tuple[float, float],
    frame_size: tuple[int, int],
) -> tuple[int, int]:
    """Map a scrolled, zoomed canvas viewport coordinate to one frame pixel."""
    validated_zoom = clamp_zoom(zoom)
    frame_width, frame_height = frame_size
    image_x = viewport_x + scroll_offset[0]
    image_y = viewport_y + scroll_offset[1]
    display_size = (frame_width * validated_zoom, frame_height * validated_zoom)
    return display_to_frame(image_x, image_y, display_size, frame_size)


def cursor_centered_zoom_offset(
    cursor: tuple[float, float],
    old_zoom: int,
    new_zoom: int,
    old_scroll: tuple[float, float],
    viewport_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> tuple[float, float]:
    """Keep the source point under the cursor stable while changing zoom."""
    old_zoom = clamp_zoom(old_zoom)
    new_zoom = clamp_zoom(new_zoom)
    anchor_x = (old_scroll[0] + cursor[0]) / old_zoom
    anchor_y = (old_scroll[1] + cursor[1]) / old_zoom
    desired_x = anchor_x * new_zoom - cursor[0]
    desired_y = anchor_y * new_zoom - cursor[1]
    content_size = (frame_size[0] * new_zoom, frame_size[1] * new_zoom)
    max_x = max(0.0, content_size[0] - viewport_size[0])
    max_y = max(0.0, content_size[1] - viewport_size[1])
    return max(0.0, min(max_x, desired_x)), max(0.0, min(max_y, desired_y))


def pan_scroll_offset(
    current: tuple[float, float],
    movement: tuple[float, float],
    content_size: tuple[int, int],
    viewport_size: tuple[int, int],
) -> tuple[float, float]:
    """Apply an arrow-key pan and clamp it to the scrollable image bounds."""
    max_x = max(0.0, content_size[0] - viewport_size[0])
    max_y = max(0.0, content_size[1] - viewport_size[1])
    return (
        max(0.0, min(max_x, current[0] + movement[0])),
        max(0.0, min(max_y, current[1] + movement[1])),
    )


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


def fill_enclosed_holes(mask: np.ndarray) -> np.ndarray:
    """Fill 4-connected False regions that cannot reach the image boundary."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.shape[0] <= 0 or binary.shape[1] <= 0:
        raise ValueError("Mask must be a non-empty 2D array")

    background = ~binary
    exterior = np.zeros(binary.shape, dtype=bool)
    pending: deque[tuple[int, int]] = deque()
    height, width = binary.shape

    def enqueue_if_background(y: int, x: int) -> None:
        if background[y, x] and not exterior[y, x]:
            exterior[y, x] = True
            pending.append((y, x))

    for x in range(width):
        enqueue_if_background(0, x)
        enqueue_if_background(height - 1, x)
    for y in range(height):
        enqueue_if_background(y, 0)
        enqueue_if_background(y, width - 1)

    while pending:
        y, x = pending.popleft()
        if y > 0:
            enqueue_if_background(y - 1, x)
        if y + 1 < height:
            enqueue_if_background(y + 1, x)
        if x > 0:
            enqueue_if_background(y, x - 1)
        if x + 1 < width:
            enqueue_if_background(y, x + 1)

    enclosed_holes = np.logical_and(background, ~exterior)
    return np.logical_or(binary, enclosed_holes)


def apply_mask_cleanup(mask: np.ndarray, fill_holes: bool) -> np.ndarray:
    """Derive an active binary mask without modifying its raw source array."""
    binary = np.asarray(mask, dtype=bool)
    return fill_enclosed_holes(binary) if fill_holes else binary.copy()


def apply_mask_cleanup_sequence(
    masks: Sequence[np.ndarray],
    fill_holes: bool,
) -> list[np.ndarray]:
    return [apply_mask_cleanup(mask, fill_holes) for mask in masks]


def derive_masks_from_sam_logits(
    raw_logits: Sequence[np.ndarray],
    frames: Sequence[Image.Image],
    threshold: float = DEFAULT_MASK_THRESHOLD,
    fill_holes: bool = True,
    authoritative_frame0: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build raw binary and active masks without mutating retained SAM logits.

    The processing order is threshold at SAM resolution, nearest-neighbor resize,
    frame-alpha clipping, then optional enclosed-hole filling. An authoritative
    frame-0 selection replaces only frame 0 for Select Area and Select All.
    """
    if len(raw_logits) != len(frames):
        raise ValueError("SAM logit and frame counts do not match")
    threshold = validate_mask_threshold(threshold)
    from smoke import threshold_sam2_logits, validate_binary_mask

    raw_binary_masks: list[np.ndarray] = []
    active_masks: list[np.ndarray] = []
    for index, (logits, frame) in enumerate(zip(raw_logits, frames, strict=True)):
        if index == 0 and authoritative_frame0 is not None:
            raw_binary = validate_binary_mask(authoritative_frame0, frame.size)
        else:
            raw_binary = threshold_sam2_logits(logits, frame.size, threshold)
        alpha_clipped = clip_mask_to_frame(raw_binary, frame)
        active = apply_mask_cleanup(alpha_clipped, fill_holes)
        raw_binary_masks.append(raw_binary.copy())
        active_masks.append(active)
    return raw_binary_masks, active_masks


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


def make_selection_overlay(
    frame: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 170, 255),
    outline: tuple[int, int, int] = (255, 220, 0),
) -> Image.Image:
    """Overlay a selection and draw a one-pixel outline without altering the source."""
    selected = clip_mask_to_frame(mask, frame)
    output = np.asarray(make_mask_overlay(frame, selected, color=color), dtype=np.uint8).copy()
    padded = np.pad(selected, 1, constant_values=False)
    interior = (
        selected
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = np.logical_and(selected, ~interior)
    output[boundary, :3] = np.asarray(outline, dtype=np.uint8)
    return Image.fromarray(output)


def recolor_frame_sequence(
    frames: Sequence[Image.Image],
    masks: Sequence[np.ndarray],
    target: tuple[int, int, int],
    strength: float = 0.85,
    shadow_protect_amount: float = 1.0,
) -> list[Image.Image]:
    if len(frames) != len(masks):
        raise ValueError("Frame and mask counts do not match")
    output: list[Image.Image] = []
    for frame, mask in zip(frames, masks, strict=True):
        clipped = clip_mask_to_frame(mask, frame)
        mask_image = Image.fromarray(clipped.astype(np.uint8) * 255)
        output.append(
            recolor_masked(
                frame,
                mask_image,
                target=target,
                strength=strength,
                shadow_protect_amount=shadow_protect_amount,
            )
        )
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


class RecolorPreviewState:
    """Own an in-memory recolor candidate until it is accepted or rejected."""

    def __init__(self) -> None:
        self._frames: tuple[Image.Image, ...] = ()
        self._contact_sheet: Image.Image | None = None

    @property
    def has_preview(self) -> bool:
        return self._contact_sheet is not None and bool(self._frames)

    @property
    def frames(self) -> tuple[Image.Image, ...]:
        return self._frames

    @property
    def contact_sheet(self) -> Image.Image | None:
        return self._contact_sheet

    def begin(self, frames: Sequence[Image.Image], contact_sheet: Image.Image) -> None:
        if not frames:
            raise ValueError("At least one recolored frame is required")
        self._frames = tuple(frame.copy() for frame in frames)
        self._contact_sheet = contact_sheet.copy()

    def reject(self) -> None:
        self._frames = ()
        self._contact_sheet = None

    def accept(self, exporter: Callable[[Sequence[Image.Image]], object]) -> object:
        if not self.has_preview:
            raise RuntimeError("No recolor preview is awaiting review")
        result = exporter(self._frames)
        self.reject()
        return result


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
        self.sam_frame0_prediction: np.ndarray | None = None
        self.frame0_logits: np.ndarray | None = None
        self.raw_logits: list[np.ndarray] | None = None
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
        threshold: float = DEFAULT_MASK_THRESHOLD,
    ) -> np.ndarray:
        if not any(click.label == 1 for click in clicks):
            raise ValueError("Add at least one positive click before generating a mask")
        self._ensure_ready(progress)
        from smoke import PromptPoint, add_sam2_prompts_logits, threshold_sam2_logits

        progress("Generating frame 0 mask...")
        points = [PromptPoint(click.x, click.y, click.label) for click in clicks]
        self.frame0_logits = add_sam2_prompts_logits(
            self.predictor,
            self.inference_state,
            points,
            self.frames[0].size,
            self.work_size,
        )
        raw_mask = threshold_sam2_logits(self.frame0_logits, self.frames[0].size, threshold)
        self.sam_frame0_prediction = raw_mask.copy()
        self.frame0_mask = clip_mask_to_frame(raw_mask, self.frames[0])
        self.raw_logits = None
        self.masks = None
        return self.frame0_mask.copy()

    def set_frame0_mask(
        self,
        initial_mask: np.ndarray,
        progress: ProgressCallback = lambda _message: None,
    ) -> np.ndarray:
        from smoke import validate_binary_mask

        validated = validate_binary_mask(initial_mask, self.frames[0].size)
        clipped = clip_mask_to_frame(validated, self.frames[0])
        if not clipped.any():
            raise ValueError("The selected area is empty after alpha clipping")
        self._ensure_ready(progress)
        from smoke import add_sam2_mask_logits, threshold_sam2_logits

        progress("Registering frame 0 selection mask with SAM 2...")
        self.frame0_logits = add_sam2_mask_logits(
            self.predictor,
            self.inference_state,
            clipped,
            self.frames[0].size,
        )
        self.sam_frame0_prediction = threshold_sam2_logits(
            self.frame0_logits,
            self.frames[0].size,
            DEFAULT_MASK_THRESHOLD,
        )
        # Preserve the exact user selection on frame 0. SAM 2 provides tracking
        # for subsequent frames, while alpha clipping remains authoritative here.
        self.frame0_mask = clipped
        self.raw_logits = None
        self.masks = None
        return self.frame0_mask.copy()

    def track_across_frames_logits(
        self,
        progress: ProgressCallback = lambda _message: None,
    ) -> list[np.ndarray]:
        """Run propagation once and retain owned CPU raw logits for every frame."""
        if self.frame0_mask is None or self.predictor is None or self.inference_state is None:
            raise RuntimeError("Generate and review the frame 0 mask before tracking")
        from smoke import propagate_sam2_logits

        progress("Propagating mask across frames...")
        propagated = propagate_sam2_logits(self.predictor, self.inference_state)
        expected = list(range(len(self.frames)))
        if sorted(propagated) != expected:
            raise RuntimeError(f"Expected propagated frames {expected}, got {sorted(propagated)}")
        self.raw_logits = [propagated[index].copy() for index in expected]
        return list(self.raw_logits)

    def track_across_frames(
        self,
        progress: ProgressCallback = lambda _message: None,
    ) -> list[np.ndarray]:
        raw_logits = self.track_across_frames_logits(progress)
        from smoke import threshold_sam2_logits

        self.masks = [
            clip_mask_to_frame(
                threshold_sam2_logits(logits, frame.size, DEFAULT_MASK_THRESHOLD),
                frame,
            )
            for logits, frame in zip(raw_logits, self.frames, strict=True)
        ]
        self.masks[0] = self.frame0_mask.copy()
        return [mask.copy() for mask in self.masks]

    def close(self) -> None:
        self.predictor = None
        self.inference_state = None
        self.device = None
        self.sam_frame0_prediction = None
        self.frame0_logits = None
        self.raw_logits = None
        self.frame0_mask = None
        self.masks = None
        shutil.rmtree(self.temp_dir, ignore_errors=True)


TEAM_COLOR_NAMES = tuple(name.title() for name in DEFAULT_TARGETS) + ("Custom",)
