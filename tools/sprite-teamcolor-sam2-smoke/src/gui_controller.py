"""GUI-independent state and image operations for the sprite team-color MVP."""

from __future__ import annotations

from collections import deque
import math
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
LOGIT_DIAGNOSTIC_THRESHOLDS = (
    -20.0,
    -10.0,
    -5.0,
    -2.0,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
)
LOGIT_DIAGNOSTIC_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
CONTINUOUS_LOGIT_UNIQUE_MIN = 16
PRE_GATE_PREVIEW_MIN_THRESHOLD = -20.0
PRE_GATE_PREVIEW_MAX_THRESHOLD = 20.0
PRE_GATE_PREVIEW_DEFAULT_THRESHOLD = 0.0
PRE_GATE_DIAGNOSTIC_THRESHOLDS = (0.0, -5.0, -10.0, 5.0)
REVERSE_ANCHOR_FRAME_INDEX = 14

# Temporal Pre-Gate continuity policy. Distances are normalized by the previous
# accepted bbox diagonal so the same policy applies to different sprite sizes.
MAX_NORMALIZED_CENTER_SHIFT = 2.5
MIN_AREA_RATIO = 0.20
MAX_AREA_GROWTH_RATIO = 3.0
MAX_BBOX_SCALE_CHANGE = 2.5
PROXIMITY_MARGIN_DIAGONAL_FACTOR = 1.25
MIN_CONTINUITY_SCORE = 0.35
BIDIRECTIONAL_MIN_IOU = 0.05
BIDIRECTIONAL_MAX_NORMALIZED_CENTER_DISTANCE = 0.75
BIDIRECTIONAL_BBOX_CENTER_DISTANCE = 1.5


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


@dataclass(frozen=True)
class SamLogitFrameDiagnostics:
    frame_index: int
    shape: tuple[int, int]
    dtype: str
    minimum: float
    maximum: float
    mean: float
    percentiles: tuple[tuple[int, float], ...]
    unique_value_count: int
    threshold_pixel_counts: tuple[tuple[float, int], ...]
    threshold_zero_compatible: bool
    monotonic_nonincreasing: bool


@dataclass(frozen=True)
class SamLogitDiagnostics:
    frames: tuple[SamLogitFrameDiagnostics, ...]
    all_float32_2d: bool
    has_continuous_values: bool
    threshold_zero_compatible: bool
    monotonic_nonincreasing: bool


@dataclass(frozen=True)
class SamPreGateDecoderCapture:
    decoder_call_index: int
    low_res_multimasks: np.ndarray
    ious: np.ndarray
    object_score_logits: np.ndarray
    best_mask_index: int
    best_iou: float
    object_score_logit: float
    pre_gate_logits: np.ndarray


@dataclass(frozen=True)
class SamPreGateFrameDiagnostics:
    frame_index: int
    capture_status: str
    captures: tuple[SamPreGateDecoderCapture, ...]
    capture_errors: tuple[str, ...]
    post_gate_is_no_obj: bool
    post_gate_min: float
    post_gate_max: float

    @property
    def decoder_call_count(self) -> int:
        return len(self.captures) + len(self.capture_errors)

    @property
    def capture(self) -> SamPreGateDecoderCapture | None:
        return self.captures[0] if self.capture_status == "captured" else None


@dataclass(frozen=True)
class MaskComponent:
    label: int
    area: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    width: int
    height: int
    diagonal: float
    mask: np.ndarray


@dataclass(frozen=True)
class ContinuityEvaluation:
    component_index: int
    center_shift: float
    normalized_center_shift: float
    predicted_center: tuple[float, float]
    predicted_center_distance: float
    normalized_predicted_distance: float
    area_ratio: float
    width_ratio: float
    height_ratio: float
    diagonal_ratio: float
    spatially_proximate: bool
    continuity_score: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class TemporalPassFrameDiagnostics:
    direction: str
    candidate_component_count: int
    selected_component: MaskComponent | None
    evaluation: ContinuityEvaluation | None
    result: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TemporalFilterFrameDiagnostics:
    frame_index: int
    trusted_anchor: bool
    raw_pre_gate_pixels: int
    left: TemporalPassFrameDiagnostics | None
    right: TemporalPassFrameDiagnostics | None
    final_result: str
    final_reasons: tuple[str, ...]


@dataclass(frozen=True)
class TemporalPreGateFilterResult:
    trusted_masks: tuple[np.ndarray, ...]
    filtered_masks: tuple[np.ndarray, ...]
    frames: tuple[TemporalFilterFrameDiagnostics, ...]


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


def derive_reverse_masks_from_sam_logits(
    raw_logits: Sequence[np.ndarray],
    frames: Sequence[Image.Image],
    regular_masks: Sequence[np.ndarray],
    threshold: float = DEFAULT_MASK_THRESHOLD,
    fill_holes: bool = True,
    anchor_frame_index: int = 14,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build Reverse masks while keeping its Regular anchor authoritative."""
    if not (len(raw_logits) == len(frames) == len(regular_masks)):
        raise ValueError("Reverse logits, frames, and Regular mask counts do not match")
    if not 0 <= anchor_frame_index < len(frames):
        raise ValueError("Reverse anchor frame is outside the frame sequence")
    raw_masks, active_masks = derive_masks_from_sam_logits(
        raw_logits,
        frames,
        threshold=threshold,
        fill_holes=fill_holes,
    )
    from smoke import validate_binary_mask

    anchor = validate_binary_mask(regular_masks[anchor_frame_index], frames[anchor_frame_index].size)
    if not anchor.any():
        raise ValueError(
            f"Reverse requires a non-empty Regular mask on frame {anchor_frame_index:03}."
        )
    anchor = clip_mask_to_frame(anchor, frames[anchor_frame_index])
    raw_masks[anchor_frame_index] = anchor.copy()
    active_masks[anchor_frame_index] = anchor.copy()
    return raw_masks, active_masks


def derive_pre_gate_masks(
    diagnostics: Sequence[SamPreGateFrameDiagnostics],
    frames: Sequence[Image.Image],
    threshold: float = PRE_GATE_PREVIEW_DEFAULT_THRESHOLD,
    fill_holes: bool = False,
) -> list[np.ndarray]:
    """Build the exact masks shown by Pre-Gate Diagnostics for source union."""
    if len(diagnostics) != len(frames):
        raise ValueError("Frame and pre-gate diagnostic counts do not match")
    output: list[np.ndarray] = []
    for diagnostic, frame in zip(diagnostics, frames, strict=True):
        preview = pre_gate_preview_mask(diagnostic, frame, threshold, fill_holes)
        output.append(
            np.zeros((frame.height, frame.width), dtype=bool)
            if preview is None
            else preview.copy()
        )
    return output


def _copy_mask_source(
    masks: Sequence[np.ndarray],
    source_name: str,
    expected_shapes: Sequence[tuple[int, int]] | None = None,
) -> list[np.ndarray]:
    if not masks:
        raise ValueError(f"{source_name} masks cannot be empty")
    copied: list[np.ndarray] = []
    for index, mask in enumerate(masks):
        values = np.asarray(mask)
        if values.ndim != 2:
            raise ValueError(f"{source_name} frame {index} mask must be 2D")
        if expected_shapes is not None and values.shape != expected_shapes[index]:
            raise ValueError(
                f"{source_name} frame {index} shape {values.shape} does not match "
                f"Regular shape {expected_shapes[index]}"
            )
        copied.append(values.astype(bool, copy=True))
    return copied


def connected_components_8(mask: np.ndarray) -> tuple[MaskComponent, ...]:
    """Split a binary mask into owned 8-neighbor components."""
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2 or source.shape[0] <= 0 or source.shape[1] <= 0:
        raise ValueError("Connected-component source must be a non-empty 2D mask")
    visited = np.zeros(source.shape, dtype=bool)
    height, width = source.shape
    components: list[MaskComponent] = []
    neighbors = tuple(
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dy == 0 and dx == 0)
    )
    for start_y in range(height):
        for start_x in range(width):
            if not source[start_y, start_x] or visited[start_y, start_x]:
                continue
            pending = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            points: list[tuple[int, int]] = []
            while pending:
                y, x = pending.popleft()
                points.append((y, x))
                for dy, dx in neighbors:
                    next_y, next_x = y + dy, x + dx
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and source[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        pending.append((next_y, next_x))
            component_mask = np.zeros(source.shape, dtype=bool)
            ys = np.fromiter((point[0] for point in points), dtype=np.int32)
            xs = np.fromiter((point[1] for point in points), dtype=np.int32)
            component_mask[ys, xs] = True
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            component_width = x1 - x0 + 1
            component_height = y1 - y0 + 1
            components.append(
                MaskComponent(
                    label=len(components),
                    area=len(points),
                    centroid=(float(xs.mean()), float(ys.mean())),
                    bbox=(x0, y0, x1, y1),
                    width=component_width,
                    height=component_height,
                    diagonal=math.hypot(component_width, component_height),
                    mask=component_mask,
                )
            )
    return tuple(components)


def mask_geometry(mask: np.ndarray, label: int = -1) -> MaskComponent:
    """Measure one possibly-disconnected trusted mask as a single target geometry."""
    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2 or not source.any():
        raise ValueError("Mask geometry requires a non-empty 2D mask")
    ys, xs = np.nonzero(source)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    return MaskComponent(
        label=label,
        area=int(source.sum()),
        centroid=(float(xs.mean()), float(ys.mean())),
        bbox=(x0, y0, x1, y1),
        width=width,
        height=height,
        diagonal=math.hypot(width, height),
        mask=source.copy(),
    )


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _scale_change(ratio: float) -> float:
    return max(ratio, 1.0 / ratio) if ratio > 0 else math.inf


def _bbox_overlaps(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def evaluate_component_continuity(
    accepted_history: Sequence[MaskComponent],
    candidate: MaskComponent,
    component_index: int | None = None,
) -> ContinuityEvaluation:
    """Score one component against accepted geometry and return explicit reasons."""
    if not accepted_history:
        raise ValueError("Continuity evaluation requires an accepted anchor")
    previous = accepted_history[-1]
    if len(accepted_history) >= 2:
        older = accepted_history[-2]
        predicted_center = (
            previous.centroid[0] + previous.centroid[0] - older.centroid[0],
            previous.centroid[1] + previous.centroid[1] - older.centroid[1],
        )
    else:
        predicted_center = previous.centroid

    normalization = max(previous.diagonal, 1.0)
    center_shift = _distance(previous.centroid, candidate.centroid)
    predicted_distance = _distance(predicted_center, candidate.centroid)
    normalized_center_shift = center_shift / normalization
    normalized_predicted_distance = predicted_distance / normalization
    area_ratio = candidate.area / previous.area
    width_ratio = candidate.width / previous.width
    height_ratio = candidate.height / previous.height
    diagonal_ratio = candidate.diagonal / max(previous.diagonal, 1e-6)
    margin = max(1.0, previous.diagonal * PROXIMITY_MARGIN_DIAGONAL_FACTOR)
    x0, y0, x1, y1 = previous.bbox
    expanded_previous = (
        math.floor(x0 - margin),
        math.floor(y0 - margin),
        math.ceil(x1 + margin),
        math.ceil(y1 + margin),
    )
    spatially_proximate = _bbox_overlaps(expanded_previous, candidate.bbox)

    area_change = _scale_change(area_ratio)
    bbox_change = max(
        _scale_change(width_ratio),
        _scale_change(height_ratio),
        _scale_change(diagonal_ratio),
    )
    center_score = max(
        0.0,
        1.0 - normalized_predicted_distance / MAX_NORMALIZED_CENTER_SHIFT,
    )
    movement_score = max(
        0.0,
        1.0 - normalized_center_shift / MAX_NORMALIZED_CENTER_SHIFT,
    )
    area_limit = max(MAX_AREA_GROWTH_RATIO, 1.0 / MIN_AREA_RATIO)
    area_score = max(0.0, 1.0 - math.log(area_change) / math.log(area_limit))
    bbox_score = max(
        0.0,
        1.0 - math.log(bbox_change) / math.log(MAX_BBOX_SCALE_CHANGE),
    )
    continuity_score = (
        0.35 * center_score
        + 0.10 * movement_score
        + 0.25 * area_score
        + 0.20 * bbox_score
        + 0.10 * float(spatially_proximate)
    )

    reasons: list[str] = []
    if normalized_predicted_distance > MAX_NORMALIZED_CENTER_SHIFT:
        reasons.append("REJECT_CENTER_SHIFT")
    if area_ratio > MAX_AREA_GROWTH_RATIO:
        reasons.append("REJECT_AREA_GROWTH")
    elif area_ratio < MIN_AREA_RATIO:
        reasons.append("REJECT_AREA_CHANGE")
    if bbox_change > MAX_BBOX_SCALE_CHANGE:
        reasons.append("REJECT_BBOX_CHANGE")
    if not spatially_proximate:
        reasons.append("REJECT_SPATIAL_PROXIMITY")
    if continuity_score < MIN_CONTINUITY_SCORE:
        reasons.append("REJECT_LOW_SCORE")
    return ContinuityEvaluation(
        component_index=candidate.label if component_index is None else component_index,
        center_shift=center_shift,
        normalized_center_shift=normalized_center_shift,
        predicted_center=predicted_center,
        predicted_center_distance=predicted_distance,
        normalized_predicted_distance=normalized_predicted_distance,
        area_ratio=area_ratio,
        width_ratio=width_ratio,
        height_ratio=height_ratio,
        diagonal_ratio=diagonal_ratio,
        spatially_proximate=spatially_proximate,
        continuity_score=continuity_score,
        accepted=not reasons,
        rejection_reasons=tuple(reasons),
    )


def find_empty_mask_gaps(masks: Sequence[np.ndarray]) -> tuple[tuple[int, int], ...]:
    """Return inclusive spans of consecutive empty trusted masks."""
    gaps: list[tuple[int, int]] = []
    start: int | None = None
    for index, source in enumerate(masks):
        empty = not np.asarray(source, dtype=bool).any()
        if empty and start is None:
            start = index
        elif not empty and start is not None:
            gaps.append((start, index - 1))
            start = None
    if start is not None:
        gaps.append((start, len(masks) - 1))
    return tuple(gaps)


def _directional_gap_pass(
    raw_pre_gate_masks: Sequence[np.ndarray],
    frame_indexes: Sequence[int],
    anchor_mask: np.ndarray | None,
    direction: str,
) -> tuple[dict[int, np.ndarray], dict[int, TemporalPassFrameDiagnostics]]:
    accepted: dict[int, np.ndarray] = {}
    diagnostics: dict[int, TemporalPassFrameDiagnostics] = {}
    if anchor_mask is None:
        for frame_index in frame_indexes:
            component_count = len(connected_components_8(raw_pre_gate_masks[frame_index]))
            diagnostics[frame_index] = TemporalPassFrameDiagnostics(
                direction=direction,
                candidate_component_count=component_count,
                selected_component=None,
                evaluation=None,
                result="NO_ANCHOR",
                reasons=("REJECT_NO_TRUSTED_ANCHOR",),
            )
        return accepted, diagnostics

    history = [mask_geometry(anchor_mask)]
    stopped = False
    for frame_index in frame_indexes:
        components = connected_components_8(raw_pre_gate_masks[frame_index])
        if stopped:
            diagnostics[frame_index] = TemporalPassFrameDiagnostics(
                direction=direction,
                candidate_component_count=len(components),
                selected_component=None,
                evaluation=None,
                result="STOPPED",
                reasons=("STOPPED_AFTER_PREVIOUS_FAILURE",),
            )
            continue
        if not components:
            diagnostics[frame_index] = TemporalPassFrameDiagnostics(
                direction=direction,
                candidate_component_count=0,
                selected_component=None,
                evaluation=None,
                result="REJECT",
                reasons=("REJECT_NO_COMPONENT",),
            )
            stopped = True
            continue

        evaluations = [
            evaluate_component_continuity(history, component, component.label)
            for component in components
        ]
        acceptable = [evaluation for evaluation in evaluations if evaluation.accepted]
        selected_evaluation = max(
            acceptable if acceptable else evaluations,
            key=lambda evaluation: evaluation.continuity_score,
        )
        selected = components[selected_evaluation.component_index]
        if selected_evaluation.accepted:
            accepted[frame_index] = selected.mask.copy()
            history.append(selected)
            result = "ACCEPT"
            reasons: tuple[str, ...] = ()
        else:
            result = "REJECT"
            reasons = selected_evaluation.rejection_reasons
            stopped = True
        diagnostics[frame_index] = TemporalPassFrameDiagnostics(
            direction=direction,
            candidate_component_count=len(components),
            selected_component=selected,
            evaluation=selected_evaluation,
            result=result,
            reasons=reasons,
        )
    return accepted, diagnostics


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return intersection / union if union else 1.0


def bidirectional_components_are_consistent(left: np.ndarray, right: np.ndarray) -> bool:
    left_geometry = mask_geometry(left)
    right_geometry = mask_geometry(right)
    normalization = max(left_geometry.diagonal, right_geometry.diagonal, 1.0)
    normalized_distance = _distance(left_geometry.centroid, right_geometry.centroid) / normalization
    bbox_overlap = _bbox_overlaps(left_geometry.bbox, right_geometry.bbox)
    return bool(
        _mask_iou(left, right) >= BIDIRECTIONAL_MIN_IOU
        or normalized_distance <= BIDIRECTIONAL_MAX_NORMALIZED_CENTER_DISTANCE
        or (bbox_overlap and normalized_distance <= BIDIRECTIONAL_BBOX_CENTER_DISTANCE)
    )


def temporal_filter_pre_gate_masks(
    regular_masks: Sequence[np.ndarray],
    raw_pre_gate_masks: Sequence[np.ndarray],
    reverse_masks: Sequence[np.ndarray] | None = None,
) -> TemporalPreGateFilterResult:
    """Filter Pre-Gate components across trusted-mask gaps in both directions."""
    regular = _copy_mask_source(regular_masks, "Regular")
    shapes = [mask.shape for mask in regular]
    if len(raw_pre_gate_masks) != len(regular):
        raise ValueError("Raw Pre-Gate and Regular mask counts do not match")
    raw = _copy_mask_source(raw_pre_gate_masks, "Raw Pre-Gate", shapes)
    if reverse_masks is None:
        reverse = [np.zeros(shape, dtype=bool) for shape in shapes]
    else:
        if len(reverse_masks) != len(regular):
            raise ValueError("Reverse and Regular mask counts do not match")
        reverse = _copy_mask_source(reverse_masks, "Reverse", shapes)
    trusted = [np.logical_or(base, backward) for base, backward in zip(regular, reverse, strict=True)]
    filtered = [np.zeros(shape, dtype=bool) for shape in shapes]
    left_diagnostics: dict[int, TemporalPassFrameDiagnostics] = {}
    right_diagnostics: dict[int, TemporalPassFrameDiagnostics] = {}

    for gap_start, gap_end in find_empty_mask_gaps(trusted):
        left_anchor = trusted[gap_start - 1] if gap_start > 0 else None
        right_anchor = trusted[gap_end + 1] if gap_end + 1 < len(trusted) else None
        left_accepted, left_frames = _directional_gap_pass(
            raw,
            tuple(range(gap_start, gap_end + 1)),
            left_anchor,
            "left_to_right",
        )
        right_accepted, right_frames = _directional_gap_pass(
            raw,
            tuple(range(gap_end, gap_start - 1, -1)),
            right_anchor,
            "right_to_left",
        )
        left_diagnostics.update(left_frames)
        right_diagnostics.update(right_frames)
        for frame_index in range(gap_start, gap_end + 1):
            left_mask = left_accepted.get(frame_index)
            right_mask = right_accepted.get(frame_index)
            if left_mask is not None and right_mask is not None:
                if bidirectional_components_are_consistent(left_mask, right_mask):
                    filtered[frame_index] = np.logical_or(left_mask, right_mask)
                continue
            if left_mask is not None:
                filtered[frame_index] = left_mask.copy()
            elif right_mask is not None:
                filtered[frame_index] = right_mask.copy()

    frame_diagnostics: list[TemporalFilterFrameDiagnostics] = []
    for frame_index, (trusted_mask, raw_mask, output) in enumerate(
        zip(trusted, raw, filtered, strict=True)
    ):
        if trusted_mask.any():
            final_result = "TRUSTED_FRAME"
            final_reasons = ("FILTER_NOT_NEEDED_ON_TRUSTED_FRAME",)
            left = right = None
        else:
            left = left_diagnostics.get(frame_index)
            right = right_diagnostics.get(frame_index)
            left_accepted = left is not None and left.result == "ACCEPT"
            right_accepted = right is not None and right.result == "ACCEPT"
            if left_accepted and right_accepted:
                if output.any():
                    final_result = "ACCEPT_BOTH"
                    final_reasons = ()
                else:
                    final_result = "AMBIGUOUS"
                    final_reasons = ("REJECT_AMBIGUOUS_DIRECTIONS",)
            elif left_accepted:
                final_result = "ACCEPT_LEFT"
                final_reasons = ()
            elif right_accepted:
                final_result = "ACCEPT_RIGHT"
                final_reasons = ()
            else:
                final_result = "REJECT"
                final_reasons = tuple(
                    dict.fromkeys(
                        reason
                        for diagnostic in (left, right)
                        if diagnostic is not None
                        for reason in diagnostic.reasons
                    )
                ) or ("REJECT_NO_COMPONENT",)
        frame_diagnostics.append(
            TemporalFilterFrameDiagnostics(
                frame_index=frame_index,
                trusted_anchor=bool(trusted_mask.any()),
                raw_pre_gate_pixels=int(raw_mask.sum()),
                left=left,
                right=right,
                final_result=final_result,
                final_reasons=final_reasons,
            )
        )
    return TemporalPreGateFilterResult(
        trusted_masks=tuple(mask.copy() for mask in trusted),
        filtered_masks=tuple(mask.copy() for mask in filtered),
        frames=tuple(frame_diagnostics),
    )


def format_temporal_filter_diagnostics(result: TemporalPreGateFilterResult) -> str:
    def describe_pass(diagnostic: TemporalPassFrameDiagnostics | None, indent: str) -> list[str]:
        if diagnostic is None:
            return [f"{indent}not evaluated (trusted frame)"]
        lines = [
            f"{indent}candidate components: {diagnostic.candidate_component_count}",
            f"{indent}result: {diagnostic.result}",
        ]
        if diagnostic.selected_component is not None:
            component = diagnostic.selected_component
            lines.extend(
                [
                    f"{indent}selected component: {component.label}",
                    f"{indent}area: {component.area}",
                    f"{indent}centroid: ({component.centroid[0]:.3f}, {component.centroid[1]:.3f})",
                    f"{indent}bbox: {component.bbox}",
                    f"{indent}bbox size/diagonal: {component.width} x {component.height} / {component.diagonal:.3f}",
                ]
            )
        if diagnostic.evaluation is not None:
            evaluation = diagnostic.evaluation
            lines.extend(
                [
                    f"{indent}center shift: {evaluation.center_shift:.4f} "
                    f"(normalized {evaluation.normalized_center_shift:.4f})",
                    f"{indent}predicted center: ({evaluation.predicted_center[0]:.3f}, "
                    f"{evaluation.predicted_center[1]:.3f})",
                    f"{indent}predicted center distance: {evaluation.predicted_center_distance:.4f} "
                    f"(normalized {evaluation.normalized_predicted_distance:.4f})",
                    f"{indent}area ratio: {evaluation.area_ratio:.4f}",
                    f"{indent}bbox width/height/diagonal ratios: {evaluation.width_ratio:.4f} / "
                    f"{evaluation.height_ratio:.4f} / {evaluation.diagonal_ratio:.4f}",
                    f"{indent}spatial proximity: {'YES' if evaluation.spatially_proximate else 'NO'}",
                    f"{indent}continuity score: {evaluation.continuity_score:.4f}",
                ]
            )
        if diagnostic.reasons:
            lines.append(f"{indent}reasons: {', '.join(diagnostic.reasons)}")
        return lines

    lines = [
        "Temporal Pre-Gate Filter diagnostics",
        "Geometry-only CPU filter; no SAM inference or appearance features are used.",
        "Trusted mask = Regular OR cached Reverse. Trusted frames emit no Filtered Pre-Gate pixels.",
        "",
    ]
    for frame in result.frames:
        lines.extend(
            [
                f"frame_{frame.frame_index:03}",
                f"  trusted anchor: {'YES' if frame.trusted_anchor else 'NO'}",
                f"  raw pre-gate pixels: {frame.raw_pre_gate_pixels}",
                "  left pass:",
                *describe_pass(frame.left, "    "),
                "  right pass:",
                *describe_pass(frame.right, "    "),
                f"  final: {frame.final_result}",
            ]
        )
        if frame.final_reasons:
            lines.append(f"  final reasons: {', '.join(frame.final_reasons)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def combine_mask_sources(
    regular_masks: Sequence[np.ndarray],
    pre_gate_masks: Sequence[np.ndarray] | None = None,
    reverse_masks: Sequence[np.ndarray] | None = None,
    filtered_pre_gate_masks: Sequence[np.ndarray] | None = None,
    *,
    include_pre_gate: bool = False,
    include_reverse: bool = False,
    use_filtered_pre_gate: bool = False,
) -> list[np.ndarray]:
    """Rebuild a non-destructive Regular-first OR union from selected sources."""
    regular = _copy_mask_source(regular_masks, "Regular")
    shapes = [mask.shape for mask in regular]
    selected: list[list[np.ndarray]] = []
    selected_pre_gate = filtered_pre_gate_masks if use_filtered_pre_gate else pre_gate_masks
    pre_gate_name = "Filtered Pre-Gate" if use_filtered_pre_gate else "Raw Pre-Gate"
    for enabled, masks, name in (
        (include_pre_gate, selected_pre_gate, pre_gate_name),
        (include_reverse, reverse_masks, "Reverse"),
    ):
        if not enabled:
            continue
        if masks is None:
            raise ValueError(f"{name} masks are not available")
        if len(masks) != len(regular):
            raise ValueError(f"{name} and Regular mask counts do not match")
        selected.append(_copy_mask_source(masks, name, shapes))

    combined = [mask.copy() for mask in regular]
    for source in selected:
        for index, mask in enumerate(source):
            combined[index] = np.logical_or(combined[index], mask)
    return combined


class MaskSourceState:
    """Keep Regular, optional sources, preview union, and adopted masks separate."""

    def __init__(self, regular_masks: Sequence[np.ndarray]) -> None:
        self.regular_masks = _copy_mask_source(regular_masks, "Regular")
        self.pre_gate_masks: list[np.ndarray] | None = None
        self.filtered_pre_gate_masks: list[np.ndarray] | None = None
        self.reverse_masks: list[np.ndarray] | None = None
        self.include_pre_gate = False
        self.include_reverse = False
        self.use_filtered_pre_gate = False
        self.preview_masks = combine_mask_sources(self.regular_masks)
        self.active_masks = [mask.copy() for mask in self.preview_masks]
        self.preview_dirty = False

    def _replace_source(self, masks: Sequence[np.ndarray], name: str) -> list[np.ndarray]:
        if len(masks) != len(self.regular_masks):
            raise ValueError(f"{name} and Regular mask counts do not match")
        return _copy_mask_source(masks, name, [mask.shape for mask in self.regular_masks])

    def set_regular_masks(self, masks: Sequence[np.ndarray]) -> None:
        replacement = _copy_mask_source(masks, "Regular")
        if len(replacement) != len(self.regular_masks) or any(
            actual.shape != previous.shape
            for actual, previous in zip(replacement, self.regular_masks, strict=True)
        ):
            raise ValueError("Updated Regular masks do not match the tracked frame layout")
        self.regular_masks = replacement
        self.rebuild_preview()

    def set_pre_gate_masks(self, masks: Sequence[np.ndarray]) -> None:
        self.pre_gate_masks = self._replace_source(masks, "Pre-Gate")
        self.rebuild_preview()

    def set_filtered_pre_gate_masks(self, masks: Sequence[np.ndarray]) -> None:
        self.filtered_pre_gate_masks = self._replace_source(masks, "Filtered Pre-Gate")
        self.rebuild_preview()

    def set_reverse_masks(self, masks: Sequence[np.ndarray]) -> None:
        self.reverse_masks = self._replace_source(masks, "Reverse")
        self.rebuild_preview()

    def set_enabled(self, *, pre_gate: bool | None = None, reverse: bool | None = None) -> None:
        if pre_gate is not None:
            if pre_gate and self.pre_gate_masks is None:
                raise ValueError("Pre-Gate masks are not available")
            self.include_pre_gate = bool(pre_gate)
        if reverse is not None:
            if reverse and self.reverse_masks is None:
                raise ValueError("Reverse masks are not available")
            self.include_reverse = bool(reverse)
        self.rebuild_preview()

    def set_pre_gate_filter_enabled(self, enabled: bool) -> None:
        if enabled and self.filtered_pre_gate_masks is None:
            raise ValueError("Filtered Pre-Gate masks are not available")
        self.use_filtered_pre_gate = bool(enabled)
        self.rebuild_preview()

    def rebuild_preview(self) -> list[np.ndarray]:
        self.preview_masks = combine_mask_sources(
            self.regular_masks,
            self.pre_gate_masks,
            self.reverse_masks,
            self.filtered_pre_gate_masks,
            include_pre_gate=self.include_pre_gate,
            include_reverse=self.include_reverse,
            use_filtered_pre_gate=self.use_filtered_pre_gate,
        )
        self.preview_dirty = any(
            not np.array_equal(preview, active)
            for preview, active in zip(self.preview_masks, self.active_masks, strict=True)
        )
        return [mask.copy() for mask in self.preview_masks]

    def adopt_preview(self) -> list[np.ndarray]:
        self.active_masks = [mask.copy() for mask in self.preview_masks]
        self.preview_dirty = False
        return [mask.copy() for mask in self.active_masks]

    @property
    def source_label(self) -> str:
        sources = ["Regular"]
        if self.include_pre_gate:
            mode = "Auto-filtered" if self.use_filtered_pre_gate else "Raw"
            sources.append(f"Pre-Gate ({mode})")
        if self.include_reverse:
            sources.append("Reverse")
        return " + ".join(sources)


def diagnose_sam_logits(
    raw_logits: Sequence[np.ndarray],
    frames: Sequence[Image.Image],
    thresholds: Sequence[float] = LOGIT_DIAGNOSTIC_THRESHOLDS,
) -> SamLogitDiagnostics:
    """Measure retained logits and alpha-clipped threshold masks without cleanup."""
    if len(raw_logits) != len(frames):
        raise ValueError("SAM logit and frame counts do not match")
    if not raw_logits:
        raise ValueError("At least one SAM logit frame is required")
    normalized_thresholds = tuple(float(value) for value in thresholds)
    if not normalized_thresholds:
        raise ValueError("At least one diagnostic threshold is required")
    if any(left >= right for left, right in zip(normalized_thresholds, normalized_thresholds[1:])):
        raise ValueError("Diagnostic thresholds must be strictly increasing")

    from smoke import threshold_sam2_logits

    frame_reports: list[SamLogitFrameDiagnostics] = []
    for frame_index, (source, frame) in enumerate(zip(raw_logits, frames, strict=True)):
        values = np.asarray(source)
        if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] <= 0:
            raise ValueError(
                f"Frame {frame_index} raw logits must be a non-empty 2D array, got {values.shape}"
            )
        numeric_values = values.astype(np.float32, copy=False)
        percentile_values = np.percentile(numeric_values, LOGIT_DIAGNOSTIC_PERCENTILES)
        counts: list[tuple[float, int]] = []
        for threshold in normalized_thresholds:
            resized = threshold_sam2_logits(values, frame.size, threshold)
            counts.append((threshold, int(clip_mask_to_frame(resized, frame).sum())))

        direct_zero_work_mask = values > 0.0
        direct_zero_image = Image.fromarray(direct_zero_work_mask.astype(np.uint8) * 255)
        direct_zero = np.asarray(
            direct_zero_image.resize(frame.size, Image.Resampling.NEAREST)
        ) > 0
        direct_zero = clip_mask_to_frame(direct_zero, frame)
        helper_zero = clip_mask_to_frame(
            threshold_sam2_logits(values, frame.size, 0.0),
            frame,
        )
        pixel_counts = [count for _threshold, count in counts]
        frame_reports.append(
            SamLogitFrameDiagnostics(
                frame_index=frame_index,
                shape=(int(values.shape[0]), int(values.shape[1])),
                dtype=str(values.dtype),
                minimum=float(np.min(numeric_values)),
                maximum=float(np.max(numeric_values)),
                mean=float(np.mean(numeric_values)),
                percentiles=tuple(
                    (percentile, float(value))
                    for percentile, value in zip(
                        LOGIT_DIAGNOSTIC_PERCENTILES,
                        percentile_values,
                        strict=True,
                    )
                ),
                unique_value_count=int(np.unique(values).size),
                threshold_pixel_counts=tuple(counts),
                threshold_zero_compatible=bool(np.array_equal(direct_zero, helper_zero)),
                monotonic_nonincreasing=all(
                    left >= right for left, right in zip(pixel_counts, pixel_counts[1:])
                ),
            )
        )

    reports = tuple(frame_reports)
    return SamLogitDiagnostics(
        frames=reports,
        all_float32_2d=all(report.dtype == "float32" for report in reports),
        has_continuous_values=any(
            report.unique_value_count > CONTINUOUS_LOGIT_UNIQUE_MIN for report in reports
        ),
        threshold_zero_compatible=all(report.threshold_zero_compatible for report in reports),
        monotonic_nonincreasing=all(report.monotonic_nonincreasing for report in reports),
    )


def format_sam_logit_diagnostics(diagnostics: SamLogitDiagnostics) -> str:
    """Render diagnostics as plain text suitable for selection or clipboard copy."""
    def result(value: bool) -> str:
        return "PASS" if value else "FAIL"

    def number(value: float) -> str:
        return f"{value:.7g}"

    def threshold_label(value: float) -> str:
        if value == 0:
            return "0"
        return f"{value:+g}"

    max_unique = max(report.unique_value_count for report in diagnostics.frames)
    lines = [
        "SAM raw logit diagnostics",
        "Diagnostic mask counts: threshold -> nearest resize -> alpha clip (Hole Fill excluded)",
        "",
        f"2D float32 retained logits: {result(diagnostics.all_float32_2d)}",
        (
            f"Continuous-value check (>{CONTINUOUS_LOGIT_UNIQUE_MIN} unique in at least one frame): "
            f"{result(diagnostics.has_continuous_values)} (max unique: {max_unique})"
        ),
        f"Threshold 0.0 compatibility: {result(diagnostics.threshold_zero_compatible)}",
        f"Threshold-count monotonicity: {result(diagnostics.monotonic_nonincreasing)}",
        "",
    ]
    for report in diagnostics.frames:
        lines.extend(
            [
                f"frame_{report.frame_index:03}",
                f"  shape: {report.shape}",
                f"  dtype: {report.dtype}",
                f"  min: {number(report.minimum)}",
                f"  max: {number(report.maximum)}",
                f"  mean: {number(report.mean)}",
                f"  unique values: {report.unique_value_count}",
            ]
        )
        for percentile, value in report.percentiles:
            lines.append(f"  p{percentile:02}: {number(value)}")
        lines.append(f"  threshold 0.0 compatibility: {result(report.threshold_zero_compatible)}")
        lines.append(f"  threshold-count monotonicity: {result(report.monotonic_nonincreasing)}")
        for threshold, count in report.threshold_pixel_counts:
            lines.append(f"  threshold {threshold_label(threshold)}: {count} px")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class SamPreGateHookCollector:
    """Observe SAM mask-decoder returns without replacing or mutating them."""

    def __init__(self, image_size: int) -> None:
        if image_size <= 0:
            raise ValueError("SAM image size must be positive")
        self.image_size = int(image_size)
        self._next_call_index = 0
        self._captures: list[SamPreGateDecoderCapture] = []
        self._errors: list[str] = []

    def hook(self, _module: object, _inputs: object, output: object) -> None:
        call_index = self._next_call_index
        self._next_call_index += 1
        try:
            import torch
            import torch.nn.functional as functional

            if not isinstance(output, (tuple, list)) or len(output) < 4:
                raise ValueError("SAM mask decoder output must contain four tensors")
            low_res_source, iou_source, _sam_tokens, object_score_source = output[:4]
            if not all(torch.is_tensor(value) for value in (low_res_source, iou_source, object_score_source)):
                raise ValueError("SAM mask decoder diagnostics require tensor outputs")
            low_res = low_res_source.detach().float().cpu().clone()
            ious = iou_source.detach().float().cpu().clone()
            object_scores = object_score_source.detach().float().cpu().clone()
            if low_res.ndim != 4 or ious.ndim != 2:
                raise ValueError(
                    f"Unexpected decoder shapes: masks {tuple(low_res.shape)}, ious {tuple(ious.shape)}"
                )
            if low_res.shape[0] != 1 or ious.shape[0] != 1 or object_scores.shape[0] != 1:
                raise ValueError("Pre-gate diagnostics currently require decoder batch size 1")
            if low_res.shape[1] != ious.shape[1]:
                raise ValueError("Mask candidate and IoU counts do not match")
            best_index = 0 if low_res.shape[1] == 1 else int(torch.argmax(ious[0]).item())
            best_low_res = low_res[0, best_index].unsqueeze(0).unsqueeze(0)
            pre_gate = functional.interpolate(
                best_low_res,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            self._captures.append(
                SamPreGateDecoderCapture(
                    decoder_call_index=call_index,
                    low_res_multimasks=low_res.numpy().copy(),
                    ious=ious.numpy().copy(),
                    object_score_logits=object_scores.numpy().copy(),
                    best_mask_index=best_index,
                    best_iou=float(ious[0, best_index].item()),
                    object_score_logit=float(object_scores.reshape(1, -1)[0, 0].item()),
                    pre_gate_logits=pre_gate.numpy().astype(np.float32, copy=True),
                )
            )
        except Exception as error:
            self._errors.append(f"decoder call {call_index}: {error}")
        return None

    def drain(self) -> tuple[tuple[SamPreGateDecoderCapture, ...], tuple[str, ...]]:
        captures = tuple(self._captures)
        errors = tuple(self._errors)
        self._captures.clear()
        self._errors.clear()
        return captures, errors

    def clear(self) -> None:
        self._captures.clear()
        self._errors.clear()


def map_pre_gate_frame_diagnostics(
    frame_index: int,
    captures: Sequence[SamPreGateDecoderCapture],
    capture_errors: Sequence[str],
    post_gate_logits: np.ndarray,
) -> SamPreGateFrameDiagnostics:
    """Associate captures observed since the previous predictor yield with one frame."""
    capture_tuple = tuple(captures)
    error_tuple = tuple(capture_errors)
    call_count = len(capture_tuple) + len(error_tuple)
    if call_count == 0:
        status = "no_capture"
    elif call_count == 1 and len(capture_tuple) == 1:
        status = "captured"
    elif call_count >= 2:
        status = "ambiguous_multiple_decoder_calls"
    else:
        status = "capture_error"
    post_gate = np.asarray(post_gate_logits)
    return SamPreGateFrameDiagnostics(
        frame_index=int(frame_index),
        capture_status=status,
        captures=capture_tuple,
        capture_errors=error_tuple,
        post_gate_is_no_obj=bool(post_gate.size and np.all(post_gate == -1024.0)),
        post_gate_min=float(np.min(post_gate)),
        post_gate_max=float(np.max(post_gate)),
    )


def propagate_sam2_logits_with_pre_gate(
    predictor: object,
    inference_state: object,
) -> tuple[dict[int, np.ndarray], list[SamPreGateFrameDiagnostics]]:
    """Run standard propagation while observing pre-gate decoder outputs."""
    decoder = getattr(predictor, "sam_mask_decoder", None)
    if decoder is None or not hasattr(decoder, "register_forward_hook"):
        raise RuntimeError("SAM predictor does not expose a hookable sam_mask_decoder")
    image_size = int(getattr(predictor, "image_size", 0))
    collector = SamPreGateHookCollector(image_size)
    handle = decoder.register_forward_hook(collector.hook)
    propagated: dict[int, np.ndarray] = {}
    diagnostics: list[SamPreGateFrameDiagnostics] = []
    try:
        from smoke import validate_sam2_logits

        for frame_index, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
            captures, errors = collector.drain()
            object_id_list = [int(object_id) for object_id in object_ids]
            object_index = object_id_list.index(1)
            post_gate = validate_sam2_logits(mask_logits[object_index])
            normalized_frame_index = int(frame_index)
            propagated[normalized_frame_index] = post_gate
            diagnostics.append(
                map_pre_gate_frame_diagnostics(
                    normalized_frame_index,
                    captures,
                    errors,
                    post_gate,
                )
            )
    finally:
        handle.remove()
        collector.clear()
    return propagated, diagnostics


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


def pre_gate_preview_mask(
    diagnostics: SamPreGateFrameDiagnostics,
    frame: Image.Image,
    threshold: float = PRE_GATE_PREVIEW_DEFAULT_THRESHOLD,
    fill_holes: bool = False,
) -> np.ndarray | None:
    """Derive a diagnostic-only mask without touching standard tracked masks."""
    threshold = float(threshold)
    if not PRE_GATE_PREVIEW_MIN_THRESHOLD <= threshold <= PRE_GATE_PREVIEW_MAX_THRESHOLD:
        raise ValueError(
            "Pre-gate preview threshold must be between "
            f"{PRE_GATE_PREVIEW_MIN_THRESHOLD:g} and {PRE_GATE_PREVIEW_MAX_THRESHOLD:g}"
        )
    capture = diagnostics.capture
    if capture is None:
        return None
    from smoke import threshold_sam2_logits

    resized = threshold_sam2_logits(capture.pre_gate_logits, frame.size, threshold)
    alpha_clipped = clip_mask_to_frame(resized, frame)
    return apply_mask_cleanup(alpha_clipped, fill_holes)


def make_pre_gate_contact_sheet(
    frames: Sequence[Image.Image],
    diagnostics: Sequence[SamPreGateFrameDiagnostics],
    threshold: float = PRE_GATE_PREVIEW_DEFAULT_THRESHOLD,
    fill_holes: bool = False,
    columns: int = 5,
) -> Image.Image:
    if len(frames) != len(diagnostics):
        raise ValueError("Frame and pre-gate diagnostic counts do not match")
    overlays: list[Image.Image] = []
    labels: list[str] = []
    for frame, diagnostic in zip(frames, diagnostics, strict=True):
        mask = pre_gate_preview_mask(diagnostic, frame, threshold, fill_holes)
        if mask is not None:
            overlay = make_mask_overlay(frame, mask, color=(255, 170, 0), opacity=0.6)
        else:
            overlay = frame.convert("RGBA").copy()
            draw = ImageDraw.Draw(overlay)
            message = (
                "NO CAPTURE"
                if diagnostic.capture_status == "no_capture"
                else "AMBIGUOUS"
                if diagnostic.capture_status == "ambiguous_multiple_decoder_calls"
                else "CAPTURE ERROR"
            )
            draw.rectangle((0, 0, overlay.width - 1, 13), fill=(32, 32, 32, 220))
            draw.text((2, 2), message, fill=(255, 220, 0, 255))
        overlays.append(overlay)
        labels.append(f"frame_{diagnostic.frame_index:03}")
    return make_contact_sheet(overlays, labels, columns=columns)


def make_temporal_filter_contact_sheet(
    frames: Sequence[Image.Image],
    result: TemporalPreGateFilterResult,
    columns: int = 5,
) -> Image.Image:
    """Show only Filtered Pre-Gate pixels; trusted frames intentionally stay clear."""
    if len(frames) != len(result.filtered_masks):
        raise ValueError("Frame and Temporal Filter result counts do not match")
    overlays = [
        make_mask_overlay(frame, mask, color=(80, 220, 120), opacity=0.65)
        for frame, mask in zip(frames, result.filtered_masks, strict=True)
    ]
    labels = [f"frame_{index:03}" for index in range(len(frames))]
    return make_contact_sheet(overlays, labels, columns=columns)


def format_pre_gate_diagnostics(
    frames: Sequence[Image.Image],
    diagnostics: Sequence[SamPreGateFrameDiagnostics],
) -> str:
    """Render captured pre-gate candidates and object scores as copyable text."""
    if len(frames) != len(diagnostics):
        raise ValueError("Frame and pre-gate diagnostic counts do not match")

    def number(value: float) -> str:
        return f"{value:.7g}"

    from smoke import threshold_sam2_logits

    lines = [
        "SAM pre-gate mask candidate diagnostics",
        "Observation only: standard post-gate masks, tracking, Recolor, and Export are unchanged.",
        "Preview/count pipeline: threshold -> nearest resize -> alpha clip (Hole Fill excluded).",
        "",
    ]
    absent_with_candidate: list[int] = []
    for frame, diagnostic in zip(frames, diagnostics, strict=True):
        lines.extend(
            [
                f"frame_{diagnostic.frame_index:03}",
                f"  capture status: {diagnostic.capture_status}",
                f"  decoder call count: {diagnostic.decoder_call_count}",
                f"  post-gate NO_OBJ_SCORE: {'YES' if diagnostic.post_gate_is_no_obj else 'NO'}",
                f"  post-gate min/max: {number(diagnostic.post_gate_min)} / {number(diagnostic.post_gate_max)}",
            ]
        )
        if diagnostic.capture_errors:
            for error in diagnostic.capture_errors:
                lines.append(f"  capture error: {error}")
        if not diagnostic.captures:
            lines.append("  pre-gate candidate: unavailable / conditioning frame")
            lines.append("")
            continue
        for capture_position, capture in enumerate(diagnostic.captures):
            prefix = "  " if len(diagnostic.captures) == 1 else f"  capture {capture_position}: "
            values = capture.pre_gate_logits
            lines.extend(
                [
                    f"{prefix}decoder call index: {capture.decoder_call_index}",
                    f"{prefix}object score logit: {number(capture.object_score_logit)}",
                    f"{prefix}object absent gate: {'YES' if capture.object_score_logit <= 0 else 'NO'}",
                    f"{prefix}best mask index: {capture.best_mask_index}",
                    f"{prefix}best IoU estimate: {number(capture.best_iou)}",
                    f"{prefix}pre-gate shape: {values.shape}",
                    f"{prefix}min: {number(float(np.min(values)))}",
                    f"{prefix}max: {number(float(np.max(values)))}",
                    f"{prefix}mean: {number(float(np.mean(values)))}",
                    f"{prefix}unique values: {int(np.unique(values).size)}",
                ]
            )
            counts: dict[float, int] = {}
            for threshold in PRE_GATE_DIAGNOSTIC_THRESHOLDS:
                resized = threshold_sam2_logits(capture.pre_gate_logits, frame.size, threshold)
                count = int(clip_mask_to_frame(resized, frame).sum())
                counts[threshold] = count
                lines.append(f"{prefix}threshold {threshold:+g}: {count} px")
            if (
                diagnostic.capture_status == "captured"
                and capture.object_score_logit <= 0
                and counts[0.0] > 0
            ):
                absent_with_candidate.append(diagnostic.frame_index)
                lines.append(
                    f"{prefix}summary: object absent gate with {counts[0.0]} pre-gate pixels at threshold 0"
                )
        lines.append("")
    frame_list = ", ".join(f"frame_{index:03}" for index in absent_with_candidate) or "none"
    lines.extend(
        [
            "Summary",
            f"  object absent gate with non-empty threshold-0 pre-gate candidate: {frame_list}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


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
        self.sam_frames_dir: Path | None = None
        self.predictor: object | None = None
        self.inference_state: object | None = None
        self.device: object | None = None
        self.frame0_mask: np.ndarray | None = None
        self.sam_frame0_prediction: np.ndarray | None = None
        self.frame0_logits: np.ndarray | None = None
        self.raw_logits: list[np.ndarray] | None = None
        self.reverse_raw_logits: list[np.ndarray] | None = None
        self.pre_gate_diagnostics: list[SamPreGateFrameDiagnostics] | None = None
        self.masks: list[np.ndarray] | None = None

    def _ensure_ready(self, progress: ProgressCallback) -> None:
        if self.predictor is None:
            try:
                from smoke import create_sam2_predictor, prepare_frame_sequence, resolve_device
            except ImportError as error:
                raise RuntimeError("Could not import the SAM 2 smoke integration") from error

            progress("Preparing SAM 2 frames...")
            self.sam_frames_dir = prepare_frame_sequence(
                self.frames,
                self.temp_dir,
                self.work_size,
            )
            progress("Loading SAM 2 model...")
            self.device = resolve_device(self.device_name)
            self.predictor = create_sam2_predictor(
                self.model_config,
                self.checkpoint_path,
                self.device,
            )

        if self.inference_state is None:
            if self.sam_frames_dir is None or self.device is None:
                raise RuntimeError("SAM 2 prepared frames and device are unavailable")
            try:
                from smoke import initialize_sam2_state
            except ImportError as error:
                raise RuntimeError("Could not import the SAM 2 smoke integration") from error
            progress("Initializing fresh SAM 2 video state...")
            self.inference_state = initialize_sam2_state(
                self.predictor,
                self.sam_frames_dir,
                self.device,
            )

    def reset_tracking_state(self) -> None:
        """Discard target-specific state while retaining model and prepared frames."""
        previous_state = self.inference_state
        try:
            reset_state = getattr(self.predictor, "reset_state", None)
            if previous_state is not None and callable(reset_state):
                reset_state(previous_state)
        finally:
            self.inference_state = None
            self.frame0_mask = None
            self.sam_frame0_prediction = None
            self.frame0_logits = None
            self.raw_logits = None
            self.reverse_raw_logits = None
            self.pre_gate_diagnostics = None
            self.masks = None

    def generate_frame0_mask(
        self,
        clicks: Sequence[PromptClick],
        progress: ProgressCallback = lambda _message: None,
        threshold: float = DEFAULT_MASK_THRESHOLD,
    ) -> np.ndarray:
        if not any(click.label == 1 for click in clicks):
            raise ValueError("Add at least one positive click before generating a mask")
        self.reset_tracking_state()
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
        self.reverse_raw_logits = None
        self.pre_gate_diagnostics = None
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
        self.reset_tracking_state()
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
        self.reverse_raw_logits = None
        self.pre_gate_diagnostics = None
        self.masks = None
        return self.frame0_mask.copy()

    def track_across_frames_logits(
        self,
        progress: ProgressCallback = lambda _message: None,
    ) -> list[np.ndarray]:
        """Run propagation once and retain owned CPU raw logits for every frame."""
        if self.frame0_mask is None or self.predictor is None or self.inference_state is None:
            raise RuntimeError("Generate and review the frame 0 mask before tracking")
        progress("Propagating mask across frames...")
        propagated, diagnostics = propagate_sam2_logits_with_pre_gate(
            self.predictor,
            self.inference_state,
        )
        expected = list(range(len(self.frames)))
        if sorted(propagated) != expected:
            raise RuntimeError(f"Expected propagated frames {expected}, got {sorted(propagated)}")
        if [diagnostic.frame_index for diagnostic in diagnostics] != expected:
            raise RuntimeError("Pre-gate diagnostic frame mapping does not match propagated frames")
        self.raw_logits = [propagated[index].copy() for index in expected]
        self.reverse_raw_logits = None
        self.pre_gate_diagnostics = list(diagnostics)
        return list(self.raw_logits)

    def track_reverse_logits(
        self,
        regular_masks: Sequence[np.ndarray],
        progress: ProgressCallback = lambda _message: None,
    ) -> list[np.ndarray]:
        """Track frame 014 to 000 in a fresh state while reusing the loaded model."""
        if (
            self.predictor is None
            or self.inference_state is None
            or self.device is None
            or self.sam_frames_dir is None
        ):
            raise RuntimeError("Complete Regular tracking before generating Reverse masks")
        if len(self.frames) != REVERSE_ANCHOR_FRAME_INDEX + 1:
            raise ValueError("Reverse tracking requires frames 000 through 014")
        if len(regular_masks) != len(self.frames):
            raise ValueError("Regular mask and frame counts do not match")

        from smoke import (
            add_sam2_mask_at_frame_logits,
            initialize_sam2_state,
            propagate_sam2_logits,
            validate_binary_mask,
        )

        anchor_frame = self.frames[REVERSE_ANCHOR_FRAME_INDEX]
        anchor = validate_binary_mask(
            regular_masks[REVERSE_ANCHOR_FRAME_INDEX],
            anchor_frame.size,
        )
        anchor = clip_mask_to_frame(anchor, anchor_frame)
        if not anchor.any():
            raise ValueError("Reverse requires a non-empty Regular mask on frame 014.")

        progress("Initializing fresh SAM 2 state for Reverse tracking...")
        reverse_state = initialize_sam2_state(
            self.predictor,
            self.sam_frames_dir,
            self.device,
        )
        try:
            progress("Registering frame 014 Regular mask as the Reverse anchor...")
            add_sam2_mask_at_frame_logits(
                self.predictor,
                reverse_state,
                anchor,
                anchor_frame.size,
                REVERSE_ANCHOR_FRAME_INDEX,
            )
            progress("Propagating Reverse masks from frame 014 to frame 000...")
            propagated = propagate_sam2_logits(
                self.predictor,
                reverse_state,
                start_frame_idx=REVERSE_ANCHOR_FRAME_INDEX,
                reverse=True,
            )
        finally:
            reset_state = getattr(self.predictor, "reset_state", None)
            if callable(reset_state):
                reset_state(reverse_state)

        expected = list(range(REVERSE_ANCHOR_FRAME_INDEX + 1))
        if sorted(propagated) != expected:
            raise RuntimeError(f"Expected Reverse frames {expected}, got {sorted(propagated)}")
        retained = [propagated[index].copy() for index in expected]
        self.reverse_raw_logits = retained
        return [logits.copy() for logits in retained]

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
        self.reset_tracking_state()
        self.predictor = None
        self.device = None
        self.sam_frames_dir = None
        shutil.rmtree(self.temp_dir, ignore_errors=True)


TEAM_COLOR_NAMES = tuple(name.title() for name in DEFAULT_TARGETS) + ("Custom",)
