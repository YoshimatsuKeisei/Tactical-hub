#!/usr/bin/env python3
"""SAM 2.1 Hiera Tiny smoke test for row 0 of the HD Knight Idle sprite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "input" / "Idle.png"
DEFAULT_CHECKPOINT = ROOT / "models" / "sam2.1_hiera_tiny.pt"
DEFAULT_OUTPUT = ROOT / "output"
DEFAULT_TEMP = ROOT / "temp" / "run"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
MODEL_NAME = "sam2.1_hiera_tiny"
FRAME_SIZE = 128
FRAME_COUNT = 15
TESTED_ROW = 0


@dataclass(frozen=True)
class PromptPoint:
    x: int
    y: int
    label: int

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y}


def parse_xy(value: str) -> tuple[int, int]:
    try:
        x_text, y_text = value.split(",", maxsplit=1)
        x, y = int(x_text), int(y_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("point must use integer x,y syntax") from error
    if not (0 <= x < FRAME_SIZE and 0 <= y < FRAME_SIZE):
        raise argparse.ArgumentTypeError("point must be inside the 128x128 frame")
    return x, y


def parse_rgb(value: str) -> tuple[int, int, int]:
    try:
        channels = tuple(int(channel) for channel in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("background must use r,g,b syntax") from error
    if len(channels) != 3 or any(channel < 0 or channel > 255 for channel in channels):
        raise argparse.ArgumentTypeError("background channels must be integers from 0 to 255")
    return channels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_input(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Input sprite not found: {path}")
    image = Image.open(path)
    image.load()
    if image.size != (1920, 1024):
        raise ValueError(f"Expected input dimensions 1920x1024, got {image.width}x{image.height}")
    if image.mode != "RGBA":
        raise ValueError(f"Expected RGBA input, got {image.mode}")
    if image.getextrema()[3] == (255, 255):
        raise ValueError("Expected an alpha channel with transparent pixels")
    return image


def prepare_frames(
    sprite: Image.Image,
    temp_dir: Path,
    work_size: int,
    background: tuple[int, int, int],
) -> tuple[list[Image.Image], Path]:
    if work_size < FRAME_SIZE or work_size % FRAME_SIZE != 0:
        raise ValueError("work size must be a multiple of 128 and at least 128")
    frames: list[Image.Image] = []
    for frame_index in range(FRAME_COUNT):
        left = frame_index * FRAME_SIZE
        frame = sprite.crop((left, 0, left + FRAME_SIZE, FRAME_SIZE)).copy()
        frames.append(frame)

    sam_dir = prepare_frame_sequence(frames, temp_dir, work_size, background)

    preview = frames[0].resize((512, 512), Image.Resampling.NEAREST)
    preview.save(temp_dir / "frame_000_prompt.png")
    return frames, sam_dir


def prepare_frame_sequence(
    frames: list[Image.Image],
    temp_dir: Path,
    work_size: int,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Path:
    """Prepare arbitrary same-size RGB/RGBA frames for the official video loader."""
    if not frames:
        raise ValueError("At least one frame is required")
    if work_size <= 0:
        raise ValueError("work size must be positive")
    if any(frame.mode not in ("RGB", "RGBA") for frame in frames):
        raise ValueError("SAM 2 frames must be RGB or RGBA")
    if any(frame.size != frames[0].size for frame in frames):
        raise ValueError("All SAM 2 frames must have the same dimensions")

    rgba_dir = temp_dir / "rgba_frames"
    sam_dir = temp_dir / "sam_jpeg_frames"
    rgba_dir.mkdir(parents=True, exist_ok=True)
    sam_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, frame in enumerate(frames):
        frame.save(rgba_dir / f"frame_{frame_index:03}.png")
        rgb = Image.new("RGB", frame.size, background)
        if frame.mode == "RGBA":
            rgb.paste(frame.convert("RGB"), mask=frame.getchannel("A"))
        else:
            rgb.paste(frame)
        # SAM 2.1 uses a 1024px image encoder. Nearest-neighbour avoids adding
        # interpolated sprite pixels before the official video loader runs.
        work = rgb.resize((work_size, work_size), Image.Resampling.NEAREST)
        work.save(
            sam_dir / f"{frame_index:05}.jpg",
            format="JPEG",
            quality=100,
            subsampling=0,
            optimize=False,
        )
    return sam_dir


def choose_points_gui(frame: Image.Image) -> list[PromptPoint]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is unavailable; use --positive/--negative") from error

    scale = 4
    base_rgba = np.asarray(frame.resize((512, 512), Image.Resampling.NEAREST))
    base_bgr = cv2.cvtColor(base_rgba, cv2.COLOR_RGBA2BGRA)
    points: list[PromptPoint] = []
    window = "SAM 2 prompt: left=positive right=negative Enter=run R=reset Esc=cancel"

    def redraw() -> None:
        canvas = base_bgr.copy()
        for point in points:
            color = (0, 255, 0, 255) if point.label == 1 else (0, 0, 255, 255)
            cv2.circle(canvas, (point.x * scale, point.y * scale), 7, color, -1)
            cv2.circle(canvas, (point.x * scale, point.y * scale), 9, (255, 255, 255, 255), 1)
        cv2.imshow(window, canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
        points.append(PromptPoint(min(x // scale, 127), min(y // scale, 127), label))
        redraw()

    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, on_mouse)
        redraw()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and any(point.label == 1 for point in points):
                return points
            if key in (ord("r"), ord("R")):
                points.clear()
                redraw()
            if key == 27:
                raise KeyboardInterrupt("prompt selection cancelled")
    except cv2.error as error:
        raise RuntimeError("OpenCV GUI is unavailable; use --positive/--negative") from error
    finally:
        cv2.destroyAllWindows()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(device_name)


def create_sam2_predictor(model_config: str, checkpoint_path: Path, device: torch.device) -> object:
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as error:
        raise RuntimeError("Official SAM 2 is not installed in this Python environment") from error
    return build_sam2_video_predictor(
        model_config,
        str(checkpoint_path),
        device=device,
        apply_postprocessing=False,
    )


def initialize_sam2_state(predictor: object, sam_frames_dir: Path, device: torch.device) -> object:
    return predictor.init_state(
        video_path=str(sam_frames_dir),
        offload_video_to_cpu=device.type == "cuda",
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )


def _resize_work_mask(mask_logits: object, frame_size: tuple[int, int]) -> np.ndarray:
    work_mask = mask_logits.detach().cpu().numpy().squeeze() > 0
    mask_image = Image.fromarray(work_mask.astype(np.uint8) * 255)
    return np.asarray(mask_image.resize(frame_size, Image.Resampling.NEAREST)) > 0


def add_sam2_prompts(
    predictor: object,
    inference_state: object,
    prompt_points: list[PromptPoint],
    frame_size: tuple[int, int],
    work_size: int,
) -> np.ndarray:
    """Replace frame-0 prompts and return its mask in original-frame coordinates."""
    if not any(point.label == 1 for point in prompt_points):
        raise ValueError("At least one positive point is required")
    frame_width, frame_height = frame_size
    scale_x = work_size / frame_width
    scale_y = work_size / frame_height
    point_array = np.array(
        [[point.x * scale_x, point.y * scale_y] for point in prompt_points],
        dtype=np.float32,
    )
    label_array = np.array([point.label for point in prompt_points], dtype=np.int32)
    _frame_index, object_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        points=point_array,
        labels=label_array,
        clear_old_points=True,
    )
    object_id_list = [int(object_id) for object_id in object_ids]
    object_index = object_id_list.index(1)
    return _resize_work_mask(mask_logits[object_index], frame_size)


def propagate_sam2_masks(
    predictor: object,
    inference_state: object,
    frame_size: tuple[int, int],
) -> dict[int, np.ndarray]:
    """Propagate object 1 and return masks in original-frame coordinates."""
    propagated: dict[int, np.ndarray] = {}
    for frame_index, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
        object_id_list = [int(object_id) for object_id in object_ids]
        object_index = object_id_list.index(1)
        propagated[int(frame_index)] = _resize_work_mask(mask_logits[object_index], frame_size)
    return propagated


def bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_iou(left: np.ndarray, right: np.ndarray) -> float | None:
    union = np.logical_or(left, right).sum()
    if union == 0:
        return None
    return float(np.logical_and(left, right).sum() / union)


def save_overlay(frame: Image.Image, mask: np.ndarray, destination: Path) -> None:
    rgba = np.array(frame, copy=True)
    selected = mask & (rgba[:, :, 3] > 0)
    original = rgba[selected, :3].astype(np.float32)
    red = np.array([255, 32, 32], dtype=np.float32)
    rgba[selected, :3] = np.rint(original * 0.45 + red * 0.55).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(destination)


def checkerboard(size: tuple[int, int], cell: int = 8) -> Image.Image:
    width, height = size
    output = Image.new("RGB", size, (50, 50, 50))
    draw = ImageDraw.Draw(output)
    for y in range(0, height, cell):
        for x in range(0, width, cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle((x, y, min(x + cell - 1, width), min(y + cell - 1, height)), fill=(72, 72, 72))
    return output


def save_contact_sheet(overlays: Iterable[Path], destination: Path) -> None:
    overlay_paths = list(overlays)
    columns, rows = 5, 3
    label_height = 18
    sheet = Image.new("RGB", (columns * FRAME_SIZE, rows * (FRAME_SIZE + label_height)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for index, overlay_path in enumerate(overlay_paths):
        x = (index % columns) * FRAME_SIZE
        y = (index // columns) * (FRAME_SIZE + label_height)
        background = checkerboard((FRAME_SIZE, FRAME_SIZE))
        overlay = Image.open(overlay_path).convert("RGBA")
        background.paste(overlay, mask=overlay.getchannel("A"))
        sheet.paste(background, (x, y))
        draw.text((x + 4, y + FRAME_SIZE + 2), f"frame_{index:03}", fill=(240, 240, 240))
    sheet.save(destination)


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_report(metrics: dict) -> str:
    checks = metrics["automated_screening"]
    warning_lines = "\n".join(f"- {warning}" for warning in metrics["warnings"]) or "- None"
    return f"""# SAM 2.1 Idle row 0 smoke report

## Run

- Status: **REQUIRES VISUAL REVIEW**
- Model: `{metrics['model']['name']}`
- Checkpoint: `{metrics['model']['checkpoint']}`
- Device: `{metrics['device']['type']}`
- Prompt positive: `{metrics['prompts']['positive']}`
- Prompt negative: `{metrics['prompts']['negative']}`
- Inference elapsed: `{metrics['timing_seconds']['inference_total']:.3f}` seconds

## Automated screening (not a PASS verdict)

- Empty-mask frames: `{checks['empty_mask_frames']}`
- Abrupt area-change frames: `{checks['abrupt_area_change_frames']}`
- Low temporal-IoU frames: `{checks['low_temporal_iou_frames']}`
- Alpha-clip spill frames: `{checks['high_preclip_alpha_spill_frames']}`

## Warnings

{warning_lines}

## Required visual review

Open `row0_contact_sheet.png` and the files in `overlays/`. Confirm that the
prompted sprite part remains selected and does not drift to armour, sword, shield,
or another body part. Automated metrics cannot establish semantic correctness.
"""


def run(args: argparse.Namespace) -> dict:
    input_path = args.input.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output.resolve()
    temp_dir = args.temp.resolve()
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    sprite = validate_input(input_path)
    frames, sam_frames_dir = prepare_frames(sprite, temp_dir, args.work_size, args.background)

    prompt_points = [PromptPoint(x, y, 1) for x, y in args.positive]
    prompt_points.extend(PromptPoint(x, y, 0) for x, y in args.negative)
    if args.gui:
        if prompt_points:
            raise ValueError("Use either --gui or CLI prompt coordinates, not both")
        prompt_points = choose_points_gui(frames[0])
    if not any(point.label == 1 for point in prompt_points):
        raise ValueError("At least one positive point is required")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM 2.1 checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    recorded_warnings: list[str] = []
    timing: dict[str, float] = {}
    inference_started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        predictor = create_sam2_predictor(args.model_config, checkpoint_path, device)
        timing["model_load"] = time.perf_counter() - started

        started = time.perf_counter()
        inference_state = initialize_sam2_state(predictor, sam_frames_dir, device)
        timing["state_init"] = time.perf_counter() - started

        started = time.perf_counter()
        add_sam2_prompts(
            predictor,
            inference_state,
            prompt_points,
            (FRAME_SIZE, FRAME_SIZE),
            args.work_size,
        )
        timing["prompt"] = time.perf_counter() - started

        started = time.perf_counter()
        propagated = propagate_sam2_masks(
            predictor,
            inference_state,
            (FRAME_SIZE, FRAME_SIZE),
        )
        timing["propagation"] = time.perf_counter() - started
        recorded_warnings.extend(str(item.message) for item in caught)

    timing["inference_total"] = time.perf_counter() - inference_started
    if sorted(propagated) != list(range(FRAME_COUNT)):
        raise RuntimeError(f"Expected propagated frames 0..14, got {sorted(propagated)}")

    frame_metrics: list[dict] = []
    prior_mask: np.ndarray | None = None
    overlay_paths: list[Path] = []
    abrupt_frames: list[int] = []
    low_iou_frames: list[int] = []
    empty_frames: list[int] = []
    spill_frames: list[int] = []

    for frame_index, frame in enumerate(frames):
        raw_mask = propagated[frame_index]
        alpha = np.asarray(frame.getchannel("A")) > 0
        outside_alpha_pixels = int(np.logical_and(raw_mask, ~alpha).sum())
        raw_area = int(raw_mask.sum())
        clipped_mask = np.logical_and(raw_mask, alpha)
        area = int(clipped_mask.sum())
        temporal_iou = mask_iou(prior_mask, clipped_mask) if prior_mask is not None else None
        area_ratio = (area / int(prior_mask.sum())) if prior_mask is not None and prior_mask.sum() else None
        spill_ratio = outside_alpha_pixels / raw_area if raw_area else 0.0

        if area == 0:
            empty_frames.append(frame_index)
        if area_ratio is not None and (area_ratio < 0.4 or area_ratio > 2.5):
            abrupt_frames.append(frame_index)
        if temporal_iou is not None and temporal_iou < 0.15:
            low_iou_frames.append(frame_index)
        if spill_ratio > 0.25:
            spill_frames.append(frame_index)

        mask_path = masks_dir / f"frame_{frame_index:03}.png"
        Image.fromarray(clipped_mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        overlay_path = overlays_dir / f"frame_{frame_index:03}.png"
        save_overlay(frame, clipped_mask, overlay_path)
        overlay_paths.append(overlay_path)

        frame_metrics.append(
            {
                "frame": frame_index,
                "mask_pixel_area": area,
                "mask_bounding_box_xyxy": bbox(clipped_mask),
                "sprite_alpha_pixel_area": int(alpha.sum()),
                "sprite_alpha_bounding_box_xyxy": bbox(alpha),
                "previous_frame_mask_iou": temporal_iou,
                "previous_frame_area_ratio": area_ratio,
                "raw_mask_pixel_area_before_alpha_clip": raw_area,
                "pixels_removed_by_alpha_clip": outside_alpha_pixels,
                "preclip_outside_alpha_ratio": spill_ratio,
            }
        )
        prior_mask = clipped_mask

    save_contact_sheet(overlay_paths, output_dir / "row0_contact_sheet.png")
    if empty_frames:
        recorded_warnings.append(f"mask disappeared on frames {empty_frames}")
    if abrupt_frames:
        recorded_warnings.append(f"abrupt mask area change on frames {abrupt_frames}")
    if low_iou_frames:
        recorded_warnings.append(f"temporal IoU below 0.15 on frames {low_iou_frames}")
    if spill_frames:
        recorded_warnings.append(f"more than 25% of raw mask was outside alpha on frames {spill_frames}")

    metrics = {
        "status": "REQUIRES_VISUAL_REVIEW",
        "input": {
            "path": str(input_path),
            "sha256": sha256(input_path),
            "dimensions": [sprite.width, sprite.height],
            "mode": sprite.mode,
        },
        "frame_dimensions": [FRAME_SIZE, FRAME_SIZE],
        "sam_work_frame_dimensions": [args.work_size, args.work_size],
        "sam_work_resize": "nearest-neighbor",
        "sam_work_format": "JPEG quality=100 subsampling=0 (required by official video loader)",
        "tested_row": TESTED_ROW,
        "frame_count": FRAME_COUNT,
        "prompts": {
            "coordinate_space": "original 128x128 frame",
            "positive": [point.as_dict() for point in prompt_points if point.label == 1],
            "negative": [point.as_dict() for point in prompt_points if point.label == 0],
        },
        "model": {
            "name": MODEL_NAME,
            "config": args.model_config,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "sam2_package_version": package_version("SAM-2"),
        },
        "device": {
            "type": device.type,
            "torch": torch.__version__,
            "torchvision": package_version("torchvision"),
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "frames": frame_metrics,
        "timing_seconds": timing,
        "automated_screening": {
            "empty_mask_frames": empty_frames,
            "abrupt_area_change_frames": abrupt_frames,
            "low_temporal_iou_frames": low_iou_frames,
            "high_preclip_alpha_spill_frames": spill_frames,
            "thresholds": {
                "abrupt_area_ratio": [0.4, 2.5],
                "low_temporal_iou": 0.15,
                "high_preclip_alpha_spill_ratio": 0.25,
            },
        },
        "warnings": recorded_warnings,
        "errors": [],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "smoke_report.md").write_text(make_report(metrics), encoding="utf-8")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temp", type=Path, default=DEFAULT_TEMP)
    parser.add_argument("--model-config", default=MODEL_CONFIG)
    parser.add_argument("--positive", action="append", type=parse_xy, default=[], metavar="X,Y")
    parser.add_argument("--negative", action="append", type=parse_xy, default=[], metavar="X,Y")
    parser.add_argument("--gui", action="store_true", help="pick points on frame 0 with OpenCV")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--background", type=parse_rgb, default=(0, 0, 0), metavar="R,G,B")
    parser.add_argument("--work-size", type=int, default=1024)
    parser.add_argument("--prepare-only", action="store_true", help="validate and extract row 0 without loading SAM 2")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.prepare_only:
        sprite = validate_input(args.input.resolve())
        args.temp.resolve().mkdir(parents=True, exist_ok=True)
        prepare_frames(sprite, args.temp.resolve(), args.work_size, args.background)
        print(f"Prepared {FRAME_COUNT} row-0 frames in {args.temp.resolve()}")
        return 0
    metrics = run(args)
    print(json.dumps({"status": metrics["status"], "timing_seconds": metrics["timing_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as error:
        print(str(error) or "cancelled", file=sys.stderr)
        raise SystemExit(130) from error
