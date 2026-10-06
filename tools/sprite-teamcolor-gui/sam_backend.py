from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from core import FRAME_SIZE, PromptPoint, clip_mask_to_alpha

MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


@dataclass(frozen=True)
class TrackingResult:
    masks: list[np.ndarray]
    elapsed_seconds: float
    device: str


class Sam2RowTracker:
    """Lazy SAM 2.1 video-predictor wrapper for one sprite-sheet row."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        work_size: int = 1024,
        background: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.requested_device = device
        self.work_size = work_size
        self.background = background
        self._predictor = None
        self._device = None

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    @property
    def device_name(self) -> str | None:
        return str(self._device) if self._device is not None else None

    def _ensure_model(self, status: Callable[[str], None] | None = None) -> None:
        if self._predictor is not None:
            return
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM 2.1 checkpoint not found: {self.checkpoint}")

        if status:
            status("SAM 2モデルを読み込み中…")

        import torch
        from sam2.build_sam import build_sam2_video_predictor

        name = self.requested_device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self._device = torch.device(name)
        self._predictor = build_sam2_video_predictor(
            MODEL_CONFIG,
            str(self.checkpoint),
            device=self._device,
            apply_postprocessing=False,
        )

    def track(
        self,
        frames: list[Image.Image],
        prompts_by_frame: dict[int, list[PromptPoint]],
        *,
        status: Callable[[str], None] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> TrackingResult:
        if not frames:
            raise ValueError("no frames supplied")
        if any(frame.size != (FRAME_SIZE, FRAME_SIZE) for frame in frames):
            raise ValueError("all frames must be 128x128")
        if not any(
            point.label == 1
            for points in prompts_by_frame.values()
            for point in points
        ):
            raise ValueError("at least one positive click is required")
        for frame_index in prompts_by_frame:
            if not 0 <= frame_index < len(frames):
                raise ValueError(f"prompt frame {frame_index} is outside the row")

        self._ensure_model(status)
        assert self._predictor is not None
        assert self._device is not None

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="sam2-teamcolor-") as temp:
            temp_dir = Path(temp)
            video_dir = temp_dir / "video"
            video_dir.mkdir()

            if status:
                status("SAM 2入力フレームを準備中…")
            for index, frame in enumerate(frames):
                rgba = frame.convert("RGBA")
                rgb = Image.new("RGB", rgba.size, self.background)
                rgb.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))
                work = rgb.resize(
                    (self.work_size, self.work_size),
                    Image.Resampling.NEAREST,
                )
                work.save(
                    video_dir / f"{index:05}.jpg",
                    "JPEG",
                    quality=100,
                    subsampling=0,
                    optimize=False,
                )

            if status:
                status("SAM 2追跡状態を初期化中…")
            state = self._predictor.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=self._device.type == "cuda",
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )

            scale = self.work_size / FRAME_SIZE
            for frame_index in sorted(prompts_by_frame):
                points = prompts_by_frame[frame_index]
                if not points:
                    continue
                if status:
                    status(f"フレーム {frame_index + 1} のクリックを反映中…")
                point_array = np.asarray(
                    [[point.x * scale, point.y * scale] for point in points],
                    dtype=np.float32,
                )
                labels = np.asarray([point.label for point in points], dtype=np.int32)
                self._predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_index,
                    obj_id=1,
                    points=point_array,
                    labels=labels,
                )

            if status:
                status("全フレームへマスクを伝播中…")

            propagated: dict[int, np.ndarray] = {}
            total = len(frames)
            for frame_index, object_ids, mask_logits in self._predictor.propagate_in_video(state):
                ids = [int(object_id) for object_id in object_ids]
                object_index = ids.index(1)
                raw_work_mask = (
                    mask_logits[object_index].detach().cpu().numpy().squeeze() > 0
                )
                raw = np.asarray(
                    Image.fromarray(raw_work_mask.astype(np.uint8) * 255, mode="L").resize(
                        (FRAME_SIZE, FRAME_SIZE),
                        Image.Resampling.NEAREST,
                    )
                ) > 0
                propagated[int(frame_index)] = clip_mask_to_alpha(
                    frames[int(frame_index)].convert("RGBA"),
                    raw,
                )
                if progress:
                    progress(len(propagated), total)

        expected = list(range(len(frames)))
        if sorted(propagated) != expected:
            raise RuntimeError(
                f"expected propagated frames {expected}, got {sorted(propagated)}"
            )
        masks = [propagated[index] for index in expected]
        return TrackingResult(
            masks=masks,
            elapsed_seconds=time.perf_counter() - started,
            device=str(self._device),
        )
