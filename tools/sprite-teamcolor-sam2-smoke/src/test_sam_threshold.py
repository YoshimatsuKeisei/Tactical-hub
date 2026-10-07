import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gui_controller import (
    DEFAULT_MASK_THRESHOLD,
    MAX_MASK_THRESHOLD,
    MIN_MASK_THRESHOLD,
    PromptClick,
    RecolorPreviewState,
    Sam2GuiSession,
    derive_masks_from_sam_logits,
    recolor_frame_sequence,
    save_results,
    validate_mask_threshold,
)
from smoke import _resize_work_mask, threshold_sam2_logits, validate_sam2_logits


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class CountingPredictor:
    def __init__(self, logits: list[np.ndarray]) -> None:
        self.logits = logits
        self.point_calls = 0
        self.mask_calls = 0
        self.propagate_calls = 0

    def add_new_points_or_box(self, **_kwargs: object) -> tuple[int, list[int], list[FakeTensor]]:
        self.point_calls += 1
        return 0, [1], [FakeTensor(self.logits[0][None, :, :])]

    def add_new_mask(self, **_kwargs: object) -> tuple[int, list[int], list[FakeTensor]]:
        self.mask_calls += 1
        return 0, [1], [FakeTensor(self.logits[0][None, :, :])]

    def propagate_in_video(self, _state: object):
        self.propagate_calls += 1
        for index, logits in enumerate(self.logits):
            yield index, [1], [FakeTensor(logits[None, :, :])]


def rgba_frame(size: tuple[int, int], alpha: int = 255) -> Image.Image:
    return Image.new("RGBA", size, (90, 90, 90, alpha))


class SamThresholdTests(unittest.TestCase):
    def test_raw_logits_validate_as_owned_2d_float32(self) -> None:
        source = np.array([[[1, -2], [3, 4]]], dtype=np.float64)

        logits = validate_sam2_logits(FakeTensor(source))

        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(logits.dtype, np.float32)
        self.assertFalse(np.shares_memory(logits, source))
        with self.assertRaisesRegex(ValueError, "2D"):
            validate_sam2_logits(np.zeros((2, 2, 2), dtype=np.float32))

    def test_threshold_zero_matches_historic_mask_pixel_for_pixel(self) -> None:
        logits = np.array([[1.0, 0.0], [-0.1, 2.0]], dtype=np.float32)

        expected = _resize_work_mask(FakeTensor(logits), (6, 4))
        actual = threshold_sam2_logits(logits, (6, 4), 0.0)

        np.testing.assert_array_equal(actual, expected)

    def test_lower_is_superset_higher_is_subset_and_logits_are_unchanged(self) -> None:
        logits = np.array([[-1.0, -0.2, 0.2, 1.0]], dtype=np.float32)
        before = logits.copy()

        lower = threshold_sam2_logits(logits, (8, 2), -0.3)
        baseline = threshold_sam2_logits(logits, (8, 2), 0.0)
        higher = threshold_sam2_logits(logits, (8, 2), 0.5)

        self.assertTrue(np.logical_or(~baseline, lower).all())
        self.assertTrue(np.logical_or(~higher, baseline).all())
        np.testing.assert_array_equal(logits, before)

    def test_threshold_then_nearest_resize_keeps_requested_frame_size(self) -> None:
        logits = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float32)

        mask = threshold_sam2_logits(logits, (4, 6), 0.0)

        self.assertEqual(mask.shape, (6, 4))
        self.assertTrue(mask[:3, :2].all())
        self.assertFalse(mask[:3, 2:].any())
        self.assertFalse(mask[3:, :2].any())
        self.assertTrue(mask[3:, 2:].all())

    def test_threshold_range_and_default(self) -> None:
        self.assertEqual(DEFAULT_MASK_THRESHOLD, 0.0)
        self.assertEqual(validate_mask_threshold(MIN_MASK_THRESHOLD), -2.0)
        self.assertEqual(validate_mask_threshold(MAX_MASK_THRESHOLD), 2.0)
        with self.assertRaisesRegex(ValueError, "between"):
            validate_mask_threshold(-2.01)
        with self.assertRaisesRegex(ValueError, "between"):
            validate_mask_threshold(2.01)

    def test_pipeline_order_is_threshold_resize_alpha_then_hole_fill(self) -> None:
        alpha = np.full((5, 5), 255, dtype=np.uint8)
        alpha[2, 2] = 0
        pixels = np.full((5, 5, 4), 90, dtype=np.uint8)
        pixels[:, :, 3] = alpha
        frame = Image.fromarray(pixels, "RGBA")
        logits = np.ones((5, 5), dtype=np.float32)

        raw, active = derive_masks_from_sam_logits([logits], [frame], 0.0, fill_holes=True)
        _raw_off, inactive = derive_masks_from_sam_logits([logits], [frame], 0.0, fill_holes=False)

        self.assertTrue(raw[0][2, 2])
        self.assertTrue(active[0][2, 2], "alpha clipping must happen before enclosed-hole filling")
        self.assertFalse(inactive[0][2, 2])

    def test_open_background_is_not_filled_after_threshold_change(self) -> None:
        logits = np.full((5, 5), -1.0, dtype=np.float32)
        logits[1:4, 1] = 1.0
        logits[1, 1:4] = 1.0
        logits[3, 1:4] = 1.0
        frame = rgba_frame((5, 5))

        _raw, active = derive_masks_from_sam_logits([logits], [frame], -0.5, fill_holes=True)

        self.assertFalse(active[0][2, 2])

    def test_quick_frame0_changes_but_authoritative_frame0_does_not(self) -> None:
        logits = [
            np.array([[-0.2, 1.0], [-1.0, 1.0]], dtype=np.float32),
            np.array([[-0.2, 1.0], [-1.0, 1.0]], dtype=np.float32),
        ]
        frames = [rgba_frame((2, 2)), rgba_frame((2, 2))]
        authoritative = np.array([[False, True], [True, False]], dtype=bool)

        quick_zero, _ = derive_masks_from_sam_logits(logits, frames, 0.0, False)
        quick_lower, _ = derive_masks_from_sam_logits(logits, frames, -0.3, False)
        area_zero, _ = derive_masks_from_sam_logits(logits, frames, 0.0, False, authoritative)
        area_lower, _ = derive_masks_from_sam_logits(logits, frames, -0.3, False, authoritative)

        self.assertFalse(np.array_equal(quick_zero[0], quick_lower[0]))
        np.testing.assert_array_equal(area_zero[0], authoritative)
        np.testing.assert_array_equal(area_lower[0], authoritative)
        self.assertFalse(np.array_equal(area_zero[1], area_lower[1]))

    def test_select_all_alpha_mask_is_authoritative_at_frame0(self) -> None:
        pixels = np.zeros((2, 2, 4), dtype=np.uint8)
        pixels[0, 1, 3] = 255
        frame = Image.fromarray(pixels, "RGBA")
        alpha_mask = np.asarray(frame.getchannel("A")) > 0
        logits = [np.full((2, 2), -0.2, dtype=np.float32)]

        raw_zero, _ = derive_masks_from_sam_logits(logits, [frame], 0.0, False, alpha_mask)
        raw_lower, _ = derive_masks_from_sam_logits(logits, [frame], -0.3, False, alpha_mask)

        np.testing.assert_array_equal(raw_zero[0], alpha_mask)
        np.testing.assert_array_equal(raw_lower[0], alpha_mask)

    def test_rethresholding_retained_tracking_logits_does_not_rerun_sam(self) -> None:
        logits = [
            np.array([[1.0, -0.2], [-1.0, 1.0]], dtype=np.float32),
            np.array([[-0.2, 1.0], [-1.0, 1.0]], dtype=np.float32),
        ]
        predictor = CountingPredictor(logits)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.touch()
            session = Sam2GuiSession([rgba_frame((2, 2)), rgba_frame((2, 2))], checkpoint, work_size=2)
            session.predictor = predictor
            session.inference_state = object()
            session.generate_frame0_mask([PromptClick(0, 0, 1)])
            retained = session.track_across_frames_logits()
            retained_before = [item.copy() for item in retained]

            zero = derive_masks_from_sam_logits(retained, session.frames, 0.0, False)
            lower = derive_masks_from_sam_logits(retained, session.frames, -0.3, False)

            self.assertEqual(predictor.point_calls, 1)
            self.assertEqual(predictor.propagate_calls, 1)
            self.assertFalse(np.array_equal(zero[0][1], lower[0][1]))
            for actual, expected in zip(retained, retained_before, strict=True):
                np.testing.assert_array_equal(actual, expected)
            session.close()
            self.assertIsNone(session.frame0_logits)
            self.assertIsNone(session.raw_logits)

    def test_accept_exports_masks_from_current_threshold(self) -> None:
        logits = [np.array([[-0.2, 1.0], [-1.0, 1.0]], dtype=np.float32)]
        frames = [rgba_frame((2, 2))]
        _raw, active = derive_masks_from_sam_logits(logits, frames, -0.3, False)
        recolored = recolor_frame_sequence(frames, active, (210, 48, 48))
        review = RecolorPreviewState()
        review.begin(recolored, Image.new("RGB", (2, 2)))

        with tempfile.TemporaryDirectory() as temporary:
            outputs = review.accept(
                lambda pending: save_results(Path(temporary), frames, active, pending, columns=1)
            )
            with Image.open(outputs["masks"] / "frame_000.png") as saved:
                exported = np.asarray(saved).copy() > 0

        np.testing.assert_array_equal(exported, active[0])

    def test_reject_does_not_change_retained_logits_threshold_or_masks(self) -> None:
        logits = [np.array([[-0.2, 1.0], [-1.0, 1.0]], dtype=np.float32)]
        threshold = -0.3
        frames = [rgba_frame((2, 2))]
        _raw, active = derive_masks_from_sam_logits(logits, frames, threshold, False)
        logits_before = logits[0].copy()
        mask_before = active[0].copy()
        review = RecolorPreviewState()
        review.begin(
            recolor_frame_sequence(frames, active, (210, 48, 48)),
            Image.new("RGB", (2, 2)),
        )

        review.reject()

        self.assertEqual(threshold, -0.3)
        np.testing.assert_array_equal(logits[0], logits_before)
        np.testing.assert_array_equal(active[0], mask_before)


if __name__ == "__main__":
    unittest.main()
