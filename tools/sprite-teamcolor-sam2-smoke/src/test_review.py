import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gui_controller import (
    MAX_ZOOM,
    MIN_ZOOM,
    RecolorPreviewState,
    clamp_zoom,
    cursor_centered_zoom_offset,
    pan_scroll_offset,
    recolor_frame_sequence,
    save_results,
)


class RecolorReviewTests(unittest.TestCase):
    def _sample(self) -> tuple[list[Image.Image], list[np.ndarray]]:
        frames = [
            Image.new("RGBA", (4, 4), (80, 80, 80, 255)),
            Image.new("RGBA", (4, 4), (160, 160, 160, 192)),
        ]
        masks = [np.ones((4, 4), dtype=bool), np.eye(4, dtype=bool)]
        return frames, masks

    def test_preview_and_reject_create_no_files_and_keep_masks(self) -> None:
        frames, masks = self._sample()
        masks_before = [mask.copy() for mask in masks]
        recolored = recolor_frame_sequence(frames, masks, (210, 48, 48))
        review = RecolorPreviewState()

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            review.begin(recolored, Image.new("RGB", (8, 8), "red"))
            self.assertEqual(list(output.iterdir()), [])
            review.reject()
            self.assertEqual(list(output.iterdir()), [])

        self.assertFalse(review.has_preview)
        for actual, expected in zip(masks, masks_before, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_accept_calls_exporter_only_after_acceptance(self) -> None:
        frames, masks = self._sample()
        review = RecolorPreviewState()
        review.begin(recolor_frame_sequence(frames, masks, (48, 92, 210)), Image.new("RGB", (8, 8)))
        calls: list[int] = []

        self.assertEqual(calls, [])
        result = review.accept(lambda pending: calls.append(len(pending)) or "saved")

        self.assertEqual(result, "saved")
        self.assertEqual(calls, [2])
        self.assertFalse(review.has_preview)

    def test_accept_saves_existing_result_set(self) -> None:
        frames, masks = self._sample()
        review = RecolorPreviewState()
        recolored = recolor_frame_sequence(frames, masks, (48, 170, 78))
        review.begin(recolored, Image.new("RGB", (8, 8)))

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "accepted"
            outputs = review.accept(
                lambda pending: save_results(destination, frames, masks, pending, columns=2)
            )

            self.assertTrue((outputs["masks"] / "frame_000.png").is_file())
            self.assertTrue((outputs["recolored_frames"] / "frame_001.png").is_file())
            self.assertTrue(outputs["recolored_contact_sheet"].is_file())
            self.assertTrue(outputs["overlay_contact_sheet"].is_file())

    def test_setting_changes_stay_in_memory_and_accept_exports_latest_result(self) -> None:
        frames, masks = self._sample()
        review = RecolorPreviewState()
        initial = recolor_frame_sequence(
            frames,
            masks,
            (210, 48, 48),
            strength=0.85,
            shadow_protect_amount=1.0,
        )
        latest = recolor_frame_sequence(
            frames,
            masks,
            (210, 48, 48),
            strength=1.0,
            shadow_protect_amount=0.0,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "accepted"
            review.begin(initial, Image.new("RGB", (8, 8), "red"))
            review.begin(latest, Image.new("RGB", (8, 8), "red"))
            self.assertEqual(list(root.iterdir()), [])

            outputs = review.accept(
                lambda pending: save_results(destination, frames, masks, pending, columns=2)
            )
            with Image.open(outputs["recolored_frames"] / "frame_000.png") as exported:
                exported_pixels = np.asarray(exported).copy()

        np.testing.assert_array_equal(exported_pixels, np.asarray(latest[0]))

    def test_reject_allows_retry_with_another_color(self) -> None:
        frames, masks = self._sample()
        review = RecolorPreviewState()
        red = recolor_frame_sequence(frames, masks, (210, 48, 48))
        blue = recolor_frame_sequence(frames, masks, (48, 92, 210))

        review.begin(red, Image.new("RGB", (8, 8), "red"))
        review.reject()
        review.begin(blue, Image.new("RGB", (8, 8), "blue"))

        self.assertTrue(review.has_preview)
        self.assertFalse(np.array_equal(np.asarray(red[0]), np.asarray(review.frames[0])))

    def test_preview_zoom_clamps_to_supported_range(self) -> None:
        self.assertEqual(clamp_zoom(0), MIN_ZOOM)
        self.assertEqual(clamp_zoom(17), MAX_ZOOM)

    def test_preview_cursor_centered_zoom_keeps_contact_point_stable(self) -> None:
        cursor = (120.0, 80.0)
        old_scroll = (240.0, 160.0)
        new_scroll = cursor_centered_zoom_offset(
            cursor,
            old_zoom=2,
            new_zoom=5,
            old_scroll=old_scroll,
            viewport_size=(500, 300),
            frame_size=(660, 420),
        )

        old_anchor = ((old_scroll[0] + cursor[0]) / 2, (old_scroll[1] + cursor[1]) / 2)
        new_anchor = ((new_scroll[0] + cursor[0]) / 5, (new_scroll[1] + cursor[1]) / 5)
        self.assertEqual(new_anchor, old_anchor)

    def test_preview_arrow_pan_clamps_to_scrollable_contact_sheet(self) -> None:
        self.assertEqual(
            pan_scroll_offset((200, 100), (64, -64), (1320, 840), (500, 300)),
            (264, 36),
        )
        self.assertEqual(
            pan_scroll_offset((810, 530), (64, 64), (1320, 840), (500, 300)),
            (820, 540),
        )


if __name__ == "__main__":
    unittest.main()
