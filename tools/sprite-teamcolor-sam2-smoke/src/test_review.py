import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gui_controller import RecolorPreviewState, recolor_frame_sequence, save_results


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


if __name__ == "__main__":
    unittest.main()
