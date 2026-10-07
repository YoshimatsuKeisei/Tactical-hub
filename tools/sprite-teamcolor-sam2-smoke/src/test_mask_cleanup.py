import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gui_controller import (
    RecolorPreviewState,
    apply_mask_cleanup,
    apply_mask_cleanup_sequence,
    fill_enclosed_holes,
    recolor_frame_sequence,
    save_results,
)


def ring_mask(size: int = 7) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[1, 1:-1] = True
    mask[-2, 1:-1] = True
    mask[1:-1, 1] = True
    mask[1:-1, -2] = True
    return mask


class MaskCleanupTests(unittest.TestCase):
    def test_fills_a_fully_enclosed_hole(self) -> None:
        raw = ring_mask()

        cleaned = fill_enclosed_holes(raw)

        self.assertTrue(cleaned[3, 3])
        self.assertFalse(cleaned[0, 0])

    def test_does_not_fill_background_connected_to_an_edge(self) -> None:
        raw = ring_mask()
        raw[1:4, 3] = False

        cleaned = fill_enclosed_holes(raw)

        np.testing.assert_array_equal(cleaned, raw)
        self.assertFalse(cleaned[3, 3])

    def test_fills_multiple_enclosed_holes(self) -> None:
        raw = np.ones((8, 10), dtype=bool)
        raw[2:4, 2:4] = False
        raw[4:7, 6:9] = False

        cleaned = fill_enclosed_holes(raw)

        self.assertTrue(cleaned.all())

    def test_mask_without_holes_is_unchanged_and_idempotent(self) -> None:
        raw = np.zeros((7, 7), dtype=bool)
        raw[2:5, 2:5] = True

        cleaned = fill_enclosed_holes(raw)

        np.testing.assert_array_equal(cleaned, raw)
        np.testing.assert_array_equal(fill_enclosed_holes(cleaned), cleaned)

    def test_cleanup_does_not_expand_the_outer_boundary(self) -> None:
        raw = ring_mask()
        cleaned = fill_enclosed_holes(raw)
        added = np.logical_and(cleaned, ~raw)
        expected_added = np.zeros(raw.shape, dtype=bool)
        expected_added[2:-2, 2:-2] = True

        np.testing.assert_array_equal(added, expected_added)
        self.assertFalse(added[0, :].any())
        self.assertFalse(added[-1, :].any())
        self.assertFalse(added[:, 0].any())
        self.assertFalse(added[:, -1].any())
        np.testing.assert_array_equal(cleaned[raw], np.ones(int(raw.sum()), dtype=bool))

    def test_quick_select_cleanup_on_and_off_derive_from_raw_mask(self) -> None:
        raw_frame0 = ring_mask()

        enabled = apply_mask_cleanup(raw_frame0, True)
        disabled = apply_mask_cleanup(raw_frame0, False)

        self.assertTrue(enabled[3, 3])
        self.assertFalse(disabled[3, 3])
        np.testing.assert_array_equal(disabled, raw_frame0)
        self.assertFalse(np.shares_memory(disabled, raw_frame0))

    def test_tracked_mask_sequence_cleanup_fills_each_frame(self) -> None:
        raw_masks = [ring_mask(), np.rot90(ring_mask()).copy()]

        cleaned = apply_mask_cleanup_sequence(raw_masks, True)

        self.assertEqual(len(cleaned), 2)
        self.assertTrue(all(mask[3, 3] for mask in cleaned))
        self.assertTrue(all(not mask[0, 0] for mask in cleaned))

    def test_cleanup_toggle_changes_recolor_only_inside_the_hole(self) -> None:
        pixels = np.full((7, 7, 4), (110, 110, 110, 173), dtype=np.uint8)
        frame = Image.fromarray(pixels, "RGBA")
        raw = ring_mask()
        disabled_mask = apply_mask_cleanup(raw, False)
        enabled_mask = apply_mask_cleanup(raw, True)

        disabled = np.asarray(
            recolor_frame_sequence(
                [frame],
                [disabled_mask],
                (210, 48, 48),
                strength=1.0,
                shadow_protect_amount=0.0,
            )[0]
        )
        enabled = np.asarray(
            recolor_frame_sequence(
                [frame],
                [enabled_mask],
                (210, 48, 48),
                strength=1.0,
                shadow_protect_amount=0.0,
            )[0]
        )

        np.testing.assert_array_equal(disabled[3, 3, :3], pixels[3, 3, :3])
        self.assertFalse(np.array_equal(enabled[3, 3, :3], pixels[3, 3, :3]))
        np.testing.assert_array_equal(disabled[0, 0, :3], pixels[0, 0, :3])
        np.testing.assert_array_equal(enabled[0, 0, :3], pixels[0, 0, :3])
        np.testing.assert_array_equal(disabled[:, :, 3], pixels[:, :, 3])
        np.testing.assert_array_equal(enabled[:, :, 3], pixels[:, :, 3])

    def test_accept_exports_the_current_cleanup_mask_and_recolor(self) -> None:
        frame = Image.new("RGBA", (7, 7), (110, 110, 110, 200))
        raw = ring_mask()

        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temporary:
                active = apply_mask_cleanup(raw, enabled)
                recolored = recolor_frame_sequence(
                    [frame],
                    [active],
                    (210, 48, 48),
                    strength=1.0,
                    shadow_protect_amount=0.0,
                )
                review = RecolorPreviewState()
                review.begin(recolored, Image.new("RGB", (7, 7)))
                destination = Path(temporary) / "accepted"

                outputs = review.accept(
                    lambda pending: save_results(destination, [frame], [active], pending, columns=1)
                )
                with Image.open(outputs["masks"] / "frame_000.png") as saved_mask:
                    saved_mask_pixel = saved_mask.getpixel((3, 3))
                with Image.open(outputs["recolored_frames"] / "frame_000.png") as saved_frame:
                    saved_rgb = saved_frame.getpixel((3, 3))[:3]

                self.assertEqual(saved_mask_pixel, 255 if enabled else 0)
                self.assertEqual(saved_rgb != (110, 110, 110), enabled)

    def test_reject_after_cleanup_preview_writes_no_files(self) -> None:
        frame = Image.new("RGBA", (7, 7), (110, 110, 110, 200))
        active = apply_mask_cleanup(ring_mask(), True)
        recolored = recolor_frame_sequence([frame], [active], (210, 48, 48))
        review = RecolorPreviewState()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            review.begin(recolored, Image.new("RGB", (7, 7)))
            review.reject()
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
