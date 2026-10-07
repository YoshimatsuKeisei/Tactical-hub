import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from recolor import recolor_masked, save_contact_sheet


class RecolorTests(unittest.TestCase):
    def test_zero_strength_keeps_masked_rgb_unchanged(self) -> None:
        source = np.array([[[32, 48, 64, 17], [120, 100, 80, 255]]], dtype=np.uint8)
        mask = Image.fromarray(np.full((1, 2), 255, dtype=np.uint8))

        result = np.asarray(recolor_masked(Image.fromarray(source), mask, strength=0.0))

        np.testing.assert_array_equal(result, source)

    def test_full_strength_without_protection_maximizes_target_hue_blend(self) -> None:
        source = Image.new("RGB", (1, 1), (128, 128, 128))
        mask = Image.new("L", (1, 1), 255)

        partial = np.asarray(
            recolor_masked(source, mask, strength=0.5, shadow_protect_amount=0.0)
        )[0, 0]
        full = np.asarray(
            recolor_masked(source, mask, strength=1.0, shadow_protect_amount=0.0)
        )[0, 0]

        self.assertGreater(int(full[0]) - int(full[1]), int(partial[0]) - int(partial[1]))
        self.assertEqual(int(full.max()), 128)

    def test_full_protection_matches_existing_default_pixels(self) -> None:
        source = Image.fromarray(
            np.array([[[10, 10, 10], [40, 40, 40], [160, 160, 160]]], dtype=np.uint8)
        )
        mask = Image.new("L", (3, 1), 255)

        existing = np.asarray(recolor_masked(source, mask))
        explicit = np.asarray(recolor_masked(source, mask, shadow_protect_amount=1.0))
        expected_legacy_pixels = np.array(
            [[[10, 10, 10], [40, 26, 26], [160, 55, 55]]],
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(existing, expected_legacy_pixels)
        np.testing.assert_array_equal(explicit, existing)

    def test_zero_protection_removes_only_dark_pixel_attenuation(self) -> None:
        source = Image.new("RGB", (1, 1), (10, 10, 10))
        mask = Image.new("L", (1, 1), 255)

        protected = np.asarray(
            recolor_masked(source, mask, strength=1.0, shadow_protect_amount=1.0)
        )[0, 0]
        unprotected = np.asarray(
            recolor_masked(source, mask, strength=1.0, shadow_protect_amount=0.0)
        )[0, 0]

        np.testing.assert_array_equal(protected, np.array([10, 10, 10], dtype=np.uint8))
        self.assertGreater(unprotected[0], unprotected[1])
        self.assertEqual(int(unprotected.max()), 10)

    def test_max_settings_keep_unmasked_rgb_alpha_and_brightness_order(self) -> None:
        levels = np.array([24, 96, 208], dtype=np.uint8)
        rgb = np.repeat(levels[:, None], 3, axis=1)[None, :, :]
        alpha = np.array([[[32], [128], [240]]], dtype=np.uint8)
        source_array = np.concatenate((rgb, alpha), axis=2)
        mask = Image.fromarray(np.array([[255, 255, 0]], dtype=np.uint8))

        result = np.asarray(
            recolor_masked(
                Image.fromarray(source_array),
                mask,
                strength=1.0,
                shadow_protect_amount=0.0,
            )
        )

        np.testing.assert_array_equal(result[0, 2, :3], source_array[0, 2, :3])
        np.testing.assert_array_equal(result[:, :, 3], source_array[:, :, 3])
        self.assertLess(int(result[0, 0, :3].max()), int(result[0, 1, :3].max()))
        self.assertLess(int(result[0, 1, :3].max()), int(result[0, 2, :3].max()))

    def test_only_masked_pixels_change_and_alpha_is_preserved(self) -> None:
        source = np.array(
            [
                [[100, 100, 100, 0], [120, 120, 120, 64]],
                [[140, 140, 140, 128], [160, 160, 160, 255]],
            ],
            dtype=np.uint8,
        )
        mask = Image.fromarray(np.array([[0, 255], [0, 255]], dtype=np.uint8))

        output = recolor_masked(Image.fromarray(source), mask)
        result = np.asarray(output)

        self.assertEqual(output.size, (2, 2))
        self.assertEqual(output.mode, "RGBA")
        np.testing.assert_array_equal(result[:, :, 3], source[:, :, 3])
        np.testing.assert_array_equal(result[:, 0, :3], source[:, 0, :3])
        self.assertFalse(np.array_equal(result[:, 1, :3], source[:, 1, :3]))
        self.assertTrue(np.all(result[:, 1, 0] > result[:, 1, 1]))
        self.assertEqual(result.shape, source.shape)

    def test_dark_mid_and_light_order_is_preserved(self) -> None:
        levels = np.array([32, 128, 224], dtype=np.uint8)
        rgb = np.repeat(levels[:, None], 3, axis=1)[None, :, :]
        alpha = np.full((1, 3, 1), 173, dtype=np.uint8)
        source = Image.fromarray(np.concatenate((rgb, alpha), axis=2))
        mask = Image.fromarray(np.full((1, 3), 255, dtype=np.uint8))

        result = np.asarray(recolor_masked(source, mask))
        luminance = result[0, :, :3] @ np.array([0.2126, 0.7152, 0.0722])

        self.assertLess(luminance[0], luminance[1])
        self.assertLess(luminance[1], luminance[2])
        np.testing.assert_array_equal(result[0, :, 3], alpha[0, :, 0])

    def test_contact_sheet_has_expected_15_frame_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_paths = []
            for index in range(15):
                path = root / f"frame_{index:03}.png"
                Image.new("RGBA", (128, 128), (index, 0, 0, 255)).save(path)
                frame_paths.append(path)
            destination = root / "contact.png"

            size = save_contact_sheet(frame_paths, destination, columns=5)

            self.assertEqual(size, (640, 438))
            with Image.open(destination) as contact:
                self.assertEqual(contact.size, (640, 438))

    def test_cli_recolors_sprite_sheet_with_existing_masks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            masks = root / "masks"
            output = root / "output"
            masks.mkdir()
            sprite = Image.new("RGBA", (4, 2), (100, 100, 100, 200))
            sprite.save(root / "Melee.png")
            for index in range(2):
                Image.new("L", (2, 2), 255 if index == 0 else 0).save(masks / f"frame_{index:03}.png")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("recolor.py")),
                    "--input",
                    str(root / "Melee.png"),
                    "--masks",
                    str(masks),
                    "--output",
                    str(output),
                    "--frame-count",
                    "2",
                    "--frame-width",
                    "2",
                    "--frame-height",
                    "2",
                    "--columns",
                    "2",
                    "--target",
                    "red",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            with Image.open(output / "recolored_frames" / "frame_000.png") as image:
                first = np.asarray(image).copy()
            with Image.open(output / "recolored_frames" / "frame_001.png") as image:
                second = np.asarray(image).copy()
            self.assertFalse(np.array_equal(first[:, :, :3], np.full((2, 2, 3), 100, dtype=np.uint8)))
            np.testing.assert_array_equal(second, np.full((2, 2, 4), (100, 100, 100, 200), dtype=np.uint8))
            self.assertTrue((output / "recolored_contact_sheet.png").is_file())


if __name__ == "__main__":
    unittest.main()
