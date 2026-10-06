from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from core import (
    FRAME_SIZE,
    TEAM_COLORS,
    PromptPoint,
    apply_row_masks,
    clip_mask_to_alpha,
    infer_grid,
    recolor_rgba,
    save_team_variants,
)


class CoreTests(unittest.TestCase):
    def test_team_colors_match_tactical_hub(self) -> None:
        self.assertEqual(
            TEAM_COLORS,
            {
                "red": "#d94a4a",
                "blue": "#3e7bd8",
                "green": "#36a166",
                "yellow": "#c58a2b",
            },
        )

    def test_grid_inference(self) -> None:
        image = Image.new("RGBA", (1920, 1024), (0, 0, 0, 0))
        grid = infer_grid(image)
        self.assertEqual(grid.columns, 15)
        self.assertEqual(grid.rows, 8)

    def test_grid_rejects_non_multiple(self) -> None:
        with self.assertRaises(ValueError):
            infer_grid(Image.new("RGBA", (100, 128), (0, 0, 0, 0)))

    def test_prompt_validation(self) -> None:
        PromptPoint(0, 0, 1)
        PromptPoint(127, 127, 0)
        with self.assertRaises(ValueError):
            PromptPoint(128, 0, 1)
        with self.assertRaises(ValueError):
            PromptPoint(10, 10, 2)

    def test_mask_clips_to_alpha(self) -> None:
        frame = Image.new("RGBA", (FRAME_SIZE, FRAME_SIZE), (100, 100, 100, 0))
        frame.putpixel((20, 30), (100, 100, 100, 255))
        mask = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        clipped = clip_mask_to_alpha(frame, mask)
        self.assertEqual(int(clipped.sum()), 1)
        self.assertTrue(clipped[30, 20])

    def test_recolor_changes_only_mask_and_preserves_alpha(self) -> None:
        frame = Image.new("RGBA", (FRAME_SIZE, FRAME_SIZE), (110, 110, 110, 255))
        frame.putpixel((4, 4), (10, 10, 10, 255))
        before = np.asarray(frame).copy()

        mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        mask[20:30, 20:30] = True
        recolored = np.asarray(
            recolor_rgba(frame, mask, TEAM_COLORS["red"], preserve_outline=True)
        )

        self.assertTrue(np.array_equal(before[..., 3], recolored[..., 3]))
        self.assertTrue(np.array_equal(before[0:10, 0:10], recolored[0:10, 0:10]))
        self.assertFalse(np.array_equal(before[25, 25, :3], recolored[25, 25, :3]))

    def test_recolor_keeps_light_dark_order(self) -> None:
        frame = Image.new("RGBA", (FRAME_SIZE, FRAME_SIZE), (0, 0, 0, 0))
        frame.putpixel((10, 10), (45, 45, 45, 255))
        frame.putpixel((11, 10), (130, 130, 130, 255))
        frame.putpixel((12, 10), (225, 225, 225, 255))

        mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        mask[10, 10:13] = True
        out = np.asarray(
            recolor_rgba(
                frame,
                mask,
                TEAM_COLORS["blue"],
                preserve_outline=False,
            )
        )

        weights = np.array([0.2126, 0.7152, 0.0722])
        lumas = [float(np.dot(out[10, x, :3], weights)) for x in (10, 11, 12)]
        self.assertLess(lumas[0], lumas[1])
        self.assertLess(lumas[1], lumas[2])

    def test_apply_row_masks_leaves_other_row_unchanged(self) -> None:
        sheet = Image.new("RGBA", (FRAME_SIZE * 2, FRAME_SIZE * 2), (120, 120, 120, 255))
        masks = []
        for _ in range(2):
            mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
            mask[30:40, 30:40] = True
            masks.append(mask)

        out = apply_row_masks(sheet, 0, masks, TEAM_COLORS["green"])
        before = np.asarray(sheet)
        after = np.asarray(out)
        self.assertTrue(np.array_equal(before[FRAME_SIZE:, :, :], after[FRAME_SIZE:, :, :]))
        self.assertFalse(np.array_equal(before[:FRAME_SIZE, :, :], after[:FRAME_SIZE, :, :]))

    def test_save_team_variants_writes_four_pngs(self) -> None:
        sheet = Image.new("RGBA", (FRAME_SIZE, FRAME_SIZE), (150, 150, 150, 255))
        mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
        mask[20:40, 20:40] = True

        with tempfile.TemporaryDirectory() as temp:
            written = save_team_variants(
                sheet,
                {0: [mask]},
                temp,
                "unit",
            )
            self.assertEqual(set(written), set(TEAM_COLORS))
            for path in written.values():
                self.assertTrue(Path(path).is_file())
                self.assertEqual(Image.open(path).size, (FRAME_SIZE, FRAME_SIZE))


if __name__ == "__main__":
    unittest.main()
