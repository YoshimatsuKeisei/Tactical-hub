import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gui_controller import (
    PromptState,
    SheetLayout,
    display_to_frame,
    extract_row_frames,
    frame_output_names,
    parse_team_color,
    recolor_frame_sequence,
    save_results,
)


class GuiControllerTests(unittest.TestCase):
    def test_extracts_requested_row_and_frame_count(self) -> None:
        pixels = np.zeros((4, 6, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255
        for row in range(2):
            for column in range(3):
                pixels[row * 2 : (row + 1) * 2, column * 2 : (column + 1) * 2, 0] = row * 10 + column
        sprite = Image.fromarray(pixels)
        layout = SheetLayout(2, 2, 3, 2, target_row=1, frame_count=2)

        frames = extract_row_frames(sprite, layout)

        self.assertEqual([frame.size for frame in frames], [(2, 2), (2, 2)])
        self.assertEqual([np.asarray(frame)[0, 0, 0] for frame in frames], [10, 11])

    def test_maps_display_coordinates_to_frame_coordinates(self) -> None:
        self.assertEqual(display_to_frame(0, 0, (512, 512), (128, 128)), (0, 0))
        self.assertEqual(display_to_frame(511, 511, (512, 512), (128, 128)), (127, 127))
        self.assertEqual(display_to_frame(162, 322, (512, 512), (128, 128)), (40, 80))

    def test_layout_rejects_invalid_row_and_sheet_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "Target row"):
            SheetLayout(2, 2, 3, 2, target_row=2, frame_count=2).validate((6, 4))
        with self.assertRaisesRegex(ValueError, "does not match"):
            SheetLayout(2, 2, 3, 2, target_row=0, frame_count=2).validate((5, 4))

    def test_prompt_state_add_undo_clear_and_validation(self) -> None:
        prompts = PromptState((8, 8))
        positive = prompts.add(2, 3, 1)
        prompts.add(5, 6, 0)
        self.assertTrue(prompts.has_positive)
        self.assertEqual(prompts.undo().label, 0)
        self.assertEqual(prompts.clicks, (positive,))
        with self.assertRaisesRegex(ValueError, "outside"):
            prompts.add(8, 0, 1)
        prompts.clear()
        self.assertEqual(prompts.clicks, ())

    def test_parses_presets_custom_colors_and_output_names(self) -> None:
        self.assertEqual(parse_team_color("Red"), (210, 48, 48))
        self.assertEqual(parse_team_color("Blue"), (48, 92, 210))
        self.assertEqual(parse_team_color("Custom", "#123456"), (18, 52, 86))
        with self.assertRaises(ValueError):
            parse_team_color("Custom", "")
        names = frame_output_names(15)
        self.assertEqual((names[0], names[-1]), ("frame_000.png", "frame_014.png"))

    def test_recolor_integration_and_saved_result_names(self) -> None:
        source = np.array(
            [
                [[40, 40, 40, 0], [100, 100, 100, 255]],
                [[160, 160, 160, 255], [220, 220, 220, 255]],
            ],
            dtype=np.uint8,
        )
        frames = [Image.fromarray(source.copy()), Image.fromarray(source.copy())]
        masks = [np.array([[True, True], [False, False]]), np.ones((2, 2), dtype=bool)]

        recolored = recolor_frame_sequence(frames, masks, parse_team_color("Green"))

        np.testing.assert_array_equal(np.asarray(recolored[0])[1], source[1])
        np.testing.assert_array_equal(np.asarray(recolored[0])[:, :, 3], source[:, :, 3])
        self.assertFalse(np.array_equal(np.asarray(recolored[0])[0, 1, :3], source[0, 1, :3]))

        with tempfile.TemporaryDirectory() as temporary:
            outputs = save_results(Path(temporary), frames, masks, recolored, columns=2)
            self.assertTrue((outputs["masks"] / "frame_000.png").is_file())
            self.assertTrue((outputs["recolored_frames"] / "frame_001.png").is_file())
            self.assertTrue(outputs["recolored_contact_sheet"].is_file())
            self.assertTrue(outputs["overlay_contact_sheet"].is_file())


if __name__ == "__main__":
    unittest.main()
