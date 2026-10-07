import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gui_controller import (
    LassoSelectionState,
    MAX_ZOOM,
    MIN_ZOOM,
    PromptState,
    SheetLayout,
    choose_display_scale,
    clamp_zoom,
    cursor_centered_zoom_offset,
    display_to_frame,
    ellipse_to_mask,
    extract_row_frames,
    frame_output_names,
    make_selection_overlay,
    parse_team_color,
    pan_scroll_offset,
    polygon_to_mask,
    percentage_to_unit,
    rectangle_to_mask,
    recolor_frame_sequence,
    save_results,
    select_all_mask,
    shape_bounds,
    viewport_to_frame,
)


class GuiControllerTests(unittest.TestCase):
    def test_zoom_clamps_to_minimum_and_maximum(self) -> None:
        self.assertEqual(clamp_zoom(-10), MIN_ZOOM)
        self.assertEqual(clamp_zoom(7), 7)
        self.assertEqual(clamp_zoom(100), MAX_ZOOM)

    def test_recolor_percentages_map_to_unit_interval(self) -> None:
        self.assertEqual(percentage_to_unit(0), 0.0)
        self.assertEqual(percentage_to_unit(85), 0.85)
        self.assertEqual(percentage_to_unit(100), 1.0)
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            percentage_to_unit(101)

    def test_zoomed_and_scrolled_coordinates_map_to_same_frame_pixel(self) -> None:
        expected = (40, 80)
        for zoom in (3, 4, 8):
            with self.subTest(zoom=zoom):
                scroll = (17.0, 29.0)
                viewport = (expected[0] * zoom - scroll[0], expected[1] * zoom - scroll[1])
                self.assertEqual(viewport_to_frame(*viewport, zoom, scroll, (128, 128)), expected)

    def test_zoomed_coordinates_support_non_square_frames(self) -> None:
        self.assertEqual(viewport_to_frame(175, 75, 5, (25, 25), (80, 40)), (40, 20))

    def test_cursor_centered_zoom_keeps_source_anchor_stable(self) -> None:
        cursor = (20.0, 30.0)
        old_scroll = (100.0, 50.0)
        new_scroll = cursor_centered_zoom_offset(
            cursor,
            old_zoom=4,
            new_zoom=8,
            old_scroll=old_scroll,
            viewport_size=(100, 100),
            frame_size=(128, 128),
        )

        old_anchor = ((old_scroll[0] + cursor[0]) / 4, (old_scroll[1] + cursor[1]) / 4)
        new_anchor = ((new_scroll[0] + cursor[0]) / 8, (new_scroll[1] + cursor[1]) / 8)
        self.assertEqual(new_anchor, old_anchor)

    def test_arrow_pan_offset_is_clamped_to_content(self) -> None:
        self.assertEqual(pan_scroll_offset((100, 100), (-30, 40), (1024, 1024), (200, 200)), (70, 140))
        self.assertEqual(pan_scroll_offset((0, 0), (-50, -50), (1024, 1024), (200, 200)), (0, 0))
        self.assertEqual(pan_scroll_offset((800, 800), (100, 100), (1024, 1024), (200, 200)), (824, 824))

    def test_main_and_mask_editor_can_share_selection_state(self) -> None:
        main_selection = LassoSelectionState((16, 16))
        editor_selection = main_selection

        editor_selection.apply(((2, 2), (10, 2), (10, 10), (2, 10)), "add")

        self.assertTrue(main_selection.mask[5, 5])
        self.assertIs(main_selection, editor_selection)

    def test_polygon_to_mask_selects_inside_not_outside(self) -> None:
        mask = polygon_to_mask(((2, 2), (7, 2), (7, 7), (2, 7)), (10, 10))

        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[4, 4])
        self.assertFalse(mask[0, 0])

    def test_lasso_add_subtract_undo_and_clear(self) -> None:
        selection = LassoSelectionState((12, 10))
        selection.apply(((1, 1), (9, 1), (9, 8), (1, 8)), "add")
        added = selection.mask
        self.assertTrue(added[4, 4])

        selection.apply(((3, 3), (6, 3), (6, 6), (3, 6)), "subtract")
        self.assertFalse(selection.mask[4, 4])
        np.testing.assert_array_equal(selection.undo(), added)

        selection.clear()
        self.assertTrue(selection.is_empty)
        np.testing.assert_array_equal(selection.undo(), added)

    def test_rectangle_selects_inside_not_outside_and_reverse_drag(self) -> None:
        forward = rectangle_to_mask((2, 3), (7, 8), (12, 11))
        reverse = rectangle_to_mask((7, 8), (2, 3), (12, 11))

        self.assertTrue(forward[5, 4])
        self.assertFalse(forward[2, 4])
        self.assertFalse(forward[5, 8])
        np.testing.assert_array_equal(reverse, forward)

    def test_rectangle_supports_one_pixel_width_and_square_constraint(self) -> None:
        one_pixel_wide = rectangle_to_mask((3, 2), (3, 7), (10, 10))
        square = rectangle_to_mask((2, 2), (8, 5), (12, 12), constrain_square=True)

        self.assertEqual(int(one_pixel_wide.sum()), 6)
        self.assertTrue(one_pixel_wide[4, 3])
        self.assertFalse(one_pixel_wide[4, 4])
        self.assertEqual(shape_bounds((2, 2), (8, 5), (12, 12), True), (2, 2, 8, 8))
        self.assertEqual(int(square.sum()), 49)

    def test_rectangle_add_subtract_share_selection_history(self) -> None:
        selection = LassoSelectionState((12, 12))
        outer = rectangle_to_mask((1, 1), (10, 10), selection.frame_size)
        inner = rectangle_to_mask((4, 4), (7, 7), selection.frame_size)

        selection.apply_mask(outer, "add")
        added = selection.mask
        selection.apply_mask(inner, "subtract")

        self.assertFalse(selection.mask[5, 5])
        np.testing.assert_array_equal(selection.undo(), added)

    def test_ellipse_selects_inside_not_outside_and_reverse_drag(self) -> None:
        forward = ellipse_to_mask((2, 2), (9, 7), (12, 10))
        reverse = ellipse_to_mask((9, 7), (2, 2), (12, 10))

        self.assertTrue(forward[4, 5])
        self.assertFalse(forward[1, 5])
        self.assertFalse(forward[2, 2])
        np.testing.assert_array_equal(reverse, forward)

    def test_ellipse_circle_constraint_and_add_subtract(self) -> None:
        circle = ellipse_to_mask((2, 2), (9, 6), (12, 12), constrain_circle=True)
        selection = LassoSelectionState((12, 12))
        selection.apply_mask(circle, "add")
        added = selection.mask
        cutout = ellipse_to_mask((3, 3), (5, 5), selection.frame_size)
        selection.apply_mask(cutout, "subtract")

        self.assertEqual(shape_bounds((2, 2), (9, 6), (12, 12), True), (2, 2, 9, 9))
        self.assertTrue(added[4, 4])
        self.assertFalse(selection.mask[4, 4])
        np.testing.assert_array_equal(selection.undo(), added)

    def test_freehand_rectangle_ellipse_use_one_undo_stack(self) -> None:
        selection = LassoSelectionState((16, 16))
        selection.apply(((1, 1), (6, 1), (6, 6), (1, 6)), "add")
        freehand = selection.mask
        selection.apply_mask(rectangle_to_mask((8, 1), (12, 5), selection.frame_size), "add")
        rectangle = selection.mask
        selection.apply_mask(ellipse_to_mask((5, 8), (10, 13), selection.frame_size), "add")

        np.testing.assert_array_equal(selection.undo(), rectangle)
        np.testing.assert_array_equal(selection.undo(), freehand)
        self.assertTrue(selection.clear().sum() == 0)
        np.testing.assert_array_equal(selection.undo(), freehand)

    def test_shape_coordinates_remain_correct_after_zoom_and_scroll(self) -> None:
        start = viewport_to_frame(9, 8, 4, (31, 52), (128, 128))
        end = viewport_to_frame(89, 68, 4, (31, 52), (128, 128))
        rectangle = rectangle_to_mask(start, end, (128, 128))
        self.assertEqual(start, (10, 15))
        self.assertEqual(end, (30, 30))
        self.assertTrue(rectangle[20, 20])

        ellipse_start = viewport_to_frame(5, 7, 8, (75, 73), (128, 128))
        ellipse_end = viewport_to_frame(165, 167, 8, (75, 73), (128, 128))
        ellipse = ellipse_to_mask(ellipse_start, ellipse_end, (128, 128))
        self.assertEqual((ellipse_start, ellipse_end), ((10, 10), (30, 30)))
        self.assertTrue(ellipse[20, 20])

    def test_select_all_uses_only_visible_rgba_pixels(self) -> None:
        pixels = np.zeros((3, 4, 4), dtype=np.uint8)
        pixels[0, 1, 3] = 1
        pixels[2, 3, 3] = 255

        frame0_mask = select_all_mask(Image.fromarray(pixels, "RGBA"))

        self.assertEqual(int(frame0_mask.sum()), 2)
        self.assertTrue(frame0_mask[0, 1])
        self.assertTrue(frame0_mask[2, 3])
        self.assertFalse(frame0_mask[1, 2])

    def test_select_all_uses_full_rgb_frame(self) -> None:
        frame0_mask = select_all_mask(Image.new("RGB", (5, 3), "black"))

        self.assertEqual(frame0_mask.shape, (3, 5))
        self.assertTrue(frame0_mask.all())

    def test_polygon_mask_supports_non_128_frame(self) -> None:
        mask = polygon_to_mask(((5, 3), (24, 3), (24, 12), (5, 12)), (30, 16))

        self.assertEqual(mask.shape, (16, 30))
        self.assertTrue(mask[8, 15])
        self.assertFalse(mask[15, 29])

    def test_selection_overlay_does_not_modify_source(self) -> None:
        source = Image.new("RGBA", (5, 5), (100, 100, 100, 255))
        before = np.asarray(source).copy()
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True

        overlay = np.asarray(make_selection_overlay(source, mask))

        np.testing.assert_array_equal(np.asarray(source), before)
        np.testing.assert_array_equal(overlay[1, 1, :3], np.array([255, 220, 0]))
        np.testing.assert_array_equal(overlay[0, 0], before[0, 0])

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
        for scale in (3, 4):
            with self.subTest(scale=scale):
                display_size = (128 * scale, 128 * scale)
                self.assertEqual(display_to_frame(0, 0, display_size, (128, 128)), (0, 0))
                self.assertEqual(
                    display_to_frame(display_size[0] - 1, display_size[1] - 1, display_size, (128, 128)),
                    (127, 127),
                )
                self.assertEqual(display_to_frame(40 * scale, 80 * scale, display_size, (128, 128)), (40, 80))
        self.assertEqual(display_to_frame(299, 149, (300, 150), (100, 50)), (99, 49))

    def test_display_scale_prefers_four_but_shrinks_to_fit(self) -> None:
        self.assertEqual(choose_display_scale((128, 128), (800, 600)), 4)
        self.assertEqual(choose_display_scale((128, 128), (500, 400)), 3)
        self.assertEqual(choose_display_scale((256, 128), (700, 400)), 2)
        self.assertEqual(choose_display_scale((256, 128), (200, 100)), 1)

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
