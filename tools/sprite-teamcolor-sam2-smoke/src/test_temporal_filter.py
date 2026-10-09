import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gui_controller import (
    MaskSourceState,
    bidirectional_components_are_consistent,
    combine_mask_sources,
    connected_components_8,
    evaluate_component_continuity,
    format_temporal_filter_diagnostics,
    make_temporal_filter_contact_sheet,
    mask_geometry,
    recolor_frame_sequence,
    save_results,
    temporal_filter_pre_gate_masks,
)


SHAPE = (40, 40)


def empty(shape: tuple[int, int] = SHAPE) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def rectangle(
    x: int,
    y: int,
    width: int = 4,
    height: int = 4,
    shape: tuple[int, int] = SHAPE,
) -> np.ndarray:
    output = empty(shape)
    output[y : y + height, x : x + width] = True
    return output


class ConnectedComponentTests(unittest.TestCase):
    def test_eight_neighbor_diagonal_pixels_are_one_component(self) -> None:
        source = empty((6, 6))
        source[1, 1] = True
        source[2, 2] = True

        components = connected_components_8(source)

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].area, 2)

    def test_component_geometry_and_mask_are_exact(self) -> None:
        source = empty((8, 9))
        source[2:4, 3:6] = True
        source[6, 8] = True
        snapshot = source.copy()

        components = connected_components_8(source)

        self.assertEqual(len(components), 2)
        component = components[0]
        self.assertEqual(component.area, 6)
        self.assertEqual(component.centroid, (4.0, 2.5))
        self.assertEqual(component.bbox, (3, 2, 5, 3))
        self.assertEqual((component.width, component.height), (3, 2))
        np.testing.assert_array_equal(component.mask, rectangle(3, 2, 3, 2, (8, 9)))
        np.testing.assert_array_equal(source, snapshot)
        self.assertFalse(np.shares_memory(component.mask, source))


class ContinuityTests(unittest.TestCase):
    def test_directional_pass_selects_the_near_component(self) -> None:
        regular = [rectangle(2, 10), empty(), rectangle(10, 10)]
        near = rectangle(6, 10)
        far = rectangle(30, 30)
        raw = [empty(), near | far, empty()]

        result = temporal_filter_pre_gate_masks(regular, raw)

        np.testing.assert_array_equal(result.filtered_masks[1], near)
        self.assertEqual(result.frames[1].left.selected_component.bbox, (6, 10, 9, 13))

    def test_near_component_scores_higher_than_far_component(self) -> None:
        anchor = mask_geometry(rectangle(3, 10))
        near = connected_components_8(rectangle(7, 10))[0]
        far = connected_components_8(rectangle(30, 30))[0]

        near_result = evaluate_component_continuity([anchor], near)
        far_result = evaluate_component_continuity([anchor], far)

        self.assertTrue(near_result.accepted)
        self.assertFalse(far_result.accepted)
        self.assertGreater(near_result.continuity_score, far_result.continuity_score)
        self.assertIn("REJECT_CENTER_SHIFT", far_result.rejection_reasons)

    def test_area_growth_and_bbox_change_are_rejected(self) -> None:
        anchor = mask_geometry(rectangle(10, 10, 3, 3))
        large = connected_components_8(rectangle(8, 8, 9, 9))[0]

        result = evaluate_component_continuity([anchor], large)

        self.assertFalse(result.accepted)
        self.assertIn("REJECT_AREA_GROWTH", result.rejection_reasons)
        self.assertIn("REJECT_BBOX_CHANGE", result.rejection_reasons)

    def test_natural_shrink_and_motion_are_allowed(self) -> None:
        first = mask_geometry(rectangle(4, 10, 6, 6))
        moved = connected_components_8(rectangle(7, 10, 5, 5))[0]
        shrunk = connected_components_8(rectangle(10, 10, 4, 4))[0]

        first_move = evaluate_component_continuity([first], moved)
        second_move = evaluate_component_continuity([first, moved], shrunk)

        self.assertTrue(first_move.accepted)
        self.assertTrue(second_move.accepted)
        self.assertEqual(second_move.predicted_center, (11.5, 11.5))

    def test_missing_candidate_rejects_and_stops_directional_chain(self) -> None:
        regular = [rectangle(2, 10), empty(), empty(), empty(), rectangle(18, 10)]
        raw = [empty(), rectangle(6, 10), empty(), rectangle(14, 10), empty()]

        result = temporal_filter_pre_gate_masks(regular, raw)

        self.assertEqual(result.frames[2].left.result, "REJECT")
        self.assertIn("REJECT_NO_COMPONENT", result.frames[2].left.reasons)
        self.assertEqual(result.frames[3].left.result, "STOPPED")
        self.assertIn("STOPPED_AFTER_PREVIOUS_FAILURE", result.frames[3].left.reasons)
        self.assertEqual(result.frames[2].right.result, "REJECT")
        self.assertEqual(result.frames[1].right.result, "STOPPED")


class BidirectionalTemporalFilterTests(unittest.TestCase):
    def test_left_and_right_passes_follow_natural_chains_and_stop_at_drift(self) -> None:
        regular = [rectangle(2, 10)] + [empty() for _ in range(7)] + [rectangle(30, 10)]
        raw = [
            empty(),
            rectangle(5, 10),
            rectangle(8, 10),
            rectangle(11, 10),
            rectangle(18, 28, 10, 8),
            rectangle(21, 10),
            rectangle(24, 10),
            rectangle(27, 10),
            empty(),
        ]

        result = temporal_filter_pre_gate_masks(regular, raw)

        self.assertEqual([result.frames[index].left.result for index in (1, 2, 3)], ["ACCEPT"] * 3)
        self.assertEqual(result.frames[4].left.result, "REJECT")
        self.assertEqual(result.frames[5].left.result, "STOPPED")
        self.assertEqual([result.frames[index].right.result for index in (7, 6, 5)], ["ACCEPT"] * 3)
        self.assertEqual(result.frames[4].right.result, "REJECT")
        self.assertEqual(result.frames[3].right.result, "STOPPED")
        self.assertTrue(all(result.filtered_masks[index].any() for index in (1, 2, 3, 5, 6, 7)))
        self.assertFalse(result.filtered_masks[4].any())

    def test_one_direction_acceptance_is_retained(self) -> None:
        regular = [rectangle(2, 10), empty(), empty()]
        raw = [empty(), rectangle(5, 10), rectangle(8, 10)]

        result = temporal_filter_pre_gate_masks(regular, raw)

        self.assertEqual(result.frames[1].final_result, "ACCEPT_LEFT")
        np.testing.assert_array_equal(result.filtered_masks[1], raw[1])

    def test_matching_bidirectional_candidates_are_unioned(self) -> None:
        regular = [rectangle(2, 10), empty(), rectangle(10, 10)]
        raw = [empty(), rectangle(6, 10), empty()]

        result = temporal_filter_pre_gate_masks(regular, raw)

        self.assertEqual(result.frames[1].final_result, "ACCEPT_BOTH")
        np.testing.assert_array_equal(result.filtered_masks[1], raw[1])

    def test_different_bidirectional_targets_are_ambiguous_and_rejected(self) -> None:
        regular = [rectangle(2, 10), empty(), rectangle(30, 10)]
        raw = [empty(), rectangle(6, 10) | rectangle(26, 10), empty()]

        result = temporal_filter_pre_gate_masks(regular, raw)

        self.assertEqual(result.frames[1].final_result, "AMBIGUOUS")
        self.assertIn("REJECT_AMBIGUOUS_DIRECTIONS", result.frames[1].final_reasons)
        self.assertFalse(result.filtered_masks[1].any())

    def test_consistency_comparison_is_symmetric(self) -> None:
        left = rectangle(10, 10)
        close = rectangle(12, 10)
        far = rectangle(30, 30)
        self.assertTrue(bidirectional_components_are_consistent(left, close))
        self.assertTrue(bidirectional_components_are_consistent(close, left))
        self.assertFalse(bidirectional_components_are_consistent(left, far))

    def test_trusted_frames_emit_no_filtered_pre_gate_pixels(self) -> None:
        regular = [rectangle(2, 10), empty(), rectangle(10, 10)]
        reverse = [empty(), rectangle(6, 10), empty()]
        raw = [rectangle(1, 1), rectangle(6, 10), rectangle(30, 30)]

        result = temporal_filter_pre_gate_masks(regular, raw, reverse)

        self.assertTrue(all(frame.trusted_anchor for frame in result.frames))
        self.assertTrue(all(not mask.any() for mask in result.filtered_masks))

    def test_synthetic_occlusion_regression_keeps_edges_and_rejects_middle(self) -> None:
        natural_left = [rectangle(5 + 3 * index, 10) for index in range(3)]
        unrelated = [rectangle(9 + index, 27, 11, 8) for index in range(5)]
        natural_right = [rectangle(22 + 3 * index, 10) for index in range(3)]
        regular = [rectangle(2, 10)] + [empty() for _ in range(11)] + [rectangle(31, 10)]
        raw = [empty(), *natural_left, *unrelated, *natural_right, empty()]

        result = temporal_filter_pre_gate_masks(regular, raw)

        accepted = [index for index, mask in enumerate(result.filtered_masks) if mask.any()]
        self.assertEqual(accepted, [1, 2, 3, 9, 10, 11])
        self.assertTrue(all(not result.filtered_masks[index].any() for index in range(4, 9)))


class TemporalMaskSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.regular = [rectangle(1, 1, 2, 2, (8, 8))]
        self.raw = [rectangle(3, 1, 2, 2, (8, 8))]
        self.filtered = [rectangle(5, 1, 2, 2, (8, 8))]
        self.reverse = [rectangle(1, 5, 2, 2, (8, 8))]

    def test_raw_and_filtered_modes_are_mutually_exclusive(self) -> None:
        state = MaskSourceState(self.regular)
        state.set_pre_gate_masks(self.raw)
        state.set_filtered_pre_gate_masks(self.filtered)
        state.set_enabled(pre_gate=True)
        np.testing.assert_array_equal(state.preview_masks[0], self.regular[0] | self.raw[0])

        state.set_pre_gate_filter_enabled(True)

        np.testing.assert_array_equal(state.preview_masks[0], self.regular[0] | self.filtered[0])
        self.assertFalse(np.logical_and(state.preview_masks[0], self.raw[0]).any())
        self.assertEqual(state.source_label, "Regular + Pre-Gate (Auto-filtered)")

    def test_pre_gate_off_excludes_both_raw_and_filtered(self) -> None:
        state = MaskSourceState(self.regular)
        state.set_pre_gate_masks(self.raw)
        state.set_filtered_pre_gate_masks(self.filtered)
        state.set_pre_gate_filter_enabled(True)
        state.set_enabled(pre_gate=False)

        np.testing.assert_array_equal(state.preview_masks[0], self.regular[0])

    def test_reverse_toggle_rebuilds_without_baking_union(self) -> None:
        state = MaskSourceState(self.regular)
        state.set_pre_gate_masks(self.raw)
        state.set_filtered_pre_gate_masks(self.filtered)
        state.set_reverse_masks(self.reverse)
        state.set_enabled(pre_gate=True, reverse=True)
        state.set_pre_gate_filter_enabled(True)
        np.testing.assert_array_equal(
            state.preview_masks[0], self.regular[0] | self.filtered[0] | self.reverse[0]
        )

        state.set_enabled(reverse=False)
        np.testing.assert_array_equal(state.preview_masks[0], self.regular[0] | self.filtered[0])
        state.set_pre_gate_filter_enabled(False)
        np.testing.assert_array_equal(state.preview_masks[0], self.regular[0] | self.raw[0])

    def test_combination_does_not_mutate_any_source(self) -> None:
        sources = (self.regular, self.raw, self.filtered, self.reverse)
        snapshots = [[mask.copy() for mask in source] for source in sources]

        combined = combine_mask_sources(
            self.regular,
            self.raw,
            self.reverse,
            self.filtered,
            include_pre_gate=True,
            include_reverse=True,
            use_filtered_pre_gate=True,
        )

        for source, snapshot in zip(sources, snapshots, strict=True):
            np.testing.assert_array_equal(source[0], snapshot[0])
        self.assertTrue(all(not np.shares_memory(combined[0], source[0]) for source in sources))

    def test_filter_toggle_requires_readoption_and_adopted_mask_drives_output(self) -> None:
        state = MaskSourceState(self.regular)
        state.set_pre_gate_masks(self.raw)
        state.set_filtered_pre_gate_masks(self.filtered)
        state.set_enabled(pre_gate=True)
        raw_active = state.adopt_preview()[0]
        state.set_pre_gate_filter_enabled(True)

        np.testing.assert_array_equal(state.active_masks[0], raw_active)
        self.assertTrue(state.preview_dirty)
        filtered_active = state.adopt_preview()
        self.assertFalse(state.preview_dirty)
        np.testing.assert_array_equal(filtered_active[0], self.regular[0] | self.filtered[0])

        frame = Image.new("RGBA", (8, 8), (120, 120, 120, 255))
        recolored = recolor_frame_sequence([frame], filtered_active, (210, 48, 48))
        recolored_pixels = np.asarray(recolored[0])
        self.assertFalse(np.array_equal(recolored_pixels[1, 1, :3], (120, 120, 120)))
        self.assertFalse(np.array_equal(recolored_pixels[1, 5, :3], (120, 120, 120)))
        np.testing.assert_array_equal(recolored_pixels[1, 3, :3], (120, 120, 120))
        with tempfile.TemporaryDirectory() as temporary:
            outputs = save_results(
                Path(temporary), [frame], filtered_active, recolored, columns=1
            )
            with Image.open(outputs["masks"] / "frame_000.png") as saved:
                exported = np.asarray(saved).copy() > 0
        np.testing.assert_array_equal(exported, filtered_active[0])

        state.set_pre_gate_filter_enabled(False)
        self.assertTrue(state.preview_dirty)
        np.testing.assert_array_equal(state.active_masks[0], filtered_active[0])

    def test_diagnostics_and_contact_sheet_are_renderable(self) -> None:
        regular = [rectangle(2, 10), empty(), rectangle(10, 10)]
        raw = [empty(), rectangle(6, 10), empty()]
        result = temporal_filter_pre_gate_masks(regular, raw)

        report = format_temporal_filter_diagnostics(result)
        contact = make_temporal_filter_contact_sheet(
            [Image.new("RGBA", (40, 40), (80, 80, 80, 255)) for _ in range(3)],
            result,
            columns=3,
        )

        self.assertIn("frame_001", report)
        self.assertIn("final: ACCEPT_BOTH", report)
        self.assertEqual(contact.size, (120, 58))


if __name__ == "__main__":
    unittest.main()
