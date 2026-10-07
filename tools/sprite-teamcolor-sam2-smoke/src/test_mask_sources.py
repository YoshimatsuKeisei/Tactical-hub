import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gui_controller import (
    MaskSourceState,
    Sam2GuiSession,
    SamPreGateDecoderCapture,
    SamPreGateFrameDiagnostics,
    combine_mask_sources,
    derive_pre_gate_masks,
    derive_reverse_masks_from_sam_logits,
    recolor_frame_sequence,
    save_results,
)


def mask(*points: tuple[int, int], size: tuple[int, int] = (3, 3)) -> np.ndarray:
    output = np.zeros(size, dtype=bool)
    for y, x in points:
        output[y, x] = True
    return output


def rgba_frames(count: int = 15, size: tuple[int, int] = (2, 2)) -> list[Image.Image]:
    return [Image.new("RGBA", size, (100, 100, 100, 255)) for _ in range(count)]


def captured_diagnostic(frame_index: int, logits: np.ndarray) -> SamPreGateFrameDiagnostics:
    capture = SamPreGateDecoderCapture(
        decoder_call_index=frame_index,
        low_res_multimasks=logits[None, None].astype(np.float32),
        ious=np.array([[0.9]], dtype=np.float32),
        object_score_logits=np.array([[-0.5]], dtype=np.float32),
        best_mask_index=0,
        best_iou=0.9,
        object_score_logit=-0.5,
        pre_gate_logits=logits.astype(np.float32, copy=True),
    )
    return SamPreGateFrameDiagnostics(
        frame_index=frame_index,
        capture_status="captured",
        captures=(capture,),
        capture_errors=(),
        post_gate_is_no_obj=True,
        post_gate_min=-1024.0,
        post_gate_max=-1024.0,
    )


class FakeReversePredictor:
    def __init__(self) -> None:
        self.init_calls = 0
        self.propagate_calls = 0
        self.add_calls: list[dict[str, object]] = []
        self.propagate_arguments: list[tuple[object, int | None, bool]] = []
        self.reset_states: list[object] = []
        self.fail_propagation = False

    def init_state(self, **_kwargs: object) -> dict[str, object]:
        self.init_calls += 1
        return {"fresh": self.init_calls}

    def add_new_mask(self, **kwargs: object):
        recorded = dict(kwargs)
        recorded["mask"] = np.asarray(kwargs["mask"]).copy()
        self.add_calls.append(recorded)
        return kwargs["frame_idx"], [1], [np.ones((1, 2, 2), dtype=np.float32)]

    def propagate_in_video(
        self,
        inference_state: object,
        start_frame_idx: int | None = None,
        reverse: bool = False,
    ):
        self.propagate_calls += 1
        self.propagate_arguments.append((inference_state, start_frame_idx, reverse))
        if self.fail_propagation:
            raise RuntimeError("simulated Reverse failure")
        for frame_index in range(14, -1, -1):
            logits = np.full((1, 2, 2), float(frame_index - 7), dtype=np.float32)
            yield frame_index, [1], [logits]

    def reset_state(self, inference_state: object) -> None:
        self.reset_states.append(inference_state)


class MaskSourceUnionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.regular = [mask((0, 0)), mask((1, 1))]
        self.pre_gate = [mask((0, 1)), mask((1, 2))]
        self.reverse = [mask((2, 0)), mask((2, 2))]

    def test_regular_only_is_pixel_identical_and_sources_are_not_mutated(self) -> None:
        snapshots = [item.copy() for item in self.regular]

        combined = combine_mask_sources(self.regular)

        for actual, expected, source in zip(combined, snapshots, self.regular, strict=True):
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(source, expected)
            self.assertFalse(np.shares_memory(actual, source))

    def test_regular_plus_pre_gate_is_or_union(self) -> None:
        combined = combine_mask_sources(
            self.regular,
            self.pre_gate,
            include_pre_gate=True,
        )
        np.testing.assert_array_equal(combined[0], self.regular[0] | self.pre_gate[0])
        np.testing.assert_array_equal(combined[1], self.regular[1] | self.pre_gate[1])

    def test_regular_plus_reverse_is_or_union(self) -> None:
        combined = combine_mask_sources(
            self.regular,
            reverse_masks=self.reverse,
            include_reverse=True,
        )
        np.testing.assert_array_equal(combined[0], self.regular[0] | self.reverse[0])
        np.testing.assert_array_equal(combined[1], self.regular[1] | self.reverse[1])

    def test_all_three_sources_are_or_union_without_source_mutation(self) -> None:
        originals = [
            [item.copy() for item in source]
            for source in (self.regular, self.pre_gate, self.reverse)
        ]
        combined = combine_mask_sources(
            self.regular,
            self.pre_gate,
            self.reverse,
            include_pre_gate=True,
            include_reverse=True,
        )
        for index in range(2):
            np.testing.assert_array_equal(
                combined[index],
                self.regular[index] | self.pre_gate[index] | self.reverse[index],
            )
        for source, expected in zip((self.regular, self.pre_gate, self.reverse), originals, strict=True):
            for actual, snapshot in zip(source, expected, strict=True):
                np.testing.assert_array_equal(actual, snapshot)

    def test_turning_source_off_rebuilds_from_regular_without_baked_pixels(self) -> None:
        state = MaskSourceState(self.regular)
        state.set_pre_gate_masks(self.pre_gate)
        state.set_reverse_masks(self.reverse)
        state.set_enabled(pre_gate=True, reverse=True)
        state.set_enabled(pre_gate=False, reverse=True)

        for actual, regular, reverse in zip(
            state.preview_masks, self.regular, self.reverse, strict=True
        ):
            np.testing.assert_array_equal(actual, regular | reverse)

    def test_count_and_shape_mismatches_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "counts"):
            combine_mask_sources(
                self.regular,
                self.pre_gate[:1],
                include_pre_gate=True,
            )
        wrong_shape = [np.zeros((2, 2), dtype=bool), self.reverse[1]]
        with self.assertRaisesRegex(ValueError, "shape"):
            combine_mask_sources(
                self.regular,
                reverse_masks=wrong_shape,
                include_reverse=True,
            )


class MaskSourceAdoptionTests(unittest.TestCase):
    def test_toggle_changes_preview_but_not_active_until_adoption(self) -> None:
        regular = [mask((0, 0))]
        pre_gate = [mask((0, 1))]
        state = MaskSourceState(regular)
        active_before = state.active_masks[0].copy()

        state.set_pre_gate_masks(pre_gate)
        state.set_enabled(pre_gate=True)

        np.testing.assert_array_equal(state.active_masks[0], active_before)
        self.assertTrue(state.preview_dirty)
        adopted = state.adopt_preview()
        np.testing.assert_array_equal(adopted[0], regular[0] | pre_gate[0])
        self.assertFalse(state.preview_dirty)

    def test_previous_union_is_not_baked_into_regular_after_source_is_removed(self) -> None:
        regular = [mask((0, 0))]
        pre_gate = [mask((0, 1))]
        state = MaskSourceState(regular)
        state.set_pre_gate_masks(pre_gate)
        state.set_enabled(pre_gate=True)
        state.adopt_preview()

        state.set_enabled(pre_gate=False)

        np.testing.assert_array_equal(state.preview_masks[0], regular[0])
        np.testing.assert_array_equal(state.regular_masks[0], regular[0])
        self.assertTrue(state.preview_dirty)

    def test_recolor_and_export_use_the_adopted_combined_masks(self) -> None:
        frame = Image.new("RGBA", (3, 3), (120, 120, 120, 255))
        state = MaskSourceState([mask((0, 0))])
        state.set_reverse_masks([mask((2, 2))])
        state.set_enabled(reverse=True)
        adopted = state.adopt_preview()
        recolored = recolor_frame_sequence([frame], adopted, (210, 48, 48))

        with tempfile.TemporaryDirectory() as temporary:
            outputs = save_results(Path(temporary), [frame], adopted, recolored, columns=1)
            with Image.open(outputs["masks"] / "frame_000.png") as saved:
                exported = np.asarray(saved).copy() > 0

        np.testing.assert_array_equal(exported, mask((0, 0), (2, 2)))


class PreGateSourceTests(unittest.TestCase):
    def test_diagnostic_and_combined_source_use_the_same_threshold_result(self) -> None:
        frame = Image.new("RGBA", (3, 3), (100, 100, 100, 255))
        logits = np.array(
            [[-1.0, 0.5, 2.0], [-1.0, 0.5, 2.0], [-1.0, 0.5, 2.0]],
            dtype=np.float32,
        )
        diagnostics = [captured_diagnostic(0, logits)]

        lower = derive_pre_gate_masks(diagnostics, [frame], threshold=0.0, fill_holes=False)
        higher = derive_pre_gate_masks(diagnostics, [frame], threshold=1.0, fill_holes=False)

        self.assertEqual(int(lower[0].sum()), 6)
        self.assertEqual(int(higher[0].sum()), 3)
        self.assertFalse(np.array_equal(lower[0], higher[0]))

    def test_shared_diagnostic_hole_fill_setting_changes_pre_gate_source(self) -> None:
        frame = Image.new("RGBA", (3, 3), (100, 100, 100, 255))
        logits = np.ones((3, 3), dtype=np.float32)
        logits[1, 1] = -1.0
        diagnostics = [captured_diagnostic(0, logits)]

        unfilled = derive_pre_gate_masks(diagnostics, [frame], fill_holes=False)
        filled = derive_pre_gate_masks(diagnostics, [frame], fill_holes=True)

        self.assertFalse(unfilled[0][1, 1])
        self.assertTrue(filled[0][1, 1])

    def test_pre_gate_off_contributes_no_pixels_and_runs_no_inference(self) -> None:
        regular = [mask((0, 0))]
        pre_gate = [mask((0, 1))]
        state = MaskSourceState(regular)
        state.set_pre_gate_masks(pre_gate)

        state.set_enabled(pre_gate=False)

        np.testing.assert_array_equal(state.preview_masks[0], regular[0])


class ReverseTrackingTests(unittest.TestCase):
    def make_session(self, temporary: str) -> tuple[Sam2GuiSession, FakeReversePredictor, object]:
        checkpoint = Path(temporary) / "checkpoint.pt"
        checkpoint.touch()
        session = Sam2GuiSession(rgba_frames(), checkpoint, work_size=2)
        predictor = FakeReversePredictor()
        forward_state = {"forward": [1, 2, 3]}
        session.predictor = predictor
        session.inference_state = forward_state
        session.device = torch.device("cpu")
        session.sam_frames_dir = Path(temporary)
        return session, predictor, forward_state

    def test_reverse_uses_frame14_regular_anchor_fresh_state_and_official_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, predictor, forward_state = self.make_session(temporary)
            regular = [np.zeros((2, 2), dtype=bool) for _ in range(15)]
            regular[14][0, 1] = True
            forward_snapshot = {"forward": list(forward_state["forward"])}

            retained = session.track_reverse_logits(regular)

            self.assertEqual(predictor.init_calls, 1)
            self.assertIsNot(predictor.propagate_arguments[0][0], forward_state)
            self.assertEqual(predictor.propagate_arguments[0][1:], (14, True))
            self.assertEqual(predictor.add_calls[0]["frame_idx"], 14)
            np.testing.assert_array_equal(predictor.add_calls[0]["mask"], regular[14])
            self.assertEqual(forward_state, forward_snapshot)
            self.assertEqual([float(item[0, 0]) for item in retained], list(map(float, range(-7, 8))))
            self.assertEqual(len(predictor.reset_states), 1)
            self.assertIs(predictor.reset_states[0], predictor.propagate_arguments[0][0])

    def test_empty_frame14_regular_mask_rejects_before_reverse_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, predictor, _forward_state = self.make_session(temporary)
            regular = [np.zeros((2, 2), dtype=bool) for _ in range(15)]

            with self.assertRaisesRegex(ValueError, "non-empty Regular mask on frame 014"):
                session.track_reverse_logits(regular)

            self.assertEqual(predictor.init_calls, 0)
            self.assertEqual(predictor.propagate_calls, 0)
            self.assertIsNone(session.reverse_raw_logits)

    def test_reverse_frame14_is_exact_authoritative_regular_anchor(self) -> None:
        frames = rgba_frames()
        raw_logits = [np.ones((2, 2), dtype=np.float32) for _ in range(15)]
        regular = [np.zeros((2, 2), dtype=bool) for _ in range(15)]
        regular[14][1, 0] = True

        raw, active = derive_reverse_masks_from_sam_logits(
            raw_logits,
            frames,
            regular,
            threshold=0.0,
            fill_holes=False,
        )

        np.testing.assert_array_equal(raw[14], regular[14])
        np.testing.assert_array_equal(active[14], regular[14])

    def test_reverse_raw_logits_can_be_rethresholded_without_propagation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, predictor, _forward_state = self.make_session(temporary)
            regular = [np.zeros((2, 2), dtype=bool) for _ in range(15)]
            regular[14][0, 0] = True
            raw_logits = session.track_reverse_logits(regular)

            _low_raw, low = derive_reverse_masks_from_sam_logits(
                raw_logits, session.frames, regular, threshold=-2.0, fill_holes=False
            )
            _high_raw, high = derive_reverse_masks_from_sam_logits(
                raw_logits, session.frames, regular, threshold=2.0, fill_holes=False
            )

            self.assertGreater(int(low[6].sum()), int(high[6].sum()))
            self.assertEqual(predictor.propagate_calls, 1)
            self.assertIsNotNone(session.reverse_raw_logits)

    def test_reverse_toggle_reuses_cached_masks_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, predictor, _forward_state = self.make_session(temporary)
            regular = [np.zeros((2, 2), dtype=bool) for _ in range(15)]
            regular[14][0, 0] = True
            raw_logits = session.track_reverse_logits(regular)
            _raw, reverse = derive_reverse_masks_from_sam_logits(
                raw_logits,
                session.frames,
                regular,
                threshold=0.0,
                fill_holes=False,
            )
            state = MaskSourceState(regular)
            state.set_reverse_masks(reverse)

            state.set_enabled(reverse=True)
            state.set_enabled(reverse=False)
            state.set_enabled(reverse=True)

            self.assertEqual(predictor.propagate_calls, 1)
            for actual, base, cached in zip(
                state.preview_masks, regular, reverse, strict=True
            ):
                np.testing.assert_array_equal(actual, base | cached)

    def test_reverse_failure_resets_fresh_state_and_keeps_cache_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, predictor, forward_state = self.make_session(temporary)
            predictor.fail_propagation = True
            regular = [np.zeros((2, 2), dtype=bool) for _ in range(15)]
            regular[14][0, 0] = True

            with self.assertRaisesRegex(RuntimeError, "simulated Reverse failure"):
                session.track_reverse_logits(regular)

            self.assertEqual(len(predictor.reset_states), 1)
            self.assertIsNot(predictor.reset_states[0], forward_state)
            self.assertIsNone(session.reverse_raw_logits)

    def test_session_close_clears_reverse_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, _predictor, _forward_state = self.make_session(temporary)
            session.reverse_raw_logits = [np.ones((2, 2), dtype=np.float32)]

            session.close()

            self.assertIsNone(session.reverse_raw_logits)
            self.assertIsNone(session.sam_frames_dir)


if __name__ == "__main__":
    unittest.main()
