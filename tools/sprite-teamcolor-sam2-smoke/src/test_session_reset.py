import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gui_controller import PromptClick, Sam2GuiSession
from gui import TeamColorApp


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class TargetAwarePredictor:
    def __init__(self) -> None:
        self.image_size = 4
        self.sam_mask_decoder = torch.nn.Identity()
        self.init_calls = 0
        self.reset_states: list[object] = []
        self.point_calls: list[tuple[object, np.ndarray, np.ndarray]] = []
        self.mask_calls: list[tuple[object, np.ndarray]] = []
        self.propagate_states: list[object] = []

    def init_state(self, **_kwargs: object) -> dict[str, object]:
        self.init_calls += 1
        return {
            "generation": self.init_calls,
            "target_logits": np.full((4, 4), -2.0, dtype=np.float32),
        }

    def reset_state(self, inference_state: object) -> None:
        self.reset_states.append(inference_state)

    def add_new_points_or_box(self, **kwargs: object):
        state = kwargs["inference_state"]
        points = np.asarray(kwargs["points"], dtype=np.float32).copy()
        labels = np.asarray(kwargs["labels"], dtype=np.int32).copy()
        logits = np.full((4, 4), -2.0, dtype=np.float32)
        for (x, y), label in zip(points, labels, strict=True):
            logits[int(y), int(x)] = 2.0 if int(label) == 1 else -2.0
        state["target_logits"] = logits
        self.point_calls.append((state, points, labels))
        return 0, [1], [FakeTensor(logits[None])]

    def add_new_mask(self, **kwargs: object):
        state = kwargs["inference_state"]
        target = np.asarray(kwargs["mask"], dtype=bool).copy()
        logits = np.where(target, 2.0, -2.0).astype(np.float32)
        state["target_logits"] = logits
        self.mask_calls.append((state, target))
        return int(kwargs["frame_idx"]), [1], [FakeTensor(logits[None])]

    def propagate_in_video(self, inference_state: object):
        self.propagate_states.append(inference_state)
        logits = np.asarray(inference_state["target_logits"], dtype=np.float32)
        for frame_index in range(3):
            yield frame_index, [1], [FakeTensor(logits[None])]


def rgba_frames() -> list[Image.Image]:
    return [Image.new("RGBA", (4, 4), (90, 90, 90, 255)) for _ in range(3)]


class ValueStub:
    def __init__(self, value: object = None) -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class SessionStub:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.reverse_raw_logits: object | None = object()

    def reset_tracking_state(self) -> None:
        self.reset_calls += 1


class SamTrackingStateResetTests(unittest.TestCase):
    def make_session(
        self,
        temporary: str,
    ) -> tuple[Sam2GuiSession, TargetAwarePredictor]:
        checkpoint = Path(temporary) / "checkpoint.pt"
        checkpoint.touch()
        session = Sam2GuiSession(rgba_frames(), checkpoint, work_size=4)
        predictor = TargetAwarePredictor()
        session.predictor = predictor
        session.device = torch.device("cpu")
        session.sam_frames_dir = Path(temporary)
        return session, predictor

    def test_reset_discards_all_target_state_but_reuses_model_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, predictor = self.make_session(temporary)
            model_identity = session.predictor
            frames_directory = session.sam_frames_dir
            state_a = predictor.init_state()
            session.inference_state = state_a
            session.frame0_mask = np.ones((4, 4), dtype=bool)
            session.sam_frame0_prediction = np.ones((4, 4), dtype=bool)
            session.frame0_logits = np.ones((4, 4), dtype=np.float32)
            session.raw_logits = [np.ones((4, 4), dtype=np.float32)]
            session.reverse_raw_logits = [np.ones((4, 4), dtype=np.float32)]
            session.pre_gate_diagnostics = [object()]
            session.masks = [np.ones((4, 4), dtype=bool)]

            session.reset_tracking_state()

            self.assertIs(session.predictor, model_identity)
            self.assertEqual(session.sam_frames_dir, frames_directory)
            self.assertEqual(session.device.type, "cpu")
            self.assertEqual(predictor.reset_states, [state_a])
            self.assertIsNone(session.inference_state)
            self.assertIsNone(session.frame0_mask)
            self.assertIsNone(session.sam_frame0_prediction)
            self.assertIsNone(session.frame0_logits)
            self.assertIsNone(session.raw_logits)
            self.assertIsNone(session.reverse_raw_logits)
            self.assertIsNone(session.pre_gate_diagnostics)
            self.assertIsNone(session.masks)
            session.close()

    def test_area_target_change_uses_fresh_state_and_only_target_b(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, predictor = self.make_session(temporary)
            target_a = np.zeros((4, 4), dtype=bool)
            target_a[0, 0] = True
            target_b = np.zeros((4, 4), dtype=bool)
            target_b[3, 3] = True

            session.set_frame0_mask(target_a)
            state_a = session.inference_state
            output_a = session.track_across_frames_logits()
            diagnostics_a = session.pre_gate_diagnostics
            session.reverse_raw_logits = [np.ones((4, 4), dtype=np.float32)]

            session.reset_tracking_state()

            self.assertIsNone(session.raw_logits)
            self.assertIsNone(session.reverse_raw_logits)
            self.assertIsNone(session.pre_gate_diagnostics)
            session.set_frame0_mask(target_b)
            state_b = session.inference_state
            output_b = session.track_across_frames_logits()

            self.assertIs(session.predictor, predictor)
            self.assertIsNot(state_b, state_a)
            self.assertEqual(predictor.init_calls, 2)
            self.assertIsNot(session.pre_gate_diagnostics, diagnostics_a)
            np.testing.assert_array_equal(session.frame0_mask, target_b)
            np.testing.assert_array_equal(predictor.mask_calls[-1][1], target_b)
            self.assertGreater(output_a[0][0, 0], 0)
            self.assertLess(output_b[0][0, 0], 0)
            self.assertGreater(output_b[0][3, 3], 0)
            self.assertIs(predictor.propagate_states[-1], state_b)
            session.close()

    def test_quick_prompt_change_starts_fresh_without_explicit_gui_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session, predictor = self.make_session(temporary)

            session.generate_frame0_mask([PromptClick(0, 0, 1)])
            state_a = session.inference_state
            session.track_across_frames_logits()
            diagnostics_a = session.pre_gate_diagnostics
            session.reverse_raw_logits = [np.ones((4, 4), dtype=np.float32)]

            mask_b = session.generate_frame0_mask([PromptClick(3, 3, 1)])
            state_b = session.inference_state

            self.assertIs(session.predictor, predictor)
            self.assertIsNot(state_b, state_a)
            self.assertEqual(predictor.init_calls, 2)
            self.assertIsNone(session.raw_logits)
            self.assertIsNone(session.reverse_raw_logits)
            self.assertIsNone(session.pre_gate_diagnostics)
            self.assertIsNot(diagnostics_a, session.pre_gate_diagnostics)
            self.assertFalse(mask_b[0, 0])
            self.assertTrue(mask_b[3, 3])
            self.assertIs(predictor.point_calls[-1][0], state_b)
            np.testing.assert_array_equal(predictor.point_calls[-1][1], [[3.0, 3.0]])

            output_b = session.track_across_frames_logits()
            self.assertLess(output_b[0][0, 0], 0)
            self.assertGreater(output_b[0][3, 3], 0)
            session.close()

    def test_selection_mode_change_resets_session_and_target_caches(self) -> None:
        app = TeamColorApp.__new__(TeamColorApp)
        app._busy = False
        app.sam_session = SessionStub()
        app.selection_mode = ValueStub("area")
        app.status = ValueStub()
        app.frame0_mask = None
        app.quick_frame0_mask = np.ones((4, 4), dtype=bool)
        app.quick_frame0_logits = np.ones((4, 4), dtype=np.float32)
        app.sam_raw_logits = [np.ones((4, 4), dtype=np.float32)]
        calls: list[str] = []
        app._clear_main_draft = lambda: calls.append("draft")
        app._clear_pre_gate_diagnostics = lambda: calls.append("pre_gate")
        app._reset_mask_sources = lambda clear_active=True: calls.append(
            f"sources:{clear_active}"
        )
        app._close_tracked_mask_preview = lambda: None
        app._close_logit_diagnostics = lambda: None
        app._discard_recolor_preview = lambda: None
        app._sync_lasso_mask = lambda: None
        app._refresh_selection_info = lambda: None
        app._show_editor_controls = lambda: None
        app._refresh_cleanup_info = lambda: None
        app._redraw_prompt = lambda: None
        app._update_buttons = lambda: None

        TeamColorApp._on_selection_mode_change(app)

        self.assertEqual(app.sam_session.reset_calls, 1)
        self.assertIsNone(app.quick_frame0_mask)
        self.assertIsNone(app.quick_frame0_logits)
        self.assertIsNone(app.sam_raw_logits)
        self.assertIn("pre_gate", calls)
        self.assertIn("sources:True", calls)

    def test_mask_source_reset_clears_reverse_temporal_and_active_target_data(self) -> None:
        app = TeamColorApp.__new__(TeamColorApp)
        app.add_pre_gate_source = ValueStub(True)
        app.add_reverse_source = ValueStub(True)
        app.auto_filter_pre_gate = ValueStub(True)
        app.preview_mode = ValueStub("identity")
        app.mask_sources_info = ValueStub()
        app.mask_adoption_info = ValueStub()
        app.tracked_mask_info = ValueStub()
        app.sam_session = SessionStub()
        app.temporal_filter_diagnostics_window = None
        app.identity_global_diagnostics_window = None
        app._close_identity_global_diagnostics = lambda: None
        app.regular_raw_masks = object()
        app.regular_masks = object()
        app.pre_gate_masks = object()
        app.filtered_pre_gate_masks = object()
        app.temporal_filter_result = object()
        app.reverse_raw_logits = object()
        app.reverse_raw_masks = object()
        app.reverse_masks = object()
        app.combined_preview_masks = object()
        app.mask_source_state = object()
        app.identity_global_result = object()
        app.identity_authoritative_frame0_mask = object()
        app.identity_reverse_warning = "warning"
        app.raw_masks = object()
        app.masks = object()

        TeamColorApp._reset_mask_sources(app, clear_active=True)

        for attribute in (
            "regular_raw_masks",
            "regular_masks",
            "pre_gate_masks",
            "filtered_pre_gate_masks",
            "temporal_filter_result",
            "reverse_raw_logits",
            "reverse_raw_masks",
            "reverse_masks",
            "combined_preview_masks",
            "mask_source_state",
            "identity_global_result",
            "identity_authoritative_frame0_mask",
            "identity_reverse_warning",
            "raw_masks",
            "masks",
        ):
            self.assertIsNone(getattr(app, attribute), attribute)
        self.assertEqual(app.preview_mode.get(), "legacy")
        self.assertIsNone(app.sam_session.reverse_raw_logits)


if __name__ == "__main__":
    unittest.main()
