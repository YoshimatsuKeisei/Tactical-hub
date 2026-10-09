import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from gui import TeamColorApp
from gui_controller import (
    format_identity_global_prerequisite_error,
    identity_global_missing_prerequisites,
)


class ValueStub:
    def __init__(self, value=None) -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class TrackingSessionStub:
    def __init__(self, target: np.ndarray) -> None:
        self.target = target.copy()
        self.frame0_mask = None
        self.pre_gate_diagnostics = [object()]
        self.reverse_raw_logits = None
        self.close_calls = 0

    def set_frame0_mask(self, mask: np.ndarray, _progress) -> np.ndarray:
        self.frame0_mask = mask.copy()
        return mask.copy()

    def track_across_frames_logits(self, _progress) -> list[np.ndarray]:
        return [np.where(self.target, 1.0, -1.0).astype(np.float32)]

    def close(self) -> None:
        self.close_calls += 1


def target_at(x: int) -> np.ndarray:
    result = np.zeros((4, 4), dtype=bool)
    result[1:3, x : x + 1] = True
    return result


class IdentityTrackingLifecycleTests(unittest.TestCase):
    def make_track_app(
        self,
        mode: str,
        checkpoint: Path,
        target: np.ndarray,
    ) -> TeamColorApp:
        app = TeamColorApp.__new__(TeamColorApp)
        app.frame0_mask = target.copy()
        app.identity_authoritative_frame0_mask = None
        app.identity_global_result = None
        app.identity_reverse_warning = None
        app.identity_global_diagnostics_window = None
        app.selection_mode = ValueStub(mode)
        app.checkpoint_path = ValueStub(str(checkpoint))
        app.device = ValueStub("cpu")
        app.preview_mode = ValueStub("legacy")
        app.status = ValueStub()
        app.sam_session = TrackingSessionStub(target) if mode == "quick" else None
        app.session_checkpoint = None
        app.session_device_name = None
        app.sam_raw_logits = None
        app.pre_gate_diagnostics = None
        app.regular_raw_masks = None
        app.regular_masks = None
        app.pre_gate_masks = None
        app.filtered_pre_gate_masks = None
        app.temporal_filter_result = None
        app.reverse_raw_logits = None
        app.reverse_raw_masks = None
        app.reverse_masks = None
        app.combined_preview_masks = None
        app.mask_source_state = None
        app.raw_masks = None
        app.masks = None
        app.quick_frame0_logits = None
        app._discard_recolor_preview = lambda: None
        app._close_logit_diagnostics = lambda: None
        app._clear_pre_gate_diagnostics = lambda: None
        app._close_identity_global_diagnostics = lambda: None
        app._queue_status = lambda _message: None
        app._refresh_cleanup_info = lambda: None
        app._redraw_prompt = lambda: None
        app._refresh_mask_editor = lambda: None
        app._open_tracked_mask_preview = lambda: None

        def reset_sources(clear_active=True) -> None:
            app.identity_authoritative_frame0_mask = None
            if clear_active:
                app.raw_masks = None
                app.masks = None

        app._reset_mask_sources = reset_sources
        tasks: list[tuple[str, object]] = []
        app._start_task = lambda name, _status, worker: tasks.append((name, worker))
        app._test_tasks = tasks

        def ensure_session(_checkpoint: Path, _device: str) -> None:
            # This is the real failing lifecycle: first-session creation closes
            # SAM-owned state after the target snapshot has been restored.
            TeamColorApp._close_session(app)
            app.sam_session = TrackingSessionStub(target)

        if mode != "quick":
            app._ensure_sam_session = ensure_session

        def rebuild() -> None:
            app.regular_masks = [target.copy()]
            app.masks = [target.copy()]

        app._rebuild_masks_from_logits = rebuild
        return app

    def run_track_success(self, mode: str, checkpoint: Path, target: np.ndarray) -> TeamColorApp:
        app = self.make_track_app(mode, checkpoint, target)
        TeamColorApp._track_frames(app)
        self.assertEqual(len(app._test_tasks), 1)
        np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, target)
        self.assertFalse(np.shares_memory(app.identity_authoritative_frame0_mask, target))

        name, worker = app._test_tasks.pop()
        self.assertEqual(name, "track")
        logits = worker()
        np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, target)
        TeamColorApp._handle_success(app, "track", logits)
        self.assertEqual(
            identity_global_missing_prerequisites(
                sam_session=app.sam_session,
                regular_masks=app.regular_masks,
                pre_gate_diagnostics=app.pre_gate_diagnostics,
                authoritative_frame0_mask=app.identity_authoritative_frame0_mask,
            ),
            (),
        )
        np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, target)
        return app

    def test_all_selection_modes_preserve_exact_authoritative_target_through_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.touch()
            for mode in ("area", "quick", "all"):
                with self.subTest(mode=mode):
                    self.run_track_success(mode, checkpoint, target_at(1))

    def test_target_change_clears_old_target_and_next_track_captures_new_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.touch()
            old_target = target_at(1)
            app = self.run_track_success("area", checkpoint, old_target)

            app._reset_mask_sources(clear_active=True)
            self.assertIsNone(app.identity_authoritative_frame0_mask)

            new_target = target_at(3)
            app.frame0_mask = new_target.copy()
            app.sam_session = None
            app = self.run_track_success("area", checkpoint, new_target)
            np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, new_target)
            self.assertFalse(np.array_equal(app.identity_authoritative_frame0_mask, old_target))

    def test_session_replacement_and_preview_close_do_not_clear_authoritative_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.touch()
            target = target_at(2)
            app = self.make_track_app("area", checkpoint, target)
            app.identity_authoritative_frame0_mask = target.copy()
            TeamColorApp._close_session(app)
            np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, target)

            preview = object()
            app.tracked_mask_window = preview
            TeamColorApp._tracked_mask_window_closed(app, preview)
            np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, target)

    def make_rebuild_app(self, target: np.ndarray) -> TeamColorApp:
        app = TeamColorApp.__new__(TeamColorApp)
        app.frames = [Image.new("RGBA", (4, 4), (80, 80, 80, 255))]
        app.sam_raw_logits = [np.where(target, 2.0, -2.0).astype(np.float32)]
        app.mask_threshold = ValueStub(0.0)
        app.mask_threshold_text = ValueStub()
        app.fill_enclosed_holes = ValueStub(False)
        app.preview_mode = ValueStub("legacy")
        app.identity_authoritative_frame0_mask = target.copy()
        app.identity_global_result = None
        app.identity_global_diagnostics_window = None
        app.mask_source_state = None
        app.regular_raw_masks = None
        app.regular_masks = None
        app.pre_gate_diagnostics = None
        app.pre_gate_masks = None
        app.reverse_raw_logits = None
        app.reverse_raw_masks = None
        app.reverse_masks = None
        app.add_pre_gate_source = ValueStub(False)
        app.add_reverse_source = ValueStub(False)
        app.auto_filter_pre_gate = ValueStub(False)
        app.frame0_mask = target.copy()
        app.raw_masks = None
        app.masks = None
        app._pending_threshold_update = None
        app._busy = False
        app.selection_mode = ValueStub("area")
        app.status = ValueStub()
        app.tracked_mask_window = None
        app._authoritative_frame0_mask = lambda: target.copy()
        app._recompute_temporal_filter = lambda: None
        app._sync_mask_source_preview = lambda: None
        app._close_identity_global_diagnostics = lambda: None
        app._discard_recolor_preview = lambda: None
        app._refresh_cleanup_info = lambda: None
        app._refresh_selection_info = lambda: None
        app._redraw_prompt = lambda: None
        app._refresh_mask_editor = lambda: None
        app._refresh_tracked_mask_preview = lambda: None
        app._update_buttons = lambda: None
        return app

    def test_rebuild_threshold_and_hole_fill_keep_authoritative_target(self) -> None:
        target = target_at(2)
        app = self.make_rebuild_app(target)

        TeamColorApp._rebuild_masks_from_logits(app)
        np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, target)

        TeamColorApp._apply_threshold_update(app)
        np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, target)

        app.fill_enclosed_holes.set(True)
        TeamColorApp._on_cleanup_toggle(app)
        np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, target)

    def test_tracked_preview_open_and_close_keep_authoritative_target(self) -> None:
        class PreviewStub:
            def __init__(self) -> None:
                self.destroyed = False

            def destroy(self) -> None:
                self.destroyed = True

        target = target_at(1)
        app = self.make_rebuild_app(target)
        app.combined_preview_masks = [target.copy()]
        preview = PreviewStub()
        with patch("gui.TrackedMaskPreviewWindow", return_value=preview):
            TeamColorApp._open_tracked_mask_preview(app)
        np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, target)
        TeamColorApp._close_tracked_mask_preview(app)
        self.assertTrue(preview.destroyed)
        np.testing.assert_array_equal(app.identity_authoritative_frame0_mask, target)


class IdentityPrerequisiteDiagnosticTests(unittest.TestCase):
    def test_each_missing_prerequisite_is_named(self) -> None:
        available = {
            "sam_session": object(),
            "regular_masks": [target_at(1)],
            "pre_gate_diagnostics": [object()],
            "authoritative_frame0_mask": target_at(1),
        }
        expected = {
            "sam_session": "SAM session",
            "regular_masks": "Regular masks",
            "pre_gate_diagnostics": "Pre-Gate diagnostics",
            "authoritative_frame0_mask": "Authoritative frame 0 target mask",
        }
        for field, label in expected.items():
            with self.subTest(field=field):
                values = dict(available)
                values[field] = None
                self.assertEqual(identity_global_missing_prerequisites(**values), (label,))

    def test_error_lists_all_missing_items(self) -> None:
        missing = identity_global_missing_prerequisites(
            sam_session=None,
            regular_masks=None,
            pre_gate_diagnostics=None,
            authoritative_frame0_mask=None,
        )
        report = format_identity_global_prerequisite_error(missing)
        self.assertTrue(report.startswith("Missing prerequisites:"))
        for item in missing:
            self.assertIn(f"- {item}", report)

    def test_gui_guard_reports_specific_missing_items_and_returns_to_legacy(self) -> None:
        app = TeamColorApp.__new__(TeamColorApp)
        app.sam_session = object()
        app.regular_masks = [target_at(1)]
        app.pre_gate_diagnostics = [object()]
        app.identity_authoritative_frame0_mask = None
        app.preview_mode = ValueStub("identity")
        with patch("gui.messagebox.showerror") as showerror:
            TeamColorApp._begin_identity_global(app)
        self.assertEqual(app.preview_mode.get(), "legacy")
        message = showerror.call_args.args[1]
        self.assertIn("Missing prerequisites:", message)
        self.assertIn("- Authoritative frame 0 target mask", message)


if __name__ == "__main__":
    unittest.main()
