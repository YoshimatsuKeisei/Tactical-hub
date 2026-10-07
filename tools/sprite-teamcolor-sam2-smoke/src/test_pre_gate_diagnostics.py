import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional

from gui_controller import (
    Sam2GuiSession,
    SamPreGateHookCollector,
    format_pre_gate_diagnostics,
    make_pre_gate_contact_sheet,
    pre_gate_preview_mask,
    propagate_sam2_logits_with_pre_gate,
)


class FakeDecoder(torch.nn.Module):
    def __init__(
        self,
        masks: torch.Tensor,
        ious: torch.Tensor,
        object_score: torch.Tensor,
    ) -> None:
        super().__init__()
        self.masks = masks
        self.ious = ious
        self.object_score = object_score
        self.calls = 0
        self.last_output: tuple[torch.Tensor, ...] | None = None

    def forward(self) -> tuple[torch.Tensor, ...]:
        self.calls += 1
        tokens = torch.zeros((self.masks.shape[0], self.masks.shape[1], 3))
        self.last_output = (self.masks, self.ious, tokens, self.object_score)
        return self.last_output


class FakePredictor:
    def __init__(self, decoder: FakeDecoder, call_counts: list[int], fail_after_calls: bool = False) -> None:
        self.sam_mask_decoder = decoder
        self.image_size = 4
        self.call_counts = call_counts
        self.fail_after_calls = fail_after_calls

    def propagate_in_video(self, _state: object):
        for frame_index, call_count in enumerate(self.call_counts):
            for _ in range(call_count):
                self.sam_mask_decoder()
            if self.fail_after_calls and frame_index == 0:
                raise RuntimeError("simulated propagation failure")
            post_value = -1024.0 if frame_index == 1 else 2.0
            yield frame_index, [1], torch.full((1, 1, 4, 4), post_value)


def decoder_with_candidates() -> FakeDecoder:
    masks = torch.tensor(
        [
            [
                [[-2.0, -1.0], [0.0, 1.0]],
                [[1.0, 2.0], [3.0, 4.0]],
                [[-4.0, -3.0], [-2.0, -1.0]],
            ]
        ],
        dtype=torch.float16,
    )
    ious = torch.tensor([[0.2, 0.9, 0.4]], dtype=torch.float32)
    object_score = torch.tensor([[-0.7]], dtype=torch.float32)
    return FakeDecoder(masks, ious, object_score)


class PreGateDiagnosticTests(unittest.TestCase):
    def test_forward_hook_observes_without_changing_decoder_return(self) -> None:
        decoder = decoder_with_candidates()
        original_masks = decoder.masks.clone()
        original_ious = decoder.ious.clone()
        original_score = decoder.object_score.clone()
        collector = SamPreGateHookCollector(image_size=4)
        handle = decoder.register_forward_hook(collector.hook)

        output = decoder()
        captures, errors = collector.drain()
        handle.remove()

        self.assertIs(output, decoder.last_output)
        self.assertIs(output[0], decoder.masks)
        self.assertIs(output[1], decoder.ious)
        self.assertIs(output[3], decoder.object_score)
        torch.testing.assert_close(decoder.masks, original_masks)
        torch.testing.assert_close(decoder.ious, original_ious)
        torch.testing.assert_close(decoder.object_score, original_score)
        self.assertEqual(errors, ())
        self.assertEqual(len(captures), 1)

    def test_capture_is_cpu_float32_and_selects_highest_iou_candidate(self) -> None:
        decoder = decoder_with_candidates()
        collector = SamPreGateHookCollector(image_size=4)
        handle = decoder.register_forward_hook(collector.hook)
        decoder()
        capture = collector.drain()[0][0]
        handle.remove()

        self.assertEqual(capture.low_res_multimasks.dtype, np.float32)
        self.assertEqual(capture.ious.dtype, np.float32)
        self.assertEqual(capture.object_score_logits.dtype, np.float32)
        self.assertEqual(capture.pre_gate_logits.dtype, np.float32)
        self.assertEqual(capture.pre_gate_logits.shape, (4, 4))
        self.assertEqual(capture.best_mask_index, 1)
        self.assertAlmostEqual(capture.best_iou, 0.9)
        self.assertAlmostEqual(capture.object_score_logit, -0.7)

    def test_single_mask_is_selected_without_argmax_guessing(self) -> None:
        decoder = FakeDecoder(
            torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
            torch.tensor([[0.3]]),
            torch.tensor([[1.2]]),
        )
        collector = SamPreGateHookCollector(image_size=4)
        handle = decoder.register_forward_hook(collector.hook)
        decoder()
        capture = collector.drain()[0][0]
        handle.remove()

        self.assertEqual(capture.best_mask_index, 0)
        self.assertAlmostEqual(capture.best_iou, 0.3)
        self.assertAlmostEqual(capture.object_score_logit, 1.2)

    def test_bilinear_upsample_matches_align_corners_false(self) -> None:
        decoder = decoder_with_candidates()
        collector = SamPreGateHookCollector(image_size=4)
        handle = decoder.register_forward_hook(collector.hook)
        decoder()
        capture = collector.drain()[0][0]
        handle.remove()
        expected = functional.interpolate(
            decoder.masks.detach().float().cpu()[:, 1:2],
            size=(4, 4),
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()

        np.testing.assert_array_equal(capture.pre_gate_logits, expected)

    def test_frame_mapping_distinguishes_zero_one_and_multiple_decoder_calls(self) -> None:
        predictor = FakePredictor(decoder_with_candidates(), [0, 1, 2])

        _post_gate, diagnostics = propagate_sam2_logits_with_pre_gate(predictor, object())

        self.assertEqual(
            [item.capture_status for item in diagnostics],
            ["no_capture", "captured", "ambiguous_multiple_decoder_calls"],
        )
        self.assertEqual([item.decoder_call_count for item in diagnostics], [0, 1, 2])
        self.assertEqual(len(diagnostics[2].captures), 2)

    def test_hook_is_removed_after_success_and_exception(self) -> None:
        successful = FakePredictor(decoder_with_candidates(), [1])
        propagate_sam2_logits_with_pre_gate(successful, object())
        self.assertEqual(len(successful.sam_mask_decoder._forward_hooks), 0)

        failing = FakePredictor(decoder_with_candidates(), [1], fail_after_calls=True)
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            propagate_sam2_logits_with_pre_gate(failing, object())
        self.assertEqual(len(failing.sam_mask_decoder._forward_hooks), 0)

    def test_no_object_post_gate_keeps_finite_pre_gate_candidate(self) -> None:
        predictor = FakePredictor(decoder_with_candidates(), [0, 1])

        post_gate, diagnostics = propagate_sam2_logits_with_pre_gate(predictor, object())
        captured = diagnostics[1].capture

        self.assertTrue(np.all(post_gate[1] == -1024.0))
        self.assertTrue(diagnostics[1].post_gate_is_no_obj)
        self.assertIsNotNone(captured)
        self.assertTrue(np.isfinite(captured.pre_gate_logits).all())
        self.assertFalse(np.all(captured.pre_gate_logits == -1024.0))
        self.assertLessEqual(captured.object_score_logit, 0.0)

    def test_preview_threshold_does_not_mutate_standard_masks_or_rerun_decoder(self) -> None:
        predictor = FakePredictor(decoder_with_candidates(), [1])
        post_gate, diagnostics = propagate_sam2_logits_with_pre_gate(predictor, object())
        post_before = post_gate[0].copy()
        calls_before = predictor.sam_mask_decoder.calls
        frame = Image.new("RGBA", (4, 4), (100, 100, 100, 255))

        low = pre_gate_preview_mask(diagnostics[0], frame, threshold=-5.0)
        high = pre_gate_preview_mask(diagnostics[0], frame, threshold=5.0)

        self.assertEqual(predictor.sam_mask_decoder.calls, calls_before)
        np.testing.assert_array_equal(post_gate[0], post_before)
        self.assertGreaterEqual(int(low.sum()), int(high.sum()))

    def test_text_and_contact_sheet_mark_no_capture_and_object_absent_candidate(self) -> None:
        predictor = FakePredictor(decoder_with_candidates(), [0, 1])
        _post_gate, diagnostics = propagate_sam2_logits_with_pre_gate(predictor, object())
        frames = [
            Image.new("RGBA", (4, 4), (100, 100, 100, 255)),
            Image.new("RGBA", (4, 4), (100, 100, 100, 255)),
        ]

        report = format_pre_gate_diagnostics(frames, diagnostics)
        contact = make_pre_gate_contact_sheet(frames, diagnostics, columns=2)

        self.assertIn("frame_000", report)
        self.assertIn("capture status: no_capture", report)
        self.assertIn("object absent gate: YES", report)
        self.assertIn("object absent gate with", report)
        self.assertEqual(contact.width, 8)
        self.assertGreater(contact.height, 4)

    def test_session_close_clears_pre_gate_diagnostic_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.touch()
            session = Sam2GuiSession([Image.new("RGBA", (2, 2))], checkpoint)
            session.pre_gate_diagnostics = []

            session.close()

            self.assertIsNone(session.pre_gate_diagnostics)


if __name__ == "__main__":
    unittest.main()
