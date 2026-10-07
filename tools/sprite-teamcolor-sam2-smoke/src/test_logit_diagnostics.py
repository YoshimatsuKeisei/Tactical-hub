import unittest

import numpy as np
from PIL import Image

from gui_controller import (
    LOGIT_DIAGNOSTIC_THRESHOLDS,
    diagnose_sam_logits,
    format_sam_logit_diagnostics,
)


def rgba_frame(size: tuple[int, int], alpha: np.ndarray | None = None) -> Image.Image:
    pixels = np.full((size[1], size[0], 4), 90, dtype=np.uint8)
    pixels[:, :, 3] = 255 if alpha is None else alpha
    return Image.fromarray(pixels, "RGBA")


class SamLogitDiagnosticsTests(unittest.TestCase):
    def test_float32_continuous_statistics_and_percentiles(self) -> None:
        logits = np.arange(100, dtype=np.float32).reshape(10, 10)

        diagnostics = diagnose_sam_logits([logits], [rgba_frame((10, 10))])
        frame = diagnostics.frames[0]
        percentiles = dict(frame.percentiles)

        self.assertTrue(diagnostics.all_float32_2d)
        self.assertTrue(diagnostics.has_continuous_values)
        self.assertEqual(frame.shape, (10, 10))
        self.assertEqual(frame.dtype, "float32")
        self.assertEqual(frame.minimum, 0.0)
        self.assertEqual(frame.maximum, 99.0)
        self.assertEqual(frame.mean, 49.5)
        self.assertAlmostEqual(percentiles[1], 0.99)
        self.assertAlmostEqual(percentiles[5], 4.95)
        self.assertAlmostEqual(percentiles[25], 24.75)
        self.assertAlmostEqual(percentiles[50], 49.5)
        self.assertAlmostEqual(percentiles[75], 74.25)
        self.assertAlmostEqual(percentiles[95], 94.05)
        self.assertAlmostEqual(percentiles[99], 98.01)

    def test_all_required_threshold_candidates_have_pixel_counts(self) -> None:
        values = np.array(
            [-30, -15, -7, -3, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 3, 7, 15, 30],
            dtype=np.float32,
        ).reshape(1, 14)

        frame = diagnose_sam_logits([values], [rgba_frame((14, 1))]).frames[0]
        counts = dict(frame.threshold_pixel_counts)

        self.assertEqual(tuple(counts), LOGIT_DIAGNOSTIC_THRESHOLDS)
        self.assertEqual(counts[-20.0], 13)
        self.assertEqual(counts[0.0], 7)
        self.assertEqual(counts[20.0], 1)

    def test_threshold_counts_are_nonincreasing_as_threshold_rises(self) -> None:
        logits = np.linspace(-30, 30, 400, dtype=np.float32).reshape(20, 20)

        diagnostics = diagnose_sam_logits([logits], [rgba_frame((20, 20))])
        counts = [count for _threshold, count in diagnostics.frames[0].threshold_pixel_counts]

        self.assertTrue(diagnostics.monotonic_nonincreasing)
        self.assertTrue(all(left >= right for left, right in zip(counts, counts[1:])))
        reversed_counts = list(reversed(counts))
        self.assertTrue(
            all(left <= right for left, right in zip(reversed_counts, reversed_counts[1:])),
            "lowering the threshold must leave the count unchanged or increase it",
        )

    def test_threshold_zero_matches_direct_legacy_comparison(self) -> None:
        logits = np.array([[1.0, 0.0], [-0.01, 2.0]], dtype=np.float32)

        diagnostics = diagnose_sam_logits([logits], [rgba_frame((6, 4))])

        self.assertTrue(diagnostics.threshold_zero_compatible)
        self.assertTrue(diagnostics.frames[0].threshold_zero_compatible)

    def test_binary_like_inputs_are_reported_without_hiding_dtype_or_unique_count(self) -> None:
        bool_logits = np.array([[True, False], [False, True]], dtype=bool)
        float_binary = bool_logits.astype(np.float32)

        bool_report = diagnose_sam_logits([bool_logits], [rgba_frame((2, 2))])
        float_report = diagnose_sam_logits([float_binary], [rgba_frame((2, 2))])

        self.assertFalse(bool_report.all_float32_2d)
        self.assertEqual(bool_report.frames[0].dtype, "bool")
        self.assertEqual(bool_report.frames[0].unique_value_count, 2)
        self.assertFalse(bool_report.has_continuous_values)
        self.assertTrue(float_report.all_float32_2d)
        self.assertEqual(float_report.frames[0].unique_value_count, 2)
        self.assertFalse(float_report.has_continuous_values)

    def test_pixel_counts_apply_alpha_clip(self) -> None:
        alpha = np.array([[255, 0, 255], [0, 255, 0], [255, 0, 255]], dtype=np.uint8)
        logits = np.ones((3, 3), dtype=np.float32)

        frame = diagnose_sam_logits([logits], [rgba_frame((3, 3), alpha)]).frames[0]
        counts = dict(frame.threshold_pixel_counts)

        self.assertEqual(counts[0.0], 5)

    def test_diagnostic_counts_do_not_apply_enclosed_hole_fill(self) -> None:
        logits = np.ones((3, 3), dtype=np.float32)
        logits[1, 1] = -1.0

        frame = diagnose_sam_logits([logits], [rgba_frame((3, 3))]).frames[0]

        self.assertEqual(dict(frame.threshold_pixel_counts)[0.0], 8)

    def test_plain_text_report_is_copyable_and_explains_integrity_checks(self) -> None:
        logits = np.linspace(-3, 3, 25, dtype=np.float32).reshape(5, 5)
        diagnostics = diagnose_sam_logits([logits], [rgba_frame((5, 5))])

        report = format_sam_logit_diagnostics(diagnostics)

        self.assertIn("frame_000", report)
        self.assertIn("p05:", report)
        self.assertIn("threshold -20: ", report)
        self.assertIn("threshold +20: ", report)
        self.assertIn("Hole Fill excluded", report)
        self.assertIn("Threshold-count monotonicity: PASS", report)


if __name__ == "__main__":
    unittest.main()
