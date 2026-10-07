import unittest

import numpy as np

from smoke import (
    add_sam2_mask,
    add_sam2_mask_logits,
    bbox,
    mask_iou,
    parse_xy,
    propagate_sam2_logits,
    propagate_sam2_masks,
    validate_binary_mask,
)


class FakeMaskLogits:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> "FakeMaskLogits":
        return self

    def cpu(self) -> "FakeMaskLogits":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeMaskPredictor:
    def __init__(self) -> None:
        self.received_mask: np.ndarray | None = None

    def add_new_mask(self, **kwargs: object) -> tuple[int, list[int], list[FakeMaskLogits]]:
        self.received_mask = kwargs["mask"]
        logits = np.array([[[1.0, -1.0], [-1.0, 1.0]]], dtype=np.float32)
        return 0, [1], [FakeMaskLogits(logits)]

    def propagate_in_video(self, _state: object):
        logits = np.array([[[1.0, -0.2], [-1.0, 1.0]]], dtype=np.float32)
        yield 0, [1], [FakeMaskLogits(logits)]


class SmokeUtilityTests(unittest.TestCase):
    def test_binary_mask_validation_rejects_wrong_shape_and_dtype(self) -> None:
        valid = np.zeros((4, 6), dtype=bool)
        np.testing.assert_array_equal(validate_binary_mask(valid, (6, 4)), valid)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_binary_mask(np.zeros((6, 4), dtype=bool), (6, 4))
        with self.assertRaisesRegex(ValueError, "dtype"):
            validate_binary_mask(np.zeros((4, 6), dtype=np.uint8), (6, 4))

    def test_sam2_mask_adapter_uses_binary_frame_mask(self) -> None:
        predictor = FakeMaskPredictor()
        initial = np.array([[True, False], [False, True]], dtype=bool)

        result = add_sam2_mask(predictor, object(), initial, (2, 2))

        np.testing.assert_array_equal(predictor.received_mask, initial)
        np.testing.assert_array_equal(result, initial)

    def test_raw_logit_adapters_preserve_cli_threshold_zero_wrapper(self) -> None:
        predictor = FakeMaskPredictor()
        initial = np.array([[True, False], [False, True]], dtype=bool)

        raw = add_sam2_mask_logits(predictor, object(), initial, (2, 2))
        propagated_raw = propagate_sam2_logits(predictor, object())
        propagated_binary = propagate_sam2_masks(predictor, object(), (2, 2))

        self.assertEqual(raw.dtype, np.float32)
        np.testing.assert_array_equal(raw > 0, initial)
        np.testing.assert_array_equal(propagated_binary[0], propagated_raw[0] > 0)

    def test_parse_xy_accepts_original_frame_coordinates(self) -> None:
        self.assertEqual(parse_xy("12,34"), (12, 34))

    def test_bbox_uses_inclusive_xyxy_coordinates(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 3:7] = True
        self.assertEqual(bbox(mask), [3, 2, 6, 4])

    def test_iou(self) -> None:
        left = np.array([[True, True], [False, False]])
        right = np.array([[False, True], [True, False]])
        self.assertEqual(mask_iou(left, right), 1 / 3)


if __name__ == "__main__":
    unittest.main()
