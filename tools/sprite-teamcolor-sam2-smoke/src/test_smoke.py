import unittest

import numpy as np

from smoke import bbox, mask_iou, parse_xy


class SmokeUtilityTests(unittest.TestCase):
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
