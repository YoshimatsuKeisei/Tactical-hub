import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gui_controller import MaskSourceState, Sam2GuiSession
from identity_tracking import AppearanceFeatureCache, IdentityMaskCandidate
from identity_tracking_v2 import (
    MAX_APPEARANCE_FEATURE_CACHE_BYTES,
    MIN_REFERENCE_SEPARATION,
    MultiScaleFeatureCache,
    SamLevelEvidence,
    build_negative_context,
    choose_multiscale_feature_levels,
    feature_level_cache_bytes,
    fuse_level_scores,
    make_multiscale_feature_cache,
    make_pixel_descriptor,
    make_shape_descriptor,
    normalized_contrastive_score,
    select_fused_identity_path,
    shape_similarity,
    format_fused_identity_diagnostics,
)


def mask_at(x: int, y: int = 4, size: int = 2, shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    result[y : y + size, x : x + size] = True
    return result


def candidate(frame: int, name: str, mask: np.ndarray, source: str = "REGULAR") -> IdentityMaskCandidate:
    ys, xs = np.nonzero(mask)
    descriptor = np.array([1.0, 0.0], dtype=np.float32)
    return IdentityMaskCandidate(
        frame,
        name,
        source,
        0,
        mask.copy(),
        int(mask.sum()),
        (float(xs.mean()), float(ys.mean())),
        (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        descriptor,
        1.0,
    )


def rgba_frame(target: np.ndarray, distractor: np.ndarray | None = None) -> Image.Image:
    values = np.zeros((*target.shape, 4), dtype=np.uint8)
    values[..., 3] = 255
    values[..., :3] = (40, 70, 210)
    values[target, :3] = (220, 60, 40)
    if distractor is not None:
        values[distractor, :3] = (40, 70, 210)
    return Image.fromarray(values, mode="RGBA")


def feature_map(target: np.ndarray, distractor: np.ndarray | None = None) -> np.ndarray:
    values = np.zeros((2, *target.shape), dtype=np.float16)
    values[1] = 1.0
    values[:, target] = np.array([[1.0], [0.0]], dtype=np.float16)
    if distractor is not None:
        values[:, distractor] = np.array([[0.0], [1.0]], dtype=np.float16)
    return values


class MultiScaleCacheTests(unittest.TestCase):
    def test_measured_shapes_select_high_resolution_and_deep_level_within_budget(self) -> None:
        shapes = ((32, 256, 256), (64, 128, 128), (256, 64, 64))
        levels = choose_multiscale_feature_levels(shapes, 15)
        self.assertEqual(levels, (0, 2))
        total = sum(feature_level_cache_bytes(shapes[level], 15) for level in levels)
        self.assertLessEqual(total, MAX_APPEARANCE_FEATURE_CACHE_BYTES)
        self.assertEqual(total, 90 * 1024 * 1024)

    def test_actual_nbytes_and_float16_arrays_are_reused(self) -> None:
        first = np.ones((2, 4, 4), dtype=np.float16)
        deep = np.ones((4, 2, 2), dtype=np.float16)
        cache = make_multiscale_feature_cache(
            (0, 1), ((2, 4, 4), (4, 2, 2)), {0: [first], 1: [deep]}
        )
        self.assertEqual(cache.total_bytes, first.nbytes + deep.nbytes)
        self.assertIs(cache.level_features(0)[0], first)

    def test_target_reset_reuses_cache_and_close_invalidates_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.touch()
            session = Sam2GuiSession([Image.new("RGBA", (4, 4))], checkpoint, work_size=4)
            cache = make_multiscale_feature_cache(
                (0,), ((2, 4, 4),), {0: [np.ones((2, 4, 4), dtype=np.float16)]}
            )
            session.multi_scale_feature_cache = cache
            session.reset_tracking_state()
            self.assertIs(session.multi_scale_feature_cache, cache)
            session.close()
            self.assertIsNone(session.multi_scale_feature_cache)

    def test_extraction_uses_fresh_state_and_shares_existing_v1_level(self) -> None:
        class Predictor:
            def __init__(self) -> None:
                self.init_calls = 0
                self.reset_states: list[object] = []

            def init_state(self, **_kwargs: object) -> dict[str, object]:
                self.init_calls += 1
                return {"images": torch.ones((2, 3, 8, 8)), "fresh": self.init_calls}

            def forward_image(self, _image: torch.Tensor) -> dict[str, list[torch.Tensor]]:
                return {
                    "backbone_fpn": [
                        torch.ones((1, 2, 8, 8)),
                        torch.ones((1, 4, 4, 4)),
                        torch.ones((1, 8, 2, 2)),
                    ]
                }

            def reset_state(self, state: object) -> None:
                self.reset_states.append(state)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.touch()
            session = Sam2GuiSession(
                [Image.new("RGBA", (4, 4)) for _ in range(2)], checkpoint, work_size=8
            )
            predictor = Predictor()
            forward_state = {"forward": "unchanged"}
            shared = tuple(np.ones((2, 8, 8), dtype=np.float16) for _ in range(2))
            session.predictor = predictor
            session.device = torch.device("cpu")
            session.sam_frames_dir = Path(temporary)
            session.inference_state = forward_state
            session.appearance_feature_cache = AppearanceFeatureCache(
                0, ((2, 8, 8), (4, 4, 4), (8, 2, 2)), shared
            )

            first = session.extract_multiscale_appearance_features()
            second = session.extract_multiscale_appearance_features()

            self.assertIs(first, second)
            self.assertEqual(first.selected_levels, (0, 2, 1))
            self.assertIs(first.level_features(0)[0], shared[0])
            self.assertIs(session.inference_state, forward_state)
            self.assertEqual(forward_state, {"forward": "unchanged"})
            self.assertEqual(predictor.init_calls, 1)
            self.assertEqual(len(predictor.reset_states), 1)
            session.close()


class NegativeContextTests(unittest.TestCase):
    def test_ring_excludes_target_and_prefers_visible_alpha(self) -> None:
        target = mask_at(6)
        alpha = np.full(target.shape, 255, dtype=np.uint8)
        alpha[:, :4] = 0
        frame = Image.new("RGBA", (16, 16), (80, 80, 80, 255))
        frame.putalpha(Image.fromarray(alpha))
        snapshot = target.copy()
        context = build_negative_context(target, frame)
        self.assertIsNotNone(context.mask)
        self.assertFalse(np.logical_and(context.mask, target).any())
        self.assertFalse(context.mask[:, :4].any())
        np.testing.assert_array_equal(target, snapshot)

    def test_ring_expands_or_falls_back_when_initial_support_is_small(self) -> None:
        target = mask_at(7, y=7, size=2)
        values = np.zeros((16, 16, 4), dtype=np.uint8)
        values[target] = (80, 80, 80, 255)
        values[7:10, 1:5] = (90, 90, 90, 255)
        context = build_negative_context(
            target, Image.fromarray(values, mode="RGBA"), min_visible_pixels=8
        )
        self.assertTrue(context.fallback_used)
        self.assertIsNotNone(context.mask)
        self.assertGreaterEqual(context.visible_pixels, 8)

    def test_transparent_background_does_not_fake_negative_signal(self) -> None:
        target = mask_at(6)
        values = np.zeros((16, 16, 4), dtype=np.uint8)
        values[target] = (100, 100, 100, 255)
        context = build_negative_context(target, Image.fromarray(values, mode="RGBA"))
        self.assertIsNone(context.mask)
        self.assertEqual(context.visible_pixels, 0)


class ContrastiveScoreTests(unittest.TestCase):
    def test_saturated_positive_cosines_are_separated_by_negative_context(self) -> None:
        true_score = normalized_contrastive_score(0.992, 0.70, 0.30)
        distractor_score = normalized_contrastive_score(0.991, 0.98, 0.30)
        self.assertGreater(true_score, distractor_score + 0.8)

    def test_small_reference_separation_level_is_automatically_ignored(self) -> None:
        weak = SamLevelEvidence(0, 0.99, 0.989, 0.001, 1.0, MIN_REFERENCE_SEPARATION / 2, False)
        useful = SamLevelEvidence(2, 0.95, 0.60, 0.35, 0.9, 0.40, True)
        self.assertAlmostEqual(fuse_level_scores((weak, useful)), 0.9)

    def test_multiple_levels_are_fused_by_reference_separation(self) -> None:
        first = SamLevelEvidence(0, 0.9, 0.5, 0.4, 1.0, 0.4, True)
        second = SamLevelEvidence(2, 0.8, 0.6, 0.2, 0.5, 0.2, True)
        self.assertAlmostEqual(fuse_level_scores((first, second)), 5.0 / 6.0)


class PixelAndShapeDescriptorTests(unittest.TestCase):
    def test_pixel_descriptor_is_finite_and_same_pixels_match(self) -> None:
        frame = Image.new("RGBA", (8, 8), (128, 128, 128, 255))
        mask = np.ones((8, 8), dtype=bool)
        descriptor = make_pixel_descriptor(frame, mask)
        self.assertIsNotNone(descriptor)
        self.assertTrue(np.isfinite(descriptor).all())
        self.assertAlmostEqual(float(np.dot(descriptor, descriptor)), 1.0, places=6)

    def test_different_pixel_distribution_has_lower_similarity(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        gray = make_pixel_descriptor(Image.new("RGB", (8, 8), (120, 120, 120)), mask)
        varied_values = np.zeros((8, 8, 3), dtype=np.uint8)
        varied_values[:, :4] = (255, 0, 0)
        varied_values[:, 4:] = (0, 0, 255)
        varied = make_pixel_descriptor(Image.fromarray(varied_values, mode="RGB"), mask)
        self.assertLess(float(np.dot(gray, varied)), 0.95)

    def test_shape_is_soft_for_occlusion_and_penalizes_large_expansion(self) -> None:
        target = make_shape_descriptor(mask_at(4, size=4))
        partial = make_shape_descriptor(mask_at(4, size=3))
        large = make_shape_descriptor(mask_at(1, y=1, size=12))
        self.assertGreater(shape_similarity(target, partial), shape_similarity(target, large))
        self.assertGreater(shape_similarity(target, large), 0.0)

    def test_aspect_change_is_soft_not_hard_reject(self) -> None:
        square = mask_at(4, size=4)
        wide = np.zeros_like(square)
        wide[4:6, 2:12] = True
        score = shape_similarity(make_shape_descriptor(square), make_shape_descriptor(wide))
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class FusedPathRegressionTests(unittest.TestCase):
    def make_result(self, banks: list[list[IdentityMaskCandidate]], frames, features):
        cache = MultiScaleFeatureCache((0,), ((2, 16, 16),), (tuple(features),))
        return select_fused_identity_path(frames, banks, banks[0][0].mask, cache)

    def test_nearby_same_color_distractor_loses_to_contrastive_target(self) -> None:
        target0 = mask_at(4)
        true1 = mask_at(8)
        wrong1 = mask_at(5)
        frames = [rgba_frame(target0), rgba_frame(true1, wrong1)]
        features = [feature_map(target0), feature_map(true1, wrong1)]
        banks = [
            [candidate(0, "AUTHORITATIVE/component0", target0, "AUTHORITATIVE")],
            [candidate(1, "true", true1), candidate(1, "nearby-wrong", wrong1)],
        ]
        result = self.make_result(banks, frames, features)
        self.assertEqual(result.selected_candidates[1].candidate_id, "true")

    def test_large_wrong_candidate_with_high_raw_similarity_loses(self) -> None:
        target0 = mask_at(2, size=4)
        true1 = mask_at(10, size=3)
        wrong1 = np.zeros((16, 16), dtype=bool)
        wrong1[1:15, 1:9] = True
        frames = [rgba_frame(target0), rgba_frame(true1, wrong1)]
        features = [feature_map(target0), feature_map(true1, wrong1)]
        banks = [
            [candidate(0, "AUTHORITATIVE/component0", target0, "AUTHORITATIVE")],
            [candidate(1, "true", true1), candidate(1, "large-wrong", wrong1)],
        ]
        result = self.make_result(banks, frames, features)
        self.assertEqual(result.selected_candidates[1].candidate_id, "true")

    def test_distractor_only_gap_uses_occluded_then_reacquires_target(self) -> None:
        target0 = mask_at(2)
        wrong1 = mask_at(3)
        wrong2 = mask_at(4)
        target3 = mask_at(12)
        frames = [
            rgba_frame(target0),
            rgba_frame(np.zeros_like(target0), wrong1),
            rgba_frame(np.zeros_like(target0), wrong2),
            rgba_frame(target3),
        ]
        features = [
            feature_map(target0),
            feature_map(np.zeros_like(target0), wrong1),
            feature_map(np.zeros_like(target0), wrong2),
            feature_map(target3),
        ]
        banks = [
            [candidate(0, "AUTHORITATIVE/component0", target0, "AUTHORITATIVE")],
            [replace(candidate(1, "wrong-1", wrong1), descriptor=np.array([0.0, 1.0], dtype=np.float32))],
            [replace(candidate(2, "wrong-2", wrong2), descriptor=np.array([0.0, 1.0], dtype=np.float32))],
            [candidate(3, "target-3", target3, "REVERSE")],
        ]
        result = self.make_result(banks, frames, features)
        self.assertEqual(
            [None if item is None else item.candidate_id for item in result.selected_candidates],
            ["AUTHORITATIVE/component0", None, None, "target-3"],
        )

    def test_target_change_rebuilds_descriptors_without_mutating_shared_cache(self) -> None:
        first_target = mask_at(2)
        second_target = mask_at(10)
        combined_features = feature_map(first_target)
        combined_features[:, second_target] = np.array([[0.0], [1.0]], dtype=np.float16)
        cache = MultiScaleFeatureCache((0,), ((2, 16, 16),), ((combined_features,),))
        first = select_fused_identity_path(
            [rgba_frame(first_target)],
            [[candidate(0, "AUTHORITATIVE/component0", first_target, "AUTHORITATIVE")]],
            first_target,
            cache,
        )
        second = select_fused_identity_path(
            [rgba_frame(second_target)],
            [[candidate(0, "AUTHORITATIVE/component0", second_target, "AUTHORITATIVE")]],
            second_target,
            cache,
        )
        self.assertIs(first.feature_cache, cache)
        self.assertIs(second.feature_cache, cache)
        self.assertFalse(
            np.array_equal(
                first.target_positive_descriptors[0][1],
                second.target_positive_descriptors[0][1],
            )
        )

    def test_v2_scoring_does_not_mutate_v1_candidates_or_masks(self) -> None:
        target = mask_at(4)
        next_target = mask_at(8)
        bank = [
            [candidate(0, "AUTHORITATIVE/component0", target, "AUTHORITATIVE")],
            [candidate(1, "target", next_target)],
        ]
        snapshots = [(item.mask.copy(), item.target_similarity) for frame in bank for item in frame]
        self.make_result(
            bank,
            [rgba_frame(target), rgba_frame(next_target)],
            [feature_map(target), feature_map(next_target)],
        )
        for item, (mask_snapshot, similarity) in zip(
            (item for frame in bank for item in frame), snapshots, strict=True
        ):
            np.testing.assert_array_equal(item.mask, mask_snapshot)
            self.assertEqual(item.target_similarity, similarity)

    def test_diagnostics_exposes_cache_context_and_candidate_evidence(self) -> None:
        target = mask_at(4)
        result = self.make_result(
            [[candidate(0, "AUTHORITATIVE/component0", target, "AUTHORITATIVE")]],
            [rgba_frame(target)],
            [feature_map(target)],
        )
        report = format_fused_identity_diagnostics(result)
        self.assertIn("Selected SAM feature levels", report)
        self.assertIn("Negative context", report)
        self.assertIn("reference separation", report)
        self.assertIn("fused identity score", report)
        self.assertIn("final selected", report)


class FusedAdoptionTests(unittest.TestCase):
    def test_v2_preview_is_non_destructive_and_legacy_can_be_restored(self) -> None:
        regular = [mask_at(1)]
        fused = [mask_at(10)]
        state = MaskSourceState(regular)
        active = state.active_masks[0].copy()
        state.set_identity_v2_masks(fused)
        state.set_preview_mode("identity_v2")
        np.testing.assert_array_equal(state.active_masks[0], active)
        self.assertTrue(state.preview_dirty)
        np.testing.assert_array_equal(state.adopt_preview()[0], fused[0])
        state.set_preview_mode("legacy")
        np.testing.assert_array_equal(state.preview_masks[0], regular[0])

    def test_v1_and_v2_states_do_not_overwrite_each_other(self) -> None:
        state = MaskSourceState([mask_at(1)])
        v1 = [mask_at(5)]
        v2 = [mask_at(10)]
        state.set_identity_masks(v1)
        state.set_identity_v2_masks(v2)
        state.set_preview_mode("identity_v1")
        np.testing.assert_array_equal(state.preview_masks[0], v1[0])
        state.set_preview_mode("identity_v2")
        np.testing.assert_array_equal(state.preview_masks[0], v2[0])


if __name__ == "__main__":
    unittest.main()
