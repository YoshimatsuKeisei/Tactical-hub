import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from gui_controller import (
    MaskSourceState,
    Sam2GuiSession,
    SamPreGateDecoderCapture,
    SamPreGateFrameDiagnostics,
    build_identity_candidate_bank,
    recolor_frame_sequence,
    save_results,
)
from identity_tracking import (
    AppearanceFeatureCache,
    IdentityMaskCandidate,
    choose_appearance_feature_level,
    cosine_similarity,
    make_appearance_feature_cache,
    pool_mask_descriptor,
    select_identity_global_path,
)


def square_mask(x: int, y: int = 1, size: int = 2, shape: tuple[int, int] = (12, 80)) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[y : y + size, x : x + size] = True
    return mask


def descriptor(similarity: float) -> np.ndarray:
    similarity = float(similarity)
    return np.array([similarity, np.sqrt(max(0.0, 1.0 - similarity**2))], dtype=np.float32)


def candidate(
    frame: int,
    name: str,
    x: int,
    similarity: float,
    source: str = "REGULAR",
) -> IdentityMaskCandidate:
    mask = square_mask(x)
    ys, xs = np.nonzero(mask)
    vector = descriptor(similarity)
    return IdentityMaskCandidate(
        frame,
        name,
        source,
        0,
        mask,
        int(mask.sum()),
        (float(xs.mean()), float(ys.mean())),
        (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        vector,
        similarity,
    )


def cache(frame_count: int, shape: tuple[int, int] = (12, 80)) -> AppearanceFeatureCache:
    features = tuple(np.ones((2, *shape), dtype=np.float16) for _ in range(frame_count))
    return AppearanceFeatureCache(0, ((2, *shape),), features)


class AppearanceDescriptorTests(unittest.TestCase):
    def test_weighted_pool_is_normalized_and_does_not_mutate_feature(self) -> None:
        feature = np.zeros((2, 4, 4), dtype=np.float32)
        feature[0, :, :2] = 2.0
        feature[1, :, 2:] = 3.0
        snapshot = feature.copy()
        mask = np.zeros((8, 8), dtype=bool)
        mask[:, :4] = True

        pooled = pool_mask_descriptor(feature, mask)

        self.assertIsNotNone(pooled)
        self.assertAlmostEqual(float(np.linalg.norm(pooled)), 1.0, places=6)
        self.assertGreater(float(pooled[0]), 0.99)
        np.testing.assert_array_equal(feature, snapshot)

    def test_empty_mask_is_invalid_and_tiny_mask_keeps_support(self) -> None:
        feature = np.zeros((2, 2, 2), dtype=np.float32)
        feature[0] = 1.0
        empty = np.zeros((32, 32), dtype=bool)
        tiny = empty.copy()
        tiny[31, 31] = True

        self.assertIsNone(pool_mask_descriptor(feature, empty))
        pooled = pool_mask_descriptor(feature, tiny)
        self.assertIsNotNone(pooled)
        self.assertTrue(np.isfinite(pooled).all())

    def test_cosine_identical_is_one_and_different_is_lower(self) -> None:
        target = np.array([1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(target, target), 1.0, places=6)
        self.assertLess(cosine_similarity(target, np.array([0.0, 1.0], dtype=np.float32)), 0.1)

    def test_feature_level_uses_highest_resolution_within_memory_limit(self) -> None:
        shapes = ((32, 256, 256), (64, 128, 128), (256, 64, 64))
        self.assertEqual(choose_appearance_feature_level(shapes, 15, 96 * 1024 * 1024), 0)
        self.assertEqual(choose_appearance_feature_level(shapes, 15, 40 * 1024 * 1024), 1)

    def test_cache_owns_float16_features(self) -> None:
        source = np.ones((2, 4, 4), dtype=np.float32)
        retained = make_appearance_feature_cache(0, ((2, 4, 4),), [source])
        source[:] = 0
        self.assertEqual(retained.features[0].dtype, np.float16)
        self.assertTrue((retained.features[0] == 1).all())


class FeatureCacheLifecycleTests(unittest.TestCase):
    def test_target_reset_keeps_cache_and_close_clears_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.touch()
            session = Sam2GuiSession([Image.new("RGBA", (4, 4))], checkpoint, work_size=4)
            retained = cache(1, (4, 4))
            session.appearance_feature_cache = retained

            session.reset_tracking_state()
            self.assertIs(session.appearance_feature_cache, retained)

            session.close()
            self.assertIsNone(session.appearance_feature_cache)

    def test_new_row_or_checkpoint_session_starts_without_previous_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first_checkpoint = Path(temporary) / "first.pt"
            second_checkpoint = Path(temporary) / "second.pt"
            first_checkpoint.touch()
            second_checkpoint.touch()
            first = Sam2GuiSession([Image.new("RGBA", (4, 4))], first_checkpoint, work_size=4)
            first.appearance_feature_cache = cache(1, (4, 4))
            first.close()

            second = Sam2GuiSession(
                [Image.new("RGBA", (6, 6))], second_checkpoint, work_size=6
            )
            self.assertIsNone(second.appearance_feature_cache)
            second.close()

    def test_extraction_uses_fresh_state_preserves_forward_state_and_reuses_cache(self) -> None:
        class FeaturePredictor:
            def __init__(self) -> None:
                self.init_calls = 0
                self.forward_calls = 0
                self.reset_states: list[object] = []

            def init_state(self, **_kwargs: object) -> dict[str, object]:
                self.init_calls += 1
                return {"images": torch.ones((2, 3, 8, 8)), "fresh": self.init_calls}

            def forward_image(self, image: torch.Tensor) -> dict[str, list[torch.Tensor]]:
                self.forward_calls += 1
                return {
                    "backbone_fpn": [
                        torch.ones((1, 2, 8, 8), dtype=torch.float32),
                        torch.ones((1, 4, 4, 4), dtype=torch.float32),
                    ]
                }

            def reset_state(self, state: object) -> None:
                self.reset_states.append(state)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.touch()
            frames = [Image.new("RGBA", (4, 4)) for _ in range(2)]
            session = Sam2GuiSession(frames, checkpoint, work_size=8)
            predictor = FeaturePredictor()
            forward_state = {"forward": "must remain untouched"}
            session.predictor = predictor
            session.device = torch.device("cpu")
            session.sam_frames_dir = Path(temporary)
            session.inference_state = forward_state

            first = session.extract_appearance_features()
            second = session.extract_appearance_features()

            self.assertIs(first, second)
            self.assertIs(session.inference_state, forward_state)
            self.assertEqual(forward_state, {"forward": "must remain untouched"})
            self.assertEqual(predictor.init_calls, 1)
            self.assertEqual(predictor.forward_calls, 2)
            self.assertEqual(len(predictor.reset_states), 1)
            self.assertEqual(first.level_index, 0)
            session.close()


class CandidateBankTests(unittest.TestCase):
    def diagnostic(self, frame: int, multimasks: np.ndarray) -> SamPreGateFrameDiagnostics:
        capture = SamPreGateDecoderCapture(
            decoder_call_index=frame,
            low_res_multimasks=multimasks[None].astype(np.float32),
            ious=np.array([[0.2, 0.8]], dtype=np.float32),
            object_score_logits=np.array([[-0.7]], dtype=np.float32),
            best_mask_index=1,
            best_iou=0.8,
            object_score_logit=-0.7,
            pre_gate_logits=multimasks[1].astype(np.float32),
        )
        return SamPreGateFrameDiagnostics(
            frame,
            "captured",
            (capture,),
            (),
            True,
            -1024.0,
            -1024.0,
        )

    def test_bank_uses_all_sources_multimasks_components_and_alpha_clip(self) -> None:
        frames = [Image.new("RGBA", (8, 8), (80, 80, 80, 255)) for _ in range(2)]
        alpha = np.full((8, 8), 255, dtype=np.uint8)
        alpha[:, 7] = 0
        frames[1].putalpha(Image.fromarray(alpha))
        authoritative = np.zeros((8, 8), dtype=bool)
        authoritative[1:3, 1:3] = True
        regular = [authoritative.copy(), authoritative.copy()]
        reverse = [authoritative.copy(), np.roll(authoritative, 2, axis=1)]
        first_logits = np.full((2, 4, 4), -2.0, dtype=np.float32)
        first_logits[:, :2, :2] = 2.0
        second_logits = np.full((2, 4, 4), -2.0, dtype=np.float32)
        second_logits[0, 0, 0] = 2.0
        second_logits[0, 3, 3] = 2.0  # two disconnected regions after upsample
        second_logits[1, 1:3, 1:3] = 2.0
        diagnostics = [self.diagnostic(0, first_logits), self.diagnostic(1, second_logits)]
        feature = np.ones((2, 8, 8), dtype=np.float16)
        feature[1] = 0
        appearance = AppearanceFeatureCache(0, ((2, 8, 8),), (feature.copy(), feature.copy()))
        snapshots = [mask.copy() for mask in regular + reverse]

        bank = build_identity_candidate_bank(
            frames,
            authoritative,
            regular,
            diagnostics,
            appearance,
            reverse_masks=reverse,
        )

        self.assertEqual(len(bank[0]), 1)
        sources = [item.source for item in bank[1]]
        self.assertIn("REGULAR", sources)
        self.assertIn("REVERSE", sources)
        self.assertIn("PRE_GATE_0", sources)
        self.assertIn("PRE_GATE_1", sources)
        self.assertGreaterEqual(sources.count("PRE_GATE_0"), 2)
        pre_gate = next(item for item in bank[1] if item.source == "PRE_GATE_1")
        self.assertAlmostEqual(pre_gate.predicted_iou, 0.8, places=5)
        self.assertAlmostEqual(pre_gate.object_score_logit, -0.7, places=5)
        self.assertFalse(any(item.mask[:, 7].any() for item in bank[1]))
        for actual, expected in zip(regular + reverse, snapshots, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_invalid_candidate_descriptor_is_reported_and_skipped(self) -> None:
        frames = [Image.new("RGBA", (4, 4), (80, 80, 80, 255)) for _ in range(2)]
        authoritative = np.zeros((4, 4), dtype=bool)
        authoritative[0, 0] = True
        regular = [authoritative.copy(), np.ones((4, 4), dtype=bool)]
        no_capture = [
            SamPreGateFrameDiagnostics(index, "no_capture", (), (), False, 0.0, 0.0)
            for index in range(2)
        ]
        target_feature = np.zeros((2, 4, 4), dtype=np.float16)
        target_feature[0, 0, 0] = 1.0
        candidate_feature = np.zeros((2, 4, 4), dtype=np.float16)
        appearance = AppearanceFeatureCache(
            0, ((2, 4, 4),), (target_feature, candidate_feature)
        )
        invalid: list[str] = []

        bank = build_identity_candidate_bank(
            frames,
            authoritative,
            regular,
            no_capture,
            appearance,
            invalid_descriptors=invalid,
        )

        self.assertEqual(bank[1], ())
        self.assertEqual(len(invalid), 1)
        self.assertIn("INVALID_DESCRIPTOR", invalid[0])


class GlobalPathTests(unittest.TestCase):
    def run_path(self, bank: list[list[IdentityMaskCandidate]]):
        authoritative = bank[0][0].mask
        return select_identity_global_path(bank, authoritative, cache(len(bank)), descriptor(1.0))

    def test_large_motion_high_identity_beats_nearby_distractor(self) -> None:
        bank = [[candidate(0, "AUTHORITATIVE/component0", 2, 1.0, "AUTHORITATIVE")]]
        for frame, target_x, distractor_x in ((1, 8, 3), (2, 38, 4), (3, 70, 5)):
            bank.append(
                [
                    candidate(frame, f"target-{frame}", target_x, 0.98, "PRE_GATE_0"),
                    candidate(frame, f"distractor-{frame}", distractor_x, 0.10, "REGULAR"),
                ]
            )
        result = self.run_path(bank)
        self.assertEqual([item.candidate_id for item in result.selected_candidates if item], [
            "AUTHORITATIVE/component0", "target-1", "target-2", "target-3"
        ])

    def test_occlusion_then_distant_reappearance(self) -> None:
        bank = [
            [candidate(0, "AUTHORITATIVE/component0", 2, 1.0, "AUTHORITATIVE")],
            [candidate(1, "wrong-1", 3, 0.05)],
            [candidate(2, "wrong-2", 4, 0.05)],
            [candidate(3, "target-3", 70, 0.99, "REVERSE")],
        ]
        result = self.run_path(bank)
        self.assertEqual(
            [None if item is None else item.candidate_id for item in result.selected_candidates],
            ["AUTHORITATIVE/component0", None, None, "target-3"],
        )

    def test_global_backtracking_prefers_sequence_over_local_first_choice(self) -> None:
        bridge = replace(
            candidate(1, "bridge", 20, 0.80, "PRE_GATE_0"),
            descriptor=np.array([0.8, 0.6, 0.0], dtype=np.float32),
        )
        local = replace(
            candidate(1, "local", 3, 0.85),
            descriptor=np.array([0.85, 0.0, np.sqrt(1.0 - 0.85**2)], dtype=np.float32),
        )
        target = replace(
            candidate(2, "target", 40, 0.80, "PRE_GATE_1"),
            descriptor=np.array([0.8, 0.6, 0.0], dtype=np.float32),
        )
        authoritative = replace(
            candidate(0, "AUTHORITATIVE/component0", 2, 1.0, "AUTHORITATIVE"),
            descriptor=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        drift = replace(
            candidate(2, "drift", 4, 0.20),
            descriptor=np.array([0.2, 0.0, np.sqrt(1.0 - 0.2**2)], dtype=np.float32),
        )
        bank = [
            [authoritative],
            [bridge, local],
            [target, drift],
        ]
        result = self.run_path(bank)
        self.assertEqual(result.selected_candidates[1].candidate_id, "bridge")
        self.assertEqual(result.selected_candidates[2].candidate_id, "target")

    def test_source_prior_is_not_a_hard_rule_and_frame0_is_exact(self) -> None:
        authoritative = square_mask(2)
        bank = [
            [candidate(0, "AUTHORITATIVE/component0", 2, 1.0, "AUTHORITATIVE")],
            [candidate(1, "regular-wrong", 3, 0.10, "REGULAR"), candidate(1, "pg-right", 60, 0.98, "PRE_GATE_2")],
        ]
        result = select_identity_global_path(bank, authoritative, cache(2), descriptor(1.0))
        self.assertEqual(result.selected_candidates[1].candidate_id, "pg-right")
        np.testing.assert_array_equal(result.masks[0], authoritative)


class IdentityAdoptionTests(unittest.TestCase):
    def test_switch_does_not_adopt_and_legacy_can_be_restored(self) -> None:
        regular = [square_mask(1)]
        identity = [square_mask(20)]
        state = MaskSourceState(regular)
        active = state.active_masks[0].copy()
        state.set_identity_masks(identity)
        state.set_preview_mode("identity")
        np.testing.assert_array_equal(state.active_masks[0], active)
        self.assertTrue(state.preview_dirty)

        adopted_identity = state.adopt_preview()
        np.testing.assert_array_equal(adopted_identity[0], identity[0])
        state.set_preview_mode("legacy")
        np.testing.assert_array_equal(state.preview_masks[0], regular[0])
        adopted_legacy = state.adopt_preview()
        np.testing.assert_array_equal(adopted_legacy[0], regular[0])

    def test_recolor_and_export_use_adopted_identity_mask(self) -> None:
        frame = Image.new("RGBA", (80, 12), (120, 120, 120, 255))
        state = MaskSourceState([square_mask(1)])
        identity = [square_mask(30)]
        state.set_identity_masks(identity)
        state.set_preview_mode("identity")
        adopted = state.adopt_preview()
        recolored = recolor_frame_sequence([frame], adopted, (210, 48, 48))

        with tempfile.TemporaryDirectory() as temporary:
            outputs = save_results(Path(temporary), [frame], adopted, recolored, columns=1)
            with Image.open(outputs["masks"] / "frame_000.png") as saved:
                exported = np.asarray(saved).copy() > 0

        np.testing.assert_array_equal(exported, identity[0])
        self.assertFalse(np.array_equal(np.asarray(recolored[0]), np.asarray(frame)))


if __name__ == "__main__":
    unittest.main()
