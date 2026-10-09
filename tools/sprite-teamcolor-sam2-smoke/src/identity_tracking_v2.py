"""Contrastive multi-scale identity scoring for the experimental v2 preview.

The v1 tracker remains in :mod:`identity_tracking`.  This module consumes the
same candidate masks but scores them against both the selected frame-0 target
and nearby, unselected sprite context.  It never runs or changes SAM tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter

from identity_tracking import (
    AREA_CHANGE_PENALTY_WEIGHT,
    BBOX_CHANGE_PENALTY_WEIGHT,
    MAX_APPEARANCE_FEATURE_CACHE_BYTES,
    MOTION_PENALTY_WEIGHT,
    OCCLUSION_ENTER_PENALTY,
    OCCLUSION_EXIT_PENALTY,
    OCCLUSION_STAY_PENALTY,
    SOURCE_PRIORS,
    SOURCE_PRIOR_WEIGHT,
    IdentityMaskCandidate,
    cosine_similarity,
    pool_mask_descriptor,
)


NEGATIVE_RING_MIN_RADIUS = 2
NEGATIVE_RING_MAX_RADIUS = 24
NEGATIVE_RING_BBOX_DIAGONAL_FACTOR = 0.25
NEGATIVE_RING_MIN_VISIBLE_PIXELS = 8
NEGATIVE_RING_EXPANSION_FACTOR = 2
MIN_REFERENCE_SEPARATION = 0.01
CONTRASTIVE_NORMALIZATION_EPSILON = 1e-6
CONTRASTIVE_SCORE_CLAMP = 2.0

SAM_CONTRASTIVE_WEIGHT = 0.60
CONTEXT_CONTRAST_WEIGHT = 0.10
PIXEL_CONTRAST_WEIGHT = 0.15
SHAPE_IDENTITY_WEIGHT = 0.15
FUSED_IDENTITY_NODE_WEIGHT = 2.25
FUSED_IDENTITY_BASELINE = 0.35
AMBIGUITY_MARGIN_SCALE = 0.15
AMBIGUITY_PENALTY_WEIGHT = 0.30
V2_TEMPORAL_APPEARANCE_WEIGHT = 0.10
V2_OCCLUSION_NODE_SCORE = 0.0

PIXEL_HISTOGRAM_BINS = 8
SHAPE_AREA_WEIGHT = 0.45
SHAPE_ASPECT_WEIGHT = 0.20
SHAPE_FILL_WEIGHT = 0.20
SHAPE_COMPACTNESS_WEIGHT = 0.15


@dataclass(frozen=True)
class MultiScaleFeatureCache:
    selected_levels: tuple[int, ...]
    level_shapes: tuple[tuple[int, int, int], ...]
    features_by_level: tuple[tuple[np.ndarray, ...], ...]
    cached_dtype: str = "float16"

    @property
    def frame_count(self) -> int:
        return len(self.features_by_level[0])

    @property
    def total_bytes(self) -> int:
        return sum(feature.nbytes for level in self.features_by_level for feature in level)

    def level_features(self, level_index: int) -> tuple[np.ndarray, ...]:
        position = self.selected_levels.index(level_index)
        return self.features_by_level[position]

    def level_bytes(self, level_index: int) -> int:
        return sum(feature.nbytes for feature in self.level_features(level_index))


@dataclass(frozen=True)
class NegativeContext:
    mask: np.ndarray | None
    radius: int
    visible_pixels: int
    bbox: tuple[int, int, int, int]
    fallback_used: bool


@dataclass(frozen=True)
class ShapeDescriptor:
    area: int
    width: int
    height: int
    aspect_ratio: float
    fill_ratio: float
    compactness: float


@dataclass(frozen=True)
class SamLevelEvidence:
    level_index: int
    positive_similarity: float
    negative_similarity: float | None
    raw_margin: float | None
    normalized_margin: float | None
    reference_separation: float | None
    valid: bool


@dataclass(frozen=True)
class FusedCandidateEvidence:
    candidate: IdentityMaskCandidate
    sam_levels: tuple[SamLevelEvidence, ...]
    sam_fused_contrastive: float
    context_contrast: float | None
    pixel_positive_similarity: float
    pixel_negative_similarity: float | None
    pixel_contrastive_score: float
    shape_similarity: float
    ambiguity_penalty: float
    fused_identity_score: float
    temporal_similarity: float | None = None
    motion_penalty: float = 0.0
    area_penalty: float = 0.0
    bbox_penalty: float = 0.0
    node_score: float = 0.0
    best_dp_score: float = -math.inf
    selected: bool = False


@dataclass(frozen=True)
class FusedFrameDiagnostics:
    frame_index: int
    candidates: tuple[FusedCandidateEvidence, ...]
    identity_margin: float | None
    occluded_score: float
    occluded_selected: bool
    selected_label: str


@dataclass(frozen=True)
class FusedIdentityResult:
    masks: tuple[np.ndarray, ...]
    selected_candidates: tuple[IdentityMaskCandidate | None, ...]
    frames: tuple[FusedFrameDiagnostics, ...]
    feature_cache: MultiScaleFeatureCache
    target_positive_descriptors: tuple[tuple[int, np.ndarray], ...]
    target_negative_descriptors: tuple[tuple[int, np.ndarray], ...]
    reference_separations: tuple[tuple[int, float | None], ...]
    negative_context: NegativeContext
    pixel_positive_descriptor: np.ndarray
    pixel_negative_descriptor: np.ndarray | None
    target_shape: ShapeDescriptor
    reverse_warning: str | None = None
    invalid_descriptors: tuple[str, ...] = ()


def _normalize_level_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in shape)
    if len(values) == 4 and values[0] == 1:
        values = values[1:]
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"Expected a CxHxW feature shape, got {tuple(shape)}")
    return values


def feature_level_cache_bytes(shape: Sequence[int], frame_count: int) -> int:
    channels, height, width = _normalize_level_shape(shape)
    if frame_count <= 0:
        raise ValueError("Frame count must be positive")
    return frame_count * channels * height * width * np.dtype(np.float16).itemsize


def choose_multiscale_feature_levels(
    level_shapes: Sequence[Sequence[int]],
    frame_count: int,
    max_bytes: int = MAX_APPEARANCE_FEATURE_CACHE_BYTES,
) -> tuple[int, ...]:
    """Choose high-resolution then deepest-semantic levels within a real byte budget."""
    if not level_shapes or frame_count <= 0 or max_bytes <= 0:
        raise ValueError("Feature shapes, frame count, and cache limit must be positive")
    normalized = tuple(_normalize_level_shape(shape) for shape in level_shapes)
    sizes = tuple(feature_level_cache_bytes(shape, frame_count) for shape in normalized)
    highest_resolution = max(
        range(len(normalized)), key=lambda index: normalized[index][1] * normalized[index][2]
    )
    deepest_semantic = min(
        range(len(normalized)),
        key=lambda index: (normalized[index][1] * normalized[index][2], -normalized[index][0]),
    )
    priority = [highest_resolution]
    if deepest_semantic != highest_resolution:
        priority.append(deepest_semantic)
    priority.extend(
        index
        for index in sorted(
            range(len(normalized)),
            key=lambda item: normalized[item][1] * normalized[item][2],
            reverse=True,
        )
        if index not in priority
    )
    selected: list[int] = []
    used = 0
    for index in priority:
        if used + sizes[index] <= max_bytes:
            selected.append(index)
            used += sizes[index]
    if not selected:
        raise MemoryError(
            f"No SAM feature level fits {max_bytes} bytes; smallest estimate is {min(sizes)}"
        )
    return tuple(selected)


def make_multiscale_feature_cache(
    selected_levels: Sequence[int],
    level_shapes: Sequence[Sequence[int]],
    features_by_level: Mapping[int, Sequence[np.ndarray]],
) -> MultiScaleFeatureCache:
    normalized = tuple(_normalize_level_shape(shape) for shape in level_shapes)
    levels = tuple(int(level) for level in selected_levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("Selected feature levels must be non-empty and unique")
    owned_levels: list[tuple[np.ndarray, ...]] = []
    frame_count: int | None = None
    for level in levels:
        if not 0 <= level < len(normalized) or level not in features_by_level:
            raise ValueError(f"Missing selected SAM feature level {level}")
        expected = normalized[level]
        owned: list[np.ndarray] = []
        for feature in features_by_level[level]:
            values = np.asarray(feature)
            if values.shape != expected:
                raise ValueError(f"Feature shape {values.shape} does not match level {expected}")
            # Avoid an unnecessary copy when the extractor already owns float16 CPU data.
            owned.append(values if values.dtype == np.float16 else values.astype(np.float16, copy=True))
        if not owned:
            raise ValueError("Every selected level needs at least one frame")
        if frame_count is None:
            frame_count = len(owned)
        elif len(owned) != frame_count:
            raise ValueError("Selected feature levels must have the same frame count")
        owned_levels.append(tuple(owned))
    return MultiScaleFeatureCache(levels, normalized, tuple(owned_levels))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Mask must be non-empty")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    selection = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return selection.copy()
    image = Image.fromarray(selection.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0


def visible_alpha_mask(frame: Image.Image) -> np.ndarray:
    if "A" not in frame.getbands():
        return np.ones((frame.height, frame.width), dtype=bool)
    return np.asarray(frame.getchannel("A"), dtype=np.uint8) > 0


def build_negative_context(
    target_mask: np.ndarray,
    frame: Image.Image,
    *,
    min_visible_pixels: int = NEGATIVE_RING_MIN_VISIBLE_PIXELS,
) -> NegativeContext:
    target = np.asarray(target_mask, dtype=bool)
    if target.shape != (frame.height, frame.width) or not target.any():
        raise ValueError("Target mask must be non-empty and match the frame")
    left, top, right, bottom = _bbox(target)
    diagonal = math.hypot(right - left + 1, bottom - top + 1)
    radius = int(round(diagonal * NEGATIVE_RING_BBOX_DIAGONAL_FACTOR))
    radius = max(NEGATIVE_RING_MIN_RADIUS, min(NEGATIVE_RING_MAX_RADIUS, radius))
    visible = visible_alpha_mask(frame)
    ring = _dilate(target, radius) & ~target & visible
    fallback_used = False
    while int(ring.sum()) < min_visible_pixels and radius < NEGATIVE_RING_MAX_RADIUS:
        fallback_used = True
        radius = min(NEGATIVE_RING_MAX_RADIUS, max(radius + 1, radius * NEGATIVE_RING_EXPANSION_FACTOR))
        ring = _dilate(target, radius) & ~target & visible
    if int(ring.sum()) < min_visible_pixels:
        fallback_used = True
        margin = NEGATIVE_RING_MAX_RADIUS
        local = np.zeros_like(target)
        local[max(0, top - margin) : min(target.shape[0], bottom + margin + 1),
              max(0, left - margin) : min(target.shape[1], right + margin + 1)] = True
        ring = local & ~target & visible
    if int(ring.sum()) < min_visible_pixels:
        ring_bbox = _bbox(ring) if ring.any() else (left, top, right, bottom)
        return NegativeContext(None, radius, int(ring.sum()), ring_bbox, True)
    return NegativeContext(ring.copy(), radius, int(ring.sum()), _bbox(ring), fallback_used)


def normalized_contrastive_score(
    positive_similarity: float,
    negative_similarity: float,
    reference_separation: float,
) -> float:
    value = (positive_similarity - negative_similarity) / max(
        reference_separation, CONTRASTIVE_NORMALIZATION_EPSILON
    )
    return float(np.clip(value, -CONTRASTIVE_SCORE_CLAMP, CONTRASTIVE_SCORE_CLAMP))


def fuse_level_scores(evidence: Sequence[SamLevelEvidence]) -> float:
    valid = [item for item in evidence if item.valid and item.normalized_margin is not None]
    if not valid:
        return 0.0
    weights = np.asarray(
        [max(float(item.reference_separation or 0.0), MIN_REFERENCE_SEPARATION) for item in valid],
        dtype=np.float64,
    )
    values = np.asarray([float(item.normalized_margin) for item in valid], dtype=np.float64)
    return float(np.average(values, weights=weights))


def make_pixel_descriptor(frame: Image.Image, mask: np.ndarray) -> np.ndarray | None:
    selection = np.asarray(mask, dtype=bool)
    if selection.shape != (frame.height, frame.width) or not selection.any():
        return None
    rgb = np.asarray(frame.convert("RGB"), dtype=np.float32)[selection] / 255.0
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    maximum = rgb.max(axis=1)
    minimum = rgb.min(axis=1)
    saturation = np.zeros_like(maximum)
    np.divide(maximum - minimum, maximum, out=saturation, where=maximum > 1e-6)
    lum_hist = np.histogram(luminance, bins=PIXEL_HISTOGRAM_BINS, range=(0.0, 1.0))[0]
    sat_hist = np.histogram(saturation, bins=PIXEL_HISTOGRAM_BINS, range=(0.0, 1.0))[0]
    values = np.concatenate(
        (
            rgb.mean(axis=0),
            rgb.std(axis=0),
            np.array([luminance.mean(), luminance.std()], dtype=np.float32),
            lum_hist.astype(np.float32) / len(rgb),
            sat_hist.astype(np.float32) / len(rgb),
        )
    ).astype(np.float32)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return values / norm


def make_shape_descriptor(mask: np.ndarray) -> ShapeDescriptor:
    selection = np.asarray(mask, dtype=bool)
    area = int(selection.sum())
    if area <= 0:
        raise ValueError("Shape mask must be non-empty")
    left, top, right, bottom = _bbox(selection)
    width = right - left + 1
    height = bottom - top + 1
    padded = np.pad(selection, 1, mode="constant")
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    perimeter = max(1, int(np.count_nonzero(selection & ~interior)))
    compactness = float(4.0 * math.pi * area / (perimeter * perimeter))
    return ShapeDescriptor(
        area,
        width,
        height,
        width / max(height, 1),
        area / float(width * height),
        compactness,
    )


def shape_similarity(target: ShapeDescriptor, candidate: ShapeDescriptor) -> float:
    area_delta = abs(math.log(candidate.area / target.area))
    aspect_delta = abs(math.log(candidate.aspect_ratio / target.aspect_ratio))
    fill_delta = abs(candidate.fill_ratio - target.fill_ratio)
    compactness_delta = abs(candidate.compactness - target.compactness)
    distance = (
        SHAPE_AREA_WEIGHT * area_delta
        + SHAPE_ASPECT_WEIGHT * aspect_delta
        + SHAPE_FILL_WEIGHT * fill_delta
        + SHAPE_COMPACTNESS_WEIGHT * compactness_delta
    )
    return float(math.exp(-distance))


def _source_prior(candidate: IdentityMaskCandidate) -> float:
    family = "PRE_GATE" if candidate.source.startswith("PRE_GATE_") else candidate.source
    return SOURCE_PRIOR_WEIGHT * SOURCE_PRIORS.get(family, 0.0)


def _transition_metrics(
    previous: IdentityMaskCandidate,
    current: IdentityMaskCandidate,
    frame_shape: tuple[int, int],
) -> tuple[float, float, float, float, float]:
    height, width = frame_shape
    diagonal = max(math.hypot(width, height), 1.0)
    motion = math.dist(previous.centroid, current.centroid) / diagonal
    area = abs(math.log(max(current.area, 1) / max(previous.area, 1)))
    previous_width = previous.bbox[2] - previous.bbox[0] + 1
    previous_height = previous.bbox[3] - previous.bbox[1] + 1
    current_width = current.bbox[2] - current.bbox[0] + 1
    current_height = current.bbox[3] - current.bbox[1] + 1
    bbox = (
        abs(math.log(current_width / max(previous_width, 1)))
        + abs(math.log(current_height / max(previous_height, 1)))
    ) / 2.0
    temporal = cosine_similarity(previous.descriptor, current.descriptor)
    motion_penalty = MOTION_PENALTY_WEIGHT * motion
    area_penalty = AREA_CHANGE_PENALTY_WEIGHT * area
    bbox_penalty = BBOX_CHANGE_PENALTY_WEIGHT * bbox
    transition = (
        V2_TEMPORAL_APPEARANCE_WEIGHT * temporal
        - motion_penalty
        - area_penalty
        - bbox_penalty
    )
    return transition, temporal, motion_penalty, area_penalty, bbox_penalty


def _candidate_ring(mask: np.ndarray, visible: np.ndarray, radius: int) -> np.ndarray | None:
    ring = _dilate(mask, max(1, radius)) & ~mask & visible
    return ring if int(ring.sum()) >= NEGATIVE_RING_MIN_VISIBLE_PIXELS else None


def score_fused_candidates(
    frames: Sequence[Image.Image],
    candidate_bank: Sequence[Sequence[IdentityMaskCandidate]],
    authoritative_mask: np.ndarray,
    feature_cache: MultiScaleFeatureCache,
) -> tuple[
    list[list[FusedCandidateEvidence]],
    tuple[tuple[int, np.ndarray], ...],
    tuple[tuple[int, np.ndarray], ...],
    tuple[tuple[int, float | None], ...],
    NegativeContext,
    np.ndarray,
    np.ndarray | None,
    ShapeDescriptor,
    tuple[str, ...],
]:
    if len(frames) != len(candidate_bank) or len(frames) != feature_cache.frame_count:
        raise ValueError("Fused identity inputs must have matching frame counts")
    target = np.asarray(authoritative_mask, dtype=bool)
    negative = build_negative_context(target, frames[0])
    target_shape = make_shape_descriptor(target)
    pixel_positive = make_pixel_descriptor(frames[0], target)
    if pixel_positive is None:
        raise ValueError("Frame 0 target has no valid pixel descriptor")
    pixel_negative = (
        None if negative.mask is None else make_pixel_descriptor(frames[0], negative.mask)
    )

    positives: list[tuple[int, np.ndarray]] = []
    negatives: list[tuple[int, np.ndarray]] = []
    separations: list[tuple[int, float | None]] = []
    invalid: list[str] = []
    for level in feature_cache.selected_levels:
        target_descriptor = pool_mask_descriptor(feature_cache.level_features(level)[0], target)
        if target_descriptor is None:
            invalid.append(f"level {level} target: INVALID_DESCRIPTOR")
            separations.append((level, None))
            continue
        positives.append((level, target_descriptor))
        negative_descriptor = None
        if negative.mask is not None:
            negative_descriptor = pool_mask_descriptor(
                feature_cache.level_features(level)[0], negative.mask
            )
        if negative_descriptor is not None:
            negatives.append((level, negative_descriptor))
            separation = 1.0 - cosine_similarity(target_descriptor, negative_descriptor)
            separations.append((level, separation))
        else:
            separations.append((level, None))
    if not positives:
        raise ValueError("No selected SAM level has a valid target descriptor")
    positive_by_level = dict(positives)
    negative_by_level = dict(negatives)
    separation_by_level = dict(separations)

    target_contrast_vectors: dict[int, np.ndarray] = {}
    for level, positive in positives:
        level_negative = negative_by_level.get(level)
        if level_negative is not None:
            contrast = positive - level_negative
            norm = float(np.linalg.norm(contrast))
            if norm > 1e-12:
                target_contrast_vectors[level] = (contrast / norm).astype(np.float32)

    scored: list[list[FusedCandidateEvidence]] = []
    for frame_index, candidates in enumerate(candidate_bank):
        frame_evidence: list[FusedCandidateEvidence] = []
        visible = visible_alpha_mask(frames[frame_index])
        for candidate in candidates:
            level_evidence: list[SamLevelEvidence] = []
            context_scores: list[float] = []
            candidate_radius = max(
                NEGATIVE_RING_MIN_RADIUS,
                min(
                    NEGATIVE_RING_MAX_RADIUS,
                    int(round(math.hypot(
                        candidate.bbox[2] - candidate.bbox[0] + 1,
                        candidate.bbox[3] - candidate.bbox[1] + 1,
                    ) * NEGATIVE_RING_BBOX_DIAGONAL_FACTOR)),
                ),
            )
            surrounding = _candidate_ring(candidate.mask, visible, candidate_radius)
            for level in feature_cache.selected_levels:
                target_positive = positive_by_level.get(level)
                if target_positive is None:
                    continue
                feature_map = feature_cache.level_features(level)[frame_index]
                descriptor = pool_mask_descriptor(feature_map, candidate.mask)
                if descriptor is None:
                    invalid.append(f"frame_{frame_index:03} {candidate.candidate_id} level {level}: INVALID_DESCRIPTOR")
                    continue
                positive_similarity = cosine_similarity(descriptor, target_positive)
                target_negative = negative_by_level.get(level)
                separation = separation_by_level.get(level)
                if target_negative is None or separation is None:
                    level_evidence.append(
                        SamLevelEvidence(level, positive_similarity, None, None, None, separation, False)
                    )
                else:
                    negative_similarity = cosine_similarity(descriptor, target_negative)
                    margin = positive_similarity - negative_similarity
                    valid = separation >= MIN_REFERENCE_SEPARATION
                    normalized = normalized_contrastive_score(
                        positive_similarity, negative_similarity, separation
                    )
                    level_evidence.append(
                        SamLevelEvidence(
                            level,
                            positive_similarity,
                            negative_similarity,
                            margin,
                            normalized,
                            separation,
                            valid,
                        )
                    )
                if surrounding is not None and level in target_contrast_vectors:
                    context_descriptor = pool_mask_descriptor(feature_map, surrounding)
                    if context_descriptor is not None:
                        contrast = descriptor - context_descriptor
                        norm = float(np.linalg.norm(contrast))
                        if norm > 1e-12:
                            context_scores.append(
                                cosine_similarity(
                                    contrast / norm, target_contrast_vectors[level]
                                )
                            )
            sam_fused = fuse_level_scores(level_evidence)
            context_score = float(np.mean(context_scores)) if context_scores else None
            candidate_pixel = make_pixel_descriptor(frames[frame_index], candidate.mask)
            if candidate_pixel is None:
                invalid.append(f"frame_{frame_index:03} {candidate.candidate_id}: INVALID_PIXEL_DESCRIPTOR")
                continue
            pixel_positive_similarity = cosine_similarity(candidate_pixel, pixel_positive)
            if pixel_negative is None:
                pixel_negative_similarity = None
                pixel_contrastive = 0.0
            else:
                pixel_negative_similarity = cosine_similarity(candidate_pixel, pixel_negative)
                pixel_reference_separation = max(
                    1.0 - cosine_similarity(pixel_positive, pixel_negative),
                    MIN_REFERENCE_SEPARATION,
                )
                pixel_contrastive = normalized_contrastive_score(
                    pixel_positive_similarity,
                    pixel_negative_similarity,
                    pixel_reference_separation,
                )
            shape_score = shape_similarity(target_shape, make_shape_descriptor(candidate.mask))
            active_context = 0.0 if context_score is None else context_score
            fused = (
                SAM_CONTRASTIVE_WEIGHT * sam_fused
                + CONTEXT_CONTRAST_WEIGHT * active_context
                + PIXEL_CONTRAST_WEIGHT * pixel_contrastive
                + SHAPE_IDENTITY_WEIGHT * shape_score
            )
            frame_evidence.append(
                FusedCandidateEvidence(
                    candidate,
                    tuple(level_evidence),
                    sam_fused,
                    context_score,
                    pixel_positive_similarity,
                    pixel_negative_similarity,
                    pixel_contrastive,
                    shape_score,
                    0.0,
                    fused,
                )
            )
        if frame_index > 0 and len(frame_evidence) > 1:
            ordered = sorted((item.fused_identity_score for item in frame_evidence), reverse=True)
            margin = ordered[0] - ordered[1]
            penalty = AMBIGUITY_PENALTY_WEIGHT * math.exp(
                -max(0.0, margin) / AMBIGUITY_MARGIN_SCALE
            )
            frame_evidence = [
                FusedCandidateEvidence(**{**item.__dict__, "ambiguity_penalty": penalty})
                for item in frame_evidence
            ]
        scored.append(frame_evidence)
    return (
        scored,
        tuple((level, descriptor.copy()) for level, descriptor in positives),
        tuple((level, descriptor.copy()) for level, descriptor in negatives),
        tuple(separations),
        negative,
        pixel_positive.copy(),
        None if pixel_negative is None else pixel_negative.copy(),
        target_shape,
        tuple(invalid),
    )


def select_fused_identity_path(
    frames: Sequence[Image.Image],
    candidate_bank: Sequence[Sequence[IdentityMaskCandidate]],
    authoritative_mask: np.ndarray,
    feature_cache: MultiScaleFeatureCache,
    *,
    reverse_warning: str | None = None,
) -> FusedIdentityResult:
    """Score the unchanged candidate bank and run global DP with OCCLUDED."""
    (
        scored,
        positives,
        negatives,
        separations,
        negative_context,
        pixel_positive,
        pixel_negative,
        target_shape,
        invalid,
    ) = score_fused_candidates(frames, candidate_bank, authoritative_mask, feature_cache)
    target = np.asarray(authoritative_mask, dtype=bool)
    if not scored or len(scored[0]) != 1 or scored[0][0].candidate.source != "AUTHORITATIVE":
        raise ValueError("Frame 0 must contain only the authoritative candidate")
    states: list[list[FusedCandidateEvidence | None]] = [list(frame) + [None] for frame in scored]
    states[0] = [scored[0][0]]
    scores: list[list[float]] = [[0.0]]
    backpointers: list[list[int]] = [[-1]]
    metrics: list[list[tuple[float | None, float, float, float]]] = [[(None, 0.0, 0.0, 0.0)]]
    for frame_index in range(1, len(states)):
        frame_scores: list[float] = []
        frame_backpointers: list[int] = []
        frame_metrics: list[tuple[float | None, float, float, float]] = []
        for current in states[frame_index]:
            best_score = -math.inf
            best_previous = -1
            best_metrics = (None, 0.0, 0.0, 0.0)
            if current is None:
                node = V2_OCCLUSION_NODE_SCORE
            else:
                node = (
                    FUSED_IDENTITY_NODE_WEIGHT
                    * (current.fused_identity_score - FUSED_IDENTITY_BASELINE)
                    - current.ambiguity_penalty
                    + _source_prior(current.candidate)
                )
            for previous_index, previous in enumerate(states[frame_index - 1]):
                if current is None:
                    transition = -(
                        OCCLUSION_STAY_PENALTY if previous is None else OCCLUSION_ENTER_PENALTY
                    )
                    current_metrics = (None, 0.0, 0.0, 0.0)
                elif previous is None:
                    transition = -OCCLUSION_EXIT_PENALTY
                    current_metrics = (None, 0.0, 0.0, 0.0)
                else:
                    transition, temporal, motion, area, bbox = _transition_metrics(
                        previous.candidate, current.candidate, target.shape
                    )
                    current_metrics = (temporal, motion, area, bbox)
                score = scores[frame_index - 1][previous_index] + node + transition
                if score > best_score:
                    best_score = score
                    best_previous = previous_index
                    best_metrics = current_metrics
            frame_scores.append(best_score)
            frame_backpointers.append(best_previous)
            frame_metrics.append(best_metrics)
        scores.append(frame_scores)
        backpointers.append(frame_backpointers)
        metrics.append(frame_metrics)
    selected_indices = [0] * len(states)
    selected_indices[-1] = int(np.argmax(scores[-1]))
    for frame_index in range(len(states) - 1, 0, -1):
        selected_indices[frame_index - 1] = backpointers[frame_index][selected_indices[frame_index]]

    selected: list[IdentityMaskCandidate | None] = []
    masks: list[np.ndarray] = []
    diagnostics: list[FusedFrameDiagnostics] = []
    for frame_index, frame_states in enumerate(states):
        selected_index = selected_indices[frame_index]
        selected_state = frame_states[selected_index]
        selected_candidate = None if selected_state is None else selected_state.candidate
        selected.append(selected_candidate)
        masks.append(np.zeros_like(target) if selected_candidate is None else selected_candidate.mask.copy())
        evaluations: list[FusedCandidateEvidence] = []
        for index, evidence in enumerate(frame_states):
            if evidence is None:
                continue
            temporal, motion, area, bbox = metrics[frame_index][index]
            node = 0.0 if frame_index == 0 else (
                FUSED_IDENTITY_NODE_WEIGHT * (evidence.fused_identity_score - FUSED_IDENTITY_BASELINE)
                - evidence.ambiguity_penalty
                + _source_prior(evidence.candidate)
            )
            evaluations.append(
                FusedCandidateEvidence(
                    **{
                        **evidence.__dict__,
                        "temporal_similarity": temporal,
                        "motion_penalty": motion,
                        "area_penalty": area,
                        "bbox_penalty": bbox,
                        "node_score": node,
                        "best_dp_score": scores[frame_index][index],
                        "selected": index == selected_index,
                    }
                )
            )
        identity_scores = sorted(
            (item.fused_identity_score for item in evaluations), reverse=True
        )
        identity_margin = (
            identity_scores[0] - identity_scores[1] if len(identity_scores) > 1 else None
        )
        occluded_index = next(
            (index for index, value in enumerate(frame_states) if value is None), None
        )
        diagnostics.append(
            FusedFrameDiagnostics(
                frame_index,
                tuple(evaluations),
                identity_margin,
                -math.inf if occluded_index is None else scores[frame_index][occluded_index],
                selected_state is None,
                "OCCLUDED" if selected_candidate is None else selected_candidate.candidate_id,
            )
        )
    masks[0] = target.copy()
    return FusedIdentityResult(
        tuple(masks),
        tuple(selected),
        tuple(diagnostics),
        feature_cache,
        positives,
        negatives,
        separations,
        negative_context,
        pixel_positive,
        pixel_negative,
        target_shape,
        reverse_warning,
        invalid,
    )


def format_fused_identity_diagnostics(result: FusedIdentityResult) -> str:
    cache = result.feature_cache
    negative = result.negative_context
    context_active = any(
        candidate.context_contrast is not None
        for frame in result.frames
        for candidate in frame.candidates
    )
    pixel_contrast_active = result.pixel_negative_descriptor is not None
    active_modalities = [
        *(f"SAM level{level}" for level, value in result.reference_separations if value is not None and value >= MIN_REFERENCE_SEPARATION),
        *(('local context',) if context_active else ()),
        *(('pixel contrastive',) if pixel_contrast_active else ()),
        "shape",
    ]
    lines = [
        "Identity Global v2 (Fused Experimental) diagnostics",
        "Selected SAM feature levels:",
    ]
    for level in cache.selected_levels:
        lines.append(
            f"  level {level}: shape={cache.level_shapes[level]} dtype={cache.cached_dtype} "
            f"cache={cache.level_bytes(level) / (1024 * 1024):.3f} MiB"
        )
    lines.extend(
        [
            f"  total cache: {cache.total_bytes / (1024 * 1024):.3f} MiB",
            "Target positive descriptors:",
            *(f"  level {level}: dimension={value.size} norm={np.linalg.norm(value):.6f}" for level, value in result.target_positive_descriptors),
            "Negative context:",
            f"  radius: {negative.radius}",
            f"  visible pixels: {negative.visible_pixels}",
            f"  bbox: {negative.bbox}",
            f"  fallback used: {'YES' if negative.fallback_used else 'NO'}",
            f"  signal available: {'YES' if negative.mask is not None else 'NO'}",
            "Per-level reference separation:",
            *(f"  level {level}: {'N/A' if value is None else f'{value:.6f}'}" for level, value in result.reference_separations),
            "Active identity modalities:",
            *(f"  {modality}" for modality in active_modalities),
            f"Reverse candidates: {'unavailable - ' + result.reverse_warning if result.reverse_warning else 'available'}",
            f"Invalid descriptors: {len(result.invalid_descriptors)}",
            *(f"  {item}" for item in result.invalid_descriptors),
            "",
        ]
    )
    for frame in result.frames:
        ranked = sorted(frame.candidates, key=lambda item: item.fused_identity_score, reverse=True)
        lines.extend([f"frame_{frame.frame_index:03}", "  Top identity candidates:"])
        for rank, item in enumerate(ranked[:3], start=1):
            lines.append(f"    {rank}. {item.candidate.candidate_id}: {item.fused_identity_score:.6f}")
        lines.extend(
            [
                "  identity top1-top2 margin: " + ("N/A" if frame.identity_margin is None else f"{frame.identity_margin:.6f}"),
                f"  OCCLUDED score: {frame.occluded_score:.6f}",
                f"  final selected: {frame.selected_label}",
            ]
        )
        for item in frame.candidates:
            candidate = item.candidate
            lines.extend(
                [
                    f"  candidate {candidate.candidate_id}:",
                    f"    source/component: {candidate.source} / {candidate.component_index}",
                    f"    source prior: {_source_prior(candidate):.6f}",
                    f"    pixels: {candidate.area}",
                    f"    centroid: ({candidate.centroid[0]:.3f}, {candidate.centroid[1]:.3f})",
                    f"    bbox: {candidate.bbox}",
                    "    pre-gate predicted IoU: " + ("N/A" if candidate.predicted_iou is None else f"{candidate.predicted_iou:.6f}"),
                    "    object score logit: " + ("N/A" if candidate.object_score_logit is None else f"{candidate.object_score_logit:.6f}"),
                    f"    source metadata: {dict(candidate.source_metadata)}",
                ]
            )
            for level in item.sam_levels:
                lines.extend(
                    [
                        f"    SAM level{level.level_index}:",
                        f"      positive cosine: {level.positive_similarity:.6f}",
                        "      negative cosine: " + ("N/A" if level.negative_similarity is None else f"{level.negative_similarity:.6f}"),
                        "      raw margin: " + ("N/A" if level.raw_margin is None else f"{level.raw_margin:.6f}"),
                        "      normalized margin: " + ("N/A" if level.normalized_margin is None else f"{level.normalized_margin:.6f}"),
                        "      reference separation: " + ("N/A" if level.reference_separation is None else f"{level.reference_separation:.6f}"),
                        f"      valid: {'YES' if level.valid else 'NO'}",
                    ]
                )
            lines.extend(
                [
                    f"    SAM fused contrastive score: {item.sam_fused_contrastive:.6f}",
                    "    context contrast score: " + ("N/A" if item.context_contrast is None else f"{item.context_contrast:.6f}"),
                    f"    pixel positive similarity: {item.pixel_positive_similarity:.6f}",
                    "    pixel negative similarity: " + ("N/A" if item.pixel_negative_similarity is None else f"{item.pixel_negative_similarity:.6f}"),
                    f"    pixel contrastive score: {item.pixel_contrastive_score:.6f}",
                    f"    shape similarity: {item.shape_similarity:.6f}",
                    f"    ambiguity penalty: {item.ambiguity_penalty:.6f}",
                    f"    fused identity score: {item.fused_identity_score:.6f}",
                    "    temporal appearance cosine: " + ("N/A" if item.temporal_similarity is None else f"{item.temporal_similarity:.6f}"),
                    f"    motion penalty: {item.motion_penalty:.6f}",
                    f"    area penalty: {item.area_penalty:.6f}",
                    f"    bbox penalty: {item.bbox_penalty:.6f}",
                    f"    node score: {item.node_score:.6f}",
                    f"    best DP score: {item.best_dp_score:.6f}",
                    f"    selected: {'YES' if item.selected else 'NO'}",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
