"""SAM-feature identity descriptors and global mask-path selection.

This module is intentionally GUI independent.  It observes cached SAM image
encoder features and existing mask sources; it never changes SAM inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from PIL import Image


MAX_APPEARANCE_FEATURE_CACHE_BYTES = 96 * 1024 * 1024

TARGET_APPEARANCE_WEIGHT = 4.0
TARGET_APPEARANCE_BASELINE = 0.50
TEMPORAL_APPEARANCE_WEIGHT = 1.25
SOURCE_PRIOR_WEIGHT = 0.20
MOTION_PENALTY_WEIGHT = 0.30
AREA_CHANGE_PENALTY_WEIGHT = 0.18
BBOX_CHANGE_PENALTY_WEIGHT = 0.12
OCCLUSION_NODE_SCORE = -0.35
OCCLUSION_ENTER_PENALTY = 0.25
OCCLUSION_STAY_PENALTY = 0.05
OCCLUSION_EXIT_PENALTY = 0.20

SOURCE_PRIORS = {
    "AUTHORITATIVE": 1.0,
    "REGULAR": 0.35,
    "REVERSE": 0.25,
    "PRE_GATE": 0.0,
}


@dataclass(frozen=True)
class AppearanceFeatureCache:
    level_index: int
    level_shapes: tuple[tuple[int, int, int], ...]
    features: tuple[np.ndarray, ...]
    cached_dtype: str = "float16"

    @property
    def channels(self) -> int:
        return int(self.features[0].shape[0])

    @property
    def spatial_size(self) -> tuple[int, int]:
        return int(self.features[0].shape[2]), int(self.features[0].shape[1])

    @property
    def total_bytes(self) -> int:
        return sum(feature.nbytes for feature in self.features)


@dataclass(frozen=True)
class IdentityMaskCandidate:
    frame_index: int
    candidate_id: str
    source: str
    component_index: int
    mask: np.ndarray
    area: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    descriptor: np.ndarray
    target_similarity: float
    predicted_iou: float | None = None
    object_score_logit: float | None = None
    source_metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class IdentityCandidateEvaluation:
    candidate_id: str
    source: str
    component_index: int
    pixels: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    target_similarity: float
    predicted_iou: float | None
    object_score_logit: float | None
    source_metadata: tuple[tuple[str, str], ...]
    temporal_similarity: float | None
    motion_penalty: float
    area_penalty: float
    bbox_penalty: float
    node_score: float
    best_dp_score: float
    selected: bool


@dataclass(frozen=True)
class IdentityFrameDiagnostics:
    frame_index: int
    candidates: tuple[IdentityCandidateEvaluation, ...]
    occluded_score: float
    occluded_selected: bool
    selected_label: str


@dataclass(frozen=True)
class IdentityGlobalResult:
    masks: tuple[np.ndarray, ...]
    selected_candidates: tuple[IdentityMaskCandidate | None, ...]
    frames: tuple[IdentityFrameDiagnostics, ...]
    feature_cache: AppearanceFeatureCache
    target_descriptor: np.ndarray
    reverse_warning: str | None = None
    invalid_descriptors: tuple[str, ...] = ()


def _normalize_level_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in shape)
    if len(values) == 4 and values[0] == 1:
        values = values[1:]
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"Expected a CxHxW feature shape, got {tuple(shape)}")
    return values


def choose_appearance_feature_level(
    level_shapes: Sequence[Sequence[int]],
    frame_count: int,
    max_bytes: int = MAX_APPEARANCE_FEATURE_CACHE_BYTES,
) -> int:
    """Choose the highest spatial-resolution float16 level within the cache cap."""
    if frame_count <= 0 or max_bytes <= 0:
        raise ValueError("Frame count and feature cache limit must be positive")
    normalized = tuple(_normalize_level_shape(shape) for shape in level_shapes)
    fitting = [
        index
        for index, (channels, height, width) in enumerate(normalized)
        if frame_count * channels * height * width * np.dtype(np.float16).itemsize <= max_bytes
    ]
    if not fitting:
        estimates = [
            frame_count * channels * height * width * np.dtype(np.float16).itemsize
            for channels, height, width in normalized
        ]
        raise MemoryError(
            f"No SAM feature level fits {max_bytes} bytes; smallest estimate is {min(estimates)}"
        )
    return max(fitting, key=lambda index: normalized[index][1] * normalized[index][2])


def make_appearance_feature_cache(
    level_index: int,
    level_shapes: Sequence[Sequence[int]],
    frame_features: Sequence[np.ndarray],
) -> AppearanceFeatureCache:
    normalized_shapes = tuple(_normalize_level_shape(shape) for shape in level_shapes)
    if not 0 <= level_index < len(normalized_shapes):
        raise ValueError("Feature level index is outside the inspected FPN levels")
    expected = normalized_shapes[level_index]
    owned: list[np.ndarray] = []
    for feature in frame_features:
        values = np.asarray(feature)
        if values.shape != expected:
            raise ValueError(f"Feature shape {values.shape} does not match selected level {expected}")
        owned.append(values.astype(np.float16, copy=True))
    if not owned:
        raise ValueError("At least one frame feature is required")
    return AppearanceFeatureCache(level_index, normalized_shapes, tuple(owned))


def pool_mask_descriptor(feature_map: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Coverage-weighted pool of CxHxW SAM features, returned L2-normalized."""
    features = np.asarray(feature_map)
    selection = np.asarray(mask, dtype=bool)
    if features.ndim != 3 or selection.ndim != 2:
        raise ValueError("Feature map must be CxHxW and mask must be 2D")
    if not selection.any():
        return None
    _, feature_height, feature_width = features.shape
    coverage = np.asarray(
        Image.fromarray(selection.astype(np.uint8) * 255, mode="L").resize(
            (feature_width, feature_height), Image.Resampling.BOX
        ),
        dtype=np.float32,
    ) / 255.0
    if not np.any(coverage > 0):
        # Preserve support for a target smaller than one feature cell.
        ys, xs = np.nonzero(selection)
        x = min(feature_width - 1, int((float(xs.mean()) + 0.5) * feature_width / selection.shape[1]))
        y = min(feature_height - 1, int((float(ys.mean()) + 0.5) * feature_height / selection.shape[0]))
        coverage[y, x] = 1.0
    weights = coverage / float(coverage.sum())
    descriptor = np.sum(features.astype(np.float32, copy=False) * weights[None, :, :], axis=(1, 2))
    norm = float(np.linalg.norm(descriptor))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return (descriptor / norm).astype(np.float32, copy=False)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_values = np.asarray(left, dtype=np.float32)
    right_values = np.asarray(right, dtype=np.float32)
    if left_values.ndim != 1 or right_values.ndim != 1 or left_values.shape != right_values.shape:
        raise ValueError("Cosine descriptors must be equal-length 1D arrays")
    denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denominator <= 1e-12:
        raise ValueError("Cosine descriptor norm must be non-zero")
    return float(np.clip(np.dot(left_values, right_values) / denominator, -1.0, 1.0))


def _candidate_prior(candidate: IdentityMaskCandidate) -> float:
    family = "PRE_GATE" if candidate.source.startswith("PRE_GATE_") else candidate.source
    return SOURCE_PRIOR_WEIGHT * SOURCE_PRIORS.get(family, 0.0)


def _node_score(candidate: IdentityMaskCandidate) -> float:
    return (
        TARGET_APPEARANCE_WEIGHT
        * (candidate.target_similarity - TARGET_APPEARANCE_BASELINE)
        + _candidate_prior(candidate)
    )


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
        TEMPORAL_APPEARANCE_WEIGHT * temporal
        - motion_penalty
        - area_penalty
        - bbox_penalty
    )
    return transition, temporal, motion_penalty, area_penalty, bbox_penalty


def select_identity_global_path(
    candidate_bank: Sequence[Sequence[IdentityMaskCandidate]],
    authoritative_frame0_mask: np.ndarray,
    feature_cache: AppearanceFeatureCache,
    target_descriptor: np.ndarray,
    reverse_warning: str | None = None,
    invalid_descriptors: Sequence[str] = (),
) -> IdentityGlobalResult:
    """Run full-sequence Viterbi selection over candidates plus OCCLUDED."""
    if not candidate_bank or len(candidate_bank) != len(feature_cache.features):
        raise ValueError("Candidate bank and feature cache frame counts must match")
    frame0 = np.asarray(authoritative_frame0_mask, dtype=bool)
    if not frame0.any():
        raise ValueError("Authoritative frame 0 mask must be non-empty")
    first = tuple(candidate_bank[0])
    if len(first) != 1 or first[0].source != "AUTHORITATIVE":
        raise ValueError("Frame 0 must contain only the authoritative candidate")
    if not np.array_equal(first[0].mask, frame0):
        raise ValueError("Frame 0 authoritative candidate does not match the user mask")

    states: list[list[IdentityMaskCandidate | None]] = [list(frame) + [None] for frame in candidate_bank]
    states[0] = [first[0]]
    scores: list[list[float]] = [[0.0]]
    backpointers: list[list[int]] = [[-1]]
    best_metrics: list[list[tuple[float | None, float, float, float]]] = [[(None, 0.0, 0.0, 0.0)]]

    for frame_index in range(1, len(states)):
        current_scores: list[float] = []
        current_backpointers: list[int] = []
        current_metrics: list[tuple[float | None, float, float, float]] = []
        frame_shape = tuple(int(value) for value in states[0][0].mask.shape)
        for current in states[frame_index]:
            best_score = -math.inf
            best_previous = -1
            chosen_metrics: tuple[float | None, float, float, float] = (None, 0.0, 0.0, 0.0)
            node = OCCLUSION_NODE_SCORE if current is None else _node_score(current)
            for previous_index, previous in enumerate(states[frame_index - 1]):
                if current is None:
                    transition = -(
                        OCCLUSION_STAY_PENALTY if previous is None else OCCLUSION_ENTER_PENALTY
                    )
                    metrics = (None, 0.0, 0.0, 0.0)
                elif previous is None:
                    transition = -OCCLUSION_EXIT_PENALTY
                    metrics = (None, 0.0, 0.0, 0.0)
                else:
                    transition, temporal, motion, area, bbox = _transition_metrics(
                        previous, current, frame_shape
                    )
                    metrics = (temporal, motion, area, bbox)
                score = scores[frame_index - 1][previous_index] + node + transition
                if score > best_score:
                    best_score = score
                    best_previous = previous_index
                    chosen_metrics = metrics
            current_scores.append(best_score)
            current_backpointers.append(best_previous)
            current_metrics.append(chosen_metrics)
        scores.append(current_scores)
        backpointers.append(current_backpointers)
        best_metrics.append(current_metrics)

    selected_indices = [0] * len(states)
    selected_indices[-1] = int(np.argmax(scores[-1]))
    for frame_index in range(len(states) - 1, 0, -1):
        selected_indices[frame_index - 1] = backpointers[frame_index][selected_indices[frame_index]]

    selected: list[IdentityMaskCandidate | None] = []
    output_masks: list[np.ndarray] = []
    diagnostics: list[IdentityFrameDiagnostics] = []
    for frame_index, frame_states in enumerate(states):
        selected_index = selected_indices[frame_index]
        selected_candidate = frame_states[selected_index]
        selected.append(selected_candidate)
        output_masks.append(
            np.zeros_like(frame0) if selected_candidate is None else selected_candidate.mask.copy()
        )
        evaluations: list[IdentityCandidateEvaluation] = []
        for index, candidate in enumerate(frame_states):
            if candidate is None:
                continue
            temporal, motion, area, bbox = best_metrics[frame_index][index]
            evaluations.append(
                IdentityCandidateEvaluation(
                    candidate.candidate_id,
                    candidate.source,
                    candidate.component_index,
                    candidate.area,
                    candidate.centroid,
                    candidate.bbox,
                    candidate.target_similarity,
                    candidate.predicted_iou,
                    candidate.object_score_logit,
                    candidate.source_metadata,
                    temporal,
                    motion,
                    area,
                    bbox,
                    0.0 if frame_index == 0 else _node_score(candidate),
                    scores[frame_index][index],
                    index == selected_index,
                )
            )
        occluded_index = next(
            (index for index, candidate in enumerate(frame_states) if candidate is None), None
        )
        diagnostics.append(
            IdentityFrameDiagnostics(
                frame_index,
                tuple(evaluations),
                -math.inf if occluded_index is None else scores[frame_index][occluded_index],
                selected_candidate is None,
                "OCCLUDED" if selected_candidate is None else selected_candidate.candidate_id,
            )
        )
    output_masks[0] = frame0.copy()
    return IdentityGlobalResult(
        tuple(output_masks),
        tuple(selected),
        tuple(diagnostics),
        feature_cache,
        np.asarray(target_descriptor, dtype=np.float32).copy(),
        reverse_warning,
        tuple(str(item) for item in invalid_descriptors),
    )


def build_identity_candidate(
    *,
    frame_index: int,
    candidate_id: str,
    source: str,
    component_index: int,
    mask: np.ndarray,
    feature_map: np.ndarray,
    target_descriptor: np.ndarray,
    predicted_iou: float | None = None,
    object_score_logit: float | None = None,
    source_metadata: Mapping[str, object] | None = None,
) -> IdentityMaskCandidate | None:
    selection = np.asarray(mask, dtype=bool)
    descriptor = pool_mask_descriptor(feature_map, selection)
    if descriptor is None:
        return None
    ys, xs = np.nonzero(selection)
    metadata = tuple(sorted((str(key), str(value)) for key, value in (source_metadata or {}).items()))
    return IdentityMaskCandidate(
        frame_index,
        candidate_id,
        source,
        component_index,
        selection.copy(),
        int(selection.sum()),
        (float(xs.mean()), float(ys.mean())),
        (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
        descriptor.copy(),
        cosine_similarity(target_descriptor, descriptor),
        predicted_iou,
        object_score_logit,
        metadata,
    )


def clone_candidate_with_similarity(
    candidate: IdentityMaskCandidate, target_descriptor: np.ndarray
) -> IdentityMaskCandidate:
    """Test/support helper for recomputing target affinity without touching masks."""
    return replace(
        candidate,
        target_similarity=cosine_similarity(target_descriptor, candidate.descriptor),
    )


def format_identity_global_diagnostics(result: IdentityGlobalResult) -> str:
    cache = result.feature_cache
    width, height = cache.spatial_size
    lines = [
        "Identity Global (Experimental) diagnostics",
        "SAM appearance feature:",
        f"  selected level: {cache.level_index}",
        f"  inspected level shapes (C,H,W): {cache.level_shapes}",
        f"  shape: {cache.channels} x {height} x {width}",
        f"  dtype: {cache.cached_dtype}",
        f"  cache size: {cache.total_bytes / (1024 * 1024):.3f} MiB",
        "target descriptor:",
        f"  dimension: {result.target_descriptor.size}",
        f"  norm: {np.linalg.norm(result.target_descriptor):.6f}",
        f"Reverse candidates: {'unavailable - ' + result.reverse_warning if result.reverse_warning else 'available'}",
        f"INVALID_DESCRIPTOR candidates: {len(result.invalid_descriptors)}",
        *(f"  {item}" for item in result.invalid_descriptors),
        "",
    ]
    for frame in result.frames:
        lines.extend([f"frame_{frame.frame_index:03}", f"  candidate count: {len(frame.candidates)}"])
        for index, candidate in enumerate(frame.candidates):
            lines.extend(
                [
                    f"  candidate {index}:",
                    f"    id/source/component: {candidate.candidate_id} / {candidate.source} / {candidate.component_index}",
                    f"    pixels: {candidate.pixels}",
                    f"    centroid: ({candidate.centroid[0]:.3f}, {candidate.centroid[1]:.3f})",
                    f"    bbox: {candidate.bbox}",
                    f"    target appearance cosine: {candidate.target_similarity:.6f}",
                    "    pre-gate predicted IoU: "
                    + ("N/A" if candidate.predicted_iou is None else f"{candidate.predicted_iou:.6f}"),
                    "    object score logit: "
                    + ("N/A" if candidate.object_score_logit is None else f"{candidate.object_score_logit:.6f}"),
                    f"    source metadata: {dict(candidate.source_metadata)}",
                    "    temporal appearance cosine: "
                    + ("N/A" if candidate.temporal_similarity is None else f"{candidate.temporal_similarity:.6f}"),
                    f"    motion penalty: {candidate.motion_penalty:.6f}",
                    f"    area penalty: {candidate.area_penalty:.6f}",
                    f"    bbox penalty: {candidate.bbox_penalty:.6f}",
                    f"    node score: {candidate.node_score:.6f}",
                    f"    best DP score: {candidate.best_dp_score:.6f}",
                    f"    selected: {'YES' if candidate.selected else 'NO'}",
                ]
            )
        lines.extend(
            [
                "  OCCLUDED:",
                f"    score: {frame.occluded_score:.6f}",
                f"    selected: {'YES' if frame.occluded_selected else 'NO'}",
                f"  final selected: {frame.selected_label}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
