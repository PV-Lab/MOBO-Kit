"""Deterministic regional comparisons for normalized candidate batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


DEFAULT_REGIONAL_THRESHOLDS = (0.10, 0.15, 0.20)


@dataclass(frozen=True)
class RegionalBatchComparison:
    """Exact and regional correspondence metrics for two candidate batches."""

    exact_overlap_count: int
    jaccard_overlap: float
    matched_reference_rows: np.ndarray
    matched_comparison_rows: np.ndarray
    matched_pair_distances: np.ndarray
    mean_matched_distance: float
    maximum_matched_distance: float
    regional_match_counts: dict[float, int]
    symmetric_chamfer_distance: float
    hausdorff_distance: float
    thresholds: tuple[float, ...]

    @property
    def exact_overlap(self) -> int:
        """Concise alias for report-building callers."""

        return self.exact_overlap_count

    @property
    def matched_distances(self) -> np.ndarray:
        """Concise alias for the optimal matched-pair distances."""

        return self.matched_pair_distances

    @property
    def max_matched_distance(self) -> float:
        """Concise alias for the maximum optimal matched-pair distance."""

        return self.maximum_matched_distance

    @property
    def chamfer_distance(self) -> float:
        """Concise alias for the symmetric Chamfer distance."""

        return self.symmetric_chamfer_distance

    def regional_match_count(self, threshold: float) -> int:
        """Return the Hungarian match count for an evaluated threshold."""

        key = float(threshold)
        if key not in self.regional_match_counts:
            raise KeyError(f"Threshold {key} was not evaluated.")
        return self.regional_match_counts[key]


def _normalized_batch(value: np.ndarray, *, name: str) -> np.ndarray:
    batch = np.asarray(value, dtype=float)
    if batch.ndim != 2 or batch.shape[0] == 0 or batch.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(batch)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(batch < 0.0) or np.any(batch > 1.0):
        raise ValueError(f"{name} must contain normalized coordinates in [0, 1].")
    return batch


def _thresholds(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("thresholds must be a non-empty sequence of finite numbers.")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "thresholds must be a non-empty sequence of finite numbers."
        ) from exc
    if not raw:
        raise ValueError("thresholds must not be empty.")
    result: list[float] = []
    for value in raw:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("thresholds must contain finite non-negative numbers.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "thresholds must contain finite non-negative numbers."
            ) from exc
        if not np.isfinite(number) or number < 0.0:
            raise ValueError("thresholds must contain finite non-negative numbers.")
        result.append(number)
    if len(set(result)) != len(result):
        raise ValueError("thresholds must not contain duplicates.")
    return tuple(sorted(result))


def _canonical_rows(batch: np.ndarray) -> np.ndarray:
    keys = tuple(batch[:, column] for column in reversed(range(batch.shape[1])))
    return batch[np.lexsort(keys)].copy()


def _exact_row_set(batch: np.ndarray) -> set[tuple[float, ...]]:
    return {tuple(float(value) for value in row) for row in batch}


def regional_match_batches(
    reference_norm: np.ndarray,
    comparison_norm: np.ndarray,
    *,
    thresholds: Sequence[float] = DEFAULT_REGIONAL_THRESHOLDS,
) -> RegionalBatchComparison:
    """Compare two normalized batches using canonical Hungarian assignment.

    Canonical lexicographic row order is applied before assignment. Therefore all
    returned coordinate pairs, distances, and aggregate metrics are invariant to
    row reordering in either input batch.
    """

    reference = _normalized_batch(reference_norm, name="reference_norm")
    comparison = _normalized_batch(comparison_norm, name="comparison_norm")
    if reference.shape[1] != comparison.shape[1]:
        raise ValueError(
            "reference_norm and comparison_norm must have the same input dimension."
        )
    evaluated_thresholds = _thresholds(thresholds)
    reference = _canonical_rows(reference)
    comparison = _canonical_rows(comparison)
    distances = cdist(reference, comparison, metric="euclidean")
    reference_indices, comparison_indices = linear_sum_assignment(distances)
    matched_distances = distances[reference_indices, comparison_indices]

    reference_set = _exact_row_set(reference)
    comparison_set = _exact_row_set(comparison)
    exact_overlap = len(reference_set & comparison_set)
    union = len(reference_set | comparison_set)
    jaccard = float(exact_overlap / union)

    directed_reference = distances.min(axis=1)
    directed_comparison = distances.min(axis=0)
    chamfer = 0.5 * (
        float(directed_reference.mean()) + float(directed_comparison.mean())
    )
    hausdorff = max(float(directed_reference.max()), float(directed_comparison.max()))
    regional_counts = {
        threshold: int(np.count_nonzero(matched_distances <= threshold))
        for threshold in evaluated_thresholds
    }

    return RegionalBatchComparison(
        exact_overlap_count=exact_overlap,
        jaccard_overlap=jaccard,
        matched_reference_rows=reference[reference_indices].copy(),
        matched_comparison_rows=comparison[comparison_indices].copy(),
        matched_pair_distances=matched_distances.copy(),
        mean_matched_distance=float(matched_distances.mean()),
        maximum_matched_distance=float(matched_distances.max()),
        regional_match_counts=regional_counts,
        symmetric_chamfer_distance=chamfer,
        hausdorff_distance=hausdorff,
        thresholds=evaluated_thresholds,
    )


compare_candidate_batches = regional_match_batches


__all__ = [
    "DEFAULT_REGIONAL_THRESHOLDS",
    "RegionalBatchComparison",
    "compare_candidate_batches",
    "regional_match_batches",
]
