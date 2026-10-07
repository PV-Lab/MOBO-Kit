"""Reusable sequential local penalization for discrete MOBO candidate pools."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable

import numpy as np

from .candidate_pool import CandidatePool


@dataclass(frozen=True)
class LocalPenalizationConfig:
    radius: float | None
    min_batch_distance: float
    min_observed_distance: float = 0.0
    dimension_weights: np.ndarray | None = None
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        for field_name in ("min_batch_distance", "min_observed_distance", "epsilon"):
            field_value = getattr(self, field_name)
            if isinstance(field_value, (bool, np.bool_)) or not isinstance(
                field_value, Real
            ):
                raise ValueError(f"{field_name} must be a real non-boolean number.")
        if self.radius is None:
            radius = None
        else:
            if isinstance(self.radius, (bool, np.bool_)) or not isinstance(
                self.radius, Real
            ):
                raise ValueError("radius must be None or a real non-boolean number.")
            radius = float(self.radius)
        minimum_batch = float(self.min_batch_distance)
        minimum_observed = float(self.min_observed_distance)
        epsilon = float(self.epsilon)
        if radius is not None and (not np.isfinite(radius) or radius <= 0):
            raise ValueError("radius must be None or finite and strictly positive.")
        if not np.isfinite(minimum_batch) or minimum_batch < 0:
            raise ValueError("min_batch_distance must be finite and non-negative.")
        if not np.isfinite(minimum_observed) or minimum_observed < 0:
            raise ValueError("min_observed_distance must be finite and non-negative.")
        if not np.isfinite(epsilon) or epsilon <= 0 or epsilon >= 1:
            raise ValueError("epsilon must be finite and between zero and one.")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "min_batch_distance", minimum_batch)
        object.__setattr__(self, "min_observed_distance", minimum_observed)
        object.__setattr__(self, "epsilon", epsilon)
        if self.dimension_weights is not None:
            raw_weights = np.asarray(self.dimension_weights)
            if np.issubdtype(raw_weights.dtype, np.bool_) or any(
                not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
                for value in raw_weights.ravel()
            ):
                raise ValueError(
                    "dimension_weights must contain real non-boolean numbers."
                )
            weights = np.asarray(self.dimension_weights, dtype=float).copy()
            if weights.ndim != 1 or weights.size == 0:
                raise ValueError("dimension_weights must be a non-empty vector.")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError(
                    "dimension_weights must be finite and strictly positive."
                )
            weights.setflags(write=False)
            object.__setattr__(self, "dimension_weights", weights)


@dataclass(frozen=True)
class BaseScoreResult:
    base_log_score: np.ndarray
    base_score: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionStep:
    order: int
    pool_index: int
    base_score: float | None
    base_log_score: float
    penalty_factor: float
    log_penalty: float
    penalized_log_score: float
    nearest_selected_distance_before: float | None
    nearest_observed_distance: float | None


@dataclass(frozen=True)
class BatchSelectionResult:
    X_norm: np.ndarray
    X_phys: np.ndarray
    selected_pool_indices: np.ndarray
    steps: tuple[SelectionStep, ...]
    method_diagnostics: dict[str, Any]
    distance_diagnostics: dict[str, Any]


class UndersizedBatchError(RuntimeError):
    """Raised when an exact batch would require relaxing a hard rule."""

    def __init__(
        self,
        *,
        requested_size: int,
        selected_size: int,
        pool_size: int,
        remaining_candidate_count: int,
        min_batch_distance: float,
        min_observed_distance: float,
        selected_pool_indices: np.ndarray,
        hard_valid_candidate_count: int | None = None,
    ) -> None:
        self.requested_size = requested_size
        self.selected_size = selected_size
        self.pool_size = pool_size
        self.remaining_candidate_count = remaining_candidate_count
        self.min_batch_distance = min_batch_distance
        self.min_observed_distance = min_observed_distance
        self.selected_pool_indices = selected_pool_indices.copy()
        self.hard_valid_candidate_count = hard_valid_candidate_count
        super().__init__(
            "Unable to select the exact requested batch without violating an "
            "eligibility or hard-distance rule: "
            f"requested={requested_size}, selected={selected_size}, "
            f"pool_size={pool_size}, remaining={remaining_candidate_count}, "
            f"min_batch_distance={min_batch_distance}, "
            f"min_observed_distance={min_observed_distance}."
        )


def _weights(config: LocalPenalizationConfig, dimension: int) -> np.ndarray:
    if config.dimension_weights is None:
        return np.ones(dimension, dtype=float)
    if config.dimension_weights.shape != (dimension,):
        raise ValueError(
            "dimension_weights must have shape "
            f"({dimension},); got {config.dimension_weights.shape}."
        )
    return config.dimension_weights


def _distances(
    X: np.ndarray, references: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    if references.shape[0] == 0:
        return np.empty((X.shape[0], 0), dtype=float)
    differences = X[:, None, :] - references[None, :, :]
    return np.sqrt(np.sum(weights * differences**2, axis=-1))


def soft_local_penalty(
    distances: np.ndarray, *, radius: float, epsilon: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Return stabilized exclusion factors and their natural logarithms."""
    distance = np.asarray(distances, dtype=float)
    if not np.all(np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError("distances must be finite and non-negative.")
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, Real):
        raise ValueError("radius must be a real non-boolean number.")
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise ValueError("epsilon must be a real non-boolean number.")
    radius_value = float(radius)
    epsilon_value = float(epsilon)
    if not np.isfinite(radius_value) or radius_value <= 0:
        raise ValueError("radius must be finite and strictly positive.")
    if not np.isfinite(epsilon_value) or epsilon_value <= 0 or epsilon_value >= 1:
        raise ValueError("epsilon must be finite and between zero and one.")
    factor = 1.0 - np.exp(-0.5 * (distance / radius_value) ** 2)
    return factor, np.log(np.maximum(factor, epsilon_value))


def _validate_pool(candidate_pool: CandidatePool) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(candidate_pool, CandidatePool):
        raise TypeError("candidate_pool must be a CandidatePool.")
    X_norm = np.asarray(candidate_pool.X_norm, dtype=float)
    X_phys = np.asarray(candidate_pool.X_phys, dtype=float)
    if X_norm.ndim != 2 or X_phys.shape != X_norm.shape:
        raise ValueError(
            "Candidate pool physical and normalized arrays must be (N, D)."
        )
    if X_norm.shape[0] == 0 or not np.all(np.isfinite(X_norm)):
        raise ValueError("Candidate pool must contain finite rows.")
    if not np.all(np.isfinite(X_phys)):
        raise ValueError(
            "Candidate pool physical rows must contain only finite values."
        )
    if np.any(X_norm < -1e-12) or np.any(X_norm > 1 + 1e-12):
        raise ValueError("Candidate pool normalized rows must lie in [0, 1].")
    grid_indices = np.asarray(candidate_pool.grid_indices)
    if grid_indices.shape != X_norm.shape or not np.issubdtype(
        grid_indices.dtype, np.integer
    ):
        raise ValueError(
            "Candidate pool grid_indices must be an integer array aligned with "
            "physical and normalized rows."
        )
    if np.unique(grid_indices, axis=0).shape[0] != X_norm.shape[0]:
        raise ValueError("Candidate pool contains duplicate grid-index tuples.")
    if np.unique(X_norm, axis=0).shape[0] != X_norm.shape[0]:
        raise ValueError("Candidate pool contains duplicate normalized rows.")
    if np.unique(X_phys, axis=0).shape[0] != X_phys.shape[0]:
        raise ValueError("Candidate pool contains duplicate physical rows.")
    return X_norm, X_phys


def select_local_penalized_batch(
    candidate_pool: CandidatePool,
    q: int,
    score_remaining: Callable[[np.ndarray, np.ndarray], BaseScoreResult],
    config: LocalPenalizationConfig,
    *,
    observed_pending_norm: np.ndarray | None = None,
) -> BatchSelectionResult:
    """Sequentially select exactly ``q`` candidates or fail without relaxation."""
    X_norm, X_phys = _validate_pool(candidate_pool)
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or int(q) <= 0:
        raise ValueError("q must be a positive integer.")
    requested = int(q)
    if requested > X_norm.shape[0]:
        raise UndersizedBatchError(
            requested_size=requested,
            selected_size=0,
            pool_size=X_norm.shape[0],
            remaining_candidate_count=X_norm.shape[0],
            min_batch_distance=config.min_batch_distance,
            min_observed_distance=config.min_observed_distance,
            selected_pool_indices=np.empty(0, dtype=int),
            hard_valid_candidate_count=X_norm.shape[0],
        )
    if not callable(score_remaining):
        raise TypeError("score_remaining must be callable.")
    weights = _weights(config, X_norm.shape[1])
    if observed_pending_norm is None:
        observed = np.empty((0, X_norm.shape[1]), dtype=float)
    else:
        observed = np.asarray(observed_pending_norm, dtype=float)
        if observed.ndim != 2 or observed.shape[1] != X_norm.shape[1]:
            raise ValueError(
                "observed_pending_norm must have shape " f"(N, {X_norm.shape[1]})."
            )
        if not np.all(np.isfinite(observed)):
            raise ValueError("observed_pending_norm must contain only finite values.")
        if np.any(observed < 0.0) or np.any(observed > 1.0):
            raise ValueError("observed_pending_norm must lie within [0, 1].")

    if observed.shape[0]:
        observed_distances = _distances(X_norm, observed, weights)
        nearest_observed_all = observed_distances.min(axis=1)
    else:
        nearest_observed_all = np.full(X_norm.shape[0], np.inf)
    nearest_selected_all = np.full(X_norm.shape[0], np.inf)
    cumulative_log_penalty = np.zeros(X_norm.shape[0], dtype=float)

    remaining = np.arange(X_norm.shape[0], dtype=int)
    selected: list[int] = []
    steps: list[SelectionStep] = []
    score_diagnostics: list[dict[str, Any]] = []
    for order in range(1, requested + 1):
        selected_array = np.asarray(selected, dtype=int)
        scored = score_remaining(remaining.copy(), selected_array.copy())
        if not isinstance(scored, BaseScoreResult):
            raise TypeError("score_remaining must return BaseScoreResult.")
        base_log = np.asarray(scored.base_log_score, dtype=float)
        if base_log.shape != (remaining.size,):
            raise ValueError(
                "base_log_score must align with remaining_indices; "
                f"expected {(remaining.size,)}, got {base_log.shape}."
            )
        if np.any(np.isnan(base_log)) or np.any(np.isposinf(base_log)):
            raise ValueError("base_log_score may be finite or -inf, not NaN/+inf.")
        base_score: np.ndarray | None = None
        if scored.base_score is not None:
            base_score = np.asarray(scored.base_score, dtype=float)
            if base_score.shape != (remaining.size,) or not np.all(
                np.isfinite(base_score)
            ):
                raise ValueError(
                    "base_score must be finite and align with remaining_indices."
                )
        score_diagnostics.append(dict(scored.diagnostics))

        if selected:
            nearest_selected = nearest_selected_all[remaining]
            log_penalty = cumulative_log_penalty[remaining]
        else:
            nearest_selected = np.full(remaining.size, np.inf)
            log_penalty = np.zeros(remaining.size, dtype=float)

        nearest_observed = nearest_observed_all[remaining]

        hard_valid = np.ones(remaining.size, dtype=bool)
        if selected:
            hard_valid &= nearest_selected >= config.min_batch_distance
        if observed.shape[0]:
            # Exact observed/pending recipes are always forbidden, even when the
            # configured distance threshold is explicitly zero.
            hard_valid &= nearest_observed > 0.0
            if config.min_observed_distance > 0:
                hard_valid &= nearest_observed >= config.min_observed_distance
        penalized = base_log + log_penalty
        penalized[~hard_valid] = -np.inf
        valid_positions = np.flatnonzero(np.isfinite(penalized))
        if valid_positions.size == 0:
            raise UndersizedBatchError(
                requested_size=requested,
                selected_size=len(selected),
                pool_size=X_norm.shape[0],
                remaining_candidate_count=int(valid_positions.size),
                min_batch_distance=config.min_batch_distance,
                min_observed_distance=config.min_observed_distance,
                selected_pool_indices=np.asarray(selected, dtype=int),
                hard_valid_candidate_count=int(np.count_nonzero(hard_valid)),
            )
        # np.argmax returns the first maximum. Remaining pool indices preserve
        # ascending/stable pool order, which is the documented tie-break.
        chosen_position = int(np.argmax(penalized))
        chosen_pool_index = int(remaining[chosen_position])
        chosen_log_penalty = float(log_penalty[chosen_position])
        steps.append(
            SelectionStep(
                order=order,
                pool_index=chosen_pool_index,
                base_score=(
                    None if base_score is None else float(base_score[chosen_position])
                ),
                base_log_score=float(base_log[chosen_position]),
                penalty_factor=float(np.exp(chosen_log_penalty)),
                log_penalty=chosen_log_penalty,
                penalized_log_score=float(penalized[chosen_position]),
                nearest_selected_distance_before=(
                    None if not selected else float(nearest_selected[chosen_position])
                ),
                nearest_observed_distance=(
                    None
                    if not observed.shape[0]
                    else float(nearest_observed[chosen_position])
                ),
            )
        )
        selected.append(chosen_pool_index)
        remaining = np.delete(remaining, chosen_position)
        new_distances = _distances(
            X_norm, X_norm[chosen_pool_index : chosen_pool_index + 1], weights
        )[:, 0]
        nearest_selected_all = np.minimum(nearest_selected_all, new_distances)
        if config.radius is not None:
            _, new_log_penalty = soft_local_penalty(
                new_distances, radius=config.radius, epsilon=config.epsilon
            )
            cumulative_log_penalty += new_log_penalty

    selected_array = np.asarray(selected, dtype=int)
    selected_distances = _distances(
        X_norm[selected_array], X_norm[selected_array], weights
    )
    if requested >= 2:
        triangle = selected_distances[np.triu_indices(requested, k=1)]
        within = {
            "minimum_within_batch_distance": float(triangle.min()),
            "mean_within_batch_distance": float(triangle.mean()),
            "maximum_within_batch_distance": float(triangle.max()),
        }
    else:
        within = {
            "minimum_within_batch_distance": None,
            "mean_within_batch_distance": None,
            "maximum_within_batch_distance": None,
        }
    return BatchSelectionResult(
        X_norm=X_norm[selected_array].copy(),
        X_phys=X_phys[selected_array].copy(),
        selected_pool_indices=selected_array,
        steps=tuple(steps),
        method_diagnostics={"score_steps": score_diagnostics},
        distance_diagnostics={
            "pairwise_distance_matrix": selected_distances,
            **within,
            "dimension_weights": weights.copy(),
            "radius": config.radius,
            "min_batch_distance": config.min_batch_distance,
            "min_observed_distance": config.min_observed_distance,
        },
    )


__all__ = [
    "BaseScoreResult",
    "BatchSelectionResult",
    "LocalPenalizationConfig",
    "SelectionStep",
    "UndersizedBatchError",
    "select_local_penalized_batch",
    "soft_local_penalty",
]
