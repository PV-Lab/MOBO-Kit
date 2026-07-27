"""Deterministic coordinate refinement on an exact finite design grid."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable, Sequence

import numpy as np

from .batch_selection import soft_local_penalty
from .candidate_pool import CandidatePool
from .design import Design


GridScoreFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RefinementConfig:
    anchors_per_selection_step: int = 64
    max_sweeps: int = 10
    improvement_tolerance: float = 1.0e-10
    radius: float | None = 0.25
    min_batch_distance: float = 0.15
    min_observed_distance: float = 0.0
    dimension_weights: np.ndarray | None = None
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        for name in ("anchors_per_selection_step", "max_sweeps"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
            ):
                raise ValueError(f"{name} must be a positive integer.")
            object.__setattr__(self, name, int(value))
        for name in (
            "improvement_tolerance",
            "min_batch_distance",
            "min_observed_distance",
            "epsilon",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a real non-boolean number.")
            number = float(value)
            if not np.isfinite(number) or number < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, number)
        if self.epsilon <= 0 or self.epsilon >= 1:
            raise ValueError("epsilon must be strictly between zero and one.")
        if self.radius is not None:
            if isinstance(self.radius, (bool, np.bool_)) or not isinstance(
                self.radius, Real
            ):
                raise ValueError("radius must be null or a real number.")
            radius = float(self.radius)
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError("radius must be null or finite and positive.")
            object.__setattr__(self, "radius", radius)
        if self.dimension_weights is not None:
            weights = np.asarray(self.dimension_weights, dtype=float).copy()
            if weights.ndim != 1 or weights.size == 0:
                raise ValueError("dimension_weights must be a non-empty vector.")
            if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("dimension_weights must be finite and positive.")
            weights.setflags(write=False)
            object.__setattr__(self, "dimension_weights", weights)


@dataclass(frozen=True)
class RefinementTraceRow:
    selection_step: int
    anchor_rank: int
    anchor_pool_index: int | None
    sweep: int
    coordinate: int | None
    coordinate_name: str | None
    start_grid_index: tuple[int, ...]
    chosen_grid_index: tuple[int, ...]
    score_before: float
    score_after: float
    base_score_before: float
    base_score_after: float
    penalized_log_score_before: float
    penalized_log_score_after: float
    accepted_move: bool
    termination_reason: str | None


@dataclass(frozen=True)
class RefinedAnchor:
    selection_step: int
    anchor_rank: int
    anchor_pool_index: int | None
    anchor_grid_index: tuple[int, ...]
    refined_grid_index: tuple[int, ...]
    start_base_score: float
    start_penalized_score: float
    start_penalized_log_score: float
    base_score: float
    penalized_score: float
    end_penalized_log_score: float
    sweeps: int
    accepted_move_count: int
    changed_dimensions: tuple[str, ...]
    termination_reason: str


@dataclass(frozen=True)
class RefinedBatchResult:
    grid_indices: np.ndarray
    X_phys: np.ndarray
    X_norm: np.ndarray
    base_scores: np.ndarray
    penalized_scores_at_selection: np.ndarray
    anchors: tuple[RefinedAnchor, ...]
    trace: tuple[RefinementTraceRow, ...]
    distinct_converged_optima: int


class CachedGridScorer:
    """Memoize a vectorized exact-grid score function by integer tuple."""

    def __init__(self, score_function: GridScoreFunction, dimension: int) -> None:
        if not callable(score_function):
            raise TypeError("score_function must be callable.")
        if isinstance(dimension, bool) or int(dimension) <= 0:
            raise ValueError("dimension must be a positive integer.")
        self._score_function = score_function
        self._dimension = int(dimension)
        self._cache: dict[tuple[int, ...], float] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def seed(self, grid_indices: np.ndarray, scores: np.ndarray) -> None:
        rows = _grid_matrix(grid_indices, self._dimension, name="grid_indices")
        values = _score_vector(scores, rows.shape[0])
        for row, value in zip(rows, values):
            key = tuple(int(item) for item in row)
            previous = self._cache.get(key)
            if previous is not None and previous != float(value):
                raise ValueError("Conflicting score supplied for a cached grid tuple.")
            self._cache[key] = float(value)

    def __call__(self, grid_indices: np.ndarray) -> np.ndarray:
        rows = _grid_matrix(grid_indices, self._dimension, name="grid_indices")
        keys = [tuple(int(item) for item in row) for row in rows]
        missing_keys = list(
            dict.fromkeys(key for key in keys if key not in self._cache)
        )
        if missing_keys:
            missing = np.asarray(missing_keys, dtype=np.int64)
            values = _score_vector(self._score_function(missing), missing.shape[0])
            for key, value in zip(missing_keys, values):
                self._cache[key] = float(value)
        return np.asarray([self._cache[key] for key in keys], dtype=float)


def _grid_matrix(value: np.ndarray, dimension: int, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != dimension:
        raise ValueError(f"{name} must have shape (N, {dimension}); got {raw.shape}.")
    if not np.issubdtype(raw.dtype, np.integer):
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            raise ValueError(f"{name} must contain integer grid indices.")
    return raw.astype(np.int64, copy=False)


def _score_vector(value: np.ndarray, size: int) -> np.ndarray:
    scores = np.asarray(value, dtype=float)
    if scores.shape != (size,) or not np.all(np.isfinite(scores)):
        raise ValueError(f"score_function must return {size} finite scores.")
    if np.any(scores < 0):
        raise ValueError("Acquisition scores must be non-negative.")
    return scores


def _validate_design(design: Design) -> tuple[np.ndarray, ...]:
    if not isinstance(design, Design):
        raise TypeError("design must be a Design.")
    grids = tuple(np.asarray(grid, dtype=float) for grid in design.var_array)
    if len(grids) != len(design.names) or not grids:
        raise ValueError("design must contain one grid per named dimension.")
    if any(grid.ndim != 1 or grid.size == 0 for grid in grids):
        raise ValueError("Every design grid must be a non-empty vector.")
    return grids


def grid_indices_to_physical_and_normalized(
    grid_indices: np.ndarray, design: Design
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve exact integer tuples without allocating the Cartesian product."""
    grids = _validate_design(design)
    rows = _grid_matrix(grid_indices, len(grids), name="grid_indices")
    physical = np.empty(rows.shape, dtype=float)
    for column, grid in enumerate(grids):
        if np.any(rows[:, column] < 0) or np.any(rows[:, column] >= grid.size):
            raise ValueError(
                f"grid_indices contains an out-of-range value for {design.names[column]!r}."
            )
        physical[:, column] = grid[rows[:, column]]

    # Canonicalize exactly as the Sobol candidate-pool path.  Index fractions are
    # mathematically equivalent on an evenly spaced grid, but decimal-valued axes
    # can differ by one floating-point bit.  Using physical bounds here keeps pool
    # anchors, refined candidates, distance checks, and exact-overlap diagnostics
    # on one representation.
    lower = np.asarray(design.lowers, dtype=float)
    upper = np.asarray(design.uppers, dtype=float)
    spans = upper - lower
    normalized = np.zeros_like(physical, dtype=float)
    changing = spans > 0.0
    normalized[:, changing] = (physical[:, changing] - lower[changing]) / spans[
        changing
    ]
    return physical, normalized


def _weights(config: RefinementConfig, dimension: int) -> np.ndarray:
    if config.dimension_weights is None:
        return np.ones(dimension, dtype=float)
    if config.dimension_weights.shape != (dimension,):
        raise ValueError(
            f"dimension_weights must have shape ({dimension},); got "
            f"{config.dimension_weights.shape}."
        )
    return config.dimension_weights


def _distance_to_references(
    X_norm: np.ndarray, references: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    if references.shape[0] == 0:
        return np.full(X_norm.shape[0], np.inf, dtype=float)
    difference = X_norm[:, None, :] - references[None, :, :]
    return np.sqrt(np.sum(weights * difference**2, axis=-1)).min(axis=1)


def _penalized_scores(
    base_scores: np.ndarray,
    X_norm: np.ndarray,
    *,
    selected_norm: np.ndarray,
    observed_norm: np.ndarray,
    grid_indices: np.ndarray,
    forbidden_grid_keys: set[tuple[int, ...]],
    config: RefinementConfig,
    positive_score_threshold: float,
) -> np.ndarray:
    weights = _weights(config, X_norm.shape[1])
    penalized = np.asarray(base_scores, dtype=float).copy()
    valid = penalized > positive_score_threshold
    if forbidden_grid_keys:
        valid &= np.asarray(
            [
                tuple(int(value) for value in row) not in forbidden_grid_keys
                for row in grid_indices
            ],
            dtype=bool,
        )
    nearest_selected = _distance_to_references(X_norm, selected_norm, weights)
    nearest_observed = _distance_to_references(X_norm, observed_norm, weights)
    if selected_norm.shape[0]:
        valid &= nearest_selected >= config.min_batch_distance
    if observed_norm.shape[0] and config.min_observed_distance > 0:
        valid &= nearest_observed >= config.min_observed_distance
    if selected_norm.shape[0] and config.radius is not None:
        difference = X_norm[:, None, :] - selected_norm[None, :, :]
        distances = np.sqrt(np.sum(weights * difference**2, axis=-1))
        factors, _ = soft_local_penalty(
            distances, radius=config.radius, epsilon=config.epsilon
        )
        penalized *= np.prod(factors, axis=1)
    penalized[~valid] = -np.inf
    return penalized


def _lexicographic_best(
    grid_indices: np.ndarray, scores: np.ndarray, *, tie_tolerance: float = 1.0e-15
) -> int:
    valid = np.flatnonzero(np.isfinite(scores))
    if valid.size == 0:
        raise RuntimeError("No eligible finite acquisition score remains.")
    maximum = float(np.max(scores[valid]))
    tied = valid[np.abs(scores[valid] - maximum) <= tie_tolerance]
    if tied.size == 1:
        return int(tied[0])
    keys = tuple(
        grid_indices[tied, column] for column in reversed(range(grid_indices.shape[1]))
    )
    return int(tied[np.lexsort(keys)[0]])


def refine_discrete_acquisition_anchors(
    design: Design,
    anchor_grid_indices: np.ndarray,
    score_function: GridScoreFunction,
    *,
    config: RefinementConfig,
    selection_step: int,
    anchor_pool_indices: Sequence[int | None] | None = None,
    selected_grid_indices: np.ndarray | None = None,
    selected_norm: np.ndarray | None = None,
    observed_grid_indices: np.ndarray | None = None,
    observed_norm: np.ndarray | None = None,
    avoid_grid_indices: np.ndarray | None = None,
    positive_score_threshold: float = 0.0,
) -> tuple[tuple[RefinedAnchor, ...], tuple[RefinementTraceRow, ...]]:
    """Coordinate-ascent every anchor using every allowed value per dimension."""
    grids = _validate_design(design)
    dimension = len(grids)
    anchors = _grid_matrix(anchor_grid_indices, dimension, name="anchor_grid_indices")
    if anchors.shape[0] == 0:
        raise ValueError("At least one refinement anchor is required.")
    if isinstance(selection_step, bool) or int(selection_step) <= 0:
        raise ValueError("selection_step must be a positive integer.")
    if not isinstance(config, RefinementConfig):
        raise TypeError("config must be a RefinementConfig.")
    if anchor_pool_indices is None:
        pool_indices: tuple[int | None, ...] = (None,) * anchors.shape[0]
    else:
        if len(anchor_pool_indices) != anchors.shape[0]:
            raise ValueError("anchor_pool_indices must align with anchor rows.")
        parsed_indices: list[int | None] = []
        for value in anchor_pool_indices:
            if value is None:
                parsed_indices.append(None)
            elif isinstance(value, (bool, np.bool_)) or int(value) < 0:
                raise ValueError(
                    "anchor_pool_indices values must be non-negative integers or None."
                )
            else:
                parsed_indices.append(int(value))
        pool_indices = tuple(parsed_indices)
    threshold = float(positive_score_threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("positive_score_threshold must be finite and non-negative.")
    scorer = (
        score_function
        if isinstance(score_function, CachedGridScorer)
        else CachedGridScorer(score_function, dimension)
    )

    empty_grid = np.empty((0, dimension), dtype=np.int64)
    selected_grid = _grid_matrix(
        empty_grid if selected_grid_indices is None else selected_grid_indices,
        dimension,
        name="selected_grid_indices",
    )
    observed_grid = _grid_matrix(
        empty_grid if observed_grid_indices is None else observed_grid_indices,
        dimension,
        name="observed_grid_indices",
    )
    avoid_grid = _grid_matrix(
        empty_grid if avoid_grid_indices is None else avoid_grid_indices,
        dimension,
        name="avoid_grid_indices",
    )
    empty_norm = np.empty((0, dimension), dtype=float)
    selected_X = (
        empty_norm if selected_norm is None else np.asarray(selected_norm, dtype=float)
    )
    observed_X = (
        empty_norm if observed_norm is None else np.asarray(observed_norm, dtype=float)
    )
    for name, value in (("selected_norm", selected_X), ("observed_norm", observed_X)):
        if (
            value.ndim != 2
            or value.shape[1] != dimension
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(f"{name} must be a finite (N, {dimension}) matrix.")
    forbidden = {
        tuple(int(value) for value in row)
        for row in np.vstack([selected_grid, observed_grid, avoid_grid])
    }

    refined: list[RefinedAnchor] = []
    trace: list[RefinementTraceRow] = []
    for anchor_rank, (anchor, anchor_pool_index) in enumerate(
        zip(anchors, pool_indices), start=1
    ):
        current = anchor.copy()
        current_phys, current_norm = grid_indices_to_physical_and_normalized(
            current[None, :], design
        )
        del current_phys
        current_base = scorer(current[None, :])[0]
        current_penalized = _penalized_scores(
            np.asarray([current_base]),
            current_norm,
            selected_norm=selected_X,
            observed_norm=observed_X,
            grid_indices=current[None, :],
            forbidden_grid_keys=forbidden,
            config=config,
            positive_score_threshold=threshold,
        )[0]
        if not np.isfinite(current_penalized):
            raise RuntimeError("A refinement anchor violates an eligibility rule.")
        start_base = float(current_base)
        start_penalized = float(current_penalized)
        accepted_move_count = 0
        changed_dimensions: set[str] = set()
        termination = "max_sweeps"
        sweeps_completed = 0
        for sweep in range(1, config.max_sweeps + 1):
            sweep_improved = False
            sweeps_completed = sweep
            for coordinate, (name, grid) in enumerate(zip(design.names, grids)):
                candidates = np.repeat(current[None, :], grid.size, axis=0)
                candidates[:, coordinate] = np.arange(grid.size, dtype=np.int64)
                _, candidates_norm = grid_indices_to_physical_and_normalized(
                    candidates, design
                )
                base = scorer(candidates)
                penalized = _penalized_scores(
                    base,
                    candidates_norm,
                    selected_norm=selected_X,
                    observed_norm=observed_X,
                    grid_indices=candidates,
                    forbidden_grid_keys=forbidden,
                    config=config,
                    positive_score_threshold=threshold,
                )
                chosen = _lexicographic_best(candidates, penalized)
                proposed = candidates[chosen]
                proposed_score = float(penalized[chosen])
                accepted = proposed_score > (
                    current_penalized + config.improvement_tolerance
                )
                start_key = tuple(int(value) for value in current)
                chosen_key = tuple(int(value) for value in proposed)
                before = float(current_penalized)
                base_before = float(current_base)
                if accepted:
                    current = proposed.copy()
                    current_base = float(base[chosen])
                    current_penalized = proposed_score
                    sweep_improved = True
                    accepted_move_count += 1
                    changed_dimensions.add(name)
                trace.append(
                    RefinementTraceRow(
                        selection_step=int(selection_step),
                        anchor_rank=anchor_rank,
                        anchor_pool_index=anchor_pool_index,
                        sweep=sweep,
                        coordinate=coordinate,
                        coordinate_name=name,
                        start_grid_index=start_key,
                        chosen_grid_index=chosen_key,
                        score_before=before,
                        score_after=float(current_penalized),
                        base_score_before=base_before,
                        base_score_after=float(current_base),
                        penalized_log_score_before=float(np.log(before)),
                        penalized_log_score_after=float(np.log(current_penalized)),
                        accepted_move=accepted,
                        termination_reason=None,
                    )
                )
            if not sweep_improved:
                termination = "no_improvement"
                break
        trace.append(
            RefinementTraceRow(
                selection_step=int(selection_step),
                anchor_rank=anchor_rank,
                anchor_pool_index=anchor_pool_index,
                sweep=sweeps_completed,
                coordinate=None,
                coordinate_name=None,
                start_grid_index=tuple(int(value) for value in current),
                chosen_grid_index=tuple(int(value) for value in current),
                score_before=float(current_penalized),
                score_after=float(current_penalized),
                base_score_before=float(current_base),
                base_score_after=float(current_base),
                penalized_log_score_before=float(np.log(current_penalized)),
                penalized_log_score_after=float(np.log(current_penalized)),
                accepted_move=False,
                termination_reason=termination,
            )
        )
        refined.append(
            RefinedAnchor(
                selection_step=int(selection_step),
                anchor_rank=anchor_rank,
                anchor_pool_index=anchor_pool_index,
                anchor_grid_index=tuple(int(value) for value in anchor),
                refined_grid_index=tuple(int(value) for value in current),
                start_base_score=start_base,
                start_penalized_score=start_penalized,
                start_penalized_log_score=float(np.log(start_penalized)),
                base_score=float(current_base),
                penalized_score=float(current_penalized),
                end_penalized_log_score=float(np.log(current_penalized)),
                sweeps=sweeps_completed,
                accepted_move_count=accepted_move_count,
                changed_dimensions=tuple(sorted(changed_dimensions)),
                termination_reason=termination,
            )
        )
    return tuple(refined), tuple(trace)


def propose_refined_discrete_batch(
    master_pool: CandidatePool,
    design: Design,
    score_function: GridScoreFunction,
    *,
    q: int,
    config: RefinementConfig,
    master_base_scores: np.ndarray | None = None,
    observed_grid_indices: np.ndarray | None = None,
    observed_norm: np.ndarray | None = None,
    avoid_grid_indices: np.ndarray | None = None,
    positive_score_threshold: float = 0.0,
) -> RefinedBatchResult:
    """Sequentially refine top pool anchors and select an exact-grid batch."""
    if not isinstance(master_pool, CandidatePool):
        raise TypeError("master_pool must be a CandidatePool.")
    grids = _validate_design(design)
    dimension = len(grids)
    pool_grid = _grid_matrix(
        master_pool.grid_indices, dimension, name="pool.grid_indices"
    )
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or int(q) <= 0:
        raise ValueError("q must be a positive integer.")
    requested = int(q)
    scorer = CachedGridScorer(score_function, dimension)
    if master_base_scores is None:
        pool_base = scorer(pool_grid)
    else:
        pool_base = _score_vector(master_base_scores, pool_grid.shape[0])
        scorer.seed(pool_grid, pool_base)
    empty_grid = np.empty((0, dimension), dtype=np.int64)
    observed_grid = _grid_matrix(
        empty_grid if observed_grid_indices is None else observed_grid_indices,
        dimension,
        name="observed_grid_indices",
    )
    avoid_grid = _grid_matrix(
        empty_grid if avoid_grid_indices is None else avoid_grid_indices,
        dimension,
        name="avoid_grid_indices",
    )
    observed_X = (
        np.empty((0, dimension), dtype=float)
        if observed_norm is None
        else np.asarray(observed_norm, dtype=float)
    )
    if (
        observed_X.ndim != 2
        or observed_X.shape[1] != dimension
        or not np.all(np.isfinite(observed_X))
    ):
        raise ValueError(f"observed_norm must be a finite (N, {dimension}) matrix.")

    selected_grid: list[np.ndarray] = []
    selected_norm: list[np.ndarray] = []
    selected_base: list[float] = []
    selected_penalized: list[float] = []
    all_anchors: list[RefinedAnchor] = []
    all_trace: list[RefinementTraceRow] = []
    discovered_optima: set[tuple[int, ...]] = set()
    pool_position_by_key = {
        tuple(int(value) for value in row): index for index, row in enumerate(pool_grid)
    }
    for selection_step in range(1, requested + 1):
        selected_grid_array = (
            np.asarray(selected_grid, dtype=np.int64)
            if selected_grid
            else empty_grid.copy()
        )
        selected_norm_array = (
            np.asarray(selected_norm, dtype=float)
            if selected_norm
            else np.empty((0, dimension), dtype=float)
        )
        forbidden = {
            tuple(int(value) for value in row)
            for row in np.vstack([observed_grid, avoid_grid, selected_grid_array])
        }
        pool_penalized = _penalized_scores(
            pool_base,
            np.asarray(master_pool.X_norm, dtype=float),
            selected_norm=selected_norm_array,
            observed_norm=observed_X,
            grid_indices=pool_grid,
            forbidden_grid_keys=forbidden,
            config=config,
            positive_score_threshold=positive_score_threshold,
        )
        valid = np.flatnonzero(np.isfinite(pool_penalized))
        if valid.size == 0:
            raise RuntimeError(
                "No master-pool anchors remain without relaxing an eligibility rule."
            )
        lex_order = np.lexsort(
            tuple(pool_grid[valid, column] for column in reversed(range(dimension)))
        )
        lex_valid = valid[lex_order]
        score_order = np.argsort(-pool_penalized[lex_valid], kind="stable")
        anchor_positions = lex_valid[score_order][
            : min(config.anchors_per_selection_step, valid.size)
        ]
        anchor_rows = [pool_grid[position].copy() for position in anchor_positions]
        anchor_pool_indices: list[int | None] = [
            int(position) for position in anchor_positions
        ]
        seen_anchor_keys = {tuple(int(value) for value in row) for row in anchor_rows}
        for previous_key in sorted(discovered_optima):
            if previous_key in seen_anchor_keys or previous_key in forbidden:
                continue
            anchor_rows.append(np.asarray(previous_key, dtype=np.int64))
            anchor_pool_indices.append(pool_position_by_key.get(previous_key))
            seen_anchor_keys.add(previous_key)
        combined_anchors = np.asarray(anchor_rows, dtype=np.int64)
        _, combined_norm = grid_indices_to_physical_and_normalized(
            combined_anchors, design
        )
        combined_penalized = _penalized_scores(
            scorer(combined_anchors),
            combined_norm,
            selected_norm=selected_norm_array,
            observed_norm=observed_X,
            grid_indices=combined_anchors,
            forbidden_grid_keys=forbidden,
            config=config,
            positive_score_threshold=positive_score_threshold,
        )
        eligible_anchor_mask = np.isfinite(combined_penalized)
        combined_anchors = combined_anchors[eligible_anchor_mask]
        eligible_pool_indices = tuple(
            value
            for value, eligible in zip(anchor_pool_indices, eligible_anchor_mask)
            if eligible
        )
        if combined_anchors.shape[0] == 0:
            raise RuntimeError("No eligible local-refinement anchor remains.")
        refined, trace = refine_discrete_acquisition_anchors(
            design,
            combined_anchors,
            scorer,
            config=config,
            selection_step=selection_step,
            anchor_pool_indices=eligible_pool_indices,
            selected_grid_indices=selected_grid_array,
            selected_norm=selected_norm_array,
            observed_grid_indices=observed_grid,
            observed_norm=observed_X,
            avoid_grid_indices=avoid_grid,
            positive_score_threshold=positive_score_threshold,
        )
        all_anchors.extend(refined)
        all_trace.extend(trace)
        discovered_optima.update(item.refined_grid_index for item in refined)
        unique_optima = np.asarray(
            sorted({item.refined_grid_index for item in refined}), dtype=np.int64
        )
        _, optima_norm = grid_indices_to_physical_and_normalized(unique_optima, design)
        optima_base = scorer(unique_optima)
        optima_penalized = _penalized_scores(
            optima_base,
            optima_norm,
            selected_norm=selected_norm_array,
            observed_norm=observed_X,
            grid_indices=unique_optima,
            forbidden_grid_keys=forbidden,
            config=config,
            positive_score_threshold=positive_score_threshold,
        )
        chosen = _lexicographic_best(unique_optima, optima_penalized)
        selected_grid.append(unique_optima[chosen].copy())
        selected_norm.append(optima_norm[chosen].copy())
        selected_base.append(float(optima_base[chosen]))
        selected_penalized.append(float(optima_penalized[chosen]))

    selected_grid_array = np.asarray(selected_grid, dtype=np.int64)
    physical, normalized = grid_indices_to_physical_and_normalized(
        selected_grid_array, design
    )
    if np.unique(selected_grid_array, axis=0).shape[0] != requested:
        raise RuntimeError("Refinement produced duplicate selected grid tuples.")
    return RefinedBatchResult(
        grid_indices=selected_grid_array,
        X_phys=physical,
        X_norm=normalized,
        base_scores=np.asarray(selected_base, dtype=float),
        penalized_scores_at_selection=np.asarray(selected_penalized, dtype=float),
        anchors=tuple(all_anchors),
        trace=tuple(all_trace),
        distinct_converged_optima=len(
            {anchor.refined_grid_index for anchor in all_anchors}
        ),
    )


__all__ = [
    "CachedGridScorer",
    "RefinedAnchor",
    "RefinedBatchResult",
    "RefinementConfig",
    "RefinementTraceRow",
    "grid_indices_to_physical_and_normalized",
    "propose_refined_discrete_batch",
    "refine_discrete_acquisition_anchors",
]
