"""Deterministic sampling of finite pools from very large discrete designs."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence

import numpy as np

from .constraints import RowConstraint, apply_row_constraints
from .design import Design


@dataclass(frozen=True)
class CandidatePool:
    """A discrete candidate pool in grid-index, physical, and normalized spaces."""

    grid_indices: np.ndarray
    X_phys: np.ndarray
    X_norm: np.ndarray
    seed: int
    draws: int
    rejected_duplicate: int
    rejected_avoid: int
    rejected_constraint: int

    @property
    def size(self) -> int:
        return int(self.grid_indices.shape[0])


class CandidatePoolSamplingError(RuntimeError):
    """Raised when an exact-size discrete pool cannot be produced safely."""

    def __init__(
        self,
        *,
        requested: int,
        accepted: int,
        draws: int,
        max_draws: int,
        rejected_duplicate: int,
        rejected_avoid: int,
        rejected_constraint: int,
        reason: str,
    ) -> None:
        self.requested = requested
        self.accepted = accepted
        self.draws = draws
        self.max_draws = max_draws
        self.rejected_duplicate = rejected_duplicate
        self.rejected_avoid = rejected_avoid
        self.rejected_constraint = rejected_constraint
        self.reason = reason
        super().__init__(
            "Could not sample the requested discrete candidate pool: "
            f"requested={requested}, accepted={accepted}, draws={draws}, "
            f"max_draws={max_draws}, duplicate_rejections={rejected_duplicate}, "
            f"avoid_rejections={rejected_avoid}, "
            f"constraint_rejections={rejected_constraint}. {reason}"
        )


def _design_grids(design: Design) -> tuple[np.ndarray, ...]:
    if not isinstance(design, Design):
        raise TypeError("design must be a Design.")
    grids = tuple(np.asarray(grid, dtype=float) for grid in design.var_array)
    if not grids or len(grids) != len(design.names):
        raise ValueError("design must contain one non-empty grid per input name.")
    for name, grid in zip(design.names, grids):
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError(f"Design grid {name!r} must be a non-empty vector.")
        if not np.all(np.isfinite(grid)) or np.unique(grid).size != grid.size:
            raise ValueError(f"Design grid {name!r} must be finite and unique.")
    return grids


def _physical_matrix(
    value: np.ndarray | None, *, name: str, dimension: int
) -> np.ndarray:
    if value is None:
        return np.empty((0, dimension), dtype=float)
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError(f"{name} must have shape (N, {dimension}); got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def physical_rows_to_grid_indices(
    X_phys: np.ndarray,
    design: Design,
    *,
    atol: float = 0.0,
) -> np.ndarray:
    """Convert physical rows to exact integer grid tuples or fail off-grid."""
    grids = _design_grids(design)
    X = _physical_matrix(X_phys, name="X_phys", dimension=len(grids))
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative.")
    indices = np.empty(X.shape, dtype=np.int64)
    for column, (name, grid) in enumerate(zip(design.names, grids)):
        differences = np.abs(X[:, column, None] - grid[None, :])
        closest = differences.argmin(axis=1)
        invalid = differences[np.arange(X.shape[0]), closest] > atol
        if np.any(invalid):
            rows = np.flatnonzero(invalid).tolist()
            raise ValueError(f"Physical rows {rows} are off-grid for input {name!r}.")
        indices[:, column] = closest
    return indices


def _indices_to_physical(
    grid_indices: np.ndarray, grids: tuple[np.ndarray, ...]
) -> np.ndarray:
    physical = np.empty(grid_indices.shape, dtype=float)
    for column, grid in enumerate(grids):
        physical[:, column] = grid[grid_indices[:, column]]
    return physical


def _normalize_physical(X_phys: np.ndarray, design: Design) -> np.ndarray:
    lower = np.asarray(design.lowers, dtype=float)
    upper = np.asarray(design.uppers, dtype=float)
    spans = upper - lower
    normalized = np.zeros_like(X_phys, dtype=float)
    changing = spans > 0
    normalized[:, changing] = (X_phys[:, changing] - lower[changing]) / spans[changing]
    return normalized


def sample_discrete_candidate_pool(
    design: Design,
    pool_size: int,
    *,
    seed: int,
    observed_phys: np.ndarray | None = None,
    pending_phys: np.ndarray | None = None,
    avoid_phys: np.ndarray | None = None,
    row_constraints: Sequence[RowConstraint] | None = None,
    max_draws: int | None = None,
) -> CandidatePool:
    """Sample an exact-size unique pool without allocating the Cartesian grid."""
    grids = _design_grids(design)
    if isinstance(pool_size, bool) or not isinstance(pool_size, (int, np.integer)):
        raise ValueError("pool_size must be a positive integer.")
    requested = int(pool_size)
    if requested <= 0:
        raise ValueError("pool_size must be a positive integer.")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer.")
    if max_draws is None:
        draw_limit = max(1000, requested * 50)
    elif isinstance(max_draws, bool) or not isinstance(max_draws, (int, np.integer)):
        raise ValueError("max_draws must be a positive integer.")
    else:
        draw_limit = int(max_draws)
    if draw_limit <= 0:
        raise ValueError("max_draws must be a positive integer.")

    dimension = len(grids)
    avoid_rows = np.vstack(
        [
            _physical_matrix(observed_phys, name="observed_phys", dimension=dimension),
            _physical_matrix(pending_phys, name="pending_phys", dimension=dimension),
            _physical_matrix(avoid_phys, name="avoid_phys", dimension=dimension),
        ]
    )
    avoid_indices = physical_rows_to_grid_indices(avoid_rows, design)
    avoid_set = {tuple(int(value) for value in row) for row in avoid_indices}
    total_grid_size = prod(int(grid.size) for grid in grids)
    available_without_constraints = total_grid_size - len(avoid_set)
    if requested > available_without_constraints:
        raise CandidatePoolSamplingError(
            requested=requested,
            accepted=0,
            draws=0,
            max_draws=draw_limit,
            rejected_duplicate=0,
            rejected_avoid=0,
            rejected_constraint=0,
            reason=(
                "The request exceeds the number of grid tuples remaining after "
                f"explicit exclusions ({available_without_constraints})."
            ),
        )

    generator = np.random.default_rng(int(seed))
    axis_sizes = np.asarray([grid.size for grid in grids], dtype=np.int64)
    seen: set[tuple[int, ...]] = set()
    accepted_indices: list[tuple[int, ...]] = []
    draws = rejected_duplicate = rejected_avoid = rejected_constraint = 0
    while len(accepted_indices) < requested and draws < draw_limit:
        index_tuple = tuple(int(generator.integers(0, high)) for high in axis_sizes)
        draws += 1
        if index_tuple in seen:
            rejected_duplicate += 1
            continue
        seen.add(index_tuple)
        if index_tuple in avoid_set:
            rejected_avoid += 1
            continue
        row_indices = np.asarray(index_tuple, dtype=np.int64)[None, :]
        row_physical = _indices_to_physical(row_indices, grids)
        if not bool(apply_row_constraints(row_physical, design, row_constraints)[0]):
            rejected_constraint += 1
            continue
        accepted_indices.append(index_tuple)

    if len(accepted_indices) != requested:
        raise CandidatePoolSamplingError(
            requested=requested,
            accepted=len(accepted_indices),
            draws=draws,
            max_draws=draw_limit,
            rejected_duplicate=rejected_duplicate,
            rejected_avoid=rejected_avoid,
            rejected_constraint=rejected_constraint,
            reason="Maximum draws reached; constraints and exclusions were not relaxed.",
        )
    index_array = np.asarray(accepted_indices, dtype=np.int64)
    physical = _indices_to_physical(index_array, grids)
    normalized = _normalize_physical(physical, design)
    return CandidatePool(
        grid_indices=index_array,
        X_phys=physical,
        X_norm=normalized,
        seed=int(seed),
        draws=draws,
        rejected_duplicate=rejected_duplicate,
        rejected_avoid=rejected_avoid,
        rejected_constraint=rejected_constraint,
    )


__all__ = [
    "CandidatePool",
    "CandidatePoolSamplingError",
    "physical_rows_to_grid_indices",
    "sample_discrete_candidate_pool",
]
