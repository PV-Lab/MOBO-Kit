"""Deterministic nested Sobol prefixes for finite discrete designs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import prod
from typing import Sequence
import warnings

import numpy as np
import scipy
from scipy.stats import qmc

from .candidate_pool import (
    CandidatePool,
    CandidatePoolSamplingError,
    physical_rows_to_grid_indices,
)
from .constraints import RowConstraint, apply_row_constraints
from .design import Design


@dataclass(frozen=True)
class NestedSobolPoolResult:
    """Accepted-unique Sobol prefixes and their reproducibility metadata."""

    pools: dict[int, CandidatePool]
    prefix_hashes: dict[int, str]
    accepted_sizes: tuple[int, ...]
    scramble_seed: int
    raw_sobol_draws: int
    rejected_duplicate: int
    rejected_avoid: int
    rejected_constraint: int
    ignored_off_grid_observed: int
    scipy_version: str

    @property
    def largest_pool(self) -> CandidatePool:
        """Return the largest accepted prefix."""

        return self.pools[self.accepted_sizes[-1]]

    @property
    def accepted_count(self) -> int:
        """Return the accepted count in the final master prefix."""

        return self.largest_pool.size

    @property
    def pools_by_size(self) -> dict[int, CandidatePool]:
        """Alias spelling useful to report-building callers."""

        return self.pools


@dataclass(frozen=True)
class _PrefixSnapshot:
    draws: int
    rejected_duplicate: int
    rejected_avoid: int
    rejected_constraint: int


def _validated_grids(design: Design) -> tuple[np.ndarray, ...]:
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


def _positive_sizes(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("accepted_sizes must be a non-empty sequence of integers.")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "accepted_sizes must be a non-empty sequence of integers."
        ) from exc
    if not raw:
        raise ValueError("accepted_sizes must not be empty.")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in raw
    ):
        raise ValueError("accepted_sizes must contain only positive integers.")
    normalized = tuple(int(value) for value in raw)
    if len(set(normalized)) != len(normalized):
        raise ValueError("accepted_sizes must not contain duplicates.")
    return tuple(sorted(normalized))


def _non_negative_seed(value: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError("scramble_seed must be a non-negative integer.")
    return int(value)


def _physical_rows(
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


def map_unit_points_to_grid_indices(
    unit_points: np.ndarray,
    design: Design,
) -> np.ndarray:
    """Map points in ``[0, 1)`` to exact discrete grid indices."""

    grids = _validated_grids(design)
    points = np.asarray(unit_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != len(grids):
        raise ValueError(
            "unit_points must have shape (N, n_design_inputs); "
            f"got {points.shape} for {len(grids)} inputs."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("unit_points must contain only finite values.")
    if np.any(points < 0.0) or np.any(points >= 1.0):
        raise ValueError("unit_points must lie in the half-open interval [0, 1).")
    axis_sizes = np.asarray([grid.size for grid in grids], dtype=np.int64)
    indices = np.floor(points * axis_sizes[None, :]).astype(np.int64)
    return np.minimum(indices, axis_sizes[None, :] - 1)


def _indices_to_physical(
    grid_indices: np.ndarray,
    grids: tuple[np.ndarray, ...],
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
    changing = spans > 0.0
    normalized[:, changing] = (X_phys[:, changing] - lower[changing]) / spans[changing]
    return normalized


def hash_grid_index_prefix(grid_indices: np.ndarray) -> str:
    """Return a platform-stable SHA-256 for one integer-index prefix."""

    indices = np.asarray(grid_indices)
    if indices.ndim != 2:
        raise ValueError("grid_indices must be a two-dimensional integer matrix.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("grid_indices must have an integer dtype.")
    canonical = np.ascontiguousarray(indices, dtype="<i8")
    shape = np.asarray(canonical.shape, dtype="<i8")
    return hashlib.sha256(shape.tobytes() + canonical.tobytes()).hexdigest().upper()


def _strict_exclusion_indices(
    rows: np.ndarray,
    design: Design,
) -> np.ndarray:
    if rows.shape[0] == 0:
        return np.empty((0, len(design.names)), dtype=np.int64)
    return physical_rows_to_grid_indices(rows, design)


def _observed_exclusion_indices(
    rows: np.ndarray,
    design: Design,
) -> tuple[np.ndarray, int]:
    """Partition observed rows, deliberately skipping exact off-grid controls."""

    accepted: list[np.ndarray] = []
    ignored = 0
    lower = np.asarray(design.lowers, dtype=float)
    upper = np.asarray(design.uppers, dtype=float)
    for row in rows:
        if np.any(row < lower) or np.any(row > upper):
            raise ValueError(
                "observed_phys rows must remain within the design bounds even "
                "when an observed recipe is off-grid."
            )
        try:
            accepted.append(physical_rows_to_grid_indices(row[None, :], design)[0])
        except ValueError as exc:
            if "off-grid" not in str(exc):
                raise
            ignored += 1
    if not accepted:
        return np.empty((0, len(design.names)), dtype=np.int64), ignored
    return np.asarray(accepted, dtype=np.int64), ignored


def _sobol_prefix(dimension: int, draws: int, seed: int) -> np.ndarray:
    engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    if draws > 0 and draws & (draws - 1) == 0:
        return engine.random_base2(int(draws.bit_length() - 1))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The balance properties of Sobol.*",
            category=UserWarning,
        )
        return engine.random(draws)


def build_nested_sobol_discrete_pool(
    design: Design,
    accepted_sizes: Sequence[int],
    *,
    scramble_seed: int,
    observed_phys: np.ndarray | None = None,
    pending_phys: np.ndarray | None = None,
    avoid_phys: np.ndarray | None = None,
    row_constraints: Sequence[RowConstraint] | None = None,
    max_raw_draws: int | None = None,
) -> NestedSobolPoolResult:
    """Build exact accepted-unique nested prefixes from one Sobol scramble.

    Observed rows that are not exact grid members are intentionally omitted from
    index exclusion (the D2D control is such a row). Pending and explicit avoid
    rows must be exact grid members and fail closed otherwise.
    """

    grids = _validated_grids(design)
    sizes = _positive_sizes(accepted_sizes)
    seed = _non_negative_seed(scramble_seed)
    largest = sizes[-1]
    if max_raw_draws is None:
        draw_limit = max(1024, largest * 50)
    elif (
        isinstance(max_raw_draws, (bool, np.bool_))
        or not isinstance(max_raw_draws, (int, np.integer))
        or int(max_raw_draws) <= 0
    ):
        raise ValueError("max_raw_draws must be a positive integer.")
    else:
        draw_limit = int(max_raw_draws)
    if draw_limit < largest:
        raise CandidatePoolSamplingError(
            requested=largest,
            accepted=0,
            draws=0,
            max_draws=draw_limit,
            rejected_duplicate=0,
            rejected_avoid=0,
            rejected_constraint=0,
            reason="max_raw_draws is smaller than the requested accepted prefix.",
        )

    dimension = len(grids)
    observed = _physical_rows(observed_phys, name="observed_phys", dimension=dimension)
    pending = _physical_rows(pending_phys, name="pending_phys", dimension=dimension)
    explicit_avoid = _physical_rows(avoid_phys, name="avoid_phys", dimension=dimension)
    observed_indices, ignored_off_grid = _observed_exclusion_indices(observed, design)
    pending_indices = _strict_exclusion_indices(pending, design)
    avoid_indices = _strict_exclusion_indices(explicit_avoid, design)
    exclusions = np.vstack([observed_indices, pending_indices, avoid_indices])
    avoid_set = {tuple(int(value) for value in row) for row in exclusions}

    total_grid_size = prod(int(grid.size) for grid in grids)
    available = total_grid_size - len(avoid_set)
    if largest > available:
        raise CandidatePoolSamplingError(
            requested=largest,
            accepted=0,
            draws=0,
            max_draws=draw_limit,
            rejected_duplicate=0,
            rejected_avoid=0,
            rejected_constraint=0,
            reason=(
                "The request exceeds the number of grid tuples remaining after "
                f"exact exclusions ({available})."
            ),
        )

    seen: set[tuple[int, ...]] = set()
    accepted: list[tuple[int, ...]] = []
    snapshots: dict[int, _PrefixSnapshot] = {}
    draws = rejected_duplicate = rejected_avoid = rejected_constraint = 0
    prefix_draws = 1 << max(0, (largest - 1).bit_length())

    while len(accepted) < largest and draws < draw_limit:
        target_draws = min(prefix_draws, draw_limit)
        points = _sobol_prefix(dimension, target_draws, seed)
        new_indices = map_unit_points_to_grid_indices(
            points[draws:target_draws], design
        )
        for row in new_indices:
            draws += 1
            key = tuple(int(value) for value in row)
            if key in seen:
                rejected_duplicate += 1
                continue
            seen.add(key)
            if key in avoid_set:
                rejected_avoid += 1
                continue
            if row_constraints:
                physical = _indices_to_physical(row[None, :], grids)
                if not bool(
                    apply_row_constraints(physical, design, row_constraints)[0]
                ):
                    rejected_constraint += 1
                    continue
            accepted.append(key)
            accepted_count = len(accepted)
            if accepted_count in sizes:
                snapshots[accepted_count] = _PrefixSnapshot(
                    draws=draws,
                    rejected_duplicate=rejected_duplicate,
                    rejected_avoid=rejected_avoid,
                    rejected_constraint=rejected_constraint,
                )
            if accepted_count == largest:
                break
        if target_draws == draw_limit:
            break
        prefix_draws *= 2

    if len(accepted) != largest:
        raise CandidatePoolSamplingError(
            requested=largest,
            accepted=len(accepted),
            draws=draws,
            max_draws=draw_limit,
            rejected_duplicate=rejected_duplicate,
            rejected_avoid=rejected_avoid,
            rejected_constraint=rejected_constraint,
            reason=(
                "Maximum raw Sobol draws reached; exact exclusions and constraints "
                "were not relaxed."
            ),
        )

    all_indices = np.asarray(accepted, dtype=np.int64)
    pools: dict[int, CandidatePool] = {}
    hashes: dict[int, str] = {}
    for size in sizes:
        indices = all_indices[:size].copy()
        physical = _indices_to_physical(indices, grids)
        snapshot = snapshots[size]
        pools[size] = CandidatePool(
            grid_indices=indices,
            X_phys=physical,
            X_norm=_normalize_physical(physical, design),
            seed=seed,
            draws=snapshot.draws,
            rejected_duplicate=snapshot.rejected_duplicate,
            rejected_avoid=snapshot.rejected_avoid,
            rejected_constraint=snapshot.rejected_constraint,
        )
        hashes[size] = hash_grid_index_prefix(indices)

    return NestedSobolPoolResult(
        pools=pools,
        prefix_hashes=hashes,
        accepted_sizes=sizes,
        scramble_seed=seed,
        raw_sobol_draws=draws,
        rejected_duplicate=rejected_duplicate,
        rejected_avoid=rejected_avoid,
        rejected_constraint=rejected_constraint,
        ignored_off_grid_observed=ignored_off_grid,
        scipy_version=scipy.__version__,
    )


__all__ = [
    "NestedSobolPoolResult",
    "build_nested_sobol_discrete_pool",
    "hash_grid_index_prefix",
    "map_unit_points_to_grid_indices",
]
