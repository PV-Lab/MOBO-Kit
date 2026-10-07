"""Deterministic, grid-safe Latin hypercube campaign design."""

from __future__ import annotations

from math import prod
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .constraints import RowConstraint, apply_row_constraints
from .design import Design


def _max_abs_corr(X: np.ndarray) -> float:
    """Return the largest absolute Pearson correlation between varying columns."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {values.shape}.")
    if values.shape[0] < 2 or values.shape[1] < 2:
        return 0.0

    varying = np.ptp(values, axis=0) > 0
    varying_values = values[:, varying]
    if varying_values.shape[1] < 2:
        return 0.0

    corr = np.corrcoef(varying_values, rowvar=False)
    absolute = np.abs(corr)
    np.fill_diagonal(absolute, 0.0)
    absolute = np.nan_to_num(absolute, nan=0.0, posinf=1.0, neginf=1.0)
    return float(np.max(absolute))


def _pick_subset_with_corr(
    X: np.ndarray,
    n: int,
    threshold: float,
    tries: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """Search deterministically (for a seeded RNG) for a qualifying subset."""
    if X.shape[0] < n:
        raise ValueError(
            f"Need at least {n} rows for subset selection; got {X.shape[0]}."
        )

    best = X[:n].copy()
    best_corr = _max_abs_corr(best)
    if best_corr <= threshold or X.shape[0] == n:
        return best, best_corr

    for _ in range(tries):
        indices = rng.choice(X.shape[0], size=n, replace=False)
        candidate = X[indices]
        correlation = _max_abs_corr(candidate)
        if correlation < best_corr:
            best = candidate.copy()
            best_corr = correlation
        if correlation <= threshold:
            return candidate.copy(), correlation

    return best, best_corr


def _lhs_select_numeric(df: pd.DataFrame, design: Design) -> pd.DataFrame:
    columns = [name for name in design.names if name in df.columns]
    return df[columns].select_dtypes(include=[np.number]).copy()


def _lhs_labels(design: Design, columns: Sequence[str]) -> list[str]:
    """Build labels from the fields Design actually owns (names and units)."""
    labels = {}
    for name, unit in zip(design.names, design.units):
        labels[name] = f"{name} [{unit}]" if unit else name
    return [labels.get(column, column) for column in columns]


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer; got {integer}.")
    return integer


def _validate_design(design: Design) -> None:
    if not isinstance(design, Design):
        raise TypeError(
            f"design must be a Design instance; got {type(design).__name__}."
        )
    dimension = len(design.names)
    if dimension == 0:
        raise ValueError("Design must contain at least one input.")
    if len(set(design.names)) != dimension:
        raise ValueError("Design input names must be unique.")
    if len(design.units) != dimension:
        raise ValueError(
            f"Design has {dimension} names but {len(design.units)} unit entries."
        )
    if len(design.var_list) != dimension:
        raise ValueError(
            f"Design has {dimension} names but {len(design.var_list)} variable grids."
        )
    for field_name in ("lowers", "uppers", "steps"):
        values = np.asarray(getattr(design, field_name), dtype=float)
        if values.shape != (dimension,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Design field '{field_name}' must contain {dimension} finite values."
            )
    if np.any(np.asarray(design.steps, dtype=float) <= 0):
        raise ValueError("Design steps must all be > 0.")

    for index, (name, raw_grid) in enumerate(zip(design.names, design.var_list)):
        grid = np.asarray(raw_grid, dtype=float)
        if grid.ndim != 1 or grid.size == 0:
            raise ValueError(
                f"Design grid for input '{name}' must be a non-empty 1D array."
            )
        if not np.all(np.isfinite(grid)):
            raise ValueError(
                f"Design grid for input '{name}' contains non-finite values."
            )
        if np.unique(grid).size != grid.size or np.any(np.diff(grid) <= 0):
            raise ValueError(
                f"Design grid for input '{name}' must be strictly increasing "
                "and unique."
            )
        if grid.size > 1:
            step = float(design.steps[index])
            if not np.allclose(
                np.diff(grid),
                step,
                rtol=0.0,
                atol=max(1e-12, abs(step) * 1e-9),
            ):
                raise ValueError(
                    f"Design grid for input '{name}' is not aligned to step {step}."
                )
        if not np.isclose(grid[0], design.lowers[index], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Design lower bound for input '{name}' does not match its grid."
            )
        if not np.isclose(grid[-1], design.uppers[index], rtol=0.0, atol=1e-12):
            raise ValueError(
                f"Design upper bound for input '{name}' does not match its grid."
            )


def _normalize_constraints(
    row_constraints: Optional[Union[RowConstraint, Sequence[RowConstraint]]],
) -> list[RowConstraint]:
    if row_constraints is None:
        return []
    if callable(row_constraints):
        return [row_constraints]
    constraints = list(row_constraints)
    if not all(callable(constraint) for constraint in constraints):
        raise ValueError("Every row constraint must be callable.")
    return constraints


def _latin_hypercube_unit(
    sample_count: int,
    dimension: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one randomized Latin hypercube using a persistent RNG."""
    jitter = rng.random((sample_count, dimension))
    unit = (np.arange(sample_count, dtype=float)[:, None] + jitter) / sample_count
    for column in range(dimension):
        rng.shuffle(unit[:, column])
    return unit


def _snap_to_design_grid(X: np.ndarray, design: Design) -> np.ndarray:
    """Snap physical values to exact configured grid values."""
    values = np.asarray(X, dtype=float)
    snapped = values.copy()
    for column, raw_grid in enumerate(design.var_list):
        grid = np.asarray(raw_grid, dtype=float)
        nearest = np.argmin(
            np.abs(snapped[:, column, None] - grid[None, :]),
            axis=1,
        )
        snapped[:, column] = grid[nearest]
    return snapped


def _generate_lhs_samples(
    design: Design,
    total_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one physical-space LHS without resetting RNG state."""
    unit = _latin_hypercube_unit(total_samples, len(design.names), rng)
    return design.lowers + unit * (design.uppers - design.lowers)


def _batch_apply_constraints(
    X_samples: np.ndarray,
    design: Design,
    constraints: Sequence[RowConstraint],
    batch_size: int,
) -> np.ndarray:
    """Apply constraints to already-snapped rows in bounded-size chunks."""
    if not constraints:
        return X_samples

    valid_batches = []
    for start in range(0, X_samples.shape[0], batch_size):
        batch = X_samples[start : start + batch_size]
        mask = apply_row_constraints(batch, design, constraints)
        if np.any(mask):
            valid_batches.append(batch[mask])
    if not valid_batches:
        return np.empty((0, X_samples.shape[1]), dtype=float)
    return np.vstack(valid_batches)


def _append_unique_rows(
    pool: list[np.ndarray],
    seen: set[tuple[float, ...]],
    rows: np.ndarray,
) -> None:
    """Append rows in encounter order, deduplicating exact snapped grid values."""
    for row in rows:
        key = tuple(float(value) for value in row)
        if key not in seen:
            seen.add(key)
            pool.append(np.asarray(row, dtype=float).copy())


def _validate_grid_membership(X: np.ndarray, design: Design) -> None:
    """Guard the campaign invariant that every emitted value is on its grid."""
    for column, (name, raw_grid) in enumerate(zip(design.names, design.var_list)):
        grid = np.asarray(raw_grid, dtype=float)
        if not np.all(np.isin(X[:, column], grid)):
            raise RuntimeError(
                "Internal LHS error: generated value outside configured grid "
                f"for '{name}'."
            )


def _strict_lhs_dataframe(
    *,
    design: Design,
    n: int,
    seed: Optional[int],
    snap_to_grids: bool,
    row_constraints: Optional[Union[RowConstraint, Sequence[RowConstraint]]],
    max_abs_corr: Optional[float],
    max_attempts: int,
    oversample: int,
    batch_size: int,
    subset_tries: int,
    samples_per_attempt: Optional[int],
    verbose: bool,
) -> pd.DataFrame:
    _validate_design(design)
    n = _validate_positive_int(n, name="n")
    max_attempts = _validate_positive_int(max_attempts, name="max_attempts")
    oversample = _validate_positive_int(oversample, name="oversample")
    batch_size = _validate_positive_int(batch_size, name="batch_size")
    subset_tries = _validate_positive_int(subset_tries, name="subset_tries")

    if not snap_to_grids:
        raise ValueError(
            "Campaign LHS requires snap_to_grids=True so every condition is exactly "
            "on the configured processing grid."
        )

    if seed is not None:
        if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)
        ):
            raise ValueError(f"seed must be an integer or None; got {seed!r}.")
        seed = int(seed)
        if seed < 0:
            raise ValueError(f"seed must be non-negative; got {seed}.")

    if max_abs_corr is not None:
        try:
            max_abs_corr = float(max_abs_corr)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_abs_corr must be a finite number in [0, 1].") from exc
        if not np.isfinite(max_abs_corr) or not 0.0 <= max_abs_corr <= 1.0:
            raise ValueError("max_abs_corr must be a finite number in [0, 1].")

    if samples_per_attempt is None:
        samples_per_attempt = max(n, n * oversample)
    else:
        samples_per_attempt = _validate_positive_int(
            samples_per_attempt, name="samples_per_attempt"
        )

    total_grid_points = prod(len(grid) for grid in design.var_list)
    if n > total_grid_points:
        raise ValueError(
            f"Requested n={n} unique conditions, but the design contains only "
            f"{total_grid_points} unique grid combinations."
        )

    constraints = _normalize_constraints(row_constraints)
    rng = np.random.default_rng(seed)
    unique_pool: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    best_corr = float("inf")

    for attempt in range(1, max_attempts + 1):
        raw = _generate_lhs_samples(
            design=design,
            total_samples=samples_per_attempt,
            rng=rng,
        )
        # Constraints deliberately see executable, snapped processing conditions.
        snapped = _snap_to_design_grid(raw, design)
        _validate_grid_membership(snapped, design)
        valid = _batch_apply_constraints(snapped, design, constraints, batch_size)
        _append_unique_rows(unique_pool, seen, valid)

        if verbose:
            print(
                f"[LHS] attempt {attempt}/{max_attempts}: "
                f"{len(unique_pool)} unique valid grid points"
            )

        if len(unique_pool) < n:
            continue

        pool_array = np.vstack(unique_pool)
        if max_abs_corr is None:
            selected = pool_array[:n].copy()
            _validate_grid_membership(selected, design)
            return pd.DataFrame(selected, columns=design.names)

        selected, correlation = _pick_subset_with_corr(
            pool_array,
            n=n,
            threshold=max_abs_corr,
            tries=subset_tries,
            rng=rng,
        )
        best_corr = min(best_corr, correlation)
        if verbose:
            print(
                f"[LHS] attempt {attempt}: best max|corr|={correlation:.6f}; "
                f"required <= {max_abs_corr:.6f}"
            )
        if correlation <= max_abs_corr:
            _validate_grid_membership(selected, design)
            return pd.DataFrame(selected, columns=design.names)

    details = (
        f"Generated {len(unique_pool)} unique constraint-valid grid points after "
        f"{max_attempts} attempt(s), using {samples_per_attempt} samples per attempt."
    )
    if max_abs_corr is not None and len(unique_pool) >= n:
        details += f" Best max|corr|={best_corr:.6f}, required <= {max_abs_corr:.6f}."
    raise RuntimeError(
        f"Unable to generate exactly n={n} campaign conditions. {details} "
        "Increase samples_per_attempt/max_attempts or revise the design constraints."
    )


def lhs_dataframe_optimized(
    design: Design,
    n: int,
    seed: Optional[int] = None,
    snap_to_grids: bool = True,
    row_constraints: Optional[Union[RowConstraint, Sequence[RowConstraint]]] = None,
    max_abs_corr: Optional[float] = None,
    max_attempts: int = 100,
    oversample: int = 5,
    batch_size: int = 100,
    subset_tries: int = 1000,
    early_stop_threshold: float = 0.1,
    verbose: bool = False,
    samples_per_attempt: Optional[int] = None,
) -> pd.DataFrame:
    """Generate exactly ``n`` deterministic, unique, constraint-valid grid rows.

    ``samples_per_attempt`` controls the candidate count directly. When omitted,
    it is ``max(n, n * oversample)``. A configured correlation limit is hard: the
    function raises instead of returning a noncompliant or unconstrained fallback.

    ``early_stop_threshold`` remains accepted for API compatibility but is not
    used; stopping occurs only when the hard ``max_abs_corr`` limit is satisfied.
    """
    try:
        compatibility_threshold = float(early_stop_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "early_stop_threshold must be a finite non-negative number."
        ) from exc
    if not np.isfinite(compatibility_threshold) or compatibility_threshold < 0:
        raise ValueError("early_stop_threshold must be a finite non-negative number.")
    return _strict_lhs_dataframe(
        design=design,
        n=n,
        seed=seed,
        snap_to_grids=snap_to_grids,
        row_constraints=row_constraints,
        max_abs_corr=max_abs_corr,
        max_attempts=max_attempts,
        oversample=oversample,
        batch_size=batch_size,
        subset_tries=subset_tries,
        samples_per_attempt=samples_per_attempt,
        verbose=verbose,
    )


def lhs_dataframe(
    design: Design,
    n: int,
    seed: Optional[int] = None,
    snap_to_grids: bool = True,
    row_constraints: Optional[Union[RowConstraint, Sequence[RowConstraint]]] = None,
    max_abs_corr: Optional[float] = None,
    max_attempts: int = 100,
    oversample: int = 3,
    subset_tries: int = 800,
    verbose: bool = False,
    samples_per_attempt: Optional[int] = None,
    batch_size: int = 100,
) -> pd.DataFrame:
    """Compatibility wrapper around the strict campaign LHS implementation."""
    return _strict_lhs_dataframe(
        design=design,
        n=n,
        seed=seed,
        snap_to_grids=snap_to_grids,
        row_constraints=row_constraints,
        max_abs_corr=max_abs_corr,
        max_attempts=max_attempts,
        oversample=oversample,
        batch_size=batch_size,
        subset_tries=subset_tries,
        samples_per_attempt=samples_per_attempt,
        verbose=verbose,
    )
