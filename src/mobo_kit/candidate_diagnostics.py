"""Numerical and plotting diagnostics for discrete candidate batches.

All distances in this module are defined in normalized input space.  Plotting
helpers are intentionally file-oriented and use a headless Matplotlib backend
so they are safe in automated, CPU-only campaign checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from .design import Design


def _matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (N, D); got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _weights(dimension_weights: np.ndarray | None, dimension: int) -> np.ndarray:
    if dimension_weights is None:
        return np.ones(dimension, dtype=float)
    weights = np.asarray(dimension_weights, dtype=float)
    if weights.shape != (dimension,):
        raise ValueError(
            "dimension_weights must have shape " f"({dimension},); got {weights.shape}."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("dimension_weights must be finite and strictly positive.")
    return weights


def pairwise_normalized_distances(
    X_norm: np.ndarray,
    *,
    dimension_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return the weighted Euclidean pairwise-distance matrix for ``(N, D)``."""
    X = _matrix(X_norm, name="X_norm")
    weights = _weights(dimension_weights, X.shape[1])
    differences = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(weights * differences**2, axis=-1))


def nearest_reference_distances(
    X_norm: np.ndarray,
    reference_norm: np.ndarray | None,
    *,
    dimension_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return each candidate's nearest observed/pending normalized distance."""
    X = _matrix(X_norm, name="X_norm")
    weights = _weights(dimension_weights, X.shape[1])
    if reference_norm is None:
        return np.full(X.shape[0], np.nan, dtype=float)
    reference = _matrix(reference_norm, name="reference_norm")
    if reference.shape[1] != X.shape[1]:
        raise ValueError(
            "reference_norm and X_norm must have the same final dimension."
        )
    if reference.shape[0] == 0:
        return np.full(X.shape[0], np.nan, dtype=float)
    differences = X[:, None, :] - reference[None, :, :]
    distances = np.sqrt(np.sum(weights * differences**2, axis=-1))
    return distances.min(axis=1)


def grid_membership_mask(
    X_phys: np.ndarray,
    design: Design,
    *,
    atol: float = 1e-9,
) -> np.ndarray:
    """Return a per-row mask indicating exact membership in every design grid."""
    X = _matrix(X_phys, name="X_phys")
    dimension = len(design.var_array)
    if X.shape[1] != dimension:
        raise ValueError(
            f"X_phys has {X.shape[1]} columns but the design has {dimension}."
        )
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative.")
    valid = np.ones(X.shape[0], dtype=bool)
    for column, grid in enumerate(design.var_array):
        grid_values = np.asarray(grid, dtype=float)
        valid &= np.any(
            np.isclose(
                X[:, column, None],
                grid_values[None, :],
                rtol=0.0,
                atol=atol,
            ),
            axis=1,
        )
    return valid


def boundary_flags(X_norm: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Flag candidate dimensions lying on either normalized boundary."""
    X = _matrix(X_norm, name="X_norm")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative.")
    return np.isclose(X, 0.0, rtol=0.0, atol=atol) | np.isclose(
        X, 1.0, rtol=0.0, atol=atol
    )


@dataclass(frozen=True)
class BatchDistanceDiagnostics:
    """Compact distance and validity summary for a selected batch."""

    pairwise_distance_matrix: np.ndarray
    minimum_within_batch_distance: float | None
    mean_within_batch_distance: float | None
    maximum_within_batch_distance: float | None
    nearest_observed_pending_distance: np.ndarray
    duplicate_row_pairs: tuple[tuple[int, int], ...]
    grid_valid_rows: np.ndarray | None
    boundary_flags: np.ndarray
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly shallow mapping."""
        return asdict(self)


def summarize_candidate_batch(
    X_norm: np.ndarray,
    *,
    observed_pending_norm: np.ndarray | None = None,
    X_phys: np.ndarray | None = None,
    design: Design | None = None,
    dimension_weights: np.ndarray | None = None,
    duplicate_atol: float = 1e-12,
    metadata: Mapping[str, Any] | None = None,
) -> BatchDistanceDiagnostics:
    """Compute batch-only ``O(q^2 D)`` diagnostics and reference distances."""
    X = _matrix(X_norm, name="X_norm")
    if not np.isfinite(duplicate_atol) or duplicate_atol < 0:
        raise ValueError("duplicate_atol must be finite and non-negative.")
    distances = pairwise_normalized_distances(X, dimension_weights=dimension_weights)
    if X.shape[0] >= 2:
        triangle = distances[np.triu_indices(X.shape[0], k=1)]
        minimum = float(triangle.min())
        mean = float(triangle.mean())
        maximum = float(triangle.max())
    else:
        minimum = mean = maximum = None

    duplicate_pairs: list[tuple[int, int]] = []
    for left in range(X.shape[0]):
        for right in range(left + 1, X.shape[0]):
            if np.allclose(X[left], X[right], rtol=0.0, atol=duplicate_atol):
                duplicate_pairs.append((left, right))

    if (X_phys is None) != (design is None):
        raise ValueError("X_phys and design must be supplied together.")
    if X_phys is not None:
        physical = _matrix(X_phys, name="X_phys")
        if physical.shape[0] != X.shape[0]:
            raise ValueError("X_phys and X_norm must contain the same row count.")
    else:
        physical = None
    grid_valid = (
        None
        if physical is None
        else grid_membership_mask(physical, design)  # type: ignore[arg-type]
    )
    return BatchDistanceDiagnostics(
        pairwise_distance_matrix=distances,
        minimum_within_batch_distance=minimum,
        mean_within_batch_distance=mean,
        maximum_within_batch_distance=maximum,
        nearest_observed_pending_distance=nearest_reference_distances(
            X,
            observed_pending_norm,
            dimension_weights=dimension_weights,
        ),
        duplicate_row_pairs=tuple(duplicate_pairs),
        grid_valid_rows=grid_valid,
        boundary_flags=boundary_flags(X),
        metadata={} if metadata is None else dict(metadata),
    )


def _output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig: Any, path: Path, watermark: str | None) -> None:
    metadata: dict[str, str] | None = None
    if watermark is not None:
        if not isinstance(watermark, str) or not watermark.strip():
            raise ValueError("watermark must be a nonblank string when provided.")
        label = watermark.strip()
        fig.text(
            0.5,
            0.015,
            label,
            ha="center",
            va="bottom",
            color="firebrick",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "firebrick", "alpha": 0.9},
        )
        metadata = {"Description": label}
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0) if watermark else None)
    fig.savefig(path, dpi=160, metadata=metadata)
    plt.close(fig)


def plot_candidate_pca(
    observed_norm: np.ndarray,
    selected_norm: np.ndarray,
    output_path: str | Path,
    *,
    pool_norm: np.ndarray | None = None,
    pool_sample_size: int = 1000,
    seed: int = 0,
    watermark: str | None = None,
) -> Path:
    """Save a two-component PCA view of observed, pool, and selected points."""
    observed = _matrix(observed_norm, name="observed_norm")
    selected = _matrix(selected_norm, name="selected_norm")
    if observed.shape[1] != selected.shape[1]:
        raise ValueError("observed_norm and selected_norm dimensions must match.")
    groups: list[tuple[str, np.ndarray]] = [("Observed", observed)]
    if pool_norm is not None:
        pool = _matrix(pool_norm, name="pool_norm")
        if pool.shape[1] != observed.shape[1]:
            raise ValueError("pool_norm and observed_norm dimensions must match.")
        if pool_sample_size <= 0:
            raise ValueError("pool_sample_size must be positive.")
        if pool.shape[0] > pool_sample_size:
            rng = np.random.default_rng(seed)
            indices = np.sort(
                rng.choice(pool.shape[0], size=pool_sample_size, replace=False)
            )
            pool = pool[indices]
        groups.append(("Pool", pool))
    groups.append(("Selected", selected))
    combined = np.vstack([values for _, values in groups])
    if combined.shape[0] < 2 or combined.shape[1] < 2:
        raise ValueError("PCA plot requires at least two rows and two inputs.")
    projected = PCA(n_components=2).fit_transform(combined)

    path = _output_path(output_path)
    fig, axis = plt.subplots(figsize=(7, 5))
    offset = 0
    styles: Mapping[str, Mapping[str, Any]] = {
        "Observed": {"marker": "o", "alpha": 0.75, "s": 38},
        "Pool": {"marker": ".", "alpha": 0.2, "s": 14},
        "Selected": {"marker": "*", "alpha": 1.0, "s": 130},
    }
    for label, values in groups:
        count = values.shape[0]
        points = projected[offset : offset + count]
        axis.scatter(points[:, 0], points[:, 1], label=label, **styles[label])
        offset += count
    axis.set(xlabel="PC1", ylabel="PC2", title="Candidate acquisition in input space")
    axis.legend()
    _save_figure(fig, path, watermark)
    return path


def plot_parallel_coordinates(
    selected_norm: np.ndarray,
    input_names: Sequence[str],
    output_path: str | Path,
    *,
    watermark: str | None = None,
) -> Path:
    """Save normalized selected conditions as a parallel-coordinates plot."""
    selected = _matrix(selected_norm, name="selected_norm")
    if len(input_names) != selected.shape[1]:
        raise ValueError("input_names must match the selected input dimension.")
    path = _output_path(output_path)
    fig, axis = plt.subplots(figsize=(max(8, selected.shape[1]), 4.5))
    positions = np.arange(selected.shape[1])
    for row_index, row in enumerate(selected):
        axis.plot(positions, row, marker="o", label=f"Selection {row_index + 1}")
    axis.set_xticks(positions, input_names, rotation=40, ha="right")
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("Normalized condition")
    axis.set_title("Selected candidate conditions")
    axis.legend(ncol=min(3, max(1, selected.shape[0])))
    _save_figure(fig, path, watermark)
    return path


def plot_distance_heatmap(
    X_norm: np.ndarray,
    output_path: str | Path,
    *,
    dimension_weights: np.ndarray | None = None,
    watermark: str | None = None,
) -> Path:
    """Save the within-batch normalized-distance matrix as a heatmap."""
    matrix = pairwise_normalized_distances(X_norm, dimension_weights=dimension_weights)
    path = _output_path(output_path)
    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(matrix, cmap="viridis")
    labels = [str(index + 1) for index in range(matrix.shape[0])]
    axis.set_xticks(range(matrix.shape[0]), labels)
    axis.set_yticks(range(matrix.shape[0]), labels)
    axis.set(xlabel="Selection", ylabel="Selection", title="Batch distances")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < matrix.max() * 0.55 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, label="Normalized Euclidean distance")
    _save_figure(fig, path, watermark)
    return path


def plot_selection_scores(
    selection_order: Sequence[int],
    base_log_scores: Sequence[float],
    penalized_log_scores: Sequence[float],
    output_path: str | Path,
    *,
    watermark: str | None = None,
) -> Path:
    """Save base-versus-penalized acquisition values by selection order."""
    order = np.asarray(selection_order)
    base = np.asarray(base_log_scores, dtype=float)
    penalized = np.asarray(penalized_log_scores, dtype=float)
    if order.ndim != 1 or base.shape != order.shape or penalized.shape != order.shape:
        raise ValueError("selection order and score arrays must share shape (q,).")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(penalized)):
        raise ValueError("selection scores must be finite.")
    path = _output_path(output_path)
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(order, base, marker="o", label="Base log acquisition")
    axis.plot(order, penalized, marker="s", label="Penalized log acquisition")
    axis.set(
        xlabel="Selection order",
        ylabel="Log acquisition",
        title="Sequential acquisition and local penalty",
    )
    axis.set_xticks(order)
    axis.legend()
    _save_figure(fig, path, watermark)
    return path
