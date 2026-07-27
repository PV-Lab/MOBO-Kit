"""Headless, watermarked plotting helpers for Step 2C robustness diagnostics."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment


DEBUG_WATERMARK = "DEBUG ONLY - NOT APPROVED FOR EXPERIMENT"


def _output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("output_path must use a .png suffix.")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_debug_figure(fig: Any, output_path: str | Path) -> Path:
    try:
        path = _output_path(output_path)
        fig.text(
            0.5,
            0.012,
            DEBUG_WATERMARK,
            ha="center",
            va="bottom",
            color="firebrick",
            fontsize=9,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "firebrick",
                "alpha": 0.92,
            },
        )
        fig.tight_layout(rect=(0.0, 0.065, 1.0, 1.0))
        fig.savefig(
            path,
            dpi=160,
            metadata={"Description": DEBUG_WATERMARK},
        )
    finally:
        plt.close(fig)
    return path


def _frame(
    value: pd.DataFrame,
    *,
    name: str,
    required_columns: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    if value.empty:
        raise ValueError(f"{name} must not be empty.")
    missing = [column for column in required_columns if column not in value.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}.")
    return value.copy()


def _numeric(frame: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    for column in columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        values = converted.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} column {column!r} must be complete and finite.")
        frame[column] = values


def _labels(frame: pd.DataFrame, column: str, *, name: str) -> None:
    if frame[column].isna().any():
        raise ValueError(f"{name} column {column!r} must be complete.")
    labels = frame[column].astype(str).str.strip()
    if labels.eq("").any():
        raise ValueError(f"{name} column {column!r} must contain nonblank labels.")
    frame[column] = labels


def _matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must have non-empty shape (N, D) with D >= 2.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError(f"{name} must contain normalized coordinates in [0, 1].")
    return matrix


def _normalized_columns(
    frame: pd.DataFrame,
    *,
    prefix: str,
    name: str,
) -> list[str]:
    columns = sorted(
        (
            column
            for column in frame.columns
            if column.startswith(prefix) and column[len(prefix) :].isdigit()
        ),
        key=lambda column: int(column[len(prefix) :]),
    )
    expected = [f"{prefix}{index}" for index in range(len(columns))]
    if len(columns) < 2 or columns != expected:
        raise ValueError(
            f"{name} must contain contiguous {prefix}0..D columns with D >= 1."
        )
    return columns


def _natural_label_key(value: object) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in parts
        if part
    )


def _validate_aligned_labels(
    labels: Sequence[object] | None,
    *,
    count: int,
    name: str,
    default_prefix: str,
) -> np.ndarray:
    if labels is None:
        return np.asarray(
            [f"{default_prefix}{index + 1}" for index in range(count)], dtype=str
        )
    values = np.asarray(tuple(labels), dtype=object)
    if values.shape != (count,):
        raise ValueError(f"{name} must contain one label per candidate row.")
    if any(value is None or not str(value).strip() for value in values):
        raise ValueError(f"{name} must contain complete nonblank labels.")
    return np.asarray([str(value).strip() for value in values], dtype=str)


def _heatmap(
    values: np.ndarray,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    title: str,
    colorbar_label: str,
    output_path: str | Path,
    cmap: str = "Blues",
) -> Path:
    matrix = np.asarray(values, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape != (len(row_labels), len(column_labels))
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
    ):
        raise ValueError("Heatmap values must align with non-empty row/column labels.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Heatmap values must be complete and finite.")
    width = max(7.5, 0.55 * len(column_labels) + 3.0)
    height = max(4.5, 0.38 * len(row_labels) + 2.2)
    fig, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
    axis.set_xticks(
        np.arange(len(column_labels)), column_labels, rotation=40, ha="right"
    )
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set(title=title)
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(colorbar_label)
    if matrix.size <= 225:
        threshold = float(matrix.min() + (matrix.max() - matrix.min()) / 2.0)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > threshold else "black",
                )
    return _save_debug_figure(fig, output_path)


def _safe_axis_range(*values: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([np.asarray(value, dtype=float) for value in values])
    lower = float(combined.min())
    upper = float(combined.max())
    if lower == upper:
        padding = max(1.0, abs(lower) * 0.05)
    else:
        padding = (upper - lower) * 0.05
    return lower - padding, upper + padding


def plot_loocv_diagnostics(
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    objective_column: str = "objective",
    observed_column: str = "observed",
    predicted_column: str = "predicted_mean",
    std_column: str = "predicted_std",
) -> Path:
    """Plot LOOCV parity, standardized residuals, and interval coverage."""

    frame = _frame(
        predictions,
        name="predictions",
        required_columns=(
            objective_column,
            observed_column,
            predicted_column,
            std_column,
        ),
    )
    _labels(frame, objective_column, name="predictions")
    _numeric(
        frame,
        (observed_column, predicted_column, std_column),
        name="predictions",
    )
    if (frame[std_column] <= 0.0).any():
        raise ValueError("predicted standard deviations must be strictly positive.")
    frame["_standardized_residual"] = (
        frame[observed_column] - frame[predicted_column]
    ) / frame[std_column]
    frame = frame.sort_values(
        [objective_column, observed_column, predicted_column], kind="mergesort"
    ).reset_index(drop=True)
    objectives = sorted(frame[objective_column].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    parity, residual, coverage = axes
    for objective in objectives:
        group = frame[frame[objective_column] == objective]
        parity.scatter(
            group[observed_column],
            group[predicted_column],
            label=objective,
            alpha=0.8,
        )
    limits = _safe_axis_range(
        frame[observed_column].to_numpy(), frame[predicted_column].to_numpy()
    )
    parity.plot(limits, limits, linestyle="--", color="black", linewidth=1)
    parity.set(
        xlim=limits,
        ylim=limits,
        xlabel="Observed objective",
        ylabel="LOOCV predicted mean",
        title="LOOCV observed versus predicted",
    )
    parity.legend(fontsize=8)

    positions = np.arange(1, len(frame) + 1)
    residual.scatter(positions, frame["_standardized_residual"], s=28)
    residual.axhline(0.0, color="black", linewidth=1)
    residual.axhline(1.96, color="grey", linestyle="--", linewidth=1)
    residual.axhline(-1.96, color="grey", linestyle="--", linewidth=1)
    residual.set(
        xlabel="Held-out prediction (stable order)",
        ylabel="Standardized residual",
        title="LOOCV standardized residuals",
    )

    x_positions = np.arange(len(objectives), dtype=float)
    coverage_68 = []
    coverage_95 = []
    for objective in objectives:
        absolute = frame.loc[
            frame[objective_column] == objective, "_standardized_residual"
        ].abs()
        coverage_68.append(float((absolute <= 1.0).mean()))
        coverage_95.append(float((absolute <= 1.96).mean()))
    width = 0.36
    coverage.bar(x_positions - width / 2, coverage_68, width, label="68% interval")
    coverage.bar(x_positions + width / 2, coverage_95, width, label="95% interval")
    coverage.axhline(0.68, color="C0", linestyle=":", linewidth=1)
    coverage.axhline(0.95, color="C1", linestyle=":", linewidth=1)
    coverage.set_xticks(x_positions, objectives, rotation=30, ha="right")
    coverage.set(
        ylim=(0.0, 1.05),
        ylabel="Empirical coverage",
        title="LOOCV predictive interval coverage",
    )
    coverage.legend(fontsize=8)
    return _save_debug_figure(fig, output_path)


def plot_nested_search_convergence(
    convergence: pd.DataFrame,
    output_path: str | Path,
    *,
    pool_size_column: str = "pool_size",
    mean_distance_column: str = "mean_matched_distance",
    max_distance_column: str = "maximum_matched_distance",
    regional_columns: Mapping[float, str] | None = None,
) -> Path:
    """Plot nested-pool regional matches and optimal matched distances."""

    region_columns = (
        {
            0.10: "regional_matches_0_10",
            0.15: "regional_matches_0_15",
            0.20: "regional_matches_0_20",
        }
        if regional_columns is None
        else {float(key): value for key, value in regional_columns.items()}
    )
    if not region_columns or any(
        not np.isfinite(threshold) or threshold < 0.0 for threshold in region_columns
    ):
        raise ValueError("regional_columns must map finite non-negative thresholds.")
    required = (
        pool_size_column,
        mean_distance_column,
        max_distance_column,
        *region_columns.values(),
    )
    frame = _frame(convergence, name="convergence", required_columns=required)
    _numeric(frame, required, name="convergence")
    if (frame[pool_size_column] <= 0.0).any():
        raise ValueError("pool sizes must be strictly positive.")
    if frame[pool_size_column].duplicated().any():
        raise ValueError("pool sizes must be unique.")
    if (frame[[mean_distance_column, max_distance_column]] < 0.0).any().any():
        raise ValueError("matched distances must be non-negative.")
    if (frame[list(region_columns.values())] < 0.0).any().any():
        raise ValueError("regional match counts must be non-negative.")
    frame = frame.sort_values(pool_size_column, kind="mergesort")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for threshold, column in sorted(region_columns.items()):
        axes[0].plot(
            frame[pool_size_column],
            frame[column],
            marker="o",
            label=f"within {threshold:.2f}",
        )
    axes[0].set(
        xlabel="Accepted unique Sobol pool size",
        ylabel="Regionally matched candidates",
        title="Nested-search regional convergence",
    )
    axes[0].legend(fontsize=8)
    axes[0].ticklabel_format(axis="x", style="plain")

    axes[1].plot(
        frame[pool_size_column],
        frame[mean_distance_column],
        marker="o",
        label="Mean matched distance",
    )
    axes[1].plot(
        frame[pool_size_column],
        frame[max_distance_column],
        marker="s",
        label="Maximum matched distance",
    )
    axes[1].set(
        xlabel="Accepted unique Sobol pool size",
        ylabel="Normalized Euclidean distance",
        title="Nested-search matched distances",
    )
    axes[1].legend(fontsize=8)
    axes[1].ticklabel_format(axis="x", style="plain")
    return _save_debug_figure(fig, output_path)


def plot_bounded_utility_comparison(
    comparison: pd.DataFrame,
    output_path: str | Path,
    *,
    policy_column: str = "policy",
    raw_value_column: str = "raw_value",
    bounded_value_column: str = "bounded_value",
) -> Path:
    """Compare raw and bounded utility coordinates without exposing recipes."""

    frame = _frame(
        comparison,
        name="comparison",
        required_columns=(policy_column, raw_value_column, bounded_value_column),
    )
    _labels(frame, policy_column, name="comparison")
    _numeric(frame, (raw_value_column, bounded_value_column), name="comparison")
    summary = (
        frame.groupby(policy_column, sort=True)[
            [raw_value_column, bounded_value_column]
        ]
        .mean()
        .sort_index()
    )
    delta = summary[bounded_value_column] - summary[raw_value_column]
    positions = np.arange(len(summary), dtype=float)
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(
        positions - width / 2,
        summary[raw_value_column],
        width,
        label="Raw utility",
    )
    axes[0].bar(
        positions + width / 2,
        summary[bounded_value_column],
        width,
        label="Bounded utility",
    )
    axes[0].set_xticks(positions, summary.index, rotation=30, ha="right")
    axes[0].set(
        ylabel="Mean utility coordinate",
        title="Bounded-utility policy comparison",
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(positions, delta, color="C2")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(positions, summary.index, rotation=30, ha="right")
    axes[1].set(
        ylabel="Mean bounded minus raw utility",
        title="Utility-bound effect",
    )
    return _save_debug_figure(fig, output_path)


def plot_local_penalty_tradeoff(
    tradeoff: pd.DataFrame,
    output_path: str | Path,
    *,
    variant_column: str = "variant",
    acquisition_column: str = "sum_base_hvi",
    diversity_column: str = "minimum_within_batch_distance",
) -> Path:
    """Plot acquisition quality against within-batch diversity by policy."""

    frame = _frame(
        tradeoff,
        name="tradeoff",
        required_columns=(variant_column, acquisition_column, diversity_column),
    )
    _labels(frame, variant_column, name="tradeoff")
    _numeric(frame, (acquisition_column, diversity_column), name="tradeoff")
    if (frame[diversity_column] < 0.0).any():
        raise ValueError("diversity distances must be non-negative.")
    summary = (
        frame.groupby(variant_column, sort=True)[[acquisition_column, diversity_column]]
        .mean()
        .sort_index()
    )
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        summary[diversity_column], summary[acquisition_column], s=70, color="C3"
    )
    for label, row in summary.iterrows():
        axis.annotate(
            label,
            (row[diversity_column], row[acquisition_column]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set(
        xlabel="Minimum normalized within-batch distance",
        ylabel="Mean summed base HVI",
        title="Local-penalty acquisition versus diversity",
    )
    return _save_debug_figure(fig, output_path)


def plot_observation_influence_ranking(
    influence: pd.DataFrame,
    output_path: str | Path,
    *,
    sample_column: str = "sample_id",
    score_column: str = "influence_score",
) -> Path:
    """Plot a deterministic all-observation influence ranking."""

    frame = _frame(
        influence,
        name="influence",
        required_columns=(sample_column, score_column),
    )
    _labels(frame, sample_column, name="influence")
    _numeric(frame, (score_column,), name="influence")
    if frame[sample_column].duplicated().any():
        raise ValueError("sample identifiers must be unique.")
    frame = frame.sort_values(
        [score_column, sample_column], ascending=[True, True], kind="mergesort"
    )
    fig, axis = plt.subplots(figsize=(7.5, max(4.5, 0.3 * len(frame))))
    axis.barh(frame[sample_column], frame[score_column], color="C4")
    axis.set(
        xlabel="Composite influence score",
        ylabel="Omitted sample",
        title="Observation-influence ranking",
    )
    return _save_debug_figure(fig, output_path)


def plot_boundary_enrichment(
    enrichment: pd.DataFrame,
    output_path: str | Path,
    *,
    input_column: str = "input_name",
    pool_lower_column: str = "pool_lower_boundary_frequency",
    top_lower_column: str = "top_lower_boundary_frequency",
    selected_lower_column: str = "selected_lower_boundary_frequency",
    pool_upper_column: str = "pool_upper_boundary_frequency",
    top_upper_column: str = "top_upper_boundary_frequency",
    selected_upper_column: str = "selected_upper_boundary_frequency",
) -> Path:
    """Compare lower/upper boundary seeking in pool, high-HVI, and selections."""

    side_columns = {
        "Configured minimum": (
            pool_lower_column,
            top_lower_column,
            selected_lower_column,
        ),
        "Configured maximum": (
            pool_upper_column,
            top_upper_column,
            selected_upper_column,
        ),
    }
    required = (
        input_column,
        *(column for columns in side_columns.values() for column in columns),
    )
    frame = _frame(enrichment, name="enrichment", required_columns=required)
    _labels(frame, input_column, name="enrichment")
    _numeric(frame, required[1:], name="enrichment")
    if frame[input_column].duplicated().any():
        raise ValueError("input names must be unique.")
    frequencies = frame.loc[:, list(required[1:])]
    if ((frequencies < 0.0) | (frequencies > 1.0)).any().any():
        raise ValueError("boundary frequencies must remain in [0, 1].")
    frame = frame.sort_values(input_column, kind="mergesort")
    positions = np.arange(len(frame), dtype=float)
    width = 0.27
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(8, 0.7 * len(frame)), 8.0),
        sharex=True,
        constrained_layout=True,
    )
    for axis, (side_label, columns) in zip(axes, side_columns.items()):
        pool_column, top_column, selected_column = columns
        axis.bar(positions - width, frame[pool_column], width, label="Full pool")
        axis.bar(positions, frame[top_column], width, label="Top acquisition")
        axis.bar(positions + width, frame[selected_column], width, label="Selected")
        axis.set(
            ylim=(0.0, 1.05),
            ylabel="Boundary frequency",
            title=side_label,
        )
        axis.legend(fontsize=8)
    axes[-1].set_xticks(positions, frame[input_column], rotation=35, ha="right")
    fig.suptitle("Lower versus upper boundary enrichment by input dimension")
    return _save_debug_figure(fig, output_path)


def _canonical_region_order(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_strings = labels.astype(str)
    keys: list[np.ndarray] = [
        points[:, column] for column in reversed(range(points.shape[1]))
    ]
    keys.append(label_strings)
    return np.lexsort(tuple(keys))


def _region_medoids(
    points: np.ndarray,
    labels: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    region_names = sorted(set(labels.astype(str)))
    medoids: list[np.ndarray] = []
    for region in region_names:
        members = points[labels.astype(str) == region]
        distances = cdist(members, members, metric="euclidean")
        totals = distances.sum(axis=1)
        minimum = totals.min()
        tied = members[np.isclose(totals, minimum, rtol=0.0, atol=1e-12)]
        keys = tuple(tied[:, column] for column in reversed(range(tied.shape[1])))
        medoids.append(tied[np.lexsort(keys)[0]])
    return region_names, np.asarray(medoids, dtype=float)


def plot_robust_region_overview(
    region_points_norm: np.ndarray,
    region_labels: Sequence[object],
    output_path: str | Path,
    *,
    persistence: Sequence[float] | None = None,
    input_names: Sequence[str] | None = None,
) -> Path:
    """Plot robust-region members in PCA space and medoids in normalized space."""

    points = _matrix(region_points_norm, name="region_points_norm")
    if points.shape[0] < 2:
        raise ValueError("region_points_norm must contain at least two rows for PCA.")
    labels = np.asarray(tuple(region_labels), dtype=object)
    if labels.shape != (points.shape[0],):
        raise ValueError("region_labels must contain one label per normalized row.")
    if any(value is None or not str(value).strip() for value in labels):
        raise ValueError("region_labels must contain complete nonblank labels.")
    labels = np.asarray([str(value).strip() for value in labels], dtype=str)
    if persistence is None:
        counts = {
            label: int(np.count_nonzero(labels == label)) for label in set(labels)
        }
        persistence_values = np.asarray(
            [counts[label] for label in labels], dtype=float
        )
    else:
        persistence_values = np.asarray(tuple(persistence), dtype=float)
        if persistence_values.shape != (points.shape[0],):
            raise ValueError("persistence must contain one value per normalized row.")
        if not np.all(np.isfinite(persistence_values)) or np.any(
            persistence_values <= 0.0
        ):
            raise ValueError("persistence values must be finite and strictly positive.")
    if input_names is None:
        names = [f"X{index + 1}" for index in range(points.shape[1])]
    else:
        names = [str(value).strip() for value in input_names]
        if len(names) != points.shape[1] or any(not value for value in names):
            raise ValueError(
                "input_names must contain one nonblank name per input dimension."
            )

    order = _canonical_region_order(points, labels)
    points = points[order]
    labels = labels[order]
    persistence_values = persistence_values[order]
    projected = PCA(n_components=2).fit_transform(points)
    region_names, medoids = _region_medoids(points, labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.get_cmap("tab10")
    for index, region in enumerate(region_names):
        mask = labels == region
        sizes = 30.0 + 20.0 * persistence_values[mask] / persistence_values.max()
        axes[0].scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=sizes,
            alpha=0.75,
            color=colors(index % 10),
            label=f"Region {region}",
        )
    axes[0].set(
        xlabel="PC1",
        ylabel="PC2",
        title="Robust-region PCA overview",
    )
    axes[0].legend(fontsize=8)

    positions = np.arange(points.shape[1])
    for index, (region, medoid) in enumerate(zip(region_names, medoids)):
        axes[1].plot(
            positions,
            medoid,
            marker="o",
            color=colors(index % 10),
            label=f"Region {region}",
        )
    axes[1].set_xticks(positions, names, rotation=35, ha="right")
    axes[1].set(
        ylim=(-0.03, 1.03),
        ylabel="Normalized input coordinate",
        title="Robust-region medoids in normalized space",
    )
    axes[1].legend(fontsize=8)
    return _save_debug_figure(fig, output_path)


def plot_ard_lengthscale_comparison(
    hyperparameters: pd.DataFrame,
    output_path: str | Path,
    *,
    variant_column: str = "variant_name",
    objective_column: str = "objective_name",
    lengthscale_columns: Mapping[str, str] | None = None,
) -> Path:
    """Compare ARD lengthscales across model variants and objectives."""

    frame = _frame(
        hyperparameters,
        name="hyperparameters",
        required_columns=(variant_column, objective_column),
    )
    _labels(frame, variant_column, name="hyperparameters")
    _labels(frame, objective_column, name="hyperparameters")
    if lengthscale_columns is None:
        detected = {
            column.removeprefix("ard_lengthscale_"): column
            for column in frame.columns
            if column.startswith("ard_lengthscale_")
            and not column.endswith(
                (
                    "_near_floor",
                    "_very_small_normalized_domain",
                    "_extremely_large_flat",
                )
            )
        }
        columns = detected
    else:
        columns = {
            str(input_name).strip(): str(column).strip()
            for input_name, column in lengthscale_columns.items()
        }
    if not columns or any(not name or not column for name, column in columns.items()):
        raise ValueError(
            "lengthscale_columns must identify at least one nonblank ARD column."
        )
    missing = sorted(set(columns.values()) - set(frame.columns))
    if missing:
        raise ValueError(f"hyperparameters is missing ARD columns: {missing}.")
    _numeric(frame, tuple(columns.values()), name="hyperparameters")
    if (frame[list(columns.values())] <= 0.0).any().any():
        raise ValueError("ARD lengthscales must be strictly positive.")

    variants = sorted(frame[variant_column].unique(), key=_natural_label_key)
    objectives = sorted(frame[objective_column].unique(), key=_natural_label_key)
    observed_pairs = set(
        frame[[variant_column, objective_column]].itertuples(index=False, name=None)
    )
    expected_pairs = {
        (variant, objective) for variant in variants for objective in objectives
    }
    if observed_pairs != expected_pairs:
        raise ValueError(
            "hyperparameters must cover every model-variant/objective pair."
        )

    figure_width = max(6.5, 5.2 * len(objectives))
    fig, axes_value = plt.subplots(
        1, len(objectives), figsize=(figure_width, 4.8), squeeze=False
    )
    axes = axes_value.ravel()
    positions = np.arange(len(columns), dtype=float)
    colors = plt.get_cmap("tab10")
    for objective_index, objective in enumerate(objectives):
        axis = axes[objective_index]
        for variant_index, variant in enumerate(variants):
            group = frame[
                (frame[variant_column] == variant)
                & (frame[objective_column] == objective)
            ]
            values = group.loc[:, list(columns.values())].to_numpy(dtype=float)
            medians = np.median(values, axis=0)
            lower = np.quantile(values, 0.25, axis=0)
            upper = np.quantile(values, 0.75, axis=0)
            axis.errorbar(
                positions,
                medians,
                yerr=np.vstack((medians - lower, upper - medians)),
                marker="o",
                capsize=3,
                color=colors(variant_index % 10),
                label=variant,
            )
        axis.set_xticks(positions, list(columns), rotation=35, ha="right")
        axis.set_yscale("log")
        axis.set(
            xlabel="Normalized input dimension",
            ylabel="ARD lengthscale (log scale)",
            title=f"{objective}: ARD comparison",
        )
        axis.grid(axis="y", which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    return _save_debug_figure(fig, output_path)


def plot_control_omission_candidate_region_comparison(
    full_candidates_norm: np.ndarray,
    control_omitted_candidates_norm: np.ndarray,
    output_path: str | Path,
    *,
    full_region_labels: Sequence[object] | None = None,
    omit_region_labels: Sequence[object] | None = None,
) -> Path:
    """Compare full-fit and configured-control-omission candidates with PCA."""

    full = _matrix(full_candidates_norm, name="full_candidates_norm")
    omitted = _matrix(
        control_omitted_candidates_norm, name="control_omitted_candidates_norm"
    )
    if full.shape[1] != omitted.shape[1]:
        raise ValueError("Full and control-omitted candidates must share dimensions.")
    full_labels = (
        np.repeat("unassigned", full.shape[0])
        if full_region_labels is None
        else _validate_aligned_labels(
            full_region_labels,
            count=full.shape[0],
            name="full_region_labels",
            default_prefix="full-",
        )
    )
    omitted_labels = (
        np.repeat("unassigned", omitted.shape[0])
        if omit_region_labels is None
        else _validate_aligned_labels(
            omit_region_labels,
            count=omitted.shape[0],
            name="omit_region_labels",
            default_prefix="omit-",
        )
    )

    full_order = np.lexsort(
        tuple(full[:, column] for column in reversed(range(full.shape[1])))
    )
    omitted_order = np.lexsort(
        tuple(omitted[:, column] for column in reversed(range(omitted.shape[1])))
    )
    full = full[full_order]
    omitted = omitted[omitted_order]
    full_labels = full_labels[full_order]
    omitted_labels = omitted_labels[omitted_order]
    combined = np.vstack((full, omitted))
    if combined.shape[0] < 2:
        raise ValueError("At least two total candidate rows are required for PCA.")
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined)
    for component in range(2):
        loading = pca.components_[component]
        anchor = int(np.argmax(np.abs(loading)))
        if loading[anchor] < 0.0:
            projected[:, component] *= -1.0
    full_projected = projected[: full.shape[0]]
    omitted_projected = projected[full.shape[0] :]
    full_match, omitted_match = linear_sum_assignment(cdist(full, omitted))

    fig, axis = plt.subplots(figsize=(8.2, 5.6))
    for full_index, omitted_index in zip(full_match, omitted_match):
        axis.plot(
            [full_projected[full_index, 0], omitted_projected[omitted_index, 0]],
            [full_projected[full_index, 1], omitted_projected[omitted_index, 1]],
            color="0.7",
            linewidth=1,
            zorder=1,
        )
    region_names = sorted(
        set(full_labels) | set(omitted_labels), key=_natural_label_key
    )
    color_lookup = {
        region: plt.get_cmap("tab10")(index % 10)
        for index, region in enumerate(region_names)
    }
    for values, labels, marker, scenario in (
        (full_projected, full_labels, "o", "Full model"),
        (omitted_projected, omitted_labels, "^", "Omit configured control"),
    ):
        axis.scatter(
            values[:, 0],
            values[:, 1],
            c=[color_lookup[label] for label in labels],
            marker=marker,
            s=75,
            edgecolor="black",
            linewidth=0.5,
            label=scenario,
            zorder=2,
        )
    if region_names != ["unassigned"] and len(region_names) <= 10:
        for region in region_names:
            axis.scatter(
                [],
                [],
                color=color_lookup[region],
                marker="s",
                s=45,
                label=region,
            )
    axis.set(
        xlabel="Common PCA coordinate 1",
        ylabel="Common PCA coordinate 2",
        title="Full versus configured-control-omission candidate regions",
    )
    axis.legend(fontsize=8)
    return _save_debug_figure(fig, output_path)


# Backward-compatible callable name; plot text and semantics are control-generic.
plot_sample1_candidate_region_comparison = (
    plot_control_omission_candidate_region_comparison
)


def plot_candidate_predictions_vs_observed_ranges(
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    candidate_column: str = "candidate_id",
    model_column: str = "model_variant",
    objective_column: str = "objective_name",
    mean_column: str = "predicted_mean",
    std_column: str = "predicted_std",
    observed_min_column: str = "observed_minimum",
    observed_max_column: str = "observed_maximum",
) -> Path:
    """Show candidate posterior predictions relative to observed score ranges."""

    required = (
        candidate_column,
        model_column,
        objective_column,
        mean_column,
        std_column,
        observed_min_column,
        observed_max_column,
    )
    frame = _frame(predictions, name="predictions", required_columns=required)
    for column in (candidate_column, model_column, objective_column):
        _labels(frame, column, name="predictions")
    _numeric(
        frame,
        (mean_column, std_column, observed_min_column, observed_max_column),
        name="predictions",
    )
    if (frame[std_column] <= 0.0).any():
        raise ValueError("candidate predictive standard deviations must be positive.")
    if (frame[observed_min_column] > frame[observed_max_column]).any():
        raise ValueError("observed minimum must not exceed observed maximum.")
    identity_columns = [candidate_column, model_column, objective_column]
    if frame.duplicated(identity_columns).any():
        raise ValueError("candidate/model/objective prediction rows must be unique.")
    range_counts = frame.groupby(objective_column, sort=False)[
        [observed_min_column, observed_max_column]
    ].nunique(dropna=False)
    if (range_counts > 1).any().any():
        raise ValueError("each objective must use one consistent observed range.")

    objectives = sorted(frame[objective_column].unique(), key=_natural_label_key)
    maximum_rows = int(frame.groupby(objective_column).size().max())
    fig, axes_value = plt.subplots(
        1,
        len(objectives),
        figsize=(max(7.0, 5.3 * len(objectives)), max(4.8, 0.3 * maximum_rows + 2.5)),
        squeeze=False,
    )
    axes = axes_value.ravel()
    colors = plt.get_cmap("tab10")
    models = sorted(frame[model_column].unique(), key=_natural_label_key)
    model_colors = {model: colors(index % 10) for index, model in enumerate(models)}
    for objective_index, objective in enumerate(objectives):
        axis = axes[objective_index]
        group = frame[frame[objective_column] == objective].sort_values(
            [candidate_column, model_column],
            key=lambda values: values.map(_natural_label_key),
            kind="mergesort",
        )
        positions = np.arange(len(group), dtype=float)
        observed_min = float(group[observed_min_column].iloc[0])
        observed_max = float(group[observed_max_column].iloc[0])
        axis.axvspan(
            observed_min,
            observed_max,
            color="0.88",
            alpha=0.8,
            label="Observed range",
        )
        for model in models:
            mask = group[model_column] == model
            if mask.any():
                axis.errorbar(
                    group.loc[mask, mean_column],
                    positions[mask.to_numpy()],
                    xerr=group.loc[mask, std_column],
                    fmt="o",
                    capsize=3,
                    color=model_colors[model],
                    label=model,
                )
        labels = [
            f"{candidate} | {model}"
            for candidate, model in group[[candidate_column, model_column]].itertuples(
                index=False, name=None
            )
        ]
        axis.set_yticks(positions, labels, fontsize=7)
        axis.set(
            xlabel="Posterior mean ± one standard deviation",
            title=f"{objective}: prediction vs observed range",
        )
        axis.invert_yaxis()
    axes[0].legend(fontsize=7)
    return _save_debug_figure(fig, output_path)


def plot_shortlist_medoid_parallel_coordinates(
    shortlist: pd.DataFrame,
    output_path: str | Path,
    *,
    region_column: str = "region_id",
    persistence_column: str = "family_weighted_persistence",
    input_names: Sequence[str] | None = None,
) -> Path:
    """Plot normalized shortlist medoids as deterministic parallel coordinates."""

    frame = _frame(shortlist, name="shortlist", required_columns=(region_column,))
    _labels(frame, region_column, name="shortlist")
    if frame[region_column].duplicated().any():
        raise ValueError("shortlist region identifiers must be unique.")
    norm_columns = _normalized_columns(frame, prefix="medoid_norm_", name="shortlist")
    _numeric(frame, norm_columns, name="shortlist")
    normalized = frame[norm_columns]
    if ((normalized < 0.0) | (normalized > 1.0)).any().any():
        raise ValueError("shortlist medoid coordinates must remain in [0, 1].")
    if persistence_column not in frame.columns:
        raise ValueError(
            f"shortlist is missing required column: {persistence_column!r}."
        )
    _numeric(frame, (persistence_column,), name="shortlist")
    if ((frame[persistence_column] < 0.0) | (frame[persistence_column] > 1.0)).any():
        raise ValueError("shortlist persistence values must remain in [0, 1].")
    if input_names is None:
        names = [f"X{index + 1}" for index in range(len(norm_columns))]
    else:
        names = [str(value).strip() for value in input_names]
        if len(names) != len(norm_columns) or any(not value for value in names):
            raise ValueError(
                "input_names must contain one nonblank name per medoid dimension."
            )
    frame = frame.sort_values(
        region_column,
        key=lambda values: values.map(_natural_label_key),
        kind="mergesort",
    )
    positions = np.arange(len(norm_columns), dtype=float)
    fig, axis = plt.subplots(figsize=(max(8.5, 0.8 * len(norm_columns)), 5.2))
    colors = plt.get_cmap("tab20")
    for index, row in frame.reset_index(drop=True).iterrows():
        persistence = float(row[persistence_column])
        axis.plot(
            positions,
            row[norm_columns].to_numpy(dtype=float),
            marker="o",
            linewidth=1.0 + 2.0 * persistence,
            alpha=0.4 + 0.55 * persistence,
            color=colors(index % 20),
            label=str(row[region_column]),
        )
    axis.set_xticks(positions, names, rotation=35, ha="right")
    axis.set(
        ylim=(-0.03, 1.03),
        ylabel="Normalized medoid coordinate",
        title="Robust-shortlist medoid parallel coordinates",
    )
    if len(frame) <= 12:
        axis.legend(fontsize=7, ncol=2)
    return _save_debug_figure(fig, output_path)


def plot_run_region_persistence_heatmap(
    membership: pd.DataFrame,
    output_path: str | Path,
    *,
    run_column: str = "run_id",
    region_column: str = "region_id",
    family_column: str = "run_family",
    value_column: str | None = None,
) -> Path:
    """Plot run-by-region selection persistence as a deterministic heatmap."""

    required = [run_column, region_column, family_column]
    if value_column is not None:
        required.append(value_column)
    frame = _frame(membership, name="membership", required_columns=required)
    for column in (run_column, region_column, family_column):
        _labels(frame, column, name="membership")
    family_counts = frame.groupby(run_column, sort=False)[family_column].nunique()
    if (family_counts != 1).any():
        raise ValueError("each run must map to exactly one run family.")
    run_metadata = frame[[run_column, family_column]].drop_duplicates()
    run_metadata = run_metadata.sort_values(
        [family_column, run_column],
        key=lambda values: values.map(_natural_label_key),
        kind="mergesort",
    )
    runs = run_metadata[run_column].tolist()
    regions = sorted(frame[region_column].unique(), key=_natural_label_key)
    if value_column is None:
        values = frame[[run_column, region_column]].drop_duplicates().assign(_value=1.0)
        value_name = "_value"
        colorbar_label = "Region represented (0/1)"
    else:
        _numeric(frame, (value_column,), name="membership")
        if (frame[value_column] < 0.0).any():
            raise ValueError("persistence heatmap values must be non-negative.")
        values = frame[[run_column, region_column, value_column]].copy()
        value_name = value_column
        colorbar_label = value_column.replace("_", " ")
    pivot = values.pivot_table(
        index=run_column,
        columns=region_column,
        values=value_name,
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=runs, columns=regions, fill_value=0.0)
    family_lookup = dict(
        run_metadata[[run_column, family_column]].itertuples(index=False, name=None)
    )
    row_labels = [f"{family_lookup[run]} | {run}" for run in runs]
    return _heatmap(
        pivot.to_numpy(dtype=float),
        row_labels=row_labels,
        column_labels=regions,
        title="Core run-by-region persistence",
        colorbar_label=colorbar_label,
        output_path=output_path,
    )


def plot_model_policy_region_correspondence(
    correspondence: pd.DataFrame,
    output_path: str | Path,
    *,
    model_column: str = "model_variant",
    policy_column: str = "bound_policy",
    region_column: str = "region_id",
    value_column: str | None = None,
) -> Path:
    """Plot candidate-region correspondence by model and utility policy."""

    required = [model_column, policy_column, region_column]
    if value_column is not None:
        required.append(value_column)
    frame = _frame(correspondence, name="correspondence", required_columns=required)
    for column in (model_column, policy_column, region_column):
        _labels(frame, column, name="correspondence")
    frame["_model_policy"] = (
        frame[model_column].astype(str) + " | " + frame[policy_column].astype(str)
    )
    row_labels = sorted(frame["_model_policy"].unique(), key=_natural_label_key)
    regions = sorted(frame[region_column].unique(), key=_natural_label_key)
    if value_column is None:
        values = frame.assign(_value=1.0)
        value_name = "_value"
        colorbar_label = "Selected candidate count"
    else:
        _numeric(frame, (value_column,), name="correspondence")
        if (frame[value_column] < 0.0).any():
            raise ValueError("correspondence values must be non-negative.")
        values = frame
        value_name = value_column
        colorbar_label = value_column.replace("_", " ")
    pivot = values.pivot_table(
        index="_model_policy",
        columns=region_column,
        values=value_name,
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=row_labels, columns=regions, fill_value=0.0)
    return _heatmap(
        pivot.to_numpy(dtype=float),
        row_labels=row_labels,
        column_labels=regions,
        title="Model/policy candidate-region correspondence",
        colorbar_label=colorbar_label,
        output_path=output_path,
        cmap="Purples",
    )


def plot_acquisition_quality_vs_persistence(
    regions: pd.DataFrame,
    output_path: str | Path,
    *,
    region_column: str = "region_id",
    persistence_column: str = "family_weighted_persistence",
    acquisition_column: str = "median_acquisition_score",
    family_count_column: str = "distinct_family_count",
) -> Path:
    """Plot regional acquisition quality against family-weighted persistence."""

    required = (
        region_column,
        persistence_column,
        acquisition_column,
        family_count_column,
    )
    frame = _frame(regions, name="regions", required_columns=required)
    _labels(frame, region_column, name="regions")
    _numeric(
        frame,
        (persistence_column, acquisition_column, family_count_column),
        name="regions",
    )
    if frame[region_column].duplicated().any():
        raise ValueError("region identifiers must be unique.")
    if ((frame[persistence_column] < 0.0) | (frame[persistence_column] > 1.0)).any():
        raise ValueError("region persistence values must remain in [0, 1].")
    if (frame[acquisition_column] < 0.0).any():
        raise ValueError("regional acquisition quality must be non-negative.")
    if (frame[family_count_column] <= 0.0).any():
        raise ValueError("distinct family counts must be strictly positive.")
    frame = frame.sort_values(
        region_column,
        key=lambda values: values.map(_natural_label_key),
        kind="mergesort",
    )
    sizes = 40.0 + 35.0 * frame[family_count_column].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(7.5, 5.2))
    axis.scatter(
        frame[persistence_column],
        frame[acquisition_column],
        s=sizes,
        c=frame[family_count_column],
        cmap="viridis",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    for _, row in frame.iterrows():
        axis.annotate(
            str(row[region_column]),
            (
                float(row[persistence_column]),
                float(row[acquisition_column]),
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set(
        xlim=(-0.03, 1.03),
        xlabel="Family-weighted persistence",
        ylabel="Median base HVI",
        title="Acquisition quality versus region persistence",
    )
    return _save_debug_figure(fig, output_path)


def plot_shortlist_region_influence_sensitivity(
    sensitivity: pd.DataFrame,
    output_path: str | Path,
    *,
    region_column: str = "region_id",
    omitted_sample_column: str = "omitted_sample_id",
    sensitivity_column: str = "prediction_change",
) -> Path:
    """Plot omission sensitivity for each shortlisted robust region."""

    frame = _frame(
        sensitivity,
        name="sensitivity",
        required_columns=(region_column, omitted_sample_column, sensitivity_column),
    )
    _labels(frame, region_column, name="sensitivity")
    _labels(frame, omitted_sample_column, name="sensitivity")
    _numeric(frame, (sensitivity_column,), name="sensitivity")
    if (frame[sensitivity_column] < 0.0).any():
        raise ValueError("shortlist-region sensitivity must be non-negative.")
    regions = sorted(frame[region_column].unique(), key=_natural_label_key)
    samples = sorted(frame[omitted_sample_column].unique(), key=_natural_label_key)
    pivot = frame.pivot_table(
        index=region_column,
        columns=omitted_sample_column,
        values=sensitivity_column,
        aggfunc="mean",
    ).reindex(index=regions, columns=samples)
    if pivot.isna().any().any():
        raise ValueError(
            "sensitivity must cover every shortlist-region/omitted-sample pair."
        )
    return _heatmap(
        pivot.to_numpy(dtype=float),
        row_labels=regions,
        column_labels=[f"Omit {sample}" for sample in samples],
        title="Shortlist-region observation-influence sensitivity",
        colorbar_label=sensitivity_column.replace("_", " "),
        output_path=output_path,
        cmap="Oranges",
    )


__all__ = [
    "DEBUG_WATERMARK",
    "plot_acquisition_quality_vs_persistence",
    "plot_ard_lengthscale_comparison",
    "plot_boundary_enrichment",
    "plot_bounded_utility_comparison",
    "plot_candidate_predictions_vs_observed_ranges",
    "plot_control_omission_candidate_region_comparison",
    "plot_local_penalty_tradeoff",
    "plot_loocv_diagnostics",
    "plot_model_policy_region_correspondence",
    "plot_nested_search_convergence",
    "plot_observation_influence_ranking",
    "plot_robust_region_overview",
    "plot_run_region_persistence_heatmap",
    "plot_sample1_candidate_region_comparison",
    "plot_shortlist_medoid_parallel_coordinates",
    "plot_shortlist_region_influence_sensitivity",
]
