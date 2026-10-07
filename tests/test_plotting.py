"""Headless smoke tests for campaign diagnostic plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mobo_kit.plotting import (
    plot_PCA,
    plot_correlation_heatmap,
    plot_distribution,
    plot_parity_np,
)


def _feature_frame():
    return pd.DataFrame(
        {
            "temperature": [80.0, 85.0, 90.0, 95.0, 100.0, 105.0],
            "speed": [1.0, 1.5, 1.2, 2.0, 1.8, 2.4],
            "ratio": [0.1, 0.3, 0.2, 0.6, 0.7, 0.9],
            "sample": ["A", "B", "C", "D", "E", "F"],
        }
    )


def test_parity_plot_returns_metrics_and_saves_png(tmp_path):
    true_y = np.array([[0.2, 1.0], [0.4, 1.5], [0.7, 2.0], [0.9, 2.5]], dtype=float)
    pred_y = true_y + np.array(
        [[0.02, -0.10], [-0.03, 0.05], [0.01, 0.08], [0.04, -0.04]]
    )
    pred_std = np.full_like(true_y, 0.05)
    output_path = tmp_path / "parity.png"

    fig, metrics = plot_parity_np(
        true_y,
        pred_y,
        pred_std=pred_std,
        objective_names=["efficiency", "stability"],
        save=str(output_path),
        show_plot=False,
    )

    assert output_path.is_file()
    assert len(fig.axes) == 2
    assert list(metrics.columns) == ["Objective", "R2", "RMSE"]
    assert metrics["Objective"].tolist() == ["efficiency", "stability"]
    assert np.isfinite(metrics[["R2", "RMSE"]].to_numpy()).all()
    plt.close(fig)


def test_tabular_diagnostic_plots_run_headlessly_and_preserve_shapes(tmp_path):
    frame = _feature_frame()
    numeric_columns = ["temperature", "speed", "ratio"]

    corr_fig, corr = plot_correlation_heatmap(
        frame,
        columns=numeric_columns,
        save=str(tmp_path / "correlation.png"),
        show_plot=False,
    )
    distribution_fig = plot_distribution(
        frame,
        columns=numeric_columns,
        n_cols=2,
        save=str(tmp_path / "distributions.png"),
        show_plot=False,
    )
    pca_fig, transformed, pca = plot_PCA(
        frame,
        columns=numeric_columns,
        n_components=2,
        save=str(tmp_path / "pca.png"),
        show_plot=False,
    )

    assert corr.shape == (3, 3)
    assert transformed.shape == (len(frame), 2)
    assert pca.n_components_ == 2
    assert (tmp_path / "correlation.png").is_file()
    assert (tmp_path / "distributions.png").is_file()
    assert (tmp_path / "pca.png").is_file()

    for fig in (corr_fig, distribution_fig, pca_fig):
        assert fig.axes
        plt.close(fig)


def test_plotting_helpers_reject_missing_numeric_data():
    frame = pd.DataFrame({"sample": ["A", "B"]})

    with pytest.raises(ValueError, match="No numeric columns"):
        plot_correlation_heatmap(frame, show_plot=False)
