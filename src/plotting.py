# src/plotting.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, root_mean_squared_error

def plot_parity_np(
    true_Y: np.ndarray,                 # (N, M) in ORIGINAL units
    pred_mean: np.ndarray,              # (N, M) in ORIGINAL units
    pred_std: Optional[np.ndarray] = None,  # (N, M) in ORIGINAL units (optional)
    objective_names: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,   # default: (5*M, 5)
    equal_axes: bool = True,
    save: Optional[str] = None,
):
    """
    Parity plots (Predicted vs True) with optional vertical error bars (predictive std).
    Returns (fig, metrics_df) where metrics_df has R² and RMSE per objective.
    """
    # Basic shape checks
    true = np.asarray(true_Y, dtype=float)
    pred = np.asarray(pred_mean, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: true {true.shape} vs pred {pred.shape}")
    if pred_std is not None:
        std = np.asarray(pred_std, dtype=float)
        if std.shape != true.shape:
            raise ValueError(f"Shape mismatch: std {std.shape} vs true {true.shape}")
    else:
        std = None

    N, M = true.shape
    names = list(objective_names) if objective_names is not None else [f"obj{j}" for j in range(M)]
    if len(names) != M:
        raise ValueError(f"objective_names length {len(names)} != M={M}")

    if figsize is None:
        figsize = (5.0 * M, 5.0)

    fig, axes = plt.subplots(1, M, figsize=figsize, squeeze=False)
    axes = axes[0]

    rows = []
    for j, name in enumerate(names):
        ax = axes[j]
        if std is not None:
            ax.errorbar(
                true[:, j], pred[:, j],
                yerr=std[:, j],
                fmt="o", color='blue', ecolor="gray", elinewidth=1, capsize=3, alpha=0.75,
            )
        else:
            sns.scatterplot(x=true[:, j], y=pred[:, j], s=35, ax=ax)

        # Parity line and axis handling
        lo = float(min(true[:, j].min(), pred[:, j].min()))
        hi = float(max(true[:, j].max(), pred[:, j].max()))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        if equal_axes:
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)

        # Metrics (original units)
        r2 = r2_score(true[:, j], pred[:, j])
        rmse = root_mean_squared_error(true[:, j], pred[:, j])
        ax.text(
            0.05, 0.95, f"R² = {r2:.2f}\nRMSE = {rmse:.2f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        rows.append({"Objective": name, "R2": round(float(r2), 3), "RMSE": round(float(rmse), 3)})

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig, pd.DataFrame(rows, columns=["Objective", "R2", "RMSE"])
