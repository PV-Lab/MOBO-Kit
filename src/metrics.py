# src/metrics.py
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.spatial.distance import pdist
from .utils import np_to_torch


def compute_ref_pareto_hv(
    Y: torch.Tensor,
    ref_point_np: Optional[np.ndarray] = None,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute the Pareto front and hypervolume of the training data.

    Args:
        Y: (N, M) array of objectives.
        ref_point: (M,) array reference point for hypervolume calculation.
        eps: small margin to ensure auto reference point is strictly dominated.
    
    Returns:
        ref_point: Reference point used for hypervolume.
        pareto_Y: Pareto front points.
        volume: Hypervolume of the Pareto front.

    Notes:
    * All objectives are to be MAXIMIZED. If any are minimized, flip sign before calling.
    * Returns tensors on the same device/dtype as `Y`.
    """
    device, dtype = Y.device, Y.dtype
    N, M = Y.shape

    pareto_mask = is_non_dominated(Y)
    pareto_Y = Y[pareto_mask]

    if ref_point_np is None:
        mins = torch.min(Y, dim=0).values
        ref_point_t = mins - eps
    else:
        if not isinstance(ref_point_np, np.ndarray):
            raise TypeError("ref_point_np must be a numpy.ndarray")
        if ref_point_np.ndim != 1:
            raise ValueError(f"ref_point must be 1D of length {M}, got shape {ref_point_np.shape}")
        if ref_point_np.size != M:
            raise ValueError(f"ref_point length {ref_point_np.size} does not match number of objectives M={M}")
        ref_point_t = torch.as_tensor(ref_point_np, device=device, dtype=dtype)

    hv = Hypervolume(ref_point=ref_point_t)
    volume = float(hv.compute(pareto_Y))

    return ref_point_t, pareto_Y, volume


def compute_metrics(
    true_Y: np.ndarray,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    objective_names: list[str] | None = None,
    add_residuals: bool = False,
    add_zscores: bool = False,
) -> pd.DataFrame:
    
    # Convert to numpy arrays if they aren't already
    true_Y = np.asarray(true_Y, dtype=float)
    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_std = np.asarray(pred_std, dtype=float)
    
    rows = []
    N, M = true_Y.shape
    names = objective_names or [f"obj{j}" for j in range(M)]
    
    for j, name in enumerate(names):
        r2 = r2_score(true_Y[:, j], pred_mean[:, j])
        rmse = root_mean_squared_error(true_Y[:, j], pred_mean[:, j])
        
        row = {"Objective": name, "R2": round(float(r2), 3), "RMSE": round(float(rmse), 3)}
        
        if add_residuals:
            residual = pred_mean[:, j] - true_Y[:, j]
            row["Residual"] = residual.tolist()  # Store as list since it's an array
            
        if add_zscores:
            if not add_residuals:
                residual = pred_mean[:, j] - true_Y[:, j]  # Calculate if not already done
            # guard against very small std
            denom = np.where(pred_std[:, j] > 1e-12, pred_std[:, j], np.nan)
            zscore = residual / denom
            row["Z-score"] = zscore.tolist()  # Store as list since it's an array
            
        rows.append(row)
    # Build column list dynamically based on what was actually added
    base_columns = ["Objective", "R2", "RMSE"]
    if add_residuals:
        base_columns.append("Residual")
    if add_zscores:
        base_columns.append("Z-score")
    
    metrics_df = pd.DataFrame(rows)
    return metrics_df

def compute_diversity_score(X: np.ndarray) -> float:
    """
    Computes the average pairwise Euclidean distance between rows in X.
    A higher value indicates greater diversity among candidate points.
    
    Args:
        X (np.ndarray): 2D array of shape (n_points, n_features)

    Returns:
        float: average pairwise distance
    """
    if len(X) < 2:
        return 0.0  # Not enough points to compute diversity
    distances = pdist(X, metric='euclidean')  # All pairwise distances
    return distances.mean()
    
