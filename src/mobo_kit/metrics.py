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
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute the Pareto front and hypervolume of the training data.

    Args:
        Y: (N, M) array of objectives.
        ref_point_np: (M,) reference point. REQUIRED, and fixed for the campaign.

    Returns:
        ref_point: Reference point used for hypervolume.
        pareto_Y: Pareto front points.
        volume: Hypervolume of the Pareto front.

    Notes:
    * All objectives are to be MAXIMIZED. If any are minimized, flip sign before calling.
    * Returns tensors on the same device/dtype as `Y`.

    The reference point is required rather than inferred, for two reasons that
    both produced wrong numbers here.

    A previous version defaulted to ``Y.min(dim=0) - 1e-8``, essentially the nadir
    of whatever data it was handed. Every hypervolume slab is then 1e-8 thick: on
    the campaign's R0 utilities that gave **6e-8 against 1.448** from BoTorch's
    ``infer_reference_point`` on the same data. It also re-derived the reference
    from the current data on every call, so two rounds' hypervolumes were measured
    against two different reference points and were never comparable -- which is
    the whole purpose of tracking hypervolume across rounds.

    Pass the campaign's declared ``reference_point_utility`` from config, in
    utility space, after the objective transforms.
    """
    if ref_point_np is None:
        raise ValueError(
            "compute_ref_pareto_hv requires an explicit ref_point_np. Pass the "
            "campaign's fixed reference: "
            "np.asarray(config['reference_point_utility'], dtype=float), in utility "
            "space after the objective transforms. A reference inferred from the "
            "current data moves between rounds, which makes hypervolumes "
            "incomparable across them."
        )

    device, dtype = Y.device, Y.dtype
    N, M = Y.shape
    del N

    pareto_mask = is_non_dominated(Y)
    pareto_Y = Y[pareto_mask]

    if not isinstance(ref_point_np, (np.ndarray, torch.Tensor)):
        raise TypeError("ref_point_np must be a numpy.ndarray or torch.Tensor")
    reference = np.asarray(
        ref_point_np.detach().cpu().numpy()
        if isinstance(ref_point_np, torch.Tensor)
        else ref_point_np,
        dtype=float,
    )
    if reference.ndim != 1:
        raise ValueError(f"ref_point must be 1D of length {M}, got shape {reference.shape}")
    if reference.size != M:
        raise ValueError(f"ref_point length {reference.size} does not match number of objectives M={M}")
    if not np.all(np.isfinite(reference)):
        raise ValueError("ref_point must be finite.")
    ref_point_t = torch.as_tensor(reference, device=device, dtype=dtype)

    # BoTorch's Hypervolume assumes maximisation and SILENTLY DROPS points that do
    # not dominate the reference -- no warning, no exception, just a smaller number
    # or 0.0. If nothing dominates, the answer would be 0.0 and indistinguishable
    # from a wrongly signed or badly placed reference, so say so instead.
    dominating = bool((pareto_Y > ref_point_t).all(dim=-1).any())
    if not dominating:
        raise ValueError(
            "No observation dominates the reference point, so the hypervolume "
            "would be 0.0 for a reason the number cannot express. Check the sign "
            "convention (every objective must be maximised here) and that the "
            "reference sits below the achievable region. Reference: "
            f"{reference.tolist()}; per-objective observed maxima: "
            f"{Y.max(dim=0).values.detach().cpu().numpy().tolist()}."
        )

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
    
