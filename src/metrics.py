# src/metrics.py
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

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
    
