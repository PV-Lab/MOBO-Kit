# src/data.py
import numpy as np
import torch
from .design import Design
from typing import Optional, Tuple

def x_normalizer_np(X, space: Design):
    X = np.asarray(X, dtype=float)
    mins = np.array([arr.min() for arr in space.var_array], dtype=float)
    maxs = np.array([arr.max() for arr in space.var_array], dtype=float)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    return (X - mins) / denom

def x_denormalizer_np(X_norm, space: Design):
    X_norm = np.asarray(X_norm, dtype=float)
    mins = np.array([arr.min() for arr in space.var_array], dtype=float)
    maxs = np.array([arr.max() for arr in space.var_array], dtype=float)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    return X_norm * denom + mins

def x_normalizer_torch(X: torch.Tensor, space: Design):
    device = X.device
    dtype = torch.double if not X.dtype.is_floating_point else X.dtype
    mins = torch.tensor([float(a.min()) for a in space.var_array], dtype=dtype, device=device)
    maxs = torch.tensor([float(a.max()) for a in space.var_array], dtype=dtype, device=device)
    denom = torch.clamp(maxs - mins, min=1e-12)
    return (X.to(dtype) - mins) / denom

def x_denormalizer_torch(X_norm: torch.Tensor, space: Design):
    device = X_norm.device
    dtype = torch.double if not X_norm.dtype.is_floating_point else X_norm.dtype
    mins = torch.tensor([float(a.min()) for a in space.var_array], dtype=dtype, device=device)
    maxs = torch.tensor([float(a.max()) for a in space.var_array], dtype=dtype, device=device)
    denom = torch.clamp(maxs - mins, min=1e-12)
    return X_norm.to(dtype) * denom + mins

def snap_to_grid_np(X, space: Design):
    """
    For each feature with a discrete grid in space.var_list[j], snap to nearest grid value.
    Continuous features (grid=None) are left unchanged.
    """
    X = np.asarray(X, dtype=float)
    out = X.copy()
    for j, grid in enumerate(space.var_list):
        if grid is None:   # continuous; do nothing
            continue
        # grid must be sorted unique 1D
        g = np.asarray(grid, dtype=float)
        idx = np.searchsorted(g, out[:, j], side='left')
        idx = np.clip(idx, 1, g.size - 1)
        left = g[idx - 1]
        right = g[idx]
        choose_right = (out[:, j] - left) > (right - out[:, j])
        out[:, j] = np.where(choose_right, right, left)
    return out

def snap_to_grid_torch(X: torch.Tensor, space: Design):
    device = X.device
    dtype = X.dtype if X.dtype.is_floating_point else torch.double
    out = X.to(dtype).clone()
    for j, grid in enumerate(space.var_list):
        if grid is None:
            continue
        g = torch.tensor(np.array(grid, dtype=float), dtype=dtype, device=device)
        diffs = torch.abs(out[:, j:j+1] - g.unsqueeze(0))  # (N,G)
        idx = torch.argmin(diffs, dim=1)
        out[:, j] = g[idx]
    return out

def y_minmax_np(Y: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Column-wise min–max scale to [0,1]. Returns (Y_scaled, Y_min, Y_max).
    """
    Y_min = np.min(Y, axis=0)
    Y_max = np.max(Y, axis=0)
    denom = np.maximum(Y_max - Y_min, eps)
    return (Y - Y_min) / denom, Y_min, Y_max

def y_minmax_denorm_np(
    Y_scaled: np.ndarray,
    Y_min: np.ndarray,
    Y_max: np.ndarray,
) -> np.ndarray:
    
    Y_scaled = np.asarray(Y_scaled, dtype=float)
    Y_min = np.asarray(Y_min, dtype=float)
    Y_max = np.asarray(Y_max, dtype=float)
    return Y_scaled * (Y_max - Y_min) + Y_min

# def to_torch_pair(
#     X: np.ndarray,
#     Y: np.ndarray,
#     device: Optional[torch.device] = None,
#     dtype: torch.dtype = torch.float64,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.device]:
#     """
#     Convert numpy arrays to torch tensors on the chosen device.
#     """
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     return torch.tensor(X, dtype=dtype, device=device), torch.tensor(Y, dtype=dtype, device=device), device
