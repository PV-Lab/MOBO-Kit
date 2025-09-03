# src/data.py
import numpy as np
import torch
from .design import Design
from typing import Optional, Tuple

def x_normalizer_np(X, space: Design):
    """
    Normalize input features X to [0,1] range using the design space bounds.
    
    Args:
        X: Input features array of any shape that can be converted to numpy
        space: Design object containing variable bounds in space.var_array
        
    Returns:
        X_normalized: X normalized to [0,1] range per feature
    """
    X = np.asarray(X, dtype=float)
    mins = np.array([arr.min() for arr in space.var_array], dtype=float)
    maxs = np.array([arr.max() for arr in space.var_array], dtype=float)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    return (X - mins) / denom

def x_denormalizer_np(X_norm, space: Design):
    """
    Reverse normalization: convert X from [0,1] range back to original scale.
    
    Args:
        X_norm: Normalized features array in [0,1] range
        space: Design object containing variable bounds in space.var_array
        
    Returns:
        X: Features in original scale
    """
    X_norm = np.asarray(X_norm, dtype=float)
    mins = np.array([arr.min() for arr in space.var_array], dtype=float)
    maxs = np.array([arr.max() for arr in space.var_array], dtype=float)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    return X_norm * denom + mins

def x_normalizer_torch(X: torch.Tensor, space: Design):
    """
    Normalize input features X to [0,1] range using PyTorch tensors.
    
    Args:
        X: Input features tensor of any shape
        space: Design object containing variable bounds in space.var_array
        
    Returns:
        X_normalized: X normalized to [0,1] range per feature, on same device as input
    """
    device = X.device
    dtype = torch.double if not X.dtype.is_floating_point else X.dtype
    mins = torch.tensor([float(a.min()) for a in space.var_array], dtype=dtype, device=device)
    maxs = torch.tensor([float(a.max()) for a in space.var_array], dtype=dtype, device=device)
    denom = torch.clamp(maxs - mins, min=1e-12)
    return (X.to(dtype) - mins) / denom

def x_denormalizer_torch(X_norm: torch.Tensor, space: Design):
    """
    Reverse normalization: convert X from [0,1] range back to original scale using PyTorch.
    
    Args:
        X_norm: Normalized features tensor in [0,1] range
        space: Design object containing variable bounds in space.var_array
        
    Returns:
        X: Features in original scale, on same device as input
    """
    device = X_norm.device
    dtype = torch.double if not X_norm.dtype.is_floating_point else X_norm.dtype
    mins = torch.tensor([float(a.min()) for a in space.var_array], dtype=dtype, device=device)
    maxs = torch.tensor([float(a.max()) for a in space.var_array], dtype=dtype, device=device)
    denom = torch.clamp(maxs - mins, min=1e-12)
    return X_norm.to(dtype) * denom + mins

def snap_to_grid_np(X, space: Design):
    """
    Snap continuous values to the nearest discrete grid values for discrete features.
    
    Args:
        X: Input features array of shape (N, D) where N is number of samples, D is number of features
        space: Design object containing variable definitions in space.var_list
        
    Returns:
        X_snapped: X with discrete features snapped to nearest grid values
        
    Notes:
        - Continuous features (grid=None) are left unchanged
        - Discrete features are snapped to the nearest valid grid value
        - Grid values in space.var_list must be sorted and unique
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
    """
    Snap continuous values to the nearest discrete grid values for discrete features using PyTorch.
    
    Args:
        X: Input features tensor of shape (N, D) where N is number of samples, D is number of features
        space: Design object containing variable definitions in space.var_list
        
    Returns:
        X_snapped: X with discrete features snapped to nearest grid values, on same device as input
        
    Notes:
        - Continuous features (grid=None) are left unchanged
        - Discrete features are snapped to the nearest valid grid value
        - Grid values in space.var_list must be sorted and unique
    """
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
    Column-wise min-max scaling to [0,1] range for each objective.
    
    Args:
        Y: Input array of shape (N, M) where N is number of samples, M is number of objectives
        eps: Small value to prevent division by zero when range is very small
        
    Returns:
        Y_scaled: Y scaled to [0,1] range per objective
        Y_min: Minimum values for each objective (shape M,)
        Y_max: Maximum values for each objective (shape M,)
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
    """
    Reverse min-max scaling: convert Y from [0,1] range back to original scale.
    
    Args:
        Y_scaled: Scaled array in [0,1] range
        Y_min: Minimum values for each objective (shape M,)
        Y_max: Maximum values for each objective (shape M,)
        
    Returns:
        Y: Array in original scale
    """
    Y_scaled = np.asarray(Y_scaled, dtype=float)
    Y_min = np.asarray(Y_min, dtype=float)
    Y_max = np.asarray(Y_max, dtype=float)
    return Y_scaled * (Y_max - Y_min) + Y_min

def y_standardize_np(Y: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Column-wise standardization to mean=0, std=1. Returns (Y_standardized, Y_mean, Y_std).
    
    Args:
        Y: Input array of shape (N, M) where N is number of samples, M is number of objectives
        eps: Small value to prevent division by zero when standard deviation is very small
        
    Returns:
        Y_standardized: Y standardized to have mean=0, std=1 per column
        Y_mean: Mean values for each column (shape M,)
        Y_std: Standard deviation values for each column (shape M,)
    """
    Y = np.asarray(Y, dtype=float)
    Y_mean = np.mean(Y, axis=0)
    Y_std = np.std(Y, axis=0)
    # Prevent division by zero
    Y_std = np.maximum(Y_std, eps)
    return (Y - Y_mean) / Y_std, Y_mean, Y_std

def y_destandardize_np(
    Y_standardized: np.ndarray,
    Y_mean: np.ndarray,
    Y_std: np.ndarray,
) -> np.ndarray:
    """
    Reverse standardization: convert back from mean=0, std=1 to original scale.
    
    Args:
        Y_standardized: Standardized array of shape (N, M)
        Y_mean: Mean values for each column (shape M,)
        Y_std: Standard deviation values for each column (shape M,)
        
    Returns:
        Y: Array in original scale
    """
    Y_standardized = np.asarray(Y_standardized, dtype=float)
    Y_mean = np.asarray(Y_mean, dtype=float)
    Y_std = np.asarray(Y_std, dtype=float)
    return Y_standardized * Y_std + Y_mean

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
