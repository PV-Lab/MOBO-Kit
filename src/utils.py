# src/utils.py
from __future__ import annotations
from typing import List, Tuple, Any
import pandas as pd
import numpy as np
import torch
import yaml

from .design import Design

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "inputs" not in cfg:
        raise ValueError("Config missing required key: 'inputs'")
    return cfg

def get_objective_names(cfg: dict) -> List[str]:
    names = cfg.get("objectives", {}).get("names")
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("Config must have objectives.names as a non-empty list.")
    return list(names)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_XY_from_cfg(df: pd.DataFrame, design: Design, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    x_cols = list(design.names)
    y_cols = get_objective_names(cfg)

    miss_x = [c for c in x_cols if c not in df.columns]
    miss_y = [c for c in y_cols if c not in df.columns]
    if miss_x or miss_y:
        parts = []
        if miss_x: parts.append(f"missing inputs: {miss_x}")
        if miss_y: parts.append(f"missing objectives: {miss_y}")
        raise KeyError("CSV column check failed: " + "; ".join(parts))

    return df[x_cols].to_numpy(), df[y_cols].to_numpy()

def select_device(prefer: str = "cuda") -> torch.device:
    return torch.device("cuda" if prefer == "cuda" and torch.cuda.is_available() else "cpu")

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def np_to_torch(
    *arrays: np.ndarray,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    return_device: bool = False,
):
    """
    Convert one or more NumPy arrays to torch tensors on the chosen device.

    Usage:
        X_t = np_to_torch(X)                                       # one array -> one tensor
        X_t, Y_t = np_to_torch(X, Y)                               # many arrays -> many tensors
        (X_t, Y_t), dev = np_to_torch(X, Y, return_device=True)    # also get the device used
    """
    if device is None:
        device = select_device("cuda")  # uses your existing helper
    tensors = tuple(torch.as_tensor(a, dtype=dtype, device=device) for a in arrays)
    out = tensors[0] if len(tensors) == 1 else tensors
    return (out, device) if return_device else out


def torch_to_np(*tensors: torch.Tensor):
    """
    Convert one or more torch tensors to NumPy arrays (detached, moved to CPU).

    Usage:
        X_np = torch_to_np(X_t)
        X_np, Y_np = torch_to_np(X_t, Y_t)
    """
    arrays = tuple(t.detach().cpu().numpy() for t in tensors)
    return arrays[0] if len(arrays) == 1 else arrays