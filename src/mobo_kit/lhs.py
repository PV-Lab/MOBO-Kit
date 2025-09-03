# src/lhs.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Callable, Tuple, Sequence, Union
from .constraints import apply_row_constraints, RowConstraint

from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs.latin_design import LatinDesign

from .design import Design
from .data import snap_to_grid_np  # snaps only dims with grids (design.var_list)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Emukit space & core sampling
# ----------------------------

def _space_from_design(design: Design) -> ParameterSpace:
    """
    Build an Emukit ParameterSpace using only ContinuousParameter.
    For grid dims, we use [min(grid), max(grid)] and snap after sampling.
    """
    params = []
    for j, name in enumerate(design.names):
        lo = float(design.lowers[j])
        hi = float(design.uppers[j])
        params.append(ContinuousParameter(name, lo, hi))
    return ParameterSpace(params)


# ----------------------------
# Dataset-level corr utilities
# ----------------------------

def _max_abs_corr(X: np.ndarray) -> float:
    """
    Maximum absolute Pearson correlation across columns of X.
    NaNs from zero-variance columns are treated as 0 correlation.
    """
    C = np.corrcoef(X, rowvar=False)  # DxD
    A = np.abs(C)
    np.fill_diagonal(A, 0.0)
    A = np.nan_to_num(A, nan=0.0, posinf=1.0, neginf=1.0)
    return float(np.max(A))


def _pick_subset_with_corr(
    X: np.ndarray,
    n: int,
    threshold: float,
    tries: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    Try random subsets of size n from X to find one with max|corr| <= threshold.
    Returns (best_subset, best_max_corr). If none meet threshold, returns best found.
    """
    m = X.shape[0]
    if m == n:
        return X, _max_abs_corr(X)

    best_X = None
    best_val = float("inf")
    for _ in range(max(1, tries)):
        idx = rng.choice(m, size=n, replace=False)
        X_sub = X[idx]
        val = _max_abs_corr(X_sub)
        if val < best_val:
            best_val, best_X = val, X_sub
            if val <= threshold:
                break
    return best_X, best_val

def _lhs_select_numeric(df: pd.DataFrame, design: Design) -> pd.DataFrame:
    cols = [c for c in design.names if c in df.columns]
    return df[cols].select_dtypes(include=[np.number]).copy()

def _lhs_labels(design: Design, cols) -> list:
    name_to_label = {n: l for n, l in zip(design.names, design.labels)}
    return [name_to_label.get(c, c) for c in cols]


# ----------------------------
# OPTIMIZED LHS IMPLEMENTATION
# ----------------------------

def _batch_generate_lhs_samples(
    design: Design, 
    total_samples: int, 
    batch_size: int = 100,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate LHS samples in batches for better memory management.
    """
    if seed is not None:
        np.random.seed(seed)
    
    space = _space_from_design(design)
    all_samples = []
    
    for i in range(0, total_samples, batch_size):
        current_batch = min(batch_size, total_samples - i)
        batch_samples = LatinDesign(space).get_samples(current_batch)
        all_samples.append(batch_samples)
    
    return np.vstack(all_samples)


def _batch_apply_constraints(
    X_samples: np.ndarray, 
    design: Design, 
    constraints: Sequence[RowConstraint],
    batch_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply constraints in batches to avoid memory issues with large sample sets.
    Returns (valid_samples, valid_indices).
    """
    if not constraints:
        return X_samples, np.arange(len(X_samples))
    
    valid_samples = []
    valid_indices = []
    
    for i in range(0, len(X_samples), batch_size):
        batch_end = min(i + batch_size, len(X_samples))
        X_batch = X_samples[i:batch_end]
        
        # Apply constraints to this batch
        mask = apply_row_constraints(X_batch, design, constraints)
        valid_batch = X_batch[mask]
        valid_indices.extend(np.arange(i, batch_end)[mask])
        
        if len(valid_batch) > 0:
            valid_samples.append(valid_batch)
    
    if valid_samples:
        return np.vstack(valid_samples), np.array(valid_indices)
    else:
        return np.empty((0, X_samples.shape[1])), np.array([], dtype=int)


def _smart_subset_selection(
    X_valid: np.ndarray,
    n_target: int,
    max_abs_corr: float,
    max_tries: int = 1000,
    early_stop_threshold: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float, int]:
    """
    Smart subset selection with early stopping and adaptive search.
    Returns (best_subset, best_correlation, tries_used).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    m = X_valid.shape[0]
    if m <= n_target:
        return X_valid, _max_abs_corr(X_valid), 0
    
    best_X = None
    best_corr = float("inf")
    tries_used = 0
    
    # Phase 1: Quick random sampling
    quick_tries = min(max_tries // 2, 100)
    for _ in range(quick_tries):
        idx = rng.choice(m, size=n_target, replace=False)
        X_sub = X_valid[idx]
        corr = _max_abs_corr(X_sub)
        
        if corr < best_corr:
            best_corr = corr
            best_X = X_sub.copy()
            tries_used += 1
            
            # Early stopping if we're close to target
            if corr <= max_abs_corr + early_stop_threshold:
                break
    
    # Phase 2: Targeted improvement if needed
    if best_corr > max_abs_corr:
        remaining_tries = max_tries - tries_used
        for _ in range(remaining_tries):
            idx = rng.choice(m, size=n_target, replace=False)
            X_sub = X_valid[idx]
            corr = _max_abs_corr(X_sub)
            
            if corr < best_corr:
                best_corr = corr
                best_X = X_sub.copy()
                tries_used += 1
                
                if corr <= max_abs_corr:
                    break
    
    return best_X, best_corr, tries_used


def lhs_dataframe_optimized(
    design: Design,
    n: int,
    seed: Optional[int] = None,
    snap_to_grids: bool = True,
    row_constraints: Optional[Union[RowConstraint, Sequence[RowConstraint]]] = None,
    max_abs_corr: Optional[float] = None,
    # Performance tuning:
    max_attempts: int = 100,
    oversample: int = 5,  # Increased for better constraint satisfaction
    batch_size: int = 100,
    subset_tries: int = 1000,
    early_stop_threshold: float = 0.1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    OPTIMIZED LHS generation with batch processing, smart subset selection, and early stopping.
    
    Key optimizations:
    1. Batch LHS generation to manage memory
    2. Batch constraint application
    3. Smart subset selection with early stopping
    4. Adaptive oversampling based on constraint strictness
    """
    if seed is not None:
        np.random.seed(int(seed))
    
    # Normalize constraints
    if row_constraints is None:
        constraints_list = []
    elif callable(row_constraints):
        constraints_list = [row_constraints]
    else:
        constraints_list = list(row_constraints)
    
    # Adjust oversampling based on constraint complexity
    if constraints_list:
        effective_oversample = max(oversample, len(constraints_list) * 2)
    else:
        effective_oversample = 1
    
    total_samples_needed = n * effective_oversample
    
    if verbose:
        print(f"[OPTIMIZED LHS] Target: {n} samples, Generating: {total_samples_needed} samples")
    
    best_candidate = None
    best_corr = float("inf")
    
    for attempt in range(1, max_attempts + 1):
        if verbose and attempt % 10 == 0:
            print(f"[OPTIMIZED LHS] Attempt {attempt}/{max_attempts}")
        
        # Generate LHS samples in batches
        X_raw = _batch_generate_lhs_samples(
            design, total_samples_needed, batch_size, seed
        )
        
        # Snap to grids if requested
        if snap_to_grids:
            X_raw = snap_to_grid_np(X_raw, design)
        
        # Apply constraints in batches
        X_valid, valid_indices = _batch_apply_constraints(
            X_raw, design, constraints_list, batch_size
        )
        
        if len(X_valid) < n:
            if verbose:
                print(f"[OPTIMIZED LHS] Attempt {attempt}: Only {len(X_valid)} valid samples")
            continue
        
        # Smart subset selection
        X_best, corr_val, tries_used = _smart_subset_selection(
            X_valid, n, max_abs_corr or 1.0, subset_tries, early_stop_threshold
        )
        
        if verbose:
            print(f"[OPTIMIZED LHS] Attempt {attempt}: max|corr|={corr_val:.3f} (tries: {tries_used})")
        
        # Check if we met the correlation target
        if max_abs_corr is None or corr_val <= max_abs_corr:
            if verbose:
                print(f"[OPTIMIZED LHS] Success! max|corr|={corr_val:.3f} <= {max_abs_corr}")
            return pd.DataFrame(X_best, columns=design.names)
        
        # Keep track of best candidate
        if corr_val < best_corr:
            best_corr = corr_val
            best_candidate = X_best.copy()
    
    # Return best candidate if we didn't meet the target
    if best_candidate is not None:
        if verbose:
            print(f"[OPTIMIZED LHS] Best achieved: max|corr|={best_corr:.3f} (> target)")
        return pd.DataFrame(best_candidate, columns=design.names)
    
    # Fallback: return whatever we have
    if verbose:
        print(f"[OPTIMIZED LHS] Failed to generate {n} samples after {max_attempts} attempts")
    
    # Try one more time with minimal constraints
    X_fallback = _batch_generate_lhs_samples(design, n, batch_size, seed)
    if snap_to_grids:
        X_fallback = snap_to_grid_np(X_fallback, design)
    
    return pd.DataFrame(X_fallback, columns=design.names)


# ----------------------------
# Original LHS (kept for backward compatibility)
# ----------------------------

def lhs_dataframe(
    design: Design,
    n: int,
    seed: Optional[int] = None,
    snap_to_grids: bool = True,
    # Constraints:
    #row_constraint_fn: Optional[Callable[[np.ndarray, Design], np.ndarray]] = None,
    row_constraints: Optional[Union[RowConstraint, Sequence[RowConstraint]]] = None,
    max_abs_corr: Optional[float] = None,
    # Sampling controls:
    max_attempts: int = 100,
    oversample: int = 3,
    subset_tries: int = 800,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate an LHS using Emukit, then enforce:
      - pointwise constraints via `row_constraints -> bool mask`
      - dataset-level Pearson correlation bound via `max_abs_corr`

    Strategy:
      1) Sample in continuous box via Emukit.
      2) Snap grid features (if requested).
      3) Apply row-wise mask (e.g., Clausius–Clapeyron); accumulate valid rows.
      4) If we have >= n rows, optionally pick a subset with max|corr| <= threshold.

    Returns DataFrame with columns == design.names (physical units).
    """
    # Seed Emukit's internal RNG
    if seed is not None:
        np.random.seed(int(seed))
    rng = np.random.default_rng(seed)

    space = _space_from_design(design)

    # Normalize row constraints to a list (None -> empty list)
    if row_constraints is None:
        row_constraints_list: Sequence[RowConstraint] = []
    elif callable(row_constraints):
        row_constraints_list = [row_constraints]  # backward-compatible single fn
    else:
        row_constraints_list = list(row_constraints)

    collected = []
    best_candidate = None
    best_corr_val = float("inf")

    for attempt in range(1, max_attempts + 1):
        # Heuristic: oversample to survive row constraints
        #batch = n if row_constraint_fn is None else max(n, oversample * n)
        batch = n if not row_constraints_list else max(n, oversample * n)
        
        X = LatinDesign(space).get_samples(batch)  # (batch, D)
        if snap_to_grids:
            X = snap_to_grid_np(X, design)

        # if row_constraint_fn is not None:
        #     mask = row_constraint_fn(X, design)
        #     if mask is None or mask.dtype != bool or mask.shape[0] != X.shape[0]:
        #         raise ValueError("row_constraint_fn must return a boolean mask of shape (batch,).")
        #     X = X[mask]

        if row_constraints_list:
            mask = apply_row_constraints(X, design, row_constraints_list)
            X = X[mask]

        if X.size == 0:
            if verbose:
                print(f"[LHS] attempt {attempt}: 0 valid after row constraints; retrying.")
            continue

        collected.append(X)
        X_all = np.vstack(collected)

        if X_all.shape[0] >= n:
            # If no correlation constraint, take first n (shuffled)
            idx = rng.permutation(X_all.shape[0])
            X_all = X_all[idx]

            if max_abs_corr is None:
                return pd.DataFrame(X_all[:n], columns=design.names)

            # Try to find subset meeting correlation threshold
            X_best, corr_val = _pick_subset_with_corr(X_all, n, max_abs_corr, subset_tries, rng)
            if verbose:
                print(f"[LHS] attempt {attempt}: candidate max|corr|={corr_val:.3f}")
            if corr_val <= max_abs_corr:
                return pd.DataFrame(X_best, columns=design.names)

            # Keep best-so-far
            if corr_val < best_corr_val:
                best_corr_val = corr_val
                best_candidate = X_best

    # If we got here, we didn't meet correlation threshold; return best we found
    if best_candidate is None:
        # Maybe we never amassed n rows; return whatever we have (possibly < n)
        X_all = np.vstack(collected) if collected else np.empty((0, len(design.names)))
        X_all = X_all[:n]
        return pd.DataFrame(X_all, columns=design.names)

    if verbose:
        print(f"[LHS] giving best candidate with max|corr|={best_corr_val:.3f} (> target).")
    return pd.DataFrame(best_candidate, columns=design.names)

