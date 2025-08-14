# src/acquisition.py
from __future__ import annotations

from typing import Callable, Optional, Dict, Any, List, Union, Sequence, Tuple

import numpy as np
import torch
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
    MCMultiOutputObjective,
)
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf

from .design import Design
from .data import x_normalizer_torch, x_denormalizer_np, snap_to_grid_torch, x_denormalizer_torch
from .constraints import apply_row_constraints, RowConstraint


# ---------------------------
# Helpers
# ---------------------------

def _unit_bounds(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Return a (2, d) tensor of bounds for the unit hypercube [0, 1]^d
    on the specified device/dtype.
    """
    lb = torch.zeros(d, device=device, dtype=dtype)
    ub = torch.ones(d, device=device, dtype=dtype)
    return torch.stack([lb, ub], dim=0)


# Outcome-space constraint builders (negative => feasible)
def outcome_ge(obj_idx: int, thresh: float):
    """Feasible when Y[..., obj_idx] >= thresh."""
    def _c(Y: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(thresh, dtype=Y.dtype, device=Y.device) - Y[..., obj_idx]
    return _c


def outcome_le(obj_idx: int, thresh: float):
    """Feasible when Y[..., obj_idx] <= thresh."""
    def _c(Y: torch.Tensor) -> torch.Tensor:
        return Y[..., obj_idx] - torch.as_tensor(thresh, dtype=Y.dtype, device=Y.device)
    return _c

def _make_snap_postproc(design: Design) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Factory for a post-processing function to pass into optimize_acqf.

    It maps normalized candidates Z in [0,1]^D to snapped normalized candidates:
      Z  --denorm-->  X_phys_cont  --snap-->  X_phys_grid  --renorm-->  Z_snapped

    This ensures the optimizer "sees" the snapped surface and prevents
    distinct Z from collapsing to duplicates *after* optimization.
    """
    def _postproc(Z: torch.Tensor) -> torch.Tensor:
        # Z may be shape (q, d) or (b, q, d) depending on optimizer internals.
        z_shape = Z.shape
        Zf = Z.reshape(-1, z_shape[-1])
        # denormalize to physical
        x_cont = x_denormalizer_torch(Zf, design)
        # snap in physical units
        x_snap = snap_to_grid_torch(x_cont, design)
        # renormalize back to [0,1]^D
        z_snap = x_normalizer_torch(x_snap, design)
        # return tensor on the right device/dtype
        return z_snap.reshape(z_shape)
    return _postproc

# ---------------------------
# Acquisition builder
# ---------------------------

def build_qnehvi(
    model,                               # ModelListGP (fitted)
    train_X: torch.Tensor,               # (N, D) normalized to [0,1]^D
    ref_point_t: torch.Tensor,           # (M,) tensor on same device/dtype as model/train_X
    sample_shape: int = 128,
    prune_baseline: bool = True,
    objective: Optional[MCMultiOutputObjective] = None,
    constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,  # c(Y) -> negative feasible
    eta: Optional[Union[float, torch.Tensor]] = None,  # scalar or (num_constraints,)
    X_pending: Optional[torch.Tensor] = None,
    use_lognehvi: bool = True,
) -> qNoisyExpectedHypervolumeImprovement:
    """
    Build qNEHVI with a Sobol QMC sampler. Assumes maximization and normalized inputs.
    """
    device = train_X.device
    dtype = train_X.dtype

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(sample_shape)]))

    # If outcome constraints are provided and eta is None, set a stable default.
    if constraints and eta is None:
        eta = 0.05
    ACQ = qLogNoisyExpectedHypervolumeImprovement if use_lognehvi else qNoisyExpectedHypervolumeImprovement
    acq = ACQ(
        model=model,
        ref_point=ref_point_t.to(device=device, dtype=dtype),
        X_baseline=train_X,
        sampler=sampler,
        objective=objective or IdentityMCMultiOutputObjective(),
        constraints=constraints,
        eta=eta,
        X_pending=X_pending,
        prune_baseline=prune_baseline,
    )
    return acq


# ---------------------------
# Optimizer wrapper
# ---------------------------

def optimize_acq_qnehvi(
    acq_function,
    d: int,
    q: int,
    num_restarts: int = 20,
    raw_samples: int = 512,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    options: Optional[Dict[str, Any]] = None,
    sequential: bool = True,
    post_processing_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize an acquisition function over the unit hypercube [0,1]^d and return q candidates and their acquisition value.

    Important: Ensure device/dtype align with the acq_function's expectations (usually train_X).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if options is None:
        options = {"retry_on_optimization_warning": True}
    bounds = _unit_bounds(d, device, dtype)

    cand, acq_val = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds,
        q=q,  # NOTE: q is the batch size here
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=options,# or {"batch_limit": 8, "maxiter": 400},
        sequential=sequential,
        post_processing_func=post_processing_func,
    )
    return cand, acq_val  # (q, d), (q,)


# ---------------------------
# Main: propose a snapped, constraint-valid batch in physical space
# ---------------------------

def propose_qnehvi_batch(
    design: Design,
    model,                               # ModelListGP (fitted)
    train_X: torch.Tensor,               # (N, D) normalized to [0,1]^D
    ref_point_t: torch.Tensor,           # (M,) on model device/dtype
    batch_size: int,
    num_restarts: int = 10,
    raw_samples: int = 512,
    sample_shape: int = 128,
    max_attempts: int = 3,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    options: Optional[Dict[str, Any]] = None,
    # Row constraint on PHYSICAL units: (np.ndarray, shape (B, D)) -> bool mask
    row_constraints: Optional[Union[RowConstraint, Sequence[RowConstraint]]] = None,
    # Outcome constraints (negative => feasible)
    objective: Optional[MCMultiOutputObjective] = None,
    constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    eta: Optional[Union[float, torch.Tensor]] = None,
    X_pending: Optional[torch.Tensor] = None,
    sequential: bool = True,
    #snap_in_optimizer: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    End-to-end:
      1) Build qNEHVI
      2) Optimize on [0,1]^D **with post-processing snap (denorm→snap→renorm)**
      3) Denormalize to physical units (already snapped)
      4) Enforce optional row-wise constraints; retry up to `max_attempts`
      5) Return candidates (physical + normalized) and diagnostics

    Returns:
      dict with:
        - X_phys: np.ndarray (K, D) physical, snapped (K == batch_size if enough valid points found)
        - X_norm: np.ndarray (K, D) corresponding normalized points
        - attempts: int
        - acq_val: np.ndarray (K,) acquisition values for returned candidates
    """
    # Device / dtype aligned to train_X
    if device is None:
        device = train_X.device
    if dtype is None:
        dtype = train_X.dtype
    if options is None:
        options = {"retry_on_optimization_warning": True}

    d = len(design.names)

    # Normalize row constraints into a list
    if row_constraints is None:
        row_constraints_list: Sequence[RowConstraint] = []
    elif callable(row_constraints):
        row_constraints_list = [row_constraints]  # backward-compatible single fn
    else:
        row_constraints_list = list(row_constraints)

    # Build acquisition
    acq_func = build_qnehvi(
        model=model,
        train_X=train_X,
        ref_point_t=ref_point_t,
        sample_shape=sample_shape,
        prune_baseline=True,
        objective=objective,
        constraints=constraints,
        eta=eta,
        X_pending=X_pending,
    )

    collected_phys: List[np.ndarray] = []
    collected_norm: List[np.ndarray] = []
    collected_acq: List[np.ndarray] = []
    attempts = 0

    # Post-processing hook: snap in normalized space by denorm->snap->renorm
    postproc = _make_snap_postproc(design=design)# if snap_in_optimizer else None

    while attempts < max_attempts:
        attempts += 1

        # 0) Assign currently collected points to pending
        #    (if any) to avoid re-evaluating the acquisition function on them.
        if collected_norm:
            Xpend = torch.tensor(np.vstack(collected_norm), device=device, dtype=dtype)
            acq_func.set_X_pending(Xpend)
        else:
            acq_func.set_X_pending(None)

        # 1) optimize on the unit hypercube
        cand_norm_t, acq_val_t = optimize_acq_qnehvi(
            acq_function=acq_func,
            d=d,
            q=batch_size,  # correct kwarg for batch size
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            device=device,
            dtype=dtype,
            options=options,# or {"batch_limit": 8, "maxiter": 400, "retry_on_optimization_warning": True},
            sequential=sequential,
            post_processing_func=postproc,
        )  # (q, d)

        # 2) denormalize to physical space via design, then snap
        cand_np = cand_norm_t.detach().cpu().numpy()
        acq_np  = acq_val_t.detach().cpu().numpy().reshape(-1)
        #X_cont = x_denormalizer_np(cand_np, design)      # physical-space continuous
        #X_snapped = snap_to_grid_np(X_cont, design)      # apply grids where present
        # if snap_in_optimizer:
        X_snapped = x_denormalizer_np(cand_np, design)  # already snapped in norm space
        Z_norm = cand_np
        # else:
        #     X_cont = x_denormalizer_np(cand_np, design)
        #     X_snapped = snap_to_grid_np(X_cont, design)
        #     Z_norm = x_normalizer_np(X_snapped, design)

        # 3) enforce optional row constraints in PHYSICAL units
        #mask = apply_row_constraints(X_snapped, design, row_constraints_list)
        #X_valid = X_snapped[mask]
        #Z_valid = cand_np[mask]  # corresponding normalized points
        mask   = apply_row_constraints(X_snapped, design, row_constraints_list)
        X_valid = X_snapped[mask]
        Z_valid = Z_norm[mask]
        A_valid = acq_np[mask]


        if X_valid.shape[0] > 0:
            collected_phys.append(X_valid)
            collected_norm.append(Z_valid)
            collected_acq.append(A_valid)

        # Enough points collected?
        total = sum(arr.shape[0] for arr in collected_phys)
        if total >= batch_size:
            break

    # Consolidate results
    if collected_phys:
        X_phys_all = np.vstack(collected_phys)
        X_norm_all = np.vstack(collected_norm)
        A_all =      np.concatenate(collected_acq)

        # # Keep first occurrence of each unique physical row
        # _, unique_idx = np.unique(X_phys_all, axis=0, return_index=True)
        # order = np.argsort(unique_idx)
        # X_phys = X_phys_all[unique_idx[order]][:batch_size]
        # X_norm = X_norm_all[unique_idx[order]][:batch_size]
        # acq_val = A_all[unique_idx[order]][:batch_size]
        X_phys = X_phys_all[:batch_size]
        X_norm = X_norm_all[:batch_size]
        acq_val = A_all[:batch_size]
    else:
        X_phys = np.empty((0, d), dtype=float)
        X_norm = np.empty((0, d), dtype=float)
        acq_val = np.empty((0,), dtype=float)
        if verbose:
            print(
                f"No valid candidates found after {attempts} attempts; returning empty batch."
            )

    out: Dict[str, Any] = dict(
        X_phys=X_phys,  # snapped, physical units
        X_norm=X_norm,  # corresponding normalized points in [0,1]^D
        acq_val=acq_val,  # acquisition values for returned candidates
        attempts=attempts,
    )
    return out