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
from botorch.optim.optimize import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_discrete_local_search,
)

from .design import Design
from .data import x_normalizer_torch, x_normalizer_np, x_denormalizer_np, snap_to_grid_torch, x_denormalizer_torch
from .constraints import apply_row_constraints, RowConstraint

LinearConstraint = Tuple[torch.Tensor, torch.Tensor, float]


# ---------------------------
# Helpers
# ---------------------------

def _unit_bounds(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Return a (2, d) tensor of bounds for the unit hypercube [0, 1]^d
    on the specified device/dtype.
    
    Args:
        d: Number of dimensions
        device: Target device for the tensor
        dtype: Target data type for the tensor
        
    Returns:
        torch.Tensor: Bounds tensor of shape (2, d) with [lower_bounds, upper_bounds]
    """
    lb = torch.zeros(d, device=device, dtype=dtype)
    ub = torch.ones(d, device=device, dtype=dtype)
    return torch.stack([lb, ub], dim=0)


# Outcome-space constraint builders (negative => feasible)
def outcome_ge(obj_idx: int, thresh: float):
    """
    Create a constraint function for Y[..., obj_idx] >= thresh.
    
    Args:
        obj_idx: Index of the objective to constrain
        thresh: Threshold value for the constraint (in the same space as Y)
        
    Returns:
        Callable: Constraint function that returns negative values when feasible
    """
    def _c(Y: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(thresh, dtype=Y.dtype, device=Y.device) - Y[..., obj_idx]
    return _c


def outcome_le(obj_idx: int, thresh: float):
    """
    Create a constraint function for Y[..., obj_idx] <= thresh.
    
    Args:
        obj_idx: Index of the objective to constrain
        thresh: Threshold value for the constraint (in the same space as Y)
        
    Returns:
        Callable: Constraint function that returns negative values when feasible
    """
    def _c(Y: torch.Tensor) -> torch.Tensor:
        return Y[..., obj_idx] - torch.as_tensor(thresh, dtype=Y.dtype, device=Y.device)
    return _c


def outcome_ge_standardized(obj_idx: int, thresh_original: float, Y_mean: float, Y_std: float):
    """
    Create a constraint function for Y[..., obj_idx] >= thresh in original units.
    Automatically converts threshold from original units to standardized space.
    
    Args:
        obj_idx: Index of the objective to constrain
        thresh_original: Threshold value in original units (before standardization)
        Y_mean: Mean value used for standardization of this objective
        Y_std: Standard deviation used for standardization of this objective
        
    Returns:
        Callable: Constraint function that returns negative values when feasible
    """
    # Convert threshold from original units to standardized space
    thresh_standardized = (thresh_original - Y_mean) / Y_std
    return outcome_ge(obj_idx, thresh_standardized)


def outcome_le_standardized(obj_idx: int, thresh_original: float, Y_mean: float, Y_std: float):
    """
    Create a constraint function for Y[..., obj_idx] <= thresh in original units.
    Automatically converts threshold from original units to standardized space.
    
    Args:
        obj_idx: Index of the objective to constrain
        thresh_original: Threshold value in original units (before standardization)
        Y_mean: Mean value used for standardization of this objective
        Y_std: Standard deviation used for standardization of this objective
        
    Returns:
        Callable: Constraint function that returns negative values when feasible
    """
    # Convert threshold from original units to standardized space
    thresh_standardized = (thresh_original - Y_mean) / Y_std
    return outcome_le(obj_idx, thresh_standardized)

def _make_snap_postproc(design: Design) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Factory for a post-processing function to pass into optimize_acqf.

    It maps normalized candidates Z in [0,1]^D to snapped normalized candidates:
      Z  --denorm-->  X_phys_cont  --snap-->  X_phys_grid  --renorm-->  Z_snapped

    This ensures the optimizer "sees" the snapped surface and prevents
    distinct Z from collapsing to duplicates *after* optimization.
    
    Args:
        design: Design object containing grid specifications
        
    Returns:
        Callable: Post-processing function that snaps candidates to grid values
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
    
    Args:
        model: Fitted ModelListGP model
        train_X: Training inputs normalized to [0,1]^D, shape (N, D)
        ref_point_t: Reference point for hypervolume calculation, shape (M,)
        sample_shape: Number of MC samples for acquisition evaluation
        prune_baseline: Whether to prune baseline points
        objective: Multi-output objective function (default: IdentityMCMultiOutputObjective)
        constraints: List of outcome constraint functions (negative => feasible)
        eta: Constraint violation penalty parameter
        X_pending: Pending points to avoid re-evaluation
        use_lognehvi: Whether to use log-transformed NEHVI
        
    Returns:
        qNoisyExpectedHypervolumeImprovement: Configured acquisition function
    """
    # Validate that X is normalized between 0 and 1
    X_min, X_max = train_X.min(), train_X.max()
    
    # Check X normalization
    if X_min < 0.0 or X_max > 1.0:
        import warnings
        warnings.warn(
            f"X is not normalized to [0,1] range. Current range: [{X_min:.4f}, {X_max:.4f}]. "
            "This may cause poor GP performance. Consider normalizing X before calling this function.",
            UserWarning,
            stacklevel=2
        )

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

def _move_linear_constraints(
    constraints: Optional[Sequence[LinearConstraint]],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[List[LinearConstraint]]:
    if constraints is None:
        return None
    moved: List[LinearConstraint] = []
    for indices, coefficients, rhs in constraints:
        moved.append(
            (
                indices.to(device=device, dtype=torch.long),
                coefficients.to(device=device, dtype=dtype),
                float(rhs),
            )
        )
    return moved


def make_normalized_linear_constraint(
    design: Design,
    names: Sequence[str],
    coefficients: Sequence[float],
    rhs: float,
    sense: str = ">=",
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> LinearConstraint:
    """
    Build a BoTorch linear constraint on normalized inputs from physical units.

    BoTorch expects constraints of the form sum_i z[i] * c[i] >= rhs, where z is
    normalized to [0, 1]. This helper lets callers express the same constraint in
    physical units. For example, SOLVENT_2ME + SOLVENT_ACN <= 100 can be written as:

        make_normalized_linear_constraint(design, ["SOLVENT_2ME", "SOLVENT_ACN"], [1, 1], 100, "<=")
    """
    if len(names) != len(coefficients):
        raise ValueError("names and coefficients must have the same length.")
    if sense not in {">=", "<=", "=="}:
        raise ValueError("sense must be one of '>=', '<=', or '=='.")

    indices_np = np.array([design.names.index(name) for name in names], dtype=np.int64)
    coeff_np = np.asarray(coefficients, dtype=float)
    mins = np.array([arr.min() for arr in design.var_array], dtype=float)
    maxs = np.array([arr.max() for arr in design.var_array], dtype=float)
    spans = np.where(maxs > mins, maxs - mins, 1.0)

    # x_phys = mins + spans * z, so c*x_phys >= rhs becomes
    # (c*spans)*z >= rhs - c*mins.
    normalized_coeff = coeff_np * spans[indices_np]
    normalized_rhs = float(rhs - np.dot(coeff_np, mins[indices_np]))

    if sense == "<=":
        normalized_coeff = -normalized_coeff
        normalized_rhs = -normalized_rhs

    return (
        torch.tensor(indices_np, device=device, dtype=torch.long),
        torch.tensor(normalized_coeff, device=device, dtype=dtype),
        normalized_rhs,
    )


def _linear_constraints_mask(
    X: np.ndarray,
    inequality_constraints: Optional[Sequence[LinearConstraint]],
    equality_constraints: Optional[Sequence[LinearConstraint]],
    tol: float = 1e-8,
) -> np.ndarray:
    mask = np.ones(X.shape[0], dtype=bool)
    if inequality_constraints:
        for indices, coefficients, rhs in inequality_constraints:
            idx = indices.detach().cpu().numpy().astype(int)
            coef = coefficients.detach().cpu().numpy().astype(float)
            mask &= X[:, idx] @ coef >= float(rhs) - tol
    if equality_constraints:
        for indices, coefficients, rhs in equality_constraints:
            idx = indices.detach().cpu().numpy().astype(int)
            coef = coefficients.detach().cpu().numpy().astype(float)
            mask &= np.abs(X[:, idx] @ coef - float(rhs)) <= tol
    return mask


def optimize_acq_function(
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
    bounds: Optional[torch.Tensor] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    equality_constraints: Optional[Sequence[LinearConstraint]] = None,
    nonlinear_inequality_constraints: Optional[List[Tuple[Callable, bool]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    batch_initial_conditions: Optional[torch.Tensor] = None,
    return_best_only: bool = True,
    retry_on_optimization_warning: bool = True,
    timeout_sec: Optional[float] = None,
    **ic_gen_kwargs: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize an acquisition function over normalized inputs.

    This is a small, explicit wrapper around BoTorch's optimize_acqf. It keeps the
    common [0, 1]^d bounds default while exposing BoTorch's linear/nonlinear
    constraint hooks for recipe constraints.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_options = dict(options or {})
    if "retry_on_optimization_warning" in opt_options:
        retry_on_optimization_warning = bool(opt_options.pop("retry_on_optimization_warning"))

    bounds_t = bounds.to(device=device, dtype=dtype) if bounds is not None else _unit_bounds(d, device, dtype)

    cand, acq_val = optimize_acqf(
        acq_function=acq_function,
        bounds=bounds_t,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options=opt_options or None,
        inequality_constraints=_move_linear_constraints(inequality_constraints, device, dtype),
        equality_constraints=_move_linear_constraints(equality_constraints, device, dtype),
        nonlinear_inequality_constraints=nonlinear_inequality_constraints,
        fixed_features=fixed_features,
        sequential=sequential,
        post_processing_func=post_processing_func,
        batch_initial_conditions=batch_initial_conditions,
        return_best_only=return_best_only,
        retry_on_optimization_warning=retry_on_optimization_warning,
        timeout_sec=timeout_sec,
        **ic_gen_kwargs,
    )
    return cand, acq_val


def optimize_acq_qnehvi(*args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible alias for optimize_acq_function."""
    return optimize_acq_function(*args, **kwargs)


def random_discrete_choices(
    design: Design,
    n: int,
    seed: Optional[int] = None,
    row_constraints: Optional[Union[RowConstraint, Sequence[RowConstraint]]] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    equality_constraints: Optional[Sequence[LinearConstraint]] = None,
    X_avoid: Optional[Union[np.ndarray, torch.Tensor]] = None,
    max_draws: int = 200000,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Sample unique valid choices from the explicit design grids.

    Returns both physical choices and normalized choices. This is intended for
    large product grids where enumerating every possible recipe is impractical.
    """
    rng = np.random.default_rng(seed)
    if row_constraints is None:
        row_constraints_list: Sequence[RowConstraint] = []
    elif callable(row_constraints):
        row_constraints_list = [row_constraints]
    else:
        row_constraints_list = list(row_constraints)

    avoid = set()
    if X_avoid is not None:
        X_avoid_np = X_avoid.detach().cpu().numpy() if isinstance(X_avoid, torch.Tensor) else np.asarray(X_avoid, dtype=float)
        for row in X_avoid_np:
            avoid.add(tuple(np.round(row, decimals=8)))

    rows: List[np.ndarray] = []
    seen = set(avoid)
    draws = 0
    while len(rows) < n and draws < max_draws:
        draws += 1
        row = np.array([rng.choice(grid) for grid in design.var_array], dtype=float)
        key = tuple(np.round(row, decimals=8))
        if key in seen:
            continue

        row_2d = row.reshape(1, -1)
        row_norm = x_normalizer_np(row_2d, design)
        if not _linear_constraints_mask(row_norm, inequality_constraints, equality_constraints)[0]:
            continue
        if not apply_row_constraints(row_2d, design, row_constraints_list)[0]:
            continue

        seen.add(key)
        rows.append(row)

    if len(rows) < n:
        raise RuntimeError(
            f"Generated only {len(rows)} valid discrete choices after {draws} draws; "
            "reduce n, increase max_draws, or relax constraints."
        )

    choices_phys = np.vstack(rows)
    choices_norm = torch.tensor(
        x_normalizer_np(choices_phys, design),
        device=device,
        dtype=dtype,
    )
    return choices_phys, choices_norm


def optimize_acq_function_discrete(
    acq_function,
    q: int,
    choices: torch.Tensor,
    max_batch_size: int = 2048,
    unique: bool = True,
    return_acq_values: bool = True,
    X_avoid: Optional[torch.Tensor] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize an acquisition function over an explicit normalized choice set.

    This uses BoTorch's optimize_acqf_discrete directly and does not perform any
    continuous optimization or snapping.
    """
    device = choices.device
    dtype = choices.dtype
    if not return_acq_values:
        raise ValueError("This BoTorch version always returns acquisition values.")
    return optimize_acqf_discrete(
        acq_function=acq_function,
        q=q,
        choices=choices,
        max_batch_size=max_batch_size,
        unique=unique,
        X_avoid=None if X_avoid is None else X_avoid.to(device=device, dtype=dtype),
        inequality_constraints=_move_linear_constraints(inequality_constraints, device, dtype),
    )


def propose_batch_discrete(
    design: Design,
    model,
    train_X: torch.Tensor,
    ref_point_t: torch.Tensor,
    batch_size: int,
    choices: Optional[torch.Tensor] = None,
    choices_phys: Optional[np.ndarray] = None,
    n_choices: int = 5000,
    seed: Optional[int] = None,
    max_draws: int = 200000,
    max_batch_size: int = 2048,
    unique: bool = True,
    avoid_train_X: bool = True,
    use_lognehvi: bool = True,
    sample_shape: int = 128,
    acq: Optional[Callable] = None,
    objective: Optional[MCMultiOutputObjective] = None,
    constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    eta: Optional[Union[float, torch.Tensor]] = None,
    X_pending: Optional[torch.Tensor] = None,
    row_constraints: Optional[Union[RowConstraint, Sequence[RowConstraint]]] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    equality_constraints: Optional[Sequence[LinearConstraint]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Propose a batch from explicit discrete choices using qNEHVI.

    Provide either `choices` in normalized units, `choices_phys` in physical units,
    or let this function draw `n_choices` random valid grid recipes from `design`.
    """
    device = train_X.device
    dtype = train_X.dtype

    if choices is not None and choices_phys is not None:
        raise ValueError("Pass either choices or choices_phys, not both.")

    if choices is not None:
        choices_norm = choices.to(device=device, dtype=dtype)
        choices_phys_np = x_denormalizer_np(choices_norm.detach().cpu().numpy(), design)
    elif choices_phys is not None:
        choices_phys_np = np.asarray(choices_phys, dtype=float)
        choices_norm = torch.tensor(
            x_normalizer_np(choices_phys_np, design),
            device=device,
            dtype=dtype,
        )
    else:
        avoid_phys = x_denormalizer_np(train_X.detach().cpu().numpy(), design) if avoid_train_X else None
        choices_phys_np, choices_norm = random_discrete_choices(
            design=design,
            n=n_choices,
            seed=seed,
            row_constraints=row_constraints,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            X_avoid=avoid_phys,
            max_draws=max_draws,
            device=device,
            dtype=dtype,
        )

    if choices is not None or choices_phys is not None:
        mask = _linear_constraints_mask(
            choices_norm.detach().cpu().numpy(),
            inequality_constraints,
            equality_constraints,
        )
        if row_constraints is not None:
            row_constraints_list = [row_constraints] if callable(row_constraints) else list(row_constraints)
            mask &= apply_row_constraints(choices_phys_np, design, row_constraints_list)
        choices_phys_np = choices_phys_np[mask]
        choices_norm = choices_norm[torch.tensor(mask, device=device, dtype=torch.bool)]

    x_avoid_parts = []
    if avoid_train_X:
        x_avoid_parts.append(train_X.to(device=device, dtype=dtype))
    if X_pending is not None:
        x_avoid_parts.append(X_pending.to(device=device, dtype=dtype))
    X_avoid = torch.cat(x_avoid_parts, dim=0) if x_avoid_parts else None

    if acq is not None:
        acq_func = acq()
    else:
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
            use_lognehvi=use_lognehvi,
        )

    cand_norm_t, acq_val_t = optimize_acq_function_discrete(
        acq_function=acq_func,
        q=batch_size,
        choices=choices_norm,
        max_batch_size=max_batch_size,
        unique=unique,
        return_acq_values=True,
        X_avoid=X_avoid,
        inequality_constraints=inequality_constraints,
    )

    cand_norm = cand_norm_t.detach().cpu().numpy()
    cand_phys = x_denormalizer_np(cand_norm, design)
    acq_val = acq_val_t.detach().cpu().numpy().reshape(-1)

    if verbose:
        print(f"[propose_batch_discrete] choices={choices_norm.shape[0]} returned={cand_phys.shape[0]}")

    return {
        "X_phys": cand_phys,
        "X_norm": cand_norm,
        "acq_val": acq_val,
        "candidate_pool_size": int(choices_norm.shape[0]),
        "mode": "discrete_optimize_acqf",
    }


def _dominated_by_observed_np(y: np.ndarray, observed_Y: np.ndarray) -> bool:
    """Return True if any observed point is >= y in all objectives and > in one."""
    return bool(np.any(np.all(observed_Y >= y, axis=1) & np.any(observed_Y > y, axis=1)))


def _estimate_prob_not_dominated_by_observed(
    model,
    X: torch.Tensor,
    observed_Y: torch.Tensor,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate candidate-level posterior diagnostics against observed outcomes.

    All objectives are assumed to be maximized and in the same outcome space used
    for model fitting/acquisition.
    """
    with torch.no_grad():
        posterior = model.posterior(X)
        mean_t = posterior.mean
        std_t = torch.sqrt(posterior.variance.clamp_min(0.0))
        samples_t = posterior.rsample(torch.Size([int(n_samples)]))

    observed = observed_Y.to(device=X.device, dtype=X.dtype).detach().cpu().numpy()
    mean = mean_t.detach().cpu().numpy()
    std = std_t.detach().cpu().numpy()
    samples = samples_t.detach().cpu().numpy()

    mean_dominated = np.array(
        [_dominated_by_observed_np(row, observed) for row in mean],
        dtype=bool,
    )
    non_dominated_counts = np.zeros(X.shape[0], dtype=float)
    for sample in samples:
        non_dominated_counts += np.array(
            [not _dominated_by_observed_np(row, observed) for row in sample],
            dtype=float,
        )
    prob_not_dominated = non_dominated_counts / float(n_samples)
    return mean, std, prob_not_dominated, mean_dominated


def _minmax01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=float)
    if not np.any(finite):
        return out
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    if hi <= lo:
        out[finite] = 0.5
    else:
        out[finite] = (values[finite] - lo) / (hi - lo)
    return out


def _greedy_diverse_indices(
    X_norm: np.ndarray,
    base_score: np.ndarray,
    k: int,
    diversity_weight: float,
    min_selected_distance: float,
    train_X_norm: Optional[np.ndarray] = None,
    observed_distance_weight: float = 0.0,
    min_observed_distance: float = 0.0,
) -> List[int]:
    """
    Greedily select a diverse set of candidate indices.

    Scoring combines:
      1. base_score: acquisition/posterior-derived candidate score
      2. within-batch diversity: distance from already-selected candidates
      3. observed-distance exploration: distance from the existing observed dataset

    Args:
        X_norm:
            Candidate inputs in normalized [0, 1] space, shape (n_candidates, d).

        base_score:
            Base score for each candidate, shape (n_candidates,). This can already
            include qNEHVI score, probability of non-domination, etc.

        k:
            Number of candidates to select.

        diversity_weight:
            Weight for distance from already-selected candidates. This controls
            within-batch diversity.

        min_selected_distance:
            Hard minimum normalized Euclidean distance from already-selected
            candidates. If a candidate is closer than this to any selected point,
            it is skipped unless fallback selection happens outside this function.

        train_X_norm:
            Existing observed training inputs in normalized [0, 1] space,
            shape (n_observed, d). If provided, candidates farther from observed
            points receive an exploration bonus.

        observed_distance_weight:
            Weight for distance from the nearest observed training point. This is
            an explicit exploration knob.

        min_observed_distance:
            Optional hard minimum distance from the observed dataset. If > 0,
            candidates closer than this to any observed point are skipped.

    Returns:
        List of selected candidate indices.
    """
    X_norm = np.asarray(X_norm, dtype=float)
    base_score = np.asarray(base_score, dtype=float).reshape(-1)

    if X_norm.ndim != 2:
        raise ValueError(f"X_norm must be 2D, got shape {X_norm.shape}.")
    if base_score.shape[0] != X_norm.shape[0]:
        raise ValueError(
            f"base_score length {base_score.shape[0]} does not match "
            f"X_norm rows {X_norm.shape[0]}."
        )

    # Precompute distance from each candidate to the nearest observed point.
    # Then min-max normalize it so observed_distance_weight has a sensible scale.
    observed_distance = np.zeros(X_norm.shape[0], dtype=float)
    observed_distance_score = np.zeros(X_norm.shape[0], dtype=float)

    if train_X_norm is not None:
        train_X_norm = np.asarray(train_X_norm, dtype=float)

        if train_X_norm.ndim != 2:
            raise ValueError(
                f"train_X_norm must be 2D when provided, got shape {train_X_norm.shape}."
            )
        if train_X_norm.shape[1] != X_norm.shape[1]:
            raise ValueError(
                f"train_X_norm dimension {train_X_norm.shape[1]} does not match "
                f"X_norm dimension {X_norm.shape[1]}."
            )
        if train_X_norm.shape[0] > 0:
            for i, x in enumerate(X_norm):
                observed_distance[i] = float(
                    np.min(np.linalg.norm(train_X_norm - x, axis=1))
                )
            observed_distance_score = _minmax01(observed_distance)

    selected: List[int] = []
    remaining = set(range(X_norm.shape[0]))

    while remaining and len(selected) < k:
        best_idx = None
        best_score = -np.inf

        for idx in remaining:
            # Optional hard constraint: avoid candidates too close to observed data.
            if (
                train_X_norm is not None
                and train_X_norm.shape[0] > 0
                and min_observed_distance > 0
                and observed_distance[idx] < min_observed_distance
            ):
                continue

            score = float(base_score[idx])

            # Explicit exploration bonus: prefer candidates farther from observed data.
            if train_X_norm is not None and train_X_norm.shape[0] > 0:
                score += float(observed_distance_weight) * float(
                    observed_distance_score[idx]
                )

            # Within-batch diversity bonus: prefer candidates farther from selected points.
            if selected:
                selected_X = X_norm[np.asarray(selected, dtype=int)]
                min_selected_dist = float(
                    np.min(np.linalg.norm(selected_X - X_norm[idx], axis=1))
                )

                if (
                    min_selected_distance > 0
                    and min_selected_dist < min_selected_distance
                ):
                    continue

                score += float(diversity_weight) * min_selected_dist

            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected

def _nearest_distance_to_reference(
    X_norm: np.ndarray,
    X_ref_norm: Optional[np.ndarray],
) -> np.ndarray:
    if X_ref_norm is None or len(X_ref_norm) == 0:
        return np.zeros(X_norm.shape[0], dtype=float)

    X_norm = np.asarray(X_norm, dtype=float)
    X_ref_norm = np.asarray(X_ref_norm, dtype=float)

    dists = np.zeros(X_norm.shape[0], dtype=float)
    for i, x in enumerate(X_norm):
        dists[i] = float(np.min(np.linalg.norm(X_ref_norm - x, axis=1)))
    return dists


def propose_batch_discrete_posterior_guarded(
    design: Design,
    model,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    ref_point_t: torch.Tensor,
    batch_size: int,
    shortlist_size: Optional[int] = None,
    choices: Optional[torch.Tensor] = None,
    choices_phys: Optional[np.ndarray] = None,
    n_choices: int = 5000,
    seed: Optional[int] = None,
    max_draws: int = 200000,
    max_batch_size: int = 2048,
    unique: bool = True,
    avoid_train_X: bool = True,
    use_lognehvi: bool = True,
    sample_shape: int = 128,
    posterior_sample_shape: int = 256,
    acq: Optional[Callable] = None,
    objective: Optional[MCMultiOutputObjective] = None,
    constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    eta: Optional[Union[float, torch.Tensor]] = None,
    X_pending: Optional[torch.Tensor] = None,
    row_constraints: Optional[Union[RowConstraint, Sequence[RowConstraint]]] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    equality_constraints: Optional[Sequence[LinearConstraint]] = None,
    prob_weight: float = 1.0,
    acq_weight: float = 0.25,
    mean_nondominated_bonus: float = 0.25,
    diversity_weight: float = 0.4,
    min_selected_distance: float = 0.45,
    observed_distance_weight: float = 0.25,
    min_observed_distance: float = 0.0,
    conservative_beta: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Propose a guarded discrete batch from a larger qNEHVI shortlist.

    This is an opt-in layer on top of `propose_batch_discrete`: first ask qNEHVI
    for a larger shortlist, then select the final batch using posterior
    diagnostics and normalized-space diversity. It is intended for rounds where
    qNEHVI candidates have dominated posterior means but nontrivial uncertainty
    upside.
    """
    if shortlist_size is None:
        shortlist_size = max(int(batch_size), int(batch_size) * 3)
    shortlist_size = max(int(shortlist_size), int(batch_size))

    shortlist = propose_batch_discrete(
        design=design,
        model=model,
        train_X=train_X,
        ref_point_t=ref_point_t,
        batch_size=shortlist_size,
        choices=choices,
        choices_phys=choices_phys,
        n_choices=n_choices,
        seed=seed,
        max_draws=max_draws,
        max_batch_size=max_batch_size,
        unique=unique,
        avoid_train_X=avoid_train_X,
        use_lognehvi=use_lognehvi,
        sample_shape=sample_shape,
        acq=acq,
        objective=objective,
        constraints=constraints,
        eta=eta,
        X_pending=X_pending,
        row_constraints=row_constraints,
        inequality_constraints=inequality_constraints,
        equality_constraints=equality_constraints,
        verbose=verbose,
    )

    X_norm = np.asarray(shortlist["X_norm"], dtype=float)
    X_phys = np.asarray(shortlist["X_phys"], dtype=float)
    acq_val = np.asarray(shortlist["acq_val"], dtype=float).reshape(-1)
    if X_norm.shape[0] == 0:
        return {**shortlist, "mode": "discrete_posterior_guarded", "selected_indices": []}

    X_t = torch.tensor(X_norm, device=train_X.device, dtype=train_X.dtype)
    mean, std, prob_not_dominated, mean_dominated = _estimate_prob_not_dominated_by_observed(
        model=model,
        X=X_t,
        observed_Y=train_Y,
        n_samples=posterior_sample_shape,
    )

    observed_distance = _nearest_distance_to_reference(
        X_norm=X_norm,
        X_ref_norm=train_X.detach().cpu().numpy(),
    )

    observed_distance_score = _minmax01(observed_distance)

    base_score = (
        float(prob_weight) * prob_not_dominated
        + float(acq_weight) * _minmax01(acq_val)
        + float(mean_nondominated_bonus) * (~mean_dominated).astype(float)
        + float(observed_distance_weight) * observed_distance_score
    )

    conservative_dominated = None
    if conservative_beta is not None:
        conservative_y = mean - float(conservative_beta) * std
        observed = train_Y.detach().cpu().numpy()
        conservative_dominated = np.array(
            [_dominated_by_observed_np(row, observed) for row in conservative_y],
            dtype=bool,
        )
        base_score += 0.25 * (~conservative_dominated).astype(float)

    selected = _greedy_diverse_indices(
        X_norm=X_norm,
        base_score=base_score,
        k=batch_size,
        diversity_weight=diversity_weight,
        min_selected_distance=min_selected_distance,
        train_X_norm=train_X.detach().cpu().numpy(),
        observed_distance_weight=0.0,  # already included in base_score
        min_observed_distance=min_observed_distance,
    )

    if len(selected) < batch_size:
        fallback = [idx for idx in np.argsort(-base_score) if idx not in selected]
        selected.extend(fallback[: batch_size - len(selected)])

    selected_idx = np.asarray(selected[:batch_size], dtype=int)
    diagnostics = {
        "posterior_mean": mean,
        "posterior_std": std,
        "prob_not_dominated_by_observed": prob_not_dominated,
        "mean_is_dominated_by_observed": mean_dominated,
        "observed_distance": observed_distance,
        "observed_distance_score": observed_distance_score,
        "selection_score": base_score,
    }
    if conservative_dominated is not None:
        diagnostics["conservative_is_dominated_by_observed"] = conservative_dominated

    if verbose:
        print(
            "[propose_batch_discrete_posterior_guarded] "
            f"shortlist={X_norm.shape[0]} selected={selected_idx.size}"
        )

    return {
        "X_phys": X_phys[selected_idx],
        "X_norm": X_norm[selected_idx],
        "acq_val": acq_val[selected_idx],
        "candidate_pool_size": shortlist.get("candidate_pool_size"),
        "shortlist_X_phys": X_phys,
        "shortlist_X_norm": X_norm,
        "shortlist_acq_val": acq_val,
        "shortlist_diagnostics": diagnostics,
        "selected_indices": selected_idx,
        "mode": "discrete_posterior_guarded",
    }


def normalized_discrete_choices_from_design(
    design: Design,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> List[torch.Tensor]:
    """
    Convert each design grid to a normalized tensor for discrete local search.

    The returned list is suitable for BoTorch's optimize_acqf_discrete_local_search.
    """
    choices: List[torch.Tensor] = []
    for grid in design.var_array:
        values = np.asarray(grid, dtype=float)
        lower = float(values.min())
        upper = float(values.max())
        denom = upper - lower if upper > lower else 1.0
        choices.append(torch.tensor((values - lower) / denom, device=device, dtype=dtype))
    return choices


def optimize_acq_function_discrete_local_search(
    acq_function,
    discrete_choices: Sequence[torch.Tensor],
    q: int,
    num_restarts: int = 20,
    raw_samples: int = 4096,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    X_avoid: Optional[torch.Tensor] = None,
    batch_initial_conditions: Optional[torch.Tensor] = None,
    max_batch_size: int = 2048,
    max_tries: int = 100,
    unique: bool = True,
    return_acq_values: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimize over a per-dimension discrete lattice with BoTorch local search.

    `discrete_choices` should be in the same normalized coordinate system as the
    acquisition function inputs. This path does not perform continuous
    optimization or snapping.
    """
    if not discrete_choices:
        raise ValueError("discrete_choices must contain one tensor per input dimension.")
    if not return_acq_values:
        raise ValueError("This BoTorch version always returns acquisition values.")

    device = discrete_choices[0].device
    dtype = discrete_choices[0].dtype
    choices = [choice.to(device=device, dtype=dtype) for choice in discrete_choices]
    bic = None
    if batch_initial_conditions is not None:
        bic = batch_initial_conditions.to(device=device, dtype=dtype)

    return optimize_acqf_discrete_local_search(
        acq_function=acq_function,
        discrete_choices=choices,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=_move_linear_constraints(inequality_constraints, device, dtype),
        X_avoid=None if X_avoid is None else X_avoid.to(device=device, dtype=dtype),
        batch_initial_conditions=bic,
        max_batch_size=max_batch_size,
        max_tries=max_tries,
        unique=unique,
    )


def propose_batch_discrete_local_search(
    design: Design,
    model,
    train_X: torch.Tensor,
    ref_point_t: torch.Tensor,
    batch_size: int,
    discrete_choices: Optional[Sequence[torch.Tensor]] = None,
    num_restarts: int = 20,
    raw_samples: int = 4096,
    max_batch_size: int = 2048,
    max_tries: int = 100,
    unique: bool = True,
    avoid_train_X: bool = True,
    use_lognehvi: bool = True,
    sample_shape: int = 128,
    acq: Optional[Callable] = None,
    objective: Optional[MCMultiOutputObjective] = None,
    constraints: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    eta: Optional[Union[float, torch.Tensor]] = None,
    X_pending: Optional[torch.Tensor] = None,
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    batch_initial_conditions: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Propose a batch by local search over the design lattice.

    This is the most direct path for large discrete grids: each dimension's
    allowed values are passed to BoTorch, and candidates are returned directly on
    that lattice in normalized coordinates.
    """
    device = train_X.device
    dtype = train_X.dtype

    choices = (
        [choice.to(device=device, dtype=dtype) for choice in discrete_choices]
        if discrete_choices is not None
        else normalized_discrete_choices_from_design(design, device=device, dtype=dtype)
    )

    x_avoid_parts = []
    if avoid_train_X:
        x_avoid_parts.append(train_X.to(device=device, dtype=dtype))
    if X_pending is not None:
        x_avoid_parts.append(X_pending.to(device=device, dtype=dtype))
    X_avoid = torch.cat(x_avoid_parts, dim=0) if x_avoid_parts else None

    if acq is not None:
        acq_func = acq()
    else:
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
            use_lognehvi=use_lognehvi,
        )

    cand_norm_t, acq_val_t = optimize_acq_function_discrete_local_search(
        acq_function=acq_func,
        discrete_choices=choices,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        inequality_constraints=inequality_constraints,
        X_avoid=X_avoid,
        batch_initial_conditions=batch_initial_conditions,
        max_batch_size=max_batch_size,
        max_tries=max_tries,
        unique=unique,
        return_acq_values=True,
    )

    cand_norm = cand_norm_t.detach().cpu().numpy()
    cand_phys = x_denormalizer_np(cand_norm, design)
    acq_val = acq_val_t.detach().cpu().numpy().reshape(-1)

    if verbose:
        sizes = [int(choice.numel()) for choice in choices]
        print(f"[propose_batch_discrete_local_search] choice_sizes={sizes} returned={cand_phys.shape[0]}")

    return {
        "X_phys": cand_phys,
        "X_norm": cand_norm,
        "acq_val": acq_val,
        "discrete_choice_sizes": [int(choice.numel()) for choice in choices],
        "mode": "discrete_local_search",
    }


# ---------------------------
# Main: propose a snapped, constraint-valid batch in physical space
# ---------------------------

def propose_batch(
    design: Design,
    model,                               # ModelListGP (fitted)
    train_X: torch.Tensor,               # (N, D) normalized to [0,1]^D
    ref_point_t: torch.Tensor,           # (M,) on model device/dtype
    batch_size: int,
    acq: Optional[Callable] = None,      # Optional custom acquisition function (must be BoTorch MOBO compatible)
    use_lognehvi: bool = True,
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
    inequality_constraints: Optional[Sequence[LinearConstraint]] = None,
    equality_constraints: Optional[Sequence[LinearConstraint]] = None,
    nonlinear_inequality_constraints: Optional[List[Tuple[Callable, bool]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    deduplicate: bool = True,
    exclude_existing: bool = True,
    min_distance: float = 0.0,
    snap_to_grid: bool = True,
    retry_on_optimization_warning: bool = True,
    timeout_sec: Optional[float] = None,
    #snap_in_optimizer: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    End-to-end:
      1) Build acquisition function (custom or default qNEHVI)
      2) Optimize on [0,1]^D **with post-processing snap (denorm→snap→renorm)**
      3) Denormalize to physical units (already snapped)
      4) Enforce optional row-wise constraints; retry up to `max_attempts`
      5) Return candidates (physical + normalized) and diagnostics

    Args:
        design: Design object containing parameter space and grid specifications
        model: Fitted ModelListGP model
        train_X: Training inputs normalized to [0,1]^D, shape (N, D)
        ref_point_t: Reference point for hypervolume calculation, shape (M,)
        batch_size: Number of candidates to generate
        acq: If provided, must be a callable that returns a BoTorch acquisition function
             compatible with multi-objective optimization. User is responsible for
             ensuring compatibility with the optimization pipeline.
        num_restarts: Number of optimization restarts
        raw_samples: Number of raw samples for initialization
        sample_shape: Number of MC samples for acquisition evaluation
        max_attempts: Maximum attempts to find valid candidates
        device: Target device (default: train_X.device)
        dtype: Target data type (default: train_X.dtype)
        options: Optimization options dictionary
        row_constraints: Optional row-wise constraints on physical units
        objective: Multi-output objective function
        constraints: List of outcome constraint functions (negative => feasible)
        eta: Constraint violation penalty parameter
        X_pending: Pending points to avoid re-evaluation
        sequential: Whether to optimize candidates sequentially
        verbose: Whether to print progress information

    Returns:
        dict: Dictionary containing:
            - X_phys: np.ndarray (K, D) physical, snapped (K == batch_size if enough valid points found)
            - X_norm: np.ndarray (K, D) corresponding normalized points in [0,1]^D
            - attempts: int - number of attempts made
            - acq_val: np.ndarray (K,) acquisition values for returned candidates
    """
    # Validate that X is normalized between 0 and 1
    X_min, X_max = train_X.min(), train_X.max()
    
    # Check X normalization
    if X_min < 0.0 or X_max > 1.0:
        import warnings
        warnings.warn(
            f"X is not normalized to [0,1] range. Current range: [{X_min:.4f}, {X_max:.4f}]. "
            "This may cause poor GP performance. Consider normalizing X before calling this function.",
            UserWarning,
            stacklevel=2
        )

    # Device / dtype aligned to train_X
    if device is None:
        device = train_X.device
    if dtype is None:
        dtype = train_X.dtype
    if options is None:
        options = {}

    d = len(design.names)

    # Normalize row constraints into a list
    if row_constraints is None:
        row_constraints_list: Sequence[RowConstraint] = []
    elif callable(row_constraints):
        row_constraints_list = [row_constraints]  # backward-compatible single fn
    else:
        row_constraints_list = list(row_constraints)

    # Build acquisition function
    if acq is not None:
        # Use custom acquisition function builder - user is responsible for compatibility
        acq_func = acq()
    else:
        # Use default qNEHVI builder
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
            use_lognehvi=use_lognehvi
        )

    collected_phys: List[np.ndarray] = []
    collected_norm: List[np.ndarray] = []
    collected_acq: List[float] = []
    rejected_norm: List[np.ndarray] = []
    attempts = 0

    # Post-processing hook: snap in normalized space by denorm->snap->renorm
    postproc = _make_snap_postproc(design=design) if snap_to_grid else None

    seen_phys = set()
    distance_reference: List[np.ndarray] = []

    if exclude_existing:
        train_np = train_X.detach().cpu().numpy()
        train_phys = x_denormalizer_np(train_np, design)
        for row in train_phys:
            seen_phys.add(tuple(np.round(row, decimals=8)))
        if min_distance > 0:
            distance_reference.extend(train_np)

    if X_pending is not None and min_distance > 0:
        distance_reference.extend(X_pending.detach().cpu().numpy())

    while attempts < max_attempts:
        attempts += 1
        needed = batch_size - len(collected_phys)
        if needed <= 0:
            break

        pending_parts = []
        if X_pending is not None:
            pending_parts.append(X_pending.to(device=device, dtype=dtype))
        if collected_norm:
            pending_parts.append(torch.tensor(np.vstack(collected_norm), device=device, dtype=dtype))
        if rejected_norm:
            pending_parts.append(torch.tensor(np.vstack(rejected_norm), device=device, dtype=dtype))
        Xpend = torch.cat(pending_parts, dim=0) if pending_parts else None
        if hasattr(acq_func, "set_X_pending"):
            acq_func.set_X_pending(Xpend)

        cand_norm_t, acq_val_t = optimize_acq_function(
            acq_function=acq_func,
            d=d,
            q=needed,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            device=device,
            dtype=dtype,
            options=options,
            sequential=sequential,
            post_processing_func=postproc,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            nonlinear_inequality_constraints=nonlinear_inequality_constraints,
            fixed_features=fixed_features,
            retry_on_optimization_warning=retry_on_optimization_warning,
            timeout_sec=timeout_sec,
        )

        cand_np = cand_norm_t.detach().cpu().numpy()
        acq_np = acq_val_t.detach().cpu().numpy().reshape(-1)
        if acq_np.size == 1 and cand_np.shape[0] > 1:
            acq_np = np.repeat(acq_np.item(), cand_np.shape[0])

        X_phys_attempt = x_denormalizer_np(cand_np, design)
        Z_norm_attempt = cand_np

        mask = _linear_constraints_mask(Z_norm_attempt, inequality_constraints, equality_constraints)
        mask &= apply_row_constraints(X_phys_attempt, design, row_constraints_list)
        X_valid = X_phys_attempt[mask]
        Z_valid = Z_norm_attempt[mask]
        A_valid = acq_np[mask]
        rejected_norm.extend(Z_norm_attempt[~mask])

        for x_row, z_row, a_val in zip(X_valid, Z_valid, A_valid):
            key = tuple(np.round(x_row, decimals=8))
            if deduplicate and key in seen_phys:
                rejected_norm.append(z_row.copy())
                continue
            if min_distance > 0 and distance_reference:
                ref = np.vstack(distance_reference)
                if np.min(np.linalg.norm(ref - z_row, axis=1)) < min_distance:
                    rejected_norm.append(z_row.copy())
                    continue
            seen_phys.add(key)
            distance_reference.append(z_row.copy())
            collected_phys.append(x_row.reshape(1, -1))
            collected_norm.append(z_row.reshape(1, -1))
            collected_acq.append(float(a_val))
            if len(collected_phys) >= batch_size:
                break

        if len(collected_phys) >= batch_size:
            break
        if verbose:
            print(f"[propose_batch] attempt {attempts}: collected {len(collected_phys)}/{batch_size}")

    # Consolidate results
    if collected_phys:
        X_phys_all = np.vstack(collected_phys)
        X_norm_all = np.vstack(collected_norm)
        A_all = np.asarray(collected_acq, dtype=float)

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
        fully_satisfied=bool(X_phys.shape[0] == batch_size),
    )
    return out