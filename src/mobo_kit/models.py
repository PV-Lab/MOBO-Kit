from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import gpytorch
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

from botorch.models import SingleTaskGP
from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, ConstantKernel
from gpytorch.priors import GammaPrior, LogNormalPrior

from .utils import torch_to_np
from .data import y_destandardize_np

_BAYBE_DIM_LIMITS = (8, 75)


def _baybe_interp(effective_dims: int, low: float, high: float) -> float:
    """Linearly interpolate a value across BayBE low/high dimensional regimes."""
    return float(np.interp(effective_dims, _BAYBE_DIM_LIMITS, [low, high]))


def _baybe_default_priors(
    input_dims: int, num_task_parameters: int = 0
) -> Dict[str, float | gpytorch.priors.Prior]:
    """
    Build BayBE-like default GP priors and initial values from effective dimensionality.

    Mirrors the interpolation strategy from:
    baybe/surrogates/gaussian_process/presets/default.py
    """
    effective_dims = max(1, int(input_dims) - int(num_task_parameters))

    lengthscale_prior = GammaPrior(
        _baybe_interp(effective_dims, 1.2, 2.5),
        _baybe_interp(effective_dims, 1.1, 0.55),
    )
    outputscale_prior = GammaPrior(
        _baybe_interp(effective_dims, 5.0, 3.5),
        _baybe_interp(effective_dims, 0.5, 0.15),
    )
    noise_prior = GammaPrior(
        _baybe_interp(effective_dims, 1.05, 1.5),
        _baybe_interp(effective_dims, 0.5, 0.1),
    )

    return {
        "lengthscale_prior": lengthscale_prior,
        "lengthscale_init": _baybe_interp(effective_dims, 0.2, 6.0),
        "outputscale_prior": outputscale_prior,
        "outputscale_init": _baybe_interp(effective_dims, 8.0, 15.0),
        "noise_prior": noise_prior,
        "noise_init": _baybe_interp(effective_dims, 0.1, 5.0),
    }

# --------------------------------------------------------------------------------------
# Core: fit one GP per output and wrap in a ModelListGP (matches your notebook function)
# --------------------------------------------------------------------------------------
def fit_gp_models_rrp(X, Y, kernel_fn=None) -> ModelListGP:
    models = []
    D, M = X.shape[1], Y.shape[1]
    for j in range(M):
        base_k = kernel_fn(D) if kernel_fn else MaternKernel(nu=2.5, ard_num_dims=D)
        covar_module = ScaleKernel(base_k)
        noise_constraint=gpytorch.constraints.GreaterThan(1e-3)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
        gp = RobustRelevancePursuitSingleTaskGP(
            train_X=X, train_Y=Y[:, j:j+1], covar_module=covar_module,
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)  # triggers relevance pursuit automatically
        models.append(gp)
    return ModelListGP(*models)


def _resolve_training_device(
    train_X: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.device:
    """Prefer an explicit device, otherwise infer from training inputs."""
    if device is not None:
        return device
    if isinstance(train_X, torch.Tensor):
        return train_X.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_gp_models(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_fn: Optional[Union[Callable[[int], gpytorch.kernels.Kernel], List[Callable[[int], gpytorch.kernels.Kernel]]]] = None,
    noise_priors: Optional[Union[gpytorch.priors.Prior, List[Optional[gpytorch.priors.Prior]]]] = None,
    min_noise: float = 1e-3,
    min_lengthscale: Optional[float] = None,
) -> ModelListGP:
    """
    Fits a list of GP models for each output dimension and returns a ModelListGP.

    Args:
        X: Input features tensor of shape (N, D), should be normalized to [0,1] range
        Y: Target values tensor of shape (N, M)
        kernel_fn: Optional kernel specification. Can be:
                  - Single function that takes input dimension D and returns a GPyTorch kernel (used for all objectives)
                  - List of functions (one per objective, length M)
                  - If None, defaults to Matern(2.5) kernel with ARD for all objectives
        noise_priors: Optional noise prior specification. Can be:
                     - Single Prior (used for all objectives)
                     - List of Priors (one per objective, length M, each item can be a Prior or None)
                     - If None, uses default noise constraint for all objectives
        min_noise: Lower bound on observation noise during fitting.
        min_lengthscale: Optional lower bound on ARD lengthscales for the default
                         Matern(2.5) kernel when `kernel_fn` is None.

    Returns:
        ModelListGP: A model list containing one SingleTaskGP per output dimension
    """
    # Sanity warnings
    X_min, X_max = X.min(), X.max()
    if X_min < 0.0 or X_max > 1.0:
        import warnings
        warnings.warn(
            f"X is not normalized to [0,1]: [{X_min:.4f}, {X_max:.4f}]",
            UserWarning, stacklevel=2
        )
    # Y_mean, Y_std = Y.mean(), Y.std()
    # if abs(Y_mean) > 0.1 or abs(Y_std - 1.0) > 0.1:
    #     import warnings
    #     warnings.warn(
    #         f"Y not ~ standardized (mean={Y_mean:.4f}, std={Y_std:.4f})",
    #         UserWarning, stacklevel=2
    #     )

    models: List[SingleTaskGP] = []
    D = X.shape[1]
    M = Y.shape[1]

    lengthscale_constraint = (
        gpytorch.constraints.GreaterThan(min_lengthscale)
        if min_lengthscale is not None
        else gpytorch.constraints.Positive()
    )

    # kernel_fn handling: single / list / default
    if kernel_fn is None:
        kernel_fns = [
            (
                lambda d, _ls=lengthscale_constraint: MaternKernel(
                    nu=2.5,
                    ard_num_dims=d,
                    lengthscale_constraint=_ls,
                )
            )
        ] * M
    elif callable(kernel_fn):
        kernel_fns = [kernel_fn] * M
    elif isinstance(kernel_fn, list):
        if len(kernel_fn) != M:
            raise ValueError(f"kernel_fn list length ({len(kernel_fn)}) must match M={M}")
        kernel_fns = kernel_fn
    else:
        raise TypeError("kernel_fn must be a callable, list of callables, or None")

    # noise_priors handling: single / list / None
    if noise_priors is None:
        noise_priors_list = [GammaPrior(1.05, 0.5)] * M
    elif isinstance(noise_priors, list):
        if len(noise_priors) != M:
            raise ValueError(f"noise_priors list length ({len(noise_priors)}) must match M={M}")
        noise_priors_list = noise_priors
    elif hasattr(noise_priors, "log_prob"):
        noise_priors_list = [noise_priors] * M
    else:
        raise TypeError("noise_priors must be a Prior, a list of Priors, or None")

    # move priors to device if possible
    for k in range(M):
        p = noise_priors_list[k]
        if p is not None and hasattr(p, "to"):
            noise_priors_list[k] = p.to(X.device)

    for j in range(M):
        base_kernel = kernel_fns[j](D)
        covar_module = ScaleKernel(base_kernel)

        pj = noise_priors_list[j]
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=pj,
            noise_constraint=gpytorch.constraints.GreaterThan(min_noise),
        )

        gp = SingleTaskGP(X, Y[:, j:j+1], covar_module=covar_module, likelihood=likelihood, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        # Try to fit; never raise
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"[fit_gp_models] obj {j}: first fit failed ({e}); retrying with higher noise floor")
            try:
                # Loosen the floor slightly and retry
                retry_noise = max(1e-2, min_noise)
                gp.likelihood.noise_covar.register_constraint(
                    "raw_noise", gpytorch.constraints.GreaterThan(retry_noise)
                )
                with torch.no_grad():
                    # small positive init can help optimizer
                    gp.likelihood.noise = torch.as_tensor(
                        retry_noise, device=X.device, dtype=X.dtype
                    )
                fit_gpytorch_mll(mll)
            except Exception as e2:
                print(f"[fit_gp_models] obj {j}: retry failed ({e2}); continuing with initial hyperparameters")

        models.append(gp)

    return ModelListGP(*models)


def fit_gp_models_baybe(
    X: torch.Tensor,
    Y: torch.Tensor,
    num_task_parameters: int = 0,
    min_noise: float = 1e-2,
) -> ModelListGP:
    """
    Fit one GP per objective using BayBE default kernel/noise preset heuristics.

    The priors and initial values are interpolated from effective dimensionality
    (D - num_task_parameters), following BayBE's default GP preset.
    """
    X_min, X_max = X.min(), X.max()
    if X_min < 0.0 or X_max > 1.0:
        import warnings
        warnings.warn(
            f"X is not normalized to [0,1]: [{X_min:.4f}, {X_max:.4f}]",
            UserWarning,
            stacklevel=2,
        )

    D = X.shape[1]
    M = Y.shape[1]
    priors_cfg = _baybe_default_priors(D, num_task_parameters=num_task_parameters)

    models: List[SingleTaskGP] = []
    for j in range(M):
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=D,
            lengthscale_prior=priors_cfg["lengthscale_prior"],
        )
        covar_module = ScaleKernel(
            base_kernel,
            outputscale_prior=priors_cfg["outputscale_prior"],
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=priors_cfg["noise_prior"],
            noise_constraint=gpytorch.constraints.GreaterThan(min_noise),
        )

        gp = SingleTaskGP(
            X,
            Y[:, j:j + 1],
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )

        # Set BayBE-style initial values before optimization.
        with torch.no_grad():
            gp.covar_module.base_kernel.lengthscale = torch.as_tensor(
                priors_cfg["lengthscale_init"], device=X.device, dtype=X.dtype
            )
            gp.covar_module.outputscale = torch.as_tensor(
                priors_cfg["outputscale_init"], device=X.device, dtype=X.dtype
            )
            gp.likelihood.noise = torch.as_tensor(
                max(float(priors_cfg["noise_init"]), min_noise),
                device=X.device,
                dtype=X.dtype,
            )

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"[fit_gp_models_baybe] obj {j}: first fit failed ({e}); retrying with higher noise floor")
            try:
                retry_noise = max(1e-2, min_noise)
                gp.likelihood.noise_covar.register_constraint(
                    "raw_noise", gpytorch.constraints.GreaterThan(retry_noise)
                )
                with torch.no_grad():
                    gp.likelihood.noise = torch.as_tensor(
                        retry_noise, device=X.device, dtype=X.dtype
                    )
                fit_gpytorch_mll(mll)
            except Exception as e2:
                print(f"[fit_gp_models_baybe] obj {j}: retry failed ({e2}); continuing with initial hyperparameters")

        models.append(gp)

    return ModelListGP(*models)


def fit_gp_models_baybe_conservative(
    X: torch.Tensor,
    Y: torch.Tensor,
    num_task_parameters: int = 0,
    min_noise: float = 3e-2,
    min_lengthscale: float = 5e-2,
) -> ModelListGP:
    """
    Fit BayBE-style GPs with more conservative noise and lengthscale floors.

    This variant is useful when qNEHVI is driven mainly by posterior uncertainty:
    the higher noise floor and minimum ARD lengthscale reduce overconfident,
    highly local extrapolation without changing the existing fit functions.
    """
    X_min, X_max = X.min(), X.max()
    if X_min < 0.0 or X_max > 1.0:
        import warnings
        warnings.warn(
            f"X is not normalized to [0,1]: [{X_min:.4f}, {X_max:.4f}]",
            UserWarning,
            stacklevel=2,
        )

    D = X.shape[1]
    M = Y.shape[1]
    priors_cfg = _baybe_default_priors(D, num_task_parameters=num_task_parameters)

    lengthscale_prior = priors_cfg["lengthscale_prior"]
    outputscale_prior = priors_cfg["outputscale_prior"]
    noise_prior = priors_cfg["noise_prior"]
    if hasattr(lengthscale_prior, "to"):
        lengthscale_prior = lengthscale_prior.to(X.device)
    if hasattr(outputscale_prior, "to"):
        outputscale_prior = outputscale_prior.to(X.device)
    if hasattr(noise_prior, "to"):
        noise_prior = noise_prior.to(X.device)

    models: List[SingleTaskGP] = []
    for j in range(M):
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=D,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=gpytorch.constraints.GreaterThan(min_lengthscale),
        )
        covar_module = ScaleKernel(
            base_kernel,
            outputscale_prior=outputscale_prior,
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=gpytorch.constraints.GreaterThan(min_noise),
        )

        gp = SingleTaskGP(
            X,
            Y[:, j:j + 1],
            covar_module=covar_module,
            likelihood=likelihood,
            outcome_transform=Standardize(m=1),
        )

        with torch.no_grad():
            gp.covar_module.base_kernel.lengthscale = torch.as_tensor(
                max(float(priors_cfg["lengthscale_init"]), min_lengthscale),
                device=X.device,
                dtype=X.dtype,
            )
            gp.covar_module.outputscale = torch.as_tensor(
                priors_cfg["outputscale_init"], device=X.device, dtype=X.dtype
            )
            gp.likelihood.noise = torch.as_tensor(
                max(float(priors_cfg["noise_init"]), min_noise),
                device=X.device,
                dtype=X.dtype,
            )

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(
                f"[fit_gp_models_baybe_conservative] obj {j}: fit failed ({e}); "
                "continuing with initialized hyperparameters"
            )

        models.append(gp)

    return ModelListGP(*models)


# --------------------------------------------------------------------------------------
# Ready-to-use option factories (so LOOCV works even if user passes nothing)
# --------------------------------------------------------------------------------------

def _kernel_factory(
    kernel_cls: Callable[..., gpytorch.kernels.Kernel],
    *,
    min_lengthscale: float,
    **kernel_kwargs: Any,
) -> Callable[[int], gpytorch.kernels.Kernel]:
    """Build a kernel factory with a stable ARD lengthscale floor."""

    def _factory(d: int) -> gpytorch.kernels.Kernel:
        return kernel_cls(
            ard_num_dims=d,
            lengthscale_constraint=gpytorch.constraints.GreaterThan(min_lengthscale),
            **kernel_kwargs,
        )

    return _factory


def default_kernel_labels() -> List[str]:
    """Human-readable labels aligned with `default_kernel_options()`."""
    return ["RBF", "Matern_nu1.5", "Matern_nu2.5"]


def default_kernel_options(
    min_lengthscale: float = 5e-2,
) -> List[Callable[[int], gpytorch.kernels.Kernel]]:
    """
    Return predefined kernel factories for LOOCV.

    Each kernel uses ARD and a minimum lengthscale floor to avoid the very sharp,
    over-interpolating fits that were destabilizing LOOCV on small batches.
    """
    return [
        _kernel_factory(RBFKernel, min_lengthscale=min_lengthscale),
        _kernel_factory(MaternKernel, min_lengthscale=min_lengthscale, nu=1.5),
        _kernel_factory(MaternKernel, min_lengthscale=min_lengthscale, nu=2.5),
    ]


def default_noise_labels() -> List[str]:
    """Human-readable labels aligned with `default_noise_options()`."""
    return [
        "BayBE_Gamma",
        "Gamma_1.05_0.5",
        "LogNormal_-3.0_1.0",
        "LogNormal_-2.0_0.5",
    ]


def default_noise_options(
    device: Optional[torch.device] = None,
    input_dims: int = 8,
    num_task_parameters: int = 0,
) -> List[gpytorch.priors.Prior]:
    """
    Return predefined noise priors for LOOCV.

    Args:
        device: Device for the returned priors. If None, priors remain on CPU and
                `fit_gp_models` will move them to the training device.
        input_dims: Input dimensionality used to build the BayBE-style prior.
        num_task_parameters: Task-parameter count passed to BayBE heuristics.
    """
    priors_cfg = _baybe_default_priors(input_dims, num_task_parameters=num_task_parameters)
    priors = [
        priors_cfg["noise_prior"],
        GammaPrior(1.05, 0.5),
        LogNormalPrior(-3.0, 1.0),
        LogNormalPrior(-2.0, 0.5),
    ]
    if device is None:
        return priors
    return [prior.to(device) if hasattr(prior, "to") else prior for prior in priors]


# --------------------------------------------------------------------------------------
# LOOCV model selection across (kernel x noise) combinations
# --------------------------------------------------------------------------------------

def loocv_select_models(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    objective_names: Optional[List[str]] = None,
    kernel_options: Optional[List[Callable[[int], gpytorch.kernels.Kernel]]] = None,
    noise_options: Optional[List[Optional[gpytorch.priors.Prior]]] = None,
    kernel_labels: Optional[List[str]] = None,
    noise_labels: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    min_noise: float = 1e-2,
    min_lengthscale: float = 5e-2,
    input_dims_for_noise: Optional[int] = None,
    num_task_parameters: int = 0,
) -> Tuple[ModelListGP, pd.DataFrame]:
    """
    Perform leave-one-out cross-validation over (kernel x noise_prior) combinations.

    Args:
        train_X: Training input features tensor of shape (N, D), normalized to [0,1].
        train_Y: Training target values tensor of shape (N, M).
        objective_names: Optional objective names (length M).
        kernel_options: Optional kernel factories. Defaults to `default_kernel_options()`.
        noise_options: Optional noise priors. Defaults to `default_noise_options()`.
        kernel_labels: Optional labels aligned with `kernel_options`.
        noise_labels: Optional labels aligned with `noise_options`.
        device: Optional device. If None, uses `train_X.device`.
        min_noise: Minimum observation noise passed to `fit_gp_models`.
        min_lengthscale: Minimum ARD lengthscale passed to `fit_gp_models`.
        input_dims_for_noise: Dimensionality used for BayBE noise prior defaults.
        num_task_parameters: Task-parameter count for BayBE noise prior defaults.

    Returns:
        best_model: ModelListGP with the best model per objective (refit on full data).
        results_df: DataFrame with LOOCV scores for each combination/objective.
    """
    device = _resolve_training_device(train_X, device)
    train_X = train_X.to(device=device)
    train_Y = train_Y.to(device=device)

    X_min, X_max = train_X.min(), train_X.max()
    if X_min < 0.0 or X_max > 1.0:
        import warnings
        warnings.warn(
            f"train_X not in [0,1]: [{X_min:.4f}, {X_max:.4f}]",
            UserWarning, stacklevel=2
        )

    N, D = train_X.shape
    M = train_Y.shape[1]
    names = objective_names or [f"obj{j}" for j in range(M)]

    if kernel_options is None:
        kernel_options = default_kernel_options(min_lengthscale=min_lengthscale)
    if noise_options is None:
        noise_options = default_noise_options(
            device=device,
            input_dims=input_dims_for_noise or D,
            num_task_parameters=num_task_parameters,
        )
    if kernel_labels is None:
        kernel_labels = default_kernel_labels()
        if len(kernel_labels) != len(kernel_options):
            kernel_labels = [f"kernel_{i}" for i in range(len(kernel_options))]
    if noise_labels is None:
        noise_labels = default_noise_labels()
        if len(noise_labels) != len(noise_options):
            noise_labels = [f"noise_{i}" for i in range(len(noise_options))]

    results: List[Dict[str, object]] = []
    best_rmse = [np.inf] * M
    best_config: List[Optional[Tuple[int, int]]] = [None] * M

    for kernel_idx, kernel_fn in enumerate(kernel_options):
        kernel_name = kernel_labels[kernel_idx]
        for noise_idx, shared_prior in enumerate(noise_options):
            noise_name = noise_labels[noise_idx]
            preds_all = [[] for _ in range(M)]
            actuals_all = [[] for _ in range(M)]

            for i in range(N):
                X_cv = torch.cat([train_X[:i], train_X[i + 1 :]], dim=0)
                Y_cv = torch.cat([train_Y[:i], train_Y[i + 1 :]], dim=0)
                x_ho = train_X[i : i + 1]

                try:
                    model_cv = fit_gp_models(
                        X_cv,
                        Y_cv,
                        kernel_fn=kernel_fn,
                        noise_priors=shared_prior,
                        min_noise=min_noise,
                        min_lengthscale=min_lengthscale,
                    )
                    for j, gp in enumerate(model_cv.models):
                        try:
                            gp.eval()
                            if hasattr(gp.likelihood, "eval"):
                                gp.likelihood.eval()
                            with torch.no_grad():
                                preds_all[j].append(gp.posterior(x_ho).mean.item())
                            actuals_all[j].append(train_Y[i, j].item())
                        except Exception as pred_e:
                            print(
                                f"[LOOCV] pred fail: {kernel_name}/{noise_name}, "
                                f"fold {i}, obj {j} ({pred_e})"
                            )
                            preds_all[j].append(np.nan)
                            actuals_all[j].append(train_Y[i, j].item())
                except Exception as e:
                    print(
                        f"[LOOCV] fold {i} fit failed for "
                        f"{kernel_name}/{noise_name} ({e})"
                    )
                    for j in range(M):
                        preds_all[j].append(np.nan)
                        actuals_all[j].append(train_Y[i, j].item())

            for j in range(M):
                mask = ~np.isnan(preds_all[j])
                if np.sum(mask) < 2:
                    results.append({
                        "kernel": kernel_name,
                        "noise_prior": noise_name,
                        "objective": names[j],
                        "r2": np.nan,
                        "rmse": np.nan,
                    })
                    continue

                valid_y = [a for a, m in zip(actuals_all[j], mask) if m]
                valid_p = [p for p, m in zip(preds_all[j], mask) if m]
                r2 = r2_score(valid_y, valid_p)
                rmse = root_mean_squared_error(valid_y, valid_p)

                results.append({
                    "kernel": kernel_name,
                    "noise_prior": noise_name,
                    "objective": names[j],
                    "r2": round(float(r2), 3),
                    "rmse": round(float(rmse), 3),
                })

                if rmse < best_rmse[j]:
                    best_rmse[j] = rmse
                    best_config[j] = (kernel_idx, noise_idx)

    best_models: List[Optional[SingleTaskGP]] = [None] * M
    for j in range(M):
        if best_config[j] is not None:
            kernel_idx, noise_idx = best_config[j]
            try:
                model_full = fit_gp_models(
                    train_X,
                    train_Y,
                    kernel_fn=kernel_options[kernel_idx],
                    noise_priors=noise_options[noise_idx],
                    min_noise=min_noise,
                    min_lengthscale=min_lengthscale,
                )
                best_models[j] = model_full.models[j]
            except Exception as e_full:
                print(
                    f"[LOOCV] refit on full data failed for obj {j} "
                    f"({kernel_labels[kernel_idx]}/{noise_labels[noise_idx]}): {e_full}"
                )

    for j in range(M):
        if best_models[j] is None:
            print(f"[LOOCV] objective {j} had no valid config; constructing fallback GP.")
            try:
                base_kernel = MaternKernel(
                    nu=2.5,
                    ard_num_dims=D,
                    lengthscale_constraint=gpytorch.constraints.GreaterThan(min_lengthscale),
                )
                covar_module = ScaleKernel(base_kernel)
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(min_noise)
                )
                fallback = SingleTaskGP(
                    train_X,
                    train_Y[:, j : j + 1],
                    covar_module=covar_module,
                    likelihood=likelihood,
                    outcome_transform=Standardize(m=1),
                )
                try:
                    mll = ExactMarginalLogLikelihood(fallback.likelihood, fallback)
                    fit_gpytorch_mll(mll)
                except Exception as e_fit:
                    print(f"[LOOCV] fallback fit skipped (obj {j}): {e_fit}")
                best_models[j] = fallback
            except Exception as e_last:
                base_kernel = MaternKernel(
                    nu=2.5,
                    ard_num_dims=D,
                    lengthscale_constraint=gpytorch.constraints.GreaterThan(min_lengthscale),
                )
                covar_module = ScaleKernel(base_kernel)
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(min_noise)
                )
                best_models[j] = SingleTaskGP(
                    train_X,
                    train_Y[:, j : j + 1],
                    covar_module=covar_module,
                    likelihood=likelihood,
                    outcome_transform=Standardize(m=1),
                )
                print(f"[LOOCV] constructed raw fallback for obj {j}: {e_last}")

    results_df = pd.DataFrame(results)
    best_modellist = ModelListGP(*best_models)
    return best_modellist, results_df


def posterior_report(
    model,
    X: torch.Tensor,
):
    """
    Computes posterior predictions on input X and returns comprehensive results including predictions, uncertainties, and performance metrics.

    Args:
        model: Fitted ModelListGP model
        X: Input features tensor of shape (N, D), should be normalized to [0,1] range

    Returns:
        pred_mean: Array of shape (N, M) containing mean predictions for each objective
        pred_std: Array of shape (N, M) containing standard deviation predictions for each objective
    """
    # Validate that X is normalized between 0 and 1
    X_min, X_max = X.min(), X.max()
    
    X_min, X_max = X.min(), X.max()
    if X_min < 0.0 or X_max > 1.0:
        import warnings
        warnings.warn(
            f"X is not normalized to [0,1] range. Current range: [{X_min:.4f}, {X_max:.4f}].",
            UserWarning, stacklevel=2
        )

    with torch.no_grad():
        post = model.posterior(X)
        pred_mean_t = post.mean
        pred_std_t  = torch.sqrt(post.variance)

    pred_mean, pred_std = torch_to_np(pred_mean_t, pred_std_t)

    return pred_mean, pred_std
