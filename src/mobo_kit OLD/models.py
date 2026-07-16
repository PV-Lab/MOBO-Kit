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


def fit_gp_models(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_fn: Optional[Union[Callable[[int], gpytorch.kernels.Kernel], List[Callable[[int], gpytorch.kernels.Kernel]]]] = None,
    noise_priors: Optional[Union[gpytorch.priors.Prior, List[Optional[gpytorch.priors.Prior]]]] = None,
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

    # kernel_fn handling: single / list / default
    if kernel_fn is None:
        kernel_fns = [(lambda d: MaternKernel(nu=2.5, ard_num_dims=d))]*M # + ConstantKernel())] * M
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
        noise_priors_list = [None] * M
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
            noise_constraint=gpytorch.constraints.GreaterThan(1e-3),
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
                gp.likelihood.noise_covar.register_constraint(
                    "raw_noise", gpytorch.constraints.GreaterThan(1e-2)
                )
                with torch.no_grad():
                    # small positive init can help optimizer
                    gp.likelihood.noise = torch.as_tensor(1e-2, device=X.device, dtype=X.dtype)
                fit_gpytorch_mll(mll)
            except Exception as e2:
                print(f"[fit_gp_models] obj {j}: retry failed ({e2}); continuing with initial hyperparameters")

        models.append(gp)

    return ModelListGP(*models)


# --------------------------------------------------------------------------------------
# Ready-to-use option factories (so LOOCV works even if user passes nothing)
# --------------------------------------------------------------------------------------

def default_kernel_options() -> List[Callable[[int], gpytorch.kernels.Kernel]]:
    """
    Returns a list of predefined kernel functions for different smoothness parameters.

    Returns:
        List[Callable]: List of kernel functions.
                       Order: RBF, Matern(0.5), Matern(1.5), Matern(2.5)
    """
    return [
        lambda d: RBFKernel(ard_num_dims=d),
        lambda d: MaternKernel(nu=0.5, ard_num_dims=d),
        lambda d: MaternKernel(nu=1.5, ard_num_dims=d),
        lambda d: MaternKernel(nu=2.5, ard_num_dims=d),
    ]

def default_noise_options(device: Optional[torch.device] = None) -> List[Optional[gpytorch.priors.Prior]]:
    """
    Returns a list of predefined noise priors for GP likelihoods.

    Args:
        device: Optional torch device to move priors to. If None, priors remain on CPU.

    Returns:
        List[Optional[Prior]]: List of noise priors.
                              Order: None, LogNormal(-4,0.5)
    """
    # Note: priors are moved to the right device again inside fit_gp_models
    return [
        None,
        GammaPrior(1.1, 0.05) if device is None else GammaPrior(1.1, 0.05).to(device),
        #LogNormalPrior(-4.0, 0.5) if device is None else LogNormalPrior(-4.0, 0.5).to(device),
        #LogNormalPrior(-3.0, 1.0) if device is None else LogNormalPrior(-3.0, 1.0).to(device),
        #LogNormalPrior(-2.0, 0.5) if device is None else LogNormalPrior(-2.0, 0.5).to(device),
    ]


# --------------------------------------------------------------------------------------
# LOOCV model selection across (kernel x noise) combinations
# --------------------------------------------------------------------------------------

def loocv_select_models(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    objective_names: Optional[List[str]] = None,
    kernel_options: Optional[List[Callable[[int], gpytorch.kernels.Kernel]]] = None,
    noise_options: Optional[List[Optional[gpytorch.priors.Prior]]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[ModelListGP, pd.DataFrame]:
    """
    Performs Leave-One-Out Cross-Validation over combinations of (kernel, noise_prior) to select the best model configuration.

    Args:
        train_X: Training input features tensor of shape (N, D), should be normalized to [0,1] range
        train_Y: Training target values tensor of shape (N, M)
        objective_names: Optional list of names for each objective (length M). Defaults to ["obj0", "obj1", ...]
        kernel_options: Optional list of kernel functions.
                       If None, uses default_kernel_options()
        noise_options: Optional list of noise priors.
                      If None, uses default_noise_options(device)
        device: Optional torch device for building priors upfront

    Returns:
        best_model: ModelListGP containing the best model per objective (fitted on full data)
        results_df: DataFrame with results for each (kernel, noise, objective) combination,
                   including r2 and rmse scores. Columns: ['kernel', 'noise_prior', 'objective', 'r2', 'rmse']
    """
    X_min, X_max = train_X.min(), train_X.max()
    if X_min < 0.0 or X_max > 1.0:
        import warnings
        warnings.warn(
            f"train_X not in [0,1]: [{X_min:.4f}, {X_max:.4f}]",
            UserWarning, stacklevel=2
        )

    if kernel_options is None:
        kernel_options = default_kernel_options()
    if noise_options is None:
        noise_options = default_noise_options(device)

    N, D = train_X.shape
    M = train_Y.shape[1]
    names = objective_names or [f"obj{j}" for j in range(M)]

    results: List[Dict[str, object]] = []
    best_rmse = [np.inf] * M
    best_models: List[Optional[SingleTaskGP]] = [None] * M

    for kernel_idx, kernel_fn in enumerate(kernel_options):
        for noise_idx, shared_prior in enumerate(noise_options):
            preds_all = [[] for _ in range(M)]
            actuals_all = [[] for _ in range(M)]

            for i in range(N):
                X_cv = torch.cat([train_X[:i], train_X[i+1:]], dim=0)
                Y_cv = torch.cat([train_Y[:i], train_Y[i+1:]], dim=0)

                try:
                    model_cv = fit_gp_models(X_cv, Y_cv, kernel_fn=kernel_fn, noise_priors=shared_prior)
                    x_ho = train_X[i:i+1]
                    for j, gp in enumerate(model_cv.models):
                        try:
                            with torch.no_grad():
                                preds_all[j].append(gp.posterior(x_ho).mean.item())  # original units
                        except Exception as pred_e:
                            print(f"[LOOCV] pred fail: fold {i}, obj {j} ({pred_e})")
                            preds_all[j].append(np.nan)
                        actuals_all[j].append(train_Y[i, j].item())
                except Exception as e:
                    print(f"[LOOCV] fold {i} fit failed ({e})")
                    for j in range(M):
                        preds_all[j].append(np.nan)
                        actuals_all[j].append(train_Y[i, j].item())

            # Score this (kernel, prior)
            for j in range(M):
                mask = ~np.isnan(preds_all[j])
                if np.sum(mask) < 2:
                    results.append({
                        "kernel": f"kernel_{kernel_idx}",
                        "noise_prior": f"noise_{noise_idx}",
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
                    "kernel": f"kernel_{kernel_idx}",
                    "noise_prior": f"noise_{noise_idx}",
                    "objective": names[j],
                    "r2": round(float(r2), 3),
                    "rmse": round(float(rmse), 3),
                })

                if rmse < best_rmse[j]:
                    best_rmse[j] = rmse
                    try:
                        model_full = fit_gp_models(train_X, train_Y, kernel_fn=kernel_fn, noise_priors=shared_prior)
                        best_models[j] = model_full.models[j]
                    except Exception as e_full:
                        print(f"[LOOCV] refit on full data failed for obj {j}: {e_full}")

    # Ensure we return a model per objective
    for j in range(M):
        if best_models[j] is None:
            print(f"[LOOCV] objective {j} had no valid config; constructing fallback GP.")
            try:
                base_kernel = MaternKernel(nu=2.5, ard_num_dims=D)
                covar_module = ScaleKernel(base_kernel)
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
                )
                fallback = SingleTaskGP(
                    train_X, train_Y[:, j:j+1],
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
                base_kernel = MaternKernel(nu=2.5, ard_num_dims=D)
                covar_module = ScaleKernel(base_kernel)
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
                )
                best_models[j] = SingleTaskGP(
                    train_X, train_Y[:, j:j+1],
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
