from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import gpytorch
import pandas as pd

from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.priors import GammaPrior, LogNormalPrior

from sklearn.metrics import r2_score, root_mean_squared_error
from .utils import torch_to_np


# --------------------------------------------------------------------------------------
# Core: fit one GP per output and wrap in a ModelListGP (matches your notebook function)
# --------------------------------------------------------------------------------------

def fit_gp_models(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_fn: Optional[Callable[[int], gpytorch.kernels.Kernel]] = None,
    noise_priors: Optional[List[Optional[gpytorch.priors.Prior]]] = None,
) -> ModelListGP:
    """
    Fits a list of GP models for each output dimension and returns a ModelListGP.

    Args:
        X: torch tensor (N, D)  - already on your chosen device/dtype
        Y: torch tensor (N, M)
        kernel_fn: function taking D and returning a GPyTorch kernel
                   (e.g., lambda d: MaternKernel(nu=2.5, ard_num_dims=d))
        noise_priors: list of length M, each item is a gpytorch Prior or None

    Returns:
        ModelListGP with one SingleTaskGP per output
    """
    models: List[SingleTaskGP] = []
    D = X.shape[1]
    M = Y.shape[1]

    for j in range(M):
        base_kernel = kernel_fn(D) if kernel_fn is not None else RBFKernel(ard_num_dims=D)
        covar_module = ScaleKernel(base_kernel)

        noise_prior_j = noise_priors[j] if (noise_priors is not None and j < len(noise_priors)) else None
        if noise_prior_j is not None and hasattr(noise_prior_j, "to"):
            noise_prior_j = noise_prior_j.to(X.device)  # ensure prior on same device

        if noise_prior_j is not None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior_j)
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()

        gp = SingleTaskGP(X, Y[:, j:j+1], covar_module=covar_module, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        models.append(gp)

    return ModelListGP(*models)


# --------------------------------------------------------------------------------------
# Ready-to-use option factories (so LOOCV works even if user passes nothing)
# --------------------------------------------------------------------------------------

def default_matern_options() -> Dict[str, Callable[[int], gpytorch.kernels.Kernel]]:
    return {
        "Matern_0.5":  lambda d: MaternKernel(nu=0.5, ard_num_dims=d),
        "Matern_1.5":  lambda d: MaternKernel(nu=1.5, ard_num_dims=d),
        "Matern_2.5":  lambda d: MaternKernel(nu=2.5, ard_num_dims=d),
    }

def default_noise_options(device: Optional[torch.device] = None) -> Dict[str, Optional[gpytorch.priors.Prior]]:
    # Note: priors are moved to the right device again inside fit_gp_models
    return {
        "None":                None,
        "Gamma(1.1,0.05)":     GammaPrior(1.1, 0.05) if device is None else GammaPrior(1.1, 0.05).to(device),
        "LogNormal(-4,0.5)":   LogNormalPrior(-4.0, 0.5) if device is None else LogNormalPrior(-4.0, 0.5).to(device),
        "LogNormal(-3,1.0)":   LogNormalPrior(-3.0, 1.0) if device is None else LogNormalPrior(-3.0, 1.0).to(device),
        "LogNormal(-2,0.5)":   LogNormalPrior(-2.0, 0.5) if device is None else LogNormalPrior(-2.0, 0.5).to(device),
    }


# --------------------------------------------------------------------------------------
# LOOCV model selection across (kernel x noise) with optional fixed per-objective prior
# --------------------------------------------------------------------------------------

def loocv_select_models(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    objective_names: Optional[List[str]] = None,
    matern_options: Optional[Dict[str, Callable[[int], gpytorch.kernels.Kernel]]] = None,
    noise_options: Optional[Dict[str, Optional[gpytorch.priors.Prior]]] = None,
    fixed_noise_priors: Optional[List[Optional[gpytorch.priors.Prior]]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[ModelListGP, pd.DataFrame]:
    """
    Leave-One-Out CV over combinations of (kernel, shared_noise_prior).
    Optionally pin certain objectives to a fixed prior via `fixed_noise_priors`.

    Args:
        train_X: (N, D) torch tensor (on device)
        train_Y: (N, M) torch tensor (on device)
        objective_names: list of names length M (for results table); defaults to ["obj0", ...]
        matern_options: dict name->kernel_fn(d)
        noise_options: dict name->prior (applied to all objectives unless overridden)
        fixed_noise_priors: list length M; if entry is not None, that prior is used
                            for that objective regardless of the shared prior.
        device: torch.device; only needed to build priors upfront (optional)

    Returns:
        best_model:   ModelListGP with best-per-objective models (fit on full data)
        results_df:   pd.DataFrame with rows for each (kernel, noise, objective) combo, incl. R2 & RMSE
    """
    if matern_options is None:
        matern_options = default_matern_options()
    if noise_options is None:
        noise_options = default_noise_options(device)

    N = train_X.shape[0]
    M = train_Y.shape[1]
    names = objective_names or [f"obj{j}" for j in range(M)]

    results: List[Dict[str, object]] = []
    best_rmse = [np.inf] * M
    best_models: List[Optional[SingleTaskGP]] = [None] * M

    for kernel_name, kernel_fn in matern_options.items():
        for noise_name, shared_noise_prior in noise_options.items():
            preds_all = [[] for _ in range(M)]
            actuals_all = [[] for _ in range(M)]

            for i in range(N):
                # LOO split
                X_cv = torch.cat([train_X[:i], train_X[i+1:]], dim=0)
                Y_cv = torch.cat([train_Y[:i], train_Y[i+1:]], dim=0)

                # Build per-objective noise prior list
                if fixed_noise_priors is not None and len(fixed_noise_priors) == M:
                    noise_prior_list = [
                        (fixed_noise_priors[j] if fixed_noise_priors[j] is not None else shared_noise_prior)
                        for j in range(M)
                    ]
                else:
                    noise_prior_list = [shared_noise_prior] * M

                model_cv = fit_gp_models(X_cv, Y_cv, kernel_fn=kernel_fn, noise_priors=noise_prior_list)

                # Predict held-out point with each objective GP
                x_ho = train_X[i:i+1]
                for j, gp in enumerate(model_cv.models):
                    pred = gp.posterior(x_ho).mean.item()
                    preds_all[j].append(pred)
                    actuals_all[j].append(train_Y[i, j].item())

            # Score each objective for this (kernel, noise) combo
            for j in range(M):
                r2 = r2_score(actuals_all[j], preds_all[j])
                rmse = root_mean_squared_error(actuals_all[j], preds_all[j])

                results.append({
                    "Kernel": kernel_name,
                    "NoisePrior": noise_name,
                    "Objective": names[j],
                    "R2": round(r2, 3),
                    "RMSE": round(rmse, 3),
                })

                # Track best model per objective; refit on full data when improved
                if rmse < best_rmse[j]:
                    best_rmse[j] = rmse
                    if fixed_noise_priors is not None and len(fixed_noise_priors) == M:
                        noise_prior_list_full = [
                            (fixed_noise_priors[j2] if fixed_noise_priors[j2] is not None else shared_noise_prior)
                            for j2 in range(M)
                        ]
                    else:
                        noise_prior_list_full = [shared_noise_prior] * M

                    model_full = fit_gp_models(train_X, train_Y, kernel_fn=kernel_fn, noise_priors=noise_prior_list_full)
                    best_models[j] = model_full.models[j]

    results_df = pd.DataFrame(results)
    best_modellist = ModelListGP(*best_models)
    return best_modellist, results_df


def posterior_report(
    model,                         # ModelListGP (fitted)
    X: torch.Tensor,               # (N, D) on the same device/dtype as training
    Y_scaled: torch.Tensor,        # (N, M) scaled to [0,1] like training targets
    Y_min: np.ndarray,             # (M,) per-objective mins (from y_minmax_np)
    Y_max: np.ndarray,             # (M,) per-objective maxes (from y_minmax_np)
    objective_names: list[str] | None = None,
    add_residuals: bool = False,
    add_zscores: bool = False,
):
    """
    Compute posterior on X and return:
      - df: DataFrame with columns [True[obj], Pred[obj], Std[obj], Residual[obj], Z[obj]]
            (Residual/Z included if enabled)
      - metrics_df: R² and RMSE per objective (in original units)

    Notes:
      * We unnormalize means as  y = y_scaled * (max - min) + min
      * We unnormalize stds as   sigma = sigma_scaled * (max - min)   (no +min)
    """
    # Posterior on training (or any) X
    post = model.posterior(X)
    pred_mean_t = post.mean
    pred_std_t  = torch.sqrt(post.variance)

    # Convert to numpy (helper handles detach/cpu)
    pred_mean, pred_std, true_scaled = torch_to_np(pred_mean_t, pred_std_t, Y_scaled)

    # Unnormalize with per-objective ranges
    Y_min = np.asarray(Y_min, dtype=float)
    Y_max = np.asarray(Y_max, dtype=float)
    scale = (Y_max - Y_min).astype(float)

    pred_mean_unnorm = pred_mean * scale + Y_min
    pred_std_unnorm  = pred_std * scale
    true_Y           = true_scaled * scale + Y_min

    N, M = true_Y.shape
    names = objective_names or [f"obj{j}" for j in range(M)]

    # Build a wide report DataFrame
    cols = {}
    for j, name in enumerate(names):
        cols[f"True[{name}]"] = true_Y[:, j]
        cols[f"Pred[{name}]"] = pred_mean_unnorm[:, j]
        cols[f"Std[{name}]"]  = pred_std_unnorm[:, j]

        if add_residuals:
            cols[f"Residual[{name}]"] = pred_mean_unnorm[:, j] - true_Y[:, j]
        if add_zscores:
            # guard against very small std
            denom = np.where(pred_std_unnorm[:, j] > 1e-12, pred_std_unnorm[:, j], np.nan)
            cols[f"Z[{name}]"] = (pred_mean_unnorm[:, j] - true_Y[:, j]) / denom

    df = pd.DataFrame(cols)

    # Metrics per objective (original units)
    rows = []
    for j, name in enumerate(names):
        r2 = r2_score(true_Y[:, j], pred_mean_unnorm[:, j])
        rmse = root_mean_squared_error(true_Y[:, j], pred_mean_unnorm[:, j])
        rows.append({"Objective": name, "R2": round(float(r2), 3), "RMSE": round(float(rmse), 3)})
    metrics_df = pd.DataFrame(rows, columns=["Objective", "R2", "RMSE"])

    return df, metrics_df
