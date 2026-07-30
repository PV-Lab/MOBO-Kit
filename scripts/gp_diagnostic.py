"""Phase 1.1 baseline GP diagnostic.

Answers one question: is the GP learning anything from the 15 R0 observations?

For each objective and each model variant it reports
  * the fitted ARD lengthscales, outputscale and noise;
  * exact leave-one-out MAE / RMSE / R2 / Spearman / coverage;
  * the spread of the posterior mean over a Sobol sample of the design space,
    as a fraction of the observed range of that objective.

A GP that has learned nothing shows large lengthscales, Spearman near zero and
near-zero posterior-mean spread: the posterior has collapsed to its prior mean.

Reads the workbook read-only. Writes nothing except an optional CSV.

Usage:
    python scripts/gp_diagnostic.py --workbook "local_inputs/Summary Table.xlsx"
    python scripts/gp_diagnostic.py --variants current dim_scaled_prior --csv out.csv
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from openpyxl import load_workbook
from scipy.stats import qmc

from mobo_kit.model_validation import compute_prediction_metrics

# Canonical grid, configs/campaign_d2d_perovskite.yaml. (name, start, stop)
DESIGN = (
    ("speed_1", 1000.0, 6000.0),
    ("time_1", 5.0, 50.0),
    ("speed_2", 0.0, 5000.0),
    ("time_2", 10.0, 60.0),
    ("precur_conc", 1.0, 2.0),
    ("precur_vol", 40.0, 200.0),
    ("anneal_temp", 100.0, 185.0),
    ("anneal_time", 10.0, 60.0),
    ("anti_vol", 100.0, 200.0),
    ("anti_time", 9.0, 25.0),
)
INPUT_NAMES = tuple(d[0] for d in DESIGN)
OBJECTIVE_NAMES = ("Uniformity", "Optoelectronic", "Thickness")

# 0-based workbook column offsets: inputs B:K, objectives Z/AA/AB.
INPUT_COLS = tuple(range(1, 11))
OBJECTIVE_COLS = (25, 26, 27)

DTYPE = torch.double


# --------------------------------------------------------------------------- #
# model variants
# --------------------------------------------------------------------------- #


def _build_current(X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    """The retired contract, now model_validation.LEGACY_NO_PRIOR.

    ScaleKernel(Matern 2.5 ARD) with no lengthscale prior. Kept here so the
    before/after comparison stays runnable from one script.
    """
    covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=X.shape[1]))
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-3))
    return SingleTaskGP(
        X,
        y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )


def _build_conservative(X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    """The repo's 'conservative' variant: noise floor 1e-2, lengthscale floor 0.05."""
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=X.shape[1],
            lengthscale_constraint=GreaterThan(0.05),
        )
    )
    likelihood = GaussianLikelihood(noise_constraint=GreaterThan(0.01))
    return SingleTaskGP(
        X,
        y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )


def _build_dim_scaled_prior(X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    """Keep Matern 2.5 ARD, add BoTorch's dimension-scaled LogNormal lengthscale prior.

    The prior is LogNormal(loc=sqrt(2) + log(d)/2, scale=sqrt(3)), which at d=10
    concentrates lengthscales around exp(loc) ~ 12.9 in raw units but with enough
    mass at moderate values to stop the unbounded drift the prior-free fit shows.
    """
    base = get_covar_module_with_dim_scaled_prior(
        ard_num_dims=X.shape[1], use_rbf_kernel=False
    )
    covar_module = ScaleKernel(base)
    likelihood = get_gaussian_likelihood_with_lognormal_prior()
    return SingleTaskGP(
        X,
        y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )


def _build_lengthscale_prior_only(X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    """Lengthscale prior but a bare noise floor: the degenerate configuration.

    Kept so the outputscale-collapse mode stays reproducible from this script.
    """
    base = get_covar_module_with_dim_scaled_prior(
        ard_num_dims=X.shape[1], use_rbf_kernel=False
    )
    return SingleTaskGP(
        X,
        y,
        covar_module=ScaleKernel(base),
        likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-3)),
        outcome_transform=Standardize(m=1),
    )


def _build_botorch_default(X: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
    """Pure BoTorch 0.15.1 default: RBF ARD + dim-scaled LogNormal lengthscale prior,
    LogNormal(-4, 1) noise prior, no ScaleKernel. Nothing overridden."""
    return SingleTaskGP(X, y, outcome_transform=Standardize(m=1))


BUILDERS = {
    "current": _build_current,
    "conservative": _build_conservative,
    "dim_scaled_prior": _build_dim_scaled_prior,
    "lengthscale_prior_only": _build_lengthscale_prior_only,
    "botorch_default": _build_botorch_default,
}


def _fit(model: SingleTaskGP) -> SingleTaskGP:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_gpytorch_mll(mll)
    return model


# --------------------------------------------------------------------------- #
# hyperparameter readout
# --------------------------------------------------------------------------- #


@dataclass
class Hypers:
    lengthscales: np.ndarray
    outputscale: float | None
    noise: float


def _read_hypers(model: SingleTaskGP) -> Hypers:
    covar = model.covar_module
    if isinstance(covar, ScaleKernel):
        ls = covar.base_kernel.lengthscale
        outputscale = float(covar.outputscale.detach().reshape(-1)[0])
    else:  # BoTorch default returns a bare kernel
        ls = covar.lengthscale
        outputscale = None
    noise = float(model.likelihood.noise.detach().reshape(-1)[0])
    return Hypers(
        lengthscales=ls.detach().cpu().numpy().reshape(-1).copy(),
        outputscale=outputscale,
        noise=noise,
    )


def _posterior(
    model: SingleTaskGP, X: torch.Tensor, *, observation_noise: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and sd.

    Interval coverage and NLPD must use the *predictive* sd, which includes the
    observation noise; the latent sd alone understates the interval and makes a
    well-calibrated model look overconfident.  Posterior-mean spread uses the
    latent sd, since noise is constant across the design space.
    """
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = model.posterior(X, observation_noise=observation_noise)
        mean = post.mean.detach().cpu().numpy().reshape(-1)
        std = post.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy().reshape(-1)
    return mean, std


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #


def loocv(
    X: np.ndarray, y: np.ndarray, builder, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact leave-one-out: N fits on N-1 rows, each predicting the held-out row."""
    n = X.shape[0]
    mean = np.empty(n)
    std = np.empty(n)
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        torch.manual_seed(seed)
        Xt = torch.tensor(X[keep], dtype=DTYPE)
        yt = torch.tensor(y[keep], dtype=DTYPE).unsqueeze(-1)
        model = _fit(builder(Xt, yt))
        m, s = _posterior(
            model, torch.tensor(X[i : i + 1], dtype=DTYPE), observation_noise=True
        )
        mean[i], std[i] = m[0], s[0]
    return mean, std


def posterior_spread(
    model: SingleTaskGP, d: int, n: int, seed: int
) -> tuple[float, float]:
    """(max-min, std) of the posterior mean over a Sobol sample of the unit cube."""
    sob = qmc.Sobol(d=d, scramble=True, seed=seed)
    P = sob.random(n)
    mean, _ = _posterior(model, torch.tensor(P, dtype=DTYPE))
    return float(mean.max() - mean.min()), float(mean.std())


# --------------------------------------------------------------------------- #


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ws = load_workbook(path, data_only=True)["Sheet1"]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            break
        rows.append(r)
    ids = np.array([int(r[0]) for r in rows])
    X_phys = np.array([[float(r[j]) for j in INPUT_COLS] for r in rows])
    Y = np.array([[float(r[j]) for j in OBJECTIVE_COLS] for r in rows])
    lo = np.array([d[1] for d in DESIGN])
    hi = np.array([d[2] for d in DESIGN])
    X = (X_phys - lo) / (hi - lo)
    if X.min() < -1e-9 or X.max() > 1 + 1e-9:
        raise ValueError("Inputs fall outside the declared design bounds.")
    return ids, X, Y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook", default="local_inputs/Summary Table.xlsx")
    ap.add_argument(
        "--variants", nargs="+", default=["current"], choices=sorted(BUILDERS)
    )
    ap.add_argument("--sobol-n", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--csv", default=None, help="optional path for a tidy metrics CSV")
    args = ap.parse_args()

    ids, X, Y = load_data(Path(args.workbook))
    n, d = X.shape
    print(f"workbook : {args.workbook}")
    print(f"data     : {n} observations, {d} inputs, {Y.shape[1]} objectives")
    print(f"samples  : {ids.tolist()}")

    records = []
    for variant in args.variants:
        builder = BUILDERS[variant]
        print(f"\n{'='*78}\nVARIANT: {variant}\n{'='*78}")

        for k, obj_name in enumerate(OBJECTIVE_NAMES):
            y = Y[:, k]
            obs_range = float(y.max() - y.min())

            torch.manual_seed(args.seed)
            full = _fit(
                builder(
                    torch.tensor(X, dtype=DTYPE),
                    torch.tensor(y, dtype=DTYPE).unsqueeze(-1),
                )
            )
            hp = _read_hypers(full)
            span, sd = posterior_spread(full, d, args.sobol_n, args.seed)
            lo_mean, lo_std = loocv(X, y, builder, args.seed)
            met = compute_prediction_metrics(
                y,
                lo_mean,
                lo_std,
                variant_name=variant,
                objective_index=k,
                objective_name=obj_name,
            )

            print(f"\n--- {obj_name}  (observed range {obs_range:.4f}) ---")
            print("  ARD lengthscales (normalised input space):")
            for nm, v in zip(INPUT_NAMES, hp.lengthscales):
                flag = "  <-- flat" if v >= 10.0 else ""
                print(f"      {nm:>13}: {v:>12.4f}{flag}")
            n_flat = int(np.sum(hp.lengthscales >= 10.0))
            print(
                f"      {'median':>13}: {np.median(hp.lengthscales):>12.4f}"
                f"   ({n_flat}/{d} at or above 10)"
            )
            os_txt = "n/a" if hp.outputscale is None else f"{hp.outputscale:.4f}"
            print(f"  outputscale : {os_txt}    noise : {hp.noise:.6f}")
            print(
                f"  LOOCV       : MAE {met.mae:.4f}  RMSE {met.rmse:.4f}  "
                f"R2 {met.r_squared:+.4f}  Spearman {met.spearman_rank_correlation:+.4f}"
            )
            print(
                f"  coverage    : 68% {met.coverage_68_percent:.3f}   "
                f"95% {met.coverage_95_percent:.3f}   NLPD {met.mean_gaussian_nlpd:.3f}"
            )
            print(
                f"  posterior mean spread over {args.sobol_n} Sobol pts: "
                f"range {span:.5f} ({100*span/obs_range:.2f}% of observed range), "
                f"sd {sd:.5f}"
            )

            records.append(
                {
                    "variant": variant,
                    "objective": obj_name,
                    "median_lengthscale": float(np.median(hp.lengthscales)),
                    "n_lengthscale_ge_10": n_flat,
                    "outputscale": hp.outputscale,
                    "noise": hp.noise,
                    "loocv_mae": met.mae,
                    "loocv_rmse": met.rmse,
                    "loocv_r2": met.r_squared,
                    "loocv_spearman": met.spearman_rank_correlation,
                    "coverage_68": met.coverage_68_percent,
                    "coverage_95": met.coverage_95_percent,
                    "nlpd": met.mean_gaussian_nlpd,
                    "post_mean_range": span,
                    "post_mean_range_frac_of_observed": span / obs_range,
                    **{
                        f"ls_{nm}": float(v)
                        for nm, v in zip(INPUT_NAMES, hp.lengthscales)
                    },
                }
            )

    if len(args.variants) > 1:
        print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
        print(
            f"{'variant':>18} {'objective':>15} {'med LS':>9} {'flat':>5} "
            f"{'R2':>8} {'Spearman':>9} {'spread%':>9}"
        )
        for r in records:
            print(
                f"{r['variant']:>18} {r['objective']:>15} "
                f"{r['median_lengthscale']:>9.3f} {r['n_lengthscale_ge_10']:>5d} "
                f"{r['loocv_r2']:>+8.3f} {r['loocv_spearman']:>+9.3f} "
                f"{100*r['post_mean_range_frac_of_observed']:>9.2f}"
            )

    if args.csv:
        import pandas as pd

        pd.DataFrame(records).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
