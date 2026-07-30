"""Strict GP fitting and exact leave-one-out validation for robustness studies.

This module intentionally does not use the historical model-selection helper in
``models.py``.  That helper prints fitting failures and continues with fallback
hyperparameters, which is appropriate for its exploratory notebook use but not
for an auditable robustness study.  Every fit here either returns a structured
record or raises :class:`ModelFitError` without silently changing its contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from math import log, pi
from numbers import Real
from time import perf_counter
from typing import Any, Hashable, Iterable, Mapping, Sequence
import warnings

import gpytorch
import numpy as np
import pandas as pd
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_lognormal_prior,
)
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import spearmanr
import torch


DIM_SCALED_PRIOR_NAME = "dim_scaled_prior"
LEGACY_NO_PRIOR_NAME = "legacy_matern_no_prior"
CONSERVATIVE_NAME = "conservative"
#: BoTorch's MIN_INFERRED_NOISE_LEVEL, the floor that ships with its LogNormal
#: noise prior.  The prior, not the floor, is what stops the variance collapse.
DIM_SCALED_PRIOR_MIN_NOISE = 1.0e-4
LEGACY_NO_PRIOR_MIN_NOISE = 1.0e-3
CONSERVATIVE_MIN_NOISE = 0.01
CONSERVATIVE_MIN_LENGTHSCALE = 0.05
GAUSSIAN_95_Z = 1.959963984540054
VERY_SMALL_NORMALIZED_LENGTHSCALE = 0.05
EXTREMELY_LARGE_NORMALIZED_LENGTHSCALE = 10.0


def _finite_positive(value: Any, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a real non-boolean number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be finite and strictly positive.")
    return result


def _seed(value: Any) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError("seed must be a non-negative integer.")
    return int(value)


@dataclass(frozen=True)
class ModelVariantSpec:
    """One explicit GP model contract.

    ``use_dim_scaled_prior`` selects BoTorch's dimension-scaled LogNormal
    lengthscale prior.  Without it, an unregularised ARD kernel fitted to 15
    observations in 10 dimensions drives lengthscales to bimodal extremes
    (measured: 0.13 to 3.8e4) and pins the likelihood noise at its floor, which
    is interpolation rather than learning.

    ``use_lognormal_noise_prior`` selects BoTorch's ``LogNormal(-4, 1)`` noise
    prior.  It is required alongside the lengthscale prior, not optional: with a
    bare noise floor the fit has a second degenerate mode in which the
    outputscale collapses to zero and the model declares the data pure noise.
    That mode was observed on 10 of 15 leave-one-out folds of the thickness
    score, producing a latent predictive sd of 1e-4 against a fitted noise of
    0.93, 68% interval coverage of 0.133, and a mean NLPD of 3.1e6.

    See docs/GP_MODEL_DECISION.md.
    """

    name: str
    min_noise: float
    min_lengthscale: float | None
    kernel_name: str = "matern_2.5_ard"
    use_dim_scaled_prior: bool = False
    use_lognormal_noise_prior: bool = False

    def __post_init__(self) -> None:
        if self.name not in {
            DIM_SCALED_PRIOR_NAME,
            LEGACY_NO_PRIOR_NAME,
            CONSERVATIVE_NAME,
        }:
            raise ValueError(
                "Model variant name must be 'dim_scaled_prior', "
                "'legacy_matern_no_prior' or 'conservative'."
            )
        noise = _finite_positive(self.min_noise, field_name="min_noise")
        lengthscale = (
            None
            if self.min_lengthscale is None
            else _finite_positive(self.min_lengthscale, field_name="min_lengthscale")
        )
        if self.kernel_name != "matern_2.5_ard":
            raise ValueError("Only the audited matern_2.5_ard kernel is supported.")
        if self.name == DIM_SCALED_PRIOR_NAME and (
            noise != DIM_SCALED_PRIOR_MIN_NOISE
            or lengthscale is not None
            or not self.use_dim_scaled_prior
            or not self.use_lognormal_noise_prior
        ):
            raise ValueError(
                "dim_scaled_prior must use min_noise=1e-4, no lengthscale floor, "
                "and BOTH the dimension-scaled lengthscale prior and the "
                "LogNormal noise prior."
            )
        if self.name == LEGACY_NO_PRIOR_NAME and (
            noise != LEGACY_NO_PRIOR_MIN_NOISE
            or lengthscale is not None
            or self.use_dim_scaled_prior
            or self.use_lognormal_noise_prior
        ):
            raise ValueError(
                "legacy_matern_no_prior must preserve the retired Step 2B contract: "
                "min_noise=1e-3, no lengthscale floor, and no priors."
            )
        if self.name == CONSERVATIVE_NAME and (
            noise != CONSERVATIVE_MIN_NOISE
            or lengthscale != CONSERVATIVE_MIN_LENGTHSCALE
            or self.use_dim_scaled_prior
            or self.use_lognormal_noise_prior
        ):
            raise ValueError(
                "conservative must use min_noise=0.01, min_lengthscale=0.05 and "
                "no priors."
            )
        object.__setattr__(self, "min_noise", noise)
        object.__setattr__(self, "min_lengthscale", lengthscale)


#: The campaign default.  Matern 2.5 ARD with BoTorch's dimension-scaled
#: LogNormal lengthscale prior and its LogNormal(-4, 1) noise prior.  Both are
#: required; see ModelVariantSpec for what happens with only the former.
DIM_SCALED_PRIOR = ModelVariantSpec(
    DIM_SCALED_PRIOR_NAME,
    min_noise=DIM_SCALED_PRIOR_MIN_NOISE,
    min_lengthscale=None,
    use_dim_scaled_prior=True,
    use_lognormal_noise_prior=True,
)
#: Retired.  The prior-free contract used through Step 2C, kept only so archived
#: runs remain reproducible and interpretable.  Do not select for new work.
LEGACY_NO_PRIOR = ModelVariantSpec(
    LEGACY_NO_PRIOR_NAME,
    min_noise=LEGACY_NO_PRIOR_MIN_NOISE,
    min_lengthscale=None,
)
CONSERVATIVE = ModelVariantSpec(
    CONSERVATIVE_NAME,
    min_noise=CONSERVATIVE_MIN_NOISE,
    min_lengthscale=CONSERVATIVE_MIN_LENGTHSCALE,
)

#: What new runs get unless a caller deliberately asks for something else.
PRIMARY_VARIANT = DIM_SCALED_PRIOR


def model_variant_spec(name: str) -> ModelVariantSpec:
    """Return one of the fixed model contracts by name."""
    if name == DIM_SCALED_PRIOR_NAME:
        return DIM_SCALED_PRIOR
    if name == LEGACY_NO_PRIOR_NAME:
        return LEGACY_NO_PRIOR
    if name == CONSERVATIVE_NAME:
        return CONSERVATIVE
    raise ValueError(f"Unsupported model variant {name!r}.")


@dataclass(frozen=True)
class ModelFitWarning:
    variant_name: str
    fit_key: str
    omitted_sample_id: Hashable | None
    objective_index: int
    objective_name: str
    stage: str
    warning_category: str
    message: str


class ModelFitError(RuntimeError):
    """A strict model-fit failure with all warnings observed before failure."""

    def __init__(
        self,
        *,
        variant_name: str,
        fit_key: str,
        omitted_sample_id: Hashable | None,
        objective_index: int,
        objective_name: str,
        stage: str,
        cause: BaseException,
        fit_warnings: Sequence[ModelFitWarning],
    ) -> None:
        self.variant_name = variant_name
        self.fit_key = fit_key
        self.omitted_sample_id = omitted_sample_id
        self.objective_index = objective_index
        self.objective_name = objective_name
        self.stage = stage
        self.cause = cause
        self.fit_warnings = tuple(fit_warnings)
        super().__init__(
            "Strict GP fit failed: "
            f"variant={variant_name!r}, fit_key={fit_key!r}, "
            f"objective={objective_name!r}, stage={stage!r}, "
            f"cause={type(cause).__name__}: {cause}"
        )


@dataclass(frozen=True)
class ModelFitCacheKey:
    variant_name: str
    cohort_fingerprint: str
    omitted_sample_key: str
    objective_names: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class FittedModelRecord:
    variant: ModelVariantSpec
    model: ModelListGP
    train_X: torch.Tensor
    train_Y: torch.Tensor
    sample_ids: tuple[Hashable, ...]
    objective_names: tuple[str, ...]
    fit_key: str
    omitted_sample_id: Hashable | None
    seed: int
    cohort_fingerprint: str
    training_fingerprint: str
    fit_runtime_seconds: float
    warnings: tuple[ModelFitWarning, ...]


@dataclass
class ModelFitCache:
    """In-memory fit cache shared by LOOCV and observation influence."""

    records: dict[ModelFitCacheKey, FittedModelRecord] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: ModelFitCacheKey) -> FittedModelRecord | None:
        record = self.records.get(key)
        if record is None:
            self.misses += 1
        else:
            self.hits += 1
        return record

    def store(self, key: ModelFitCacheKey, record: FittedModelRecord) -> None:
        existing = self.records.get(key)
        if existing is not None and existing is not record:
            raise ValueError(f"A different fit already exists for cache key {key}.")
        self.records[key] = record


@dataclass(frozen=True)
class LOOCVPrediction:
    variant_name: str
    omitted_sample_id: Hashable
    row_role: str
    is_control: bool
    objective_index: int
    objective_name: str
    observed: float
    predicted_mean: float
    latent_std: float
    predictive_std: float
    prediction_error: float
    residual: float
    standardized_residual: float
    within_68_percent_interval: bool
    within_95_percent_interval: bool
    gaussian_nlpd: float
    fold_fit_key: str
    fold_fit_warning_count: int


@dataclass(frozen=True)
class PredictionMetricRecord:
    variant_name: str
    objective_index: int
    objective_name: str
    prediction_count: int
    mae: float
    rmse: float
    r_squared: float
    r_squared_warning: str
    spearman_rank_correlation: float
    mean_signed_error: float
    median_absolute_error: float
    coverage_68_percent: float
    coverage_95_percent: float
    mean_standardized_residual: float
    maximum_absolute_standardized_residual: float
    mean_gaussian_nlpd: float


@dataclass(frozen=True)
class HyperparameterRecord:
    variant_name: str
    fit_key: str
    omitted_sample_id: Hashable | None
    objective_index: int
    objective_name: str
    kernel_type: str
    likelihood_noise: float
    outputscale: float
    ard_lengthscales: tuple[float, ...]
    input_names: tuple[str, ...]
    configured_min_noise: float
    configured_min_lengthscale: float | None
    noise_constraint_lower_bound: float
    lengthscale_constraint_lower_bound: float
    noise_near_floor: bool
    lengthscales_near_floor: tuple[bool, ...]
    lengthscales_very_small_normalized_domain: tuple[bool, ...]
    lengthscales_extremely_large_flat: tuple[bool, ...]
    very_small_lengthscale_threshold: float = VERY_SMALL_NORMALIZED_LENGTHSCALE
    extremely_large_lengthscale_threshold: float = (
        EXTREMELY_LARGE_NORMALIZED_LENGTHSCALE
    )
    input_parameter_space: str = "normalized_0_1"
    outcome_parameter_space: str = "standardized_internal"

    def as_flat_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("ard_lengthscales")
        result.pop("input_names")
        result.pop("lengthscales_near_floor")
        result.pop("lengthscales_very_small_normalized_domain")
        result.pop("lengthscales_extremely_large_flat")
        result["any_lengthscale_very_small_normalized_domain"] = any(
            self.lengthscales_very_small_normalized_domain
        )
        result["any_lengthscale_extremely_large_flat"] = any(
            self.lengthscales_extremely_large_flat
        )
        for input_name, value, near_floor, very_small, extremely_large in zip(
            self.input_names,
            self.ard_lengthscales,
            self.lengthscales_near_floor,
            self.lengthscales_very_small_normalized_domain,
            self.lengthscales_extremely_large_flat,
        ):
            result[f"ard_lengthscale_{input_name}"] = value
            result[f"ard_lengthscale_{input_name}_near_floor"] = near_floor
            result[f"ard_lengthscale_{input_name}_very_small_normalized_domain"] = (
                very_small
            )
            result[f"ard_lengthscale_{input_name}_extremely_large_flat"] = (
                extremely_large
            )
        return result


@dataclass(frozen=True)
class ExactLOOCVResult:
    variant: ModelVariantSpec
    predictions: tuple[LOOCVPrediction, ...]
    metrics: tuple[PredictionMetricRecord, ...]
    fold_records: Mapping[Hashable, FittedModelRecord]
    cohort_fingerprint: str
    cache: ModelFitCache

    def predictions_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(row) for row in self.predictions)

    def metrics_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(row) for row in self.metrics)


@dataclass(frozen=True)
class ModelValidationResult:
    variant: ModelVariantSpec
    full_fit: FittedModelRecord
    loocv: ExactLOOCVResult
    hyperparameters: tuple[HyperparameterRecord, ...]

    def hyperparameters_frame(self) -> pd.DataFrame:
        return pd.DataFrame(row.as_flat_dict() for row in self.hyperparameters)

    def warnings_frame(self) -> pd.DataFrame:
        records = [self.full_fit, *self.loocv.fold_records.values()]
        return fit_warnings_frame(records)


def _sample_key(value: Hashable | None) -> str:
    if value is None:
        return "<full>"
    return f"{type(value).__name__}:{value!r}"


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.numpy().tobytes()


def dataset_fingerprint(
    X: torch.Tensor,
    Y: torch.Tensor,
    sample_ids: Sequence[Hashable],
) -> str:
    """Return a deterministic fingerprint for cache and provenance checks."""
    digest = hashlib.sha256()
    for tensor in (X, Y):
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(json.dumps(tuple(tensor.shape)).encode("utf-8"))
        digest.update(_tensor_bytes(tensor))
    serialized_ids = [f"{type(value).__name__}:{value!r}" for value in sample_ids]
    digest.update(json.dumps(serialized_ids, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _validate_dataset(
    X: torch.Tensor,
    Y: torch.Tensor,
    sample_ids: Sequence[Hashable],
    objective_names: Sequence[str],
    *,
    minimum_rows: int,
) -> tuple[tuple[Hashable, ...], tuple[str, ...]]:
    if not isinstance(X, torch.Tensor) or X.ndim != 2:
        raise ValueError("X must be a torch tensor with shape (N, D).")
    if not isinstance(Y, torch.Tensor) or Y.ndim != 2:
        raise ValueError("Y must be a torch tensor with shape (N, M).")
    if not X.is_floating_point() or not Y.is_floating_point():
        raise TypeError("X and Y must use floating dtypes.")
    if X.device != Y.device or X.dtype != Y.dtype:
        raise ValueError("X and Y must share dtype and device.")
    if X.shape[0] != Y.shape[0] or X.shape[0] < minimum_rows:
        raise ValueError(
            f"X and Y must share at least {minimum_rows} rows; "
            f"got {X.shape[0]} and {Y.shape[0]}."
        )
    if X.shape[1] == 0 or Y.shape[1] == 0:
        raise ValueError("X and Y must each contain at least one column.")
    if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
        raise ValueError("X and Y must contain only finite values.")
    if torch.any(X < 0.0) or torch.any(X > 1.0):
        raise ValueError("X must be normalized to [0, 1].")
    ids = tuple(sample_ids)
    if len(ids) != X.shape[0] or len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique and aligned with X/Y rows.")
    names = tuple(objective_names)
    if (
        len(names) != Y.shape[1]
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "objective_names must be unique non-empty strings aligned with Y columns."
        )
    return ids, tuple(name.strip() for name in names)


def _warning_rows(
    caught: Sequence[warnings.WarningMessage],
    *,
    variant: ModelVariantSpec,
    fit_key: str,
    omitted_sample_id: Hashable | None,
    objective_index: int,
    objective_name: str,
    stage: str,
) -> list[ModelFitWarning]:
    return [
        ModelFitWarning(
            variant_name=variant.name,
            fit_key=fit_key,
            omitted_sample_id=omitted_sample_id,
            objective_index=objective_index,
            objective_name=objective_name,
            stage=stage,
            warning_category=warning.category.__name__,
            message=str(warning.message),
        )
        for warning in caught
    ]


class SignalCollapseError(RuntimeError):
    """The fitted outputscale went to zero; the model has no signal component."""


#: A fit whose latent (signal) sd falls below this multiple of the fitted noise
#: sd has no signal component left in its GP.  Acquisition reads the latent
#: posterior, so the exploration term degenerates, even though predictive
#: intervals look fine because the inflated noise hides it.
MINIMUM_LATENT_TO_NOISE_SD_RATIO = 1.0e-2

#: How much the posterior MEAN must vary across the training inputs, as a fraction
#: of how much the OBSERVATIONS vary, for the fit to be able to rank candidates.
#:
#: This is the second half of the diagnosis, and what separates two very different
#: situations that share one numeric signature.
#:
#: The observed spread is the yardstick rather than the fitted noise sd, which was
#: the first attempt and is wrong: the noise is inflated precisely in the
#: degenerate case, so a noise-relative test co-varies with the thing it is trying
#: to detect.  Measured instance -- a linear mean on `anneal_temp` against a forced
#: noise of 0.9 gave a mean/noise ratio of 0.38 and would have been called
#: "effectively constant" while it was in fact tracking the data.
#:
#: Against the observed spread the question is scale-free and stable: does the
#: model's mean move with the measurements, or not at all?
#:
#: This is a DIAGNOSTIC threshold and it never touches utility space, so it does
#: not violate the campaign-fixed-scaling rule in `assert_scaling_is_campaign_fixed`.
#: That rule governs the objective scales that feed hypervolume, where a
#: data-derived scale would make rounds incomparable.  Nothing here reaches a
#: utility, a reference point or a hypervolume; it only asks whether one fit's
#: mean moved.  Do not "correct" it to a fixed constant.
MINIMUM_MEAN_SPREAD_TO_TARGET_RATIO = 0.05

#: Fit stage name for the guard, so warnings and errors are filterable.
SIGNAL_COLLAPSE_STAGE = "signal_collapse_guard"

#: Warning category raised when the GP's signal component has collapsed but the
#: mean function still carries a usable trend.
EXPLORATION_DEGENERATE_CATEGORY = "ExplorationTermDegenerate"


def _assert_signal_not_collapsed(
    gp: SingleTaskGP,
    X: torch.Tensor,
    target: torch.Tensor,
    *,
    variant: ModelVariantSpec,
    objective_index: int,
    objective_name: str,
    fit_key: str,
    omitted_sample_id: Hashable | None,
    fit_warnings: list[ModelFitWarning] | Sequence[ModelFitWarning],
) -> None:
    """Judge a fit whose GP signal component has gone to zero.

    This is a numerical guard, not a configuration check.  Naming a variant
    correctly cannot prevent a degenerate optimum: the same contract refitted on
    different data -- more observations, replicate-derived ``train_Yvar``, a new
    round -- can land there again.  So it runs on every fit.

    **Two situations share the collapsed-latent-sd signature, and they need
    different answers.**

    *True collapse.*  A zero-mean GP whose outputscale went to zero: the
    posterior mean is flat, nothing can be ranked, acquisition is meaningless.
    Observed instance: 10 of 15 leave-one-out folds of the thickness score fitted
    a noise of 0.93 against a latent sd of 1e-4.  This must fail.

    *The mean function did its job.*  With a ``StructuredMean`` carrying the
    trend, the residual GP can legitimately have nothing left to model.  The
    posterior *mean* still varies -- the mean module is not part of the covariance
    and so never enters ``posterior().variance`` -- so candidates still rank and
    the round is still worth proposing.  Refusing here would dead-end the campaign
    at the moment the physics model started working, with no remedy available:
    better data cannot be collected without first proposing conditions.  So this
    warns instead, and the human review artifact is the gate.

    The warning is not a formality.  Two things are genuinely wrong with such a
    fit and both are named in its message: UCB's exploration term has degenerated,
    and the mean module's coefficients are frozen buffers carrying no uncertainty
    of their own, so the narrow intervals the model reports are **understated
    rather than earned**.
    """
    with torch.no_grad():
        posterior = gp.posterior(X)
        latent_sd = float(posterior.variance.clamp_min(0.0).sqrt().min())
        mean_values = posterior.mean.detach().reshape(-1)
        mean_spread = float(mean_values.std()) if mean_values.numel() > 1 else 0.0
        # a FixedNoiseGaussianLikelihood (measured train_Yvar) carries one noise per
        # observation rather than one for the model, so take the average rather than
        # whichever row happens to be first
        noise_values = gp.likelihood.noise.detach().reshape(-1)
        noise_sd = float(noise_values.mean() ** 0.5)
        observed = target.detach().reshape(-1)
        target_spread = float(observed.std()) if observed.numel() > 1 else 0.0
    if noise_sd <= 0.0:
        return
    latent_ratio = latent_sd / noise_sd
    if latent_ratio >= MINIMUM_LATENT_TO_NOISE_SD_RATIO:
        return

    # a constant objective has nothing to rank by and nothing to diagnose
    mean_ratio = mean_spread / target_spread if target_spread > 0.0 else 0.0
    measured = (
        f"minimum latent sd {latent_sd:.3e} is {latent_ratio:.3e} of the fitted "
        f"noise sd {noise_sd:.3e}, below the "
        f"{MINIMUM_LATENT_TO_NOISE_SD_RATIO:g} floor"
    )

    if mean_ratio < MINIMUM_MEAN_SPREAD_TO_TARGET_RATIO:
        raise ModelFitError(
            variant_name=variant.name,
            fit_key=fit_key,
            omitted_sample_id=omitted_sample_id,
            objective_index=objective_index,
            objective_name=objective_name,
            stage=SIGNAL_COLLAPSE_STAGE,
            cause=SignalCollapseError(
                f"{measured}, and the posterior mean varies by only "
                f"{mean_spread:.3e} across the training inputs "
                f"({mean_ratio:.3e} of the observed spread {target_spread:.3e}, "
                f"floor {MINIMUM_MEAN_SPREAD_TO_TARGET_RATIO:g}). The model has explained "
                "the data as pure noise: its posterior mean is effectively "
                "constant, so it cannot order two candidates and its acquisition "
                "scores are meaningless. Predictive intervals do NOT reveal this, "
                "because the inflated noise masks the collapse."
            ),
            fit_warnings=tuple(fit_warnings),
        )

    warning = ModelFitWarning(
        variant_name=variant.name,
        fit_key=fit_key,
        omitted_sample_id=omitted_sample_id,
        objective_index=objective_index,
        objective_name=objective_name,
        stage=SIGNAL_COLLAPSE_STAGE,
        warning_category=EXPLORATION_DEGENERATE_CATEGORY,
        message=(
            f"{objective_name}: {measured}, but the posterior mean still varies by "
            f"{mean_spread:.3e} across the training inputs, {mean_ratio:.2f} of the "
            f"observed spread, so the mean function is carrying the signal and "
            "candidates can still be ranked. "
            "Two consequences to distrust: UCB's exploration term has degenerated, "
            "because it reads the latent posterior that just collapsed; and the "
            "mean module's coefficients are frozen buffers with no uncertainty of "
            "their own, so the narrow intervals this model reports are UNDERSTATED "
            "rather than earned. Treat its confidence, especially away from the "
            "observed points, as unproven."
        ),
    )
    if isinstance(fit_warnings, list):
        fit_warnings.append(warning)


def _build_single_task_gp(
    X: torch.Tensor,
    y: torch.Tensor,
    variant: ModelVariantSpec,
    mean_module: Any = None,
    train_Yvar: torch.Tensor | None = None,
) -> SingleTaskGP:
    if variant.use_dim_scaled_prior:
        # BoTorch's dimension-scaled LogNormal lengthscale prior, the same one
        # SingleTaskGP applies by default when no covar_module is supplied.  We
        # still pass an explicit module so the ScaleKernel wrapper (which the
        # hyperparameter readout and plots depend on) stays in place.
        base_kernel = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=X.shape[1],
            use_rbf_kernel=False,
        )
    else:
        lengthscale_constraint = (
            None
            if variant.min_lengthscale is None
            else GreaterThan(variant.min_lengthscale)
        )
        kernel_kwargs: dict[str, Any] = {
            "nu": 2.5,
            "ard_num_dims": X.shape[1],
        }
        if lengthscale_constraint is not None:
            kernel_kwargs["lengthscale_constraint"] = lengthscale_constraint
        base_kernel = MaternKernel(**kernel_kwargs)
    covar_module = ScaleKernel(base_kernel)
    if train_Yvar is not None:
        # Measured observation noise replaces fitted noise, so no noise prior or
        # constraint applies -- there is nothing left to fit.
        #
        # BoTorch will accept BOTH `train_Yvar` and an explicit `likelihood` and
        # then SILENTLY IGNORE the variance: the likelihood wins, stays a
        # single-element GaussianLikelihood, and the replicate information is
        # dropped with no error. Verified on 0.15.1. Hence the either/or here.
        #
        # `train_Yvar` is in the ORIGINAL target units; `Standardize` rescales it
        # along with the targets. Passing an already-standardized variance would be
        # wrong by var(Y) and would also fail silently.
        model = SingleTaskGP(
            X,
            y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
        )
        if mean_module is not None:
            model.mean_module = mean_module
        return model
    if variant.use_lognormal_noise_prior:
        # LogNormal(-4, 1) with a GreaterThan(1e-4) floor.  Without the prior the
        # marginal likelihood is free to drive the outputscale to zero and call
        # the data pure noise, which yields a degenerate near-zero latent
        # variance.  The floor alone does not prevent that.
        likelihood = get_gaussian_likelihood_with_lognormal_prior()
    else:
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(variant.min_noise))
    model = SingleTaskGP(
        X,
        y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    if mean_module is not None:
        # a frozen structured mean: its coefficients are registered as buffers,
        # so the marginal likelihood still fits only the GP hyperparameters
        model.mean_module = mean_module
    return model


def fit_model_variant(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    sample_ids: Sequence[Hashable],
    objective_names: Sequence[str],
    variant: ModelVariantSpec,
    seed: int = 73,
    fit_key: str = "full",
    omitted_sample_id: Hashable | None = None,
    cohort_fingerprint: str | None = None,
    cache: ModelFitCache | None = None,
    mean_module: Any = None,
    train_Yvar: torch.Tensor | None = None,
) -> FittedModelRecord:
    """Fit one strict independent GP per objective and return an audit record.

    ``train_Yvar`` is measured observation variance, shaped like ``Y``, in the
    ORIGINAL target units.  Supplying it replaces the fitted noise entirely: there
    is no noise hyperparameter left to optimise, so the variant's noise prior and
    floor no longer apply to that fit.
    """
    ids, names = _validate_dataset(X, Y, sample_ids, objective_names, minimum_rows=2)
    if not isinstance(variant, ModelVariantSpec):
        raise TypeError("variant must be a ModelVariantSpec.")
    resolved_seed = _seed(seed)
    if not isinstance(fit_key, str) or not fit_key.strip():
        raise ValueError("fit_key must be a non-empty string.")
    fit_key = fit_key.strip()
    training_hash = dataset_fingerprint(X, Y, ids)
    cohort_hash = training_hash if cohort_fingerprint is None else cohort_fingerprint
    if not isinstance(cohort_hash, str) or not cohort_hash.strip():
        raise ValueError("cohort_fingerprint must be None or a non-empty string.")
    if mean_module is not None and cache is not None:
        # the cache key is built from the data and the variant, not the mean
        # module, so a cached fit could be returned for a different trend.
        # Refuse rather than silently serve the wrong model.
        raise ValueError(
            "A structured mean_module cannot be combined with a ModelFitCache: "
            "the cache key does not capture the mean, so a fold could be served "
            "a fit built from a different trend."
        )
    cache_key = ModelFitCacheKey(
        variant_name=variant.name,
        cohort_fingerprint=cohort_hash,
        omitted_sample_key=_sample_key(omitted_sample_id),
        objective_names=names,
        seed=resolved_seed,
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            if cached.training_fingerprint != training_hash:
                raise ValueError(
                    "Cached model training fingerprint does not match supplied data."
                )
            return cached

    np.random.seed(resolved_seed)
    torch.manual_seed(resolved_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved_seed)

    started = perf_counter()
    models: list[SingleTaskGP] = []
    fit_warning_rows: list[ModelFitWarning] = []
    for objective_index, objective_name in enumerate(names):
        caught: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                gp = _build_single_task_gp(
                    X,
                    Y[:, objective_index : objective_index + 1],
                    variant,
                    mean_module=mean_module,
                    train_Yvar=(
                        None
                        if train_Yvar is None
                        else train_Yvar[:, objective_index : objective_index + 1]
                    ),
                )
            fit_warning_rows.extend(
                _warning_rows(
                    caught,
                    variant=variant,
                    fit_key=fit_key,
                    omitted_sample_id=omitted_sample_id,
                    objective_index=objective_index,
                    objective_name=objective_name,
                    stage="construct",
                )
            )
        except Exception as exc:
            fit_warning_rows.extend(
                _warning_rows(
                    caught,
                    variant=variant,
                    fit_key=fit_key,
                    omitted_sample_id=omitted_sample_id,
                    objective_index=objective_index,
                    objective_name=objective_name,
                    stage="construct",
                )
            )
            raise ModelFitError(
                variant_name=variant.name,
                fit_key=fit_key,
                omitted_sample_id=omitted_sample_id,
                objective_index=objective_index,
                objective_name=objective_name,
                stage="construct",
                cause=exc,
                fit_warnings=fit_warning_rows,
            ) from exc

        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit_gpytorch_mll(mll)
            fit_warning_rows.extend(
                _warning_rows(
                    caught,
                    variant=variant,
                    fit_key=fit_key,
                    omitted_sample_id=omitted_sample_id,
                    objective_index=objective_index,
                    objective_name=objective_name,
                    stage="optimize",
                )
            )
        except Exception as exc:
            fit_warning_rows.extend(
                _warning_rows(
                    caught,
                    variant=variant,
                    fit_key=fit_key,
                    omitted_sample_id=omitted_sample_id,
                    objective_index=objective_index,
                    objective_name=objective_name,
                    stage="optimize",
                )
            )
            raise ModelFitError(
                variant_name=variant.name,
                fit_key=fit_key,
                omitted_sample_id=omitted_sample_id,
                objective_index=objective_index,
                objective_name=objective_name,
                stage="optimize",
                cause=exc,
                fit_warnings=fit_warning_rows,
            ) from exc
        gp.eval()
        gp.likelihood.eval()
        _assert_signal_not_collapsed(
            gp,
            X,
            Y[:, objective_index],
            variant=variant,
            objective_index=objective_index,
            objective_name=objective_name,
            fit_key=fit_key,
            omitted_sample_id=omitted_sample_id,
            fit_warnings=fit_warning_rows,
        )
        models.append(gp)

    record = FittedModelRecord(
        variant=variant,
        model=ModelListGP(*models),
        train_X=X.detach().clone(),
        train_Y=Y.detach().clone(),
        sample_ids=ids,
        objective_names=names,
        fit_key=fit_key,
        omitted_sample_id=omitted_sample_id,
        seed=resolved_seed,
        cohort_fingerprint=cohort_hash,
        training_fingerprint=training_hash,
        fit_runtime_seconds=perf_counter() - started,
        warnings=tuple(fit_warning_rows),
    )
    record.model.eval()
    if cache is not None:
        cache.store(cache_key, record)
    return record


def _posterior_prediction(
    record: FittedModelRecord,
    X: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.no_grad():
        latent = record.model.posterior(X, observation_noise=False)
        predictive = record.model.posterior(X, observation_noise=True)
    mean = latent.mean.detach().cpu().double().numpy()
    latent_std = latent.variance.clamp_min(0.0).sqrt().detach().cpu().double().numpy()
    predictive_std = (
        predictive.variance.clamp_min(0.0).sqrt().detach().cpu().double().numpy()
    )
    if mean.shape != latent_std.shape or mean.shape != predictive_std.shape:
        raise RuntimeError("Posterior mean and uncertainty shapes do not match.")
    if not (
        np.all(np.isfinite(mean))
        and np.all(np.isfinite(latent_std))
        and np.all(np.isfinite(predictive_std))
        and np.all(predictive_std > 0.0)
    ):
        raise RuntimeError("Posterior predictions must be finite with positive noise.")
    if np.any(predictive_std + 1.0e-12 < latent_std):
        raise RuntimeError("Predictive uncertainty cannot be below latent uncertainty.")
    return mean, latent_std, predictive_std


def compute_prediction_metrics(
    observed: Sequence[float] | np.ndarray,
    predicted_mean: Sequence[float] | np.ndarray,
    predictive_std: Sequence[float] | np.ndarray,
    *,
    variant_name: str = "unspecified",
    objective_index: int = 0,
    objective_name: str = "objective",
    small_n_warning_threshold: int = 20,
) -> PredictionMetricRecord:
    """Compute declared predictive metrics from original-unit predictions."""
    actual = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted_mean, dtype=float)
    uncertainty = np.asarray(predictive_std, dtype=float)
    if (
        actual.ndim != 1
        or predicted.shape != actual.shape
        or uncertainty.shape != actual.shape
    ):
        raise ValueError("observed, predicted_mean, and predictive_std must align.")
    if actual.size == 0:
        raise ValueError("At least one prediction is required.")
    if not (
        np.all(np.isfinite(actual))
        and np.all(np.isfinite(predicted))
        and np.all(np.isfinite(uncertainty))
    ):
        raise ValueError("Prediction metrics require finite inputs.")
    if np.any(uncertainty <= 0.0):
        raise ValueError("predictive_std must be strictly positive.")
    if (
        isinstance(small_n_warning_threshold, bool)
        or not isinstance(small_n_warning_threshold, (int, np.integer))
        or int(small_n_warning_threshold) < 2
    ):
        raise ValueError("small_n_warning_threshold must be an integer >= 2.")

    errors = predicted - actual
    residuals = -errors
    standardized = residuals / uncertainty
    squared_error = errors**2
    centered = actual - actual.mean()
    denominator = float(np.sum(centered**2))
    if actual.size < 2 or denominator <= 0.0:
        r_squared = np.nan
        r_squared_warning = (
            "R² is undefined for fewer than two or constant observations."
        )
    else:
        r_squared = 1.0 - float(np.sum(squared_error)) / denominator
        r_squared_warning = (
            f"R² is unstable with small N={actual.size}."
            if actual.size < int(small_n_warning_threshold)
            else ""
        )
    if actual.size < 2 or np.unique(actual).size < 2 or np.unique(predicted).size < 2:
        spearman = np.nan
    else:
        spearman = float(spearmanr(actual, predicted).statistic)
    nlpd = 0.5 * np.log(2.0 * pi * uncertainty**2) + 0.5 * standardized**2
    return PredictionMetricRecord(
        variant_name=str(variant_name),
        objective_index=int(objective_index),
        objective_name=str(objective_name),
        prediction_count=int(actual.size),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(squared_error))),
        r_squared=float(r_squared),
        r_squared_warning=r_squared_warning,
        spearman_rank_correlation=float(spearman),
        mean_signed_error=float(np.mean(errors)),
        median_absolute_error=float(np.median(np.abs(errors))),
        coverage_68_percent=float(np.mean(np.abs(standardized) <= 1.0)),
        coverage_95_percent=float(np.mean(np.abs(standardized) <= GAUSSIAN_95_Z)),
        mean_standardized_residual=float(np.mean(standardized)),
        maximum_absolute_standardized_residual=float(np.max(np.abs(standardized))),
        mean_gaussian_nlpd=float(np.mean(nlpd)),
    )


def _summarize_predictions(
    predictions: Sequence[LOOCVPrediction],
    variant: ModelVariantSpec,
    objective_names: Sequence[str],
) -> tuple[PredictionMetricRecord, ...]:
    rows: list[PredictionMetricRecord] = []
    for objective_index, objective_name in enumerate(objective_names):
        selected = [
            row for row in predictions if row.objective_index == objective_index
        ]
        rows.append(
            compute_prediction_metrics(
                [row.observed for row in selected],
                [row.predicted_mean for row in selected],
                [row.predictive_std for row in selected],
                variant_name=variant.name,
                objective_index=objective_index,
                objective_name=objective_name,
            )
        )
    return tuple(rows)


def run_exact_loocv(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    sample_ids: Sequence[Hashable],
    objective_names: Sequence[str],
    variant: ModelVariantSpec,
    seed: int = 73,
    row_roles: Sequence[str] | None = None,
    control_sample_ids: Iterable[Hashable] = (),
    cache: ModelFitCache | None = None,
) -> ExactLOOCVResult:
    """Fit exactly one N-1 model per row and predict every held-out objective."""
    ids, names = _validate_dataset(X, Y, sample_ids, objective_names, minimum_rows=3)
    roles = tuple("observation" for _ in ids) if row_roles is None else tuple(row_roles)
    if len(roles) != len(ids) or any(
        not isinstance(role, str) or not role.strip() for role in roles
    ):
        raise ValueError("row_roles must contain one non-empty string per row.")
    control_ids = set(control_sample_ids)
    resolved_cache = ModelFitCache() if cache is None else cache
    cohort_hash = dataset_fingerprint(X, Y, ids)
    fold_records: dict[Hashable, FittedModelRecord] = {}
    prediction_rows: list[LOOCVPrediction] = []
    for omitted_index, omitted_id in enumerate(ids):
        mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
        mask[omitted_index] = False
        fold_ids = tuple(
            sample_id for index, sample_id in enumerate(ids) if index != omitted_index
        )
        fit_key = f"omit:{_sample_key(omitted_id)}"
        record = fit_model_variant(
            X[mask],
            Y[mask],
            sample_ids=fold_ids,
            objective_names=names,
            variant=variant,
            seed=seed,
            fit_key=fit_key,
            omitted_sample_id=omitted_id,
            cohort_fingerprint=cohort_hash,
            cache=resolved_cache,
        )
        fold_records[omitted_id] = record
        mean, latent_std, predictive_std = _posterior_prediction(
            record, X[omitted_index : omitted_index + 1]
        )
        for objective_index, objective_name in enumerate(names):
            observed = float(Y[omitted_index, objective_index].item())
            predicted = float(mean[0, objective_index])
            latent_uncertainty = float(latent_std[0, objective_index])
            predictive_uncertainty = float(predictive_std[0, objective_index])
            error = predicted - observed
            residual = -error
            standardized = residual / predictive_uncertainty
            nlpd = 0.5 * log(2.0 * pi * predictive_uncertainty**2) + 0.5 * (
                standardized**2
            )
            prediction_rows.append(
                LOOCVPrediction(
                    variant_name=variant.name,
                    omitted_sample_id=omitted_id,
                    row_role=roles[omitted_index].strip(),
                    is_control=omitted_id in control_ids,
                    objective_index=objective_index,
                    objective_name=objective_name,
                    observed=observed,
                    predicted_mean=predicted,
                    latent_std=latent_uncertainty,
                    predictive_std=predictive_uncertainty,
                    prediction_error=error,
                    residual=residual,
                    standardized_residual=standardized,
                    within_68_percent_interval=abs(standardized) <= 1.0,
                    within_95_percent_interval=(abs(standardized) <= GAUSSIAN_95_Z),
                    gaussian_nlpd=nlpd,
                    fold_fit_key=record.fit_key,
                    fold_fit_warning_count=len(record.warnings),
                )
            )
    expected_count = X.shape[0] * Y.shape[1]
    if len(prediction_rows) != expected_count:
        raise RuntimeError(
            f"LOOCV produced {len(prediction_rows)} rows; expected {expected_count}."
        )
    metrics = _summarize_predictions(prediction_rows, variant, names)
    return ExactLOOCVResult(
        variant=variant,
        predictions=tuple(prediction_rows),
        metrics=metrics,
        fold_records=fold_records,
        cohort_fingerprint=cohort_hash,
        cache=resolved_cache,
    )


def _constraint_lower_bound(constraint: Any) -> float:
    value = constraint.lower_bound.detach().cpu().double().reshape(-1)
    return float(value[0].item())


def extract_model_hyperparameters(
    record: FittedModelRecord,
    *,
    input_names: Sequence[str],
) -> tuple[HyperparameterRecord, ...]:
    """Extract comparable full/fold hyperparameters from a fitted model list."""
    names = tuple(input_names)
    if len(names) != record.train_X.shape[1] or any(
        not isinstance(name, str) or not name.strip() for name in names
    ):
        raise ValueError("input_names must align with the fitted input dimension.")
    rows: list[HyperparameterRecord] = []
    for objective_index, (objective_name, gp) in enumerate(
        zip(record.objective_names, record.model.models)
    ):
        base_kernel = gp.covar_module.base_kernel
        lengthscales = tuple(
            float(value)
            for value in base_kernel.lengthscale.detach()
            .cpu()
            .double()
            .reshape(-1)
            .tolist()
        )
        if len(lengthscales) != len(names):
            raise RuntimeError("ARD lengthscales do not align with input names.")
        noise = float(gp.likelihood.noise.detach().cpu().double().reshape(-1)[0].item())
        outputscale = float(
            gp.covar_module.outputscale.detach().cpu().double().reshape(-1)[0].item()
        )
        noise_floor = _constraint_lower_bound(
            gp.likelihood.noise_covar.raw_noise_constraint
        )
        lengthscale_floor = _constraint_lower_bound(
            base_kernel.raw_lengthscale_constraint
        )
        noise_near = noise <= noise_floor * 1.05 + 1.0e-12
        lengthscale_near = tuple(
            value <= lengthscale_floor * 1.05 + 1.0e-12 for value in lengthscales
        )
        lengthscale_very_small = tuple(
            value <= VERY_SMALL_NORMALIZED_LENGTHSCALE for value in lengthscales
        )
        lengthscale_extremely_large = tuple(
            value >= EXTREMELY_LARGE_NORMALIZED_LENGTHSCALE for value in lengthscales
        )
        rows.append(
            HyperparameterRecord(
                variant_name=record.variant.name,
                fit_key=record.fit_key,
                omitted_sample_id=record.omitted_sample_id,
                objective_index=objective_index,
                objective_name=objective_name,
                kernel_type=type(base_kernel).__name__,
                likelihood_noise=noise,
                outputscale=outputscale,
                ard_lengthscales=lengthscales,
                input_names=tuple(name.strip() for name in names),
                configured_min_noise=record.variant.min_noise,
                configured_min_lengthscale=record.variant.min_lengthscale,
                noise_constraint_lower_bound=noise_floor,
                lengthscale_constraint_lower_bound=lengthscale_floor,
                noise_near_floor=noise_near,
                lengthscales_near_floor=lengthscale_near,
                lengthscales_very_small_normalized_domain=lengthscale_very_small,
                lengthscales_extremely_large_flat=lengthscale_extremely_large,
            )
        )
    return tuple(rows)


def fit_warnings_frame(records: Iterable[FittedModelRecord]) -> pd.DataFrame:
    columns = [field.name for field in ModelFitWarning.__dataclass_fields__.values()]
    rows = [asdict(warning) for record in records for warning in record.warnings]
    return pd.DataFrame(rows, columns=columns)


def validate_model_variant(
    X: torch.Tensor,
    Y: torch.Tensor,
    *,
    sample_ids: Sequence[Hashable],
    input_names: Sequence[str],
    objective_names: Sequence[str],
    variant: ModelVariantSpec,
    seed: int = 73,
    row_roles: Sequence[str] | None = None,
    control_sample_ids: Iterable[Hashable] = (),
    cache: ModelFitCache | None = None,
) -> ModelValidationResult:
    """Fit the full model, run exact LOOCV, and extract every hyperparameter."""
    ids, names = _validate_dataset(X, Y, sample_ids, objective_names, minimum_rows=3)
    resolved_cache = ModelFitCache() if cache is None else cache
    cohort_hash = dataset_fingerprint(X, Y, ids)
    full = fit_model_variant(
        X,
        Y,
        sample_ids=ids,
        objective_names=names,
        variant=variant,
        seed=seed,
        fit_key="full",
        omitted_sample_id=None,
        cohort_fingerprint=cohort_hash,
        cache=resolved_cache,
    )
    loocv = run_exact_loocv(
        X,
        Y,
        sample_ids=ids,
        objective_names=names,
        variant=variant,
        seed=seed,
        row_roles=row_roles,
        control_sample_ids=control_sample_ids,
        cache=resolved_cache,
    )
    all_records = [full, *loocv.fold_records.values()]
    hyperparameters = tuple(
        hyperparameter
        for record in all_records
        for hyperparameter in extract_model_hyperparameters(
            record, input_names=input_names
        )
    )
    return ModelValidationResult(
        variant=variant,
        full_fit=full,
        loocv=loocv,
        hyperparameters=hyperparameters,
    )


__all__ = [
    "CONSERVATIVE",
    "MINIMUM_LATENT_TO_NOISE_SD_RATIO",
    "SignalCollapseError",
    "DIM_SCALED_PRIOR",
    "LEGACY_NO_PRIOR",
    "PRIMARY_VARIANT",
    "ExactLOOCVResult",
    "FittedModelRecord",
    "HyperparameterRecord",
    "LOOCVPrediction",
    "ModelFitCache",
    "ModelFitError",
    "ModelFitWarning",
    "ModelValidationResult",
    "ModelVariantSpec",
    "PredictionMetricRecord",
    "compute_prediction_metrics",
    "dataset_fingerprint",
    "extract_model_hyperparameters",
    "fit_model_variant",
    "fit_warnings_frame",
    "model_variant_spec",
    "run_exact_loocv",
    "validate_model_variant",
]
