"""Physics-informed mean functions for small-data GPs.

With 15 observations in 10 dimensions a zero-mean GP spends most of its capacity
rediscovering a trend that is already known from process physics.  Giving it that
trend as a mean function, and letting the GP model only the residual, roughly
doubles the leave-one-out fit on this campaign's data.

EVERY NUMBER BELOW IS FROM THE FIRST CAMPAIGN, contract
``d2d-objectives-v2-nm-thickness``, on ``Summary Table.xlsx``.  Objectives have
been redefined twice since.  Read them as the record of how these shapes were
chosen, NOT as current fits -- and see the second bullet for one that has since
been measured false.

* **thickness** -- ``log T ~ log(speed_1) + log(precur_conc)``.  Spin-coating
  theory gives ``T ~ omega^-0.5``; the measured exponent was -0.38 on v2 and is
  -0.255 on the current workbook.  Neither term alone is worth much (LOO R2
  +0.159 and +0.187); the *pair* carries the signal (+0.449 on v2, and the shape
  still holds on v4).  Modelled in log space, so the response is lognormal.
  **WITHDRAWN FROM THE LIVE CONFIG 2026-09-06.**  The shape transfers, but the
  physics justification does not: the exponent's 95% interval on v4 is
  [-0.385, -0.126], which EXCLUDES the textbook -0.5 by 4.1 standard errors, and
  fixing the exponents at theory scores +0.5600 against +0.5823 for no trend at
  all.  What survives is that the VARIABLE choice beats matched-flexibility
  controls (four unmotivated pairs scored +0.30 to +0.40, all below the plain GP)
  -- the magnitudes were fitted, not predicted.  This module stays wired and
  tested for a prior that clears the bar; nothing currently does.
* **optoelectronic** -- a single linear term on ``anneal_temp`` and nothing else,
  LOO R2 +0.244 ON V2.  **IT DOES NOT TRANSFER AND IS NO LONGER DECLARED
  ANYWHERE.**  v2's optoelectronic was ``log10(Voc x Photoconductance)``; the
  current contract's is a different quantity, and on it the same mean function
  scores **-1.0721** (and -0.7452 on raw Voc), far below even a constant.  The
  key was removed from the live config on the v3 intake verdict and nothing since
  has argued for reinstating it.  Measured 2026-09-04; reproduce with
  ``scripts/raw_component_screen.py --spec
  '[{"name":"x","expr":"score_opto","mean_features":["anneal_temp"]}]'``.

Do not assume the two-term shape generalises: these are opposite patterns, and
one of them turned out to be about an objective that no longer exists.

The linear coefficients are refit on the training rows of every fold, so
cross-validation stays honest.  Feature *choice* is fixed in configuration from
physics, before fitting -- it is not selected against the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import gpytorch
import numpy as np
import torch

__all__ = [
    "MeanFeature",
    "StructuredMeanSpec",
    "StructuredPosterior",
    "StructuredMean",
    "build_structured_mean",
    "fit_structured_mean",
    "mean_spec_from_config",
]

Link = Literal["identity", "log"]


@dataclass(frozen=True)
class MeanFeature:
    """One column of the linear mean's design matrix."""

    column: str
    transform: Link = "identity"

    def evaluate(self, values: np.ndarray) -> np.ndarray:
        if self.transform == "identity":
            return values
        if self.transform == "log":
            if np.any(values <= 0):
                raise ValueError(
                    f"Mean feature {self.column!r} uses a log transform but the "
                    "column contains non-positive values."
                )
            return np.log(values)
        raise ValueError(f"Unsupported mean-feature transform {self.transform!r}.")


@dataclass(frozen=True)
class StructuredMeanSpec:
    """A linear trend removed before GP fitting and added back after.

    ``response`` is the space the GP works in.  ``log`` means the GP models
    ``log y``, which makes ``y`` lognormal -- the utility expectation must then
    use Gauss-Hermite quadrature, not the Gaussian closed form.
    """

    response: Link
    features: tuple[MeanFeature, ...]

    def __post_init__(self) -> None:
        if self.response not in ("identity", "log"):
            raise ValueError(f"Unsupported response link {self.response!r}.")
        if not self.features:
            raise ValueError("A structured mean needs at least one feature.")
        names = [feature.column for feature in self.features]
        if len(set(names)) != len(names):
            raise ValueError(f"Mean features must be unique; got {names}.")

    def design_matrix(
        self, X_phys: np.ndarray, input_names: Sequence[str]
    ) -> np.ndarray:
        columns = []
        for feature in self.features:
            try:
                index = list(input_names).index(feature.column)
            except ValueError as exc:
                raise ValueError(
                    f"Mean feature {feature.column!r} is not a declared input."
                ) from exc
            columns.append(feature.evaluate(np.asarray(X_phys, float)[:, index]))
        return np.column_stack(columns)


@dataclass(frozen=True)
class StructuredPosterior:
    """Posterior in the response space, plus the link needed to interpret it.

    When ``link == "log"`` these are the mean and variance of ``log y``, so the
    utility expectation must integrate a lognormal.
    """

    mean: np.ndarray
    variance: np.ndarray
    link: Link


def _ols(F: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least squares with an intercept, returned as coefficients on [1, F]."""
    design = np.column_stack([np.ones(len(F)), F])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def fit_structured_mean(
    X_phys: np.ndarray,
    y: np.ndarray,
    spec: StructuredMeanSpec,
    input_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the linear-mean coefficients and the residuals to hand the GP.

    Call this with the TRAINING rows only.  Refitting per fold is what keeps
    cross-validation honest; fitting once on everything and holding it fixed
    leaks the held-out value into the trend.
    """
    values = np.asarray(y, dtype=float)
    if spec.response == "log":
        if np.any(values <= 0):
            raise ValueError("A log response requires strictly positive observations.")
        target = np.log(values)
    else:
        target = values
    F = spec.design_matrix(X_phys, input_names)
    coefficients = _ols(F, target)
    fitted = np.column_stack([np.ones(len(F)), F]) @ coefficients
    return coefficients, target - fitted


def apply_structured_mean(
    coefficients: np.ndarray,
    X_phys: np.ndarray,
    spec: StructuredMeanSpec,
    input_names: Sequence[str],
    residual_mean: np.ndarray,
    residual_variance: np.ndarray,
) -> StructuredPosterior:
    """Add the linear trend back to a GP residual posterior."""
    F = spec.design_matrix(X_phys, input_names)
    trend = np.column_stack([np.ones(len(F)), F]) @ coefficients
    return StructuredPosterior(
        mean=np.asarray(residual_mean, float) + trend,
        variance=np.asarray(residual_variance, float),
        link=spec.response,
    )


def mean_spec_from_config(entry: Mapping[str, Any]) -> StructuredMeanSpec | None:
    """Build a spec from one objective's ``mean_function`` block, if present."""
    block = entry.get("mean_function")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError("mean_function must be a mapping.")
    raw_features = block.get("features")
    if not isinstance(raw_features, Sequence) or not raw_features:
        raise ValueError("mean_function.features must be a non-empty list.")
    features = []
    for item in raw_features:
        if isinstance(item, str):
            features.append(MeanFeature(item))
        elif isinstance(item, Mapping):
            features.append(
                MeanFeature(str(item["column"]), str(item.get("transform", "identity")))
            )
        else:
            raise ValueError("Each mean feature must be a string or a mapping.")
    return StructuredMeanSpec(
        response=str(block.get("response", "identity")),
        features=tuple(features),
    )


# --------------------------------------------------------------------------- #
# as a GPyTorch mean module
# --------------------------------------------------------------------------- #


class StructuredMean(gpytorch.means.Mean):
    """The linear trend as a GP mean module, with the coefficients frozen.

    This is the two-stage pipeline -- OLS detrend, zero-mean GP on the residual --
    expressed as a single model.  ``posterior()`` is then correct by
    construction: there is no trend to add back afterwards and therefore no code
    path that can forget to.

    Coefficients are registered as **buffers, not parameters**, so the marginal
    likelihood fits only the GP hyperparameters and leaves the OLS fit alone.

    The module operates in the model's *standardized* outcome space, because
    ``SingleTaskGP`` is built with ``Standardize(m=1)``.  Getting that wrong is
    silent: the trend comes out shifted and scaled, and the model still looks
    plausible.  :func:`build_structured_mean` handles the conversion.
    """

    def __init__(
        self,
        lowers: torch.Tensor,
        uppers: torch.Tensor,
        feature_columns: Sequence[int],
        feature_logs: Sequence[bool],
        coefficients: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("lowers", lowers)
        self.register_buffer("uppers", uppers)
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("bias", bias)
        self.feature_columns = tuple(int(c) for c in feature_columns)
        self.feature_logs = tuple(bool(f) for f in feature_logs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the GP sees normalized inputs; the trend is defined on physical ones
        physical = self.lowers + x * (self.uppers - self.lowers)
        columns = []
        for position, use_log in zip(self.feature_columns, self.feature_logs):
            value = physical[..., position]
            columns.append(torch.log(value) if use_log else value)
        features = torch.stack(columns, dim=-1)
        return (features * self.coefficients).sum(dim=-1) + self.bias


def build_structured_mean(
    X_phys: np.ndarray,
    y: np.ndarray,
    spec: StructuredMeanSpec,
    input_names: Sequence[str],
    lowers: np.ndarray,
    uppers: np.ndarray,
) -> tuple["StructuredMean", np.ndarray]:
    """Fit the trend on these rows and return it as a mean module.

    Returns the module plus the response-space target the GP should train on
    (``log y`` for a log response, ``y`` otherwise).  Pass only TRAINING rows;
    refitting per fold is what keeps cross-validation honest.
    """
    values = np.asarray(y, dtype=float)
    if spec.response == "log":
        if np.any(values <= 0):
            raise ValueError("A log response requires strictly positive observations.")
        target = np.log(values)
    else:
        target = values

    coefficients = _ols(spec.design_matrix(X_phys, input_names), target)
    intercept, slopes = float(coefficients[0]), coefficients[1:]

    # SingleTaskGP standardizes the outcome, so express the trend in that space:
    #   m_std(x) = (m_raw(x) - mu) / sigma
    mu = float(np.mean(target))
    sigma = float(np.std(target, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0

    names = list(input_names)
    module = StructuredMean(
        lowers=torch.tensor(np.asarray(lowers, float), dtype=torch.double),
        uppers=torch.tensor(np.asarray(uppers, float), dtype=torch.double),
        feature_columns=[names.index(f.column) for f in spec.features],
        feature_logs=[f.transform == "log" for f in spec.features],
        coefficients=torch.tensor(slopes / sigma, dtype=torch.double),
        bias=torch.tensor((intercept - mu) / sigma, dtype=torch.double),
    )
    return module, target
