"""Versioned, all-maximize objective transformations for MOBO acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal, Sequence

import numpy as np
import torch
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective


Goal = Literal["maximize", "minimize", "target"]
TransformName = Literal[
    "identity", "affine", "gaussian_target", "negative_absolute_target"
]
UtilityBound = tuple[float | None, float | None]


def _finite_optional(value: float | None, *, field: str, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(
            f"Objective {name!r} field {field!r} must be a real non-boolean number."
        )
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Objective {name!r} field {field!r} must be finite.")
    return number


@dataclass(frozen=True)
class ObjectiveSpec:
    """Immutable definition of one raw-output-to-utility transformation."""

    name: str
    goal: Goal
    transform: TransformName
    #: What the MODEL emits, relative to the physical quantity the utility is
    #: defined on.  ``log`` means the GP was fitted in log space (see
    #: ``structured_mean``), so a model output must be exponentiated before the
    #: utility applies, and a model *posterior* is lognormal rather than normal.
    model_link: Literal["identity", "log"] = "identity"
    source_column: str | None = None
    lower_anchor: float | None = None
    upper_anchor: float | None = None
    target: float | None = None
    sigma: float | None = None
    scale: float | None = None
    clip: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Objective name must be a non-empty string.")
        object.__setattr__(self, "name", self.name.strip())
        if self.goal not in {"maximize", "minimize", "target"}:
            raise ValueError(
                f"Objective {self.name!r} has unsupported goal {self.goal!r}."
            )
        supported = {
            "identity",
            "affine",
            "gaussian_target",
            "negative_absolute_target",
        }
        if self.transform not in supported:
            raise ValueError(
                f"Objective {self.name!r} has unsupported transform "
                f"{self.transform!r}."
            )
        if self.source_column is not None and (
            not isinstance(self.source_column, str) or not self.source_column.strip()
        ):
            raise ValueError("source_column must be None or a non-empty string.")
        if self.source_column is not None:
            object.__setattr__(self, "source_column", self.source_column.strip())
        if not isinstance(self.clip, bool):
            raise ValueError("clip must be a boolean.")

        numeric = {
            field: _finite_optional(getattr(self, field), field=field, name=self.name)
            for field in (
                "lower_anchor",
                "upper_anchor",
                "target",
                "sigma",
                "scale",
            )
        }
        for field, value in numeric.items():
            object.__setattr__(self, field, value)

        if self.transform == "identity":
            if self.goal != "maximize":
                raise ValueError("identity is supported only for maximize utilities.")
            self._require_unused(
                "lower_anchor", "upper_anchor", "target", "sigma", "scale"
            )
            if self.clip:
                raise ValueError(
                    "identity utilities cannot enable clip; approve their scale "
                    "upstream or use an explicit affine transform."
                )
        elif self.transform == "affine":
            if self.goal not in {"maximize", "minimize"}:
                raise ValueError("affine requires goal 'maximize' or 'minimize'.")
            if self.lower_anchor is None or self.upper_anchor is None:
                raise ValueError("affine requires lower_anchor and upper_anchor.")
            if self.lower_anchor >= self.upper_anchor:
                raise ValueError("affine requires lower_anchor < upper_anchor.")
            self._require_unused("target", "sigma", "scale")
        elif self.transform == "gaussian_target":
            if self.goal != "target":
                raise ValueError("gaussian_target requires goal 'target'.")
            if self.target is None or self.sigma is None:
                raise ValueError("gaussian_target requires target and sigma.")
            if self.sigma <= 0:
                raise ValueError("gaussian_target sigma must be strictly positive.")
            self._require_unused("lower_anchor", "upper_anchor", "scale")
            if self.clip:
                raise ValueError(
                    "gaussian_target is naturally bounded; clip is invalid."
                )
        else:
            if self.goal != "target":
                raise ValueError("negative_absolute_target requires goal 'target'.")
            if self.target is None or self.scale is None:
                raise ValueError(
                    "negative_absolute_target requires target and explicit scale."
                )
            if self.scale <= 0:
                raise ValueError(
                    "negative_absolute_target scale must be strictly positive."
                )
            self._require_unused("lower_anchor", "upper_anchor", "sigma")
            if self.clip:
                raise ValueError("clip is not supported for negative_absolute_target.")

    def _require_unused(self, *fields: str) -> None:
        used = [field for field in fields if getattr(self, field) is not None]
        if used:
            raise ValueError(
                f"Objective {self.name!r} transform {self.transform!r} does not "
                f"accept parameter(s): {', '.join(used)}."
            )


class ObjectiveTransform:
    """Apply an ordered objective contract to floating tensors ``[..., M]``."""

    def __init__(self, specs: Sequence[ObjectiveSpec], *, version: str) -> None:
        if not isinstance(version, str) or not version.strip():
            raise ValueError("Objective contract version must be a non-empty string.")
        if not specs:
            raise ValueError("At least one ObjectiveSpec is required.")
        validated = tuple(specs)
        if not all(isinstance(spec, ObjectiveSpec) for spec in validated):
            raise TypeError("Every objective specification must be an ObjectiveSpec.")
        names = [spec.name for spec in validated]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Objective names must be unique; got {duplicates}.")
        self.specs = validated
        self.version = version.strip()

    @property
    def objective_count(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def transform(self, Y: torch.Tensor) -> torch.Tensor:
        """Transform raw/model outputs while preserving shape, dtype, and device."""
        if not isinstance(Y, torch.Tensor):
            raise TypeError("Y must be a torch.Tensor.")
        if not Y.is_floating_point():
            raise TypeError("Y must use a floating dtype.")
        if Y.ndim < 1 or Y.shape[-1] != self.objective_count:
            raise ValueError(
                f"Y final dimension must be {self.objective_count}; "
                f"got shape {tuple(Y.shape)}."
            )
        if not torch.isfinite(Y).all():
            raise ValueError("Y must contain only finite values.")
        outputs: list[torch.Tensor] = []
        for index, spec in enumerate(self.specs):
            outputs.append(self._utility(Y[..., index], spec))
        transformed = torch.stack(outputs, dim=-1)
        if transformed.shape != Y.shape:
            raise RuntimeError("Internal objective transform shape error.")
        return transformed

    def _utility(self, model_output: torch.Tensor, spec: ObjectiveSpec):
        """Utility for one objective, given that objective's MODEL output.

        The link decode happens exactly here and nowhere else.  Quadrature and
        Monte-Carlo paths must both route through this, or one of them will
        exponentiate twice.
        """
        raw = torch.exp(model_output) if spec.model_link == "log" else model_output
        return self._utility_from_physical(raw, spec)

    @staticmethod
    def _utility_from_physical(raw: torch.Tensor, spec: ObjectiveSpec):
        """Utility for one objective from its PHYSICAL value (link already undone)."""
        if True:
            if spec.transform == "identity":
                utility = raw
            elif spec.transform == "affine":
                lower = raw.new_tensor(spec.lower_anchor)
                upper = raw.new_tensor(spec.upper_anchor)
                if spec.goal == "maximize":
                    utility = (raw - lower) / (upper - lower)
                else:
                    utility = (upper - raw) / (upper - lower)
                if spec.clip:
                    utility = utility.clamp(0.0, 1.0)
            elif spec.transform == "gaussian_target":
                target = raw.new_tensor(spec.target)
                sigma = raw.new_tensor(spec.sigma)
                utility = torch.exp(-0.5 * ((raw - target) / sigma).square())
            else:
                target = raw.new_tensor(spec.target)
                scale = raw.new_tensor(spec.scale)
                utility = -(raw - target).abs() / scale
            return utility

    __call__ = transform

    def encode_measurements(self, Y_measured: torch.Tensor) -> torch.Tensor:
        """MEASUREMENT-space values into the MODEL space :meth:`transform` expects.

        :meth:`transform` is a *model-output* decoder: its first act is to undo the
        link, so a ``log`` objective is exponentiated before the utility is
        computed.  Handing it a raw measurement therefore exponentiates a value
        that was never a logarithm.

        **This fails silently and it has.** ``run_r1_ucb`` passed thickness in
        nanometres here until 2026-07-31; ``exp(360…1303)`` saturates the 650 nm
        Gaussian to exactly ``0.0``, which is finite, so neither the transform's
        own finiteness check nor the caller's noticed. Every observation's
        thickness utility was zero, and the UCB-HVI baseline hypervolume came out
        0.004659 where the correct value is 0.436442 -- a factor of 94, against
        which every candidate looked like a large improvement.

        Use this, or :meth:`transform_measurements`, wherever the values in hand
        are what the workbook reports rather than what the GP emits.
        """
        if not isinstance(Y_measured, torch.Tensor):
            raise TypeError("Y_measured must be a torch.Tensor.")
        if not Y_measured.is_floating_point():
            raise TypeError("Y_measured must use a floating dtype.")
        if Y_measured.ndim < 1 or Y_measured.shape[-1] != self.objective_count:
            raise ValueError(
                f"Y_measured final dimension must be {self.objective_count}; "
                f"got shape {tuple(Y_measured.shape)}."
            )
        if not torch.isfinite(Y_measured).all():
            raise ValueError("Y_measured must contain only finite values.")
        columns: list[torch.Tensor] = []
        for index, spec in enumerate(self.specs):
            column = Y_measured[..., index]
            if spec.model_link == "log":
                if not bool((column > 0).all()):
                    raise ValueError(
                        f"Objective {spec.name!r} has a log link, so its measured "
                        "values must be strictly positive."
                    )
                column = torch.log(column)
            columns.append(column)
        return torch.stack(columns, dim=-1)

    def transform_measurements(self, Y_measured: torch.Tensor) -> torch.Tensor:
        """Utility straight from MEASUREMENT-space values.

        The one-call safe route: :meth:`encode_measurements` then
        :meth:`transform`.  Prefer it at any call site holding workbook values, so
        the encoding step cannot be forgotten.
        """
        return self.transform(self.encode_measurements(Y_measured))

    def expected_transform(
        self, mean: torch.Tensor, variance: torch.Tensor
    ) -> torch.Tensor:
        """Expected utility ``E[transform(Y)]`` for ``Y ~ N(mean, variance)``.

        Use this when the GP is trained on a *raw* measurement and the utility is
        a nonlinear function of it.  Applying :meth:`transform` to the posterior
        mean is wrong in that case: it is biased by Jensen's inequality and it
        discards the posterior variance entirely, which for a target-seeking
        utility is precisely the information that matters.

        ``identity`` and ``affine`` are linear, so their expectation is just the
        transform of the mean.  The two target transforms are nonlinear and have
        exact closed forms:

        * ``gaussian_target`` with target ``c`` and width ``s``::

              E = s / sqrt(s^2 + v) * exp(-0.5 * (mu - c)^2 / (s^2 + v))

          At ``mu == c`` this decays from 1 as the posterior widens, so a
          confidently on-target candidate outranks an uncertain one.

        * ``negative_absolute_target`` uses the folded-normal mean.

        Both reduce to :meth:`transform` as ``variance -> 0``.
        """
        if not isinstance(mean, torch.Tensor) or not isinstance(variance, torch.Tensor):
            raise TypeError("mean and variance must be torch.Tensors.")
        if not mean.is_floating_point() or not variance.is_floating_point():
            raise TypeError("mean and variance must use a floating dtype.")
        if mean.shape != variance.shape:
            raise ValueError(
                f"mean and variance must share a shape; got {tuple(mean.shape)} "
                f"and {tuple(variance.shape)}."
            )
        if mean.ndim < 1 or mean.shape[-1] != self.objective_count:
            raise ValueError(
                f"mean final dimension must be {self.objective_count}; "
                f"got shape {tuple(mean.shape)}."
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise ValueError("mean and variance must contain only finite values.")
        if (variance < 0).any():
            raise ValueError("variance must be non-negative.")

        outputs: list[torch.Tensor] = []
        for index, spec in enumerate(self.specs):
            mu = mean[..., index]
            var = variance[..., index]
            if spec.model_link == "log":
                # the posterior is lognormal, so no Gaussian closed form applies;
                # integrate in log space by quadrature
                utility = self._quadrature_expectation(mu, var, spec, nodes=20)
            elif spec.transform in {"identity", "affine"}:
                # linear in the physical value, and the link is identity in this
                # branch, so E[f(Y)] = f(E[Y])
                utility = self._utility_from_physical(mu, spec)
            elif spec.transform == "gaussian_target":
                target = mu.new_tensor(spec.target)
                s2 = mu.new_tensor(spec.sigma) ** 2
                denom = s2 + var
                utility = torch.sqrt(s2 / denom) * torch.exp(
                    -0.5 * (mu - target).square() / denom
                )
            else:
                target = mu.new_tensor(spec.target)
                scale = mu.new_tensor(spec.scale)
                sd = var.clamp_min(0.0).sqrt()
                delta = mu - target
                # folded-normal mean; the sd == 0 branch degenerates to |delta|
                safe_sd = torch.where(sd > 0, sd, torch.ones_like(sd))
                folded = safe_sd * np.sqrt(2.0 / np.pi) * torch.exp(
                    -0.5 * (delta / safe_sd).square()
                ) + delta * torch.erf(delta / (safe_sd * np.sqrt(2.0)))
                folded = torch.where(sd > 0, folded, delta.abs())
                utility = -folded / scale
            outputs.append(utility)
        expected = torch.stack(outputs, dim=-1)
        if expected.shape != mean.shape:
            raise RuntimeError("Internal expected-objective shape error.")
        return expected

    def _quadrature_expectation(
        self,
        log_mean: torch.Tensor,
        log_variance: torch.Tensor,
        spec: ObjectiveSpec,
        *,
        nodes: int,
    ) -> torch.Tensor:
        """E[utility] for one log-link objective, by Gauss-Hermite in log space.

            E[g(Y)] = int g(exp(z)) N(z; m, s^2) dz
                    ~ (1/sqrt(pi)) sum_i w_i g(exp(m + sqrt(2) s x_i))

        Exact for the Gaussian weight, deterministic, differentiable, and
        cheaper than sampling.  Moment-matching the lognormal to a Gaussian and
        reusing the closed form is ~500x less accurate here, and its bias
        changes sign across the range, which reorders candidates rather than
        merely shifting them.
        """
        raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(nodes)
        abscissa = log_mean.new_tensor(raw_nodes)
        weights = log_mean.new_tensor(raw_weights) / float(np.sqrt(np.pi))
        sd = log_variance.clamp_min(0.0).sqrt()
        shape = (-1, *([1] * log_mean.ndim))
        shifted = log_mean.unsqueeze(0) + np.sqrt(2.0) * sd.unsqueeze(
            0
        ) * abscissa.view(shape)
        utilities = self._utility_from_physical(torch.exp(shifted), spec)
        return (utilities * weights.view(shape)).sum(dim=0)

    def expected_transform_lognormal(
        self,
        log_mean: torch.Tensor,
        log_variance: torch.Tensor,
        *,
        nodes: int = 20,
    ) -> torch.Tensor:
        """Expected utility treating EVERY objective as log-link.

        Prefer :meth:`expected_transform`, which dispatches per objective from
        each spec's ``model_link``.  This method is kept for the single-objective
        case where the caller knows the posterior is lognormal.
        """
        if not isinstance(log_mean, torch.Tensor) or not isinstance(
            log_variance, torch.Tensor
        ):
            raise TypeError("log_mean and log_variance must be torch.Tensors.")
        if not log_mean.is_floating_point() or not log_variance.is_floating_point():
            raise TypeError("log_mean and log_variance must use a floating dtype.")
        if log_mean.shape != log_variance.shape:
            raise ValueError(
                f"log_mean and log_variance must share a shape; got "
                f"{tuple(log_mean.shape)} and {tuple(log_variance.shape)}."
            )
        if log_mean.ndim < 1 or log_mean.shape[-1] != self.objective_count:
            raise ValueError(
                f"log_mean final dimension must be {self.objective_count}; "
                f"got shape {tuple(log_mean.shape)}."
            )
        if not torch.isfinite(log_mean).all() or not torch.isfinite(log_variance).all():
            raise ValueError("log_mean and log_variance must be finite.")
        if (log_variance < 0).any():
            raise ValueError("log_variance must be non-negative.")
        if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 2:
            raise ValueError("nodes must be an integer of at least 2.")
        columns = [
            self._quadrature_expectation(
                log_mean[..., index], log_variance[..., index], spec, nodes=nodes
            )
            for index, spec in enumerate(self.specs)
        ]
        return torch.stack(columns, dim=-1)


class ConfiguredMCMultiOutputObjective(MCMultiOutputObjective):
    """BoTorch MC objective backed by the exact same `ObjectiveTransform`."""

    def __init__(self, objective_transform: ObjectiveTransform) -> None:
        super().__init__()
        if not isinstance(objective_transform, ObjectiveTransform):
            raise TypeError("objective_transform must be an ObjectiveTransform.")
        self.objective_transform = objective_transform

    def forward(
        self, samples: torch.Tensor, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        del X
        return self.objective_transform.transform(samples)


def _validate_posterior_sample_bounds(
    bounds: Sequence[UtilityBound], objective_transform: ObjectiveTransform
) -> tuple[UtilityBound, ...]:
    if isinstance(bounds, (str, bytes)):
        raise TypeError("bounds must be an ordered sequence of (lower, upper) pairs.")
    try:
        raw_bounds = tuple(bounds)
    except TypeError as exc:
        raise TypeError(
            "bounds must be an ordered sequence of (lower, upper) pairs."
        ) from exc
    if len(raw_bounds) != objective_transform.objective_count:
        raise ValueError(
            "bounds must contain one (lower, upper) pair per objective; "
            f"expected {objective_transform.objective_count}, got {len(raw_bounds)}."
        )

    validated: list[UtilityBound] = []
    bounded_count = 0
    for index, (raw_bound, spec) in enumerate(
        zip(raw_bounds, objective_transform.specs)
    ):
        if isinstance(raw_bound, (str, bytes)):
            raise TypeError(f"bounds[{index}] must be a (lower, upper) pair.")
        try:
            pair = tuple(raw_bound)
        except TypeError as exc:
            raise TypeError(f"bounds[{index}] must be a (lower, upper) pair.") from exc
        if len(pair) != 2:
            raise ValueError(f"bounds[{index}] must contain exactly two values.")
        lower = _finite_optional(
            pair[0], field="posterior_sample_lower_bound", name=spec.name
        )
        upper = _finite_optional(
            pair[1], field="posterior_sample_upper_bound", name=spec.name
        )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"Objective {spec.name!r} posterior-sample lower bound must not "
                "exceed its upper bound."
            )
        if lower is not None or upper is not None:
            bounded_count += 1
            if spec.transform != "identity" or spec.goal != "maximize":
                raise ValueError(
                    "Posterior-sample bounds are supported only for explicit "
                    f"identity/maximize utilities; objective {spec.name!r} uses "
                    f"{spec.transform!r}/{spec.goal!r}."
                )
        validated.append((lower, upper))
    if bounded_count == 0:
        raise ValueError("At least one posterior-sample utility bound is required.")
    return tuple(validated)


class BoundedPosteriorSampleTransform:
    """Clamp declared identity utilities only after transforming MC samples.

    This acquisition-only wrapper never changes observed training targets or the
    underlying versioned objective contract.  It is intended for explicit
    synthetic qLogNEHVI policy comparisons.
    """

    def __init__(
        self,
        objective_transform: ObjectiveTransform,
        bounds: Sequence[UtilityBound],
    ) -> None:
        if not isinstance(objective_transform, ObjectiveTransform):
            raise TypeError("objective_transform must be an ObjectiveTransform.")
        self.objective_transform = objective_transform
        self.bounds = _validate_posterior_sample_bounds(bounds, objective_transform)
        self.version = f"{objective_transform.version}+posterior-sample-bounds-v1"

    def transform(self, samples: torch.Tensor) -> torch.Tensor:
        utilities = self.objective_transform.transform(samples)
        bounded_columns: list[torch.Tensor] = []
        for index, (lower, upper) in enumerate(self.bounds):
            utility = utilities[..., index]
            if lower is not None or upper is not None:
                utility = torch.clamp(utility, min=lower, max=upper)
            bounded_columns.append(utility)
        bounded = torch.stack(bounded_columns, dim=-1)
        if bounded.shape != utilities.shape:
            raise RuntimeError("Internal posterior-sample bounds shape error.")
        return bounded

    __call__ = transform


class BoundedMCMultiOutputObjective(MCMultiOutputObjective):
    """BoTorch objective applying explicit bounds to posterior utility samples."""

    def __init__(
        self,
        objective_transform: ObjectiveTransform,
        bounds: Sequence[UtilityBound],
    ) -> None:
        super().__init__()
        self.posterior_sample_transform = BoundedPosteriorSampleTransform(
            objective_transform, bounds
        )
        self.objective_transform = objective_transform
        self.bounds = self.posterior_sample_transform.bounds
        self.version = self.posterior_sample_transform.version

    def forward(
        self, samples: torch.Tensor, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        del X
        return self.posterior_sample_transform.transform(samples)


__all__ = [
    "BoundedMCMultiOutputObjective",
    "BoundedPosteriorSampleTransform",
    "ConfiguredMCMultiOutputObjective",
    "ObjectiveSpec",
    "ObjectiveTransform",
    "UtilityBound",
]
