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
            raw = Y[..., index]
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
            outputs.append(utility)
        transformed = torch.stack(outputs, dim=-1)
        if transformed.shape != Y.shape:
            raise RuntimeError("Internal objective transform shape error.")
        return transformed

    __call__ = transform


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
