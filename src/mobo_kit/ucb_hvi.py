"""Discrete multi-objective UCB-HVI scoring in transformed utility space.

The module deliberately separates posterior sampling from deterministic
hypervolume arithmetic.  Nonlinear objective transforms are applied to every
posterior Monte Carlo sample before utility moments are formed.  All objective
dimensions are maximized after transformation, and the reference point is
always supplied explicitly by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from numbers import Real
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated


TensorTransform = Callable[[torch.Tensor], torch.Tensor]
MomentMethod = Literal["monte_carlo", "analytic_identity"]
UCBBoundPolicy = Literal["none", "clip_ucb"]
UtilityBound = tuple[float | None, float | None]


@dataclass(frozen=True)
class PosteriorIdentityMoments:
    """Exact posterior moments for an all-identity maximize contract."""

    utility_mean: torch.Tensor
    utility_std: torch.Tensor
    observation_noise: bool
    objective_contract_version: str
    moment_method: str = "analytic_identity"


@dataclass(frozen=True)
class BoundedUCBResult:
    """Raw and policy-effective UCB vectors plus non-negative clip amounts."""

    utility_ucb_raw: np.ndarray
    utility_ucb_effective: np.ndarray
    utility_ucb_clip_amount: np.ndarray
    policy: str
    bounds: tuple[UtilityBound, ...]


@dataclass(frozen=True)
class PosteriorUtilityMoments:
    """Monte Carlo moments in all-maximize transformed utility space."""

    utility_mean: np.ndarray
    utility_std: np.ndarray
    mc_samples: int
    seed: int
    observation_noise: bool
    standard_deviation_correction: int = 0
    moment_method: str = "monte_carlo"


@dataclass(frozen=True)
class UCBHVIScoreResult:
    """Candidate-wise optimistic utilities and hypervolume improvements."""

    base_score: np.ndarray
    base_log_score: np.ndarray
    utility_mean: np.ndarray
    utility_std: np.ndarray
    utility_ucb: np.ndarray
    baseline_hypervolume: float
    pareto_utility: np.ndarray
    reference_point_utility: np.ndarray
    beta: float
    kappa: float
    mc_samples: int | None
    seed: int | None
    observation_noise: bool
    objective_contract_version: str
    moment_method: str = "monte_carlo"
    bound_policy: str = "none"
    utility_bounds: tuple[UtilityBound, ...] | None = None
    utility_ucb_raw: np.ndarray | None = None
    utility_ucb_effective: np.ndarray | None = None
    utility_ucb_clip_amount: np.ndarray | None = None
    method: str = "ucb_hvi"
    method_version: str = "step2a-v1"

    def __post_init__(self) -> None:
        effective = (
            np.asarray(self.utility_ucb, dtype=float)
            if self.utility_ucb_effective is None
            else np.asarray(self.utility_ucb_effective, dtype=float)
        )
        raw = (
            effective.copy()
            if self.utility_ucb_raw is None
            else np.asarray(self.utility_ucb_raw, dtype=float)
        )
        clip_amount = (
            np.abs(raw - effective)
            if self.utility_ucb_clip_amount is None
            else np.asarray(self.utility_ucb_clip_amount, dtype=float)
        )
        expected_shape = np.asarray(self.utility_mean).shape
        if (
            effective.shape != expected_shape
            or raw.shape != expected_shape
            or clip_amount.shape != expected_shape
        ):
            raise ValueError(
                "Raw/effective UCB and clip amounts must align with utility_mean."
            )
        object.__setattr__(self, "utility_ucb", effective)
        object.__setattr__(self, "utility_ucb_raw", raw)
        object.__setattr__(self, "utility_ucb_effective", effective)
        object.__setattr__(self, "utility_ucb_clip_amount", clip_amount)


@dataclass(frozen=True)
class UCBHVIBatchProposal:
    """A locally penalized batch plus its complete static UCB-HVI score table."""

    selection: Any
    scoring: UCBHVIScoreResult
    positive_score_tolerance: float
    metadata: dict[str, Any]


def _finite_nonnegative(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real non-boolean number; got {value!r}.")
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative; got {value!r}.")
    return number


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return result


def _apply_objective_transform(transform: Any, samples: torch.Tensor) -> torch.Tensor:
    if hasattr(transform, "transform"):
        utility = transform.transform(samples)
    elif callable(transform):
        utility = transform(samples)
    else:
        raise TypeError("objective_transform must be callable or expose transform().")
    if not isinstance(utility, torch.Tensor):
        raise TypeError("objective_transform must return a torch.Tensor.")
    if utility.shape != samples.shape:
        raise ValueError(
            "objective_transform must preserve shape; "
            f"got {tuple(samples.shape)} -> {tuple(utility.shape)}."
        )
    if not torch.isfinite(utility).all():
        raise ValueError("objective_transform produced non-finite utility values.")
    return utility


def _posterior(model: Any, X: torch.Tensor, observation_noise: bool) -> Any:
    try:
        return model.posterior(X, observation_noise=observation_noise)
    except TypeError as exc:
        raise TypeError(
            "model.posterior must accept the explicit observation_noise keyword."
        ) from exc


def _identity_objective_contract(objective_transform: Any) -> Any:
    if hasattr(objective_transform, "bounds"):
        raise ValueError(
            "analytic identity moments cannot be used with a posterior-sample "
            "bounds wrapper; apply an explicit UCB bound policy instead."
        )
    contract = getattr(objective_transform, "objective_transform", objective_transform)
    specs = getattr(contract, "specs", None)
    if not specs:
        raise ValueError(
            "analytic identity moments require an explicit versioned objective "
            "contract with objective specifications."
        )
    if any(
        getattr(spec, "transform", None) != "identity"
        or getattr(spec, "goal", None) != "maximize"
        or bool(getattr(spec, "clip", False))
        for spec in specs
    ):
        raise ValueError(
            "analytic identity moments require every objective to use the "
            "identity transform with maximize direction and no clipping."
        )
    version = getattr(contract, "version", None)
    if not isinstance(version, str) or not version.strip():
        raise ValueError(
            "analytic identity moments require a non-empty objective contract version."
        )
    return contract


def posterior_identity_moments(
    model: Any,
    X_pool_norm: torch.Tensor,
    objective_transform: Any,
    *,
    chunk_size: int = 512,
    observation_noise: bool = False,
) -> PosteriorIdentityMoments:
    """Return exact posterior mean and standard deviation for identity utilities.

    Tensors remain on the posterior's device and retain its dtype and any model
    batch dimensions. Candidate rows are concatenated along the posterior q
    dimension, making the result invariant to the requested evaluation chunk.
    """
    if not isinstance(X_pool_norm, torch.Tensor) or X_pool_norm.ndim != 2:
        shape = getattr(X_pool_norm, "shape", None)
        raise ValueError(
            f"X_pool_norm must be a tensor with shape (N, D); got {shape}."
        )
    if not X_pool_norm.is_floating_point():
        raise TypeError("X_pool_norm must use a floating dtype.")
    if X_pool_norm.shape[0] == 0:
        raise ValueError("X_pool_norm must contain at least one candidate.")
    if not torch.isfinite(X_pool_norm).all():
        raise ValueError("X_pool_norm must contain only finite values.")
    chunk = _positive_integer(chunk_size, name="chunk_size")
    if not isinstance(observation_noise, bool):
        raise ValueError("observation_noise must be a boolean.")
    contract = _identity_objective_contract(objective_transform)
    objective_count = len(contract.specs)

    mean_parts: list[torch.Tensor] = []
    std_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, X_pool_norm.shape[0], chunk):
            X_chunk = X_pool_norm[start : start + chunk]
            posterior = _posterior(model, X_chunk, observation_noise)
            mean = posterior.mean
            variance = posterior.variance
            if not isinstance(mean, torch.Tensor) or not isinstance(
                variance, torch.Tensor
            ):
                raise TypeError("model.posterior mean and variance must be tensors.")
            if mean.shape != variance.shape:
                raise ValueError("Posterior mean and variance shapes must match.")
            if mean.ndim < 2 or mean.shape[-2] != X_chunk.shape[0]:
                raise ValueError(
                    "Posterior candidate dimension must align with the input chunk; "
                    f"got mean shape {tuple(mean.shape)} for {X_chunk.shape[0]} rows."
                )
            if mean.shape[-1] != objective_count:
                raise ValueError(
                    "Posterior objective dimension does not match the identity "
                    f"contract; expected {objective_count}, got {mean.shape[-1]}."
                )
            if not mean.is_floating_point() or not variance.is_floating_point():
                raise TypeError("Posterior mean and variance must use floating dtypes.")
            if mean.dtype != variance.dtype or mean.device != variance.device:
                raise ValueError(
                    "Posterior mean and variance must share dtype and device."
                )
            if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
                raise ValueError("Posterior mean and variance must be finite.")
            tolerance = 100.0 * torch.finfo(variance.dtype).eps
            if torch.any(variance < -tolerance):
                raise ValueError("Posterior variance cannot be materially negative.")
            mean_parts.append(mean)
            std_parts.append(variance.clamp_min(0.0).sqrt())

    try:
        means = torch.cat(mean_parts, dim=-2)
        standard_deviations = torch.cat(std_parts, dim=-2)
    except RuntimeError as exc:
        raise ValueError(
            "Posterior batch shape, dtype, and device must remain stable across chunks."
        ) from exc
    return PosteriorIdentityMoments(
        utility_mean=means,
        utility_std=standard_deviations,
        observation_noise=observation_noise,
        objective_contract_version=contract.version.strip(),
    )


def posterior_utility_moments(
    model: Any,
    X_pool_norm: torch.Tensor,
    objective_transform: TensorTransform | Any,
    *,
    mc_samples: int = 128,
    seed: int = 0,
    chunk_size: int = 512,
    observation_noise: bool = False,
) -> PosteriorUtilityMoments:
    """Estimate utility moments for a normalized ``(N, D)`` candidate pool.

    Candidates are represented as independent singleton q-batches ``N x 1 x D``.
    A seeded Sobol sampler is rebuilt for each chunk.  BoTorch collapses the
    singleton batch dimensions in its base-sample shape, so every candidate sees
    the same reproducible QMC normal draws and results are invariant to chunking.
    Population standard deviation (``correction=0``) is reported.
    """
    if not isinstance(X_pool_norm, torch.Tensor) or X_pool_norm.ndim != 2:
        shape = getattr(X_pool_norm, "shape", None)
        raise ValueError(
            f"X_pool_norm must be a tensor with shape (N, D); got {shape}."
        )
    if not X_pool_norm.is_floating_point():
        raise TypeError("X_pool_norm must use a floating dtype.")
    if not torch.isfinite(X_pool_norm).all():
        raise ValueError("X_pool_norm must contain only finite values.")
    sample_count = _positive_integer(mc_samples, name="mc_samples")
    chunk = _positive_integer(chunk_size, name="chunk_size")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer.")
    if not isinstance(observation_noise, bool):
        raise ValueError("observation_noise must be a boolean.")

    mean_parts: list[torch.Tensor] = []
    std_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, X_pool_norm.shape[0], chunk):
            X_chunk = X_pool_norm[start : start + chunk].unsqueeze(-2)
            posterior = _posterior(model, X_chunk, observation_noise)
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([sample_count]), seed=int(seed)
            )
            raw_samples = sampler(posterior)
            utility_samples = _apply_objective_transform(
                objective_transform, raw_samples
            )
            if utility_samples.shape[-2] != 1:
                raise ValueError(
                    "Singleton posterior evaluation must retain q=1 in the "
                    f"penultimate dimension; got {tuple(utility_samples.shape)}."
                )
            utility_samples = utility_samples.squeeze(-2)
            mean_parts.append(utility_samples.mean(dim=0))
            std_parts.append(utility_samples.std(dim=0, correction=0))

    if mean_parts:
        means = torch.cat(mean_parts, dim=0)
        standard_deviations = torch.cat(std_parts, dim=0)
    else:
        # The output dimension cannot be discovered safely without a posterior.
        raise ValueError("X_pool_norm must contain at least one candidate.")
    return PosteriorUtilityMoments(
        utility_mean=means.detach().cpu().double().numpy(),
        utility_std=standard_deviations.detach().cpu().double().numpy(),
        mc_samples=sample_count,
        seed=int(seed),
        observation_noise=bool(observation_noise),
    )


def _utility_matrix(value: np.ndarray | torch.Tensor, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().double().numpy()
    else:
        array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (N, M); got {array.shape}.")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must include at least one objective.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validated_utility_bounds(
    bounds: Sequence[UtilityBound] | None,
    objective_count: int,
    *,
    require_bounded: bool,
) -> tuple[UtilityBound, ...]:
    if bounds is None:
        if require_bounded:
            raise ValueError("clip_ucb requires explicit per-objective utility bounds.")
        return tuple((None, None) for _ in range(objective_count))
    if isinstance(bounds, (str, bytes)):
        raise TypeError("bounds must be an ordered sequence of (lower, upper) pairs.")
    try:
        raw_bounds = tuple(bounds)
    except TypeError as exc:
        raise TypeError(
            "bounds must be an ordered sequence of (lower, upper) pairs."
        ) from exc
    if len(raw_bounds) != objective_count:
        raise ValueError(
            "bounds must contain one (lower, upper) pair per objective; "
            f"expected {objective_count}, got {len(raw_bounds)}."
        )
    validated: list[UtilityBound] = []
    bounded_count = 0
    for index, raw_bound in enumerate(raw_bounds):
        if isinstance(raw_bound, (str, bytes)):
            raise TypeError(f"bounds[{index}] must be a (lower, upper) pair.")
        try:
            pair = tuple(raw_bound)
        except TypeError as exc:
            raise TypeError(f"bounds[{index}] must be a (lower, upper) pair.") from exc
        if len(pair) != 2:
            raise ValueError(f"bounds[{index}] must contain exactly two values.")
        converted: list[float | None] = []
        for label, value in zip(("lower", "upper"), pair):
            if value is None:
                converted.append(None)
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(
                    f"bounds[{index}] {label} must be finite, real, and non-boolean."
                )
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"bounds[{index}] {label} must be finite.")
            converted.append(number)
        lower, upper = converted
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(
                f"bounds[{index}] lower value must not exceed its upper value."
            )
        if lower is not None or upper is not None:
            bounded_count += 1
        validated.append((lower, upper))
    if require_bounded and bounded_count == 0:
        raise ValueError("clip_ucb requires at least one finite utility bound.")
    return tuple(validated)


def apply_ucb_bound_policy(
    raw_ucb: np.ndarray | torch.Tensor,
    bounds: Sequence[UtilityBound] | None,
    policy: UCBBoundPolicy | str,
) -> BoundedUCBResult:
    """Apply an explicit optimistic-utility policy without changing targets."""
    raw = _utility_matrix(raw_ucb, name="raw_ucb").copy()
    if policy not in {"none", "clip_ucb"}:
        raise ValueError("policy must be exactly 'none' or 'clip_ucb'.")
    validated = _validated_utility_bounds(
        bounds, raw.shape[1], require_bounded=policy == "clip_ucb"
    )
    effective = raw.copy()
    if policy == "clip_ucb":
        for index, (lower, upper) in enumerate(validated):
            effective[:, index] = np.clip(
                effective[:, index],
                -np.inf if lower is None else lower,
                np.inf if upper is None else upper,
            )
    return BoundedUCBResult(
        utility_ucb_raw=raw,
        utility_ucb_effective=effective,
        utility_ucb_clip_amount=np.abs(raw - effective),
        policy=str(policy),
        bounds=validated,
    )


def _reference_point(
    value: np.ndarray | torch.Tensor | None, objective_count: int
) -> np.ndarray:
    if value is None:
        raise ValueError(
            "reference_point_utility is required and is never derived from data."
        )
    if isinstance(value, torch.Tensor):
        reference = value.detach().cpu().double().numpy()
    else:
        reference = np.asarray(value, dtype=float)
    if reference.shape != (objective_count,):
        raise ValueError(
            "reference_point_utility must have shape "
            f"({objective_count},); got {reference.shape}."
        )
    if not np.all(np.isfinite(reference)):
        raise ValueError("reference_point_utility must contain only finite values.")
    return reference


def pareto_utility_above_reference(
    observed_utility: np.ndarray | torch.Tensor,
    reference_point_utility: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Return finite non-dominated utilities that strictly dominate the reference."""
    observed = _utility_matrix(observed_utility, name="observed_utility")
    reference = _reference_point(reference_point_utility, observed.shape[1])
    contributing = observed[np.all(observed > reference, axis=1)]
    if contributing.shape[0] == 0:
        return np.empty((0, observed.shape[1]), dtype=float)
    tensor = torch.as_tensor(contributing, dtype=torch.double)
    return tensor[is_non_dominated(tensor)].numpy()


def hypervolume_improvement_scores(
    optimistic_utility: np.ndarray | torch.Tensor,
    observed_utility: np.ndarray | torch.Tensor,
    reference_point_utility: np.ndarray | torch.Tensor | None,
    *,
    numeric_tolerance: float = 1e-12,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Compute exact singleton HVI for optimistic all-maximize utility vectors."""
    candidates = _utility_matrix(optimistic_utility, name="optimistic_utility")
    observed = _utility_matrix(observed_utility, name="observed_utility")
    if candidates.shape[1] != observed.shape[1]:
        raise ValueError("Candidate and observed utility dimensions must match.")
    reference = _reference_point(reference_point_utility, observed.shape[1])
    tolerance = _finite_nonnegative(numeric_tolerance, name="numeric_tolerance")
    chunk = _positive_integer(chunk_size, name="chunk_size")
    pareto = pareto_utility_above_reference(observed, reference)
    ref_tensor = torch.as_tensor(reference, dtype=torch.double)
    hypervolume = Hypervolume(ref_point=ref_tensor)
    baseline = (
        0.0
        if pareto.shape[0] == 0
        else float(hypervolume.compute(torch.as_tensor(pareto, dtype=torch.double)))
    )

    scores = np.zeros(candidates.shape[0], dtype=float)
    for start in range(0, candidates.shape[0], chunk):
        stop = min(start + chunk, candidates.shape[0])
        for index in range(start, stop):
            candidate = candidates[index]
            if not np.all(candidate > reference):
                continue
            if pareto.shape[0] and np.any(
                np.all(pareto >= candidate - tolerance, axis=1)
            ):
                continue
            augmented = np.vstack([pareto, candidate[None, :]])
            augmented_tensor = torch.as_tensor(augmented, dtype=torch.double)
            augmented_pareto = augmented_tensor[is_non_dominated(augmented_tensor)]
            improvement = float(hypervolume.compute(augmented_pareto)) - baseline
            if improvement < -tolerance:
                raise RuntimeError(
                    "Hypervolume improvement was negative beyond numeric "
                    f"tolerance: candidate_index={index}, improvement={improvement}, "
                    f"tolerance={tolerance}."
                )
            if improvement > tolerance:
                scores[index] = improvement
    return scores, baseline, pareto, reference


def score_ucb_hvi_from_moments(
    utility_mean: np.ndarray,
    utility_std: np.ndarray,
    observed_utility: np.ndarray,
    reference_point_utility: np.ndarray | None,
    *,
    beta: float,
    numeric_tolerance: float = 1e-12,
    chunk_size: int = 1024,
    log_epsilon: float = 1e-12,
    mc_samples: int | None = None,
    seed: int | None = None,
    observation_noise: bool = False,
    objective_contract_version: str = "direct-moments",
    moment_method: str = "direct_moments",
    bound_policy: UCBBoundPolicy | str = "none",
    utility_bounds: Sequence[UtilityBound] | None = None,
) -> UCBHVIScoreResult:
    """Form utility UCB vectors and score their singleton hypervolume gain."""
    means = _utility_matrix(utility_mean, name="utility_mean")
    standard_deviations = _utility_matrix(utility_std, name="utility_std")
    if standard_deviations.shape != means.shape:
        raise ValueError("utility_std must have the same shape as utility_mean.")
    if np.any(standard_deviations < 0):
        raise ValueError("utility_std cannot contain negative values.")
    beta_value = _finite_nonnegative(beta, name="beta")
    if isinstance(log_epsilon, (bool, np.bool_)):
        raise ValueError("log_epsilon must be a real non-boolean number.")
    epsilon = float(log_epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("log_epsilon must be finite and strictly positive.")
    if (
        not isinstance(objective_contract_version, str)
        or not objective_contract_version.strip()
    ):
        raise ValueError("objective_contract_version must be a non-empty string.")
    if not isinstance(moment_method, str) or not moment_method.strip():
        raise ValueError("moment_method must be a non-empty string.")
    kappa = sqrt(beta_value)
    optimistic_raw = means + kappa * standard_deviations
    bounded = apply_ucb_bound_policy(optimistic_raw, utility_bounds, bound_policy)
    scores, baseline, pareto, reference = hypervolume_improvement_scores(
        bounded.utility_ucb_effective,
        observed_utility,
        reference_point_utility,
        numeric_tolerance=numeric_tolerance,
        chunk_size=chunk_size,
    )
    return UCBHVIScoreResult(
        base_score=scores,
        base_log_score=np.log(np.maximum(scores, epsilon)),
        utility_mean=means,
        utility_std=standard_deviations,
        utility_ucb=bounded.utility_ucb_effective,
        baseline_hypervolume=baseline,
        pareto_utility=pareto,
        reference_point_utility=reference,
        beta=beta_value,
        kappa=kappa,
        mc_samples=mc_samples,
        seed=seed,
        observation_noise=bool(observation_noise),
        objective_contract_version=objective_contract_version.strip(),
        moment_method=moment_method.strip(),
        bound_policy=bounded.policy,
        utility_bounds=bounded.bounds,
        utility_ucb_raw=bounded.utility_ucb_raw,
        utility_ucb_effective=bounded.utility_ucb_effective,
        utility_ucb_clip_amount=bounded.utility_ucb_clip_amount,
    )


def score_ucb_hvi_pool(
    model: Any,
    X_pool_norm: torch.Tensor,
    observed_raw: np.ndarray | torch.Tensor,
    objective_transform: TensorTransform | Any,
    reference_point_utility: np.ndarray | torch.Tensor | None,
    *,
    beta: float,
    mc_samples: int = 128,
    seed: int = 0,
    posterior_chunk_size: int = 512,
    hvi_chunk_size: int = 1024,
    observation_noise: bool = False,
    numeric_tolerance: float = 1e-12,
    log_epsilon: float = 1e-12,
    moment_method: MomentMethod | str = "monte_carlo",
    bound_policy: UCBBoundPolicy | str = "none",
    utility_bounds: Sequence[UtilityBound] | None = None,
) -> UCBHVIScoreResult:
    """Score a normalized discrete pool from raw-output model posteriors."""
    if moment_method == "monte_carlo":
        moments = posterior_utility_moments(
            model,
            X_pool_norm,
            objective_transform,
            mc_samples=mc_samples,
            seed=seed,
            chunk_size=posterior_chunk_size,
            observation_noise=observation_noise,
        )
        utility_mean = moments.utility_mean
        utility_std = moments.utility_std
        resolved_mc_samples: int | None = moments.mc_samples
        resolved_seed: int | None = moments.seed
        resolved_method = moments.moment_method
        resolved_observation_noise = moments.observation_noise
    elif moment_method == "analytic_identity":
        analytic = posterior_identity_moments(
            model,
            X_pool_norm,
            objective_transform,
            chunk_size=posterior_chunk_size,
            observation_noise=observation_noise,
        )
        if analytic.utility_mean.ndim != 2 or analytic.utility_std.ndim != 2:
            raise ValueError(
                "UCB-HVI pool scoring requires unbatched analytic posterior moments "
                "with shape (N, M)."
            )
        utility_mean = analytic.utility_mean.detach().cpu().numpy()
        utility_std = analytic.utility_std.detach().cpu().numpy()
        resolved_mc_samples = None
        resolved_seed = None
        resolved_method = analytic.moment_method
        resolved_observation_noise = analytic.observation_noise
    else:
        raise ValueError(
            "moment_method must be exactly 'monte_carlo' or 'analytic_identity'."
        )
    raw = torch.as_tensor(
        observed_raw,
        dtype=X_pool_norm.dtype,
        device=X_pool_norm.device,
    )
    if raw.ndim != 2:
        raise ValueError(
            f"observed_raw must have shape (N, M); got {tuple(raw.shape)}."
        )
    observed_utility_tensor = _apply_objective_transform(objective_transform, raw)
    contract_version = getattr(objective_transform, "version", None)
    if contract_version is None and hasattr(objective_transform, "objective_transform"):
        contract_version = getattr(
            objective_transform.objective_transform, "version", None
        )
    if not isinstance(contract_version, str) or not contract_version.strip():
        raise ValueError(
            "objective_transform must expose a non-empty objective contract version."
        )
    return score_ucb_hvi_from_moments(
        utility_mean,
        utility_std,
        observed_utility_tensor,
        reference_point_utility,
        beta=beta,
        numeric_tolerance=numeric_tolerance,
        chunk_size=hvi_chunk_size,
        log_epsilon=log_epsilon,
        mc_samples=resolved_mc_samples,
        seed=resolved_seed,
        observation_noise=resolved_observation_noise,
        objective_contract_version=contract_version,
        moment_method=resolved_method,
        bound_policy=bound_policy,
        utility_bounds=utility_bounds,
    )


def propose_ucb_hvi_batch(
    candidate_pool: Any,
    model: Any,
    observed_raw: np.ndarray | torch.Tensor,
    objective_transform: TensorTransform | Any,
    reference_point_utility: np.ndarray | torch.Tensor | None,
    *,
    q: int,
    beta: float,
    local_penalization_config: Any,
    observed_pending_norm: np.ndarray | None = None,
    positive_score_tolerance: float = 1e-12,
    mc_samples: int = 128,
    seed: int = 0,
    posterior_chunk_size: int = 512,
    hvi_chunk_size: int = 1024,
    observation_noise: bool = False,
    numeric_tolerance: float = 1e-12,
    log_epsilon: float = 1e-12,
    moment_method: MomentMethod | str = "monte_carlo",
    bound_policy: UCBBoundPolicy | str = "none",
    utility_bounds: Sequence[UtilityBound] | None = None,
) -> UCBHVIBatchProposal:
    """Select exactly ``q`` locally penalized positive-HVI pool candidates.

    The UCB-HVI score is static for the pool, while the shared selector updates
    the soft local penalty and hard distance masks after every selection.  A
    candidate whose true raw HVI is not greater than
    ``positive_score_tolerance`` is ineligible (log score ``-inf``).
    """
    from .batch_selection import BaseScoreResult, select_local_penalized_batch

    if isinstance(positive_score_tolerance, (bool, np.bool_)):
        raise ValueError("positive_score_tolerance must be a real non-boolean number.")
    tolerance = float(positive_score_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            "positive_score_tolerance must be finite and strictly positive."
        )
    try:
        model_parameter = next(model.parameters())
        model_dtype = model_parameter.dtype
        model_device = model_parameter.device
    except (AttributeError, StopIteration):
        model_dtype = torch.double
        model_device = torch.device("cpu")
    X_pool = torch.as_tensor(
        candidate_pool.X_norm, dtype=model_dtype, device=model_device
    )
    scoring = score_ucb_hvi_pool(
        model,
        X_pool,
        observed_raw,
        objective_transform,
        reference_point_utility,
        beta=beta,
        mc_samples=mc_samples,
        seed=seed,
        posterior_chunk_size=posterior_chunk_size,
        hvi_chunk_size=hvi_chunk_size,
        observation_noise=observation_noise,
        numeric_tolerance=numeric_tolerance,
        log_epsilon=log_epsilon,
        moment_method=moment_method,
        bound_policy=bound_policy,
        utility_bounds=utility_bounds,
    )

    def score_remaining(
        remaining_indices: np.ndarray, selected_indices: np.ndarray
    ) -> Any:
        del selected_indices
        raw_scores = scoring.base_score[remaining_indices]
        log_scores = scoring.base_log_score[remaining_indices].copy()
        log_scores[raw_scores <= tolerance] = -np.inf
        return BaseScoreResult(
            base_log_score=log_scores,
            base_score=raw_scores,
            diagnostics={
                "utility_mean": scoring.utility_mean[remaining_indices],
                "utility_std": scoring.utility_std[remaining_indices],
                "utility_ucb": scoring.utility_ucb[remaining_indices],
                "utility_ucb_raw": scoring.utility_ucb_raw[remaining_indices],
                "utility_ucb_effective": scoring.utility_ucb_effective[
                    remaining_indices
                ],
                "utility_ucb_clip_amount": scoring.utility_ucb_clip_amount[
                    remaining_indices
                ],
                "eligible_positive_hvi": raw_scores > tolerance,
            },
        )

    selection = select_local_penalized_batch(
        candidate_pool,
        q,
        score_remaining,
        local_penalization_config,
        observed_pending_norm=observed_pending_norm,
    )
    return UCBHVIBatchProposal(
        selection=selection,
        scoring=scoring,
        positive_score_tolerance=tolerance,
        metadata={
            "method": scoring.method,
            "method_version": scoring.method_version,
            "objective_contract_version": scoring.objective_contract_version,
            "reference_point_utility": scoring.reference_point_utility.copy(),
            "pool_seed": candidate_pool.seed,
            "pool_size": candidate_pool.size,
            "pool_draws": candidate_pool.draws,
            "pool_rejected_duplicate": candidate_pool.rejected_duplicate,
            "pool_rejected_avoid": candidate_pool.rejected_avoid,
            "pool_rejected_constraint": candidate_pool.rejected_constraint,
            "posterior_seed": scoring.seed,
            "mc_samples": scoring.mc_samples,
            "moment_method": scoring.moment_method,
            "bound_policy": scoring.bound_policy,
            "utility_bounds": scoring.utility_bounds,
            "beta": scoring.beta,
            "kappa": scoring.kappa,
            "observation_noise": scoring.observation_noise,
            "positive_score_tolerance": tolerance,
            "local_penalization": {
                "radius": local_penalization_config.radius,
                "min_batch_distance": local_penalization_config.min_batch_distance,
                "min_observed_distance": local_penalization_config.min_observed_distance,
                "dimension_weights": local_penalization_config.dimension_weights,
                "epsilon": local_penalization_config.epsilon,
            },
            "selected_pool_indices": selection.selected_pool_indices.copy(),
        },
    )
