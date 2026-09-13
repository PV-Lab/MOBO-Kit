"""Discrete singleton-pool qLogNEHVI scoring and sequential batch proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler


@dataclass(frozen=True)
class QLogNEHVIPoolScoreResult:
    """Singleton qLogNEHVI values and the pending context used to compute them."""

    base_log_score: np.ndarray
    evaluated_shape: tuple[int, int, int]
    pending_count: int
    mc_samples: int
    seed: int
    reference_point_utility: np.ndarray
    objective_contract_version: str
    method: str = "qlognehvi"
    method_version: str = "step2a-v1"


@dataclass(frozen=True)
class QLogNEHVIBatchProposal:
    """Sequentially selected qLogNEHVI batch and per-step score histories."""

    selection: Any
    score_history: tuple[QLogNEHVIPoolScoreResult, ...]
    preexisting_pending_count: int
    metadata: dict[str, Any]


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return result


def _input_matrix(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        shape = getattr(value, "shape", None)
        raise ValueError(f"{name} must be a tensor with shape (N, D); got {shape}.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating dtype.")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _reference(
    value: np.ndarray | torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        raise ValueError(
            "reference_point_utility is required and is never derived from data."
        )
    reference = torch.as_tensor(value, dtype=dtype, device=device)
    if reference.ndim != 1 or reference.numel() == 0:
        raise ValueError(
            "reference_point_utility must have shape (M,) with at least one objective."
        )
    if not torch.isfinite(reference).all():
        raise ValueError("reference_point_utility must contain only finite values.")
    return reference


def _build_qlognehvi(
    *,
    model: Any,
    train_X: torch.Tensor,
    reference_point: torch.Tensor,
    objective: Any,
    mc_samples: int,
    seed: int,
    X_pending: torch.Tensor | None,
    constraints: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None,
    eta: float | torch.Tensor,
    prune_baseline: bool,
) -> qLogNoisyExpectedHypervolumeImprovement:
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([mc_samples]), seed=int(seed)
    )
    return qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=reference_point,
        X_baseline=train_X,
        sampler=sampler,
        objective=objective,
        constraints=None if constraints is None else list(constraints),
        eta=eta,
        X_pending=X_pending,
        prune_baseline=prune_baseline,
    )


def score_qlognehvi_singletons(
    model: Any,
    train_X_norm: torch.Tensor,
    X_pool_norm: torch.Tensor,
    objective: Any,
    reference_point_utility: np.ndarray | torch.Tensor | None,
    *,
    mc_samples: int = 128,
    seed: int = 0,
    chunk_size: int = 512,
    X_pending_norm: torch.Tensor | None = None,
    constraints: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    eta: float | torch.Tensor = 0.001,
    prune_baseline: bool = False,
) -> QLogNEHVIPoolScoreResult:
    """Evaluate qLogNEHVI on a normalized pool with explicit ``N x 1 x D`` shape."""
    from .objectives import (
        BoundedMCMultiOutputObjective,
        ConfiguredMCMultiOutputObjective,
    )

    train_X = _input_matrix(train_X_norm, name="train_X_norm")
    pool = _input_matrix(X_pool_norm, name="X_pool_norm")
    if pool.shape[0] == 0:
        raise ValueError("X_pool_norm must contain at least one candidate.")
    if pool.shape[1] != train_X.shape[1]:
        raise ValueError("train_X_norm and X_pool_norm dimensions must match.")
    sample_count = _positive_integer(mc_samples, name="mc_samples")
    chunk = _positive_integer(chunk_size, name="chunk_size")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a non-negative integer.")
    approved_objective_types = (
        ConfiguredMCMultiOutputObjective,
        BoundedMCMultiOutputObjective,
    )
    if not isinstance(objective, approved_objective_types):
        raise TypeError(
            "objective must be a configured or bounded configured multi-output "
            "objective so raw outcomes cannot silently bypass the approved utility "
            "transform."
        )
    pending: torch.Tensor | None = None
    if X_pending_norm is not None:
        pending = _input_matrix(X_pending_norm, name="X_pending_norm")
        if pending.shape[1] != train_X.shape[1]:
            raise ValueError("X_pending_norm and train_X_norm dimensions must match.")
        pending = pending.to(dtype=train_X.dtype, device=train_X.device)
    reference = _reference(
        reference_point_utility, dtype=train_X.dtype, device=train_X.device
    )
    objective_count = objective.objective_transform.objective_count
    if reference.numel() != objective_count:
        raise ValueError(
            "reference_point_utility dimension must match the configured objective "
            f"count ({objective_count}); got {reference.numel()}."
        )
    model_output_count = getattr(model, "num_outputs", None)
    if model_output_count is not None and int(model_output_count) != objective_count:
        raise ValueError(
            f"Model has {int(model_output_count)} outputs but objective contract "
            f"has {objective_count}."
        )
    objective_version = getattr(
        objective, "version", objective.objective_transform.version
    )
    acquisition = _build_qlognehvi(
        model=model,
        train_X=train_X,
        reference_point=reference,
        objective=objective,
        mc_samples=sample_count,
        seed=int(seed),
        X_pending=pending,
        constraints=constraints,
        eta=eta,
        prune_baseline=bool(prune_baseline),
    )

    values: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, pool.shape[0], chunk):
            singleton_batch = (
                pool[start : start + chunk]
                .to(dtype=train_X.dtype, device=train_X.device)
                .unsqueeze(-2)
            )
            chunk_values = acquisition(singleton_batch)
            if chunk_values.shape != (singleton_batch.shape[0],):
                raise RuntimeError(
                    "qLogNEHVI singleton evaluation returned unexpected shape "
                    f"{tuple(chunk_values.shape)} for input "
                    f"{tuple(singleton_batch.shape)}."
                )
            values.append(chunk_values.detach().cpu().double())
    scores = torch.cat(values).numpy()
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise RuntimeError("qLogNEHVI returned NaN or positive-infinite scores.")
    return QLogNEHVIPoolScoreResult(
        base_log_score=scores,
        evaluated_shape=(int(pool.shape[0]), 1, int(pool.shape[1])),
        pending_count=0 if pending is None else int(pending.shape[0]),
        mc_samples=sample_count,
        seed=int(seed),
        reference_point_utility=reference.detach().cpu().double().numpy(),
        objective_contract_version=objective_version,
    )


def _assert_no_reference_overlap(
    pool_norm: np.ndarray,
    reference_norm: np.ndarray | None,
    *,
    name: str,
    atol: float = 1e-12,
) -> None:
    if reference_norm is None:
        return
    reference = np.asarray(reference_norm, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != pool_norm.shape[1]:
        raise ValueError(
            f"{name} must have shape (N, {pool_norm.shape[1]}); got {reference.shape}."
        )
    if not np.all(np.isfinite(reference)):
        raise ValueError(f"{name} must contain only finite values.")
    if reference.shape[0] == 0:
        return
    overlap = np.all(
        np.isclose(
            pool_norm[:, None, :],
            reference[None, :, :],
            rtol=0.0,
            atol=atol,
        ),
        axis=-1,
    )
    if np.any(overlap):
        pool_indices = np.flatnonzero(np.any(overlap, axis=1)).tolist()
        raise ValueError(
            f"Candidate pool overlaps {name} at pool indices {pool_indices}; "
            "resample with those recipes in the avoid set."
        )


def propose_qlognehvi_penalized_batch(
    candidate_pool: Any,
    model: Any,
    train_X_norm: torch.Tensor,
    objective: Any,
    reference_point_utility: np.ndarray | torch.Tensor | None,
    *,
    q: int,
    local_penalization_config: Any,
    X_pending_norm: torch.Tensor | None = None,
    mc_samples: int = 128,
    seed: int = 0,
    chunk_size: int = 512,
    constraints: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    eta: float | torch.Tensor = 0.001,
    prune_baseline: bool = False,
) -> QLogNEHVIBatchProposal:
    """Select a discrete batch, rebuilding qLogNEHVI pending state each step."""
    from .batch_selection import BaseScoreResult, select_local_penalized_batch

    train_X = _input_matrix(train_X_norm, name="train_X_norm")
    pool_norm = np.asarray(candidate_pool.X_norm, dtype=float)
    if pool_norm.ndim != 2 or pool_norm.shape[1] != train_X.shape[1]:
        raise ValueError(
            "candidate_pool.X_norm and train_X_norm must share input dimension."
        )
    train_numpy = train_X.detach().cpu().double().numpy()
    _assert_no_reference_overlap(pool_norm, train_numpy, name="observed train_X_norm")

    pending_tensor: torch.Tensor | None = None
    pending_numpy: np.ndarray | None = None
    if X_pending_norm is not None:
        pending_tensor = _input_matrix(X_pending_norm, name="X_pending_norm").to(
            dtype=train_X.dtype, device=train_X.device
        )
        pending_numpy = pending_tensor.detach().cpu().double().numpy()
        _assert_no_reference_overlap(
            pool_norm, pending_numpy, name="pre-existing X_pending_norm"
        )

    histories: list[QLogNEHVIPoolScoreResult] = []

    def score_remaining(
        remaining_indices: np.ndarray, selected_indices: np.ndarray
    ) -> Any:
        selected_tensor = torch.as_tensor(
            pool_norm[selected_indices], dtype=train_X.dtype, device=train_X.device
        )
        if pending_tensor is None:
            current_pending = selected_tensor if selected_indices.size else None
        elif selected_indices.size:
            current_pending = torch.cat([pending_tensor, selected_tensor], dim=0)
        else:
            current_pending = pending_tensor
        result = score_qlognehvi_singletons(
            model,
            train_X,
            torch.as_tensor(
                pool_norm[remaining_indices],
                dtype=train_X.dtype,
                device=train_X.device,
            ),
            objective,
            reference_point_utility,
            mc_samples=mc_samples,
            seed=seed,
            chunk_size=chunk_size,
            X_pending_norm=current_pending,
            constraints=constraints,
            eta=eta,
            prune_baseline=prune_baseline,
        )
        histories.append(result)
        raw_base_score = np.exp(np.clip(result.base_log_score, -745.0, 709.0))
        raw_base_score[np.isneginf(result.base_log_score)] = 0.0
        return BaseScoreResult(
            base_log_score=result.base_log_score,
            base_score=raw_base_score,
            diagnostics={
                "pending_count": result.pending_count,
                "evaluated_shape": result.evaluated_shape,
                "remaining_pool_indices": remaining_indices.copy(),
            },
        )

    observed_pending = (
        train_numpy
        if pending_numpy is None
        else np.vstack([train_numpy, pending_numpy])
    )
    selection = select_local_penalized_batch(
        candidate_pool,
        q,
        score_remaining,
        local_penalization_config,
        observed_pending_norm=observed_pending,
    )
    return QLogNEHVIBatchProposal(
        selection=selection,
        score_history=tuple(histories),
        preexisting_pending_count=(
            0 if pending_tensor is None else int(pending_tensor.shape[0])
        ),
        metadata={
            "method": "qlognehvi",
            "method_version": "step2a-v1",
            "objective_contract_version": getattr(
                objective, "version", objective.objective_transform.version
            ),
            "reference_point_utility": np.asarray(
                reference_point_utility, dtype=float
            ).copy(),
            "pool_seed": candidate_pool.seed,
            "pool_size": candidate_pool.size,
            "pool_draws": candidate_pool.draws,
            "pool_rejected_duplicate": candidate_pool.rejected_duplicate,
            "pool_rejected_avoid": candidate_pool.rejected_avoid,
            "pool_rejected_constraint": candidate_pool.rejected_constraint,
            "mc_seed": int(seed),
            "mc_samples": int(mc_samples),
            "preexisting_pending_count": (
                0 if pending_tensor is None else int(pending_tensor.shape[0])
            ),
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
