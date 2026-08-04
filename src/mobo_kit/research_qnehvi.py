"""qNEHVI as a research-only R2 variant, for comparison against qLogNEHVI.

**This is not part of the campaign.** ``campaign.py`` proposes R2 with qLogNEHVI
and nothing here changes that; the frozen acquisition modules
(``qlognehvi_batch.py``, ``ucb_hvi.py``, ``batch_selection.py``) are untouched.
This module exists so the two acquisitions can be compared on identical inputs,
which is a question about the tooling rather than about the chemistry.

**BoTorch itself recommends against qNEHVI.** Constructing one emits a
``NumericsWarning`` saying it "has known numerical issues that lead to suboptimal
optimization performance" and to use qLogNEHVI instead (arXiv:2310.20708). That
warning is deliberately not silenced here -- if this module is used, the caller
should see it.

Measured on this campaign at the default cell (radius 0.25, beta 4.0, seed 73),
the two acquisitions propose the **identical batch**. That is the useful finding,
and it is what makes the comparison worth having run once rather than repeatedly.

Derived from the structure of ``qlognehvi_batch.propose_qlognehvi_penalized_batch``
and from the ``qnehvi_batch`` module on Annie Xu's ``ax_plots_simulation`` branch,
which is where the idea of carrying a second acquisition came from. Two things are
reviewed against current conventions rather than copied: the observed baseline
reaches the objective transform in MODEL space, and the hypervolume reference is
explicit rather than inferred.

THE ONE REAL DIFFERENCE, and why it is handled the way it is. qLogNEHVI returns
the *logarithm* of the expected improvement; qNEHVI returns the improvement
itself. ``select_local_penalized_batch`` applies its soft penalty in log space, so
a raw qNEHVI value must be logged before it enters the selector or the two
variants would be penalised on different scales and would not be comparable. That
conversion is exactly the numerically fragile step qLogNEHVI exists to avoid: for
a small Monte-Carlo estimate, ``log(mean(...))`` loses precision where qLogNEHVI
computes the log directly.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler

from .batch_selection import LocalPenalizationConfig
from .campaign import (
    RoundResult,
    build_objective_transform,
    expand_replicates,
    validate_batch,
)
from .candidate_pool import sample_discrete_candidate_pool
from .constraints import constraints_from_config
from .design import build_design_from_config

__all__ = [
    "propose_qnehvi_penalized_batch",
    "run_r2_qnehvi_research",
    "R2_ACQUISITIONS",
]

#: Selectable R2 acquisitions. ``qlognehvi`` is the campaign's own.
R2_ACQUISITIONS = ("qlognehvi", "qnehvi")

#: Below this the log of a Monte-Carlo improvement estimate is meaningless.
_LOG_FLOOR = 1e-12


def _score_qnehvi_singletons(
    model: Any,
    train_X: torch.Tensor,
    pool_norm: np.ndarray,
    objective: Any,
    reference: torch.Tensor,
    *,
    mc_samples: int,
    seed: int,
    chunk_size: int,
    pending: torch.Tensor | None,
    constraints: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None,
    eta: float | torch.Tensor,
    prune_baseline: bool,
) -> np.ndarray:
    """Evaluate qNEHVI on each pool row as its own ``1 x D`` batch."""
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([int(mc_samples)]), seed=int(seed)
    )
    acquisition = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=reference,
        X_baseline=train_X,
        sampler=sampler,
        objective=objective,
        constraints=None if constraints is None else list(constraints),
        eta=eta,
        X_pending=pending,
        prune_baseline=bool(prune_baseline),
    )
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, pool_norm.shape[0], chunk_size):
            batch = torch.as_tensor(
                pool_norm[start : start + chunk_size],
                dtype=train_X.dtype,
                device=train_X.device,
            ).unsqueeze(-2)
            chunk_values = acquisition(batch)
            if chunk_values.shape != (batch.shape[0],):
                raise RuntimeError(
                    "qNEHVI singleton evaluation returned unexpected shape "
                    f"{tuple(chunk_values.shape)} for input {tuple(batch.shape)}."
                )
            values.append(chunk_values.detach().cpu().double())
    scores = torch.cat(values).numpy()
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise RuntimeError("qNEHVI returned NaN or positive-infinite scores.")
    return scores


def propose_qnehvi_penalized_batch(
    candidate_pool: Any,
    model: Any,
    train_X_norm: torch.Tensor,
    objective: Any,
    reference_point_utility: np.ndarray | torch.Tensor,
    *,
    q: int,
    local_penalization_config: LocalPenalizationConfig,
    mc_samples: int = 128,
    seed: int = 0,
    chunk_size: int = 512,
    constraints: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    eta: float | torch.Tensor = 0.001,
    prune_baseline: bool = False,
) -> Any:
    """Select ``q`` locally penalised pool candidates by qNEHVI.

    Mirrors ``propose_qlognehvi_penalized_batch`` step for step -- same pool, same
    selector, same pending-state rebuild after each pick -- so any difference in
    the batch is attributable to the acquisition and nothing else.
    """
    from .batch_selection import BaseScoreResult, select_local_penalized_batch

    train_X = torch.as_tensor(train_X_norm, dtype=torch.double)
    if train_X.ndim != 2:
        raise ValueError("train_X_norm must be a 2-D (N, D) tensor.")
    pool_norm = np.asarray(candidate_pool.X_norm, dtype=float)
    if pool_norm.ndim != 2 or pool_norm.shape[1] != train_X.shape[1]:
        raise ValueError(
            "candidate_pool.X_norm and train_X_norm must share input dimension."
        )
    reference = torch.as_tensor(
        np.asarray(reference_point_utility, dtype=float), dtype=torch.double
    )

    def score_remaining(
        remaining_indices: np.ndarray, selected_indices: np.ndarray
    ) -> Any:
        pending = (
            torch.as_tensor(pool_norm[selected_indices], dtype=torch.double)
            if selected_indices.size
            else None
        )
        raw = _score_qnehvi_singletons(
            model,
            train_X,
            pool_norm[remaining_indices],
            objective,
            reference,
            mc_samples=mc_samples,
            seed=seed,
            chunk_size=chunk_size,
            pending=pending,
            constraints=constraints,
            eta=eta,
            prune_baseline=prune_baseline,
        )
        # qNEHVI returns the improvement; the selector penalises in log space.
        base_score = np.clip(raw, 0.0, None)
        with np.errstate(divide="ignore"):
            base_log_score = np.where(
                base_score > _LOG_FLOOR, np.log(np.maximum(base_score, _LOG_FLOOR)),
                -np.inf,
            )
        return BaseScoreResult(
            base_log_score=base_log_score,
            base_score=base_score,
            diagnostics={"remaining_pool_indices": remaining_indices.copy()},
        )

    return select_local_penalized_batch(
        candidate_pool,
        q,
        score_remaining,
        local_penalization_config,
        observed_pending_norm=train_X.detach().cpu().double().numpy(),
    )


def run_r2_qnehvi_research(
    config: Any,
    observed_X_phys: np.ndarray,
    observed_Y_raw: np.ndarray,
    *,
    n: int | None = None,
    seed: int | None = None,
) -> RoundResult:
    """An R2 round proposed by qNEHVI, for comparison only.

    Same contract as ``campaign.run_r2_qlognehvi``: ``observed_Y_raw`` holds the
    MODEL SOURCE values in objective order, so thickness arrives in nanometres.
    """
    from .campaign import _fit_models, _normalise, _penalization, _reference_point
    from .campaign import _on_grid_mask, _round_settings
    from .objectives import ConfiguredMCMultiOutputObjective

    design = build_design_from_config(dict(config))
    settings = _round_settings(config, "r2")
    q = int(settings["batch_size"]) if n is None else int(n)
    resolved_seed = (
        int(config.get("reproducibility", {}).get("seed", 0)) if seed is None else seed
    )
    transform = build_objective_transform(config)
    reference = _reference_point(config, transform.objective_count)
    penalization = _penalization(config)

    observed_norm = _normalise(design, observed_X_phys)
    model, fit_warnings, raw_fit_warnings = _fit_models(
        config,
        np.asarray(observed_X_phys, dtype=float),
        observed_norm,
        observed_Y_raw,
        resolved_seed,
    )

    on_grid = _on_grid_mask(design, observed_X_phys)
    pool = sample_discrete_candidate_pool(
        design,
        int(settings.get("candidate_pool_size", 32768)),
        seed=resolved_seed,
        observed_phys=np.asarray(observed_X_phys, dtype=float)[on_grid],
        row_constraints=constraints_from_config(dict(config), design) or None,
    )

    selection = propose_qnehvi_penalized_batch(
        pool,
        model,
        torch.tensor(observed_norm, dtype=torch.double),
        ConfiguredMCMultiOutputObjective(transform),
        reference,
        q=q,
        local_penalization_config=penalization,
        mc_samples=int(settings.get("mc_samples", 128)),
        seed=resolved_seed,
    )

    conditions = pd.DataFrame(
        np.asarray(selection.X_phys, dtype=float), columns=list(design.names)
    )
    report = validate_batch(
        conditions,
        design,
        expected_count=q,
        min_pairwise_distance=penalization.min_batch_distance,
    )
    replicates_per = int(settings.get("replicates_per_condition", 1))
    return RoundResult(
        round_name="R2",
        conditions=conditions,
        replicates=expand_replicates(
            conditions, replicates=replicates_per, round_name="R2"
        ),
        diagnostics={
            "method": "qnehvi",
            "research_only": True,
            "seed": resolved_seed,
            "pool_size": pool.size,
            "objective_contract": transform.version,
            "model_fit_warnings": list(fit_warnings),
            "fit_warnings_raw": list(raw_fit_warnings),
            "validity": report,
        },
    )
