"""Read-only, debug-only D2D campaign adapter for the completed R0 workbook."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any

import botorch
import gpytorch
import numpy as np
import pandas as pd
import torch

from .batch_selection import LocalPenalizationConfig
from .candidate_diagnostics import (
    plot_candidate_pca,
    plot_distance_heatmap,
    plot_parallel_coordinates,
    plot_selection_scores,
    summarize_candidate_batch,
)
from .candidate_pool import CandidatePool, sample_discrete_candidate_pool
from .d2d_campaign import (
    D2D_DEBUG_WATERMARK,
    D2D_INPUT_COLUMNS,
    D2D_OBJECTIVE_COLUMNS,
    D2D_REFERENCE_POINT_UTILITY,
    D2DTrainingData,
    ResolvedD2DDebugConfig,
    build_d2d_objective_transform,
    expand_candidates_to_replicates,
    load_d2d_debug_config,
    load_d2d_workbook_frame,
    prepare_d2d_training_data,
    sha256_file,
)
from .d2d_scores import validate_supplied_d2d_scores
from .models import fit_gp_models, posterior_report
from .ucb_hvi import UCBHVIBatchProposal, propose_ucb_hvi_batch


@dataclass(frozen=True)
class ProposalComputation:
    candidate_pool: CandidatePool
    proposal: UCBHVIBatchProposal
    model: Any
    train_X: torch.Tensor
    train_Y: torch.Tensor
    runtime_seconds: float


@dataclass(frozen=True)
class D2DDebugRunResult:
    output_dir: Path
    candidates_unique: pd.DataFrame
    replicate_worklist: pd.DataFrame
    sensitivity_summary: pd.DataFrame
    sensitivity_candidates: pd.DataFrame
    control_ablation: pd.DataFrame
    model_diagnostics: pd.DataFrame
    run_manifest: dict[str, Any]


@dataclass(frozen=True)
class SensitivityArtifacts:
    summary: pd.DataFrame
    candidates: pd.DataFrame
    control_ablation: pd.DataFrame


def _fit_debug_model(
    training: D2DTrainingData,
    *,
    seed: int,
    include_mask: np.ndarray | None = None,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    mask = (
        training.include_in_model if include_mask is None else np.asarray(include_mask)
    )
    if mask.shape != training.include_in_model.shape or mask.dtype != bool:
        raise ValueError(
            "include_mask must be a boolean vector aligned with training rows."
        )
    if np.count_nonzero(mask) < 2:
        raise ValueError("At least two condition-level observations are required.")
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    X = torch.as_tensor(training.X_norm_all[mask], dtype=torch.double, device="cpu")
    Y = torch.as_tensor(training.Y_objectives[mask], dtype=torch.double, device="cpu")
    model = fit_gp_models(X, Y)
    return model, X, Y


def _compute_proposal(
    config: ResolvedD2DDebugConfig,
    training: D2DTrainingData,
    model: Any,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    *,
    pool_size: int,
    pool_seed: int,
    mc_seed: int,
    beta: float,
    posterior_samples: int,
    radius: float,
    min_batch_distance: float,
) -> ProposalComputation:
    started = perf_counter()
    pool = sample_discrete_candidate_pool(
        config.design,
        pool_size,
        seed=pool_seed,
        observed_phys=training.X_phys_all[training.on_grid_mask],
        row_constraints=[],
    )
    local = LocalPenalizationConfig(
        radius=radius,
        min_batch_distance=min_batch_distance,
        min_observed_distance=config.min_observed_distance,
        dimension_weights=config.dimension_weights,
    )
    proposal = propose_ucb_hvi_batch(
        pool,
        model,
        train_Y,
        build_d2d_objective_transform(),
        config.reference_point_utility,
        q=config.r1_batch_size,
        beta=beta,
        local_penalization_config=local,
        observed_pending_norm=training.X_norm_all,
        positive_score_tolerance=1e-12,
        mc_samples=posterior_samples,
        seed=mc_seed,
        posterior_chunk_size=config.score_chunk_size,
        hvi_chunk_size=1024,
        observation_noise=False,
    )
    if proposal.selection.X_phys.shape != (5, len(D2D_INPUT_COLUMNS)):
        raise RuntimeError("The debug adapter did not produce exactly five conditions.")
    return ProposalComputation(
        candidate_pool=pool,
        proposal=proposal,
        model=model,
        train_X=train_X,
        train_Y=train_Y,
        runtime_seconds=perf_counter() - started,
    )


def _candidate_key_set(X_phys: np.ndarray) -> set[tuple[float, ...]]:
    return {tuple(float(value) for value in row) for row in np.asarray(X_phys)}


def _objective_prefix(objective: str) -> str:
    return objective.lower().replace(" ", "_")


def _ordered_batch_sha256(X_phys: np.ndarray) -> str:
    ordered = np.ascontiguousarray(np.asarray(X_phys, dtype=np.float64))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def _proposal_settings(
    config: ResolvedD2DDebugConfig,
    *,
    pool_size: int,
    pool_seed: int,
    mc_seed: int,
    beta: float,
    posterior_samples: int,
    radius: float,
    min_batch_distance: float,
    include_control: bool,
) -> dict[str, Any]:
    return {
        "candidate_pool_size": int(pool_size),
        "pool_seed": int(pool_seed),
        "mc_seed": int(mc_seed),
        "model_seed": int(config.seed),
        "beta": float(beta),
        "posterior_samples": int(posterior_samples),
        "local_radius": float(radius),
        "min_batch_distance": float(min_batch_distance),
        "min_observed_distance": float(config.min_observed_distance),
        "include_control": bool(include_control),
    }


def _observed_pareto_sample_ids(
    train_Y: torch.Tensor,
    sample_ids: np.ndarray,
) -> list[int | float | str]:
    values = train_Y.detach().cpu().numpy()
    ids = np.asarray(sample_ids)
    if values.shape[0] != ids.shape[0]:
        raise ValueError("sample_ids must align with train_Y rows.")
    nondominated = np.ones(values.shape[0], dtype=bool)
    for index, row in enumerate(values):
        dominates = np.all(values >= row, axis=1) & np.any(values > row, axis=1)
        nondominated[index] = not bool(np.any(dominates))
    result: list[int | float | str] = []
    for value in ids[nondominated]:
        item = value.item() if isinstance(value, np.generic) else value
        if isinstance(item, float) and item.is_integer():
            item = int(item)
        result.append(item)
    return result


def _pareto_summary_fields(
    observed_ids: list[int | float | str],
    baseline_ids: list[int | float | str],
) -> dict[str, Any]:
    observed_set = {str(value) for value in observed_ids}
    baseline_set = {str(value) for value in baseline_ids}
    overlap = len(observed_set & baseline_set)
    union = len(observed_set | baseline_set)
    return {
        "observed_pareto_count": len(observed_ids),
        "observed_pareto_sample_ids": json.dumps(observed_ids),
        "baseline_observed_pareto_count": len(baseline_ids),
        "baseline_observed_pareto_sample_ids": json.dumps(baseline_ids),
        "observed_pareto_overlap_count": overlap,
        "observed_pareto_jaccard_with_baseline": (
            float(overlap / union) if union else 1.0
        ),
    }


def _batch_summary(
    label: str,
    computation: ProposalComputation,
    baseline_phys: np.ndarray,
    baseline_norm: np.ndarray,
    *,
    parameter: str,
    value: Any,
    settings: dict[str, Any],
    fit_runtime_seconds: float,
    observed_pareto_ids: list[int | float | str],
    baseline_pareto_ids: list[int | float | str],
) -> dict[str, Any]:
    selected = computation.proposal.selection.X_phys
    selected_norm = computation.proposal.selection.X_norm
    baseline_keys = _candidate_key_set(baseline_phys)
    keys = _candidate_key_set(selected)
    intersection = len(keys & baseline_keys)
    union = len(keys | baseline_keys)
    differences = selected_norm[:, None, :] - baseline_norm[None, :, :]
    nearest = np.sqrt(np.sum(differences**2, axis=-1)).min(axis=1)
    distance = computation.proposal.selection.distance_diagnostics
    row = {
        "run_label": label,
        "parameter": parameter,
        "value": value,
        "status": "pass",
        "debug_only": True,
        "approved_for_experiment": False,
        "candidate_status": D2D_DEBUG_WATERMARK,
        "selected_count": int(selected.shape[0]),
        "exact_overlap_with_baseline": intersection,
        "jaccard_overlap_with_baseline": float(intersection / union),
        "mean_nearest_batch_distance_to_baseline": float(nearest.mean()),
        "minimum_within_batch_distance": distance["minimum_within_batch_distance"],
        "mean_within_batch_distance": distance["mean_within_batch_distance"],
        "maximum_within_batch_distance": distance["maximum_within_batch_distance"],
        "boundary_coordinate_count": int(
            np.count_nonzero(
                np.isclose(selected_norm, 0.0, atol=1e-12)
                | np.isclose(selected_norm, 1.0, atol=1e-12)
            )
        ),
        "ordered_batch_sha256": _ordered_batch_sha256(selected),
        **settings,
        "fit_runtime_seconds": float(fit_runtime_seconds),
        "proposal_runtime_seconds": float(computation.runtime_seconds),
        "total_fit_proposal_runtime_seconds": float(
            fit_runtime_seconds + computation.runtime_seconds
        ),
        "warning": "",
    }
    row.update(_pareto_summary_fields(observed_pareto_ids, baseline_pareto_ids))
    return row


def _sensitivity_candidate_rows(
    label: str,
    computation: ProposalComputation,
    config: ResolvedD2DDebugConfig,
    training: D2DTrainingData,
    *,
    settings: dict[str, Any],
    fit_runtime_seconds: float,
) -> list[dict[str, Any]]:
    selection = computation.proposal.selection
    predicted_mean, predicted_std = posterior_report(
        computation.model,
        torch.as_tensor(selection.X_norm, dtype=torch.double),
    )
    diagnostics = summarize_candidate_batch(
        selection.X_norm,
        observed_pending_norm=training.X_norm_all,
        X_phys=selection.X_phys,
        design=config.design,
        dimension_weights=config.dimension_weights,
        metadata={"debug_only": True, "run_label": label},
    )
    pairwise = diagnostics.pairwise_distance_matrix
    pairwise_nearest = pairwise.copy()
    np.fill_diagonal(pairwise_nearest, np.inf)
    ordered_hash = _ordered_batch_sha256(selection.X_phys)
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(selection.steps):
        boundary_names = [
            name
            for name, is_boundary in zip(
                D2D_INPUT_COLUMNS, diagnostics.boundary_flags[index]
            )
            if is_boundary
        ]
        row: dict[str, Any] = {
            "record_type": "selected_candidate",
            "run_label": label,
            "candidate_id": f"{label}-C{index + 1:02d}",
            "selection_order": index + 1,
            "selected_pool_index": int(step.pool_index),
            "status": "pass",
            "warning": "",
            "debug_only": True,
            "approved_for_experiment": False,
            "candidate_status": D2D_DEBUG_WATERMARK,
        }
        for column, candidate_value in zip(D2D_INPUT_COLUMNS, selection.X_phys[index]):
            row[column] = float(candidate_value)
        for objective_index, objective in enumerate(D2D_OBJECTIVE_COLUMNS):
            prefix = _objective_prefix(objective)
            row[f"predicted_{prefix}_mean"] = float(
                predicted_mean[index, objective_index]
            )
            row[f"predicted_{prefix}_std"] = float(
                predicted_std[index, objective_index]
            )
        final_penalized_score = (
            np.nan
            if step.base_score is None
            else float(step.base_score * step.penalty_factor)
        )
        row.update(
            {
                "base_raw_score": step.base_score,
                "base_log_score": step.base_log_score,
                "penalty_factor": step.penalty_factor,
                "log_penalty": step.log_penalty,
                "penalized_log_score": step.penalized_log_score,
                "final_penalized_score": final_penalized_score,
                "pairwise_distance_row": json.dumps(
                    [float(distance) for distance in pairwise[index]]
                ),
                "nearest_selected_distance": float(pairwise_nearest[index].min()),
                "nearest_observed_distance": float(
                    diagnostics.nearest_observed_pending_distance[index]
                ),
                "boundary_coordinates": json.dumps(boundary_names),
                "boundary_coordinate_count": len(boundary_names),
                "grid_valid": bool(diagnostics.grid_valid_rows[index]),
                "bounds_valid": bool(
                    np.all(selection.X_norm[index] >= 0.0)
                    and np.all(selection.X_norm[index] <= 1.0)
                ),
                "ordered_batch_sha256": ordered_hash,
                **settings,
                "candidate_pool_accepted": int(computation.candidate_pool.size),
                "candidate_pool_draws": int(computation.candidate_pool.draws),
                "candidate_pool_duplicate_rejections": int(
                    computation.candidate_pool.rejected_duplicate
                ),
                "candidate_pool_avoid_rejections": int(
                    computation.candidate_pool.rejected_avoid
                ),
                "candidate_pool_constraint_rejections": int(
                    computation.candidate_pool.rejected_constraint
                ),
                "fit_runtime_seconds": float(fit_runtime_seconds),
                "proposal_runtime_seconds": float(computation.runtime_seconds),
                "total_fit_proposal_runtime_seconds": float(
                    fit_runtime_seconds + computation.runtime_seconds
                ),
            }
        )
        rows.append(row)
    return rows


def _failed_summary_row(
    label: str,
    *,
    parameter: str,
    value: Any,
    settings: dict[str, Any],
    fit_runtime_seconds: float,
    proposal_runtime_seconds: float,
    warning: str,
    observed_pareto_ids: list[int | float | str],
    baseline_pareto_ids: list[int | float | str],
) -> dict[str, Any]:
    row = {
        "run_label": label,
        "parameter": parameter,
        "value": value,
        "status": "failed",
        "debug_only": True,
        "approved_for_experiment": False,
        "candidate_status": D2D_DEBUG_WATERMARK,
        "selected_count": 0,
        "exact_overlap_with_baseline": 0,
        "jaccard_overlap_with_baseline": 0.0,
        "mean_nearest_batch_distance_to_baseline": np.nan,
        "minimum_within_batch_distance": np.nan,
        "mean_within_batch_distance": np.nan,
        "maximum_within_batch_distance": np.nan,
        "boundary_coordinate_count": np.nan,
        "ordered_batch_sha256": "",
        **settings,
        "fit_runtime_seconds": float(fit_runtime_seconds),
        "proposal_runtime_seconds": float(proposal_runtime_seconds),
        "total_fit_proposal_runtime_seconds": float(
            fit_runtime_seconds + proposal_runtime_seconds
        ),
        "warning": warning,
    }
    row.update(_pareto_summary_fields(observed_pareto_ids, baseline_pareto_ids))
    return row


def _failed_sensitivity_candidate_row(
    label: str,
    *,
    settings: dict[str, Any],
    fit_runtime_seconds: float,
    proposal_runtime_seconds: float,
    warning: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_type": "failed_run_marker",
        "run_label": label,
        "candidate_id": "",
        "selection_order": np.nan,
        "selected_pool_index": np.nan,
        "status": "failed",
        "warning": warning,
        "debug_only": True,
        "approved_for_experiment": False,
        "candidate_status": D2D_DEBUG_WATERMARK,
        "base_raw_score": np.nan,
        "base_log_score": np.nan,
        "penalty_factor": np.nan,
        "log_penalty": np.nan,
        "penalized_log_score": np.nan,
        "final_penalized_score": np.nan,
        "pairwise_distance_row": "[]",
        "nearest_selected_distance": np.nan,
        "nearest_observed_distance": np.nan,
        "boundary_coordinates": "[]",
        "boundary_coordinate_count": np.nan,
        "grid_valid": False,
        "bounds_valid": False,
        "ordered_batch_sha256": "",
        **settings,
        "fit_runtime_seconds": float(fit_runtime_seconds),
        "proposal_runtime_seconds": float(proposal_runtime_seconds),
        "total_fit_proposal_runtime_seconds": float(
            fit_runtime_seconds + proposal_runtime_seconds
        ),
    }
    for column in D2D_INPUT_COLUMNS:
        row[column] = np.nan
    for objective in D2D_OBJECTIVE_COLUMNS:
        prefix = _objective_prefix(objective)
        row[f"predicted_{prefix}_mean"] = np.nan
        row[f"predicted_{prefix}_std"] = np.nan
    return row


def _control_ablation_columns() -> list[str]:
    columns = [
        "record_type",
        "candidate_id",
        "selection_order",
        *D2D_INPUT_COLUMNS,
    ]
    for objective in D2D_OBJECTIVE_COLUMNS:
        prefix = _objective_prefix(objective)
        columns.extend(
            [
                f"baseline_predicted_{prefix}_mean",
                f"baseline_predicted_{prefix}_std",
                f"control_excluded_predicted_{prefix}_mean",
                f"control_excluded_predicted_{prefix}_std",
                f"delta_{prefix}_mean",
                f"delta_{prefix}_std",
            ]
        )
    columns.extend(
        [
            "selected_exact_overlap_count",
            "selected_jaccard_overlap",
            "selected_mean_nearest_distance_to_baseline",
            "control_excluded_ordered_batch_sha256",
            "baseline_observed_pareto_count",
            "baseline_observed_pareto_sample_ids",
            "control_excluded_observed_pareto_count",
            "control_excluded_observed_pareto_sample_ids",
            "observed_pareto_overlap_count",
            "observed_pareto_jaccard_with_baseline",
            "fit_runtime_seconds",
            "proposal_runtime_seconds",
            "total_fit_proposal_runtime_seconds",
            "status",
            "warning",
            "debug_only",
            "approved_for_experiment",
            "candidate_status",
        ]
    )
    return columns


def _empty_control_ablation() -> pd.DataFrame:
    return pd.DataFrame(columns=_control_ablation_columns())


def _stamp_uniformity_mismatch(
    artifacts: SensitivityArtifacts,
    known_uniformity_score_mismatch: bool,
) -> SensitivityArtifacts:
    frames = [
        artifacts.summary.copy(),
        artifacts.candidates.copy(),
        artifacts.control_ablation.copy(),
    ]
    for frame in frames:
        frame["known_uniformity_score_mismatch"] = bool(known_uniformity_score_mismatch)
    return SensitivityArtifacts(*frames)


def _control_ablation_rows(
    baseline: ProposalComputation,
    ablation: ProposalComputation,
    baseline_summary: dict[str, Any],
    *,
    baseline_pareto_ids: list[int | float | str],
    ablation_pareto_ids: list[int | float | str],
    fit_runtime_seconds: float,
) -> list[dict[str, Any]]:
    baseline_phys = baseline.proposal.selection.X_phys
    baseline_norm = baseline.proposal.selection.X_norm
    baseline_mean, baseline_std = posterior_report(
        baseline.model, torch.as_tensor(baseline_norm, dtype=torch.double)
    )
    ablation_mean, ablation_std = posterior_report(
        ablation.model, torch.as_tensor(baseline_norm, dtype=torch.double)
    )
    pareto_fields = _pareto_summary_fields(ablation_pareto_ids, baseline_pareto_ids)
    rows: list[dict[str, Any]] = []
    for index in range(baseline_phys.shape[0]):
        row: dict[str, Any] = {
            "record_type": "baseline_candidate_prediction_comparison",
            "candidate_id": f"R1-C{index + 1:02d}",
            "selection_order": index + 1,
        }
        for column, candidate_value in zip(D2D_INPUT_COLUMNS, baseline_phys[index]):
            row[column] = float(candidate_value)
        for objective_index, objective in enumerate(D2D_OBJECTIVE_COLUMNS):
            prefix = _objective_prefix(objective)
            row.update(
                {
                    f"baseline_predicted_{prefix}_mean": float(
                        baseline_mean[index, objective_index]
                    ),
                    f"baseline_predicted_{prefix}_std": float(
                        baseline_std[index, objective_index]
                    ),
                    f"control_excluded_predicted_{prefix}_mean": float(
                        ablation_mean[index, objective_index]
                    ),
                    f"control_excluded_predicted_{prefix}_std": float(
                        ablation_std[index, objective_index]
                    ),
                    f"delta_{prefix}_mean": float(
                        ablation_mean[index, objective_index]
                        - baseline_mean[index, objective_index]
                    ),
                    f"delta_{prefix}_std": float(
                        ablation_std[index, objective_index]
                        - baseline_std[index, objective_index]
                    ),
                }
            )
        row.update(
            {
                "selected_exact_overlap_count": baseline_summary[
                    "exact_overlap_with_baseline"
                ],
                "selected_jaccard_overlap": baseline_summary[
                    "jaccard_overlap_with_baseline"
                ],
                "selected_mean_nearest_distance_to_baseline": baseline_summary[
                    "mean_nearest_batch_distance_to_baseline"
                ],
                "control_excluded_ordered_batch_sha256": baseline_summary[
                    "ordered_batch_sha256"
                ],
                "baseline_observed_pareto_count": len(baseline_pareto_ids),
                "baseline_observed_pareto_sample_ids": json.dumps(baseline_pareto_ids),
                "control_excluded_observed_pareto_count": len(ablation_pareto_ids),
                "control_excluded_observed_pareto_sample_ids": json.dumps(
                    ablation_pareto_ids
                ),
                "observed_pareto_overlap_count": pareto_fields[
                    "observed_pareto_overlap_count"
                ],
                "observed_pareto_jaccard_with_baseline": pareto_fields[
                    "observed_pareto_jaccard_with_baseline"
                ],
                "fit_runtime_seconds": float(fit_runtime_seconds),
                "proposal_runtime_seconds": float(ablation.runtime_seconds),
                "total_fit_proposal_runtime_seconds": float(
                    fit_runtime_seconds + ablation.runtime_seconds
                ),
                "status": "pass",
                "warning": "",
                "debug_only": True,
                "approved_for_experiment": False,
                "candidate_status": D2D_DEBUG_WATERMARK,
            }
        )
        rows.append(row)
    return rows


def _failed_control_ablation_row(
    *,
    fit_runtime_seconds: float,
    proposal_runtime_seconds: float,
    warning: str,
) -> dict[str, Any]:
    row = {column: np.nan for column in _control_ablation_columns()}
    row.update(
        {
            "record_type": "failed_run_marker",
            "candidate_id": "",
            "status": "failed",
            "warning": warning,
            "debug_only": True,
            "approved_for_experiment": False,
            "candidate_status": D2D_DEBUG_WATERMARK,
            "fit_runtime_seconds": float(fit_runtime_seconds),
            "proposal_runtime_seconds": float(proposal_runtime_seconds),
            "total_fit_proposal_runtime_seconds": float(
                fit_runtime_seconds + proposal_runtime_seconds
            ),
        }
    )
    return row


def _run_sensitivity(
    config: ResolvedD2DDebugConfig,
    training: D2DTrainingData,
    baseline: ProposalComputation,
    *,
    baseline_fit_runtime_seconds: float,
    known_uniformity_score_mismatch: bool,
) -> SensitivityArtifacts:
    baseline_phys = baseline.proposal.selection.X_phys
    baseline_norm = baseline.proposal.selection.X_norm
    baseline_pareto_ids = _observed_pareto_sample_ids(
        baseline.train_Y,
        training.sample_ids[training.include_in_model],
    )
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    baseline_settings = _proposal_settings(
        config,
        pool_size=config.candidate_pool_size,
        pool_seed=config.seed,
        mc_seed=config.seed,
        beta=config.beta,
        posterior_samples=config.posterior_samples,
        radius=config.local_radius,
        min_batch_distance=config.min_batch_distance,
        include_control=True,
    )
    baseline_row = _batch_summary(
        "baseline",
        baseline,
        baseline_phys,
        baseline_norm,
        parameter="baseline",
        value="configured",
        settings=baseline_settings,
        fit_runtime_seconds=baseline_fit_runtime_seconds,
        observed_pareto_ids=baseline_pareto_ids,
        baseline_pareto_ids=baseline_pareto_ids,
    )
    rows.append(baseline_row)
    candidate_rows.extend(
        _sensitivity_candidate_rows(
            "baseline",
            baseline,
            config,
            training,
            settings=baseline_settings,
            fit_runtime_seconds=baseline_fit_runtime_seconds,
        )
    )

    settings = {
        "pool_size": config.candidate_pool_size,
        "pool_seed": config.seed,
        "mc_seed": config.seed,
        "beta": config.beta,
        "posterior_samples": config.posterior_samples,
        "radius": config.local_radius,
        "min_batch_distance": config.min_batch_distance,
    }
    variations = [
        ("beta_1", "beta", 1.0),
        ("beta_9", "beta", 9.0),
        ("pool_5000", "pool_size", 5000),
        ("pool_20000", "pool_size", 20000),
        ("pool_seed_137", "pool_seed", 137),
        ("pool_seed_911", "pool_seed", 911),
        ("radius_0.15", "radius", 0.15),
        ("radius_0.35", "radius", 0.35),
        ("min_batch_0.10", "min_batch_distance", 0.10),
        ("min_batch_0.20", "min_batch_distance", 0.20),
        ("posterior_128", "posterior_samples", 128),
    ]
    for label, parameter, value in variations:
        current = dict(settings)
        current[parameter] = value
        actual_settings = _proposal_settings(
            config,
            pool_size=int(current["pool_size"]),
            pool_seed=int(current["pool_seed"]),
            mc_seed=int(current["mc_seed"]),
            beta=float(current["beta"]),
            posterior_samples=int(current["posterior_samples"]),
            radius=float(current["radius"]),
            min_batch_distance=float(current["min_batch_distance"]),
            include_control=True,
        )
        variation_started = perf_counter()
        try:
            computation = _compute_proposal(
                config,
                training,
                baseline.model,
                baseline.train_X,
                baseline.train_Y,
                pool_size=int(current["pool_size"]),
                pool_seed=int(current["pool_seed"]),
                mc_seed=int(current["mc_seed"]),
                beta=float(current["beta"]),
                posterior_samples=int(current["posterior_samples"]),
                radius=float(current["radius"]),
                min_batch_distance=float(current["min_batch_distance"]),
            )
            row = _batch_summary(
                label,
                computation,
                baseline_phys,
                baseline_norm,
                parameter=parameter,
                value=value,
                settings=actual_settings,
                fit_runtime_seconds=0.0,
                observed_pareto_ids=baseline_pareto_ids,
                baseline_pareto_ids=baseline_pareto_ids,
            )
            candidate_rows.extend(
                _sensitivity_candidate_rows(
                    label,
                    computation,
                    config,
                    training,
                    settings=actual_settings,
                    fit_runtime_seconds=0.0,
                )
            )
        except Exception as exc:  # sensitivity failures are reported, never hidden
            warning = f"{type(exc).__name__}: {exc}"
            failed_runtime = perf_counter() - variation_started
            row = _failed_summary_row(
                label,
                parameter=parameter,
                value=value,
                settings=actual_settings,
                fit_runtime_seconds=0.0,
                proposal_runtime_seconds=failed_runtime,
                warning=warning,
                observed_pareto_ids=baseline_pareto_ids,
                baseline_pareto_ids=baseline_pareto_ids,
            )
            candidate_rows.append(
                _failed_sensitivity_candidate_row(
                    label,
                    settings=actual_settings,
                    fit_runtime_seconds=0.0,
                    proposal_runtime_seconds=failed_runtime,
                    warning=warning,
                )
            )
        rows.append(row)

    ablation_mask = training.include_in_model.copy()
    ablation_mask[np.isin(training.sample_ids, config.control_sample_ids)] = False
    ablation_settings = _proposal_settings(
        config,
        pool_size=config.candidate_pool_size,
        pool_seed=config.seed,
        mc_seed=config.seed,
        beta=config.beta,
        posterior_samples=config.posterior_samples,
        radius=config.local_radius,
        min_batch_distance=config.min_batch_distance,
        include_control=False,
    )
    ablation_started = perf_counter()
    ablation_fit_runtime = 0.0
    try:
        fit_started = perf_counter()
        model, train_X, train_Y = _fit_debug_model(
            training, seed=config.seed, include_mask=ablation_mask
        )
        ablation_fit_runtime = perf_counter() - fit_started
        ablation = _compute_proposal(
            config,
            training,
            model,
            train_X,
            train_Y,
            pool_size=config.candidate_pool_size,
            pool_seed=config.seed,
            mc_seed=config.seed,
            beta=config.beta,
            posterior_samples=config.posterior_samples,
            radius=config.local_radius,
            min_batch_distance=config.min_batch_distance,
        )
        row = _batch_summary(
            "control_excluded",
            ablation,
            baseline_phys,
            baseline_norm,
            parameter="include_control",
            value=False,
            settings=ablation_settings,
            fit_runtime_seconds=ablation_fit_runtime,
            observed_pareto_ids=_observed_pareto_sample_ids(
                train_Y, training.sample_ids[ablation_mask]
            ),
            baseline_pareto_ids=baseline_pareto_ids,
        )
        ablation_pareto_ids = _observed_pareto_sample_ids(
            train_Y, training.sample_ids[ablation_mask]
        )
        candidate_rows.extend(
            _sensitivity_candidate_rows(
                "control_excluded",
                ablation,
                config,
                training,
                settings=ablation_settings,
                fit_runtime_seconds=ablation_fit_runtime,
            )
        )
        control_rows.extend(
            _control_ablation_rows(
                baseline,
                ablation,
                row,
                baseline_pareto_ids=baseline_pareto_ids,
                ablation_pareto_ids=ablation_pareto_ids,
                fit_runtime_seconds=ablation_fit_runtime,
            )
        )
        mean_delta_columns = [
            f"delta_{_objective_prefix(objective)}_mean"
            for objective in D2D_OBJECTIVE_COLUMNS
        ]
        row["mean_abs_prediction_change_on_baseline_candidates"] = float(
            pd.DataFrame(control_rows)
            .loc[:, mean_delta_columns]
            .abs()
            .to_numpy()
            .mean()
        )
    except Exception as exc:
        warning = f"{type(exc).__name__}: {exc}"
        total_failure_runtime = perf_counter() - ablation_started
        proposal_failure_runtime = max(
            0.0, total_failure_runtime - ablation_fit_runtime
        )
        row = _failed_summary_row(
            "control_excluded",
            parameter="include_control",
            value=False,
            settings=ablation_settings,
            fit_runtime_seconds=ablation_fit_runtime,
            proposal_runtime_seconds=proposal_failure_runtime,
            warning=warning,
            observed_pareto_ids=[],
            baseline_pareto_ids=baseline_pareto_ids,
        )
        row["mean_abs_prediction_change_on_baseline_candidates"] = np.nan
        candidate_rows.append(
            _failed_sensitivity_candidate_row(
                "control_excluded",
                settings=ablation_settings,
                fit_runtime_seconds=ablation_fit_runtime,
                proposal_runtime_seconds=proposal_failure_runtime,
                warning=warning,
            )
        )
        control_rows.append(
            _failed_control_ablation_row(
                fit_runtime_seconds=ablation_fit_runtime,
                proposal_runtime_seconds=proposal_failure_runtime,
                warning=warning,
            )
        )
    rows.append(row)
    return _stamp_uniformity_mismatch(
        SensitivityArtifacts(
            summary=pd.DataFrame(rows),
            candidates=pd.DataFrame(candidate_rows),
            control_ablation=pd.DataFrame(control_rows).reindex(
                columns=_control_ablation_columns()
            ),
        ),
        known_uniformity_score_mismatch,
    )


def _model_diagnostics(
    model: Any,
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
) -> pd.DataFrame:
    predicted, uncertainty = posterior_report(model, train_X)
    observed = train_Y.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for index, objective in enumerate(D2D_OBJECTIVE_COLUMNS):
        residual = predicted[:, index] - observed[:, index]
        denominator = float(
            np.sum((observed[:, index] - observed[:, index].mean()) ** 2)
        )
        r_squared = (
            np.nan
            if denominator <= 0
            else 1.0 - float(np.sum(residual**2)) / denominator
        )
        rows.append(
            {
                "objective": objective,
                "training_count": int(observed.shape[0]),
                "r_squared_training_posterior": r_squared,
                "rmse_training_posterior": float(np.sqrt(np.mean(residual**2))),
                "mae_training_posterior": float(np.mean(np.abs(residual))),
                "mean_posterior_std": float(np.mean(uncertainty[:, index])),
                "maximum_abs_residual": float(np.max(np.abs(residual))),
            }
        )
    return pd.DataFrame(rows)


def _candidates_frame(
    computation: ProposalComputation,
    config: ResolvedD2DDebugConfig,
    training: D2DTrainingData,
    *,
    known_uniformity_score_mismatch: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = computation.proposal.selection
    selected_t = torch.as_tensor(selection.X_norm, dtype=torch.double)
    predicted_mean, predicted_std = posterior_report(computation.model, selected_t)
    diagnostics = summarize_candidate_batch(
        selection.X_norm,
        observed_pending_norm=training.X_norm_all,
        X_phys=selection.X_phys,
        design=config.design,
        dimension_weights=config.dimension_weights,
        metadata={"debug_only": True},
    )
    pairwise = diagnostics.pairwise_distance_matrix.copy()
    np.fill_diagonal(pairwise, np.inf)
    nearest_selected = pairwise.min(axis=1)
    rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for index, step in enumerate(selection.steps):
        candidate_id = f"R1-C{index + 1:02d}"
        row: dict[str, Any] = {
            "campaign_id": str(config.raw["campaign_id"]),
            "round": "R1",
            "sample_id": pd.NA,
            "candidate_id": candidate_id,
            "row_role": "candidate_condition",
            "replicate_group": candidate_id,
            "replicate_number": pd.NA,
            "selection_order": index + 1,
            "debug_only": True,
            "approved_for_experiment": False,
            "candidate_status": D2D_DEBUG_WATERMARK,
            "include_in_model": False,
            "measurement_provenance": "proposed_unmeasured",
            "off_grid_exception": False,
            "exclusion_reason": "awaiting_measurement",
            "known_uniformity_score_mismatch": bool(known_uniformity_score_mismatch),
        }
        for column, value in zip(D2D_INPUT_COLUMNS, selection.X_phys[index]):
            row[column] = float(value)
        for objective_index, objective in enumerate(D2D_OBJECTIVE_COLUMNS):
            prefix = _objective_prefix(objective)
            row[f"predicted_{prefix}_mean"] = float(
                predicted_mean[index, objective_index]
            )
            row[f"predicted_{prefix}_std"] = float(
                predicted_std[index, objective_index]
            )
        row.update(
            {
                "base_ucb_hvi": step.base_score,
                "base_log_ucb_hvi": step.base_log_score,
                "penalty_factor": step.penalty_factor,
                "penalized_log_score": step.penalized_log_score,
                "final_penalized_score": (
                    np.nan
                    if step.base_score is None
                    else float(step.base_score * step.penalty_factor)
                ),
                "nearest_selected_distance": float(nearest_selected[index]),
                "nearest_observed_distance": float(
                    diagnostics.nearest_observed_pending_distance[index]
                ),
                "grid_valid": bool(diagnostics.grid_valid_rows[index]),
                "bounds_valid": bool(
                    np.all(selection.X_norm[index] >= 0.0)
                    and np.all(selection.X_norm[index] <= 1.0)
                ),
                "boundary_coordinate_count": int(
                    np.count_nonzero(diagnostics.boundary_flags[index])
                ),
                "pool_seed": config.seed,
                "mc_seed": config.seed,
                "beta": config.beta,
                "kappa": float(np.sqrt(config.beta)),
                "candidate_pool_size": config.candidate_pool_size,
                "posterior_samples": config.posterior_samples,
                "reference_point_utility": json.dumps(
                    config.reference_point_utility.tolist()
                ),
                "config_sha256": config.config_hash,
            }
        )
        rows.append(row)
        diagnostic_rows.append(
            {
                "candidate_id": candidate_id,
                "selection_order": index + 1,
                "nearest_selected_distance": nearest_selected[index],
                "nearest_observed_distance": diagnostics.nearest_observed_pending_distance[
                    index
                ],
                "grid_valid": diagnostics.grid_valid_rows[index],
                "bounds_valid": row["bounds_valid"],
                "boundary_dimensions": "|".join(
                    name
                    for name, is_boundary in zip(
                        D2D_INPUT_COLUMNS, diagnostics.boundary_flags[index]
                    )
                    if is_boundary
                ),
                "debug_only": True,
                "approved_for_experiment": False,
                "candidate_status": D2D_DEBUG_WATERMARK,
                "known_uniformity_score_mismatch": bool(
                    known_uniformity_score_mismatch
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame["grid_valid"].all() or not frame["bounds_valid"].all():
        raise RuntimeError(
            "A selected debug candidate failed grid or bounds validation."
        )
    if frame.loc[:, list(D2D_INPUT_COLUMNS)].duplicated().any():
        raise RuntimeError("The selected debug batch contains duplicate conditions.")
    if any(
        step.base_score is None or step.base_score <= 1e-12 for step in selection.steps
    ):
        raise RuntimeError("Every selected debug candidate must have positive UCB-HVI.")
    return frame, pd.DataFrame(diagnostic_rows)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _git_commit(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _validate_output_destination(
    destination: Path,
    config_file: Path,
    config: ResolvedD2DDebugConfig,
) -> None:
    """Keep in-repository debug writes below the configured ignored output root."""
    del config_file  # The guard must also apply when callers use a copied config.
    repository_root = Path(__file__).resolve().parents[2]
    if destination != repository_root and repository_root not in destination.parents:
        # Pytest and other callers may deliberately use an isolated temp directory.
        return
    allowed_root = (repository_root / config.output_root).resolve()
    if destination != allowed_root and allowed_root not in destination.parents:
        raise ValueError(
            "An output destination inside the repository must be under the "
            f"configured debug output root: {allowed_root}."
        )


def _write_debug_bundle(
    output_dir: Path,
    *,
    audit_payload: dict[str, Any],
    score_validation: pd.DataFrame,
    training_manifest: pd.DataFrame,
    model_diagnostics: pd.DataFrame,
    candidates: pd.DataFrame,
    worklist: pd.DataFrame,
    candidate_diagnostics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    sensitivity_candidates: pd.DataFrame,
    control_ablation: pd.DataFrame,
    run_manifest: dict[str, Any],
    computation: ProposalComputation,
    training: D2DTrainingData,
    config: ResolvedD2DDebugConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    (output_dir / "DEBUG_ONLY_NOT_APPROVED_FOR_EXPERIMENT.txt").write_text(
        D2D_DEBUG_WATERMARK
        + "\nThese candidates are generated only to test the algorithm.\n",
        encoding="utf-8",
    )
    (output_dir / "workbook_audit.json").write_text(
        json.dumps(_jsonable(audit_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    score_validation.to_csv(output_dir / "score_validation.csv", index=False)
    training_manifest.to_csv(output_dir / "training_row_manifest.csv", index=False)
    model_diagnostics.to_csv(output_dir / "model_diagnostics.csv", index=False)
    candidates.to_csv(output_dir / "r1_debug_candidates_unique.csv", index=False)
    worklist.to_csv(output_dir / "r1_debug_replicate_worklist.csv", index=False)
    candidate_diagnostics.to_csv(output_dir / "candidate_diagnostics.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity_summary.csv", index=False)
    sensitivity_candidates.to_csv(
        output_dir / "sensitivity_candidates_long.csv", index=False
    )
    control_ablation.to_csv(output_dir / "control_ablation.csv", index=False)

    selection = computation.proposal.selection
    plot_candidate_pca(
        training.X_norm_all,
        selection.X_norm,
        plots / "r1_candidate_pca.png",
        pool_norm=computation.candidate_pool.X_norm,
        seed=config.seed,
        watermark=D2D_DEBUG_WATERMARK,
    )
    plot_parallel_coordinates(
        selection.X_norm,
        D2D_INPUT_COLUMNS,
        plots / "r1_parallel_coordinates.png",
        watermark=D2D_DEBUG_WATERMARK,
    )
    plot_distance_heatmap(
        selection.X_norm,
        plots / "r1_distance_heatmap.png",
        dimension_weights=config.dimension_weights,
        watermark=D2D_DEBUG_WATERMARK,
    )
    plot_selection_scores(
        [step.order for step in selection.steps],
        [step.base_log_score for step in selection.steps],
        [step.penalized_log_score for step in selection.steps],
        plots / "r1_acquisition_scores.png",
        watermark=D2D_DEBUG_WATERMARK,
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(_jsonable(run_manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_d2d_step2b_debug(
    workbook_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    run_sensitivity: bool = True,
) -> D2DDebugRunResult:
    """Generate a fully watermarked five-condition R1 algorithm-debug bundle.

    The source workbook is opened read-only and never saved.  The legacy
    ``run_mobo_experiment(..., propose_candidates=True)`` path is not used.
    """
    started = perf_counter()
    workbook = Path(workbook_path).resolve()
    config_file = Path(config_path).resolve()
    destination = Path(output_dir).resolve()
    if (
        destination == workbook
        or destination == workbook.parent
        or workbook.parent in destination.parents
    ):
        raise ValueError(
            "Debug output must not target the source workbook or its directory."
        )
    config = load_d2d_debug_config(config_file)
    _validate_output_destination(destination, config_file, config)
    source_hash_before = sha256_file(workbook)
    source_mtime_before = workbook.stat().st_mtime_ns
    if source_hash_before != config.expected_workbook_sha256:
        raise ValueError(
            "Workbook hash does not match the resolved Step 2B config: "
            f"expected={config.expected_workbook_sha256}, actual={source_hash_before}."
        )
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Debug output directory is not empty: {destination}. "
            "Use a new run directory or set overwrite=True."
        )

    frame, audit = load_d2d_workbook_frame(
        workbook,
        expected_profile=config.workbook_profile,
        expected_sample_ids=config.expected_sample_ids,
        allowed_input_exceptions=config.off_grid_exceptions,
    )
    if audit.used_range != config.expected_content_range:
        raise ValueError(
            f"Workbook range must be {config.expected_content_range}; found {audit.used_range}."
        )
    validation = validate_supplied_d2d_scores(frame)
    validation.raise_for_errors()
    training = prepare_d2d_training_data(frame, config, include_control=True)
    if np.count_nonzero(training.include_in_model) != 15:
        raise RuntimeError(
            "The primary debug model must contain all 15 R0 observations."
        )
    fit_started = perf_counter()
    model, train_X, train_Y = _fit_debug_model(training, seed=config.seed)
    baseline_fit_runtime = perf_counter() - fit_started
    computation = _compute_proposal(
        config,
        training,
        model,
        train_X,
        train_Y,
        pool_size=config.candidate_pool_size,
        pool_seed=config.seed,
        mc_seed=config.seed,
        beta=config.beta,
        posterior_samples=config.posterior_samples,
        radius=config.local_radius,
        min_batch_distance=config.min_batch_distance,
    )
    candidates, candidate_diagnostics = _candidates_frame(
        computation,
        config,
        training,
        known_uniformity_score_mismatch=(validation.known_uniformity_score_mismatch),
    )
    worklist = expand_candidates_to_replicates(
        candidates,
        replicates_per_condition=config.replicates_per_condition,
        round_name="R1",
    )
    worklist["known_uniformity_score_mismatch"] = bool(
        validation.known_uniformity_score_mismatch
    )
    if worklist.shape[0] != 15:
        raise RuntimeError("Five R1 conditions must expand to exactly 15 executions.")
    diagnostics = _model_diagnostics(model, train_X, train_Y)
    if run_sensitivity:
        sensitivity_artifacts = _run_sensitivity(
            config,
            training,
            computation,
            baseline_fit_runtime_seconds=baseline_fit_runtime,
            known_uniformity_score_mismatch=(
                validation.known_uniformity_score_mismatch
            ),
        )
    else:
        baseline_settings = _proposal_settings(
            config,
            pool_size=config.candidate_pool_size,
            pool_seed=config.seed,
            mc_seed=config.seed,
            beta=config.beta,
            posterior_samples=config.posterior_samples,
            radius=config.local_radius,
            min_batch_distance=config.min_batch_distance,
            include_control=True,
        )
        baseline_pareto_ids = _observed_pareto_sample_ids(
            train_Y, training.sample_ids[training.include_in_model]
        )
        baseline_summary = _batch_summary(
            "baseline",
            computation,
            computation.proposal.selection.X_phys,
            computation.proposal.selection.X_norm,
            parameter="baseline",
            value="configured",
            settings=baseline_settings,
            fit_runtime_seconds=baseline_fit_runtime,
            observed_pareto_ids=baseline_pareto_ids,
            baseline_pareto_ids=baseline_pareto_ids,
        )
        sensitivity_artifacts = _stamp_uniformity_mismatch(
            SensitivityArtifacts(
                summary=pd.DataFrame([baseline_summary]),
                candidates=pd.DataFrame(
                    _sensitivity_candidate_rows(
                        "baseline",
                        computation,
                        config,
                        training,
                        settings=baseline_settings,
                        fit_runtime_seconds=baseline_fit_runtime,
                    )
                ),
                control_ablation=_empty_control_ablation(),
            ),
            validation.known_uniformity_score_mismatch,
        )
    sensitivity = sensitivity_artifacts.summary

    source_hash_after_compute = sha256_file(workbook)
    source_mtime_after_compute = workbook.stat().st_mtime_ns
    if (
        source_hash_after_compute != source_hash_before
        or source_mtime_after_compute != source_mtime_before
    ):
        raise RuntimeError(
            "The source workbook changed during the read-only debug run."
        )

    training_manifest = training.manifest_frame()
    training_manifest["measurement_provenance"] = "measured_in_current_campaign"
    training_manifest["debug_only"] = True
    training_manifest["approved_for_experiment"] = False
    training_manifest["candidate_status"] = D2D_DEBUG_WATERMARK
    training_manifest["known_uniformity_score_mismatch"] = bool(
        validation.known_uniformity_score_mismatch
    )
    audit_payload = asdict(audit)
    audit_payload.update(
        {
            "source_sha256_before": source_hash_before,
            "source_sha256_after_compute": source_hash_after_compute,
            "source_mtime_ns_before": source_mtime_before,
            "source_mtime_ns_after_compute": source_mtime_after_compute,
            "source_workbook_modified": False,
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": "d2d-step2b-debug-run-v1",
        "debug_only": True,
        "approved_for_experiment": False,
        "approved_for_production": False,
        "watermark": D2D_DEBUG_WATERMARK,
        "known_uniformity_score_mismatch": (validation.known_uniformity_score_mismatch),
        "uniformity_warning_count": validation.uniformity_warning_count,
        "score_validation_error_count": len(validation.errors),
        "objective_order": list(D2D_OBJECTIVE_COLUMNS),
        "objective_transforms": ["identity", "identity", "identity"],
        "objective_directions": ["maximize", "maximize", "maximize"],
        "reference_point_utility": D2D_REFERENCE_POINT_UTILITY.tolist(),
        "ignored_model_columns": ["Stability score?", "AD:AI"],
        "training_row_count": int(train_X.shape[0]),
        "control_sample_ids": list(config.control_sample_ids),
        "control_included": True,
        "off_grid_observed_exceptions": [
            asdict(exception) for exception in config.off_grid_exceptions
        ],
        "candidate_count_unique": int(candidates.shape[0]),
        "replicate_execution_count": int(worklist.shape[0]),
        "baseline_hypervolume": computation.proposal.scoring.baseline_hypervolume,
        "observed_pareto_count": int(
            computation.proposal.scoring.pareto_utility.shape[0]
        ),
        "candidate_pool": {
            "requested": config.candidate_pool_size,
            "accepted": computation.candidate_pool.size,
            "draws": computation.candidate_pool.draws,
            "duplicate_rejections": computation.candidate_pool.rejected_duplicate,
            "avoid_rejections": computation.candidate_pool.rejected_avoid,
            "constraint_rejections": computation.candidate_pool.rejected_constraint,
        },
        "r1_settings": {
            "model_seed": config.seed,
            "pool_seed": config.seed,
            "mc_seed": config.seed,
            "beta": config.beta,
            "kappa": float(np.sqrt(config.beta)),
            "posterior_samples": config.posterior_samples,
            "score_chunk_size": config.score_chunk_size,
            "local_radius": config.local_radius,
            "min_batch_distance": config.min_batch_distance,
            "min_observed_distance": config.min_observed_distance,
        },
        "config_sha256": config.config_hash,
        "workbook_sha256": source_hash_before,
        "git_commit": _git_commit(Path(__file__).resolve().parents[2]),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "botorch": botorch.__version__,
            "gpytorch": gpytorch.__version__,
        },
        "warnings": [*validation.warnings, *training.warnings],
        "sensitivity_run_count": int(sensitivity.shape[0]),
        "sensitivity_failure_count": int((sensitivity["status"] == "failed").sum()),
        "sensitivity_candidate_record_count": int(
            sensitivity_artifacts.candidates.shape[0]
        ),
        "control_ablation_record_count": int(
            sensitivity_artifacts.control_ablation.shape[0]
        ),
        "runtime_seconds_before_output": perf_counter() - started,
        "source_workbook_modified": False,
    }

    _write_debug_bundle(
        destination,
        audit_payload=audit_payload,
        score_validation=validation.frame,
        training_manifest=training_manifest,
        model_diagnostics=diagnostics,
        candidates=candidates,
        worklist=worklist,
        candidate_diagnostics=candidate_diagnostics,
        sensitivity=sensitivity,
        sensitivity_candidates=sensitivity_artifacts.candidates,
        control_ablation=sensitivity_artifacts.control_ablation,
        run_manifest=manifest,
        computation=computation,
        training=training,
        config=config,
    )
    source_hash_after = sha256_file(workbook)
    source_mtime_after = workbook.stat().st_mtime_ns
    if (
        source_hash_after != source_hash_before
        or source_mtime_after != source_mtime_before
    ):
        raise RuntimeError(
            "The source workbook changed while writing the debug bundle."
        )
    manifest["source_sha256_after"] = source_hash_after
    manifest["source_mtime_ns_after"] = source_mtime_after
    manifest["runtime_seconds_total"] = perf_counter() - started
    (destination / "run_manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    return D2DDebugRunResult(
        output_dir=destination,
        candidates_unique=candidates,
        replicate_worklist=worklist,
        sensitivity_summary=sensitivity,
        sensitivity_candidates=sensitivity_artifacts.candidates,
        control_ablation=sensitivity_artifacts.control_ablation,
        model_diagnostics=diagnostics,
        run_manifest=manifest,
    )


__all__ = ["D2DDebugRunResult", "run_d2d_step2b_debug"]
