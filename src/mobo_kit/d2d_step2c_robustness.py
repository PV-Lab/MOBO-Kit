"""Read-only D2D Step 2C robustness, convergence, and stabilization study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import botorch
import gpytorch
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import yaml

from .batch_comparison import RegionalBatchComparison, regional_match_batches
from .batch_selection import (
    BaseScoreResult,
    LocalPenalizationConfig,
    select_local_penalized_batch,
)
from .candidate_diagnostics import summarize_candidate_batch
from .candidate_pool import CandidatePool
from .d2d_campaign import (
    D2D_DEBUG_WATERMARK,
    D2D_INPUT_COLUMNS,
    D2D_OBJECTIVE_COLUMNS,
    D2D_OBJECTIVE_NAMES,
    build_d2d_objective_transform,
    load_d2d_workbook_frame,
    prepare_d2d_training_data,
    sha256_file,
)
from .d2d_scores import validate_supplied_d2d_scores
from .d2d_step2c_config import (
    ExecutionModeSettings,
    PenaltyVariant,
    ResolvedStep2CConfig,
    load_step2c_config,
)
from .discrete_refinement import (
    CachedGridScorer,
    RefinedBatchResult,
    RefinementConfig,
    grid_indices_to_physical_and_normalized,
    propose_refined_discrete_batch,
)
from .model_validation import (
    CONSERVATIVE,
    DIM_SCALED_PRIOR,
    ModelFitCache,
    ModelValidationResult,
    extract_model_hyperparameters,
    fit_warnings_frame,
    validate_model_variant,
)
from .observation_influence import (
    InfluenceRunInput,
    ObservationInfluenceStudyResult,
    run_observation_influence_study,
)
from .robust_regions import (
    ConsensusCriteriaResult,
    RobustRegionResult,
    cluster_candidate_regions,
    evaluate_consensus_criteria,
    select_robust_shortlist,
)
from .robustness_plots import (
    plot_acquisition_quality_vs_persistence,
    plot_ard_lengthscale_comparison,
    plot_boundary_enrichment,
    plot_bounded_utility_comparison,
    plot_candidate_predictions_vs_observed_ranges,
    plot_local_penalty_tradeoff,
    plot_loocv_diagnostics,
    plot_model_policy_region_correspondence,
    plot_nested_search_convergence,
    plot_observation_influence_ranking,
    plot_robust_region_overview,
    plot_run_region_persistence_heatmap,
    plot_control_omission_candidate_region_comparison,
    plot_shortlist_medoid_parallel_coordinates,
    plot_shortlist_region_influence_sensitivity,
)
from .sobol_pool import NestedSobolPoolResult, build_nested_sobol_discrete_pool
from .step2c_artifacts import (
    PUBLIC_SUMMARY_ARCHIVE_ROOT,
    PUBLIC_SUMMARY_CSV_FILES,
    PUBLIC_SUMMARY_DEBUG_MARKER_FILE,
    PUBLIC_SUMMARY_MANIFEST_FILE,
    PUBLIC_SUMMARY_README_FILE,
    PUBLIC_SUMMARY_SCHEMA_VERSION,
    validate_step2c_artifact_bundle,
    validate_step2c_public_summary_archive,
)
from .ucb_hvi import (
    PosteriorIdentityMoments,
    PosteriorUtilityMoments,
    UCBHVIScoreResult,
    posterior_identity_moments,
    posterior_utility_moments,
    score_ucb_hvi_from_moments,
)


STEP2C_METHOD_VERSION = "d2d-step2c-robustness-v1"
STEP2C_BATCH_SIZE = 5


@dataclass(frozen=True)
class StudyBatch:
    """One five-condition result in the one-factor-at-a-time registry."""

    run_id: str
    run_family: str
    core_run: bool
    grid_indices: np.ndarray
    X_phys: np.ndarray
    X_norm: np.ndarray
    base_scores: np.ndarray
    penalized_scores: np.ndarray
    model_variant: str
    pool_seed: int
    pool_size: int
    pool_hash: str
    beta: float
    bound_policy: str
    penalty_label: str
    refinement_enabled: bool
    refinement_runtime_seconds: float
    proposal_runtime_seconds: float
    scoring: UCBHVIScoreResult | None = None
    refinement: RefinedBatchResult | None = None

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid_indices)
        physical = np.asarray(self.X_phys, dtype=float)
        normalized = np.asarray(self.X_norm, dtype=float)
        base = np.asarray(self.base_scores, dtype=float)
        penalized = np.asarray(self.penalized_scores, dtype=float)
        if grid.shape != physical.shape or physical.shape != normalized.shape:
            raise ValueError("StudyBatch coordinate arrays must align.")
        if grid.shape != (STEP2C_BATCH_SIZE, len(D2D_INPUT_COLUMNS)):
            raise ValueError("Every Step 2C study batch must contain five D2D rows.")
        if not np.issubdtype(grid.dtype, np.integer):
            raise ValueError("StudyBatch grid indices must be integers.")
        if base.shape != (STEP2C_BATCH_SIZE,) or penalized.shape != base.shape:
            raise ValueError("StudyBatch acquisition arrays must contain five values.")
        if not (
            np.all(np.isfinite(physical))
            and np.all(np.isfinite(normalized))
            and np.all(np.isfinite(base))
            and np.all(np.isfinite(penalized))
        ):
            raise ValueError("StudyBatch arrays must be finite.")
        if np.any(base <= 0) or np.unique(grid, axis=0).shape[0] != STEP2C_BATCH_SIZE:
            raise ValueError("StudyBatch rows must be unique with positive HVI.")
        if np.any(normalized < 0) or np.any(normalized > 1):
            raise ValueError("StudyBatch normalized rows must lie in [0, 1].")
        if (
            not np.isfinite(self.refinement_runtime_seconds)
            or self.refinement_runtime_seconds < 0.0
        ):
            raise ValueError(
                "refinement_runtime_seconds must be finite and non-negative."
            )
        if (
            not np.isfinite(self.proposal_runtime_seconds)
            or self.proposal_runtime_seconds < 0.0
        ):
            raise ValueError(
                "proposal_runtime_seconds must be finite and non-negative."
            )
        for name, value in (("run_id", self.run_id), ("run_family", self.run_family)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonblank string.")


@dataclass(frozen=True)
class Step2CRobustnessResult:
    output_dir: Path
    mode: str
    run_manifest: dict[str, Any]
    consensus: ConsensusCriteriaResult
    robust_regions: pd.DataFrame
    shortlist: pd.DataFrame
    influence_summary: pd.DataFrame
    artifact_hashes: dict[str, str]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _git_state(repository_root: Path) -> tuple[str, bool, tuple[str, ...]]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        status_text = subprocess.run(
            ["git", "status", "--short"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Step 2C requires readable Git provenance.") from exc
    status = tuple(line for line in status_text.splitlines() if line.strip())
    return commit, bool(status), status


def _hardware_summary() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "botorch": botorch.__version__,
        "gpytorch": gpytorch.__version__,
    }


def _validate_output_destination(
    destination: Path, config: ResolvedStep2CConfig
) -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    allowed_root = (repository_root / config.output_root).resolve()
    resolved = destination.resolve()
    if resolved == repository_root or repository_root not in resolved.parents:
        raise ValueError("Step 2C output must remain inside the repository.")
    if resolved == allowed_root:
        raise ValueError("Choose a run directory below the Step 2C output root.")
    if allowed_root not in resolved.parents:
        raise ValueError(
            f"Step 2C output must remain below the ignored root {allowed_root}."
        )
    return resolved


def _stamp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["debug_only"] = True
    result["approved_for_experiment"] = False
    result["approved_for_production"] = False
    result["candidate_status"] = D2D_DEBUG_WATERMARK
    return result


def _declared_bound_flags(
    values: np.ndarray,
    bounds: tuple[float | None, float | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower, upper, and combined declared-support flags."""
    array = np.asarray(values, dtype=float)
    lower, upper = bounds
    below = np.zeros(array.shape, dtype=bool)
    above = np.zeros(array.shape, dtype=bool)
    if lower is not None:
        below = array < lower
    if upper is not None:
        above = array > upper
    return below, above, below | above


def _frame_from_dataclasses(
    rows: Iterable[Any], *, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    values = [asdict(row) if is_dataclass(row) else dict(row) for row in rows]
    return pd.DataFrame(values, columns=columns)


def _safe_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    stamped = dict(payload)
    stamped.update(
        {
            "debug_only": True,
            "approved_for_experiment": False,
            "approved_for_production": False,
            "candidate_status": D2D_DEBUG_WATERMARK,
        }
    )
    path.write_text(
        json.dumps(_jsonable(stamped), indent=2, sort_keys=True), encoding="utf-8"
    )


def _pairwise_minimum(X_norm: np.ndarray) -> float:
    matrix = np.asarray(X_norm, dtype=float)
    if matrix.shape[0] < 2:
        return np.nan
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=-1)
    return float(distances[np.triu_indices(matrix.shape[0], k=1)].min())


def _pool_with_rows(
    pool: CandidatePool, extra_grid_indices: np.ndarray, design
) -> CandidatePool:
    extras = np.asarray(extra_grid_indices, dtype=np.int64)
    combined = np.vstack([pool.grid_indices, extras])
    _, first = np.unique(combined, axis=0, return_index=True)
    ordered = combined[np.sort(first)]
    physical, normalized = grid_indices_to_physical_and_normalized(ordered, design)
    return CandidatePool(
        grid_indices=ordered,
        X_phys=physical,
        X_norm=normalized,
        seed=pool.seed,
        draws=pool.draws,
        rejected_duplicate=pool.rejected_duplicate,
        rejected_avoid=pool.rejected_avoid,
        rejected_constraint=pool.rejected_constraint,
    )


def _analytic_numpy_moments(
    model: Any,
    X_norm: np.ndarray,
    *,
    objective_transform: Any,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, PosteriorIdentityMoments, float]:
    started = perf_counter()
    result = posterior_identity_moments(
        model,
        torch.as_tensor(X_norm, dtype=torch.double, device="cpu"),
        objective_transform,
        chunk_size=chunk_size,
        observation_noise=False,
    )
    if result.utility_mean.ndim != 2 or result.utility_std.ndim != 2:
        raise RuntimeError("Step 2C requires unbatched (N, M) analytic moments.")
    return (
        result.utility_mean.detach().cpu().double().numpy(),
        result.utility_std.detach().cpu().double().numpy(),
        result,
        perf_counter() - started,
    )


def _score_moments(
    utility_mean: np.ndarray,
    utility_std: np.ndarray,
    observed_y: np.ndarray,
    config: ResolvedStep2CConfig,
    *,
    beta: float,
    bound_policy: str,
) -> tuple[UCBHVIScoreResult, float]:
    started = perf_counter()
    result = score_ucb_hvi_from_moments(
        utility_mean,
        utility_std,
        observed_y,
        config.reference_point_utility,
        beta=beta,
        numeric_tolerance=config.numeric_tolerance,
        chunk_size=2048,
        log_epsilon=1.0e-300,
        mc_samples=None,
        seed=None,
        observation_noise=False,
        objective_contract_version=build_d2d_objective_transform().version,
        moment_method="analytic_identity",
        bound_policy=bound_policy,
        utility_bounds=config.objective_bounds,
    )
    return result, perf_counter() - started


def _grid_score_function(
    model: Any,
    design: Any,
    observed_y: np.ndarray,
    config: ResolvedStep2CConfig,
    *,
    beta: float,
    bound_policy: str,
):
    transform = build_d2d_objective_transform()

    def score(grid_indices: np.ndarray) -> np.ndarray:
        _, X_norm = grid_indices_to_physical_and_normalized(grid_indices, design)
        means, stds, _, _ = _analytic_numpy_moments(
            model,
            X_norm,
            objective_transform=transform,
            chunk_size=config.score_chunk_size,
        )
        scoring, _ = _score_moments(
            means,
            stds,
            observed_y,
            config,
            beta=beta,
            bound_policy=bound_policy,
        )
        return scoring.base_score

    return score


def _static_select(
    pool: CandidatePool,
    base_scores: np.ndarray,
    *,
    q: int,
    penalty: PenaltyVariant,
    observed_norm: np.ndarray,
) -> Any:
    scores = np.asarray(base_scores, dtype=float)
    if scores.shape != (pool.size,) or not np.all(np.isfinite(scores)):
        raise ValueError("base_scores must be finite and align with the pool.")
    if np.count_nonzero(scores > 0.0) < q:
        raise RuntimeError("Fewer than q candidates have positive UCB-HVI.")
    log_scores = np.full(scores.shape, -np.inf, dtype=float)
    positive = scores > 0.0
    log_scores[positive] = np.log(scores[positive])

    def score_remaining(
        remaining_indices: np.ndarray, selected_indices: np.ndarray
    ) -> BaseScoreResult:
        del selected_indices
        return BaseScoreResult(
            base_log_score=log_scores[remaining_indices],
            base_score=scores[remaining_indices],
        )

    return select_local_penalized_batch(
        pool,
        q,
        score_remaining,
        LocalPenalizationConfig(
            radius=penalty.radius,
            min_batch_distance=penalty.min_batch_distance,
            min_observed_distance=0.0,
            dimension_weights=None,
        ),
        observed_pending_norm=observed_norm,
    )


def _penalty_by_label(config: ResolvedStep2CConfig, label: str) -> PenaltyVariant:
    for variant in config.penalty_variants:
        if variant.label == label:
            return variant
    raise KeyError(f"Unknown penalty variant {label!r}.")


def _study_batch_from_refined(
    *,
    run_id: str,
    run_family: str,
    core_run: bool,
    refined: RefinedBatchResult,
    model_variant: str,
    pool: CandidatePool,
    pool_hash: str,
    beta: float,
    bound_policy: str,
    penalty_label: str,
    scoring: UCBHVIScoreResult | None,
    refinement_runtime_seconds: float,
) -> StudyBatch:
    return StudyBatch(
        run_id=run_id,
        run_family=run_family,
        core_run=core_run,
        grid_indices=refined.grid_indices.copy(),
        X_phys=refined.X_phys.copy(),
        X_norm=refined.X_norm.copy(),
        base_scores=refined.base_scores.copy(),
        penalized_scores=refined.penalized_scores_at_selection.copy(),
        model_variant=model_variant,
        pool_seed=pool.seed,
        pool_size=pool.size,
        pool_hash=pool_hash,
        beta=float(beta),
        bound_policy=bound_policy,
        penalty_label=penalty_label,
        refinement_enabled=True,
        refinement_runtime_seconds=refinement_runtime_seconds,
        proposal_runtime_seconds=refinement_runtime_seconds,
        scoring=scoring,
        refinement=refined,
    )


def _study_batch_from_static(
    *,
    run_id: str,
    run_family: str,
    core_run: bool,
    selection: Any,
    model_variant: str,
    pool: CandidatePool,
    pool_hash: str,
    beta: float,
    bound_policy: str,
    penalty_label: str,
    scoring: UCBHVIScoreResult | None,
    selection_runtime_seconds: float = 0.0,
) -> StudyBatch:
    return StudyBatch(
        run_id=run_id,
        run_family=run_family,
        core_run=core_run,
        grid_indices=pool.grid_indices[selection.selected_pool_indices].copy(),
        X_phys=selection.X_phys.copy(),
        X_norm=selection.X_norm.copy(),
        base_scores=np.asarray(
            [step.base_score for step in selection.steps], dtype=float
        ),
        penalized_scores=np.exp(
            np.asarray([step.penalized_log_score for step in selection.steps])
        ),
        model_variant=model_variant,
        pool_seed=pool.seed,
        pool_size=pool.size,
        pool_hash=pool_hash,
        beta=float(beta),
        bound_policy=bound_policy,
        penalty_label=penalty_label,
        refinement_enabled=False,
        refinement_runtime_seconds=0.0,
        proposal_runtime_seconds=selection_runtime_seconds,
        scoring=scoring,
        refinement=None,
    )


def _run_refined_batch(
    *,
    run_id: str,
    run_family: str,
    core_run: bool,
    pool: CandidatePool,
    pool_hash: str,
    master_base_scores: np.ndarray,
    shared_grid_scorer: CachedGridScorer,
    config: ResolvedStep2CConfig,
    mode_settings: ExecutionModeSettings,
    training: Any,
    model_variant: str,
    beta: float,
    bound_policy: str,
    penalty_label: str,
    scoring: UCBHVIScoreResult | None,
) -> StudyBatch:
    penalty = _penalty_by_label(config, penalty_label)
    refinement_started = perf_counter()
    refined = propose_refined_discrete_batch(
        pool,
        config.design,
        shared_grid_scorer,
        q=STEP2C_BATCH_SIZE,
        config=RefinementConfig(
            anchors_per_selection_step=mode_settings.anchors_per_selection_step,
            max_sweeps=config.refinement_max_sweeps,
            improvement_tolerance=config.refinement_tolerance,
            radius=penalty.radius,
            min_batch_distance=penalty.min_batch_distance,
            min_observed_distance=0.0,
            dimension_weights=None,
        ),
        master_base_scores=master_base_scores,
        observed_grid_indices=training.on_grid_grid_indices,
        observed_norm=training.X_norm_all,
        positive_score_threshold=config.positive_hvi_threshold,
    )
    refinement_runtime = perf_counter() - refinement_started
    return _study_batch_from_refined(
        run_id=run_id,
        run_family=run_family,
        core_run=core_run,
        refined=refined,
        model_variant=model_variant,
        pool=pool,
        pool_hash=pool_hash,
        beta=beta,
        bound_policy=bound_policy,
        penalty_label=penalty_label,
        scoring=scoring,
        refinement_runtime_seconds=refinement_runtime,
    )


def _comparison_row(
    reference: StudyBatch,
    comparison: StudyBatch,
    *,
    comparison_type: str,
) -> dict[str, Any]:
    match = regional_match_batches(reference.X_norm, comparison.X_norm)
    reference_sum = float(reference.base_scores.sum())
    comparison_sum = float(comparison.base_scores.sum())
    reference_boundary_count = int(
        np.count_nonzero(
            np.isclose(reference.X_norm, 0.0) | np.isclose(reference.X_norm, 1.0)
        )
    )
    comparison_boundary_count = int(
        np.count_nonzero(
            np.isclose(comparison.X_norm, 0.0) | np.isclose(comparison.X_norm, 1.0)
        )
    )
    row: dict[str, Any] = {
        "comparison_type": comparison_type,
        "reference_run_id": reference.run_id,
        "comparison_run_id": comparison.run_id,
        "reference_pool_size": reference.pool_size,
        "comparison_pool_size": comparison.pool_size,
        "exact_overlap_count": match.exact_overlap_count,
        "jaccard_overlap": match.jaccard_overlap,
        "mean_matched_distance": match.mean_matched_distance,
        "maximum_matched_distance": match.maximum_matched_distance,
        "symmetric_chamfer_distance": match.symmetric_chamfer_distance,
        "hausdorff_distance": match.hausdorff_distance,
        "reference_acquisition_sum": reference_sum,
        "comparison_acquisition_sum": comparison_sum,
        "acquisition_regret_vs_reference": reference_sum - comparison_sum,
        "relative_acquisition_regret_vs_reference": (
            (reference_sum - comparison_sum) / reference_sum
            if reference_sum > 0.0
            else np.nan
        ),
        "reference_boundary_coordinate_count": reference_boundary_count,
        "comparison_boundary_coordinate_count": comparison_boundary_count,
        "boundary_coordinate_count_difference": (
            comparison_boundary_count - reference_boundary_count
        ),
    }
    for threshold, count in match.regional_match_counts.items():
        row[f"regional_matches_within_{threshold:.2f}"] = count
    row["matched_pair_distances"] = "|".join(
        f"{value:.12g}" for value in match.matched_pair_distances
    )
    return row


def _study_summary(batches: Sequence[StudyBatch]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        pairwise = np.linalg.norm(
            batch.X_norm[:, None, :] - batch.X_norm[None, :, :], axis=-1
        )
        triangle = pairwise[np.triu_indices(STEP2C_BATCH_SIZE, k=1)]
        rows.append(
            {
                "run_id": batch.run_id,
                "run_family": batch.run_family,
                "core_run": batch.core_run,
                "model_variant": batch.model_variant,
                "pool_seed": batch.pool_seed,
                "pool_size": batch.pool_size,
                "pool_hash": batch.pool_hash,
                "beta": batch.beta,
                "bound_policy": batch.bound_policy,
                "penalty_label": batch.penalty_label,
                "refinement_enabled": batch.refinement_enabled,
                "refinement_runtime_seconds": batch.refinement_runtime_seconds,
                "proposal_runtime_seconds": batch.proposal_runtime_seconds,
                "acquisition_sum": float(batch.base_scores.sum()),
                "acquisition_mean": float(batch.base_scores.mean()),
                "minimum_within_batch_distance": float(triangle.min()),
                "mean_within_batch_distance": float(triangle.mean()),
                "maximum_within_batch_distance": float(triangle.max()),
                "mean_penalty_factor": float(
                    np.mean(batch.penalized_scores / batch.base_scores)
                ),
                "distinct_refined_optima": (
                    batch.refinement.distinct_converged_optima
                    if batch.refinement is not None
                    else np.nan
                ),
            }
        )
    return _stamp_frame(pd.DataFrame(rows))


def _batch_candidate_rows(
    batches: Sequence[StudyBatch],
    *,
    models: Mapping[str, Any],
    config: ResolvedStep2CConfig,
    training: Any,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    transform = build_d2d_objective_transform()
    control_mask = np.isin(training.sample_ids, config.control_sample_ids)
    control_norm = training.X_norm_all[control_mask]
    for batch in batches:
        model = models[batch.model_variant]
        means, stds, _, _ = _analytic_numpy_moments(
            model,
            batch.X_norm,
            objective_transform=transform,
            chunk_size=config.score_chunk_size,
        )
        selected_scoring, _ = _score_moments(
            means,
            stds,
            training.Y_objectives,
            config,
            beta=batch.beta,
            bound_policy=batch.bound_policy,
        )
        if not np.allclose(
            selected_scoring.base_score,
            batch.base_scores,
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"Selected score recomputation changed for run {batch.run_id}."
            )
        diagnostics = summarize_candidate_batch(
            batch.X_norm,
            observed_pending_norm=training.X_norm_all,
            X_phys=batch.X_phys,
            design=config.design,
            metadata={"debug_only": True, "run_id": batch.run_id},
        )
        control_distances = np.linalg.norm(
            batch.X_norm[:, None, :] - control_norm[None, :, :], axis=-1
        ).min(axis=1)
        minimum_distance = _pairwise_minimum(batch.X_norm)
        required_distance = _penalty_by_label(
            config, batch.penalty_label
        ).min_batch_distance
        for index in range(STEP2C_BATCH_SIZE):
            row: dict[str, Any] = {
                "run_id": batch.run_id,
                "run_family": batch.run_family,
                "core_run": batch.core_run,
                "candidate_id": f"{batch.run_id}-C{index + 1:02d}",
                "selection_order": index + 1,
                "model_variant": batch.model_variant,
                "pool_seed": batch.pool_seed,
                "pool_size": batch.pool_size,
                "pool_hash": batch.pool_hash,
                "beta": batch.beta,
                "bound_policy": batch.bound_policy,
                "penalty_label": batch.penalty_label,
                "refinement_enabled": batch.refinement_enabled,
                "acquisition_score": float(batch.base_scores[index]),
                "penalized_acquisition_score": float(batch.penalized_scores[index]),
                "penalty_factor": float(
                    batch.penalized_scores[index] / batch.base_scores[index]
                ),
                "nearest_observed_distance": float(
                    diagnostics.nearest_observed_pending_distance[index]
                ),
                "nearest_control_distance": float(control_distances[index]),
                "minimum_within_batch_distance": minimum_distance,
                "hard_distance_required": required_distance,
                "hard_distance_valid": bool(
                    minimum_distance + 1.0e-12 >= required_distance
                ),
                "grid_valid": bool(diagnostics.grid_valid_rows[index]),
                "bounds_valid": bool(
                    np.all(batch.X_norm[index] >= 0.0)
                    and np.all(batch.X_norm[index] <= 1.0)
                    and np.all(np.isfinite(batch.X_phys[index]))
                ),
                "boundary_coordinate_count": int(
                    np.count_nonzero(diagnostics.boundary_flags[index])
                ),
                "boundary_dimensions": "|".join(
                    name
                    for name, flag in zip(
                        D2D_INPUT_COLUMNS, diagnostics.boundary_flags[index]
                    )
                    if flag
                ),
                "lower_boundary_coordinate_count": int(
                    np.count_nonzero(np.isclose(batch.X_norm[index], 0.0))
                ),
                "upper_boundary_coordinate_count": int(
                    np.count_nonzero(np.isclose(batch.X_norm[index], 1.0))
                ),
                "lower_boundary_dimensions": "|".join(
                    name
                    for name, value in zip(D2D_INPUT_COLUMNS, batch.X_norm[index])
                    if np.isclose(value, 0.0)
                ),
                "upper_boundary_dimensions": "|".join(
                    name
                    for name, value in zip(D2D_INPUT_COLUMNS, batch.X_norm[index])
                    if np.isclose(value, 1.0)
                ),
            }
            for dimension, name in enumerate(D2D_INPUT_COLUMNS):
                row[name] = float(batch.X_phys[index, dimension])
                row[f"phys_{dimension}"] = float(batch.X_phys[index, dimension])
                row[f"norm_{dimension}"] = float(batch.X_norm[index, dimension])
                row[f"grid_{dimension}"] = int(batch.grid_indices[index, dimension])
            for objective, objective_name in enumerate(D2D_OBJECTIVE_NAMES):
                observed_minimum = float(training.Y_objectives[:, objective].min())
                observed_maximum = float(training.Y_objectives[:, objective].max())
                row[f"pred_mean_{objective}"] = float(means[index, objective])
                row[f"pred_std_{objective}"] = float(stds[index, objective])
                row[f"observed_minimum_{objective}"] = observed_minimum
                row[f"observed_maximum_{objective}"] = observed_maximum
                row[f"pred_mean_below_observed_{objective}"] = bool(
                    means[index, objective] < observed_minimum
                )
                row[f"pred_mean_above_observed_{objective}"] = bool(
                    means[index, objective] > observed_maximum
                )
                lower, upper = config.objective_bounds[objective]
                below, above, outside = _declared_bound_flags(
                    means[index, objective], config.objective_bounds[objective]
                )
                row[f"pred_mean_below_declared_bounds_{objective}"] = bool(below)
                row[f"pred_mean_above_declared_bounds_{objective}"] = bool(above)
                row[f"pred_mean_outside_declared_bounds_{objective}"] = bool(outside)
                row[f"ucb_raw_{objective}"] = float(
                    selected_scoring.utility_ucb_raw[index, objective]
                )
                row[f"ucb_effective_{objective}"] = float(
                    selected_scoring.utility_ucb_effective[index, objective]
                )
                row[f"ucb_clip_amount_{objective}"] = float(
                    selected_scoring.utility_ucb_clip_amount[index, objective]
                )
                row[f"ucb_effective_outside_declared_bounds_{objective}"] = bool(
                    (
                        lower is not None
                        and selected_scoring.utility_ucb_effective[index, objective]
                        < lower
                    )
                    or (
                        upper is not None
                        and selected_scoring.utility_ucb_effective[index, objective]
                        > upper
                    )
                )
                row[f"objective_name_{objective}"] = objective_name
            rows.append(row)
    return _stamp_frame(pd.DataFrame(rows))


def _refinement_trace_frame(batches: Sequence[StudyBatch]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        if batch.refinement is None:
            continue
        for trace in batch.refinement.trace:
            row = asdict(trace)
            row.update(
                {
                    "record_type": "coordinate_move",
                    "run_id": batch.run_id,
                    "run_family": batch.run_family,
                    "pool_size": batch.pool_size,
                    "pool_seed": batch.pool_seed,
                    "model_variant": batch.model_variant,
                    "beta": batch.beta,
                    "bound_policy": batch.bound_policy,
                    "penalty_label": batch.penalty_label,
                    "start_grid_index": "|".join(
                        str(value) for value in trace.start_grid_index
                    ),
                    "chosen_grid_index": "|".join(
                        str(value) for value in trace.chosen_grid_index
                    ),
                }
            )
            rows.append(row)
        for anchor in batch.refinement.anchors:
            row = asdict(anchor)
            row.update(
                {
                    "record_type": "anchor_summary",
                    "run_id": batch.run_id,
                    "run_family": batch.run_family,
                    "pool_size": batch.pool_size,
                    "pool_seed": batch.pool_seed,
                    "model_variant": batch.model_variant,
                    "beta": batch.beta,
                    "bound_policy": batch.bound_policy,
                    "penalty_label": batch.penalty_label,
                    "start_grid_index": "|".join(
                        str(value) for value in anchor.anchor_grid_index
                    ),
                    "chosen_grid_index": "|".join(
                        str(value) for value in anchor.refined_grid_index
                    ),
                    "changed_dimensions": "|".join(anchor.changed_dimensions),
                    "total_coordinate_moves": anchor.accepted_move_count,
                    "total_sweeps": anchor.sweeps,
                    "end_base_score": anchor.base_score,
                    "end_penalized_score": anchor.penalized_score,
                }
            )
            rows.append(row)
    return _stamp_frame(pd.DataFrame(rows))


def _nested_pool_manifest(
    results: Sequence[NestedSobolPoolResult], *, role_by_seed: Mapping[int, str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        for size in result.accepted_sizes:
            pool = result.pools[size]
            rows.append(
                {
                    "pool_role": role_by_seed[result.scramble_seed],
                    "scramble_seed": result.scramble_seed,
                    "accepted_size": size,
                    "prefix_sha256": result.prefix_hashes[size],
                    "draws_at_prefix": pool.draws,
                    "duplicate_rejections_at_prefix": pool.rejected_duplicate,
                    "avoid_rejections_at_prefix": pool.rejected_avoid,
                    "constraint_rejections_at_prefix": pool.rejected_constraint,
                    "final_raw_sobol_draws": result.raw_sobol_draws,
                    "ignored_off_grid_observed_rows": result.ignored_off_grid_observed,
                    "scipy_version": result.scipy_version,
                    "exact_accepted_prefix": True,
                    "full_cartesian_grid_materialized": False,
                }
            )
    return _stamp_frame(pd.DataFrame(rows))


def _nested_convergence_table(
    nested_batches: Sequence[StudyBatch],
    raw_pool_batches: Sequence[StudyBatch],
) -> pd.DataFrame:
    ordered = sorted(nested_batches, key=lambda batch: batch.pool_size)
    raw_by_size = {batch.pool_size: batch for batch in raw_pool_batches}
    if set(raw_by_size) != {batch.pool_size for batch in ordered}:
        raise RuntimeError("Raw and refined nested-pool batches must align by size.")
    largest = ordered[-1]
    rows: list[dict[str, Any]] = []
    for index, batch in enumerate(ordered):
        reference = largest
        row = _comparison_row(
            reference,
            batch,
            comparison_type="prefix_vs_largest",
        )
        row["prefix_order"] = index + 1
        row["is_largest_prefix"] = batch is largest
        raw_batch = raw_by_size[batch.pool_size]
        raw_comparison = _comparison_row(
            raw_batch,
            batch,
            comparison_type="raw_pool_vs_refined",
        )
        row.update(
            {
                "raw_pool_acquisition_sum": float(raw_batch.base_scores.sum()),
                "raw_pool_first_acquisition": float(raw_batch.base_scores[0]),
                "raw_pool_median_acquisition": float(np.median(raw_batch.base_scores)),
                "refined_acquisition_sum": float(batch.base_scores.sum()),
                "refined_first_acquisition": float(batch.base_scores[0]),
                "refined_median_acquisition": float(np.median(batch.base_scores)),
                "refinement_acquisition_gain": float(
                    batch.base_scores.sum() - raw_batch.base_scores.sum()
                ),
                "refinement_runtime_seconds": batch.refinement_runtime_seconds,
                "raw_vs_refined_exact_overlap_count": raw_comparison[
                    "exact_overlap_count"
                ],
                "raw_vs_refined_mean_matched_distance": raw_comparison[
                    "mean_matched_distance"
                ],
                "raw_vs_refined_maximum_matched_distance": raw_comparison[
                    "maximum_matched_distance"
                ],
                "raw_pool_boundary_coordinate_count": raw_comparison[
                    "reference_boundary_coordinate_count"
                ],
                "refined_boundary_coordinate_count": raw_comparison[
                    "comparison_boundary_coordinate_count"
                ],
            }
        )
        if index > 0:
            adjacent = _comparison_row(
                ordered[index - 1], batch, comparison_type="adjacent_prefixes"
            )
            for key, value in adjacent.items():
                if key not in {
                    "comparison_type",
                    "reference_run_id",
                    "comparison_run_id",
                }:
                    row[f"adjacent_{key}"] = value
            row["adjacent_reference_run_id"] = ordered[index - 1].run_id
        rows.append(row)
    return _stamp_frame(pd.DataFrame(rows))


def _sobol_scramble_table(
    baseline: StudyBatch, secondary_batches: Sequence[StudyBatch]
) -> pd.DataFrame:
    return _stamp_frame(
        pd.DataFrame(
            [
                _comparison_row(
                    baseline, batch, comparison_type="secondary_sobol_scramble"
                )
                for batch in secondary_batches
            ]
        )
    )


def _bounded_utility_table(
    clip_batch: StudyBatch,
    none_batch: StudyBatch,
    clip_scoring: UCBHVIScoreResult,
    selected_candidate_rows: pd.DataFrame,
) -> pd.DataFrame:
    row = _comparison_row(
        clip_batch, none_batch, comparison_type="clip_ucb_vs_unbounded"
    )
    clip_amount = np.asarray(clip_scoring.utility_ucb_clip_amount, dtype=float)
    selected = selected_candidate_rows[
        selected_candidate_rows["run_id"].eq(clip_batch.run_id)
    ].sort_values("selection_order")
    selected_clip_columns = [
        f"ucb_clip_amount_{index}" for index in range(len(D2D_OBJECTIVE_NAMES))
    ]
    if selected.shape[0] != STEP2C_BATCH_SIZE or any(
        column not in selected for column in selected_clip_columns
    ):
        raise RuntimeError(
            "Bounded-utility selected-candidate diagnostics are incomplete."
        )
    selected_clip = selected.loc[:, selected_clip_columns].to_numpy(dtype=float)
    row.update(
        {
            "bounded_reference_run": clip_batch.run_id,
            "unbounded_comparison_run": none_batch.run_id,
            "clipped_pool_coordinate_count": int(np.count_nonzero(clip_amount > 0)),
            "clipped_pool_row_count": int(
                np.count_nonzero(np.any(clip_amount > 0, axis=1))
            ),
            "maximum_pool_clip_amount": float(clip_amount.max()),
            "mean_positive_pool_clip_amount": (
                float(clip_amount[clip_amount > 0].mean())
                if np.any(clip_amount > 0)
                else 0.0
            ),
            "uniformity_pool_clipped_count": int(
                np.count_nonzero(clip_amount[:, 0] > 0)
            ),
            "optoelectronic_pool_clipped_count": int(
                np.count_nonzero(clip_amount[:, 1] > 0)
            ),
            "thickness_pool_clipped_count": int(
                np.count_nonzero(clip_amount[:, 2] > 0)
            ),
            "clipped_selected_coordinate_count": int(
                np.count_nonzero(selected_clip > 0)
            ),
            "clipped_selected_candidate_count": int(
                np.count_nonzero(np.any(selected_clip > 0, axis=1))
            ),
            "maximum_selected_clip_amount": float(selected_clip.max()),
            "training_targets_mutated": False,
        }
    )
    return _stamp_frame(pd.DataFrame([row]))


def _penalty_tradeoff_table(
    batches: Sequence[StudyBatch],
    *,
    observed_norm: np.ndarray,
    reference_label: str = "no_soft_no_hard",
) -> pd.DataFrame:
    by_label = {batch.penalty_label: batch for batch in batches}
    no_hard_reference = by_label[reference_label]
    fixed_hard_reference = by_label["no_soft_hard_0_15"]
    rows: list[dict[str, Any]] = []
    for batch in batches:
        reference = (
            no_hard_reference
            if batch.penalty_label in {reference_label, "no_soft_hard_0_15"}
            else fixed_hard_reference
        )
        row = _comparison_row(
            reference, batch, comparison_type="local_penalty_isolation"
        )
        pairwise = np.linalg.norm(
            batch.X_norm[:, None, :] - batch.X_norm[None, :, :], axis=-1
        )
        triangle = pairwise[np.triu_indices(STEP2C_BATCH_SIZE, k=1)]
        reference_pairwise = np.linalg.norm(
            reference.X_norm[:, None, :] - reference.X_norm[None, :, :], axis=-1
        )
        reference_triangle = reference_pairwise[np.triu_indices(STEP2C_BATCH_SIZE, k=1)]
        factors = batch.penalized_scores / batch.base_scores
        acquisition_sacrifice = float(
            reference.base_scores.sum() - batch.base_scores.sum()
        )
        nearest_observed = np.linalg.norm(
            batch.X_norm[:, None, :] - observed_norm[None, :, :], axis=-1
        ).min(axis=1)
        boundary_count = int(
            np.count_nonzero(
                np.isclose(batch.X_norm, 0.0) | np.isclose(batch.X_norm, 1.0)
            )
        )
        minimum_distance_change = float(triangle.min() - reference_triangle.min())
        mean_distance_change = float(triangle.mean() - reference_triangle.mean())
        regional_matches = int(row["regional_matches_within_0.15"])
        materially_changes_diversity = bool(
            regional_matches <= 3
            or abs(minimum_distance_change) >= 0.05
            or abs(mean_distance_change) >= 0.05
        )
        modestly_changes_diversity = bool(
            materially_changes_diversity
            or regional_matches < STEP2C_BATCH_SIZE
            or abs(minimum_distance_change) >= 0.01
            or abs(mean_distance_change) >= 0.01
        )
        if materially_changes_diversity:
            activity = "active and materially changes diversity"
        elif modestly_changes_diversity:
            activity = "active but only modestly changes diversity"
        else:
            activity = (
                "implemented but effectively inactive because base optima "
                "are already separated"
            )
        row.update(
            {
                "penalty_label": batch.penalty_label,
                "comparison_reference_penalty_label": reference.penalty_label,
                "soft_radius": (
                    np.nan
                    if batch.penalty_label.startswith("no_soft")
                    else float(".".join(batch.penalty_label.split("_")[-2:]))
                ),
                "minimum_within_batch_distance": float(triangle.min()),
                "mean_within_batch_distance": float(triangle.mean()),
                "maximum_within_batch_distance": float(triangle.max()),
                "minimum_within_batch_distance_change_vs_reference": (
                    minimum_distance_change
                ),
                "mean_within_batch_distance_change_vs_reference": (
                    mean_distance_change
                ),
                "acquisition_sum": float(batch.base_scores.sum()),
                "acquisition_median": float(np.median(batch.base_scores)),
                "acquisition_minimum": float(batch.base_scores.min()),
                "acquisition_sacrifice": acquisition_sacrifice,
                "relative_acquisition_sacrifice": (
                    acquisition_sacrifice / float(reference.base_scores.sum())
                ),
                "minimum_penalty_factor": float(factors.min()),
                "mean_penalty_factor": float(factors.mean()),
                "maximum_penalty_factor": float(factors.max()),
                "minimum_nearest_observed_distance": float(nearest_observed.min()),
                "mean_nearest_observed_distance": float(nearest_observed.mean()),
                "boundary_coordinate_count": boundary_count,
                "proposal_runtime_seconds": batch.proposal_runtime_seconds,
                "candidate_region_changed_vs_reference": bool(
                    regional_matches < STEP2C_BATCH_SIZE
                ),
                "material_diversity_distance_change_threshold": 0.05,
                "modest_diversity_distance_change_threshold": 0.01,
                "material_diversity_maximum_regional_matches": 3,
                "materially_changes_diversity": materially_changes_diversity,
                "modestly_changes_diversity": modestly_changes_diversity,
                "penalty_activity_classification": activity,
                "hard_distance_relaxed": False,
            }
        )
        rows.append(row)
    return _stamp_frame(pd.DataFrame(rows))


def _beta_robustness_table(
    beta_four: StudyBatch, beta_variants: Sequence[StudyBatch]
) -> pd.DataFrame:
    rows = [
        _comparison_row(beta_four, batch, comparison_type="beta_robustness")
        | {"beta": batch.beta}
        for batch in (beta_four, *beta_variants)
    ]
    return _stamp_frame(pd.DataFrame(rows))


def _boundary_enrichment_table(
    pool: CandidatePool,
    base_scores: np.ndarray,
    selected: StudyBatch,
    comparison_batches: Sequence[StudyBatch],
) -> pd.DataFrame:
    scores = np.asarray(base_scores, dtype=float)
    top_count = max(1, int(np.ceil(pool.size * 0.01)))
    top_indices = np.argsort(-scores, kind="stable")[:top_count]
    groups: list[tuple[str, np.ndarray, StudyBatch | None]] = [
        ("pool", np.asarray(pool.X_norm), None),
        (
            "top_1_percent_acquisition",
            np.asarray(pool.X_norm)[top_indices],
            None,
        ),
        ("selected_batch", selected.X_norm, selected),
        *[
            (f"selected_run:{batch.run_id}", batch.X_norm, batch)
            for batch in comparison_batches
            if batch.run_id != selected.run_id
        ],
    ]
    pool_lower_rates = np.mean(np.isclose(pool.X_norm, 0.0), axis=0)
    pool_upper_rates = np.mean(np.isclose(pool.X_norm, 1.0), axis=0)
    pool_rates = np.mean(
        np.isclose(pool.X_norm, 0.0) | np.isclose(pool.X_norm, 1.0), axis=0
    )
    rows: list[dict[str, Any]] = []
    for dimension, name in enumerate(D2D_INPUT_COLUMNS):
        for group_name, X, batch in groups:
            lower_flags = np.isclose(X[:, dimension], 0.0)
            upper_flags = np.isclose(X[:, dimension], 1.0)
            flags = lower_flags | upper_flags
            rate = float(flags.mean())
            lower_rate = float(lower_flags.mean())
            upper_rate = float(upper_flags.mean())
            rows.append(
                {
                    "dimension_index": dimension,
                    "input_name": name,
                    "group": group_name,
                    "group_kind": (
                        "search_pool"
                        if batch is None and group_name == "pool"
                        else ("top_acquisition" if batch is None else "selected_batch")
                    ),
                    "run_id": batch.run_id if batch is not None else None,
                    "run_family": batch.run_family if batch is not None else None,
                    "model_variant": (
                        batch.model_variant if batch is not None else None
                    ),
                    "beta": batch.beta if batch is not None else np.nan,
                    "bound_policy": (batch.bound_policy if batch is not None else None),
                    "row_count": int(X.shape[0]),
                    "boundary_count": int(flags.sum()),
                    "boundary_rate": rate,
                    "lower_boundary_count": int(lower_flags.sum()),
                    "lower_boundary_rate": lower_rate,
                    "upper_boundary_count": int(upper_flags.sum()),
                    "upper_boundary_rate": upper_rate,
                    "pool_boundary_rate": float(pool_rates[dimension]),
                    "pool_lower_boundary_rate": float(pool_lower_rates[dimension]),
                    "pool_upper_boundary_rate": float(pool_upper_rates[dimension]),
                    "enrichment_ratio_vs_pool": (
                        rate / float(pool_rates[dimension])
                        if pool_rates[dimension] > 0
                        else np.nan
                    ),
                    "lower_enrichment_ratio_vs_pool": (
                        lower_rate / float(pool_lower_rates[dimension])
                        if pool_lower_rates[dimension] > 0
                        else np.nan
                    ),
                    "upper_enrichment_ratio_vs_pool": (
                        upper_rate / float(pool_upper_rates[dimension])
                        if pool_upper_rates[dimension] > 0
                        else np.nan
                    ),
                }
            )
    return _stamp_frame(pd.DataFrame(rows))


def _model_validation_frames(
    validation_results: Sequence[ModelValidationResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    fit_records: list[Any] = []
    for result in validation_results:
        summary_rows.extend(asdict(metric) for metric in result.loocv.metrics)
        prediction_rows.extend(asdict(row) for row in result.loocv.predictions)
        hyperparameter_rows.extend(row.as_flat_dict() for row in result.hyperparameters)
        fit_records.extend([result.full_fit, *result.loocv.fold_records.values()])
    warnings_frame = fit_warnings_frame(fit_records)
    return (
        _stamp_frame(pd.DataFrame(summary_rows)),
        _stamp_frame(pd.DataFrame(prediction_rows)),
        _stamp_frame(pd.DataFrame(hyperparameter_rows)),
        _stamp_frame(warnings_frame),
    )


def _augment_model_summary_with_candidate_diagnostics(
    summary: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    training_y: np.ndarray,
    objective_bounds: Sequence[tuple[float | None, float | None]],
) -> pd.DataFrame:
    result = summary.copy()
    for objective_index, objective_name in enumerate(D2D_OBJECTIVE_NAMES):
        result.loc[result["objective_index"] == objective_index, "observed_minimum"] = (
            float(training_y[:, objective_index].min())
        )
        result.loc[result["objective_index"] == objective_index, "observed_maximum"] = (
            float(training_y[:, objective_index].max())
        )
        for variant in result["variant_name"].unique():
            selected = candidate_rows[
                (candidate_rows["run_id"] == "baseline")
                & (candidate_rows["model_variant"] == variant)
            ]
            if selected.empty:
                selected = candidate_rows[
                    candidate_rows["model_variant"] == variant
                ].head(STEP2C_BATCH_SIZE)
            values = selected[f"pred_mean_{objective_index}"].to_numpy(dtype=float)
            mask = (result["variant_name"] == variant) & (
                result["objective_index"] == objective_index
            )
            if values.size:
                result.loc[mask, "selected_prediction_minimum"] = float(values.min())
                result.loc[mask, "selected_prediction_maximum"] = float(values.max())
                result.loc[mask, "selected_prediction_outside_observed_count"] = int(
                    np.count_nonzero(
                        (values < training_y[:, objective_index].min())
                        | (values > training_y[:, objective_index].max())
                    )
                )
                _, _, outside_declared = _declared_bound_flags(
                    values, objective_bounds[objective_index]
                )
                result.loc[
                    mask, "selected_prediction_outside_declared_bounds_count"
                ] = int(np.count_nonzero(outside_declared))
            result.loc[mask, "objective_name_contract"] = objective_name
    return _stamp_frame(result)


def _hyperparameter_stability_summary(
    hyperparameters: pd.DataFrame,
) -> pd.DataFrame:
    parameter_columns = [
        "likelihood_noise",
        "outputscale",
        *[
            column
            for column in hyperparameters.columns
            if column.startswith("ard_lengthscale_")
            and not column.endswith(
                (
                    "_near_floor",
                    "_very_small_normalized_domain",
                    "_extremely_large_flat",
                )
            )
        ],
    ]
    rows: list[dict[str, Any]] = []
    for (variant, objective), group in hyperparameters.groupby(
        ["variant_name", "objective_name"], sort=True
    ):
        full = group[group["omitted_sample_id"].isna()]
        leave_one_out = group[group["omitted_sample_id"].notna()]
        if full.shape[0] != 1 or leave_one_out.empty:
            raise RuntimeError(
                "Hyperparameter stability requires one full fit and LOOCV fits."
            )
        for parameter in parameter_columns:
            values = leave_one_out[parameter].to_numpy(dtype=float)
            full_value = float(full.iloc[0][parameter])
            log_displacement = np.abs(np.log(values / full_value))
            rows.append(
                {
                    "variant_name": variant,
                    "objective_name": objective,
                    "parameter_name": parameter,
                    "full_fit_value": full_value,
                    "loocv_minimum": float(values.min()),
                    "loocv_q1": float(np.quantile(values, 0.25)),
                    "loocv_median": float(np.median(values)),
                    "loocv_q3": float(np.quantile(values, 0.75)),
                    "loocv_maximum": float(values.max()),
                    "loocv_maximum_absolute_log_ratio_vs_full": float(
                        log_displacement.max()
                    ),
                    "loocv_fit_count": int(values.size),
                }
            )
    return _stamp_frame(pd.DataFrame(rows))


def _analytic_mc_comparison(
    *,
    model: Any,
    pool: CandidatePool,
    training_y: np.ndarray,
    observed_norm: np.ndarray,
    config: ResolvedStep2CConfig,
    mode_settings: ExecutionModeSettings,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    comparison_count = min(
        pool.size,
        512 if mode_settings.nested_unique_sizes[-1] <= 4096 else 2048,
    )
    indices = pool.grid_indices[:comparison_count].copy()
    physical = pool.X_phys[:comparison_count].copy()
    normalized = pool.X_norm[:comparison_count].copy()
    comparison_pool = CandidatePool(
        grid_indices=indices,
        X_phys=physical,
        X_norm=normalized,
        seed=pool.seed,
        draws=comparison_count,
        rejected_duplicate=0,
        rejected_avoid=0,
        rejected_constraint=0,
    )
    transform = build_d2d_objective_transform()
    analytic_mean, analytic_std, _, analytic_runtime = _analytic_numpy_moments(
        model,
        normalized,
        objective_transform=transform,
        chunk_size=config.score_chunk_size,
    )
    mc_started = perf_counter()
    mc: PosteriorUtilityMoments = posterior_utility_moments(
        model,
        torch.as_tensor(normalized, dtype=torch.double),
        transform,
        mc_samples=mode_settings.mc_comparison_samples,
        seed=config.mc_comparison_seed,
        chunk_size=min(config.score_chunk_size, 512),
        observation_noise=False,
    )
    mc_runtime = perf_counter() - mc_started
    analytic_scoring, _ = _score_moments(
        analytic_mean,
        analytic_std,
        training_y,
        config,
        beta=config.primary_beta,
        bound_policy=config.primary_bound_policy,
    )
    mc_scoring = score_ucb_hvi_from_moments(
        mc.utility_mean,
        mc.utility_std,
        training_y,
        config.reference_point_utility,
        beta=config.primary_beta,
        numeric_tolerance=config.numeric_tolerance,
        chunk_size=2048,
        log_epsilon=1.0e-300,
        mc_samples=mc.mc_samples,
        seed=mc.seed,
        observation_noise=False,
        objective_contract_version=transform.version,
        moment_method="monte_carlo",
        bound_policy=config.primary_bound_policy,
        utility_bounds=config.objective_bounds,
    )
    penalty = _penalty_by_label(config, config.primary_penalty_variant)
    analytic_selection = _static_select(
        comparison_pool,
        analytic_scoring.base_score,
        q=STEP2C_BATCH_SIZE,
        penalty=penalty,
        observed_norm=observed_norm,
    )
    mc_selection = _static_select(
        comparison_pool,
        mc_scoring.base_score,
        q=STEP2C_BATCH_SIZE,
        penalty=penalty,
        observed_norm=observed_norm,
    )
    match = regional_match_batches(analytic_selection.X_norm, mc_selection.X_norm)
    mean_delta = np.abs(analytic_mean - mc.utility_mean)
    std_delta = np.abs(analytic_std - mc.utility_std)
    observed_ranges = np.ptp(np.asarray(training_y, dtype=float), axis=0)
    moment_difference_tolerances = np.maximum(
        0.02, 0.02 * np.maximum(observed_ranges, 1.0)
    )
    maximum_moment_difference_tolerance = float(moment_difference_tolerances.max())
    minimum_regional_matches = 4
    maximum_selection_mean_distance = 0.15
    moment_check_passed = bool(
        np.all(mean_delta.max(axis=0) <= moment_difference_tolerances)
        and np.all(std_delta.max(axis=0) <= moment_difference_tolerances)
    )
    selection_check_passed = bool(
        match.regional_match_count(0.15) >= minimum_regional_matches
        and match.mean_matched_distance <= maximum_selection_mean_distance
    )
    row: dict[str, Any] = {
        "comparison_candidate_count": comparison_count,
        "mc_samples": mode_settings.mc_comparison_samples,
        "mc_seed": config.mc_comparison_seed,
        "analytic_runtime_seconds": analytic_runtime,
        "mc_runtime_seconds": mc_runtime,
        "runtime_speedup_mc_over_analytic": (
            mc_runtime / analytic_runtime if analytic_runtime > 0 else np.nan
        ),
        "mean_absolute_mean_difference": float(mean_delta.mean()),
        "maximum_absolute_mean_difference": float(mean_delta.max()),
        "mean_absolute_std_difference": float(std_delta.mean()),
        "maximum_absolute_std_difference": float(std_delta.max()),
        "exact_selection_overlap": match.exact_overlap_count,
        "mean_selection_matched_distance": match.mean_matched_distance,
        "maximum_selection_matched_distance": match.maximum_matched_distance,
        "analytic_selection_indices": "|".join(
            str(value) for value in analytic_selection.selected_pool_indices
        ),
        "mc_selection_indices": "|".join(
            str(value) for value in mc_selection.selected_pool_indices
        ),
        "analytic_deterministic_without_mc_seed": True,
        "primary_step2c_moment_method": "analytic_identity",
        "maximum_moment_difference_tolerance": (maximum_moment_difference_tolerance),
        "moment_difference_tolerance_policy": (
            "per objective max(0.02, 0.02 * max(observed_range, 1.0))"
        ),
        "moment_difference_tolerances": "|".join(
            f"{value:.17g}" for value in moment_difference_tolerances
        ),
        "minimum_regional_matches_within_0_15": minimum_regional_matches,
        "maximum_selection_mean_distance_tolerance": (maximum_selection_mean_distance),
        "moment_comparison_passed": moment_check_passed,
        "selection_comparison_passed": selection_check_passed,
        "analytic_mc_debug_check_passed": (
            moment_check_passed and selection_check_passed
        ),
    }
    for threshold, count in match.regional_match_counts.items():
        row[f"regional_matches_within_{threshold:.2f}"] = count
    for objective, objective_name in enumerate(D2D_OBJECTIVE_NAMES):
        row[f"{objective_name}_maximum_absolute_mean_difference"] = float(
            mean_delta[:, objective].max()
        )
        row[f"{objective_name}_maximum_absolute_std_difference"] = float(
            std_delta[:, objective].max()
        )
        row[f"{objective_name}_moment_difference_tolerance"] = float(
            moment_difference_tolerances[objective]
        )
    payload = {
        "comparison_candidate_count": comparison_count,
        "mc_samples": mode_settings.mc_comparison_samples,
        "mean_absolute_mean_difference": row["mean_absolute_mean_difference"],
        "maximum_absolute_mean_difference": row["maximum_absolute_mean_difference"],
        "mean_absolute_std_difference": row["mean_absolute_std_difference"],
        "maximum_absolute_std_difference": row["maximum_absolute_std_difference"],
        "exact_selection_overlap": match.exact_overlap_count,
        "regional_matches_within_0_15": match.regional_match_count(0.15),
        "analytic_runtime_seconds": analytic_runtime,
        "mc_runtime_seconds": mc_runtime,
        "maximum_moment_difference_tolerance": (maximum_moment_difference_tolerance),
        "moment_difference_tolerance_policy": row["moment_difference_tolerance_policy"],
        "moment_difference_tolerances": {
            objective_name: float(moment_difference_tolerances[objective])
            for objective, objective_name in enumerate(D2D_OBJECTIVE_NAMES)
        },
        "minimum_regional_matches_within_0_15": minimum_regional_matches,
        "maximum_selection_mean_distance_tolerance": (maximum_selection_mean_distance),
        "moment_comparison_passed": moment_check_passed,
        "selection_comparison_passed": selection_check_passed,
        "analytic_mc_debug_check_passed": (
            moment_check_passed and selection_check_passed
        ),
    }
    return _stamp_frame(pd.DataFrame([row])), payload


def _pareto_sample_ids(Y: np.ndarray, sample_ids: Sequence[Any]) -> tuple[Any, ...]:
    from botorch.utils.multi_objective.pareto import is_non_dominated

    values = torch.as_tensor(Y, dtype=torch.double)
    mask = is_non_dominated(values).detach().cpu().numpy()
    ids = np.asarray(tuple(sample_ids), dtype=object)
    if ids.shape != (values.shape[0],):
        raise ValueError("sample_ids must align with objective rows.")
    return tuple(ids[mask].tolist())


def _hyperparameter_mapping(record: Any) -> dict[str, float]:
    rows = extract_model_hyperparameters(record, input_names=D2D_INPUT_COLUMNS)
    result: dict[str, float] = {}
    for row in rows:
        prefix = f"objective_{row.objective_index}"
        result[f"{prefix}.noise"] = row.likelihood_noise
        result[f"{prefix}.outputscale"] = row.outputscale
        for input_name, value in zip(row.input_names, row.ard_lengthscales):
            result[f"{prefix}.lengthscale.{input_name}"] = value
    if not result or any(
        not np.isfinite(value) or value <= 0 for value in result.values()
    ):
        raise RuntimeError("Influence hyperparameters must be finite and positive.")
    return result


def _influence_candidate_rows(
    batch: StudyBatch,
    *,
    omitted_sample_id: int | None,
    model: Any,
    model_training_y: np.ndarray,
    observed_y: np.ndarray,
    observed_norm: np.ndarray,
    config: ResolvedStep2CConfig,
) -> list[dict[str, Any]]:
    means, stds, _, _ = _analytic_numpy_moments(
        model,
        batch.X_norm,
        objective_transform=build_d2d_objective_transform(),
        chunk_size=config.score_chunk_size,
    )
    scoring, _ = _score_moments(
        means,
        stds,
        np.asarray(model_training_y, dtype=float),
        config,
        beta=config.primary_beta,
        bound_policy=config.primary_bound_policy,
    )
    observed_values = np.asarray(observed_y, dtype=float)
    observed_coordinates = np.asarray(observed_norm, dtype=float)
    nearest_observed = np.linalg.norm(
        batch.X_norm[:, None, :] - observed_coordinates[None, :, :], axis=-1
    ).min(axis=1)
    rows: list[dict[str, Any]] = []
    for index in range(STEP2C_BATCH_SIZE):
        row: dict[str, Any] = {
            "run_id": batch.run_id,
            "omitted_sample_id": omitted_sample_id,
            "candidate_id": f"{batch.run_id}-C{index + 1:02d}",
            "selection_order": index + 1,
            "common_pool_sha256": batch.pool_hash,
            "acquisition_score": float(batch.base_scores[index]),
            "penalized_acquisition_score": float(batch.penalized_scores[index]),
            "nearest_observed_distance": float(nearest_observed[index]),
            "lower_boundary_coordinate_count": int(
                np.count_nonzero(np.isclose(batch.X_norm[index], 0.0))
            ),
            "upper_boundary_coordinate_count": int(
                np.count_nonzero(np.isclose(batch.X_norm[index], 1.0))
            ),
            "boundary_coordinate_count": int(
                np.count_nonzero(
                    np.isclose(batch.X_norm[index], 0.0)
                    | np.isclose(batch.X_norm[index], 1.0)
                )
            ),
            "lower_boundary_dimensions": "|".join(
                name
                for name, value in zip(D2D_INPUT_COLUMNS, batch.X_norm[index])
                if np.isclose(value, 0.0)
            ),
            "upper_boundary_dimensions": "|".join(
                name
                for name, value in zip(D2D_INPUT_COLUMNS, batch.X_norm[index])
                if np.isclose(value, 1.0)
            ),
        }
        for dimension, name in enumerate(D2D_INPUT_COLUMNS):
            row[name] = float(batch.X_phys[index, dimension])
            row[f"grid_{dimension}"] = int(batch.grid_indices[index, dimension])
            row[f"norm_{dimension}"] = float(batch.X_norm[index, dimension])
        for objective, objective_name in enumerate(D2D_OBJECTIVE_NAMES):
            observed_minimum = float(observed_values[:, objective].min())
            observed_maximum = float(observed_values[:, objective].max())
            below, above, outside = _declared_bound_flags(
                means[index, objective], config.objective_bounds[objective]
            )
            lower, upper = config.objective_bounds[objective]
            row[f"pred_mean_{objective}"] = float(means[index, objective])
            row[f"pred_std_{objective}"] = float(stds[index, objective])
            row[f"observed_minimum_{objective}"] = observed_minimum
            row[f"observed_maximum_{objective}"] = observed_maximum
            row[f"pred_mean_below_observed_{objective}"] = bool(
                means[index, objective] < observed_minimum
            )
            row[f"pred_mean_above_observed_{objective}"] = bool(
                means[index, objective] > observed_maximum
            )
            row[f"pred_mean_below_declared_bounds_{objective}"] = bool(below)
            row[f"pred_mean_above_declared_bounds_{objective}"] = bool(above)
            row[f"pred_mean_outside_declared_bounds_{objective}"] = bool(outside)
            row[f"ucb_raw_{objective}"] = float(
                scoring.utility_ucb_raw[index, objective]
            )
            row[f"ucb_effective_{objective}"] = float(
                scoring.utility_ucb_effective[index, objective]
            )
            row[f"ucb_clip_amount_{objective}"] = float(
                scoring.utility_ucb_clip_amount[index, objective]
            )
            row[f"ucb_effective_outside_declared_bounds_{objective}"] = bool(
                (
                    lower is not None
                    and scoring.utility_ucb_effective[index, objective] < lower
                )
                or (
                    upper is not None
                    and scoring.utility_ucb_effective[index, objective] > upper
                )
            )
            row[f"objective_name_{objective}"] = objective_name
        rows.append(row)
    return rows


def _influence_boundary_summary(
    influence_batches: Sequence[StudyBatch],
) -> pd.DataFrame:
    """Compare full versus each omission's lower/upper boundary preference."""
    by_id = {batch.run_id: batch for batch in influence_batches}
    full = by_id.get("influence_full")
    if full is None:
        raise RuntimeError("Influence boundary summary requires influence_full.")
    omission_batches = sorted(
        (
            batch
            for batch in influence_batches
            if batch.run_id.startswith("influence_omit_")
        ),
        key=lambda batch: int(batch.run_id.rsplit("_", 1)[1]),
    )
    if not omission_batches:
        raise RuntimeError("Influence boundary summary requires omission batches.")

    rows: list[dict[str, Any]] = []
    for batch in omission_batches:
        row: dict[str, Any] = {"omitted_sample_id": int(batch.run_id.rsplit("_", 1)[1])}
        for dimension, name in enumerate(D2D_INPUT_COLUMNS):
            for prefix, source in (("full", full), ("omitted", batch)):
                lower = np.isclose(source.X_norm[:, dimension], 0.0)
                upper = np.isclose(source.X_norm[:, dimension], 1.0)
                boundary = lower | upper
                row[f"{prefix}_lower_boundary_count_{dimension}"] = int(lower.sum())
                row[f"{prefix}_lower_boundary_rate_{dimension}"] = float(lower.mean())
                row[f"{prefix}_upper_boundary_count_{dimension}"] = int(upper.sum())
                row[f"{prefix}_upper_boundary_rate_{dimension}"] = float(upper.mean())
                row[f"{prefix}_boundary_count_{dimension}"] = int(boundary.sum())
                row[f"{prefix}_boundary_rate_{dimension}"] = float(boundary.mean())
            row[f"lower_boundary_rate_change_{dimension}"] = (
                row[f"omitted_lower_boundary_rate_{dimension}"]
                - row[f"full_lower_boundary_rate_{dimension}"]
            )
            row[f"upper_boundary_rate_change_{dimension}"] = (
                row[f"omitted_upper_boundary_rate_{dimension}"]
                - row[f"full_upper_boundary_rate_{dimension}"]
            )
            row[f"boundary_rate_change_{dimension}"] = (
                row[f"omitted_boundary_rate_{dimension}"]
                - row[f"full_boundary_rate_{dimension}"]
            )
            row[f"input_name_{dimension}"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def _run_observation_influence(
    *,
    validation: ModelValidationResult,
    common_pool: CandidatePool,
    common_pool_hash: str,
    full_pool_scores: np.ndarray,
    config: ResolvedStep2CConfig,
    mode_settings: ExecutionModeSettings,
    training: Any,
) -> tuple[
    ObservationInfluenceStudyResult,
    list[StudyBatch],
    pd.DataFrame,
    pd.DataFrame,
]:
    full_grid_scorer = CachedGridScorer(
        _grid_score_function(
            validation.full_fit.model,
            config.design,
            validation.full_fit.train_Y.detach().cpu().numpy(),
            config,
            beta=config.primary_beta,
            bound_policy=config.primary_bound_policy,
        ),
        len(D2D_INPUT_COLUMNS),
    )
    full_grid_scorer.seed(common_pool.grid_indices, full_pool_scores)
    full_batch = _run_refined_batch(
        run_id="influence_full",
        run_family="observation_influence",
        core_run=False,
        pool=common_pool,
        pool_hash=common_pool_hash,
        master_base_scores=full_pool_scores,
        shared_grid_scorer=full_grid_scorer,
        config=config,
        mode_settings=mode_settings,
        training=training,
        model_variant="dim_scaled_prior",
        beta=config.primary_beta,
        bound_policy=config.primary_bound_policy,
        penalty_label=config.primary_penalty_variant,
        scoring=None,
    )
    transform = build_d2d_objective_transform()
    full_mean, full_std, _, _ = _analytic_numpy_moments(
        validation.full_fit.model,
        full_batch.X_norm,
        objective_transform=transform,
        chunk_size=config.score_chunk_size,
    )
    full_input = InfluenceRunInput(
        run_label="full",
        omitted_sample_id=None,
        common_pool_sha256=common_pool_hash,
        selected_X_norm=full_batch.X_norm,
        pool_acquisition_scores=full_pool_scores,
        prediction_mean_at_full_candidates=full_mean,
        prediction_std_at_full_candidates=full_std,
        pareto_sample_ids=_pareto_sample_ids(
            validation.full_fit.train_Y.detach().cpu().numpy(),
            validation.full_fit.sample_ids,
        ),
        hyperparameters=_hyperparameter_mapping(validation.full_fit),
        prediction_uncertainty_kind="latent",
    )
    omission_inputs: list[InfluenceRunInput] = []
    study_batches: list[StudyBatch] = [full_batch]
    candidate_rows = _influence_candidate_rows(
        full_batch,
        omitted_sample_id=None,
        model=validation.full_fit.model,
        model_training_y=validation.full_fit.train_Y.detach().cpu().numpy(),
        observed_y=training.Y_objectives,
        observed_norm=training.X_norm_all,
        config=config,
    )
    for sample_id in mode_settings.omitted_sample_ids:
        record = validation.loocv.fold_records[sample_id]
        means, stds, _, _ = _analytic_numpy_moments(
            record.model,
            common_pool.X_norm,
            objective_transform=transform,
            chunk_size=config.score_chunk_size,
        )
        scoring, _ = _score_moments(
            means,
            stds,
            record.train_Y.detach().cpu().numpy(),
            config,
            beta=config.primary_beta,
            bound_policy=config.primary_bound_policy,
        )
        omission_grid_scorer = CachedGridScorer(
            _grid_score_function(
                record.model,
                config.design,
                record.train_Y.detach().cpu().numpy(),
                config,
                beta=config.primary_beta,
                bound_policy=config.primary_bound_policy,
            ),
            len(D2D_INPUT_COLUMNS),
        )
        omission_grid_scorer.seed(common_pool.grid_indices, scoring.base_score)
        batch = _run_refined_batch(
            run_id=f"influence_omit_{int(sample_id):02d}",
            run_family="observation_influence",
            core_run=False,
            pool=common_pool,
            pool_hash=common_pool_hash,
            master_base_scores=scoring.base_score,
            shared_grid_scorer=omission_grid_scorer,
            config=config,
            mode_settings=mode_settings,
            training=training,
            model_variant="dim_scaled_prior",
            beta=config.primary_beta,
            bound_policy=config.primary_bound_policy,
            penalty_label=config.primary_penalty_variant,
            scoring=scoring,
        )
        prediction_mean, prediction_std, _, _ = _analytic_numpy_moments(
            record.model,
            full_batch.X_norm,
            objective_transform=transform,
            chunk_size=config.score_chunk_size,
        )
        omission_inputs.append(
            InfluenceRunInput(
                run_label=batch.run_id,
                omitted_sample_id=sample_id,
                common_pool_sha256=common_pool_hash,
                selected_X_norm=batch.X_norm,
                pool_acquisition_scores=scoring.base_score,
                prediction_mean_at_full_candidates=prediction_mean,
                prediction_std_at_full_candidates=prediction_std,
                pareto_sample_ids=_pareto_sample_ids(
                    record.train_Y.detach().cpu().numpy(), record.sample_ids
                ),
                hyperparameters=_hyperparameter_mapping(record),
                prediction_uncertainty_kind="latent",
            )
        )
        study_batches.append(batch)
        candidate_rows.extend(
            _influence_candidate_rows(
                batch,
                omitted_sample_id=sample_id,
                model=record.model,
                model_training_y=record.train_Y.detach().cpu().numpy(),
                observed_y=training.Y_objectives,
                observed_norm=training.X_norm_all,
                config=config,
            )
        )
    objective_scales = np.ptp(training.Y_objectives, axis=0)
    if np.any(objective_scales <= 0):
        raise RuntimeError("Influence normalization requires nonconstant objectives.")
    roles = {
        int(sample_id): str(role)
        for sample_id, role in zip(training.sample_ids, training.row_roles)
    }
    include_policy = {
        int(sample_id): bool(include)
        for sample_id, include in zip(training.sample_ids, training.include_in_model)
    }
    influence = run_observation_influence_study(
        full_input,
        omission_inputs,
        expected_common_pool_sha256=common_pool_hash,
        objective_names=D2D_OBJECTIVE_NAMES,
        objective_scales=objective_scales,
        row_roles=roles,
        primary_include_policy=include_policy,
        regional_thresholds=config.regional_thresholds,
        top_k=min(config.influence_top_k, common_pool.size),
        require_complete_coverage=(set(mode_settings.omitted_sample_ids) == set(roles)),
    )
    return (
        influence,
        study_batches,
        _stamp_frame(pd.DataFrame(candidate_rows)),
        _stamp_frame(influence.prediction_changes_frame()),
    )


def _augment_shortlist_model_and_influence_diagnostics(
    shortlist: pd.DataFrame,
    *,
    validations_by_name: Mapping[str, ModelValidationResult],
    config: ResolvedStep2CConfig,
    mode_settings: ExecutionModeSettings,
    training: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach model/policy reports and omission sensitivity at region medoids."""
    result = shortlist.copy().reset_index(drop=True)
    norm_columns = [f"medoid_norm_{index}" for index in range(len(D2D_INPUT_COLUMNS))]
    missing = sorted(set(norm_columns) - set(result.columns))
    if missing:
        raise RuntimeError(f"Shortlist is missing medoid coordinates: {missing}.")
    X_norm = result.loc[:, norm_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(X_norm)) or np.any(X_norm < 0.0) or np.any(X_norm > 1.0):
        raise RuntimeError("Shortlist medoids must be finite normalized coordinates.")
    lower_boundary_flags = np.isclose(X_norm, 0.0)
    upper_boundary_flags = np.isclose(X_norm, 1.0)
    result["lower_boundary_coordinate_count"] = lower_boundary_flags.sum(axis=1)
    result["upper_boundary_coordinate_count"] = upper_boundary_flags.sum(axis=1)
    result["lower_boundary_dimensions"] = [
        "|".join(name for name, flag in zip(D2D_INPUT_COLUMNS, candidate_flags) if flag)
        for candidate_flags in lower_boundary_flags
    ]
    result["upper_boundary_dimensions"] = [
        "|".join(name for name, flag in zip(D2D_INPUT_COLUMNS, candidate_flags) if flag)
        for candidate_flags in upper_boundary_flags
    ]

    transform = build_d2d_objective_transform()
    full_default_mean: np.ndarray | None = None
    full_default_std: np.ndarray | None = None
    for variant_name in config.model_variant_names:
        validation = validations_by_name[variant_name]
        means, stds, _, _ = _analytic_numpy_moments(
            validation.full_fit.model,
            X_norm,
            objective_transform=transform,
            chunk_size=config.score_chunk_size,
        )
        scoring, _ = _score_moments(
            means,
            stds,
            validation.full_fit.train_Y.detach().cpu().numpy(),
            config,
            beta=config.primary_beta,
            bound_policy=config.primary_bound_policy,
        )
        if variant_name == config.primary_model_variant:
            full_default_mean = means
            full_default_std = stds
        for objective, objective_name in enumerate(D2D_OBJECTIVE_NAMES):
            prefix = f"{variant_name}_{objective_name}"
            result[f"{prefix}_pred_mean"] = means[:, objective]
            result[f"{prefix}_pred_std"] = stds[:, objective]
            result[f"{prefix}_ucb_raw"] = scoring.utility_ucb_raw[:, objective]
            result[f"{prefix}_ucb_effective"] = scoring.utility_ucb_effective[
                :, objective
            ]
            result[f"{prefix}_ucb_clip_amount"] = scoring.utility_ucb_clip_amount[
                :, objective
            ]
            observed_minimum = float(training.Y_objectives[:, objective].min())
            observed_maximum = float(training.Y_objectives[:, objective].max())
            result[f"{prefix}_observed_minimum"] = observed_minimum
            result[f"{prefix}_observed_maximum"] = observed_maximum
            result[f"{prefix}_pred_mean_below_observed"] = (
                means[:, objective] < observed_minimum
            )
            result[f"{prefix}_pred_mean_above_observed"] = (
                means[:, objective] > observed_maximum
            )
            below_declared, above_declared, outside_declared = _declared_bound_flags(
                means[:, objective], config.objective_bounds[objective]
            )
            result[f"{prefix}_pred_mean_below_declared_bounds"] = below_declared
            result[f"{prefix}_pred_mean_above_declared_bounds"] = above_declared
            result[f"{prefix}_pred_mean_outside_declared_bounds"] = outside_declared

    if full_default_mean is None or full_default_std is None:
        raise RuntimeError("The primary model variant was not evaluated at medoids.")
    objective_scales = np.ptp(training.Y_objectives, axis=0)
    if np.any(objective_scales <= 0.0):
        raise RuntimeError("Shortlist influence requires nonconstant objectives.")

    per_omission_mean: list[np.ndarray] = []
    per_omission_std: list[np.ndarray] = []
    if len(config.control_sample_ids) != 1:
        raise RuntimeError("Step 2C requires exactly one configured control sample.")
    control_sample_id = int(config.control_sample_ids[0])
    control_omission_mean: np.ndarray | None = None
    control_omission_std: np.ndarray | None = None
    prediction_rows: list[dict[str, Any]] = []
    default_validation = validations_by_name[config.primary_model_variant]
    for sample_id in mode_settings.omitted_sample_ids:
        fit = default_validation.loocv.fold_records[sample_id]
        omitted_mean, omitted_std, _, _ = _analytic_numpy_moments(
            fit.model,
            X_norm,
            objective_transform=transform,
            chunk_size=config.score_chunk_size,
        )
        mean_delta = omitted_mean - full_default_mean
        std_delta = omitted_std - full_default_std
        normalized_mean = np.abs(mean_delta) / objective_scales[None, :]
        normalized_std = np.abs(std_delta) / objective_scales[None, :]
        per_omission_mean.append(normalized_mean)
        per_omission_std.append(normalized_std)
        if int(sample_id) == control_sample_id:
            control_omission_mean = normalized_mean
            control_omission_std = normalized_std
        for candidate_index, region_id in enumerate(result["region_id"].astype(str)):
            for objective, objective_name in enumerate(D2D_OBJECTIVE_NAMES):
                prediction_rows.append(
                    {
                        "omitted_sample_id": int(sample_id),
                        "candidate_index": candidate_index,
                        "objective_index": objective,
                        "objective_name": objective_name,
                        "full_mean": float(
                            full_default_mean[candidate_index, objective]
                        ),
                        "omitted_mean": float(omitted_mean[candidate_index, objective]),
                        "mean_delta": float(mean_delta[candidate_index, objective]),
                        "absolute_mean_delta": float(
                            abs(mean_delta[candidate_index, objective])
                        ),
                        "full_std": float(full_default_std[candidate_index, objective]),
                        "omitted_std": float(omitted_std[candidate_index, objective]),
                        "std_delta": float(std_delta[candidate_index, objective]),
                        "absolute_std_delta": float(
                            abs(std_delta[candidate_index, objective])
                        ),
                        "objective_scale": float(objective_scales[objective]),
                        "normalized_absolute_mean_delta": float(
                            normalized_mean[candidate_index, objective]
                        ),
                        "normalized_absolute_std_delta": float(
                            normalized_std[candidate_index, objective]
                        ),
                        "prediction_location_kind": "robust_region_medoid",
                        "location_id": region_id,
                    }
                )

    mean_stack = np.stack(per_omission_mean, axis=0)
    std_stack = np.stack(per_omission_std, axis=0)
    result["influence_omission_count"] = len(mode_settings.omitted_sample_ids)
    result["influence_mean_normalized_prediction_mean_change"] = mean_stack.mean(
        axis=(0, 2)
    )
    result["influence_max_normalized_prediction_mean_change"] = mean_stack.max(
        axis=(0, 2)
    )
    result["influence_mean_normalized_prediction_std_change"] = std_stack.mean(
        axis=(0, 2)
    )
    result["influence_max_normalized_prediction_std_change"] = std_stack.max(
        axis=(0, 2)
    )
    if control_omission_mean is not None and control_omission_std is not None:
        result["control_omission_mean_normalized_prediction_mean_change"] = (
            control_omission_mean.mean(axis=1)
        )
        result["control_omission_mean_normalized_prediction_std_change"] = (
            control_omission_std.mean(axis=1)
        )
    nearest_observed = np.linalg.norm(
        X_norm[:, None, :] - training.X_norm_all[None, :, :], axis=-1
    ).min(axis=1)
    result["nearest_observed_distance"] = nearest_observed
    boundary_flags = np.isclose(X_norm, 0.0) | np.isclose(X_norm, 1.0)
    result["boundary_coordinate_count"] = boundary_flags.sum(axis=1)
    result["boundary_dimensions"] = [
        "|".join(
            name for name, is_boundary in zip(D2D_INPUT_COLUMNS, flags) if is_boundary
        )
        for flags in boundary_flags
    ]
    return _stamp_frame(result), _stamp_frame(pd.DataFrame(prediction_rows))


def _augment_regions_with_control_influence_correspondence(
    regions: pd.DataFrame,
    *,
    influence_batches: Sequence[StudyBatch],
    control_sample_id: int,
    threshold: float,
) -> pd.DataFrame:
    by_id = {batch.run_id: batch for batch in influence_batches}
    omit_control_run_id = f"influence_omit_{int(control_sample_id):02d}"
    if "influence_full" not in by_id or omit_control_run_id not in by_id:
        raise RuntimeError(
            "Control correspondence requires full and omit-control runs."
        )
    medoid_columns = sorted(
        (column for column in regions if column.startswith("medoid_norm_")),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    medoids = regions.loc[:, medoid_columns].to_numpy(dtype=float)
    full = by_id["influence_full"].X_norm
    omit_control = by_id[omit_control_run_id].X_norm
    full_distance = np.linalg.norm(medoids[:, None, :] - full[None, :, :], axis=-1).min(
        axis=1
    )
    omit_distance = np.linalg.norm(
        medoids[:, None, :] - omit_control[None, :, :], axis=-1
    ).min(axis=1)
    result = regions.copy()
    result["control_sample_id"] = int(control_sample_id)
    result["control_influence_correspondence_threshold"] = float(threshold)
    result["full_model_batch_minimum_distance"] = full_distance
    result["omit_control_batch_minimum_distance"] = omit_distance
    result["full_model_batch_region_hit"] = full_distance <= float(threshold)
    result["omit_control_batch_region_hit"] = omit_distance <= float(threshold)
    result["control_included_vs_omit_control_correspondence"] = (
        result["full_model_batch_region_hit"] & result["omit_control_batch_region_hit"]
    )
    result["control_omit_correspondence_available"] = True
    result["control_omit_correspondence_definition"] = (
        "both_full_and_omit_control_batches_within_region_threshold"
    )
    return _stamp_frame(result)


def _require_debug_stamps(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Fail closed if an exported table has lost its debug-only contract."""
    required = {
        "debug_only",
        "approved_for_experiment",
        "approved_for_production",
        "candidate_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Artifact table {name!r} is missing stamps: {missing}.")
    if not frame.empty:
        if not frame["debug_only"].astype(bool).all():
            raise RuntimeError(f"Artifact table {name!r} is not entirely debug-only.")
        if frame["approved_for_experiment"].astype(bool).any():
            raise RuntimeError(
                f"Artifact table {name!r} contains experimental approval."
            )
        if frame["approved_for_production"].astype(bool).any():
            raise RuntimeError(f"Artifact table {name!r} contains production approval.")
        if not frame["candidate_status"].eq(D2D_DEBUG_WATERMARK).all():
            raise RuntimeError(f"Artifact table {name!r} lost the watermark.")
    return frame


def _bounded_plot_frame(scoring: UCBHVIScoreResult) -> pd.DataFrame:
    raw = np.asarray(scoring.utility_ucb_raw, dtype=float)
    effective = np.asarray(scoring.utility_ucb_effective, dtype=float)
    if raw.shape != effective.shape or raw.ndim != 2:
        raise RuntimeError("Bounded-utility plotting arrays must have shape (N, M).")
    if raw.shape[1] != len(D2D_OBJECTIVE_NAMES):
        raise RuntimeError("Bounded-utility plotting objective count changed.")
    rows = [
        {
            "policy": f"clip_ucb:{name}",
            "raw_value": float(raw[:, objective].mean()),
            "bounded_value": float(effective[:, objective].mean()),
        }
        for objective, name in enumerate(D2D_OBJECTIVE_NAMES)
    ]
    return pd.DataFrame(rows)


def _boundary_plot_frame(enrichment: pd.DataFrame) -> pd.DataFrame:
    required_groups = (
        "pool",
        "top_1_percent_acquisition",
        "selected_batch",
    )
    result: pd.DataFrame | None = None
    for side in ("lower", "upper"):
        pivot = enrichment.pivot(
            index="input_name",
            columns="group",
            values=f"{side}_boundary_rate",
        )
        missing = sorted(set(required_groups) - set(pivot.columns))
        if missing:
            raise RuntimeError(
                f"Boundary-enrichment plot is missing {side} groups: {missing}."
            )
        side_frame = pivot.loc[:, list(required_groups)].rename(
            columns={
                "pool": f"pool_{side}_boundary_frequency",
                "top_1_percent_acquisition": f"top_{side}_boundary_frequency",
                "selected_batch": f"selected_{side}_boundary_frequency",
            }
        )
        result = side_frame if result is None else result.join(side_frame)
    if result is None:
        raise RuntimeError("Boundary-enrichment plot has no lower/upper data.")
    return result.reset_index()


def _shortlist_prediction_plot_frame(shortlist: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, candidate in shortlist.iterrows():
        for variant_name in ("dim_scaled_prior", "conservative"):
            for objective_name in D2D_OBJECTIVE_NAMES:
                prefix = f"{variant_name}_{objective_name}"
                rows.append(
                    {
                        "candidate_id": candidate["shortlist_id"],
                        "model_variant": variant_name,
                        "objective_name": objective_name,
                        "predicted_mean": candidate[f"{prefix}_pred_mean"],
                        "predicted_std": candidate[f"{prefix}_pred_std"],
                        "observed_minimum": candidate[f"{prefix}_observed_minimum"],
                        "observed_maximum": candidate[f"{prefix}_observed_maximum"],
                    }
                )
    return pd.DataFrame(rows)


def _consensus_debug_frame(
    shortlist: pd.DataFrame,
    *,
    consensus_batch_size: int,
) -> pd.DataFrame:
    if shortlist.shape[0] < consensus_batch_size:
        raise RuntimeError("Consensus passed without enough shortlist rows.")
    result = shortlist.head(consensus_batch_size).copy().reset_index(drop=True)
    missing_inputs = sorted(set(D2D_INPUT_COLUMNS) - set(result.columns))
    if missing_inputs:
        raise RuntimeError(
            f"Consensus shortlist is missing physical inputs: {missing_inputs}."
        )
    result.insert(
        0,
        "consensus_candidate_id",
        [f"R1-CONSENSUS-DEBUG-{index:02d}" for index in range(1, len(result) + 1)],
    )
    result.insert(1, "selection_order", np.arange(1, len(result) + 1))
    result["consensus_basis"] = "robust_region_medoid"
    result["experimental_release_blocked"] = True
    return _stamp_frame(result)


def _build_private_evidence_zip_staging(
    source_dir: Path,
    temporary_zip: Path,
    *,
    archive_root_name: str,
    overwrite: bool,
) -> Path:
    """Build the complete local evidence ZIP with an unambiguous warning."""
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Private evidence ZIP source directory is missing: {source_dir}."
        )
    if (
        not archive_root_name
        or Path(archive_root_name).name != archive_root_name
        or archive_root_name in {".", ".."}
    ):
        raise ValueError("archive_root_name must be one safe path component.")
    if temporary_zip.exists():
        if not overwrite:
            raise FileExistsError(
                f"Private evidence ZIP staging path exists: {temporary_zip}."
            )
        temporary_zip.unlink()
    try:
        with zipfile.ZipFile(
            temporary_zip, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(
                f"{archive_root_name}/PRIVATE_EVIDENCE_DO_NOT_SHARE.txt",
                (
                    "PRIVATE EVIDENCE - DO NOT SHARE\n\n"
                    "This archive contains the complete local Step 2C audit "
                    "surface, including private source provenance, local paths, "
                    "sample-level tables, and exact candidate recipes. It is not "
                    "a portable or public export. Use the sibling public summary "
                    "ZIP for sharing.\n"
                ),
            )
            for path in sorted(
                (item for item in source_dir.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(source_dir).as_posix(),
            ):
                archive.write(
                    path,
                    arcname=(
                        f"{archive_root_name}/"
                        f"{path.relative_to(source_dir).as_posix()}"
                    ),
                )
        with zipfile.ZipFile(temporary_zip, mode="r") as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Private evidence ZIP failed its integrity check.")
    except Exception:
        if temporary_zip.exists():
            temporary_zip.unlink()
        raise
    return temporary_zip


def _public_summary_manifest(
    private_manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> dict[str, Any]:
    input_data_kind = private_manifest.get("input_data_kind")
    if input_data_kind == "sanitized_synthetic_ci":
        public_data_kind = "generated_synthetic_dataset"
    elif input_data_kind == "private_pinned_workbook":
        public_data_kind = "private_campaign_dataset_redacted"
    else:
        raise RuntimeError("Step 2C public export received an unknown data kind.")
    return {
        "schema_version": PUBLIC_SUMMARY_SCHEMA_VERSION,
        "method_version": private_manifest["method_version"],
        "mode": private_manifest["mode"],
        "input_data_kind": public_data_kind,
        "source_dataset": public_data_kind,
        "public_export": True,
        "debug_only": True,
        "approved_for_experiment": False,
        "approved_for_production": False,
        "candidate_status": D2D_DEBUG_WATERMARK,
        "contains_candidate_recipes": False,
        "contains_sample_level_data": False,
        "contains_local_paths": False,
        "full_private_evidence_included": False,
        "git_commit": private_manifest["git_commit"],
        "objective_order": private_manifest["objective_order"],
        "reference_point": private_manifest["reference_point"],
        "objective_bounds": private_manifest["objective_bounds"],
        "moment_method": private_manifest["moment_method"],
        "robust_region_count": private_manifest["robust_region_count"],
        "shortlist_count": private_manifest["shortlist_count"],
        "consensus_passed": private_manifest["consensus_passed"],
        "consensus_checks": private_manifest["consensus_checks"],
        "consensus_observed": private_manifest["consensus_observed"],
        "runtime_versions": private_manifest["runtime_versions"],
        "archive_root": PUBLIC_SUMMARY_ARCHIVE_ROOT,
        "included_files": {
            relative: hashlib.sha256(payload).hexdigest().upper()
            for relative, payload in sorted(payloads.items())
        },
    }


def _build_public_summary_zip_staging(
    source_dir: Path,
    temporary_zip: Path,
    *,
    private_manifest: Mapping[str, Any],
    overwrite: bool,
) -> Path:
    """Build and validate a strict aggregate-only public summary ZIP."""
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Public summary ZIP source directory is missing: {source_dir}."
        )
    if temporary_zip.exists():
        if not overwrite:
            raise FileExistsError(
                f"Public summary ZIP staging path exists: {temporary_zip}."
            )
        temporary_zip.unlink()

    payloads: dict[str, bytes] = {
        PUBLIC_SUMMARY_DEBUG_MARKER_FILE: (D2D_DEBUG_WATERMARK + "\n").encode("utf-8"),
        PUBLIC_SUMMARY_README_FILE: (
            "MOBO-Kit D2D Step 2C sanitized public summary\n\n"
            "This debug-only export contains allowlisted aggregate model and "
            "search diagnostics. Candidate recipes, row-level measurements, "
            "source provenance, local paths, and the complete evidence surface "
            "are intentionally excluded. This summary cannot independently "
            "validate the private source data and is not approved for experiment "
            "or production use.\n"
        ).encode("utf-8"),
    }
    for relative in PUBLIC_SUMMARY_CSV_FILES:
        source = source_dir / relative
        if not source.is_file():
            raise FileNotFoundError(
                f"Public summary source table is missing: {relative}."
            )
        payloads[relative] = source.read_bytes()
    public_manifest = _public_summary_manifest(private_manifest, payloads)
    payloads[PUBLIC_SUMMARY_MANIFEST_FILE] = json.dumps(
        _jsonable(public_manifest), indent=2, sort_keys=True
    ).encode("utf-8")

    try:
        with zipfile.ZipFile(
            temporary_zip, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for relative, payload in sorted(payloads.items()):
                archive.writestr(f"{PUBLIC_SUMMARY_ARCHIVE_ROOT}/{relative}", payload)
        validate_step2c_public_summary_archive(temporary_zip)
    except Exception:
        if temporary_zip.exists():
            temporary_zip.unlink()
        raise
    return temporary_zip


def _remove_publication_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _publish_validated_bundle_transaction(
    staging: Path,
    destination: Path,
    backup: Path,
    *,
    zip_staging: Path,
    zip_path: Path,
    zip_backup: Path,
    overwrite: bool,
    publish_zip: bool,
    validate_published: Callable[[], Any],
    private_zip_staging: Path | None = None,
    private_zip_path: Path | None = None,
    private_zip_backup: Path | None = None,
    publish_private_zip: bool = False,
    validate_public_zip: Callable[[], Any] | None = None,
) -> Any:
    """Publish local evidence and public/private ZIPs as one transaction."""
    private_paths = (private_zip_staging, private_zip_path, private_zip_backup)
    private_paths_supplied = tuple(path is not None for path in private_paths)
    if any(private_paths_supplied) and not all(private_paths_supplied):
        raise ValueError("Private evidence ZIP publication paths are all-or-none.")
    if publish_private_zip and not all(private_paths_supplied):
        raise ValueError("Private evidence ZIP paths are required for publication.")
    private_enabled = all(private_paths_supplied)
    if not staging.is_dir():
        raise FileNotFoundError(f"Step 2C staging directory is missing: {staging}.")
    if (
        backup.exists()
        or zip_backup.exists()
        or (
            private_enabled
            and private_zip_backup is not None
            and private_zip_backup.exists()
        )
    ):
        raise FileExistsError("Step 2C publication backup paths must not pre-exist.")
    if publish_zip and not zip_staging.is_file():
        raise FileNotFoundError(
            f"Public summary ZIP staging file is missing: {zip_staging}."
        )
    if (
        publish_private_zip
        and private_zip_staging is not None
        and not private_zip_staging.is_file()
    ):
        raise FileNotFoundError(
            f"Private evidence ZIP staging file is missing: {private_zip_staging}."
        )
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(
                f"Step 2C output destination is not a directory: {destination}."
            )
        if any(destination.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Step 2C output directory is not empty: {destination}."
            )
    if zip_path.exists() and not overwrite:
        raise FileExistsError(f"Public summary ZIP already exists: {zip_path}.")
    if (
        private_enabled
        and private_zip_path is not None
        and private_zip_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"Private evidence ZIP already exists: {private_zip_path}."
        )

    original_directory_moved = False
    original_zip_moved = False
    original_private_zip_moved = False
    new_directory_published = False
    new_zip_published = False
    new_private_zip_published = False
    try:
        if destination.exists():
            destination.rename(backup)
            original_directory_moved = True
        if zip_path.exists():
            zip_path.rename(zip_backup)
            original_zip_moved = True
        if (
            private_enabled
            and private_zip_path is not None
            and private_zip_path.exists()
        ):
            assert private_zip_backup is not None
            private_zip_path.rename(private_zip_backup)
            original_private_zip_moved = True
        staging.rename(destination)
        new_directory_published = True
        if publish_zip:
            os.replace(zip_staging, zip_path)
            new_zip_published = True
        if publish_private_zip:
            assert private_zip_staging is not None
            assert private_zip_path is not None
            os.replace(private_zip_staging, private_zip_path)
            new_private_zip_published = True
        validated = validate_published()
        if publish_zip and validate_public_zip is not None:
            validate_public_zip()
    except Exception:
        if new_private_zip_published and private_zip_path is not None:
            _remove_publication_path(private_zip_path)
        if new_zip_published:
            _remove_publication_path(zip_path)
        if new_directory_published:
            _remove_publication_path(destination)
        if original_directory_moved and backup.exists() and not destination.exists():
            backup.rename(destination)
        if original_zip_moved and zip_backup.exists() and not zip_path.exists():
            zip_backup.rename(zip_path)
        if (
            original_private_zip_moved
            and private_zip_backup is not None
            and private_zip_path is not None
            and private_zip_backup.exists()
            and not private_zip_path.exists()
        ):
            private_zip_backup.rename(private_zip_path)
        raise
    finally:
        if zip_staging.exists():
            _remove_publication_path(zip_staging)
        if private_zip_staging is not None and private_zip_staging.exists():
            _remove_publication_path(private_zip_staging)
    if original_directory_moved:
        _remove_publication_path(backup)
    if original_zip_moved:
        _remove_publication_path(zip_backup)
    if original_private_zip_moved and private_zip_backup is not None:
        _remove_publication_path(private_zip_backup)
    return validated


def _write_step2c_bundle(
    *,
    destination: Path,
    config: ResolvedStep2CConfig,
    workbook: Path,
    workbook_audit: dict[str, Any],
    tables: Mapping[str, pd.DataFrame],
    manifest: dict[str, Any],
    consensus: ConsensusCriteriaResult,
    shortlist: pd.DataFrame,
    consensus_candidates: pd.DataFrame,
    loocv_predictions: pd.DataFrame,
    model_hyperparameters: pd.DataFrame,
    nested_convergence: pd.DataFrame,
    bounded_scoring: UCBHVIScoreResult,
    penalty_tradeoff: pd.DataFrame,
    influence_summary: pd.DataFrame,
    influence_candidates: pd.DataFrame,
    influence_predictions: pd.DataFrame,
    boundary_enrichment: pd.DataFrame,
    robust_regions: pd.DataFrame,
    robust_membership: pd.DataFrame,
    create_portable_zip: bool,
    overwrite: bool,
    total_started: float,
) -> dict[str, str]:
    """Write, validate, then atomically publish a private Step 2C bundle."""
    repository_root = Path(__file__).resolve().parents[2]
    allowed_root = (repository_root / config.output_root).resolve()
    if allowed_root not in destination.parents:
        raise RuntimeError("Step 2C bundle escaped the configured ignored root.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging")
    backup = destination.with_name(f".{destination.name}.backup")
    # The legacy sibling ``<run>.zip`` path is retained for callers, but now
    # contains only the sanitized public summary. Full private evidence receives
    # an explicit filename and is produced only for the pinned campaign source.
    zip_path = destination.with_suffix(".zip")
    zip_staging = zip_path.with_name(f".{zip_path.name}.staging")
    zip_backup = zip_path.with_name(f".{zip_path.name}.backup")
    private_zip_path = destination.with_name(
        f"{destination.name}_PRIVATE_EVIDENCE_DO_NOT_SHARE.zip"
    )
    private_zip_staging = private_zip_path.with_name(
        f".{private_zip_path.name}.staging"
    )
    private_zip_backup = private_zip_path.with_name(f".{private_zip_path.name}.backup")
    publish_private_evidence = (
        create_portable_zip
        and manifest.get("input_data_kind") == "private_pinned_workbook"
    )
    for internal in (
        staging,
        backup,
        zip_staging,
        zip_backup,
        private_zip_staging,
        private_zip_backup,
    ):
        if internal.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Internal Step 2C publication path exists: {internal}."
                )
            if internal.is_dir() and not internal.is_symlink():
                shutil.rmtree(internal)
            else:
                internal.unlink()

    if zip_path.exists() and not overwrite:
        raise FileExistsError(f"Public summary ZIP already exists: {zip_path}.")
    if private_zip_path.exists() and not overwrite:
        raise FileExistsError(
            f"Private evidence ZIP already exists: {private_zip_path}."
        )

    staging.mkdir(parents=False, exist_ok=False)
    try:
        (staging / "DEBUG_ONLY_NOT_APPROVED_FOR_EXPERIMENT.txt").write_text(
            D2D_DEBUG_WATERMARK + "\n",
            encoding="utf-8",
        )
        (staging / "resolved_debug_config.yaml").write_text(
            yaml.safe_dump(config.raw, sort_keys=False), encoding="utf-8"
        )
        for name, frame in sorted(tables.items()):
            if Path(name).name != name or not name.endswith(".csv"):
                raise RuntimeError(f"Unsafe Step 2C table artifact name: {name!r}.")
            _require_debug_stamps(frame, name=name).to_csv(staging / name, index=False)

        if consensus.passed:
            consensus_frame = _consensus_debug_frame(
                consensus_candidates,
                consensus_batch_size=config.consensus_batch_size,
            )
            consensus_frame.to_csv(
                staging / "r1_consensus_debug_batch.csv", index=False
            )
        else:
            _safe_write_json(
                staging / "r1_no_stable_batch_reason.json",
                {
                    "consensus_passed": False,
                    "failed_checks": list(consensus.reasons),
                    "checks": consensus.checks,
                    "observed": consensus.observed,
                    "message": (
                        "No stable R1 batch is released. Review the failed robustness "
                        "criteria before any experimental decision."
                    ),
                },
            )

        plot_loocv_diagnostics(
            loocv_predictions,
            staging / "plots/model_validation/loocv_diagnostics.png",
            objective_column="objective_name",
            std_column="predictive_std",
        )
        plot_ard_lengthscale_comparison(
            model_hyperparameters,
            staging / "plots/model_validation/ard_lengthscale_comparison.png",
        )
        plot_candidate_predictions_vs_observed_ranges(
            _shortlist_prediction_plot_frame(shortlist),
            staging
            / "plots/model_validation/candidate_predictions_vs_observed_ranges.png",
        )
        plot_nested_search_convergence(
            nested_convergence,
            staging / "plots/search_convergence/nested_pool_convergence.png",
            pool_size_column="comparison_pool_size",
            regional_columns={
                0.10: "regional_matches_within_0.10",
                0.15: "regional_matches_within_0.15",
                0.20: "regional_matches_within_0.20",
            },
        )
        plot_bounded_utility_comparison(
            _bounded_plot_frame(bounded_scoring),
            staging / "plots/bounded_utility/bounded_utility_comparison.png",
        )
        plot_local_penalty_tradeoff(
            penalty_tradeoff,
            staging / "plots/local_penalty/local_penalty_tradeoff.png",
            variant_column="penalty_label",
            acquisition_column="comparison_acquisition_sum",
        )
        plot_observation_influence_ranking(
            influence_summary,
            staging / "plots/influence/observation_influence_ranking.png",
            sample_column="omitted_sample_id",
            score_column="composite_influence_score",
        )
        influence_norm_columns = sorted(
            (
                column
                for column in influence_candidates.columns
                if column.startswith("norm_") and column[5:].isdigit()
            ),
            key=lambda value: int(value.split("_", 1)[1]),
        )
        full_influence_rows = influence_candidates[
            influence_candidates["omitted_sample_id"].isna()
        ]
        if len(config.control_sample_ids) != 1:
            raise RuntimeError(
                "Step 2C requires exactly one configured control sample."
            )
        control_sample_id = int(config.control_sample_ids[0])
        omit_control_rows = influence_candidates[
            pd.to_numeric(
                influence_candidates["omitted_sample_id"], errors="coerce"
            ).eq(control_sample_id)
        ]
        plot_control_omission_candidate_region_comparison(
            full_influence_rows.loc[:, influence_norm_columns].to_numpy(dtype=float),
            omit_control_rows.loc[:, influence_norm_columns].to_numpy(dtype=float),
            staging / "plots/influence/full_vs_omit_control_candidates.png",
        )
        plot_shortlist_region_influence_sensitivity(
            influence_predictions[
                influence_predictions["prediction_location_kind"].eq(
                    "robust_region_medoid"
                )
            ],
            staging / "plots/influence/shortlist_region_influence_sensitivity.png",
            region_column="location_id",
            sensitivity_column="normalized_absolute_mean_delta",
        )
        plot_boundary_enrichment(
            _boundary_plot_frame(boundary_enrichment),
            staging / "plots/search_convergence/boundary_enrichment.png",
        )
        norm_columns = sorted(
            (
                column
                for column in robust_membership.columns
                if column.startswith("norm_") and column[5:].isdigit()
            ),
            key=lambda value: int(value.split("_", 1)[1]),
        )
        plot_robust_region_overview(
            robust_membership.loc[:, norm_columns].to_numpy(dtype=float),
            robust_membership["region_id"].astype(str).tolist(),
            staging / "plots/robust_regions/robust_region_overview.png",
            input_names=D2D_INPUT_COLUMNS,
        )
        plot_shortlist_medoid_parallel_coordinates(
            shortlist,
            staging / "plots/robust_regions/shortlist_parallel_coordinates.png",
            input_names=D2D_INPUT_COLUMNS,
        )
        plot_run_region_persistence_heatmap(
            robust_membership,
            staging / "plots/robust_regions/run_region_persistence_heatmap.png",
        )
        plot_model_policy_region_correspondence(
            robust_membership,
            staging / "plots/robust_regions/model_policy_region_correspondence.png",
        )
        plot_acquisition_quality_vs_persistence(
            robust_regions,
            staging / "plots/robust_regions/acquisition_quality_vs_persistence.png",
        )

        workbook_hash_after = sha256_file(workbook)
        workbook_mtime_after = workbook.stat().st_mtime_ns
        if (
            workbook_hash_after != manifest["workbook_sha256_before"]
            or workbook_mtime_after != manifest["workbook_mtime_ns_before"]
        ):
            raise RuntimeError(
                "The private workbook changed while exporting artifacts."
            )
        manifest.update(
            {
                "workbook_sha256_after": workbook_hash_after,
                "workbook_mtime_ns_after": workbook_mtime_after,
                "source_workbook_modified": False,
                "output_directory": str(destination),
                "public_summary_archive_requested": create_portable_zip,
                "private_evidence_archive_requested": publish_private_evidence,
                "runtime_seconds_total": perf_counter() - total_started,
            }
        )
        workbook_audit.update(
            {
                "workbook_sha256_after": workbook_hash_after,
                "workbook_mtime_ns_after": workbook_mtime_after,
                "source_workbook_modified": False,
            }
        )
        _safe_write_json(staging / "workbook_audit.json", workbook_audit)
        _safe_write_json(staging / "run_manifest.json", manifest)

        validate_step2c_artifact_bundle(staging, repository_root=repository_root)

        if create_portable_zip:
            _build_public_summary_zip_staging(
                staging,
                zip_staging,
                private_manifest=manifest,
                overwrite=overwrite,
            )
        if publish_private_evidence:
            _build_private_evidence_zip_staging(
                staging,
                private_zip_staging,
                archive_root_name=("MOBO_Kit_Step2C_PRIVATE_EVIDENCE_DO_NOT_SHARE"),
                overwrite=overwrite,
            )
        validated = _publish_validated_bundle_transaction(
            staging,
            destination,
            backup,
            zip_staging=zip_staging,
            zip_path=zip_path,
            zip_backup=zip_backup,
            overwrite=overwrite,
            publish_zip=create_portable_zip,
            validate_published=lambda: validate_step2c_artifact_bundle(
                destination, repository_root=repository_root
            ),
            private_zip_staging=private_zip_staging,
            private_zip_path=private_zip_path,
            private_zip_backup=private_zip_backup,
            publish_private_zip=publish_private_evidence,
            validate_public_zip=(
                (lambda: validate_step2c_public_summary_archive(zip_path))
                if create_portable_zip
                else None
            ),
        )
        return validated.artifact_sha256
    except Exception:
        if staging.exists():
            _remove_publication_path(staging)
        if zip_staging.exists():
            _remove_publication_path(zip_staging)
        if private_zip_staging.exists():
            _remove_publication_path(private_zip_staging)
        raise


def _run_d2d_step2c_robustness_resolved(
    workbook_path: str | Path,
    config: ResolvedStep2CConfig,
    output_dir: str | Path,
    *,
    mode: str = "full",
    overwrite: bool = False,
    create_portable_zip: bool | None = None,
    input_data_kind: str = "private_pinned_workbook",
) -> Step2CRobustnessResult:
    """Run Step 2C from an already resolved, fail-closed input contract."""
    total_started = perf_counter()
    phase_started = total_started
    phase_runtimes: dict[str, float] = {}
    repository_root = Path(__file__).resolve().parents[2]
    mode_settings = config.mode(mode)
    if input_data_kind not in {
        "private_pinned_workbook",
        "sanitized_synthetic_ci",
    }:
        raise ValueError("input_data_kind is not an approved Step 2C source kind.")
    workbook = Path(workbook_path).resolve()
    expected_workbook = (repository_root / config.workbook_relative_path).resolve()
    if workbook != expected_workbook:
        raise ValueError(
            f"Step 2C may read only the pinned private workbook {expected_workbook}."
        )
    destination = _validate_output_destination(Path(output_dir), config)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean.")
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"Step 2C output directory is not empty: {destination}.")
    create_archives = (
        config.create_portable_zip
        if create_portable_zip is None
        else create_portable_zip
    )
    if not isinstance(create_archives, bool):
        raise ValueError("create_portable_zip must be a boolean or None.")
    source_hash_before = sha256_file(workbook)
    source_mtime_before = workbook.stat().st_mtime_ns
    if source_hash_before != config.workbook_expected_sha256:
        raise ValueError(
            "Workbook hash differs from the Step 2C contract: "
            f"expected={config.workbook_expected_sha256}, actual={source_hash_before}."
        )
    git_commit, git_dirty, git_status = _git_state(repository_root)

    frame, audit = load_d2d_workbook_frame(
        workbook,
        expected_sample_ids=config.expected_sample_ids,
        allowed_input_exceptions=config.off_grid_exceptions,
    )
    validation = validate_supplied_d2d_scores(frame)
    validation.raise_for_errors()
    training = prepare_d2d_training_data(frame, config, include_control=True)
    if np.count_nonzero(training.include_in_model) != 15:
        raise RuntimeError(
            "The Step 2C primary model must include all 15 observations."
        )
    if not validation.known_uniformity_score_mismatch:
        raise RuntimeError(
            "The pinned workbook was expected to retain the known Uniformity mismatch."
        )
    phase_runtimes["ingestion_and_validation"] = perf_counter() - phase_started

    phase_started = perf_counter()
    train_X = torch.as_tensor(training.X_norm_all, dtype=torch.double, device="cpu")
    train_Y = torch.as_tensor(training.Y_objectives, dtype=torch.double, device="cpu")
    fit_cache = ModelFitCache()
    validation_results = [
        validate_model_variant(
            train_X,
            train_Y,
            sample_ids=training.sample_ids.tolist(),
            input_names=D2D_INPUT_COLUMNS,
            objective_names=D2D_OBJECTIVE_NAMES,
            variant=variant,
            seed=config.model_seed,
            row_roles=training.row_roles,
            control_sample_ids=config.control_sample_ids,
            cache=fit_cache,
        )
        for variant in (DIM_SCALED_PRIOR, CONSERVATIVE)
    ]
    validations_by_name = {result.variant.name: result for result in validation_results}
    models = {
        name: result.full_fit.model for name, result in validations_by_name.items()
    }
    phase_runtimes["model_validation"] = perf_counter() - phase_started

    phase_started = perf_counter()
    primary_sobol = build_nested_sobol_discrete_pool(
        config.design,
        mode_settings.nested_unique_sizes,
        scramble_seed=config.primary_sobol_seed,
        observed_phys=training.X_phys_all,
        row_constraints=[],
    )
    secondary_sobol = [
        build_nested_sobol_discrete_pool(
            config.design,
            (mode_settings.nested_unique_sizes[-1],),
            scramble_seed=seed,
            observed_phys=training.X_phys_all,
            row_constraints=[],
        )
        for seed in config.secondary_sobol_seeds
    ]
    phase_runtimes["nested_sobol_pools"] = perf_counter() - phase_started

    phase_started = perf_counter()
    transform = build_d2d_objective_transform()
    largest_size = mode_settings.nested_unique_sizes[-1]
    largest_pool = primary_sobol.pools[largest_size]
    primary_mean, primary_std, primary_analytic, primary_moment_runtime = (
        _analytic_numpy_moments(
            models["dim_scaled_prior"],
            largest_pool.X_norm,
            objective_transform=transform,
            chunk_size=config.score_chunk_size,
        )
    )
    scoring_by_policy: dict[tuple[float, str], UCBHVIScoreResult] = {}
    scoring_runtime: dict[str, float] = {}
    for beta, bound_policy in (
        (config.primary_beta, config.primary_bound_policy),
        (1.0, config.primary_bound_policy),
        (9.0, config.primary_bound_policy),
        (config.primary_beta, "none"),
    ):
        scored, runtime = _score_moments(
            primary_mean,
            primary_std,
            training.Y_objectives,
            config,
            beta=beta,
            bound_policy=bound_policy,
        )
        scoring_by_policy[(beta, bound_policy)] = scored
        scoring_runtime[f"default_beta_{beta:g}_{bound_policy}"] = runtime
    baseline_scoring = scoring_by_policy[
        (config.primary_beta, config.primary_bound_policy)
    ]
    analytic_mc_frame, analytic_mc_payload = _analytic_mc_comparison(
        model=models["dim_scaled_prior"],
        pool=largest_pool,
        training_y=training.Y_objectives,
        observed_norm=training.X_norm_all,
        config=config,
        mode_settings=mode_settings,
    )
    phase_runtimes["primary_moments_and_scores"] = perf_counter() - phase_started

    phase_started = perf_counter()
    dimension = len(D2D_INPUT_COLUMNS)
    primary_scorers: dict[tuple[float, str], CachedGridScorer] = {}
    for key, scoring in scoring_by_policy.items():
        beta, bound_policy = key
        cached = CachedGridScorer(
            _grid_score_function(
                models["dim_scaled_prior"],
                config.design,
                training.Y_objectives,
                config,
                beta=beta,
                bound_policy=bound_policy,
            ),
            dimension,
        )
        cached.seed(largest_pool.grid_indices, scoring.base_score)
        primary_scorers[key] = cached

    nested_batches: list[StudyBatch] = []
    nested_raw_batches: list[StudyBatch] = []
    for size in mode_settings.nested_unique_sizes:
        pool = primary_sobol.pools[size]
        run_id = "baseline" if size == largest_size else f"nested_{size}"
        raw_selection = _static_select(
            pool,
            baseline_scoring.base_score[:size],
            q=STEP2C_BATCH_SIZE,
            penalty=_penalty_by_label(config, config.primary_penalty_variant),
            observed_norm=training.X_norm_all,
        )
        nested_raw_batches.append(
            _study_batch_from_static(
                run_id=f"nested_raw_{size}",
                run_family="nested_pool_raw",
                core_run=False,
                selection=raw_selection,
                model_variant="dim_scaled_prior",
                pool=pool,
                pool_hash=primary_sobol.prefix_hashes[size],
                beta=config.primary_beta,
                bound_policy=config.primary_bound_policy,
                penalty_label=config.primary_penalty_variant,
                scoring=baseline_scoring,
            )
        )
        nested_batches.append(
            _run_refined_batch(
                run_id=run_id,
                run_family="baseline" if size == largest_size else "nested_pool",
                core_run=True,
                pool=pool,
                pool_hash=primary_sobol.prefix_hashes[size],
                master_base_scores=baseline_scoring.base_score[:size],
                shared_grid_scorer=primary_scorers[
                    (config.primary_beta, config.primary_bound_policy)
                ],
                config=config,
                mode_settings=mode_settings,
                training=training,
                model_variant="dim_scaled_prior",
                beta=config.primary_beta,
                bound_policy=config.primary_bound_policy,
                penalty_label=config.primary_penalty_variant,
                scoring=baseline_scoring,
            )
        )
    baseline_batch = nested_batches[-1]
    phase_runtimes["nested_refinement"] = perf_counter() - phase_started

    phase_started = perf_counter()
    secondary_batches: list[StudyBatch] = []
    secondary_moment_runtime: dict[str, float] = {}
    for result in secondary_sobol:
        pool = result.largest_pool
        mean, std, _, moment_runtime = _analytic_numpy_moments(
            models["dim_scaled_prior"],
            pool.X_norm,
            objective_transform=transform,
            chunk_size=config.score_chunk_size,
        )
        scored, runtime = _score_moments(
            mean,
            std,
            training.Y_objectives,
            config,
            beta=config.primary_beta,
            bound_policy=config.primary_bound_policy,
        )
        secondary_moment_runtime[str(result.scramble_seed)] = moment_runtime + runtime
        scorer = CachedGridScorer(
            _grid_score_function(
                models["dim_scaled_prior"],
                config.design,
                training.Y_objectives,
                config,
                beta=config.primary_beta,
                bound_policy=config.primary_bound_policy,
            ),
            dimension,
        )
        scorer.seed(pool.grid_indices, scored.base_score)
        secondary_batches.append(
            _run_refined_batch(
                run_id=f"sobol_seed_{result.scramble_seed}",
                run_family="sobol_scramble",
                core_run=True,
                pool=pool,
                pool_hash=result.prefix_hashes[largest_size],
                master_base_scores=scored.base_score,
                shared_grid_scorer=scorer,
                config=config,
                mode_settings=mode_settings,
                training=training,
                model_variant="dim_scaled_prior",
                beta=config.primary_beta,
                bound_policy=config.primary_bound_policy,
                penalty_label=config.primary_penalty_variant,
                scoring=scored,
            )
        )

    conservative_mean, conservative_std, _, conservative_moment_runtime = (
        _analytic_numpy_moments(
            models["conservative"],
            largest_pool.X_norm,
            objective_transform=transform,
            chunk_size=config.score_chunk_size,
        )
    )
    conservative_scoring, conservative_score_runtime = _score_moments(
        conservative_mean,
        conservative_std,
        training.Y_objectives,
        config,
        beta=config.primary_beta,
        bound_policy=config.primary_bound_policy,
    )
    conservative_scorer = CachedGridScorer(
        _grid_score_function(
            models["conservative"],
            config.design,
            training.Y_objectives,
            config,
            beta=config.primary_beta,
            bound_policy=config.primary_bound_policy,
        ),
        dimension,
    )
    conservative_scorer.seed(largest_pool.grid_indices, conservative_scoring.base_score)
    conservative_batch = _run_refined_batch(
        run_id="model_conservative",
        run_family="model_variant",
        core_run=True,
        pool=largest_pool,
        pool_hash=primary_sobol.prefix_hashes[largest_size],
        master_base_scores=conservative_scoring.base_score,
        shared_grid_scorer=conservative_scorer,
        config=config,
        mode_settings=mode_settings,
        training=training,
        model_variant="conservative",
        beta=config.primary_beta,
        bound_policy=config.primary_bound_policy,
        penalty_label=config.primary_penalty_variant,
        scoring=conservative_scoring,
    )

    bound_none_scoring = scoring_by_policy[(config.primary_beta, "none")]
    bound_none_batch = _run_refined_batch(
        run_id="bound_none",
        run_family="bounded_utility",
        core_run=True,
        pool=largest_pool,
        pool_hash=primary_sobol.prefix_hashes[largest_size],
        master_base_scores=bound_none_scoring.base_score,
        shared_grid_scorer=primary_scorers[(config.primary_beta, "none")],
        config=config,
        mode_settings=mode_settings,
        training=training,
        model_variant="dim_scaled_prior",
        beta=config.primary_beta,
        bound_policy="none",
        penalty_label=config.primary_penalty_variant,
        scoring=bound_none_scoring,
    )
    beta_batches: list[StudyBatch] = []
    for beta in (1.0, 9.0):
        scored = scoring_by_policy[(beta, config.primary_bound_policy)]
        beta_batches.append(
            _run_refined_batch(
                run_id=f"beta_{int(beta)}",
                run_family="beta",
                core_run=True,
                pool=largest_pool,
                pool_hash=primary_sobol.prefix_hashes[largest_size],
                master_base_scores=scored.base_score,
                shared_grid_scorer=primary_scorers[(beta, config.primary_bound_policy)],
                config=config,
                mode_settings=mode_settings,
                training=training,
                model_variant="dim_scaled_prior",
                beta=beta,
                bound_policy=config.primary_bound_policy,
                penalty_label=config.primary_penalty_variant,
                scoring=scored,
            )
        )
    phase_runtimes["secondary_model_bound_beta_studies"] = (
        perf_counter() - phase_started
    )

    phase_started = perf_counter()
    baseline_optima = np.asarray(
        [anchor.refined_grid_index for anchor in baseline_batch.refinement.anchors],
        dtype=np.int64,
    )
    converged_pool = _pool_with_rows(largest_pool, baseline_optima, config.design)
    converged_pool_hash = _canonical_json_sha256(converged_pool.grid_indices.tolist())
    converged_scores = primary_scorers[
        (config.primary_beta, config.primary_bound_policy)
    ](converged_pool.grid_indices)
    penalty_batches: list[StudyBatch] = []
    penalty_core_labels = {
        "no_soft_hard_0_15",
        "radius_0_15",
        "radius_0_35",
    }
    for penalty in config.penalty_variants:
        penalty_selection_started = perf_counter()
        selection = _static_select(
            converged_pool,
            converged_scores,
            q=STEP2C_BATCH_SIZE,
            penalty=penalty,
            observed_norm=training.X_norm_all,
        )
        penalty_selection_runtime = perf_counter() - penalty_selection_started
        penalty_batches.append(
            _study_batch_from_static(
                run_id=f"penalty_{penalty.label}",
                run_family="local_penalty",
                core_run=penalty.label in penalty_core_labels,
                selection=selection,
                model_variant="dim_scaled_prior",
                pool=converged_pool,
                pool_hash=converged_pool_hash,
                beta=config.primary_beta,
                bound_policy=config.primary_bound_policy,
                penalty_label=penalty.label,
                scoring=baseline_scoring,
                selection_runtime_seconds=penalty_selection_runtime,
            )
        )
    phase_runtimes["local_penalty_isolation"] = perf_counter() - phase_started

    core_batches = [
        *nested_batches,
        *secondary_batches,
        conservative_batch,
        bound_none_batch,
        *beta_batches,
        *[batch for batch in penalty_batches if batch.core_run],
    ]
    if len(core_batches) != 13 or len({batch.run_id for batch in core_batches}) != 13:
        raise RuntimeError(
            "The Step 2C core one-factor run registry must contain 13 runs."
        )
    all_batches_by_id = {batch.run_id: batch for batch in core_batches}
    for batch in penalty_batches:
        all_batches_by_id.setdefault(batch.run_id, batch)
    all_batches = list(all_batches_by_id.values())

    phase_started = perf_counter()
    study_summary = _study_summary(all_batches)
    candidate_rows = _batch_candidate_rows(
        all_batches, models=models, config=config, training=training
    )
    nested_candidates = pd.concat(
        [
            candidate_rows[
                candidate_rows["run_id"].isin(
                    [batch.run_id for batch in nested_batches]
                )
            ],
            _batch_candidate_rows(
                nested_raw_batches,
                models=models,
                config=config,
                training=training,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    nested_candidates["search_stage"] = np.where(
        nested_candidates["run_id"].str.startswith("nested_raw_"),
        "raw_pool_selection",
        "locally_refined_selection",
    )
    refinement_trace = _refinement_trace_frame(all_batches)
    nested_convergence = _nested_convergence_table(nested_batches, nested_raw_batches)
    nested_convergence["shared_largest_pool_moment_runtime_seconds"] = (
        primary_moment_runtime
    )
    nested_convergence["shared_largest_pool_hvi_runtime_seconds"] = scoring_runtime[
        f"default_beta_{config.primary_beta:g}_{config.primary_bound_policy}"
    ]
    nested_convergence["prefix_scores_reused_from_exact_largest_prefix"] = True
    scramble_comparison = _sobol_scramble_table(baseline_batch, secondary_batches)
    bounded_comparison = _bounded_utility_table(
        baseline_batch,
        bound_none_batch,
        baseline_scoring,
        candidate_rows,
    )
    penalty_tradeoff = _penalty_tradeoff_table(
        penalty_batches, observed_norm=training.X_norm_all
    )
    beta_robustness = _beta_robustness_table(baseline_batch, beta_batches)
    model_summary, loocv_predictions, model_hyperparameters, model_fit_warnings = (
        _model_validation_frames(validation_results)
    )
    model_summary = _augment_model_summary_with_candidate_diagnostics(
        model_summary,
        candidate_rows,
        training.Y_objectives,
        config.objective_bounds,
    )
    hyperparameter_stability = _hyperparameter_stability_summary(model_hyperparameters)
    phase_runtimes["diagnostic_tables"] = perf_counter() - phase_started

    phase_started = perf_counter()
    influence_size = (
        config.influence_pool_size
        if config.influence_pool_size in primary_sobol.pools
        else max(
            size
            for size in primary_sobol.accepted_sizes
            if size <= min(config.influence_pool_size, largest_size)
        )
    )
    common_pool = primary_sobol.pools[influence_size]
    common_pool_hash = primary_sobol.prefix_hashes[influence_size]
    influence, influence_batches, influence_candidates, influence_predictions = (
        _run_observation_influence(
            validation=validations_by_name["dim_scaled_prior"],
            common_pool=common_pool,
            common_pool_hash=common_pool_hash,
            full_pool_scores=baseline_scoring.base_score[:influence_size],
            config=config,
            mode_settings=mode_settings,
            training=training,
        )
    )
    influence_runtime_rows = []
    influence_batch_by_id = {
        int(batch.run_id.rsplit("_", 1)[1]): batch
        for batch in influence_batches
        if batch.run_id.startswith("influence_omit_")
    }
    for sample_id in mode_settings.omitted_sample_ids:
        fit = validations_by_name["dim_scaled_prior"].loocv.fold_records[sample_id]
        batch = influence_batch_by_id[int(sample_id)]
        influence_runtime_rows.append(
            {
                "omitted_sample_id": int(sample_id),
                "fit_runtime_seconds": fit.fit_runtime_seconds,
                "fit_warning_count": len(fit.warnings),
                "proposal_runtime_seconds": batch.proposal_runtime_seconds,
                "refinement_anchor_count": len(batch.refinement.anchors),
                "refinement_trace_row_count": len(batch.refinement.trace),
            }
        )
    influence_boundary = _influence_boundary_summary(influence_batches)
    influence_summary = _stamp_frame(
        influence.summary_frame()
        .merge(
            pd.DataFrame(influence_runtime_rows),
            on="omitted_sample_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            influence_boundary,
            on="omitted_sample_id",
            how="left",
            validate="one_to_one",
        )
    )
    boundary_enrichment = _boundary_enrichment_table(
        largest_pool,
        baseline_scoring.base_score,
        baseline_batch,
        [*core_batches, *influence_batches],
    )
    phase_runtimes["observation_influence"] = perf_counter() - phase_started

    phase_started = perf_counter()
    core_candidate_rows = candidate_rows[
        candidate_rows["run_id"].isin([batch.run_id for batch in core_batches])
    ].reset_index(drop=True)
    core_registry = {batch.run_id: batch.run_family for batch in core_batches}
    region_result: RobustRegionResult = cluster_candidate_regions(
        core_candidate_rows,
        distance_threshold=config.region_threshold,
        core_run_registry=core_registry,
    )
    region_result = RobustRegionResult(
        regions=_augment_regions_with_control_influence_correspondence(
            region_result.regions,
            influence_batches=influence_batches,
            control_sample_id=int(config.control_sample_ids[0]),
            threshold=config.region_threshold,
        ),
        membership=region_result.membership,
        threshold=region_result.threshold,
        region_count=region_result.region_count,
    )
    region_sensitivity = {
        f"threshold_{threshold:.2f}": cluster_candidate_regions(
            core_candidate_rows,
            distance_threshold=threshold,
            core_run_registry=core_registry,
        ).region_count
        for threshold in config.region_sensitivity_thresholds
    }
    robust_regions = region_result.regions.copy()
    robust_membership = region_result.membership.copy()
    shortlist = select_robust_shortlist(
        region_result,
        minimum_count=config.shortlist_min,
        maximum_count=config.shortlist_max,
        minimum_normalized_distance=0.15,
    )
    for dimension, name in enumerate(D2D_INPUT_COLUMNS):
        source = f"medoid_phys_{dimension}"
        if source in shortlist:
            shortlist[name] = shortlist[source]
    shortlist, medoid_influence_predictions = (
        _augment_shortlist_model_and_influence_diagnostics(
            shortlist,
            validations_by_name=validations_by_name,
            config=config,
            mode_settings=mode_settings,
            training=training,
        )
    )
    full_batch_predictions = influence_predictions.copy()
    full_batch_predictions["prediction_location_kind"] = "full_model_batch"
    full_batch_predictions["location_id"] = [
        f"FULL-C{int(index) + 1:02d}"
        for index in full_batch_predictions["candidate_index"]
    ]
    influence_predictions = _stamp_frame(
        pd.concat(
            [full_batch_predictions, medoid_influence_predictions],
            ignore_index=True,
            sort=False,
        )
    )
    medoid_influence_summary = (
        medoid_influence_predictions.groupby("omitted_sample_id", sort=True)
        .agg(
            medoid_mean_normalized_prediction_mean_change=(
                "normalized_absolute_mean_delta",
                "mean",
            ),
            medoid_max_normalized_prediction_mean_change=(
                "normalized_absolute_mean_delta",
                "max",
            ),
            medoid_mean_normalized_prediction_std_change=(
                "normalized_absolute_std_delta",
                "mean",
            ),
            medoid_max_normalized_prediction_std_change=(
                "normalized_absolute_std_delta",
                "max",
            ),
        )
        .reset_index()
    )
    influence_summary = _stamp_frame(
        influence_summary.drop(
            columns=[
                "debug_only",
                "approved_for_experiment",
                "approved_for_production",
                "candidate_status",
            ]
        ).merge(
            medoid_influence_summary,
            on="omitted_sample_id",
            how="left",
            validate="one_to_one",
        )
    )
    family_count_column = (
        "distinct_nonbaseline_family_count"
        if "distinct_nonbaseline_family_count" in shortlist
        else "distinct_family_count"
    )
    consensus_candidates = shortlist[
        (
            pd.to_numeric(shortlist[family_count_column], errors="coerce")
            >= config.consensus_family_coverage_min
        )
        & shortlist["all_grid_valid"].astype(bool)
        & shortlist["all_hard_distance_valid"].astype(bool)
    ].head(config.consensus_batch_size)
    largest_match: RegionalBatchComparison = regional_match_batches(
        nested_batches[-2].X_norm,
        nested_batches[-1].X_norm,
        thresholds=config.regional_thresholds,
    )
    consensus = evaluate_consensus_criteria(
        robust_regions,
        consensus_candidates,
        largest_two_regional_matches_within_0_15=(
            largest_match.regional_match_count(0.15)
        ),
        largest_two_mean_matched_distance=largest_match.mean_matched_distance,
        nested_match_minimum=config.consensus_nested_match_min,
        mean_distance_maximum=config.consensus_mean_distance_max,
        minimum_family_coverage=config.consensus_family_coverage_min,
        consensus_batch_size=config.consensus_batch_size,
        required_minimum_distance=0.15,
        full_mode_eligible=(
            mode == "full"
            and mode_settings.nested_unique_sizes == config.nested_pool_sizes
            and mode_settings.omitted_sample_ids == config.influence_sample_ids
        ),
    )
    phase_runtimes["robust_regions_and_consensus"] = perf_counter() - phase_started

    source_hash_after_compute = sha256_file(workbook)
    source_mtime_after_compute = workbook.stat().st_mtime_ns
    if (
        source_hash_after_compute != source_hash_before
        or source_mtime_after_compute != source_mtime_before
    ):
        raise RuntimeError("The private workbook changed during Step 2C computation.")

    pool_manifest = _nested_pool_manifest(
        [primary_sobol, *secondary_sobol],
        role_by_seed={
            config.primary_sobol_seed: "primary_nested",
            **{seed: "secondary_scramble" for seed in config.secondary_sobol_seeds},
        },
    )
    training_manifest = _stamp_frame(training.manifest_frame())
    score_validation = _stamp_frame(validation.frame)
    workbook_audit = asdict(audit)
    workbook_audit.update(
        {
            "workbook_path": str(workbook),
            "workbook_sha256_before": source_hash_before,
            "workbook_sha256_after_compute": source_hash_after_compute,
            "workbook_mtime_ns_before": source_mtime_before,
            "workbook_mtime_ns_after_compute": source_mtime_after_compute,
            "source_workbook_modified": False,
            "known_uniformity_score_mismatch": True,
            "control_off_grid_exception_retained": True,
        }
    )
    if len(config.control_sample_ids) != 1:
        raise RuntimeError("Step 2C requires exactly one configured control sample.")
    control_sample_id = int(config.control_sample_ids[0])
    control_influence = influence.rank_for_sample(control_sample_id)
    penalty_classifications = dict(
        zip(
            penalty_tradeoff["penalty_label"],
            penalty_tradeoff["penalty_activity_classification"],
        )
    )
    classification_priority = {
        (
            "implemented but effectively inactive because base optima are already "
            "separated"
        ): 0,
        "active but only modestly changes diversity": 1,
        "active and materially changes diversity": 2,
    }
    soft_penalty_classifications = {
        label: value
        for label, value in penalty_classifications.items()
        if str(label).startswith("radius_")
    }
    if not soft_penalty_classifications:
        raise RuntimeError("The local soft-penalty study produced no radius variants.")
    overall_penalty_interpretation = max(
        soft_penalty_classifications.values(),
        key=lambda value: classification_priority[str(value)],
    )
    hard_spacing_interpretation = penalty_classifications["no_soft_hard_0_15"]
    manifest: dict[str, Any] = {
        "schema_version": "d2d-step2c-robustness-run-v1",
        "method_version": STEP2C_METHOD_VERSION,
        "mode": mode,
        "input_data_kind": input_data_kind,
        "debug_only": True,
        "approved_for_experiment": False,
        "approved_for_production": False,
        "candidate_status": D2D_DEBUG_WATERMARK,
        "real_r2_proposal_generated": False,
        "workbook_writeback_performed": False,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_status_at_start": git_status,
        "step2b_checkpoint": "1c6a83a9dfd7e6ed5e69ce66765b8dc0fd8a86af",
        "workbook_path": str(workbook),
        "workbook_sha256_before": source_hash_before,
        "workbook_sha256_after": source_hash_after_compute,
        "workbook_mtime_ns_before": source_mtime_before,
        "workbook_mtime_ns_after": source_mtime_after_compute,
        "source_workbook_modified": False,
        "config_path": str(config.config_path),
        "config_sha256": config.config_sha256,
        "resolved_config_hash": config.resolved_config_hash,
        "objective_order": D2D_OBJECTIVE_NAMES,
        "objective_source_columns": D2D_OBJECTIVE_COLUMNS,
        "reference_point": config.reference_point_utility,
        "objective_bounds": config.objective_bounds,
        "moment_method": "analytic_identity",
        "analytic_mc_comparison": analytic_mc_payload,
        "primary_analytic_moment_runtime_seconds": primary_moment_runtime,
        "scoring_runtime_seconds": scoring_runtime,
        "secondary_scoring_runtime_seconds": secondary_moment_runtime,
        "conservative_moment_runtime_seconds": conservative_moment_runtime,
        "conservative_score_runtime_seconds": conservative_score_runtime,
        "sobol_seeds": [config.primary_sobol_seed, *config.secondary_sobol_seeds],
        "nested_pool_sizes": mode_settings.nested_unique_sizes,
        "pool_prefix_hashes": {
            str(result.scramble_seed): result.prefix_hashes
            for result in [primary_sobol, *secondary_sobol]
        },
        "local_refinement": {
            "anchors_per_selection_step": mode_settings.anchors_per_selection_step,
            "max_sweeps": config.refinement_max_sweeps,
            "improvement_tolerance": config.refinement_tolerance,
            "coordinate_values": "all_allowed_grid_values",
        },
        "beta_values": config.beta_values,
        "bound_policies": config.bound_policies,
        "local_penalty_variants": [asdict(value) for value in config.penalty_variants],
        "local_penalty_activity_by_variant": penalty_classifications,
        "local_penalty_overall_interpretation": overall_penalty_interpretation,
        "hard_spacing_interpretation": hard_spacing_interpretation,
        "model_variants": [asdict(result.variant) for result in validation_results],
        "influence_common_pool_hash": common_pool_hash,
        "influence_common_pool_size": influence_size,
        "influence_omitted_sample_ids": mode_settings.omitted_sample_ids,
        "control_sample_id": control_sample_id,
        "control_influence_rank": control_influence.influence_rank,
        "control_influence_percentile": control_influence.influence_percentile,
        "robust_region_clustering": "agglomerative_complete_link",
        "robust_region_threshold": config.region_threshold,
        "robust_region_sensitivity_counts": region_sensitivity,
        "robust_region_count": region_result.region_count,
        "shortlist_count": int(shortlist.shape[0]),
        "consensus_passed": consensus.passed,
        "consensus_checks": consensus.checks,
        "consensus_observed": consensus.observed,
        "stability_criteria": {
            "consensus_batch_size": config.consensus_batch_size,
            "regional_match_threshold": 0.15,
            "largest_two_nested_regional_match_minimum": (
                config.consensus_nested_match_min
            ),
            "largest_two_nested_mean_matched_distance_maximum": (
                config.consensus_mean_distance_max
            ),
            "minimum_nonbaseline_core_family_coverage": (
                config.consensus_family_coverage_min
            ),
            "required_pairwise_minimum_distance": 0.15,
            "require_finite_and_bounded": True,
            "require_unique_and_on_grid": True,
            "require_debug_only_and_approval_false": True,
        },
        "known_uniformity_score_mismatch": True,
        "uniformity_warning_count": validation.uniformity_warning_count,
        "control_assumption": config.control_measurement_provenance_assumption,
        "off_grid_control_exception": [
            asdict(value) for value in config.off_grid_exceptions
        ],
        "runtime_versions": _runtime_versions(),
        "hardware": _hardware_summary(),
        "phase_runtime_seconds": phase_runtimes,
        "runtime_seconds_before_output": perf_counter() - total_started,
    }
    tables: dict[str, pd.DataFrame] = {
        "score_validation.csv": score_validation,
        "training_row_manifest.csv": training_manifest,
        "model_validation_summary.csv": model_summary,
        "loocv_predictions_long.csv": loocv_predictions,
        "model_hyperparameters.csv": model_hyperparameters,
        "model_hyperparameter_stability_summary.csv": hyperparameter_stability,
        "model_fit_warnings.csv": model_fit_warnings,
        "nested_pool_manifest.csv": pool_manifest,
        "nested_pool_convergence_summary.csv": nested_convergence,
        "nested_pool_candidates_long.csv": _stamp_frame(nested_candidates),
        "sobol_scramble_comparison.csv": scramble_comparison,
        "local_refinement_trace.csv": refinement_trace,
        "bounded_utility_comparison.csv": bounded_comparison,
        "bounded_utility_candidates_long.csv": _stamp_frame(
            robust_membership[
                robust_membership["run_id"].isin(
                    [baseline_batch.run_id, bound_none_batch.run_id]
                )
            ].reset_index(drop=True)
        ),
        "local_penalty_tradeoff.csv": penalty_tradeoff,
        "beta_robustness.csv": beta_robustness,
        "boundary_enrichment.csv": boundary_enrichment,
        "observation_influence_summary.csv": influence_summary,
        "observation_influence_candidates_long.csv": influence_candidates,
        "observation_influence_predictions.csv": influence_predictions,
        "robust_regions.csv": _stamp_frame(robust_regions),
        "r1_robust_shortlist_debug.csv": _stamp_frame(shortlist),
        "study_run_summary.csv": study_summary,
        "study_candidates_long.csv": candidate_rows,
        "robust_region_membership.csv": _stamp_frame(robust_membership),
        "analytic_mc_comparison.csv": analytic_mc_frame,
    }
    artifact_hashes = _write_step2c_bundle(
        destination=destination,
        config=config,
        workbook=workbook,
        workbook_audit=workbook_audit,
        tables=tables,
        manifest=manifest,
        consensus=consensus,
        shortlist=shortlist,
        consensus_candidates=consensus_candidates,
        loocv_predictions=loocv_predictions,
        model_hyperparameters=model_hyperparameters,
        nested_convergence=nested_convergence,
        bounded_scoring=baseline_scoring,
        penalty_tradeoff=penalty_tradeoff,
        influence_summary=influence_summary,
        influence_candidates=influence_candidates,
        influence_predictions=influence_predictions,
        boundary_enrichment=boundary_enrichment,
        robust_regions=robust_regions,
        robust_membership=robust_membership,
        create_portable_zip=create_archives,
        overwrite=overwrite,
        total_started=total_started,
    )
    source_hash_after = sha256_file(workbook)
    source_mtime_after = workbook.stat().st_mtime_ns
    if (
        source_hash_after != source_hash_before
        or source_mtime_after != source_mtime_before
    ):
        raise RuntimeError(
            "The private workbook changed while writing Step 2C outputs."
        )
    return Step2CRobustnessResult(
        output_dir=destination,
        mode=mode,
        run_manifest=manifest,
        consensus=consensus,
        robust_regions=tables["robust_regions.csv"],
        shortlist=tables["r1_robust_shortlist_debug.csv"],
        influence_summary=influence_summary,
        artifact_hashes=artifact_hashes,
    )


def run_d2d_step2c_robustness(
    workbook_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    *,
    mode: str = "full",
    overwrite: bool = False,
    create_portable_zip: bool | None = None,
) -> Step2CRobustnessResult:
    """Run the private-workbook Step 2C audit through its pinned contract.

    The public campaign command always resolves the tracked fail-closed config
    itself.  Sanitized CI exercises use a separate helper and cannot redirect this
    command to an alternate workbook.
    """
    config = load_step2c_config(config_path)
    return _run_d2d_step2c_robustness_resolved(
        workbook_path,
        config,
        output_dir,
        mode=mode,
        overwrite=overwrite,
        create_portable_zip=create_portable_zip,
        input_data_kind="private_pinned_workbook",
    )


__all__ = [
    "STEP2C_METHOD_VERSION",
    "Step2CRobustnessResult",
    "StudyBatch",
    "_run_d2d_step2c_robustness_resolved",
    "run_d2d_step2c_robustness",
]
