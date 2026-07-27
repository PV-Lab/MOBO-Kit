"""Pure all-observation influence metrics for Step 2C robustness studies.

The module consumes proposals, common-pool acquisition scores, posterior reports,
Pareto membership, and hyperparameters that were computed elsewhere.  It never
fits a model and never changes row roles or primary include-in-model policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from numbers import Real
from types import MappingProxyType
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .batch_comparison import (
    DEFAULT_REGIONAL_THRESHOLDS,
    RegionalBatchComparison,
    regional_match_batches,
)


INFLUENCE_COMPONENT_NAMES = (
    "batch_displacement",
    "prediction_change",
    "acquisition_change",
    "pareto_change",
    "hyperparameter_change",
)


def _sample_sort_key(value: Hashable) -> tuple[int, str, Any]:
    """Order common scalar Sample IDs naturally and all other IDs stably."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return 0, "", int(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return 1, "", float(value)
    if isinstance(value, str):
        return 2, "", value
    return 3, type(value).__name__, repr(value)


def _readonly_array(
    value: Any,
    *,
    name: str,
    dimensions: int,
    finite: bool = True,
) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    if result.ndim != dimensions:
        raise ValueError(
            f"{name} must be {dimensions}-dimensional; got {result.shape}."
        )
    if any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must not contain an empty dimension.")
    if finite and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex string.")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex string.")
    return normalized


def _positive_scales(value: Sequence[float], *, count: int) -> np.ndarray:
    scales = np.asarray(value, dtype=float).copy()
    if scales.shape != (count,) or not np.all(np.isfinite(scales)):
        raise ValueError(f"objective_scales must contain {count} finite values.")
    if np.any(scales <= 0.0):
        raise ValueError("objective_scales must be strictly positive.")
    scales.setflags(write=False)
    return scales


def _immutable_numeric_mapping(
    value: Mapping[str, float], *, name: str
) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping.")
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings.")
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, Real):
            raise ValueError(f"{name} values must be real non-boolean numbers.")
        number = float(raw)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} values must be finite and strictly positive.")
        result[key.strip()] = number
    if len(result) != len(value):
        raise ValueError(f"{name} keys must be unique after trimming.")
    return MappingProxyType(result)


@dataclass(frozen=True)
class InfluenceRunInput:
    """Already-computed arrays for one full or leave-one-out proposal run."""

    run_label: str
    omitted_sample_id: Hashable | None
    common_pool_sha256: str
    selected_X_norm: np.ndarray
    pool_acquisition_scores: np.ndarray
    prediction_mean_at_full_candidates: np.ndarray
    prediction_std_at_full_candidates: np.ndarray
    pareto_sample_ids: tuple[Hashable, ...]
    hyperparameters: Mapping[str, float]
    prediction_uncertainty_kind: str = "latent"

    def __post_init__(self) -> None:
        if not isinstance(self.run_label, str) or not self.run_label.strip():
            raise ValueError("run_label must be a non-empty string.")
        uncertainty_kind = self.prediction_uncertainty_kind
        if uncertainty_kind not in {"latent", "predictive"}:
            raise ValueError(
                "prediction_uncertainty_kind must be 'latent' or 'predictive'."
            )
        selected = _readonly_array(
            self.selected_X_norm, name="selected_X_norm", dimensions=2
        )
        if np.any(selected < 0.0) or np.any(selected > 1.0):
            raise ValueError("selected_X_norm must lie within [0, 1].")
        if np.unique(selected, axis=0).shape[0] != selected.shape[0]:
            raise ValueError("selected_X_norm must contain unique candidate rows.")
        scores = _readonly_array(
            self.pool_acquisition_scores,
            name="pool_acquisition_scores",
            dimensions=1,
        )
        means = _readonly_array(
            self.prediction_mean_at_full_candidates,
            name="prediction_mean_at_full_candidates",
            dimensions=2,
        )
        standard_deviations = _readonly_array(
            self.prediction_std_at_full_candidates,
            name="prediction_std_at_full_candidates",
            dimensions=2,
        )
        if means.shape != standard_deviations.shape:
            raise ValueError(
                "Prediction mean and standard deviation shapes must match."
            )
        if means.shape[0] != selected.shape[0]:
            raise ValueError(
                "Prediction rows must align with the full-candidate batch size."
            )
        if np.any(standard_deviations < 0.0):
            raise ValueError("Prediction standard deviations cannot be negative.")
        pareto_ids = tuple(self.pareto_sample_ids)
        if len(set(pareto_ids)) != len(pareto_ids):
            raise ValueError("pareto_sample_ids must be unique.")
        object.__setattr__(self, "run_label", self.run_label.strip())
        object.__setattr__(
            self,
            "common_pool_sha256",
            _sha256(self.common_pool_sha256, name="common_pool_sha256"),
        )
        object.__setattr__(self, "selected_X_norm", selected)
        object.__setattr__(self, "pool_acquisition_scores", scores)
        object.__setattr__(self, "prediction_mean_at_full_candidates", means)
        object.__setattr__(
            self, "prediction_std_at_full_candidates", standard_deviations
        )
        object.__setattr__(self, "pareto_sample_ids", pareto_ids)
        object.__setattr__(
            self,
            "hyperparameters",
            _immutable_numeric_mapping(self.hyperparameters, name="hyperparameters"),
        )


@dataclass(frozen=True)
class PredictionChangeRecord:
    omitted_sample_id: Hashable
    candidate_index: int
    objective_index: int
    objective_name: str
    full_mean: float
    omitted_mean: float
    mean_delta: float
    absolute_mean_delta: float
    full_std: float
    omitted_std: float
    std_delta: float
    absolute_std_delta: float
    objective_scale: float
    normalized_absolute_mean_delta: float
    normalized_absolute_std_delta: float


@dataclass(frozen=True)
class PredictionChangeMetrics:
    mean_absolute_mean_change: float
    maximum_absolute_mean_change: float
    mean_absolute_std_change: float
    maximum_absolute_std_change: float
    mean_normalized_absolute_mean_change: float
    mean_normalized_absolute_std_change: float
    component_value: float


@dataclass(frozen=True)
class AcquisitionRankMetrics:
    pool_size: int
    top_k: int
    spearman_rank_correlation: float
    top_k_overlap_count: int
    top_k_jaccard: float
    mean_absolute_rank_change: float
    maximum_absolute_rank_change: float
    normalized_mean_absolute_rank_change: float
    mean_absolute_score_change: float
    maximum_absolute_score_change: float
    component_value: float


@dataclass(frozen=True)
class ParetoMembershipChange:
    full_count: int
    omitted_count: int
    intersection_count: int
    union_count: int
    jaccard: float
    added_sample_ids: tuple[Hashable, ...]
    removed_sample_ids: tuple[Hashable, ...]
    symmetric_difference_count: int
    component_value: float


@dataclass(frozen=True)
class HyperparameterDisplacement:
    parameter_count: int
    mean_absolute_log_ratio: float
    maximum_absolute_log_ratio: float
    absolute_log_ratio_by_parameter: Mapping[str, float]
    component_value: float


@dataclass(frozen=True)
class InfluenceRunMetrics:
    omitted_sample_id: Hashable
    omitted_row_role: str
    omitted_primary_include_in_model: bool
    batch: RegionalBatchComparison
    prediction: PredictionChangeMetrics
    acquisition: AcquisitionRankMetrics
    pareto: ParetoMembershipChange
    hyperparameter: HyperparameterDisplacement
    raw_components: Mapping[str, float]
    normalized_components: Mapping[str, float]
    composite_score: float
    influence_rank: int
    influence_percentile: float


@dataclass(frozen=True)
class ObservationInfluenceStudyResult:
    common_pool_sha256: str
    objective_names: tuple[str, ...]
    objective_scales: np.ndarray
    regional_thresholds: tuple[float, ...]
    top_k: int
    component_weights: Mapping[str, float]
    row_roles: Mapping[Hashable, str]
    primary_include_policy: Mapping[Hashable, bool]
    runs: tuple[InfluenceRunMetrics, ...]
    prediction_changes: tuple[PredictionChangeRecord, ...]

    def rank_for_sample(self, sample_id: Hashable) -> InfluenceRunMetrics:
        for run in self.runs:
            if run.omitted_sample_id == sample_id:
                return run
        raise KeyError(f"No omission run exists for Sample ID {sample_id!r}.")

    def sample_1_rank(self) -> tuple[int, float]:
        record = self.rank_for_sample(1)
        return record.influence_rank, record.influence_percentile

    def summary_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for run in self.runs:
            row: dict[str, Any] = {
                "omitted_sample_id": run.omitted_sample_id,
                "omitted_row_role": run.omitted_row_role,
                "omitted_primary_include_in_model": (
                    run.omitted_primary_include_in_model
                ),
                "common_pool_sha256": self.common_pool_sha256,
                "exact_overlap_count": run.batch.exact_overlap_count,
                "jaccard_overlap": run.batch.jaccard_overlap,
                "matched_pair_distances": json.dumps(
                    run.batch.matched_pair_distances.tolist()
                ),
                "mean_matched_distance": run.batch.mean_matched_distance,
                "maximum_matched_distance": run.batch.maximum_matched_distance,
                "symmetric_chamfer_distance": (run.batch.symmetric_chamfer_distance),
                "hausdorff_distance": run.batch.hausdorff_distance,
                "mean_absolute_prediction_mean_change": (
                    run.prediction.mean_absolute_mean_change
                ),
                "maximum_absolute_prediction_mean_change": (
                    run.prediction.maximum_absolute_mean_change
                ),
                "mean_absolute_prediction_std_change": (
                    run.prediction.mean_absolute_std_change
                ),
                "maximum_absolute_prediction_std_change": (
                    run.prediction.maximum_absolute_std_change
                ),
                "acquisition_spearman_rank_correlation": (
                    run.acquisition.spearman_rank_correlation
                ),
                "acquisition_top_k": run.acquisition.top_k,
                "acquisition_top_k_overlap_count": (
                    run.acquisition.top_k_overlap_count
                ),
                "acquisition_top_k_jaccard": run.acquisition.top_k_jaccard,
                "acquisition_mean_absolute_rank_change": (
                    run.acquisition.mean_absolute_rank_change
                ),
                "acquisition_normalized_mean_absolute_rank_change": (
                    run.acquisition.normalized_mean_absolute_rank_change
                ),
                "full_pareto_count": run.pareto.full_count,
                "omitted_pareto_count": run.pareto.omitted_count,
                "pareto_jaccard": run.pareto.jaccard,
                "pareto_added_sample_ids": json.dumps(
                    list(run.pareto.added_sample_ids)
                ),
                "pareto_removed_sample_ids": json.dumps(
                    list(run.pareto.removed_sample_ids)
                ),
                "hyperparameter_mean_absolute_log_ratio": (
                    run.hyperparameter.mean_absolute_log_ratio
                ),
                "hyperparameter_maximum_absolute_log_ratio": (
                    run.hyperparameter.maximum_absolute_log_ratio
                ),
                "hyperparameter_absolute_log_ratios": json.dumps(
                    dict(run.hyperparameter.absolute_log_ratio_by_parameter),
                    sort_keys=True,
                ),
                "composite_influence_score": run.composite_score,
                "influence_rank": run.influence_rank,
                "influence_percentile": run.influence_percentile,
            }
            for threshold, count in run.batch.regional_match_counts.items():
                suffix = f"{threshold:.2f}".replace(".", "_")
                row[f"regional_matches_within_{suffix}"] = count
            for component in INFLUENCE_COMPONENT_NAMES:
                row[f"{component}_raw_component"] = run.raw_components[component]
                row[f"{component}_normalized_component"] = run.normalized_components[
                    component
                ]
                row[f"{component}_weight"] = self.component_weights[component]
            rows.append(row)
        return pd.DataFrame(rows)

    def prediction_changes_frame(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(row) for row in self.prediction_changes)


def prediction_change_metrics(
    full_mean: np.ndarray,
    full_std: np.ndarray,
    omitted_mean: np.ndarray,
    omitted_std: np.ndarray,
    *,
    omitted_sample_id: Hashable,
    objective_names: Sequence[str],
    objective_scales: Sequence[float],
) -> tuple[PredictionChangeMetrics, tuple[PredictionChangeRecord, ...]]:
    """Compare posterior reports evaluated at the same full-model candidates."""
    baseline_mean = _readonly_array(full_mean, name="full_mean", dimensions=2)
    baseline_std = _readonly_array(full_std, name="full_std", dimensions=2)
    comparison_mean = _readonly_array(omitted_mean, name="omitted_mean", dimensions=2)
    comparison_std = _readonly_array(omitted_std, name="omitted_std", dimensions=2)
    if not (
        baseline_mean.shape
        == baseline_std.shape
        == comparison_mean.shape
        == comparison_std.shape
    ):
        raise ValueError("All prediction matrices must have identical shapes.")
    if np.any(baseline_std < 0.0) or np.any(comparison_std < 0.0):
        raise ValueError("Prediction standard deviations cannot be negative.")
    names = tuple(
        name.strip() if isinstance(name, str) else name for name in objective_names
    )
    if len(names) != baseline_mean.shape[1] or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("objective_names must align with prediction columns.")
    if len(set(names)) != len(names):
        raise ValueError("objective_names must be unique.")
    scales = _positive_scales(objective_scales, count=len(names))
    mean_delta = comparison_mean - baseline_mean
    std_delta = comparison_std - baseline_std
    normalized_mean = np.abs(mean_delta) / scales[None, :]
    normalized_std = np.abs(std_delta) / scales[None, :]
    records: list[PredictionChangeRecord] = []
    for candidate_index in range(baseline_mean.shape[0]):
        for objective_index, objective_name in enumerate(names):
            records.append(
                PredictionChangeRecord(
                    omitted_sample_id=omitted_sample_id,
                    candidate_index=candidate_index,
                    objective_index=objective_index,
                    objective_name=objective_name,
                    full_mean=float(baseline_mean[candidate_index, objective_index]),
                    omitted_mean=float(
                        comparison_mean[candidate_index, objective_index]
                    ),
                    mean_delta=float(mean_delta[candidate_index, objective_index]),
                    absolute_mean_delta=float(
                        abs(mean_delta[candidate_index, objective_index])
                    ),
                    full_std=float(baseline_std[candidate_index, objective_index]),
                    omitted_std=float(comparison_std[candidate_index, objective_index]),
                    std_delta=float(std_delta[candidate_index, objective_index]),
                    absolute_std_delta=float(
                        abs(std_delta[candidate_index, objective_index])
                    ),
                    objective_scale=float(scales[objective_index]),
                    normalized_absolute_mean_delta=float(
                        normalized_mean[candidate_index, objective_index]
                    ),
                    normalized_absolute_std_delta=float(
                        normalized_std[candidate_index, objective_index]
                    ),
                )
            )
    mean_normalized = float(normalized_mean.mean())
    std_normalized = float(normalized_std.mean())
    metrics = PredictionChangeMetrics(
        mean_absolute_mean_change=float(np.abs(mean_delta).mean()),
        maximum_absolute_mean_change=float(np.abs(mean_delta).max()),
        mean_absolute_std_change=float(np.abs(std_delta).mean()),
        maximum_absolute_std_change=float(np.abs(std_delta).max()),
        mean_normalized_absolute_mean_change=mean_normalized,
        mean_normalized_absolute_std_change=std_normalized,
        component_value=0.5 * (mean_normalized + std_normalized),
    )
    return metrics, tuple(records)


def acquisition_rank_metrics(
    full_scores: np.ndarray,
    omitted_scores: np.ndarray,
    *,
    top_k: int,
) -> AcquisitionRankMetrics:
    """Compare acquisition ranks on one hash-verified ordered common pool."""
    baseline = _readonly_array(full_scores, name="full_scores", dimensions=1)
    comparison = _readonly_array(omitted_scores, name="omitted_scores", dimensions=1)
    if baseline.shape != comparison.shape:
        raise ValueError("full_scores and omitted_scores must have identical shape.")
    if (
        isinstance(top_k, (bool, np.bool_))
        or not isinstance(top_k, (int, np.integer))
        or int(top_k) <= 0
    ):
        raise ValueError("top_k must be a positive integer.")
    effective_top_k = min(int(top_k), baseline.size)
    baseline_rank = rankdata(-baseline, method="average")
    comparison_rank = rankdata(-comparison, method="average")
    rank_delta = np.abs(comparison_rank - baseline_rank)
    if np.array_equal(baseline, comparison):
        correlation = 1.0
    elif np.unique(baseline).size < 2 or np.unique(comparison).size < 2:
        correlation = np.nan
    else:
        correlation = float(spearmanr(baseline_rank, comparison_rank).statistic)
    baseline_top = set(np.argsort(-baseline, kind="stable")[:effective_top_k].tolist())
    comparison_top = set(
        np.argsort(-comparison, kind="stable")[:effective_top_k].tolist()
    )
    overlap = len(baseline_top & comparison_top)
    union = len(baseline_top | comparison_top)
    top_k_jaccard = float(overlap / union)
    denominator = max(1, baseline.size - 1)
    normalized_rank_change = float(rank_delta.mean() / denominator)
    component = 0.5 * (normalized_rank_change + (1.0 - top_k_jaccard))
    score_delta = np.abs(comparison - baseline)
    return AcquisitionRankMetrics(
        pool_size=int(baseline.size),
        top_k=effective_top_k,
        spearman_rank_correlation=correlation,
        top_k_overlap_count=overlap,
        top_k_jaccard=top_k_jaccard,
        mean_absolute_rank_change=float(rank_delta.mean()),
        maximum_absolute_rank_change=float(rank_delta.max()),
        normalized_mean_absolute_rank_change=normalized_rank_change,
        mean_absolute_score_change=float(score_delta.mean()),
        maximum_absolute_score_change=float(score_delta.max()),
        component_value=component,
    )


def pareto_membership_change(
    full_sample_ids: Sequence[Hashable],
    omitted_sample_ids: Sequence[Hashable],
) -> ParetoMembershipChange:
    """Return transparent set changes for observed Pareto membership."""
    full_ids = tuple(full_sample_ids)
    comparison_ids = tuple(omitted_sample_ids)
    full = set(full_ids)
    comparison = set(comparison_ids)
    if len(full) != len(full_ids) or len(comparison) != len(comparison_ids):
        raise ValueError("Pareto sample ID sequences must be unique.")
    intersection = full & comparison
    union = full | comparison
    jaccard = 1.0 if not union else float(len(intersection) / len(union))
    added = tuple(sorted(comparison - full, key=_sample_sort_key))
    removed = tuple(sorted(full - comparison, key=_sample_sort_key))
    return ParetoMembershipChange(
        full_count=len(full),
        omitted_count=len(comparison),
        intersection_count=len(intersection),
        union_count=len(union),
        jaccard=jaccard,
        added_sample_ids=added,
        removed_sample_ids=removed,
        symmetric_difference_count=len(full ^ comparison),
        component_value=1.0 - jaccard,
    )


def hyperparameter_displacement(
    full_parameters: Mapping[str, float],
    omitted_parameters: Mapping[str, float],
) -> HyperparameterDisplacement:
    """Compare positive hyperparameters with scale-free absolute log ratios."""
    baseline = _immutable_numeric_mapping(full_parameters, name="full_parameters")
    comparison = _immutable_numeric_mapping(
        omitted_parameters, name="omitted_parameters"
    )
    if set(baseline) != set(comparison):
        missing = sorted(set(baseline) - set(comparison))
        extra = sorted(set(comparison) - set(baseline))
        raise ValueError(
            "Hyperparameter keys must match exactly; "
            f"missing={missing}, extra={extra}."
        )
    changes = {
        key: float(abs(np.log(comparison[key]) - np.log(baseline[key])))
        for key in sorted(baseline)
    }
    values = np.asarray(list(changes.values()), dtype=float)
    mean_change = float(values.mean())
    return HyperparameterDisplacement(
        parameter_count=len(changes),
        mean_absolute_log_ratio=mean_change,
        maximum_absolute_log_ratio=float(values.max()),
        absolute_log_ratio_by_parameter=MappingProxyType(changes),
        component_value=mean_change,
    )


def _component_weights(
    value: Mapping[str, float] | None,
) -> Mapping[str, float]:
    raw = (
        {component: 1.0 for component in INFLUENCE_COMPONENT_NAMES}
        if value is None
        else dict(value)
    )
    if set(raw) != set(INFLUENCE_COMPONENT_NAMES):
        raise ValueError(
            "component_weights must contain exactly "
            f"{list(INFLUENCE_COMPONENT_NAMES)}."
        )
    parsed: dict[str, float] = {}
    for component in INFLUENCE_COMPONENT_NAMES:
        weight = raw[component]
        if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, Real):
            raise ValueError("Influence component weights must be real numbers.")
        number = float(weight)
        if not np.isfinite(number) or number < 0.0:
            raise ValueError(
                "Influence component weights must be finite and non-negative."
            )
        parsed[component] = number
    total = sum(parsed.values())
    if total <= 0.0:
        raise ValueError("At least one influence component weight must be positive.")
    return MappingProxyType(
        {component: parsed[component] / total for component in parsed}
    )


def _normalize_component_rows(
    rows: Sequence[Mapping[str, float]],
) -> list[dict[str, float]]:
    normalized = [dict() for _ in rows]
    for component in INFLUENCE_COMPONENT_NAMES:
        values = np.asarray([row[component] for row in rows], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                "Raw influence components must be finite and non-negative."
            )
        lower = float(values.min())
        upper = float(values.max())
        if upper - lower <= 1.0e-15:
            scaled = np.zeros(values.shape, dtype=float)
        else:
            scaled = (values - lower) / (upper - lower)
        for index, value in enumerate(scaled):
            normalized[index][component] = float(value)
    return normalized


def _policy_mapping(
    row_roles: Mapping[Hashable, str],
    include_policy: Mapping[Hashable, bool],
) -> tuple[Mapping[Hashable, str], Mapping[Hashable, bool]]:
    if not isinstance(row_roles, Mapping) or not row_roles:
        raise ValueError("row_roles must be a non-empty mapping.")
    if not isinstance(include_policy, Mapping) or set(include_policy) != set(row_roles):
        raise ValueError("primary_include_policy keys must exactly match row_roles.")
    roles: dict[Hashable, str] = {}
    includes: dict[Hashable, bool] = {}
    for sample_id, role in row_roles.items():
        if not isinstance(role, str) or not role.strip():
            raise ValueError("Every row role must be a non-empty string.")
        include = include_policy[sample_id]
        if not isinstance(include, (bool, np.bool_)):
            raise ValueError("Every include-in-model policy value must be boolean.")
        roles[sample_id] = role.strip()
        includes[sample_id] = bool(include)
    return MappingProxyType(roles), MappingProxyType(includes)


def run_observation_influence_study(
    full_run: InfluenceRunInput,
    omission_runs: Sequence[InfluenceRunInput],
    *,
    expected_common_pool_sha256: str,
    objective_names: Sequence[str],
    objective_scales: Sequence[float],
    row_roles: Mapping[Hashable, str],
    primary_include_policy: Mapping[Hashable, bool],
    regional_thresholds: Sequence[float] = DEFAULT_REGIONAL_THRESHOLDS,
    top_k: int = 100,
    component_weights: Mapping[str, float] | None = None,
    require_complete_coverage: bool = True,
) -> ObservationInfluenceStudyResult:
    """Compare full and all omission runs without fitting or mutating policy.

    Each raw component is min-max normalized over the omission cohort, then the
    normalized components are combined with the reported weights.  Rank 1 is the
    largest composite score; its percentile is 100 (ties share a rank).
    """
    if (
        not isinstance(full_run, InfluenceRunInput)
        or full_run.omitted_sample_id is not None
    ):
        raise ValueError(
            "full_run must be an InfluenceRunInput with no omitted sample."
        )
    runs = tuple(omission_runs)
    if not runs or not all(isinstance(run, InfluenceRunInput) for run in runs):
        raise ValueError("omission_runs must be a non-empty sequence of run inputs.")
    omitted_ids = tuple(run.omitted_sample_id for run in runs)
    if any(sample_id is None for sample_id in omitted_ids) or len(
        set(omitted_ids)
    ) != len(omitted_ids):
        raise ValueError("Every omission run must identify one unique omitted sample.")
    roles, include_policy = _policy_mapping(row_roles, primary_include_policy)
    if not isinstance(require_complete_coverage, bool):
        raise ValueError("require_complete_coverage must be a boolean.")
    omitted_set = set(omitted_ids)
    if require_complete_coverage and omitted_set != set(roles):
        raise ValueError(
            "Omission runs must cover every fixed row-role Sample ID exactly once."
        )
    if not require_complete_coverage and not omitted_set <= set(roles):
        raise ValueError("Omission runs contain a Sample ID absent from row_roles.")
    expected_hash = _sha256(
        expected_common_pool_sha256, name="expected_common_pool_sha256"
    )
    if full_run.common_pool_sha256 != expected_hash or any(
        run.common_pool_sha256 != expected_hash for run in runs
    ):
        raise ValueError("Every influence run must use the verified common pool hash.")
    if any(
        run.pool_acquisition_scores.shape != full_run.pool_acquisition_scores.shape
        for run in runs
    ):
        raise ValueError("Common-pool acquisition score arrays must have equal length.")
    if any(run.selected_X_norm.shape != full_run.selected_X_norm.shape for run in runs):
        raise ValueError(
            "Every influence proposal batch must have the full batch shape."
        )
    if any(
        run.prediction_mean_at_full_candidates.shape
        != full_run.prediction_mean_at_full_candidates.shape
        or run.prediction_std_at_full_candidates.shape
        != full_run.prediction_std_at_full_candidates.shape
        for run in runs
    ):
        raise ValueError(
            "Every prediction report must use the full candidate locations."
        )
    if any(
        run.prediction_uncertainty_kind != full_run.prediction_uncertainty_kind
        for run in runs
    ):
        raise ValueError("All runs must report the same uncertainty kind.")
    names = tuple(
        name.strip() if isinstance(name, str) else name for name in objective_names
    )
    objective_count = full_run.prediction_mean_at_full_candidates.shape[1]
    if len(names) != objective_count or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("objective_names must align with prediction columns.")
    if len(set(names)) != len(names):
        raise ValueError("objective_names must be unique.")
    scales = _positive_scales(objective_scales, count=objective_count)
    weights = _component_weights(component_weights)
    thresholds = regional_match_batches(
        full_run.selected_X_norm,
        full_run.selected_X_norm,
        thresholds=regional_thresholds,
    ).thresholds
    known_sample_ids = set(roles)
    if not set(full_run.pareto_sample_ids) <= known_sample_ids:
        raise ValueError("Full-run Pareto membership contains an unknown Sample ID.")

    provisional: list[dict[str, Any]] = []
    all_prediction_rows: list[PredictionChangeRecord] = []
    for run in sorted(runs, key=lambda item: _sample_sort_key(item.omitted_sample_id)):
        omitted_id = run.omitted_sample_id
        if omitted_id in run.pareto_sample_ids:
            raise ValueError(
                f"Omission run {omitted_id!r} still lists its omitted row as Pareto."
            )
        if not set(run.pareto_sample_ids) <= known_sample_ids - {omitted_id}:
            raise ValueError(
                "Omission Pareto membership contains an unknown Sample ID."
            )
        batch = regional_match_batches(
            full_run.selected_X_norm,
            run.selected_X_norm,
            thresholds=thresholds,
        )
        prediction, prediction_rows = prediction_change_metrics(
            full_run.prediction_mean_at_full_candidates,
            full_run.prediction_std_at_full_candidates,
            run.prediction_mean_at_full_candidates,
            run.prediction_std_at_full_candidates,
            omitted_sample_id=omitted_id,
            objective_names=names,
            objective_scales=scales,
        )
        all_prediction_rows.extend(prediction_rows)
        acquisition = acquisition_rank_metrics(
            full_run.pool_acquisition_scores,
            run.pool_acquisition_scores,
            top_k=top_k,
        )
        pareto = pareto_membership_change(
            full_run.pareto_sample_ids, run.pareto_sample_ids
        )
        hyperparameter = hyperparameter_displacement(
            full_run.hyperparameters, run.hyperparameters
        )
        raw_components = {
            "batch_displacement": batch.mean_matched_distance
            / np.sqrt(full_run.selected_X_norm.shape[1]),
            "prediction_change": prediction.component_value,
            "acquisition_change": acquisition.component_value,
            "pareto_change": pareto.component_value,
            "hyperparameter_change": hyperparameter.component_value,
        }
        provisional.append(
            {
                "omitted_sample_id": omitted_id,
                "batch": batch,
                "prediction": prediction,
                "acquisition": acquisition,
                "pareto": pareto,
                "hyperparameter": hyperparameter,
                "raw_components": raw_components,
            }
        )

    normalized_rows = _normalize_component_rows(
        [row["raw_components"] for row in provisional]
    )
    composite_scores = [
        float(
            sum(
                normalized[component] * weights[component]
                for component in INFLUENCE_COMPONENT_NAMES
            )
        )
        for normalized in normalized_rows
    ]
    count = len(composite_scores)
    completed: list[InfluenceRunMetrics] = []
    for index, row in enumerate(provisional):
        score = composite_scores[index]
        rank = 1 + sum(other > score + 1.0e-15 for other in composite_scores)
        percentile = 100.0 if count == 1 else 100.0 * (count - rank) / (count - 1)
        omitted_id = row["omitted_sample_id"]
        completed.append(
            InfluenceRunMetrics(
                omitted_sample_id=omitted_id,
                omitted_row_role=roles[omitted_id],
                omitted_primary_include_in_model=include_policy[omitted_id],
                batch=row["batch"],
                prediction=row["prediction"],
                acquisition=row["acquisition"],
                pareto=row["pareto"],
                hyperparameter=row["hyperparameter"],
                raw_components=MappingProxyType(dict(row["raw_components"])),
                normalized_components=MappingProxyType(normalized_rows[index]),
                composite_score=score,
                influence_rank=rank,
                influence_percentile=percentile,
            )
        )
    completed.sort(
        key=lambda row: (
            -row.composite_score,
            _sample_sort_key(row.omitted_sample_id),
        )
    )
    return ObservationInfluenceStudyResult(
        common_pool_sha256=expected_hash,
        objective_names=names,
        objective_scales=scales,
        regional_thresholds=thresholds,
        top_k=min(int(top_k), full_run.pool_acquisition_scores.size),
        component_weights=weights,
        row_roles=roles,
        primary_include_policy=include_policy,
        runs=tuple(completed),
        prediction_changes=tuple(all_prediction_rows),
    )


__all__ = [
    "AcquisitionRankMetrics",
    "HyperparameterDisplacement",
    "INFLUENCE_COMPONENT_NAMES",
    "InfluenceRunInput",
    "InfluenceRunMetrics",
    "ObservationInfluenceStudyResult",
    "ParetoMembershipChange",
    "PredictionChangeMetrics",
    "PredictionChangeRecord",
    "acquisition_rank_metrics",
    "hyperparameter_displacement",
    "pareto_membership_change",
    "prediction_change_metrics",
    "run_observation_influence_study",
]
