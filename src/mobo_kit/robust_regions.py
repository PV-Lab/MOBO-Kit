"""Persistent candidate-region clustering and debug-only consensus gating."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform


DEBUG_WATERMARK = "DEBUG ONLY - NOT APPROVED FOR EXPERIMENT"


# The Step 2C primary baseline is the reference level for every one-factor study.
# Keeping that relationship explicit prevents the larger nested and penalty families
# from dominating persistence merely because they contain more runs.
_STUDY_FAMILY_ALIASES: dict[str, frozenset[str]] = {
    "model": frozenset({"model", "model_variant"}),
    "nested_pool": frozenset({"nested", "nested_pool", "pool"}),
    "sobol_scramble": frozenset({"scramble", "sobol_scramble"}),
    "bound_policy": frozenset({"bound", "bound_policy", "bounded_utility"}),
    "beta": frozenset({"beta"}),
    "local_penalty": frozenset({"local_penalty", "penalty"}),
}


@dataclass(frozen=True)
class RobustRegionResult:
    regions: pd.DataFrame
    membership: pd.DataFrame
    threshold: float
    region_count: int


@dataclass(frozen=True)
class ConsensusCriteriaResult:
    passed: bool
    checks: dict[str, bool]
    observed: dict[str, float | int | bool]
    reasons: tuple[str, ...]


def _candidate_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("candidate_records must be a non-empty DataFrame.")
    required = {"run_id", "run_family", "acquisition_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate_records is missing required columns: {missing}.")
    norm_columns = sorted(
        (
            column
            for column in frame.columns
            if column.startswith("norm_") and column[5:].isdigit()
        ),
        key=lambda value: int(value.split("_", 1)[1]),
    )
    grid_columns = sorted(
        (
            column
            for column in frame.columns
            if column.startswith("grid_") and column[5:].isdigit()
        ),
        key=lambda value: int(value.split("_", 1)[1]),
    )
    if not norm_columns or len(norm_columns) != len(grid_columns):
        raise ValueError(
            "candidate_records must contain aligned norm_0..D and grid_0..D columns."
        )
    result = frame.copy(deep=True).reset_index(drop=True)
    result["run_id"] = result["run_id"].astype(str)
    result["run_family"] = result["run_family"].astype(str)
    if (
        result["run_id"].str.strip().eq("").any()
        or result["run_family"].str.strip().eq("").any()
    ):
        raise ValueError("run_id and run_family values must be nonblank.")
    norm = result.loc[:, norm_columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(norm)) or np.any(norm < 0) or np.any(norm > 1):
        raise ValueError("Normalized candidate coordinates must be finite in [0, 1].")
    grid = result.loc[:, grid_columns].to_numpy()
    if not np.all(np.isfinite(grid)) or not np.all(grid == np.floor(grid)):
        raise ValueError("Grid candidate coordinates must be finite integers.")
    acquisition = pd.to_numeric(result["acquisition_score"], errors="coerce").to_numpy()
    if not np.all(np.isfinite(acquisition)) or np.any(acquisition < 0):
        raise ValueError("acquisition_score values must be finite and non-negative.")
    result["acquisition_score"] = acquisition
    base_hvi_column = "base_hvi" if "base_hvi" in result else "acquisition_score"
    base_hvi = pd.to_numeric(result[base_hvi_column], errors="coerce").to_numpy()
    if not np.all(np.isfinite(base_hvi)) or np.any(base_hvi < 0):
        raise ValueError("base HVI values must be finite and non-negative.")
    result["base_hvi"] = base_hvi
    run_maximum = result.groupby("run_id", sort=False)["base_hvi"].transform("max")
    result["run_maximum_base_hvi"] = run_maximum
    result["base_hvi_normalized_within_run"] = np.divide(
        base_hvi,
        run_maximum.to_numpy(dtype=float),
        out=np.zeros_like(base_hvi, dtype=float),
        where=run_maximum.to_numpy(dtype=float) > 0,
    )
    if "selection_order" in result:
        selection_order = pd.to_numeric(result["selection_order"], errors="coerce")
        if (
            selection_order.isna().any()
            or (~np.isfinite(selection_order.to_numpy(dtype=float))).any()
            or (selection_order <= 0).any()
        ):
            raise ValueError("selection_order values must be finite and positive.")
        result["selection_order"] = selection_order
    return result, norm_columns, grid_columns


def _canonical_order(frame: pd.DataFrame, grid_columns: Sequence[str]) -> np.ndarray:
    tie_breakers = [
        column for column in ("selection_order", "candidate_id") if column in frame
    ]
    return frame.sort_values(
        [*grid_columns, "run_family", "run_id", *tie_breakers], kind="stable"
    ).index.to_numpy(dtype=int)


def _study_family_runs(
    registry: Mapping[str, str],
) -> tuple[dict[str, set[str]], set[str]]:
    normalized = {
        str(run_id): str(family).strip().lower() for run_id, family in registry.items()
    }
    baseline_runs = {
        run_id for run_id, family in normalized.items() if family == "baseline"
    }
    study_runs: dict[str, set[str]] = {}
    for name, aliases in _STUDY_FAMILY_ALIASES.items():
        study_runs[name] = baseline_runs | {
            run_id for run_id, family in normalized.items() if family in aliases
        }
    return study_runs, baseline_runs


def _coverage(
    represented_runs: set[str], family_runs: set[str]
) -> tuple[int, int, float]:
    total = len(family_runs)
    represented = len(represented_runs & family_runs)
    return represented, total, represented / total if total else np.nan


def _control_run_groups(
    frame: pd.DataFrame,
    *,
    control_sample_id: int | None,
) -> tuple[set[str], set[str], bool]:
    control_column = next(
        (
            column
            for column in ("control_included", "include_control")
            if column in frame
        ),
        None,
    )
    omitted_column = "omitted_sample_id" if "omitted_sample_id" in frame else None
    if control_column is None and omitted_column is None:
        return set(), set(), False

    if control_column is not None:
        values = frame[control_column]
        if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError(f"{control_column} values must be boolean.")
        control_included = values.astype(bool)
        control_omitted = ~control_included
    elif omitted_column is not None:
        if isinstance(control_sample_id, bool) or not isinstance(
            control_sample_id, (int, np.integer)
        ):
            raise ValueError(
                "control_sample_id is required when only omitted_sample_id is "
                "available."
            )
        omitted = pd.to_numeric(frame[omitted_column], errors="coerce")
        control_omitted = omitted.eq(int(control_sample_id))
        control_included = ~control_omitted
    else:  # pragma: no cover - guarded by the early return above
        raise RuntimeError("Control-run grouping reached an invalid state.")

    control_runs = set(frame.loc[control_included, "run_id"].astype(str))
    omitted_control_runs = set(frame.loc[control_omitted, "run_id"].astype(str))
    return (
        control_runs,
        omitted_control_runs,
        bool(control_runs and omitted_control_runs),
    )


def _finite_median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def _medoid_index(norm: np.ndarray, grid: np.ndarray) -> int:
    if norm.shape[0] == 1:
        return 0
    distances = squareform(pdist(norm, metric="euclidean"))
    means = distances.mean(axis=1)
    minimum = float(means.min())
    tied = np.flatnonzero(np.abs(means - minimum) <= 1.0e-15)
    if tied.size == 1:
        return int(tied[0])
    keys = tuple(grid[tied, column] for column in reversed(range(grid.shape[1])))
    return int(tied[np.lexsort(keys)[0]])


def _diameter(norm: np.ndarray) -> float:
    if norm.shape[0] < 2:
        return 0.0
    return float(np.max(pdist(norm, metric="euclidean")))


def cluster_candidate_regions(
    candidate_records: pd.DataFrame,
    *,
    distance_threshold: float = 0.15,
    core_run_registry: Mapping[str, str] | None = None,
    control_sample_id: int | None = None,
) -> RobustRegionResult:
    """Cluster candidate selections by deterministic complete linkage."""
    threshold = float(distance_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("distance_threshold must be finite and positive.")
    frame, norm_columns, grid_columns = _candidate_frame(candidate_records)
    order = _canonical_order(frame, grid_columns)
    sorted_frame = (
        frame.iloc[order]
        .reset_index(drop=False)
        .rename(columns={"index": "source_row"})
    )
    norm = sorted_frame.loc[:, norm_columns].to_numpy(dtype=float)
    grid = sorted_frame.loc[:, grid_columns].to_numpy(dtype=np.int64)
    if norm.shape[0] == 1:
        raw_labels = np.ones(1, dtype=int)
    else:
        hierarchy = linkage(
            norm, method="complete", metric="euclidean", optimal_ordering=True
        )
        raw_labels = fcluster(hierarchy, t=threshold, criterion="distance")

    registry = (
        {str(run): str(family) for run, family in core_run_registry.items()}
        if core_run_registry is not None
        else dict(
            sorted_frame.loc[:, ["run_id", "run_family"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    )
    if not registry:
        raise ValueError("core_run_registry must contain at least one run.")
    for run_id, family in registry.items():
        observed = sorted_frame.loc[sorted_frame["run_id"] == run_id, "run_family"]
        if not observed.empty and not observed.eq(family).all():
            raise ValueError(f"Run {run_id!r} has inconsistent family metadata.")
    family_runs: dict[str, set[str]] = {}
    for run_id, family in registry.items():
        family_runs.setdefault(family, set()).add(run_id)
    study_family_runs, baseline_runs = _study_family_runs(registry)
    (
        control_runs,
        omitted_control_runs,
        control_correspondence_available,
    ) = _control_run_groups(sorted_frame, control_sample_id=control_sample_id)

    raw_region_rows: list[dict[str, object]] = []
    membership_parts: list[pd.DataFrame] = []
    for raw_label in sorted(set(int(value) for value in raw_labels)):
        positions = np.flatnonzero(raw_labels == raw_label)
        member_frame = sorted_frame.iloc[positions].copy()
        member_norm = norm[positions]
        member_grid = grid[positions]
        medoid_local = _medoid_index(member_norm, member_grid)
        medoid_row = member_frame.iloc[medoid_local]
        medoid_norm = member_norm[medoid_local]
        member_distances = np.linalg.norm(member_norm - medoid_norm, axis=1)
        represented_runs = set(member_frame["run_id"].astype(str)) & set(registry)
        coverage_by_family: dict[str, float] = {}
        for family, runs in sorted(family_runs.items()):
            covered = len(runs & represented_runs)
            coverage_by_family[family] = covered / len(runs)
        raw_family_persistence = float(np.mean(list(coverage_by_family.values())))
        study_family_coverage: dict[str, float] = {}
        study_family_counts: dict[str, tuple[int, int]] = {}
        for family in _STUDY_FAMILY_ALIASES:
            covered, total, coverage = _coverage(
                represented_runs, study_family_runs[family]
            )
            study_family_counts[family] = (covered, total)
            study_family_coverage[family] = coverage
        available_study_coverage = [
            value for value in study_family_coverage.values() if np.isfinite(value)
        ]
        study_family_persistence = (
            float(np.mean(available_study_coverage))
            if available_study_coverage
            else np.nan
        )
        # A Step 2C registry has an explicit shared baseline. Equal weighting of
        # its six study families is then the primary persistence measure. Generic
        # callers without that convention retain the historical raw-family score.
        persistence = (
            study_family_persistence if baseline_runs else raw_family_persistence
        )
        represented_families = {
            registry[run_id] for run_id in represented_runs if run_id in registry
        }
        represented_nonbaseline_families = {
            registry[run_id]
            for run_id in represented_runs
            if run_id in registry and run_id not in baseline_runs
        }
        acquisition = member_frame["acquisition_score"].to_numpy(dtype=float)
        normalized_hvi = member_frame["base_hvi_normalized_within_run"].to_numpy(
            dtype=float
        )
        hvi_quartiles = np.quantile(normalized_hvi, [0.25, 0.75])
        boundary_flags = np.isclose(
            member_norm, 0.0, rtol=0.0, atol=1.0e-12
        ) | np.isclose(member_norm, 1.0, rtol=0.0, atol=1.0e-12)
        boundary_frequency = boundary_flags.mean(axis=0)
        control_covered, control_total, control_coverage = _coverage(
            set(member_frame["run_id"].astype(str)), control_runs
        )
        omit_covered, omit_total, omit_coverage = _coverage(
            set(member_frame["run_id"].astype(str)), omitted_control_runs
        )
        # The lesser cohort coverage is a conservative, symmetric regional
        # correspondence measure: it is high only when both control policies
        # repeatedly select candidates in this same complete-link region.
        control_correspondence = (
            float(min(control_coverage, omit_coverage))
            if control_correspondence_available
            else np.nan
        )
        row: dict[str, object] = {
            "raw_region_label": raw_label,
            "member_count": int(len(positions)),
            "distinct_run_count": int(member_frame["run_id"].nunique()),
            "distinct_core_run_count": int(len(represented_runs)),
            "distinct_family_count": int(len(represented_families)),
            "distinct_nonbaseline_family_count": int(
                len(represented_nonbaseline_families)
            ),
            "run_ids": "|".join(sorted(set(member_frame["run_id"].astype(str)))),
            "run_families": "|".join(
                sorted(set(member_frame["run_family"].astype(str)))
            ),
            "family_weighted_persistence": persistence,
            "registry_family_weighted_persistence": raw_family_persistence,
            "study_family_weighted_persistence": study_family_persistence,
            "persistence_basis": (
                "six_equal_weight_step2c_study_families"
                if baseline_runs
                else "registry_run_families"
            ),
            "family_coverage": "|".join(
                f"{family}:{coverage:.6f}"
                for family, coverage in sorted(coverage_by_family.items())
            ),
            "study_family_coverage": "|".join(
                f"{family}:{coverage:.6f}"
                for family, coverage in study_family_coverage.items()
                if np.isfinite(coverage)
            ),
            "median_acquisition_score": float(np.median(acquisition)),
            "maximum_acquisition_score": float(np.max(acquisition)),
            "median_selection_order": _finite_median(member_frame, "selection_order"),
            "base_hvi_normalization": "divide_by_run_maximum",
            "median_run_normalized_base_hvi": float(np.median(normalized_hvi)),
            "q1_run_normalized_base_hvi": float(hvi_quartiles[0]),
            "q3_run_normalized_base_hvi": float(hvi_quartiles[1]),
            "iqr_run_normalized_base_hvi": float(hvi_quartiles[1] - hvi_quartiles[0]),
            "mean_distance_to_medoid": float(member_distances.mean()),
            "maximum_distance_to_medoid": float(member_distances.max()),
            "cluster_diameter": _diameter(member_norm),
            "maximum_within_region_distance": _diameter(member_norm),
            "medoid_source_row": int(medoid_row["source_row"]),
            "all_grid_valid": (
                bool(member_frame.get("grid_valid", True).astype(bool).all())
                if "grid_valid" in member_frame
                else True
            ),
            "all_hard_distance_valid": (
                bool(member_frame.get("hard_distance_valid", True).astype(bool).all())
                if "hard_distance_valid" in member_frame
                else True
            ),
            "minimum_nearest_control_distance": (
                float(member_frame["nearest_control_distance"].min())
                if "nearest_control_distance" in member_frame
                else np.nan
            ),
            "mean_boundary_coordinate_count": (
                float(member_frame["boundary_coordinate_count"].mean())
                if "boundary_coordinate_count" in member_frame
                else np.nan
            ),
            "boundary_dimension_frequency": "|".join(
                f"{column}:{frequency:.6f}"
                for column, frequency in zip(norm_columns, boundary_frequency)
            ),
            "median_nearest_observed_distance": _finite_median(
                member_frame, "nearest_observed_distance"
            ),
            "control_omission_correspondence_available": (
                control_correspondence_available
            ),
            "control_omission_correspondence_definition": (
                "minimum_of_control_included_and_control_omitted_run_coverage"
            ),
            "control_included_represented_run_count": control_covered,
            "control_included_total_run_count": control_total,
            "control_included_run_coverage": control_coverage,
            "control_omitted_represented_run_count": omit_covered,
            "control_omitted_total_run_count": omit_total,
            "control_omitted_run_coverage": omit_coverage,
            "control_included_vs_control_omitted_correspondence": (
                control_correspondence
            ),
        }
        for family, (covered, total) in study_family_counts.items():
            row[f"{family}_represented_run_count"] = covered
            row[f"{family}_total_run_count"] = total
            row[f"{family}_coverage"] = study_family_coverage[family]
        for short_name, family in {
            "nested": "nested_pool",
            "scramble": "sobol_scramble",
            "bound": "bound_policy",
            "penalty": "local_penalty",
        }.items():
            row[f"{short_name}_coverage"] = study_family_coverage[family]
        for dimension, frequency in enumerate(boundary_frequency):
            row[f"boundary_dimension_{dimension}_frequency"] = float(frequency)
        if "boundary_dimensions" in member_frame:
            boundary_names = [
                token.strip()
                for value in member_frame["boundary_dimensions"].fillna("").astype(str)
                for token in value.split("|")
                if token.strip()
            ]
            row["boundary_dimension_name_frequency"] = "|".join(
                f"{name}:{boundary_names.count(name) / len(member_frame):.6f}"
                for name in sorted(set(boundary_names))
            )
        for column in grid_columns:
            row[f"medoid_{column}"] = int(medoid_row[column])
        for column in norm_columns:
            row[f"medoid_{column}"] = float(medoid_row[column])
        for column in member_frame.columns:
            if column.startswith("phys_"):
                row[f"medoid_{column}"] = float(medoid_row[column])
            elif column.startswith("pred_mean_") or column.startswith("pred_std_"):
                numeric = pd.to_numeric(member_frame[column], errors="coerce")
                if numeric.notna().all():
                    row[f"medoid_{column}"] = float(medoid_row[column])
                    row[f"mean_{column}"] = float(numeric.mean())
                    row[f"median_{column}"] = float(numeric.median())
        raw_region_rows.append(row)
        member_frame["raw_region_label"] = raw_label
        member_frame["distance_to_region_medoid"] = member_distances
        membership_parts.append(member_frame)

    regions = pd.DataFrame(raw_region_rows)
    medoid_grid_columns = [f"medoid_{column}" for column in grid_columns]
    ranking = regions.sort_values(
        [
            "family_weighted_persistence",
            "distinct_family_count",
            "distinct_core_run_count",
            "median_run_normalized_base_hvi",
            "median_acquisition_score",
            "member_count",
            *medoid_grid_columns,
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            *([True] * len(grid_columns)),
        ],
        kind="stable",
    ).reset_index(drop=True)
    # Canonical region IDs follow robustness rank, with medoid grid tuple as the
    # deterministic final tie break.
    ranking["region_id"] = [
        f"REGION-{index:03d}" for index in range(1, len(ranking) + 1)
    ]
    label_to_id = dict(zip(ranking["raw_region_label"], ranking["region_id"]))
    membership = pd.concat(membership_parts, ignore_index=True)
    membership["region_id"] = membership["raw_region_label"].map(label_to_id)
    membership = membership.sort_values(
        ["region_id", "run_family", "run_id", *grid_columns], kind="stable"
    ).reset_index(drop=True)
    ranking["debug_only"] = True
    ranking["approved_for_experiment"] = False
    ranking["approved_for_production"] = False
    ranking["candidate_status"] = DEBUG_WATERMARK
    membership["debug_only"] = True
    membership["approved_for_experiment"] = False
    membership["approved_for_production"] = False
    membership["candidate_status"] = DEBUG_WATERMARK
    return RobustRegionResult(
        regions=ranking,
        membership=membership,
        threshold=threshold,
        region_count=int(ranking.shape[0]),
    )


def select_robust_shortlist(
    region_result: RobustRegionResult,
    *,
    minimum_count: int = 8,
    maximum_count: int = 12,
    minimum_normalized_distance: float = 0.15,
) -> pd.DataFrame:
    """Choose diverse region medoids; return fewer only when geometry requires it."""
    if not isinstance(region_result, RobustRegionResult):
        raise TypeError("region_result must be a RobustRegionResult.")
    if not (1 <= int(minimum_count) <= int(maximum_count)):
        raise ValueError("Require 1 <= minimum_count <= maximum_count.")
    distance = float(minimum_normalized_distance)
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("minimum_normalized_distance must be finite and non-negative.")
    regions = region_result.regions.copy()
    medoid_norm_columns = sorted(
        (column for column in regions if column.startswith("medoid_norm_")),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if not medoid_norm_columns:
        raise ValueError("Region table has no medoid normalized coordinates.")
    selected_rows: list[pd.Series] = []
    selected_norm: list[np.ndarray] = []
    for _, row in regions.iterrows():
        candidate = row.loc[medoid_norm_columns].to_numpy(dtype=float)
        if selected_norm:
            nearest = float(
                np.linalg.norm(
                    np.asarray(selected_norm) - candidate[None, :], axis=1
                ).min()
            )
            if nearest < distance:
                continue
        selected_rows.append(row)
        selected_norm.append(candidate)
        if len(selected_rows) == int(maximum_count):
            break
    if not selected_rows:
        raise RuntimeError("No robust-region medoid passed the shortlist rules.")
    shortlist = pd.DataFrame(selected_rows).reset_index(drop=True)
    shortlist.insert(
        0,
        "shortlist_id",
        [f"R1-RS{index:02d}" for index in range(1, len(shortlist) + 1)],
    )
    shortlist["shortlist_target_minimum_met"] = len(shortlist) >= int(minimum_count)
    shortlist["shortlist_count"] = len(shortlist)
    shortlist["minimum_normalized_distance_required"] = distance
    shortlist["debug_only"] = True
    shortlist["approved_for_experiment"] = False
    shortlist["approved_for_production"] = False
    shortlist["candidate_status"] = DEBUG_WATERMARK
    return shortlist


def evaluate_consensus_criteria(
    regions: pd.DataFrame,
    consensus_candidates: pd.DataFrame,
    *,
    largest_two_regional_matches_within_0_15: int,
    largest_two_mean_matched_distance: float,
    nested_match_minimum: int = 4,
    mean_distance_maximum: float = 0.10,
    minimum_family_coverage: int = 3,
    consensus_batch_size: int = 5,
    required_minimum_distance: float = 0.15,
    full_mode_eligible: bool = True,
) -> ConsensusCriteriaResult:
    """Evaluate every gate on the exact medoids eligible for publication."""
    if not isinstance(regions, pd.DataFrame) or not isinstance(
        consensus_candidates, pd.DataFrame
    ):
        raise TypeError("regions and consensus_candidates must be DataFrames.")
    minimum_distance = float(required_minimum_distance)
    if not np.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError("required_minimum_distance must be finite and non-negative.")
    if not isinstance(full_mode_eligible, (bool, np.bool_)):
        raise TypeError("full_mode_eligible must be a boolean.")
    family_count_column = (
        "distinct_nonbaseline_family_count"
        if "distinct_nonbaseline_family_count" in regions
        else "distinct_family_count"
    )
    robust_region_count = int(
        (
            pd.to_numeric(regions.get(family_count_column), errors="coerce")
            >= int(minimum_family_coverage)
        ).sum()
    )
    exact_count = int(consensus_candidates.shape[0]) == int(consensus_batch_size)
    family_qualified = bool(
        exact_count
        and (
            pd.to_numeric(
                consensus_candidates.get(
                    family_count_column,
                    pd.Series(np.nan, index=consensus_candidates.index),
                ),
                errors="coerce",
            )
            >= int(minimum_family_coverage)
        ).all()
    )
    grid_valid = bool(
        exact_count
        and consensus_candidates.get("all_grid_valid", pd.Series([False]))
        .astype(bool)
        .all()
    )
    norm_columns = sorted(
        (
            column
            for column in consensus_candidates
            if column.startswith("medoid_norm_")
        ),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    grid_columns = sorted(
        (
            column
            for column in consensus_candidates
            if column.startswith("medoid_grid_")
        ),
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    coordinates = (
        consensus_candidates.loc[:, norm_columns].to_numpy(dtype=float)
        if exact_count and norm_columns
        else np.empty((0, 0), dtype=float)
    )
    finite_bounded = bool(
        exact_count
        and coordinates.shape[1] > 0
        and np.all(np.isfinite(coordinates))
        and np.all(coordinates >= 0.0)
        and np.all(coordinates <= 1.0)
    )
    unique = bool(
        exact_count
        and len(grid_columns) == len(norm_columns)
        and np.unique(
            consensus_candidates.loc[:, grid_columns].to_numpy(dtype=np.int64),
            axis=0,
        ).shape[0]
        == int(consensus_batch_size)
    )
    if exact_count and coordinates.shape[0] > 1:
        pairwise_minimum = float(np.min(pdist(coordinates, metric="euclidean")))
    else:
        pairwise_minimum = 0.0
    hard_valid = bool(
        exact_count
        and pairwise_minimum + 1.0e-12 >= minimum_distance
        and consensus_candidates.get("all_hard_distance_valid", pd.Series([False]))
        .astype(bool)
        .all()
    )
    checks = {
        "full_mode_eligible_for_consensus": bool(full_mode_eligible),
        "largest_two_nested_regional_matches": int(
            largest_two_regional_matches_within_0_15
        )
        >= int(nested_match_minimum),
        "largest_two_nested_mean_matched_distance": float(
            largest_two_mean_matched_distance
        )
        <= float(mean_distance_maximum),
        "five_regions_cover_three_core_families": robust_region_count
        >= int(consensus_batch_size),
        "exact_five_candidates": exact_count,
        "chosen_regions_cover_three_core_families": family_qualified,
        "finite_and_bounded": finite_bounded,
        "unique_and_on_grid": unique and grid_valid,
        "chosen_pairwise_hard_distance_valid": hard_valid,
        "debug_only": bool(
            exact_count
            and consensus_candidates.get("debug_only", pd.Series([False]))
            .astype(bool)
            .all()
        ),
        "experimental_approval_false": bool(
            exact_count
            and (
                ~consensus_candidates.get(
                    "approved_for_experiment", pd.Series([True])
                ).astype(bool)
            ).all()
        ),
        "production_approval_false": bool(
            exact_count
            and (
                ~consensus_candidates.get(
                    "approved_for_production", pd.Series([True])
                ).astype(bool)
            ).all()
        ),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    observed: dict[str, float | int | bool] = {
        "largest_two_regional_matches_within_0_15": int(
            largest_two_regional_matches_within_0_15
        ),
        "largest_two_mean_matched_distance": float(largest_two_mean_matched_distance),
        "regions_with_minimum_family_coverage": robust_region_count,
        "consensus_candidate_count": int(consensus_candidates.shape[0]),
        "chosen_regions_family_qualified": family_qualified,
        "chosen_candidates_finite_and_bounded": finite_bounded,
        "chosen_candidates_unique_and_on_grid": unique and grid_valid,
        "chosen_pairwise_minimum_distance": pairwise_minimum,
        "required_pairwise_minimum_distance": minimum_distance,
        "chosen_hard_distance_valid": hard_valid,
        "full_mode_eligible_for_consensus": bool(full_mode_eligible),
    }
    return ConsensusCriteriaResult(
        passed=all(checks.values()),
        checks=checks,
        observed=observed,
        reasons=reasons,
    )


__all__ = [
    "ConsensusCriteriaResult",
    "RobustRegionResult",
    "cluster_candidate_regions",
    "evaluate_consensus_criteria",
    "select_robust_shortlist",
]
