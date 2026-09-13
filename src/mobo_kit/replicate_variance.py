"""Turn replicate films into the observation variance the GP is told about.

Each proposed condition is run in triplicate.  Those three films are one design
point, so they are aggregated to a single observation -- and the scatter that
aggregation discards is the only direct measurement this campaign has of how
reproducible its own process is.  Handing it to the GP as ``train_Yvar`` stops the
marginal likelihood from having to guess the noise from 15 points in 10 dimensions.

**Two variances live in this campaign and they are not interchangeable.**

*Between-film* variance is the quantity ``train_Yvar`` needs: two films made from
the same recipe differ by everything that varies run to run -- ambient conditions,
the operator, the substrate, the anneal.  It only becomes measurable when the R1
triplicates land.

*Within-film* variance is the scatter of the 3-4 thickness points measured across
one film.  It is measurement plus spatial nonuniformity.  It is **not** a
substitute.  It excludes run-to-run variation entirely, so it sets a *floor* -- but
only in the same units as what it is compared against.  The pooled between-film
variance is a variance of film MEANS, each averaging n readings, so the floor is
the within-film variance of one reading divided by the readings per film.
Comparing it with the variance of a single reading instead fired a false alarm on
the first real triplicates (2026-09-10).  If the pooled between-film variance ever
comes out below the correct floor, something is wrong with the measurement or the
pooling.  :func:`sanity_floor_findings` says so rather than assuming it.

**Space matters.**  Variance must be in the space the GP trains in.  Every v4
objective trains on its own scale -- thickness in nanometres since its prior was
withdrawn on 2026-09-06 -- so its variance is in its own units, nm^2 for
thickness.  Only an objective with a ``response: log`` mean function trains on
``log y`` and takes a variance of ``log y``.  A variance in the wrong space is
wrong by a factor of y^2, not even a constant rescaling, and ``Standardize``
rescales it without complaint.  :class:`workbook_io.CandidateResults` reports
``replicate_spread`` in each objective's aggregation space, and
:func:`campaign.replicate_aggregates` refuses an aggregation space that differs
from the model's link, which is what keeps the two the same.

**What the GP is told is the variance of the MEAN**, not of a single film.  The
observation handed to the model is an average of ``n`` films, so its variance is
``pooled / n``.  Passing the single-film variance instead would hand the model a
number three times too large on a triplicate -- overstating its uncertainty, so it
would trust a well-replicated condition less than it has earned -- and BoTorch will
not complain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "PooledVariance",
    "REPLICATE_POOLED",
    "WITHIN_FILM_LOG_THICKNESS_VARIANCE",
    "pool_between_film_variance",
    "sanity_floor_findings",
    "train_yvar_for_rows",
    "variance_config",
    "yvar_for_campaign",
]

#: ARCHIVED, from the FIRST campaign (v2), whose model trained thickness on
#: ``log T``: a within-row figure on ``log T`` across its 15 rows, 24 dof. Kept
#: because that contract's tests pin it. It is NOT a floor for the live contract,
#: which trains thickness in nanometres and declares its own floor --
#: ``model.replicate_variance.sanity_floor`` -- as the within-film variance of a
#: film mean, in nm^2.
WITHIN_FILM_LOG_THICKNESS_VARIANCE = 0.0593


@dataclass(frozen=True)
class PooledVariance:
    """One objective's between-film variance, pooled across conditions."""

    objective: str
    variance: float
    dof: int
    n_conditions: int
    space: str
    """``value`` or ``log`` -- the space the aggregation and the GP both work in."""

    @property
    def sd(self) -> float:
        return math.sqrt(self.variance)

    def variance_of_mean(self, n_films: int) -> float:
        """Variance of an observation that is the mean of ``n_films`` films."""
        if n_films < 1:
            raise ValueError("n_films must be at least 1.")
        return self.variance / float(n_films)


def pool_between_film_variance(
    replicate_spread: pd.DataFrame,
    films_used: pd.DataFrame,
    *,
    aggregates: Mapping[str, str] | Sequence[str] | None = None,
) -> dict[str, PooledVariance]:
    """Pool per-condition replicate scatter into one variance per objective.

    ``replicate_spread`` holds the per-condition sample sd in each objective's
    aggregation space, and ``films_used`` how many films each came from -- both
    straight off :func:`workbook_io.read_candidate_results`.

    Pooling is the usual dof-weighted estimate, ``sum((n_i - 1) * s_i^2) /
    sum(n_i - 1)``.  Conditions with one usable film contribute no dof and are
    skipped rather than counted as zero variance: one film measures no
    reproducibility, and treating that as perfect reproducibility is how a model
    ends up certain about a process nobody has measured twice.
    """
    if not isinstance(replicate_spread, pd.DataFrame):
        raise TypeError("replicate_spread must be a pandas DataFrame.")
    if not isinstance(films_used, pd.DataFrame):
        raise TypeError("films_used must be a pandas DataFrame.")
    if list(replicate_spread.columns) != list(films_used.columns):
        raise ValueError(
            "replicate_spread and films_used must describe the same objectives; got "
            f"{list(replicate_spread.columns)} and {list(films_used.columns)}."
        )
    if isinstance(aggregates, Mapping):
        spaces = dict(aggregates)
    elif aggregates is None:
        spaces = {}
    else:
        spaces = dict(zip(replicate_spread.columns, aggregates))

    pooled: dict[str, PooledVariance] = {}
    for name in replicate_spread.columns:
        weighted = 0.0
        dof = 0
        used = 0
        for spread, films in zip(replicate_spread[name], films_used[name]):
            count = int(films)
            if count < 2 or not np.isfinite(spread):
                continue
            weighted += (count - 1) * float(spread) ** 2
            dof += count - 1
            used += 1
        if dof == 0:
            raise ValueError(
                f"Objective {name!r} has no condition with two or more usable films, "
                "so between-film variance cannot be estimated. Replicates are what "
                "make it measurable; a single film per condition measures no "
                "reproducibility at all."
            )
        rule = str(spaces.get(name, "mean"))
        pooled[name] = PooledVariance(
            objective=name,
            variance=weighted / dof,
            dof=dof,
            n_conditions=used,
            space="log" if rule == "mean_of_log" else "value",
        )
    return pooled


def sanity_floor_findings(
    pooled: Mapping[str, PooledVariance], floors: Mapping[str, float]
) -> tuple[str, ...]:
    """Report any objective whose between-film variance falls below its floor.

    A floor comes from within-film scatter, which contains no run-to-run variation.
    Between-film variance below it means films are apparently more reproducible than
    points on one film, which is not a thing -- so it indicates a measurement or
    pooling mistake, not a very good process.
    """
    messages: list[str] = []
    for name, floor in floors.items():
        estimate = pooled.get(name)
        if estimate is None:
            continue
        if estimate.variance < float(floor):
            messages.append(
                f"{name}: pooled between-film variance {estimate.variance:.4g} "
                f"({estimate.dof} dof) is BELOW the within-film floor {float(floor):.4g}. "
                "Films cannot be more reproducible than points measured on a single "
                "film, so this points at the measurements or the pooling, not at a "
                "very reproducible process. Check before trusting train_Yvar."
            )
    return tuple(messages)


def train_yvar_for_rows(
    pooled: Mapping[str, PooledVariance],
    films_per_row: pd.DataFrame | np.ndarray,
    objective_names: Sequence[str],
    *,
    rows_without_replicates: int = 1,
) -> np.ndarray:
    """The ``(n_rows, n_objectives)`` variance array to hand the model.

    Each entry is the variance of that row's observation, which is a mean of
    ``n`` films, so ``pooled / n``.

    ``rows_without_replicates`` is the film count assumed for rows that have none --
    the R0 rows, which predate the triplicate policy.  It defaults to 1: their
    observation is a single film, so it carries the full between-film variance
    rather than a third of it. That assumes the measurement process is unchanged
    between rounds, which is an assumption and is recorded in the config as one.
    """
    names = list(objective_names)
    counts = (
        films_per_row[names].to_numpy(dtype=float)
        if isinstance(films_per_row, pd.DataFrame)
        else np.asarray(films_per_row, dtype=float)
    )
    if counts.ndim == 1:
        counts = np.repeat(counts[:, None], len(names), axis=1)
    if counts.shape[1] != len(names):
        raise ValueError(
            f"films_per_row must have one column per objective ({len(names)}); "
            f"got {counts.shape[1]}."
        )
    missing = [name for name in names if name not in pooled]
    if missing:
        raise ValueError(f"No pooled variance for objective(s) {missing}.")

    # A pooled variance of zero says every replicate of every condition agreed to
    # the last digit. For a physical measurement that means the values were copied
    # rather than measured -- and handing zero to the model asserts the observation
    # is exact, which makes it interpolate through a number nobody verified.
    degenerate = [name for name in names if pooled[name].variance <= 0.0]
    if degenerate:
        raise ValueError(
            f"Pooled between-film variance is zero for {degenerate}. Replicate films "
            "that agree exactly are a transcription, not a measurement, and a zero "
            "train_Yvar tells the model the observation is exact. Check the entries "
            "before enabling measured observation noise."
        )

    counts = np.where(counts >= 1, counts, float(rows_without_replicates))
    variances = np.empty_like(counts, dtype=float)
    for index, name in enumerate(names):
        variances[:, index] = pooled[name].variance / counts[:, index]
    return variances


#: ``model.observation_noise`` value that switches measured replicate variance on.
REPLICATE_POOLED = "replicate_pooled"


def yvar_for_campaign(
    config: Mapping[str, Any],
    results: Any,
    *,
    n_rows_without_replicates: int,
    objective_names: Sequence[str],
    aggregates: Sequence[str] | Mapping[str, str] | None = None,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """Build ``train_Yvar`` for the whole observed set, or return ``None``.

    ``None`` unless ``model.observation_noise`` is ``replicate_pooled``: until the
    triplicates land there is nothing to pool, and the marginal likelihood keeps
    fitting the noise as it does today.  Once they do land, turning this on is a
    config edit, which is the point of wiring it before the data exists.

    Rows without replicates -- the R0 block, which predates the triplicate policy --
    come first and take the declared film count.  The replicated conditions follow
    in the order :func:`workbook_io.read_candidate_results` returned them, which is
    the order they are appended to the observation matrix.
    """
    if str((config.get("model") or {}).get("observation_noise")) != REPLICATE_POOLED:
        return None, ()

    settings = variance_config(config)
    names = list(objective_names)
    pooled = pool_between_film_variance(
        results.replicate_spread, results.films_used, aggregates=aggregates
    )
    findings = sanity_floor_findings(pooled, settings["sanity_floor"])

    prior = np.full((int(n_rows_without_replicates), len(names)), 0.0)
    replicated = results.films_used[names].to_numpy(dtype=float)
    films = np.vstack([prior, replicated])
    yvar = train_yvar_for_rows(
        pooled,
        films,
        names,
        rows_without_replicates=settings["rows_without_replicates"],
    )
    return yvar, findings


def variance_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The ``model.replicate_variance`` block, with its defaults filled in."""
    block = (config.get("model") or {}).get("replicate_variance") or {}
    if not isinstance(block, Mapping):
        raise ValueError("model.replicate_variance must be a mapping.")
    floors = block.get("sanity_floor") or {}
    if not isinstance(floors, Mapping):
        raise ValueError("model.replicate_variance.sanity_floor must be a mapping.")
    return {
        "rows_without_replicates": int(block.get("rows_without_replicates", 1)),
        "sanity_floor": {str(k): float(v) for k, v in floors.items()},
    }
