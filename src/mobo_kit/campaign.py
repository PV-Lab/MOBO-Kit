"""The campaign path: three rounds, one function each.

    run_r0_lhs(config, n=15)        -> space-filling initial worklist
    run_r1_ucb(config, ..., n=5)    -> UCB-HVI batch with local penalisation
    run_r2_qlognehvi(config, ..., n=3) -> qLogNEHVI batch

Each returns a :class:`RoundResult` carrying the proposed conditions in physical
units plus a diagnostics dict.  These functions orchestrate; the mathematics
lives in ``lhs``, ``candidate_pool``, ``ucb_hvi``, ``batch_selection``,
``qlognehvi_batch`` and ``discrete_refinement``, which are not reimplemented
here.

Two things are worth knowing before reading further.

**Thickness trains on nanometres.**  The campaign's thickness utility is a
Gaussian on a 650 nm target, and that map is 2-to-1: films at 400 nm and 900 nm
score alike from opposite sides of the peak.  Training on the score forces the
GP to represent a folded ridge; training on nanometres leaves a smooth trend.
So ``model_source_column`` (what the GP sees) and the utility transform are
declared separately per objective.  See docs/GP_MODEL_DECISION.md.

**Utility moments come from posterior samples.**  Because the thickness utility
is nonlinear, the mean of the transform is not the transform of the mean.
``ucb_hvi.posterior_utility_moments`` already applies the transform to posterior
samples, so ``moment_method="monte_carlo"`` is correct and required here; the
``analytic_identity`` fast path is only valid when every transform is identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .batch_selection import LocalPenalizationConfig
from .candidate_pool import CandidatePool, sample_discrete_candidate_pool
from .constraints import constraints_from_config
from .design import Design, build_design_from_config
from .lhs import lhs_dataframe_optimized
from botorch.models.model_list_gp_regression import ModelListGP

from .model_validation import fit_model_variant, model_variant_spec
from .structured_mean import build_structured_mean, mean_spec_from_config
from .objectives import ObjectiveSpec, ObjectiveTransform
from .qlognehvi_batch import propose_qlognehvi_penalized_batch
from .ucb_hvi import propose_ucb_hvi_batch

__all__ = [
    "CampaignConfigError",
    "EXCEL_CSV_ENCODING",
    "write_worklist_csv",
    "FIXED_SCALING_MODES",
    "assert_scaling_is_campaign_fixed",
    "RoundResult",
    "build_objective_transform",
    "expand_replicates",
    "load_campaign_config",
    "run_r0_lhs",
    "run_r1_ucb",
    "run_r2_qlognehvi",
    "validate_batch",
]


#: Encoding for CSVs a human will open in Excel.  Excel does not detect UTF-8
#: without a BOM and falls back to the system ANSI codepage, which mangles any
#: non-ASCII cell -- and does so on the reader's machine, not the writer's, so
#: it is invisible during development.  ``utf-8-sig`` writes the BOM; pandas and
#: every other reader strip it transparently.
EXCEL_CSV_ENCODING = "utf-8-sig"


def write_worklist_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a worklist CSV that Excel will open correctly."""
    destination = Path(path)
    frame.to_csv(destination, index=False, encoding=EXCEL_CSV_ENCODING)
    return destination


class CampaignConfigError(ValueError):
    """The configuration cannot support the requested round."""


class BatchValidityError(RuntimeError):
    """A proposed batch failed a validity check and must not be issued."""


@dataclass
class RoundResult:
    """One round's proposal."""

    round_name: str
    conditions: pd.DataFrame
    """Distinct proposed conditions, physical units, columns == design.names."""
    replicates: pd.DataFrame
    """One row per physical film, with candidate_id and replicate_group."""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_conditions(self) -> int:
        return len(self.conditions)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


def load_campaign_config(path: str | Path) -> dict[str, Any]:
    """Read a campaign YAML.  UTF-8 is explicit: the default codec is locale
    dependent and silently fails on non-ASCII under some Windows locales."""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise CampaignConfigError(f"{path} did not parse to a mapping.")
    return dict(config)


def _objective_specs(config: Mapping[str, Any]) -> tuple[ObjectiveSpec, ...]:
    objectives = config.get("objectives")
    if not isinstance(objectives, Mapping) or "specs" not in objectives:
        raise CampaignConfigError(
            "config['objectives'] must be a mapping containing 'specs'. A config "
            "with an empty objective list cannot propose candidates."
        )
    raw_specs = objectives["specs"]
    if not isinstance(raw_specs, Sequence) or not raw_specs:
        raise CampaignConfigError("config['objectives']['specs'] must be non-empty.")

    specs: list[ObjectiveSpec] = []
    for entry in raw_specs:
        if not isinstance(entry, Mapping):
            raise CampaignConfigError("Every objective spec must be a mapping.")
        if not entry.get("model_source_column"):
            raise CampaignConfigError(
                f"Objective {entry.get('name')!r} must declare "
                "model_source_column: the GP trains on that column, which is not "
                "always the final score."
            )
        mean_spec = mean_spec_from_config(entry)
        specs.append(
            ObjectiveSpec(
                name=str(entry["name"]),
                goal=str(entry["goal"]),
                transform=str(entry["transform"]),
                # a log-response mean function means the GP emits log(y), so the
                # utility must exponentiate and its posterior is lognormal
                model_link=(
                    "log"
                    if mean_spec is not None and mean_spec.response == "log"
                    else "identity"
                ),
                source_column=str(entry["model_source_column"]),
                lower_anchor=entry.get("lower_anchor"),
                upper_anchor=entry.get("upper_anchor"),
                target=entry.get("target"),
                sigma=entry.get("sigma"),
                scale=entry.get("scale"),
            )
        )
    return tuple(specs)


#: Scaling modes that are fixed for the whole campaign.  Anything else means the
#: scale would be re-derived from whatever data happens to exist this round.
FIXED_SCALING_MODES = frozenset({"already_normalized", "fixed_affine"})


def assert_scaling_is_campaign_fixed(config: Mapping[str, Any]) -> None:
    """Refuse objective scales that are re-derived from observed data.

    This is the single most consequential check inherited from
    ``production_gate.py``, and it is not about approval.  If an objective's
    scale moves with the data each round -- observed min/max, a percentile, a
    round-local standardisation -- then the utility space itself moves, and
    hypervolume computed in round N is not comparable with round N+1.  The
    progress plot silently stops meaning anything.

    The temptation is immediate and specific: once R1 measurements land, the
    observed ranges will look like better anchors than the declared ones.  They
    are not.  Widen a declared range deliberately and version it; never let it
    track the data.
    """
    for spec in _objective_specs(config):
        if spec.transform == "affine":
            if spec.lower_anchor is None or spec.upper_anchor is None:
                raise CampaignConfigError(
                    f"Objective {spec.name!r} uses an affine transform but does "
                    "not declare fixed lower_anchor/upper_anchor. Round-by-round "
                    "min/max scaling makes hypervolume incomparable across rounds."
                )
            if not (spec.lower_anchor < spec.upper_anchor):
                raise CampaignConfigError(
                    f"Objective {spec.name!r} requires lower_anchor < upper_anchor."
                )

    declared = (config.get("objectives") or {}).get("scaling_mode", "fixed_affine")
    if declared not in FIXED_SCALING_MODES:
        raise CampaignConfigError(
            f"objectives.scaling_mode must be one of {sorted(FIXED_SCALING_MODES)}; "
            f"got {declared!r}. Observed or data-derived scaling is forbidden "
            "because it makes hypervolume incomparable between rounds."
        )


def build_objective_transform(config: Mapping[str, Any]) -> ObjectiveTransform:
    """Build the versioned raw-output-to-utility contract from a campaign config."""
    version = config.get("objectives", {}).get("contract_version")
    if not version:
        raise CampaignConfigError(
            "config['objectives']['contract_version'] is required so that "
            "hypervolume stays comparable across rounds."
        )
    assert_scaling_is_campaign_fixed(config)
    return ObjectiveTransform(_objective_specs(config), version=str(version))


def model_source_columns(config: Mapping[str, Any]) -> tuple[str, ...]:
    """The workbook columns the GP trains on, in objective order."""
    return tuple(spec.source_column for spec in _objective_specs(config))


def _reference_point(config: Mapping[str, Any], n_objectives: int) -> np.ndarray:
    raw = config.get("reference_point_utility")
    if raw is None:
        raise CampaignConfigError(
            "reference_point_utility is required and must be declared in UTILITY "
            "space, after the objective transforms."
        )
    point = np.asarray(raw, dtype=float)
    if point.shape != (n_objectives,):
        raise CampaignConfigError(
            f"reference_point_utility must have {n_objectives} entries; "
            f"got {point.shape}."
        )
    if not np.all(np.isfinite(point)):
        raise CampaignConfigError("reference_point_utility must be finite.")
    return point


def _penalization(config: Mapping[str, Any]) -> LocalPenalizationConfig:
    raw = config.get("local_penalization") or {}
    weights = raw.get("dimension_weights")
    return LocalPenalizationConfig(
        radius=raw.get("radius"),
        min_batch_distance=float(raw.get("min_batch_distance", 0.0)),
        min_observed_distance=float(raw.get("min_observed_distance", 0.0)),
        dimension_weights=None if weights is None else np.asarray(weights, float),
    )


def _round_settings(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    rounds = config.get("rounds") or {}
    settings = rounds.get(key)
    if not isinstance(settings, Mapping):
        raise CampaignConfigError(f"config['rounds']['{key}'] is required.")
    return dict(settings)


# --------------------------------------------------------------------------- #
# validity
# --------------------------------------------------------------------------- #


def validate_batch(
    conditions: pd.DataFrame,
    design: Design,
    *,
    expected_count: int,
    min_pairwise_distance: float = 0.0,
) -> dict[str, Any]:
    """Refuse to issue a batch that is malformed.

    Five checks, all of which catch real bugs: the batch is the requested size,
    its rows are distinct, every value sits exactly on the declared grid, every
    value is finite and in bounds, and the rows are at least
    ``min_pairwise_distance`` apart in normalised space.

    This replaces the previous debug/production approval tiers.  Whether a batch
    is approved for fabrication is a human decision recorded outside the code;
    it is not something this function can compute, so it does not pretend to.
    """
    report: dict[str, Any] = {}
    values = conditions.to_numpy(dtype=float)

    report["expected_count"] = expected_count
    report["actual_count"] = len(conditions)
    if len(conditions) != expected_count:
        raise BatchValidityError(
            f"Expected exactly {expected_count} conditions; got {len(conditions)}."
        )

    if not np.all(np.isfinite(values)):
        raise BatchValidityError("Proposed conditions contain non-finite values.")
    report["finite"] = True

    duplicates = pd.DataFrame(values).duplicated().to_numpy()
    if duplicates.any():
        raise BatchValidityError(
            f"Proposed conditions must be unique; {int(duplicates.sum())} duplicate "
            "row(s) found."
        )
    report["unique"] = True

    off_grid: list[str] = []
    for j, name in enumerate(design.names):
        grid = np.asarray(design.var_array[j], dtype=float)
        for value in values[:, j]:
            if not np.any(np.isclose(grid, value, rtol=0.0, atol=1e-9)):
                off_grid.append(f"{name}={value!r}")
    if off_grid:
        raise BatchValidityError(
            "Proposed values must lie exactly on the declared grid; off-grid: "
            f"{sorted(set(off_grid))}"
        )
    report["on_grid"] = True

    lowers = np.asarray(design.lowers, dtype=float)
    uppers = np.asarray(design.uppers, dtype=float)
    if np.any(values < lowers - 1e-9) or np.any(values > uppers + 1e-9):
        raise BatchValidityError("Proposed conditions fall outside design bounds.")
    report["in_bounds"] = True

    span = np.where(uppers > lowers, uppers - lowers, 1.0)
    norm = (values - lowers) / span
    if len(norm) > 1:
        from scipy.spatial.distance import pdist

        distances = pdist(norm)
        report["min_pairwise_distance"] = float(distances.min())
        if min_pairwise_distance > 0 and distances.min() < min_pairwise_distance:
            raise BatchValidityError(
                f"Minimum pairwise distance {distances.min():.4f} is below the "
                f"configured floor {min_pairwise_distance:.4f}."
            )
    else:
        report["min_pairwise_distance"] = float("inf")

    boundary = (np.isclose(norm, 0.0, atol=1e-9)) | (np.isclose(norm, 1.0, atol=1e-9))
    report["boundary_coords_per_condition"] = boundary.sum(axis=1).tolist()
    return report


def expand_replicates(
    conditions: pd.DataFrame, *, replicates: int, round_name: str
) -> pd.DataFrame:
    """One row per physical film, sharing a replicate_group per condition.

    The experimentalists run each proposed condition ``replicates`` times to
    measure reproducibility.  Those films are separate experimental rows, but
    they are one design point: aggregate them to a condition-level mean before
    the next round trains on them, and pool their within-condition variance
    across conditions for an observation-noise estimate.
    """
    if replicates < 1:
        raise ValueError("replicates must be at least 1.")
    rows = []
    for index, (_, condition) in enumerate(conditions.iterrows(), start=1):
        candidate_id = f"{round_name}_C{index:02d}"
        for replicate in range(1, replicates + 1):
            row = dict(condition)
            row["candidate_id"] = candidate_id
            row["replicate_group"] = candidate_id
            row["replicate_index"] = replicate
            row["round"] = round_name
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# rounds
# --------------------------------------------------------------------------- #


def run_r0_lhs(
    config: Mapping[str, Any], *, n: int = 15, seed: int | None = None
) -> RoundResult:
    """Space-filling initial worklist.  No model is involved."""
    design = build_design_from_config(dict(config))
    constraints = constraints_from_config(dict(config), design)
    resolved_seed = (
        int(config.get("reproducibility", {}).get("seed", 0)) if seed is None else seed
    )
    conditions = lhs_dataframe_optimized(
        design,
        n,
        seed=resolved_seed,
        snap_to_grids=True,
        row_constraints=constraints or None,
    )
    replicates_per = int(
        _round_settings(config, "r1").get("replicates_per_condition", 1)
    )
    report = validate_batch(conditions, design, expected_count=n)
    return RoundResult(
        round_name="R0",
        conditions=conditions,
        replicates=expand_replicates(
            conditions, replicates=replicates_per, round_name="R0"
        ),
        diagnostics={"seed": resolved_seed, "validity": report, "method": "lhs"},
    )


def _fit_models(
    config: Mapping[str, Any],
    X_phys: np.ndarray,
    X_norm: np.ndarray,
    Y_raw: np.ndarray,
    seed: int,
) -> Any:
    """One GP per objective, each with its declared structured mean.

    Objectives with a ``mean_function`` train on the response-space target with
    the OLS trend frozen into the mean module, so ``posterior()`` already carries
    it and no caller has to add it back.  Objectives without one are unchanged.
    """
    variant = model_variant_spec(str(config.get("model", {}).get("variant")))
    specs = _objective_specs(config)
    entries = config["objectives"]["specs"]
    design = build_design_from_config(dict(config))
    lowers = np.asarray(design.lowers, dtype=float)
    uppers = np.asarray(design.uppers, dtype=float)
    names = list(design.names)

    models = []
    for index, (spec, entry) in enumerate(zip(specs, entries)):
        mean_spec = mean_spec_from_config(entry)
        y = np.asarray(Y_raw, dtype=float)[:, index]
        mean_module = None
        if mean_spec is not None:
            mean_module, y = build_structured_mean(
                X_phys, y, mean_spec, names, lowers, uppers
            )
        torch.manual_seed(seed)
        record = fit_model_variant(
            torch.tensor(X_norm, dtype=torch.double),
            torch.tensor(y, dtype=torch.double).unsqueeze(-1),
            sample_ids=tuple(range(len(X_norm))),
            objective_names=(spec.name,),
            variant=variant,
            seed=seed,
            mean_module=mean_module,
        )
        models.append(record.model.models[0])
    return ModelListGP(*models)


def _normalise(design: Design, X_phys: np.ndarray) -> np.ndarray:
    lowers = np.asarray(design.lowers, dtype=float)
    uppers = np.asarray(design.uppers, dtype=float)
    span = np.where(uppers > lowers, uppers - lowers, 1.0)
    return (np.asarray(X_phys, dtype=float) - lowers) / span


def _on_grid_mask(design: Design, X_phys: np.ndarray) -> np.ndarray:
    """Which observed rows sit exactly on the declared grid.

    The R0 control is a declared off-grid exception (anti_time = 12 against a
    9/11/13... grid).  It stays in the GP and in the distance references, but it
    cannot take part in grid-index bookkeeping, and it does not need to: an
    off-grid point can never collide with a pool candidate by construction.
    """
    values = np.asarray(X_phys, dtype=float)
    mask = np.ones(len(values), dtype=bool)
    for j in range(values.shape[1]):
        grid = np.asarray(design.var_array[j], dtype=float)
        for i, value in enumerate(values[:, j]):
            if not np.any(np.isclose(grid, value, rtol=0.0, atol=1e-9)):
                mask[i] = False
    return mask


def run_r1_ucb(
    config: Mapping[str, Any],
    observed_X_phys: np.ndarray,
    observed_Y_raw: np.ndarray,
    *,
    n: int | None = None,
    seed: int | None = None,
) -> RoundResult:
    """UCB-HVI batch with local penalisation.

    ``observed_Y_raw`` holds the MODEL SOURCE values in objective order, not the
    final scores: see :func:`model_source_columns`.  For this campaign that means
    thickness arrives in nanometres.
    """
    design = build_design_from_config(dict(config))
    settings = _round_settings(config, "r1")
    q = int(settings["batch_size"]) if n is None else int(n)
    resolved_seed = (
        int(config.get("reproducibility", {}).get("seed", 0)) if seed is None else seed
    )
    transform = build_objective_transform(config)
    reference = _reference_point(config, transform.objective_count)
    penalization = _penalization(config)

    observed_norm = _normalise(design, observed_X_phys)
    model = _fit_models(
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

    proposal = propose_ucb_hvi_batch(
        pool,
        model,
        observed_Y_raw,
        transform,
        reference,
        q=q,
        beta=float(settings.get("beta", 4.0)),
        local_penalization_config=penalization,
        mc_samples=int(settings.get("posterior_samples", 256)),
        seed=resolved_seed,
        moment_method=str(settings.get("moment_method", "monte_carlo")),
    )

    conditions = pd.DataFrame(
        np.asarray(proposal.selection.X_phys, dtype=float), columns=list(design.names)
    )
    report = validate_batch(
        conditions,
        design,
        expected_count=q,
        min_pairwise_distance=penalization.min_batch_distance,
    )
    replicates_per = int(settings.get("replicates_per_condition", 1))
    return RoundResult(
        round_name="R1",
        conditions=conditions,
        replicates=expand_replicates(
            conditions, replicates=replicates_per, round_name="R1"
        ),
        diagnostics={
            "method": "ucb_hvi",
            "seed": resolved_seed,
            "beta": float(settings.get("beta", 4.0)),
            "pool_size": pool.size,
            "objective_contract": transform.version,
            "moment_method": str(settings.get("moment_method", "monte_carlo")),
            "off_grid_observations_excluded_from_pool_bookkeeping": int(
                (~on_grid).sum()
            ),
            "validity": report,
        },
    )


def run_r2_qlognehvi(
    config: Mapping[str, Any],
    observed_X_phys: np.ndarray,
    observed_Y_raw: np.ndarray,
    *,
    n: int | None = None,
    seed: int | None = None,
) -> RoundResult:
    """qLogNEHVI batch.  ``observed_Y_raw`` follows the same contract as R1.

    qLogNEHVI is the numerically stable formulation of qNEHVI and is the correct
    choice here; it is not a deviation from the brief.
    """
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
    model = _fit_models(
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

    from .objectives import ConfiguredMCMultiOutputObjective

    proposal = propose_qlognehvi_penalized_batch(
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
        np.asarray(proposal.selection.X_phys, dtype=float), columns=list(design.names)
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
            "method": "qlognehvi",
            "seed": resolved_seed,
            "pool_size": pool.size,
            "objective_contract": transform.version,
            "off_grid_observations_excluded_from_pool_bookkeeping": int(
                (~on_grid).sum()
            ),
            "validity": report,
        },
    )
