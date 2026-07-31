"""Simulate the campaign loop against a frozen GP oracle, and plot what it did.

Derived from Annie Xu's ``examples/round_simulations.py`` on her
``ax_plots_simulation`` branch, which established the approach, the output
directory convention (``{pair}/qlognehvi/radius_*__beta_*/``), the round legend,
and the boxplot-with-overlaid-points figure.  See ``docs/ROUND_SIM_DELTA.md`` for
what changed between her branch and this one, and why.

WHAT THIS IS.  There is exactly one round of real measurements (15 R0 films), so
the loop from R0 to R2 has never been run end to end on this campaign.  This
script runs it against an *oracle*: a GP fitted once on the 15 real observations,
then frozen and used to answer "what would this recipe have measured?" for every
condition the optimiser proposes.

WHAT IT IS NOT.  The oracle is a model, so every number downstream of it is a
model prediction.  A condition that scores well here has scored well against
MOBO-Kit's own beliefs -- which is a test of the optimiser loop on a data-shaped
landscape, and is not evidence about the chemistry.  Both figures and manifest say
so; do not quote a thickness from this script as a measurement.

THE LOOP, per parameter cell::

    GP_exp        fit once on the 15 real rows (fit_campaign_models, seed 73)
      -> R0       the 15 REAL recipes, re-scored by the oracle
      -> R1       UCB-HVI, 5 conditions, this cell's beta and radius
      -> oracle   score them
      -> R2       qLogNEHVI, 3 conditions
      -> oracle   score them
      -> final GP refit on all 23, which is what the heatmaps render

qLogNEHVI only.  It is the numerically stable formulation of qNEHVI and the one
``campaign.py`` ships; Annie's branch carried a ``run_r2_qnehvi`` alternative,
which is deliberately not used here.

THE R1 BASELINE, and why the manifest carries three numbers for it.  When this
script was written, ``campaign.run_r1_ucb`` handed its observed HVI baseline to
the objective transform in the WRONG SPACE -- measurement-space nanometres to a
transform that applies ``exp()`` to log-link objectives.  That pinned every
observation's thickness utility to exactly 0.0 and made the baseline hypervolume
**0.004659 against a true 0.436442**.  The script carried its own corrected R1
until the defect was fixed in ``campaign.py`` (commit ``4b76670``, promoting the
fix Annie's branch already carried as ``_physical_to_model_output``).

It now calls the public ``run_r1_ucb``, verified to reproduce the private
version's batches hash-for-hash, and keeps the contrast in the manifest as a
standing tripwire: the baseline the acquisition REPORTS must equal the one
recomputed here by an independent route, and both must stay far away from the
unencoded value.  The assertion runs on every cell of every sweep.

Usage::

    python scripts/plot_round_simulation.py --workbook "local_inputs/Summary Table.xlsx"
    python scripts/plot_round_simulation.py --workbook ... --no-figures    # manifest only
    python scripts/plot_round_simulation.py --workbook ... --pairs speed_1,precur_conc
    python scripts/plot_round_simulation.py --workbook ... --full-grid     # 45 cells
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import warnings
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from mobo_kit.campaign import (  # noqa: E402
    build_objective_transform,
    fit_campaign_models,
    load_campaign_config,
    normalise_inputs,
    objective_names,
    run_r1_ucb,
    run_r2_qlognehvi,
)
from mobo_kit.design import Design, build_design_from_config  # noqa: E402
from mobo_kit.metrics import compute_ref_pareto_hv  # noqa: E402
from mobo_kit.objectives import ObjectiveTransform  # noqa: E402
from mobo_kit.workbook_io import read_campaign_workbook  # noqa: E402

# Scoped rather than blanket, exactly as scripts/plot_dtlz2_report.py does it: the
# GP fits emit numerical and deprecation chatter that would bury a real message.
# Anything the fit guard says still comes through, which is the point.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="botorch")
warnings.filterwarnings("ignore", category=UserWarning, module="gpytorch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
torch.set_num_threads(1)

# --------------------------------------------------------------------------- #
# figure style -- the scripts/plot_dtlz2_report.py palette, so every figure this
# project ships reads as one set
# --------------------------------------------------------------------------- #
R0_COLOR, R1_COLOR, R2_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SPINE = "#d8d7d2"

ROUND_STYLE = {
    "R0": (R0_COLOR, "R0 LHS (GP_exp scored)"),
    "R1": (R1_COLOR, "R1 simulated"),
    "R2": (R2_COLOR, "R2 simulated"),
}

#: Every figure carries two caveat lines.  The oracle caveat is on all of them --
#: it is the one that silently produces a wrong conclusion.  The other line is
#: whichever caveat is TRUE of that figure: a slice figure gets the slice caveat,
#: which is the one that silently produces a wrong reading; a round summary gets
#: the small-n caveat instead.
#:
#: The brief asked for one fixed two-line footer everywhere.  Printing the slice
#: caveat on a boxplot, which has no slice, would be a false statement in the
#: place a reader looks for true ones -- and this project's own rule (see
#: CAMPAIGN_STATUS.md issue 6) is that padding a warning channel with
#: inapplicable text is how people learn to ignore it.  Both lines are still
#: fixed and still on every figure.
SLICE_CAVEAT = (
    "Slice: the other 8 inputs are held at the fixed values named above. Plotted "
    "points are shown at their own (x, y) only -- their remaining coordinates are "
    "generally NOT on this slice."
)
ROUND_N_CAVEAT = (
    "Small n: the rounds hold 15 / 5 / 3 points. A box over three numbers reports "
    "little more than those numbers, which is why every raw point is drawn on top."
)
ORACLE_CAVEAT = (
    "Oracle: surface and point values are predictions from a GP fitted to 15 real "
    "films, not measurements. This validates the optimiser loop on a data-shaped "
    "landscape, not the chemistry."
)

#: OFAT, per the brief.  The two arms share the (0.25, 4.0) cell, so the union is
#: 13 distinct cells rather than 14.
OFAT_RADII = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
OFAT_BETAS = (1.0, 4.0, 9.0, 16.0, 25.0)
ANCHOR_RADIUS, ANCHOR_BETA = 0.25, 4.0

#: The parameter sweep pinned this and so does the brief, so that "spacing" means
#: one thing in every cell.  It is NOT swept, and a cell that changed it would not
#: be comparable with the others.
PINNED_MIN_BATCH_DISTANCE = 0.15


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _safe_filename(value: str) -> str:
    """Annie's slug rule, kept so her output paths stay recognisable."""
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in str(value))
    return "_".join(part for part in cleaned.split("_") if part)


def _slug_number(value: float) -> str:
    """Annie's number slug: 0.25 -> '0p25', 4.0 -> '4'."""
    return f"{value:g}".replace("-", "m").replace(".", "p")


def cell_slug(radius: float, beta: float) -> str:
    return f"radius_{_slug_number(radius)}__beta_{_slug_number(beta)}"


def to_model_space(Y_physical: np.ndarray, transform: ObjectiveTransform) -> np.ndarray:
    """Measurement space -> model space, as a numpy convenience.

    Delegates to ``ObjectiveTransform.encode_measurements``, which is the public
    contract for this step since commit ``4b76670``.  It exists as a separate
    function here only because the rest of this script works in numpy.
    """
    values = torch.tensor(np.asarray(Y_physical, dtype=float), dtype=torch.double)
    with torch.no_grad():
        return transform.encode_measurements(values).detach().cpu().numpy()


def utilities(Y_physical: np.ndarray, transform: ObjectiveTransform) -> np.ndarray:
    """Campaign utility from measurement-space values.  Higher is better, always."""
    values = torch.tensor(np.asarray(Y_physical, dtype=float), dtype=torch.double)
    with torch.no_grad():
        return transform.transform_measurements(values).detach().cpu().numpy()


def hypervolume(Y_physical: np.ndarray, transform: ObjectiveTransform,
                reference: np.ndarray) -> float:
    """Hypervolume at the campaign's declared reference, in utility space."""
    U = torch.tensor(utilities(Y_physical, transform), dtype=torch.double)
    _ref, _pareto, volume = compute_ref_pareto_hv(U, reference)
    return float(volume)


def batch_hash(conditions: pd.DataFrame) -> str:
    """Order-independent identity of a proposed batch.

    Sorted before hashing because the question the manifest asks is "did these two
    cells propose the same SET of conditions", not "in the same order".  Rounded to
    12 decimals so a float representation difference cannot masquerade as a
    different batch.
    """
    values = np.round(np.asarray(conditions, dtype=float), 12)
    ordered = values[np.lexsort(values.T[::-1])]
    return hashlib.sha256(ordered.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# the oracle
# --------------------------------------------------------------------------- #


def oracle_predict(
    model: Any,
    config: Mapping[str, Any],
    X_phys: np.ndarray,
    transform: ObjectiveTransform,
) -> np.ndarray:
    """Deterministic measurement-space prediction for each row of ``X_phys``.

    Returns values in the same space the workbook reports and the campaign trains
    on: nanometres for thickness, the score itself for the other two.

    THICKNESS IS THE POSTERIOR MEDIAN, ``exp(mu)``, and is labelled median
    everywhere.  Two other choices were considered and rejected:

    * ``exp(mu + v/2)`` is the lognormal *mean*, and is what Annie's branch used.
      It is the correct mean, and her "physical mean" colorbar label was accurate
      for it.  It is nonetheless the wrong choice for an ORACLE, because it makes
      the oracle's value a function of the posterior VARIANCE -- which is large
      wherever the 15 real films are sparse.  The simulated ground truth would then
      bulge in exactly the regions the optimiser is about to explore, and the
      landscape would encode where R0 happened to look rather than what the model
      believes.  ``exp(mu)`` depends on the mean surface alone.
    * Drawing a posterior sample makes the oracle stochastic, so two cells that
      propose the same batch could still be scored differently, and the batch
      identity question this sweep exists to answer would be unanswerable.

    Determinism matters beyond tidiness: the manifest compares batches ACROSS
    cells, and that comparison is only meaningful if the oracle is a fixed
    function.
    """
    X_norm = normalise_inputs(config, np.asarray(X_phys, dtype=float))
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(
            torch.tensor(X_norm, dtype=torch.double), observation_noise=False
        )
        mean = posterior.mean.detach().cpu().double().numpy()
    if mean.ndim != 2 or mean.shape[1] != transform.objective_count:
        raise RuntimeError(f"Oracle posterior mean has unexpected shape {mean.shape}.")
    out = np.empty_like(mean)
    for index, spec in enumerate(transform.specs):
        out[:, index] = np.exp(mean[:, index]) if spec.model_link == "log" else mean[:, index]
    return out


# --------------------------------------------------------------------------- #
# one parameter cell
# --------------------------------------------------------------------------- #


def cell_config(base: Mapping[str, Any], *, radius: float, beta: float) -> dict[str, Any]:
    """Base config with this cell's two knobs set and min_batch_distance pinned."""
    config = deepcopy(dict(base))
    penalization = config.setdefault("local_penalization", {})
    penalization["radius"] = float(radius)
    penalization["min_batch_distance"] = PINNED_MIN_BATCH_DISTANCE
    config.setdefault("rounds", {}).setdefault("r1", {})["beta"] = float(beta)
    return config


def run_cell(
    base_config: Mapping[str, Any],
    oracle: Any,
    X_r0: np.ndarray,
    transform: ObjectiveTransform,
    reference: np.ndarray,
    *,
    radius: float,
    beta: float,
    seed: int,
) -> dict[str, Any]:
    """R0 -> R1 -> R2 for one (radius, beta), everything scored by the oracle."""
    config = cell_config(base_config, radius=radius, beta=beta)

    # R0: the REAL 15 recipes, re-scored by the oracle so the whole loop lives on
    # one consistent landscape. Using the real measured Y here instead would mix a
    # measured R0 with a simulated R1/R2 and make the round comparison incoherent.
    Y_r0 = oracle_predict(oracle, config, X_r0, transform)

    r1 = run_r1_ucb(config, X_r0, Y_r0, seed=seed)
    r1_warnings = tuple(r1.diagnostics.get("model_fit_warnings", ()))

    # STANDING TRIPWIRE for the defect this script was written alongside.
    # run_r1_ucb reports the HVI baseline it actually used; `hypervolume` recomputes
    # it here through metrics.compute_ref_pareto_hv, a different Pareto filter and a
    # different call path. They agree only if the observed values were encoded into
    # model space before being transformed. If that encoding is ever dropped again,
    # this fires on the first cell of the next sweep instead of quietly producing a
    # plausible manifest.
    reported_baseline = float(r1.diagnostics["observed_baseline_hypervolume"])
    independent_baseline = hypervolume(Y_r0, transform, reference)
    if not math.isclose(reported_baseline, independent_baseline, rel_tol=1e-9):
        raise RuntimeError(
            "The R1 acquisition's observed baseline does not match an independent "
            f"computation: reported {reported_baseline!r} against "
            f"{independent_baseline!r}. The most likely cause is measurement-space "
            "values reaching ObjectiveTransform.transform without going through "
            "encode_measurements first -- see docs/ROUND_SIM_DELTA.md."
        )

    X_r1 = r1.conditions.to_numpy(dtype=float)
    Y_r1 = oracle_predict(oracle, config, X_r1, transform)

    X_01 = np.vstack([X_r0, X_r1])
    Y_01 = np.vstack([Y_r0, Y_r1])

    r2 = run_r2_qlognehvi(config, X_01, Y_01, seed=seed)
    X_r2 = r2.conditions.to_numpy(dtype=float)
    Y_r2 = oracle_predict(oracle, config, X_r2, transform)

    X_all = np.vstack([X_01, X_r2])
    Y_all = np.vstack([Y_01, Y_r2])

    # The model the heatmaps render: refitted on all 23 oracle-scored points.
    final_model, final_warnings = fit_campaign_models(config, X_all, Y_all, seed=seed)

    return {
        "radius": float(radius),
        "beta": float(beta),
        "slug": cell_slug(radius, beta),
        "config": config,
        "X": {"R0": X_r0, "R1": X_r1, "R2": X_r2, "all": X_all},
        "Y": {"R0": Y_r0, "R1": Y_r1, "R2": Y_r2, "all": Y_all},
        "U": {
            "R0": utilities(Y_r0, transform),
            "R1": utilities(Y_r1, transform),
            "R2": utilities(Y_r2, transform),
        },
        "r1": r1,
        "r2": r2,
        "final_model": final_model,
        "final_fit_warnings": tuple(final_warnings),
        "r1_fit_warnings": r1_warnings,
        "r2_fit_warnings": tuple(r2.diagnostics.get("model_fit_warnings", ())),
        "baseline": {
            "reported": reported_baseline,
            "independent": independent_baseline,
            "pareto_size": int(r1.diagnostics["observed_baseline_pareto_size"]),
        },
        "hv": {
            "R0": hypervolume(Y_r0, transform, reference),
            "R0+R1": hypervolume(Y_01, transform, reference),
            "R0+R1+R2": hypervolume(Y_all, transform, reference),
        },
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def fixed_slice_values(design: Design, X_r0: np.ndarray) -> np.ndarray:
    """Median of the 15 R0 values per input, snapped onto the declared grid.

    Median rather than mean: a mean can land between grid values in a way that no
    recipe could realise, and the campaign's own diagnostics use the median.  The
    snap keeps the held-fixed slice a recipe the group could actually run.
    """
    fixed = np.median(np.asarray(X_r0, dtype=float), axis=0)
    for index, grid in enumerate(design.var_array):
        allowed = np.asarray(grid, dtype=float)
        fixed[index] = allowed[np.argmin(np.abs(allowed - fixed[index]))]
    return fixed


def surface_grid(
    model: Any,
    config: Mapping[str, Any],
    design: Design,
    transform: ObjectiveTransform,
    pair: tuple[str, str],
    fixed: np.ndarray,
    *,
    points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior median surface over one input pair, other inputs held at ``fixed``."""
    names = list(design.names)
    xi, yi = names.index(pair[0]), names.index(pair[1])
    x_values = np.linspace(design.lowers[xi], design.uppers[xi], points)
    y_values = np.linspace(design.lowers[yi], design.uppers[yi], points)
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)

    rows = np.repeat(fixed[None, :], mesh_x.size, axis=0)
    rows[:, xi] = mesh_x.ravel()
    rows[:, yi] = mesh_y.ravel()

    # normalise_inputs, not an observed-range rescale: the model was told about
    # config-grid coordinates and must be asked about the same ones.
    X_norm = normalise_inputs(config, rows)
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(
            torch.tensor(X_norm, dtype=torch.double), observation_noise=False
        )
        mean = posterior.mean.detach().cpu().double().numpy()

    surfaces = np.empty((mesh_x.size, transform.objective_count))
    for index, spec in enumerate(transform.specs):
        surfaces[:, index] = (
            np.exp(mean[:, index]) if spec.model_link == "log" else mean[:, index]
        )
    return mesh_x, mesh_y, surfaces.reshape(*mesh_x.shape, transform.objective_count)


def _footer(
    fig: plt.Figure, seed: int, first_caveat: str, extra: str | None = None
) -> None:
    text = f"{first_caveat}\n{ORACLE_CAVEAT}"
    if extra is not None:
        text = f"{extra}\n{text}"
    fig.text(
        0.008,
        0.008,
        f"seed {seed}  |  {text}",
        fontsize=6.4,
        color=INK_MUTED,
        va="bottom",
        ha="left",
        wrap=True,
    )


def _objective_axis_label(spec: Any) -> str:
    if spec.model_link == "log":
        return f"{spec.name} -- posterior median (nm)"
    return f"{spec.name} -- posterior mean"


def plot_surface(
    path: Path,
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    surface: np.ndarray,
    pair: tuple[str, str],
    spec: Any,
    rounds: Mapping[str, np.ndarray],
    design: Design,
    fixed: np.ndarray,
    *,
    radius: float,
    beta: float,
    seed: int,
    warning_banner: str | None,
) -> None:
    names = list(design.names)
    xi, yi = names.index(pair[0]), names.index(pair[1])

    fig, axis = plt.subplots(figsize=(8.8, 7.4), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    filled = axis.contourf(mesh_x, mesh_y, surface, levels=16, cmap=SEQ)
    bar = fig.colorbar(filled, ax=axis, fraction=0.046, pad=0.03)
    bar.set_label(_objective_axis_label(spec), fontsize=8.5, color=INK_MUTED)
    bar.ax.tick_params(labelsize=7, colors=INK_MUTED)

    # R0's categorical colour is the same blue the magnitude ramp is built from, so
    # on the dark end of the surface a plain blue marker disappears into it. A white
    # stroke around a dark marker edge reads on both ends of the ramp; a single
    # white edge does not, which is what the first draft of this figure showed.
    halo = [path_effects.withStroke(linewidth=3.0, foreground="white")]
    for round_name in ("R0", "R1", "R2"):
        points = rounds[round_name]
        colour, label = ROUND_STYLE[round_name]
        axis.scatter(
            points[:, xi],
            points[:, yi],
            s=70 if round_name == "R0" else 104,
            c=colour,
            edgecolors=INK,
            linewidths=0.9,
            path_effects=halo,
            zorder=3 + ("R0", "R1", "R2").index(round_name),
            label=f"{label} (n={len(points)})",
        )

    fixed_text = ", ".join(
        f"{name}={fixed[index]:g}"
        for index, name in enumerate(names)
        if name not in pair
    )
    axis.set_xlabel(pair[0], fontsize=9.5, color=INK_MUTED)
    axis.set_ylabel(pair[1], fontsize=9.5, color=INK_MUTED)
    axis.tick_params(labelsize=8, colors=INK_MUTED, length=3)
    for spine in axis.spines.values():
        spine.set_color(SPINE)
    axis.legend(
        fontsize=8, frameon=True, facecolor="white", edgecolor=SPINE, loc="best"
    )

    fig.suptitle(
        f"Final GP after R0+R1+R2 -- {spec.name}   |   radius {radius:g}, beta {beta:g}",
        fontsize=12, color=INK, y=0.985,
    )
    axis.set_title(
        "other 8 inputs fixed at the median of the 15 R0 values, snapped to grid:\n"
        + fixed_text,
        fontsize=7.6, color=INK_MUTED, pad=8,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.96))
    _footer(fig, seed, SLICE_CAVEAT, warning_banner)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_boxplots(
    path: Path,
    cell: Mapping[str, Any],
    transform: ObjectiveTransform,
    *,
    seed: int,
    warning_banner: str | None,
) -> None:
    """Utility by round, three panels, with every raw point drawn over the box.

    The overlay is not decoration.  R2 has n = 3: a box drawn from three numbers
    reports quartiles that are essentially the three numbers, and reads as a
    distribution when it is a handful of points.  Annie's branch drew the points
    for the same reason and the convention is kept.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.4), facecolor=SURFACE)
    rng = np.random.default_rng(0)

    for index, spec in enumerate(transform.specs):
        axis = axes[index]
        axis.set_facecolor(SURFACE)
        series = [cell["U"][name][:, index] for name in ("R0", "R1", "R2")]
        boxes = axis.boxplot(
            series,
            tick_labels=[
                f"{name}\nn={len(series[position])}"
                for position, name in enumerate(("R0", "R1", "R2"))
            ],
            showmeans=True,
            # Every raw point is drawn below, so matplotlib's flier markers would
            # draw a second, differently-styled copy of the same observation.
            showfliers=False,
            widths=0.55,
            patch_artist=True,
        )
        for patch, name in zip(boxes["boxes"], ("R0", "R1", "R2")):
            patch.set_facecolor(ROUND_STYLE[name][0])
            patch.set_alpha(0.22)
            patch.set_edgecolor(ROUND_STYLE[name][0])
        for key in ("whiskers", "caps", "medians"):
            for artist in boxes[key]:
                artist.set_color(INK_MUTED)
        # the default mean marker is green, which is R2's categorical colour
        for marker in boxes.get("means", ()):
            marker.set_markerfacecolor(INK)
            marker.set_markeredgecolor(INK)
            marker.set_markersize(6)

        for position, values in enumerate(series, start=1):
            colour = ROUND_STYLE[("R0", "R1", "R2")[position - 1]][0]
            axis.scatter(
                np.full(values.shape, position) + rng.normal(0, 0.045, values.shape),
                values,
                s=34, c=colour, edgecolors="white", linewidths=0.8, zorder=3,
            )
        axis.set_title(spec.name, fontsize=10.5, color=INK, pad=6)
        axis.set_ylabel("utility (higher is better)", fontsize=9, color=INK_MUTED)
        axis.tick_params(labelsize=8, colors=INK_MUTED, length=3)
        axis.grid(axis="y", color="#e8e7e2", lw=0.8)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_color(SPINE)

    fig.suptitle(
        f"Objective utility by round   |   radius {cell['radius']:g}, "
        f"beta {cell['beta']:g}",
        fontsize=12.5, color=INK, y=0.98,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    _footer(fig, seed, ROUND_N_CAVEAT, warning_banner)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_hypervolume(
    path: Path,
    cell: Mapping[str, Any],
    reference: np.ndarray,
    *,
    seed: int,
    warning_banner: str | None,
) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 5.0), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    stages = ("R0", "R0+R1", "R0+R1+R2")
    values = [cell["hv"][stage] for stage in stages]
    axis.plot(
        range(len(stages)), values, color=R1_COLOR, lw=2.0, marker="o", ms=9,
        markeredgecolor="white", markeredgewidth=1.4, zorder=3,
    )
    for position, value in enumerate(values):
        axis.annotate(
            f"{value:.4f}", (position, value), textcoords="offset points",
            xytext=(0, 11), ha="center", fontsize=8.5, color=INK,
        )
    axis.set_xticks(range(len(stages)))
    axis.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(stages, (15, 20, 23))],
                         fontsize=9)
    axis.set_ylabel("hypervolume (utility space)", fontsize=9.5, color=INK_MUTED)
    axis.tick_params(labelsize=8, colors=INK_MUTED, length=3)
    axis.grid(axis="y", color="#e8e7e2", lw=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(SPINE)
    axis.set_title(
        f"Cumulative hypervolume   |   radius {cell['radius']:g}, "
        f"beta {cell['beta']:g}\n"
        f"reference {np.asarray(reference).tolist()} (campaign-fixed, utility space)",
        fontsize=10.5, color=INK, pad=10,
    )
    # Cumulative hypervolume rises monotonically by construction: adding points can
    # only grow a Pareto front. This panel shows the size of each step, not that
    # optimisation happened.
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    _footer(fig, seed, ROUND_N_CAVEAT, warning_banner)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

MANIFEST_COLUMNS: tuple[str, ...] = (
    "condition_id",
    "arm",
    "radius",
    "beta",
    "min_batch_distance",
    "seed",
    "r1_batch_hash",
    "r2_batch_hash",
    "r1_min_pairwise_distance",
    "r2_min_pairwise_distance",
    "r1_boundary_coords_total",
    "r2_boundary_coords_total",
    "r1_boundary_coords_per_condition",
    "r2_boundary_coords_per_condition",
    "hv_r0",
    "hv_r0_r1",
    "hv_r0_r1_r2",
    "hv_gain_r1",
    "hv_gain_r2",
    "baseline_hv_reported_by_r1",
    "baseline_hv_independent",
    "baseline_hv_pareto_size",
    "baseline_hv_unencoded_contrast",
    "r1_fit_warnings",
    "r2_fit_warnings",
    "final_fit_warnings",
    "mean_utility_r0",
    "mean_utility_r1",
    "mean_utility_r2",
)


def manifest_row(
    cell: Mapping[str, Any],
    *,
    condition_id: int,
    arm: str,
    seed: int,
    baseline_unencoded: float,
) -> dict[str, Any]:
    r1_validity = cell["r1"].diagnostics["validity"]
    r2_validity = cell["r2"].diagnostics["validity"]
    return {
        "condition_id": condition_id,
        "arm": arm,
        "radius": cell["radius"],
        "beta": cell["beta"],
        "min_batch_distance": PINNED_MIN_BATCH_DISTANCE,
        "seed": seed,
        "r1_batch_hash": batch_hash(cell["r1"].conditions),
        "r2_batch_hash": batch_hash(cell["r2"].conditions),
        "r1_min_pairwise_distance": float(r1_validity["min_pairwise_distance"]),
        "r2_min_pairwise_distance": float(r2_validity["min_pairwise_distance"]),
        "r1_boundary_coords_total": int(sum(r1_validity["boundary_coords_per_condition"])),
        "r2_boundary_coords_total": int(sum(r2_validity["boundary_coords_per_condition"])),
        "r1_boundary_coords_per_condition": json.dumps(
            r1_validity["boundary_coords_per_condition"]
        ),
        "r2_boundary_coords_per_condition": json.dumps(
            r2_validity["boundary_coords_per_condition"]
        ),
        "hv_r0": cell["hv"]["R0"],
        "hv_r0_r1": cell["hv"]["R0+R1"],
        "hv_r0_r1_r2": cell["hv"]["R0+R1+R2"],
        "hv_gain_r1": cell["hv"]["R0+R1"] - cell["hv"]["R0"],
        "hv_gain_r2": cell["hv"]["R0+R1+R2"] - cell["hv"]["R0+R1"],
        "baseline_hv_reported_by_r1": cell["baseline"]["reported"],
        "baseline_hv_independent": cell["baseline"]["independent"],
        "baseline_hv_pareto_size": cell["baseline"]["pareto_size"],
        "baseline_hv_unencoded_contrast": baseline_unencoded,
        "r1_fit_warnings": len(cell["r1_fit_warnings"]),
        "r2_fit_warnings": len(cell["r2_fit_warnings"]),
        "final_fit_warnings": len(cell["final_fit_warnings"]),
        "mean_utility_r0": float(cell["U"]["R0"].mean()),
        "mean_utility_r1": float(cell["U"]["R1"].mean()),
        "mean_utility_r2": float(cell["U"]["R2"].mean()),
    }


def rounds_frame(
    cell: Mapping[str, Any], names: Sequence[str], transform: ObjectiveTransform
) -> pd.DataFrame:
    """All 23 simulated design points, one row each, labelled by round.

    23 = 15 R0 + 5 R1 + 3 R2, distinct CONDITIONS rather than films.  The campaign
    runs each condition in triplicate, so the same 23 rows correspond to 39 films;
    plotting films would overplot three identical markers per condition.
    """
    frames = []
    for round_name in ("R0", "R1", "R2"):
        frame = pd.DataFrame(cell["X"][round_name], columns=list(names))
        frame.insert(0, "round", round_name)
        for index, spec in enumerate(transform.specs):
            frame[f"oracle_{spec.name}"] = cell["Y"][round_name][:, index]
            frame[f"utility_{spec.name}"] = cell["U"][round_name][:, index]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_expectations(
    manifest: pd.DataFrame,
    cells: Sequence[Mapping[str, Any]],
    transform: ObjectiveTransform,
) -> list[dict[str, Any]]:
    """The three pre-registered expectations, stated before the run and checked after.

    Each returns a verdict of HELD / FAILED plus the evidence, so a reader can
    disagree with the rule rather than only with the conclusion.
    """
    results: list[dict[str, Any]] = []

    # 1. Uniformity is exploration-only: it has no learnable signal from the 15 real
    #    rows (permutation p = 0.82), so the campaign should not be able to climb it.
    #    RULE: held if uniformity's mean-utility change from R0 to R2 is smaller in
    #    magnitude, median across cells, than BOTH other objectives'.
    index_by_name = {spec.name: i for i, spec in enumerate(transform.specs)}
    deltas: dict[str, list[float]] = {name: [] for name in index_by_name}
    for cell in cells:
        for name, index in index_by_name.items():
            deltas[name].append(
                float(cell["U"]["R2"][:, index].mean() - cell["U"]["R0"][:, index].mean())
            )
    medians = {name: float(np.median(values)) for name, values in deltas.items()}
    others = [abs(medians[n]) for n in medians if n != "uniformity"]
    held = abs(medians.get("uniformity", 0.0)) < min(others) if others else False
    results.append({
        "expectation": "uniformity is flat (exploration-only, no learnable signal)",
        "rule": "|median delta R0->R2| smallest of the three objectives",
        "verdict": "HELD" if held else "FAILED",
        "evidence": ", ".join(f"{n} {v:+.4f}" for n, v in medians.items()),
    })

    # 2. radius is inert on this problem: achieved batch spacings are far above every
    #    radius tested, so local penalization rarely has two candidates close enough
    #    to penalise. RULE: held if every cell in the radius arm shares one R1 hash
    #    and one R2 hash.
    arm = manifest[manifest["arm"].isin(("radius", "both"))]
    r1_unique = sorted(set(arm["r1_batch_hash"]))
    r2_unique = sorted(set(arm["r2_batch_hash"]))
    held = len(r1_unique) == 1 and len(r2_unique) == 1
    results.append({
        "expectation": "radius produces identical batches (the knob is inert here)",
        "rule": "one distinct R1 hash and one distinct R2 hash across the radius arm",
        "verdict": "HELD" if held else "FAILED",
        "evidence": (
            f"{len(arm)} cells -> {len(r1_unique)} distinct R1 batch(es), "
            f"{len(r2_unique)} distinct R2 batch(es); "
            f"min spacing {arm['r1_min_pairwise_distance'].min():.3f}-"
            f"{arm['r1_min_pairwise_distance'].max():.3f} against radii "
            f"{arm['radius'].min():g}-{arm['radius'].max():g}"
        ),
    })

    # 3. beta trades exploitation against exploration, so it should move the batch.
    #    RULE: held if the beta arm produces more than one distinct R1 batch.
    arm = manifest[manifest["arm"].isin(("beta", "both"))]
    r1_unique = sorted(set(arm["r1_batch_hash"]))
    held = len(r1_unique) > 1
    results.append({
        "expectation": "beta changes the R1 batch",
        "rule": "more than one distinct R1 hash across the beta arm",
        "verdict": "HELD" if held else "FAILED",
        "evidence": (
            f"{len(arm)} cells -> {len(r1_unique)} distinct R1 batch(es) "
            f"at betas {sorted(set(arm['beta']))}"
        ),
    })
    return results


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def ofat_conditions() -> list[tuple[float, float, str]]:
    """13 distinct cells: 9 radii at beta 4, 5 betas at radius 0.25, sharing one."""
    cells: list[tuple[float, float, str]] = []
    for radius in OFAT_RADII:
        arm = "both" if radius == ANCHOR_RADIUS else "radius"
        cells.append((radius, ANCHOR_BETA, arm))
    for beta in OFAT_BETAS:
        if beta == ANCHOR_BETA:
            continue  # already present as the shared anchor cell
        cells.append((ANCHOR_RADIUS, beta, "beta"))
    return cells


def full_grid_conditions() -> list[tuple[float, float, str]]:
    return [(r, b, "grid") for r in OFAT_RADII for b in OFAT_BETAS]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/campaign_d2d_perovskite.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("local_outputs/round_simulations")
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--slice-points", type=int, default=41)
    parser.add_argument(
        "--full-grid", action="store_true",
        help="45-cell radius x beta cross instead of the 13-cell OFAT set.",
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Restrict heatmaps to these input pairs, each 'input_x,input_y'.",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="Restrict to these cell slugs, e.g. radius_0p25__beta_4.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Manifest only. The whole sweep in a couple of minutes.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.time()

    config = load_campaign_config(args.config)
    seed = (
        int((config.get("reproducibility") or {}).get("seed", 0))
        if args.seed is None
        else int(args.seed)
    )
    design = build_design_from_config(dict(config))
    transform = build_objective_transform(config)
    reference = np.asarray(config["reference_point_utility"], dtype=float)
    names = list(design.names)

    print("=" * 79)
    print("ROUND SIMULATION -- campaign loop against a frozen GP oracle")
    print("=" * 79)

    # ------------------------------------------------------------------ read --
    contents = read_campaign_workbook(args.workbook, config)
    if contents.errors:
        for finding in contents.errors:
            print(f"  ERROR  {finding}")
        print("\nRefusing to fit: the objectives cannot be computed for every row.")
        return 1
    X_r0 = contents.inputs.to_numpy(float)
    Y_r0_measured = contents.model_values.to_numpy(float)
    print(f"\n1. WORKBOOK  {len(X_r0)} rows, objectives {objective_names(config)}")
    print(f"   errors 0, warnings {len(contents.warnings)}")

    # ---------------------------------------------------------------- oracle --
    print("\n2. ORACLE  fit_campaign_models on the real rows, then frozen")
    oracle, oracle_warnings = fit_campaign_models(config, X_r0, Y_r0_measured, seed=seed)
    if oracle_warnings:
        print("\n   ABORTING -- the oracle fit raised guard warnings:")
        for message in oracle_warnings:
            print(f"     {message}")
        print(
            "\n   Every surface and every simulated measurement below would be built\n"
            "   on this fit. A collapsed oracle must not render silently, so this is\n"
            "   a hard stop rather than a banner on the figures."
        )
        return 1
    print("   fit guard clean")

    # The unencoded contrast: what the R1 baseline WOULD be if measurement-space
    # values reached the transform directly. It is a property of the observed set,
    # not of any cell, so it is computed once. It is never asserted equal to
    # anything -- it is the size of a mistake, kept on record. The equality that IS
    # asserted, per cell, is reported == independent, inside run_cell.
    def _unencoded_baseline(Y: np.ndarray) -> float:
        with torch.no_grad():
            raw = transform.transform(torch.tensor(Y, dtype=torch.double))
        values = raw.detach().cpu().numpy()
        if not bool((values > reference).all(axis=1).any()):
            return 0.0
        return float(
            compute_ref_pareto_hv(torch.tensor(values, dtype=torch.double), reference)[2]
        )

    Y_r0_oracle = oracle_predict(oracle, config, X_r0, transform)
    baseline_unencoded = _unencoded_baseline(Y_r0_oracle)

    print("\n   R1 observed baseline hypervolume")
    print(f"     on the oracle-scored R0 this sweep uses : "
          f"{hypervolume(Y_r0_oracle, transform, reference):.6f}"
          f"   (unencoded would be {baseline_unencoded:.6f})")
    print(f"     on the real measured R0                 : "
          f"{hypervolume(Y_r0_measured, transform, reference):.6f}"
          f"   (unencoded would be {_unencoded_baseline(Y_r0_measured):.6f})")
    print("     the unencoded figures are what run_r1_ucb produced before "
          "commit 4b76670")

    # ------------------------------------------------------------ conditions --
    cells_spec = full_grid_conditions() if args.full_grid else ofat_conditions()
    if args.conditions:
        wanted = set(args.conditions)
        cells_spec = [c for c in cells_spec if cell_slug(c[0], c[1]) in wanted]
        if not cells_spec:
            print(f"\nNo cell matched --conditions {args.conditions}.")
            return 1

    if args.pairs:
        pairs = []
        for item in args.pairs:
            parts = [p.strip() for p in item.split(",")]
            if len(parts) != 2 or any(p not in names for p in parts):
                print(f"\n--pairs entry {item!r} must be 'input_x,input_y' from {names}.")
                return 1
            pairs.append((parts[0], parts[1]))
    else:
        pairs = list(combinations(names, 2))

    n_figures = 0 if args.no_figures else len(cells_spec) * (len(pairs) * 3 + 2)
    print(f"\n3. PLAN  {len(cells_spec)} cells x {len(pairs)} input pairs")
    print(f"   min_batch_distance pinned at {PINNED_MIN_BATCH_DISTANCE} in every cell")
    print(f"   figures to render: {n_figures}")

    # ------------------------------------------------------------- first cell --
    fixed = fixed_slice_values(design, X_r0)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    cell_seconds: float | None = None

    for position, (radius, beta, arm) in enumerate(cells_spec, start=1):
        slug = cell_slug(radius, beta)
        cell_started = time.time()
        print(f"\n   [{position}/{len(cells_spec)}] {slug} ({arm})", flush=True)
        try:
            cell = run_cell(
                config, oracle, X_r0, transform, reference,
                radius=radius, beta=beta, seed=seed,
            )
        except Exception as error:  # noqa: BLE001 -- one bad cell must not cost the sweep
            # A cell can legitimately fail: validate_batch refuses a batch that
            # breaches the spacing floor, and an extreme radius could in principle
            # leave the selector nothing to pick. Losing the other twelve cells to
            # that would be the wrong trade at ~3 minutes each, so record and move
            # on -- and report at the end rather than only in the scrollback.
            failures.append({"slug": slug, "arm": arm, "error": f"{type(error).__name__}: {error}"})
            print(f"       FAILED: {type(error).__name__}: {error}")
            continue
        cells.append(cell)
        rows.append(manifest_row(
            cell, condition_id=position, arm=arm, seed=seed,
            baseline_unencoded=baseline_unencoded,
        ))
        elapsed = time.time() - cell_started
        print(
            f"       R1 spacing {rows[-1]['r1_min_pairwise_distance']:.3f}  "
            f"HV {cell['hv']['R0']:.4f} -> {cell['hv']['R0+R1']:.4f} -> "
            f"{cell['hv']['R0+R1+R2']:.4f}   ({elapsed:.1f}s)"
        )
        if cell_seconds is None:
            cell_seconds = elapsed
            if not args.no_figures:
                remaining = cell_seconds * (len(cells_spec) - 1)
                # ~0.35 s per rendered figure, measured on this stack
                estimate = (remaining + n_figures * 0.35) / 60.0
                print(f"       estimated total remaining: ~{estimate:.0f} min")
                if estimate > 30:
                    print(
                        "       NOTE: over 30 minutes. Narrow it with --pairs / "
                        "--conditions, or use --no-figures for the manifest alone."
                    )

        if args.no_figures:
            continue

        banner = None
        if cell["final_fit_warnings"]:
            banner = (
                "FIT GUARD: the final GP raised "
                f"{len(cell['final_fit_warnings'])} warning(s) -- this surface is "
                "drawn from a fit worth distrusting."
            )

        condition_dir = output_root / "by_condition" / "qlognehvi" / slug
        condition_dir.mkdir(parents=True, exist_ok=True)
        plot_boxplots(
            condition_dir / "round_boxplots.png", cell, transform,
            seed=seed, warning_banner=banner,
        )
        plot_hypervolume(
            condition_dir / "hypervolume_by_round.png", cell, reference,
            seed=seed, warning_banner=banner,
        )
        rounds_frame(cell, names, transform).to_csv(
            condition_dir / "all_rounds.csv", index=False, encoding="utf-8-sig"
        )

        for pair in pairs:
            mesh_x, mesh_y, surfaces = surface_grid(
                cell["final_model"], cell["config"], design, transform, pair, fixed,
                points=args.slice_points,
            )
            pair_dir = (
                output_root
                / f"{_safe_filename(pair[0])}__{_safe_filename(pair[1])}"
                / "qlognehvi"
                / slug
            )
            pair_dir.mkdir(parents=True, exist_ok=True)
            for index, spec in enumerate(transform.specs):
                plot_surface(
                    pair_dir / f"final_surface_{_safe_filename(spec.name)}.png",
                    mesh_x, mesh_y, surfaces[..., index], pair, spec,
                    cell["X"], design, fixed,
                    radius=radius, beta=beta, seed=seed, warning_banner=banner,
                )

    # -------------------------------------------------------------- manifest --
    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    manifest_path = output_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    print(f"\n4. MANIFEST  {manifest_path}  ({len(manifest)} rows)")

    if failures:
        # Said here, not only in the scrollback: a manifest with rows missing must
        # not read as a manifest of every cell that was asked for.
        print(f"\n   {len(failures)} of {len(cells_spec)} cell(s) FAILED and are "
              "absent from the manifest:")
        for failure in failures:
            print(f"     {failure['slug']} ({failure['arm']}): {failure['error']}")
    if not rows:
        print("\nNo cell completed, so there is nothing to check. Stopping.")
        return 1

    identical: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row["r1_batch_hash"], row["r2_batch_hash"])
        identical.setdefault(key, []).append(cell_slug(row["radius"], row["beta"]))
    print("\n   BATCH IDENTITY -- cells that proposed the same R1 and R2 batches")
    for (r1_hash, r2_hash), members in sorted(identical.items(), key=lambda kv: -len(kv[1])):
        print(f"     R1 {r1_hash} / R2 {r2_hash}  <- {len(members)} cell(s)")
        print(f"       {', '.join(members)}")

    print("\n5. PRE-REGISTERED EXPECTATIONS")
    for check in check_expectations(manifest, cells, transform):
        print(f"\n   {check['verdict']}  {check['expectation']}")
        print(f"     rule     {check['rule']}")
        print(f"     evidence {check['evidence']}")

    print(f"\nDone in {(time.time() - started) / 60.0:.1f} min. Outputs under {output_root}")
    print("Every number above is a model prediction, not a measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
