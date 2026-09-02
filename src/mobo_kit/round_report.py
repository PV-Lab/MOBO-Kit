"""Figures an experimentalist sees when a round is proposed.

One orchestrator, :func:`generate_round_report`, pure and headless.  The launcher
calls it after a successful propose; ``scripts/generate_round_report.py`` calls it
from a terminal; neither owns any of the logic.

**Every figure writes the numbers behind it.**  A PNG whose data cannot be
re-derived is the next plausible-finite-number bug waiting to happen -- this
project has had three, and all three were quantities nothing recomputed.  So each
figure emits at least one CSV, ``manifest.json`` records what was produced, and
determinism is checked against the CSVs rather than against PNG bytes.

**Nothing here decides anything, and several figures exist to say so.**  Two of
the three objectives on the current campaign carry no learnable signal; their
panels look exactly as convincing as thickness's and mean nothing.  Each such panel
is labelled on its face rather than in a caption somewhere else, because a figure
travels without its documentation.

**Output goes beside the workbook, never into it.**  ``openpyxl`` discards cached
formula values on save, so the source workbook is opened read-only for the life of
this module.

Three notebook conventions are deliberately NOT ported; see
``docs/CAMPAIGN_STATUS.md``:

* in-sample parity -- a model is being asked about points it was fitted on, which
  measures memorisation. Parity here is leave-one-out.
* ad-hoc sign flips at plot time -- objective polarity is a config contract
  (``goal:``), and flipping it in a figure makes the figure disagree with the
  optimiser.
* auto-referenced hypervolume -- the reference point is required and campaign-fixed,
  because a reference re-derived per call makes rounds incomparable.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .campaign import (  # noqa: E402
    build_design_from_config,
    build_objective_transform,
    fit_campaign_models,
    normalise_inputs,
    objective_names,
)
from .candidate_diagnostics import (  # noqa: E402
    nearest_reference_distances,
    pairwise_normalized_distances,
)
from .loocv import loo_predictions, null_loo_r2  # noqa: E402
from .metrics import compute_ref_pareto_hv  # noqa: E402
from .scores import ScoreSeverity  # noqa: E402
from .workbook_io import (  # noqa: E402
    candidate_workbook_path,
    read_campaign_workbook,
    read_candidate_results,
)

__all__ = [
    "FigureRecord",
    "ReportManifest",
    "generate_round_report",
    "report_directory",
]

#: The palette every figure in this project shares, so they read as one set.
R0_COLOR, R1_COLOR, R2_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
PROPOSED_COLOR = "#7b3fbf"
REFERENCE_COLOR = "#c0392b"
GRID_COLOR = "#e6e5e1"
SPINE = "#d8d7d2"
OBSERVED_GREY = "#9a9894"

ROUND_COLORS = {"R0": R0_COLOR, "R1": R1_COLOR, "R2": R2_COLOR}

#: Posterior draws for the batch hypervolume diagnostic. Fixed, and recorded in
#: the manifest: a distribution whose sample count moves between runs is not a
#: distribution anyone can compare.
HV_POSTERIOR_SAMPLES = 512


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FigureRecord:
    """One rendered figure and the data files that reproduce it."""

    key: str
    title: str
    caption: str
    png: str
    data: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "caption": self.caption,
            "png": self.png,
            "data": list(self.data),
            "caveats": list(self.caveats),
        }


@dataclass
class ReportManifest:
    """What one report run produced, and what a reader must know about it."""

    directory: Path
    round_name: str
    mode: str
    """``proposal`` when a batch was supplied, ``data_only`` otherwise."""
    figures: tuple[FigureRecord, ...] = ()
    skipped: tuple[tuple[str, str], ...] = ()
    notices: tuple[str, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_name,
            "mode": self.mode,
            "generated_utc": self.context.get("generated_utc"),
            "runtime_seconds": round(self.runtime_seconds, 2),
            "context": self.context,
            "figures": [figure.as_dict() for figure in self.figures],
            "skipped": [{"key": key, "why": why} for key, why in self.skipped],
            "notices": list(self.notices),
        }

    def summary(self) -> str:
        """One line per figure, for the launcher pane."""
        lines = [f"Round report ({self.mode}) -> {self.directory}"]
        for figure in self.figures:
            lines.append(f"  {figure.png:<34} {figure.title}")
        for key, why in self.skipped:
            lines.append(f"  {key:<34} SKIPPED: {why}")
        if self.notices:
            lines.append("")
            lines.append("  Read with the figures:")
            for notice in self.notices:
                lines.append(f"   - {notice}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #


def report_directory(workbook: str | Path, round_name: str, *, when: str) -> Path:
    """``<workbook stem>_reports/<round>_<UTC timestamp>/``, beside the workbook."""
    source = Path(workbook)
    return source.with_name(f"{source.stem}_reports") / f"{round_name}_{when}"


def _git_describe() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover - git absent or not a checkout
        return "unknown"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)


def _wrapped_caveats(fig: plt.Figure, caveats: Sequence[str]) -> list[str]:
    """Hard-wrap to the figure width; matplotlib's own ``wrap=True`` is unreliable."""
    import textwrap

    columns = max(60, int(fig.get_size_inches()[0] * 17))
    lines: list[str] = []
    for caveat in caveats:
        wrapped = textwrap.wrap(caveat, width=columns) or [""]
        lines.append(f"* {wrapped[0]}")
        lines.extend(f"  {piece}" for piece in wrapped[1:])
    return lines


def _save(
    fig: plt.Figure, path: Path, caveats: Sequence[str], *, tight: bool = True
) -> None:
    """Reserve the footer's space BEFORE laying out, so nothing lands on an axis label.

    The caveats are part of the figure rather than a caption in a document, because
    a PNG gets pasted into a slide and the caption does not travel with it. That
    only helps if they are legible, hence the explicit reservation rather than
    trusting a default margin.

    ``tight=False`` for any figure holding a 3-D axes: ``tight_layout`` does not
    support them and warns that its result may be wrong, which on this project's
    rules means not using it rather than ignoring the warning.
    """
    lines = _wrapped_caveats(fig, caveats)
    reserved = float(min(0.42, (len(lines) * 0.155 + 0.30) / fig.get_size_inches()[1]))
    if tight:
        fig.tight_layout(rect=(0.0, reserved, 1.0, 0.99))
    else:
        fig.subplots_adjust(bottom=reserved + 0.09, top=0.9, left=0.055, right=0.985)
    if lines:
        fig.text(
            0.008,
            0.008,
            "\n".join(lines),
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="#5a5854",
            family="monospace",
            linespacing=1.35,
        )
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def _write_csv(frame: pd.DataFrame, path: Path) -> str:
    frame.to_csv(path, index=False)
    return path.name


# --------------------------------------------------------------------------- #
# data gathering
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _RoundBlock:
    """One round's observations, in the order they entered the campaign."""

    name: str
    X_phys: np.ndarray
    Y_measured: np.ndarray
    labels: tuple[str, ...]


def _observations_by_round(
    workbook: Path, config: Mapping[str, Any]
) -> list[_RoundBlock]:
    """R0 from Sheet1, then whichever candidate sheets are filled in.

    A round whose sheet exists but is not fully measured is skipped rather than
    partially included: a hypervolume computed on half a round is not that round's
    hypervolume, and it would silently make the trajectory wrong rather than short.
    """
    contents = read_campaign_workbook(workbook, config)
    blocks = [
        _RoundBlock(
            "R0",
            contents.inputs.to_numpy(float),
            contents.model_values.to_numpy(float),
            tuple(str(value) for value in contents.sample_ids),
        )
    ]
    for round_name in ("R1", "R2"):
        path = candidate_workbook_path(workbook, round_name)
        if not path.exists():
            break
        try:
            results = read_candidate_results(workbook, config, round_name)
        except Exception:
            break
        values = results.model_values.to_numpy(float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            break
        blocks.append(
            _RoundBlock(
                round_name,
                results.conditions.to_numpy(float),
                values,
                tuple(results.candidate_ids),
            )
        )
    return blocks


def _utility(transform: Any, Y_measured: np.ndarray) -> np.ndarray:
    """Measurement space -> utility, by the one call that cannot forget the link."""
    block = torch.tensor(np.asarray(Y_measured, dtype=float), dtype=torch.double)
    return transform.transform_measurements(block).detach().cpu().numpy()


def _pareto_mask(utility: np.ndarray) -> np.ndarray:
    """Non-dominated rows, maximisation. Small N, so the O(n^2) form is fine."""
    n = len(utility)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = np.all(utility >= utility[i], axis=1) & np.any(
            utility > utility[i], axis=1
        )
        if dominated.any():
            mask[i] = False
    return mask


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _figure_batch_placement(
    directory: Path,
    config: Mapping[str, Any],
    observed_X: np.ndarray,
    proposed_X: np.ndarray,
    round_name: str,
) -> FigureRecord:
    """Where in recipe space the algorithm is asking to go.

    The figure a coater operator actually reads. Parallel coordinates because ten
    inputs will not fit on two axes and a projection would invent structure; the
    distance panel because "is this batch spread out" is the question local
    penalisation exists to answer and it is not visible in the lines.
    """
    design = build_design_from_config(dict(config))
    names = list(design.names)
    observed_norm = normalise_inputs(config, observed_X)
    proposed_norm = normalise_inputs(config, proposed_X)

    fig = plt.figure(figsize=(13.5, 5.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.85, 1.0], wspace=0.28)

    ax = fig.add_subplot(grid[0, 0])
    _style(ax)
    xs = np.arange(len(names))
    for row in observed_norm:
        ax.plot(xs, row, color=OBSERVED_GREY, linewidth=1.0, alpha=0.55, zorder=2)
    colour = ROUND_COLORS.get(round_name, PROPOSED_COLOR)
    for index, row in enumerate(proposed_norm, start=1):
        ax.plot(
            xs,
            row,
            color=colour,
            linewidth=2.2,
            marker="o",
            markersize=4.5,
            zorder=3,
            label=f"{round_name}_C{index:02d}",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("normalised to the declared grid")
    ax.set_title(
        f"{round_name} proposal against {len(observed_norm)} measured recipes",
        fontsize=11,
    )
    ax.legend(fontsize=7.5, ncol=2, framealpha=0.9)

    ax2 = fig.add_subplot(grid[0, 1])
    within = pairwise_normalized_distances(proposed_norm)
    image = ax2.imshow(within, cmap="magma_r", vmin=0.0)
    ax2.set_xticks(range(len(proposed_norm)))
    ax2.set_yticks(range(len(proposed_norm)))
    labels = [f"C{i:02d}" for i in range(1, len(proposed_norm) + 1)]
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_yticklabels(labels, fontsize=8)
    for i in range(len(proposed_norm)):
        for j in range(len(proposed_norm)):
            ax2.text(
                j,
                i,
                f"{within[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if within[i, j] > within.max() * 0.55 else "#333333",
            )
    ax2.set_title("pairwise distance within the batch", fontsize=11)
    fig.colorbar(image, ax=ax2, fraction=0.046, pad=0.04)

    nearest = nearest_reference_distances(proposed_norm, observed_norm)
    frame = pd.DataFrame(proposed_norm, columns=[f"{name}_norm" for name in names])
    frame.insert(0, "candidate", labels)
    frame["distance_to_nearest_observed"] = nearest
    frame["min_distance_within_batch"] = [
        np.min(np.delete(within[i], i)) if len(within) > 1 else np.nan
        for i in range(len(within))
    ]
    data = _write_csv(frame, directory / "00_batch_placement.csv")

    penalization = (config.get("local_penalization") or {})
    caveats = [
        "Normalised against the DECLARED GRID, not the observed range: 0 and 1 are "
        "the range edges the config allows, so a line touching them is at a bound.",
        f"Local penalisation radius {penalization.get('radius')}, minimum batch "
        f"spacing {penalization.get('min_batch_distance')}; achieved minimum "
        f"{np.min(within[within > 0]) if (within > 0).any() else float('nan'):.3f}.",
    ]
    _save(fig, directory / "00_batch_placement.png", caveats, tight=False)
    return FigureRecord(
        key="00_batch_placement",
        title="Where the proposed batch sits in recipe space",
        caption=(
            "Each line is one recipe across the ten inputs, normalised to the "
            "declared grid. Grey lines are what has been measured; coloured lines "
            "are what is proposed. The heatmap is the batch's internal spacing."
        ),
        png="00_batch_placement.png",
        data=(data,),
        caveats=tuple(caveats),
    )


def _figure_loo_parity(
    directory: Path,
    config: Mapping[str, Any],
    X_phys: np.ndarray,
    Y_measured: np.ndarray,
    names: Sequence[str],
    labels: Sequence[str],
    seed: int,
) -> FigureRecord:
    """Predicted against measured, leave-one-out, in the measurement's own units.

    **Leave-one-out and not in-sample.** An in-sample parity plot asks the model
    about points it was fitted on and therefore measures memorisation; at N=15 in
    10 dimensions it is close to a straight line no matter what the model knows.
    That is one of the three notebook conventions deliberately not carried over.

    The numbers come from :mod:`mobo_kit.loocv`, which is the same fold loop
    ``scripts/intake_new_data.py`` uses -- not a reimplementation that agrees today.
    """
    entries = config["objectives"]["specs"]
    results = {
        name: loo_predictions(config, entries[index], X_phys, Y_measured[:, index], seed=seed)
        for index, name in enumerate(names)
    }
    null = null_loo_r2(len(Y_measured))

    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 5.6))
    axes = np.atleast_1d(axes)
    rows: list[dict[str, Any]] = []
    for ax, name in zip(axes, names):
        result = results[name]
        _style(ax)
        entry = entries[names.index(name)]
        learnable = str(entry.get("signal_status", "")) == "learnable"
        colour = R0_COLOR if learnable else OBSERVED_GREY
        ax.errorbar(
            result.observed,
            result.predicted,
            yerr=result.predictive_sd,
            fmt="o",
            markersize=6,
            color=colour,
            ecolor=colour,
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.9,
            zorder=3,
        )
        lo = float(min(result.observed.min(), result.predicted.min()))
        hi = float(max(result.observed.max(), result.predicted.max()))
        pad = 0.07 * (hi - lo if hi > lo else 1.0)
        line = np.array([lo - pad, hi + pad])
        ax.plot(line, line, color="#666666", linewidth=1.0, linestyle="--", zorder=2)
        ax.set_xlim(*line)
        ax.set_ylim(*line)
        ax.set_xlabel(f"measured {name}")
        ax.set_ylabel("leave-one-out prediction")
        ax.set_title(
            f"{name}\nLOO R2 {result.r2:+.4f}   null {null:+.4f}",
            fontsize=11,
            color="#222222" if learnable else "#8a3b2f",
        )
        if not learnable:
            # inside the axes, not in the title: this is the single most important
            # thing about the panel and it must not be croppable
            ax.text(
                0.5,
                0.955,
                "NO LEARNABLE SIGNAL",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                color="#8a3b2f",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fdeeea", ec="#e0b4a8"),
            )
        for position, label in enumerate(labels):
            ax.annotate(
                label,
                (result.observed[position], result.predicted[position]),
                fontsize=6.5,
                color="#555555",
                xytext=(3, 3),
                textcoords="offset points",
            )
        for position, label in enumerate(labels):
            rows.append(
                {
                    "objective": name,
                    "sample": label,
                    "observed": result.observed[position],
                    "loo_predicted": result.predicted[position],
                    "loo_predictive_sd": result.predictive_sd[position],
                    "model_link": result.model_link,
                    "loo_r2": result.r2,
                    "loo_spearman": result.spearman,
                    "null_loo_r2": null,
                    "has_mean_function": result.has_mean_function,
                }
            )

    data = _write_csv(pd.DataFrame(rows), directory / "01_loo_parity.csv")
    caveats = [
        "Leave-one-out, not in-sample: every point is predicted by a model that "
        "never saw it. An in-sample version of this plot looks far better and "
        "measures memorisation.",
        f"The bar to clear is the NULL, {null:+.4f}, not zero. Predicting the "
        "leave-one-out mean scores exactly that, whatever the data.",
        "An axis marked NO LEARNABLE SIGNAL has a model that does not beat the "
        "null. Its scatter is not a weak trend; it is nothing.",
    ]
    if any(results[name].model_link == "log" for name in names):
        caveats.append(
            "Where the model emits log(y), the point shown is the median exp(mu) "
            "and the bar is the lognormal sd, which is asymmetric in the original "
            "units."
        )
    _save(fig, directory / "01_loo_parity.png", caveats)
    return FigureRecord(
        key="01_loo_parity",
        title="How well the model predicts a film it has not seen",
        caption=(
            "Leave-one-out prediction against measurement, one panel per objective, "
            "in the measurement's own units. Points on the dashed line are perfect."
        ),
        png="01_loo_parity.png",
        data=(data,),
        caveats=tuple(caveats),
    )


def _figure_attribution(
    directory: Path,
    config: Mapping[str, Any],
    model: Any,
    transform: Any,
    X_phys: np.ndarray,
    names: Sequence[str],
    seed: int,
    max_instances: int,
) -> FigureRecord:
    """Mean |SHAP| per input per objective, from the campaign's own fitted model."""
    from .attribution import mean_absolute_shap, shap_values_for

    design = build_design_from_config(dict(config))
    feature_names = list(design.names)
    instances = X_phys[: max(1, min(max_instances, len(X_phys)))]

    entries = config["objectives"]["specs"]
    rows: list[dict[str, Any]] = []
    magnitudes: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        values = shap_values_for(
            model, config, transform, index, X_phys, instances, seed=seed
        )
        magnitude = mean_absolute_shap(values)
        magnitudes[name] = magnitude
        declared = {
            str(feature["column"])
            for feature in (entries[index].get("mean_function") or {}).get(
                "features", []
            )
        }
        order = np.argsort(magnitude)[::-1]
        for rank, position in enumerate(order, start=1):
            rows.append(
                {
                    "objective": name,
                    "feature": feature_names[position],
                    "mean_abs_shap": float(magnitude[position]),
                    "mean_shap": float(values[:, position].mean()),
                    "rank": rank,
                    "in_mean_function": feature_names[position] in declared,
                    "signal_status": str(entries[index].get("signal_status", "")),
                }
            )

    fig, axes = plt.subplots(1, len(names), figsize=(5.4 * len(names), 5.6))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        _style(ax)
        index = names.index(name)
        magnitude = magnitudes[name]
        order = np.argsort(magnitude)
        declared = {
            str(feature["column"])
            for feature in (entries[index].get("mean_function") or {}).get(
                "features", []
            )
        }
        learnable = str(entries[index].get("signal_status", "")) == "learnable"
        colours = [
            R1_COLOR if feature_names[position] in declared else (
                R0_COLOR if learnable else OBSERVED_GREY
            )
            for position in order
        ]
        ax.barh(
            range(len(order)),
            magnitude[order],
            color=colours,
            edgecolor="white",
            zorder=3,
        )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([feature_names[position] for position in order], fontsize=8)
        ax.set_xlabel("mean |SHAP| in utility units")
        verdict = "" if learnable else "  [fitted noise]"
        ax.set_title(f"{name}{verdict}", fontsize=10.5,
                     color="#222222" if learnable else "#8a3b2f")

    data = _write_csv(pd.DataFrame(rows), directory / "02_attribution.csv")
    caveats = [
        "Attributions explain the MODEL, not the world. Orange bars are features "
        "the config TOLD the model about through a mean function, so recovering "
        "them is a consistency check rather than a discovery.",
        "On an axis with no learnable signal the bars are structure fitted to "
        "noise. They have real magnitude and orderly ranking and mean nothing.",
        "Explains E[utility] through the campaign transform, so thickness is "
        "attributed on its 650 nm target and not on nanometres.",
        f"Exact Shapley values: all 2^{len(feature_names)} coalitions are "
        f"enumerated over {len(instances)} instances, so these do not depend on "
        "the seed.",
    ]
    _save(fig, directory / "02_attribution.png", caveats)
    return FigureRecord(
        key="02_attribution",
        title="Which process inputs move each objective, in the model",
        caption=(
            "Mean absolute SHAP value per input, per objective, computed on the "
            "campaign's own fitted model in utility units."
        ),
        png="02_attribution.png",
        data=(data,),
        caveats=tuple(caveats),
    )


def _figure_batch_predictions(
    directory: Path,
    config: Mapping[str, Any],
    review: Any,
    names: Sequence[str],
    hv_frame: pd.DataFrame,
    round_name: str,
) -> FigureRecord:
    """What the model expects from each proposed condition, physical and utility.

    **The numbers are read from the batch-review artifact, not recomputed.** The
    Review sheet and this figure must not be able to disagree; one of them is the
    source and it is the one already attached to the worklist.
    """
    candidates = review.candidates
    labels = [f"C{i:02d}" for i in range(1, len(candidates) + 1)]
    colour = ROUND_COLORS.get(round_name, PROPOSED_COLOR)

    fig, axes = plt.subplots(2, len(names), figsize=(4.7 * len(names), 8.2))
    axes = np.atleast_2d(axes)
    rows: list[dict[str, Any]] = []
    positions = np.arange(len(candidates))
    for column, name in enumerate(names):
        physical = candidates[f"{name}_predicted"].to_numpy(float)
        lo = candidates[f"{name}_lo68"].to_numpy(float)
        hi = candidates[f"{name}_hi68"].to_numpy(float)
        utility = candidates[f"{name}_utility"].to_numpy(float)
        utility_sd = candidates[f"{name}_sd"].to_numpy(float)

        ax = axes[0, column]
        _style(ax)
        ax.bar(positions, physical, color=colour, edgecolor="white", zorder=3)
        ax.errorbar(
            positions,
            physical,
            yerr=[physical - lo, hi - physical],
            fmt="none",
            ecolor="#333333",
            elinewidth=1.1,
            capsize=3.5,
            zorder=4,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"{name} - predicted measurement", fontsize=10.5)
        ax.set_ylabel("measurement units")

        ax = axes[1, column]
        _style(ax)
        ax.bar(positions, utility, color=colour, edgecolor="white", zorder=3)
        ax.errorbar(
            positions,
            utility,
            yerr=utility_sd,
            fmt="none",
            ecolor="#333333",
            elinewidth=1.1,
            capsize=3.5,
            zorder=4,
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"{name} - utility (higher is better)", fontsize=10.5)
        ax.set_ylabel("utility")

        for position, label in enumerate(labels):
            rows.append(
                {
                    "candidate": label,
                    "objective": name,
                    "predicted_measurement": physical[position],
                    "lo68": lo[position],
                    "hi68": hi[position],
                    "utility_mean": utility[position],
                    "utility_sd": utility_sd[position],
                }
            )

    data = _write_csv(pd.DataFrame(rows), directory / "03_batch_predictions.csv")
    hv_data = _write_csv(hv_frame, directory / "03_batch_hypervolume.csv")
    overall = hv_frame[hv_frame["candidate"] == "BATCH"]
    caveats = [
        "Predictions, not measurements. The bars are what the model expects before "
        "anything is fabricated, and the whiskers are its own uncertainty.",
        "The top row is in each measurement's units; the bottom row is utility, "
        "which is what the optimiser maximises. Thickness utility peaks at the "
        "650 nm target, so a thicker film is not a better one.",
        "Numbers are read from the Review sheet's artifact, not recomputed here, "
        "so the two cannot disagree.",
    ]
    if not overall.empty:
        row = overall.iloc[0]
        caveats.append(
            f"Expected hypervolume gain {row['delta_hv_p50']:+.4f} "
            f"(p05 {row['delta_hv_p05']:+.4f}, p95 {row['delta_hv_p95']:+.4f}), "
            f"P(gain > 0) = {row['p_gain_positive']:.2f}, over "
            f"{HV_POSTERIOR_SAMPLES} posterior draws."
        )
    _save(fig, directory / "03_batch_predictions.png", caveats)
    return FigureRecord(
        key="03_batch_predictions",
        title="What the model expects from each proposed condition",
        caption=(
            "Per condition and objective: predicted measurement with a 68% interval "
            "above, utility with its posterior sd below. The companion CSV carries "
            "the batch's hypervolume-gain distribution."
        ),
        png="03_batch_predictions.png",
        data=(data, hv_data),
        caveats=tuple(caveats),
    )


def _batch_hypervolume_diagnostic(
    config: Mapping[str, Any],
    model: Any,
    transform: Any,
    observed_utility: np.ndarray,
    proposed_X: np.ndarray,
    reference: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """How much this batch could add, as a distribution rather than a point.

    A single expected utility per candidate cannot answer "is this batch worth
    fabricating": hypervolume gain is a joint, nonlinear function of all of them.
    So draw from the posterior at the proposed points, transform each draw to
    utility, and recompute the hypervolume of ``observed + batch`` per draw.

    ``P(non-dominated)`` per candidate is the share of draws in which that
    condition is not dominated by anything already measured -- the question "is
    this one pulling its weight" for a specific row.
    """
    X_norm = normalise_inputs(config, np.asarray(proposed_X, dtype=float))
    torch.manual_seed(int(seed))
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(torch.tensor(X_norm, dtype=torch.double))
        draws = posterior.rsample(
            torch.Size([HV_POSTERIOR_SAMPLES])
        )  # (S, q, m) in MODEL space
        utility_draws = transform.transform(draws).detach().cpu().numpy()

    base_ref = np.asarray(reference, dtype=float)
    _, _, base_hv = compute_ref_pareto_hv(
        torch.tensor(observed_utility, dtype=torch.double), base_ref
    )
    base_hv = float(base_hv)

    gains = np.empty(HV_POSTERIOR_SAMPLES)
    non_dominated = np.zeros(utility_draws.shape[1])
    for s in range(HV_POSTERIOR_SAMPLES):
        combined = np.vstack([observed_utility, utility_draws[s]])
        _, _, volume = compute_ref_pareto_hv(
            torch.tensor(combined, dtype=torch.double), base_ref
        )
        gains[s] = float(volume) - base_hv
        for q in range(utility_draws.shape[1]):
            point = utility_draws[s, q]
            dominated = np.all(observed_utility >= point, axis=1) & np.any(
                observed_utility > point, axis=1
            )
            if not dominated.any():
                non_dominated[q] += 1.0
    non_dominated /= HV_POSTERIOR_SAMPLES

    rows = [
        {
            "candidate": f"C{q + 1:02d}",
            "p_non_dominated": float(non_dominated[q]),
            "delta_hv_p05": float("nan"),
            "delta_hv_p50": float("nan"),
            "delta_hv_p95": float("nan"),
            "p_gain_positive": float("nan"),
            "baseline_hv": base_hv,
            "posterior_draws": HV_POSTERIOR_SAMPLES,
        }
        for q in range(utility_draws.shape[1])
    ]
    rows.append(
        {
            "candidate": "BATCH",
            "p_non_dominated": float("nan"),
            "delta_hv_p05": float(np.percentile(gains, 5)),
            "delta_hv_p50": float(np.percentile(gains, 50)),
            "delta_hv_p95": float(np.percentile(gains, 95)),
            "p_gain_positive": float(np.mean(gains > 0.0)),
            "baseline_hv": base_hv,
            "posterior_draws": HV_POSTERIOR_SAMPLES,
        }
    )
    return pd.DataFrame(rows)


def _figure_hv_trajectory(
    directory: Path,
    blocks: Sequence[_RoundBlock],
    transform: Any,
    reference: np.ndarray,
) -> FigureRecord:
    """Cumulative observed hypervolume, one point per completed round."""
    rows: list[dict[str, Any]] = []
    cumulative_X: list[np.ndarray] = []
    previous = 0.0
    for block in blocks:
        cumulative_X.append(block.Y_measured)
        utility = _utility(transform, np.vstack(cumulative_X))
        _, pareto, volume = compute_ref_pareto_hv(
            torch.tensor(utility, dtype=torch.double), np.asarray(reference, float)
        )
        rows.append(
            {
                "round": block.name,
                "cumulative_points": len(utility),
                "points_added": len(block.Y_measured),
                "hypervolume": float(volume),
                "gain": float(volume) - previous,
                "pareto_size": int(pareto.shape[0]),
            }
        )
        previous = float(volume)

    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    _style(ax)
    xs = np.arange(len(frame))
    ax.plot(xs, frame["hypervolume"], color="#444444", linewidth=1.4, zorder=2)
    ax.scatter(
        xs,
        frame["hypervolume"],
        s=110,
        c=[ROUND_COLORS.get(name, PROPOSED_COLOR) for name in frame["round"]],
        edgecolor="white",
        linewidth=1.5,
        zorder=4,
    )
    for position, row in frame.iterrows():
        ax.annotate(
            f"{row['hypervolume']:.4f}"
            + ("" if position == 0 else f"\n(+{row['gain']:.4f})"),
            (position, row["hypervolume"]),
            fontsize=8.5,
            ha="center",
            va="bottom",
            xytext=(0, 9),
            textcoords="offset points",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{row['round']}\nn={row['cumulative_points']}" for _, row in frame.iterrows()]
    )
    ax.set_ylabel("cumulative hypervolume, utility space")
    ax.set_title("Learning progress across measured rounds", fontsize=11.5)
    if len(frame) == 1:
        ax.set_xlim(-0.6, 0.6)
        ax.text(
            0,
            frame["hypervolume"].iloc[0],
            "  only R0 is measured, so there is\n  no trajectory yet",
            fontsize=9,
            va="center",
            ha="left",
            color="#8a3b2f",
        )

    data = _write_csv(frame, directory / "04_hv_trajectory.csv")
    caveats = [
        "Cumulative hypervolume rises monotonically BY CONSTRUCTION -- adding "
        "points can only grow a Pareto front. Random sampling produces a rising "
        "line too, so this shows progress and is not evidence of optimisation.",
        "OBSERVED outcomes only. No predicted point appears on this line.",
        # `list(np.round(...))` yields np.float64 objects whose repr leaks the
        # type into the caption. A figure that prints "np.float64(-0.01)" at an
        # experimentalist is telling them about numpy, not about the campaign.
        "Fixed campaign reference point "
        + str([round(float(value), 4) for value in np.asarray(reference, float)])
        + " in utility space. Re-deriving it per round would make these numbers "
        "incomparable with each other.",
    ]
    _save(fig, directory / "04_hv_trajectory.png", caveats)
    return FigureRecord(
        key="04_hv_trajectory",
        title="Hypervolume after each measured round",
        caption=(
            "Cumulative hypervolume of everything measured up to and including each "
            "round, in utility space, against the campaign's fixed reference point."
        ),
        png="04_hv_trajectory.png",
        data=(data,),
        caveats=tuple(caveats),
    )


def _figure_objective_space(
    directory: Path,
    blocks: Sequence[_RoundBlock],
    transform: Any,
    reference: np.ndarray,
    names: Sequence[str],
    proposed_utility: np.ndarray | None,
    proposed_sd: np.ndarray | None,
) -> FigureRecord:
    """The trade-off itself: pairwise panels, plus one 3D view for orientation.

    Pairwise 2D is primary because a static 3D scatter cannot be read for
    position -- depth is ambiguous without rotation, and "which point dominates
    which" is exactly a position question. The 3D panel is kept for the shape of
    the front, which the pairs do not convey.
    """
    all_utility = np.vstack([_utility(transform, block.Y_measured) for block in blocks])
    round_of = [name for block in blocks for name in [block.name] * len(block.Y_measured)]
    labels = [label for block in blocks for label in block.labels]
    pareto = _pareto_mask(all_utility)

    rows = [
        {
            "point": labels[i],
            "round": round_of[i],
            "kind": "observed",
            "on_pareto": bool(pareto[i]),
            **{f"utility_{name}": float(all_utility[i, j]) for j, name in enumerate(names)},
        }
        for i in range(len(all_utility))
    ]
    if proposed_utility is not None:
        for i, point in enumerate(proposed_utility):
            rows.append(
                {
                    "point": f"C{i + 1:02d}",
                    "round": "proposed",
                    "kind": "proposed",
                    "on_pareto": False,
                    **{f"utility_{name}": float(point[j]) for j, name in enumerate(names)},
                }
            )

    pairs = [(0, 1), (0, 2), (1, 2)][: max(1, len(names) * (len(names) - 1) // 2)]
    fig = plt.figure(figsize=(5.2 * len(pairs) + 6.0, 5.8))
    grid = fig.add_gridspec(1, len(pairs) + 1, wspace=0.34, width_ratios=[1] * len(pairs) + [1.25])

    for position, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(grid[0, position])
        _style(ax)
        for block_name in dict.fromkeys(round_of):
            mask = np.array([name == block_name for name in round_of])
            ax.scatter(
                all_utility[mask, i],
                all_utility[mask, j],
                s=52,
                color=ROUND_COLORS.get(block_name, OBSERVED_GREY),
                edgecolor="white",
                linewidth=1.1,
                label=block_name,
                zorder=3,
            )
        # The 2-D front for THIS PAIR, computed on this pair alone. Sorting the
        # 3-D Pareto set by one axis and joining it produces a zigzag that is not
        # a front in any space -- a point can be non-dominated in 3-D while sitting
        # well inside the 2-D trade-off, and the line then crosses itself and
        # invites exactly the wrong reading.
        pair_mask = _pareto_mask(all_utility[:, [i, j]])
        pair_front = all_utility[pair_mask][np.argsort(all_utility[pair_mask][:, i])]
        ax.step(
            pair_front[:, i],
            pair_front[:, j],
            where="post",
            color="#1baf7a",
            linewidth=1.6,
            alpha=0.85,
            zorder=2,
            label="front for this pair" if position == 0 else None,
        )
        # Points on the FULL 3-objective front, ringed rather than joined: they are
        # what the optimiser is trading off, and several of them are interior here.
        ax.scatter(
            all_utility[pareto, i],
            all_utility[pareto, j],
            s=150,
            facecolor="none",
            edgecolor="#1baf7a",
            linewidth=1.6,
            zorder=3,
            label="on the 3-objective front" if position == 0 else None,
        )
        if proposed_utility is not None:
            ax.errorbar(
                proposed_utility[:, i],
                proposed_utility[:, j],
                xerr=None if proposed_sd is None else proposed_sd[:, i],
                yerr=None if proposed_sd is None else proposed_sd[:, j],
                fmt="o",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor=PROPOSED_COLOR,
                markeredgewidth=1.8,
                ecolor=PROPOSED_COLOR,
                elinewidth=1.0,
                zorder=4,
                label="proposed" if position == 0 else None,
            )
        # The reference point is at (-0.01, -0.01) while every observation lives
        # above 0.35, so plotting it in scale spends about 40% of the panel on
        # empty space and squeezes the region anyone needs to read. It is marked
        # at the corner instead, labelled with its real coordinates, because
        # hypervolume is measured from it and hiding it entirely would be worse.
        drawn = [all_utility[:, [i, j]]]
        if proposed_utility is not None:
            drawn.append(proposed_utility[:, [i, j]])
        visible = np.vstack(drawn)
        lo = visible.min(axis=0)
        hi = visible.max(axis=0)
        pad = np.where(hi > lo, (hi - lo) * 0.09, 0.05)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        # A star drawn at the corner sits at real data coordinates and reads as an
        # observation. Text only, below the axes, where nothing can be mistaken
        # for a measurement.
        ax.annotate(
            f"* reference ({reference[i]:g}, {reference[j]:g}) is off-scale, "
            "down and to the left",
            xy=(0.0, -0.155),
            xycoords="axes fraction",
            fontsize=7.4,
            color=REFERENCE_COLOR,
            ha="left",
            va="top",
            annotation_clip=False,
        )
        ax.set_xlabel(f"{names[i]} utility")
        ax.set_ylabel(f"{names[j]} utility")
        if position == 0:
            ax.legend(fontsize=7.0, loc="upper left", framealpha=0.92)

    ax3d = fig.add_subplot(grid[0, len(pairs)], projection="3d")
    ax3d.scatter(
        all_utility[~pareto, 0],
        all_utility[~pareto, 1],
        all_utility[~pareto, 2],
        s=26,
        color=OBSERVED_GREY,
        alpha=0.75,
    )
    ax3d.scatter(
        all_utility[pareto, 0],
        all_utility[pareto, 1],
        all_utility[pareto, 2],
        s=62,
        color="#1baf7a",
        edgecolor="white",
        label="Pareto set",
    )
    if proposed_utility is not None:
        ax3d.scatter(
            proposed_utility[:, 0],
            proposed_utility[:, 1],
            proposed_utility[:, 2],
            s=62,
            color=PROPOSED_COLOR,
            marker="^",
            label="proposed",
        )
    ax3d.scatter(
        [reference[0]], [reference[1]], [reference[2]],
        marker="*", s=200, color=REFERENCE_COLOR, label="reference",
    )
    ax3d.set_xlabel(f"{names[0]}", fontsize=8)
    ax3d.set_ylabel(f"{names[1]}", fontsize=8)
    ax3d.set_zlabel(f"{names[2]}", fontsize=8)
    ax3d.view_init(elev=25, azim=-45)
    ax3d.set_title("utility space, one fixed view", fontsize=10)
    ax3d.legend(fontsize=7, loc="upper left")

    data = _write_csv(pd.DataFrame(rows), directory / "05_objective_space.csv")
    caveats = [
        "Axes are cropped to the data. The reference point sits below every "
        "observation, so drawing it in scale would spend most of the panel on "
        "empty space; it is marked at the corner instead.",
        "The green step line is the front FOR THAT PAIR. Rings mark points on the "
        "full three-objective front -- several of those sit inside the pairwise "
        "trade-off, which is what a three-way trade-off looks like in projection.",
        "UTILITY space, not measurement units. Thickness utility peaks at the "
        "650 nm target, so a point high on that axis is near the target rather "
        "than thick.",
        "The 3D panel is a single fixed view and cannot be read for position -- "
        "depth is ambiguous without rotation. Read the pairwise panels for which "
        "point dominates which.",
        "A point on the Pareto set is non-dominated among what has been MEASURED. "
        "It is not a claim about the whole design space.",
    ]
    _save(fig, directory / "05_objective_space.png", caveats, tight=False)
    return FigureRecord(
        key="05_objective_space",
        title="The trade-off between objectives",
        caption=(
            "Every measured film in utility space: pairwise panels with the Pareto "
            "set traced, plus one 3D view. Open markers are the proposed batch; the "
            "red star is the campaign's reference point."
        ),
        png="05_objective_space.png",
        data=(data,),
        caveats=tuple(caveats),
    )


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #


def generate_round_report(
    workbook: str | Path,
    config: Mapping[str, Any],
    *,
    proposal: Any = None,
    review: Any = None,
    outdir: str | Path | None = None,
    seed: int | None = None,
    shap_max_instances: int = 15,
    progress: Callable[[str], None] | None = None,
    when: str | None = None,
) -> ReportManifest:
    """Render the round's figures beside the workbook and return the manifest.

    ``proposal`` is a :class:`campaign.RoundResult`; supplying it turns on the two
    batch figures. Without one this runs in ``data_only`` mode, which is what the
    "Figures from current data" button uses once measurements are entered and
    before anything is proposed.

    A figure that fails is recorded in ``skipped`` and in ``notices`` and the rest
    of the report still renders. That is deliberate: losing the attribution panel
    should not cost the parity plot, and a silent absence is prevented by the
    manifest naming what went wrong.
    """

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    started = time.perf_counter()
    workbook = Path(workbook)
    names = list(objective_names(config))
    transform = build_objective_transform(config)
    reference = np.asarray(config["reference_point_utility"], dtype=float)
    resolved_seed = (
        int(config.get("reproducibility", {}).get("seed", 0)) if seed is None else seed
    )
    stamp = when or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    mode = "proposal" if proposal is not None else "data_only"
    round_name = str(getattr(proposal, "round_name", None) or "current")
    directory = (
        Path(outdir)
        if outdir is not None
        else report_directory(workbook, round_name, when=stamp)
    )
    directory.mkdir(parents=True, exist_ok=True)

    say("Reading the workbook...")
    contents = read_campaign_workbook(workbook, config)
    X_phys = contents.inputs.to_numpy(float)
    Y_measured = contents.model_values.to_numpy(float)
    labels = [str(value) for value in contents.sample_ids]
    blocks = _observations_by_round(workbook, config)

    say("Fitting the model the round used...")
    model, model_warnings = fit_campaign_models(
        config, X_phys, Y_measured, seed=resolved_seed
    )

    figures: list[FigureRecord] = []
    skipped: list[tuple[str, str]] = []
    notices: list[str] = []

    def attempt(key: str, build: Callable[[], FigureRecord]) -> None:
        say(f"Rendering {key}...")
        try:
            figures.append(build())
        except Exception as exc:  # noqa: BLE001 - one bad figure must not cost the rest
            detail = f"{type(exc).__name__}: {exc}"
            skipped.append((key, detail))
            notices.append(f"{key} could not be rendered -- {detail}")
            (directory / f"{key}.error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )

    proposed_X = None
    if proposal is not None:
        proposed_X = proposal.conditions.to_numpy(float)
        attempt(
            "00_batch_placement",
            lambda: _figure_batch_placement(
                directory, config, X_phys, proposed_X, round_name
            ),
        )
    else:
        skipped.append(
            ("00_batch_placement", "no proposal supplied (data-only mode)")
        )

    attempt(
        "01_loo_parity",
        lambda: _figure_loo_parity(
            directory, config, X_phys, Y_measured, names, labels, resolved_seed
        ),
    )
    attempt(
        "02_attribution",
        lambda: _figure_attribution(
            directory,
            config,
            model,
            transform,
            X_phys,
            names,
            resolved_seed,
            shap_max_instances,
        ),
    )

    if proposal is not None and review is not None:
        observed_utility = _utility(transform, Y_measured)

        def build_batch_figure() -> FigureRecord:
            hv_frame = _batch_hypervolume_diagnostic(
                config,
                model,
                transform,
                observed_utility,
                proposed_X,
                reference,
                resolved_seed,
            )
            return _figure_batch_predictions(
                directory, config, review, names, hv_frame, round_name
            )

        attempt("03_batch_predictions", build_batch_figure)
    else:
        skipped.append(
            (
                "03_batch_predictions",
                "no proposal supplied (data-only mode)"
                if proposal is None
                else "no batch review supplied",
            )
        )

    attempt(
        "04_hv_trajectory",
        lambda: _figure_hv_trajectory(directory, blocks, transform, reference),
    )

    proposed_utility = None
    proposed_sd = None
    if review is not None:
        proposed_utility = np.column_stack(
            [review.candidates[f"{name}_utility"].to_numpy(float) for name in names]
        )
        proposed_sd = np.column_stack(
            [review.candidates[f"{name}_sd"].to_numpy(float) for name in names]
        )
    attempt(
        "05_objective_space",
        lambda: _figure_objective_space(
            directory,
            blocks,
            transform,
            reference,
            names,
            proposed_utility,
            proposed_sd,
        ),
    )

    # ---------------------------------------------------------- the notices --
    entries = config["objectives"]["specs"]
    frozen = [
        str(entry.get("name"))
        for entry in entries
        if str((entry.get("measurement") or {}).get("recipe")) == "stored"
    ]
    if frozen:
        notices.append(
            f"{' and '.join(frozen)} are taken from the workbook as stored; no "
            "independent recomputation exists under this contract, so a stale "
            "value in those columns would not be caught here."
        )
    for index, name in enumerate(names):
        status = str(entries[index].get("signal_status", ""))
        if status and status != "learnable":
            notices.append(
                f"{name}: {status.replace('_', ' ')} -- its model does not beat the "
                "leave-one-out null, so its predictions carry no signal."
            )
    for warning in model_warnings:
        notices.append(f"model fit warning: {warning}")
    for finding in contents.findings:
        if finding.severity is ScoreSeverity.WARNING:
            notices.append(f"data: {finding}")

    runtime = time.perf_counter() - started
    manifest = ReportManifest(
        directory=directory,
        round_name=round_name,
        mode=mode,
        figures=tuple(figures),
        skipped=tuple(skipped),
        notices=tuple(dict.fromkeys(notices)),
        context={
            "workbook": workbook.name,
            "generated_utc": stamp,
            "objective_contract": config["objectives"]["contract_version"],
            "campaign": config.get("campaign", {}).get("name"),
            "seed": resolved_seed,
            "reference_point_utility": [float(value) for value in reference],
            "observed_rows": int(len(X_phys)),
            "rounds_measured": [block.name for block in blocks],
            "git": _git_describe(),
            "python": platform.python_version(),
            "posterior_draws_for_hv": HV_POSTERIOR_SAMPLES,
            "shap_instances": int(min(shap_max_instances, len(X_phys))),
        },
        runtime_seconds=runtime,
    )
    (directory / "manifest.json").write_text(
        json.dumps(manifest.as_dict(), indent=2), encoding="utf-8"
    )
    (directory / "README.txt").write_text(manifest.summary() + "\n", encoding="utf-8")
    say("Report written.")
    return manifest
