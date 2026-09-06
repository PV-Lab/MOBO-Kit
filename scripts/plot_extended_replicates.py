"""What 45 rows of REPEATED recipes say about the three scores.

The sheet behind this script (``Extended Summary Table C1C2``) is the first one in
the project where the same recipe appears more than once: samples 1-15, 16-30 and
31-45 carry identical inputs, recipe for recipe, and block 1 is bit-identical to
the current campaign workbook on thickness. So the three blocks are the SAME 15
RECIPES REMADE, not 45 designs.

That buys the one thing no amount of modelling can buy: a separation of

    "the score changed because the recipe changed"   (what BO can chase)

from

    "the score changed because the film was made and measured again"  (what it cannot).

TWO FIGURES.

``boxplot_extended`` is the measurement, not the model. The top row boxes each
score by campaign block, which is where a systematic between-campaign shift shows
up as three boxes that do not overlap. The bottom row boxes each RECIPE's three
repeats, sorted by recipe mean: tall boxes that overlap everything mean the repeat
spread swamps the recipe spread, and no model can beat that.

``01_loo_parity_extended`` is the model. Both rows are leave-one-out predictions
plotted against the measurement, but they leave out different things, and the
difference is the point:

  * ROW-WISE (top) holds out one ROW. Its two repeats stay in the training set
    carrying the same inputs, so the GP interpolates its own repeat. That number
    measures REPRODUCIBILITY and is not a prediction score. It is plotted because
    it is what a naive run on this sheet reports, and it looks excellent.
  * RECIPE-WISE (bottom) holds out all THREE rows of a recipe. Nothing with those
    inputs remains. That is the honest question -- can the model predict a recipe
    it has never made -- and it is the number to quote.

Run::

    python scripts/plot_extended_replicates.py \
        --workbook "local_inputs/Extended Summary Table C1C2.xlsx" \
        --config configs/campaign_d2d_perovskite_extended_c1c2.yaml \
        --outdir local_inputs/extended_c1c2_reports

Outputs stay local; the workbook is gitignored and so is its report directory.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import f as fdist  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from mobo_kit.campaign import load_campaign_config, normalise_inputs  # noqa: E402
from mobo_kit.model_validation import DIM_SCALED_PRIOR, fit_model_variant  # noqa: E402
from mobo_kit.round_report import _save, _style  # noqa: E402
from mobo_kit.workbook_io import read_campaign_workbook  # noqa: E402

warnings.filterwarnings("ignore")

#: One colour per campaign block. Deliberately NOT the round palette: these are
#: three makings of the same designs, not three rounds of a campaign.
BLOCK_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
BLOCK_LABELS = ("block 1 (= current campaign)", "block 2", "block 3")
DEAD = "#8a3b2f"


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def variance_decomposition(y: np.ndarray, recipe: np.ndarray, block: np.ndarray) -> dict:
    """One-way ANOVA with recipe as the factor, plus the block's share.

    ``icc`` is the fraction of variance the RECIPE owns. It is the ceiling on any
    model that sees only the recipe: predict every repeat by its recipe's true
    mean and the leftover is repeat variance, by construction. A score with an ICC
    near zero cannot be optimised, however good the optimiser.
    """
    y = np.asarray(y, float)
    groups = sorted(set(recipe.tolist()))
    grand = y.mean()
    means = np.array([y[recipe == g].mean() for g in groups])
    counts = np.array([int((recipe == g).sum()) for g in groups])
    ss_between = float((counts * (means - grand) ** 2).sum())
    ss_within = float(
        sum(((y[recipe == g] - means[i]) ** 2).sum() for i, g in enumerate(groups))
    )
    df_b, df_w = len(groups) - 1, len(y) - len(groups)
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    n0 = counts.mean()
    block_ids = sorted(set(block.tolist()))
    block_means = np.array([y[block == b].mean() for b in block_ids])
    block_counts = np.array([int((block == b).sum()) for b in block_ids])
    return {
        "sd_total": float(y.std(ddof=1)),
        "sd_within_recipe": float(np.sqrt(ms_w)),
        "sd_between_recipe": float(np.sqrt(max(0.0, (ms_b - ms_w) / n0))),
        "icc": float(max(0.0, (ms_b - ms_w) / (ms_b + (n0 - 1) * ms_w))),
        "f": float(ms_b / ms_w),
        "p": float(1.0 - fdist.cdf(ms_b / ms_w, df_b, df_w)),
        "block_variance_share": float(
            (block_counts * (block_means - grand) ** 2).sum() / ((y - grand) ** 2).sum()
        ),
    }


def fold_predictions(config, X_phys, y, groups, *, seed=73):
    """Predict every row from a model fitted without ANY row of its group.

    A fold that will not fit is not skipped and not retried on a looser variant:
    it falls back to the training mean and is COUNTED. A collapsed fold means the
    GP explained that objective as pure noise, which is a result about the data
    and must not be hidden behind a model that happens to fit.
    """
    Xn = normalise_inputs(config, np.asarray(X_phys, float))
    y = np.asarray(y, float)
    n = len(y)
    mu, sd = np.empty(n), np.empty(n)
    collapsed: list[int] = []
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for g in sorted(set(np.asarray(groups).tolist())):
            held = [i for i in range(n) if groups[i] == g]
            keep = [i for i in range(n) if groups[i] != g]
            torch.manual_seed(seed)
            try:
                record = fit_model_variant(
                    torch.tensor(Xn[keep], dtype=torch.double),
                    torch.tensor(y[keep], dtype=torch.double).unsqueeze(-1),
                    sample_ids=tuple(range(len(keep))),
                    objective_names=("y",),
                    variant=DIM_SCALED_PRIOR,
                    seed=seed,
                )
            except Exception:
                collapsed.append(int(g))
                mu[held] = y[keep].mean()
                sd[held] = y[keep].std(ddof=1)
                continue
            gp = record.model.models[0]
            gp.eval()
            with torch.no_grad():
                posterior = gp.posterior(torch.tensor(Xn[held], dtype=torch.double))
                mu[held] = posterior.mean.reshape(-1).numpy()
                sd[held] = np.sqrt(
                    np.clip(posterior.variance.reshape(-1).numpy(), 0.0, None)
                )
    finally:
        torch.set_num_threads(previous)
    return mu, sd, tuple(collapsed)


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, float)
    residual = ((observed - predicted) ** 2).sum()
    total = ((observed - observed.mean()) ** 2).sum()
    return float(1.0 - residual / total)


def grouped_null(n: int, n_folds: int) -> float:
    """What predicting the held-out-group mean scores, at this fold size.

    The familiar ``1 - (N/(N-1))^2`` is the k=1 case. Dropping k rows at a time
    shifts the training mean further, so the bar to clear MOVES with the fold
    size and the two rows of the parity figure do not share one null.
    """
    k = n / n_folds
    return float(1.0 - (n / (n - k)) ** 2)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _display(name: str, values: np.ndarray) -> tuple[np.ndarray, str, bool]:
    """Optoelectronic spans five orders of magnitude; plot it in log10."""
    if name == "optoelectronic":
        return np.log10(np.clip(values, 1e-300, None)), "log10(optoelectronic score)", True
    return values, f"{name} score", False


def plot_boxes(directory: Path, names, Y, recipe, block, stats) -> Path:
    fig, axes = plt.subplots(2, len(names), figsize=(5.6 * len(names), 9.6))
    for column, name in enumerate(names):
        values, label, _ = _display(name, Y[:, column])
        stat = stats[name]

        top = axes[0, column]
        _style(top)
        data = [values[block == b] for b in range(3)]
        boxes = top.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, colour in zip(boxes["boxes"], BLOCK_COLORS):
            patch.set_facecolor(colour)
            patch.set_alpha(0.22)
            patch.set_edgecolor(colour)
        for key in ("whiskers", "caps", "medians"):
            for line in boxes[key]:
                line.set_color("#555555")
        for g in sorted(set(recipe.tolist())):
            top.plot(
                [1, 2, 3],
                [values[(recipe == g) & (block == b)][0] for b in range(3)],
                color="#c4c3bf",
                linewidth=0.7,
                zorder=2,
            )
        for b in range(3):
            jitter = np.random.default_rng(73 + b).normal(0, 0.04, size=len(data[b]))
            top.scatter(
                1 + b + jitter, data[b], s=26, color=BLOCK_COLORS[b], zorder=3, alpha=0.9
            )
        top.set_xticks([1, 2, 3])
        top.set_xticklabels(
            ["block 1\n(current campaign)", "block 2", "block 3"], fontsize=9
        )
        top.set_ylabel(label)
        share = stat["block_variance_share"]
        top.set_title(
            f"{name}\nthe block owns {share:.1%} of the variance",
            fontsize=11,
            color=DEAD if share > 0.25 else "#222222",
        )
        if share > 0.25:
            top.text(
                0.5,
                0.955,
                "SYSTEMATIC BETWEEN-CAMPAIGN SHIFT",
                transform=top.transAxes,
                ha="center",
                va="top",
                fontsize=9.5,
                color=DEAD,
                bbox=dict(boxstyle="round,pad=0.35", fc="#fdeeea", ec="#e0b4a8"),
            )

        bottom = axes[1, column]
        _style(bottom)
        order = list(np.argsort([values[recipe == g].mean() for g in sorted(set(recipe.tolist()))]))
        per_recipe = [values[recipe == g] for g in order]
        boxes = bottom.boxplot(per_recipe, patch_artist=True, widths=0.6, showfliers=False)
        for patch in boxes["boxes"]:
            patch.set_facecolor("#9a9894")
            patch.set_alpha(0.18)
            patch.set_edgecolor("#9a9894")
        for key in ("whiskers", "caps", "medians"):
            for line in boxes[key]:
                line.set_color("#555555")
        for position, g in enumerate(order, start=1):
            for b in range(3):
                mask = (recipe == g) & (block == b)
                bottom.scatter(
                    np.full(int(mask.sum()), position),
                    values[mask],
                    s=24,
                    color=BLOCK_COLORS[b],
                    zorder=3,
                    alpha=0.9,
                )
        bottom.set_xticks(range(1, len(order) + 1))
        bottom.set_xticklabels([str(int(g) + 1) for g in order], fontsize=8)
        bottom.set_xlabel("recipe, sorted by its mean")
        bottom.set_ylabel(label)
        learnable = stat["icc"] >= 0.5
        bottom.set_title(
            f"repeat spread {stat['sd_within_recipe']:.3g}   "
            f"recipe spread {stat['sd_between_recipe']:.3g}\n"
            f"ICC {stat['icc']:.3f}   F {stat['f']:.2f}   p {stat['p']:.4f}",
            fontsize=10,
            color="#222222" if learnable else DEAD,
        )
        if not learnable:
            bottom.text(
                0.5,
                0.955,
                "REPEATS SWAMP THE RECIPE",
                transform=bottom.transAxes,
                ha="center",
                va="top",
                fontsize=9.5,
                color=DEAD,
                bbox=dict(boxstyle="round,pad=0.35", fc="#fdeeea", ec="#e0b4a8"),
            )

    fig.suptitle(
        "The same 15 recipes, made three times: what actually moves the score",
        fontsize=13,
        y=0.985,
    )
    caveats = [
        "These are MEASUREMENTS, not model output. Nothing here has been fitted.",
        "Top row: one box per campaign block, grey lines joining the three makings "
        "of one recipe. Boxes at different heights with the lines all sloping the "
        "same way is a systematic shift between campaigns, not repeat scatter.",
        "Bottom row: one box per recipe over its three repeats. ICC is the share of "
        "variance the RECIPE owns and it is a CEILING on any model -- an ICC near "
        "zero means the recipe explains none of the score and no optimiser, "
        "acquisition or kernel can chase it.",
        "Optoelectronic is drawn in log10 because it spans five orders of magnitude "
        "on this sheet.",
    ]
    path = directory / "boxplot_extended.png"
    _save(fig, path, caveats)
    return path


def plot_parity(directory: Path, names, Y, recipe, block, folds) -> Path:
    n = len(Y)
    n_recipes = len(set(recipe.tolist()))
    fig, axes = plt.subplots(2, len(names), figsize=(5.4 * len(names), 10.6))
    rows = [
        ("row-wise LOO", "rowwise", grouped_null(n, n), n),
        ("leave-one-RECIPE-out", "recipe", grouped_null(n, n_recipes), n_recipes),
    ]
    for r, (title, key, null, n_folds) in enumerate(rows):
        for column, name in enumerate(names):
            ax = axes[r, column]
            _style(ax)
            observed, _, logged = _display(name, Y[:, column])
            fold = folds[key][name]
            predicted = np.asarray(fold["predicted"], float)
            if logged:
                predicted = np.log10(np.clip(predicted, 1e-300, None))
            dead = bool(fold["collapsed"])
            for b in range(3):
                mask = block == b
                ax.scatter(
                    observed[mask],
                    predicted[mask],
                    s=34,
                    color=BLOCK_COLORS[b],
                    alpha=0.35 if dead else 0.9,
                    zorder=3,
                    label=BLOCK_LABELS[b] if (r == 0 and column == 0) else None,
                )
            lo = float(min(observed.min(), predicted.min()))
            hi = float(max(observed.max(), predicted.max()))
            pad = 0.07 * (hi - lo if hi > lo else 1.0)
            line = np.array([lo - pad, hi + pad])
            ax.plot(line, line, color="#666666", linewidth=1.0, linestyle="--", zorder=2)
            ax.set_xlim(*line)
            ax.set_ylim(*line)
            ax.set_xlabel(f"measured {name}" + (" (log10)" if logged else ""))
            ax.set_ylabel(f"{title} prediction")
            beats = fold["r2"] > null and not dead
            ax.set_title(
                f"{name} - {title}\n"
                f"R2 {fold['r2']:+.4f}   null {null:+.4f}   rho {fold['spearman']:+.3f}",
                fontsize=10.5,
                color="#222222" if beats else DEAD,
            )
            if dead:
                banner = f"MODEL COLLAPSED IN {len(fold['collapsed'])}/{n_folds} FOLDS"
            elif not beats:
                banner = "DOES NOT BEAT THE NULL"
            else:
                banner = ""
            if banner:
                ax.text(
                    0.5,
                    0.955,
                    banner,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9.5,
                    color=DEAD,
                    bbox=dict(boxstyle="round,pad=0.35", fc="#fdeeea", ec="#e0b4a8"),
                )
    # Inside the first panel rather than on the figure: a figure-level legend at
    # the top right lands on the suptitle, and one at the bottom lands on the
    # caveats, both of which `_save` has already reserved space for.
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle(
        "Predicted against measured. The top row leaks a repeat; the bottom row does not.",
        fontsize=13,
        y=0.985,
    )
    caveats = [
        "TOP ROW IS NOT A PREDICTION SCORE. Holding out one row leaves that "
        "recipe's other two repeats in the training set with identical inputs, so "
        "the GP interpolates its own repeat. It measures reproducibility, and it is "
        "shown because it is what a naive leave-one-out on this sheet reports.",
        "BOTTOM ROW is the honest question: all three repeats of a recipe held out "
        "together, so the model has never seen those inputs. Quote this one.",
        "The two rows have DIFFERENT nulls. Dropping 3 rows of 45 moves the training "
        "mean further than dropping 1, so the bar is lower for the top row. "
        "Predicting the held-out mean scores exactly the null, whatever the data.",
        "A collapsed fold is one whose GP explained the objective as pure noise and "
        "refused to fit; it falls back to the training mean and is counted rather "
        "than retried on a looser model.",
    ]
    path = directory / "01_loo_parity_extended.png"
    _save(fig, path, caveats)
    return path


# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/campaign_d2d_perovskite_extended_c1c2.yaml"),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="repeats per recipe; the sheet must be that many equal blocks",
    )
    parser.add_argument(
        "--align-blocks-to-first",
        action="store_true",
        help=(
            "copy block 1's inputs onto the later blocks where they disagree. "
            "On this sheet that applies ONE correction the group already made: "
            "sample 2 was re-encoded from `speed_2 = 0, time_2 = 60` to "
            "`time_2 = 0`, and the edit reached block 1 only. A second stage of "
            "60 s at 0 rpm is not a second stage, so the later blocks describe "
            "the same film under the old encoding. Without this flag the "
            "mismatch is reported and the rows are grouped by POSITION anyway -- "
            "the flag changes what the GP is told, not what is grouped."
        ),
    )
    args = parser.parse_args(argv)

    config = load_campaign_config(args.config)
    seed = args.seed if args.seed is not None else int(config["reproducibility"]["seed"])
    contents = read_campaign_workbook(args.workbook, config)
    X = contents.inputs.to_numpy(float)
    Y = contents.model_values.to_numpy(float)
    names = list(contents.model_values.columns)
    n = len(X)
    k = args.replicates
    if n % k:
        raise SystemExit(f"{n} rows is not {k} equal blocks.")
    per_block = n // k

    recipe = np.tile(np.arange(per_block), k)
    block = np.repeat(np.arange(k), per_block)
    # Verify the assumed layout rather than trusting it. Grouping is by POSITION,
    # which is what makes a block a block; input equality is the check on that
    # assumption, and a failure is reported by name rather than absorbed.
    input_names = [item["name"] for item in config["inputs"]]
    drift: list[str] = []
    for g in range(per_block):
        rows = np.flatnonzero(recipe == g)
        if not np.allclose(X[rows], X[rows[0]]):
            differing = [
                input_names[c]
                for c in range(X.shape[1])
                if not np.allclose(X[rows, c], X[rows[0], c])
            ]
            drift.append(
                f"  recipe {g + 1}: sheet rows {(rows + 1).tolist()} disagree on "
                f"{', '.join(differing)} -- "
                + " vs ".join(
                    "/".join(f"{X[r, c]:g}" for c in range(X.shape[1]) if input_names[c] in differing)
                    for r in rows
                )
            )
    if drift:
        print("INPUT DRIFT BETWEEN BLOCKS (grouped by position regardless):")
        print("\n".join(drift))
        if args.align_blocks_to_first:
            for g in range(per_block):
                rows = np.flatnonzero(recipe == g)
                X[rows] = X[rows[0]]
            print("  --align-blocks-to-first: later blocks re-encoded to block 1.")
        else:
            print(
                "  Not aligned. The GP is told these are different recipes, which "
                "understates the repeat evidence. Re-run with "
                "--align-blocks-to-first to apply block 1's encoding."
            )

    args.outdir.mkdir(parents=True, exist_ok=True)
    # On the DISPLAYED scale, which for optoelectronic is log10. A variance share
    # computed on the raw product and printed on a log axis describes a different
    # quantity from the one the reader is looking at: raw gives the block 49.3% and
    # log10 gives it 84.5%, because the raw scale is dominated by the handful of
    # largest values. The panel's numbers must be about the panel.
    stats = {
        name: variance_decomposition(_display(name, Y[:, j])[0], recipe, block)
        for j, name in enumerate(names)
    }

    folds: dict[str, dict] = {"rowwise": {}, "recipe": {}}
    for key, groups in (("rowwise", np.arange(n)), ("recipe", recipe)):
        for j, name in enumerate(names):
            mu, sd, collapsed = fold_predictions(config, X, Y[:, j], groups, seed=seed)
            folds[key][name] = {
                "predicted": mu.tolist(),
                "predictive_sd": sd.tolist(),
                "r2": r_squared(Y[:, j], mu),
                "spearman": float(spearmanr(Y[:, j], mu).statistic),
                "collapsed": list(collapsed),
            }
            print(
                f"  {key:8} {name:16} R2 {folds[key][name]['r2']:+.4f}"
                f"   collapsed {len(collapsed)} folds",
                flush=True,
            )

    frame = pd.DataFrame(
        {
            "recipe": recipe + 1,
            "block": block + 1,
            **{f"measured_{name}": Y[:, j] for j, name in enumerate(names)},
            **{
                f"{key}_pred_{name}": folds[key][name]["predicted"]
                for key in folds
                for name in names
            },
        }
    )
    frame.to_csv(args.outdir / "01_loo_parity_extended.csv", index=False)
    pd.DataFrame(stats).T.rename_axis("objective").to_csv(
        args.outdir / "boxplot_extended.csv"
    )
    (args.outdir / "extended_folds.json").write_text(
        json.dumps({"stats": stats, "folds": folds, "seed": seed}, indent=1),
        encoding="utf-8",
    )

    for path in (
        plot_boxes(args.outdir, names, Y, recipe, block, stats),
        plot_parity(args.outdir, names, Y, recipe, block, folds),
    ):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
