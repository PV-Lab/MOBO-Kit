"""Round-by-round utility boxplots across a beta x radius sweep, over three trials.

Same construction as ``scripts/plot_round_simulation.py``'s boxplots -- utility by
round, one panel per objective, every raw point drawn over its box -- swept over a
wider grid and replicated over three starting designs.

THE GRID.  beta in {9, 25, 36, 49} x radius in {0.05 ... 0.45} = 36 cells, chosen
after beta 9 and 25 at radius 0.25 showed the behaviour the group wanted to see
more of.

THE THREE TRIALS differ in ONE thing: where R0 comes from.

* ``real``   -- the 15 real recipes from the workbook, oracle-scored. The anchor.
* ``lhs_a``  -- a fresh 15-point Latin hypercube, seed 101.
* ``lhs_b``  -- a fresh 15-point Latin hypercube, seed 202.

Everything downstream is identical: the same frozen oracle scores every design,
the acquisitions run at the campaign seed 73 in every trial, and the candidate
pool is therefore the same 32768 recipes throughout. So a difference between
trials is attributable to the starting design and to nothing else. That is a
sensitivity check on R0, which is what was asked for -- it is NOT three
independent replicates of the whole pipeline, and the spread between trials
understates true run-to-run variability for that reason.

WHAT THE NUMBERS ARE.  Every value is a GP prediction. The oracle is fitted once
to the 15 real films and then frozen; R1 and R2 conditions were never fabricated.
This compares acquisition settings on a data-shaped landscape. It is not evidence
about the chemistry, and a tall box does not mean a good film.

RUNNING IT.  108 cells at ~195 s each is about six hours in one process, so the
work is sharded::

    # 12 workers, ~30 min wall clock
    for i in 0..11:  python scripts/plot_boxplot_sweep.py --workbook ... --shard i --num-shards 12
    python scripts/plot_boxplot_sweep.py --workbook ... --compose

``--compose`` reads the per-cell files and renders 12 pages (3 trials x 4 betas),
each page holding 9 radii x 3 objectives, plus a combined PDF and a summary CSV.
Y-limits are shared per objective across ALL pages, so any two panels anywhere in
the deliverable are directly comparable.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_round_simulation as prs  # noqa: E402

from mobo_kit.campaign import (  # noqa: E402
    build_objective_transform,
    fit_campaign_models,
    load_campaign_config,
    run_r0_lhs,
)
from mobo_kit.workbook_io import read_campaign_workbook  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="botorch")
warnings.filterwarnings("ignore", category=UserWarning, module="gpytorch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
torch.set_num_threads(1)

BETAS = (9.0, 25.0, 36.0, 49.0)
RADII = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)

#: (name, R0 source, LHS seed). The acquisition seed stays at the campaign's 73
#: for every trial, so only the starting design moves.
TRIALS = (
    ("real", "workbook", None),
    ("lhs_a", "lhs", 101),
    ("lhs_b", "lhs", 202),
)

ROUND_STYLE = prs.ROUND_STYLE
INK, INK_MUTED, SURFACE, SPINE = prs.INK, prs.INK_MUTED, prs.SURFACE, prs.SPINE

FOOTER = (
    "Every value is a GP prediction, not a measurement: the oracle is fitted once "
    "to the 15 real films and frozen, and no R1/R2 condition was ever fabricated. "
    "This compares acquisition settings on a data-shaped landscape, not chemistry.\n"
    "Rounds hold 15 / 5 / 3 conditions. A box over three numbers reports little "
    "more than those numbers, which is why every raw point is drawn on top."
)


def signal_caveat(config: Any) -> str:
    """Name the dead axes from THIS config, never from a remembered campaign.

    The line here used to read "uniformity carries no validated signal
    (permutation p = 0.82)", which is a fact about the first campaign's uniformity
    score on the first campaign's films. On the v3 contract that objective is a
    different construction and optoelectronic is dead as well, so a hard-coded
    caveat would have shipped the wrong evidence attached to the right warning --
    which is worse than no caveat, because it looks checked.
    """
    dead = [
        str(spec["name"])
        for spec in config["objectives"]["specs"]
        if str(spec.get("signal_status", "")) not in ("learnable", "")
    ]
    if not dead:
        return ""
    listed = " and ".join(dead) if len(dead) < 3 else ", ".join(dead)
    verb = "carries" if len(dead) == 1 else "carry"
    return (
        f"\n{listed} {verb} no learnable signal on this contract "
        f"({config['objectives']['contract_version']}): the model does not beat "
        "the leave-one-out null, so read those panels as exploration and not as a "
        "result."
    )


def cell_key(trial: str, beta: float, radius: float) -> str:
    return f"{trial}__beta_{beta:g}__radius_{radius:g}".replace(".", "p")


def all_cells(
    betas: Sequence[float] | None = None,
    radii: Sequence[float] | None = None,
) -> list[tuple[str, float, float]]:
    """Every (trial, beta, radius) to run. Filters keep the trial axis intact.

    Restricting the knobs never drops a trial: the trials are what turn three
    numbers per round into a distribution worth boxing, so a "one cell" run is
    still three campaigns from three starting designs.
    """
    return [
        (trial, beta, radius)
        for trial, _source, _seed in TRIALS
        for beta in (BETAS if betas is None else tuple(float(b) for b in betas))
        for radius in (RADII if radii is None else tuple(float(r) for r in radii))
    ]


def r0_for_trial(
    config: Any, trial: str, source: str, lhs_seed: int | None, workbook: Path
) -> np.ndarray:
    """The 15 starting conditions for a trial, in physical units."""
    if source == "workbook":
        contents = read_campaign_workbook(workbook, config)
        if contents.errors:
            raise SystemExit(f"Workbook read failed: {contents.errors}")
        return contents.inputs.to_numpy(float)
    return run_r0_lhs(config, n=15, seed=int(lhs_seed)).conditions.to_numpy(float)


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #


def run_shard(args: argparse.Namespace) -> int:
    config = load_campaign_config(args.config)
    seed = int((config.get("reproducibility") or {}).get("seed", 0))
    transform = build_objective_transform(config)
    reference = np.asarray(config["reference_point_utility"], dtype=float)

    contents = read_campaign_workbook(args.workbook, config)
    if contents.errors:
        for finding in contents.errors:
            print(f"  ERROR  {finding}")
        return 1
    X_real = contents.inputs.to_numpy(float)

    oracle, oracle_warnings = fit_campaign_models(
        config, X_real, contents.model_values.to_numpy(float), seed=seed
    )
    if oracle_warnings:
        print("ABORTING -- the oracle fit raised guard warnings:")
        for message in oracle_warnings:
            print(f"  {message}")
        return 1

    r0_by_trial = {
        name: r0_for_trial(config, name, source, lhs_seed, args.workbook)
        for name, source, lhs_seed in TRIALS
    }

    cells = all_cells(args.betas, args.radii)
    mine = cells[args.shard :: args.num_shards]
    out_dir = Path(args.output_dir) / "cells"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} of {len(cells)} cells")

    started = time.time()
    for position, (trial, beta, radius) in enumerate(mine, start=1):
        key = cell_key(trial, beta, radius)
        target = out_dir / f"{key}.npz"
        if target.exists() and not args.overwrite:
            print(f"  [{position}/{len(mine)}] {key} exists, skipping")
            continue
        cell_started = time.time()
        cell = prs.run_cell(
            config, oracle, r0_by_trial[trial], transform, reference,
            radius=radius, beta=beta, seed=seed,
        )
        np.savez_compressed(
            target,
            U_R0=cell["U"]["R0"], U_R1=cell["U"]["R1"], U_R2=cell["U"]["R2"],
            X_R0=cell["X"]["R0"], X_R1=cell["X"]["R1"], X_R2=cell["X"]["R2"],
            hv=np.array([cell["hv"]["R0"], cell["hv"]["R0+R1"], cell["hv"]["R0+R1+R2"]]),
            r1_spacing=float(cell["r1"].diagnostics["validity"]["min_pairwise_distance"]),
            r2_spacing=float(cell["r2"].diagnostics["validity"]["min_pairwise_distance"]),
            r1_edge=int(sum(cell["r1"].diagnostics["validity"]["boundary_coords_per_condition"])),
            r2_edge=int(sum(cell["r2"].diagnostics["validity"]["boundary_coords_per_condition"])),
            r1_hash=prs.batch_hash(cell["r1"].conditions),
            r2_hash=prs.batch_hash(cell["r2"].conditions),
            fit_warnings=len(cell["final_fit_warnings"]),
        )
        elapsed = time.time() - cell_started
        print(f"  [{position}/{len(mine)}] {key}  {elapsed:.0f}s", flush=True)
        if position == 1:
            remaining = elapsed * (len(mine) - 1) / 60.0
            print(f"      shard estimate: ~{remaining:.0f} min remaining", flush=True)
    print(f"shard {args.shard} done in {(time.time() - started) / 60:.1f} min")
    return 0


# --------------------------------------------------------------------------- #
# compose
# --------------------------------------------------------------------------- #


def _panel(axis, series, objective_index) -> None:
    values = [series[name][:, objective_index] for name in ("R0", "R1", "R2")]
    boxes = axis.boxplot(
        values,
        tick_labels=[f"{n}\nn={len(v)}" for n, v in zip(("R0", "R1", "R2"), values)],
        showmeans=True, showfliers=False, widths=0.55, patch_artist=True,
    )
    for patch, name in zip(boxes["boxes"], ("R0", "R1", "R2")):
        patch.set_facecolor(ROUND_STYLE[name][0])
        patch.set_alpha(0.22)
        patch.set_edgecolor(ROUND_STYLE[name][0])
    for key in ("whiskers", "caps", "medians"):
        for artist in boxes[key]:
            artist.set_color(INK_MUTED)
    for marker in boxes.get("means", ()):
        marker.set_markerfacecolor(INK)
        marker.set_markeredgecolor(INK)
        marker.set_markersize(5)
    rng = np.random.default_rng(0)
    for position, block in enumerate(values, start=1):
        colour = ROUND_STYLE[("R0", "R1", "R2")[position - 1]][0]
        axis.scatter(
            np.full(block.shape, position) + rng.normal(0, 0.045, block.shape),
            block, s=20, c=colour, edgecolors="white", linewidths=0.6, zorder=3,
        )
    axis.tick_params(labelsize=7, colors=INK_MUTED, length=2)
    axis.grid(axis="y", color="#e8e7e2", lw=0.7)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(SPINE)


def compose(args: argparse.Namespace) -> int:
    config = load_campaign_config(args.config)
    transform = build_objective_transform(config)
    names = list(transform.names)
    # Compose exactly what was run. Iterating the full sweep constants here while
    # the run was filtered would draw a page of "missing" panels around the one
    # cell anybody asked for.
    betas_used = BETAS if args.betas is None else tuple(float(b) for b in args.betas)
    radii_used = RADII if args.radii is None else tuple(float(r) for r in args.radii)
    cells_dir = Path(args.output_dir) / "cells"

    loaded: dict[str, Any] = {}
    missing: list[str] = []
    for trial, beta, radius in all_cells(args.betas, args.radii):
        key = cell_key(trial, beta, radius)
        path = cells_dir / f"{key}.npz"
        if path.exists():
            loaded[key] = np.load(path, allow_pickle=False)
        else:
            missing.append(key)
    print(f"loaded {len(loaded)} cells, missing {len(missing)}")
    if missing:
        for key in missing[:10]:
            print(f"  MISSING {key}")
        if not args.allow_partial:
            print("\nRefusing to compose an incomplete deliverable. Re-run the "
                  "missing shards, or pass --allow-partial.")
            return 1

    # Shared y-limits per objective across every page, so any two panels in the
    # whole deliverable are directly comparable. Without this a flatter cell can
    # look identical to a better one.
    limits = []
    for index in range(len(names)):
        stack = np.concatenate([
            np.concatenate([
                data["U_R0"][:, index], data["U_R1"][:, index], data["U_R2"][:, index]
            ])
            for data in loaded.values()
        ])
        low, high = float(stack.min()), float(stack.max())
        pad = 0.06 * (high - low if high > low else 1.0)
        limits.append((low - pad, high + pad))

    figures_dir = Path(args.output_dir) / "pages"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pages = len(TRIALS) * len(betas_used)
    pdf_path = Path(args.output_dir) / f"boxplot_sweep_{pages}pages.pdf"

    with PdfPages(pdf_path) as pdf:
        for trial, _source, lhs_seed in TRIALS:
            for beta in betas_used:
                fig, axes = plt.subplots(
                    len(radii_used), len(names),
                    figsize=(12.5, max(5.0, 26.0 * len(radii_used) / len(RADII))),
                    facecolor=SURFACE, squeeze=False,
                )
                for row, radius in enumerate(radii_used):
                    key = cell_key(trial, beta, radius)
                    data = loaded.get(key)
                    for column, objective in enumerate(names):
                        axis = axes[row, column]
                        axis.set_facecolor(SURFACE)
                        if data is None:
                            axis.text(0.5, 0.5, "missing", ha="center", va="center",
                                      fontsize=9, color=INK_MUTED)
                            axis.set_xticks([])
                            continue
                        series = {
                            "R0": data["U_R0"], "R1": data["U_R1"], "R2": data["U_R2"]
                        }
                        _panel(axis, series, column)
                        axis.set_ylim(*limits[column])
                        if row == 0:
                            axis.set_title(objective, fontsize=11, color=INK, pad=8)
                        if column == 0:
                            axis.set_ylabel(
                                f"radius {radius:g}\nutility",
                                fontsize=9, color=INK,
                            )
                        for round_name in ("R0", "R1", "R2"):
                            block = series[round_name][:, column]
                            rows.append({
                                "trial": trial, "beta": beta, "radius": radius,
                                "objective": objective, "round": round_name,
                                "n": int(block.size),
                                "mean_utility": float(block.mean()),
                                "median_utility": float(np.median(block)),
                                "max_utility": float(block.max()),
                                "hv_r0": float(data["hv"][0]),
                                "hv_r0_r1": float(data["hv"][1]),
                                "hv_r0_r1_r2": float(data["hv"][2]),
                                "r1_min_spacing": float(data["r1_spacing"]),
                                "r1_edge_coords": int(data["r1_edge"]),
                                "r1_batch_hash": data["r1_hash"].item(),
                                "r2_batch_hash": data["r2_hash"].item(),
                                "final_fit_warnings": int(data["fit_warnings"]),
                            })
                source = "15 real recipes" if trial == "real" else f"LHS seed {lhs_seed}"
                shown = sorted(radii_used)
                span = (
                    f"radius {shown[0]:g}"
                    if len(shown) == 1
                    else f"radius {shown[0]:g} → {shown[-1]:g}"
                )
                fig.suptitle(
                    f"Utility by round   |   trial {trial} ({source})   |   "
                    f"beta = {beta:g}   |   {span}",
                    fontsize=15, color=INK, y=0.995,
                )
                fig.tight_layout(rect=(0, 0.035, 1, 0.982))
                fig.text(
                    0.008, 0.006, "seed 73  |  " + FOOTER + signal_caveat(config),
                    fontsize=6.6, color=INK_MUTED, va="bottom", ha="left", wrap=True,
                )
                stem = f"page_{trial}_beta_{beta:g}".replace(".", "p")
                if len(radii_used) == 1:
                    stem += f"_radius_{radii_used[0]:g}".replace(".", "p")
                page = figures_dir / f"{stem}.png"
                fig.savefig(page, dpi=110, facecolor=SURFACE)
                pdf.savefig(fig, facecolor=SURFACE)
                plt.close(fig)
                print(f"  page {page.name}")

    summary = pd.DataFrame(rows).drop_duplicates()
    summary_path = Path(args.output_dir) / "boxplot_sweep_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nPDF     {pdf_path}")
    print(f"summary {summary_path}  ({len(summary)} rows)")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/campaign_d2d_perovskite_test.yaml"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("local_outputs/boxplot_sweep")
    )
    # A ratified cell needs no sweep. The grid is a decision, not a default:
    # running 108 cells to look at one is not thoroughness, it is 36x the
    # compute for the same answer.
    parser.add_argument(
        "--betas", nargs="+", type=float, default=None,
        help="restrict to these betas; default is the full sweep set",
    )
    parser.add_argument(
        "--radii", nargs="+", type=float, default=None,
        help="restrict to these radii; default is the full sweep set",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Compose with cells missing; each gap is drawn as 'missing'.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compose:
        return compose(args)
    return run_shard(args)


if __name__ == "__main__":
    raise SystemExit(main())
