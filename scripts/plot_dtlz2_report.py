"""Figures for the DTLZ2 acceptance run: how the optimiser actually moves.

Renders the progression the campaign goes through -- initial design, GP fit,
acquisition surface, selected batch, refit, repeat -- plus objective-space and
hypervolume views.

    python scripts/plot_dtlz2_report.py --out local_outputs/dtlz2_report

The design space is 10-dimensional, so every contour is a 2-D slice on
(x0, x1) with x2..x9 held at 0.5. That slice is chosen deliberately: for DTLZ2
the last k=8 inputs are "distance" variables whose optimum is exactly 0.5, and
x0/x1 are the "position" variables that move you along the Pareto front. So the
slice contains the true optimal surface, and a well-behaved optimiser should be
seen concentrating on it.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions.multi_objective import DTLZ2
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from matplotlib.colors import LinearSegmentedColormap

from mobo_kit.campaign import (
    _fit_models,
    _normalise,
    build_objective_transform,
    run_r0_lhs,
    run_r1_ucb,
    run_r2_qlognehvi,
)
from mobo_kit.design import build_design_from_config
from mobo_kit.ucb_hvi import score_ucb_hvi_pool

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

# dataviz reference palette, categorical slots 1-3 (the documented all-pairs-safe
# set: CVD dE 9.2 light / 9.4 dark, normal-vision 24.0 / 20.9)
R0_COLOR, R1_COLOR, R2_COLOR = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
# sequential blue ramp, 100 -> 700, for magnitude
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# second sequential context takes the next categorical hue (orange)
ORANGE_RAMP = ["#fbe3d5", "#f6c3a4", "#f0a074", "#eb6834", "#c14f22", "#933a17", "#66270e"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
ACQ = LinearSegmentedColormap.from_list("seq_orange", ORANGE_RAMP)

D, M, GRID = 10, 3, 45
SLICE_X, SLICE_Y = 0, 1


def config(pool: int = 1024, mc: int = 32) -> dict:
    return {
        "inputs": [
            {"name": f"x{i}", "start": 0.0, "stop": 1.0, "step": 0.05} for i in range(D)
        ],
        "objectives": {
            "contract_version": "TEST_ONLY-dtlz2-v1",
            "scaling_mode": "fixed_affine",
            "specs": [
                {
                    "name": f"f{i}", "goal": "maximize", "transform": "affine",
                    "model_source_column": f"f{i}",
                    "lower_anchor": -2.0, "upper_anchor": 0.0,
                }
                for i in range(M)
            ],
        },
        "reference_point_utility": [-0.01] * M,
        "rounds": {
            "r1": {"method": "ucb_hvi", "batch_size": 5, "replicates_per_condition": 3,
                   "beta": 4.0, "candidate_pool_size": pool, "posterior_samples": 256,
                   "moment_method": "monte_carlo"},
            "r2": {"method": "qlognehvi", "batch_size": 3,
                   "replicates_per_condition": 3, "candidate_pool_size": pool,
                   "mc_samples": mc},
        },
        "local_penalization": {"radius": 0.25, "min_batch_distance": 0.15,
                               "min_observed_distance": 0.0, "dimension_weights": None},
        "model": {"variant": "dim_scaled_prior"},
        "reproducibility": {"seed": 73},
        "constraints": [],
    }


CFG = config()
PROBLEM = DTLZ2(dim=D, num_objectives=M, negate=True).to(dtype=torch.double)
TRANSFORM = build_objective_transform(CFG)
DESIGN = build_design_from_config(CFG)
REF = torch.tensor(CFG["reference_point_utility"], dtype=torch.double)


def evaluate(X):
    return PROBLEM(torch.tensor(np.asarray(X, float), dtype=torch.double)).numpy()


def hypervolume(Y):
    U = TRANSFORM(torch.tensor(np.asarray(Y, float), dtype=torch.double))
    return Hypervolume(ref_point=REF).compute(U[is_non_dominated(U)])


def slice_grid():
    axis = np.linspace(0.0, 1.0, GRID)
    xx, yy = np.meshgrid(axis, axis)
    pts = np.full((GRID * GRID, D), 0.5)
    pts[:, SLICE_X] = xx.ravel()
    pts[:, SLICE_Y] = yy.ravel()
    return axis, xx, yy, pts


def surfaces(X_phys, Y_raw, seed=73):
    """Posterior utility mean/sd and the UCB-HVI acquisition over the slice."""
    axis, xx, yy, pts = slice_grid()
    model = _fit_models(CFG, X_phys, _normalise(DESIGN, X_phys), Y_raw, seed)
    grid_t = torch.tensor(pts, dtype=torch.double)

    model.eval()
    with torch.no_grad():
        post = model.posterior(grid_t)
        util = TRANSFORM.expected_transform(post.mean, post.variance).numpy()
        sd = post.variance.sqrt().numpy()

    scored = score_ucb_hvi_pool(
        model, grid_t, Y_raw, TRANSFORM, REF.numpy(),
        beta=CFG["rounds"]["r1"]["beta"], mc_samples=64, seed=seed,
    )
    # base_score is the raw hypervolume improvement per candidate
    acq = np.asarray(scored.base_score, dtype=float)
    return {
        "axis": axis, "xx": xx, "yy": yy,
        "mean": util.mean(axis=1).reshape(GRID, GRID),
        "sd": sd.mean(axis=1).reshape(GRID, GRID),
        "acq": acq.reshape(GRID, GRID),
    }


def style(ax, title, xlabel=True, ylabel=True):
    ax.set_title(title, fontsize=10, color=INK, pad=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("x0" if xlabel else "", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("x1" if ylabel else "", fontsize=9, color=INK_MUTED)
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)
    for spine in ax.spines.values():
        spine.set_color("#d8d7d2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="local_outputs/dtlz2_report")
    ap.add_argument("--seed", type=int, default=73)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("running the campaign...")
    r0 = run_r0_lhs(CFG, n=15, seed=args.seed)
    X0 = r0.conditions.to_numpy(float); Y0 = evaluate(X0)
    r1 = run_r1_ucb(CFG, X0, Y0, seed=args.seed)
    X1 = r1.conditions.to_numpy(float); Y1 = evaluate(X1)
    X01, Y01 = np.vstack([X0, X1]), np.vstack([Y0, Y1])
    r2 = run_r2_qlognehvi(CFG, X01, Y01, seed=args.seed)
    X2 = r2.conditions.to_numpy(float); Y2 = evaluate(X2)
    X012, Y012 = np.vstack([X01, X2]), np.vstack([Y01, Y2])
    hv = [hypervolume(Y0), hypervolume(Y01), hypervolume(Y012)]
    print(f"  hypervolume: {hv[0]:.4f} -> {hv[1]:.4f} -> {hv[2]:.4f}")

    print("building surfaces (3 GP fits over a 45x45 slice)...")
    stages = [
        ("After R0: 15 LHS points", surfaces(X0, Y0, args.seed), X0, X1, "R1"),
        ("After R1: 20 points", surfaces(X01, Y01, args.seed), X01, X2, "R2"),
        ("After R2: 23 points", surfaces(X012, Y012, args.seed), X012, None, None),
    ]

    # ---------------- figure 1: the progression ----------------
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.2), facecolor=SURFACE)
    for col, (title, s, seen, chosen, label) in enumerate(stages):
        # row 0 -- posterior mean utility
        ax = axes[0, col]
        cf = ax.contourf(s["xx"], s["yy"], s["mean"], levels=14, cmap=SEQ)
        ax.scatter(seen[:, SLICE_X], seen[:, SLICE_Y], s=26, c="white",
                   edgecolors=INK, linewidths=0.9, zorder=3, label="observed")
        style(ax, f"{title}\nGP posterior mean utility", xlabel=False)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

        # row 1 -- posterior uncertainty
        ax = axes[1, col]
        cf = ax.contourf(s["xx"], s["yy"], s["sd"], levels=14, cmap=SEQ)
        ax.scatter(seen[:, SLICE_X], seen[:, SLICE_Y], s=26, c="white",
                   edgecolors=INK, linewidths=0.9, zorder=3)
        style(ax, "GP posterior uncertainty (sd)", xlabel=False)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

        # row 2 -- acquisition + what it picked
        ax = axes[2, col]
        cf = ax.contourf(s["xx"], s["yy"], s["acq"], levels=14, cmap=ACQ)
        ax.scatter(seen[:, SLICE_X], seen[:, SLICE_Y], s=20, c="white",
                   edgecolors=INK_MUTED, linewidths=0.7, zorder=3)
        if chosen is not None:
            colour = R1_COLOR if label == "R1" else R2_COLOR
            ax.scatter(chosen[:, SLICE_X], chosen[:, SLICE_Y], s=150, marker="*",
                       c=colour, edgecolors="white", linewidths=1.4, zorder=4,
                       label=f"{label} selected ({len(chosen)})")
            ax.legend(loc="upper right", fontsize=8, frameon=True,
                      facecolor="white", edgecolor="#d8d7d2")
            style(ax, f"UCB-HVI acquisition -> {label} batch")
        else:
            style(ax, "UCB-HVI acquisition (final model)")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=7)

    fig.suptitle(
        "DTLZ2 acceptance run: 2-D slice at (x0, x1), x2..x9 = 0.5\n"
        "Stars mark the batch each acquisition surface selected",
        fontsize=12.5, color=INK, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out / "01_progression.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---------------- figure 2: objective space ----------------
    fig = plt.figure(figsize=(12.5, 5.4), facecolor=SURFACE)
    U0 = TRANSFORM(torch.tensor(Y0)).numpy()
    U1 = TRANSFORM(torch.tensor(Y1)).numpy()
    U2 = TRANSFORM(torch.tensor(Y2)).numpy()
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(1, 3, i + 1, facecolor=SURFACE)
        for U, c, name in ((U0, R0_COLOR, "R0 (15)"), (U1, R1_COLOR, "R1 (5)"),
                           (U2, R2_COLOR, "R2 (3)")):
            ax.scatter(U[:, a], U[:, b], s=52, c=c, edgecolors="white",
                       linewidths=1.2, label=name, zorder=3)
        # Ring the Pareto-optimal points instead of connecting them: this is a
        # 2-D projection of a 3-D front, so the points carry no ordering along
        # either axis and a connecting line would invent one.
        allU = np.vstack([U0, U1, U2])
        front = allU[is_non_dominated(torch.tensor(allU)).numpy()]
        ax.scatter(front[:, a], front[:, b], s=170, facecolors="none",
                   edgecolors=INK, linewidths=1.5, zorder=2,
                   label="Pareto optimal (3-D)" if i == 0 else None)
        ax.set_xlabel(f"utility f{a}", fontsize=9, color=INK_MUTED)
        ax.set_ylabel(f"utility f{b}", fontsize=9, color=INK_MUTED)
        ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)
        for spine in ax.spines.values():
            spine.set_color("#d8d7d2")
        if i == 0:
            ax.legend(fontsize=8, frameon=True, facecolor="white",
                      edgecolor="#d8d7d2", loc="lower left")
    fig.suptitle("Objective space: where each round landed (higher is better)",
                 fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "02_objective_space.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---------------- figure 3: hypervolume vs random ----------------
    print("running the random baseline across 5 seeds...")
    grids = [np.asarray(g, float) for g in DESIGN.var_array]
    bo_curves, rand_curves = [], []
    for seed in (1, 2, 3, 4, 5):
        a0 = run_r0_lhs(CFG, n=15, seed=seed)
        Xa = a0.conditions.to_numpy(float); Ya = evaluate(Xa)
        b1 = run_r1_ucb(CFG, Xa, Ya, seed=seed)
        Yb = evaluate(b1.conditions.to_numpy(float))
        Xab = np.vstack([Xa, b1.conditions.to_numpy(float)])
        Yab = np.vstack([Ya, Yb])
        b2 = run_r2_qlognehvi(CFG, Xab, Yab, seed=seed)
        Yc = evaluate(b2.conditions.to_numpy(float))
        bo_curves.append([hypervolume(Ya), hypervolume(Yab),
                          hypervolume(np.vstack([Yab, Yc]))])
        rng = np.random.default_rng(seed)
        Xr1 = np.column_stack([rng.choice(g, size=5) for g in grids])
        Xr2 = np.column_stack([rng.choice(g, size=3) for g in grids])
        Yr1, Yr2 = evaluate(Xr1), evaluate(Xr2)
        rand_curves.append([hypervolume(Ya), hypervolume(np.vstack([Ya, Yr1])),
                            hypervolume(np.vstack([Ya, Yr1, Yr2]))])
    bo = np.array(bo_curves); rand = np.array(rand_curves)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), facecolor=SURFACE)
    x = np.array([15, 20, 23])
    for arr, colour, name in ((bo, R1_COLOR, "Bayesian optimisation"),
                              (rand, R0_COLOR, "Random on-grid search")):
        ax.fill_between(x, arr.min(axis=0), arr.max(axis=0), color=colour, alpha=0.14)
        ax.plot(x, arr.mean(axis=0), color=colour, lw=2.0, marker="o", ms=8,
                markeredgecolor="white", markeredgewidth=1.4, label=name, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["R0\n15 points", "+R1\n20 points", "+R2\n23 points"],
                       fontsize=9, color=INK_MUTED)
    ax.set_ylabel("hypervolume (utility space)", fontsize=9.5, color=INK_MUTED)
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=3)
    ax.grid(axis="y", color="#e8e7e2", lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#d8d7d2")
    ax.legend(fontsize=9, frameon=True, facecolor="white", edgecolor="#d8d7d2",
              loc="upper left")
    ax.set_title(
        f"Hypervolume gain at equal budget, 5 seeds (band = min-max)\n"
        f"mean gain: BO +{(bo[:,2]-bo[:,0]).mean():.3f}   "
        f"random +{(rand[:,2]-rand[:,0]).mean():.3f}",
        fontsize=11.5, color=INK, pad=10,
    )
    fig.tight_layout()
    fig.savefig(out / "03_hypervolume.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    np.savetxt(out / "hypervolume_bo.csv", bo, delimiter=",",
               header="hv_R0,hv_R1,hv_R2", comments="")
    np.savetxt(out / "hypervolume_random.csv", rand, delimiter=",",
               header="hv_R0,hv_R1,hv_R2", comments="")
    print(f"\nwrote 3 figures + 2 CSVs to {out}")


if __name__ == "__main__":
    main()
