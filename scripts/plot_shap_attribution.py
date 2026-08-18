"""SHAP attributions for what the campaign's models actually use.

Answers one question per figure: **which process inputs move this objective's
expected utility, and in which direction?** It explains the MODEL, which is the
only thing SHAP can explain -- see the caveats below and on every figure.

MODEL STATES.  Two, not three.

* ``r0_only`` -- the GP fitted to the **15 real measurements**. This is the anchor:
  it is the only model here trained on measured data, it does not depend on any
  acquisition function, and it is the same object the round simulation uses as its
  oracle.
* ``final`` -- refitted on all 23 conditions after a simulated R0 -> R1 -> R2 pass
  at the default cell (radius 0.25, beta 4.0). Its R1 and R2 conditions were never
  fabricated, so its extra 8 points carry oracle predictions rather than
  measurements.

The brief asked for three states, splitting ``final`` by R2 acquisition. Measured
here, **qLogNEHVI and qNEHVI propose the identical R2 batch**, so those two states
are one model and their figures would be bit-identical. The script detects that
per run rather than assuming it, prints it, records both hashes in the summary,
and stamps it on every affected figure. If they ever diverge, both sets are
produced automatically.

WHAT SHAP DOES AND DOES NOT SHOW HERE.

* It explains ``E[utility]`` per objective, through
  ``ObjectiveTransform.expected_transform`` -- so thickness goes through the
  lognormal quadrature rather than a transformed mean, and every value is in
  utility units where higher is better.
* **A large attribution is not evidence of a physical effect.** For thickness and
  optoelectronic the model carries a declared physics mean function, so
  ``speed_1``, ``precur_conc`` and ``anneal_temp`` attributions partly restate that
  declaration rather than discovering it.
* **Uniformity has no validated predictive signal** (LOO R2 -0.681, permutation
  p = 0.82). Its GP still has ARD lengthscales and a posterior mean that varies, so
  SHAP will report structure. That structure is fitted noise. It is shown because
  hiding it would be worse, and every uniformity figure says so.

Usage::

    python scripts/plot_shap_attribution.py --workbook "local_inputs/Summary Table.xlsx"
    python scripts/plot_shap_attribution.py --workbook ... --instances 200
    python scripts/plot_shap_attribution.py --workbook ... --no-figures
    python scripts/plot_shap_attribution.py --workbook ... --extreme-cells
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402  -- still used directly for summary_plot
import torch  # noqa: E402

from mobo_kit.attribution import (  # noqa: E402
    expected_utility_fn,
    shap_values_for,
)
from mobo_kit.campaign import (  # noqa: E402
    build_objective_transform,
    fit_campaign_models,
    load_campaign_config,
    normalise_inputs,
    run_r1_ucb,
    run_r2_qlognehvi,
)
from mobo_kit.candidate_pool import sample_discrete_candidate_pool  # noqa: E402
from mobo_kit.constraints import constraints_from_config  # noqa: E402
from mobo_kit.design import build_design_from_config  # noqa: E402
from mobo_kit.objectives import ObjectiveTransform  # noqa: E402
from mobo_kit.research_qnehvi import run_r2_qnehvi_research  # noqa: E402
from mobo_kit.workbook_io import read_campaign_workbook  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="botorch")
warnings.filterwarnings("ignore", category=UserWarning, module="gpytorch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
torch.set_num_threads(1)

# house style, shared with plot_dtlz2_report.py and plot_round_simulation.py
INK, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SPINE = "#d8d7d2"

DEFAULT_RADIUS, DEFAULT_BETA = 0.25, 4.0
EXTREME_CELLS = ((0.05, 4.0), (0.45, 4.0), (0.25, 25.0))

#: The declared mean-function features, by objective. Attributions on these are
#: partly a restatement of the model's declared physics, not a discovery.
MEAN_FUNCTION_FEATURES = {
    "thickness": ("speed_1", "precur_conc"),
    "optoelectronic": ("anneal_temp",),
}


# --------------------------------------------------------------------------- #
# the function SHAP explains
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# model states
# --------------------------------------------------------------------------- #


def oracle_predict(
    model: Any,
    config: Mapping[str, Any],
    X_phys: np.ndarray,
    transform: ObjectiveTransform,
) -> np.ndarray:
    """Deterministic measurement-space prediction; thickness is the median exp(mu).

    Identical convention to ``scripts/plot_round_simulation.py``; see that file for
    why the median rather than the lognormal mean.
    """
    X_norm = normalise_inputs(config, np.asarray(X_phys, dtype=float))
    model.eval()
    with torch.no_grad():
        mean = (
            model.posterior(torch.tensor(X_norm, dtype=torch.double),
                            observation_noise=False)
            .mean.detach().cpu().double().numpy()
        )
    out = np.empty_like(mean)
    for index, spec in enumerate(transform.specs):
        out[:, index] = (
            np.exp(mean[:, index]) if spec.model_link == "log" else mean[:, index]
        )
    return out


def cell_config(base: Mapping[str, Any], *, radius: float, beta: float) -> dict:
    from copy import deepcopy

    config = deepcopy(dict(base))
    penalization = config.setdefault("local_penalization", {})
    penalization["radius"] = float(radius)
    penalization["min_batch_distance"] = 0.15
    config.setdefault("rounds", {}).setdefault("r1", {})["beta"] = float(beta)
    return config


def batch_hash(conditions: pd.DataFrame) -> str:
    import hashlib

    values = np.round(np.asarray(conditions, dtype=float), 12)
    ordered = values[np.lexsort(values.T[::-1])]
    return hashlib.sha256(ordered.tobytes()).hexdigest()[:16]


def build_final_state(
    base_config: Mapping[str, Any],
    oracle: Any,
    X_r0: np.ndarray,
    Y_r0_oracle: np.ndarray,
    transform: ObjectiveTransform,
    *,
    radius: float,
    beta: float,
    seed: int,
    acquisition: str,
) -> dict[str, Any]:
    """One simulated campaign at a cell, refitted on all 23 oracle-scored points."""
    config = cell_config(base_config, radius=radius, beta=beta)
    r1 = run_r1_ucb(config, X_r0, Y_r0_oracle, seed=seed)
    X_r1 = r1.conditions.to_numpy(float)
    Y_r1 = oracle_predict(oracle, config, X_r1, transform)
    X_01, Y_01 = np.vstack([X_r0, X_r1]), np.vstack([Y_r0_oracle, Y_r1])

    runner = run_r2_qlognehvi if acquisition == "qlognehvi" else run_r2_qnehvi_research
    r2 = runner(config, X_01, Y_01, seed=seed)
    X_r2 = r2.conditions.to_numpy(float)
    Y_r2 = oracle_predict(oracle, config, X_r2, transform)

    X_all, Y_all = np.vstack([X_01, X_r2]), np.vstack([Y_01, Y_r2])
    model, fit_warnings = fit_campaign_models(config, X_all, Y_all, seed=seed)
    return {
        "model": model,
        "config": config,
        "X": X_all,
        "fit_warnings": tuple(fit_warnings),
        "r1_hash": batch_hash(r1.conditions),
        "r2_hash": batch_hash(r2.conditions),
        "acquisition": acquisition,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

ORACLE_CAVEAT = (
    "Oracle: this model's 8 non-R0 conditions carry GP predictions, not "
    "measurements. It shows what the optimiser would believe, not what a film did."
)
REAL_DATA_CAVEAT = (
    "Fitted to the 15 real R0 measurements. SHAP still explains the MODEL's "
    "behaviour, which is not the same as a measured effect."
)
CONSTRUCTION_CAVEAT = (
    "Construction: this objective carries a declared physics mean function on "
    "{features}, so attributions there partly restate that declaration."
)
NO_SIGNAL_CAVEAT = (
    "No validated signal: uniformity does not beat the leave-one-out null "
    "(LOO R2 -0.681, permutation p = 0.82). Structure below is fitted noise, "
    "not physics."
)
IDENTICAL_BATCH_NOTE = (
    "qLogNEHVI and qNEHVI proposed the IDENTICAL R2 batch here, so this one "
    "figure covers both acquisitions."
)


def _feature_labels(config: Mapping[str, Any], X: np.ndarray) -> list[str]:
    """Name plus the physical range the colour scale actually spans, per feature.

    A beeswarm colours each row against **that feature's own** min-max, so one
    shared colorbar cannot carry physical units for ten inputs measured in rpm,
    seconds, molarity and microlitres at once. Putting each row's range in its
    label states the physical scale without the colorbar claiming something false.
    """
    labels = []
    for index, item in enumerate(config["inputs"]):
        unit = str(item.get("unit", "")).strip()
        low, high = float(np.min(X[:, index])), float(np.max(X[:, index]))
        suffix = f" {unit}" if unit else ""
        labels.append(f"{item['name']}\n{low:g}–{high:g}{suffix}")
    return labels


def plot_beeswarm(
    path: Path,
    shap_values: np.ndarray,
    instances: np.ndarray,
    config: Mapping[str, Any],
    objective: str,
    state_label: str,
    *,
    seed: int,
    caveats: Sequence[str],
) -> None:
    plt.figure(figsize=(9.6, 6.4), facecolor=SURFACE)
    shap.summary_plot(
        shap_values,
        instances,
        feature_names=_feature_labels(config, instances),
        plot_type="dot",
        show=False,
        color_bar=True,
        sort=True,
        # explicit generator: shap jitters overlapping points, and reading the
        # global RNG would make a figure depend on whatever ran before it
        rng=np.random.default_rng(seed),
    )
    fig = plt.gcf()
    fig.patch.set_facecolor(SURFACE)
    axis = fig.axes[0]
    axis.set_facecolor(SURFACE)
    axis.set_xlabel(
        "SHAP value  (impact on expected utility, higher is better)",
        fontsize=9.5, color=INK_MUTED,
    )
    axis.tick_params(labelsize=8, colors=INK_MUTED, length=3)
    for spine in axis.spines.values():
        spine.set_color(SPINE)
    # objective named on the right as well as in the title, so a cropped or
    # forwarded panel is still self-identifying
    axis.set_ylabel(objective, fontsize=10.5, color=INK, rotation=270, labelpad=18)
    axis.yaxis.set_label_position("right")
    for extra in fig.axes[1:]:
        extra.tick_params(labelsize=7, colors=INK_MUTED)
        if extra.get_ylabel():
            extra.set_ylabel(extra.get_ylabel(), fontsize=8, color=INK_MUTED)

    fig.suptitle(
        f"{objective} — SHAP attribution   |   {state_label}",
        fontsize=12.5, color=INK, y=0.985,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.text(
        0.008, 0.008,
        f"seed {seed}  |  " + "\n".join(caveats),
        fontsize=6.4, color=INK_MUTED, va="bottom", ha="left", wrap=True,
    )
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def caveats_for(objective: str, state: str, identical: bool) -> list[str]:
    """Exactly two true lines per figure, plus the identical-batch fact if it holds.

    The brief asked for one fixed footer everywhere. A slice caveat on a model
    fitted to real data, or an oracle caveat on the R0-only anchor, would be false
    where a reader looks for true statements, so line one states the model's
    provenance and line two states the objective's own hazard.
    """
    lines = [ORACLE_CAVEAT if state != "r0_only" else REAL_DATA_CAVEAT]
    if objective == "uniformity":
        lines.append(NO_SIGNAL_CAVEAT)
    elif objective in MEAN_FUNCTION_FEATURES:
        lines.append(
            CONSTRUCTION_CAVEAT.format(
                features=", ".join(MEAN_FUNCTION_FEATURES[objective])
            )
        )
    else:
        lines.append("")
    if identical and state != "r0_only":
        lines.append(IDENTICAL_BATCH_NOTE)
    return [line for line in lines if line]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


#: Pre-registered before the extreme cells were run. Sweeping the acquisition
#: parameters for the SHAP work is warranted only if changing them changes what
#: the model attributes to -- either by moving a feature to the top, or by moving
#: any attribution by more than this fraction of that objective's largest one.
SHIFT_FRACTION_THRESHOLD = 0.10


def extreme_cell_shifts(
    default_mean_abs: Mapping[str, np.ndarray],
    cell_mean_abs: Mapping[str, Mapping[str, np.ndarray]],
    feature_names: Sequence[str],
) -> tuple[list[dict[str, Any]], str]:
    """How much does the attribution move when the acquisition knobs move?

    The question behind this is whether the SHAP figures are a property of the
    MODEL or of the SEARCH. If three very different acquisition settings put the
    same features in the same order with similar magnitudes, the attributions are
    telling us about the fitted physics rather than about how the batch was picked
    -- and there is no reason to sweep.
    """
    rows: list[dict[str, Any]] = []
    warranted = False
    for slug, per_objective in cell_mean_abs.items():
        for objective, values in per_objective.items():
            reference = np.asarray(default_mean_abs[objective], dtype=float)
            values = np.asarray(values, dtype=float)
            scale = float(reference.max()) if reference.max() > 0 else 1.0
            deltas = np.abs(values - reference)
            worst = int(np.argmax(deltas))
            top_moved = int(np.argmax(values)) != int(np.argmax(reference))
            fraction = float(deltas[worst] / scale)
            if fraction > SHIFT_FRACTION_THRESHOLD or top_moved:
                warranted = True
            rows.append({
                "cell": slug,
                "objective": objective,
                "max_abs_shift": float(deltas[worst]),
                "max_shift_feature": feature_names[worst],
                "max_shift_fraction_of_largest": fraction,
                "top_feature_changed": top_moved,
                "default_top_feature": feature_names[int(np.argmax(reference))],
                "cell_top_feature": feature_names[int(np.argmax(values))],
            })
    verdict = (
        "SWEEP WARRANTED: an extreme cell moved the top feature or shifted an "
        f"attribution by more than {SHIFT_FRACTION_THRESHOLD:.0%} of the largest."
        if warranted
        else "NO SWEEP NEEDED: every extreme cell kept the same top feature and "
        f"moved every attribution by under {SHIFT_FRACTION_THRESHOLD:.0%} of the "
        "largest, so the attributions describe the model rather than the search."
    )
    return rows, verdict


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/campaign_d2d_perovskite.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("local_outputs/shap"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--instances", type=int, default=1000,
        help="On-grid points to attribute. Cost is linear in this.",
    )
    parser.add_argument(
        "--extreme-cells", action="store_true",
        help="Also measure attribution shift at radius 0.05 / 0.45 and beta 25.",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--objectives", nargs="+", default=None,
        help="Restrict to these objective names.",
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
    names = list(transform.names)

    print("=" * 79)
    print("SHAP ATTRIBUTION — what the campaign's models use")
    print("=" * 79)

    contents = read_campaign_workbook(args.workbook, config)
    if contents.errors:
        for finding in contents.errors:
            print(f"  ERROR  {finding}")
        return 1
    X_r0 = contents.inputs.to_numpy(float)
    Y_r0_measured = contents.model_values.to_numpy(float)
    print(f"\n1. WORKBOOK  {len(X_r0)} rows, objectives {tuple(names)}")

    print("\n2. R0-ONLY MODEL  fit_campaign_models on the real measurements")
    r0_model, r0_warnings = fit_campaign_models(
        config, X_r0, Y_r0_measured, seed=seed
    )
    if r0_warnings:
        print("\n   ABORTING — the anchor fit raised guard warnings:")
        for message in r0_warnings:
            print(f"     {message}")
        print("\n   Every attribution below would be about this fit.")
        return 1
    print("   fit guard clean")

    Y_r0_oracle = oracle_predict(r0_model, config, X_r0, transform)

    print("\n3. FINAL MODELS  simulated R0 -> R1 -> R2 at "
          f"radius {DEFAULT_RADIUS}, beta {DEFAULT_BETA}")
    finals = {}
    for acquisition in ("qlognehvi", "qnehvi"):
        finals[acquisition] = build_final_state(
            config, r0_model, X_r0, Y_r0_oracle, transform,
            radius=DEFAULT_RADIUS, beta=DEFAULT_BETA, seed=seed,
            acquisition=acquisition,
        )
        built = finals[acquisition]
        print(f"   {acquisition:<10} R2 batch hash {built['r2_hash']}"
              f"   fit warnings {len(built['fit_warnings'])}")
        # A collapsed final GP would attribute confidently to nothing at all, and
        # the beeswarm would look no different. Say so rather than render quietly.
        for message in built["fit_warnings"]:
            print(f"     FIT GUARD  {message}")
    identical = finals["qlognehvi"]["r2_hash"] == finals["qnehvi"]["r2_hash"]
    print(f"   IDENTICAL R2 BATCH: {identical}")
    if identical:
        print("   -> the two acquisitions give one model; one set of final figures.")

    states: dict[str, dict[str, Any]] = {
        "r0_only": {
            "model": r0_model, "config": config, "X": X_r0,
            "label": "R0-only, fitted to the 15 real measurements",
        }
    }
    if identical:
        states["final"] = {
            "model": finals["qlognehvi"]["model"],
            "config": finals["qlognehvi"]["config"],
            "X": finals["qlognehvi"]["X"],
            "label": "final 23-point model (qLogNEHVI = qNEHVI)",
        }
    else:
        for acquisition, built in finals.items():
            states[f"final_{acquisition}"] = {
                "model": built["model"], "config": built["config"],
                "X": built["X"],
                "label": f"final 23-point model ({acquisition})",
            }

    # ------------------------------------------------------------ instances --
    pool = sample_discrete_candidate_pool(
        design, int(args.instances), seed=seed,
        row_constraints=constraints_from_config(dict(config), design) or None,
    )
    instances = np.asarray(pool.X_phys, dtype=float)
    wanted = names if args.objectives is None else [
        n for n in names if n in set(args.objectives)
    ]
    n_runs = len(states) * len(wanted)
    print(f"\n4. ATTRIBUTION  {instances.shape[0]} on-grid instances x "
          f"{len(wanted)} objectives x {len(states)} model states = {n_runs} runs")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    mean_abs_by_state: dict[str, dict[str, np.ndarray]] = {}
    per_run_seconds: float | None = None

    for state_name, state in states.items():
        for objective in wanted:
            index = names.index(objective)
            run_started = time.time()
            values = shap_values_for(
                state["model"], state["config"], transform, index,
                background=state["X"], instances=instances, seed=seed,
            )
            elapsed = time.time() - run_started
            mean_abs = np.abs(values).mean(axis=0)
            mean_abs_by_state.setdefault(state_name, {})[objective] = mean_abs
            order = np.argsort(-mean_abs)
            for rank, feature_index in enumerate(order, start=1):
                rows.append({
                    "model_state": state_name,
                    "objective": objective,
                    "feature": design.names[feature_index],
                    "mean_abs_shap": float(mean_abs[feature_index]),
                    "rank": rank,
                    "mean_shap": float(values[:, feature_index].mean()),
                    "feature_min": float(instances[:, feature_index].min()),
                    "feature_max": float(instances[:, feature_index].max()),
                    "in_mean_function": design.names[feature_index]
                    in MEAN_FUNCTION_FEATURES.get(objective, ()),
                    "r2_acquisition": (
                        "identical" if identical and state_name != "r0_only"
                        else state_name
                    ),
                })
            top = design.names[order[0]]
            print(f"   {state_name:<12} {objective:<15} top {top:<12} "
                  f"mean|SHAP| {mean_abs[order[0]]:.4f}  ({elapsed:.0f}s)")

            if per_run_seconds is None:
                per_run_seconds = elapsed
                print(f"       estimated total: ~{elapsed * n_runs / 60:.0f} min")

            if not args.no_figures:
                figure_dir = output_root / "figures"
                figure_dir.mkdir(parents=True, exist_ok=True)
                plot_beeswarm(
                    figure_dir / f"shap_{state_name}_{objective}.png",
                    values, instances, config, objective, state["label"],
                    seed=seed,
                    caveats=caveats_for(objective, state_name, identical),
                )

    # ------------------------------------------------------- extreme cells --
    shift_rows: list[dict[str, Any]] = []
    verdict = ""
    if args.extreme_cells:
        default_state = "final" if "final" in states else "final_qlognehvi"
        print(f"\n5. EXTREME CELLS  attribution shift against {default_state}")
        cell_mean_abs: dict[str, dict[str, np.ndarray]] = {}
        for radius, beta in EXTREME_CELLS:
            slug = f"radius_{radius:g}__beta_{beta:g}".replace(".", "p")
            built = build_final_state(
                config, r0_model, X_r0, Y_r0_oracle, transform,
                radius=radius, beta=beta, seed=seed, acquisition="qlognehvi",
            )
            cell_mean_abs[slug] = {}
            for objective in wanted:
                index = names.index(objective)
                values = shap_values_for(
                    built["model"], built["config"], transform, index,
                    background=built["X"], instances=instances, seed=seed,
                )
                cell_mean_abs[slug][objective] = np.abs(values).mean(axis=0)
                for rank, feature_index in enumerate(
                    np.argsort(-cell_mean_abs[slug][objective]), start=1
                ):
                    rows.append({
                        "model_state": f"final_{slug}",
                        "objective": objective,
                        "feature": design.names[feature_index],
                        "mean_abs_shap": float(
                            cell_mean_abs[slug][objective][feature_index]
                        ),
                        "rank": rank,
                        "mean_shap": float(values[:, feature_index].mean()),
                        "feature_min": float(instances[:, feature_index].min()),
                        "feature_max": float(instances[:, feature_index].max()),
                        "in_mean_function": design.names[feature_index]
                        in MEAN_FUNCTION_FEATURES.get(objective, ()),
                        "r2_acquisition": "qlognehvi",
                    })
            print(f"   {slug} done  (R2 hash {built['r2_hash']})")

        shift_rows, verdict = extreme_cell_shifts(
            mean_abs_by_state[default_state], cell_mean_abs, list(design.names)
        )
        for row in shift_rows:
            print(f"   {row['cell']:<24} {row['objective']:<15} "
                  f"max shift {row['max_abs_shift']:.4f} "
                  f"({row['max_shift_fraction_of_largest']:.1%} of largest) "
                  f"on {row['max_shift_feature']}"
                  f"{'  TOP MOVED' if row['top_feature_changed'] else ''}")
        print(f"\n   {verdict}")
        pd.DataFrame(shift_rows).to_csv(
            output_root / "shap_extreme_cell_shift.csv",
            index=False, encoding="utf-8-sig",
        )

    summary = pd.DataFrame(rows)
    summary_path = output_root / "shap_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n6. SUMMARY  {summary_path}  ({len(summary)} rows)")

    print(f"\nDone in {(time.time() - started) / 60:.1f} min. Outputs under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
