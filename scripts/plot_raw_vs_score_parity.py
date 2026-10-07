"""Does the model learn better from a RAW measurement than from a combined score?

THE QUESTION, IN THE GROUP'S OWN WORDS. R1 and R2 stopped improving on uniformity
and optoelectronic, and the parity plot showed the model learning neither. Both
are composites of several raw measurements. So: is the COMBINATION the problem?
Would feeding the GP one raw measurement per axis -- just photoconductance, just
Voc, just phase purity -- give it something it can learn?

This draws the answer. Twelve panels, one per candidate objective, every one an
exact leave-one-out parity plot on the same 15 films with the same model:

    row 1   the three COMPOSITE SCORES the campaign runs, plus raw thickness as
            the positive control -- the one axis that does work
    row 2   the OPTOELECTRONIC score taken apart: Voc, photoconductance (raw and
            log), photosensitivity
    row 3   the UNIFORMITY score taken apart: coverage, 1-uniformity, phase
            purity, and raw uniformity

Every point is a film predicted by a model that never saw it. Points on the
dashed line are perfect. A cloud with no slope is a model that has learned
nothing, whatever its R2 says.

HOW TO READ THE NUMBER, which is not how this project read it until 2026-09-04.
``1-(N/(N-1))^2 = -0.1480`` is NOT a significance threshold. It is the score of
one specific predictor -- predict every held-out film with the average of the
other fourteen -- and a fitted GP does not behave like it. Measured on this
campaign, **28.7% of pure-noise shuffles score above -0.1480**. The honest bar is
each candidate's own permutation p95, roughly +0.23 here, and the adjudicator for
a real verdict is the rank permutation test. Panels are therefore marked from the
permutation verdict, not from a comparison against -0.1480.

    python scripts/plot_raw_vs_score_parity.py \
        --workbook "local_inputs/Final Summary Table.xlsx" \
        --outdir local_outputs/raw_vs_score
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mobo_kit.campaign import load_campaign_config  # noqa: E402
from mobo_kit.round_report import _save, _style  # noqa: E402

LEARNS = "#1baf7a"
FAILS = "#8a3b2f"
GREY = "#9a9894"

#: (panel title, expression, plain-English label, verdict)
#: `verdict` is "learns" only where the RANK PERMUTATION TEST said so. Nothing is
#: marked learnable on the strength of clearing -0.1480, which a quarter of pure
#: noise does.
PANELS = [
    # row 1 -- what the campaign runs, plus the control
    ("Uniformity SCORE", "score_uniformity", "the composite you run now", "fails"),
    ("Optoelectronic SCORE", "score_opto", "the composite you run now", "fails"),
    ("Thickness SCORE", "score_thickness", "the composite you run now", "fails"),
    ("Thickness, RAW nm", "thickness_nm", "CONTROL: the axis that works", "learns"),
    # row 2 -- the optoelectronic score, taken apart
    ("Voc (raw)", "voc_raw", "optoelectronic part", "fails"),
    ("Photoconductance", "photocond", "optoelectronic part", "fails"),
    ("log Photoconductance", "np.log(photocond)", "optoelectronic part, log scale", "fails"),
    ("Photosensitivity", "photosens_ratio", "optoelectronic part", "fails"),
    # row 3 -- the uniformity score, taken apart
    ("Coverage", "coverage", "uniformity part", "fails"),
    ("1 - Uniformity", "one_minus_unif", "uniformity part", "fails"),
    ("Phase purity", "phase_purity", "uniformity part, the best of them", "fails"),
    ("Uniformity (raw)", "uniformity_raw", "uniformity part", "fails"),
]


def _load_screen():
    path = Path("scripts") / "raw_component_screen.py"
    spec = importlib.util.spec_from_file_location("_screen_for_parity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workbook", default="local_inputs/Final Summary Table.xlsx")
    parser.add_argument("--config", default="configs/campaign_d2d_perovskite_final.yaml")
    parser.add_argument("--sheet", default="R0")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--outdir", type=Path, default=Path("local_outputs/raw_vs_score"))
    args = parser.parse_args(argv)

    screen = _load_screen()
    config = load_campaign_config(args.config)
    space = screen.read_measurements(Path(args.workbook), args.sheet)
    X = np.column_stack([space[item["name"]] for item in config["inputs"]])
    n = len(X)
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(19.5, 14.4))
    rows: list[dict] = []
    for index, (title, expr, blurb, verdict) in enumerate(PANELS):
        ax = axes[index // 4][index % 4]
        _style(ax)
        y = screen.evaluate(expr, space)
        result = screen.loo_r2(config, X, y, seed=args.seed)
        predicted = np.asarray(result["predicted"], float)
        learns = verdict == "learns"
        colour = LEARNS if learns else FAILS
        ax.scatter(y, predicted, s=42, color=colour, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.6)
        lo = float(min(y.min(), predicted.min()))
        hi = float(max(y.max(), predicted.max()))
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        line = np.array([lo - pad, hi + pad])
        ax.plot(line, line, color="#666666", linewidth=1.0, linestyle="--", zorder=2)
        ax.set_xlim(*line)
        ax.set_ylim(*line)
        for position in range(n):
            ax.annotate(str(position + 1), (y[position], predicted[position]),
                        fontsize=6, color="#555555", xytext=(4, 3),
                        textcoords="offset points")
        ax.set_xlabel("measured")
        ax.set_ylabel("predicted (never saw this film)")
        ax.set_title(
            f"{title}\n{blurb}\nLOO R2 {result['r2']:+.4f}    rank {result['spearman']:+.3f}",
            fontsize=10.5,
            color="#1a6b4c" if learns else FAILS,
        )
        ax.text(
            0.5, 0.965,
            "MODEL LEARNS THIS" if learns else "MODEL LEARNS NOTHING",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5,
            color="#1a6b4c" if learns else FAILS,
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc="#e8f7f1" if learns else "#fdeeea",
                ec="#9fd8c3" if learns else "#e0b4a8",
            ),
        )
        rows.append({
            "panel": title, "expression": expr, "loo_r2": result["r2"],
            "spearman": result["spearman"], "verdict": verdict,
            "collapsed_folds": result["collapsed_folds"],
        })

    fig.suptitle(
        "Would raw measurements work better than the combined scores?  "
        "One panel per candidate objective, all on the same 15 films.",
        fontsize=14, y=0.988,
    )
    caveats = [
        "Every point is a film predicted by a model that never saw it "
        "(exact leave-one-out). Points on the dashed line are perfect; a "
        "shapeless cloud is a model that learned nothing. Numbers are the film's "
        "sample number.",
        "THE ANSWER: taking the scores apart does not help. Every part of the "
        "optoelectronic score fails on its own, and so does every part of the "
        "uniformity score. Only raw thickness -- unchanged from what you already "
        "run -- is learnable. The combination was not the problem on these two "
        "axes; the underlying measurements are.",
        "-0.1480 IS NOT THE BAR, though this project long read it as one. It is "
        "the score of predicting the average of the other 14 films, and 28.7% of "
        "pure-noise shuffles beat it. Panels are marked from the rank permutation "
        "test instead.",
        "Phase purity is the closest thing to an exception (R2 +0.0571, and "
        "+0.3244 once a precursor-concentration trend is declared) but it was "
        "refuted on verification: the whole effect is three films below 1.25 M, "
        "and among the ten high-purity films the model ranks them BACKWARDS "
        "(rank -0.754).",
    ]
    path = args.outdir / "raw_vs_score_parity.png"
    _save(fig, path, caveats)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.outdir / "raw_vs_score_parity.csv", index=False)
    print(frame.to_string(index=False))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
