"""One command to run when the experimental group returns new or corrected data.

    python scripts/intake_new_data.py --workbook "local_inputs/Final Summary Table.xlsx"

The group has always described the current numbers as test data, so a replacement
was expected from the start.  When it arrives, the question is not "does the code
still run" -- the tests answer that -- but "does the model this campaign committed
to still earn its place on THIS data".  Several of those commitments were justified
by measurements on 15 specific rows, and a new dataset does not inherit them.

So this checks, per objective:

* whether the objectives can be computed at all, and what the read notices;
* whether each declared ``mean_function`` still beats the leave-one-out null by
  more than the resolution floor -- and if it does not, names the exact config
  block to delete;
* whether the fit guard has anything to say, including the case where the mean
  function explains so much that the residual GP collapses;
* whether the fixed objective anchors still span the data;
* whether the campaign-fixed scaling guard still passes.

**Both floors are recomputed at the new N rather than reused.**  The null is
``1 - (N/(N-1))^2``, which moves with N: -0.148 at 15, -0.105 at 21, -0.069 at 31.
The resolution floor of +-0.236 was a parametric bootstrap at N=15 and shrinks
roughly as ``1/sqrt(N)``; the estimate printed here is scaled that way and is
labelled as an estimate, because the honest version is to re-run the bootstrap.

Nothing here decides anything. It prints what the data supports so a human can.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mobo_kit.campaign import (
    assert_scaling_is_campaign_fixed,
    build_design_from_config,
    build_objective_transform,
    load_campaign_config,
    objective_names,
)
from mobo_kit.constraints import constraint_violations, constraints_from_config
from mobo_kit.loocv import (
    RESOLUTION_SD_AT_15,
    loo_predictions,
    null_loo_r2,
    resolution_sd,
)
from mobo_kit.model_validation import ModelFitError
from mobo_kit.scores import ScoreSeverity
from mobo_kit.structured_mean import mean_spec_from_config
from mobo_kit.workbook_io import read_campaign_workbook

# The fold loop lives in `mobo_kit.loocv`, shared with the round report and the
# permutation test. It used to live here, and the moment a second caller needed it
# there were two copies of a number this document calls canonical.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", required=True)
    # Defaults to the ACTIVE campaign. The first campaign's config is archived, and
    # defaulting to it would quietly audit new rows against a retired contract --
    # different recipes, different anchors, different grids.
    parser.add_argument("--config", default="configs/campaign_d2d_perovskite_final.yaml")
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="audit and anchors only; skip the leave-one-out refits",
    )
    args = parser.parse_args()

    config = load_campaign_config(args.config)
    names = list(objective_names(config))
    print("=" * 78)
    print(f"INTAKE: {Path(args.workbook).name}")
    print(f"CONFIG: {Path(args.config).name}  "
          f"({config.get('objectives', {}).get('contract_version')})")
    print("=" * 78)
    if str(config.get("campaign", {}).get("status")) == "archived":
        print("\n   NOTE: this config is archived. Its recipes, anchors and grids")
        print("   describe a retired contract, so every number below is about that")
        print("   contract rather than about the active campaign.")

    # ---------------------------------------------------------------- audit --
    contents = read_campaign_workbook(args.workbook, config)
    n = contents.n_rows
    print(f"\n1. READ  {n} rows, objectives {tuple(names)}")
    errors = contents.errors
    warnings_found = contents.warnings
    notes = [f for f in contents.findings if f.severity is ScoreSeverity.NOTE]
    print(f"   errors {len(errors)}   warnings {len(warnings_found)}   notes {len(notes)}")
    for finding in errors:
        print(f"   ERROR   {finding}")
    for finding in warnings_found:
        print(f"   warning {finding}")
    if errors:
        print("\n   Objectives cannot be computed for every row. Stopping: every")
        print("   number below would be about a subset nobody chose.")
        return 1

    # ------------------------------------------------------------- contract --
    print("\n2. CONTRACT")
    try:
        assert_scaling_is_campaign_fixed(config)
        print("   scaling guard          PASS (scales are campaign-fixed)")
    except Exception as exc:
        print(f"   scaling guard          FAIL: {exc}")
        return 1

    transform = build_objective_transform(config)
    for index, spec in enumerate(transform.specs):
        column = contents.model_values[names[index]]
        low, high = float(column.min()), float(column.max())
        if spec.transform == "affine":
            inside = spec.lower_anchor <= low and high <= spec.upper_anchor
            verdict = "PASS" if inside else "OUT OF RANGE"
            print(
                f"   {spec.name:<16} anchors [{spec.lower_anchor:g}, "
                f"{spec.upper_anchor:g}] vs data [{low:.4g}, {high:.4g}]  {verdict}"
            )
            if not inside:
                print(
                    "        -> widen the anchors DELIBERATELY and bump "
                    "objectives.contract_version; do not let them track the data."
                )
        else:
            print(f"   {spec.name:<16} target {spec.target:g} vs data [{low:.4g}, {high:.4g}]")

    # ------------------------------------------------------ design and rules --
    print("\n3. DESIGN AND CONSTRAINTS")
    design = build_design_from_config(dict(config))
    X_observed = contents.inputs.to_numpy(float)
    off_grid = [
        (contents.sample_ids[row], name, float(value))
        for column, name in enumerate(design.names)
        for row, value in enumerate(X_observed[:, column])
        if not np.any(
            np.isclose(design.var_array[column], value, rtol=0.0, atol=1e-9)
        )
    ]
    if off_grid:
        # An off-grid observation stays in the GP and in the distance references,
        # but it cannot take part in grid-index bookkeeping. Worth knowing which,
        # because the usual cause is a grid that no longer describes the process.
        print(f"   on-grid check          {len(off_grid)} observed value(s) OFF GRID")
        for sample, name, value in off_grid:
            print(f"        sample {sample}: {name} = {value:g}")
    else:
        print(f"   on-grid check          PASS, all {n} rows land on the declared grid")

    constraints = constraints_from_config(dict(config), design)
    if not constraints:
        print("   constraints            none declared")
    else:
        for item in constraints:
            print(f"   constraint             {item.name}: {item.description}")
        violations = constraint_violations(X_observed, design, constraints)
        broken = [
            (contents.sample_ids[row], names_broken)
            for row, names_broken in enumerate(violations)
            if names_broken
        ]
        if broken:
            # History is history: a row measured before a rule existed is not an
            # error and must not block anything. It is worth saying, though -- a
            # constraint that rejects a film the group actually ran is much more
            # likely to be wrong than the film is.
            print(f"   observed rows          {len(broken)} of {n} break a constraint")
            for sample, names_broken in broken:
                print(f"        sample {sample}: {names_broken}")
            print("        -> not an error. Check the RULE before the films.")
        else:
            print(f"   observed rows          PASS, all {n} satisfy every constraint")

    # ---------------------------------------------------------------- floors --
    null = null_loo_r2(n)
    floor = resolution_sd(n)
    print(f"\n4. FLOORS AT N={n}")
    print(f"   null LOO R2            {null:+.4f}   (was {null_loo_r2(15):+.4f} at N=15)")
    print(f"   resolution sd          +-{floor:.4f}  (estimated by sqrt(15/N) from "
          f"{RESOLUTION_SD_AT_15}; re-run the bootstrap if a call is close)")

    if args.skip_model:
        print("\n5. MODEL  skipped (--skip-model)")
        return 0

    # ----------------------------------------------------------------- model --
    # The rule has two parts, and printing only the first one is what made the
    # thickness verdict read as a dead end rather than as a question for a
    # different instrument.
    print("\n5. PER-OBJECTIVE VERDICT")
    print(f"   (i)  the structured fit must beat the null, {null:+.4f}")
    print(f"   (ii) if structured-vs-plain is inside the floor ({floor:.3f}), R2 cannot")
    print("        decide and the RANK PERMUTATION adjudicates")
    X_phys = contents.inputs.to_numpy(float)

    entries = config["objectives"]["specs"]
    for index, (name, entry) in enumerate(zip(names, entries)):
        y = contents.model_values[name].to_numpy(float)
        mean_spec = mean_spec_from_config(entry)
        print(f"\n   {name}")
        try:
            plain_loo = loo_predictions(
                config, entry, X_phys, y, use_mean_function=False
            )
            plain, plain_warnings = plain_loo.r2, plain_loo.collapse_warnings
            print(f"     plain GP             LOO R2 {plain:+.4f}")
        except ModelFitError as exc:
            print(f"     plain GP             REFUSED: {exc.cause}")
            plain, plain_warnings = float("nan"), []

        if mean_spec is None:
            print("     no mean function declared")
            verdict = "beats the null" if plain > null else "does NOT beat the null"
            print(f"     verdict              {verdict} ({plain:+.4f} vs {null:+.4f})")
            continue

        try:
            structured_loo = loo_predictions(config, entry, X_phys, y)
            structured = structured_loo.r2
            structured_warnings = structured_loo.collapse_warnings
            print(f"     with mean function   LOO R2 {structured:+.4f}")
        except ModelFitError as exc:
            print(f"     with mean function   REFUSED: {exc.cause}")
            print("     verdict              DELETE the mean_function block: the fit")
            print(f"                          is refused outright for {name}.")
            continue

        for message in dict.fromkeys(plain_warnings + structured_warnings):
            print(f"     GUARD                {message}")
        if not structured_warnings:
            print("     guard status         clean")

        swing = structured - plain
        clears_floor = swing > floor
        beats_null = structured > null
        print(f"     swing                {swing:+.4f}  (floor {floor:.4f})")
        if clears_floor and beats_null:
            print("     verdict              KEEP the mean function: it clears the")
            print("                          resolution floor and beats the null.")
        elif beats_null:
            print("     verdict              INCONCLUSIVE ON R2: beats the null, but the")
            print("                          swing is inside the floor, so R2 cannot")
            print("                          resolve plain against structured at this N.")
            print("                          That is not a verdict. Adjudicate on RANK,")
            print("                          which is what the acquisition consumes:")
            print("                            python scripts/permutation_rank_test.py \\")
            print(f"                              --objective {name} --permutations 1800")
        else:
            features = ", ".join(f.column for f in mean_spec.features)
            print("     verdict              DELETE the mean function. It does not beat")
            print(f"                          the null at N={n}. Remove this block from")
            print(f"                          {Path(args.config).name}:")
            print(f"                            objectives.specs[{index}].mean_function")
            print(f"                            (response: {mean_spec.response}, "
                  f"features: {features})")

    print("\n" + "=" * 78)
    print("Nothing above is a decision. It is what the new data supports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
