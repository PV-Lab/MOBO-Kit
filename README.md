# MOBO-Kit

initially designed by Ethan Schwartz, Daniel Abdoue, Nicky Evans, and Tonio Buonassisi<br>
updated and reconstructed by Ziyang (Colin) Qi and Annie Xu

<h1>
<p align="center">
    <img src="assets/mobo-fom-logo.jpg" alt="Slot-die optimization logo" width="600"/>
</p>
</h1>

<h4 align="center">

[![Requires Python 3.11-3.12](https://img.shields.io/badge/Python-3.11--3.12-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)

</h4>

**MOBO-Kit** accelerates design of experiments with **multi-objective Bayesian
optimization**. It proposes small batches of experimental conditions that trade
off several objectives at once, for problems with more than two inputs and more
than two outputs. Developed across the University of Washington, UC San Diego and
MIT, and demonstrated on slot-die coated perovskite films.

## Three contracts, one live campaign

| | v2 — test data | v3 — test data | v4 — **the real campaign** |
|---|---|---|---|
| config | `campaign_d2d_perovskite.yaml` (archived) | `campaign_d2d_perovskite_test.yaml` (archived) | `campaign_d2d_perovskite_final.yaml` |
| contract | `d2d-objectives-v2-nm-thickness` | `d2d-objectives-v3-test` | `d2d-objectives-v4-final` |
| workbook | `Summary Table.xlsx` | `Summary Table Test.xlsx` | `Final Summary Table.xlsx` |
| sheet | `Sheet1` | `Sheet1` | `R0` |
| purpose | early toolkit testing | rehearsing this contract's shape | **the experiment being run** |

Uniformity and optoelectronic have been renormalised twice, so **none of v2's or
v3's fitted numbers carry over** — they are about quantities that were redefined.
Each earlier contract is kept as a record, with a banner on every document that
describes it. The launcher and every script default to v4.

**In v4 the uniformity and optoelectronic scores are FROZEN**: they are read from
the workbook as stored, with no recomputation in Python, because the group is
still revising how they are defined. Thickness is still computed, because its
definition has been stable and the recomputation is what lets an operator-flagged
reading be excluded and reported. See `docs/CAMPAIGN_STATUS.md` for what freezing
costs and what replaces the missing cross-check.

## The campaign loop

A campaign runs in three rounds. Each proposed condition is run in triplicate so
reproducibility can be measured.

| Round | Method | Conditions | Films |
|---|---|---:|---:|
| R0 | Latin hypercube sampling | 15 | 45 |
| R1 | UCB-HVI + local penalization | 5 | 15 |
| R2 | qLogNEHVI | 3 | 9 |

```python
from mobo_kit.campaign import load_campaign_config, run_r0_lhs, run_r1_ucb, run_r2_qlognehvi
from mobo_kit.workbook_io import read_campaign_workbook

config = load_campaign_config("configs/campaign_d2d_perovskite_final.yaml")

r0 = run_r0_lhs(config, n=15)                        # space-filling, no model

# uniformity and optoelectronic are read from the workbook as stored (frozen);
# thickness is computed from the raw readings and cross-checked
contents = read_campaign_workbook("local_inputs/Final Summary Table.xlsx", config)
X_phys = contents.inputs.to_numpy(float)
Y_model = contents.model_values.to_numpy(float)      # in objective order
assert contents.errors == ()                         # fail closed before fitting

r1 = run_r1_ucb(config, X_phys, Y_model, n=5)        # after R0 is measured
r2 = run_r2_qlognehvi(config, X_phys, Y_model, n=3)  # after R1 is measured
```

Each call returns a `RoundResult` with `conditions` (distinct recipes, physical
units), `replicates` (one row per film, grouped), and `diagnostics` (seed, pool
size, fit warnings, and a validity report).

**New to this repo?** Read `docs/HANDOFF.md` first — reading order, where the
project stands, what is genuinely open, and the questions already settled.
`docs/CAMPAIGN_STATUS.md` is the working guide: what to pass, what comes back, and
the evidence behind each decision.

## Running a round without writing code

Double-click **`launch_mobo_kit.bat`** (Windows) or **`launch_mobo_kit.command`**
(macOS — `chmod +x` it once first). A small window opens:

1. **Browse** to the campaign workbook. It is remembered next time.
2. **Check workbook** — reports which round is due, and anything the read
   noticed: a stored score that no longer matches its measurements, a reading the
   operator flagged, a film whose thickness readings disagree with each other.
3. **Propose R1** (or R2) — fits the model, scores the candidate pool, and writes
   the batch to a **new file beside the workbook**, never into it. That file gets
   two sheets: the worklist to fill in, and a **`Review`** sheet giving each
   proposed condition's predicted objectives with uncertainties, its predicted
   thickness in nanometres, its distance from anything already measured, and which
   settings sit at the edge of their range. The same text appears in the window, so
   it can be forwarded to the group as-is.

4. **Figures.** The same press renders six figures beside the workbook, under
   `<name>_reports/<round>_<timestamp>/`: where the batch sits in recipe space,
   how well the model predicts a film it has not seen, which inputs move each
   objective, what the batch is expected to produce, hypervolume so far, and the
   trade-off itself. Each one writes the CSV behind it. A second button,
   **Figures from current data**, renders the four that need no batch — useful the
   moment measurements are entered.

Then run the films, fill in the highlighted columns of that new sheet, and press
the button again. R2 reads the R1 measurements back and aggregates each condition's
three films into one observation.

The window approves nothing. It shows the proposed conditions in physical units
with the batch's spacing diagnostics; a human decides whether to fabricate.
Everything it does is available as plain functions in `mobo_kit.launcher`
(`inspect_campaign`, `gather_observations`, `generate_next_round`) for anyone who
would rather script it.

## Installation

```bash
conda create -n mobo-kit python=3.12
conda activate mobo-kit
git clone https://github.com/PV-Lab/MOBO-Kit.git
cd MOBO-Kit
python -m pip install -r requirements/dev.txt
```

Tested on CPU with Python 3.12, PyTorch 2.8.0, BoTorch 0.15.1, GPyTorch 1.14.
The exact stack is pinned in `requirements/constraints.txt`.

## Does the optimizer actually work?

`tests/test_dtlz2_acceptance.py` runs the whole loop on **DTLZ2** — a synthetic
3-objective, 10-input problem with a known Pareto front — so the algorithm can be
checked independently of any experimental data.

```bash
pytest tests/test_dtlz2_acceptance.py -m "not slow"
pytest tests/test_dtlz2_acceptance.py -m slow
```

Cumulative hypervolume rises monotonically **by construction**, so that alone
proves nothing — random sampling passes it too. The informative comparison is
against a random baseline at equal budget: mean hypervolume gain **+0.075 (BO)
against +0.056 (random)**, winning on 5 of 8 seeds. BO wins on the mean, not on
every seed, which is the honest expectation for 8 added points in 10 dimensions.

`python scripts/plot_dtlz2_report.py` renders the round-by-round GP fit,
uncertainty, acquisition surface and selected batch.

## How beta and radius were chosen

The live campaign runs **beta = 4.0** and **radius = 0.25**. They were determined
by a sweep over two instruments on the campaign's own data: per-round utility
**box plots** across a grid of **beta from 9 to 49** and **radius from 0.05 to
0.45**, and **heat maps** -- 2-D slices through the higher-dimensional
Gaussian-process model -- at the same cells. `scripts/plot_boxplot_sweep.py` and
`scripts/plot_round_simulation.py` produce them; the outputs stay local, because
they are how the group picks a setting rather than a result about the chemistry.

Two things to know before quoting that choice.

**The sweep could not rank the cells.** The whole spread across betas was 0.0065
against a trial-to-trial standard deviation of 0.010--0.027, and the best cell was
a different (beta, radius) in every trial. So this is a declared policy about how
much to explore, not a measured optimum.

**The campaign ran at beta = 36 from 2026-08 to 2026-09-03**, on the argument that
two of three objectives carried no learnable signal and heavy exploration was
therefore the right posture. That was retired when 45 rows of repeated recipes
showed *why* those two axes are unlearnable -- one is dominated by
between-campaign measurement drift, the other is reproducible but too sparsely
sampled -- neither of which more exploration reaches. At beta = 36 the radius knob
was also provably inert: radii 0.15, 0.25 and 0.35 returned bit-identical batches,
and 18 of 50 proposed coordinates sat on a grid bound. At beta = 4 / radius 0.25
that falls to 11. See `docs/CAMPAIGN_STATUS.md` for the table and its caveats.

`docs/CAMPAIGN_STATUS.md` carries the full record, including the two triggers for
revisiting the choice.

## Repository layout

```
src/mobo_kit/
  campaign.py             the three rounds; start here
  design.py               input grid and bounds
  lhs.py                  Latin hypercube sampling (R0)
  candidate_pool.py       discrete candidate sampling
  sobol_pool.py           nested Sobol pools (alternative sampler)
  models.py               GP construction
  model_validation.py     strict fitting, exact LOOCV, fit guards
  structured_mean.py      physics-informed GP mean functions
  scores.py               measurement columns -> objective values, cross-checked
  objectives.py           objective value -> utility contract
  replicate_variance.py   replicate films -> observation variance (train_Yvar)
  batch_review.py         what a proposed batch says, before anyone fabricates it
  round_report.py         the six figures a round produces, and their data
  loocv.py                the one leave-one-out fold loop, shared by all callers
  attribution.py          exact Shapley values over the campaign's own models
  launcher.py             the one-button loop, and the tkinter window over it
  ucb_hvi.py              UCB hypervolume-improvement scoring (R1)
  qlognehvi_batch.py      qLogNEHVI batch selection (R2)
  batch_selection.py      local penalization, shared by both
  discrete_refinement.py  exact-grid local search
  workbook_io.py          Excel read / candidate-sheet write / read results back
  metrics.py              Pareto front and hypervolume
  plotting.py             diagnostic plots
  candidate_diagnostics.py, acquisition.py, cli.py, main.py,
  data.py, constraints.py, utils.py

  research_qnehvi.py      qNEHVI as a research-only R2 variant, NOT the campaign

configs/   campaign_d2d_perovskite_test.yaml (the live campaign),
           campaign_d2d_perovskite.yaml (archived, first campaign) + two examples
docs/      HANDOFF.md, CAMPAIGN_STATUS.md, GP_MODEL_DECISION.md,
           R1_BATCH_WITHDRAWAL.md, ROUND_SIM_DELTA.md, ROUND_SIM_MANIFEST.md,
           SHAP_SUMMARY.md
scripts/   diagnostics, report figures, intake_new_data.py,
           dtlz2_parameter_sweep.py, plot_round_simulation.py,
           plot_shap_attribution.py, permutation_rank_test.py,
           generate_round_report.py
tests/     601 tests
launch_mobo_kit.bat, launch_mobo_kit.command   double-click entry points
```

## Configuration

Objectives declare **what the model trains on** separately from **how that
becomes a utility**, because the two are not always the same column:

```yaml
objectives:
  contract_version: d2d-objectives-v2-nm-thickness
  scaling_mode: fixed_affine
  specs:
    - name: thickness
      model_source_column: "Thickness (avg)"   # stored cell: cross-check only
      transform: gaussian_target               # utility peaks at the target
      target: 650.0
      sigma: 176.7766952966369
      measurement:                             # what the GP actually trains on
        recipe: mean_of_present                # mean of whichever were measured
        inputs: [{column: T1}, {column: T2}, {column: T3}, {column: T4}]
        excluded: [{column: "T anom"}]         # operator-flagged, never averaged
        cross_check: [{column: "Thickness (avg)", atol: 0.5}]
      mean_function:                           # physics-informed trend
        response: log
        features:
          - {column: speed_1, transform: log}
          - {column: precur_conc, transform: log}
```

The `measurement` block exists because several of the workbook's derived score
cells are pasted literals rather than formulas, so they do not update when the
measurements behind them are edited. `scores.py` recomputes each objective from
the raw columns and demotes the stored cell to a cross-check that warns on
disagreement — see `docs/CAMPAIGN_STATUS.md` issue 2 for the audit.

Objective scales are **fixed for the whole campaign** and must never be
re-derived from observed data — otherwise utility space moves between rounds and
hypervolume stops being comparable across them.
`assert_scaling_is_campaign_fixed` enforces this and runs inside
`build_objective_transform`, so no transform can bypass it.

## Current parameters

| Setting | Value | Config key |
|---|---|---|
| UCB beta (R1) | 36.0 | `rounds.r1.beta` |
| Local penalization radius | 0.35 | `local_penalization.radius` |
| Minimum batch spacing | 0.15 | `local_penalization.min_batch_distance` |
| Candidate pool | 32768 | `rounds.*.candidate_pool_size` |
| Posterior samples (R1) | 256 | `rounds.r1.posterior_samples` |
| MC samples (R2) | 128 | `rounds.r2.mc_samples` |
| GP variant | `dim_scaled_prior` | `model.variant` |
| Seed | 73 | `reproducibility.seed` |

All of these are campaign configuration, not code. Tuning them does not require
touching the algorithm.

The ten input grids hold 11/10/11/13/21/17/18/11/21/17 values, so the full
Cartesian product is 396,945,008,460 recipes. It must never be materialised —
that is what the sampled candidate pool and the discrete local search are for.

**Constraints are config too, and the live campaign declares three.** They are
enforced by filtering the candidate pool before any acquisition scores it, and
re-checked independently when the batch is validated:

```yaml
constraints:
  # a second spin stage either happens or it does not
  - zero_coupled: [speed_2, time_2]
  # the antisolvent has to land while the substrate is still spinning
  - sum_upper_strict: {lhs: anti_time, rhs: [time_1, time_2]}
  # and if it happens, it runs for at least 10 s
  - nonzero_minimum: {column: time_2, minimum: 10}
```

## License

MIT — see [LICENSE](LICENSE).

## Get in touch

Open an issue on GitHub, or contact the development team.
