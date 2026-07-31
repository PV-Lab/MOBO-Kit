# MOBO-Kit

by Ethan Schwartz, Daniel Abdoue, Nicky Evans, and Tonio Buonassisi

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

config = load_campaign_config("configs/campaign_d2d_perovskite.yaml")

r0 = run_r0_lhs(config, n=15)                        # space-filling, no model

# objective values are computed from the raw measurement columns, not read from
# the workbook's stored score cells -- three of those are pasted literals
contents = read_campaign_workbook("local_inputs/Summary Table.xlsx", config)
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

The two tuned parameters were checked the same way. `scripts/dtlz2_parameter_sweep.py`
sweeps `beta` against the local-penalization `radius`, 8 seeds per cell, under a
decision rule written before the numbers existed — and the answer was to keep
`beta = 4.0` and `radius = 0.25`, because the whole grid is flat within one
per-seed standard deviation. That is the outcome that says a default was not a
lucky pick.

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

configs/   campaign_d2d_perovskite.yaml (the live campaign) + two examples
docs/      HANDOFF.md, CAMPAIGN_STATUS.md, GP_MODEL_DECISION.md,
           R1_BATCH_WITHDRAWAL.md, ROUND_SIM_DELTA.md, ROUND_SIM_MANIFEST.md
scripts/   diagnostics, report figures, intake_new_data.py,
           dtlz2_parameter_sweep.py, plot_round_simulation.py
tests/     465 tests
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
| UCB beta (R1) | 4.0 | `rounds.r1.beta` |
| Local penalization radius | 0.25 | `local_penalization.radius` |
| Minimum batch spacing | 0.15 | `local_penalization.min_batch_distance` |
| Candidate pool | 32768 | `rounds.*.candidate_pool_size` |
| Posterior samples (R1) | 256 | `rounds.r1.posterior_samples` |
| MC samples (R2) | 128 | `rounds.r2.mc_samples` |
| GP variant | `dim_scaled_prior` | `model.variant` |
| Seed | 73 | `reproducibility.seed` |

All of these are campaign configuration, not code. Tuning them does not require
touching the algorithm.

The ten input grids hold 11/10/11/11/21/17/18/11/21/9 values, so the full
Cartesian product is 177,816,994,740 recipes. It must never be materialised —
that is what the sampled candidate pool and the discrete local search are for.

## History

The Step 1 / 2A / 2B / 2C audit apparatus was removed from this branch on
2026-07-29, once its findings were recorded in `docs/GP_MODEL_DECISION.md`. It
validated a GP model that has since been replaced. To recover any of it:

```bash
git show pre-cleanup-2026-07-29:src/mobo_kit/<file>.py
```

## License

MIT — see [LICENSE](LICENSE).

## Get in touch

Open an issue on GitHub, or contact the development team.
