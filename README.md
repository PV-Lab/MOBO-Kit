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

config = load_campaign_config("configs/campaign_d2d_perovskite.yaml")

r0 = run_r0_lhs(config, n=15)                       # space-filling, no model
r1 = run_r1_ucb(config, X_phys, Y_model, n=5)       # after R0 is measured
r2 = run_r2_qlognehvi(config, X_phys, Y_model, n=3) # after R1 is measured
```

Each call returns a `RoundResult` with `conditions` (distinct recipes, physical
units), `replicates` (one row per film, grouped), and `diagnostics` (seed, pool
size, and a validity report).

`docs/CAMPAIGN_STATUS.md` is the working guide: what to pass, what comes back,
how to plot it, and the current open issues.

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
  objectives.py           raw measurement -> utility contract
  ucb_hvi.py              UCB hypervolume-improvement scoring (R1)
  qlognehvi_batch.py      qLogNEHVI batch selection (R2)
  batch_selection.py      local penalization, shared by both
  discrete_refinement.py  exact-grid local search
  workbook_io.py          Excel read / candidate-sheet write
  metrics.py              Pareto front and hypervolume
  plotting.py             diagnostic plots
  candidate_diagnostics.py, acquisition.py, cli.py, main.py,
  data.py, constraints.py, utils.py

configs/   campaign_d2d_perovskite.yaml (the live campaign) + two examples
docs/      CAMPAIGN_STATUS.md, GP_MODEL_DECISION.md, D2D_CAMPAIGN_SPEC.md
scripts/   diagnostics and report figures
tests/     280 tests
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
      model_source_column: "Thickness (avg)"   # the GP trains on nanometres
      transform: gaussian_target               # utility peaks at the target
      target: 650.0
      sigma: 176.7766952966369
      mean_function:                           # physics-informed trend
        response: log
        features:
          - {column: speed_1, transform: log}
          - {column: precur_conc, transform: log}
```

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
