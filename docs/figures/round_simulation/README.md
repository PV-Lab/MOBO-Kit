# Curated round-simulation figures

Six figures from one run of `scripts/plot_round_simulation.py`, seed 73, kept as
examples of what the script produces. The full run writes 1,781 figures and 222 MB
to the gitignored `local_outputs/round_simulations/`; these are the ones worth
looking at without re-running it.

**Every value in these figures is a model prediction, not a measurement.** The
oracle is a GP fitted to the 15 real R0 films and then frozen, so a condition that
scores well here has scored well against MOBO-Kit's own beliefs. This validates
the optimiser loop on a data-shaped landscape; it says nothing about the
chemistry. Each figure repeats that in its footer.

| file | what it shows |
|---|---|
| `01_thickness_radius_0p05_tight_batch.png` | radius 0.05. R1 (orange) clusters — two conditions land on `speed_1 = 2500`. Achieved batch spacing 0.455. |
| `02_thickness_radius_0p25_anchor.png` | radius 0.25, the campaign's current setting. The same five conditions have been pushed apart; spacing 0.720. |
| `03_thickness_radius_0p45_saturated.png` | radius 0.45. Spacing 0.921, identical to radius 0.30–0.40 — the knob has saturated and stops doing anything. |
| `04_optoelectronic_anneal_temp_range_edge.png` | why every proposed condition pins `anneal_temp` to its lower bound. The surface is monotone in temperature because the objective carries a monotone linear mean function, so its optimum is at a range edge by construction. This reproduces `CAMPAIGN_STATUS.md` issue 4 from the model side. |
| `05_utility_by_round_anchor.png` | utility by round, three objectives, n = 15 / 5 / 3, with every raw point drawn over its box. |
| `06_hypervolume_by_round_anchor.png` | cumulative hypervolume at the campaign-fixed reference. It rises monotonically **by construction** — adding points can only grow a Pareto front — so this panel shows the size of each step, not that optimisation happened. |

Figures 01 → 02 → 03 are the same slice of the same input pair at three radii, and
are the visual form of the sweep's main finding: on this landscape `radius`
**binds** below about 0.30 and is inert above it. That contradicts the expectation
carried over from the DTLZ2 sweep, where achieved spacings of 0.72–0.98 left the
knob nothing to act on.

Reproduce any of them with, for example:

```bash
python scripts/plot_round_simulation.py --workbook "local_inputs/Summary Table.xlsx" --conditions radius_0p05__beta_4 --pairs speed_1,precur_conc
```
