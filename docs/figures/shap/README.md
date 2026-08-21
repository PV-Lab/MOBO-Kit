# SHAP attribution figures

> **This document describes the FIRST campaign, which was algorithm testing.**
> Its data and workbook (`Summary Table.xlsx`, contract
> `d2d-objectives-v2-nm-thickness`) existed to validate the toolkit, not to run an
> experiment. **The real campaign is the second one** --
> `configs/campaign_d2d_perovskite_test.yaml` on `Summary Table Test.xlsx`,
> contract `d2d-objectives-v3-test`. Two of the three objectives are computed
> differently there, so **no number below transfers**; they describe quantities
> that were redefined. Start from `docs/CAMPAIGN_STATUS.md` for the live campaign.

Six figures from one run of `scripts/plot_shap_attribution.py`, seed 73, 1,000
on-grid instances. Schema and reading notes: `docs/SHAP_SUMMARY.md`.

Each answers: **which process inputs move this objective's expected utility, and
in which direction?** They explain the *model*, which is the only thing SHAP can
explain.

| file | what it shows |
|---|---|
| `01_thickness_r0_only.png` | The clearest real result. `precur_conc` and `speed_1` lead by 3.7× over the third feature, with the sign pattern the fitted physics predicts: high concentration and low spin speed both push the film thicker, away from the 650 nm target, so both reduce utility. |
| `02_optoelectronic_r0_only.png` | `anneal_temp` leads, monotone and negative — the declared linear trend (marginal ρ = −0.651, p = 0.009). This is why every proposed condition pins the temperature to its lower bound. |
| `03_uniformity_r0_only_is_fitted_noise.png` | **The cautionary figure.** It looks like a textbook result — a clean monotone gradient on `time_1`, an orderly ranking, magnitudes of ±0.15. Uniformity does not beat the leave-one-out null (LOO R² −0.681, permutation p = 0.82). Every bit of that structure is fitted noise. |
| `04`–`06` | The same three objectives after a simulated R0 → R1 → R2 pass, refitted on 23 conditions. Rankings are unchanged; magnitudes grow slightly. Only 15 of those 23 conditions were ever measured. |

## Three things these figures are not

**Not evidence of a physical effect.** Where a feature appears in that objective's
declared `mean_function` — `speed_1` and `precur_conc` for thickness,
`anneal_temp` for optoelectronic — the model was *told* that relationship by the
config. SHAP recovering it is a consistency check, not a discovery. The
`in_mean_function` column in `shap_summary.csv` marks exactly which rows those are.

**Not a cross-objective comparison.** Thickness attributions are larger than
uniformity's, but the two utilities are different constructions (a Gaussian on a
650 nm target against an affine product). Compare within an objective.

**Not sensitive to how the batch was chosen.** Measured across three extreme
acquisition cells (radius 0.05, radius 0.45, beta 25), no objective changed its
top feature and the largest attribution shift was **0.0996 of that objective's
largest** — and that near-miss was on uniformity, whose attributions are noise
anyway. The two objectives carrying real structure moved by at most 6.3%. So these
are properties of the fitted model, not of the search.

## qLogNEHVI and qNEHVI give the same model

The brief expected two distinct final states, one per R2 acquisition. They propose
the **identical R2 batch** — verified across four cells (default, radius 0.05,
radius 0.45, beta 25) — so figures `04`–`06` cover both, and each says so in its
footer. BoTorch itself warns against qNEHVI in favour of qLogNEHVI.

Reproduce with:

```bash
python scripts/plot_shap_attribution.py --workbook "local_inputs/Summary Table.xlsx" --instances 1000 --extreme-cells
```
