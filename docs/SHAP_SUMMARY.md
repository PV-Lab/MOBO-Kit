# `shap_summary.csv` schema

Written by `scripts/plot_shap_attribution.py` to
`local_outputs/shap/shap_summary.csv`. One row per **feature × objective × model
state**.

The beeswarms show shape; this file is what you sort, diff and quote. It is also
what makes the extreme-cell comparison possible, since ranking two pictures by eye
is not a measurement.

## What is being explained

`E[utility]` for one objective, through
`ObjectiveTransform.expected_transform` — so thickness goes through the lognormal
quadrature rather than a transformed posterior mean, and every SHAP value is in
**utility units where higher is better**, comparable across features within an
objective.

**Not comparable across objectives.** Uniformity and thickness utilities are
different constructions (an affine product against a Gaussian on a 650 nm target),
so a larger mean |SHAP| on one does not mean that objective is more sensitive.
Compare rows within an objective, or compare the same objective across model
states.

## Columns

| column | type | meaning |
|---|---|---|
| `model_state` | str | `r0_only` (fitted to the 15 real measurements), `final` (23 points after a simulated R0→R1→R2 at radius 0.25, beta 4), or `final_radius_*__beta_*` for an extreme cell. If the two R2 acquisitions ever diverge, `final_qlognehvi` and `final_qnehvi` appear instead of `final`. |
| `objective` | str | `uniformity`, `optoelectronic` or `thickness`. |
| `feature` | str | One of the ten campaign inputs. |
| `mean_abs_shap` | float | Mean absolute SHAP value over the attributed instances — the magnitude the beeswarm ranks by. |
| `rank` | int | 1 = largest `mean_abs_shap` within that objective and model state. |
| `mean_shap` | float | Signed mean. Near zero with a large `mean_abs_shap` means the feature matters in both directions — a non-monotone effect, not a weak one. |
| `feature_min` / `feature_max` | float | The physical range spanned by the attributed instances, in that input's own units. Present because a beeswarm's colour is normalised **per feature row**, so one colorbar cannot carry physical units for ten inputs at once. |
| `in_mean_function` | bool | True if this feature appears in that objective's declared `mean_function`. **A True row is partly a restatement of the model's declared physics, not a discovery.** |
| `r2_acquisition` | str | `identical` when qLogNEHVI and qNEHVI proposed the same R2 batch (so the row covers both), otherwise the acquisition that produced the state. |

## `shap_extreme_cell_shift.csv`

Written alongside when `--extreme-cells` is passed. One row per extreme cell ×
objective, answering **whether the attributions describe the model or the search**.

| column | meaning |
|---|---|
| `cell` | the extreme cell, e.g. `radius_0p05__beta_4` |
| `max_abs_shift` | largest change in `mean_abs_shap` for any feature, against the default cell |
| `max_shift_feature` | which feature moved most |
| `max_shift_fraction_of_largest` | that shift as a fraction of the objective's largest default-cell attribution |
| `top_feature_changed` | whether the rank-1 feature differs from the default cell |
| `default_top_feature` / `cell_top_feature` | the two rank-1 features |

**The verdict rule was fixed before the cells were run**: sweeping the acquisition
parameters is warranted only if an extreme cell moves the top feature, or moves any
attribution by more than **10%** of that objective's largest. Otherwise the
attributions are a property of the fitted model rather than of how the batch was
selected, and there is nothing to sweep.

## Three things to read carefully

**A large attribution is not evidence of a physical effect.** SHAP explains the
model. Where `in_mean_function` is True, the model was *told* that relationship by
`configs/campaign_d2d_perovskite.yaml`; SHAP recovering it is a consistency check,
not a discovery.

**Uniformity attributions are not signal.** Uniformity does not beat the
leave-one-out null (LOO R² −0.681, permutation p = 0.82). Its GP still fits ARD
lengthscales and has a posterior mean that varies, so SHAP reports structure with
real magnitude. That structure is fitted noise. It is included rather than
suppressed because a reader who sees only the beeswarm would otherwise conclude
the opposite — and every uniformity figure says so in its footer.

**`final` rows describe a simulated campaign.** Only the 15 R0 conditions were
measured; the other 8 carry oracle predictions. `r0_only` is the state fitted
entirely to real data and is the anchor for anything quoted outside this analysis.
