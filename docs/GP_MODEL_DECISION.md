# GP model decision record

Measured 2026-07-28 on the corrected `Summary Table.xlsx` (15 R0 observations,
10 inputs). Reproduce with `python scripts/gp_diagnostic.py`.

## What was wrong

`ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=10))` was built with no lengthscale
prior. GPyTorch's default constraint is `Positive()` — lower bound 0, upper bound
infinity — so nothing bounded the fit. At N=15 in 10 dimensions the marginal
likelihood interpolates every observation by making a few directions extremely
wiggly and switching the rest off:

| Objective | Fitted ARD lengthscales | Directions ≥ 10 | Noise |
|---|---|---|---|
| Uniformity | 0.22 … 2229 | 6/10 | pinned at 1e-3 floor |
| Optoelectronic | 3.0 … 38105 | 9/10 | pinned at 1e-3 floor |
| Thickness | 0.13 … 2150 | 7/10 | pinned at 1e-3 floor |

This is overfitting, not the prior-mean collapse originally hypothesised: the
posterior mean varied over 104–131% of each observed range across the design
space. Noise at its floor means the model believed the data were noiseless.

## The fix

BoTorch 0.15.1's `SingleTaskGP` already applies a dimension-scaled LogNormal
lengthscale prior — `LogNormal(loc = √2 + ln(d)/2, scale = √3)` — whenever
`covar_module` is not supplied. MOBO-Kit was discarding it by passing an explicit
module. `dim_scaled_prior` restores the prior while keeping the `ScaleKernel`
wrapper that the hyperparameter readout and plots depend on.

| Variant | Median lengthscale | Flat directions | Uniformity LOO R² |
|---|---|---|---|
| `legacy_matern_no_prior` | 1121 | 6/10 | −1.557 |
| `conservative` | 2735 | 6/10 | −1.453 |
| **`dim_scaled_prior`** | **0.88** | **0/10** | **−0.537** |

## Naming

`default_current` was retired to `legacy_matern_no_prior` rather than redefined,
so archived Step 2B/2C artifacts stay interpretable. `dim_scaled_prior` is
`PRIMARY_VARIANT`; the retired contract must be requested by name and is kept
only for reproducing old runs.

## Two resolution floors. Check both before comparing anything.

**Null: −0.148.** Predicting the leave-one-out mean of the other N−1 gives
LOO R² = 1 − (N/(N−1))² and Spearman exactly −1, independent of the data. A model
below this learned nothing.

**Resolution: ±0.236.** Parametric bootstrap at N=15, 4000 resamples from the
same underlying truth, gives an LOO R² standard deviation of **0.236** and a
central 95% range of **[−0.231, +0.666]**. The identical relationship produces
anything in that range purely by resampling.

So **two LOO R² values less than about half a point apart are not a comparison at
this N.** This has been walked into twice in this project — once arguing −0.017
against −0.145, once arguing +0.355 against +0.244 — both times because the
number moved in the pleasing direction. The floor is written down here so the
next person can check before reaching for a difference.

Differences that survive the floor: the plain-vs-structured swings below
(0.20 and 0.70). Differences that do not: anything in the second decimal place.

## The bar for "the model learned something"

For N observations, predicting the leave-one-out mean of the other N−1 gives

    LOO R² = 1 − (N/(N−1))²      (−0.1480 at N=15)
    Spearman = −1 exactly

both independent of the data, because that prediction is a strictly decreasing
function of the held-out value. So a **negative LOOCV Spearman is the signature
of a model that learned nothing**, and the bar to clear is −0.148, not 0.

## What the data supports

- **Uniformity** — no learnable signal. Nothing beat the null across ~240 model
  configurations, 7 target transforms, or modelling Coverage / (1−Uniformity) /
  Phase purity separately. Permutation p = 0.82. Treat as exploration-only.
- **Optoelectronic** — weak but real, via `anneal_temp` (single-input LOO
  R² +0.244). `=LOG10(P*Q)` is monotone, so model the score directly.
- **Thickness** — see below.

## Thickness: model nanometres, not the score

`Normalized thickness = EXP(-(((T-650)/250)^2))` is a peaked Gaussian on a 650 nm
target. The map T → score is **2-to-1**: films at 400 nm and 900 nm receive
near-identical scores from opposite sides of the peak, and the observed films
straddle the target (4 below, 11 above, 360–1303 nm). A GP trained on the score
must represent a folded bimodal ridge in process space; a GP trained on
nanometres sees a smooth trend.

Raw thickness is the most predictable quantity in the campaign:
`log T ~ log(speed_1) + log(precur_conc)` gives LOO R² **+0.449**, Spearman
**+0.714**, permutation p **0.0067**, with fitted speed exponent −0.38 against
spin-coating theory's −0.5.

Predicting the score, exact leave-one-out (`dim_scaled_prior`):

| Approach | LOO R² | Spearman |
|---|---|---|
| train on the score directly | −0.444 | −0.764 |
| train on nm → E[score], analytic | **−0.147** | **+0.279** |
| train on nm → score(mean) only | −0.492 | +0.161 |
| null | −0.148 | −1.000 |

Rank correlation flips from actively misleading to usable, which is what drives
candidate selection. R² only reaching the null is the honest outcome: the raw-nm
posterior is wide (median 157 nm against a Gaussian width of 250/√2 ≈ 177 nm), so
expected scores are correctly pulled toward the middle.

Note the third row. Transforming only the posterior *mean* is worse than the null
— it is biased by Jensen's inequality and blind to variance. Use
`ObjectiveTransform.expected_transform`, which has a closed form for
`Y ~ N(μ, v)`:

    E[exp(-½((Y-c)/s)²)] = √(s²/(s²+v)) · exp(-½(μ-c)²/(s²+v))

verified against Monte Carlo to <1e-3. It reduces to the plain transform at v = 0
and penalises uncertainty at the target: at μ = 650 exactly, expected score is
0.994 / 0.870 / 0.508 for posterior σ of 20 / 100 / 300 nm.

The workbook's `exp(-((T-650)/250)²)` has no ½, so in this parameterisation
`sigma = 250/√2 ≈ 176.78`.

## Both priors are required, not just the lengthscale one

With the lengthscale prior but a bare noise floor, the fit has a *second*
degenerate mode: the outputscale collapses to ~0 and the model declares the data
pure noise. Measured on the thickness score, 10 of 15 leave-one-out folds landed
there — fitted noise 0.93 against a latent predictive sd of 1e-4, giving
z-scores in the thousands. Adding BoTorch's `LogNormal(-4, 1)` noise prior
removes it entirely.

The collapse is specific to the thickness *score*, the folded 2-to-1 objective —
uniformity, optoelectronic and raw nm are stable either way, and the noise prior
costs them nothing (latent sd 0.1226 vs 0.1236). It matters because acquisition
consumes the *latent* posterior: a predictive interval can look well calibrated
while the latent variance has collapsed, because the large fitted noise hides it.

Calibration, exact leave-one-out, predictive (noise-inclusive) intervals against
nominal 0.68 / 0.95:

| Variant | Uniformity | Optoelectronic | Thickness | Mean NLPD |
|---|---|---|---|---|
| `legacy_matern_no_prior` | 0.067 / 0.533 | 0.467 / 0.667 | 0.467 / 0.600 | 1.76 / 2.82 / 1.35 |
| `dim_scaled_prior` | 0.400 / 0.800 | 0.533 / 0.800 | 0.533 / 0.867 | 0.58 / 1.44 / 0.82 |

Interval coverage must use the predictive sd, not the latent sd. Using the latent
sd understates every interval and makes a calibrated model look overconfident.

## Structured means: two objectives, opposite shapes

Physics fixes the features in advance, so this is not selection on the outcome.
Linear coefficients are refit inside every fold.

| Objective | Mean function | LOO R² | Spearman |
|---|---|---:|---:|
| thickness nm | none | +0.183 | +0.586 |
| thickness nm | `log T ~ log(speed_1) + log(precur_conc)` | **+0.384** | +0.682 |
| optoelectronic | none | **−0.342** | −0.100 |
| optoelectronic | linear `anneal_temp` | **+0.355** | +0.618 |

Null −0.148. Both swings (0.20 and 0.70) clear the ±0.236 resolution floor.

**The optoelectronic result is the larger finding.** Its plain GP sat *below the
null* — actively worse than predicting the mean — so the seven irrelevant inputs
were not merely diluting the fit, they were doing damage. Removing the
temperature trend first fixes it.

The two shapes are **opposite** and neither generalises. Thickness needs a pair
of log terms and neither alone is worth much (+0.159, +0.187). Optoelectronic
needs exactly one linear term: every addition tested made it worse, and an
Arrhenius `1/T` form bought nothing over plain linear temperature (+0.226 vs
+0.244). `anneal_temp` is also the least search-contaminated choice available —
it came from a marginal correlation already on record (ρ = −0.651, p = 0.009),
not from the six-form search that was run afterwards.

**Not claimed:** that linear-mean-plus-GP beats linear-mean-alone. The observed
gap (+0.355 vs +0.244) is 0.47 sd of the ±0.236 resolution floor — indistinguishable
from sampling noise. It is an open hypothesis with a specific test attached: does
linear-mean-plus-GP beat linear-mean-alone under the permutation null? That
question rides along with the optoelectronic permutation run.

## Structured mean for thickness

Physics fixes the two predictors in advance, so this is not selection on the
outcome. Refitting the linear coefficients inside every fold:

| Raw-nm model | LOO R² | Spearman |
|---|---|---|
| plain GP, 10 inputs | +0.183 | +0.586 |
| **GP + linear mean on log(speed_1), log(precur_conc)** | **+0.384** | **+0.682** |
| 2-input log-log reference | +0.449 | +0.714 |

Carried through to the score: R² −0.145 → **−0.019**, Spearman +0.243 → **+0.461**.
The structured mean legitimately reaches what the legacy model reached by
accident of overconfidence.

**Not yet wired into `campaign.py`.** R1 candidates generated before this lands
use the plain GP.

The linear coefficients are refit **inside every fold**, on the 14 training rows
only (`scripts/thickness_permutation_and_mean.py`, in the fold loop). The held-out
value never touches them. The two predictors are fixed from physics before any
fitting, so this is not selection on the outcome.

## Significance of the rank improvement

Permutation test, 200 shuffles, permuting the nm measurements and redoing the
full leave-one-out fit plus transform. When the structured mean is used, its
linear coefficients are refit inside every null fold too, so the null is not
flattered.

| Pipeline | Shuffles | Observed ρ | Null mean | Null sd | p | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| plain GP | 200 | +0.243 | −0.241 | 0.349 | 0.12 | — |
| structured mean | 200 | +0.461 | −0.167 | 0.295 | 0.020 | [0.006, 0.050] |
| **structured mean** | **1800** | **+0.461** | −0.131 | 0.305 | **0.0350** | **[0.0270, 0.0446]** |

The 1800-shuffle run is reported **standalone**, not pooled with the earlier 200.
Pooling would be defensible but carries an optional-stopping flavour, since the
larger run was commissioned because the first result was borderline. Reporting
the fresh run alone sidesteps the question at no cost.

**The result holds and the interval clears.** 63 exceedances in 1800, upper bound
0.0446, below 0.05. Note the point estimate moved 0.020 → 0.035: the 200-shuffle
figure was optimistic, which is exactly why the re-run was worth doing. Its
interval did contain the final value.

The plain GP does not clear p < 0.05. **The structured mean does.** That earns
"thickness is genuinely predictive" rather than "directionally right". On R² the
structured mean reaches p = 0.144 (95% CI [0.129, 0.162]), not significant — but
rank drives candidate selection, and rank is significant.

The null mean is −0.17, not 0: the leave-one-out shrinkage artifact drags it
negative, which is why a positive observed value carries information.

## Sample 1 stays in. Do not exclude it. (decided, closed)

**This section exists because "excluding the control nearly doubles R²" is a true
sentence that will get rediscovered and acted on. It is the wrong action.**

Dropping sample 1 does improve the thickness fit — raw-nm LOO R² goes +0.384 →
+0.632, Spearman +0.682 → +0.824, and the score prediction reaches +0.207 against
a −0.160 null. Sample 1 is also the off-grid literature control, so there is a
ready-made provenance story for excluding it.

That story is wrong. Ranking every point by how much dropping it improves the fit:

| Dropped sample | Leverage | LOO R² after drop | Δ |
|---|---:|---:|---:|
| **12** | **0.462** | **+0.879** | **+0.429** |
| 1 | 0.297 | +0.676 | +0.226 |
| 13 | 0.199 | +0.465 | +0.015 |
| … | | | |
| 7 | 0.316 | +0.261 | −0.188 |

Sample 1 is not the drag. **Sample 12 is, by nearly double**, and it has the
highest leverage in the design.

The mechanism is visible in the inputs. Samples 1 and 12 are the *only* two
points at `speed_1 = 1000`, the minimum, so between them they anchor the entire
low-speed end of the strongest predictor — and they contradict each other:
sample 1 has the higher concentration (1.4 vs 1.1) but the *thinner* film
(687 vs 1155 nm), inverting the expected relationship.

So the gain from dropping sample 1 is a **high-leverage-endpoint artifact**, not
a signal about its provenance. Acting on it would commit you to also dropping
sample 12 — an ordinary LHS point with no provenance justification at all. The
declared provenance difference is real; it is simply not what the residual was
reporting.

A tempting explanation for the inversion — sample 1 runs `speed_2 = 5000` against
sample 12's 500, so a fast second stage could be thinning the film — **does not
survive testing**: adding `log(speed_2 + 1)` to the structured mean drops LOO R²
from +0.449 to −0.827. Do not add it.

The low-speed corner therefore remains genuinely unexplained, with two
contradictory observations in it. That makes it something **R1 should probe**,
not something to model around. When the real R1 batch is generated: if the
acquisition function proposes nothing near `speed_1 = 1000`, notice it. It may
mean the model has concluded the region is bad when what it actually has is two
points that disagree.

## Open

- Wire the structured mean into the campaign path before generating R1
  candidates for fabrication.
- Hypervolume reference: fixed. `configs/FA0.9CS0.1PbI3_260407_Config.yaml` now
  declares `reference_point_utility` in utility space after the transforms, so
  no axis dominates. The old raw-scale `[-0.01, -10.0, -0.01]` gave the
  optoelectronic axis 4.01x the uniformity axis.
