# D2D Step 2A Computational Core

## Scope and safety boundary

Step 2A supplies a generic, synthetic-tested engine for discrete R1 UCB-HVI
and R2 qLogNEHVI batch construction. It does **not** authorize a D2D campaign
proposal. The workbook objective mapping, utility formulas, fixed scales,
reference point, QC/control policy, physical constraints, and acquisition
settings remain subject to approval.

The production gate is separate from the mathematical functions. Low-level
functions may run with visibly test-only synthetic settings; a workbook- or
campaign-facing run must pass `validate_production_config` first. During Step
2A, `run_mobo_experiment(..., propose_candidates=True)` is blocked before CSV
parsing, model fitting, or output creation. Even an approved configuration is
rejected there because the old runner does not use this objective contract,
discrete pool, or shared selector; wiring those pieces is a Step 2B task.

## Coordinate and outcome spaces

| Space | Representation | Used for |
|---|---|---|
| Physical input | Configured laboratory units on exact grids | worklists and physical constraints |
| Normalized input | Each input mapped to `[0, 1]` | GP inputs, candidate distances, local penalties |
| Raw/model outcome | Measurements fitted by the GP models | posterior sampling |
| Transformed utility | Every objective oriented so larger is better | Pareto filtering, hypervolume, UCB-HVI, qLogNEHVI |

Raw outcomes and transformed utilities are never interchanged. In particular,
the fixed reference point is always expressed in transformed utility space.

## Objective transformations

`mobo_kit.objectives` defines an immutable `ObjectiveSpec`, a validated
`ObjectiveTransform`, and a BoTorch-compatible
`ConfiguredMCMultiOutputObjective`. Tensors use the shape contract
`[..., M] -> [..., M]`, preserving floating dtype and device.

For fixed anchors `lo < hi`:

```text
maximize: u(y) = (y - lo) / (hi - lo)
minimize: u(y) = (hi - y) / (hi - lo)
```

The target transformations are:

```text
Gaussian:         u(y) = exp(-0.5 * ((y - target) / sigma)^2)
Negative absolute u(y) = -abs(y - target) / scale
```

`sigma` or `scale` must be explicit, finite, and positive. Identity is valid
only for an already approved maximize utility. No anchor or reference point is
estimated from campaign observations.

Nonlinear transformations are applied to every raw posterior Monte Carlo
sample. For candidate `x`:

```text
Y_raw^(s)(x) ~ posterior(model, x)
U^(s)(x) = transform(Y_raw^(s)(x))
mu_U(x) = mean_s U^(s)(x)
sigma_U(x) = population_std_s U^(s)(x)
```

The default is the latent posterior (`observation_noise=False`). Posterior
sampling uses a caller-supplied seed and Sobol QMC samples.

## Discrete candidate pool

`sample_discrete_candidate_pool` samples integer index tuples directly from the
configured axes. It never materializes the 177,816,994,740-point D2D Cartesian
product. Each accepted index tuple is converted exactly to a physical grid row
and then normalized.

Observed, pending, and explicit avoid rows are converted back to integer grid
indices and excluded. Opt-in physical row constraints are evaluated only after
conversion to laboratory units. A request either returns the exact requested
pool in deterministic order for its seed or raises
`CandidatePoolSamplingError` with draw and rejection statistics. Constraints or
duplicate rules are never relaxed.

## R1 UCB-HVI

`posterior_utility_moments` evaluates candidates as singleton q-batches in
CPU-safe chunks. `score_ucb_hvi_pool` then forms, for each transformed utility
dimension,

```text
kappa = sqrt(beta)
UCB_j(x) = mu_U,j(x) + kappa * sigma_U,j(x), beta >= 0
```

For the non-dominated observed utility set `P` and explicit utility-space
reference `r`, the deterministic base score is

```text
a_UCB-HVI(x) = HV(P union {UCB(x)}; r) - HV(P; r).
```

True dominated or non-contributing points retain a raw score of zero. A small
epsilon is used only to represent scores in log space. The proposal wrapper
requires enough candidates above its explicit positive-HVI threshold; it does
not fill a batch with arbitrary zero-HVI points.

Public scoring APIs:

- `posterior_utility_moments`
- `hypervolume_improvement_scores`
- `score_ucb_hvi_from_moments`
- `score_ucb_hvi_pool`
- `propose_ucb_hvi_batch`

## Shared local-penalized selector

Both acquisition methods use `select_local_penalized_batch`. For normalized
inputs and optional positive dimension weights `w`, distance is

```text
d_w(x, z) = sqrt(sum_j w_j * (x_j - z_j)^2).
```

After selecting `x_i`, the soft exclusion factor applied to a remaining point
is

```text
phi_i(x) = 1 - exp(-0.5 * (d_w(x, x_i) / rho)^2), rho > 0.
```

The selector operates in log space:

```text
log a_pen(x) = log a_base(x) + sum_i log(max(phi_i(x), epsilon)).
```

Hard minimum selected-to-selected and optional selected-to-observed/pending
distances are masked before selection. Ties retain stable candidate-pool order.
If the exact batch is impossible, `UndersizedBatchError` reports the requested
and selected sizes, active thresholds, and remaining count; no fallback relaxes
the rules.

## R2 qLogNEHVI

`score_qlognehvi_singletons` constructs BoTorch 0.15.1
`qLogNoisyExpectedHypervolumeImprovement` with:

- the fitted `ModelListGP`;
- normalized `train_X` as `X_baseline`;
- the same configured Monte Carlo objective used by UCB-HVI;
- the same explicit transformed-utility reference point;
- a seeded Sobol sampler; and
- explicit pending points.

The scorer requires `ConfiguredMCMultiOutputObjective`; passing `None` or an
unversioned arbitrary objective fails before BoTorch can silently operate in raw
outcome space. The utility reference dimension is checked against both the
objective contract and the model output count.

The pool is evaluated in shape `N x 1 x D` and in caller-controlled chunks.
`propose_qlognehvi_penalized_batch` recomputes singleton base scores at every
selection step. Its pending set is the pre-existing pending set plus all points
already selected in the new batch. The shared selector then applies the same
soft penalty and hard distance policy used by UCB-HVI.

## Diagnostics

`mobo_kit.candidate_diagnostics` provides:

- within-batch pairwise normalized distances and min/mean/max summaries;
- nearest observed/pending distance per candidate;
- duplicate, grid-membership, and normalized-boundary checks;
- PCA, selected-condition parallel coordinates, distance heatmap, and
  base-versus-penalized acquisition plots.

Selection results retain pool indices, order, base/log scores, penalty factors,
nearest distances, acquisition-specific diagnostics, seeds, and settings.
Plot functions use the headless `Agg` backend and write only to caller-supplied
paths. The Step 2A example uses ignored `local_outputs/` and writes a strict
JSON provenance report containing method/contract versions, every seed,
pool-draw statistics, beta/kappa or MC settings, local-penalty settings,
per-selection scores, UCB utility moments, qLogNEHVI pending counts, and runtime
versions. It contains no campaign recipes.

## Determinism and performance

- Pool sampling uses a seeded NumPy `Generator` and stable acceptance order.
- Posterior and qLogNEHVI Monte Carlo sampling use explicit Sobol seeds.
- Candidate-pool evaluation has explicit chunk sizes.
- Sequential distance work is `O(N*q*D)`; no pool-wide `N x N` matrix is made.
- Hypervolume work is singleton candidate scoring against the fixed observed
  Pareto set.
- CPU floating-point reductions can differ below normal numerical tolerances
  across chunk shapes; selected indices are tested for deterministic reruns.

## Limitations before Step 2B

Step 2A does not ingest completed R0 outcomes, write workbooks, manage campaign
state, or generate real candidates. The exact production objective contract and
all fields rejected by the gate must be resolved at the experimental-team
meeting and frozen in a versioned configuration before a dry run. The read-only
v2 workbook profile requires the exact 29-column header tuple, sample rows 2-16,
numeric sample identifiers 1-15, blank row 17, and content range A1:AC18. A
local off-grid discrepancy is reported without snapping or copying the private
condition into tracked fixtures.
