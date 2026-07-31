# `manifest.csv` schema

Written by `scripts/plot_round_simulation.py` to
`local_outputs/round_simulations/manifest.csv`. One row per parameter cell.

This file, not the figures, is the decision instrument. The figures show what a
landscape looks like; the manifest answers whether changing a knob changed
anything, which is the question the sweep exists to settle. A knob that produces a
byte-identical batch at every setting is inert on this problem, and no amount of
looking at contour plots will tell you that.

Column order is pinned by `MANIFEST_COLUMNS` in the script and asserted by
`tests/test_plot_round_simulation.py::test_manifest_row_has_exactly_the_declared_columns`.

| column | type | meaning |
|---|---|---|
| `condition_id` | int | 1-based position in the run. Not stable across different `--conditions` filters; use the slug or `(radius, beta)` to join. |
| `arm` | str | `radius` (beta held at 4), `beta` (radius held at 0.25), `both` (the shared 0.25/4 anchor), or `grid` under `--full-grid`. |
| `radius` | float | `local_penalization.radius` for this cell. |
| `beta` | float | `rounds.r1.beta` for this cell. |
| `min_batch_distance` | float | Always 0.15. Pinned, never swept, so "spacing" means one thing in every row. |
| `seed` | int | 73 unless `--seed` overrides. The oracle, both acquisitions, and every pool draw use it. |
| `r1_batch_hash` | str | 16 hex chars. SHA-256 of the R1 conditions, **sorted** and rounded to 12 dp. Two cells sharing a hash proposed the same set of recipes; order is not part of the identity. |
| `r2_batch_hash` | str | The same for R2. |
| `r1_min_pairwise_distance` | float | Smallest normalised distance within the R1 batch. Compare against `radius` to see whether local penalisation had anything to act on. |
| `r2_min_pairwise_distance` | float | The same for R2. |
| `r1_boundary_coords_total` | int | How many coordinates across the whole R1 batch sit exactly at a range edge. On the live campaign this is driven by the monotone `anneal_temp` mean function, not by either knob. |
| `r2_boundary_coords_total` | int | The same for R2. |
| `r1_boundary_coords_per_condition` | JSON list | Per-condition breakdown, so one pinned condition is distinguishable from five mildly-pinned ones. |
| `r2_boundary_coords_per_condition` | JSON list | The same for R2. |
| `hv_r0` | float | Hypervolume of the 15 oracle-scored R0 points, utility space, at the campaign's declared `reference_point_utility`. |
| `hv_r0_r1` | float | After adding the 5 R1 conditions. |
| `hv_r0_r1_r2` | float | After adding the 3 R2 conditions. |
| `hv_gain_r1` | float | `hv_r0_r1 - hv_r0`. |
| `hv_gain_r2` | float | `hv_r0_r1_r2 - hv_r0_r1`. |
| `baseline_hv_model_space` | float | The R1 observed baseline hypervolume with the objective link decoded once — what this script uses. Constant across rows; it is a property of the observed set. |
| `baseline_hv_as_run_r1_ucb_calls_it` | float | The same quantity as `campaign.run_r1_ucb` currently encodes it. Constant across rows. See `ROUND_SIM_DELTA.md` §1: the gap is not a rounding difference. |
| `r1_fit_warnings` | int | Fit-guard warnings raised by the GP that proposed R1. Guard warnings only, not the ~18 numpy deprecation notices per fit. |
| `r2_fit_warnings` | int | The same for the R2 model. |
| `final_fit_warnings` | int | The same for the 23-point model the heatmaps render. Non-zero puts a banner on that cell's figures. |
| `mean_utility_r0` | float | Mean utility over all objectives and all 15 R0 points. A summary, not a ranking: it averages three objectives that are not commensurable. |
| `mean_utility_r1` | float | The same over the 5 R1 conditions. |
| `mean_utility_r2` | float | The same over the 3 R2 conditions. |

## Reading it

**Hypervolume rises monotonically by construction.** Adding points can only grow a
Pareto front, so `hv_r0 <= hv_r0_r1 <= hv_r0_r1_r2` holds in every row and proves
nothing on its own — random sampling satisfies it too. What carries information is
the *size* of `hv_gain_r1` compared across cells, and the batch hashes.

**Cells with equal `(r1_batch_hash, r2_batch_hash)` are the same experiment.**
Their hypervolumes and utilities are then identical by construction, not by
agreement, and quoting them as independent replicates would be double counting.

**Both baseline columns are constant within a run.** They are recorded per row so a
single row is self-describing when it is pasted somewhere else.
