# The R1 batch was withdrawn and reissued, 2026-07-31

**No films were fabricated from the withdrawn batch.** The defect was caught while
the batch was still awaiting human review, which is what the review gate is for.

## What was wrong

`campaign.run_r1_ucb` handed its observed hypervolume-improvement baseline to
`ObjectiveTransform.transform` in **measurement space**. That transform is a
model-output decoder: it applies `exp()` to a log-link objective before computing
utility. Thickness in nanometres was therefore exponentiated a second time.
`exp(360…1303)` saturates the 650 nm Gaussian to exactly `0.0` — a finite number,
so neither the transform's own finiteness check nor the caller's fired.

Every one of the 15 observations scored **thickness utility 0.0**, so R1 chose its
candidates against a baseline front with no thickness axis at all.

| | withdrawn | reissued |
|---|---:|---:|
| observed baseline hypervolume | **0.004659** | **0.436442** |
| baseline Pareto set | 2 points | 5 points |
| minimum pairwise spacing | 0.9209 | 0.6337 |
| boundary coordinates | 13 | 12 |

Fixed in commit `4b76670` by `ObjectiveTransform.encode_measurements`, with
`transform_measurements` as the one-call safe route. Annie Xu had already found
and fixed this independently on `ax_plots_simulation`, as
`_physical_to_model_output`, before we knew it existed.

## What actually changed in the batch

**Four of the five conditions are identical.** One was replaced:

| | speed_1 | time_1 | speed_2 | time_2 | precur_conc | precur_vol | anneal_temp | anneal_time | anti_vol | anti_time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **dropped** | 4000 | 45 | 1500 | 30 | 1.70 | 40 | 105 | 10 | 170 | 15 |
| **added** | 2500 | 50 | 3500 | 35 | 1.45 | 70 | 105 | 15 | 135 | 13 |

The reissued batch in full:

| # | speed_1 | time_1 | speed_2 | time_2 | precur_conc | precur_vol | anneal_temp | anneal_time | anti_vol | anti_time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2500 | 50 | 1000 | 50 | 1.45 | 50 | 100 | 15 | 100 | 11 |
| 2 | 2500 | 50 | 3500 | 35 | 1.45 | 70 | 105 | 15 | 135 | 13 |
| 3 | 1500 | 50 | 3500 | 45 | 1.30 | 50 | 105 | 25 | 200 | 11 |
| 4 | 3000 | 50 | 2000 | 20 | 1.55 | 80 | 105 | 60 | 130 | 11 |
| 5 | 2000 | 50 | 2500 | 10 | 1.35 | 90 | 100 | 10 | 110 | 13 |

The withdrawn batch was more spread out — 0.9209 against 0.6337 — which is worth
saying plainly: **the defect made the batch look better diversified than the model
actually justified.** With the thickness axis of the baseline pinned at zero,
candidates were being separated on a distorted score.

## What did not change

Two standing observations survive the fix, so nothing that rests on them needs
revisiting:

- **The `speed_1 = 1000` corner is still skipped.** The reissued batch's minimum
  `speed_1` is 1500, as before. The low-speed corner remains a measurement
  question — samples 1 and 12 still contradict each other and sample 12's 1155 nm
  is still `ROUND(mean(1600, 709))`.
- **`anneal_temp` still pins to its lower bound**, at 100–105 across all five
  conditions. That is the monotone linear mean function speaking, exactly as
  recorded, and the open question remains chemical rather than numerical.

## Numbers that came from the withdrawn batch

Anything quoting the old batch's *diagnostics* is void and has been corrected in
place:

- the minimum spacing of **0.921** in `CAMPAIGN_STATUS.md`, which was used to argue
  that `radius` is "probably inert" on the live campaign. It is not — see the
  measured staircase in that file.
- the probe numbers in issue 4 (thickness utility 0.786 → 0.223 at an sd ratio of
  1.02). Those came from the withdrawn review artifact and must be re-read from
  the reissued one.

## Where the artifacts are

`local_inputs/Summary Table_R1_Candidates.xlsx`, regenerated through
`launcher.generate_next_round` — the same path the double-click launcher uses —
with the `R1_Candidates` worklist and the `Review` sheet. The withdrawal is
declared in `configs/campaign_d2d_perovskite.yaml` under `review.notes`, so it
travels with the Review sheet if that is forwarded on its own. **Delete that note
once R1 is measured.**

No prior candidate workbook existed on disk to archive: the withdrawn batch was
described in `CAMPAIGN_STATUS.md` and echoed to the launcher pane, but
`Summary Table_R1_Candidates.xlsx` had never been written.
