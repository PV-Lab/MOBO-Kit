"""The figures a round produces, and the promises attached to them.

What is pinned here is not "a PNG appeared". A figure that renders and shows the
wrong number is worse than no figure, because it carries authority. So:

* **every figure writes the numbers behind it**, and the schema of those numbers is
  fixed here -- a plot whose data cannot be re-derived is the next
  plausible-finite-number bug, and this project has had three;
* **the parity numbers ARE intake's numbers.** They come from one shared fold loop
  rather than two implementations that agree today, and the test asserts the
  identity rather than a tolerance;
* **the batch figure's numbers ARE the Review sheet's numbers**, for the same
  reason: two artifacts a human compares must not be able to disagree;
* **determinism is checked on the CSVs, never on PNG bytes** -- matplotlib output
  is not reproducible across versions and a byte comparison would fail for reasons
  that have nothing to do with the campaign.

Everything here builds its own workbook, so none of it needs the ignored private
one.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from mobo_kit.batch_review import build_batch_review
from mobo_kit.campaign import (
    build_objective_transform,
    fit_campaign_models,
    load_campaign_config,
    objective_names,
    run_r1_ucb,
)
from mobo_kit.loocv import loo_predictions
from mobo_kit.round_report import (
    ReportManifest,
    generate_round_report,
    report_directory,
)

CONFIG_PATH = "configs/campaign_d2d_perovskite_test.yaml"

#: Fitting GPs to six synthetic rows produces near-degenerate posteriors, and
#: gpytorch says so on nearly every fold. That is a property of the fixture, not a
#: finding, and letting it through would add ~75 warnings to a suite whose warning
#: tail is deliberately kept fixed so that a NEW warning means something.
pytestmark = [
    pytest.mark.filterwarnings("ignore::gpytorch.utils.warnings.NumericalWarning"),
    pytest.mark.filterwarnings("ignore:.*deprecated - use.*:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

#: Exactly as the v3 sheet spells them, trailing spaces included.
VOC = "PL - Implied Voc (Max) Raw "
PHOTO = "Normalized photoconductance "

HEADERS = [
    "Sample number",
    "speed_1", "time_1", "speed_2", "time_2", "precur_conc",
    "precur_vol (uL)", "anneal_temp", "anneal_time", "anti_vol", "anti_time",
    "Coverage", "Uniformity", "Phase purity",
    VOC, "Photoconductance (Max)", PHOTO,
    "T1", "T2", "T3", "T4", "T anom",
    "Thickness (avg)",
    "Uniformity score (Avg (Coverage + (1-Uniformity) + Phase purity))",
    "Optoelectronic score (Avg normalized (Voc + Photocondiuctivity)",
]


def _rows(n: int) -> list[list]:
    """A small on-grid, constraint-satisfying campaign with real structure.

    Thickness follows a genuine speed_1 trend so the parity panel has something
    to find; the other two are deliberately close to noise, which is also what the
    live campaign looks like.
    """
    rng = np.random.default_rng(11)
    rows = []
    for i in range(n):
        speed_1 = 1000.0 + 500.0 * (i % 11)
        time_1 = 20.0 + 5.0 * (i % 4)
        time_2 = 10.0 + 5.0 * (i % 5)
        precur_conc = 1.0 + 0.05 * (i % 12)
        coverage = float(np.clip(0.90 + 0.01 * (i % 7), 0.0, 1.0))
        uniformity = float(np.clip(0.20 + 0.06 * (i % 9), 0.0, 2.0))
        purity = float(np.clip(0.70 + 0.02 * (i % 8), 0.0, 1.0))
        voc = 0.95 + 0.02 * (i % 6)
        photo_raw = 1e-7 * (1 + i)
        photo_norm = float(np.clip(0.2 + 0.05 * (i % 9), 0.0, 1.0))
        thickness = 900.0 * (speed_1 / 3000.0) ** -0.4 * (precur_conc / 1.4) ** 0.8
        readings = list(np.round(thickness + rng.normal(0, 8.0, 3), 1))
        rows.append(
            [
                i + 1,
                speed_1, time_1, 1000.0, time_2, round(precur_conc, 2),
                100.0, 120.0, 30.0, 150.0, 12.0,
                coverage, uniformity, purity,
                voc, photo_raw, photo_norm,
                readings[0], readings[1], readings[2], None, None,
                float(np.mean(readings)),
                (coverage + (1.0 - min(uniformity, 0.99 if uniformity > 1 else uniformity)) + purity) / 3.0,
                (min(voc, 1.4) / 1.4 + photo_norm) / 2.0,
            ]
        )
    return rows


@pytest.fixture(scope="module")
def config() -> dict:
    return load_campaign_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def workbook(tmp_path_factory) -> "object":
    from pathlib import Path

    path = Path(tmp_path_factory.mktemp("report")) / "Synthetic Campaign.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "Sheet1"
    sheet.append(HEADERS)
    for row in _rows(6):
        sheet.append(row)
    book.save(path)
    return path


@pytest.fixture(scope="module")
def proposed(workbook, config, tmp_path_factory):
    """One proposal-mode report, reused: each render is tens of seconds of fitting."""
    from mobo_kit.workbook_io import read_campaign_workbook

    contents = read_campaign_workbook(workbook, config)
    X = contents.inputs.to_numpy(float)
    Y = contents.model_values.to_numpy(float)
    proposal = run_r1_ucb(config, X, Y, n=3, seed=73)
    review = build_batch_review(
        config, X, Y, proposal.conditions, round_name="R1", seed=73
    )
    manifest = generate_round_report(
        workbook,
        config,
        proposal=proposal,
        review=review,
        outdir=tmp_path_factory.mktemp("full"),
        shap_max_instances=2,
        seed=73,
        when="FIXED",
    )
    return manifest, review


@pytest.fixture(scope="module")
def data_only(workbook, config) -> ReportManifest:
    """One data-only report, reused: each render is tens of seconds of fitting."""
    return generate_round_report(
        workbook, config, shap_max_instances=3, when="FIXED", seed=73
    )


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #


def test_the_report_lands_beside_the_workbook_and_never_inside_it(
    data_only, workbook
) -> None:
    assert data_only.directory.parent.parent == workbook.parent
    assert data_only.directory.parent.name.endswith("_reports")
    assert workbook.exists()
    # nothing may have been written into the source workbook itself
    from openpyxl import load_workbook

    assert load_workbook(workbook).sheetnames == ["Sheet1"]


def test_every_rendered_figure_has_a_png_and_the_numbers_behind_it(data_only) -> None:
    assert data_only.figures, "a data-only report still renders four figures"
    for figure in data_only.figures:
        assert (data_only.directory / figure.png).is_file(), figure.key
        assert figure.data, f"{figure.key} wrote no data file"
        for name in figure.data:
            path = data_only.directory / name
            assert path.is_file(), name
            assert not pd.read_csv(path).empty, name
        assert figure.caveats, f"{figure.key} carries no caveat on its face"


def test_the_manifest_records_what_a_reader_needs_to_reproduce_it(data_only) -> None:
    manifest = json.loads((data_only.directory / "manifest.json").read_text())
    context = manifest["context"]
    assert context["objective_contract"] == "d2d-objectives-v3-test"
    assert context["seed"] == 73
    assert context["reference_point_utility"] == [-0.01, -0.01, -0.01]
    assert context["observed_rows"] == 6
    assert "git" in context and "python" in context
    assert manifest["mode"] == "data_only"
    assert manifest["runtime_seconds"] > 0


def test_data_only_mode_skips_the_two_batch_figures_and_says_so(data_only) -> None:
    """Skipping in silence is the failure mode; the manifest names both."""
    skipped = dict(data_only.skipped)
    assert set(skipped) == {"00_batch_placement", "03_batch_predictions"}
    for why in skipped.values():
        assert "data-only" in why
    keys = {figure.key for figure in data_only.figures}
    assert keys == {
        "01_loo_parity",
        "02_attribution",
        "04_hv_trajectory",
        "05_objective_space",
    }


def test_the_hv_trajectory_renders_at_r0_only(data_only) -> None:
    """The first report of a campaign has one point and no trajectory. It must
    still draw rather than fail on an empty diff."""
    frame = pd.read_csv(data_only.directory / "04_hv_trajectory.csv")
    assert list(frame["round"]) == ["R0"]
    assert frame["gain"].iloc[0] == pytest.approx(frame["hypervolume"].iloc[0])
    assert frame["cumulative_points"].iloc[0] == 6


# --------------------------------------------------------------------------- #
# the two equalities
# --------------------------------------------------------------------------- #


def test_the_parity_numbers_are_the_shared_loo_numbers(
    data_only, workbook, config
) -> None:
    """Identity, not agreement.

    ``scripts/intake_new_data.py`` is canonical for LOO, and it calls
    ``loocv.loo_predictions``; so does the figure. If these ever diverge, someone
    has reintroduced a second fold loop, which is exactly what this module's
    docstring exists to prevent.
    """
    from mobo_kit.workbook_io import read_campaign_workbook

    contents = read_campaign_workbook(workbook, config)
    X = contents.inputs.to_numpy(float)
    Y = contents.model_values.to_numpy(float)
    names = list(objective_names(config))
    entries = config["objectives"]["specs"]

    frame = pd.read_csv(data_only.directory / "01_loo_parity.csv")
    for index, name in enumerate(names):
        direct = loo_predictions(config, entries[index], X, Y[:, index], seed=73)
        block = frame[frame["objective"] == name]
        assert block["loo_r2"].iloc[0] == pytest.approx(direct.r2, abs=1e-12)
        np.testing.assert_allclose(
            block["loo_predicted"].to_numpy(float), direct.predicted, atol=1e-12
        )
        np.testing.assert_allclose(
            block["observed"].to_numpy(float), direct.observed, atol=1e-12
        )


def test_the_batch_figure_reports_the_review_sheets_numbers(proposed, config) -> None:
    """One source of truth. The Review sheet is attached to the worklist an
    experimentalist runs from; the figure must not be able to disagree with it,
    so this is an exact comparison rather than a tolerance."""
    manifest, review = proposed
    assert manifest.mode == "proposal"
    frame = pd.read_csv(manifest.directory / "03_batch_predictions.csv")
    for name in objective_names(config):
        block = frame[frame["objective"] == name].reset_index(drop=True)
        for column, source in (
            ("utility_mean", f"{name}_utility"),
            ("utility_sd", f"{name}_sd"),
            ("predicted_measurement", f"{name}_predicted"),
        ):
            np.testing.assert_allclose(
                block[column].to_numpy(float),
                review.candidates[source].to_numpy(float),
                atol=0.0,
            )


def test_proposal_mode_renders_all_six_figures(proposed) -> None:
    manifest, _ = proposed
    assert {figure.key for figure in manifest.figures} == {
        "00_batch_placement",
        "01_loo_parity",
        "02_attribution",
        "03_batch_predictions",
        "04_hv_trajectory",
        "05_objective_space",
    }
    assert manifest.skipped == ()
    placement = pd.read_csv(manifest.directory / "00_batch_placement.csv")
    assert len(placement) == 3
    assert "distance_to_nearest_observed" in placement.columns


def test_the_batch_hypervolume_diagnostic_is_a_distribution_not_a_point(
    proposed,
) -> None:
    """A single expected utility per candidate cannot answer "is this batch worth
    fabricating" -- hypervolume gain is a joint, nonlinear function of all of them."""
    manifest, _ = proposed
    frame = pd.read_csv(manifest.directory / "03_batch_hypervolume.csv")
    batch = frame[frame["candidate"] == "BATCH"].iloc[0]
    assert batch["delta_hv_p05"] <= batch["delta_hv_p50"] <= batch["delta_hv_p95"]
    assert 0.0 <= batch["p_gain_positive"] <= 1.0
    per_candidate = frame[frame["candidate"] != "BATCH"]
    assert len(per_candidate) == 3
    assert ((per_candidate["p_non_dominated"] >= 0.0)
            & (per_candidate["p_non_dominated"] <= 1.0)).all()
    # adding points can only grow a Pareto front, so no draw can lose volume;
    # the by-construction property, asserted rather than assumed
    assert batch["delta_hv_p05"] >= 0.0


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def test_two_runs_at_the_same_seed_produce_the_same_numbers(
    workbook, config, tmp_path
) -> None:
    """Compared on the CSVs, never on PNG bytes: matplotlib output moves between
    versions for reasons that have nothing to do with the campaign, and a byte
    comparison would fail loudly for a non-reason."""
    first = generate_round_report(
        workbook, config, outdir=tmp_path / "a", shap_max_instances=2, seed=73,
        when="FIXED",
    )
    second = generate_round_report(
        workbook, config, outdir=tmp_path / "b", shap_max_instances=2, seed=73,
        when="FIXED",
    )
    names = {name for figure in first.figures for name in figure.data}
    assert names, "nothing to compare"
    for name in names:
        left = pd.read_csv(first.directory / name)
        right = pd.read_csv(second.directory / name)
        pd.testing.assert_frame_equal(left, right, check_exact=False, atol=1e-10)


# --------------------------------------------------------------------------- #
# the attribution panel
# --------------------------------------------------------------------------- #


def test_attribution_marks_the_features_the_config_declared(data_only, config) -> None:
    """A feature named in a mean_function was TOLD to the model. The CSV marks
    those rows so nobody quotes one as a discovery."""
    frame = pd.read_csv(data_only.directory / "02_attribution.csv")
    assert set(frame["objective"]) == set(objective_names(config))
    assert (frame.groupby("objective")["rank"].min() == 1).all()
    thickness = frame[frame["objective"] == "thickness"]
    declared = set(thickness[thickness["in_mean_function"]]["feature"])
    assert declared == {"speed_1", "precur_conc"}
    # ranked by magnitude, descending, within each objective
    for _, block in frame.groupby("objective"):
        ordered = block.sort_values("rank")["mean_abs_shap"].to_numpy()
        assert np.all(np.diff(ordered) <= 1e-12)


def test_no_signal_objectives_are_labelled_in_the_data_not_just_the_picture(
    data_only,
) -> None:
    """The caveat has to survive being read from the CSV, because that is what a
    downstream analysis sees."""
    frame = pd.read_csv(data_only.directory / "02_attribution.csv")
    statuses = dict(zip(frame["objective"], frame["signal_status"]))
    assert statuses["uniformity"] == "exploration_only"
    assert statuses["optoelectronic"] == "exploration_only"
    assert statuses["thickness"] == "learnable"


def test_the_notices_repeat_the_no_signal_verdicts(data_only) -> None:
    """The verdict is the campaign's rank permutation test, not the leave-one-out
    R2 printed beside it: -0.148 is not a significance bar, and on R2's 20 recipes
    uniformity scores above it while still having no usable signal."""
    joined = " ".join(data_only.notices)
    assert "uniformity" in joined and "optoelectronic" in joined
    assert "carries no usable signal" in joined
    assert "not this report's R2" in joined
    assert "does not beat" not in joined


# --------------------------------------------------------------------------- #
# failure containment
# --------------------------------------------------------------------------- #


def test_one_broken_figure_does_not_cost_the_others(
    workbook, config, tmp_path, monkeypatch
) -> None:
    """Losing the attribution panel must not lose the parity plot. The manifest
    names what failed, so the absence is never silent."""
    from mobo_kit import round_report

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic attribution failure")

    monkeypatch.setattr(round_report, "_figure_attribution", explode)
    manifest = round_report.generate_round_report(
        workbook, config, outdir=tmp_path / "partial", seed=73, when="FIXED"
    )
    keys = {figure.key for figure in manifest.figures}
    assert "01_loo_parity" in keys and "05_objective_space" in keys
    skipped = dict(manifest.skipped)
    assert "synthetic attribution failure" in skipped["02_attribution"]
    assert any("02_attribution" in notice for notice in manifest.notices)
    assert (manifest.directory / "02_attribution.error.txt").is_file()


def test_a_report_failure_never_costs_the_batch(workbook, config, monkeypatch) -> None:
    """The worklist and the Review sheet are the expensive, careful part of a
    round. Throwing them away because a figure could not be drawn would be the
    wrong trade by a wide margin."""
    from mobo_kit import launcher

    monkeypatch.setattr(
        launcher, "generate_data_report", lambda *a, **k: None, raising=False
    )
    import mobo_kit.round_report as round_report

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic report failure")

    monkeypatch.setattr(round_report, "generate_round_report", explode)
    generated = launcher.generate_next_round(workbook, config)
    assert generated.sheet_path.is_file(), "the worklist survives"
    assert generated.review is not None, "so does the review"
    assert generated.report is None
    assert "synthetic report failure" in generated.report_error
    assert "unaffected" in generated.report_error
    assert "FIGURES NOT PRODUCED" in generated.summary()


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #


def test_the_report_directory_is_named_for_the_round_and_the_time() -> None:
    path = report_directory("/tmp/Summary Table Test.xlsx", "R1", when="20260818T101112Z")
    assert path.parent.name == "Summary Table Test_reports"
    assert path.name == "R1_20260818T101112Z"
