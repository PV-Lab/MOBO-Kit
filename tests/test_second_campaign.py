"""The second campaign's objective contract, end to end.

`configs/campaign_d2d_perovskite_test.yaml` is a new contract on a new workbook:
uniformity and optoelectronic are computed differently from the first campaign,
the column layout moved, two grids changed and three constraints are active for
the first time in this project.

The synthetic half builds a sheet with the v3 headers -- INCLUDING their trailing
spaces -- so the contract is exercised without the ignored workbook. The real-
workbook half is marked `local_input` and skips without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mobo_kit.campaign import (
    BatchValidityError,
    build_design_from_config,
    build_objective_transform,
    load_campaign_config,
    measurement_specs,
    objective_names,
    run_r1_ucb,
    validate_batch,
)
from mobo_kit.constraints import constraint_violations, constraints_from_config
from mobo_kit.scores import ScoreSeverity, compute_measurements

CONFIG_PATH = "configs/campaign_d2d_perovskite_test.yaml"
ARCHIVED_CONFIG_PATH = "configs/campaign_d2d_perovskite.yaml"
SOURCE = "local_inputs/Summary Table Test.xlsx"

#: Exactly as Sheet1 spells them. Two carry a trailing space, which is not a typo
#: in this file -- it is what the header cell contains, and resolution has to cope
#: with it from both directions.
VOC_HEADER = "PL - Implied Voc (Max) Raw "
PHOTOCONDUCTANCE_HEADER = "Normalized photoconductance "


@pytest.fixture(scope="module")
def config() -> dict:
    return load_campaign_config(CONFIG_PATH)


# --------------------------------------------------------------------------- #
# the contract itself
# --------------------------------------------------------------------------- #


def test_the_new_contract_is_distinct_from_the_archived_one(config) -> None:
    """Every contract's utility space is its own, and they must not be confused.

    Uniformity is a three-term mean here and a three-term product in v2;
    optoelectronic is a mean of normalised values here and a log10 product there.
    A shared contract_version would make hypervolumes look comparable when they
    measure different spaces. v3 is itself archived now -- superseded by v4 -- so
    all that is asserted here is that the three versions are distinct.
    """
    archived = load_campaign_config(ARCHIVED_CONFIG_PATH)
    assert config["objectives"]["contract_version"] == "d2d-objectives-v3-test"
    assert (
        config["objectives"]["contract_version"]
        != archived["objectives"]["contract_version"]
    )
    assert archived["campaign"]["status"] == "archived"
    assert config["campaign"]["status"] == "archived"
    # objective ORDER is part of the contract: Y columns are positional
    assert objective_names(config) == ("uniformity", "optoelectronic", "thickness")


def test_both_new_scores_live_in_zero_to_one(config) -> None:
    """A mean of terms that are each in [0, 1] is in [0, 1] by construction, so
    these anchors are the objective's range and not a guess about the data."""
    specs = {spec["name"]: spec for spec in config["objectives"]["specs"]}
    for name in ("uniformity", "optoelectronic"):
        assert specs[name]["lower_anchor"] == 0.0
        assert specs[name]["upper_anchor"] == 1.0
    build_objective_transform(config)  # runs assert_scaling_is_campaign_fixed


def test_the_signal_verdicts_are_the_ones_measured_on_these_rows(config) -> None:
    """Nothing was inherited: every verdict here came from an intake run on this
    workbook, and two of the three came out differently from the first campaign's.

    Both non-thickness objectives sit below the leave-one-out null, so a batch is
    chosen on one informative axis and two uninformative ones.
    """
    status = {
        spec["name"]: spec["signal_status"] for spec in config["objectives"]["specs"]
    }
    assert status == {
        "uniformity": "exploration_only",
        "optoelectronic": "exploration_only",
        "thickness": "learnable",
    }


def test_the_optoelectronic_mean_function_stays_deleted(config) -> None:
    """The intake verdict was DELETE: the first campaign's linear anneal_temp trend
    made this objective's fit WORSE here (-0.5842 -> -0.6977), because the target
    was redefined underneath it. Reinstating it from the archived config is the
    obvious mistake, so it is pinned."""
    specs = {spec["name"]: spec for spec in config["objectives"]["specs"]}
    assert "mean_function" not in specs["optoelectronic"]
    # thickness keeps its block: inconclusive, but it clears the null either way
    assert specs["thickness"]["mean_function"]["response"] == "log"
    assert [
        feature["column"] for feature in specs["thickness"]["mean_function"]["features"]
    ] == ["speed_1", "precur_conc"]


def test_the_thickness_cross_check_is_tight_now(config) -> None:
    """The first campaign's `Thickness (avg)` was ROUND(mean(T1..T4)), so half a
    nanometre of disagreement was legitimate. This sheet's is a live unrounded
    AVERAGE, so anything above floating-point noise is real."""
    thickness = next(
        spec
        for spec in config["objectives"]["specs"]
        if spec["name"] == "thickness"
    )
    (check,) = thickness["measurement"]["cross_check"]
    assert check["atol"] == pytest.approx(1e-9)


# --------------------------------------------------------------------------- #
# the grid edits
# --------------------------------------------------------------------------- #


def test_the_two_grid_edits_and_nothing_else(config) -> None:
    archived = load_campaign_config(ARCHIVED_CONFIG_PATH)
    before = {item["name"]: item for item in archived["inputs"]}
    after = {item["name"]: item for item in config["inputs"]}
    assert list(before) == list(after), "input order is positional; it must not move"

    changed = {
        name
        for name in after
        if (after[name]["start"], after[name]["stop"], after[name]["step"])
        != (before[name]["start"], before[name]["stop"], before[name]["step"])
    }
    assert changed == {"time_2", "anti_time"}
    # time_2 reaches 0 so a one-step film is on-grid; anti_time steps by 1 so
    # sample 1's anti_time = 12 is an ordinary observation rather than a declared
    # off-grid exception
    assert (after["time_2"]["start"], after["time_2"]["step"]) == (0, 5)
    assert (after["anti_time"]["start"], after["anti_time"]["step"]) == (9, 1)


def test_the_grid_hole_at_time_2_equals_5_is_declared_not_silent(config) -> None:
    """Reaching 0 with step 5 also reaches 5, which the first campaign's grid
    excluded and no film has run. The constraint is what keeps it out."""
    design = build_design_from_config(dict(config))
    time_2_grid = design.var_array[design.names.index("time_2")]
    assert 5.0 in set(time_2_grid), "the arithmetic grid does contain it"

    constraints = constraints_from_config(dict(config), design)
    row = {name: design.var_array[i][1] for i, name in enumerate(design.names)}
    row.update({"speed_2": 1000.0, "time_1": 50.0, "time_2": 5.0, "anti_time": 9.0})
    values = np.asarray([[row[name] for name in design.names]], dtype=float)
    assert constraint_violations(values, design, constraints) == [
        ["second_stage_runs_at_least_10s"]
    ], "and the constraint is what excludes it"


# --------------------------------------------------------------------------- #
# recipes, on a synthetic sheet with the v3 headers
# --------------------------------------------------------------------------- #


def _sheet(rows: list[dict]) -> pd.DataFrame:
    """A frame keyed by the v3 headers, trailing spaces and all."""
    return pd.DataFrame(rows, dtype=object)


def test_the_recipes_reproduce_the_stored_scores_on_a_synthetic_sheet(config) -> None:
    """Sample 1 and sample 4 of the real sheet, transcribed. Sample 4 is one of
    the two clamped rows, so this covers the clamp on the way through as well."""
    specs = [spec for spec in measurement_specs(config) if spec is not None]
    frame = _sheet(
        [
            {
                "Coverage": 0.989,
                "Uniformity": 0.324584,
                "Phase purity": 0.9685,
                VOC_HEADER: 1.02683981553478,
                PHOTOCONDUCTANCE_HEADER: 0.763425,
                "Photoconductance (Max)": 5e-07,
                "T1": 584.4,
                "T2": 418.5,
                "T3": 692.0,
                "T4": 624.6,
                "T anom": None,
            },
            {
                "Coverage": 0.992,
                "Uniformity": 1.658775,  # clamped to 0.99
                "Phase purity": 0.786,
                VOC_HEADER: 1.13513544854332,
                PHOTOCONDUCTANCE_HEADER: 1.0,
                "Photoconductance (Max)": 3.42e-08,
                "T1": 657.7,
                "T2": 586.4,
                "T3": 693.2,
                "T4": 667.1,
                "T anom": 832.1,
            },
        ]
    )
    result = compute_measurements(frame, specs, sample_ids=[1, 4])

    # the workbook's own AB, AC and Z for those two rows
    assert result.values["uniformity"].tolist() == pytest.approx(
        [0.8776386666666668, 0.596], abs=1e-15
    )
    assert result.values["optoelectronic"].tolist() == pytest.approx(
        [0.7484410055481358, 0.9054055173369], abs=1e-15
    )
    assert result.values["thickness"].tolist() == pytest.approx(
        [579.875, 651.1], abs=1e-12
    )
    assert not result.has_errors


def _filler(**overrides) -> dict:
    """A row that satisfies every objective, so one can be varied at a time."""
    row = {
        "Coverage": 1.0,
        "Uniformity": 0.0,
        "Phase purity": 1.0,
        VOC_HEADER: 1.4,
        PHOTOCONDUCTANCE_HEADER: 1.0,
        "T1": 650.0,
        "T2": 650.0,
        "T3": 650.0,
        "T4": 650.0,
        "T anom": None,
    }
    row.update(overrides)
    return row


def test_a_variable_number_of_thickness_readings_is_normal(config) -> None:
    """Eleven of the fifteen rows carry three readings and four carry four, so a
    recipe demanding all four would reject two thirds of the campaign."""
    specs = [spec for spec in measurement_specs(config) if spec is not None]
    frame = _sheet(
        [
            _filler(T1=413.0, T2=430.2, T3=439.4, T4=None),
            _filler(T1=962.6, T2=961.1, T3=947.7, T4=942.8),
        ]
    )
    result = compute_measurements(frame, specs, sample_ids=[2, 3])
    assert result.inputs_used["thickness"].tolist() == [3, 4]
    assert result.values["thickness"].tolist() == pytest.approx(
        [427.5333333333333, 953.55], abs=1e-12
    )
    assert not result.has_errors


def test_the_v3_headers_resolve_despite_their_trailing_spaces(config) -> None:
    """Two of the sheet's headers end in a space, and the config quotes them
    verbatim. Names are compared stripped on BOTH sides, so either spelling
    resolves and neither silently reports a present column as missing."""
    specs = [spec for spec in measurement_specs(config) if spec is not None]
    declared = [item.column for spec in specs for item in spec.inputs]
    assert VOC_HEADER.rstrip() in declared, "the config side is stripped"
    assert VOC_HEADER not in declared

    with_spaces = _sheet([_filler()])
    without_spaces = with_spaces.rename(columns=lambda name: name.strip())
    assert list(with_spaces.columns) != list(without_spaces.columns)

    from_spaced = compute_measurements(with_spaces, specs, sample_ids=[1])
    from_stripped = compute_measurements(without_spaces, specs, sample_ids=[1])
    assert not from_spaced.has_errors
    pd.testing.assert_frame_equal(from_spaced.values, from_stripped.values)


def test_t_anom_is_excluded_from_the_mean_and_still_reported(config) -> None:
    specs = [spec for spec in measurement_specs(config) if spec is not None]
    frame = _sheet(
        [_filler(T1=657.7, T2=586.4, T3=693.2, T4=667.1, **{"T anom": 832.1})]
    )
    result = compute_measurements(frame, specs, sample_ids=[4])
    assert result.values["thickness"][0] == pytest.approx(651.1)
    codes = [f.code for f in result.findings if f.severity is ScoreSeverity.NOTE]
    assert "reading_excluded" in codes


# --------------------------------------------------------------------------- #
# constraints reach the batch gate
# --------------------------------------------------------------------------- #


def test_validate_batch_refuses_a_condition_that_breaks_a_constraint(config) -> None:
    """Deliberately redundant with the pool filter. The pool is the mechanism;
    this is the independent second route to the same answer, which is the check
    this project's three finite-but-wrong-number bugs all lacked."""
    design = build_design_from_config(dict(config))
    constraints = constraints_from_config(dict(config), design)
    good = {
        "speed_1": 2500.0,
        "time_1": 30.0,
        "speed_2": 1000.0,
        "time_2": 20.0,
        "precur_conc": 1.4,
        "precur_vol": 100.0,
        "anneal_temp": 120.0,
        "anneal_time": 30.0,
        "anti_vol": 150.0,
        "anti_time": 12.0,
    }
    frame = pd.DataFrame([good], columns=design.names)
    report = validate_batch(frame, design, expected_count=1, constraints=constraints)
    assert report["constraints_satisfied"] is True
    assert report["constraint_violations_per_condition"] == [[]]

    broken = dict(good, speed_2=0.0)  # time_2 still 20: exactly one of the pair is 0
    with pytest.raises(BatchValidityError, match="second_stage_all_or_nothing"):
        validate_batch(
            pd.DataFrame([broken], columns=design.names),
            design,
            expected_count=1,
            constraints=constraints,
        )


def test_constraints_are_inert_when_unconfigured(config) -> None:
    """DTLZ2 declares none, and its acceptance suite must be unaffected."""
    design = build_design_from_config(dict(config))
    frame = pd.DataFrame(
        [
            {
                "speed_1": 2500.0,
                "time_1": 30.0,
                "speed_2": 0.0,
                "time_2": 20.0,  # would break second_stage_all_or_nothing
                "precur_conc": 1.4,
                "precur_vol": 100.0,
                "anneal_temp": 120.0,
                "anneal_time": 30.0,
                "anti_vol": 150.0,
                "anti_time": 12.0,
            }
        ],
        columns=design.names,
    )
    report = validate_batch(frame, design, expected_count=1)
    assert report["constraint_violations_per_condition"] == [[]]
    assert report["constraints_declared"] == []


# --------------------------------------------------------------------------- #
# the launcher points at the campaign that is actually running
# --------------------------------------------------------------------------- #


def test_this_contract_is_archived_and_the_launcher_has_moved_on() -> None:
    """v3 was the DRY RUN -- it rehearsed this contract's shape on a workbook
    literally called "Test". The live campaign is v4, and the launcher points
    there; the pinning of that default lives in `test_final_campaign.py`.

    The 2026-08-18 regression this guards against is unchanged in kind: archiving
    a config without moving the launcher's default leaves the double-click path
    reading a new workbook against a retired contract, which surfaces as a
    missing-column error on an intact workbook.
    """
    from mobo_kit.launcher import DEFAULT_CONFIG

    assert load_campaign_config(CONFIG_PATH)["campaign"]["status"] == "archived"
    assert DEFAULT_CONFIG != CONFIG_PATH
    assert load_campaign_config(DEFAULT_CONFIG)["campaign"]["status"] == "active"


def test_reading_a_workbook_against_the_wrong_contract_says_which_contract() -> None:
    """A column mismatch is almost never a broken workbook; it is a config
    describing a different campaign. The message has to say so, because the
    obvious reading of "missing column" sends someone to edit the sheet."""
    from openpyxl import Workbook

    from mobo_kit.workbook_io import CandidateSheetError, read_campaign_workbook

    archived = load_campaign_config(ARCHIVED_CONFIG_PATH)
    book = Workbook()
    sheet = book.active
    sheet.title = "Sheet1"
    # a v3-shaped sheet: the archived config wants "PL - Implied Voc (Max)"
    sheet.append(["Sample number", VOC_HEADER])
    sheet.append([1, 1.0])
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        path = _Path(tmp) / "Summary Table Test.xlsx"
        book.save(path)
        with pytest.raises(CandidateSheetError) as caught:
            read_campaign_workbook(path, archived)

    message = str(caught.value)
    assert "archived" in message.lower()
    assert "d2d-objectives-v2-nm-thickness" in message
    # and it points at the column that is almost certainly the same measurement
    assert "PL - Implied Voc (Max) Raw" in message


def test_a_column_level_finding_does_not_pretend_to_have_a_row(config) -> None:
    """`sample ?` reads as a row whose identity was lost. The rank-agreement
    finding is about a column and says so."""
    from mobo_kit.scores import ScoreFinding, ScoreSeverity

    finding = ScoreFinding(
        severity=ScoreSeverity.WARNING,
        code="agreement_not_monotonic",
        objective="optoelectronic",
        row_position=-1,
        sample_id=None,
        message="ranks backwards",
    )
    assert finding.is_column_level
    assert "sample ?" not in str(finding)
    assert "all rows, optoelectronic" in str(finding)

    per_row = ScoreFinding(
        severity=ScoreSeverity.WARNING,
        code="readings_disagree",
        objective="thickness",
        row_position=0,
        sample_id=1,
        message="readings disagree",
    )
    assert not per_row.is_column_level
    assert "sample 1, thickness" in str(per_row)


# --------------------------------------------------------------------------- #
# the real workbook
# --------------------------------------------------------------------------- #

requires_workbook = pytest.mark.skipif(
    not Path(SOURCE).is_file(), reason=f"{SOURCE} is not present in this checkout"
)


@pytest.mark.local_input
@requires_workbook
def test_every_measured_row_is_on_grid_and_satisfies_every_constraint(config) -> None:
    """Both halves matter. Off-grid observations drop out of pool bookkeeping, and
    a constraint that rejects a film the group actually ran is far more likely to
    be wrong than the film is."""
    from mobo_kit.workbook_io import read_campaign_workbook

    contents = read_campaign_workbook(SOURCE, config)
    assert contents.n_rows == 15
    assert contents.errors == ()

    design = build_design_from_config(dict(config))
    X = contents.inputs.to_numpy(float)
    off_grid = [
        (contents.sample_ids[i], name, value)
        for j, name in enumerate(design.names)
        for i, value in enumerate(X[:, j])
        if not np.any(np.isclose(design.var_array[j], value, rtol=0.0, atol=1e-9))
    ]
    assert off_grid == []

    constraints = constraints_from_config(dict(config), design)
    assert constraint_violations(X, design, constraints) == [[] for _ in range(15)]


@pytest.mark.local_input
@requires_workbook
def test_the_computed_objectives_match_the_stored_score_columns(config) -> None:
    """The policy for this workbook is that the stored scores are authoritative and
    the recompute is the cross-check, so the two agreeing is the whole claim."""
    from mobo_kit.workbook_io import read_campaign_workbook

    contents = read_campaign_workbook(SOURCE, config)
    computed = contents.model_values.to_numpy(float)
    stored = contents.workbook_values.to_numpy(float)
    assert computed.shape == stored.shape == (15, 3)
    assert np.abs(computed - stored).max() < 1e-9
    assert not [
        f for f in contents.findings if f.code == "cross_check_mismatch"
    ]


@pytest.mark.local_input
@requires_workbook
def test_the_photoconductance_normalization_warns_on_the_real_rows(config) -> None:
    """The live defect, pinned so it cannot be quietly resolved by editing config.

    When the group supplies the real formula this test should start failing, and
    that failure is the signal to update the recorded number rather than to
    loosen the check.
    """
    from mobo_kit.workbook_io import read_campaign_workbook

    contents = read_campaign_workbook(SOURCE, config)
    warning = next(
        f for f in contents.findings if f.code == "agreement_not_monotonic"
    )
    assert warning.severity is ScoreSeverity.WARNING
    assert "-0.5484" in warning.message
    assert contents.errors == (), "a finding, never a gate"


@pytest.mark.local_input
@pytest.mark.slow
@requires_workbook
def test_r1_on_the_real_workbook_is_valid_and_deterministic(config) -> None:
    """One full R1 at production settings: five conditions, on grid, constraint
    satisfying, and identical on a second run at the same seed."""
    from mobo_kit.workbook_io import read_campaign_workbook

    contents = read_campaign_workbook(SOURCE, config)
    X = contents.inputs.to_numpy(float)
    Y = contents.model_values.to_numpy(float)

    first = run_r1_ucb(config, X, Y, n=5, seed=73)
    assert len(first.conditions) == 5
    assert first.diagnostics["validity"]["constraints_satisfied"] is True
    assert first.diagnostics["validity"]["constraint_violations_per_condition"] == [
        [] for _ in range(5)
    ]
    assert first.diagnostics["observed_rows_violating_constraints"] == []
    assert len(first.diagnostics["constraints_declared"]) == 3
    # the sampler draws until the pool is full, so a constraint that gutted the
    # space would still yield a normal-looking pool; the survival rate is the only
    # place that shows
    assert 0.0 < first.diagnostics["constraint_pool_survival_rate"] <= 1.0

    second = run_r1_ucb(config, X, Y, n=5, seed=73)
    pd.testing.assert_frame_equal(first.conditions, second.conditions)
