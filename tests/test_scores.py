from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mobo_kit.scores import (
    CrossCheck,
    MeasurementInput,
    MeasurementSpec,
    ScoreSeverity,
    ScoreValidationError,
    compute_measurements,
    entry_columns,
    measurement_spec_from_config,
    row_completeness,
)

#: The character Excel leaves in cells that look empty.
NBSP = "\u00a0"


def _uniformity() -> MeasurementSpec:
    return MeasurementSpec(
        name="uniformity",
        recipe="product",
        inputs=(
            MeasurementInput("Coverage"),
            MeasurementInput("Uniformity", "complement"),
            MeasurementInput("Phase purity"),
        ),
        cross_checks=(CrossCheck("Uniformity score", 1e-9),),
    )


def _optoelectronic() -> MeasurementSpec:
    return MeasurementSpec(
        name="optoelectronic",
        recipe="log10_product",
        inputs=(MeasurementInput("PL"), MeasurementInput("PC")),
        cross_checks=(CrossCheck("Optoelectronic score", 1e-9),),
    )


def _thickness(**kwargs) -> MeasurementSpec:
    return MeasurementSpec(
        name="thickness",
        recipe="mean_of_present",
        inputs=tuple(MeasurementInput(f"T{i}") for i in (1, 2, 3, 4)),
        cross_checks=(CrossCheck("Thickness (avg)", 0.5),),
        excluded=("T anom",),
        **kwargs,
    )


def _codes(result, severity: ScoreSeverity) -> list[str]:
    return [f.code for f in result.findings if f.severity is severity]


# --------------------------------------------------------------------------- #
# the recipes
# --------------------------------------------------------------------------- #


def test_product_multiplies_and_takes_the_complement() -> None:
    frame = pd.DataFrame({"Coverage": [1.0], "Uniformity": [0.33], "Phase purity": [0.98]})
    result = compute_measurements(frame, [_uniformity()])
    assert result.values["uniformity"][0] == pytest.approx(1.0 * 0.67 * 0.98)
    assert not result.has_errors


def test_log10_product_sums_logs_rather_than_logging_a_product() -> None:
    """Algebraically identical, but the sum cannot overflow on the way there.
    Photoconductance runs to 1e-7, so the product is small but the logs are not."""
    frame = pd.DataFrame({"PL": [0.0459], "PC": [6.73e-07]})
    result = compute_measurements(frame, [_optoelectronic()])
    assert result.values["optoelectronic"][0] == pytest.approx(
        math.log10(0.0459 * 6.73e-07)
    )


def test_log10_product_survives_inputs_whose_product_would_underflow() -> None:
    frame = pd.DataFrame({"PL": [1e-200], "PC": [1e-200]})
    result = compute_measurements(frame, [_optoelectronic()])
    assert result.values["optoelectronic"][0] == pytest.approx(-400.0)


def test_mean_of_present_averages_only_what_was_measured() -> None:
    frame = pd.DataFrame({"T1": [674], "T2": [700], "T3": [None], "T4": [None]})
    result = compute_measurements(frame, [_thickness()])
    assert result.values["thickness"][0] == pytest.approx(687.0)
    assert result.inputs_used["thickness"][0] == 2


def test_mean_of_present_is_unrounded() -> None:
    """The workbook stores ROUND(mean(T1..T4)); the model gets the mean itself."""
    frame = pd.DataFrame({"T1": [650], "T2": [655], "T3": [670], "T4": [680]})
    result = compute_measurements(frame, [_thickness()])
    assert result.values["thickness"][0] == pytest.approx(663.75)


# --------------------------------------------------------------------------- #
# "blank means not measured, never zero"
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("blank", [None, "", " ", NBSP, f" {NBSP} ", np.nan, "n/a"])
def test_every_spelling_of_empty_is_treated_as_unmeasured(blank) -> None:
    """Excel leaves non-breaking spaces in cells that look empty. `str.strip` does
    not remove one, so an unnormalised blank test would read it as data."""
    frame = pd.DataFrame({"T1": [700], "T2": [blank], "T3": [blank], "T4": [blank]})
    result = compute_measurements(frame, [_thickness()])
    assert result.values["thickness"][0] == pytest.approx(700.0)
    assert result.inputs_used["thickness"][0] == 1
    assert not result.has_errors


def test_a_blank_is_not_a_zero() -> None:
    """The failure this guards: averaging a blank as 0 halves the thickness."""
    blank = pd.DataFrame({"T1": [700], "T2": [None], "T3": [None], "T4": [None]})
    zero = pd.DataFrame({"T1": [700], "T2": [0], "T3": [None], "T4": [None]})
    assert compute_measurements(blank, [_thickness()]).values["thickness"][0] == 700.0
    assert compute_measurements(zero, [_thickness()]).values["thickness"][0] == 350.0


def test_no_measured_thickness_at_all_is_an_error_not_a_nan_average() -> None:
    frame = pd.DataFrame({"T1": [None], "T2": [None], "T3": [None], "T4": [NBSP]})
    result = compute_measurements(frame, [_thickness()])
    assert "no_inputs_measured" in _codes(result, ScoreSeverity.ERROR)
    assert math.isnan(result.values["thickness"][0])


def test_a_recipe_needing_every_input_errors_on_a_blank_one() -> None:
    frame = pd.DataFrame({"Coverage": [1.0], "Uniformity": [None], "Phase purity": [0.98]})
    result = compute_measurements(frame, [_uniformity()])
    assert "input_missing" in _codes(result, ScoreSeverity.ERROR)
    assert math.isnan(result.values["uniformity"][0])


def test_zero_photoconductance_is_an_error_with_an_actionable_message() -> None:
    """log10(0) is -inf. A failed film must be blank, not zero, and the message
    has to say so or someone will type a zero."""
    frame = pd.DataFrame({"PL": [0.0459], "PC": [0.0]})
    result = compute_measurements(frame, [_optoelectronic()])
    errors = [f for f in result.findings if f.severity is ScoreSeverity.ERROR]
    assert errors and "blank, not as zero" in errors[0].message
    assert math.isnan(result.values["optoelectronic"][0])


def test_text_in_a_measurement_cell_is_reported_not_coerced() -> None:
    frame = pd.DataFrame({"T1": ["about 700"], "T2": [None], "T3": [None], "T4": [None]})
    result = compute_measurements(frame, [_thickness()])
    assert "input_not_numeric" in _codes(result, ScoreSeverity.ERROR)


def test_a_numeric_string_is_accepted() -> None:
    frame = pd.DataFrame({"T1": ["700"], "T2": [f"710{NBSP}"], "T3": [None], "T4": [None]})
    result = compute_measurements(frame, [_thickness()])
    assert result.values["thickness"][0] == pytest.approx(705.0)


def test_one_bad_row_does_not_hide_the_others() -> None:
    frame = pd.DataFrame(
        {"T1": [700, None, 500], "T2": [None, None, None], "T3": [None] * 3, "T4": [None] * 3}
    )
    result = compute_measurements(frame, [_thickness()], sample_ids=[1, 2, 3])
    assert result.values["thickness"].tolist()[0] == 700.0
    assert math.isnan(result.values["thickness"][1])
    assert result.values["thickness"].tolist()[2] == 500.0
    assert [f.sample_id for f in result.errors] == [2]


def test_raise_for_errors_fails_closed() -> None:
    frame = pd.DataFrame({"T1": [None], "T2": [None], "T3": [None], "T4": [None]})
    result = compute_measurements(frame, [_thickness()])
    with pytest.raises(ScoreValidationError, match="cannot be computed"):
        result.raise_for_errors()


# --------------------------------------------------------------------------- #
# cross-checks
# --------------------------------------------------------------------------- #


def test_a_matching_stored_cell_says_nothing() -> None:
    frame = pd.DataFrame(
        {
            "Coverage": [1.0],
            "Uniformity": [0.33],
            "Phase purity": [0.98],
            "Uniformity score": [1.0 * 0.67 * 0.98],
        }
    )
    result = compute_measurements(frame, [_uniformity()])
    assert result.findings == ()


def test_a_stale_paste_is_caught() -> None:
    """The whole point: a literal that no longer matches its inputs."""
    frame = pd.DataFrame(
        {
            "Coverage": [1.0],
            "Uniformity": [0.33],
            "Phase purity": [0.98],
            "Uniformity score": [0.5],
        }
    )
    result = compute_measurements(frame, [_uniformity()], sample_ids=[7])
    mismatch = [f for f in result.warnings if f.code == "cross_check_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0].sample_id == 7
    assert "the model uses" in mismatch[0].message
    # and the computed value is what comes out
    assert result.values["uniformity"][0] == pytest.approx(0.6566)


def test_a_rounded_stored_cell_within_tolerance_is_accepted() -> None:
    """`Thickness (avg)` is ROUND(mean), so half a nanometre is not a mismatch."""
    frame = pd.DataFrame(
        {"T1": [650], "T2": [655], "T3": [670], "T4": [680], "Thickness (avg)": [664]}
    )
    result = compute_measurements(frame, [_thickness()])
    assert [f.code for f in result.warnings] == []


def test_a_rounded_stored_cell_beyond_tolerance_is_not() -> None:
    frame = pd.DataFrame(
        {"T1": [650], "T2": [655], "T3": [670], "T4": [680], "Thickness (avg)": [700]}
    )
    result = compute_measurements(frame, [_thickness()])
    assert "cross_check_mismatch" in _codes(result, ScoreSeverity.WARNING)


def test_an_emptied_formula_column_names_the_cause() -> None:
    """openpyxl discards cached formula values on save. If a cross-check column
    reads empty, that is the likely reason and the message should say it."""
    frame = pd.DataFrame(
        {
            "Coverage": [1.0],
            "Uniformity": [0.33],
            "Phase purity": [0.98],
            "Uniformity score": [None],
        }
    )
    result = compute_measurements(frame, [_uniformity()])
    empty = [f for f in result.warnings if f.code == "cross_check_empty"]
    assert empty and "non-Excel tool" in empty[0].message


def test_a_missing_cross_check_column_is_a_note_not_a_failure() -> None:
    """A replacement dataset may not carry the score columns at all."""
    frame = pd.DataFrame({"Coverage": [1.0], "Uniformity": [0.33], "Phase purity": [0.98]})
    result = compute_measurements(frame, [_uniformity()])
    assert "cross_check_absent" in _codes(result, ScoreSeverity.NOTE)
    assert not result.has_errors


# --------------------------------------------------------------------------- #
# excluded readings and disagreement
# --------------------------------------------------------------------------- #


def test_an_excluded_reading_is_recorded_rather_than_averaged() -> None:
    frame = pd.DataFrame(
        {"T1": [650], "T2": [655], "T3": [670], "T4": [680], "T anom": [1618]}
    )
    result = compute_measurements(frame, [_thickness()], sample_ids=[4])
    assert result.values["thickness"][0] == pytest.approx(663.75)
    notes = [f for f in result.notes if f.code == "reading_excluded"]
    assert notes and notes[0].sample_id == 4 and "1618" in notes[0].message


def test_an_empty_anomaly_column_says_nothing() -> None:
    frame = pd.DataFrame({"T1": [700], "T2": [710], "T3": [None], "T4": [None], "T anom": [NBSP]})
    result = compute_measurements(frame, [_thickness()])
    assert [f.code for f in result.notes if f.code == "reading_excluded"] == []


def test_readings_that_split_into_two_clusters_warn() -> None:
    """Sample 12's recorded 1155 nm is the midpoint of 1600 and 709. The mean is
    computed either way, but nobody should act on it without knowing."""
    frame = pd.DataFrame({"T1": [1600], "T2": [709], "T3": [None], "T4": [None]})
    result = compute_measurements(
        frame, [_thickness(spread_warning_ratio=0.25)], sample_ids=[12]
    )
    warned = [f for f in result.warnings if f.code == "readings_disagree"]
    assert warned and warned[0].sample_id == 12
    assert "1600" in warned[0].message and "709" in warned[0].message
    assert result.values["thickness"][0] == pytest.approx(1154.5)


def test_ordinary_scatter_does_not_warn() -> None:
    frame = pd.DataFrame({"T1": [751], "T2": [754], "T3": [752], "T4": [None]})
    result = compute_measurements(frame, [_thickness(spread_warning_ratio=0.25)])
    assert [f.code for f in result.warnings] == []


def test_the_spread_warning_is_off_unless_configured() -> None:
    frame = pd.DataFrame({"T1": [1600], "T2": [709], "T3": [None], "T4": [None]})
    result = compute_measurements(frame, [_thickness()])
    assert "readings_disagree" not in _codes(result, ScoreSeverity.WARNING)


# --------------------------------------------------------------------------- #
# completeness and entry columns
# --------------------------------------------------------------------------- #


def test_row_completeness_needs_every_input_for_a_product() -> None:
    frame = pd.DataFrame(
        {
            "Coverage": [1.0, 1.0],
            "Uniformity": [0.33, None],
            "Phase purity": [0.98, 0.98],
        }
    )
    assert row_completeness(frame, [_uniformity()]).tolist() == [True, False]


def test_row_completeness_needs_only_one_thickness_reading() -> None:
    """Nine of the fifteen R0 rows have two readings. Demanding all four would
    report a finished sheet as half-filled and block the next round."""
    frame = pd.DataFrame(
        {
            "T1": [674, None],
            "T2": [700, None],
            "T3": [None, None],
            "T4": [None, None],
        }
    )
    assert row_completeness(frame, [_thickness()]).tolist() == [True, False]


def test_entry_columns_split_required_from_optional() -> None:
    required, optional = entry_columns([_uniformity(), _optoelectronic(), _thickness()])
    assert required == (
        "Coverage",
        "Uniformity",
        "Phase purity",
        "PL",
        "PC",
    )
    assert optional == ("T1", "T2", "T3", "T4", "T anom")


def test_an_absent_required_column_is_refused_up_front() -> None:
    frame = pd.DataFrame({"Coverage": [1.0], "Phase purity": [0.98]})
    with pytest.raises(ValueError, match="missing from the sheet"):
        compute_measurements(frame, [_uniformity()])


def test_an_absent_optional_column_is_reported_once_not_per_row() -> None:
    frame = pd.DataFrame({"T1": [700, 800], "T2": [710, 810]})
    result = compute_measurements(frame, [_thickness()])
    absent = [f for f in result.notes if f.code == "input_column_absent"]
    assert {f.column for f in absent} == {"T3", "T4"}
    assert len(absent) == 2
    assert result.values["thickness"].tolist() == [705.0, 805.0]


# --------------------------------------------------------------------------- #
# config parsing
# --------------------------------------------------------------------------- #


def test_no_measurement_block_means_no_spec() -> None:
    assert measurement_spec_from_config({"name": "uniformity"}) is None


def test_inputs_accept_bare_strings_and_mappings() -> None:
    spec = measurement_spec_from_config(
        {
            "name": "uniformity",
            "measurement": {
                "recipe": "product",
                "inputs": ["Coverage", {"column": "Uniformity", "transform": "complement"}],
                "cross_check": "Uniformity score",
            },
        }
    )
    assert spec is not None
    assert [i.column for i in spec.inputs] == ["Coverage", "Uniformity"]
    assert [i.transform for i in spec.inputs] == ["identity", "complement"]
    assert spec.cross_checks[0].column == "Uniformity score"
    assert spec.cross_checks[0].atol == pytest.approx(0.005)


def test_a_single_cross_check_mapping_is_accepted() -> None:
    spec = measurement_spec_from_config(
        {
            "name": "thickness",
            "measurement": {
                "recipe": "mean_of_present",
                "inputs": ["T1"],
                "cross_check": {"column": "Thickness (avg)", "atol": 0.5},
            },
        }
    )
    assert spec.cross_checks == (CrossCheck("Thickness (avg)", 0.5),)


@pytest.mark.parametrize(
    "block, match",
    [
        ({"recipe": "nonsense", "inputs": ["a"]}, "unknown recipe"),
        ({"recipe": "product", "inputs": []}, "non-empty list"),
        ({"recipe": "product", "inputs": ["a", "a"]}, "repeats"),
        (
            {"recipe": "product", "inputs": [{"column": "a", "transform": "sqrt"}]},
            "Unsupported measurement transform",
        ),
        (
            {"recipe": "mean_of_present", "inputs": ["a"], "spread_warning_ratio": 0},
            "positive finite",
        ),
    ],
)
def test_a_malformed_measurement_block_is_refused(block, match) -> None:
    with pytest.raises(ValueError, match=match):
        measurement_spec_from_config({"name": "x", "measurement": block})


def test_findings_frame_is_exportable() -> None:
    frame = pd.DataFrame({"T1": [1600], "T2": [709], "T3": [None], "T4": [None]})
    result = compute_measurements(
        frame, [_thickness(spread_warning_ratio=0.25)], sample_ids=[12]
    )
    exported = result.findings_frame()
    assert list(exported.columns) == [
        "severity",
        "code",
        "objective",
        "sample_id",
        "column",
        "message",
    ]
    assert (exported["objective"] == "thickness").all()
