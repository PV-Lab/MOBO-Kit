import math

import numpy as np
import pandas as pd
import pytest

from mobo_kit.d2d_scores import (
    D2DScoreTolerances,
    D2DScoreValidationError,
    OPTOELECTRONIC_CHECK_COLUMN,
    THICKNESS_CHECK_COLUMN,
    compute_optoelectronic_score,
    compute_thickness_average,
    compute_thickness_score,
    compute_uniformity_score,
    raise_for_errors,
    validate_supplied_d2d_scores,
)


def _valid_frame() -> pd.DataFrame:
    implied_voc = 0.9
    photoconductance = 1000.0
    optoelectronic = compute_optoelectronic_score(implied_voc, photoconductance)
    thicknesses = (600.0, 650.0, 700.0, None)
    thickness = compute_thickness_score(thicknesses)
    return pd.DataFrame(
        {
            "Sample number": [101],
            "Coverage": [0.8],
            "1 - Uniformity": [0.5],
            "Phase purity": [0.75],
            "PL - Implied Voc (Max)": [implied_voc],
            "Photoconductance (Max)": [photoconductance],
            OPTOELECTRONIC_CHECK_COLUMN: [optoelectronic],
            "T1": [thicknesses[0]],
            "T2": [thicknesses[1]],
            "T3": [thicknesses[2]],
            "T4": [thicknesses[3]],
            "T anom": [1_000_000.0],
            "Thickness (avg)": [-123.0],
            THICKNESS_CHECK_COLUMN: [thickness],
            "Uniformity score": [0.3],
            "Optoelectronic score": [optoelectronic],
            "Thickness score": [thickness],
        },
        index=pd.Index(["synthetic-row"], name="source_row"),
    )


def test_uniformity_support_equation_and_finite_validation():
    assert compute_uniformity_score(0.8, 0.5, 0.75) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="finite"):
        compute_uniformity_score(np.nan, 0.5, 0.75)
    with pytest.raises(TypeError, match="non-boolean"):
        compute_uniformity_score(True, 0.5, 0.75)


def test_optoelectronic_equation_and_extreme_products():
    assert compute_optoelectronic_score(0.9, 1000.0) == pytest.approx(
        math.log10(0.9 * 1000.0)
    )
    # Algebraic evaluation remains finite even when the intermediate product
    # would overflow double precision.
    assert compute_optoelectronic_score(1e300, 1e300) == pytest.approx(600.0)


@pytest.mark.parametrize(
    "implied_voc, photoconductance",
    [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)],
)
def test_optoelectronic_rejects_nonpositive_components(implied_voc, photoconductance):
    with pytest.raises(ValueError, match="strictly positive"):
        compute_optoelectronic_score(implied_voc, photoconductance)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_optoelectronic_rejects_nonfinite_components(invalid):
    with pytest.raises(ValueError, match="finite"):
        compute_optoelectronic_score(invalid, 1.0)


def test_thickness_average_ignores_all_documented_blank_forms():
    assert compute_thickness_average(600.0, None, "   \t", "\u00a0\u00a0") == 600.0
    assert compute_thickness_average(600.0, np.nan, "", 700.0) == 650.0
    with pytest.raises(ValueError, match="At least one valid"):
        compute_thickness_average(None, "", "  ", "\u00a0")


def test_thickness_average_rejects_nonblank_nonnumeric_measurement():
    with pytest.raises(TypeError, match="T2"):
        compute_thickness_average(600.0, "not measured", None, None)


def test_thickness_score_uses_unrounded_mean_and_no_half_factor():
    values = (600.4, 600.4, 600.4, 601.6)
    unrounded_mean = 600.7
    assert compute_thickness_average(*values) == pytest.approx(unrounded_mean)
    expected = math.exp(-(((unrounded_mean - 650.0) / 250.0) ** 2))
    assert compute_thickness_score(values) == pytest.approx(expected)

    one_scale_from_target = compute_thickness_score((900.0, None, None, None))
    assert one_scale_from_target == pytest.approx(math.exp(-1.0))
    assert one_scale_from_target != pytest.approx(math.exp(-0.5))
    assert compute_thickness_score(values, 650.0, 250.0) == pytest.approx(expected)


def test_thickness_api_accepts_exactly_t1_through_t4_and_excludes_t_anom():
    expected = compute_thickness_score((600.0, 650.0, 700.0, None))
    assert expected == pytest.approx(1.0)
    with pytest.raises(ValueError, match="exactly T1:T4"):
        compute_thickness_score((600.0, 650.0, 700.0, None, 1_000_000.0))
    with pytest.raises(ValueError, match="strictly positive"):
        compute_thickness_score((650.0, None, None, None), scale=0.0)


def test_clean_validation_is_exportable_and_does_not_mutate_authoritative_scores():
    frame = _valid_frame()
    original = frame.copy(deep=True)

    result = validate_supplied_d2d_scores(frame)

    assert result.has_errors is False
    assert result.warnings == ()
    assert result.raise_for_errors() is result
    assert raise_for_errors(result) is result
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.uniformity_score_authoritative == pytest.approx(0.3)
    assert row.uniformity_score_calculated == pytest.approx(0.3)
    assert row.thickness_average_calculated == pytest.approx(650.0)
    assert row.thickness_score_calculated == pytest.approx(1.0)
    assert row.optoelectronic_check_matches is True
    assert row.optoelectronic_objective_matches is True
    assert row.thickness_check_matches is True
    assert row.thickness_objective_matches is True
    assert result.frame.loc[0, "row_label"] == "synthetic-row"
    assert result.findings_frame().empty
    pd.testing.assert_frame_equal(frame, original)


def test_uniformity_mismatch_warns_and_keeps_supplied_score_authoritative():
    frame = _valid_frame()
    frame.loc["synthetic-row", "Uniformity score"] = 0.91
    original = frame.copy(deep=True)

    result = validate_supplied_d2d_scores(frame)

    assert result.has_errors is False
    assert result.known_uniformity_score_mismatch is True
    assert result.uniformity_warning_count == 1
    assert [finding.code for finding in result.warnings] == ["uniformity_mismatch"]
    assert result.rows[0].uniformity_score_authoritative == pytest.approx(0.91)
    assert result.rows[0].uniformity_score_calculated == pytest.approx(0.3)
    assert result.rows[0].uniformity_matches is False
    result.raise_for_errors()
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("invalid_score", [-0.01, 1.01, 1.5])
def test_authoritative_uniformity_score_must_remain_in_unit_interval(
    invalid_score,
):
    frame = _valid_frame()
    frame.loc["synthetic-row", "Uniformity score"] = invalid_score

    result = validate_supplied_d2d_scores(frame)

    assert "uniformity_score_out_of_range" in {
        finding.code for finding in result.errors
    }
    with pytest.raises(D2DScoreValidationError, match=r"must remain in \[0, 1\]"):
        result.raise_for_errors()


@pytest.mark.parametrize(
    "column, code",
    [
        (OPTOELECTRONIC_CHECK_COLUMN, "optoelectronic_check_mismatch"),
        ("Optoelectronic score", "optoelectronic_objective_mismatch"),
        (THICKNESS_CHECK_COLUMN, "thickness_check_mismatch"),
        ("Thickness score", "thickness_objective_mismatch"),
    ],
)
def test_unexpected_optoelectronic_and_thickness_mismatches_are_errors(column, code):
    frame = _valid_frame()
    frame.loc["synthetic-row", column] = -20.0

    result = validate_supplied_d2d_scores(frame)

    assert code in [finding.code for finding in result.errors]
    with pytest.raises(D2DScoreValidationError, match="score validation failed"):
        result.raise_for_errors()


@pytest.mark.parametrize(
    "column, invalid",
    [
        ("Uniformity score", None),
        ("Optoelectronic score", np.nan),
        ("Thickness score", np.inf),
    ],
)
def test_missing_or_nonfinite_authoritative_scores_are_errors(column, invalid):
    frame = _valid_frame()
    frame.loc["synthetic-row", column] = invalid

    result = validate_supplied_d2d_scores(frame)

    assert result.has_errors
    assert any(
        finding.column == column
        and finding.code
        in {"authoritative_score_missing", "authoritative_score_invalid"}
        for finding in result.errors
    )


def test_invalid_raw_optoelectronic_and_empty_thickness_are_errors():
    frame = _valid_frame()
    frame = frame.astype({column: object for column in ("T1", "T2", "T3", "T4")})
    frame.loc["synthetic-row", "PL - Implied Voc (Max)"] = 0.0
    frame.loc["synthetic-row", ["T1", "T2", "T3", "T4"]] = [
        None,
        "",
        " ",
        "\u00a0",
    ]

    result = validate_supplied_d2d_scores(frame)
    codes = {finding.code for finding in result.errors}

    assert "optoelectronic_support_invalid" in codes
    assert "thickness_support_invalid" in codes


def test_optional_check_fields_are_compared_only_when_available():
    frame = _valid_frame().drop(
        columns=[OPTOELECTRONIC_CHECK_COLUMN, THICKNESS_CHECK_COLUMN]
    )

    result = validate_supplied_d2d_scores(frame)

    assert result.has_errors is False
    assert result.rows[0].optoelectronic_check_matches is None
    assert result.rows[0].thickness_check_matches is None


def test_raw_t1_t4_override_neither_t_anom_nor_displayed_rounded_average():
    frame = _valid_frame()
    values = (600.4, 600.4, 600.4, 601.6)
    score = compute_thickness_score(values)
    frame.loc["synthetic-row", ["T1", "T2", "T3", "T4"]] = values
    frame.loc["synthetic-row", "T anom"] = 650.0
    frame.loc["synthetic-row", "Thickness (avg)"] = 650.0
    frame.loc["synthetic-row", THICKNESS_CHECK_COLUMN] = score
    frame.loc["synthetic-row", "Thickness score"] = score

    result = validate_supplied_d2d_scores(frame).raise_for_errors()

    assert result.rows[0].thickness_average_calculated == pytest.approx(600.7)
    assert result.rows[0].thickness_score_calculated == pytest.approx(score)


def test_default_tolerance_is_two_decimal_rounding_aware_and_configurable():
    frame = _valid_frame()
    for column in (OPTOELECTRONIC_CHECK_COLUMN, "Optoelectronic score"):
        frame.loc["synthetic-row", column] = round(
            frame.loc["synthetic-row", column], 2
        )
    for column in (THICKNESS_CHECK_COLUMN, "Thickness score"):
        frame.loc["synthetic-row", column] = round(
            frame.loc["synthetic-row", column], 2
        )

    assert validate_supplied_d2d_scores(frame).has_errors is False
    strict = D2DScoreTolerances(
        uniformity_atol=0.0,
        optoelectronic_atol=0.0,
        thickness_atol=0.0,
        rounding_slack=0.0,
    )
    strict_result = validate_supplied_d2d_scores(frame, strict)
    assert "optoelectronic_check_mismatch" in {
        finding.code for finding in strict_result.errors
    }


def test_tolerance_and_dataframe_contract_validation():
    with pytest.raises(ValueError, match="non-negative"):
        D2DScoreTolerances(thickness_atol=-1.0)
    with pytest.raises(TypeError, match="D2DScoreTolerances"):
        validate_supplied_d2d_scores(_valid_frame(), {"thickness_atol": 0.1})
    with pytest.raises(ValueError, match="Missing required"):
        validate_supplied_d2d_scores(_valid_frame().drop(columns=["T4"]))

    duplicate = pd.concat([_valid_frame(), _valid_frame()[["Thickness score"]]], axis=1)
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        validate_supplied_d2d_scores(duplicate)
