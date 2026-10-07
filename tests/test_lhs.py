import numpy as np
import pandas as pd
import pytest

from mobo_kit.design import InputSpec, build_design
from mobo_kit.lhs import lhs_dataframe, lhs_dataframe_optimized


def _design():
    return build_design(
        [
            InputSpec("speed", 0.0, 1.0, 0.25, unit="m/min"),
            InputSpec("temperature", 20.0, 24.0, 1.0, unit="C"),
            InputSpec("time", 5.0, 15.0, 5.0, unit="s"),
        ]
    )


def _assert_exact_grid_membership(frame: pd.DataFrame, design) -> None:
    for column, grid in zip(design.names, design.var_list):
        assert np.all(np.isin(frame[column].to_numpy(), grid))


def test_seeded_lhs_is_deterministic_unique_and_exactly_sized():
    design = _design()
    kwargs = dict(
        design=design,
        n=12,
        seed=123,
        samples_per_attempt=40,
        max_attempts=5,
    )

    first = lhs_dataframe_optimized(**kwargs)
    second = lhs_dataframe_optimized(**kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (12, 3)
    assert list(first.columns) == design.names
    assert len(first.drop_duplicates()) == 12
    _assert_exact_grid_membership(first, design)


def test_different_seed_changes_at_least_one_condition():
    design = _design()
    first = lhs_dataframe_optimized(
        design=design,
        n=12,
        seed=123,
        samples_per_attempt=40,
        max_attempts=5,
    )
    second = lhs_dataframe_optimized(
        design=design,
        n=12,
        seed=124,
        samples_per_attempt=40,
        max_attempts=5,
    )

    assert not first.equals(second)


def test_compatibility_wrapper_has_the_same_strict_contract():
    design = _design()
    frame = lhs_dataframe(
        design,
        n=8,
        seed=77,
        samples_per_attempt=24,
        max_attempts=5,
    )

    assert frame.shape == (8, 3)
    assert len(frame.drop_duplicates()) == 8
    _assert_exact_grid_membership(frame, design)


def test_constraints_are_applied_after_grid_snapping():
    design = _design()
    calls = []

    def snapped_constraint(X, constrained_design):
        for column, grid in enumerate(constrained_design.var_list):
            assert np.all(np.isin(X[:, column], grid))
        calls.append(X.copy())
        speed_index = constrained_design.names.index("speed")
        return X[:, speed_index] >= 0.75

    frame = lhs_dataframe_optimized(
        design,
        n=5,
        seed=5,
        row_constraints=[snapped_constraint],
        samples_per_attempt=30,
        max_attempts=5,
    )

    assert calls
    assert frame.shape == (5, 3)
    assert np.all(frame["speed"] >= 0.75)
    _assert_exact_grid_membership(frame, design)


def test_impossible_constraint_raises_without_unconstrained_fallback():
    design = _design()

    def reject_everything(X, _design):
        return np.zeros(X.shape[0], dtype=bool)

    with pytest.raises(RuntimeError, match="Unable to generate exactly n=3"):
        lhs_dataframe_optimized(
            design,
            n=3,
            seed=9,
            row_constraints=reject_everything,
            samples_per_attempt=20,
            max_attempts=2,
        )


def test_request_larger_than_grid_raises_before_sampling():
    design = build_design([InputSpec("x", 0, 1, 1), InputSpec("y", 0, 1, 1)])

    with pytest.raises(ValueError, match="only 4 unique grid combinations"):
        lhs_dataframe_optimized(design, n=5, seed=1)


def test_max_abs_corr_is_a_hard_requirement():
    design = build_design([InputSpec("x", 0, 1, 1), InputSpec("y", 0, 1, 1)])

    # A two-row Latin design on two binary dimensions varies in both columns;
    # their absolute Pearson correlation is necessarily 1.
    with pytest.raises(RuntimeError, match=r"required <= 0\.500000"):
        lhs_dataframe_optimized(
            design,
            n=2,
            seed=42,
            max_abs_corr=0.5,
            samples_per_attempt=2,
            batch_size=2,
            subset_tries=10,
            max_attempts=1,
        )


def test_returned_design_satisfies_configured_correlation_limit():
    design = build_design([InputSpec("x", 0, 1, 1), InputSpec("y", 0, 1, 1)])

    frame = lhs_dataframe_optimized(
        design,
        n=4,
        seed=3,
        max_abs_corr=0.0,
        samples_per_attempt=4,
        max_attempts=10,
    )

    correlation = np.corrcoef(frame.to_numpy(), rowvar=False)[0, 1]
    assert abs(correlation) <= 0.0


def test_continuous_unsnapped_output_is_rejected_for_campaign_use():
    with pytest.raises(ValueError, match="requires snap_to_grids=True"):
        lhs_dataframe_optimized(_design(), n=3, seed=1, snap_to_grids=False)
