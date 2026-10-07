import numpy as np
import pytest

from mobo_kit.design import (
    InputSpec,
    build_design,
    build_design_from_config,
    build_input_spec_list,
    make_linspace,
)


def _valid_inputs():
    return [
        {
            "name": "speed",
            "unit": "rpm",
            "start": 1000,
            "stop": 2000,
            "step": 250,
        },
        {
            "name": "time",
            "unit": "s",
            "start": 5,
            "stop": 20,
            "step": 5,
        },
    ]


def test_make_linspace_preserves_requested_step_and_endpoints():
    grid = make_linspace(0.0, 1.0, 0.2)

    assert np.array_equal(grid, np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    assert np.allclose(np.diff(grid), 0.2)


def test_make_linspace_allows_a_single_fixed_value():
    assert np.array_equal(make_linspace(3.5, 3.5, 0.25), np.array([3.5]))


@pytest.mark.parametrize(
    ("start", "stop", "step", "message"),
    [
        (0.0, 1.0, 0.3, "not aligned"),
        (1.0, 0.0, 0.1, "stop must be"),
        (0.0, 1.0, 0.0, "step must be > 0"),
        (0.0, 1.0, -0.1, "step must be > 0"),
        (0.0, np.inf, 0.1, "must be finite"),
    ],
)
def test_make_linspace_rejects_ambiguous_grids(start, stop, step, message):
    with pytest.raises(ValueError, match=message):
        make_linspace(start, stop, step)


def test_build_design_from_config_constructs_exact_grids():
    design = build_design_from_config({"inputs": _valid_inputs()})

    assert design.names == ["speed", "time"]
    assert design.units == ["rpm", "s"]
    assert np.array_equal(
        design.var_list[0], np.array([1000.0, 1250.0, 1500.0, 1750.0, 2000.0])
    )
    assert np.array_equal(design.var_list[1], np.array([5.0, 10.0, 15.0, 20.0]))
    assert np.array_equal(design.lowers, np.array([1000.0, 5.0]))
    assert np.array_equal(design.uppers, np.array([2000.0, 20.0]))


@pytest.mark.parametrize(
    ("inputs", "message"),
    [
        ([], "non-empty list"),
        ([{"name": "", "start": 0, "stop": 1, "step": 1}], "non-empty string"),
        ([{"name": "x", "start": 0, "stop": 1}], "missing required"),
        ([{"name": "x", "start": "bad", "stop": 1, "step": 1}], "finite number"),
        ([{"name": "x", "start": 0, "stop": np.nan, "step": 1}], "must be finite"),
        ([{"name": "x", "start": 1, "stop": 0, "step": 1}], "stop >= start"),
        ([{"name": "x", "start": 0, "stop": 1, "step": -1}], "step.*> 0"),
        ([{"name": "x", "start": 0, "stop": 1, "step": 0.3}], "not aligned"),
        (
            [
                {"name": " x ", "start": 0, "stop": 1, "step": 1},
                {"name": "x", "start": 0, "stop": 1, "step": 1},
            ],
            "duplicate name",
        ),
    ],
)
def test_input_schema_validation_is_explicit(inputs, message):
    with pytest.raises(ValueError, match=message):
        build_input_spec_list(inputs)


def test_rounding_that_collapses_grid_points_is_rejected():
    with pytest.raises(ValueError, match="duplicate values"):
        InputSpec(name="x", start=0.0, stop=0.02, step=0.005, decimals=2)


def test_build_design_revalidates_mutated_specs_and_unique_names():
    first = InputSpec("x", 0, 1, 1)
    second = InputSpec("y", 0, 1, 1)
    second.name = "x"

    with pytest.raises(ValueError, match="duplicate name"):
        build_design([first, second])


def test_build_design_requires_input_specs():
    with pytest.raises(ValueError, match="At least one"):
        build_design([])
    with pytest.raises(TypeError, match="InputSpec"):
        build_design([{"name": "x"}])
