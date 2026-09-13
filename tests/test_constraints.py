"""The campaign's physical-space constraints.

These are the first real constraints this project has carried -- the first
campaign declared an empty list deliberately -- so the boundary cases are pinned
here rather than left to the round tests, where a constraint bug would show up as
a batch that merely looks unusual.
"""

from __future__ import annotations

import numpy as np
import pytest

from mobo_kit.constraints import (
    NamedConstraint,
    apply_row_constraints,
    constraint_violations,
    constraints_from_config,
)
from mobo_kit.design import build_design_from_config

CONFIG_INPUTS = [
    {"name": "speed_2", "start": 0, "stop": 5000, "step": 500},
    {"name": "time_1", "start": 5, "stop": 50, "step": 5},
    {"name": "time_2", "start": 0, "stop": 60, "step": 5},
    {"name": "anti_time", "start": 9, "stop": 25, "step": 1},
]


def _design():
    return build_design_from_config({"inputs": CONFIG_INPUTS})


def _row(speed_2=1000.0, time_1=30.0, time_2=20.0, anti_time=12.0):
    return [speed_2, time_1, time_2, anti_time]


def _mask(config_entries, rows):
    design = _design()
    constraints = constraints_from_config({"constraints": config_entries}, design)
    return apply_row_constraints(np.asarray(rows, dtype=float), design, constraints)


# --------------------------------------------------------------------------- #
# zero_coupled
# --------------------------------------------------------------------------- #

ZERO_COUPLED = [{"zero_coupled": ["speed_2", "time_2"]}]


def test_both_zero_is_valid_because_a_one_step_film_is_a_real_recipe() -> None:
    """Sample 2 of the v3 workbook runs no second stage at all. A plain lower
    bound on either column would delete that recipe, which is why the rule is an
    iff and not two bounds."""
    assert _mask(ZERO_COUPLED, [_row(speed_2=0.0, time_2=0.0)]).tolist() == [True]


def test_both_nonzero_is_valid() -> None:
    assert _mask(ZERO_COUPLED, [_row(speed_2=3500.0, time_2=30.0)]).tolist() == [True]


@pytest.mark.parametrize(
    "speed_2, time_2, what",
    [
        (0.0, 30.0, "a second stage that spins at 0 rpm for 30 s"),
        (3500.0, 0.0, "a second stage that spins at 3500 rpm for 0 s"),
    ],
)
def test_exactly_one_zero_is_invalid_in_both_orientations(
    speed_2, time_2, what
) -> None:
    """Both orientations, because a constraint written as a single implication
    catches only one of them and the other stays proposable."""
    assert _mask(ZERO_COUPLED, [_row(speed_2=speed_2, time_2=time_2)]).tolist() == [
        False
    ], what


# --------------------------------------------------------------------------- #
# sum_upper_strict
# --------------------------------------------------------------------------- #

SUM_STRICT = [{"sum_upper_strict": {"lhs": "anti_time", "rhs": ["time_1", "time_2"]}}]


def test_anti_time_below_the_total_spin_is_valid() -> None:
    rows = [_row(time_1=30.0, time_2=20.0, anti_time=49.0)]
    assert _mask(SUM_STRICT, rows).tolist() == [True]


def test_equality_is_a_violation_because_the_bound_is_strict() -> None:
    """The antisolvent has to land while the substrate is still spinning, so
    dropping it exactly at the end is already too late. Strictness is the whole
    point of this constraint type existing separately from a bounds check."""
    rows = [_row(time_1=30.0, time_2=20.0, anti_time=50.0)]
    assert _mask(SUM_STRICT, rows).tolist() == [False]


def test_anti_time_above_the_total_spin_is_a_violation() -> None:
    rows = [_row(time_1=10.0, time_2=10.0, anti_time=25.0)]
    assert _mask(SUM_STRICT, rows).tolist() == [False]


def test_a_one_step_film_still_has_to_satisfy_the_sum() -> None:
    """time_2 = 0 does not exempt the row; the sum is just time_1."""
    valid = _row(speed_2=0.0, time_2=0.0, time_1=30.0, anti_time=25.0)
    invalid = _row(speed_2=0.0, time_2=0.0, time_1=20.0, anti_time=25.0)
    assert _mask(SUM_STRICT, [valid, invalid]).tolist() == [True, False]


# --------------------------------------------------------------------------- #
# nonzero_minimum
# --------------------------------------------------------------------------- #

NONZERO_MIN = [{"nonzero_minimum": {"column": "time_2", "minimum": 10}}]


@pytest.mark.parametrize(
    "time_2, expected",
    [(0.0, True), (5.0, False), (10.0, True), (55.0, True)],
)
def test_the_declared_hole_in_the_grid(time_2, expected) -> None:
    """0 is allowed and 10 upwards is allowed; only the gap between them is not.

    The grid is arithmetic, so reaching 0 with step 5 also reaches 5. This is what
    keeps 5 out without widening the design space in silence.
    """
    assert _mask(NONZERO_MIN, [_row(time_2=time_2)]).tolist() == [expected]


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #


def test_constraints_are_anded_together() -> None:
    entries = ZERO_COUPLED + SUM_STRICT + NONZERO_MIN
    rows = [
        _row(speed_2=1000.0, time_2=20.0, time_1=30.0, anti_time=12.0),  # all pass
        _row(speed_2=1000.0, time_2=5.0, time_1=30.0, anti_time=12.0),  # minimum
        _row(speed_2=0.0, time_2=20.0, time_1=30.0, anti_time=12.0),  # coupling
        _row(speed_2=1000.0, time_2=20.0, time_1=30.0, anti_time=50.0),  # sum
    ]
    assert _mask(entries, rows).tolist() == [True, False, False, False]


def test_violations_are_reported_by_name_not_by_index() -> None:
    """A reviewer told "condition 3 is invalid" cannot act on it. The rule can."""
    design = _design()
    entries = [
        {"zero_coupled": ["speed_2", "time_2"], "name": "second_stage_all_or_nothing"},
        {"sum_upper_strict": {"lhs": "anti_time", "rhs": ["time_1", "time_2"]}},
    ]
    constraints = constraints_from_config({"constraints": entries}, design)
    rows = np.asarray(
        [
            _row(speed_2=1000.0, time_2=20.0, time_1=30.0, anti_time=12.0),
            _row(speed_2=0.0, time_2=20.0, time_1=30.0, anti_time=99.0),
        ],
        dtype=float,
    )
    assert constraint_violations(rows, design, constraints) == [
        [],
        ["second_stage_all_or_nothing", "sum_upper_strict"],
    ]


def test_a_named_constraint_is_still_an_ordinary_row_constraint() -> None:
    """The candidate pool takes plain callables and must stay unaware of naming."""
    design = _design()
    constraint = constraints_from_config({"constraints": ZERO_COUPLED}, design)[0]
    assert isinstance(constraint, NamedConstraint)
    rows = np.asarray([_row(speed_2=0.0, time_2=0.0)], dtype=float)
    assert constraint(rows, design).tolist() == [True]


def test_the_description_states_the_rule() -> None:
    design = _design()
    entries = ZERO_COUPLED + SUM_STRICT + NONZERO_MIN
    descriptions = [
        item.description for item in constraints_from_config({"constraints": entries}, design)
    ]
    assert descriptions == [
        "speed_2 and time_2 are all zero or all nonzero",
        "anti_time < time_1 + time_2",
        "time_2 is 0 or at least 10",
    ]


def test_no_constraints_means_every_row_passes() -> None:
    """Constraints must be inert when unconfigured -- DTLZ2 declares none."""
    design = _design()
    assert constraints_from_config({}, design) == []
    assert constraints_from_config({"constraints": []}, design) == []
    rows = np.asarray([_row(speed_2=0.0, time_2=30.0, anti_time=99.0)], dtype=float)
    assert apply_row_constraints(rows, design, []).tolist() == [True]
    assert constraint_violations(rows, design, None) == [[]]


# --------------------------------------------------------------------------- #
# configuration errors fail loudly
# --------------------------------------------------------------------------- #


def test_an_unknown_column_names_itself() -> None:
    design = _design()
    with pytest.raises(KeyError, match="speed_9"):
        constraints_from_config(
            {"constraints": [{"zero_coupled": ["speed_9", "time_2"]}]}, design
        )


def test_an_unknown_constraint_type_is_refused_rather_than_ignored() -> None:
    design = _design()
    with pytest.raises(KeyError, match="no supported type"):
        constraints_from_config({"constraints": [{"nonsense": [1, 2]}]}, design)


def test_zero_coupled_needs_two_columns_to_couple() -> None:
    design = _design()
    with pytest.raises(KeyError, match="at least two"):
        constraints_from_config({"constraints": [{"zero_coupled": ["time_2"]}]}, design)


@pytest.mark.parametrize(
    "entry, match",
    [
        ({"sum_upper_strict": {"rhs": ["time_1"]}}, "lhs"),
        ({"sum_upper_strict": {"lhs": "anti_time"}}, "rhs"),
        ({"nonzero_minimum": {"minimum": 10}}, "column"),
        ({"nonzero_minimum": {"column": "time_2"}}, "minimum"),
    ],
)
def test_a_half_specified_constraint_is_an_error(entry, match) -> None:
    """Silently skipping a malformed entry would leave everyone believing a rule
    is enforced while nothing enforces it."""
    with pytest.raises(KeyError, match=match):
        constraints_from_config({"constraints": [entry]}, _design())


def test_a_nonpositive_minimum_is_refused() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        constraints_from_config(
            {"constraints": [{"nonzero_minimum": {"column": "time_2", "minimum": 0}}]},
            _design(),
        )
