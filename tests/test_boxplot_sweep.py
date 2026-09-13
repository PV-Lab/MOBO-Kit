"""The beta x radius boxplot sweep.

The sweep itself is 108 full campaign runs, so nothing here runs one. What is
pinned is the bookkeeping around them, where a silent error would be expensive and
invisible: a sharding bug that drops or duplicates cells would produce a
deliverable with quietly missing panels after six hours of compute.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load():
    path = Path("scripts") / "plot_boxplot_sweep.py"
    spec = importlib.util.spec_from_file_location("_script_boxplot_sweep", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sweep = _load()


def test_the_grid_is_the_one_the_group_asked_for() -> None:
    assert sweep.BETAS == (9.0, 25.0, 36.0, 49.0)
    assert sweep.RADII == (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
    assert len(sweep.TRIALS) == 3
    assert [name for name, _s, _seed in sweep.TRIALS] == ["real", "lhs_a", "lhs_b"]


def test_the_cell_count_is_108() -> None:
    cells = sweep.all_cells()
    assert len(cells) == len(sweep.TRIALS) * len(sweep.BETAS) * len(sweep.RADII) == 108
    assert len(set(cells)) == 108, "no duplicate cells"


@pytest.mark.parametrize("num_shards", [1, 2, 5, 7, 12, 108])
def test_sharding_covers_every_cell_exactly_once(num_shards: int) -> None:
    """The property that makes a six-hour parallel run trustworthy.

    ``cells[shard::num_shards]`` must partition the grid. A stride that dropped or
    repeated cells would leave gaps the compose step draws as 'missing', or would
    burn compute recomputing the same cell in two workers.
    """
    cells = sweep.all_cells()
    seen: list[tuple[str, float, float]] = []
    for shard in range(num_shards):
        seen.extend(cells[shard::num_shards])
    assert len(seen) == len(cells)
    assert set(seen) == set(cells)
    assert len(set(seen)) == len(seen), "a cell was assigned to two shards"


def test_shards_are_balanced_within_one_cell() -> None:
    """Wall clock is the slowest shard, so an unbalanced split wastes it."""
    cells = sweep.all_cells()
    for num_shards in (8, 12, 16):
        sizes = [len(cells[shard::num_shards]) for shard in range(num_shards)]
        assert max(sizes) - min(sizes) <= 1


def test_cell_keys_are_unique_and_filesystem_safe() -> None:
    keys = [sweep.cell_key(t, b, r) for t, b, r in sweep.all_cells()]
    assert len(set(keys)) == len(keys) == 108
    for key in keys:
        assert "." not in key, "a dot would collide with the .npz suffix"
        assert all(c.isalnum() or c in "_" for c in key)


def test_cell_key_round_trips_the_parameters() -> None:
    assert sweep.cell_key("real", 9.0, 0.05) == "real__beta_9__radius_0p05"
    assert sweep.cell_key("lhs_b", 49.0, 0.45) == "lhs_b__beta_49__radius_0p45"
    # 0.10 and 0.1 must not produce two different keys for one radius
    assert sweep.cell_key("real", 25.0, 0.10) == sweep.cell_key("real", 25.0, 0.1)


def test_only_the_starting_design_differs_between_trials() -> None:
    """The comparison this sweep makes is only clean if nothing else moves.

    Trial 1 reads the workbook; trials 2 and 3 draw a Latin hypercube at distinct
    seeds. No trial carries its own acquisition seed -- that stays at the
    campaign's, so the candidate pool is identical throughout.
    """
    sources = {name: (source, seed) for name, source, seed in sweep.TRIALS}
    assert sources["real"][0] == "workbook"
    assert sources["lhs_a"] == ("lhs", 101)
    assert sources["lhs_b"] == ("lhs", 202)
    assert sources["lhs_a"][1] != sources["lhs_b"][1], "trials must differ"


def test_the_footer_states_both_hazards() -> None:
    """The two hazards that do not depend on which campaign is loaded."""
    assert "not a measurement" in sweep.FOOTER
    assert "15 / 5 / 3" in sweep.FOOTER


def test_the_no_signal_caveat_is_read_from_the_config_not_remembered() -> None:
    """It used to be hard-coded as "uniformity ... permutation p = 0.82".

    That is a fact about the FIRST campaign's uniformity score on the FIRST
    campaign's films. On the v3 contract that objective is a different
    construction and optoelectronic is dead as well, so the constant would have
    put the wrong evidence under the right warning -- which is worse than no
    caveat, because it looks checked.
    """
    from mobo_kit.campaign import load_campaign_config

    active = load_campaign_config("configs/campaign_d2d_perovskite_test.yaml")
    caveat = sweep.signal_caveat(active)
    assert "uniformity" in caveat and "optoelectronic" in caveat
    assert "d2d-objectives-v3-test" in caveat
    assert "leave-one-out null" in caveat
    # and it must not carry the previous campaign's evidence
    assert "0.82" not in caveat

    every_axis_learnable = {
        "objectives": {
            "contract_version": "synthetic",
            "specs": [{"name": "a", "signal_status": "learnable"}],
        }
    }
    assert sweep.signal_caveat(every_axis_learnable) == ""


def test_a_single_ratified_cell_keeps_all_three_trials() -> None:
    """Filtering the knobs must never drop a trial: the trials are what turn
    three numbers per round into a distribution worth boxing."""
    cells = sweep.all_cells([36.0], [0.35])
    assert len(cells) == len(sweep.TRIALS)
    assert {trial for trial, _b, _r in cells} == {t[0] for t in sweep.TRIALS}
    assert {(b, r) for _t, b, r in cells} == {(36.0, 0.35)}
    # and the unfiltered default is unchanged
    assert len(sweep.all_cells()) == len(sweep.TRIALS) * len(sweep.BETAS) * len(sweep.RADII)
