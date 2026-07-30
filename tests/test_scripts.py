"""The two operator-facing scripts.

Neither is a test — one sweeps parameters, one is what you run when new data
arrives — but both encode commitments that should not drift silently: the sweep's
pre-committed decision rule, and the intake's floors. Those are pinned here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path("scripts")


def _load(name: str):
    """Import a script by path. Both guard their entry point with __main__, so
    importing runs no work."""
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_both_scripts_import_cleanly() -> None:
    assert _load("dtlz2_parameter_sweep") is not None
    assert _load("intake_new_data") is not None


# --------------------------------------------------------------------------- #
# the sweep's pre-committed rule
# --------------------------------------------------------------------------- #


def test_the_sweep_grid_and_current_default_are_what_was_agreed() -> None:
    sweep = _load("dtlz2_parameter_sweep")
    assert sweep.BETAS == (2.0, 4.0, 8.0)
    assert sweep.RADII == (0.15, 0.25, 0.35)
    assert sweep.CURRENT_BETA == 4.0
    assert sweep.CURRENT_RADIUS == 0.25
    # a hard floor on spacing, held fixed so "spacing" means the same thing in
    # every cell
    assert sweep.MIN_BATCH_DISTANCE == 0.15


def test_the_sweep_scores_the_added_budget_not_the_whole_campaign() -> None:
    """The comparison is BO against random at EQUAL budget: the 8 points R1 and R2
    add. Scoring the whole campaign would credit BO with the shared R0 start."""
    sweep = _load("dtlz2_parameter_sweep")
    assert sweep.ADDED == sweep.R1_SIZE + sweep.R2_SIZE == 8


# --------------------------------------------------------------------------- #
# the intake's floors
# --------------------------------------------------------------------------- #


def test_the_null_moves_with_n() -> None:
    """1 - (N/(N-1))^2, independent of the data. Reusing the N=15 value on a bigger
    dataset would hold the model to the wrong bar."""
    intake = _load("intake_new_data")
    assert intake.null_loo_r2(15) == pytest.approx(-0.1480, abs=1e-4)
    assert intake.null_loo_r2(21) == pytest.approx(-0.1025, abs=1e-4)
    assert intake.null_loo_r2(31) == pytest.approx(-0.0678, abs=1e-4)
    # it approaches zero from below as N grows, never crossing it
    assert intake.null_loo_r2(1000) < 0.0


def test_the_resolution_floor_shrinks_with_n() -> None:
    intake = _load("intake_new_data")
    assert intake.resolution_sd(15) == pytest.approx(0.236)
    assert intake.resolution_sd(60) == pytest.approx(0.118)
    assert intake.resolution_sd(15) > intake.resolution_sd(30)


def test_the_bootstrap_reference_is_the_measured_one() -> None:
    """0.236 was measured by parametric bootstrap at N=15, 4000 resamples. The
    sqrt(15/N) rescaling is an approximation and the script says so."""
    intake = _load("intake_new_data")
    assert intake.RESOLUTION_SD_AT_15 == 0.236
    assert intake.RESOLUTION_REFERENCE_N == 15
    assert "estimate" in intake.__doc__ or "approximation" in intake.__doc__
