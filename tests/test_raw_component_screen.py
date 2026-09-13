"""The raw-component screen, and the two ways it could quietly mislead.

This harness exists to answer "does the model learn better from the measurement
than from the score", and it answers by ``eval``-ing a candidate expression over a
namespace of workbook columns. Two things therefore have to be pinned rather than
trusted:

* **the expression validator**, because a screen that evaluates arbitrary text is
  a bad instrument regardless of who is typing into it -- and because these
  expressions are increasingly written by agents rather than by hand;
* **the leave-one-out loop**, because it is a SECOND implementation of a fold loop
  this project has already had to consolidate once. It is not the same function
  object as ``mobo_kit.loocv.loo_predictions`` -- it takes a free-form ``y``
  rather than an objective spec -- so the identity trick used elsewhere does not
  apply and agreement has to be asserted numerically instead.

The fold loop is also where an honest screen and a flattering one diverge: the
structured mean must be refit INSIDE every fold. Fitting it once on all rows leaks
the held-out value into the mean function, which on 15 rows is worth more than any
real effect anyone has found here. That is pinned by construction below.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook

from mobo_kit.loocv import loo_predictions


def _load():
    path = Path("scripts") / "raw_component_screen.py"
    spec = importlib.util.spec_from_file_location("_script_raw_component_screen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


screen = _load()


# --------------------------------------------------------------------------- #
# the expression validator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "expr",
    [
        "coverage",
        "np.log(photocond)",
        "(coverage + phase_purity) / 2",
        "np.clip(uniformity_raw, 0, 1)",
        "phase_purity ** 2",
        "np.log(phase_purity / (1 - phase_purity))",
    ],
)
def test_ordinary_candidate_expressions_are_accepted(expr):
    screen._check_expression(expr)


@pytest.mark.parametrize(
    ("expr", "because"),
    [
        ("__import__('os').system('echo hi')", "no dunder, no import"),
        ("coverage.__class__", "attribute access outside the np namespace"),
        ("np.load('x.npy')", "np.load is not on the numeric allow-list"),
        ("open('secrets')", "open is not in the namespace"),
        ("[c for c in coverage]", "comprehensions are not expressions we screen"),
        ("thickness", "not a name in the measurement namespace"),
        ("coverage if phase_purity else 0", "no conditionals"),
    ],
)
def test_expressions_outside_the_measurement_namespace_are_refused(expr, because):
    with pytest.raises((ValueError, SyntaxError)):
        screen._check_expression(expr)


def test_the_error_names_the_namespace_rather_than_just_refusing():
    """A rejected candidate must say what IS available, or the next attempt is a guess."""
    with pytest.raises(ValueError, match="phase_purity"):
        screen._check_expression("phase_purty")


def test_evaluate_uses_only_the_supplied_columns():
    space = {"coverage": np.array([0.5, 1.0]), "phase_purity": np.array([2.0, 4.0])}
    got = screen.evaluate("coverage * phase_purity", space)
    assert got.tolist() == [1.0, 4.0]


# --------------------------------------------------------------------------- #
# the fold loop
# --------------------------------------------------------------------------- #


#: Eight inputs and twelve rows, deliberately. A three-input version of this
#: fixture made the trend test vacuous: with that much data per dimension the
#: plain GP already scored 0.9980 and a correct mean function could not show any
#: improvement. The campaign's real shape is ten inputs and fifteen rows, where
#: the GP is starved and the trend is worth a great deal -- which is the regime
#: the harness has to be right in.
_INPUT_NAMES = ("a", "b", "c", "d", "e", "f", "g", "h")


@pytest.fixture(scope="module")
def tiny_campaign():
    """Eight inputs, twelve rows, one objective with a real linear trend in `a`."""
    config = {
        "inputs": [
            {"name": name, "start": 0.0, "stop": 10.0, "step": 1.0}
            for name in _INPUT_NAMES
        ],
        "objectives": {
            "contract_version": "test",
            "scaling_mode": "fixed_affine",
            "specs": [
                {
                    "name": "y",
                    "model_source_column": "y",
                    "transform": "affine",
                    "goal": "maximize",
                    "lower_anchor": 0.0,
                    "upper_anchor": 100.0,
                }
            ],
        },
    }
    rng = np.random.default_rng(11)
    X = rng.uniform(0.0, 10.0, size=(12, len(_INPUT_NAMES)))
    y = 3.0 * X[:, 0] + rng.normal(0.0, 0.4, size=12) + 20.0
    return config, X, y


def test_loo_r2_agrees_with_the_shared_fold_loop(tiny_campaign):
    """Two implementations of leave-one-out must not be able to disagree.

    ``loocv.loo_predictions`` is canonical and takes an objective spec; this
    harness takes a bare array so that a candidate expression can be screened
    without inventing a config entry for it. They must still produce the same
    predictions, or the screen is measuring a different model from the campaign.
    """
    config, X, y = tiny_campaign
    entry = config["objectives"]["specs"][0]
    canonical = loo_predictions(config, entry, X, y, seed=73, use_mean_function=False)
    ours = screen.loo_r2(config, X, y, seed=73)
    np.testing.assert_allclose(ours["predicted"], canonical.predicted, rtol=0, atol=1e-9)
    assert ours["r2"] == pytest.approx(canonical.r2, abs=1e-9)


def test_the_mean_function_is_refit_inside_every_fold(tiny_campaign, monkeypatch):
    """The single most consequential detail, asserted by counting calls.

    If the trend were fitted once and reused, the held-out row would be inside the
    data the mean function saw, and every score this harness reports would be
    inflated. One fit per fold is the only correct count.
    """
    config, X, y = tiny_campaign
    spec = screen.StructuredMeanSpec(
        response="identity", features=(screen.MeanFeature("a"),)
    )
    calls = []
    original = screen.build_structured_mean

    def counted(X_phys, values, *args, **kwargs):
        calls.append(len(values))
        return original(X_phys, values, *args, **kwargs)

    monkeypatch.setattr(screen, "build_structured_mean", counted)
    screen.loo_r2(config, X, y, seed=73, mean_spec=spec)
    assert len(calls) == len(y), "one mean fit per fold"
    assert set(calls) == {len(y) - 1}, "each fit sees N-1 rows, never all N"


def test_a_declared_trend_helps_when_the_trend_is_real(tiny_campaign):
    """The point of a mean function, in the starved regime where it matters.

    Eight inputs and twelve rows: the plain GP cannot find the one dimension that
    matters, and declaring it is worth a large jump. If this ever stops holding,
    the mean-function path is not doing what the screen reports it as doing.
    """
    config, X, y = tiny_campaign
    spec = screen.StructuredMeanSpec(
        response="identity", features=(screen.MeanFeature("a"),)
    )
    plain = screen.loo_r2(config, X, y, seed=73)["r2"]
    structured = screen.loo_r2(config, X, y, seed=73, mean_spec=spec)["r2"]
    assert structured > plain


# --------------------------------------------------------------------------- #
# reading the workbook
# --------------------------------------------------------------------------- #


def _sheet_with(rows):
    book = Workbook()
    sheet = book.active
    sheet.title = "R0"
    sheet["A1"] = "Sample number"
    sheet["L1"] = "Coverage"
    for index, (sample, coverage) in enumerate(rows, start=2):
        sheet[f"A{index}"] = sample
        sheet[f"L{index}"] = coverage
    return book


def test_reading_stops_at_the_first_blank_sample_number(tmp_path):
    """Rows below the data block are notes, and notes are not films."""
    path = tmp_path / "book.xlsx"
    book = _sheet_with([(1, 0.9), (2, 0.8), (3, 0.7)])
    book["R0"]["A6"] = 99          # a stray row below a gap
    book["R0"]["L6"] = 0.1
    book.save(path)
    space = screen.read_measurements(path, "R0")
    assert space["coverage"].tolist() == [0.9, 0.8, 0.7]


def test_missing_cells_become_nan_rather_than_zero(tmp_path):
    """A blank measurement is unknown, not zero. Zero would be a plausible number."""
    path = tmp_path / "book.xlsx"
    book = _sheet_with([(1, 0.9), (2, None), (3, 0.7)])
    book.save(path)
    space = screen.read_measurements(path, "R0")
    assert np.isnan(space["coverage"][1])
    assert space["coverage"][[0, 2]].tolist() == [0.9, 0.7]


# --------------------------------------------------------------------------- #
# the built-in screen
# --------------------------------------------------------------------------- #


def test_every_built_in_candidate_is_a_valid_expression():
    for item in screen.BUILT_IN:
        screen._check_expression(item["expr"])


def test_the_built_in_screen_covers_both_forms_of_thickness():
    """The screen's whole argument rests on this contrast, so it must be in it.

    Raw nanometres against the stored Gaussian-squashed score, on identical films.
    If either disappears from the built-ins, the headline comparison stops being
    reproducible from a bare run of the script.
    """
    names = {item["name"] for item in screen.BUILT_IN}
    assert {"thickness_nm", "STORED_score_thickness"} <= names


def _full_workbook(path, n_rows=9):
    """A workbook carrying every column the measurement namespace names."""
    rng = np.random.default_rng(5)
    book = Workbook()
    sheet = book.active
    sheet.title = "R0"
    sheet["A1"] = "Sample number"
    for name, letter in screen.COLUMNS.items():
        sheet[f"{letter}1"] = name
    for row in range(2, n_rows + 2):
        sheet[f"A{row}"] = row - 1
        for name, letter in screen.COLUMNS.items():
            sheet[f"{letter}{row}"] = float(rng.uniform(0.2, 0.9))
    book.save(path)
    return path


def test_the_screen_prints_its_family_size_and_the_selection_warning(tmp_path, capsys):
    """Silent multiplicity is how a screen of thirty reports a discovery.

    The count of candidates screened, and the warning that the winner was chosen
    by looking at this data, are part of the OUTPUT rather than of the docstring.
    A reader who sees only the table must still see how many it was picked from.
    """
    workbook = _full_workbook(tmp_path / "full.xlsx")
    config = tmp_path / "screen.yaml"
    design = [
        "speed_1", "time_1", "speed_2", "time_2", "precur_conc",
        "precur_vol", "anneal_temp", "anneal_time", "anti_vol", "anti_time",
    ]
    lines = ["inputs:"]
    lines += [
        "  - {name: %s, start: 0.0, stop: 1.0, step: 0.05}" % name for name in design
    ]
    lines += [
        "objectives:",
        "  contract_version: screen-test",
        "  scaling_mode: fixed_affine",
        "  specs:",
        "  - name: y",
        "    model_source_column: y",
        "    transform: affine",
        "    goal: maximize",
        "    lower_anchor: 0.0",
        "    upper_anchor: 1.0",
    ]
    config.write_text("\n".join(lines), encoding="utf-8")

    screen.main([
        "--workbook", str(workbook),
        "--config", str(config),
        "--candidates", "coverage,phase_purity",
    ])
    printed = capsys.readouterr().out
    assert "screening  2 candidates" in printed
    assert "Bonferroni family size K = 2" in printed
    assert "chosen by looking at this data" in printed
    assert "null LOO R2" in printed


def test_a_totally_collapsed_run_refuses_rather_than_reporting_the_null():
    """The trap this project walked into, closed by construction.

    When every fold falls back to its training mean, the predictions ARE the
    leave-one-out mean predictor, whose R2 is exactly ``1-(n/(n-1))^2`` with
    Spearman -1. That is the number this project used as its null for a year, so
    a completely broken run would have reported an ordinary-looking no-signal
    result. It must raise instead.
    """
    config = {
        "inputs": [{"name": "a", "start": 0.0, "stop": 1.0, "step": 0.1}],
        "objectives": {
            "contract_version": "t", "scaling_mode": "fixed_affine",
            "specs": [{"name": "y", "model_source_column": "y", "transform": "affine",
                       "goal": "maximize", "lower_anchor": 0.0, "upper_anchor": 1.0}],
        },
    }
    X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    y = np.linspace(1.0, 2.0, 8)

    def always_fails(*args, **kwargs):
        raise RuntimeError("fit refused")

    import unittest.mock as mock
    with mock.patch.object(screen, "fit_model_variant", always_fails):
        with pytest.raises(RuntimeError, match="All 8 folds failed"):
            screen.loo_r2(config, X, y, seed=73)


def test_the_refusal_names_the_number_it_would_otherwise_have_printed():
    """So whoever hits it recognises the value from the project's own docs."""
    config = {
        "inputs": [{"name": "a", "start": 0.0, "stop": 1.0, "step": 0.1}],
        "objectives": {
            "contract_version": "t", "scaling_mode": "fixed_affine",
            "specs": [{"name": "y", "model_source_column": "y", "transform": "affine",
                       "goal": "maximize", "lower_anchor": 0.0, "upper_anchor": 1.0}],
        },
    }
    X = np.linspace(0.0, 1.0, 15).reshape(-1, 1)
    y = np.linspace(1.0, 2.0, 15)
    import unittest.mock as mock
    with mock.patch.object(
        screen, "fit_model_variant", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    ):
        with pytest.raises(RuntimeError, match=r"-0\.1480"):
            screen.loo_r2(config, X, y, seed=73)


def test_a_mean_feature_may_declare_a_log_transform():
    """Without this the screen cannot express the mean function the campaign runs.

    ``configs/campaign_d2d_perovskite_final.yaml`` declares
    ``log(speed_1) + log(precur_conc)`` on a log response. The screen originally
    built every feature with the default identity transform, so it silently
    measured a DIFFERENT model and then compared candidates against it as though
    it were the incumbent.
    """
    spec = screen._mean_spec(
        {"mean_features": [{"column": "speed_1", "transform": "log"}, "precur_conc"]},
        "log",
    )
    assert spec.response == "log"
    assert [(f.column, f.transform) for f in spec.features] == [
        ("speed_1", "log"),
        ("precur_conc", "identity"),
    ]


def test_no_mean_features_means_no_mean_spec():
    assert screen._mean_spec({"expr": "coverage"}, "identity") is None
    assert screen._mean_spec({"mean_features": []}, "identity") is None


def test_the_docstring_no_longer_teaches_the_wrong_bar():
    """-0.1480 must not be presented as a significance threshold anywhere here.

    It is the score of the leave-one-out mean predictor. Measured on this
    campaign, 28.7% of pure-noise shuffles beat it. The docstring has to say so,
    because the docstring is what the next person reads before quoting an R2.
    """
    doc = screen.__doc__
    assert "NOT A SIGNIFICANCE THRESHOLD" in doc
    assert "28.7%" in doc
    assert "--calibrate" in doc or "`--calibrate`" in doc
