"""The v4 contract: frozen scores, a per-round source sheet, and what freezing costs.

Three objective contracts have existed and this is the live one. What is new here
is a deliberate reversal of this project's usual polarity: uniformity and
optoelectronic are READ from the workbook rather than recomputed, because the
group is still revising how they are defined.

That reversal removes a cross-check, and the tests below are mostly about the
consequences of removing it:

* a frozen score must still be *validated* -- numeric, present, inside its
  declared anchors -- because nothing else looks at it;
* a frozen score's DEFINITION must be watched, since its value cannot be;
* and the one thing freezing cannot see -- a stale literal that has stopped
  tracking its inputs -- is asserted to be exactly what the fingerprint does
  **not** catch, so nobody later mistakes the fingerprint for a value check.

The synthetic workbook here uses the v4 layout, so none of it needs the ignored
private one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from mobo_kit.campaign import load_campaign_config, objective_names
from mobo_kit.scores import (
    FormulaFingerprint,
    MeasurementInput,
    MeasurementSpec,
    ScoreSeverity,
    compute_measurements,
    entry_columns,
    measurement_spec_from_config,
)
from mobo_kit.workbook_io import (
    CandidateSheetError,
    formula_findings,
    read_campaign_workbook,
    source_sheet,
)

CONFIG_PATH = "configs/campaign_d2d_perovskite_final.yaml"
V3_CONFIG_PATH = "configs/campaign_d2d_perovskite_test.yaml"
SOURCE = "local_inputs/Final Summary Table.xlsx"

UNIFORMITY_COLUMN = "Uniformity score (Avg (Coverage + (1-Uniformity) + Phase purity))"
OPTO_COLUMN = (
    "Optoelectronic score (Normalized (Voc + (0.75*Photoconductance + "
    "0.25*Photosensitivity))/2"
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_campaign_config(CONFIG_PATH)


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #


def test_three_contracts_exist_and_only_one_is_active(config) -> None:
    """Naming them is not bookkeeping. An objective that keeps its name while
    changing its construction makes every cross-contract number incomparable
    while every plot still renders.

    The live contract became ``-nomean`` on 2026-09-06, when the thickness mean
    function was withdrawn. That is a MODEL change rather than an objective
    redefinition -- the three quantities are unchanged -- but it moves every
    fitted number on the learnable axis (+0.7423 to +0.5814) and therefore every
    hypervolume, so it earns a version. This assertion failing is this test doing
    its job; update it deliberately, never to make a run go green.
    """
    v3 = load_campaign_config(V3_CONFIG_PATH)
    archived = load_campaign_config("configs/campaign_d2d_perovskite.yaml")

    assert config["objectives"]["contract_version"] == "d2d-objectives-v4-final-nomean"
    assert v3["objectives"]["contract_version"] == "d2d-objectives-v3-test"
    assert archived["objectives"]["contract_version"] == "d2d-objectives-v2-nm-thickness"

    assert config["campaign"]["status"] == "active"
    assert v3["campaign"]["status"] == "archived"
    assert archived["campaign"]["status"] == "archived"


def test_no_objective_declares_a_mean_function(config) -> None:
    """The live campaign carries NO physics prior, by decision on 2026-09-06.

    The thickness prior ``log T ~ log(speed_1) + log(precur_conc)`` was withdrawn
    after its justification failed: the fitted speed exponent's 95% interval is
    [-0.385, -0.126], which excludes spin-coating theory's -0.5 by 4.1 standard
    errors, and fixing the exponents at their theoretical values scores +0.5600
    against +0.5823 for no trend at all.

    ``structured_mean`` stays wired and tested for a prior that clears the bar --
    established physics, declared before fitting, beating matched-flexibility
    controls, and surviving a permutation test. Nothing currently does. If this
    test fails, someone has added one; make them show the four pieces of evidence
    before updating it.
    """
    from mobo_kit.structured_mean import mean_spec_from_config

    declared = {
        entry["name"]: mean_spec_from_config(entry)
        for entry in config["objectives"]["specs"]
    }
    assert declared == {name: None for name in declared}, (
        f"a mean function reappeared: "
        f"{ {k: v for k, v in declared.items() if v is not None} }"
    )


def test_measured_replicate_noise_is_on_and_in_the_model_s_units(config) -> None:
    """Option C, switched on 2026-09-10 when the R1 triplicates came back. It is
    three settings, not one: the noise mode; a replicate rule that matches every
    objective's model space (thickness trains in nm, so `mean`); and a floor in
    the same units as the variance it guards, the within-film variance of a film
    MEAN in nm^2. Getting the second wrong handed the model a log-space variance
    as nm^2; getting the third wrong fired a false alarm on the first real data."""
    from mobo_kit.campaign import replicate_aggregates
    from mobo_kit.replicate_variance import REPLICATE_POOLED, variance_config

    assert config["model"]["observation_noise"] == REPLICATE_POOLED
    rules = dict(zip(objective_names(config), replicate_aggregates(config)))
    assert rules == {"uniformity": "mean", "optoelectronic": "mean", "thickness": "mean"}
    assert variance_config(config)["sanity_floor"]["thickness"] == pytest.approx(91.2)


def test_r2_observations_carry_nanometre_noise_and_no_floor_alarm(config) -> None:
    """On the real triplicates: 20 observations, thickness noise in nm^2 (a single
    R0 film takes the pooled 662.7, an R1 mean of three a third of it), and no
    floor warning -- the false alarm the old per-reading log floor raised here."""
    from pathlib import Path

    from mobo_kit.launcher import gather_observations

    source = Path(SOURCE)
    r1 = source.with_name(f"{source.stem}_R1_Candidates.xlsx")
    if not (source.exists() and r1.exists()):
        pytest.skip("needs the private R0 workbook and its filled R1 worklist")

    X, Y, Yvar, provenance = gather_observations(source, config, for_round="R2")
    assert X.shape == (20, 10)
    assert Yvar.shape == (20, 3)
    column = list(objective_names(config)).index("thickness")
    assert Yvar[0, column] == pytest.approx(662.7, abs=0.1)
    assert Yvar[-1, column] == pytest.approx(662.7 / 3, abs=0.1)
    assert not any(item.startswith("WARNING") for item in provenance)


def test_the_launcher_defaults_to_the_live_contract() -> None:
    """The one path an experimentalist reaches by double-clicking. Archiving a
    config without moving this is how a user once got a missing-column error on
    an intact workbook."""
    from mobo_kit.launcher import DEFAULT_CONFIG

    assert DEFAULT_CONFIG == CONFIG_PATH
    assert load_campaign_config(DEFAULT_CONFIG)["campaign"]["status"] == "active"


def test_the_source_sheet_is_configuration_not_a_constant(config) -> None:
    """The v4 workbook names its sheets by round, so `Sheet1` stopped being true.
    Older contracts must keep working without declaring the key."""
    assert source_sheet(config) == "R0"
    assert source_sheet(load_campaign_config(V3_CONFIG_PATH)) == "Sheet1"
    assert source_sheet({}) == "Sheet1"


def test_both_score_objectives_are_frozen_and_thickness_is_not(config) -> None:
    """Thickness stays computed: its definition has been stable across all three
    contracts, and the recomputation is what lets `T anom` be excluded and
    reported rather than silently dropped."""
    recipes = {
        spec["name"]: spec["measurement"]["recipe"]
        for spec in config["objectives"]["specs"]
    }
    assert recipes == {
        "uniformity": "stored",
        "optoelectronic": "stored",
        "thickness": "mean_of_present",
    }


# --------------------------------------------------------------------------- #
# the stored recipe
# --------------------------------------------------------------------------- #


def _stored_spec(**kwargs) -> MeasurementSpec:
    return MeasurementSpec(
        name="uniformity",
        recipe="stored",
        inputs=(MeasurementInput("Uniformity score"),),
        **kwargs,
    )


def test_a_stored_score_is_taken_exactly_as_the_workbook_computed_it() -> None:
    frame = pd.DataFrame({"Uniformity score": [0.877272, 0.599033]})
    result = compute_measurements(frame, [_stored_spec()], sample_ids=[1, 4])
    assert result.values["uniformity"].tolist() == [0.877272, 0.599033]
    assert not result.has_errors


def test_a_blank_frozen_score_is_an_error_not_a_gap() -> None:
    """`mean_of_present` tolerates a missing reading because a film can carry
    three instead of four. A missing SCORE is different: nothing can recompute it
    under this contract, so the row simply has no objective value."""
    frame = pd.DataFrame({"Uniformity score": [0.87, None]})
    result = compute_measurements(frame, [_stored_spec()], sample_ids=[1, 2])
    codes = [f.code for f in result.findings if f.severity is ScoreSeverity.ERROR]
    assert "input_missing" in codes
    assert result.values["uniformity"].tolist()[0] == 0.87
    assert np.isnan(result.values["uniformity"].tolist()[1])


def test_a_formula_cell_with_no_cached_value_reads_as_blank_and_errors() -> None:
    """openpyxl discards cached formula values on save, so a workbook written by
    a non-Excel tool hands back None for every formula column. Under a freeze that
    is every objective at once, and it must stop the round rather than train on
    nothing."""
    frame = pd.DataFrame({"Uniformity score": [None, None, None]})
    result = compute_measurements(frame, [_stored_spec()], sample_ids=[1, 2, 3])
    assert result.has_errors
    assert all(np.isnan(v) for v in result.values["uniformity"])


def test_a_non_numeric_frozen_score_is_an_error() -> None:
    frame = pd.DataFrame({"Uniformity score": ["n/a", "not a number"]})
    result = compute_measurements(frame, [_stored_spec()], sample_ids=[1, 2])
    codes = [f.code for f in result.findings if f.severity is ScoreSeverity.ERROR]
    assert codes, "a score column full of text must not pass silently"


def test_stored_takes_exactly_one_column() -> None:
    """Two columns would mean something is being combined, which is precisely what
    a freeze exists to avoid."""
    with pytest.raises(ValueError, match="one score column"):
        MeasurementSpec(
            name="uniformity",
            recipe="stored",
            inputs=(MeasurementInput("a"), MeasurementInput("b")),
        )


def test_the_v3_recipes_survive_unwired_for_when_the_group_unfreezes() -> None:
    """`mean`, `clamped_complement` and `capped_ratio` are not deleted. The freeze
    is temporary by the group's own description, and deleting the code would mean
    rebuilding it from a doc rather than un-commenting it."""
    from mobo_kit.scores import RECIPES

    assert {"stored", "mean", "mean_of_present", "product", "log10_product"} <= set(RECIPES)
    spec = measurement_spec_from_config(
        {
            "name": "uniformity",
            "measurement": {
                "recipe": "mean",
                "inputs": [
                    {"column": "Coverage"},
                    {
                        "column": "Uniformity",
                        "transform": "clamped_complement",
                        "clamp_above": 1.0,
                        "clamp_to": 0.99,
                    },
                    {"column": "Phase purity"},
                ],
            },
        }
    )
    frame = pd.DataFrame(
        {"Coverage": [0.989], "Uniformity": [0.324584], "Phase purity": [0.9674]}
    )
    result = compute_measurements(frame, [spec], sample_ids=[1])
    # the v4 workbook's own AJ for sample 1
    assert result.values["uniformity"][0] == pytest.approx(0.877272, abs=1e-6)


# --------------------------------------------------------------------------- #
# the fingerprint: what replaces the cross-check, and what it cannot replace
# --------------------------------------------------------------------------- #


def _workbook_with(tmp_path, formula, *, column=UNIFORMITY_COLUMN, rows=3):
    path = tmp_path / "Fingerprint.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "R0"
    sheet.append(["Sample number", "Coverage", column])
    for i in range(rows):
        value = formula.replace("2", str(i + 2)) if formula else 0.5
        sheet.append([i + 1, 0.9, value])
    book.save(path)
    return path


def _fingerprint_config(formula="=(L2+O2+P2)/3", column=UNIFORMITY_COLUMN) -> dict:
    return {
        "campaign": {"source_sheet": "R0"},
        "inputs": [{"name": "x", "start": 0, "stop": 1, "step": 1}],
        "objectives": {
            "contract_version": "synthetic",
            "specs": [
                {
                    "name": "uniformity",
                    "model_source_column": column,
                    "transform": "affine",
                    "goal": "maximize",
                    "lower_anchor": 0.0,
                    "upper_anchor": 1.0,
                    "measurement": {
                        "recipe": "stored",
                        "inputs": [{"column": column}],
                        "formula_fingerprint": {"column": column, "formula": formula},
                    },
                }
            ],
        },
    }


def test_an_unchanged_definition_is_a_note(tmp_path) -> None:
    path = _workbook_with(tmp_path, "=(L2+O2+P2)/3")
    (finding,) = formula_findings(path, _fingerprint_config())
    assert finding.code == "formula_fingerprint_unchanged"
    assert finding.severity is ScoreSeverity.NOTE


def test_a_changed_definition_is_a_warning_that_names_both_formulas(tmp_path) -> None:
    """The value is still read and still used -- the change is not an error. But
    every number computed under the old definition is about a different quantity,
    so it has to be audible."""
    path = _workbook_with(tmp_path, "=(L2+O2+P2+Q2)/4")
    (finding,) = formula_findings(path, _fingerprint_config())
    assert finding.code == "formula_fingerprint_changed"
    assert finding.severity is ScoreSeverity.WARNING
    assert "(L2+O2+P2)/3" in finding.message
    assert "contract_version" in finding.message


def test_the_same_formula_copied_down_a_column_is_not_a_change(tmp_path) -> None:
    """Fingerprinting per row would report fifteen changes for one edit."""
    path = _workbook_with(tmp_path, "=(L2+O2+P2)/3", rows=5)
    (finding,) = formula_findings(path, _fingerprint_config())
    assert finding.code == "formula_fingerprint_unchanged"
    assert "5 rows" in finding.message


def test_a_pasted_literal_score_is_flagged_as_uncheckable(tmp_path) -> None:
    """The one failure this contract cannot see, called out rather than left
    silent: a literal cannot be checked against anything at all."""
    path = _workbook_with(tmp_path, None)
    (finding,) = formula_findings(path, _fingerprint_config())
    assert finding.code == "fingerprint_no_formula"
    assert finding.severity is ScoreSeverity.WARNING


def test_the_fingerprint_cannot_catch_a_stale_value(tmp_path) -> None:
    """Asserted deliberately, so nobody later mistakes the fingerprint for a value
    check. A formula whose inputs have changed still matches its own text; only a
    recomputation would notice, and a freeze is the decision not to have one."""
    path = _workbook_with(tmp_path, "=(L2+O2+P2)/3")
    (finding,) = formula_findings(path, _fingerprint_config())
    assert finding.severity is ScoreSeverity.NOTE, (
        "the definition is unchanged, so the fingerprint is silent -- whatever the "
        "values behind it have done"
    )


def test_no_fingerprint_declared_means_no_second_workbook_read(tmp_path) -> None:
    """The check needs data_only=False, a second full read. Configs that do not
    freeze anything must not pay for it."""
    config = _fingerprint_config()
    del config["objectives"]["specs"][0]["measurement"]["formula_fingerprint"]
    assert formula_findings(tmp_path / "does-not-exist.xlsx", config) == ()


def test_the_agreement_check_columns_are_offered_even_when_neither_is_an_input() -> None:
    """This broke when optoelectronic was frozen: the check listed only its `raw`
    column, on the assumption that `normalized` was a recipe input. Under a freeze
    the only input is the score column, so the check reported "column absent" on a
    sheet that had it."""
    from mobo_kit.scores import AgreementCheck

    spec = MeasurementSpec(
        name="optoelectronic",
        recipe="stored",
        inputs=(MeasurementInput("Optoelectronic score"),),
        agreement_check=AgreementCheck(raw="Photoconductance", normalized="Normalized"),
    )
    required, optional = entry_columns([spec])
    assert "Photoconductance" in optional
    assert "Normalized" in optional


# --------------------------------------------------------------------------- #
# the wrong workbook
# --------------------------------------------------------------------------- #


def test_a_workbook_without_the_configured_sheet_says_which_sheet(tmp_path, config) -> None:
    path = tmp_path / "Wrong.xlsx"
    book = Workbook()
    book.active.title = "Sheet1"
    book.active.append(["Sample number"])
    book.save(path)
    with pytest.raises(CandidateSheetError) as caught:
        read_campaign_workbook(path, config)
    message = str(caught.value)
    assert "'R0'" in message and "campaign.source_sheet" in message
    assert "Sheet1" in message


# --------------------------------------------------------------------------- #
# the real workbook
# --------------------------------------------------------------------------- #

requires_workbook = pytest.mark.skipif(
    not __import__("pathlib").Path(SOURCE).is_file(),
    reason=f"{SOURCE} is not present in this checkout",
)


@pytest.mark.local_input
@requires_workbook
def test_the_final_workbook_reads_clean(config) -> None:
    contents = read_campaign_workbook(SOURCE, config)
    assert contents.n_rows == 15
    assert contents.errors == ()
    assert contents.warnings == ()
    values = contents.model_values
    assert list(values.columns) == list(objective_names(config))
    # the frozen scores are the sheet's own numbers, not a recomputation
    stored = contents.workbook_values
    for name in ("uniformity", "optoelectronic"):
        column = [c for c in stored.columns if c.lower().startswith(name[:6])][0]
        np.testing.assert_allclose(
            values[name].to_numpy(float), stored[column].to_numpy(float), atol=0.0
        )


@pytest.mark.local_input
@requires_workbook
def test_the_recorded_fingerprints_match_the_final_workbook(config) -> None:
    findings = formula_findings(SOURCE, config)
    assert len(findings) == 2
    assert {f.code for f in findings} == {"formula_fingerprint_unchanged"}


@pytest.mark.local_input
@requires_workbook
def test_the_photoconductance_inversion_is_fixed(config) -> None:
    """The v3 contract's optoelectronic axis was provisional because its
    normalised column ranked BACKWARDS against its own raw measurement (Spearman
    -0.5484). The group's fix landed; this pins that it did."""
    contents = read_campaign_workbook(SOURCE, config)
    finding = next(f for f in contents.findings if f.code.startswith("agreement_"))
    assert finding.code == "agreement_monotonic"
    assert "+1.0000" in finding.message
