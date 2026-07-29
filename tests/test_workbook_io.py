from __future__ import annotations

import shutil

import pandas as pd
import pytest
from openpyxl import load_workbook

from mobo_kit.campaign import load_campaign_config, model_source_columns
from mobo_kit.workbook_io import (
    candidate_workbook_path,
    CandidateSheetError,
    detect_round,
    read_campaign_workbook,
    sheet_name_for_round,
    write_candidate_sheet,
)

CONFIG_PATH = "configs/FA0.9CS0.1PbI3_260407_Config.yaml"
SOURCE = "local_inputs/Summary Table.xlsx"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(SOURCE).exists(),
    reason="requires the ignored private campaign workbook",
)


@pytest.fixture(scope="module")
def config() -> dict:
    return load_campaign_config(CONFIG_PATH)


@pytest.fixture
def workbook(tmp_path):
    destination = tmp_path / "Summary Table.xlsx"
    shutil.copy2(SOURCE, destination)
    return destination


def _conditions(config: dict, n: int = 5) -> pd.DataFrame:
    names = [item["name"] for item in config["inputs"]]
    rows = [
        [float(item["start"]) + i * float(item["step"]) for item in config["inputs"]]
        for i in range(n)
    ]
    return pd.DataFrame(rows, columns=names)


def test_reads_inputs_and_the_declared_model_columns(workbook, config) -> None:
    contents = read_campaign_workbook(workbook, config)
    assert contents.n_rows == 15
    assert list(contents.inputs.columns) == [i["name"] for i in config["inputs"]]
    assert list(contents.model_values.columns) == list(model_source_columns(config))
    # thickness must arrive in nanometres, not as its score
    assert contents.model_values["Thickness (avg)"].max() > 100.0


def test_stops_at_the_first_blank_sample_number(workbook, config) -> None:
    """Rows below the data block are notes, not observations."""
    contents = read_campaign_workbook(workbook, config)
    assert contents.sample_ids == tuple(range(1, 16))


def test_candidate_sheet_has_an_entry_column_per_model_source(workbook, config) -> None:
    """The reader contract changed when thickness moved to nanometres: the sheet
    needs an nm entry column or R2 has nothing to train on."""
    write_candidate_sheet(workbook, config, _conditions(config), round_name="R1")
    sheet = load_workbook(candidate_workbook_path(workbook, "R1"))[
        sheet_name_for_round("R1")
    ]
    headers = [c.value for c in sheet[1]]
    for column in model_source_columns(config):
        assert column in headers
    assert "Thickness (avg)" in headers


def test_writing_leaves_the_source_byte_identical(workbook, config) -> None:
    """openpyxl discards cached formula values on save, and Uniformity score is a
    formula column. Writing beside the workbook makes that impossible rather than
    merely unlikely."""
    import hashlib

    before = hashlib.sha256(workbook.read_bytes()).hexdigest()
    out = write_candidate_sheet(workbook, config, _conditions(config), round_name="R1")
    assert out != workbook
    assert out.exists()
    assert hashlib.sha256(workbook.read_bytes()).hexdigest() == before


def test_formula_columns_survive_because_the_source_is_not_rewritten(
    workbook, config
) -> None:
    """The regression this design exists to prevent."""
    write_candidate_sheet(workbook, config, _conditions(config), round_name="R1")
    contents = read_campaign_workbook(workbook, config)
    # Uniformity score is =L*N*O; a rewritten workbook reads it as NaN
    assert contents.model_values["Uniformity score"].notna().all()
    assert contents.model_values["Uniformity score"].max() > 0.0


def test_three_replicate_rows_per_condition(workbook, config) -> None:
    write_candidate_sheet(
        workbook, config, _conditions(config, 5), round_name="R1", replicates=3
    )
    sheet = load_workbook(candidate_workbook_path(workbook, "R1"))[
        sheet_name_for_round("R1")
    ]
    rows = [r for r in sheet.iter_rows(min_row=2, values_only=True) if r[0]]
    assert len(rows) == 15
    assert len({r[1] for r in rows}) == 5


def test_refuses_to_overwrite_an_existing_sheet(workbook, config) -> None:
    write_candidate_sheet(workbook, config, _conditions(config), round_name="R1")
    with pytest.raises(CandidateSheetError, match="already exists"):
        write_candidate_sheet(
            workbook, config, _conditions(config), round_name="R1", make_backup=False
        )


def test_round_detection_walks_r1_then_r2(workbook, config) -> None:
    assert detect_round(workbook, config).next_round == "R1"
    write_candidate_sheet(workbook, config, _conditions(config), round_name="R1")
    state = detect_round(workbook, config)
    assert state.next_round is None
    assert "no results have been entered" in state.reason


def test_partially_scored_sheet_is_refused_with_a_readable_message(
    workbook, config
) -> None:
    """Fail closed: guessing at a half-filled sheet is how a round gets built on
    data the experimentalist had not finished entering."""
    out = write_candidate_sheet(workbook, config, _conditions(config), round_name="R1")
    book = load_workbook(out)
    sheet = book[sheet_name_for_round("R1")]
    headers = [c.value for c in sheet[1]]
    for column in model_source_columns(config):
        sheet.cell(row=2, column=headers.index(column) + 1).value = 1.0
    book.save(out)

    state = detect_round(workbook, config)
    assert state.next_round is None
    assert state.scored_rows == 1 and state.total_rows == 15
    assert "partly filled in" in state.reason
    assert "1 of 15" in state.reason


def test_missing_source_sheet_is_a_plain_sentence(tmp_path, config) -> None:
    from openpyxl import Workbook

    path = tmp_path / "wrong.xlsx"
    book = Workbook()
    book.active.title = "Renamed"
    book.save(path)
    with pytest.raises(CandidateSheetError, match="rename it back"):
        read_campaign_workbook(path, config)
