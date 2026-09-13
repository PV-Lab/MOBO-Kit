from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from mobo_kit.utils import (
    ParsedCampaignCSV,
    csv_to_config,
    load_csv,
    parse_campaign_csv,
    split_XY,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CAMPAIGN_CSV = REPOSITORY_ROOT / "data" / "processed" / "configCSV_example.csv"


def _campaign_text(*, blank_separator: bool = True) -> str:
    separator_header = "," if blank_separator else ""
    separator_cell = "," if blank_separator else ""
    blank_row = ",,,,,\n" if blank_separator else ""
    return (
        f",x_speed,x_time{separator_header},yield,stability\n"
        f"units,rpm,s{separator_cell},%,h\n"
        f"start,1000,5{separator_cell},,\n"
        f"stop,2000,15{separator_cell},,\n"
        f"step,500,5{separator_cell},,\n"
        f"{blank_row}"
        f",1000,5{separator_cell},1.5,10\n"
        f",1500,10{separator_cell},2.5,11\n"
    )


def _write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    path.write_bytes(text.encode(encoding))
    return path


def test_parser_detects_metadata_and_preserves_first_experiment(tmp_path: Path) -> None:
    path = _write_text(tmp_path / "campaign.csv", _campaign_text())

    parsed = parse_campaign_csv(path)

    assert isinstance(parsed, ParsedCampaignCSV)
    assert parsed.input_columns == ["x_speed", "x_time"]
    assert parsed.objective_columns == ["yield", "stability"]
    assert parsed.metadata_row_count == 5
    assert parsed.duplicate_headers == {}
    assert parsed.config["constraints"] == []
    assert parsed.data.iloc[0].to_dict() == {
        "x_speed": "1000",
        "x_time": "5",
        "yield": "1.5",
        "stability": "10",
    }
    assert len(parsed.data) == 2


def test_repository_example_retains_its_first_experiment() -> None:
    parsed = parse_campaign_csv(EXAMPLE_CAMPAIGN_CSV)

    assert parsed.metadata_row_count == 5
    assert parsed.input_columns == [
        "speed_inorg",
        "speed_org",
        "inkfl_inorg",
        "inkfl_org",
        "conc_inorg",
        "conc_org",
        "temperature_c",
        "absolute_humidity",
    ]
    assert parsed.objective_columns == ["PCE", "Stability", "Repeatability"]
    assert len(parsed.data) == 12
    assert parsed.data.iloc[0]["speed_inorg"] == "0.58"


def test_parser_accepts_no_blank_separator_row_or_column(tmp_path: Path) -> None:
    path = _write_text(
        tmp_path / "no_separator.csv", _campaign_text(blank_separator=False)
    )

    parsed = parse_campaign_csv(path)

    assert parsed.metadata_row_count == 4
    assert parsed.objective_columns == ["yield", "stability"]
    assert parsed.data.iloc[0]["x_speed"] == "1000"


def test_load_csv_returns_only_experimental_rows(tmp_path: Path) -> None:
    path = _write_text(tmp_path / "campaign.csv", _campaign_text())

    data = load_csv(path)

    assert list(data.columns) == ["x_speed", "x_time", "yield", "stability"]
    assert len(data) == 2
    assert "units" not in data.astype(str).to_numpy()


def test_utf8_bom_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + _campaign_text().encode("utf-8"))

    parsed = parse_campaign_csv(path)

    assert parsed.encoding == "utf-8-sig"
    assert parsed.input_columns[0] == "x_speed"


def test_cp1252_is_supported(tmp_path: Path) -> None:
    text = _campaign_text().replace("rpm", "\N{DEGREE SIGN}C")
    path = _write_text(tmp_path / "cp1252.csv", text, encoding="cp1252")

    parsed = parse_campaign_csv(path)

    assert parsed.encoding == "cp1252"
    assert parsed.config["inputs"][0]["unit"] == "\N{DEGREE SIGN}C"


def test_latin1_is_used_when_cp1252_cannot_decode(tmp_path: Path) -> None:
    payload = _campaign_text().replace("rpm", "UNIT_MARKER").encode("ascii")
    path = tmp_path / "latin1.csv"
    path.write_bytes(payload.replace(b"UNIT_MARKER", b"\x81"))

    parsed = parse_campaign_csv(path)

    assert parsed.encoding == "latin-1"
    assert parsed.config["inputs"][0]["unit"] == "\x81"


def test_duplicate_headers_are_rejected_before_pandas_mangling(tmp_path: Path) -> None:
    text = _campaign_text().replace(",yield,stability", ",yield,yield", 1)
    path = _write_text(tmp_path / "duplicates.csv", text)

    with pytest.raises(ValueError, match=r"Duplicate CSV headers.*columns \[5, 6\]"):
        parse_campaign_csv(path)


def test_plain_data_csv_is_rejected_explicitly(tmp_path: Path) -> None:
    path = _write_text(tmp_path / "plain.csv", "x,y\n1,2\n")

    with pytest.raises(ValueError, match="Plain data CSVs are not supported"):
        parse_campaign_csv(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("start,1000,5", "start,not-a-number,5", "Malformed start metadata"),
        ("step,500,5", "step,,5", "Incomplete numeric metadata"),
    ],
)
def test_malformed_or_missing_numeric_metadata_is_rejected(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _write_text(
        tmp_path / "bad_metadata.csv", _campaign_text().replace(old, new)
    )

    with pytest.raises(ValueError, match=message):
        parse_campaign_csv(path)


def test_expected_objectives_are_validated_and_ordered(tmp_path: Path) -> None:
    path = _write_text(tmp_path / "campaign.csv", _campaign_text())

    parsed = parse_campaign_csv(path, expected_objectives=["stability", "yield"])
    assert parsed.objective_columns == ["stability", "yield"]

    with pytest.raises(ValueError, match="Missing objective columns.*missing_score"):
        parse_campaign_csv(path, expected_objectives=["yield", "missing_score"])


def test_missing_objectives_and_empty_experiment_section_are_clear(
    tmp_path: Path,
) -> None:
    no_objective = ",x\nunits,rpm\nstart,0\nstop,1\nstep,0.5\n\n,0\n"
    path = _write_text(tmp_path / "no_objective.csv", no_objective)
    with pytest.raises(ValueError, match="no named objective columns"):
        parse_campaign_csv(path)

    empty = (
        ",x_speed,x_time,,yield,stability\n"
        "units,rpm,s,,%,h\n"
        "start,1000,5,,,\n"
        "stop,2000,15,,,\n"
        "step,500,5,,,\n"
        ",,,,,\n"
    )
    empty_path = _write_text(tmp_path / "empty.csv", empty)
    with pytest.raises(ValueError, match="empty experimental section"):
        parse_campaign_csv(empty_path)


def test_csv_to_config_is_opt_in_for_output_and_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_text(tmp_path / "campaign.csv", _campaign_text())
    monkeypatch.chdir(tmp_path)

    config = csv_to_config(path)

    assert config["constraints"] == []
    assert not (tmp_path / "configs").exists()

    output_path = tmp_path / "generated" / "campaign.yaml"
    written = csv_to_config(path, output_path)
    assert yaml.safe_load(output_path.read_text(encoding="utf-8")) == written


def _design() -> SimpleNamespace:
    return SimpleNamespace(names=["x_speed", "x_time"])


def _model_config() -> dict:
    return {"objectives": {"names": ["yield", "stability"]}}


def test_split_xy_returns_named_numeric_dataframes() -> None:
    data = pd.DataFrame(
        {
            "x_speed": ["1000", "1500"],
            "x_time": ["5", "10"],
            "yield": ["1.5", "2.5"],
            "stability": ["10", "11"],
        },
        index=["sample-a", "sample-b"],
    )

    X, Y = split_XY(data, _design(), _model_config())

    assert isinstance(X, pd.DataFrame)
    assert isinstance(Y, pd.DataFrame)
    assert X.columns.tolist() == ["x_speed", "x_time"]
    assert Y.columns.tolist() == ["yield", "stability"]
    assert X.index.tolist() == ["sample-a", "sample-b"]
    assert X.dtypes.tolist() == ["float64", "float64"]
    assert Y.iloc[0].tolist() == [1.5, 10.0]


def test_split_xy_rejects_missing_columns() -> None:
    data = pd.DataFrame({"x_speed": [1], "yield": [2], "stability": [3]})

    with pytest.raises(KeyError, match="missing inputs.*x_time"):
        split_XY(data, _design(), _model_config())


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            pd.DataFrame(
                {
                    "x_speed": [1000],
                    "x_time": [5],
                    "yield": [pd.NA],
                    "stability": [pd.NA],
                }
            ),
            "All objective values are blank",
        ),
        (
            pd.DataFrame(
                {
                    "x_speed": [1000, 1500],
                    "x_time": [5, 10],
                    "yield": [1.5, pd.NA],
                    "stability": [10, pd.NA],
                }
            ),
            "Objective values are blank for rows",
        ),
        (
            pd.DataFrame(
                {
                    "x_speed": [1000],
                    "x_time": [5],
                    "yield": [1.5],
                    "stability": [pd.NA],
                }
            ),
            "Partially completed objective rows",
        ),
        (
            pd.DataFrame(
                {
                    "x_speed": ["invalid"],
                    "x_time": [5],
                    "yield": [1.5],
                    "stability": [10],
                }
            ),
            "Input model data contains",
        ),
        (
            pd.DataFrame(
                {
                    "x_speed": [1000],
                    "x_time": [5],
                    "yield": ["invalid"],
                    "stability": [10],
                }
            ),
            "Objective model data contains",
        ),
    ],
)
def test_split_xy_rejects_incomplete_or_nonnumeric_model_rows(
    data: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        split_XY(data, _design(), _model_config())
