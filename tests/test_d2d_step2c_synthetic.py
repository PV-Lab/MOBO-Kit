from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from mobo_kit.d2d_campaign import OffGridObservedException, load_d2d_workbook_frame
from mobo_kit.d2d_step2c_config import load_step2c_config
from mobo_kit.d2d_step2c_synthetic import (
    run_synthetic_step2c_fast_ci,
    write_sanitized_step2c_workbook,
)
from mobo_kit.step2c_artifacts import validate_step2c_artifact_bundle


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "d2d_step2c_debug.yaml"
OUTPUT_ROOT = REPOSITORY_ROOT / "local_outputs" / "d2d_step2c_robustness"


def _proof(path: Path) -> tuple[str, int, int]:
    return (
        hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        path.stat().st_mtime_ns,
        path.stat().st_size,
    )


def test_sanitized_workbook_uses_real_v3_read_only_adapter(tmp_path: Path) -> None:
    config = load_step2c_config(CONFIG_PATH)
    workbook = write_sanitized_step2c_workbook(
        tmp_path / "sanitized_step2c.xlsx", config
    )
    before = _proof(workbook)
    inherited = config.off_grid_exceptions[0]
    dimension = tuple(config.design.names).index(inherited.input_name)
    grid = np.asarray(config.design.var_array[dimension], dtype=float)
    synthetic_exception = OffGridObservedException(
        sample_id=inherited.sample_id,
        input_name=inherited.input_name,
        observed_value=float(inherited.observed_value),
        reason="sanitized synthetic off-grid fixture",
    )

    frame, audit = load_d2d_workbook_frame(
        workbook,
        expected_sample_ids=config.expected_sample_ids,
        allowed_input_exceptions=(synthetic_exception,),
    )

    assert frame["Sample number"].tolist() == list(config.expected_sample_ids)
    assert audit.profile == config.base.workbook_profile
    assert audit.active_sheet == config.base.workbook_sheet
    assert audit.used_range == config.base.expected_content_range
    assert audit.input_rows_valid is True
    assert before == _proof(workbook)
    observed = float(
        frame.loc[
            frame["Sample number"].eq(synthetic_exception.sample_id),
            synthetic_exception.input_name,
        ].iloc[0]
    )
    assert not np.any(np.isclose(observed, grid, rtol=0.0, atol=1.0e-12))
    assert observed == inherited.observed_value


def test_synthetic_fast_ci_runs_full_orchestration_and_atomic_publication() -> None:
    run_name = f"pytest_synthetic_ci_{uuid4().hex}"
    output = OUTPUT_ROOT / run_name
    source = OUTPUT_ROOT / "synthetic_ci_sources" / f"{run_name}.xlsx"
    resolved_source = (
        OUTPUT_ROOT / "synthetic_ci_sources" / f"{run_name}_resolved_config.yaml"
    )
    archive = output.with_suffix(".zip")
    try:
        result = run_synthetic_step2c_fast_ci(
            CONFIG_PATH,
            output,
            create_portable_zip=True,
            nested_unique_sizes=(32, 64, 128, 256),
            anchors_per_selection_step=1,
            omitted_sample_ids=None,
            mc_comparison_samples=64,
        )
        validated = validate_step2c_artifact_bundle(
            output, repository_root=REPOSITORY_ROOT
        )

        assert result.mode == "fast"
        assert result.consensus.passed is False
        assert result.run_manifest["input_data_kind"] == "sanitized_synthetic_ci"
        assert result.run_manifest["workbook_writeback_performed"] is False
        assert result.run_manifest["real_r2_proposal_generated"] is False
        assert validated.workbook_path == source.resolve()
        assert validated.artifact_sha256 == result.artifact_hashes
        assert archive.is_file()
        assert source.is_file()
        assert resolved_source.is_file()
    finally:
        if output.is_dir():
            shutil.rmtree(output)
        if archive.is_file():
            archive.unlink()
        if source.is_file():
            source.unlink()
        if resolved_source.is_file():
            resolved_source.unlink()
        source_parent = source.parent
        if source_parent.is_dir() and not any(source_parent.iterdir()):
            source_parent.rmdir()
