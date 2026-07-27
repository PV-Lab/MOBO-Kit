from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPOSITORY_ROOT / "examples" / "d2d_step2a_synthetic.py"


def _load_example():
    spec = spec_from_file_location("d2d_step2a_synthetic", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_cpu_end_to_end_writes_only_to_tmp_path(tmp_path: Path):
    module = _load_example()
    repository_output = REPOSITORY_ROOT / "local_outputs"
    before_repository_outputs = (
        set(repository_output.glob("**/*")) if repository_output.exists() else set()
    )
    summary = module.run_synthetic_step2a(
        tmp_path,
        ucb_pool_size=80,
        qlognehvi_pool_size=64,
        posterior_samples=16,
        qlognehvi_samples=8,
    )
    assert summary.observed_count == 15
    assert summary.ucb_selected_pool_indices.shape == (5,)
    assert summary.qlognehvi_selected_pool_indices.shape == (3,)
    assert np.unique(summary.ucb_selected_pool_indices).size == 5
    assert np.unique(summary.qlognehvi_selected_pool_indices).size == 3
    assert summary.ucb_minimum_distance >= 0.2 - 1e-12
    assert summary.qlognehvi_minimum_distance >= 0.2 - 1e-12
    assert len(summary.plot_paths) == 8
    assert summary.ucb_metadata["method"] == "ucb_hvi"
    assert summary.qlognehvi_metadata["method"] == "qlognehvi"
    assert all(
        path.is_file() and path.parent == tmp_path for path in summary.plot_paths
    )
    report_path = tmp_path / "synthetic_summary.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["production_candidate_generation"] == "NOT_RUN"
    assert report["seeds"] == {
        "global": module.TEST_ONLY_SEED,
        "observed_pool": module.TEST_ONLY_SEED,
        "ucb_pool": module.TEST_ONLY_SEED + 1,
        "ucb_posterior": module.TEST_ONLY_SEED + 2,
        "qlognehvi_pool": module.TEST_ONLY_SEED + 3,
        "qlognehvi_mc": module.TEST_ONLY_SEED + 4,
    }
    ucb_report = report["ucb_hvi"]
    assert ucb_report["metadata"]["beta"] == 1.0
    assert ucb_report["metadata"]["kappa"] == 1.0
    assert ucb_report["metadata"]["pool_draws"] >= ucb_report["pool_size"]
    assert len(ucb_report["selection_steps"]) == 5
    assert len(ucb_report["selected_utility_diagnostics"]) == 5
    assert all(
        len(row[key]) == 3
        for row in ucb_report["selected_utility_diagnostics"]
        for key in ("utility_mean", "utility_std", "utility_ucb")
    )
    qlog_report = report["qlognehvi"]
    assert qlog_report["metadata"]["mc_samples"] == 8
    assert qlog_report["metadata"]["pool_draws"] >= qlog_report["pool_size"]
    assert qlog_report["pending_counts_by_selection_step"] == [5, 6, 7]
    assert len(qlog_report["selection_steps"]) == 3
    assert len(report["plots"]) == 8
    assert report["runtime"]["device"] == "cpu"
    after_repository_outputs = (
        set(repository_output.glob("**/*")) if repository_output.exists() else set()
    )
    assert after_repository_outputs == before_repository_outputs
