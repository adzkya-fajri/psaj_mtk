"""
End-to-end integration tests for LaborCastPipeline.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from laborcast.pipeline import LaborCastPipeline


def test_pipeline_run_integration(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    assets_dir = tmp_path / "assets"

    pipeline = LaborCastPipeline(
        base_year=2020,
        nairu_floor=3.50,
        artifacts_dir=artifacts_dir,
        assets_dir=assets_dir,
    )

    result = pipeline.run(
        dataset_path="experiments/datasets/tpt_nasional_timeseries_2015_2025.csv",
        target_year=2026,
        export_visualizations=True,
        export_artifacts=True,
    )

    # 1. Check preprocessed data
    assert len(result.preprocessed_data.years) == 6
    assert result.preprocessed_data.years[-1] == 2025

    # 2. Check models & equations
    assert result.primary_model.r2_ > 0.95
    assert pytest.approx(result.primary_model.slope_, abs=1e-3) == -0.4680

    # 3. Check benchmark dataframe
    assert len(result.benchmark_df) >= 4
    assert "OLS_Linear" in result.benchmark_df["Model Key"].values

    # 4. Check residual diagnostics
    assert result.residual_diagnostics.shapiro_passed is True

    # 5. Check 2026 forecast
    fc = result.forecast_result
    assert pytest.approx(fc.point_forecast, abs=1e-2) == 4.11
    assert fc.pi_lower_95 >= 3.50
    assert fc.pi_upper_95 > fc.point_forecast

    # 6. Check files created on disk
    assert result.model_artifact_path.is_file()
    assert result.metadata_artifact_path.is_file()
    for chart_path in result.chart_paths.values():
        assert chart_path.is_file()
        assert chart_path.stat().st_size > 1000

    # 7. Check summary text generation
    summary = result.summary_text()
    assert "LABORCAST MACHINE LEARNING PIPELINE" in summary
    assert "4.11" in summary
