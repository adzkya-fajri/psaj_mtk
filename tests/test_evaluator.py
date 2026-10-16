"""
Unit tests for laborcast.evaluator module.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from laborcast.evaluator import (
    Evaluator,
    LOOCVEvaluator,
    ModelBenchmarkEvaluator,
    ResidualDiagnostics,
    TemporalSplitEvaluator,
)
from laborcast.models import (
    ExponentialSmoothingModel,
    LinearTrendModel,
    MovingAverageModel,
    NAIRUAsymptoticModel,
    PolynomialTrendModel,
)


@pytest.fixture
def sample_data():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([7.07, 6.49, 5.86, 5.32, 4.91, 4.85], dtype=float)
    years = [2020, 2021, 2022, 2023, 2024, 2025]
    return X, y, years


def test_core_metrics():
    y_true = np.array([5.0, 6.0, 7.0])
    y_pred = np.array([5.2, 5.8, 7.1])

    mae = Evaluator.mae(y_true, y_pred)
    assert pytest.approx(mae, abs=1e-4) == (0.2 + 0.2 + 0.1) / 3

    rmse = Evaluator.rmse(y_true, y_pred)
    assert rmse > mae

    r2 = Evaluator.r2(y_true, y_pred)
    assert 0.90 <= r2 <= 1.0


def test_temporal_split_evaluator(sample_data):
    X, y, years = sample_data
    evaluator = TemporalSplitEvaluator(test_size=1)
    model = LinearTrendModel()

    res = evaluator.evaluate(model, X, y, years=years)
    assert res.test_year == 2025
    assert res.test_actual == 4.85
    # Model trained on 2020-2024 predicts 2025
    assert res.test_predicted > 0
    assert res.test_absolute_error >= 0
    assert res.test_accuracy > 80.0


def test_temporal_split_evaluator_multi_step(sample_data):
    X, y, years = sample_data
    evaluator = TemporalSplitEvaluator(test_size=2)
    model = LinearTrendModel()

    res = evaluator.evaluate(model, X, y, years=years)
    assert res.test_year == 2025
    assert res.test_metrics is not None
    assert "RMSE" in res.test_metrics
    assert "MAE" in res.test_metrics
    assert res.test_absolute_error > 0.0


def test_loocv_evaluator(sample_data):
    X, y, _ = sample_data
    evaluator = LOOCVEvaluator()
    model = LinearTrendModel()

    res = evaluator.evaluate(model, X, y)
    assert len(res.predictions) == len(y)
    assert res.loocv_rmse > 0
    assert res.loocv_mae > 0
    assert res.loocv_mape > 0
    assert res.loocv_mape < 10.0  # High accuracy threshold


def test_residual_diagnostics(sample_data):
    X, y, _ = sample_data
    model = LinearTrendModel().fit(X, y)
    res = ResidualDiagnostics.analyze(model, X, y)

    assert len(res.residuals) == 6
    assert pytest.approx(res.mean_residual, abs=1e-3) == 0.0

    # Shapiro-Wilk test for normality
    assert 0.0 <= res.shapiro_stat <= 1.0
    assert res.shapiro_p > 0.05  # Normality satisfied
    assert res.shapiro_passed is True

    # Durbin-Watson statistic
    assert 0.0 <= res.durbin_watson <= 4.0

    # Summary dictionary test
    summary = res.summary()
    assert "shapiro_wilk" in summary
    assert "durbin_watson" in summary


def test_benchmark_evaluator(sample_data):
    X, y, years = sample_data
    models = {
        "OLS": LinearTrendModel(),
        "Poly2": PolynomialTrendModel(degree=2),
        "NAIRU": NAIRUAsymptoticModel(fixed_floor=3.50),
        "Holt": ExponentialSmoothingModel(),
        "MA3": MovingAverageModel(window=3),
    }

    evaluator = ModelBenchmarkEvaluator(test_size=1)
    df_bench = evaluator.compare_models(models, X, y, years=years)

    assert len(df_bench) == 5
    assert "Model Name" in df_bench.columns
    assert "LOOCV RMSE (%)" in df_bench.columns
    assert "Test APE (%)" in df_bench.columns
