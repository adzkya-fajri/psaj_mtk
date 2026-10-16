"""
Unit tests for laborcast.models module.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
    return X, y


def test_linear_trend_model(sample_data):
    X, y = sample_data
    model = LinearTrendModel()
    model.fit(X, y)

    # Check fitted parameters
    assert pytest.approx(model.slope_, abs=1e-3) == -0.4680
    assert pytest.approx(model.intercept_, abs=1e-3) == 6.9200
    assert model.r2_ > 0.95

    # Check calendar parameters
    cal_int, cal_slope = model.calendar_params(base_year=2020)
    assert pytest.approx(cal_int, abs=1e-1) == 952.28
    assert pytest.approx(cal_slope, abs=1e-3) == -0.4680

    # Test 2026 forecast (X = 6)
    pred_2026 = model.predict(np.array([[6]]))[0]
    assert pytest.approx(pred_2026, abs=1e-3) == 4.1120

    # Test prediction interval
    y_pred, se_pred, moe, (pi_low, pi_high) = model.predict_interval(np.array([[6]]))
    assert se_pred[0] > 0
    assert moe[0] > 0
    assert pi_low[0] < y_pred[0] < pi_high[0]
    assert pytest.approx(pi_low[0], abs=0.01) == 3.33
    assert pytest.approx(pi_high[0], abs=0.01) == 4.89


def test_polynomial_trend_model(sample_data):
    X, y = sample_data
    model = PolynomialTrendModel(degree=2)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(y)
    assert model.coeffs_ is not None
    assert len(model.coeffs_) == 3

    # Prediction for 2026 (X = 6)
    pred_2026 = model.predict(np.array([6]))[0]
    assert 4.0 <= pred_2026 <= 5.0
    assert "X^2" in model.equation_str()


def test_nairu_asymptotic_model_bounds(sample_data):
    X, y = sample_data
    model = NAIRUAsymptoticModel(fixed_floor=3.50)
    model.fit(X, y)

    assert model.floor_ == 3.50
    assert model.amplitude_ is not None and model.amplitude_ > 0
    assert model.decay_rate_ is not None and model.decay_rate_ > 0

    # In-sample predictions should decrease smoothly
    preds = model.predict(X)
    assert np.all(np.diff(preds) < 0)

    # 2026 (X = 6)
    pred_2026 = model.predict(np.array([6]))[0]
    assert 4.0 <= pred_2026 <= 5.0

    # Distant future (e.g. X = 50, Year 2070) must never fall below NAIRU floor
    pred_distant = model.predict(np.array([50]))[0]
    assert pred_distant >= 3.50
    assert pytest.approx(pred_distant, abs=0.05) == 3.50


def test_exponential_smoothing_model(sample_data):
    X, y = sample_data
    model = ExponentialSmoothingModel()
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(y)

    pred_2026 = model.predict(np.array([6]))[0]
    assert 3.5 <= pred_2026 <= 5.0


def test_moving_average_model(sample_data):
    X, y = sample_data
    model = MovingAverageModel(window=3)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(y)

    pred_2026 = model.predict(np.array([6]))[0]
    # For future, SMA-3 should equal the mean of the last 3 values: (5.32 + 4.91 + 4.85) / 3 = 5.0267
    expected = (4.85 + 4.91 + 5.32) / 3.0
    assert pytest.approx(pred_2026, abs=1e-3) == expected


def test_models_calendar_year_invariance():
    X_rel = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    X_cal = np.array([[2020], [2021], [2022], [2023], [2024], [2025]], dtype=float)
    y = np.array([7.07, 6.49, 5.86, 5.32, 4.91, 4.85], dtype=float)

    # 1. Linear
    ols_rel = LinearTrendModel().fit(X_rel, y)
    ols_cal = LinearTrendModel().fit(X_cal, y)
    assert pytest.approx(ols_rel.predict([[6]])[0], abs=1e-3) == ols_cal.predict([[2026]])[0]

    # 2. NAIRU
    nairu_rel = NAIRUAsymptoticModel(fixed_floor=3.5).fit(X_rel, y)
    nairu_cal = NAIRUAsymptoticModel(fixed_floor=3.5).fit(X_cal, y)
    assert pytest.approx(nairu_rel.predict([[6]])[0], abs=1e-2) == nairu_cal.predict([[2026]])[0]

    # 3. Exponential Smoothing
    es_rel = ExponentialSmoothingModel().fit(X_rel, y)
    es_cal = ExponentialSmoothingModel().fit(X_cal, y)
    assert pytest.approx(es_rel.predict([[6]])[0], abs=1e-2) == es_cal.predict([[2026]])[0]

    # 4. Moving Average
    ma_rel = MovingAverageModel(3).fit(X_rel, y)
    ma_cal = MovingAverageModel(3).fit(X_cal, y)
    assert pytest.approx(ma_rel.predict([[6]])[0], abs=1e-3) == ma_cal.predict([[2026]])[0]


def test_predict_interval_contract(sample_data):
    X, y = sample_data
    models = [
        LinearTrendModel(),
        PolynomialTrendModel(degree=2),
        NAIRUAsymptoticModel(fixed_floor=3.5),
        ExponentialSmoothingModel(),
        MovingAverageModel(window=3),
    ]

    for model in models:
        model.fit(X, y)
        y_pred, se_pred, moe, (pi_low, pi_high) = model.predict_interval(np.array([[6]]))

        assert len(y_pred) == 1
        assert len(se_pred) == 1
        assert len(moe) == 1
        assert len(pi_low) == 1
        assert len(pi_high) == 1

        assert se_pred[0] > 0.0
        assert moe[0] > 0.0
        assert pi_low[0] <= y_pred[0] <= pi_high[0]


def test_continuous_prediction_interpolation(sample_data):
    X, y = sample_data
    es = ExponentialSmoothingModel().fit(X, y)
    ma = MovingAverageModel(3).fit(X, y)

    # Interpolate at continuous half-step
    pred_es_half = es.predict(np.array([0.5]))[0]
    pred_ma_half = ma.predict(np.array([0.5]))[0]

    assert es.predict(np.array([1.0]))[0] <= pred_es_half <= es.predict(np.array([0.0]))[0]
    assert ma.predict(np.array([1.0]))[0] <= pred_ma_half <= ma.predict(np.array([0.0]))[0]
