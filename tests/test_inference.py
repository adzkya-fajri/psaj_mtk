"""
Unit tests for laborcast.inference module.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from laborcast.inference import LaborCastInference
from laborcast.models import LinearTrendModel, NAIRUAsymptoticModel


@pytest.fixture
def fitted_ols():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([7.07, 6.49, 5.86, 5.32, 4.91, 4.85], dtype=float)
    return LinearTrendModel().fit(X, y)


def test_inference_target_year_2026(fitted_ols):
    engine = LaborCastInference(
        model=fitted_ols,
        base_year=2020,
        seasonal_gap=0.09,
        nairu_floor=3.50,
    )
    result = engine.predict_target_year(target_year=2026, confidence_level=0.95)

    assert result.target_year == 2026
    assert result.base_year == 2020
    assert result.relative_step == 6.0

    # Point forecast checks
    assert pytest.approx(result.point_forecast, abs=1e-2) == 4.11
    assert pytest.approx(result.point_forecast_february, abs=1e-2) == 4.02

    # Prediction Interval checks
    assert result.pi_lower_95 < result.point_forecast < result.pi_upper_95
    assert result.pi_lower_95 >= result.nairu_floor  # Bound check
    assert result.margin_of_error_95 > 0

    # Scenarios checks
    scenarios = result.scenarios
    assert "baseline" in scenarios
    assert "optimistic" in scenarios
    assert "pessimistic" in scenarios

    opt = scenarios["optimistic"].tpt_agustus
    base = scenarios["baseline"].tpt_agustus
    pess = scenarios["pessimistic"].tpt_agustus

    # Economic consistency: Optimistic <= Baseline <= Pessimistic
    assert opt <= base <= pess
    assert opt >= result.nairu_floor

    # Multi-year horizon checks
    horizon = result.multi_year_horizon
    assert len(horizon) == 5
    assert [item["year"] for item in horizon] == [2026, 2027, 2028, 2029, 2030]
    for step in horizon:
        assert step["pi_lower"] <= step["forecast_agustus"] <= step["pi_upper"]


def test_inference_with_asymptotic_model():
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([7.07, 6.49, 5.86, 5.32, 4.91, 4.85], dtype=float)
    nairu_model = NAIRUAsymptoticModel(fixed_floor=3.50).fit(X, y)

    engine = LaborCastInference(
        model=nairu_model,
        base_year=2020,
        seasonal_gap=0.09,
        nairu_floor=3.50,
    )
    result = engine.predict_target_year(target_year=2026)
    assert result.point_forecast >= 3.50
    assert result.pi_lower_95 >= 3.50
    # Standard error should be dynamically computed from NAIRU model residuals (~0.1382), not hardcoded OLS (0.2814)
    assert pytest.approx(result.se_prediction, abs=1e-2) == 0.14
    assert result.margin_of_error_95 > 0.0


def test_inference_with_custom_scenario_deltas(fitted_ols):
    engine = LaborCastInference(
        model=fitted_ols,
        base_year=2020,
        seasonal_gap=0.09,
        nairu_floor=3.50,
        optimistic_delta=0.50,
        pessimistic_delta=0.60,
    )
    result = engine.predict_target_year(target_year=2026)
    opt = result.scenarios["optimistic"].tpt_agustus
    pess = result.scenarios["pessimistic"].tpt_agustus
    base = result.point_forecast

    assert pytest.approx(base - opt, abs=1e-2) == 0.50
    assert pytest.approx(pess - base, abs=1e-2) == 0.60
