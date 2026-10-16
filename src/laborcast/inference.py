"""
Multi-scenario inference engine and prediction interval estimation for LaborCast.
Target: 2026 1-step ahead forecast & medium-term projections (2026-2030).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from laborcast.models import BaseModel


@dataclass
class ScenarioForecast:
    name: str
    scenario_type: str  # 'baseline', 'optimistic', 'pessimistic'
    tpt_agustus: float
    tpt_februari: float
    description: str
    assumptions: str


@dataclass
class ForecastResult:
    target_year: int
    base_year: int
    relative_step: float
    point_forecast: float
    point_forecast_february: float
    se_prediction: float
    margin_of_error_95: float
    pi_lower_95: float
    pi_upper_95: float
    nairu_floor: float
    scenarios: Dict[str, ScenarioForecast]
    multi_year_horizon: List[Dict[str, Any]]
    confidence_level: float = 0.95
    margin_of_error: float = 0.0
    pi_lower: float = 0.0
    pi_upper: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


class LaborCastInference:
    """Inference engine orchestrating point forecasts, prediction intervals, and scenarios."""

    def __init__(
        self,
        model: BaseModel,
        base_year: int = 2020,
        seasonal_gap: float = 0.09,
        nairu_floor: float = 3.50,
        optimistic_delta: Optional[float] = None,
        pessimistic_delta: Optional[float] = None,
    ):
        self.model = model
        self.base_year = base_year
        self.seasonal_gap = seasonal_gap
        self.nairu_floor = nairu_floor
        self.optimistic_delta = optimistic_delta
        self.pessimistic_delta = pessimistic_delta

    def predict_target_year(
        self,
        target_year: int = 2026,
        confidence_level: float = 0.95,
        horizon_years: int = 5,
        optimistic_delta: Optional[float] = None,
        pessimistic_delta: Optional[float] = None,
        start_year: Optional[int] = None,
    ) -> ForecastResult:
        """
        Generate comprehensive multi-scenario forecast for target_year (e.g. 2026).
        """
        x_target = float(target_year - self.base_year)
        x_arr = np.array([[x_target]])

        # 1. Point forecast (bounded by structural NAIRU floor)
        y_pred_raw = float(self.model.predict(x_arr)[0])
        y_pred = max(self.nairu_floor, y_pred_raw)

        # 2. Prediction standard error and interval dynamically evaluated from fitted model
        _, se_pred_arr, moe_arr, (pi_low_arr, pi_high_arr) = self.model.predict_interval(
            x_arr, confidence_level=confidence_level
        )
        se_pred = float(se_pred_arr[0])
        moe = float(moe_arr[0])
        # Preserving uncertainty band even when y_pred is bounded at the NAIRU floor
        pi_low = max(self.nairu_floor, y_pred - moe)
        pi_high = max(y_pred + moe, float(pi_high_arr[0]))

        point_forecast_feb = max(self.nairu_floor, y_pred - self.seasonal_gap)

        # 3. Scenario definitions
        # Resolve macroeconomic scenario shifts
        opt_shift = optimistic_delta or self.optimistic_delta
        if opt_shift is None:
            opt_shift = 0.35  # Empirical 2026 accelerated recovery scenario
        pess_shift = pessimistic_delta or self.pessimistic_delta
        if pess_shift is None:
            pess_shift = 0.40  # Empirical 2026 external turbulence scenario

        # Baseline: Pure trend projection
        baseline = ScenarioForecast(
            name="Baseline (Rezim Tren Berjalan)",
            scenario_type="baseline",
            tpt_agustus=round(y_pred, 4),
            tpt_februari=round(point_forecast_feb, 4),
            description="Pertumbuhan ekonomi stabil 5.0%-5.2%, penyerapan tenaga kerja mengikuti laju tren pemulihan pascapandemi normal.",
            assumptions="Elastisitas tenaga kerja stabil, sektor jasa dan digital menyerap angkatan kerja baru tanpa guncangan makro.",
        )

        # Optimistic: Accelerated downstreaming & labor-intensive investments
        opt_tpt_aug = max(self.nairu_floor, round(y_pred - opt_shift, 4))
        opt_tpt_feb = max(self.nairu_floor, round(opt_tpt_aug - self.seasonal_gap, 4))
        optimistic = ScenarioForecast(
            name="Optimis (Akselerasi Hilirisasi & Investasi Padat Karya)",
            scenario_type="optimistic",
            tpt_agustus=opt_tpt_aug,
            tpt_februari=opt_tpt_feb,
            description="Pertumbuhan ekonomi melaju di atas 5.5%, investasi padat karya dan revitalisasi vokasi SMK-DUDI mempercepat penyerapan kerja.",
            assumptions="Program link-and-match vokasi optimal, ekspansi industri manufaktur pengolahan, efisiensi penempatan kerja BKK tinggi.",
        )

        # Pessimistic: Global geopolitical turbulence & manufacturing slowdown
        pess_tpt_aug = round(y_pred + pess_shift, 4)
        pess_tpt_feb = round(pess_tpt_aug - self.seasonal_gap, 4)
        pessimistic = ScenarioForecast(
            name="Pesimis (Turbulensi Global & Perlambatan Manufaktur)",
            scenario_type="pessimistic",
            tpt_agustus=pess_tpt_aug,
            tpt_februari=pess_tpt_feb,
            description="Perlambatan ekonomi global, fluktuasi komoditas, dan tekanan deindustrialisasi dini menahan ekspansi rekrutmen tenaga kerja.",
            assumptions="Penurunan permintaan ekspor manufaktur, kontraksi penyerapan angkatan kerja formal, peningkatan pekerja informal.",
        )

        scenarios = {
            "baseline": baseline,
            "optimistic": optimistic,
            "pessimistic": pessimistic,
        }

        # 4. Multi-year horizon projections
        if start_year is not None:
            s_yr = start_year
            e_yr = max(target_year + 1, start_year + horizon_years)
        else:
            s_yr = target_year
            e_yr = target_year + horizon_years

        multi_year = []
        for yr in range(s_yr, e_yr):
            x_step = float(yr - self.base_year)
            x_step_arr = np.array([[x_step]])
            val_pt_raw = float(self.model.predict(x_step_arr)[0])
            val_pt = max(self.nairu_floor, val_pt_raw)

            _, s_pred_arr, m_err_arr, (p_low_arr, p_hi_arr) = self.model.predict_interval(
                x_step_arr, confidence_level=confidence_level
            )
            m_val = float(m_err_arr[0])
            s_val = float(s_pred_arr[0])
            low_val = max(self.nairu_floor, val_pt - m_val)
            hi_val = max(val_pt + m_val, float(p_hi_arr[0]))
            feb_val = max(self.nairu_floor, val_pt - self.seasonal_gap)

            multi_year.append({
                "year": yr,
                "relative_step": x_step,
                "forecast_agustus": round(val_pt, 4),
                "forecast_februari": round(feb_val, 4),
                "pi_lower": round(low_val, 4),
                "pi_upper": round(hi_val, 4),
                "margin_of_error": round(m_val, 4),
                "se_prediction": round(s_val, 4),
                "optimistic_agustus": max(self.nairu_floor, round(val_pt - opt_shift, 4)),
                "optimistic_februari": max(self.nairu_floor, round(feb_val - opt_shift, 4)),
                "pessimistic_agustus": round(val_pt + pess_shift, 4),
                "pessimistic_februari": round(feb_val + pess_shift, 4),
            })

        return ForecastResult(
            target_year=target_year,
            base_year=self.base_year,
            relative_step=x_target,
            point_forecast=round(y_pred, 4),
            point_forecast_february=round(point_forecast_feb, 4),
            se_prediction=round(se_pred, 4),
            margin_of_error_95=round(moe, 4),
            pi_lower_95=round(pi_low, 4),
            pi_upper_95=round(pi_high, 4),
            nairu_floor=self.nairu_floor,
            scenarios=scenarios,
            multi_year_horizon=multi_year,
            confidence_level=confidence_level,
            margin_of_error=round(moe, 4),
            pi_lower=round(pi_low, 4),
            pi_upper=round(pi_high, 4),
        )
