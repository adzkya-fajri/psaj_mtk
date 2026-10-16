"""
End-to-end pipeline orchestrator for LaborCast ML solution.
Coordinates:
- Ingestion & validation
- Feature preprocessing
- Multi-model training
- Walk-forward temporal split & LOOCV evaluation
- Residual assumption diagnostics
- Multi-scenario inference engine for 2026
- Artifact persistence (joblib + JSON metadata)
- Publication chart rendering
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union
import pandas as pd

from laborcast.data import DataLoader, DataPreprocessor, PreprocessedData
from laborcast.evaluator import (
    ModelBenchmarkEvaluator,
    ResidualDiagnostics,
    ResidualDiagnosticsResult,
)
from laborcast.inference import ForecastResult, LaborCastInference
from laborcast.models import (
    BaseModel,
    ExponentialSmoothingModel,
    LinearTrendModel,
    MovingAverageModel,
    NAIRUAsymptoticModel,
    PolynomialTrendModel,
)
from laborcast.persistence import ModelPersistence
from laborcast.visualizer import Visualizer


@dataclass
class PipelineResult:
    preprocessed_data: PreprocessedData
    benchmark_df: pd.DataFrame
    primary_model: LinearTrendModel
    asymptotic_model: NAIRUAsymptoticModel
    residual_diagnostics: ResidualDiagnosticsResult
    forecast_result: ForecastResult
    model_artifact_path: Path
    metadata_artifact_path: Path
    chart_paths: Dict[str, Path]

    def summary_text(self) -> str:
        """Formatted terminal summary report."""
        lines = []
        lines.append("=" * 72)
        lines.append(" LABORCAST MACHINE LEARNING PIPELINE - EXECUTION REPORT ")
        lines.append("=" * 72)

        # 1. Primary Model Specification
        cal_int, cal_slope = self.primary_model.calendar_params(self.preprocessed_data.base_year)
        lines.append("\n[1] MODEL REGRESI PRIMER (ORDINARY LEAST SQUARES):")
        lines.append(f"    - Persamaan Relatif (X = t - {self.preprocessed_data.base_year}) : Y = {self.primary_model.intercept_:.4f} - {abs(self.primary_model.slope_):.4f} * X")
        lines.append(f"    - Persamaan Kalender Asli (t) : Y = {cal_int:.2f} - {abs(cal_slope):.4f} * t")
        lines.append(f"    - Koefisien Determinasi (R^2) : {self.primary_model.r2_:.4f} ({self.primary_model.r2_*100:.2f}%)")
        lines.append(f"    - Standard Error Regresi (se) : {self.primary_model.se_:.4f}%")

        # 2. Benchmark Comparison
        lines.append("\n[2] PERBANDINGAN MODEL BENCHMARK:")
        lines.append(self.benchmark_df.to_string(index=False))

        # 3. Residual Diagnostics
        res = self.residual_diagnostics
        lines.append("\n[3] UJI ASUMSI RESIDUAL (OLS):")
        lines.append(f"    - Uji Normalitas (Shapiro-Wilk)  : W = {res.shapiro_stat:.4f}, p = {res.shapiro_p:.4f} (Normal: {res.shapiro_passed})")
        lines.append(f"    - Uji Normalitas (Jarque-Bera)   : JB = {res.jarque_bera_stat:.4f}, p = {res.jarque_bera_p:.4f}")
        lines.append(f"    - Uji Autokorelasi (Durbin-Watson): DW = {res.durbin_watson:.4f} (Ideal: ~2.0)")
        lines.append(f"    - Uji Homoskedastisitas (B-Pagan): p = {res.breusch_pagan_p:.4f} (Homoskedastis: {res.breusch_pagan_passed})")

        # 4. Scenario Projections 2026
        fc = self.forecast_result
        lines.append(f"\n[4] PROYEKSI TPT TAHUN {fc.target_year} (1-STEP AHEAD):")
        lines.append(f"    - Point Forecast (Agustus)       : {fc.point_forecast:.2f}%")
        lines.append(f"    - Koreksi Musiman (Februari)     : {fc.point_forecast_february:.2f}%")
        lines.append(f"    - Standard Error Prediksi (SEpred): {fc.se_prediction:.4f}%")
        lines.append(f"    - Margin of Error 95% (MoE)      : +/- {fc.margin_of_error_95:.4f}%")
        lines.append(f"    - 95% Prediction Interval (PI)   : [{fc.pi_lower_95:.2f}% s.d. {fc.pi_upper_95:.2f}%]")
        lines.append(f"    - Lantai Struktural (NAIRU Floor): {fc.nairu_floor:.2f}%")

        lines.append("\n[5] MATRIKS SKENARIO MAKROEKONOMI 2026:")
        for key, sc in fc.scenarios.items():
            lines.append(f"    * {sc.name}:")
            lines.append(f"        TPT Agustus : {sc.tpt_agustus:.2f}% | TPT Februari: {sc.tpt_februari:.2f}%")
            lines.append(f"        Deskripsi   : {sc.description}")

        lines.append("\n[6] ARTIFAK & VISUALISASI:")
        lines.append(f"    - Model Binary   : {self.model_artifact_path}")
        lines.append(f"    - Metadata JSON  : {self.metadata_artifact_path}")
        for name, path in self.chart_paths.items():
            lines.append(f"    - Plot ({name}): {path}")
        lines.append("=" * 72)
        return "\n".join(lines)


class LaborCastPipeline:
    """End-to-end pipeline executor."""

    def __init__(
        self,
        base_year: int = 2020,
        nairu_floor: float = 3.50,
        artifacts_dir: Union[str, Path] = "experiments/artifacts",
        assets_dir: Union[str, Path] = "assets",
    ):
        self.base_year = base_year
        self.nairu_floor = nairu_floor
        self.artifacts_dir = Path(artifacts_dir)
        self.assets_dir = Path(assets_dir)
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor(base_year=base_year)

    def run(
        self,
        dataset_path: Union[str, Path] = "experiments/datasets/tpt_nasional_timeseries_2015_2025.csv",
        target_year: int = 2026,
        export_visualizations: bool = True,
        export_artifacts: bool = True,
    ) -> PipelineResult:
        # Step 1: Load and Preprocess
        df_raw = self.loader.load_national_timeseries(dataset_path)
        prep_data = self.preprocessor.prepare_features(df_raw, filter_post_pandemic=True)

        X = prep_data.X
        y = prep_data.y
        years = prep_data.years

        # Step 2: Initialize Models
        primary_model = LinearTrendModel().fit(X, y)
        poly_model = PolynomialTrendModel(degree=2).fit(X, y)
        holt_model = ExponentialSmoothingModel().fit(X, y)
        ma_model = MovingAverageModel(window=3).fit(X, y)
        nairu_model = NAIRUAsymptoticModel(fixed_floor=self.nairu_floor).fit(X, y)

        benchmark_models: Dict[str, BaseModel] = {
            "OLS_Linear": primary_model,
            "Polynomial_Deg2": poly_model,
            "Holt_ExpSmoothing": holt_model,
            "Moving_Average_3": ma_model,
            "NAIRU_Asymptotic": nairu_model,
        }

        # Step 3: Run Benchmark Evaluations
        evaluator = ModelBenchmarkEvaluator(test_size=1)
        bench_df = evaluator.compare_models(benchmark_models, X, y, years=years)

        # Step 4: Residual Diagnostics on Primary Model
        res_diag = ResidualDiagnostics.analyze(primary_model, X, y)

        # Step 5: Inference & Scenario Generation
        engine = LaborCastInference(
            model=primary_model,
            base_year=self.base_year,
            seasonal_gap=prep_data.seasonal_gap,
            nairu_floor=self.nairu_floor,
        )
        forecast_res = engine.predict_target_year(target_year=target_year)

        # Step 6: Visualizations
        chart_paths: Dict[str, Path] = {}
        if export_visualizations:
            self.assets_dir.mkdir(parents=True, exist_ok=True)
            p_fan = Visualizer.plot_scenario_forecast(
                df_hist=df_raw,
                forecast_res=forecast_res,
                output_path=self.assets_dir / "laborcast_scenario_forecast_2026.png",
            )
            p_bench = Visualizer.plot_benchmark_comparison(
                models=benchmark_models,
                X=X,
                y=y,
                years=years,
                bench_df=bench_df,
                output_path=self.assets_dir / "laborcast_benchmark_comparison.png",
            )
            p_diag = Visualizer.plot_residual_diagnostics(
                residuals=res_diag.residuals,
                fitted_values=res_diag.fitted_values,
                years=years,
                dw_stat=res_diag.durbin_watson,
                sw_p=res_diag.shapiro_p,
                output_path=self.assets_dir / "laborcast_residual_diagnostics.png",
            )
            chart_paths["scenario_forecast"] = p_fan
            chart_paths["benchmark_comparison"] = p_bench
            chart_paths["residual_diagnostics"] = p_diag

        # Step 7: Persistence
        model_art_path = self.artifacts_dir / "laborcast_model.joblib"
        meta_art_path = self.artifacts_dir / "laborcast_metadata.json"

        if export_artifacts:
            cal_int, cal_slope = primary_model.calendar_params(self.base_year)
            metadata = {
                "project": "LaborCast TPT Forecasting",
                "version": "1.0.0",
                "target_year": target_year,
                "base_year": self.base_year,
                "dataset_source": str(dataset_path),
                "n_samples": len(y),
                "primary_model": {
                    "type": "LinearTrendModel (OLS)",
                    "slope_relative": primary_model.slope_,
                    "intercept_relative": primary_model.intercept_,
                    "calendar_slope": cal_slope,
                    "calendar_intercept": cal_int,
                    "r2": primary_model.r2_,
                    "standard_error": primary_model.se_,
                    "relative_equation": f"Y = {primary_model.intercept_:.4f} + ({primary_model.slope_:.4f}) * X",
                    "calendar_equation": f"Y = {cal_int:.2f} + ({cal_slope:.4f}) * t",
                },
                "asymptotic_model": {
                    "type": "NAIRUAsymptoticModel",
                    "floor": nairu_model.floor_,
                    "amplitude": nairu_model.amplitude_,
                    "decay_rate": nairu_model.decay_rate_,
                    "equation": nairu_model.equation_str(self.base_year),
                },
                "residual_diagnostics": res_diag.summary(),
                "benchmark_rankings": bench_df.to_dict(orient="records"),
                "forecast_2026": forecast_res.to_dict(),
            }
            ModelPersistence.save(
                model=primary_model,
                metadata=metadata,
                model_path=model_art_path,
                metadata_path=meta_art_path,
            )

        return PipelineResult(
            preprocessed_data=prep_data,
            benchmark_df=bench_df,
            primary_model=primary_model,
            asymptotic_model=nairu_model,
            residual_diagnostics=res_diag,
            forecast_result=forecast_res,
            model_artifact_path=model_art_path,
            metadata_artifact_path=meta_art_path,
            chart_paths=chart_paths,
        )
