"""
Publication-quality visualization module for LaborCast.
Generates:
1. Scenario forecast fan chart (2015-2030) with 95% Prediction Interval
2. Multi-model benchmark trajectory and error metric comparison
3. Econometric 2x2 residual assumption diagnostic plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from laborcast.inference import ForecastResult
from laborcast.models import BaseModel


def _set_chart_style() -> None:
    """Set aesthetic defaults for academic report visualizations."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
    plt.rcParams["axes.edgecolor"] = "#cbd5e1"
    plt.rcParams["axes.linewidth"] = 0.8


class Visualizer:
    """Renders and exports publication-quality charts for the PSAJ project."""

    @classmethod
    def plot_scenario_forecast(
        cls,
        df_hist: pd.DataFrame,
        forecast_res: ForecastResult,
        output_path: Union[str, Path] = "assets/laborcast_scenario_forecast_2026.png",
    ) -> Path:
        """
        Render the main forecasting fan chart showing:
        - Actual historical TPT (2015-2025)
        - Exogenous COVID-19 shock marker
        - 2020-2025 post-pandemic recovery fit
        - 2026-2030 projection horizon
        - 95% Prediction Interval shaded area
        - Baseline, Optimistic, and Pessimistic scenarios
        - NAIRU asymptotic boundary floor (3.5%)
        """
        _set_chart_style()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)

        # 1. Historical Actual Series
        hist_years = df_hist["Tahun"].to_numpy()
        hist_aug = df_hist["TPT_Agustus"].to_numpy()

        ax.plot(
            hist_years,
            hist_aug,
            "o-",
            color="#1e3a8a",
            linewidth=2.4,
            markersize=6,
            label="TPT Aktual Sakernas BPS (Agustus)",
            zorder=5,
        )

        # 2. Exogenous Shock Annotation
        idx_2020 = np.where(hist_years == 2020)[0]
        if len(idx_2020) > 0:
            val_2020 = hist_aug[idx_2020[0]]
            ax.annotate(
                f"Syok Pandemi COVID-19\nAgustus 2020 ({val_2020:.2f}%)",
                xy=(2020, val_2020),
                xytext=(2016.8, 6.75),
                arrowprops=dict(facecolor="#dc2626", edgecolor="#b91c1c", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold",
                color="#dc2626",
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fee2e2", edgecolor="#f87171", alpha=0.9),
            )

        # 3. Horizon Scenarios
        horizon = forecast_res.multi_year_horizon
        hz_years = np.array([item["year"] for item in horizon])
        hz_base = np.array([item["forecast_agustus"] for item in horizon])
        hz_opt = np.array([item["optimistic_agustus"] for item in horizon])
        hz_pess = np.array([item["pessimistic_agustus"] for item in horizon])
        hz_pi_low = np.array([item["pi_lower"] for item in horizon])
        hz_pi_high = np.array([item["pi_upper"] for item in horizon])

        # Connect last actual observation to future horizon
        last_hist_yr = int(hist_years[-1])
        connect_years = np.insert(hz_years, 0, last_hist_yr)
        connect_base = np.insert(hz_base, 0, hist_aug[-1])
        connect_opt = np.insert(hz_opt, 0, hist_aug[-1])
        connect_pess = np.insert(hz_pess, 0, hist_aug[-1])
        connect_low = np.insert(hz_pi_low, 0, hist_aug[-1])
        connect_high = np.insert(hz_pi_high, 0, hist_aug[-1])

        # Shaded 95% Prediction Interval Band
        ax.fill_between(
            connect_years,
            connect_low,
            connect_high,
            color="#93c5fd",
            alpha=0.35,
            label="95% Interval Prediksi (PI)",
            zorder=2,
        )

        # Baseline Scenario Line
        tgt_yr = forecast_res.target_year
        ax.plot(
            connect_years,
            connect_base,
            "--s",
            color="#16a34a",
            linewidth=2.2,
            markersize=5,
            label=f"Skenario Baseline {tgt_yr}: {forecast_res.point_forecast:.2f}%",
            zorder=4,
        )

        # Optimistic Scenario Line
        opt_tgt = forecast_res.scenarios["optimistic"].tpt_agustus
        ax.plot(
            connect_years,
            connect_opt,
            ":^",
            color="#0284c7",
            linewidth=1.8,
            markersize=5,
            label=f"Skenario Optimis {tgt_yr}: {opt_tgt:.2f}%",
            zorder=3,
        )

        # Pessimistic Scenario Line
        pess_tgt = forecast_res.scenarios["pessimistic"].tpt_agustus
        ax.plot(
            connect_years,
            connect_pess,
            ":v",
            color="#d97706",
            linewidth=1.8,
            markersize=5,
            label=f"Skenario Pesimis {tgt_yr}: {pess_tgt:.2f}%",
            zorder=3,
        )

        # Target Year Highlight Point
        ax.scatter(
            [tgt_yr],
            [forecast_res.point_forecast],
            color="#dc2626",
            s=80,
            edgecolor="black",
            linewidth=1.5,
            zorder=6,
        )
        ax.annotate(
            f"Target {tgt_yr}\nAgustus: {forecast_res.point_forecast:.2f}%\nFebruari: {forecast_res.point_forecast_february:.2f}%\nPI: [{forecast_res.pi_lower_95:.2f}%, {forecast_res.pi_upper_95:.2f}%]",
            xy=(tgt_yr, forecast_res.point_forecast),
            xytext=(tgt_yr + 0.4, 4.8),
            arrowprops=dict(facecolor="#16a34a", shrink=0.08, width=1.2, headwidth=5),
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0fdf4", edgecolor="#86efac", alpha=0.95),
        )

        # NAIRU Floor Reference Line
        ax.axhline(
            forecast_res.nairu_floor,
            color="#7c3aed",
            linestyle="-.",
            linewidth=1.6,
            label=f"Lantai Alami (NAIRU ~{forecast_res.nairu_floor:.1f}%)",
            zorder=2,
        )

        ax.set_title(
            f"Tren Historis & Peramalan Multi-Skenario TPT Nasional {tgt_yr}-{hz_years[-1]} (LaborCast Engine)",
            fontsize=13,
            fontweight="bold",
            pad=14,
            color="#0f172a",
        )
        ax.set_xlabel("Tahun Amatan / Proyeksi", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_ylabel("Tingkat Pengangguran Terbuka (%)", fontsize=11, fontweight="bold", labelpad=8)
        ax.set_xticks(range(int(hist_years[0]), int(hz_years[-1]) + 1))
        ax.set_ylim(2.5, 7.8)
        ax.grid(True, linestyle="--", alpha=0.55)
        ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", framealpha=0.95, fontsize=9)

        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        return out

    @classmethod
    def plot_benchmark_comparison(
        cls,
        models: Mapping[str, BaseModel],
        X: np.ndarray,
        y: np.ndarray,
        years: Sequence[int],
        bench_df: pd.DataFrame,
        output_path: Union[str, Path] = "assets/laborcast_benchmark_comparison.png",
    ) -> Path:
        """
        Generate two-panel benchmark comparison:
        - Top: Curves of all fitted models extrapolated to target year
        - Bottom: Bar chart comparing performance metrics (RMSE, MAE, LOOCV)
        """
        _set_chart_style()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 9.5), dpi=300, gridspec_kw={"height_ratios": [1.2, 1]})

        # Panel 1: Trajectory Comparison
        x_min = float(np.min(X))
        x_max = float(np.max(X))
        step = float(np.median(np.diff(np.sort(X.ravel())))) if len(X) > 1 else 1.0
        x_target = x_max + step
        x_dense = np.linspace(x_min, x_target, 150)
        base_yr = int(years[0] - round(x_min))
        dense_years = base_yr + x_dense
        tgt_year = int(round(base_yr + x_target))

        # Plot Actual Points
        ax1.scatter(years, y, color="#0f172a", s=65, zorder=5, label=f"Data Aktual ({years[0]}-{years[-1]})", edgecolor="black")

        palette = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#e11d48"]
        for idx, (m_key, model) in enumerate(models.items()):
            color = palette[idx % len(palette)]
            y_curve = model.predict(x_dense)
            pred_tgt = float(model.predict(np.array([[x_target]]))[0])
            ax1.plot(
                dense_years,
                y_curve,
                label=f"{model.name} ({tgt_year}: {pred_tgt:.2f}%)",
                color=color,
                linewidth=2.0,
            )

        ax1.set_title(
            f"Perbandingan Trajektori Model Benchmark Menuju {tgt_year}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax1.set_xlabel("Tahun", fontsize=10, fontweight="bold")
        ax1.set_ylabel("TPT (%)", fontsize=10, fontweight="bold")
        ax1.set_xticks(range(int(years[0]), tgt_year + 1))
        ax1.set_ylim(3.4, 7.5)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

        # Panel 2: Error Metrics Bar Chart
        model_names = [name[:24] for name in bench_df["Model Name"]]
        x_pos = np.arange(len(model_names))
        bar_width = 0.22

        r1 = ax2.bar(x_pos - bar_width, bench_df["In-Sample RMSE (%)"], width=bar_width, label="In-Sample RMSE (%)", color="#3b82f6")
        r2 = ax2.bar(x_pos, bench_df["LOOCV RMSE (%)"], width=bar_width, label="LOOCV RMSE (%)", color="#10b981")
        r3 = ax2.bar(x_pos + bar_width, bench_df["Test APE (%)"] / 10.0, width=bar_width, label="Test APE / 10 (%)", color="#f59e0b")

        ax2.set_title("Evaluasi Metrik Akurasi Antarmodel (In-Sample, LOOCV, & Holdout Uji 2025)", fontsize=12, fontweight="bold", pad=10)
        ax2.set_ylabel("Nilai Metrik Error", fontsize=10, fontweight="bold")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=8.5)

        # Add value labels above bars
        for rect in list(r1) + list(r2) + list(r3):
            height = rect.get_height()
            if height > 0:
                ax2.annotate(
                    f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        return out

    @classmethod
    def plot_residual_diagnostics(
        cls,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
        years: Sequence[int],
        dw_stat: float,
        sw_p: float,
        output_path: Union[str, Path] = "assets/laborcast_residual_diagnostics.png",
    ) -> Path:
        """
        Generate classical 2x2 econometric residual diagnostic grid:
        1. Residuals vs Fitted values (linearity & homoscedasticity)
        2. Normal Q-Q plot (normality)
        3. Scale-Location plot (homoscedasticity)
        4. Residuals vs Time (autocorrelation with DW statistic)
        """
        _set_chart_style()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.0), dpi=300)

        # 1. Residuals vs Fitted
        ax1 = axs[0, 0]
        ax1.scatter(fitted_values, residuals, color="#2563eb", edgecolor="black", s=60, zorder=4)
        ax1.axhline(0, color="#dc2626", linestyle="--", linewidth=1.5, zorder=2)
        ax1.set_title("1. Residuals vs Fitted (Linearity)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Fitted Values (%)", fontsize=9.5)
        ax1.set_ylabel("Residuals", fontsize=9.5)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Normal Q-Q Plot
        ax2 = axs[0, 1]
        std_res = (residuals - np.mean(residuals)) / (np.std(residuals, ddof=1) if len(residuals) > 1 else 1.0)
        osm, osr = stats.probplot(std_res, dist="norm")[:2]
        ax2.scatter(osm[0], osm[1], color="#059669", edgecolor="black", s=60, zorder=4)
        ax2.plot(osm[0], osm[0], color="#dc2626", linestyle="--", linewidth=1.5, zorder=2)
        norm_label = "Normal (p > 0.05)" if sw_p > 0.05 else "Non-normal (p <= 0.05)"
        ax2.set_title(f"2. Normal Q-Q Plot (Shapiro p={sw_p:.3f} | {norm_label})", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Theoretical Quantiles", fontsize=9.5)
        ax2.set_ylabel("Standardized Residuals", fontsize=9.5)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. Scale-Location Plot
        ax3 = axs[1, 0]
        sqrt_abs_std = np.sqrt(np.abs(std_res))
        ax3.scatter(fitted_values, sqrt_abs_std, color="#7c3aed", edgecolor="black", s=60, zorder=4)
        ax3.set_title("3. Scale-Location Plot (Homoscedasticity)", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Fitted Values (%)", fontsize=9.5)
        ax3.set_ylabel(r"$\sqrt{|\text{Standardized Residuals}|}$", fontsize=9.5)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # 4. Residuals vs Time Order
        ax4 = axs[1, 1]
        ax4.plot(years, residuals, "o-", color="#d97706", linewidth=2.0, markersize=6, zorder=4)
        ax4.axhline(0, color="#dc2626", linestyle="--", linewidth=1.5, zorder=2)
        dw_label = "Tidak Ada Autokorelasi Signifikan" if 1.0 <= dw_stat <= 3.0 else "Potensi Autokorelasi"
        ax4.set_title(f"4. Residuals vs Waktu (Durbin-Watson = {dw_stat:.3f})", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Tahun Observasi", fontsize=9.5)
        ax4.set_ylabel("Residuals", fontsize=9.5)
        ax4.set_xticks(years)
        ax4.annotate(dw_label, xy=(0.05, 0.88), xycoords="axes fraction", fontsize=8.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef3c7", edgecolor="#f59e0b", alpha=0.9))
        ax4.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("Uji Diagnostik Asumsi Residual OLS (LaborCast)", fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        return out
