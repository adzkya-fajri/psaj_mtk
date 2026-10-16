"""
Evaluation framework for LaborCast models.
Provides:
- Point metrics: MAE, RMSE, MAPE, R2, AIC, BIC
- Temporal train/test split evaluation (train 2020-2024, test 2025)
- Leave-One-Out Cross-Validation (LOOCV)
- Econometric residual assumption diagnostics (Normality, Autocorrelation, Homoscedasticity)
- Comprehensive multi-model benchmark suite
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence
import numpy as np
import pandas as pd
import scipy.stats as stats

from laborcast.models import BaseModel, _ensure_1d, _ensure_2d


class Evaluator:
    """Core statistical performance metrics."""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = _ensure_1d(y_true)
        yp = _ensure_1d(y_pred)
        return float(np.mean(np.abs(yt - yp)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = _ensure_1d(y_true)
        yp = _ensure_1d(y_pred)
        return float(np.sqrt(np.mean((yt - yp) ** 2)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = _ensure_1d(y_true)
        yp = _ensure_1d(y_pred)
        denom = np.where(np.abs(yt) < 1e-9, 1e-9, yt)
        return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        yt = _ensure_1d(y_true)
        yp = _ensure_1d(y_pred)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        if ss_tot < 1e-9:
            return 1.0
        ss_res = np.sum((yt - yp) ** 2)
        return float(1.0 - (ss_res / ss_tot))

    @classmethod
    def compute_all_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_params: int = 2,
    ) -> Dict[str, float]:
        yt = _ensure_1d(y_true)
        yp = _ensure_1d(y_pred)
        n = len(yt)
        mae_val = cls.mae(yt, yp)
        rmse_val = cls.rmse(yt, yp)
        mape_val = cls.mape(yt, yp)
        r2_val = cls.r2(yt, yp)

        mse = float(np.mean((yt - yp) ** 2))
        mse_safe = max(mse, 1e-9)
        # AIC = n * ln(MSE) + 2 * k
        # BIC = n * ln(MSE) + k * ln(n)
        aic_val = float(n * np.log(mse_safe) + 2 * n_params)
        bic_val = float(n * np.log(mse_safe) + n_params * np.log(max(n, 1)))

        return {
            "MAE": round(mae_val, 4),
            "RMSE": round(rmse_val, 4),
            "MAPE": round(mape_val, 2),
            "R2": round(r2_val, 4),
            "AIC": round(aic_val, 2),
            "BIC": round(bic_val, 2),
        }


@dataclass
class TemporalSplitResult:
    train_metrics: Dict[str, float]
    test_year: int
    test_actual: float
    test_predicted: float
    test_absolute_error: float
    test_ape: float
    test_accuracy: float
    test_metrics: Optional[Dict[str, float]] = None


class TemporalSplitEvaluator:
    """Evaluates time-series walk-forward holdout on recent observation(s)."""

    def __init__(self, test_size: int = 1):
        if test_size < 1:
            raise ValueError(f"test_size must be >= 1, got {test_size}")
        self.test_size = test_size

    def evaluate(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        years: Optional[Sequence[int]] = None,
    ) -> TemporalSplitResult:
        X_arr = _ensure_2d(X)
        y_arr = _ensure_1d(y)
        n = len(y_arr)

        if n <= self.test_size + 1:
            raise ValueError(f"Need at least {self.test_size + 2} samples for temporal split.")

        train_slice = slice(0, n - self.test_size)
        test_slice = slice(n - self.test_size, n)

        X_train, y_train = X_arr[train_slice], y_arr[train_slice]
        X_test, y_test = X_arr[test_slice], y_arr[test_slice]

        # Clone and fit model on training slice
        model_copy = copy.deepcopy(model)
        model_copy.fit(X_train, y_train)

        y_train_pred = model_copy.predict(X_train)
        train_metrics = Evaluator.compute_all_metrics(y_train, y_train_pred)

        y_test_pred = _ensure_1d(model_copy.predict(X_test))
        test_metrics = Evaluator.compute_all_metrics(y_test, y_test_pred)

        if self.test_size == 1:
            y_test_scalar = float(y_test[0])
            y_pred_scalar = float(y_test_pred[0])
            abs_err = float(abs(y_test_scalar - y_pred_scalar))
            ape = float((abs_err / max(1e-9, y_test_scalar)) * 100.0)
            test_yr = int(years[-1]) if years is not None and len(years) >= n else 2025
        else:
            abs_err = float(np.mean(np.abs(y_test - y_test_pred)))
            ape = float(np.mean(np.abs((y_test - y_test_pred) / np.maximum(1e-9, y_test))) * 100.0)
            y_test_scalar = float(y_test[-1])
            y_pred_scalar = float(y_test_pred[-1])
            test_yr = int(years[-1]) if years is not None and len(years) >= n else 2025

        return TemporalSplitResult(
            train_metrics=train_metrics,
            test_year=test_yr,
            test_actual=round(y_test_scalar, 4),
            test_predicted=round(y_pred_scalar, 4),
            test_absolute_error=round(abs_err, 4),
            test_ape=round(ape, 2),
            test_accuracy=round(100.0 - ape, 2),
            test_metrics=test_metrics,
        )


@dataclass
class LOOCVResult:
    loocv_mae: float
    loocv_rmse: float
    loocv_mape: float
    actuals: np.ndarray
    predictions: np.ndarray


class LOOCVEvaluator:
    """Leave-One-Out Cross-Validation evaluator."""

    def evaluate(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> LOOCVResult:
        X_arr = _ensure_2d(X)
        y_arr = _ensure_1d(y)
        n = len(y_arr)

        preds = []
        for i in range(n):
            train_idx = [j for j in range(n) if j != i]
            test_idx = [i]

            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_te = X_arr[test_idx]

            m_clone = copy.deepcopy(model)
            m_clone.fit(X_tr, y_tr)
            y_hat = float(m_clone.predict(X_te)[0])
            preds.append(y_hat)

        preds_arr = np.array(preds)
        mae = Evaluator.mae(y_arr, preds_arr)
        rmse = Evaluator.rmse(y_arr, preds_arr)
        mape = Evaluator.mape(y_arr, preds_arr)

        return LOOCVResult(
            loocv_mae=round(mae, 4),
            loocv_rmse=round(rmse, 4),
            loocv_mape=round(mape, 2),
            actuals=y_arr,
            predictions=preds_arr,
        )


@dataclass
class ResidualDiagnosticsResult:
    residuals: np.ndarray
    fitted_values: np.ndarray
    shapiro_stat: float
    shapiro_p: float
    shapiro_passed: bool      # Normal distribution (p > 0.05)
    jarque_bera_stat: float
    jarque_bera_p: float
    durbin_watson: float      # Autocorrelation (ideal near 2.0)
    breusch_pagan_stat: Optional[float]
    breusch_pagan_p: Optional[float]
    breusch_pagan_passed: bool  # Homoscedasticity (p > 0.05)
    mean_residual: float
    std_residual: float

    def summary(self) -> Dict[str, Any]:
        return {
            "mean_residual": round(self.mean_residual, 6),
            "std_residual": round(self.std_residual, 4),
            "shapiro_wilk": {
                "statistic": round(self.shapiro_stat, 4),
                "p_value": round(self.shapiro_p, 4),
                "normality_satisfied": bool(self.shapiro_passed),
            },
            "jarque_bera": {
                "statistic": round(self.jarque_bera_stat, 4),
                "p_value": round(self.jarque_bera_p, 4),
            },
            "durbin_watson": round(self.durbin_watson, 4),
            "breusch_pagan": {
                "statistic": round(self.breusch_pagan_stat, 4) if self.breusch_pagan_stat is not None else None,
                "p_value": round(self.breusch_pagan_p, 4) if self.breusch_pagan_p is not None else None,
                "homoscedasticity_satisfied": bool(self.breusch_pagan_passed),
            },
        }


class ResidualDiagnostics:
    """Diagnoses econometric and statistical assumptions on regression residuals."""

    @classmethod
    def analyze(
        cls,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float = 0.05,
    ) -> ResidualDiagnosticsResult:
        X_arr = _ensure_2d(X)
        y_arr = _ensure_1d(y)
        y_hat = _ensure_1d(model.predict(X_arr))
        residuals = y_arr - y_hat
        n = len(residuals)

        # 1. Normality: Shapiro-Wilk Test
        sw_stat, sw_p = stats.shapiro(residuals)

        # 2. Normality: Jarque-Bera Test
        jb_stat, jb_p = stats.jarque_bera(residuals)

        # 3. Autocorrelation: Durbin-Watson statistic
        diff_res = np.diff(residuals)
        sum_sq_diff = np.sum(diff_res ** 2)
        sum_sq_res = np.sum(residuals ** 2)
        dw_val = float(sum_sq_diff / sum_sq_res) if sum_sq_res > 1e-9 else 2.0

        # 4. Homoscedasticity: Breusch-Pagan Test
        bp_stat: Optional[float] = None
        bp_p: Optional[float] = None
        bp_passed = True

        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            import statsmodels.api as sm

            X_const = sm.add_constant(X_arr)
            lm_stat, p_val, _, _ = het_breuschpagan(residuals, X_const)
            bp_stat = float(lm_stat)
            bp_p = float(p_val)
            bp_passed = bool(bp_p > alpha)
        except Exception:
            # Fallback manual correlation test between squared residuals and fitted values
            sq_res = residuals ** 2
            corr, p_val = stats.pearsonr(y_hat, sq_res) if len(y_hat) >= 3 else (0.0, 1.0)
            bp_stat = float(abs(corr))
            bp_p = float(p_val)
            bp_passed = bool(p_val > alpha)

        return ResidualDiagnosticsResult(
            residuals=residuals,
            fitted_values=y_hat,
            shapiro_stat=float(sw_stat),
            shapiro_p=float(sw_p),
            shapiro_passed=bool(sw_p > alpha),
            jarque_bera_stat=float(jb_stat),
            jarque_bera_p=float(jb_p),
            durbin_watson=float(dw_val),
            breusch_pagan_stat=bp_stat,
            breusch_pagan_p=bp_p,
            breusch_pagan_passed=bp_passed,
            mean_residual=float(np.mean(residuals)),
            std_residual=float(np.std(residuals, ddof=1)) if n > 1 else float(np.std(residuals)),
        )


class ModelBenchmarkEvaluator:
    """Runs all benchmark models through in-sample, temporal split, and LOOCV regimes."""

    def __init__(self, test_size: int = 1):
        self.temporal_eval = TemporalSplitEvaluator(test_size=test_size)
        self.loocv_eval = LOOCVEvaluator()

    def compare_models(
        self,
        models: Mapping[str, BaseModel],
        X: np.ndarray,
        y: np.ndarray,
        years: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        records = []
        for key, model in models.items():
            # Fit on full data for in-sample evaluation
            m_full = copy.deepcopy(model)
            m_full.fit(X, y)
            y_pred = m_full.predict(X)

            in_metrics = Evaluator.compute_all_metrics(y, y_pred)
            temp_res = self.temporal_eval.evaluate(model, X, y, years=years)
            loo_res = self.loocv_eval.evaluate(model, X, y)

            records.append({
                "Model Key": key,
                "Model Name": m_full.name,
                "In-Sample RMSE (%)": in_metrics["RMSE"],
                "In-Sample MAE (%)": in_metrics["MAE"],
                "In-Sample MAPE (%)": in_metrics["MAPE"],
                "In-Sample R2": in_metrics["R2"],
                "Test Actual (2025) (%)": temp_res.test_actual,
                "Test Pred (2025) (%)": temp_res.test_predicted,
                "Test MAE (%)": temp_res.test_absolute_error,
                "Test APE (%)": temp_res.test_ape,
                "LOOCV RMSE (%)": loo_res.loocv_rmse,
                "LOOCV MAE (%)": loo_res.loocv_mae,
                "LOOCV MAPE (%)": loo_res.loocv_mape,
            })

        df_bench = pd.DataFrame(records)
        return df_bench.sort_values(by="LOOCV RMSE (%)").reset_index(drop=True)
