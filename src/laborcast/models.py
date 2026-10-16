"""
Benchmark and primary machine learning models for LaborCast TPT forecasting.
Includes:
- LinearTrendModel (Ordinary Least Squares)
- PolynomialTrendModel (Quadratic curvature)
- ExponentialSmoothingModel (Holt's Linear Trend)
- MovingAverageModel (SMA / EMA baseline)
- NAIRUAsymptoticModel (Bounded economic decay toward natural rate)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def _ensure_2d(X: Union[np.ndarray, list, float, int]) -> np.ndarray:
    """Ensure input X is a 2D numpy array of floats."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _ensure_1d(y: Union[np.ndarray, list, float, int]) -> np.ndarray:
    """Ensure target y is a 1D numpy array of floats."""
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr.ravel()


class BaseModel(ABC):
    """Abstract base class for all forecasting models in LaborCast."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseModel:
        """Fit model parameters."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point forecasts."""
        pass

    @abstractmethod
    def predict_interval(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute point prediction, standard error of prediction, margin of error,
        and prediction interval (PI).

        Returns:
            (point_pred, se_pred, margin_of_error, (pi_lower, pi_upper))
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return fitted model parameters as a dictionary."""
        pass

    @abstractmethod
    def equation_str(self, base_year: int = 2020) -> str:
        """Return formatted mathematical equation string."""
        pass


class LinearTrendModel(BaseModel):
    """
    Ordinary Least Squares (OLS) Linear Trend Model.
    Equation:
      Y(X) = beta_0 + beta_1 * X
      where X = t - base_year
    Calendar year formulation:
      Y(t) = a_t + beta_1 * t
      where a_t = beta_0 - beta_1 * base_year
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self._model = LinearRegression(fit_intercept=fit_intercept)
        self.slope_: Optional[float] = None
        self.intercept_: Optional[float] = None
        self.se_: Optional[float] = None  # Residual standard error
        self.r2_: Optional[float] = None
        self.n_samples_: Optional[int] = None
        self.x_mean_: Optional[float] = None
        self.ss_x_: Optional[float] = None
        self.x_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "Linear Regression (OLS)"

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearTrendModel:
        X_arr = _ensure_2d(X)
        y_arr = _ensure_1d(y)
        n = len(y_arr)
        if n < 2:
            raise ValueError(f"Need at least 2 samples to fit OLS, got {n}")

        self._model.fit(X_arr, y_arr)
        self.slope_ = float(self._model.coef_[0])
        self.intercept_ = float(self._model.intercept_)
        self.n_samples_ = n
        self.x_train_ = X_arr.ravel().copy()
        self.y_train_ = y_arr.copy()

        # Residual standard error and statistics
        y_pred = self._model.predict(X_arr)
        residuals = y_arr - y_pred
        self.fitted_values_ = y_pred
        self.residuals_ = residuals
        df_e = max(1, n - 2)
        self.se_ = float(np.sqrt(np.sum(residuals**2) / df_e))

        ss_tot = np.sum((y_arr - np.mean(y_arr))**2)
        ss_res = np.sum(residuals**2)
        self.r2_ = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

        x_1d = X_arr.ravel()
        self.x_mean_ = float(np.mean(x_1d))
        self.ss_x_ = float(np.sum((x_1d - self.x_mean_)**2))
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict()")
        X_arr = _ensure_2d(X)
        return self._model.predict(X_arr)

    def predict_interval(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute point prediction, standard error of prediction, margin of error,
        and prediction interval (PI).

        Returns:
            (point_pred, se_pred, margin_of_error, (pi_lower, pi_upper))
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict_interval()")

        X_arr = _ensure_2d(X)
        y_pred = self.predict(X_arr)
        x_vals = X_arr.ravel()

        n = self.n_samples_
        df = max(1, n - 2)
        alpha = 1.0 - confidence_level
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))

        # SE of individual prediction: s_e * sqrt(1 + 1/n + (x0 - x_bar)^2 / sum((xi - x_bar)^2))
        if self.ss_x_ and self.ss_x_ > 1e-9:
            leverage = 1.0 + (1.0 / n) + ((x_vals - self.x_mean_)**2) / self.ss_x_
        else:
            leverage = np.ones_like(x_vals) * (1.0 + 1.0 / n)

        se_pred = self.se_ * np.sqrt(leverage)
        moe = t_crit * se_pred
        pi_low = y_pred - moe
        pi_high = y_pred + moe

        return y_pred, se_pred, moe, (pi_low, pi_high)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "slope": self.slope_,
            "intercept": self.intercept_,
            "se": self.se_,
            "r2": self.r2_,
            "n_samples": self.n_samples_,
        }

    def calendar_params(self, base_year: int = 2020) -> Tuple[float, float]:
        """
        Returns (calendar_intercept, calendar_slope) where:
          Y(t) = calendar_intercept + calendar_slope * t
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted")
        calendar_intercept = self.intercept_ - (self.slope_ * base_year)
        return float(calendar_intercept), float(self.slope_)

    def equation_str(self, base_year: int = 2020) -> str:
        if not self._fitted:
            return "LinearTrendModel (unfitted)"
        cal_int, cal_slope = self.calendar_params(base_year)
        sign_rel = "+" if self.slope_ >= 0 else "-"
        sign_cal = "+" if cal_slope >= 0 else "-"
        return (
            f"Relatif (X = t - {base_year}): Y = {self.intercept_:.4f} {sign_rel} {abs(self.slope_):.4f} * X | "
            f"Kalender (t): Y = {cal_int:.2f} {sign_cal} {abs(cal_slope):.4f} * t"
        )


class PolynomialTrendModel(BaseModel):
    """
    Polynomial Regression Model (Degree d, default d=2).
    Equation:
      Y(X) = c_0 + c_1 * X + c_2 * X^2 + ...
    Captures economic deceleration in unemployment decline as recovery stabilizes.
    """

    def __init__(self, degree: int = 2):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.coeffs_: Optional[np.ndarray] = None
        self.x_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.se_: Optional[float] = None
        self.r2_: Optional[float] = None
        self.v_inv_: Optional[np.ndarray] = None
        self.n_samples_: Optional[int] = None
        self._fitted = False

    @property
    def name(self) -> str:
        return f"Polynomial Regression (Degree {self.degree})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> PolynomialTrendModel:
        X_arr = _ensure_1d(X)
        y_arr = _ensure_1d(y)
        n = len(y_arr)
        p = self.degree + 1
        if n <= self.degree:
            raise ValueError(f"Need at least {self.degree + 1} samples for degree {self.degree}, got {n}")

        self.x_train_ = X_arr.copy()
        self.y_train_ = y_arr.copy()
        self.n_samples_ = n

        # np.polyfit returns coefficients in descending order: [c_d, c_{d-1}, ..., c_0]
        self.coeffs_ = np.polyfit(X_arr, y_arr, self.degree)
        self.fitted_values_ = np.polyval(self.coeffs_, X_arr)
        self.residuals_ = y_arr - self.fitted_values_

        df_e = max(1, n - p)
        ss_res = np.sum(self.residuals_**2)
        self.se_ = float(np.sqrt(ss_res / df_e))

        ss_tot = np.sum((y_arr - np.mean(y_arr))**2)
        self.r2_ = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

        # Vandermonde matrix in descending order [X^d, ..., 1] for exact prediction interval leverage
        V = np.vander(X_arr, p)
        self.v_inv_ = np.linalg.pinv(V.T @ V)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict()")
        X_arr = _ensure_1d(X)
        return np.polyval(self.coeffs_, X_arr)

    def predict_interval(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute point prediction, standard error of prediction, margin of error,
        and prediction interval (PI).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict_interval()")
        X_arr = _ensure_1d(X)
        y_pred = self.predict(X_arr)

        n = self.n_samples_
        p = self.degree + 1
        df = max(1, n - p)
        alpha = 1.0 - confidence_level
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))

        V_new = np.vander(X_arr, p)
        leverage = np.sum((V_new @ self.v_inv_) * V_new, axis=1)
        leverage = np.maximum(leverage, 0.0)

        se_pred = self.se_ * np.sqrt(1.0 + leverage)
        moe = t_crit * se_pred
        pi_low = y_pred - moe
        pi_high = y_pred + moe

        return y_pred, se_pred, moe, (pi_low, pi_high)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "degree": self.degree,
            "coefficients": self.coeffs_.tolist() if self.coeffs_ is not None else None,
            "se": self.se_,
            "r2": self.r2_,
            "n_samples": self.n_samples_,
        }

    def equation_str(self, base_year: int = 2020) -> str:
        if not self._fitted:
            return "PolynomialTrendModel (unfitted)"
        terms = []
        d = self.degree
        for idx, coef in enumerate(self.coeffs_):
            power = d - idx
            if power == 0:
                terms.append(f"{coef:+.4f}")
            elif power == 1:
                terms.append(f"{coef:+.4f}*X")
            else:
                terms.append(f"{coef:+.4f}*X^{power}")
        return f"Y = {' '.join(terms)} (X = t - {base_year})"


class NAIRUAsymptoticModel(BaseModel):
    """
    Nonlinear Asymptotic Bounded Model targeting structural NAIRU.
    Macroeconomic foundation:
      Unemployment cannot drop to 0% due to frictional/structural unemployment (NAIRU).
      Formula:
        Y(X) = Y_floor + A * exp(-lambda * (X - X_origin))
      where:
        Y_floor : Asymptotic floor (e.g. natural unemployment ~3.50%)
        A       : Initial post-pandemic shock amplitude (TPT(0) - Y_floor)
        lambda  : Speed of labor absorption / decay rate (lambda > 0)
        X_origin: Base coordinate (X[0] of training series) ensuring invariance to calendar/relative scale
    """

    def __init__(
        self,
        fixed_floor: Optional[float] = 3.50,
        floor_bounds: Tuple[float, float] = (3.0, 4.5),
    ):
        self.fixed_floor = fixed_floor
        self.floor_bounds = floor_bounds
        self.floor_: Optional[float] = None
        self.amplitude_: Optional[float] = None
        self.decay_rate_: Optional[float] = None
        self.x_origin_: Optional[float] = None
        self.x_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.se_: Optional[float] = None
        self.r2_: Optional[float] = None
        self.n_samples_: Optional[int] = None
        self._fitted = False

    @property
    def name(self) -> str:
        return "NAIRU Asymptotic Bounded Model"

    @staticmethod
    def _func_fixed(x: np.ndarray, floor: float, a: float, lam: float) -> np.ndarray:
        return floor + a * np.exp(-lam * x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NAIRUAsymptoticModel:
        X_arr = _ensure_1d(X)
        y_arr = _ensure_1d(y)
        n = len(y_arr)
        if n < 3:
            raise ValueError("Need at least 3 points to fit asymptotic model.")

        self.x_train_ = X_arr.copy()
        self.y_train_ = y_arr.copy()
        self.n_samples_ = n
        self.x_origin_ = float(X_arr[0])
        X_shifted = X_arr - self.x_origin_

        if self.fixed_floor is not None:
            floor = float(self.fixed_floor)

            def objective(x_val, a, lam):
                return self._func_fixed(x_val, floor, a, lam)

            p0 = [max(0.1, y_arr[0] - floor), 0.2]
            bounds = ([0.0, 0.001], [15.0, 3.0])

            try:
                popt, _ = curve_fit(objective, X_shifted, y_arr, p0=p0, bounds=bounds, maxfev=5000)
                self.floor_ = floor
                self.amplitude_ = float(popt[0])
                self.decay_rate_ = float(popt[1])
            except Exception:
                # Fallback to analytical estimate if optimizer fails
                self.floor_ = floor
                self.amplitude_ = float(max(0.1, y_arr[0] - floor))
                self.decay_rate_ = 0.21
        else:
            def objective_free(x_val, flr, a, lam):
                return flr + a * np.exp(-lam * x_val)

            p0 = [3.5, max(0.1, y_arr[0] - 3.5), 0.2]
            bounds = ([self.floor_bounds[0], 0.0, 0.001], [self.floor_bounds[1], 15.0, 3.0])

            try:
                popt, _ = curve_fit(objective_free, X_shifted, y_arr, p0=p0, bounds=bounds, maxfev=5000)
                self.floor_ = float(popt[0])
                self.amplitude_ = float(popt[1])
                self.decay_rate_ = float(popt[2])
            except Exception:
                self.floor_ = 3.5
                self.amplitude_ = float(max(0.1, y_arr[0] - 3.5))
                self.decay_rate_ = 0.21

        self._fitted = True
        self.fitted_values_ = self.predict(X_arr)
        self.residuals_ = y_arr - self.fitted_values_
        p = 2 if self.fixed_floor is not None else 3
        df_e = max(1, n - p)
        ss_res = np.sum(self.residuals_**2)
        self.se_ = float(np.sqrt(ss_res / df_e))
        ss_tot = np.sum((y_arr - np.mean(y_arr))**2)
        self.r2_ = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict()")
        X_arr = _ensure_1d(X)
        X_shifted = X_arr - self.x_origin_
        pred = self._func_fixed(X_shifted, self.floor_, self.amplitude_, self.decay_rate_)
        # Guarantee lower bound >= floor
        return np.maximum(pred, self.floor_)

    def predict_interval(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute point prediction, standard error of prediction, margin of error,
        and prediction interval (PI).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict_interval()")
        X_arr = _ensure_1d(X)
        y_pred = self.predict(X_arr)

        n = self.n_samples_
        p = 2 if self.fixed_floor is not None else 3
        df = max(1, n - p)
        alpha = 1.0 - confidence_level
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))

        # Leverage in time domain
        x_mean = float(np.mean(self.x_train_))
        ss_x = float(np.sum((self.x_train_ - x_mean)**2))
        if ss_x > 1e-9:
            leverage = (1.0 / n) + ((X_arr - x_mean)**2) / ss_x
        else:
            leverage = np.ones_like(X_arr) * (1.0 / n)

        se_pred = self.se_ * np.sqrt(1.0 + leverage)
        moe = t_crit * se_pred
        pi_low = np.maximum(self.floor_, y_pred - moe)
        pi_high = y_pred + moe

        return y_pred, se_pred, moe, (pi_low, pi_high)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nairu_floor": self.floor_,
            "amplitude": self.amplitude_,
            "decay_rate": self.decay_rate_,
            "x_origin": self.x_origin_,
            "se": self.se_,
            "r2": self.r2_,
            "n_samples": self.n_samples_,
        }

    def equation_str(self, base_year: int = 2020) -> str:
        if not self._fitted:
            return "NAIRUAsymptoticModel (unfitted)"
        return (
            f"Y = {self.floor_:.2f} + {self.amplitude_:.4f} * exp(-{self.decay_rate_:.4f} * (X - {self.x_origin_:.0f})) "
            f"[Lantai NAIRU: {self.floor_:.2f}%, Base: {self.x_origin_:.0f}]"
        )


class ExponentialSmoothingModel(BaseModel):
    """
    Holt's Linear Exponential Smoothing Model.
    Models level (l_t) and trend (b_t):
      l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1} * step)
      b_t = beta * (l_t - l_{t-1}) / step + (1 - beta) * b_{t-1}
      y_hat_{t+h} = l_t + h * b_t
    """

    def __init__(self, alpha: Optional[float] = None, beta: Optional[float] = None):
        self.alpha = alpha
        self.beta = beta
        self.level_: Optional[float] = None
        self.trend_: Optional[float] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.se_: Optional[float] = None
        self.r2_: Optional[float] = None
        self.n_samples_: Optional[int] = None
        self.x_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.step_size_: float = 1.0
        self._fitted = False

    @property
    def name(self) -> str:
        return "Exponential Smoothing (Holt's Linear Trend)"

    def fit(self, X: np.ndarray, y: np.ndarray) -> ExponentialSmoothingModel:
        X_arr = _ensure_1d(X)
        y_arr = _ensure_1d(y)
        n = len(y_arr)
        if n < 2:
            raise ValueError("Need at least 2 observations for Holt's model.")
        self.n_samples_ = n
        self.x_train_ = X_arr.copy()
        self.y_train_ = y_arr.copy()

        diffs = np.diff(X_arr)
        self.step_size_ = float(np.median(diffs)) if len(diffs) > 0 and np.median(diffs) > 0 else 1.0

        # Try statsmodels Holt if available
        fitted = None
        try:
            from statsmodels.tsa.api import Holt
            stats_model = Holt(y_arr, initialization_method="estimated")
            fit_res = stats_model.fit(smoothing_level=self.alpha, smoothing_trend=self.beta, optimized=True)
            fitted = np.asarray(fit_res.fittedvalues)
            self.level_ = float(fit_res.level[-1])
            # Trend adjusted to step size of X
            self.trend_ = float(fit_res.trend[-1]) / self.step_size_
            self._fit_res = fit_res
        except Exception:
            fitted = None

        if fitted is None:
            # Fallback manual implementation of Holt's Linear Trend
            a = self.alpha if self.alpha is not None else 0.4
            b = self.beta if self.beta is not None else 0.2

            level = y_arr[0]
            trend = (y_arr[1] - y_arr[0]) / self.step_size_
            fitted_list = [level]

            for t in range(1, n):
                val = y_arr[t]
                last_level = level
                level = a * val + (1 - a) * (level + trend * self.step_size_)
                trend = b * (level - last_level) / self.step_size_ + (1 - b) * trend
                fitted_list.append(level)

            self.level_ = float(level)
            self.trend_ = float(trend)
            fitted = np.array(fitted_list)
            self._fit_res = None

        self.fitted_values_ = fitted
        self.residuals_ = y_arr - self.fitted_values_
        df_e = max(1, n - 2)
        ss_res = np.sum(self.residuals_**2)
        self.se_ = float(np.sqrt(ss_res / df_e))
        ss_tot = np.sum((y_arr - np.mean(y_arr))**2)
        self.r2_ = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict()")
        X_arr = _ensure_1d(X)
        x_min = self.x_train_[0]
        x_max = self.x_train_[-1]
        preds = []

        for x_val in X_arr:
            if x_val < x_min:
                h = (x_val - x_min) / self.step_size_
                preds.append(self.fitted_values_[0] + h * self.trend_)
            elif x_val > x_max:
                h = (x_val - x_max) / self.step_size_
                preds.append(self.level_ + h * self.trend_)
            else:
                # Continuous interpolation between in-sample points (no discrete staircase jumps)
                val = float(np.interp(x_val, self.x_train_, self.fitted_values_))
                preds.append(val)

        return np.array(preds, dtype=float)

    def predict_interval(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute point prediction, standard error of prediction, margin of error,
        and prediction interval (PI).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict_interval()")
        X_arr = _ensure_1d(X)
        y_pred = self.predict(X_arr)

        n = self.n_samples_
        df = max(1, n - 2)
        alpha = 1.0 - confidence_level
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))

        x_max = self.x_train_[-1]
        steps_ahead = np.maximum(1.0, (X_arr - x_max) / self.step_size_)
        leverage = steps_ahead / max(1, n)
        se_pred = self.se_ * np.sqrt(1.0 + leverage)
        moe = t_crit * se_pred
        pi_low = y_pred - moe
        pi_high = y_pred + moe

        return y_pred, se_pred, moe, (pi_low, pi_high)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "final_level": self.level_,
            "final_trend": self.trend_,
            "se": self.se_,
            "r2": self.r2_,
            "n_samples": self.n_samples_,
        }

    def equation_str(self, base_year: int = 2020) -> str:
        if not self._fitted:
            return "ExponentialSmoothingModel (unfitted)"
        return f"Holt: Level = {self.level_:.4f}, Trend = {self.trend_:.4f}/step"


class MovingAverageModel(BaseModel):
    """
    Moving Average Benchmark Model (Simple Moving Average / SMA).
    Serves as an empirical baseline for time series comparison.
    """

    def __init__(self, window: int = 3):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.history_: Optional[np.ndarray] = None
        self.x_train_: Optional[np.ndarray] = None
        self.fitted_values_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.se_: Optional[float] = None
        self.r2_: Optional[float] = None
        self.n_samples_: Optional[int] = None
        self._fitted = False

    @property
    def name(self) -> str:
        return f"Moving Average (SMA-{self.window})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> MovingAverageModel:
        X_arr = _ensure_1d(X)
        y_arr = _ensure_1d(y)
        n = len(y_arr)
        if n < 1:
            raise ValueError("Need at least 1 sample for MovingAverageModel.")
        self.history_ = y_arr.copy()
        self.x_train_ = X_arr.copy()
        self.n_samples_ = n

        w = min(self.window, n)
        fitted = []
        for i in range(n):
            start = max(0, i - w + 1)
            fitted.append(float(np.mean(y_arr[start : i + 1])))
        self.fitted_values_ = np.array(fitted)
        self.residuals_ = y_arr - self.fitted_values_

        df_e = max(1, n - 1)
        ss_res = np.sum(self.residuals_**2)
        self.se_ = float(np.sqrt(ss_res / df_e))
        ss_tot = np.sum((y_arr - np.mean(y_arr))**2)
        self.r2_ = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 1.0

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict()")
        X_arr = _ensure_1d(X)
        n = len(self.history_)
        w = min(self.window, n)
        forecast_val = float(np.mean(self.history_[-w:]))

        x_min = self.x_train_[0]
        x_max = self.x_train_[-1]
        preds = []
        for x_val in X_arr:
            if x_val < x_min:
                preds.append(self.fitted_values_[0])
            elif x_val > x_max:
                preds.append(forecast_val)
            else:
                preds.append(float(np.interp(x_val, self.x_train_, self.fitted_values_)))

        return np.array(preds, dtype=float)

    def predict_interval(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute point prediction, standard error of prediction, margin of error,
        and prediction interval (PI).
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict_interval()")
        X_arr = _ensure_1d(X)
        y_pred = self.predict(X_arr)

        n = self.n_samples_
        df = max(1, n - 1)
        alpha = 1.0 - confidence_level
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))

        w = min(self.window, n)
        se_scalar = float(self.se_ * np.sqrt(1.0 + (1.0 / max(1, w))))
        moe_scalar = float(t_crit * se_scalar)

        se_pred = np.full_like(y_pred, se_scalar)
        moe = np.full_like(y_pred, moe_scalar)
        pi_low = y_pred - moe
        pi_high = y_pred + moe

        return y_pred, se_pred, moe, (pi_low, pi_high)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "window": self.window,
            "last_ma": float(np.mean(self.history_[-min(self.window, len(self.history_)) :])) if self.history_ is not None else None,
            "se": self.se_,
            "r2": self.r2_,
            "n_samples": self.n_samples_,
        }

    def equation_str(self, base_year: int = 2020) -> str:
        if not self._fitted:
            return "MovingAverageModel (unfitted)"
        w = min(self.window, len(self.history_))
        return f"SMA({w}) = Mean(Observasi {w} tahun terakhir)"
