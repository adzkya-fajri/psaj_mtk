"""
Data ingestion, validation, and feature preprocessing module for LaborCast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessedData:
    """Dataclass holding preprocessed data and metadata."""

    df_raw: pd.DataFrame
    df_filtered: pd.DataFrame
    years: np.ndarray
    X: np.ndarray  # 2D array: shape (n, 1) of relative years (year - base_year)
    y: np.ndarray  # 1D array: shape (n,) of target unemployment rates (%)
    base_year: int
    target_col: str
    seasonal_gap: float  # Average seasonal differential: TPT_Agustus - TPT_Februari


class DatasetValidator:
    """Validates schema, data types, and temporal monotonicity for TPT datasets."""

    REQUIRED_COLUMNS: Sequence[str] = ("Tahun", "TPT_Agustus")

    @classmethod
    def validate(cls, df: pd.DataFrame, require_february: bool = False) -> None:
        """
        Validate dataset integrity.

        Raises:
            TypeError: if df is not a pandas DataFrame.
            ValueError: if columns are missing, data is empty, or values violate domain constraints.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Input must be a pandas DataFrame, got {type(df).__name__}"
            )

        if df.empty:
            raise ValueError("Dataset is empty.")

        # Check required columns
        for col in cls.REQUIRED_COLUMNS:
            if col not in df.columns:
                raise ValueError(
                    f"Missing required column: '{col}'. Available: {list(df.columns)}"
                )

        if require_february and "TPT_Februari" not in df.columns:
            raise ValueError("Missing required column: 'TPT_Februari'.")

        # Check null values in key columns
        cols_to_check = list(cls.REQUIRED_COLUMNS)
        if "TPT_Februari" in df.columns:
            cols_to_check.append("TPT_Februari")

        for col in cols_to_check:
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}' contains NaN/null values.")

        # Validate Tahun column
        if not pd.api.types.is_numeric_dtype(df["Tahun"]):
            raise TypeError("Column 'Tahun' must be numeric.")

        years = df["Tahun"].to_numpy()
        if len(years) < 2:
            raise ValueError("Dataset must contain at least 2 time points.")

        # Check strictly increasing chronological order
        if not np.all(np.diff(years) > 0):
            raise ValueError(
                "Column 'Tahun' must be strictly monotonically increasing without duplicates."
            )

        # Validate target unemployment rate (must be between 0% and 100%)
        tpt_aug = df["TPT_Agustus"].to_numpy()
        if not pd.api.types.is_numeric_dtype(df["TPT_Agustus"]):
            raise TypeError("Column 'TPT_Agustus' must be numeric.")

        if np.any((tpt_aug < 0.0) | (tpt_aug > 100.0)):
            raise ValueError(
                "Unemployment rates in 'TPT_Agustus' must be bounded between 0 and 100."
            )

        if "TPT_Februari" in df.columns:
            if not pd.api.types.is_numeric_dtype(df["TPT_Februari"]):
                raise TypeError("Column 'TPT_Februari' must be numeric.")
            tpt_feb = df["TPT_Februari"].to_numpy()
            if np.any((tpt_feb < 0.0) | (tpt_feb > 100.0)):
                raise ValueError(
                    "Unemployment rates in 'TPT_Februari' must be bounded between 0 and 100."
                )


def _resolve_data_path(filepath: Union[str, Path]) -> Path:
    """Resolve data filepath checking current directory, parent directory, or repository root."""
    p = Path(filepath)
    if p.is_file():
        return p
    # Try relative to parent if running inside a subfolder like experiments/
    p_parent = Path.cwd().parent / filepath
    if p_parent.is_file():
        return p_parent
    # If filepath starts with experiments/ and cwd is experiments
    if Path.cwd().name == "experiments" and str(filepath).startswith("experiments/"):
        p_sub = Path.cwd() / str(filepath).replace("experiments/", "", 1)
        if p_sub.is_file():
            return p_sub
    # Try finding relative to package repository root
    pkg_root = Path(__file__).resolve().parent.parent.parent
    p_root = pkg_root / filepath
    if p_root.is_file():
        return p_root
    return p


class DataLoader:
    """Responsible for reading and ingesting Sakernas TPT datasets."""

    def __init__(self, validator: type[DatasetValidator] = DatasetValidator):
        self.validator = validator

    def load_national_timeseries(
        self,
        filepath: Union[
            str, Path
        ] = "experiments/datasets/tpt_nasional_timeseries_2015_2025.csv",
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Load Sakernas national time series CSV file and optionally run validation.
        """
        path = _resolve_data_path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

        try:
            df = pd.read_csv(path)
        except Exception as err:
            raise ValueError(f"Failed to parse CSV file at {path}: {err}") from err

        if validate:
            self.validator.validate(df)

        return df

    def load_provincial_tpt(
        self,
        filepath: Union[
            str, Path
        ] = "experiments/datasets/Tingkat Pengangguran Terbuka Menurut Provinsi, 2025 (dataset Y).csv",
    ) -> pd.DataFrame:
        """
        Load and parse Sakernas BPS 38 provinces unemployment dataset (Dataset Y).
        """
        path = _resolve_data_path(filepath)
        if not path.is_file():
            raise FileNotFoundError(
                f"Provincial TPT dataset not found at: {path.resolve()}"
            )

        try:
            df = pd.read_csv(path, skiprows=3)
        except Exception as err:
            raise ValueError(
                f"Failed to parse provincial CSV file at {path}: {err}"
            ) from err

        df.columns = ["Provinsi", "TPT_Februari", "TPT_Agustus", "Tahunan"]
        df["Provinsi"] = df["Provinsi"].astype(str).str.strip()
        df = df[
            ~df["Provinsi"].str.upper().isin(["INDONESIA", "38 PROVINSI", "NAN", ""])
        ].dropna(subset=["Provinsi"])
        df["TPT_Februari"] = pd.to_numeric(df["TPT_Februari"], errors="coerce")
        df["TPT_Agustus"] = pd.to_numeric(df["TPT_Agustus"], errors="coerce")
        df = df.dropna(subset=["TPT_Agustus"]).reset_index(drop=True)
        return df[["Provinsi", "TPT_Februari", "TPT_Agustus"]]

    def load_provincial_education(
        self,
        filepath: Union[
            str, Path
        ] = "experiments/datasets/Persentase penduduk usia 25 Tahun keatas dengan pendidikan SMA ke atas menurut jenis kelamin, 2025 (Dataset X).csv",
    ) -> pd.DataFrame:
        """
        Load and parse BPS educational attainment dataset (Dataset X) for provinces.
        """
        path = _resolve_data_path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Education dataset not found at: {path.resolve()}")

        try:
            df = pd.read_csv(path, skiprows=3)
        except Exception as err:
            raise ValueError(
                f"Failed to parse education CSV file at {path}: {err}"
            ) from err

        df.columns = ["Wilayah", "SMA_Laki_Laki", "SMA_Perempuan"]
        df["Wilayah"] = df["Wilayah"].astype(str).str.strip()
        df["SMA_Laki_Laki"] = pd.to_numeric(df["SMA_Laki_Laki"], errors="coerce")
        df["SMA_Perempuan"] = pd.to_numeric(df["SMA_Perempuan"], errors="coerce")

        name_map = {
            "D I YOGYAKARTA": "DI YOGYAKARTA",
            "KEPULAUAN RIAU": "KEP. RIAU",
        }
        df["Wilayah"] = df["Wilayah"].replace(name_map)
        df["SMA_Rata_Rata"] = (df["SMA_Laki_Laki"] + df["SMA_Perempuan"]) / 2.0
        return df.rename(columns={"Wilayah": "Provinsi"})

    def load_provincial_merged(
        self,
        tpt_path: Union[
            str, Path
        ] = "experiments/datasets/Tingkat Pengangguran Terbuka Menurut Provinsi, 2025 (dataset Y).csv",
        edu_path: Union[
            str, Path
        ] = "experiments/datasets/Persentase penduduk usia 25 Tahun keatas dengan pendidikan SMA ke atas menurut jenis kelamin, 2025 (Dataset X).csv",
    ) -> pd.DataFrame:
        """
        Merge 38 provinces TPT and educational attainment into a unified cross-sectional DataFrame.
        """
        df_tpt = self.load_provincial_tpt(tpt_path)
        df_edu = self.load_provincial_education(edu_path)
        return pd.merge(df_tpt, df_edu, on="Provinsi", how="inner")


class DataPreprocessor:
    """Preprocesses time series data and builds model features."""

    def __init__(self, base_year: int = 2020):
        self.base_year = base_year

    def prepare_features(
        self,
        df: pd.DataFrame,
        filter_post_pandemic: bool = True,
        start_year: int = 2020,
        target_col: str = "TPT_Agustus",
    ) -> PreprocessedData:
        """
        Transform raw time series dataframe into model features.

        Args:
            df: Validated pandas DataFrame with Tahun and target column.
            filter_post_pandemic: Whether to isolate post-pandemic regime (default True).
            start_year: Starting year for post-pandemic regime (default 2020).
            target_col: Target column name (default 'TPT_Agustus').

        Returns:
            PreprocessedData instance.
        """
        if filter_post_pandemic:
            df_filtered = df[df["Tahun"] >= start_year].copy().reset_index(drop=True)
            if df_filtered.empty:
                raise ValueError(f"No records found with Tahun >= {start_year}")
        else:
            df_filtered = df.copy().reset_index(drop=True)

        years = df_filtered["Tahun"].to_numpy(dtype=int)
        X = (years - self.base_year).reshape(-1, 1).astype(float)
        y = df_filtered[target_col].to_numpy(dtype=float)

        # Compute seasonal differential: Agustus - Februari
        # Using normal years (excluding 2020 COVID shock which had an abnormal 2.13% surge in August)
        if "TPT_Februari" in df.columns:
            non_shock_df = df[df["Tahun"] != 2020]
            if not non_shock_df.empty and "TPT_Agustus" in non_shock_df.columns:
                diffs = non_shock_df["TPT_Agustus"] - non_shock_df["TPT_Februari"]
                seasonal_gap = float(np.mean(diffs))
            else:
                seasonal_gap = 0.09
        else:
            seasonal_gap = 0.09

        return PreprocessedData(
            df_raw=df,
            df_filtered=df_filtered,
            years=years,
            X=X,
            y=y,
            base_year=self.base_year,
            target_col=target_col,
            seasonal_gap=round(seasonal_gap, 4),
        )
