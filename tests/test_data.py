"""
Unit tests for laborcast.data module.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure src/ is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from laborcast.data import DataLoader, DataPreprocessor, DatasetValidator


@pytest.fixture
def sample_valid_df():
    return pd.DataFrame(
        {
            "Tahun": [2020, 2021, 2022, 2023, 2024, 2025],
            "TPT_Februari": [4.94, 6.26, 5.83, 5.45, 4.82, 4.76],
            "TPT_Agustus": [7.07, 6.49, 5.86, 5.32, 4.91, 4.85],
        }
    )


def test_dataset_validator_success(sample_valid_df):
    # Should not raise any exception
    DatasetValidator.validate(sample_valid_df, require_february=True)


def test_dataset_validator_non_df():
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        DatasetValidator.validate([1, 2, 3])  # type: ignore


def test_dataset_validator_empty():
    with pytest.raises(ValueError, match="Dataset is empty"):
        DatasetValidator.validate(pd.DataFrame())


def test_dataset_validator_missing_column():
    bad_df = pd.DataFrame({"Tahun": [2020, 2021]})
    with pytest.raises(ValueError, match="Missing required column"):
        DatasetValidator.validate(bad_df)


def test_dataset_validator_non_monotonic_years():
    bad_df = pd.DataFrame(
        {
            "Tahun": [2021, 2020, 2022],
            "TPT_Agustus": [6.0, 5.5, 5.0],
        }
    )
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        DatasetValidator.validate(bad_df)


def test_dataset_validator_invalid_unemployment_bounds():
    bad_df = pd.DataFrame(
        {
            "Tahun": [2020, 2021],
            "TPT_Agustus": [7.0, 105.0],  # Out of bounds > 100
        }
    )
    with pytest.raises(ValueError, match="bounded between 0 and 100"):
        DatasetValidator.validate(bad_df)


def test_dataset_validator_null_values():
    bad_df = pd.DataFrame(
        {
            "Tahun": [2020, 2021],
            "TPT_Agustus": [7.0, np.nan],
        }
    )
    with pytest.raises(ValueError, match="contains NaN/null"):
        DatasetValidator.validate(bad_df)


def test_data_loader_real_dataset():
    loader = DataLoader()
    dataset_path = Path("experiments/datasets/tpt_nasional_timeseries_2015_2025.csv")
    df = loader.load_national_timeseries(dataset_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 11
    assert "TPT_Agustus" in df.columns
    assert "TPT_Februari" in df.columns


def test_data_loader_missing_file():
    loader = DataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_national_timeseries("non_existent_path.csv")


def test_data_preprocessor_post_pandemic(sample_valid_df):
    preprocessor = DataPreprocessor(base_year=2020)
    prep = preprocessor.prepare_features(sample_valid_df, filter_post_pandemic=True)

    assert prep.base_year == 2020
    assert len(prep.years) == 6
    assert np.array_equal(prep.years, np.array([2020, 2021, 2022, 2023, 2024, 2025]))
    assert prep.X.shape == (6, 1)
    assert np.array_equal(prep.X.ravel(), np.array([0, 1, 2, 3, 4, 5]))
    assert prep.y.shape == (6,)
    assert prep.seasonal_gap > 0.0


def test_data_preprocessor_custom_base_year(sample_valid_df):
    preprocessor = DataPreprocessor(base_year=2015)
    prep = preprocessor.prepare_features(sample_valid_df, filter_post_pandemic=True)

    assert prep.base_year == 2015
    assert np.array_equal(prep.X.ravel(), np.array([5, 6, 7, 8, 9, 10]))


def test_data_preprocessor_with_temporal_gaps():
    df_gap = pd.DataFrame(
        {
            "Tahun": [2020, 2021, 2023, 2025],
            "TPT_Februari": [4.94, 6.26, 5.45, 4.76],
            "TPT_Agustus": [7.07, 6.49, 5.32, 4.85],
        }
    )
    preprocessor = DataPreprocessor(base_year=2020)
    prep = preprocessor.prepare_features(df_gap, filter_post_pandemic=True)

    assert len(prep.years) == 4
    assert np.array_equal(prep.X.ravel(), np.array([0, 1, 3, 5]))
    assert len(prep.y) == 4


def test_data_preprocessor_full_series():
    loader = DataLoader()
    df = loader.load_national_timeseries()
    preprocessor = DataPreprocessor(base_year=2020)
    prep = preprocessor.prepare_features(df, filter_post_pandemic=False)

    assert len(prep.years) == 11
    assert prep.X[0, 0] == -5.0  # 2015 - 2020
    assert prep.X[-1, 0] == 5.0  # 2025 - 2020


def test_data_loader_provincial_datasets():
    loader = DataLoader()
    df_tpt = loader.load_provincial_tpt()
    assert len(df_tpt) == 38
    assert "Provinsi" in df_tpt.columns
    assert "TPT_Agustus" in df_tpt.columns
    assert "TPT_Februari" in df_tpt.columns
    assert "INDONESIA" not in df_tpt["Provinsi"].values

    df_edu = loader.load_provincial_education()
    assert len(df_edu) >= 38
    assert "SMA_Rata_Rata" in df_edu.columns

    merged = loader.load_provincial_merged()
    assert len(merged) == 38
    assert "TPT_Agustus" in merged.columns
    assert "SMA_Rata_Rata" in merged.columns
    # Check positive correlation between educational attainment and TPT
    corr = merged["SMA_Rata_Rata"].corr(merged["TPT_Agustus"])
    assert 0.5 < corr < 0.8
