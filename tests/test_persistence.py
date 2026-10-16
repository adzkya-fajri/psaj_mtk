"""
Unit tests for laborcast.persistence module.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from laborcast.models import LinearTrendModel
from laborcast.persistence import ModelPersistence


def test_save_and_load_roundtrip(tmp_path):
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([7.07, 6.49, 5.86, 5.32, 4.91, 4.85], dtype=float)
    model = LinearTrendModel().fit(X, y)

    meta = {
        "project": "LaborCast",
        "target_year": 2026,
        "sample_size": 6,
    }

    model_file = tmp_path / "test_model.joblib"
    meta_file = tmp_path / "test_metadata.json"

    saved_m, saved_meta = ModelPersistence.save(
        model=model,
        metadata=meta,
        model_path=model_file,
        metadata_path=meta_file,
    )

    assert saved_m.is_file()
    assert saved_meta.is_file()

    loaded_model, loaded_meta = ModelPersistence.load(model_file, meta_file)

    # Check model functionality
    x_test = np.array([[6], [7]])
    orig_preds = model.predict(x_test)
    loaded_preds = loaded_model.predict(x_test)

    assert np.allclose(orig_preds, loaded_preds)
    assert loaded_meta is not None
    assert loaded_meta["project"] == "LaborCast"
    assert loaded_meta["model_type"] == "LinearTrendModel"


def test_save_and_load_pickle_roundtrip(tmp_path):
    X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
    y = np.array([7.07, 6.49, 5.86, 5.32, 4.91, 4.85], dtype=float)
    model = LinearTrendModel().fit(X, y)

    meta = {"project": "LaborCast", "target_year": 2026}
    model_file = tmp_path / "test_model.pkl"
    meta_file = tmp_path / "test_metadata.json"

    saved_m, saved_meta = ModelPersistence.save(
        model=model,
        metadata=meta,
        model_path=model_file,
        metadata_path=meta_file,
        format="pickle",
    )

    assert saved_m.is_file()
    assert saved_meta.is_file()

    loaded_model, loaded_meta = ModelPersistence.load(model_file, meta_file)
    assert np.allclose(model.predict([[6]]), loaded_model.predict([[6]]))
    assert loaded_meta["serialization_format"] == "pickle"


def test_load_non_existent():
    with pytest.raises(FileNotFoundError):
        ModelPersistence.load("non_existent_file.joblib")
