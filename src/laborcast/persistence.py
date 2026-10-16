"""
Model artifact persistence and metadata serialization module for LaborCast.
Supports joblib and pickle binary export alongside human-readable JSON schema metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Tuple, Union
import warnings
import joblib

from laborcast.models import BaseModel


def _resolve_artifact_path(filepath: Union[str, Path]) -> Path:
    """Resolve artifact path checking cwd, parent dir, or package repository root."""
    p = Path(filepath)
    if p.is_file():
        return p
    p_parent = Path.cwd().parent / filepath
    if p_parent.is_file():
        return p_parent
    repo_root = Path(__file__).resolve().parent.parent.parent
    p_root = repo_root / filepath
    if p_root.is_file():
        return p_root
    return p


class ModelPersistence:
    """Handles serializing and deserializing LaborCast models and run metadata using joblib/pickle."""

    @classmethod
    def save(
        cls,
        model: BaseModel,
        metadata: Dict[str, Any],
        model_path: Union[str, Path] = "experiments/artifacts/laborcast_model.joblib",
        metadata_path: Union[str, Path] = "experiments/artifacts/laborcast_metadata.json",
        format: str = "auto",
    ) -> Tuple[Path, Path]:
        """
        Save model to binary (joblib or pickle) and metadata to JSON.

        Args:
            model: Trained BaseModel instance.
            metadata: Run metadata dictionary.
            model_path: Destination binary path.
            metadata_path: Destination JSON metadata path.
            format: 'auto', 'joblib', or 'pickle'.
        """
        m_path = Path(model_path)
        meta_path = Path(metadata_path)

        m_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        chosen_format = format.lower()
        if chosen_format == "auto":
            chosen_format = "pickle" if m_path.suffix.lower() in [".pkl", ".pickle"] else "joblib"

        # 1. Save model binary
        if chosen_format == "pickle":
            with open(m_path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            joblib.dump(model, m_path)

        # 2. Enrich and save metadata
        meta_to_save = dict(metadata)
        meta_to_save["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
        meta_to_save["model_type"] = type(model).__name__
        meta_to_save["model_name"] = model.name
        meta_to_save["serialization_format"] = chosen_format
        meta_to_save["model_params"] = model.get_params()

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_to_save, f, indent=2, ensure_ascii=False)

        return m_path, meta_path

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path] = "experiments/artifacts/laborcast_model.joblib",
        metadata_path: Optional[Union[str, Path]] = "experiments/artifacts/laborcast_metadata.json",
    ) -> Tuple[BaseModel, Optional[Dict[str, Any]]]:
        """
        Load model and optional metadata.
        """
        import sys

        src_dir = Path(__file__).resolve().parent.parent
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        m_path = _resolve_artifact_path(model_path)
        if not m_path.is_file():
            raise FileNotFoundError(f"Model file not found: {m_path.resolve()}")

        suffix = m_path.suffix.lower()
        if suffix in [".pkl", ".pickle"]:
            with open(m_path, "rb") as f:
                model = pickle.load(f)
        else:
            # Filter upstream NumPy 2.5 DeprecationWarning inside joblib.numpy_pickle
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="joblib")
                warnings.filterwarnings("ignore", message=".*Setting the shape on a NumPy array has been deprecated.*")
                model = joblib.load(m_path)

        metadata = None
        if metadata_path is not None:
            meta_path = _resolve_artifact_path(metadata_path)
            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

        return model, metadata
