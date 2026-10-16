#!/usr/bin/env python3
"""
Pipeline Machine Learning: Peramalan Tingkat Pengangguran Terbuka (TPT) Nasional
Proyek: PSAJ Matematika Terintegrasi PPLG SMK Telkom Purwokerto
Engine: LaborCast ML Production Suite
Target Prediksi: Tahun 2026 (1-Step Ahead Forecast) & Proyeksi Jangka Menengah (2026-2030)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure src/ is accessible in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from laborcast.inference import LaborCastInference  # noqa: E402
from laborcast.persistence import ModelPersistence  # noqa: E402
from laborcast.pipeline import LaborCastPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LaborCast ML Pipeline: Peramalan Tingkat Pengangguran Terbuka (TPT) Nasional"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="experiments/datasets/tpt_nasional_timeseries_2015_2025.csv",
        help="Path to historical Sakernas TPT CSV file.",
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=2026,
        help="Target year for 1-step ahead forecasting (default: 2026).",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=2020,
        help="Base year for relative temporal indexing (default: 2020).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="experiments/artifacts",
        help="Output directory for saved model binaries and JSON metadata.",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default="assets",
        help="Output directory for publication figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = LaborCastPipeline(
        base_year=args.base_year,
        nairu_floor=3.50,
        artifacts_dir=args.artifacts_dir,
        assets_dir=args.assets_dir,
    )

    # Run the full pipeline
    result = pipeline.run(
        dataset_path=args.dataset,
        target_year=args.target_year,
        export_visualizations=True,
        export_artifacts=True,
    )

    # Print formatted terminal report
    print(result.summary_text())

    # Verify model artifact persistence by loading back and running inference
    print("\n[VERIFIKASI INTEGRITAS ARTIFAK PERSISTENCE]")
    loaded_model, loaded_meta = ModelPersistence.load(
        model_path=result.model_artifact_path,
        metadata_path=result.metadata_artifact_path,
    )
    loaded_engine = LaborCastInference(
        model=loaded_model,
        base_year=result.preprocessed_data.base_year,
        seasonal_gap=result.preprocessed_data.seasonal_gap,
        nairu_floor=3.50,
    )
    test_fc = loaded_engine.predict_target_year(args.target_year)
    print(f"  * Status: SUKSES MEMUAT DARI DISK ({result.model_artifact_path})")
    print(f"  * Verifikasi Prediksi Ulang 2026: {test_fc.point_forecast:.4f}% (Agustus)")
    print(f"  * Metadata Timestamp UTC: {loaded_meta.get('exported_at_utc') if loaded_meta else 'N/A'}")
    print("\n[PIPELINE SELESAI DENGAN SUKSES]")


if __name__ == "__main__":
    main()
