"""
src/models/validate.py

Phase 5 entry point — complete model validation pipeline.

Runs four analyses in sequence:
    1. Standard evaluation on held-out test set (all metrics)
    2. Calibration curves (one per outcome class × model)
    3. SHAP feature importance plots
    4. Walk-forward validation on WC 2010, 2014, 2018, 2022

All outputs saved to models/metrics/.

Run:
    python -m src.models.validate
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.models.ensemble import StackingEnsemble
from src.models.evaluate import (
    evaluate_all_models,
    plot_calibration_curves,
    plot_feature_importance_comparison,
    plot_shap_summary,
)
from src.models.walk_forward import run_walk_forward_validation
from src.utils.helpers import initialize_project


def run_validation(config: dict) -> None:
    """Full Phase 5 validation pipeline."""
    processed_dir = Path(config["paths"]["processed_data"])
    models_dir    = Path(config["paths"]["models"])
    metrics_dir   = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    test_df  = pd.read_parquet(processed_dir / "test_set.parquet")
    features = pd.read_parquet(processed_dir / "features.parquet")
    matches  = pd.read_parquet(processed_dir / "matches_full.parquet")
    logger.info(f"Test set: {len(test_df):,} matches")

    ensemble = StackingEnsemble.load(models_dir, config)

    # ── 1. Standard evaluation ────────────────────────────────────────────────
    logger.info("\n[1/4] Standard evaluation on held-out test set")
    metrics_df = evaluate_all_models(ensemble, test_df)
    metrics_df.to_csv(metrics_dir / "model_metrics.csv", index=False)
    logger.info(f"Saved → {metrics_dir}/model_metrics.csv")

    # ── 2. Calibration curves ─────────────────────────────────────────────────
    logger.info("\n[2/4] Calibration curves")
    plot_calibration_curves(ensemble, test_df, metrics_dir)

    # ── 3. SHAP + feature importance ──────────────────────────────────────────
    logger.info("\n[3/4] SHAP and feature importance")
    plot_shap_summary(ensemble, test_df, metrics_dir)
    plot_feature_importance_comparison(ensemble, metrics_dir)

    # ── 4. Walk-forward validation ────────────────────────────────────────────
    logger.info("\n[4/4] Walk-forward validation on WC 2010–2022")
    wf_results = run_walk_forward_validation(config, features, matches)

    if not wf_results.empty:
        wf_results.to_csv(metrics_dir / "walk_forward_results.csv", index=False)
        logger.info(f"Saved → {metrics_dir}/walk_forward_results.csv")

    logger.info(f"\n✅ Phase 5 complete — all outputs in {metrics_dir}/")


if __name__ == "__main__":
    config = initialize_project()
    run_validation(config)