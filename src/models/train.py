"""
src/models/train.py

Phase 4 training pipeline.

Loads features.parquet and matches_full.parquet, applies a temporal
train/test split, trains the full stacking ensemble, saves all models,
and prints a quick accuracy sanity check.

Key design: temporal split (not random).
  - Random split: a 2019 match in the test set uses Elo ratings
    computed from 2020 data in the training set. This is leakage.
  - Temporal split: everything before holdout_year trains, everything
    from holdout_year onward tests. This mirrors real deployment.

Run:
    python -m src.models.train
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.ensemble import StackingEnsemble
from src.utils.helpers import initialize_project


# ─────────────────────────────────────────────────────────────────────────────
# Temporal split
# ─────────────────────────────────────────────────────────────────────────────

def temporal_train_test_split(
    features: pd.DataFrame,
    holdout_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the feature matrix by calendar year, not randomly.

    Args:
        features:     Full feature DataFrame (must have a 'date' column).
        holdout_year: First year held out for testing.
                      Everything before this year = training set.

    Returns:
        (train_df, test_df)
    """
    train_mask = features["date"].dt.year < holdout_year
    train = features[train_mask].copy()
    test  = features[~train_mask].copy()

    logger.info(f"Temporal split — holdout from {holdout_year}:")
    logger.info(
        f"  Train : {len(train):,} matches "
        f"({features.loc[train_mask, 'date'].min().date()} → "
        f"{features.loc[train_mask, 'date'].max().date()})"
    )
    logger.info(
        f"  Test  : {len(test):,} matches "
        f"({features.loc[~train_mask, 'date'].min().date()} → "
        f"{features.loc[~train_mask, 'date'].max().date()})"
    )
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_training(config: dict) -> StackingEnsemble:
    """
    Full Phase 4 training pipeline.

    Steps:
      1. Load features.parquet and matches_full.parquet.
      2. Apply temporal train/test split.
      3. Save splits to disk (test set used in Phase 5).
      4. Train the stacking ensemble.
      5. Save all fitted models.
      6. Log a quick test-set accuracy check.

    Returns:
        Fitted StackingEnsemble.
    """
    processed_dir = Path(config["paths"]["processed_data"])
    models_dir    = Path(config["paths"]["models"])
    holdout_year  = config["models"]["holdout_year"]

    # ── Load ──────────────────────────────────────────────────────────────────
    features_path = processed_dir / "features.parquet"
    matches_path  = processed_dir / "matches_full.parquet"

    for path in [features_path, matches_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run Phases 1-3 first:\n"
                "  python -m src.data.download\n"
                "  python -m src.data.preprocess\n"
                "  python -m src.features.build_features"
            )

    features = pd.read_parquet(features_path)
    matches  = pd.read_parquet(matches_path)
    logger.info(f"Loaded features.parquet  : {len(features):,} rows, {features.shape[1]} cols")
    logger.info(f"Loaded matches_full.parquet : {len(matches):,} rows")

    # ── Split ─────────────────────────────────────────────────────────────────
    train_df, test_df = temporal_train_test_split(features, holdout_year)

    train_df.to_parquet(processed_dir / "train_set.parquet", index=False)
    test_df.to_parquet( processed_dir / "test_set.parquet",  index=False)
    logger.info("Saved train_set.parquet and test_set.parquet")

    # Filter matches to training period only (for Dixon-Coles)
    train_matches = matches[matches["date"].dt.year < holdout_year].copy()
    logger.info(f"Training matches for Dixon-Coles: {len(train_matches):,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    ensemble = StackingEnsemble(config)
    ensemble.fit(train_df, train_matches)

    # ── Save ──────────────────────────────────────────────────────────────────
    ensemble.save(models_dir)

    # ── Sanity check ──────────────────────────────────────────────────────────
    y_test     = test_df["outcome"].values
    y_pred     = ensemble.predict(test_df)
    y_proba    = ensemble.predict_proba(test_df)
    accuracy   = (y_pred == y_test).mean()
    outcome_dist = np.bincount(y_test, minlength=3) / len(y_test)

    logger.info("\n" + "─" * 50)
    logger.info("TRAINING COMPLETE — quick test-set check")
    logger.info(f"  Accuracy        : {accuracy:.3f}")
    logger.info(f"  Outcome dist    : HW={outcome_dist[0]:.3f} D={outcome_dist[1]:.3f} AW={outcome_dist[2]:.3f}")
    logger.info("  (Full evaluation with proper metrics → run Phase 5)")
    logger.info("─" * 50)

    return ensemble


if __name__ == "__main__":
    config = initialize_project()
    run_training(config)