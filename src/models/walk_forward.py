"""
src/models/walk_forward.py

Walk-forward validation on past FIFA World Cups.

For each World Cup year in [2010, 2014, 2018, 2022]:
    1. Train on all data strictly before that year.
    2. Predict only that tournament's matches.
    3. Compute metrics.

This is the honest evaluation — it mirrors exactly how the model
will be used for 2026: trained on past data, predicting future matches.
It also answers the most important question you can ask of any
prediction model: how good would you have been if deployed in real time?

Group stage vs knockout breakdown:
    Knockout matches are harder to predict — single elimination, higher
    stakes, more tactical conservatism, more importance of luck. If your
    model has meaningfully better RPS on group stage than knockout rounds,
    that's expected and worth documenting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.models.ensemble import StackingEnsemble
from src.models.evaluate import compute_all_metrics, ranked_probability_score


def run_walk_forward_validation(
    config: dict,
    features: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Walk-forward validation across four historical World Cups.

    For each WC year:
        - Training set: all matches with date.year < wc_year
        - Test set: all matches with date.year == wc_year AND match_type == 'world_cup'
        - A fresh ensemble is trained from scratch on the training set

    Args:
        config:   Project configuration dict.
        features: Full feature DataFrame from features.parquet.
        matches:  Full match DataFrame from matches_full.parquet.

    Returns:
        DataFrame with one row per World Cup year.
    """
    wc_years = config["models"]["test_tournaments"]
    all_results = []

    for wc_year in wc_years:
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-forward: {wc_year} FIFA World Cup")
        logger.info(f"{'='*60}")

        # ── Split ─────────────────────────────────────────────────────────────
        train_features = features[features["date"].dt.year < wc_year].copy()
        train_matches  = matches[matches["date"].dt.year < wc_year].copy()

        # Test: only that year's World Cup matches
        # Use match_type column if available, otherwise filter by tournament name
        if "match_type" in features.columns:
            test_mask = (
                (features["date"].dt.year == wc_year)
                & (features["match_type"] == "world_cup")
            )
        else:
            test_mask = features["date"].dt.year == wc_year

        test_features = features[test_mask].copy()

        logger.info(f"  Train: {len(train_features):,} matches (through {wc_year - 1})")
        logger.info(f"  Test : {len(test_features)} World Cup {wc_year} matches")

        if len(test_features) < 5:
            logger.warning(f"  Too few test matches ({len(test_features)}) — skipping {wc_year}")
            continue

        if len(train_features) < 100:
            logger.warning(f"  Insufficient training data ({len(train_features)}) — skipping {wc_year}")
            continue

        # ── Train fresh ensemble ──────────────────────────────────────────────
        ensemble = StackingEnsemble(config)
        try:
            ensemble.fit(train_features, train_matches)
        except Exception as exc:
            logger.error(f"  Training failed for {wc_year}: {exc}")
            continue

        # ── Evaluate ─────────────────────────────────────────────────────────
        y_true  = test_features["outcome"].values
        y_proba = ensemble.predict_proba(test_features)

        metrics = compute_all_metrics(y_true, y_proba, model_name=f"wc_{wc_year}")
        metrics["year"]         = wc_year
        metrics["n_wc_matches"] = len(test_features)

        # Per-model breakdown for this WC
        for name, model in ensemble.base_models_.items():
            m = compute_all_metrics(y_true, model.predict_proba(test_features), name)
            metrics[f"rps_{name}"] = m["rps"]

        dc_proba = ensemble._get_dc_predictions_batch(test_features)
        metrics["rps_dixon_coles"] = round(
            ranked_probability_score(y_true, dc_proba), 4
        )

        # Stage breakdown if available
        if "match_stage" in test_features.columns:
            stage_metrics = _stage_breakdown(test_features, y_true, y_proba)
            metrics.update(stage_metrics)

        all_results.append(metrics)

        logger.info(
            f"  Accuracy={metrics['accuracy']:.3f} | "
            f"RPS={metrics['rps']:.4f} | "
            f"Log Loss={metrics['log_loss']:.4f}"
        )

    if not all_results:
        logger.warning("No walk-forward results — check your match_type column.")
        return pd.DataFrame()

    summary = pd.DataFrame(all_results)

    # Summary statistics across all WC years
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD VALIDATION SUMMARY")
    logger.info("=" * 70)
    display_cols = ["year", "n_wc_matches", "accuracy", "rps", "brier", "log_loss"]
    logger.info(summary[display_cols].to_string(index=False))
    logger.info(f"\nMean RPS (all WCs) : {summary['rps'].mean():.4f}")
    logger.info(f"Best WC (lowest RPS): {summary.loc[summary['rps'].idxmin(), 'year']}")
    logger.info(f"Worst WC (highest RPS): {summary.loc[summary['rps'].idxmax(), 'year']}")
    logger.info("=" * 70)

    return summary


def _stage_breakdown(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """
    Compute per-stage accuracy and RPS (group stage vs knockout).

    Returns an empty dict if 'match_stage' column is missing or empty.
    """
    result = {}
    for stage in test_df["match_stage"].dropna().unique():
        mask = (test_df["match_stage"] == stage).values
        if mask.sum() < 3:
            continue
        key = stage.lower().replace(" ", "_").replace("-", "_")
        rps = ranked_probability_score(y_true[mask], y_proba[mask])
        acc = (np.argmax(y_proba[mask], axis=1) == y_true[mask]).mean()
        result[f"rps_{key}"]      = round(rps, 4)
        result[f"accuracy_{key}"] = round(float(acc), 4)
    return result