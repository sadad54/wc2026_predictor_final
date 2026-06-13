"""
src/models/evaluate.py

Phase 5 evaluation: metrics, calibration curves, SHAP plots,
and feature importance comparison.

Metrics implemented:
    Accuracy         — fraction of correctly predicted outcomes
    Log Loss         — penalises confident wrong predictions heavily
    Brier Score      — mean squared error of probability predictions
    RPS              — Ranked Probability Score (gold standard for football)
    ROC-AUC          — discrimination ability (one-vs-rest, macro-averaged)

Why RPS is the standard:
    Unlike accuracy (ignores probability) or log loss (treats all outcomes
    as independent), RPS respects the natural ordering Win > Draw > Loss
    and rewards being right AND being right by a lot.

    Reference: Constantinou & Fenton (2012), "Solving the problem of inadequate
    scoring rules for assessing probabilistic football forecast models."

Calibration curve interpretation:
    X-axis: what the model said (predicted probability)
    Y-axis: what actually happened (observed frequency)
    Diagonal = perfect calibration.
    Curve above diagonal = model is under-confident (could be bolder).
    Curve below diagonal = model is over-confident (hedge more).
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers and CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from loguru import logger

from src.models.ensemble import StackingEnsemble
from src.models.outcome_models import FEATURE_COLS


# ─────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────────────────────

def ranked_probability_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """
    Ranked Probability Score (RPS) — the gold standard for football prediction.

    Measures the squared distance between the cumulative predicted probability
    distribution and the cumulative true distribution, averaged over all outcome
    classes and all matches.

    Formula for K=3 outcomes:
        RPS_i = (1/(K-1)) * Σ_{k=1}^{K-1} (CDF_pred_ik − CDF_true_ik)²
        RPS = mean(RPS_i) over all matches i

    Properties:
        - Perfect model: RPS = 0.0
        - Uniform (1/3, 1/3, 1/3) baseline: ~0.333
        - Lower is better
        - Respects ordering: predicting "draw" when outcome is "home win"
          is penalised less than predicting "away win"

    Args:
        y_true:  (n,) integer labels (0=HW, 1=D, 2=AW).
        y_proba: (n, 3) predicted probability matrix.

    Returns:
        float — mean RPS across all n matches.
    """
    n_classes = y_proba.shape[1]

    # One-hot encode true outcomes
    y_onehot = np.zeros_like(y_proba)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0

    # Cumulative sums along outcome axis
    cum_pred = np.cumsum(y_proba,  axis=1)
    cum_true = np.cumsum(y_onehot, axis=1)

    # RPS per match: mean squared diff of cumulative distributions
    # We only sum the first K-1 terms (the last term cancels to 0)
    rps_per_match = np.mean((cum_pred[:, :-1] - cum_true[:, :-1]) ** 2, axis=1)

    return float(rps_per_match.mean())


def multiclass_brier_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """
    Multiclass Brier score.

    Mean squared error between the predicted probability vector and the
    one-hot encoded true outcome. Equivalent to averaging three binary
    Brier scores (one per outcome class).

    Range: [0, 2]. Lower is better. Perfect model = 0.
    A model that always predicts (1/3, 1/3, 1/3) scores ~0.667.

    Args:
        y_true:  (n,) integer labels.
        y_proba: (n, 3) probability matrix.

    Returns:
        float — mean multiclass Brier score.
    """
    y_onehot = np.zeros_like(y_proba)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
) -> dict:
    """
    Compute the full suite of evaluation metrics for one model.

    Args:
        y_true:     True outcome labels.
        y_proba:    (n, 3) predicted probabilities.
        model_name: String identifier included in the result dict.

    Returns:
        Dictionary with all metric values.
    """
    y_pred = np.argmax(y_proba, axis=1)

    metrics = {
        "model":     model_name,
        "n_matches": int(len(y_true)),
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "log_loss":  round(float(log_loss(y_true, y_proba)), 4),
        "brier":     round(float(multiclass_brier_score(y_true, y_proba)), 4),
        "rps":       round(float(ranked_probability_score(y_true, y_proba)), 4),
        "roc_auc":   round(float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        ), 4),
    }

    # Per-outcome Brier score — shows where the model struggles most
    for idx, label in enumerate(["home_win", "draw", "away_win"]):
        y_bin = (y_true == idx).astype(int)
        metrics[f"brier_{label}"] = round(
            float(brier_score_loss(y_bin, y_proba[:, idx])), 4
        )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_models(
    ensemble: StackingEnsemble,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate the ensemble and all individual base models on the held-out test set.

    Also evaluates a naive baseline (always predict historical averages)
    so you have a floor to compare against.

    Args:
        ensemble: Fitted StackingEnsemble.
        test_df:  Held-out test DataFrame (must have 'outcome' column).

    Returns:
        DataFrame of metrics, one row per model, sorted by RPS ascending.
    """
    y_true = test_df["outcome"].values
    results = []

    # Stacking ensemble
    logger.info("Evaluating ensemble...")
    results.append(compute_all_metrics(
        y_true, ensemble.predict_proba(test_df), "ensemble"
    ))

    # Individual base models
    for name, model in ensemble.base_models_.items():
        logger.info(f"Evaluating {name}...")
        results.append(compute_all_metrics(
            y_true, model.predict_proba(test_df), name
        ))

    # Dixon-Coles
    logger.info("Evaluating dixon_coles...")
    results.append(compute_all_metrics(
        y_true, ensemble._get_dc_predictions_batch(test_df), "dixon_coles"
    ))

    # Naive baseline: predict historical class frequencies
    hw_rate, d_rate, aw_rate = [(y_true == i).mean() for i in range(3)]
    naive_proba = np.tile([hw_rate, d_rate, aw_rate], (len(y_true), 1))
    results.append(compute_all_metrics(y_true, naive_proba, "naive_baseline"))

    metrics_df = pd.DataFrame(results).sort_values("rps")

    # Print formatted comparison table
    display_cols = ["model", "n_matches", "accuracy", "rps", "brier", "log_loss", "roc_auc"]
    logger.info("\n" + "=" * 75)
    logger.info("MODEL EVALUATION — held-out test set (sorted by RPS ↑ = better)")
    logger.info("=" * 75)
    logger.info(metrics_df[display_cols].to_string(index=False))
    logger.info("=" * 75)
    logger.info("RPS baseline (uniform): ~0.333")

    return metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_curves(
    ensemble: StackingEnsemble,
    test_df: pd.DataFrame,
    output_dir: Path,
    n_bins: int = 10,
) -> None:
    """
    Plot calibration curves for all models, one panel per outcome class.

    Reading the curves:
        Diagonal = perfect calibration
        Below diagonal = over-confident (predicts 0.7 but team only wins 0.5)
        Above diagonal = under-confident (predicts 0.3 but team actually wins 0.4)

    Args:
        ensemble:   Fitted StackingEnsemble.
        test_df:    Test DataFrame with 'outcome' column.
        output_dir: Directory where PNG is saved.
        n_bins:     Number of probability bins (10 = each bin is 0-10%, 10-20%, ...).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = test_df["outcome"].values

    models_to_plot = {
        "Ensemble":     ensemble.predict_proba(test_df),
        "XGBoost":      ensemble.base_models_["xgb"].predict_proba(test_df),
        "Dixon-Coles":  ensemble._get_dc_predictions_batch(test_df),
        "Elo baseline": ensemble.base_models_["elo"].predict_proba(test_df),
    }

    colors    = ["#2E86AB", "#A23B72", "#F18F01", "#6B4226"]
    outcome_labels = ["Home win", "Draw", "Away win"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Calibration curves by outcome class", fontsize=13, fontweight="bold")

    for ax, outcome_idx in zip(axes, range(3)):
        y_bin = (y_true == outcome_idx).astype(int)

        for (model_name, y_proba), color in zip(models_to_plot.items(), colors):
            frac_pos, mean_pred = calibration_curve(
                y_bin,
                y_proba[:, outcome_idx],
                n_bins=n_bins,
                strategy="uniform",
            )
            ax.plot(
                mean_pred, frac_pos,
                marker="o", markersize=5,
                label=model_name, color=color, linewidth=1.8,
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect")
        ax.set_title(outcome_labels[outcome_idx], fontsize=11)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = output_dir / "calibration_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved calibration curves → {out_path}")


def plot_shap_summary(
    ensemble: StackingEnsemble,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Generate SHAP summary plots for the XGBoost model.

    SHAP (SHapley Additive exPlanations) decomposes each prediction
    into the contribution of each individual feature. This is the
    interpretability story you tell in a FAANG interview:

        "My model gave France a 71% win probability. Looking at the
         SHAP values, elo_diff contributed +0.18, form_points_diff
         contributed +0.09, and is_neutral penalised them by −0.04."

    Args:
        ensemble:   Fitted StackingEnsemble (XGBoost has SHAP explainer).
        test_df:    Test data for SHAP computation.
        output_dir: Output directory for saved PNGs.
    """
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xgb_model = ensemble.base_models_["xgb"]
    X_test = test_df[FEATURE_COLS].values

    logger.info("Computing SHAP values (~30-60s for large test sets)...")
    shap_values = xgb_model.compute_shap_values(test_df)  # list of 3 arrays

    outcome_names = ["home_win", "draw", "away_win"]

    for outcome_idx, outcome_name in enumerate(outcome_names):
        plt.figure(figsize=(10, 5))
        shap.summary_plot(
            shap_values[outcome_idx],
            feature_names=FEATURE_COLS,
            plot_type="bar",
            show=False,
        )
        plt.suptitle(
            f"SHAP feature importance — {outcome_name.replace('_', ' ')}",
            fontsize=12, fontweight="bold", y=1.00
        )
        plt.tight_layout()
        fname = f"shap_{outcome_name}.png"
        plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Saved SHAP plots → {output_dir}/")


def plot_feature_importance_comparison(
    ensemble: StackingEnsemble,
    output_dir: Path,
) -> None:
    """
    Side-by-side XGBoost vs Random Forest feature importances.

    If both models agree on the top features, you have strong signal.
    If they disagree, you need to investigate why.
    """
    output_dir = Path(output_dir)

    xgb_imp = ensemble.base_models_["xgb"].feature_importance_df()
    rf_imp  = ensemble.base_models_["rf"].feature_importance_df()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.barh(xgb_imp["feature"], xgb_imp["importance"], color="#2E86AB", height=0.6)
    ax1.set_title("XGBoost feature importance", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Importance score")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.25, axis="x")

    ax2.barh(rf_imp["feature"], rf_imp["importance"], color="#F18F01", height=0.6)
    ax2.set_title("Random Forest feature importance", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Importance score")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.25, axis="x")

    plt.tight_layout()
    out_path = output_dir / "feature_importance_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance comparison → {out_path}")