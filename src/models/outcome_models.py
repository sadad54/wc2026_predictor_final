"""
src/models/outcome_models.py

Four sklearn-compatible outcome models, all implementing BaseOutcomeModel.

Every model:
  - accepts features from FEATURE_COLS
  - outputs a (n, 3) probability matrix [P(HW), P(D), P(AW)]
  - is independently trainable and evaluable

FEATURE_COLS (14 features):

  Base Elo/form (8):
    elo_diff                    — home Elo minus away Elo (long-run quality)
    rank_diff                   — away rank minus home rank (positive = home better)
    form_points_diff            — recent form gap (exponentially weighted)
    form_goals_scored_diff      — recent attacking threat gap
    form_goals_conceded_diff    — recent defensive solidity gap
    is_neutral                  — 1 if played on neutral ground
    year                        — calendar year (era effects)
    month                       — month of year (international window effects)

  Squad features (6, all as home - away differences):
    squad_attack_rating_diff    — forward goal threat gap
    squad_defense_rating_diff   — defensive quality gap
    squad_depth_rating_diff     — squad depth gap (bench strength)
    squad_experience_rating_diff — international experience gap
    squad_form_rating_diff      — recent club form gap
    squad_age_balance_diff      — squad age distribution gap

Squad feature diffs are 0.0 for historical training matches (no squad CSV
for past tournaments), and carry real signal once wc2026_squads.csv is loaded.
This means the model is pre-trained on history and the squad features ADD
marginal uplift at prediction time — they don't dominate training signal.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseOutcomeModel

# ── Feature columns consumed by sklearn-based models ─────────────────────────

# Base features (always present, computed from historical match data)
_BASE_FEATURE_COLS: list[str] = [
    "elo_diff",
    "rank_diff",
    "form_points_diff",
    "form_goals_scored_diff",
    "form_goals_conceded_diff",
    "is_neutral",
    "year",
    "month",
]

# Squad features (present when wc2026_squads.csv exists; zero-filled otherwise)
_SQUAD_FEATURE_COLS: list[str] = [
    "squad_attack_rating_diff",
    "squad_defense_rating_diff",
    "squad_depth_rating_diff",
    "squad_experience_rating_diff",
    "squad_form_rating_diff",
    "squad_age_balance_diff",
]

# Full feature set used by all models
FEATURE_COLS: list[str] = _BASE_FEATURE_COLS + _SQUAD_FEATURE_COLS


def get_available_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Return the subset of FEATURE_COLS present in df.

    This allows models to work correctly whether or not squad features
    are in the DataFrame, avoiding KeyError on missing columns.
    Squad feature columns missing from df are implicitly 0.0.
    """
    return [c for c in FEATURE_COLS if c in df.columns]


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with all FEATURE_COLS, filling missing squad columns with 0.0.

    This ensures every model always receives a consistent 14-column matrix,
    even when squad data isn't available yet.
    """
    out = df.copy()
    for col in FEATURE_COLS:
        if col not in out.columns:
            out[col] = 0.0
    return out[FEATURE_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Elo baseline
# ─────────────────────────────────────────────────────────────────────────────

class EloOutcomeModel(BaseOutcomeModel):
    """
    Logistic regression trained exclusively on elo_diff.

    This is the honest baseline. If a complex model can't beat this,
    it's overfitting — all those extra features are adding noise, not signal.
    """

    name = "elo_model"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pipeline_: Optional[Pipeline] = None

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=self.random_state,
            )),
        ])

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "EloOutcomeModel":
        self.pipeline_ = self._build_pipeline()
        self.pipeline_.fit(X[["elo_diff"]], y)
        logger.info(f"  Fitted {self.name}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict_proba(X[["elo_diff"]])


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: XGBoost
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostOutcomeModel(BaseOutcomeModel):
    """
    Gradient boosting outcome classifier with SHAP explainability.

    Trained on all 14 features. The squad features will have near-zero
    importance during training (all zeros in historical data) but their
    weights in the trained model will still respond correctly to non-zero
    squad feature diffs at prediction time — because XGBoost learns split
    thresholds, and a threshold at 0.0 means "squad data present" triggers
    the squad-aware branch automatically.
    """

    name = "xgboost"

    def __init__(self, config: dict):
        self.config = config
        self.model_: Optional[xgb.XGBClassifier] = None
        self.explainer_: Optional[shap.TreeExplainer] = None
        self._fitted_feature_cols: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        **kwargs,
    ) -> "XGBoostOutcomeModel":
        cfg = self.config["models"]["xgboost"]
        X_prepared = prepare_feature_matrix(X)
        self._fitted_feature_cols = list(X_prepared.columns)

        self.model_ = xgb.XGBClassifier(
            n_estimators      = cfg["n_estimators"],
            max_depth         = cfg["max_depth"],
            learning_rate     = cfg["learning_rate"],
            subsample         = cfg["subsample"],
            colsample_bytree  = cfg["colsample_bytree"],
            min_child_weight  = cfg["min_child_weight"],
            gamma             = cfg["gamma"],
            reg_alpha         = cfg["reg_alpha"],
            reg_lambda        = cfg["reg_lambda"],
            objective         = "multi:softprob",
            num_class         = 3,
            random_state      = self.config["models"]["random_state"],
            eval_metric       = "mlogloss",
            early_stopping_rounds = 20 if eval_set else None,
            verbosity         = 0,
        )

        fit_kwargs: dict = {}
        if eval_set:
            X_val, y_val = eval_set[0]
            X_val_prepared = prepare_feature_matrix(X_val)
            fit_kwargs["eval_set"] = [(X_val_prepared.values, y_val)]
            fit_kwargs["verbose"] = False

        self.model_.fit(X_prepared.values, y, **fit_kwargs)
        self.explainer_ = shap.TreeExplainer(self.model_)

        best_iter = getattr(self.model_, "best_iteration", cfg["n_estimators"])
        logger.info(f"  Fitted {self.name} | best_iteration={best_iter}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(prepare_feature_matrix(X).values)

    def compute_shap_values(self, X: pd.DataFrame) -> list[np.ndarray]:
        """Return SHAP values for all three outcomes."""
        if self.explainer_ is None:
            raise RuntimeError(f"{self.name}: call fit() before compute_shap_values().")
        return self.explainer_.shap_values(prepare_feature_matrix(X).values)

    def feature_importance_df(self) -> pd.DataFrame:
        """Return feature importances as a tidy DataFrame (gain-based)."""
        cols = self._fitted_feature_cols if self._fitted_feature_cols else FEATURE_COLS
        return (
            pd.DataFrame({
                "feature":    cols,
                "importance": self.model_.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model 3: Random Forest
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestOutcomeModel(BaseOutcomeModel):
    """
    Random Forest outcome classifier.

    Builds trees independently (bagging). Essential for ensemble diversity —
    XGBoost and RF make different types of errors, so their combination
    through the meta-learner is more robust than either alone.
    """

    name = "random_forest"

    def __init__(self, config: dict):
        self.config = config
        cfg = config["models"]["random_forest"]
        self.model_ = RandomForestClassifier(
            n_estimators    = cfg["n_estimators"],
            max_depth       = cfg["max_depth"],
            min_samples_split = cfg["min_samples_split"],
            min_samples_leaf  = cfg["min_samples_leaf"],
            max_features    = cfg["max_features"],
            random_state    = config["models"]["random_state"],
            n_jobs          = -1,
            class_weight    = "balanced",
        )
        self._fitted_feature_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "RandomForestOutcomeModel":
        X_prepared = prepare_feature_matrix(X)
        self._fitted_feature_cols = list(X_prepared.columns)
        self.model_.fit(X_prepared.values, y)
        logger.info(f"  Fitted {self.name}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(prepare_feature_matrix(X).values)

    def feature_importance_df(self) -> pd.DataFrame:
        cols = self._fitted_feature_cols if self._fitted_feature_cols else FEATURE_COLS
        return (
            pd.DataFrame({
                "feature":    cols,
                "importance": self.model_.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Model 4: Calibrated Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────

class LogisticOutcomeModel(BaseOutcomeModel):
    """
    Calibrated logistic regression — the probability calibration anchor.

    Isotonic calibration ensures that when the model says "70% home win",
    teams actually win ~70% of the time. Miscalibration compounds across
    10,000 Monte Carlo runs, so this is critical for simulation quality.
    """

    name = "logistic"

    def __init__(self, config: dict):
        self.config = config
        cfg = config["models"]["logistic_regression"]

        base_clf = LogisticRegression(
            C        = cfg["C"],
            max_iter = cfg["max_iter"],
            solver   = cfg["solver"],
            random_state = config["models"]["random_state"],
        )

        calibrated_clf = CalibratedClassifierCV(
            estimator = base_clf,
            method    = "isotonic",
            cv        = 5,
        )

        self.model_ = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    calibrated_clf),
        ])

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "LogisticOutcomeModel":
        self.model_.fit(prepare_feature_matrix(X).values, y)
        logger.info(f"  Fitted {self.name}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(prepare_feature_matrix(X).values)