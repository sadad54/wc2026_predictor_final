"""
src/models/outcome_models.py

Four sklearn-compatible outcome models, all implementing BaseOutcomeModel.

Every model:
  - accepts features from FEATURE_COLS
  - outputs a (n, 3) probability matrix [P(HW), P(D), P(AW)]
  - is independently trainable and evaluable

FEATURE_COLS (8 features):
    elo_diff                  — home Elo minus away Elo (captures long-run quality)
    rank_diff                 — away rank minus home rank (positive = home ranked better)
    form_points_diff          — recent form gap (exponentially weighted)
    form_goals_scored_diff    — recent attacking threat gap
    form_goals_conceded_diff  — recent defensive solidity gap
    is_neutral                — 1 if played on neutral ground
    year                      — calendar year (captures era effects)
    month                     — month of year (captures international window effects)
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
FEATURE_COLS: list[str] = [
    "elo_diff",
    "rank_diff",
    "form_points_diff",
    "form_goals_scored_diff",
    "form_goals_conceded_diff",
    "is_neutral",
    "year",
    "month",
]


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Elo baseline
# ─────────────────────────────────────────────────────────────────────────────

class EloOutcomeModel(BaseOutcomeModel):
    """
    Logistic regression trained exclusively on elo_diff.

    This is the honest baseline. If a complex model can't beat this,
    it's overfitting — all those extra features are adding noise, not signal.

    FiveThirtyEight's Club Soccer Predictions use exactly this principle:
    one carefully-constructed strength number → win probability.
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

    XGBoost is the workhorse of the ensemble. It builds 300 decision trees
    sequentially, each tree correcting the errors of the previous one.

    This captures interactions Elo alone can't: form matters more when
    Elo difference is small; ranking matters more in competitive tournaments.

    After fitting, a TreeExplainer is built so SHAP values can be computed
    for any prediction — telling you exactly which feature drove each result.
    """

    name = "xgboost"

    def __init__(self, config: dict):
        self.config = config
        self.model_: Optional[xgb.XGBClassifier] = None
        self.explainer_: Optional[shap.TreeExplainer] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        **kwargs,
    ) -> "XGBoostOutcomeModel":
        cfg = self.config["models"]["xgboost"]

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
            fit_kwargs["eval_set"] = [(X_val[FEATURE_COLS].values, y_val)]
            fit_kwargs["verbose"] = False

        self.model_.fit(X[FEATURE_COLS].values, y, **fit_kwargs)

        # Build SHAP explainer immediately — TreeExplainer is fast and exact
        self.explainer_ = shap.TreeExplainer(self.model_)

        best_iter = getattr(self.model_, "best_iteration", cfg["n_estimators"])
        logger.info(f"  Fitted {self.name} | best_iteration={best_iter}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[FEATURE_COLS].values)

    def compute_shap_values(self, X: pd.DataFrame) -> list[np.ndarray]:
        """
        Compute SHAP values for all three outcomes.

        Returns:
            List of three (n_samples, n_features) arrays,
            one per outcome class [home win, draw, away win].
            Use shap.summary_plot(values[0], X, feature_names=FEATURE_COLS)
            to visualise.
        """
        if self.explainer_ is None:
            raise RuntimeError(f"{self.name}: call fit() before compute_shap_values().")
        return self.explainer_.shap_values(X[FEATURE_COLS].values)

    def feature_importance_df(self) -> pd.DataFrame:
        """Return feature importances as a tidy DataFrame (gain-based)."""
        return (
            pd.DataFrame({
                "feature":    FEATURE_COLS,
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

    Builds trees independently (bagging), not sequentially like XGBoost.
    This means it makes different kinds of errors — essential for
    ensemble diversity. If XGBoost and RF disagree strongly on a match,
    that's a signal the match is genuinely uncertain.

    class_weight='balanced' accounts for the fact that home wins (~46%)
    are more common than draws (~25%) or away wins (~29%).
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

    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs) -> "RandomForestOutcomeModel":
        self.model_.fit(X[FEATURE_COLS].values, y)
        logger.info(f"  Fitted {self.name}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[FEATURE_COLS].values)

    def feature_importance_df(self) -> pd.DataFrame:
        """Return feature importances (mean decrease in impurity)."""
        return (
            pd.DataFrame({
                "feature":    FEATURE_COLS,
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

    CalibratedClassifierCV wraps the base logistic regression with
    isotonic regression calibration fitted via cross-validation.

    Why this matters: raw logistic regression and XGBoost both tend to
    be over-confident. Calibration ensures that when the model says
    '70% chance of a home win', teams actually win approximately 70%
    of the time. Miscalibrated models produce incorrect expected values
    in the simulation — a systematic error that compounds across 10,000 runs.
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

        # Isotonic calibration is non-parametric and more powerful than Platt
        # scaling (sigmoid) when there's enough data (>1000 samples).
        # cv=5: calibration itself is cross-validated to avoid leakage.
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
        self.model_.fit(X[FEATURE_COLS].values, y)
        logger.info(f"  Fitted {self.name}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[FEATURE_COLS].values)