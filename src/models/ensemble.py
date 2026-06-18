"""
src/models/ensemble.py

Two-level stacking ensemble.

Level 0 — Base models (trained on features):
    EloOutcomeModel          — single-feature baseline (elo_diff only)
    XGBoostOutcomeModel      — complex non-linear interactions
    RandomForestOutcomeModel — diversity through bagging
    LogisticOutcomeModel     — calibration anchor

    DixonColesModel          — statistical goals model (team-name based)
                               provides predictions by looking up fitted
                               attack/defense parameters per team pair

Level 1 — Meta-learner:
    LogisticRegression trained on out-of-fold predictions from all 5 base models
    Input dimension: 5 models × 3 outcomes = 15 features
    Learns which model to trust under which conditions

Training procedure:
    1. Fit Dixon-Coles on full historical goals data.
    2. Generate OOF predictions from all 4 sklearn base models via StratifiedKFold.
    3. Add DC predictions (DC is goals-based, not outcome-based, so it doesn't
       leak outcome information — it can be fitted on all data and used directly).
    4. Train meta-learner on the 15-feature OOF matrix.
    5. Re-fit all base models on the full training set.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseOutcomeModel
from src.models.dixon_coles import DixonColesModel
from src.models.outcome_models import (
    FEATURE_COLS,
    EloOutcomeModel,
    LogisticOutcomeModel,
    RandomForestOutcomeModel,
    XGBoostOutcomeModel,
)

# Historical base rates — used as fallback when a team isn't in the DC model
_FALLBACK_PROBA = np.array([0.40, 0.27, 0.33])


class StackingEnsemble:
    """
    Two-level stacking ensemble for match outcome prediction.

    Usage:
        ensemble = StackingEnsemble(config)
        ensemble.fit(features_df, matches_df)

        # Batch prediction
        proba = ensemble.predict_proba(features_df)   # shape (n, 3)

        # Single match with per-model breakdown
        result = ensemble.predict_match("France", "Brazil", features_row)
    """

    name = "stacking_ensemble"

    def __init__(self, config: dict):
        self.config       = config
        self.n_splits     = config["models"]["cv_folds"]
        self.random_state = config["models"]["random_state"]

        self.base_models_: dict[str, BaseOutcomeModel] = self._create_base_models()
        self.dc_model_: Optional[DixonColesModel] = None

        # Meta-learner: learns optimal combination of base model outputs
        self.meta_learner_ = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C            = 0.5,
                max_iter     = 1000,
                solver       = "lbfgs",
                random_state = self.random_state,
            )),
        ])

        self.is_fitted_: bool = False

    # ── Factory ───────────────────────────────────────────────────────────────

    def _create_base_models(self) -> dict[str, BaseOutcomeModel]:
        """
        Instantiate fresh base model objects.
        Called once at __init__ and once per fold inside OOF generation.
        Using a factory prevents accidental state sharing between folds.
        """
        return {
            "elo": EloOutcomeModel(random_state=self.config["models"]["random_state"]),
            "xgb": XGBoostOutcomeModel(self.config),
            "rf":  RandomForestOutcomeModel(self.config),
            "lr":  LogisticOutcomeModel(self.config),
        }

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        features: pd.DataFrame,
        matches: pd.DataFrame,
    ) -> "StackingEnsemble":
        """
        Train the full stacking ensemble.

        Args:
            features: Feature matrix from features.parquet.
                      Must contain FEATURE_COLS + outcome + home_team + away_team.
            matches:  Full match data from matches_full.parquet.
                      Used for Dixon-Coles (needs home_score, away_score, date).

        Returns:
            self (fitted)
        """
        logger.info("=" * 60)
        logger.info("Training Stacking Ensemble")
        logger.info("=" * 60)

        y = features["outcome"].values

        # ── Step 1: Fit Dixon-Coles on full historical goals data ─────────────
        # DC is a goals model — it doesn't use outcome labels, so fitting on
        # the full training set introduces no leakage.
        logger.info("\n[1/4] Fitting Dixon-Coles on historical goals")
        xi = self.config["models"]["dixon_coles"]["time_decay_xi"]
        self.dc_model_ = DixonColesModel(xi=xi)
        self.dc_model_.fit(
            matches[["home_team", "away_team", "home_score", "away_score", "date"]]
        )

        # ── Step 2: Out-of-fold predictions from all base models ──────────────
        logger.info("\n[2/4] Generating out-of-fold predictions")
        oof_matrix = self._generate_oof_predictions(features, y)
        logger.info(f"  OOF matrix shape: {oof_matrix.shape}  (n_matches × 15 features)")

        # ── Step 3: Train meta-learner on OOF predictions ─────────────────────
        logger.info("\n[3/4] Training meta-learner on OOF matrix")
        self.meta_learner_.fit(oof_matrix, y)

        # ── Step 4: Re-train all base models on the FULL training set ─────────
        # OOF base models were only ever trained on 4/5 of the data.
        # Re-training on 100% gives the strongest possible base models.
        logger.info("\n[4/4] Re-fitting base models on full training data")
        for name, model in self.base_models_.items():
            logger.info(f"  Fitting {name}...")
            model.fit(features, y)

        self.is_fitted_ = True
        logger.info("\n✅ Ensemble training complete")
        return self

    def _generate_oof_predictions(
        self,
        features: pd.DataFrame,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Generate out-of-fold (OOF) predictions for all base models.

        For each cross-validation fold:
          - Base models are trained on the training split
          - They predict on the held-out validation split
          - Those predictions are stored in the OOF matrix

        Because no model ever predicts on rows it trained on,
        the meta-learner receives unbiased predictions.

        Layout of the returned matrix (15 columns):
            cols 0-2:   Elo model probabilities   [P(HW), P(D), P(AW)]
            cols 3-5:   XGBoost probabilities
            cols 6-8:   Random Forest probabilities
            cols 9-11:  Logistic Regression probabilities
            cols 12-14: Dixon-Coles probabilities

        Returns:
            np.ndarray of shape (n_samples, 15)
        """
        n_models  = len(self.base_models_) + 1  # +1 for Dixon-Coles
        oof_matrix = np.zeros((len(features), n_models * 3))

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        for fold, (train_idx, val_idx) in enumerate(skf.split(features, y)):
            logger.info(f"  Fold {fold + 1}/{self.n_splits} — "
                        f"train={len(train_idx):,}  val={len(val_idx):,}")

            X_tr  = features.iloc[train_idx]
            y_tr  = y[train_idx]
            X_val = features.iloc[val_idx]

            col = 0
            # Fresh model instances per fold — no state carried over
            fold_models = self._create_base_models()

            for name, model in fold_models.items():
                model.fit(X_tr, y_tr)
                oof_matrix[val_idx, col : col + 3] = model.predict_proba(X_val)
                col += 3

            # Dixon-Coles predictions (no fold-level retraining needed)
            dc_preds = self._get_dc_predictions_batch(X_val)
            oof_matrix[val_idx, col : col + 3] = dc_preds

        return oof_matrix

    # ── Dixon-Coles batch helper ──────────────────────────────────────────────

    def _get_dc_predictions_batch(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get Dixon-Coles [P(HW), P(D), P(AW)] for every row in a DataFrame.

        Falls back to historical base rates for teams not in the fitted model
        (e.g. new nations, or teams that only appear in the test set).
        """
        out = np.zeros((len(features), 3))

        for i, (_, row) in enumerate(features.iterrows()):
            home  = row["home_team"]
            away  = row["away_team"]
            is_neutral = bool(row.get("is_neutral", 1))

            if self.dc_model_.has_team(home) and self.dc_model_.has_team(away):
                squad_attack_diff, squad_defense_diff = self._get_squad_diffs_from_row(row)
                self._ensure_dc_squad_weights(squad_attack_diff, squad_defense_diff)
                out[i] = self.dc_model_.predict_proba(
                    home,
                    away,
                    is_neutral,
                    squad_attack_diff=squad_attack_diff,
                    squad_defense_diff=squad_defense_diff,
                )
            else:
                out[i] = _FALLBACK_PROBA

        return out

    def _build_meta_input(self, features: pd.DataFrame) -> np.ndarray:
        """Build the 15-column meta-learner input from all base models."""
        parts = []
        for model in self.base_models_.values():
            parts.append(model.predict_proba(features))
        parts.append(self._get_dc_predictions_batch(features))
        return np.hstack(parts)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Return ensemble probability predictions.

        Args:
            features: DataFrame with FEATURE_COLS + home_team + away_team.

        Returns:
            np.ndarray of shape (n_samples, 3): [P(HW), P(D), P(AW)]
        """
        self._check_fitted()
        meta_input = self._build_meta_input(features)
        return self.meta_learner_.predict_proba(meta_input)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return predicted outcome class (0=HW, 1=D, 2=AW)."""
        return np.argmax(self.predict_proba(features), axis=1)

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        features_row: pd.Series,
        is_neutral: bool = True,
    ) -> dict:
        """
        Predict a single match with full per-model breakdown.

        This is what the dashboard calls for the "explain this prediction" view.

        Args:
            home_team:    First team name.
            away_team:    Second team name.
            features_row: One row from features.parquet as a Series.
            is_neutral:   True for World Cup group stage (neutral venue).

        Returns:
            Dict containing ensemble prediction and per-model breakdown.
        """
        self._check_fitted()
        row_df = features_row.to_frame().T

        breakdown: dict = {}
        for name, model in self.base_models_.items():
            p = model.predict_proba(row_df)[0]
            breakdown[name] = {
                "home_win": round(float(p[0]), 4),
                "draw":     round(float(p[1]), 4),
                "away_win": round(float(p[2]), 4),
            }

        squad_attack_diff, squad_defense_diff = self._get_squad_diffs_from_row(
            features_row
        )
        self._ensure_dc_squad_weights(squad_attack_diff, squad_defense_diff)
        dc_p = self.dc_model_.predict_proba(
            home_team,
            away_team,
            is_neutral,
            squad_attack_diff=squad_attack_diff,
            squad_defense_diff=squad_defense_diff,
        )
        breakdown["dixon_coles"] = {
            "home_win": round(float(dc_p[0]), 4),
            "draw":     round(float(dc_p[1]), 4),
            "away_win": round(float(dc_p[2]), 4),
        }

        ens_p = self.predict_proba(row_df)[0]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "is_neutral": is_neutral,
            "ensemble": {
                "home_win": round(float(ens_p[0]), 4),
                "draw":     round(float(ens_p[1]), 4),
                "away_win": round(float(ens_p[2]), 4),
            },
            "model_breakdown": breakdown,
        }

    @staticmethod
    def _get_squad_diffs_from_row(row: pd.Series) -> tuple[float, float]:
        """Extract squad attack/defense diffs from a feature row if present."""
        return (
            float(row.get("squad_attack_rating_diff", 0.0)),
            float(row.get("squad_defense_rating_diff", 0.0)),
        )

    def _ensure_dc_squad_weights(
        self,
        squad_attack_diff: float,
        squad_defense_diff: float,
    ) -> None:
        """Use conservative DC squad weights when non-zero squad diffs are present."""
        if abs(squad_attack_diff) < 1e-12 and abs(squad_defense_diff) < 1e-12:
            return
        if abs(getattr(self.dc_model_, "w_attack_", 0.0)) < 1e-8:
            self.dc_model_.w_attack_ = 0.20
        if abs(getattr(self.dc_model_, "w_defense_", 0.0)) < 1e-8:
            self.dc_model_.w_defense_ = 0.12

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Save all models to a directory (one file per model)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for name, model in self.base_models_.items():
            model.save(directory / f"{name}.pkl")

        if self.dc_model_:
            self.dc_model_.save(directory / "dixon_coles.pkl")

        joblib.dump(self.meta_learner_, directory / "meta_learner.pkl")
        logger.info(f"Ensemble saved → {directory}/")

    @classmethod
    def load(cls, directory: Path, config: dict) -> "StackingEnsemble":
        """Load all models from a directory."""
        directory = Path(directory)
        ensemble = cls(config)

        loader_map = {
            "elo": EloOutcomeModel,
            "xgb": XGBoostOutcomeModel,
            "rf":  RandomForestOutcomeModel,
            "lr":  LogisticOutcomeModel,
        }
        for name in loader_map:
            ensemble.base_models_[name] = joblib.load(directory / f"{name}.pkl")

        ensemble.dc_model_    = DixonColesModel.load(directory / "dixon_coles.pkl")
        ensemble.meta_learner_ = joblib.load(directory / "meta_learner.pkl")
        ensemble.is_fitted_   = True
        logger.info(f"Ensemble loaded ← {directory}/")
        return ensemble

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                "StackingEnsemble is not fitted. Call fit() before predict_proba()."
            )
