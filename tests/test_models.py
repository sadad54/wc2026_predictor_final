"""
tests/test_models.py

Unit tests for Phase 4 models and Phase 5 metrics.

Tests cover:
    - Dixon-Coles: fitting, scoreline matrix properties, sampling reproducibility
    - Outcome models: output shape, probability validity, individual behaviour
    - Evaluation metrics: mathematical correctness against hand-computed values

Run with: pytest tests/test_models.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.models.dixon_coles import DixonColesModel
from src.models.evaluate import (
    compute_all_metrics,
    multiclass_brier_score,
    ranked_probability_score,
)
from src.models.outcome_models import (
    FEATURE_COLS,
    EloOutcomeModel,
    LogisticOutcomeModel,
    RandomForestOutcomeModel,
    XGBoostOutcomeModel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dc_match_df():
    """Synthetic match data for Dixon-Coles fitting."""
    rng = np.random.default_rng(42)
    teams = ["Spain", "France", "Brazil", "Germany", "England",
             "Argentina", "Italy", "Portugal", "Netherlands"]
    n = 300
    home = rng.choice(teams, n)
    away = rng.choice(teams, n)
    return pd.DataFrame({
        "date":       pd.date_range("2015-01-01", periods=n, freq="7D"),
        "home_team":  home,
        "away_team":  away,
        "home_score": rng.integers(0, 5, n),
        "away_score": rng.integers(0, 4, n),
    })


@pytest.fixture
def feature_df():
    """Synthetic feature data matching features.parquet schema."""
    rng = np.random.default_rng(99)
    n = 150
    dates = pd.date_range("2015-01-01", periods=n, freq="7D")
    teams_h = rng.choice(["Spain", "France", "Brazil", "Germany"], n)
    teams_a = rng.choice(["Argentina", "Italy", "Portugal", "England"], n)
    return pd.DataFrame({
        "date":                    dates,
        "home_team":               teams_h,
        "away_team":               teams_a,
        "elo_diff":                rng.normal(0, 120, n),
        "rank_diff":               rng.normal(0, 15, n).astype(int),
        "form_points_diff":        rng.normal(0, 0.4, n),
        "form_goals_scored_diff":  rng.normal(0, 0.4, n),
        "form_goals_conceded_diff":rng.normal(0, 0.3, n),
        "is_neutral":              rng.integers(0, 2, n),
        "year":                    [d.year for d in dates],
        "month":                   [d.month for d in dates],
        "outcome":                 rng.integers(0, 3, n),
        "home_score":              rng.integers(0, 4, n),
        "away_score":              rng.integers(0, 4, n),
        "match_type":              rng.choice(
            ["world_cup", "world_cup_qualifier", "friendly"], n
        ),
    })


@pytest.fixture
def minimal_config():
    return {
        "models": {
            "random_state": 42,
            "cv_folds": 3,
            "holdout_year": 2022,
            "test_tournaments": [2018, 2022],
            "xgboost": {
                "n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "min_child_weight": 1, "gamma": 0.0,
                "reg_alpha": 0.0, "reg_lambda": 1.0,
            },
            "random_forest": {
                "n_estimators": 20, "max_depth": 4,
                "min_samples_split": 2, "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            "logistic_regression": {
                "C": 1.0, "max_iter": 200,
                "solver": "lbfgs", "multi_class": "multinomial",
            },
        },
        "features": {
            "dixon_coles": {"time_decay_xi": 0.0018},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dixon-Coles tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDixonColesModel:

    def test_fits_without_error(self, dc_match_df):
        model = DixonColesModel(xi=0.001)
        model.fit(dc_match_df)
        assert model.is_fitted_

    def test_scoreline_matrix_shape(self, dc_match_df):
        model = DixonColesModel().fit(dc_match_df)
        m = model.predict_scoreline_matrix("Spain", "France")
        assert m.shape == (11, 11)  # 0..10 inclusive = 11 values

    def test_scoreline_matrix_sums_to_one(self, dc_match_df):
        model = DixonColesModel().fit(dc_match_df)
        m = model.predict_scoreline_matrix("Spain", "France")
        assert abs(m.sum() - 1.0) < 1e-6

    def test_scoreline_matrix_non_negative(self, dc_match_df):
        model = DixonColesModel().fit(dc_match_df)
        m = model.predict_scoreline_matrix("Spain", "France")
        assert (m >= 0).all()

    def test_predict_proba_shape(self, dc_match_df):
        model = DixonColesModel().fit(dc_match_df)
        p = model.predict_proba("Spain", "France")
        assert p.shape == (3,)

    def test_predict_proba_sums_to_one(self, dc_match_df):
        model = DixonColesModel().fit(dc_match_df)
        p = model.predict_proba("Spain", "France")
        assert abs(p.sum() - 1.0) < 1e-6

    def test_predict_proba_non_negative(self, dc_match_df):
        model = DixonColesModel().fit(dc_match_df)
        p = model.predict_proba("Spain", "France")
        assert (p >= 0).all()

    def test_home_advantage_increases_home_win_prob(self, dc_match_df):
        """Removing home advantage should decrease the home win probability."""
        model = DixonColesModel().fit(dc_match_df)
        p_home    = model.predict_proba("Spain", "France", is_neutral=False)
        p_neutral = model.predict_proba("Spain", "France", is_neutral=True)
        assert p_home[0] > p_neutral[0]

    def test_sample_scoreline_valid_tuple(self, dc_match_df):
        model = DixonColesModel().fit(dc_match_df)
        rng = np.random.default_rng(0)
        h, a = model.sample_scoreline("Spain", "France", rng=rng)
        assert h >= 0
        assert a >= 0
        assert h <= model.max_goals
        assert a <= model.max_goals

    def test_sample_scoreline_reproducible(self, dc_match_df):
        """Same seed → same scoreline."""
        model = DixonColesModel().fit(dc_match_df)
        s1 = model.sample_scoreline("Spain", "France", rng=np.random.default_rng(42))
        s2 = model.sample_scoreline("Spain", "France", rng=np.random.default_rng(42))
        assert s1 == s2

    def test_different_seeds_can_differ(self, dc_match_df):
        """Different seeds should eventually produce different results."""
        model = DixonColesModel().fit(dc_match_df)
        results = {
            model.sample_scoreline("Spain", "France", rng=np.random.default_rng(i))
            for i in range(50)
        }
        assert len(results) > 1, "50 samples should not all be identical"

    def test_not_fitted_raises(self):
        model = DixonColesModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba("Spain", "France")

    def test_has_team_true_after_fit(self, dc_match_df):
        model = DixonColesModel().fit(dc_match_df)
        assert model.has_team("Spain")

    def test_has_team_false_for_unknown(self, dc_match_df):
        model = DixonColesModel().fit(dc_match_df)
        assert not model.has_team("Atlantis FC")

    def test_get_team_strengths_returns_df(self, dc_match_df):
        model = DixonColesModel().fit(dc_match_df)
        df = model.get_team_strengths()
        assert isinstance(df, pd.DataFrame)
        assert set(["team", "log_attack", "log_defense", "net_strength"]).issubset(df.columns)
        assert len(df) == len(model.teams_)

    def test_save_load_roundtrip(self, dc_match_df, tmp_path):
        model = DixonColesModel().fit(dc_match_df)
        path = tmp_path / "dc.pkl"
        model.save(path)
        loaded = DixonColesModel.load(path)
        assert loaded.is_fitted_
        np.testing.assert_array_almost_equal(model.params_, loaded.params_)
        assert model.teams_ == loaded.teams_


# ─────────────────────────────────────────────────────────────────────────────
# Outcome model tests (shared across all 4 models)
# ─────────────────────────────────────────────────────────────────────────────

class TestOutcomeModels:
    """
    Property-based tests that must hold for ALL outcome models.
    Add a new model → just add it to the models() fixture.
    """

    @pytest.fixture(params=["elo", "xgb", "rf", "lr"])
    def model_and_data(self, request, feature_df, minimal_config):
        y = feature_df["outcome"].values
        models = {
            "elo": EloOutcomeModel(random_state=42),
            "xgb": XGBoostOutcomeModel(minimal_config),
            "rf":  RandomForestOutcomeModel(minimal_config),
            "lr":  LogisticOutcomeModel(minimal_config),
        }
        model = models[request.param]
        model.fit(feature_df, y)
        return model, feature_df, y

    def test_predict_proba_shape(self, model_and_data):
        model, X, y = model_and_data
        p = model.predict_proba(X)
        assert p.shape == (len(X), 3)

    def test_probabilities_sum_to_one(self, model_and_data):
        model, X, y = model_and_data
        p = model.predict_proba(X)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-5)

    def test_probabilities_non_negative(self, model_and_data):
        model, X, y = model_and_data
        p = model.predict_proba(X)
        assert (p >= 0).all()

    def test_predict_is_argmax_of_proba(self, model_and_data):
        model, X, y = model_and_data
        proba = model.predict_proba(X)
        pred  = model.predict(X)
        np.testing.assert_array_equal(pred, np.argmax(proba, axis=1))

    def test_fit_returns_self(self, feature_df, minimal_config):
        """fit() must return self to allow method chaining."""
        y = feature_df["outcome"].values
        model = EloOutcomeModel()
        returned = model.fit(feature_df, y)
        assert returned is model

    def test_xgboost_feature_importance(self, feature_df, minimal_config):
        y = feature_df["outcome"].values
        model = XGBoostOutcomeModel(minimal_config)
        model.fit(feature_df, y)
        imp = model.feature_importance_df()
        assert list(imp.columns) == ["feature", "importance"]
        assert len(imp) == len(FEATURE_COLS)
        assert (imp["importance"] >= 0).all()

    def test_elo_model_works_with_elo_col_only(self, feature_df):
        """EloOutcomeModel must work when passed a single-column DataFrame."""
        y = feature_df["outcome"].values
        model = EloOutcomeModel()
        model.fit(feature_df, y)
        p = model.predict_proba(feature_df[["elo_diff"]])
        assert p.shape == (len(feature_df), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metric tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluationMetrics:

    def test_rps_perfect_is_zero(self):
        y_true  = np.array([0, 1, 2, 0])
        y_proba = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        assert ranked_probability_score(y_true, y_proba) == pytest.approx(0.0, abs=1e-8)

    def test_rps_manual_calculation(self):
        """
        Manual check for one match.
        Outcome: home win (0) → one-hot [1, 0, 0]
        Prediction: [0.6, 0.3, 0.1]
        Cumulative prediction: [0.6, 0.9]   (drop last term)
        Cumulative truth:      [1.0, 1.0]
        RPS = mean((0.6-1.0)^2, (0.9-1.0)^2)
            = mean(0.16, 0.01)
            = 0.085
        """
        y_true  = np.array([0])
        y_proba = np.array([[0.6, 0.3, 0.1]])
        expected = ((0.6 - 1.0) ** 2 + (0.9 - 1.0) ** 2) / 2
        assert ranked_probability_score(y_true, y_proba) == pytest.approx(expected, abs=1e-8)

    def test_rps_ordered_penalty(self):
        """
        RPS penalises 'away win' prediction for a 'home win' outcome MORE
        than a 'draw' prediction (because draw is closer on the ordering).
        """
        y_true = np.array([0])   # home win
        p_draw     = np.array([[0.0, 1.0, 0.0]])  # predicted draw
        p_away_win = np.array([[0.0, 0.0, 1.0]])  # predicted away win (further)

        rps_draw     = ranked_probability_score(y_true, p_draw)
        rps_away_win = ranked_probability_score(y_true, p_away_win)

        assert rps_draw < rps_away_win

    def test_rps_better_than_worse(self):
        y_true = np.array([0, 0, 0, 0, 0])
        good   = np.tile([0.8, 0.1, 0.1], (5, 1))
        bad    = np.tile([0.1, 0.1, 0.8], (5, 1))
        assert ranked_probability_score(y_true, good) < ranked_probability_score(y_true, bad)

    def test_brier_perfect_is_zero(self):
        y_true  = np.array([0, 1, 2])
        y_proba = np.eye(3)  # identity matrix = perfect 1-hot predictions
        assert multiclass_brier_score(y_true, y_proba) == pytest.approx(0.0, abs=1e-8)

    def test_brier_worst_case(self):
        """Predicting the opposite of truth for all matches."""
        y_true  = np.array([0])
        y_proba = np.array([[0.0, 0.0, 1.0]])  # home win, predicted away win
        # One-hot truth = [1, 0, 0]; diff = [-1, 0, 1]; sum of squares = 2
        assert multiclass_brier_score(y_true, y_proba) == pytest.approx(2.0, abs=1e-8)

    def test_compute_all_metrics_required_keys(self):
        y_true  = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.array([
            [0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7],
            [0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5],
        ])
        metrics = compute_all_metrics(y_true, y_proba, "test")
        required = {"model", "n_matches", "accuracy", "log_loss", "brier", "rps", "roc_auc"}
        assert required.issubset(metrics.keys())

    def test_compute_all_metrics_values_in_range(self):
        rng = np.random.default_rng(0)
        y_true  = rng.integers(0, 3, 100)
        raw     = rng.dirichlet(np.ones(3), 100)
        y_proba = raw / raw.sum(axis=1, keepdims=True)
        metrics = compute_all_metrics(y_true, y_proba, "test")
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["log_loss"]  > 0
        assert metrics["brier"]     >= 0
        assert metrics["rps"]       >= 0
        assert 0 <= metrics["roc_auc"] <= 1