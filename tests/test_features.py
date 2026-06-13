"""
tests/test_features.py

Unit tests for Phase 2 preprocessing and Phase 3 feature engineering.

Run with: pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data.team_names import normalize_team_name
from src.data.preprocess import classify_tournament, clean_results
from src.features.elo import EloEngine
from src.features.form import compute_form_features


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_matches(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal matches DataFrame for testing."""
    defaults = {
        "date": pd.Timestamp("2020-01-01"),
        "home_team": "TeamA",
        "away_team": "TeamB",
        "home_score": 1,
        "away_score": 0,
        "tournament": "FIFA World Cup",
        "match_type": "world_cup",
        "neutral": False,
        "outcome": "W",
    }
    records = [{**defaults, **r} for r in rows]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def minimal_config() -> dict:
    return {
        "features": {
            "elo": {
                "initial_rating": 1500,
                "home_advantage": 65,
                "k_factors": {
                    "world_cup_final":    60,
                    "world_cup":          50,
                    "world_cup_qualifier": 40,
                    "continental":        35,
                    "friendly":           20,
                },
            },
            "form": {
                "window_matches": 5,
                "decay_factor": 0.85,
            },
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Team name normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeTeamName:

    def test_known_variant_maps_correctly(self):
        assert normalize_team_name("Korea Republic") == "South Korea"

    def test_unknown_name_returned_unchanged(self):
        assert normalize_team_name("France") == "France"

    def test_usa_variant(self):
        assert normalize_team_name("United States") == "USA"

    def test_ivory_coast_variant(self):
        assert normalize_team_name("Côte d'Ivoire") == "Ivory Coast"


# ─────────────────────────────────────────────────────────────────────────────
# Tournament classification
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyTournament:

    def test_world_cup(self):
        assert classify_tournament("FIFA World Cup") == "world_cup"

    def test_qualifier(self):
        assert classify_tournament("UEFA World Cup Qualifying") == "world_cup_qualifier"

    def test_continental(self):
        assert classify_tournament("UEFA Euro 2024") == "continental"

    def test_friendly(self):
        assert classify_tournament("Friendly") == "friendly"

    def test_unknown_defaults_to_friendly(self):
        assert classify_tournament("Some Unknown Tournament") == "friendly"


# ─────────────────────────────────────────────────────────────────────────────
# Elo engine
# ─────────────────────────────────────────────────────────────────────────────

class TestEloEngine:

    def test_new_teams_start_at_initial_rating(self):
        config = minimal_config()
        elo = EloEngine(config)
        matches = make_matches([
            {"home_team": "NewA", "away_team": "NewB", "home_score": 1, "away_score": 0}
        ])
        result = elo.compute(matches)
        assert result.iloc[0]["home_elo_before"] == 1500
        assert result.iloc[0]["away_elo_before"] == 1500

    def test_winner_gains_rating(self):
        config = minimal_config()
        elo = EloEngine(config)
        matches = make_matches([
            {"home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0,
             "match_type": "world_cup"},
        ])
        result = elo.compute(matches)
        assert result.iloc[0]["home_elo_after"] > result.iloc[0]["home_elo_before"]
        assert result.iloc[0]["away_elo_after"] < result.iloc[0]["away_elo_before"]

    def test_ratings_are_zero_sum(self):
        """Total Elo in the system is conserved after each match."""
        config = minimal_config()
        elo = EloEngine(config)
        matches = make_matches([
            {"home_team": "A", "away_team": "B", "home_score": 3, "away_score": 1},
            {"home_team": "A", "away_team": "B", "home_score": 0, "away_score": 0,
             "date": "2020-02-01"},
        ])
        result = elo.compute(matches)
        for _, row in result.iterrows():
            before_total = row["home_elo_before"] + row["away_elo_before"]
            after_total  = row["home_elo_after"]  + row["away_elo_after"]
            assert abs(before_total - after_total) < 0.001

    def test_draw_gives_smaller_delta_than_win(self):
        config = minimal_config()
        elo = EloEngine(config)

        win_matches = make_matches([
            {"home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0}
        ])
        draw_matches = make_matches([
            {"home_team": "A", "away_team": "B", "home_score": 0, "away_score": 0}
        ])

        win_result  = elo.compute(win_matches)
        draw_result = elo.compute(draw_matches)

        win_delta  = abs(win_result.iloc[0]["home_elo_after"]  - win_result.iloc[0]["home_elo_before"])
        draw_delta = abs(draw_result.iloc[0]["home_elo_after"] - draw_result.iloc[0]["home_elo_before"])

        assert win_delta > draw_delta

    def test_expected_score_symmetric(self):
        config = minimal_config()
        elo = EloEngine(config)
        assert abs(elo.expected_score(1500, 1500) - 0.5) < 0.001
        e_ab = elo.expected_score(1600, 1400)
        e_ba = elo.expected_score(1400, 1600)
        assert abs(e_ab + e_ba - 1.0) < 0.001

    def test_pre_match_ratings_not_contaminated(self):
        """home_elo_before must equal home_elo_after of the previous match."""
        config = minimal_config()
        elo = EloEngine(config)
        matches = make_matches([
            {"home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0,
             "date": "2020-01-01"},
            {"home_team": "A", "away_team": "C", "home_score": 1, "away_score": 1,
             "date": "2020-02-01"},
        ])
        result = elo.compute(matches)
        assert result.iloc[1]["home_elo_before"] == result.iloc[0]["home_elo_after"]


# ─────────────────────────────────────────────────────────────────────────────
# Form computation
# ─────────────────────────────────────────────────────────────────────────────

class TestFormFeatures:

    def test_first_match_has_zero_form(self):
        matches = make_matches([
            {"home_team": "A", "away_team": "B"}
        ])
        result = compute_form_features(matches, window=5, decay_factor=0.85)
        assert result.iloc[0]["home_form_matches"] == 0
        assert result.iloc[0]["home_form_points"] == 0.0

    def test_form_accumulates_over_matches(self):
        matches = make_matches([
            {"home_team": "A", "away_team": "B", "home_score": 3, "away_score": 0,
             "date": "2020-01-01"},
            {"home_team": "A", "away_team": "C", "home_score": 2, "away_score": 1,
             "date": "2020-02-01"},
        ])
        result = compute_form_features(matches, window=5, decay_factor=0.85)
        # Second match should see one previous result for team A
        assert result.iloc[1]["home_form_matches"] == 1

    def test_form_points_range(self):
        """Weighted form points should be between 0 and 3."""
        matches = make_matches([
            {"home_team": "A", "away_team": "B", "home_score": i % 3, "away_score": 0,
             "date": f"2020-0{i+1}-01"}
            for i in range(1, 6)
        ])
        result = compute_form_features(matches, window=5, decay_factor=0.85)
        assert (result["home_form_points"] >= 0).all()
        assert (result["home_form_points"] <= 3).all()

    def test_no_leakage_current_match_not_in_form(self):
        """A team's form before match N must not include match N's result."""
        matches = make_matches([
            {"home_team": "A", "away_team": "B", "home_score": 5, "away_score": 0,
             "date": "2020-01-01"},
            {"home_team": "A", "away_team": "C", "home_score": 0, "away_score": 0,
             "date": "2020-02-01"},
        ])
        result = compute_form_features(matches, window=5, decay_factor=0.85)
        # Before match 2, team A has 1 match: a 5-0 win = 3 points
        assert result.iloc[1]["home_form_matches"] == 1
        assert result.iloc[1]["home_form_points"] == pytest.approx(3.0)