"""
src/features/form.py

Exponentially weighted recent form computation.

For each match, we compute the last N results for each team, weighting
recent matches more heavily using exponential decay. A team that was
strong in 2022 but poor in 2024 should reflect the 2024 form, not
a 3-year average that flatters them.

Features computed per team per match (always looking backward):
  - form_points:         Exponentially weighted points (win=3, draw=1, loss=0)
  - form_goals_scored:   Exponentially weighted goals scored
  - form_goals_conceded: Exponentially weighted goals conceded
  - form_matches:        Number of matches in the form window (can be < N at start)
"""

import numpy as np
import pandas as pd
from src.utils import logger

def compute_form_features(matches_df: pd.DataFrame, window: int, decay_factor: float) -> pd.DataFrame:
    """
    Compute exponentially weighted form features for each team in each match.
    
    The form window looks BACKWARD from each match — we use only
    matches that happened before the current one (no leakage).

    Args:
        matches:      Cleaned matches DataFrame, sorted by date ascending.
                      Must have columns: date, home_team, away_team,
                      home_score, away_score.
        window:       Number of recent matches to include in form window.
        decay_factor: Weight of each additional match back in time.
                      0.85 means match N-1 counts 85% as much as match N.

    Returns:
        matches with 8 new columns (4 for home, 4 for away):
            home_form_points, home_form_goals_scored, home_form_goals_conceded,
            home_form_matches, away_form_*, away_form_*...
    
    """
    logger.info(f"Computing form features window={window} and decay={decay_factor}...")

    # Build a per-team match history as we process chronologically.
    # Each team's history is a list of (goals_for, goals_against) tuples.
    team_history: dict[str, list[tuple[int, int]]] = {}

    home_form_pts    = np.zeros(len(matches_df))
    home_form_gf     = np.zeros(len(matches_df))
    home_form_ga     = np.zeros(len(matches_df))
    home_form_n      = np.zeros(len(matches_df), dtype=int)

    away_form_pts    = np.zeros(len(matches_df))
    away_form_gf     = np.zeros(len(matches_df))
    away_form_ga     = np.zeros(len(matches_df))
    away_form_n      = np.zeros(len(matches_df), dtype=int)

    def compute_weighted_stats(
        history: list[tuple[int, int]],
        window: int,
        decay: float,
    ) -> tuple[float, float, float, int]:
        """
        Given a team's recent match history (most recent last), compute
        exponentially weighted form statistics.

        Returns: (weighted_points, weighted_gf, weighted_ga, n_matches)
        """
        recent = history[-window:]  # Last N matches only
        n = len(recent)
        if n == 0:
            return 0.0, 0.0, 0.0, 0

        # Weights: most recent match gets weight 1.0, then decay^1, decay^2, ...
        # history[-1] is most recent, history[-n] is oldest
        weights = np.array([decay ** (n - 1 - j) for j in range(n)])
        weights /= weights.sum()  # Normalize so they sum to 1

        pts = np.array([
            3 if gf > ga else (1 if gf == ga else 0)
            for gf, ga in recent
        ])
        gf_arr = np.array([gf for gf, _ in recent])
        ga_arr = np.array([ga for _, ga in recent])

        return (
            float(np.dot(weights, pts)),
            float(np.dot(weights, gf_arr)),
            float(np.dot(weights, ga_arr)),
            n,
        )

    for i, row in enumerate(matches_df.itertuples()):
        home = row.home_team
        away = row.away_team

        # Initialize history for new teams
        if home not in team_history:
            team_history[home] = []
        if away not in team_history:
            team_history[away] = []

        # Compute form BEFORE recording this match (no leakage)
        h_pts, h_gf, h_ga, h_n = compute_weighted_stats(team_history[home], window, decay_factor)
        a_pts, a_gf, a_ga, a_n = compute_weighted_stats(team_history[away], window, decay_factor)

        home_form_pts[i] = h_pts
        home_form_gf[i]  = h_gf
        home_form_ga[i]  = h_ga
        home_form_n[i]   = h_n

        away_form_pts[i] = a_pts
        away_form_gf[i]  = a_gf
        away_form_ga[i]  = a_ga
        away_form_n[i]   = a_n

        # Now record the result into each team's history
        team_history[home].append((row.home_score, row.away_score))
        team_history[away].append((row.away_score, row.home_score))  # Away perspective

    matches = matches_df.copy()
    matches["home_form_points"]           = home_form_pts
    matches["home_form_goals_scored"]     = home_form_gf
    matches["home_form_goals_conceded"]   = home_form_ga
    matches["home_form_matches"]          = home_form_n

    matches["away_form_points"]           = away_form_pts
    matches["away_form_goals_scored"]     = away_form_gf
    matches["away_form_goals_conceded"]   = away_form_ga
    matches["away_form_matches"]          = away_form_n

    logger.info("  Form features computed")
    return matches
