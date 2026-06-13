"""
src/features/elo.py

Custom Elo rating engine for international football.

Elo is a system for measuring team strength where every team has a number.
When two teams play, the expected result is determined by the gap in their
ratings. After the match, ratings update: winner gains, loser loses.
The amount gained/lost scales with how unexpected the result was.

Key design decisions:
  - K-factor varies by match type (World Cup > qualifier > friendly)
  - Time decay: ratings regress to 1500 if a team goes inactive
  - Home advantage: adds a fixed bonus to the home team's expected score
  - The engine processes matches chronologically, one at a time

Usage:
    elo = EloEngine(config)
    ratings_history = elo.compute(matches_df)
"""
import numpy as np
import pandas as pd

from src.utils import logger

class EloEngine:
    """
    Computes Elo ratings for all teams across all historical matches.

    Attributes:
        initial_rating:  Starting rating for all teams (default 1500).
        home_advantage:  Rating bonus for the home team (default 65).
        k_factors:       Dict mapping match_type → K-factor value.
    """
    def __init__(self, config: dict)-> None:
        elo_cfg = config["features"]["elo"]
        self.initial_rating: float = elo_cfg["initial_rating"]
        self.home_advantage: float = elo_cfg["home_advantage"]
        self.k_factors: dict = elo_cfg["k_factors"]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Compute the expected score for team A against team B.

        This is the standard Elo formula. The output is a probability
        between 0 and 1 representing team A's expected result, where
        1 = win, 0.5 = draw, 0 = loss.

        The 400 divisor is the standard in most Elo implementations and
        means a 400-point gap translates to a ~91% win probability.

        Args:
            rating_a: Current Elo rating of team A.
            rating_b: Current Elo rating of team B.

        Returns:
            Expected score for team A (float between 0 and 1).
        """
        return 1/(1 + 10 **((rating_b - rating_a) / 400))
    
    def actual_score(self, home_goals: int, away_goals: int)-> float:
        """
        Convert a match result to an Elo 'actual score' for the home team.
        Win = 1.0, Draw = 0.5, Loss = 0.0
        
        Args:
            home_goals: Goals scored by the home team.
            away_goals: Goals scored by the away team.

        Returns:
            Actual score for the home team (1.0, 0.5, or 0.0).
        """
        if home_goals > away_goals:
            return 1.0
        elif home_goals == away_goals:
            return 0.5
        else:
            return 0.0
    
    def get_k_factor(self, match_type: str) -> float:
        """
        Return the k-factor for a given match type.

        K controls how much ratings move per match. Higher K = more volatile .
        World cup matches move ratings more than friendlies because they are higher-stakes and more indicative of true quality.

        Falls back to 'friendly' K if match_type is not in config.
        """
        return self.k_factors.get(match_type, self.k_factors["friendly"])   
    
    def compute(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Process all matches chronologically and compute Elo ratings.

        For each match, we:
        1. Record both teams' Elo ratings BEFORE the match
        (these become features - never use post-match ratings as features)
        2. Compute expected score based on pre-match ratings
        3. Update ratings based on the actual result and K-factor.

        Args:
            matches: Cleaned matches Dataframe, sorted by data ascending. Must have columns: date,
            home_team, away_team, home_score, away_score, match_type, neutral.

        Returns:
            matches with four new columns:
            home_elo_before, away_elo_before,
            home_elo_after, away_elo_after
        
        """
        logger.info("Computing Elo ratings across all matches...")

        # Current ratings dict: {team_name: float}
        ratings: dict[str, float] = {}

        home_elo_before = np.zeros(len(matches))
        away_elo_before = np.zeros(len(matches))
        home_elo_after  = np.zeros(len(matches))
        away_elo_after  = np.zeros(len(matches))

        for i, row in enumerate(matches.itertuples()):
            home = row.home_team
            away = row.away_team

            # Initialize new teams at the starting rating
            if home not in ratings:
                ratings[home] = self.initial_rating
            if away not in ratings:
                ratings[away] = self.initial_rating

            r_home = ratings[home]
            r_away = ratings[away]

            # Apply home advantage (skip for neutral ground matches)
            is_neutral = getattr(row, "neutral", False)
            adj_home = r_home + (0 if is_neutral else self.home_advantage)

            # Expected and actual scores (home team perspective)
            expected = self.expected_score(adj_home, r_away)
            actual   = self.actual_score(row.home_score, row.away_score)

            # K-factor for this match type
            k = self.get_k_factor(row.match_type)

            # Rating delta
            delta = k * (actual - expected)

            # Record pre-match ratings (these are the FEATURES)
            home_elo_before[i] = r_home
            away_elo_before[i] = r_away

            # Update ratings
            ratings[home] = r_home + delta
            ratings[away] = r_away - delta

            # Record post-match ratings (used to seed next match)
            home_elo_after[i] = ratings[home]
            away_elo_after[i] = ratings[away]

        matches = matches.copy()
        matches["home_elo_before"] = home_elo_before
        matches["away_elo_before"] = away_elo_before
        matches["home_elo_after"]  = home_elo_after
        matches["away_elo_after"]  = away_elo_after

        # Current ratings snapshot (useful for predictions and diagnostics)
        self.current_ratings = ratings

        n_teams = len(ratings)
        top5 = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"  Elo computed for {n_teams} teams")
        logger.info(f"  Top 5 current ratings: {top5}")

        return matches

    def get_current_rating(self, team: str) -> float:
        """
        Return the most recent Elo rating for a team.

        Call this after compute() has been run.

        Args:
            team: Canonical team name.

        Returns:
            Current Elo rating, or initial_rating if team not seen.
        """
        if not hasattr(self, "current_ratings"):
            raise RuntimeError("Call compute() before get_current_rating()")
        return self.current_ratings.get(team, self.initial_rating)