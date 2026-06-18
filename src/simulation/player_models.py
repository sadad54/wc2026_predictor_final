"""
src/simulation/player_models.py

Player-level scoring probability distributions for Golden Boot prediction.

Design philosophy: this module must work even with incomplete squad data,
because squad lists for 2026 finalise close to the tournament and your
ML pipeline needs to run end-to-end well before then.

Two modes:
  - With squad data: each player has a scoring weight derived from a blend
    of their historical international goals-per-appearance rate and their
    recent club season form. Market value is used as a quality multiplier.
  - Without squad data (fallback): generic placeholder players per team,
    weighted by positional scoring probability. Swap in real squads once
    available without touching any other module.

Schema expected in wc2026_squads.csv:
    team                  — canonical team name
    player                — player name
    position              — GK, DF, MF, or FW
    career_goals          — international career goals
    career_appearances    — international career caps
    club                  — current club (unused here, kept for reference)
    market_value_eur      — transfermarkt value in EUR (optional but recommended)
    recent_season_goals   — goals in most recent club season (optional)
    recent_season_apps    — apps in most recent club season (optional)
    age                   — player age (optional)

Usage:
    player_model = PlayerScoringModel.from_squad_data(squads_df, config)
    # or, if no squad data yet:
    player_model = PlayerScoringModel.placeholder(team_names)

    scorer = player_model.sample_scorer("Brazil", rng)
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

_PLACEHOLDER_SQUAD_SIZE = 11

# Position-based forward scoring weights (used in placeholder mode)
_POSITION_FW_WEIGHT: dict[str, float] = {
    "FW": 1.00,
    "MF": 0.50,
    "DF": 0.12,
    "GK": 0.01,
}

# Rough positional scoring-weight template for placeholder (forwards score most)
_PLACEHOLDER_WEIGHTS = np.array([0.18, 0.16, 0.14, 0.10, 0.09, 0.08,
                                   0.07, 0.06, 0.05, 0.04, 0.03])
_PLACEHOLDER_WEIGHTS /= _PLACEHOLDER_WEIGHTS.sum()


class PlayerScoringModel:
    """
    Holds per-team player scoring-probability distributions.

    Attributes:
        team_players: dict mapping team_name → list of player names
        team_weights: dict mapping team_name → np.ndarray of scoring weights
                       (same length as team_players[team], sums to 1.0)
    """

    def __init__(
        self,
        team_players: dict[str, list[str]],
        team_weights: dict[str, np.ndarray],
        player_team: Optional[dict[str, str]] = None,
    ):
        self.team_players = team_players
        self.team_weights = team_weights
        self.player_team = player_team or {
            player: team
            for team, players in team_players.items()
            for player in players
        }

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_squad_data(
        cls,
        squads_df: pd.DataFrame,
        min_appearances: int = 5,
    ) -> "PlayerScoringModel":
        """
        Build the model from real squad data.

        Scoring weight for each player combines:
            60% — international career goal rate (proven scorer at this level)
            40% — recent club season goal rate (current form)

        Both components are multiplied by:
            - Position weight (FW > MF > DF > GK)
            - Market value multiplier: log(market_value) normalised
              (richer players in this position = higher quality)

        Players with fewer than min_appearances caps get a small fallback
        rate (0.03 goals/game) instead of career rate, preventing zero-weight
        players from becoming impossible scorers.

        Args:
            squads_df:        DataFrame matching the schema above.
            min_appearances:  Minimum caps before career rate is used.

        Returns:
            Fitted PlayerScoringModel.
        """
        logger.info(f"Building player scoring model from {len(squads_df)} players across "
                    f"{squads_df['team'].nunique()} teams")

        squads_df = _prepare_squad_df(squads_df, min_appearances)

        team_players: dict[str, list[str]] = {}
        team_weights: dict[str, np.ndarray] = {}
        player_team: dict[str, str] = {}

        for team, group in squads_df.groupby("team"):
            players = group["player"].tolist()

            # Blend career and recent form goal rates
            blended_rate = (
                0.6 * group["career_goal_rate"].values
                + 0.4 * group["recent_goal_rate"].values
            )

            # Position multiplier
            pos_mult = group["position"].map(_POSITION_FW_WEIGHT).fillna(0.1).values

            # Market value multiplier (log scale, normalised within team)
            mv = np.log1p(group["market_value_eur"].values.clip(0))
            mv_sum = mv.sum()
            mv_mult = mv / mv_sum if mv_sum > 0 else np.ones(len(group)) / len(group)

            # Final score: blend × position × market_value
            raw_weights = blended_rate * pos_mult * (1 + mv_mult)
            raw_weights = np.clip(raw_weights, 1e-4, None)  # never exactly zero
            weights = raw_weights / raw_weights.sum()

            team_players[team] = players
            team_weights[team] = weights
            for player in players:
                player_team[player] = team

        logger.info(f"  Player model built for {len(team_players)} teams")

        # Log top scorer probabilities for 3 random teams as a sanity check
        for team in list(team_players.keys())[:3]:
            players = team_players[team]
            weights = team_weights[team]
            top_idx = int(np.argmax(weights))
            logger.debug(f"  {team}: top scorer = {players[top_idx]} ({weights[top_idx]:.3f})")

        return cls(team_players, team_weights, player_team)

    @classmethod
    def placeholder(cls, team_names: list[str]) -> "PlayerScoringModel":
        """
        Build a placeholder model when real squad data isn't available yet.

        Each team gets 11 generic players ("Player 1" .. "Player 11") with
        weights that mimic a realistic team — forwards score more than defenders.
        Once real squads are added, swap the constructor and nothing downstream changes.

        Args:
            team_names: All 48 team names in the tournament.

        Returns:
            PlayerScoringModel with placeholder players.
        """
        logger.warning(
            "Using PLACEHOLDER player model — Golden Boot results will not "
            "reflect real players. Replace with from_squad_data() once "
            "wc2026_squads.csv is populated."
        )

        team_players: dict[str, list[str]] = {}
        team_weights: dict[str, np.ndarray] = {}

        for team in team_names:
            players = [f"{team} Player {i+1}" for i in range(_PLACEHOLDER_SQUAD_SIZE)]
            team_players[team] = players
            team_weights[team] = _PLACEHOLDER_WEIGHTS.copy()

        return cls(team_players, team_weights)

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        min_appearances: int = 5,
    ) -> "PlayerScoringModel":
        """
        Convenience constructor: load from CSV path directly.

        Args:
            csv_path:        Path to wc2026_squads.csv.
            min_appearances: Minimum caps for career rate computation.

        Returns:
            Fitted PlayerScoringModel.
        """
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded squad CSV: {len(df)} rows from {csv_path}")
        return cls.from_squad_data(df, min_appearances)

    # ── Sampling ───────────────────────────────────────────────────────────────

    def sample_scorer(self, team: str, rng: np.random.Generator) -> str:
        """
        Sample one player from `team` weighted by their scoring probability.

        Args:
            team: Team name.
            rng:  Seeded NumPy Generator.

        Returns:
            Player name. If `team` is unknown, returns a generic placeholder
            so the simulation never crashes on an unexpected team name.
        """
        if team not in self.team_players:
            return f"{team} Unknown Player"

        players = self.team_players[team]
        weights = self.team_weights[team]
        return str(rng.choice(players, p=weights))

    def get_top_scorers(self, team: str, top_n: int = 5) -> list[tuple[str, float]]:
        """
        Return the top N scorers for a team with their scoring weights.

        Useful for dashboard display and debugging.

        Args:
            team:  Team name.
            top_n: Number of players to return.

        Returns:
            List of (player_name, weight) tuples, sorted by weight descending.
        """
        if team not in self.team_players:
            return []
        players = self.team_players[team]
        weights = self.team_weights[team]
        sorted_pairs = sorted(zip(players, weights), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_n]

    def has_team(self, team: str) -> bool:
        """Return True if this team has player data (real or placeholder)."""
        return team in self.team_players

    def get_player_team(self, player: str) -> str:
        """Return the team for a player, falling back for placeholder names."""
        if player in self.player_team:
            return self.player_team[player]
        if " Player " in player:
            return player.split(" Player ")[0]
        return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_squad_df(df: pd.DataFrame, min_appearances: int) -> pd.DataFrame:
    """
    Standardise types, fill NaNs, compute derived rate columns.

    This is a local helper (not exported) that makes from_squad_data()
    robust to the full range of data quality issues in scraped squad data.
    """
    df = df.copy()

    if "position" not in df.columns:
        df["position"] = "MF"
    if "career_goals" not in df.columns and "goals" in df.columns:
        df["career_goals"] = df["goals"]
    if "career_appearances" not in df.columns and "caps" in df.columns:
        df["career_appearances"] = df["caps"]
    for col in ["career_goals", "career_appearances"]:
        if col not in df.columns:
            df[col] = 0

    # Normalise position
    df["position"] = df["position"].str.upper().str.strip()
    df["position"] = df["position"].where(
        df["position"].isin(["GK", "DF", "MF", "FW"]), other="MF"
    )

    # Numeric coercion with safe defaults
    for col in ["career_goals", "career_appearances"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    for col in ["recent_season_goals", "recent_season_apps", "age"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    if "market_value_eur" not in df.columns:
        df["market_value_eur"] = 0.0
    df["market_value_eur"] = pd.to_numeric(
        df["market_value_eur"], errors="coerce"
    ).fillna(0.0).clip(lower=0)

    # Career international goal rate
    df["career_goal_rate"] = np.where(
        df["career_appearances"] >= min_appearances,
        df["career_goals"] / df["career_appearances"].clip(lower=1),
        0.03,  # fallback: ~1 goal per 33 games
    )

    # Recent club season goal rate
    df["recent_goal_rate"] = np.where(
        df["recent_season_apps"] >= 5,
        df["recent_season_goals"] / df["recent_season_apps"].clip(lower=1),
        df["career_goal_rate"],  # fall back to career rate if limited club data
    )

    return df
