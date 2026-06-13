"""
src/simulation/player_models.py

Player-level scoring probability distributions for Golden Boot prediction.

Design philosophy: this module must work even with incomplete squad data,
because squad lists for 2026 finalise close to the tournament and your
ML pipeline needs to run end-to-end well before then.

Two modes:
  - With squad data: each player has a scoring weight derived from their
    historical international goals-per-appearance rate.
  - Without squad data (fallback): a small set of generic placeholder
    "players" per team, weighted equally. This keeps the simulator
    functional; swap in real squads once available without touching
    any other module.

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

_PLACEHOLDER_SQUAD_SIZE = 11  # one "player" per outfield position, simple and even


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
    ):
        self.team_players = team_players
        self.team_weights = team_weights

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_squad_data(
        cls,
        squads_df: pd.DataFrame,
        min_appearances: int = 5,
    ) -> "PlayerScoringModel":
        """
        Build the model from real squad data.

        Args:
            squads_df: DataFrame with columns:
                           team (str), player (str),
                           career_goals (int), career_appearances (int)
            min_appearances: Players below this appearance count get a
                             small uniform fallback weight, preventing
                             zero-probability players (a player with 0
                             goals in 1 cap shouldn't be impossible —
                             they could still score in a simulation).

        Returns:
            Fitted PlayerScoringModel.
        """
        logger.info(f"Building player scoring model from {len(squads_df)} players")

        team_players: dict[str, list[str]] = {}
        team_weights: dict[str, np.ndarray] = {}

        for team, group in squads_df.groupby("team"):
            players = group["player"].tolist()

            # Goals-per-appearance, with a small floor to avoid zero weights
            rates = np.where(
                group["career_appearances"] >= min_appearances,
                group["career_goals"] / group["career_appearances"].clip(lower=1),
                0.05,  # fallback rate for low-cap players
            )
            rates = np.clip(rates, 1e-3, None)  # never exactly zero
            weights = rates / rates.sum()

            team_players[team] = players
            team_weights[team] = weights

        logger.info(f"  Player model built for {len(team_players)} teams")
        return cls(team_players, team_weights)

    @classmethod
    def placeholder(cls, team_names: list[str]) -> "PlayerScoringModel":
        """
        Build a placeholder model when real squad data isn't available yet.

        Each team gets 11 generic players ("Player 1" .. "Player 11") with
        weights that mimic a realistic team — forwards score more than
        defenders. This keeps Golden Boot tracking structurally correct;
        once real squads are added, swap the constructor and nothing
        downstream changes.

        Args:
            team_names: All 48 team names in the tournament.

        Returns:
            PlayerScoringModel with placeholder players.
        """
        logger.warning(
            "Using PLACEHOLDER player model — Golden Boot results will not "
            "reflect real players. Replace with from_squad_data() once "
            "squad data is collected."
        )

        team_players: dict[str, list[str]] = {}
        team_weights: dict[str, np.ndarray] = {}

        # Rough positional scoring-weight template (forwards/attackers score most)
        # Positions 1-3: forwards/attacking midfielders (high)
        # Positions 4-7: midfielders (medium)
        # Positions 8-11: defenders/GK (low)
        position_weights = np.array([0.18, 0.16, 0.14, 0.10, 0.09, 0.08,
                                       0.07, 0.06, 0.05, 0.04, 0.03])
        position_weights /= position_weights.sum()

        for team in team_names:
            players = [f"{team} Player {i+1}" for i in range(_PLACEHOLDER_SQUAD_SIZE)]
            team_players[team] = players
            team_weights[team] = position_weights.copy()

        return cls(team_players, team_weights)

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

    def has_team(self, team: str) -> bool:
        """Return True if this team has player data (real or placeholder)."""
        return team in self.team_players