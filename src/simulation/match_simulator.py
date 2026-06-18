"""
src/simulation/match_simulator.py

Phase 6: Single-match Monte Carlo simulator.

Given two teams and a fitted DixonColesModel, simulates one complete match:

  1. Regulation time (90 min)   — scoreline drawn from Dixon-Coles distribution
  2. Extra time (knockout only) — if drawn after 90, simulate 30 more minutes
                                   at a reduced goal rate (teams tire, play
                                   more defensively)
  3. Penalty shootout            — if still drawn after ET, simulate kicks
                                   using historical success rates
  4. Goal scorers                — attribute each goal to a player using
                                   that team's scoring-probability distribution
  5. Yellow cards                — sampled per player for suspension tracking

The output is a MatchResult dataclass — a clean, typed object the tournament
simulator consumes without caring about the internals.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.models.dixon_coles import DixonColesModel
from .player_models import PlayerScoringModel


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """
    Complete outcome of one simulated match.

    Attributes:
        home_team, away_team:    Team names.
        home_goals, away_goals:  Regulation-time goals (90 min).
        went_to_extra_time:      True if the match was a knockout draw after 90.
        et_home_goals, et_away_goals: Extra-time goals (0 if no ET).
        went_to_penalties:       True if still drawn after ET.
        pen_home_score, pen_away_score: Penalty shootout score (None if no pens).
        winner:                   Team name of the match winner.
                                   For group matches that end in a draw,
                                   winner is None.
        is_draw:                  True if the match (in regulation, for group
                                   stage) ended level — irrelevant for knockout
                                   since those always produce a winner.
        scorers:                  List of (team_name, player_name) tuples,
                                   one entry per goal scored (regulation + ET).
        yellow_cards:             List of (team_name, player_name) tuples,
                                   one entry per yellow card issued.
    """
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    went_to_extra_time: bool = False
    et_home_goals: int = 0
    et_away_goals: int = 0
    went_to_penalties: bool = False
    pen_home_score: Optional[int] = None
    pen_away_score: Optional[int] = None
    winner: Optional[str] = None
    is_draw: bool = False
    scorers: list[tuple[str, str]] = field(default_factory=list)
    yellow_cards: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_home_goals(self) -> int:
        """Home goals across regulation + extra time (for Golden Boot tally)."""
        return self.home_goals + self.et_home_goals

    @property
    def total_away_goals(self) -> int:
        """Away goals across regulation + extra time."""
        return self.away_goals + self.et_away_goals


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────

class MatchSimulator:
    """
    Simulates individual matches using a fitted Dixon-Coles model.

    Attributes:
        dc_model:             Fitted DixonColesModel for scoreline probabilities.
        player_model:         PlayerScoringModel for goal scorer attribution.
        et_goal_rate:         Multiplier applied to lambda during extra time
                               (teams tire and play more conservatively).
        pen_base_success:     Historical penalty success rate, used when a
                               team-specific rate isn't available.
    """

    def __init__(
        self,
        dc_model: DixonColesModel,
        player_model: Optional[PlayerScoringModel] = None,
        squad_features: Optional[pd.DataFrame] = None,
        et_goal_rate: float = 0.75,
        pen_base_success: float = 0.753,
        fallback_squad_attack_weight: float = 0.20,
        fallback_squad_defense_weight: float = 0.12,
    ):
        self.dc_model = dc_model
        self.player_model = player_model
        self.squad_strengths = self._build_squad_strength_lookup(squad_features)
        self.et_goal_rate = et_goal_rate
        self.pen_base_success = pen_base_success
        self._activate_fallback_squad_weights(
            fallback_squad_attack_weight, fallback_squad_defense_weight
        )

    # ── Public: full match simulation ─────────────────────────────────────────

    def simulate(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool,
        is_knockout: bool,
        rng: np.random.Generator,
    ) -> MatchResult:
        """
        Simulate one complete match.

        Args:
            home_team:   First team (Dixon-Coles "home" slot — most WC matches
                         are neutral, so this mainly affects model lookup, not
                         a real home advantage).
            away_team:   Second team.
            is_neutral:  Passed to the Dixon-Coles model. True for all
                         WC 2026 matches except (rare) host-nation games.
            is_knockout: If True and the match is drawn after 90 minutes,
                         simulate extra time and penalties.
            rng:         Seeded NumPy Generator — pass the SAME generator
                         through an entire tournament simulation for full
                         reproducibility of one run.

        Returns:
            MatchResult with all fields populated.
        """
        # ── Step 1: regulation time ──────────────────────────────────────────
        home_goals, away_goals = self._sample_scoreline(
            home_team, away_team, is_neutral, rng, time_fraction=1.0
        )

        result = MatchResult(
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
        )

        is_drawn = home_goals == away_goals

        if not is_knockout:
            # Group stage: draws are a valid final result
            result.is_draw = is_drawn
            if not is_drawn:
                result.winner = home_team if home_goals > away_goals else away_team
        else:
            if not is_drawn:
                result.winner = home_team if home_goals > away_goals else away_team
            else:
                # ── Step 2: extra time ───────────────────────────────────────
                et_home, et_away = self._simulate_extra_time(
                    home_team, away_team, is_neutral, rng
                )
                result.went_to_extra_time = True
                result.et_home_goals = et_home
                result.et_away_goals = et_away

                if et_home != et_away:
                    result.winner = home_team if et_home > et_away else away_team
                else:
                    # ── Step 3: penalty shootout ─────────────────────────────
                    pen_home, pen_away = self._simulate_penalties(
                        home_team, away_team, rng
                    )
                    result.went_to_penalties = True
                    result.pen_home_score = pen_home
                    result.pen_away_score = pen_away
                    result.winner = home_team if pen_home > pen_away else away_team

        # ── Step 4: goal scorers ─────────────────────────────────────────────
        if self.player_model is not None:
            result.scorers = self._assign_scorers(result, rng)
            result.yellow_cards = self._assign_yellow_cards(
                home_team, away_team, rng
            )

        return result

    # ── Private: scoreline sampling ───────────────────────────────────────────

    def _sample_scoreline(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool,
        rng: np.random.Generator,
        time_fraction: float,
    ) -> tuple[int, int]:
        """
        Draw a scoreline from the Dixon-Coles distribution.

        Args:
            time_fraction: Scales the scoreline matrix's underlying lambdas.
                           1.0 = full 90 minutes. Used for extra time (1/3,
                           since 30 min = 1/3 of 90).

        Returns:
            (home_goals, away_goals)
        """
        squad_attack_diff, squad_defense_diff = self._get_squad_diffs(
            home_team, away_team
        )

        if time_fraction == 1.0:
            return self.dc_model.sample_scoreline(
                home_team,
                away_team,
                is_neutral,
                rng,
                squad_attack_diff=squad_attack_diff,
                squad_defense_diff=squad_defense_diff,
            )

        # For extra time, build a scaled-down scoreline matrix
        matrix = self.dc_model.predict_scoreline_matrix(
            home_team,
            away_team,
            is_neutral,
            squad_attack_diff=squad_attack_diff,
            squad_defense_diff=squad_defense_diff,
        )
        scaled_matrix = self._scale_scoreline_matrix(matrix, time_fraction)

        flat = scaled_matrix.flatten()
        flat = flat / flat.sum()
        idx = rng.choice(len(flat), p=flat)
        n = self.dc_model.max_goals + 1
        return int(idx // n), int(idx % n)

    def _scale_scoreline_matrix(self, matrix: np.ndarray, fraction: float) -> np.ndarray:
        """
        Re-derive a scoreline matrix for a shorter time window.

        Rather than re-running the full Poisson construction, we approximate
        by redistributing probability mass toward lower-scoring outcomes —
        scaling each marginal Poisson rate by `fraction` and re-deriving
        the joint matrix via outer product. This is an approximation but
        captures the key effect: fewer goals expected in 30 minutes than 90.

        Args:
            matrix:   Full 90-minute scoreline probability matrix.
            fraction: Time fraction (e.g. 1/3 for extra time's 30 min vs 90).

        Returns:
            Re-normalised scoreline matrix for the shorter window.
        """
        from scipy.stats import poisson

        n = matrix.shape[0]
        goals = np.arange(n)

        # Recover approximate marginal lambdas from the 90-min matrix
        home_marginal = matrix.sum(axis=1)
        away_marginal = matrix.sum(axis=0)
        lambda_h_90 = float(np.dot(goals, home_marginal))
        lambda_a_90 = float(np.dot(goals, away_marginal))

        # Scale by time fraction and the extra-time goal-rate multiplier
        lambda_h_et = max(lambda_h_90 * fraction * self.et_goal_rate, 1e-6)
        lambda_a_et = max(lambda_a_90 * fraction * self.et_goal_rate, 1e-6)

        p_h = poisson.pmf(goals, lambda_h_et)
        p_a = poisson.pmf(goals, lambda_a_et)

        scaled = np.outer(p_h, p_a)
        scaled = np.clip(scaled, 0.0, None)
        scaled /= scaled.sum()
        return scaled

    # ── Private: extra time ────────────────────────────────────────────────────

    def _simulate_extra_time(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """
        Simulate 30 minutes of extra time.

        Goal rates are reduced (et_goal_rate, default 0.75) relative to the
        90-minute rate, scaled to the shorter duration (30/90 = 1/3),
        reflecting tired legs and more conservative tactics.

        Returns:
            (et_home_goals, et_away_goals)
        """
        return self._sample_scoreline(
            home_team, away_team, is_neutral, rng, time_fraction=1.0 / 3.0
        )

    # ── Private: penalty shootout ───────────────────────────────────────────────

    def _simulate_penalties(
        self,
        home_team: str,
        away_team: str,
        rng: np.random.Generator,
    ) -> tuple[int, int]:
        """
        Simulate a best-of-5 penalty shootout (with sudden death if tied).

        Each kick is a Bernoulli trial using pen_base_success (historical
        ~75.3% conversion rate). Team-specific rates could be added later
        via player_model, but the base rate is a reasonable approximation —
        shootout outcomes are notoriously close to a coin-flip around this rate.

        Returns:
            (home_score, away_score) — guaranteed home_score != away_score.
        """
        p = self.pen_base_success

        def take_kicks(n: int) -> int:
            return int(rng.binomial(n, p, size=1).sum()) if n > 0 else 0

        # Standard best-of-5
        home_score = sum(rng.random() < p for _ in range(5))
        away_score = sum(rng.random() < p for _ in range(5))

        # Sudden death — alternate single kicks until someone misses and the other scores
        while home_score == away_score:
            h_kick = rng.random() < p
            a_kick = rng.random() < p
            if h_kick != a_kick:
                if h_kick:
                    home_score += 1
                else:
                    away_score += 1
            # If both score or both miss, the round is repeated (no change)

        return int(home_score), int(away_score)

    # ── Private: goal scorers ────────────────────────────────────────────────────

    def _assign_scorers(
        self,
        result: MatchResult,
        rng: np.random.Generator,
    ) -> list[tuple[str, str]]:
        """
        Attribute each goal in the match to a player.

        For each team, draws `total_goals` players (with replacement,
        weighted by each player's scoring probability) from the squad.

        Returns:
            List of (team_name, player_name) — one entry per goal, in
            no particular order. Regulation + extra time goals combined.
        """
        scorers = []

        home_total = result.total_home_goals
        away_total = result.total_away_goals

        for _ in range(home_total):
            player = self.player_model.sample_scorer(result.home_team, rng)
            scorers.append((result.home_team, player))

        for _ in range(away_total):
            player = self.player_model.sample_scorer(result.away_team, rng)
            scorers.append((result.away_team, player))

        return scorers

    # ── Private: yellow cards ────────────────────────────────────────────────────

    def _assign_yellow_cards(
        self,
        home_team: str,
        away_team: str,
        rng: np.random.Generator,
        cards_per_team_rate: float = 1.8,
    ) -> list[tuple[str, str]]:
        """
        Sample yellow cards for both teams.

        The number of cards per team is drawn from a Poisson distribution
        (average ~1.8 cards/team/match across recent World Cups), then
        each card is attributed to a random squad player weighted by
        the player's typical involvement (using the same scoring-rate
        distribution as a proxy for minutes played — imperfect but
        directionally reasonable without dedicated discipline data).

        Args:
            cards_per_team_rate: Mean yellow cards per team per match.

        Returns:
            List of (team_name, player_name) — one entry per card.
        """
        cards = []
        for team in (home_team, away_team):
            n_cards = rng.poisson(cards_per_team_rate)
            for _ in range(n_cards):
                player = self.player_model.sample_scorer(team, rng)
                cards.append((team, player))
        return cards

    @staticmethod
    def _build_squad_strength_lookup(
        squad_features: Optional[pd.DataFrame],
    ) -> dict[str, tuple[float, float]]:
        """Build team -> (attack_rating, defense_rating) lookup."""
        if squad_features is None or squad_features.empty:
            return {}

        required = {"team", "squad_attack_rating", "squad_defense_rating"}
        if not required.issubset(squad_features.columns):
            missing = required - set(squad_features.columns)
            logger.warning(f"Squad features missing columns {missing}; using neutral diffs")
            return {}

        lookup = {}
        for _, row in squad_features.iterrows():
            lookup[str(row["team"])] = (
                float(row["squad_attack_rating"]),
                float(row["squad_defense_rating"]),
            )
        return lookup

    def _get_squad_diffs(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return home-away squad attack and defense rating differences."""
        home = self.squad_strengths.get(home_team, (0.5, 0.5))
        away = self.squad_strengths.get(away_team, (0.5, 0.5))
        return home[0] - away[0], home[1] - away[1]

    def _activate_fallback_squad_weights(
        self,
        attack_weight: float,
        defense_weight: float,
    ) -> None:
        """
        Give squad diffs a conservative effect when the saved DC model learned none.

        Historical training rows usually have zero squad diffs, so the fitted
        squad weights can remain at zero. In tournament simulation, non-zero
        2026 squad ratings should still nudge expected goals modestly.
        """
        if not self.squad_strengths:
            return
        if abs(getattr(self.dc_model, "w_attack_", 0.0)) < 1e-8:
            self.dc_model.w_attack_ = attack_weight
        if abs(getattr(self.dc_model, "w_defense_", 0.0)) < 1e-8:
            self.dc_model.w_defense_ = defense_weight
