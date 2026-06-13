"""
src/models/dixon_coles.py

Dixon-Coles Poisson model for football scoreline prediction.

Reference:
    Dixon, M.J. & Coles, S.G. (1997).
    "Modelling Association Football Scores and Inefficiencies
     in the Football Betting Market."
    Applied Statistics, 46(2), pp. 265-280.

The model parameterises each team with:
    α_i  — log attack strength (higher = more goals scored)
    β_i  — log defense weakness (higher = more goals conceded)

Plus two global parameters:
    γ    — log home advantage multiplier
    ρ    — Dixon-Coles low-score correction (handles the fact that
             0-0, 1-0, 0-1, and 1-1 occur at different rates than
             pure Poisson independence would predict)

Expected goals:
    λ_home = exp(α_home − β_away + γ)
    λ_away = exp(α_away − β_home)

Goals follow Poisson(λ) with a multiplicative correction τ applied
to the four low-scoring cells in the scoreline probability matrix.

Parameters are estimated via L-BFGS-B maximisation of a
time-decayed log-likelihood: matches from further in the past
receive exponentially lower weight, controlled by xi.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson


class DixonColesModel:
    """
    Dixon-Coles Poisson model.

    After fitting, provides three key methods:
        predict_scoreline_matrix()  →  (N+1 × N+1) scoreline probabilities
        predict_proba()             →  [P(HW), P(D), P(AW)]
        sample_scoreline()          →  (home_goals, away_goals) — for simulation

    Attributes set after fit():
        teams_       : sorted list of all team names seen in training data
        team_index_  : dict mapping team name → parameter index
        params_      : fitted parameter vector [attack, defense, home_adv, rho]
        is_fitted_   : bool
    """

    name = "dixon_coles"

    def __init__(self, xi: float = 0.0018, max_goals: int = 10):
        """
        Args:
            xi:        Time decay rate (per day).
                       xi=0.0018 → a match 3 years ago has ~50% weight.
                       xi=0.0    → no decay (all matches equal weight).
            max_goals: Highest scoreline considered in the probability matrix.
                       Goals above this are truncated (extremely rare beyond 10).
        """
        self.xi = xi
        self.max_goals = max_goals

        self.teams_: list[str] = []
        self.team_index_: dict[str, int] = {}
        self.params_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    # ── Private: time decay ───────────────────────────────────────────────────

    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        """
        Exponential time decay: w(t) = exp(−ξ · days_since_match).
        More recent matches get higher weight.
        """
        reference = pd.Timestamp.today()
        days_ago = (reference - dates).dt.days.values.astype(float)
        return np.exp(-self.xi * days_ago)

    # ── Private: Dixon-Coles τ correction ────────────────────────────────────

    @staticmethod
    def _tau(
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        lambda_h: np.ndarray,
        lambda_a: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """
        Vectorised Dixon-Coles correction factor τ.

        Pure Poisson independence systematically under-predicts 0-0 draws
        and over-predicts 1-1 draws in real football data. τ corrects this.

        For most scorelines τ = 1.0 (no adjustment needed).
        Only four cells are adjusted:

            τ(0,0) = 1 − λ_h · λ_a · ρ
            τ(1,0) = 1 + λ_a · ρ
            τ(0,1) = 1 + λ_h · ρ
            τ(1,1) = 1 − ρ

        ρ is estimated from data; typically small and positive (~0.1).
        """
        tau = np.ones(len(home_goals), dtype=np.float64)

        m00 = (home_goals == 0) & (away_goals == 0)
        m10 = (home_goals == 1) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m11 = (home_goals == 1) & (away_goals == 1)

        tau[m00] = 1.0 - lambda_h[m00] * lambda_a[m00] * rho
        tau[m10] = 1.0 + lambda_a[m10] * rho
        tau[m01] = 1.0 + lambda_h[m01] * rho
        tau[m11] = 1.0 - rho

        return tau

    # ── Private: log-likelihood ───────────────────────────────────────────────

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        home_idx: np.ndarray,
        away_idx: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Time-decayed negative log-likelihood passed to scipy.optimize.minimize.

        Parameter vector layout:
            params[0 : n]        → attack  (log scale) for each team
            params[n : 2n]       → defense (log scale) for each team
            params[2n]           → home advantage (log scale)
            params[2n + 1]       → rho (low-score correction)

        Using log-scale parameters ensures λ_h, λ_a > 0 always,
        making the optimisation numerically stable.
        """
        n = len(self.teams_)
        attack   = params[:n]
        defense  = params[n : 2 * n]
        home_adv = params[2 * n]
        rho      = params[2 * n + 1]

        # Vectorised expected goals
        lambda_h = np.exp(attack[home_idx] - defense[away_idx] + home_adv)
        lambda_a = np.exp(attack[away_idx] - defense[home_idx])

        # Poisson log-PMF: x·log(λ) − λ − log(x!)
        # gammaln(x+1) = log(x!) — numerically stable for large x
        log_p_h = (
            home_goals * np.log(lambda_h + 1e-10)
            - lambda_h
            - gammaln(home_goals + 1)
        )
        log_p_a = (
            away_goals * np.log(lambda_a + 1e-10)
            - lambda_a
            - gammaln(away_goals + 1)
        )

        # Dixon-Coles τ correction
        tau = self._tau(home_goals, away_goals, lambda_h, lambda_a, rho)
        log_tau = np.log(np.clip(tau, 1e-10, None))  # guard against τ < 0

        weighted_ll = weights * (log_p_h + log_p_a + log_tau)
        return -weighted_ll.sum()

    # ── Public: fit ───────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        """
        Fit the Dixon-Coles model on historical match data.

        Args:
            df: DataFrame with columns:
                    home_team (str), away_team (str),
                    home_score (int), away_score (int),
                    date (datetime)

        Returns:
            self — fitted model (enables method chaining: model.fit(df).predict_proba(...))
        """
        logger.info("Fitting Dixon-Coles model...")

        # Build team universe
        all_teams = sorted(
            set(df["home_team"].tolist() + df["away_team"].tolist())
        )
        self.teams_ = all_teams
        self.team_index_ = {t: i for i, t in enumerate(all_teams)}
        n = len(all_teams)
        logger.info(f"  Teams in model: {n}")

        # Encode to integer indices for vectorised operations
        home_idx   = np.array([self.team_index_[t] for t in df["home_team"]])
        away_idx   = np.array([self.team_index_[t] for t in df["away_team"]])
        home_goals = df["home_score"].values.astype(float)
        away_goals = df["away_score"].values.astype(float)
        weights    = self._time_weights(df["date"])

        logger.info(
            f"  Matches: {len(df):,} | "
            f"Weight range: [{weights.min():.4f}, {weights.max():.2f}]"
        )

        # Initial parameter vector: all zeros → exp(0) = 1.0 (neutral teams)
        x0 = np.concatenate([
            np.zeros(n),   # attack
            np.zeros(n),   # defense
            [0.10],        # home_adv (small positive — home teams score slightly more)
            [0.10],        # rho (small positive — 0-0 slightly over-predicted by Poisson)
        ])

        # Bounds: only rho is constrained to (-1, 1) to keep τ positive
        bounds = (
            [(None, None)] * n      # attack: unconstrained
            + [(None, None)] * n    # defense: unconstrained
            + [(None, None)]        # home_adv: unconstrained
            + [(-0.99, 0.99)]       # rho: bounded
        )

        logger.info("  Running L-BFGS-B optimisation (~30-90s for 30k+ matches)...")
        result = minimize(
            fun=self._neg_log_likelihood,
            x0=x0,
            args=(home_idx, away_idx, home_goals, away_goals, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-8},
        )

        if not result.success:
            logger.warning(f"  Optimisation warning: {result.message}")
        else:
            logger.info(f"  Converged — {result.nit} iterations | NLL = {result.fun:.4f}")

        self.params_ = result.x
        self.is_fitted_ = True

        # Show top-5 attack teams as a sanity check
        attack_params = self.params_[:n]
        top5 = sorted(zip(self.teams_, attack_params), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"  Top-5 attack (log): {[(t, round(v, 3)) for t, v in top5]}")

        # Show fitted rho and home advantage
        home_adv_exp = np.exp(float(self.params_[2 * n]))
        rho_val = float(self.params_[2 * n + 1])
        logger.info(f"  Home advantage multiplier: {home_adv_exp:.4f} | ρ = {rho_val:.4f}")

        return self

    # ── Public: prediction ────────────────────────────────────────────────────

    def _get_lambdas(
        self, home_team: str, away_team: str, is_neutral: bool
    ) -> tuple[float, float]:
        """Compute (λ_home, λ_away) expected goals for one match."""
        self._check_fitted()
        n = len(self.teams_)
        
        # Fallback to average attack/defense for teams not in training data
        if home_team not in self.team_index_:
            h = None  # Will use average
        else:
            h = self.team_index_[home_team]
            
        if away_team not in self.team_index_:
            a = None  # Will use average
        else:
            a = self.team_index_[away_team]

        attack   = self.params_[:n]
        defense  = self.params_[n : 2 * n]
        home_adv = self.params_[2 * n] if not is_neutral else 0.0
        
        # Use average attack/defense for unknown teams
        avg_attack = np.mean(attack)
        avg_defense = np.mean(defense)
        
        h_attack = attack[h] if h is not None else avg_attack
        a_attack = attack[a] if a is not None else avg_attack
        h_defense = defense[h] if h is not None else avg_defense
        a_defense = defense[a] if a is not None else avg_defense

        lambda_h = np.exp(h_attack - a_defense + home_adv)
        lambda_a = np.exp(a_attack - h_defense)
        return float(lambda_h), float(lambda_a)

    def predict_scoreline_matrix(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,
    ) -> np.ndarray:
        """
        Return a (max_goals+1 × max_goals+1) scoreline probability matrix.

        Entry [i, j] = P(home scores i goals, away scores j goals).
        Matrix sums to 1.0 (after Dixon-Coles τ correction and renormalisation).

        Args:
            home_team:  First team (advantages apply if not neutral).
            away_team:  Second team.
            is_neutral: True for World Cup matches (most are on neutral soil).

        Returns:
            np.ndarray of shape (max_goals+1, max_goals+1).
        """
        lambda_h, lambda_a = self._get_lambdas(home_team, away_team, is_neutral)
        rho = float(self.params_[2 * len(self.teams_) + 1])

        goals = np.arange(0, self.max_goals + 1)
        p_h = poisson.pmf(goals, lambda_h)  # P(home = i) for i in 0..max_goals
        p_a = poisson.pmf(goals, lambda_a)  # P(away = j) for j in 0..max_goals

        # Outer product: matrix[i,j] = P(home=i) × P(away=j) under independence
        matrix = np.outer(p_h, p_a)

        # Apply Dixon-Coles τ correction to the four low-scoring cells
        matrix[0, 0] *= 1.0 - lambda_h * lambda_a * rho
        matrix[1, 0] *= 1.0 + lambda_a * rho
        matrix[0, 1] *= 1.0 + lambda_h * rho
        matrix[1, 1] *= 1.0 - rho

        # Re-normalise: τ adjustment shifts the sum slightly away from 1.0
        matrix = np.clip(matrix, 0.0, None)
        matrix /= matrix.sum()

        return matrix

    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,
    ) -> np.ndarray:
        """
        Return [P(home win), P(draw), P(away win)] for a single match.

        Sums the appropriate cells of the scoreline matrix:
            P(home win) = Σ P(i,j) for i > j  (lower triangle)
            P(draw)     = Σ P(i,i) for all i   (diagonal)
            P(away win) = Σ P(i,j) for j > i   (upper triangle)

        Args:
            home_team:  Name of home/first team.
            away_team:  Name of away/second team.
            is_neutral: True for neutral-venue matches.

        Returns:
            np.ndarray([p_home_win, p_draw, p_away_win])
        """
        matrix = self.predict_scoreline_matrix(home_team, away_team, is_neutral)

        p_home_win = float(np.tril(matrix, -1).sum())  # i > j → home scores more
        p_draw     = float(np.diag(matrix).sum())       # i = j → equal
        p_away_win = float(np.triu(matrix, 1).sum())    # j > i → away scores more

        return np.array([p_home_win, p_draw, p_away_win])

    def sample_scoreline(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[int, int]:
        """
        Draw a single scoreline from the model's distribution.

        This is the method the Monte Carlo simulator calls 10,000+ times.
        Using a pre-seeded Generator ensures reproducibility.

        Args:
            home_team:  First team.
            away_team:  Second team.
            is_neutral: True for World Cup matches.
            rng:        NumPy Generator (np.random.default_rng(seed)).
                        If None, creates a fresh unseeded Generator.

        Returns:
            (home_goals, away_goals) as a tuple of ints.
        """
        if rng is None:
            rng = np.random.default_rng()

        matrix = self.predict_scoreline_matrix(home_team, away_team, is_neutral)
        flat = matrix.flatten()
        flat = flat / flat.sum()  # numerical safety

        idx = rng.choice(len(flat), p=flat)
        n = self.max_goals + 1
        return int(idx // n), int(idx % n)

    # ── Public: inspection ────────────────────────────────────────────────────

    def get_team_strengths(self) -> pd.DataFrame:
        """
        Return a DataFrame of fitted attack/defense parameters for all teams.
        Useful for portfolio visualisations — shows which teams the model
        considers strongest in attack and which are defensively solid.
        """
        self._check_fitted()
        n = len(self.teams_)
        return (
            pd.DataFrame({
                "team":         self.teams_,
                "log_attack":   self.params_[:n],
                "log_defense":  self.params_[n : 2 * n],
                "net_strength": self.params_[:n] - self.params_[n : 2 * n],
            })
            .sort_values("net_strength", ascending=False)
            .reset_index(drop=True)
        )

    def has_team(self, team: str) -> bool:
        """Return True if this team was seen during fit()."""
        return team in self.team_index_

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        joblib.dump(self, path)
        logger.info(f"Saved DixonColesModel → {path}")

    @classmethod
    def load(cls, path: Path) -> "DixonColesModel":
        model = joblib.load(path)
        logger.info(f"Loaded DixonColesModel ← {path}")
        return model

    # ── Private: guard ────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )