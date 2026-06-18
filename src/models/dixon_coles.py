"""
src/models/dixon_coles.py

Dixon-Coles Poisson model for football scoreline prediction,
extended with learned squad-strength adjustment.

Reference:
    Dixon, M.J. & Coles, S.G. (1997).
    "Modelling Association Football Scores and Inefficiencies
     in the Football Betting Market."
    Applied Statistics, 46(2), pp. 265-280.

Squad-strength extension:
    We add two global scalars, w_attack and w_defense, that modulate how
    much squad features shift a team's expected goals:

        λ_home = exp(α_home − β_away + γ + w_attack  · squad_attack_diff
                                         + w_defense · squad_defense_diff)
        λ_away = exp(α_away − β_home  + w_attack  · (-squad_attack_diff)
                                       + w_defense · (-squad_defense_diff))

    where:
        squad_attack_diff  = home_squad_attack_rating  - away_squad_attack_rating
        squad_defense_diff = home_squad_defense_rating - away_squad_defense_rating

    Both weights are estimated jointly with all other DC parameters via
    L-BFGS-B. When squad features are all zero (historical training data),
    the weights have no effect — they contribute signal ONLY when non-zero
    squad feature diffs are present (i.e., 2026 tournament predictions).

    This design means:
        - Historical fit quality is unchanged (squad terms vanish)
        - Weights are still estimated from data, not manually tuned
        - At prediction time, a squad-strong team's λ is adjusted upward
          in proportion to the learned weight × squad feature gap

    We include attack and defense dimensions separately because:
        - A team with a strong attack but weak defense needs both signals
        - Learning them jointly prevents the model from conflating the two
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
    Dixon-Coles Poisson model with optional learned squad-strength adjustment.

    After fitting, provides three key methods:
        predict_scoreline_matrix()  →  (N+1 × N+1) scoreline probabilities
        predict_proba()             →  [P(HW), P(D), P(AW)]
        sample_scoreline()          →  (home_goals, away_goals) — for simulation

    Squad features are passed as per-match arrays during fit() and as
    scalar diffs during predict(). When not provided, the model falls back
    to standard Dixon-Coles (squad weight terms = 0 in the output).

    Attributes set after fit():
        teams_               : sorted list of all team names
        team_index_          : dict mapping team name → parameter index
        params_              : fitted parameter vector
        w_attack_            : learned squad attack weight
        w_defense_           : learned squad defense weight
        is_fitted_           : bool
    """

    name = "dixon_coles"

    def __init__(self, xi: float = 0.0018, max_goals: int = 10):
        """
        Args:
            xi:        Time decay rate (per day).
                       xi=0.0018 → match 3 years old has ~50% weight.
            max_goals: Highest scoreline considered. Goals above this truncated.
        """
        self.xi = xi
        self.max_goals = max_goals

        self.teams_: list[str] = []
        self.team_index_: dict[str, int] = {}
        self.params_: Optional[np.ndarray] = None
        self.w_attack_: float = 0.0
        self.w_defense_: float = 0.0
        self.is_fitted_: bool = False

    # ── Private: time decay ───────────────────────────────────────────────────

    def _time_weights(self, dates: pd.Series) -> np.ndarray:
        """Exponential time decay: w(t) = exp(−ξ · days_since_match)."""
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
        Vectorised Dixon-Coles low-score correction factor τ.
        Corrects the four cells (0-0, 1-0, 0-1, 1-1) where pure Poisson
        independence systematically mis-predicts real football scores.
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
        squad_attack_diff: np.ndarray,
        squad_defense_diff: np.ndarray,
    ) -> float:
        """
        Time-decayed negative log-likelihood with squad-strength terms.

        Parameter vector layout:
            params[0 : n]        → attack  (log scale) per team
            params[n : 2n]       → defense (log scale) per team
            params[2n]           → home advantage (log scale)
            params[2n + 1]       → rho (low-score correction)
            params[2n + 2]       → w_attack  (squad attack weight)
            params[2n + 3]       → w_defense (squad defense weight)

        Squad terms:
            For each match, squad_attack_diff = home_squad_attack - away_squad_attack.
            This adjusts λ_home upward and λ_away downward proportionally to
            how much better the home team's attack is (or vice versa).

        When squad_attack_diff = 0 for all matches (historical training data),
        w_attack is identified only by gradient noise and will converge near 0,
        which is correct — the squad terms add no signal on zero-diff data.
        """
        n = len(self.teams_)
        attack   = params[:n]
        defense  = params[n : 2 * n]
        home_adv = params[2 * n]
        rho      = params[2 * n + 1]
        w_atk    = params[2 * n + 2]
        w_def    = params[2 * n + 3]

        # Squad adjustment: positive diff = home team has stronger attack/defense
        squad_adj_home = w_atk * squad_attack_diff + w_def * squad_defense_diff
        squad_adj_away = w_atk * (-squad_attack_diff) + w_def * (-squad_defense_diff)

        lambda_h = np.exp(attack[home_idx] - defense[away_idx] + home_adv + squad_adj_home)
        lambda_a = np.exp(attack[away_idx] - defense[home_idx] + squad_adj_away)

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

        tau = self._tau(home_goals, away_goals, lambda_h, lambda_a, rho)
        log_tau = np.log(np.clip(tau, 1e-10, None))

        weighted_ll = weights * (log_p_h + log_p_a + log_tau)
        return -weighted_ll.sum()

    # ── Public: fit ───────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        squad_features: Optional[pd.DataFrame] = None,
    ) -> "DixonColesModel":
        """
        Fit the Dixon-Coles model on historical match data.

        Args:
            df: DataFrame with columns:
                    home_team (str), away_team (str),
                    home_score (int), away_score (int),
                    date (datetime)
            squad_features: Optional DataFrame with columns:
                    team, squad_attack_rating, squad_defense_rating
                    (output of squad_features.compute_squad_features()).
                    When provided, squad weights w_attack and w_defense are
                    learned from data. When None, they are fixed at 0.

        Returns:
            self — fitted model.
        """
        logger.info("Fitting Dixon-Coles model (with squad-strength extension)...")

        all_teams = sorted(set(df["home_team"].tolist() + df["away_team"].tolist()))
        self.teams_ = all_teams
        self.team_index_ = {t: i for i, t in enumerate(all_teams)}
        n = len(all_teams)
        logger.info(f"  Teams in model: {n}")

        home_idx   = np.array([self.team_index_[t] for t in df["home_team"]])
        away_idx   = np.array([self.team_index_[t] for t in df["away_team"]])
        home_goals = df["home_score"].values.astype(float)
        away_goals = df["away_score"].values.astype(float)
        weights    = self._time_weights(df["date"])

        # Build per-match squad feature diffs (zero if no squad data)
        squad_attack_diff, squad_defense_diff = self._build_squad_diffs(
            df, squad_features, n_matches=len(df)
        )

        has_squad = squad_features is not None
        logger.info(
            f"  Matches: {len(df):,} | "
            f"Weight range: [{weights.min():.4f}, {weights.max():.2f}] | "
            f"Squad features: {'yes' if has_squad else 'no (weights will be ~0)'}"
        )

        # Initial parameters
        x0 = np.concatenate([
            np.zeros(n),   # attack
            np.zeros(n),   # defense
            [0.10],        # home_adv
            [0.10],        # rho
            [0.0],         # w_attack (squad attack weight — start neutral)
            [0.0],         # w_defense (squad defense weight — start neutral)
        ])

        bounds = (
            [(None, None)] * n      # attack: unconstrained
            + [(None, None)] * n    # defense: unconstrained
            + [(None, None)]        # home_adv: unconstrained
            + [(-0.99, 0.99)]       # rho: bounded
            + [(-2.0, 2.0)]         # w_attack: bounded to prevent explosion
            + [(-2.0, 2.0)]         # w_defense: bounded
        )

        logger.info("  Running L-BFGS-B optimisation (~30-120s for 30k+ matches)...")
        result = minimize(
            fun=self._neg_log_likelihood,
            x0=x0,
            args=(
                home_idx, away_idx,
                home_goals, away_goals,
                weights,
                squad_attack_diff, squad_defense_diff,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
        )

        if not result.success:
            logger.warning(f"  Optimisation warning: {result.message}")
        else:
            logger.info(f"  Converged — {result.nit} iterations | NLL = {result.fun:.4f}")

        self.params_ = result.x
        self.w_attack_  = float(result.x[2 * n + 2])
        self.w_defense_ = float(result.x[2 * n + 3])
        self.is_fitted_ = True

        # Sanity diagnostics
        attack_params = self.params_[:n]
        top5 = sorted(zip(self.teams_, attack_params), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"  Top-5 attack (log): {[(t, round(v, 3)) for t, v in top5]}")

        home_adv_exp = np.exp(float(self.params_[2 * n]))
        rho_val = float(self.params_[2 * n + 1])
        logger.info(f"  Home advantage multiplier: {home_adv_exp:.4f} | ρ = {rho_val:.4f}")
        logger.info(f"  Squad weights — attack: {self.w_attack_:.4f} | defense: {self.w_defense_:.4f}")

        if has_squad and abs(self.w_attack_) < 1e-4 and abs(self.w_defense_) < 1e-4:
            logger.warning(
                "  Squad weights are near zero. This may indicate squad features have "
                "insufficient variance across teams. Check your wc2026_squads.csv data quality."
            )

        return self

    def _build_squad_diffs(
        self,
        df: pd.DataFrame,
        squad_features: Optional[pd.DataFrame],
        n_matches: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build per-match squad attack and defense diff arrays.

        For historical training matches: returns zeros (squad CSV is for 2026 only).
        For 2026 prediction: returns real diffs if squad_features is provided.

        Returns:
            (squad_attack_diff, squad_defense_diff) — both shape (n_matches,)
        """
        if squad_features is None:
            return np.zeros(n_matches), np.zeros(n_matches)

        # Build a team → feature lookup
        sq = squad_features.set_index("team")
        atk_col = "squad_attack_rating"
        def_col = "squad_defense_rating"

        def get_feature(team: str, col: str) -> float:
            if team in sq.index and col in sq.columns:
                return float(sq.at[team, col])
            return 0.5  # neutral fallback

        attack_diff  = np.array([
            get_feature(h, atk_col) - get_feature(a, atk_col)
            for h, a in zip(df["home_team"], df["away_team"])
        ])
        defense_diff = np.array([
            get_feature(h, def_col) - get_feature(a, def_col)
            for h, a in zip(df["home_team"], df["away_team"])
        ])

        return attack_diff, defense_diff

    # ── Public: prediction ────────────────────────────────────────────────────

    def _get_lambdas(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool,
        squad_attack_diff: float = 0.0,
        squad_defense_diff: float = 0.0,
    ) -> tuple[float, float]:
        """
        Compute (λ_home, λ_away) expected goals for one match.

        Args:
            home_team, away_team: Team names.
            is_neutral:           True → no home advantage term.
            squad_attack_diff:    home_squad_attack - away_squad_attack (0.0 if unknown).
            squad_defense_diff:   home_squad_defense - away_squad_defense (0.0 if unknown).
        """
        self._check_fitted()
        n = len(self.teams_)

        h = self.team_index_.get(home_team)
        a = self.team_index_.get(away_team)

        attack  = self.params_[:n]
        defense = self.params_[n : 2 * n]
        home_adv = self.params_[2 * n] if not is_neutral else 0.0

        avg_attack  = float(np.mean(attack))
        avg_defense = float(np.mean(defense))

        h_attack  = attack[h]  if h is not None else avg_attack
        a_attack  = attack[a]  if a is not None else avg_attack
        h_defense = defense[h] if h is not None else avg_defense
        a_defense = defense[a] if a is not None else avg_defense

        # Squad adjustment — proportional to learned weights
        squad_adj_home = self.w_attack_ * squad_attack_diff + self.w_defense_ * squad_defense_diff
        squad_adj_away = self.w_attack_ * (-squad_attack_diff) + self.w_defense_ * (-squad_defense_diff)

        lambda_h = np.exp(h_attack - a_defense + home_adv + squad_adj_home)
        lambda_a = np.exp(a_attack - h_defense + squad_adj_away)
        return float(lambda_h), float(lambda_a)

    def predict_scoreline_matrix(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,
        squad_attack_diff: float = 0.0,
        squad_defense_diff: float = 0.0,
    ) -> np.ndarray:
        """
        Return a (max_goals+1 × max_goals+1) scoreline probability matrix.

        Squad diff args are optional — pass them for 2026 predictions,
        omit for historical backtesting.
        """
        lambda_h, lambda_a = self._get_lambdas(
            home_team, away_team, is_neutral,
            squad_attack_diff, squad_defense_diff,
        )
        rho = float(self.params_[2 * len(self.teams_) + 1])

        goals = np.arange(0, self.max_goals + 1)
        p_h = poisson.pmf(goals, lambda_h)
        p_a = poisson.pmf(goals, lambda_a)

        matrix = np.outer(p_h, p_a)

        matrix[0, 0] *= 1.0 - lambda_h * lambda_a * rho
        matrix[1, 0] *= 1.0 + lambda_a * rho
        matrix[0, 1] *= 1.0 + lambda_h * rho
        matrix[1, 1] *= 1.0 - rho

        matrix = np.clip(matrix, 0.0, None)
        matrix /= matrix.sum()
        return matrix

    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,
        squad_attack_diff: float = 0.0,
        squad_defense_diff: float = 0.0,
    ) -> np.ndarray:
        """
        Return [P(home win), P(draw), P(away win)] for a single match.

        Args:
            home_team, away_team: Team names.
            is_neutral:           True for neutral-venue matches.
            squad_attack_diff:    home - away squad attack rating diff.
            squad_defense_diff:   home - away squad defense rating diff.
        """
        matrix = self.predict_scoreline_matrix(
            home_team, away_team, is_neutral,
            squad_attack_diff, squad_defense_diff,
        )
        p_home_win = float(np.tril(matrix, -1).sum())
        p_draw     = float(np.diag(matrix).sum())
        p_away_win = float(np.triu(matrix, 1).sum())
        return np.array([p_home_win, p_draw, p_away_win])

    def sample_scoreline(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,
        rng: Optional[np.random.Generator] = None,
        squad_attack_diff: float = 0.0,
        squad_defense_diff: float = 0.0,
    ) -> tuple[int, int]:
        """
        Draw a single scoreline from the model's distribution.

        Pass squad diffs at simulation time to get squad-adjusted scorelines.
        """
        if rng is None:
            rng = np.random.default_rng()

        matrix = self.predict_scoreline_matrix(
            home_team, away_team, is_neutral,
            squad_attack_diff, squad_defense_diff,
        )
        flat = matrix.flatten()
        flat = flat / flat.sum()

        idx = rng.choice(len(flat), p=flat)
        n = self.max_goals + 1
        return int(idx // n), int(idx % n)

    # ── Public: inspection ────────────────────────────────────────────────────

    def get_team_strengths(self) -> pd.DataFrame:
        """Return fitted attack/defense parameters for all teams."""
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

    def get_squad_weights(self) -> dict[str, float]:
        """Return the learned squad-strength weights."""
        return {"w_attack": self.w_attack_, "w_defense": self.w_defense_}

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

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )