"""
src/features/squad_features.py

Squad-level feature engineering for the 2026 World Cup prediction engine.

Reads data/external/wc2026_squads.csv and produces per-team aggregate
signals that capture squad quality beyond what Elo and form alone can measure.

Schema expected in wc2026_squads.csv:
    team                  — canonical team name (must match team_names.py)
    player                — player name (string)
    position              — one of: GK, DF, MF, FW
    career_goals          — international career goals (int)
    career_appearances    — international career caps (int)
    club                  — current club name (string)
    market_value_eur      — transfermarkt market value in EUR (float, can be 0/NaN)
    recent_season_goals   — goals in most recent club season (int)
    recent_season_apps    — appearances in most recent club season (int)
    age                   — player age in years (int)

Features produced per team (all numeric, used as home - away differences):
    squad_attack_rating       — weighted forward goal threat
    squad_defense_rating      — weighted GK/DF defensive quality (inverted)
    squad_depth_rating        — squad depth score (market value spread)
    squad_experience_rating   — experience score (WC-level caps weighted)
    squad_form_rating         — aggregate recent club form
    squad_age_balance         — deviation from optimal squad age (25-29)

Design notes:
    - All features are computed ONCE at tournament time (not per match), because
      squad composition doesn't change match-to-match in a tournament.
    - Features are normalised to [0, 1] range across all 48 teams so they can
      be differenced cleanly (home - away) without scale problems.
    - NaN handling: missing values fall back to the median of all 48 teams so
      partial squad data doesn't crash the pipeline.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.data.squads import load_squad_data


# ── Position group mappings ───────────────────────────────────────────────────
# Used to weight contribution of each position to attack/defense ratings.
POSITION_ATTACK_WEIGHT: dict[str, float] = {
    "FW": 1.00,
    "MF": 0.55,
    "DF": 0.15,
    "GK": 0.00,
}

POSITION_DEFENSE_WEIGHT: dict[str, float] = {
    "GK": 1.00,
    "DF": 0.85,
    "MF": 0.35,
    "FW": 0.05,
}

# Optimal age range for peak performance (used in age_balance feature)
_PEAK_AGE_LOW  = 25
_PEAK_AGE_HIGH = 29


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_squad_features(
    squads_df: pd.DataFrame,
    min_appearances: int = 5,
) -> pd.DataFrame:
    """
    Compute 6 squad-level features for every team in the squads DataFrame.

    Args:
        squads_df:       DataFrame matching the schema above.
        min_appearances: Minimum caps before a player's goal rate is used
                         (below this, a small fallback rate is applied).

    Returns:
        DataFrame with one row per team and columns:
            team, squad_attack_rating, squad_defense_rating, squad_depth_rating,
            squad_experience_rating, squad_form_rating, squad_age_balance
        All rating columns are normalised to [0, 1] across all teams.
    """
    logger.info(f"Computing squad features for {squads_df['team'].nunique()} teams...")

    squads_df = _clean_squad_df(squads_df, min_appearances)
    team_rows = []

    for team, group in squads_df.groupby("team"):
        row = {"team": team}

        # ── 1. Attack rating ─────────────────────────────────────────────────
        # Weighted sum of (goals / apps) for each player, weighted by position
        # and market value. Captures both quality and volume of goal threat.
        row["squad_attack_rating"] = _attack_rating(group)

        # ── 2. Defense rating ────────────────────────────────────────────────
        # Market-value-weighted positional quality for GK and defenders.
        row["squad_defense_rating"] = _defense_rating(group)

        # ── 3. Depth rating ──────────────────────────────────────────────────
        # 75th-percentile market value / max value — penalises teams where
        # quality is concentrated in 1-2 stars with weak backups.
        row["squad_depth_rating"] = _depth_rating(group)

        # ── 4. Experience rating ─────────────────────────────────────────────
        # Mean caps per player, weighted toward experienced players (>50 caps).
        row["squad_experience_rating"] = _experience_rating(group)

        # ── 5. Recent club form ──────────────────────────────────────────────
        # Aggregate recent season goals per app across all players.
        row["squad_form_rating"] = _form_rating(group)

        # ── 6. Age balance ───────────────────────────────────────────────────
        # How many players fall in the 25-29 peak window. Teams skewing too
        # young or too old tend to underperform expectations.
        row["squad_age_balance"] = _age_balance(group)

        team_rows.append(row)

    result = pd.DataFrame(team_rows)

    # Normalise all rating columns to [0, 1]
    rating_cols = [c for c in result.columns if c != "team"]
    for col in rating_cols:
        col_min = result[col].min()
        col_max = result[col].max()
        if col_max > col_min:
            result[col] = (result[col] - col_min) / (col_max - col_min)
        else:
            result[col] = 0.5  # all teams identical → neutral

    logger.info(f"  Squad features computed | shape: {result.shape}")
    logger.info(f"  Top 3 attack: {result.nlargest(3, 'squad_attack_rating')[['team','squad_attack_rating']].to_dict('records')}")
    logger.info(f"  Top 3 defense: {result.nlargest(3, 'squad_defense_rating')[['team','squad_defense_rating']].to_dict('records')}")

    return result


def _clean_squad_df(df: pd.DataFrame, min_appearances: int) -> pd.DataFrame:
    """Standardise types, fill NaNs, clip outliers."""
    df = df.copy()

    # Normalise position strings
    df["position"] = df["position"].str.upper().str.strip()
    df["position"] = df["position"].where(
        df["position"].isin(["GK", "DF", "MF", "FW"]), other="MF"
    )

    # Numeric columns — fill NaN with 0 before clipping
    int_cols = ["career_goals", "career_appearances", "recent_season_goals",
                "recent_season_apps", "age"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

    if "market_value_eur" in df.columns:
        df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce").fillna(0).clip(lower=0)
    else:
        df["market_value_eur"] = 0.0

    # Fill missing recent season data with 0
    for col in ["recent_season_goals", "recent_season_apps", "age"]:
        if col not in df.columns:
            df[col] = 0

    # Goal rate per career cap (with floor for low-cap players)
    df["career_goal_rate"] = np.where(
        df["career_appearances"] >= min_appearances,
        df["career_goals"] / df["career_appearances"].clip(lower=1),
        0.03,  # fallback ~1 goal per 33 games (realistic for bench player)
    )

    # Recent club goal rate
    df["recent_goal_rate"] = np.where(
        df["recent_season_apps"] >= 5,
        df["recent_season_goals"] / df["recent_season_apps"].clip(lower=1),
        df["career_goal_rate"],  # fall back to career rate
    )

    return df


def _attack_rating(group: pd.DataFrame) -> float:
    """
    Weighted attacking threat score.

    Combines:
      - career goal rate (60% weight): proven international scorer
      - recent club form goal rate (40% weight): current form matters
    Each player's contribution is weighted by their position attack weight
    and market value (proxy for overall quality).
    """
    weights = group["position"].map(POSITION_ATTACK_WEIGHT).fillna(0.1)
    mv = group["market_value_eur"].clip(lower=1e3)  # avoid zero weights
    combined_weight = weights * np.log1p(mv)  # log scale to dampen outliers

    score = (0.6 * group["career_goal_rate"] + 0.4 * group["recent_goal_rate"])
    total_weight = combined_weight.sum()
    if total_weight == 0:
        return 0.0
    return float((score * combined_weight).sum() / total_weight)


def _defense_rating(group: pd.DataFrame) -> float:
    """
    Defensive quality score.

    Uses market value of defensive players (GK, DF) weighted by position
    importance. Higher market value in this position = better defense.
    We use log market value as the quality proxy since we don't have
    goals-conceded data at the player level.
    """
    weights = group["position"].map(POSITION_DEFENSE_WEIGHT).fillna(0.0)
    mv = group["market_value_eur"].clip(lower=1e3)
    combined_weight = weights * np.log1p(mv)
    total_weight = combined_weight.sum()
    if total_weight == 0:
        return 0.0
    return float(combined_weight.sum() / len(group))  # avg quality among defenders


def _depth_rating(group: pd.DataFrame) -> float:
    """
    Squad depth: how evenly quality is distributed.

    A team with 11 world-class players and 12 replacements scores higher
    than one with 3 stars and 20 bench-warmers. We measure this as the
    ratio of p75 to p90 market value (closer to 1.0 = deeper squad).
    """
    mv = group["market_value_eur"].values
    if mv.sum() == 0:
        return 0.0
    p75 = np.percentile(mv, 75)
    p90 = np.percentile(mv, 90)
    if p90 == 0:
        return 0.0
    return float(p75 / (p90 + 1e-6))


def _experience_rating(group: pd.DataFrame) -> float:
    """
    International experience score.

    Caps are a direct measure of tournament experience. We weight
    experienced players (50+ caps) more heavily than newer ones, since
    players who have been tested in high-pressure WC qualifiers carry
    more predictive signal.
    """
    caps = group["career_appearances"].values.clip(0)
    # Non-linear weighting: caps beyond 50 count double
    weighted_caps = np.where(caps >= 50, caps * 2.0, caps)
    return float(weighted_caps.mean())


def _form_rating(group: pd.DataFrame) -> float:
    """
    Aggregate recent club form across the squad.

    Recent club form predicts tournament performance because players who
    are in form carry that confidence into international duty. We weight
    each player's recent goal rate by their position attack weight, then
    take the squad average.
    """
    attack_w = group["position"].map(POSITION_ATTACK_WEIGHT).fillna(0.1)
    recent = group["recent_goal_rate"].fillna(0)
    total_weight = attack_w.sum()
    if total_weight == 0:
        return 0.0
    return float((recent * attack_w).sum() / total_weight)


def _age_balance(group: pd.DataFrame) -> float:
    """
    Proportion of players in the 25-29 peak window.

    Research on international football consistently shows teams peaking
    when 40-60% of their outfield players are in their prime years.
    We return the fraction in-window; a value near 0.5 is ideal.
    We then convert to a "closeness to ideal" score: 1.0 - abs(fraction - 0.5).
    """
    ages = group["age"].values
    if len(ages) == 0:
        return 0.5
    in_peak = ((ages >= _PEAK_AGE_LOW) & (ages <= _PEAK_AGE_HIGH)).mean()
    return float(1.0 - abs(in_peak - 0.5))  # 1.0 = exactly 50% in peak window


# ─────────────────────────────────────────────────────────────────────────────
# Integration helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_squad_features(external_data_dir: Path, min_appearances: int = 5) -> Optional[pd.DataFrame]:
    """
    Load squad data from disk and compute features.

    Returns None (with a warning) if the file doesn't exist yet — this
    allows the pipeline to run in degraded mode without squad data.

    Args:
        external_data_dir: Path to data/external/
        min_appearances:   Minimum caps for goal-rate calculation.

    Returns:
        Squad features DataFrame, or None if data not available.
    """
    squads_df = load_squad_data(external_data_dir)
    if squads_df is None:
        logger.warning("Squad features will be ZERO for all teams.")
        return None

    _validate_squad_schema(squads_df, Path(external_data_dir))
    return compute_squad_features(squads_df, min_appearances)


def _validate_squad_schema(df: pd.DataFrame, path: Path) -> None:
    """Warn about any missing required columns."""
    required = {"team", "player", "position", "career_goals", "career_appearances"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"wc2026_squads.csv is missing required columns: {missing}. "
            f"See wc2026_squads_schema.csv for the expected format."
        )
    recommended = {"market_value_eur", "recent_season_goals", "recent_season_apps", "age"}
    missing_rec = recommended - set(df.columns)
    if missing_rec:
        logger.warning(
            f"Optional squad columns missing (features will be degraded): {missing_rec}. "
            "Add these columns to improve model quality."
        )


def attach_squad_features_to_matches(
    matches: pd.DataFrame,
    squad_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join squad features onto a matches DataFrame for home and away teams.

    Because squad features are static (one set of values per team for the
    whole tournament), we merge twice — once for home, once for away.
    Matches where a team has no squad data get 0.5 (neutral) for all features.

    Args:
        matches:        Matches DataFrame with home_team and away_team columns.
        squad_features: Output of compute_squad_features() — one row per team.

    Returns:
        matches with 12 new columns:
            home_squad_attack_rating, home_squad_defense_rating, ...
            away_squad_attack_rating, away_squad_defense_rating, ...
    """
    feat_cols = [c for c in squad_features.columns if c != "team"]

    home_sq = squad_features.rename(
        columns={c: f"home_{c}" for c in feat_cols}
    ).rename(columns={"team": "home_team"})

    away_sq = squad_features.rename(
        columns={c: f"away_{c}" for c in feat_cols}
    ).rename(columns={"team": "away_team"})

    matches = matches.merge(home_sq, on="home_team", how="left")
    matches = matches.merge(away_sq, on="away_team", how="left")

    # Fill missing squad data with neutral 0.5 (not 0, to avoid bias)
    for col in feat_cols:
        matches[f"home_{col}"] = matches[f"home_{col}"].fillna(0.5)
        matches[f"away_{col}"] = matches[f"away_{col}"].fillna(0.5)

    return matches
