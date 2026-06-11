"""
src/data/preprocess.py

Phase 2: Clean, merge, and structure raw data for feature engineering.

Reads from:  data/raw/results.csv, data/raw/rankings.csv
Writes to:   data/processed/matches.parquet
             data/processed/rankings_clean.parquet

Run from the project root:
    python -m src.data.preprocess
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.team_names import normalize_team_name
from src.utils import initialize_project, logger


# ─────────────────────────────────────────────────────────────────────────────
# Match type classification
# ─────────────────────────────────────────────────────────────────────────────

# Keywords in tournament names that tell us the importance of the match.
# These map to the weight keys in config['data']['match_type_weights'].
TOURNAMENT_CLASSIFICATION: list[tuple[str, str]] = [
    # Most important first — order matters (first match wins)
    ("FIFA World Cup",          "world_cup"),
    ("World Cup",               "world_cup"),
    ("UEFA Euro",               "continental"),
    ("Copa America",            "continental"),
    ("Africa Cup of Nations",   "continental"),
    ("Asian Cup",               "continental"),
    ("CONCACAF Gold Cup",       "continental"),
    ("Copa América",            "continental"),
    ("Confederations Cup",      "continental"),
    ("Nations League",          "continental"),
    ("qualifier",               "world_cup_qualifier"),
    ("qualification",           "world_cup_qualifier"),
    ("Qualifying",              "world_cup_qualifier"),
    ("friendly",                "friendly"),
    ("Friendly",                "friendly"),
]


def classify_tournament(tournament_name: str) -> str:
    """
    Map a raw tournament name to one of four match-type categories.

    Categories: 'world_cup', 'world_cup_qualifier', 'continental', 'friendly'
    Falls back to 'friendly' if no keyword matches.

    Args:
        tournament_name: Raw string from results.csv 'tournament' column.

    Returns:
        Match type string.
    """
    for keyword, match_type in TOURNAMENT_CLASSIFICATION:
        if keyword.lower() in tournament_name.lower():
            return match_type
    return "friendly"  # Default for anything unclassified


# ─────────────────────────────────────────────────────────────────────────────
# Results cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_results(raw_path: Path, start_year: int) -> pd.DataFrame:
    """
    Load and clean the match results dataset.

    Steps:
      1. Parse dates and filter to start_year+
      2. Drop rows with null scores
      3. Normalize team names
      4. Add match_type and importance_weight columns
      5. Add outcome column (home perspective: W/D/L)

    Args:
        raw_path: Path to results.csv
        start_year: Only keep matches from this year onward.

    Returns:
        Cleaned DataFrame with one row per match.
    """
    logger.info("Cleaning results.csv...")
    df = pd.read_csv(raw_path, parse_dates=["date"])

    # Filter to modern era
    df = df[df["date"].dt.year >= start_year].copy()
    logger.info(f"  After year filter ({start_year}+): {len(df):,} rows")

    # Drop matches with missing scores (unplayed/postponed matches)
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"  Dropped {dropped} rows with null scores")

    # Cast scores to int
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Normalize team names
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)

    # Add match type and Elo K-factor weight
    df["match_type"] = df["tournament"].apply(classify_tournament)

    # Outcome from home team perspective
    df["outcome"] = np.select(
        [df["home_score"] > df["away_score"],
         df["home_score"] == df["away_score"]],
        ["W", "D"],
        default="L",
    )

    # Sort chronologically — critical for any time-based feature engineering
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(f"  Clean results: {len(df):,} matches | match types: {df['match_type'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rankings cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_rankings(raw_path: Path) -> pd.DataFrame:
    """
    Load and clean the FIFA rankings dataset.

    Steps:
      1. Parse dates
      2. Normalize team names
      3. Keep only the columns we actually use

    Args:
        raw_path: Path to rankings.csv

    Returns:
        Cleaned rankings DataFrame with columns: rank_date, team, rank, points
    """
    logger.info("Cleaning rankings.csv...")
    df = pd.read_csv(raw_path, parse_dates=["rank_date"])

    # Normalize the team name column (Kaggle dataset uses 'country_full')
    if "country_full" in df.columns:
        df = df.rename(columns={"country_full": "team"})
    elif "Team" in df.columns:
        df = df.rename(columns={"Team": "team"})

    df["team"] = df["team"].map(normalize_team_name)

    # Points column varies by Kaggle dataset version
    if "total_points" in df.columns:
        df = df.rename(columns={"total_points": "points"})
    elif "total_points_after_substitution" in df.columns:
        df = df.rename(columns={"total_points_after_substitution": "points"})

    # Keep only what we need
    keep_cols = [c for c in ["rank_date", "team", "rank", "points"] if c in df.columns]
    df = df[keep_cols].sort_values(["team", "rank_date"]).reset_index(drop=True)

    logger.info(f"  Clean rankings: {len(df):,} rows | {df['team'].nunique()} teams")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Merge: attach ranking at time of match
# ─────────────────────────────────────────────────────────────────────────────

def attach_rankings_at_match_date(
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each match, find the most recent FIFA ranking for both teams
    available BEFORE that match date (no data leakage).

    This is a backward-looking merge: for a match on 2018-06-15, we use
    the ranking published on or before 2018-06-15, not the next update.

    Args:
        matches:  Cleaned matches DataFrame.
        rankings: Cleaned rankings DataFrame.

    Returns:
        matches with four new columns:
            home_rank, home_rank_points, away_rank, away_rank_points
    """
    logger.info("Attaching FIFA rankings at match date (no-leakage merge)...")

    def get_rank_at_date(team: str, date: pd.Timestamp, rankings: pd.DataFrame):
        """Return (rank, points) for team on or before date."""
        team_rankings = rankings[rankings["team"] == team]
        past = team_rankings[team_rankings["rank_date"] <= date]
        if past.empty:
            return np.nan, np.nan
        latest = past.sort_values("rank_date").iloc[-1]
        rank = latest["rank"]
        points = latest["points"] if "points" in latest.index else np.nan
        return rank, points

    # Vectorised approach: merge_asof is much faster than row-by-row lookup
    # We need to do this separately for home and away teams.

    # Prepare rankings for merge_asof
    rankings_sorted = rankings.sort_values("rank_date").copy()

    # Home team rankings
    home_merge = pd.merge_asof(
        matches[["date", "home_team"]].sort_values("date"),
        rankings_sorted.rename(columns={"team": "home_team", "rank": "home_rank", "points": "home_rank_points"}),
        left_on="date",
        right_on="rank_date",
        by="home_team",
        direction="backward",
    )[["date", "home_team", "home_rank", "home_rank_points"]]

    # Away team rankings
    away_merge = pd.merge_asof(
        matches[["date", "away_team"]].sort_values("date"),
        rankings_sorted.rename(columns={"team": "away_team", "rank": "away_rank", "points": "away_rank_points"}),
        left_on="date",
        right_on="rank_date",
        by="away_team",
        direction="backward",
    )[["date", "away_team", "away_rank", "away_rank_points"]]

    # Merge back onto matches
    matches = matches.sort_values("date")
    matches = matches.merge(home_merge, on=["date", "home_team"], how="left")
    matches = matches.merge(away_merge, on=["date", "away_team"], how="left")

    missing_home = matches["home_rank"].isnull().sum()
    missing_away = matches["away_rank"].isnull().sum()
    if missing_home > 0 or missing_away > 0:
        logger.warning(f"  Missing rankings: {missing_home} home, {missing_away} away (teams pre-date FIFA rankings)")

    return matches.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(config: dict) -> None:
    raw_dir = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_data"])
    start_year = config["data"]["date_filter"]["start_year"]

    # Clean
    matches = clean_results(raw_dir / "results.csv", start_year)
    rankings = clean_rankings(raw_dir / "rankings.csv")

    # Merge rankings onto matches (no leakage)
    matches = attach_rankings_at_match_date(matches, rankings)

    # Save
    processed_dir.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(processed_dir / "matches.parquet", index=False)
    rankings.to_parquet(processed_dir / "rankings_clean.parquet", index=False)

    logger.info(f"Saved matches.parquet: {len(matches):,} rows, {matches.columns.tolist()}")
    logger.info(f"Saved rankings_clean.parquet: {len(rankings):,} rows")
    logger.info("✅ Phase 2 preprocessing complete")


if __name__ == "__main__":
    config = initialize_project()
    run_preprocessing(config)