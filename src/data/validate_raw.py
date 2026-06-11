"""
src/data/validate_raw.py

Phase 1: Validate downloaded raw data.

Checks row counts, column presence, date ranges, and obvious anomalies.
Exits with code 1 if any critical check fails.

Run from the project root:
    python -m src.data.validate_raw
"""

import sys
from pathlib import Path

import pandas as pd

from src.utils import initialize_project, logger


def validate_results(path: Path, start_year: int) -> int:
    """Validate the match results CSV. Returns number of failures."""
    failures = 0
    df = pd.read_csv(path, parse_dates=["date"])

    logger.info(f"results.csv: {len(df):,} rows, {df['date'].min().year}–{df['date'].max().year}")

    if len(df) < 40_000:
        logger.error(f"results.csv has only {len(df):,} rows — expected 40,000+")
        failures += 1

    required_cols = {"date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"results.csv missing columns: {missing}")
        failures += 1

    recent = df[df["date"].dt.year >= start_year]
    logger.info(f"  Rows from {start_year}+: {len(recent):,}")
    if len(recent) < 5_000:
        logger.error(f"Too few rows from {start_year}+: {len(recent):,}")
        failures += 1

    null_scores = df[["home_score", "away_score"]].isnull().sum().sum()
    if null_scores > 0:
        logger.warning(f"  {null_scores} null scores (will be handled in preprocessing)")

    return failures


def validate_rankings(path: Path) -> int:
    """Validate the FIFA rankings CSV. Returns number of failures."""
    failures = 0
    df = pd.read_csv(path, parse_dates=["rank_date"])

    logger.info(f"rankings.csv: {len(df):,} rows")

    required_cols = {"rank_date", "country_full", "rank", "total_points"}
    # Accept either total_points or confederation_points — Kaggle dataset varies
    alt_cols = {"rank_date", "country_full", "rank"}
    has_required = required_cols.issubset(set(df.columns)) or alt_cols.issubset(set(df.columns))

    if not has_required:
        logger.error(f"rankings.csv missing expected columns. Found: {list(df.columns)}")
        failures += 1
    else:
        logger.info(f"  Date range: {df['rank_date'].min().year}–{df['rank_date'].max().year}")
        logger.info(f"  Unique teams: {df['country_full'].nunique()}")

    return failures


def main() -> None:
    config = initialize_project()
    raw_dir = Path(config["paths"]["raw_data"])
    start_year = config["data"]["date_filter"]["start_year"]

    total_failures = 0

    results_path = raw_dir / "results.csv"
    rankings_path = raw_dir / "rankings.csv"

    if not results_path.exists():
        logger.error("results.csv not found — run download.py first")
        sys.exit(1)

    if not rankings_path.exists():
        logger.error("rankings.csv not found — run download.py first")
        sys.exit(1)

    total_failures += validate_results(results_path, start_year)
    total_failures += validate_rankings(rankings_path)

    if total_failures == 0:
        logger.info("✅ All raw data validation checks passed")
    else:
        logger.error(f"❌ {total_failures} validation check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()