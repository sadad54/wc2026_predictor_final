"""
src/features/build_features.py

Phase 3: Build the final feature matrix for model training.

Takes the preprocessed matches + Elo + form, then constructs
difference-based features (Team A minus Team B) that are the
actual inputs to the ML models.

The key insight: models learn from DIFFERENCES, not raw values.
A team with Elo 1700 vs a team with Elo 1400 is what matters,
not either rating in isolation.

Outputs:
  data/processed/features.parquet  ← model-ready feature matrix
  data/processed/matches_full.parquet  ← everything merged

Run from the project root:
    python -m src.features.build_features
"""

from pathlib import Path

import pandas as pd

# Robust import for Elo engine: different modules may expose different class names
try:
    from src.features.elo import EloEngine
except Exception:
    # Fallback: try common alternative names
    import importlib
    _elo_mod = importlib.import_module("src.features.elo")
    for _name in ("EloEngine", "Elo", "EloRating", "EloCalculator"):
        if hasattr(_elo_mod, _name):
            EloEngine = getattr(_elo_mod, _name)
            break
    else:
        raise ImportError("No Elo engine class found in src.features.elo (tried EloEngine, Elo, EloRating, EloCalculator)")
from src.features.form import compute_form_features
from src.utils import initialize_project, logger


def build_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct difference-based features from home/away paired columns.

    For each paired feature (home_X, away_X), we compute:
      - X_diff = home_X - away_X
      This is what the model sees: a signed difference where positive
      values favor the home team.

    We also keep a few absolute features (neutral ground flag,
    ranking values for imputation diagnostics).

    Args:
        df: Matches DataFrame with Elo and form columns.

    Returns:
        Feature DataFrame with one row per match.
    """
    features = pd.DataFrame()

    # ── Match identity (kept for validation/diagnostics, not model inputs) ──
    features["match_id"]    = df.index
    features["date"]        = df["date"]
    features["home_team"]   = df["home_team"]
    features["away_team"]   = df["away_team"]
    features["tournament"]  = df["tournament"]
    features["match_type"]  = df["match_type"]
    features["is_neutral"]  = df["neutral"].astype(int)

    # ── Target variable ──────────────────────────────────────────────────────
    # 0 = home win, 1 = draw, 2 = away win
    # This is the label we train classifiers to predict.
    outcome_map = {"W": 0, "D": 1, "L": 2}
    features["outcome"] = df["outcome"].map(outcome_map)

    # Also store raw scores for Dixon-Coles (which needs goal counts)
    features["home_score"] = df["home_score"]
    features["away_score"] = df["away_score"]

    # ── Elo features ─────────────────────────────────────────────────────────
    features["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]
    features["home_elo"] = df["home_elo_before"]   # kept for DC model
    features["away_elo"] = df["away_elo_before"]   # kept for DC model

    # ── FIFA ranking features ─────────────────────────────────────────────────
    # Rank difference: lower rank number = stronger team.
    # We flip the sign so positive = home team is ranked higher (lower number)
    features["rank_diff"] = df["away_rank"] - df["home_rank"]   # positive = home team ranked better
    features["home_rank"]  = df["home_rank"]
    features["away_rank"]  = df["away_rank"]

    # ── Form features ─────────────────────────────────────────────────────────
    features["form_points_diff"]         = df["home_form_points"]         - df["away_form_points"]
    features["form_goals_scored_diff"]   = df["home_form_goals_scored"]   - df["away_form_goals_scored"]
    features["form_goals_conceded_diff"] = df["home_form_goals_conceded"] - df["away_form_goals_conceded"]

    # ── Match context ─────────────────────────────────────────────────────────
    features["year"]  = df["date"].dt.year
    features["month"] = df["date"].dt.month   # captures seasonal effects

    return features


def impute_missing_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing FIFA ranking values.

    Teams that predate FIFA rankings (pre-1992) or newer nations
    won't have a ranking. We impute with the median rank for that
    confederation — failing that, global median.

    For simplicity here we use global median since confederation
    membership isn't in our dataset at this stage. A future improvement
    would join confederation data.

    Args:
        df: Features DataFrame with home_rank and away_rank columns.

    Returns:
        df with nulls filled.
    """
    median_rank = df[["home_rank", "away_rank"]].stack().median()
    before = df[["home_rank", "away_rank"]].isnull().sum().sum()

    df["home_rank"] = df["home_rank"].fillna(median_rank)
    df["away_rank"] = df["away_rank"].fillna(median_rank)
    df["rank_diff"] = df["away_rank"] - df["home_rank"]

    if before > 0:
        logger.info(f"  Imputed {before} missing ranking values with median ({median_rank:.0f})")

    return df


def run_feature_engineering(config: dict) -> None:
    processed_dir = Path(config["paths"]["processed_data"])

    # ── Load preprocessed matches ─────────────────────────────────────────────
    matches_path = processed_dir / "matches.parquet"
    if not matches_path.exists():
        raise FileNotFoundError("matches.parquet not found — run preprocess.py first")

    matches = pd.read_parquet(matches_path)
    logger.info(f"Loaded matches.parquet: {len(matches):,} rows")

    # ── Step 1: Elo ratings ───────────────────────────────────────────────────
    elo_engine = EloEngine(config)
    matches = elo_engine.compute(matches)

    # ── Step 2: Form features ─────────────────────────────────────────────────
    form_cfg = config["features"]["form"]
    matches = compute_form_features(
        matches,
        window=form_cfg["window_matches"],
        decay_factor=form_cfg["decay_factor"],
    )

    # ── Step 3: Build difference features ────────────────────────────────────
    features = build_difference_features(matches)
    features = impute_missing_rankings(features)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    # Full merged dataset (everything, for debugging and Dixon-Coles)
    matches.to_parquet(processed_dir / "matches_full.parquet", index=False)

    # Model-ready feature matrix
    features.to_parquet(processed_dir / "features.parquet", index=False)

    logger.info(f"Saved matches_full.parquet: {len(matches):,} rows, {len(matches.columns)} columns")
    logger.info(f"Saved features.parquet: {len(features):,} rows, {len(features.columns)} columns")

    # ── Quick sanity checks ───────────────────────────────────────────────────
    null_counts = features[["elo_diff", "form_points_diff", "rank_diff"]].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Null values in key features:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("  No nulls in key model features ✅")

    outcome_dist = features["outcome"].value_counts(normalize=True).sort_index()
    logger.info(f"  Outcome distribution (0=HW, 1=D, 2=AW): {outcome_dist.round(3).to_dict()}")

    logger.info("✅ Phase 3 feature engineering complete")


if __name__ == "__main__":
    config = initialize_project()
    run_feature_engineering(config)