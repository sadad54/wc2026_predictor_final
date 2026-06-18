"""
src/features/build_features.py

Phase 3: Build the final feature matrix for model training.

Takes the preprocessed matches + Elo + form + squad features, then constructs
difference-based features (Team A minus Team B) that are the actual inputs
to the ML models.

The key insight: models learn from DIFFERENCES, not raw values.
A team with Elo 1700 vs a team with Elo 1400 is what matters,
not either rating in isolation. The same applies to squad features.

Feature matrix size:
    Base (8):   elo_diff, rank_diff, form_points_diff,
                form_goals_scored_diff, form_goals_conceded_diff,
                is_neutral, year, month
    Squad (6):  squad_attack_diff, squad_defense_diff, squad_depth_diff,
                squad_experience_diff, squad_form_diff, squad_age_balance_diff
    Total: 14 features

Squad features are squad-level signals computed from wc2026_squads.csv.
When squad data is unavailable (file not yet created), squad features are
set to 0.0 for all matches so the pipeline remains fully operational.

Outputs:
  data/processed/features.parquet         ← model-ready feature matrix
  data/processed/matches_full.parquet     ← everything merged

Run from the project root:
    python -m src.features.build_features
"""

from pathlib import Path

import pandas as pd

# Robust import for Elo engine
try:
    from src.features.elo import EloEngine
except Exception:
    import importlib
    _elo_mod = importlib.import_module("src.features.elo")
    for _name in ("EloEngine", "Elo", "EloRating", "EloCalculator"):
        if hasattr(_elo_mod, _name):
            EloEngine = getattr(_elo_mod, _name)
            break
    else:
        raise ImportError("No Elo engine class found in src.features.elo")

from src.features.form import compute_form_features
from src.features.squad_features import load_squad_features, attach_squad_features_to_matches
from src.utils import initialize_project, logger


# Squad feature column names (base names, without home_/away_ prefix)
SQUAD_FEATURE_COLS = [
    "squad_attack_rating",
    "squad_defense_rating",
    "squad_depth_rating",
    "squad_experience_rating",
    "squad_form_rating",
    "squad_age_balance",
]


def build_difference_features(df: pd.DataFrame, has_squad_features: bool) -> pd.DataFrame:
    """
    Construct difference-based features from home/away paired columns.

    For each paired feature (home_X, away_X), we compute:
        X_diff = home_X - away_X
    This is what the model sees: a signed difference where positive
    values favour the home team.

    Args:
        df:                 Matches DataFrame with Elo, form, and optionally squad columns.
        has_squad_features: If True, compute squad difference features too.

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
    outcome_map = {"W": 0, "D": 1, "L": 2}
    features["outcome"] = df["outcome"].map(outcome_map)

    # Raw scores for Dixon-Coles
    features["home_score"] = df["home_score"]
    features["away_score"] = df["away_score"]

    # ── Elo features ─────────────────────────────────────────────────────────
    features["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]
    features["home_elo"] = df["home_elo_before"]
    features["away_elo"] = df["away_elo_before"]

    # ── FIFA ranking features ─────────────────────────────────────────────────
    features["rank_diff"] = df["away_rank"] - df["home_rank"]
    features["home_rank"] = df["home_rank"]
    features["away_rank"] = df["away_rank"]

    # ── Form features ─────────────────────────────────────────────────────────
    features["form_points_diff"]         = df["home_form_points"]         - df["away_form_points"]
    features["form_goals_scored_diff"]   = df["home_form_goals_scored"]   - df["away_form_goals_scored"]
    features["form_goals_conceded_diff"] = df["home_form_goals_conceded"] - df["away_form_goals_conceded"]

    # ── Squad difference features ─────────────────────────────────────────────
    # These are 0.0 for all historical matches (no historical squad CSV exists).
    # For 2026 prediction matches, they carry real signal once the CSV is loaded.
    if has_squad_features:
        for col in SQUAD_FEATURE_COLS:
            home_col = f"home_{col}"
            away_col = f"away_{col}"
            if home_col in df.columns and away_col in df.columns:
                features[f"{col}_diff"] = df[home_col] - df[away_col]
            else:
                features[f"{col}_diff"] = 0.0
    else:
        # No squad data available: fill with 0.0 (neutral — no squad signal)
        for col in SQUAD_FEATURE_COLS:
            features[f"{col}_diff"] = 0.0

    # ── Match context ─────────────────────────────────────────────────────────
    features["year"]  = df["date"].dt.year
    features["month"] = df["date"].dt.month

    return features


def impute_missing_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing FIFA ranking values with global median.

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
    external_dir  = Path(config["paths"]["external_data"])

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

    # ── Step 3: Squad features (optional — graceful fallback if CSV missing) ──
    squad_features = load_squad_features(
        external_dir,
        min_appearances=config.get("data", {}).get("players", {}).get("min_appearances_for_rate", 5),
    )

    has_squad_features = squad_features is not None
    if has_squad_features:
        matches = attach_squad_features_to_matches(matches, squad_features)
        logger.info("  Squad features attached to matches ✅")
    else:
        logger.warning(
            "  Running WITHOUT squad features — add data/external/wc2026_squads.csv "
            "to unlock 6 additional model features."
        )

    # ── Step 4: Build difference features ────────────────────────────────────
    features = build_difference_features(matches, has_squad_features)
    features = impute_missing_rankings(features)

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    matches.to_parquet(processed_dir / "matches_full.parquet", index=False)
    features.to_parquet(processed_dir / "features.parquet", index=False)

    logger.info(f"Saved matches_full.parquet: {len(matches):,} rows, {len(matches.columns)} columns")
    logger.info(f"Saved features.parquet: {len(features):,} rows, {len(features.columns)} columns")

    # ── Quick sanity checks ───────────────────────────────────────────────────
    key_features = ["elo_diff", "form_points_diff", "rank_diff"]
    null_counts = features[key_features].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Null values in key features:\n{null_counts[null_counts > 0]}")
    else:
        logger.info("  No nulls in key model features ✅")

    outcome_dist = features["outcome"].value_counts(normalize=True).sort_index()
    logger.info(f"  Outcome distribution (0=HW, 1=D, 2=AW): {outcome_dist.round(3).to_dict()}")

    n_squad_cols = len([c for c in features.columns if "squad" in c])
    logger.info(f"  Squad feature columns: {n_squad_cols} ({'active' if has_squad_features else 'zero-filled — add CSV to activate'})")
    logger.info(f"  Total feature columns: {len(features.columns)}")
    logger.info("✅ Phase 3 feature engineering complete")


if __name__ == "__main__":
    config = initialize_project()
    run_feature_engineering(config)