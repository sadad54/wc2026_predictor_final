"""
scripts/squads/02_fetch_transfermarkt_kaggle.py

Stage 2 of the squad-data pipeline: download the David Cariboo
"Football Data from Transfermarkt" dataset from Kaggle and extract:

    1. Market value (latest known value per player)
    2. Recent club-season stats (goals, appearances) from the last 2
       seasons that have non-empty data, to work around a known scraper
       gap in the 2024/25 season (see dcaribou/transfermarkt-datasets#349)

Why Kaggle instead of live Transfermarkt scraping:
    - No anti-bot/rate-limiting risk — this is a pre-scraped, weekly-
      refreshed bulk dataset (davidcariboo/player-scores on Kaggle).
    - Already structured into clean relational CSVs (players, appearances,
      player_valuations, clubs) — no HTML parsing needed.
    - Uses Kaggle credentials your project already has in .env.

Requires:
    KAGGLE_USERNAME and KAGGLE_KEY set (via .env, same as src/data/download.py)
    pip install kagglehub

Output:
    data/external/raw/transfermarkt_players.csv
        Columns: tm_player_id, tm_name, market_value_eur, current_club_name

    data/external/raw/transfermarkt_recent_form.csv
        Columns: tm_player_id, tm_name, recent_season_goals, recent_season_apps,
                  season_used

Run:
    python scripts/squads/02_fetch_transfermarkt_kaggle.py
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

OUTPUT_DIR = Path("data/external/raw")
KAGGLE_DATASET = "davidcariboo/player-scores"

# Only consider the last N distinct seasons when looking for recent form,
# to skip over the known 2024/25 data gap if a player's most recent
# season happens to be affected.
N_RECENT_SEASONS_TO_CHECK = 3
MIN_APPS_FOR_VALID_SEASON = 3  # a season with <3 apps isn't a meaningful form signal


def download_dataset() -> Path:
    """
    Download the Transfermarkt dataset via kagglehub (handles caching,
    auth, and versioning automatically). Falls back to the kaggle CLI
    package if kagglehub isn't available.
    """
    load_dotenv()
    os.environ.setdefault("KAGGLE_USERNAME", os.getenv("KAGGLE_USERNAME", ""))
    os.environ.setdefault("KAGGLE_KEY", os.getenv("KAGGLE_KEY", ""))

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise EnvironmentError(
            "KAGGLE_USERNAME/KAGGLE_KEY not set. Copy .env.example to .env "
            "and fill in your Kaggle credentials (same ones used by src/data/download.py)."
        )

    try:
        import kagglehub
        print(f"Downloading {KAGGLE_DATASET} via kagglehub (this may take a few minutes — ~223MB)...")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        return Path(path)
    except ImportError:
        pass

    # Fallback: use the kaggle CLI package (already a project dependency
    # path via src/data/download.py's pattern)
    import kaggle
    dest = OUTPUT_DIR / "transfermarkt_raw"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} via kaggle API (this may take a few minutes)...")
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(dest), unzip=True)
    return dest


def extract_market_values(dataset_dir: Path) -> pd.DataFrame:
    """
    Build a clean market-value table from players.csv.

    players.csv already has a `market_value_in_eur` column reflecting the
    most recent known valuation, so no need to join player_valuations.csv
    unless a historical trend is needed (not required here).

    Returns:
        DataFrame: tm_player_id, tm_name, market_value_eur, current_club_name
    """
    players_path = _find_file(dataset_dir, "players.csv")
    print(f"  Reading {players_path.name}...")
    players = pd.read_csv(players_path, low_memory=False)

    # Column names in this dataset are stable but verify the key ones exist
    name_col = "name" if "name" in players.columns else "player_name"
    mv_col = "market_value_in_eur"
    club_col = "current_club_name" if "current_club_name" in players.columns else "current_club_id"

    keep_cols = ["player_id", name_col, mv_col]
    if club_col in players.columns:
        keep_cols.append(club_col)

    out = players[keep_cols].copy()
    out = out.rename(columns={
        "player_id": "tm_player_id",
        name_col: "tm_name",
        mv_col: "market_value_eur",
        club_col: "current_club_name",
    })
    out["market_value_eur"] = out["market_value_eur"].fillna(0)

    print(f"  Extracted market values for {len(out):,} players "
          f"({(out['market_value_eur'] > 0).sum():,} with non-zero value)")
    return out


def extract_recent_form(dataset_dir: Path) -> pd.DataFrame:
    """
    Compute recent-season goals/appearances per player from appearances.csv.

    Strategy (to handle the known 2024/25 data gap):
        1. Group appearances by player and season.
        2. For each player, walk backward from the most recent season.
        3. Use the first season with >= MIN_APPS_FOR_VALID_SEASON appearances
           as the "recent form" season — skipping seasons with suspiciously
           low/zero data that likely reflect the known scraper gap rather
           than the player actually being unused.

    Returns:
        DataFrame: tm_player_id, recent_season_goals, recent_season_apps, season_used
    """
    appearances_path = _find_file(dataset_dir, "appearances.csv")
    print(f"  Reading {appearances_path.name} (this is the largest file, ~1.8M rows)...")
    appearances = pd.read_csv(appearances_path, low_memory=False)

    # Expected columns: player_id, game_id, date, goals, assists, minutes_played, etc.
    # 'date' lets us derive season; some versions of this dataset have a
    # direct 'season' column on games.csv we could join, but deriving from
    # date is more robust to schema drift.
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances = appearances.dropna(subset=["date"])

    # European football season convention: Aug-Dec = season starting that year,
    # Jan-Jul = season that started the previous year.
    appearances["season_year"] = appearances["date"].dt.year.where(
        appearances["date"].dt.month >= 8,
        appearances["date"].dt.year - 1,
    )

    season_agg = (
        appearances.groupby(["player_id", "season_year"])
        .agg(goals=("goals", "sum"), apps=("player_id", "count"))
        .reset_index()
    )

    print(f"  Aggregated to {len(season_agg):,} player-season rows")

    # For each player, pick the most recent VALID season (apps >= threshold),
    # checking up to N_RECENT_SEASONS_TO_CHECK seasons back.
    season_agg = season_agg.sort_values(["player_id", "season_year"], ascending=[True, False])

    rows = []
    for player_id, group in season_agg.groupby("player_id"):
        candidates = group.head(N_RECENT_SEASONS_TO_CHECK)
        valid = candidates[candidates["apps"] >= MIN_APPS_FOR_VALID_SEASON]

        if not valid.empty:
            best = valid.iloc[0]  # already sorted descending by season_year
            rows.append({
                "tm_player_id": player_id,
                "recent_season_goals": int(best["goals"]),
                "recent_season_apps": int(best["apps"]),
                "season_used": int(best["season_year"]),
            })
        elif not candidates.empty:
            # No season met the threshold — use the best available anyway,
            # flagged via season_used, rather than dropping the player entirely.
            best = candidates.iloc[0]
            rows.append({
                "tm_player_id": player_id,
                "recent_season_goals": int(best["goals"]),
                "recent_season_apps": int(best["apps"]),
                "season_used": int(best["season_year"]),
            })

    out = pd.DataFrame(rows)
    print(f"  Resolved recent-form season for {len(out):,} players")
    return out


def _find_file(dataset_dir: Path, filename: str) -> Path:
    """
    Locate a file within the downloaded dataset directory, since kagglehub
    and the kaggle CLI may nest files differently across versions.
    """
    matches = list(dataset_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"{filename} not found under {dataset_dir}. "
            f"Available files: {[p.name for p in dataset_dir.rglob('*.csv')]}"
        )
    return matches[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_dir = download_dataset()
    print(f"Dataset available at: {dataset_dir}")

    print("\n[1/2] Extracting market values...")
    market_values = extract_market_values(dataset_dir)
    mv_path = OUTPUT_DIR / "transfermarkt_players.csv"
    market_values.to_csv(mv_path, index=False)
    print(f"  Saved → {mv_path}")

    print("\n[2/2] Computing recent club-season form...")
    recent_form = extract_recent_form(dataset_dir)
    # Join names back in for the fuzzy-matching stage
    recent_form = recent_form.merge(
        market_values[["tm_player_id", "tm_name"]], on="tm_player_id", how="left"
    )
    form_path = OUTPUT_DIR / "transfermarkt_recent_form.csv"
    recent_form.to_csv(form_path, index=False)
    print(f"  Saved → {form_path}")

    print("\n✅ Stage 2 complete")


if __name__ == "__main__":
    main()