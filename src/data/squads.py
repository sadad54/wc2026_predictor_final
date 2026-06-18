"""
Shared squad-data loading and normalisation helpers.

The squad scrape currently lands in data/external/raw/wc2026_squads.csv and
uses Wikipedia-oriented column names. Downstream modelling code expects a
canonical tournament schema, so this module is the single place that adapts
raw squad files into that shape.
"""

from pathlib import Path
from typing import Iterable

import pandas as pd
from loguru import logger

from src.data.team_names import normalize_team_name


SQUAD_FILENAME = "wc2026_squads.csv"

_COLUMN_ALIASES = {
    "caps": "career_appearances",
    "goals": "career_goals",
}

_TOURNAMENT_NAME_FIXES = {
    "Curaçao": "Curacao",
    "CuraÃ§ao": "Curacao",
    "Czech Republic": "Czechia",
}


def resolve_squad_path(external_data_dir: Path) -> Path | None:
    """
    Return the first squad CSV path that exists.

    Preferred path is data/external/wc2026_squads.csv. The current scraper
    writes to data/external/raw/wc2026_squads.csv, so that is supported as a
    fallback until the pipeline is consolidated.
    """
    external_data_dir = Path(external_data_dir)
    candidates = [
        external_data_dir / SQUAD_FILENAME,
        external_data_dir / "raw" / SQUAD_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def canonicalize_squad_team_name(name: str) -> str:
    """Normalise team names to the tournament naming used by wc2026_groups.csv."""
    normalized = normalize_team_name(str(name).strip())
    return _TOURNAMENT_NAME_FIXES.get(normalized, normalized)


def load_squad_data(
    external_data_dir: Path,
    required_teams: Iterable[str] | None = None,
    strict: bool = False,
) -> pd.DataFrame | None:
    """
    Load and normalise WC 2026 squad data.

    Args:
        external_data_dir: Path to data/external.
        required_teams: Optional teams that should be present in the squad file.
        strict: If True, missing required teams raise ValueError. If False, they
            are logged as warnings and the normalised DataFrame is still returned.

    Returns:
        Normalised squad DataFrame, or None when no squad file exists.
    """
    squad_path = resolve_squad_path(external_data_dir)
    if squad_path is None:
        logger.warning(
            "Squad data not found at data/external/wc2026_squads.csv or "
            "data/external/raw/wc2026_squads.csv."
        )
        return None

    squads = pd.read_csv(squad_path)
    squads = normalise_squad_schema(squads)

    logger.info(
        f"Loaded squad data: {len(squads)} players, "
        f"{squads['team'].nunique()} teams from {squad_path}"
    )

    if required_teams is not None:
        validate_squad_coverage(squads, required_teams, strict=strict)

    return squads


def normalise_squad_schema(squads_df: pd.DataFrame) -> pd.DataFrame:
    """Return squad data with canonical column names and tournament team names."""
    df = squads_df.copy()
    df = df.rename(columns={k: v for k, v in _COLUMN_ALIASES.items() if k in df.columns})

    if "team" not in df.columns:
        raise ValueError("Squad data is missing required column: team")
    if "player" not in df.columns:
        raise ValueError("Squad data is missing required column: player")

    df["team"] = df["team"].map(canonicalize_squad_team_name)
    df["player"] = df["player"].astype(str).str.strip()

    if "position" not in df.columns:
        df["position"] = "MF"

    for col in ["career_goals", "career_appearances"]:
        if col not in df.columns:
            df[col] = 0

    return df


def validate_squad_coverage(
    squads_df: pd.DataFrame,
    required_teams: Iterable[str],
    strict: bool = False,
) -> None:
    """Validate that every tournament team has squad rows."""
    required = {canonicalize_squad_team_name(team) for team in required_teams}
    present = set(squads_df["team"].dropna().map(canonicalize_squad_team_name))
    missing = sorted(required - present)
    if not missing:
        return

    message = (
        f"Squad data is missing {len(missing)} tournament teams: {missing}. "
        "Affected teams will use neutral squad features and generic scorers."
    )
    if strict:
        raise ValueError(message)
    logger.warning(message)
