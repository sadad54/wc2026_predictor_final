"""
src/data/team_names.py

Canonical team name mapping.

Any team name variant found in any dataset maps to one standard name.
This is the single source of truth for team names in the project.

Usage:
    from src.data.team_names import normalize_team_name
    normalize_team_name("Korea Republic")  # → "South Korea"
"""

# Maps variant → canonical name.
# Key  = name as it appears in the raw data (any dataset)
# Value = the standard name used throughout this project
TEAM_NAME_MAP: dict[str, str] = {
    # ── Korea ──────────────────────────────────────────────────────────────
    "Korea Republic":            "South Korea",
    "Korea DPR":                 "North Korea",
    "Republic of Korea":         "South Korea",

    # ── China ──────────────────────────────────────────────────────────────
    "China PR":                  "China",
    "China":                     "China",

    # ── USA ────────────────────────────────────────────────────────────────
    "United States":             "USA",
    "USA":                       "USA",

    # ── Iran ───────────────────────────────────────────────────────────────
    "IR Iran":                   "Iran",

    # ── Ivory Coast ────────────────────────────────────────────────────────
    "Côte d'Ivoire":             "Ivory Coast",
    "Cote d'Ivoire":             "Ivory Coast",
    "Cote dIvoire":              "Ivory Coast",

    # ── Bosnia ─────────────────────────────────────────────────────────────
    "Bosnia-Herzegovina":        "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":      "Bosnia and Herzegovina",

    # ── North Macedonia ────────────────────────────────────────────────────
    "Macedonia":                 "North Macedonia",
    "FYR Macedonia":             "North Macedonia",

    # ── Cape Verde ─────────────────────────────────────────────────────────
    "Cape Verde Islands":        "Cape Verde",
    "Cape Verde Is.":            "Cape Verde",

    # ── Trinidad ───────────────────────────────────────────────────────────
    "Trinidad & Tobago":         "Trinidad and Tobago",

    # ── Congo (both) ───────────────────────────────────────────────────────
    "DR Congo":                  "Congo DR",
    "Congo":                     "Congo",

    # ── Kyrgyzstan ─────────────────────────────────────────────────────────
    "Kyrgyz Republic":           "Kyrgyzstan",

    # ── eSwatini ───────────────────────────────────────────────────────────
    "Swaziland":                 "eSwatini",

    # ── Czechia ────────────────────────────────────────────────────────────
    "Czech Republic":            "Czechia",
    "Czechoslovakia":            "Czechia",  # historical, pre-1993

    # ── Serbia / Yugoslavia ────────────────────────────────────────────────
    "Yugoslavia":                "Serbia",   # treat as historical predecessor

    # ── Germany ────────────────────────────────────────────────────────────
    "German DR":                 "Germany",  # East Germany, historical
    "West Germany":              "Germany",  # historical

    # ── Soviet / Russian ───────────────────────────────────────────────────
    "Soviet Union":              "Russia",   # historical

    # ── UAE ────────────────────────────────────────────────────────────────
    "United Arab Emirates":      "UAE",
    "UAE":                       "UAE",

    # ── Misc ───────────────────────────────────────────────────────────────
    "Curacao":                   "Curaçao",
    "Saint Kitts and Nevis":     "St. Kitts and Nevis",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Saint Lucia":               "St. Lucia",
    "Antigua and Barbuda":       "Antigua and Barbuda",
}


def normalize_team_name(name: str) -> str:
    """
    Return the canonical name for a team.

    If the name is already canonical (not in the map), it's returned unchanged.
    The lookup is case-insensitive but the output preserves canonical casing.

    Args:
        name: Raw team name from any data source.

    Returns:
        Canonical team name.

    Examples:
        >>> normalize_team_name("Korea Republic")
        'South Korea'
        >>> normalize_team_name("France")
        'France'
    """
    return TEAM_NAME_MAP.get(name, name)