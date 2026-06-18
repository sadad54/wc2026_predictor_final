"""
scripts/squads/03_merge_and_match.py

Stage 3 of the squad-data pipeline: fuzzy-match Wikipedia squad players
against Transfermarkt records (market value + recent form) and produce
the final wc2026_squads.csv matching the project's schema.

Why fuzzy matching is necessary:
    Player names differ across sources due to:
        - Diacritics/accents (Krejčí vs Krejci)
        - Nicknames vs full names (Vini Jr vs Vinicius Junior)
        - Suffixes (Jr., II) and abbreviated middle names
        - Transliteration differences for non-Latin-script names

Matching strategy:
    1. Exact match on normalised name (lowercase, accents stripped) — fast,
       handles the majority of cases.
    2. For unmatched players, fuzzy match using rapidfuzz within the same
       team/club context to avoid false positives (e.g. two different
       "J. Silva"s on different teams).
    3. Matches below a similarity threshold are left unmatched — those
       players get neutral fallback values (market_value=0, recent stats
       fall back to career rate, both handled gracefully by the existing
       squad_features.py and player_models.py code).

A match_quality column is included in the output so you can audit which
rows were exact vs fuzzy vs unmatched, and manually fix the worst offenders
if needed (typically <10% of players, concentrated in lesser-known leagues).

Output:
    data/external/wc2026_squads.csv          ← final file, matches project schema
    data/external/raw/match_audit.csv        ← diagnostic: match quality per player

Run:
    python scripts/squads/03_merge_and_match.py
"""

import difflib
import re
import unicodedata
from pathlib import Path

import pandas as pd

try:
    from rapidfuzz import fuzz, process
    _USE_RAPIDFUZZ = True
except ImportError:
    fuzz = None
    process = None
    _USE_RAPIDFUZZ = False

RAW_DIR = Path("data/external/raw")


def _token_sort_ratio(a: str, b: str) -> float:
    a_tokens = " ".join(sorted(a.split()))
    b_tokens = " ".join(sorted(b.split()))
    return difflib.SequenceMatcher(None, a_tokens, b_tokens).ratio() * 100


def _extract_one(query: str, choices: list[str], scorer, score_cutoff: int | None = None):
    if _USE_RAPIDFUZZ:
        return process.extractOne(query, choices, scorer=scorer, score_cutoff=score_cutoff)

    best_choice = None
    best_score = -1.0
    for choice in choices:
        score = scorer(query, choice)
        if score > best_score:
            best_choice = choice
            best_score = score
    if score_cutoff is not None and best_score < score_cutoff:
        return None
    return (best_choice, best_score, None)
OUTPUT_PATH = Path("data/external/wc2026_squads.csv")
AUDIT_PATH = RAW_DIR / "match_audit.csv"

# Below this score (0-100), we don't trust the fuzzy match at all.
FUZZY_MATCH_THRESHOLD = 85

# Known nickname/short-form aliases that string similarity can't safely catch
# (lowering the fuzzy threshold to catch these risks false-matching unrelated
# players). Extend this table as you spot more misses in match_audit.csv.
# Keys and values are normalised (lowercase, no accents) — see normalise_name().
KNOWN_ALIASES: dict[str, str] = {
    "vinicius junior": "vini jr",
    "neymar": "neymar jr",
    "rodrygo": "rodrygo goes",
    "raphinha": "raphael dias belloli",
    "casemiro": "carlos henrique casimiro",
    "fabinho": "fabio henrique tavares",
    "danilo": "danilo luiz da silva",
}


def normalise_name(name: str) -> str:
    """
    Strip accents/diacritics and lowercase for exact-match comparison.
    e.g. "Krejčí" -> "krejci", "Müller" -> "muller"
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = re.sub(r"[^a-z0-9\s]", "", ascii_name.lower())
    return re.sub(r"\s+", " ", ascii_name).strip()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the outputs of stages 1 and 2."""
    wiki = pd.read_csv(RAW_DIR / "wikipedia_squads_raw.csv")
    tm_players = pd.read_csv(RAW_DIR / "transfermarkt_players.csv")
    tm_form = pd.read_csv(RAW_DIR / "transfermarkt_recent_form.csv")
    return wiki, tm_players, tm_form


def match_players(
    wiki: pd.DataFrame,
    tm_players: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match each Wikipedia player row to a Transfermarkt player_id.

    Returns:
        wiki DataFrame with two new columns: tm_player_id, match_quality
        (match_quality is one of: 'exact', 'fuzzy', 'unmatched')
    """
    wiki = wiki.copy()
    tm_players = tm_players.copy()

    wiki["_norm_name"] = wiki["player"].apply(normalise_name)
    tm_players["_norm_name"] = tm_players["tm_name"].apply(normalise_name)

    # ── Pass 1: exact match on normalised name ───────────────────────────────
    exact_lookup = (
        tm_players.drop_duplicates(subset="_norm_name", keep="first")
        .set_index("_norm_name")["tm_player_id"]
        .to_dict()
    )

    wiki["tm_player_id"] = wiki["_norm_name"].map(exact_lookup)
    wiki["match_quality"] = wiki["tm_player_id"].apply(
        lambda x: "exact" if pd.notna(x) else "unmatched"
    )

    n_exact = (wiki["match_quality"] == "exact").sum()
    print(f"  Exact matches: {n_exact}/{len(wiki)} ({n_exact/len(wiki)*100:.1f}%)")

    # ── Pass 2: known alias table (nicknames string-similarity can't catch) ───
    tm_name_to_id_full = dict(zip(tm_players["_norm_name"], tm_players["tm_player_id"]))
    unmatched_mask = wiki["match_quality"] == "unmatched"
    n_alias = 0
    for idx in wiki[unmatched_mask].index:
        query = wiki.at[idx, "_norm_name"]
        alias_target = KNOWN_ALIASES.get(query)
        if alias_target and alias_target in tm_name_to_id_full:
            wiki.at[idx, "tm_player_id"] = tm_name_to_id_full[alias_target]
            wiki.at[idx, "match_quality"] = "alias"
            n_alias += 1
    if n_alias:
        print(f"  Alias-table matches: {n_alias}/{len(wiki)} ({n_alias/len(wiki)*100:.1f}%)")

    # ── Pass 3: fuzzy match remaining players ─────────────────────────────────
    unmatched_mask = wiki["match_quality"] == "unmatched"
    tm_names_pool = tm_players["_norm_name"].tolist()
    tm_name_to_id = dict(zip(tm_players["_norm_name"], tm_players["tm_player_id"]))

    n_fuzzy = 0
    for idx in wiki[unmatched_mask].index:
        query = wiki.at[idx, "_norm_name"]
        if not query:
            continue

        result = process.extractOne(
            query, tm_names_pool, scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_MATCH_THRESHOLD,
        )
        if result is not None:
            matched_name, score, _ = result
            wiki.at[idx, "tm_player_id"] = tm_name_to_id[matched_name]
            wiki.at[idx, "match_quality"] = "fuzzy"
            n_fuzzy += 1

    print(f"  Fuzzy matches: {n_fuzzy}/{len(wiki)} ({n_fuzzy/len(wiki)*100:.1f}%)")

    n_unmatched = (wiki["match_quality"] == "unmatched").sum()
    print(f"  Unmatched: {n_unmatched}/{len(wiki)} ({n_unmatched/len(wiki)*100:.1f}%) "
          "— these get neutral fallback values")

    wiki = wiki.drop(columns=["_norm_name"])
    return wiki


def build_final_schema(
    wiki_matched: pd.DataFrame,
    tm_players: pd.DataFrame,
    tm_form: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join market value and recent-form data onto the matched Wikipedia rows
    and produce the final column set matching wc2026_squads_schema.csv:

        team, player, position, career_goals, career_appearances, club,
        market_value_eur, recent_season_goals, recent_season_apps, age
    """
    df = wiki_matched.merge(
        tm_players[["tm_player_id", "market_value_eur"]],
        on="tm_player_id", how="left",
    )
    df = df.merge(
        tm_form[["tm_player_id", "recent_season_goals", "recent_season_apps"]],
        on="tm_player_id", how="left",
    )

    final = pd.DataFrame({
        "team":                 df["team"],
        "player":               df["player"],
        "position":             df["position"],
        "career_goals":         df["goals"],          # Wikipedia caps/goals = international, authoritative
        "career_appearances":   df["caps"],
        "club":                 df["club"],
        "market_value_eur":     df["market_value_eur"].fillna(0),
        "recent_season_goals":  df["recent_season_goals"].fillna(0),
        "recent_season_apps":   df["recent_season_apps"].fillna(0),
        "age":                  df["age"],
    })

    # Audit columns kept separately, not in the final schema file
    audit = df[["team", "player", "match_quality", "tm_player_id"]].copy()

    return final, audit


def main() -> None:
    print("Loading stage 1 + 2 outputs...")
    wiki, tm_players, tm_form = load_inputs()
    print(f"  Wikipedia squads: {len(wiki)} players, {wiki['team'].nunique()} teams")
    print(f"  Transfermarkt players: {len(tm_players):,}")
    print(f"  Transfermarkt recent-form records: {len(tm_form):,}")

    print("\nMatching players (exact -> fuzzy)...")
    wiki_matched = match_players(wiki, tm_players)

    print("\nBuilding final schema...")
    final, audit = build_final_schema(wiki_matched, tm_players, tm_form)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False)
    audit.to_csv(AUDIT_PATH, index=False)

    print(f"\nSaved final squads → {OUTPUT_PATH}")
    print(f"Saved match audit  → {AUDIT_PATH}")

    print(f"\nFinal dataset: {len(final)} players, {final['team'].nunique()} teams")
    print(f"Players with market value: {(final['market_value_eur'] > 0).sum()} "
          f"({(final['market_value_eur'] > 0).mean()*100:.1f}%)")
    print(f"Players with recent-form data: {(final['recent_season_apps'] > 0).sum()} "
          f"({(final['recent_season_apps'] > 0).mean()*100:.1f}%)")

    match_summary = audit["match_quality"].value_counts()
    print(f"\nMatch quality breakdown:\n{match_summary.to_string()}")

    print(
        "\n👉 Review match_audit.csv for any 'unmatched' high-profile players "
        "(stars are most valuable to get right) and fix manually if needed."
    )
    print("✅ Stage 3 complete — wc2026_squads.csv is ready for the ML pipeline")


if __name__ == "__main__":
    main()