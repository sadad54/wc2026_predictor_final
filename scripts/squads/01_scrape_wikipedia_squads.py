"""
scripts/squads/01_scrape_wikipedia_squads.py

Stage 1 of the squad-data pipeline: scrape the official 2026 FIFA World Cup
squad lists from Wikipedia.

Source: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads
This single page contains one wikitable per team (48 tables total), each
with columns: No., Pos., Player, Date of birth (age), Caps, Goals, Club.

Why Wikipedia and not Transfermarkt for this stage:
    - Wikipedia's squad data is sourced directly from official FIFA
      submissions, is current as of the official June 2026 announcement,
      and the page structure is stable (one <h3>/<h4> team heading + one
      <table class="wikitable"> per team, in document order).
    - No rate limiting, no anti-bot measures, no auth required.

Output:
    data/external/raw/wikipedia_squads_raw.csv
    Columns: team, jersey_no, position, player, dob, age, caps, goals, club

Run:
    python scripts/squads/01_scrape_wikipedia_squads.py
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
OUTPUT_PATH = Path("data/external/raw/wikipedia_squads_raw.csv")

# Wikipedia requires a descriptive User-Agent or it may 403/throttle.
HEADERS = {
    "User-Agent": "wc2026-research-script/1.0 (personal ML project; contact: none)"
}

POSITION_NORMALISE = {
    "GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW",
}


def fetch_page_html(url: str) -> str:
    """Download the raw HTML of the Wikipedia squads page."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_age(dob_age_text: str) -> tuple[str, int | None]:
    """
    Wikipedia formats date-of-birth as: "May 17, 2000 (aged 26)"
    Returns (dob_string, age_int).
    """
    match = re.search(r"\(aged (\d+)\)", dob_age_text)
    age = int(match.group(1)) if match else None
    dob = dob_age_text.split(" (")[0].strip()
    return dob, age


def clean_player_name(raw_name: str) -> str:
    """
    Remove footnote markers, captain tags, and extra whitespace.
    e.g. "Ladislav Krejčí (captain)" -> "Ladislav Krejčí"
         "Lionel Messi[note 2]" -> "Lionel Messi"
    """
    name = re.sub(r"\(captain\)", "", raw_name, flags=re.IGNORECASE)
    name = re.sub(r"\[.*?\]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def extract_team_name(heading_text: str) -> str:
    """
    Team headings on this page are usually just the country name,
    sometimes followed by a footnote marker. Strip those out.
    """
    name = re.sub(r"\[.*?\]", "", heading_text)
    return name.strip()


def scrape_squads(html: str) -> pd.DataFrame:
    """
    Parse all team squad tables from the Wikipedia page HTML.

    Strategy:
        The page structure alternates: a team heading (h3, occasionally h2
        for confederation dividers — skipped) followed by prose, followed
        by a <table class="wikitable">. We walk the DOM in order, tracking
        the most recently seen heading as the "current team", and attach
        every wikitable's rows to that team until the next heading appears.

    Returns:
        DataFrame with columns: team, jersey_no, position, player, dob,
        age, caps, goals, club
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output")
    if content is None:
        raise RuntimeError("Could not find main content div — Wikipedia page structure may have changed")

    rows_out = []
    current_team = None

    # Walk all relevant elements in document order
    for element in content.find_all(["h2", "h3", "h4", "table"]):
        if element.name in ("h2", "h3", "h4"):
            heading_span = element.find("span", class_="mw-headline") or element
            heading_text = heading_span.get_text(strip=True)

            # Skip non-team headings (confederation dividers, "See also", etc.)
            skip_keywords = ["contents", "see also", "references", "notes", "external links"]
            if any(kw in heading_text.lower() for kw in skip_keywords):
                current_team = None
                continue

            # Heuristic: team headings are short (1-4 words), no colons
            if len(heading_text.split()) <= 5 and ":" not in heading_text:
                current_team = extract_team_name(heading_text)

        elif element.name == "table" and "wikitable" in (element.get("class") or []):
            if current_team is None:
                continue  # table not under a recognised team heading — skip

            headers = [th.get_text(strip=True) for th in element.find_all("th")]
            if not any("Player" in h for h in headers):
                continue  # not a squad table (could be a different kind of table)

            for tr in element.find_all("tr")[1:]:  # skip header row
                cells = tr.find_all(["td", "th"])
                if len(cells) < 6:
                    continue

                texts = [c.get_text(strip=True) for c in cells]

                # Expected column order: No., Pos., Player, DOB(age), Caps, Goals, Club
                # Some teams omit jersey numbers (empty first cell) — handle gracefully
                if len(texts) == 7:
                    jersey_no, position, player_raw, dob_age, caps, goals, club = texts
                elif len(texts) == 6:
                    # Missing jersey number column
                    jersey_no = ""
                    position, player_raw, dob_age, caps, goals, club = texts
                else:
                    continue

                dob, age = parse_age(dob_age)
                player = clean_player_name(player_raw)
                position = POSITION_NORMALISE.get(position.upper().strip(), position.upper().strip())

                rows_out.append({
                    "team": current_team,
                    "jersey_no": jersey_no,
                    "position": position,
                    "player": player,
                    "dob": dob,
                    "age": age,
                    "caps": pd.to_numeric(caps, errors="coerce"),
                    "goals": pd.to_numeric(goals, errors="coerce"),
                    "club": club,
                })

    df = pd.DataFrame(rows_out)
    return df


def main() -> None:
    print(f"Fetching {WIKI_URL} ...")
    html = fetch_page_html(WIKI_URL)
    time.sleep(1)  # polite delay, though this is a single request

    print("Parsing squad tables...")
    df = scrape_squads(html)

    if df.empty:
        raise RuntimeError(
            "No squad rows extracted. The Wikipedia page structure may have "
            "changed — inspect the HTML manually and adjust scrape_squads()."
        )

    n_teams = df["team"].nunique()
    print(f"Extracted {len(df)} players across {n_teams} teams")

    if n_teams < 40:
        print(
            f"⚠️  WARNING: only found {n_teams} teams (expected 48). "
            "Some teams may not have announced squads yet, or the heading "
            "heuristic missed some — check the output CSV before proceeding."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")
    print(f"\nTeams found: {sorted(df['team'].unique().tolist())}")


if __name__ == "__main__":
    main()