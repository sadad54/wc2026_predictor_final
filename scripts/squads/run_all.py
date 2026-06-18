"""
scripts/squads/01_scrape_wikipedia_squads.py  (v2 — wikitext-based)

Stage 1 of the squad-data pipeline: extract the official 2026 FIFA World Cup
squad lists from Wikipedia's raw WIKITEXT source (not the rendered HTML).

Why wikitext instead of HTML scraping (v1 of this script used HTML and
broke in practice — rendered page structure/heading nesting is harder to
detect reliably than the source markup):
    - Headings are unambiguous: "=== Czech Republic ===" is a level-3
      heading, full stop. No DOM class-name guessing.
    - Tables use regular {| ... |} / |- / | syntax — straightforward to
      split into rows and cells with simple parsing rules.
    - We fetch wikitext via the official MediaWiki Action API
      (action=parse&prop=wikitext), which is the documented, bot-friendly
      way to read page source — not scraping the rendered site.

Output:
    data/external/raw/wikipedia_squads_raw.csv
    Columns: team, jersey_no, position, player, dob, age, caps, goals, club

Run:
    python scripts/squads/01_scrape_wikipedia_squads.py
"""

import re
import time
from pathlib import Path

import mwparserfromhell
import pandas as pd
import requests

API_URL = "https://en.wikipedia.org/w/api.php"
PAGE_TITLE = "2026_FIFA_World_Cup_squads"
OUTPUT_PATH = Path("data/external/raw/wc2026_squads.csv")

HEADERS = {
    "User-Agent": "wc2026-research-script/1.0 (personal ML project; non-commercial)"
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

POSITION_NORMALISE = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}


# ─────────────────────────────────────────────────────────────────────────────
# Fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_wikitext(page_title: str) -> str:
    """
    Fetch raw wikitext source for a page via the MediaWiki Action API.

    This is the documented API endpoint for reading page content —
    distinct from scraping the rendered HTML page, and not subject to
    the same bot-detection rendering pipeline.
    """
    resp = requests.get(
        API_URL,
        params={
            "action": "parse",
            "page": page_title,
            "prop": "wikitext",
            "format": "json",
            "formatversion": "2",
        },
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"MediaWiki API error: {data['error']}")

    return data["parse"]["wikitext"]


# ─────────────────────────────────────────────────────────────────────────────
# Wikitext table parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_wikitable_rows(table_wikicode_str: str) -> list[list[str]]:
    """
    Parse a {| ... |} wikitable string into a list of rows of raw cell text
    (still containing wikicode markup — resolved separately).

    Handles the standard MediaWiki table syntax:
        {| attributes
        ! header | header | header
        |-
        | cell | cell | cell
        |-
        |}
    """
    body = table_wikicode_str.strip()
    body = re.sub(r"^\{\|[^\n]*\n", "", body)   # strip opening {| line
    body = re.sub(r"\n\|\}\s*$", "", body)       # strip closing |}

    row_blocks = re.split(r"\n\|-", body)

    rows: list[list[str]] = []
    for block in row_blocks:
        block = block.strip()
        if not block:
            continue

        cells: list[str] = []
        current_cell: str | None = None

        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("!"):
                continue  # header row content — column order is hardcoded below
            if line.startswith("|"):
                if current_cell is not None:
                    cells.append(current_cell)
                current_cell = line[1:].strip()
            elif current_cell is not None:
                current_cell += " " + line

        if current_cell is not None:
            cells.append(current_cell)
        if cells:
            rows.append(cells)

    return rows


def resolve_wikicode_cell(cell_text: str) -> str:
    """
    Convert one cell's raw wikicode into clean plain text.

    Handles the templates actually used in these tables:
        {{birth date and age|YYYY|M|D}}  -> "Month D, YYYY"
        {{fbc|Club Name|COUNTRY}}        -> "Club Name"
        [[Player Name]]                  -> "Player Name" (wikilinks resolved
                                             automatically by strip_code)
    Strips footnote markers and "(captain)" tags.
    """
    code = mwparserfromhell.parse(cell_text)

    for template in code.filter_templates():
        name = template.name.strip().lower()

        if name == "birth date and age":
            params = [p.value.strip_code().strip() for p in template.params
                       if not p.showkey]  # positional params only
            try:
                y, m, d = int(params[0]), int(params[1]), int(params[2])
                code.replace(template, f"{MONTH_NAMES[m - 1]} {d}, {y}")
            except (ValueError, IndexError):
                code.replace(template, "")

        elif name in ("fbc", "flagicon"):
            # Club/flag templates: first positional param is the display name
            positional = [p for p in template.params if not p.showkey]
            if positional:
                code.replace(template, positional[0].value.strip_code().strip())
            else:
                code.replace(template, "")

        else:
            # Unknown template — drop it rather than leaving raw markup
            code.replace(template, "")

    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"\(\s*captain\s*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[.*?\]", "", text)  # stray footnote refs like [1]
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_age_from_dob_string(dob_text: str) -> tuple[str, int | None]:
    """
    The resolved {{birth date and age}} template gives "Month D, YYYY".
    We don't get age directly this way (the template renders age only in
    HTML, not wikitext) — so age is computed separately using the
    tournament start date (June 11, 2026), matching how Wikipedia itself
    defines "age" for this page (age as of the first day of the tournament).
    """
    from datetime import date

    try:
        dob = pd.to_datetime(dob_text)
    except (ValueError, TypeError):
        return dob_text, None

    tournament_start = date(2026, 6, 11)
    age = tournament_start.year - dob.year - (
        (tournament_start.month, tournament_start.day) < (dob.month, dob.day)
    )
    return dob_text, age


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_squads(wikitext: str) -> pd.DataFrame:
    """
    Extract squads from MediaWiki template-based format.
    
    The 2026 FIFA World Cup squads page uses {{nat fs g player|...}} templates
    to store player data, not wikitables. Each team section has:
        ===Team Name===
        {{nat fs g start}}
        {{nat fs g player|no=...|pos=...|name=...|age={{birth date and age2|YYYY|M|D|YYYY|M|D}}|caps=...|goals=...|club=...}}
        ...
        {{nat fs end}}
    
    This function extracts the player data from these templates and computes age.
    """
    from datetime import date
    
    wikicode = mwparserfromhell.parse(wikitext)
    headings = wikicode.filter_headings()

    skip_keywords = ["see also", "references", "notes", "external links", "further reading"]

    rows_out = []
    current_team = None
    
    for h in headings:
        if h.level != 3:
            continue

        team_name = h.title.strip_code().strip()
        team_name = re.sub(r"\[.*?\]", "", team_name).strip()

        if any(kw in team_name.lower() for kw in skip_keywords):
            continue

        current_team = team_name

        # Find the position of this heading in the wikitext
        heading_str = str(h)
        heading_pos = wikitext.find(heading_str)
        
        # Find the next heading position (to limit our search range)
        next_heading_pos = len(wikitext)
        for next_h in headings:
            next_h_str = str(next_h)
            next_h_pos = wikitext.find(next_h_str, heading_pos + len(heading_str))
            if next_h_pos > heading_pos:
                next_heading_pos = min(next_heading_pos, next_h_pos)
        
        # Extract the section text
        section_text = wikitext[heading_pos:next_heading_pos]
        
        # Parse nat fs player templates in this section
        section_code = mwparserfromhell.parse(section_text)
        templates = section_code.filter_templates(matches=lambda t: t.name.strip().lower() == "nat fs g player")
        
        for template in templates:
            try:
                # Extract parameters from the template
                jersey_no = ""
                position = ""
                player = ""
                age = None
                dob = ""
                caps_str = ""
                goals_str = ""
                club = ""
                
                for param in template.params:
                    key = param.name.strip().lower()
                    value = param.value
                    
                    if key == 'no':
                        jersey_no = value.strip_code().strip()
                    elif key == 'pos':
                        position = value.strip_code().strip().upper()
                        position = POSITION_NORMALISE.get(position, position)
                    elif key == 'name':
                        player = value.strip_code().strip()
                        # Clean up wikilinks
                        player = re.sub(r'\[\[([^\]|]+)(\|[^\]]+)?\]\]', r'\1', player)
                    elif key == 'age':
                        # The age parameter contains a template like {{birth date and age2|2026|6|11|2000|5|17}}
                        # Extract parameters from this nested template
                        age_template_str = str(value)
                        match = re.search(r'birth\s+date\s+and\s+age2\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)', age_template_str)
                        if match:
                            tourn_year, tourn_month, tourn_day, birth_year, birth_month, birth_day = [int(x) for x in match.groups()]
                            dob = f"{MONTH_NAMES[birth_month - 1]} {birth_day}, {birth_year}"
                            try:
                                tourn_date = date(tourn_year, tourn_month, tourn_day)
                                birth_date = date(birth_year, birth_month, birth_day)
                                age = tourn_date.year - birth_date.year - (
                                    (tourn_date.month, tourn_date.day) < (birth_date.month, birth_date.day)
                                )
                            except ValueError:
                                age = None
                    elif key == 'caps':
                        caps_str = value.strip_code().strip()
                    elif key == 'goals':
                        goals_str = value.strip_code().strip()
                    elif key == 'club':
                        club = value.strip_code().strip()
                        # Clean up wikilinks
                        club = re.sub(r'\[\[([^\]|]+)(\|[^\]]+)?\]\]', r'\1', club)
                
                # Clean up player and club names (remove extra whitespace)
                player = re.sub(r'\s+', ' ', player).strip()
                club = re.sub(r'\s+', ' ', club).strip()
                
                if not player:
                    continue
                
                rows_out.append({
                    "team": current_team,
                    "jersey_no": jersey_no,
                    "position": position,
                    "player": player,
                    "dob": dob,
                    "age": age,
                    "caps": pd.to_numeric(re.sub(r"\D", "", str(caps_str)) or None, errors="coerce"),
                    "goals": pd.to_numeric(re.sub(r"\D", "", str(goals_str)) or None, errors="coerce"),
                    "club": club,
                })
            except Exception as e:
                # Skip malformed templates
                continue

    return pd.DataFrame(rows_out)


def main() -> None:
    print(f"Fetching wikitext for '{PAGE_TITLE}' via MediaWiki API...")
    wikitext = fetch_wikitext(PAGE_TITLE)
    time.sleep(0.5)

    print(f"  Fetched {len(wikitext):,} characters of wikitext")
    print("Parsing squad tables...")

    df = extract_squads(wikitext)

    if df.empty:
        raise RuntimeError(
            "No squad rows extracted. Inspect the wikitext manually — "
            "save it to a file and check the heading/table structure:\n"
            "    import requests\n"
            "    r = requests.get('https://en.wikipedia.org/w/api.php', "
            "params={'action':'parse','page':'2026_FIFA_World_Cup_squads',"
            "'prop':'wikitext','format':'json','formatversion':'2'})\n"
            "    open('debug.wikitext','w',encoding='utf-8').write(r.json()['parse']['wikitext'])"
        )

    n_teams = df["team"].nunique()
    print(f"Extracted {len(df)} players across {n_teams} teams")

    if n_teams < 40:
        print(
            f"⚠️  WARNING: only found {n_teams} teams (expected 48). "
            f"Teams found: {sorted(df['team'].unique().tolist())}\n"
            "Check for non-team headings being picked up, or team sections "
            "using a different heading level."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()