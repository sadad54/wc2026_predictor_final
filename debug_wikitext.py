"""
Debug script to inspect the wikitext structure for 2026 FIFA World Cup squads.
"""

import requests
import re
import mwparserfromhell

API_URL = "https://en.wikipedia.org/w/api.php"
PAGE_TITLE = "2026_FIFA_World_Cup_squads"
HEADERS = {
    "User-Agent": "wc2026-research-script/1.0 (personal ML project; non-commercial)"
}

print(f"Fetching wikitext for '{PAGE_TITLE}'...")
resp = requests.get(
    API_URL,
    params={
        "action": "parse",
        "page": PAGE_TITLE,
        "prop": "wikitext",
        "format": "json",
        "formatversion": "2",
    },
    headers=HEADERS,
    timeout=30,
)
resp.raise_for_status()
wikitext = resp.json()["parse"]["wikitext"]

print(f"Fetched {len(wikitext):,} characters")

# Save to file
with open("debug.wikitext", "w", encoding="utf-8") as f:
    f.write(wikitext)
print("Saved to debug.wikitext")

# Parse and inspect structure
wikicode = mwparserfromhell.parse(wikitext)
headings = list(wikicode.filter_headings())
tables = list(wikicode.filter_tags(matches=lambda node: node.tag == "table"))

print(f"\n📊 Wikitext structure:")
print(f"  Total headings: {len(headings)}")
print(f"  Total tables: {len(tables)}")

# Show heading levels and titles
print(f"\n🏆 Headings by level:")
level_counts = {}
for h in headings:
    level = h.level
    level_counts[level] = level_counts.get(level, 0) + 1

for level in sorted(level_counts.keys()):
    print(f"  Level {level}: {level_counts[level]} headings")

# Show first 20 headings with their levels
print(f"\n📝 First 20 headings:")
for i, h in enumerate(headings[:20]):
    title = h.title.strip_code().strip()[:80]
    print(f"  Level {h.level}: {title}")

# Check level-3 headings specifically
level_3_headings = [h for h in headings if h.level == 3]
print(f"\n✅ Level-3 headings (team sections): {len(level_3_headings)}")
if level_3_headings:
    print("  First 10:")
    for h in level_3_headings[:10]:
        title = h.title.strip_code().strip()
        print(f"    - {title}")

# Check for tables
print(f"\n📋 Table analysis:")
print(f"  Total tables found: {len(tables)}")
if tables:
    print(f"  First table (first 500 chars):")
    first_table = str(tables[0])[:500]
    print(f"    {first_table}...")

# Try to find level-2 headings as well
level_2_headings = [h for h in headings if h.level == 2]
print(f"\n🔍 Level-2 headings: {len(level_2_headings)}")
if level_2_headings:
    for h in level_2_headings[:10]:
        title = h.title.strip_code().strip()[:80]
        print(f"    - {title}")
