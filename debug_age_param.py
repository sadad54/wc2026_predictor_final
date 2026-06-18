"""
Debug script to inspect the age parameter format
"""

import mwparserfromhell
import re

# Read a section of the wikitext
with open("debug.wikitext", "r", encoding="utf-8") as f:
    wikitext = f.read()

# Find the Czech Republic section (first team)
start = wikitext.find("===Czech Republic===")
end = wikitext.find("===Mexico===")
section = wikitext[start:end]

# Parse templates
code = mwparserfromhell.parse(section)
templates = list(code.filter_templates(matches=lambda t: t.name.strip().lower() == "nat fs g player"))

print(f"Found {len(templates)} player templates\n")

# Show first 3 players' age parameters
for i, template in enumerate(templates[:3]):
    print(f"Player {i+1}:")
    for param in template.params:
        key = param.name.strip().lower()
        if key in ["name", "age", "no"]:
            value = param.value
            print(f"  {key}: {repr(str(value))}")
    print()
