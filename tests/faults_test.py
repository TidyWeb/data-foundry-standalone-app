"""Checks the messy-data layer: the clean table is untouched, the mess is reproducible, and the
answer key is complete (undoing every change in the key gives back exactly the clean table).

Run from the project folder:   python tests/faults_test.py
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from foundry.core import generate  # noqa: E402
from foundry.faults import make_messy, summarise, text_of  # noqa: E402
from foundry.recipes import RECIPES  # noqa: E402

problems = []


def expect(ok, message):
    if not ok:
        problems.append(message)
        print("FAIL", message)


def undo(messy, key, clean_columns):
    """Rebuild the clean table (as text) from the messy table and the answer key."""
    extra_rows = set(key.loc[key["problem"].isin(
        ["duplicate_row", "conflicting_duplicate_id", "repeated_header_row", "blank_row"]), "row"])
    headers = dict(zip(key.loc[key["problem"] == "messy_header", "messy"],
                       key.loc[key["problem"] == "messy_header", "original"]))
    table = messy.copy()
    table.columns = [headers.get(c, c) for c in table.columns]
    table.index = range(1, len(table) + 1)
    table = table.drop(index=sorted(extra_rows)).reset_index(drop=True)
    kept = [r for r in range(1, len(messy) + 1) if r not in extra_rows]
    position = {r: i for i, r in enumerate(kept)}
    text = table.astype(object).map(text_of)
    cell_faults = key[~key["problem"].isin(
        ["duplicate_row", "conflicting_duplicate_id", "repeated_header_row", "blank_row", "messy_header"])]
    for row in cell_faults.itertuples():
        text.at[position[row.row], row.column] = row.original
    return text[clean_columns]


snapshot = date(2026, 1, 1)
for key_name in sorted(RECIPES):
    for rows, cols in ((200, 99), (12, 6), (1, 3)):
        result = generate(rows, cols, seed=7, theme=key_name, snapshot=snapshot)
        before = result.df.copy()
        messy, key = make_messy(result.df, result.seed)
        tag = f"{key_name} {rows}x{cols}"
        expect(result.df.equals(before), f"{tag}: the clean table was changed")
        messy2, key2 = make_messy(result.df, result.seed)
        expect(messy.astype(str).equals(messy2.astype(str)) and key.equals(key2), f"{tag}: not reproducible")
        other, _ = make_messy(result.df, result.seed + 1)
        if rows >= 12:
            expect(not messy.astype(str).equals(other.astype(str)), f"{tag}: different seeds gave the same mess")
        expect(len(key) > 0, f"{tag}: no problems planted")
        expect(len(messy) >= rows + 2, f"{tag}: stray rows missing")
        expected = before.astype(object).map(text_of)
        expect(undo(messy, key, list(before.columns)).equals(expected), f"{tag}: the answer key does not undo the mess")
        if rows == 200:
            counts = summarise(key)
            cells = rows * len(before.columns)
            damaged = sum(v for k, v in counts.items()
                          if k not in ("duplicate_row", "conflicting_duplicate_id", "repeated_header_row",
                                       "blank_row", "messy_header"))
            expect(damaged <= 0.16 * cells, f"{tag}: too much damage ({damaged} of {cells} cells)")
            expect(damaged >= 0.02 * cells, f"{tag}: too little damage ({damaged} of {cells} cells)")

print(f"subjects: {len(RECIPES)} | problems: {len(problems)}")
sys.exit(1 if problems else 0)
