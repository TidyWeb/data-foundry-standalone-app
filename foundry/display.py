"""Presentation helpers shared by the Flask app and the static web build.

Nothing here generates data; it only decides how subjects are grouped, coloured,
named and formatted for people to read.
"""
import re
from datetime import date

import numpy as np
import pandas as pd

from .core import generate
from .recipes import RECIPES

PREVIEW_DATE = date(2026, 1, 1)       # fixed, so the cards on the home page never change

FAMILY_NAMES = {"automotive": "Automotive", "real": "Real Estate", "retail": "Retail",
                "manufacturing": "Manufacturing", "finance": "Finance", "hr": "HR",
                "saas": "SaaS", "logistics": "Logistics"}
FAMILY_HUES = {"Automotive": "#e4572e", "Real Estate": "#2a9d8f", "Retail": "#ef476f",
               "Manufacturing": "#6d597a", "Finance": "#1d6fa5", "HR": "#e09f3e",
               "SaaS": "#5f4bb6", "Logistics": "#2d6a4f"}


def family_of(key):
    return FAMILY_NAMES[key.split("_")[0]]


def fmt(value):
    """Format one cell for the web page (the downloads use the raw values)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        value = int(value)
        return f"{value:,}" if abs(value) >= 10000 else str(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        text = f"{value:,.3f}" if abs(value) >= 10000 else f"{value:.3f}"
        text = text.rstrip("0").rstrip(".")
        return "0" if text in ("-0", "") else text
    return str(value)


def slug(title):
    text = title.lower().replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def build_shelves():
    """The home page: every subject grouped by family, with a three-row preview."""
    families = {}
    for key, recipe in sorted(RECIPES.items(), key=lambda kv: kv[1].title):
        result = generate(3, 3, seed=1, theme=key, snapshot=PREVIEW_DATE)
        cols = list(result.df.columns[:3])
        families.setdefault(family_of(key), []).append({
            "key": key,
            "title": recipe.title,
            "row_meaning": recipe.row_meaning,
            "min_cols": len(recipe.core),
            "max_cols": len(recipe.core) + len(recipe.priority),
            "preview_cols": cols,
            "preview_rows": [[fmt(v) for v in row] for row in result.df[cols].itertuples(index=False)],
        })
    return [{"name": name, "hue": FAMILY_HUES[name], "cards": families[name]} for name in sorted(families)]
