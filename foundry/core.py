"""foundry/core.py - shared plumbing for every theme recipe.

A *recipe* (one file per theme in foundry/recipes/) owns the business logic:
it simulates a coherent situation and returns one table.  This file owns only
the plumbing: seeds, small helper functions, choosing which columns to show,
and tidying dates.

Reading order for a student:  Recipe -> generate() -> the helpers.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Callable
import secrets

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class Check:
    """One validation result.  kind is 'exact' (must always hold) or
    'relationship' (a statistical tendency that should hold in most seeds)."""
    name: str
    ok: bool
    detail: str = ""
    kind: str = "exact"


@dataclass
class Recipe:
    """Everything the app needs to know about one theme."""
    key: str                      # e.g. "real_estate_listings"
    title: str                    # e.g. "Property Listings"
    row_meaning: str              # plain English: "One row is ..."
    simulate: Callable            # simulate(rng, n_rows, snapshot) -> DataFrame (>= n_rows rows)
    columns: list                 # every column a student can see, in display order
    core: list                    # columns that are always shown
    priority: list                # non-core columns, in the order they get added
    docs: dict                    # column -> plain-English description (with unit)
    validate: Callable            # validate(df, snapshot) -> list[Check]  (df = full table)
    date_cols: list = field(default_factory=list)      # shown as YYYY-MM-DD
    datetime_cols: list = field(default_factory=list)  # shown with a time
    targets: dict = field(default_factory=dict)        # target column -> columns that leak it


@dataclass
class Result:
    """What generate() hands back to the web app."""
    df: pd.DataFrame              # the columns the student asked for
    full: pd.DataFrame            # every column the simulation produced (incl. hidden ones)
    recipe: Recipe
    seed: int
    snapshot: pd.Timestamp
    column_note: str | None
    columns_shown: list


# --------------------------------------------------------------------------
# Small helpers used by the recipes
# --------------------------------------------------------------------------

def make_rng(seed):
    """The one source of randomness.  Same seed -> same dataset."""
    return np.random.default_rng(seed)


def check(name, ok, detail="", kind="exact"):
    return Check(name, bool(ok), str(detail), kind)


def lognormal(rng, median, sigma, size=None):
    """Right-skewed positive numbers whose middle value is `median`."""
    return rng.lognormal(np.log(median), sigma, size)


def pick(rng, options, weights, size=None):
    """Choose from `options` with the given (not necessarily normalised) weights."""
    w = np.asarray(weights, dtype=float)
    return rng.choice(np.asarray(options, dtype=object), size=size, p=w / w.sum())


def bernoulli(rng, p):
    """True with probability p.  p may be a single number or an array."""
    p = np.asarray(p, dtype=float)
    return rng.random(p.shape) < p


def logistic(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def random_dates(rng, start, end, size):
    """`size` random dates between start and end (inclusive), as a Series."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    days = (end - start).days
    offsets = rng.integers(0, days + 1, size)
    return pd.Series(start + pd.to_timedelta(offsets, unit="D"))


def ordinal(score, cuts, labels):
    """Turn a numeric score into ordered labels.  cuts are the boundaries."""
    return np.asarray(labels, dtype=object)[np.digitize(score, cuts)]


def add_days(dates, days):
    """dates (Series of timestamps) + days (array of numbers)."""
    return pd.Series(pd.to_datetime(dates)) + pd.to_timedelta(np.asarray(days), unit="D")


# --------------------------------------------------------------------------
# Choosing columns
# --------------------------------------------------------------------------

def choose_columns(recipe, column_count):
    """Core columns are always shown; extras are added in priority order.
    Returns (columns_in_display_order, note_or_None)."""
    n_core = len(recipe.core)
    n_max = n_core + len(recipe.priority)
    wanted = int(column_count)
    actual = min(max(wanted, n_core), n_max)

    chosen = set(recipe.core) | set(recipe.priority[: actual - n_core])
    columns = [c for c in recipe.columns if c in chosen]

    note = None
    if wanted < n_core:
        note = (f"{recipe.title} needs at least {n_core} columns to make sense - "
                f"generated {actual} instead of {wanted}.")
    elif wanted > n_max:
        note = (f"{recipe.title} can supply at most {n_max} columns - "
                f"generated {actual} instead of {wanted}.")
    return columns, note


def tidy_dates(df, recipe):
    """Show plain dates as YYYY-MM-DD and date-times to the second."""
    df = df.copy()
    for c in recipe.date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.date
    for c in recipe.datetime_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c]).dt.floor("s")
    return df


# --------------------------------------------------------------------------
# The one entry point the web app (and the validator) calls
# --------------------------------------------------------------------------

def new_seed():
    return secrets.randbelow(1_000_000_000)


def simulate_full(recipe, row_count, seed, snapshot=None):
    """Run one recipe and return (full_table_with_exactly_row_count_rows, snapshot)."""
    snapshot = pd.Timestamp(snapshot or date.today()).normalize()
    rng = make_rng(seed)
    rng.integers(1)                      # keep the stream aligned with generate()
    full = recipe.simulate(rng, int(row_count), snapshot)
    if len(full) < row_count:
        raise RuntimeError(f"{recipe.key} produced {len(full)} rows, needed {row_count}")
    return full.iloc[: int(row_count)].reset_index(drop=True), snapshot


def generate(row_count, column_count, seed=None, theme=None, snapshot=None):
    from .recipes import RECIPES

    seed = new_seed() if seed in (None, "") else int(seed)
    keys = sorted(RECIPES)

    # The first random draw decides the subject, so the seed also fixes the theme.
    pick_rng = make_rng(seed)
    chosen = keys[int(pick_rng.integers(len(keys)))]
    key = theme if theme else chosen
    recipe = RECIPES[key]

    full, snapshot = simulate_full(recipe, row_count, seed, snapshot)
    columns, note = choose_columns(recipe, column_count)
    shown = tidy_dates(full, recipe)[columns]
    return Result(df=shown, full=full, recipe=recipe, seed=seed,
                  snapshot=snapshot, column_note=note, columns_shown=columns)
