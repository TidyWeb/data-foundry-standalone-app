"""Deliberate faults, for cleaning practice.

`make_messy(clean_table, seed)` takes a finished, clean table and returns a damaged copy plus an
answer key listing every change. The clean generator is never touched: the same seed always gives the
same clean table and, on top of it, the same mess.

Rates are per affected column (each column draws its own rate) and are documented next to each fault.
"""
from datetime import date, datetime

import numpy as np
import pandas as pd

KEY_COLUMNS = ["row", "column", "problem", "original", "messy", "note"]
MISSING_MARKERS = ["", "N/A", "null", "-", "?", "n/a", "NaN"]
DATE_FORMATS = ["%d/%m/%Y", "%m-%d-%Y", "%d %b %Y", "%Y%m%d"]
BOOL_STYLES = [("Yes", "No"), ("Y", "N"), ("1", "0"), ("true", "false"), ("TRUE", "FALSE")]
CELL_BUDGET = 0.14        # at most about this share of all cells is damaged by the cell-level faults


def _py(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def text_of(value):
    """The plain-text form of a clean value, as it is written in the answer key."""
    value = _py(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == datetime.min.time() else str(value)
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _kind(name, series):
    if name == "id" or name.endswith("_id"):
        return "id"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    non_null = series.dropna()
    if len(non_null) and isinstance(non_null.iloc[0], (date, datetime, pd.Timestamp)):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "text"


def make_messy(clean, seed):
    """Return (messy_table, answer_key). Values are left as they were unless a fault changes them."""
    rng = np.random.default_rng([int(seed), 7919])
    columns = list(clean.columns)
    n = len(clean)
    data = {c: [_py(v) for v in clean[c].astype(object).tolist()] for c in columns}
    kinds = {c: _kind(c, clean[c]) for c in columns}
    used = set()
    cells = []                                   # (source row, column, problem, original, messy, note)
    state = {"budget": max(2, int(round(CELL_BUDGET * n * len(columns))))}

    def eligible(*wanted):
        return [c for c in columns if kinds[c] in wanted]

    def choose_columns(pool, low, high):
        if not pool:
            return []
        k = int(rng.integers(low, high + 1))
        return list(rng.choice(pool, size=min(k, len(pool)), replace=False))

    def choose_rows(column, rate):
        free = [i for i in range(n) if (i, column) not in used]
        count = min(max(1, int(round(rate * n))), len(free), state["budget"])
        if count <= 0:
            return []
        return [int(i) for i in rng.choice(free, size=count, replace=False)]

    def hit(i, column, problem, new, note=""):
        cells.append((i, column, problem, text_of(data[column][i]), text_of(new), note))
        data[column][i] = new
        used.add((i, column))
        state["budget"] -= 1

    # ---- cell-level faults ------------------------------------------------------------------
    def missing_values():                         # 3-8% of cells in 2-4 non-ID columns
        for c in choose_columns([c for c in columns if kinds[c] != "id"], 2, 4):
            markers = list(rng.choice(MISSING_MARKERS, size=2, replace=False))
            for i in choose_rows(c, rng.uniform(0.03, 0.08)):
                hit(i, c, "missing_value", str(rng.choice(markers)))

    def numbers_as_text():                        # 4-8% of cells in 1-2 numeric columns
        for c in choose_columns(eligible("number"), 1, 2):
            for i in choose_rows(c, rng.uniform(0.04, 0.08)):
                v = data[c][i]
                if isinstance(v, int):
                    options = [f"{v:,}", f"{v} ", f"£{v:,}"]
                else:
                    options = [f"£{v:,.2f}", f"{v:,.2f}", f"{v:.2f}".replace(".", ","), f"{v} "]
                options = [o for o in options if o != text_of(v)]
                hit(i, c, "number_stored_as_text", str(rng.choice(options)))

    def wrong_dates():                            # 20-40% of rows in 1-2 date columns, plus 1-2% invalid
        for c in choose_columns(eligible("date"), 1, 2):
            for i in choose_rows(c, rng.uniform(0.01, 0.02)):
                stamp = pd.Timestamp(data[c][i])
                bad = [f"31/02/{stamp.year}", "TBC", "00/00/0000"]
                hit(i, c, "invalid_date", str(rng.choice(bad)))
            for i in choose_rows(c, rng.uniform(0.20, 0.40)):
                stamp = pd.Timestamp(data[c][i])
                hit(i, c, "date_format", stamp.strftime(str(rng.choice(DATE_FORMATS))))

    def casing_spaces_typos():                    # casing 5-10%, stray spaces 3-6%, typos 1-3% in 1-2 text columns
        for c in choose_columns(eligible("text"), 1, 2):
            for i in choose_rows(c, rng.uniform(0.05, 0.10)):
                v = str(data[c][i])
                options = [o for o in (v.upper(), v.lower(), v.title(), v.swapcase()) if o != v]
                if options:
                    hit(i, c, "inconsistent_case", str(rng.choice(options)))
            for i in choose_rows(c, rng.uniform(0.03, 0.06)):
                v = str(data[c][i])
                hit(i, c, "stray_spaces", str(rng.choice([f" {v}", f"{v} ", f" {v} "])))
            for i in choose_rows(c, rng.uniform(0.01, 0.03)):
                v = str(data[c][i])
                if len(v) < 4:
                    continue
                p = int(rng.integers(0, len(v) - 1))
                if rng.random() < 0.5 and v[p] != v[p + 1]:
                    typo = v[:p] + v[p + 1] + v[p] + v[p + 2:]
                else:
                    typo = v[:p] + v[p + 1:]
                if typo != v:
                    hit(i, c, "typo", typo)

    def mixed_booleans():                         # 15-30% of rows in one Boolean column
        for c in choose_columns(eligible("bool"), 1, 1):
            for i in choose_rows(c, rng.uniform(0.15, 0.30)):
                pair = BOOL_STYLES[int(rng.integers(len(BOOL_STYLES)))]
                hit(i, c, "mixed_boolean", pair[0] if data[c][i] else pair[1])

    def outliers():                               # 0.5-1.5% of cells in 1-2 numeric columns
        for c in choose_columns(eligible("number"), 1, 2):
            for i in choose_rows(c, rng.uniform(0.005, 0.015)):
                v = data[c][i]
                op = str(rng.choice(["x10", "x100", "negative", "sentinel"]))
                if op == "x10":
                    new = v * 10
                elif op == "x100":
                    new = v * 100
                elif op == "negative":
                    new = -abs(v) if v != 0 else -1
                else:
                    new = int(rng.choice([999999, -1]))
                if isinstance(new, float):
                    new = round(new, 2)
                if new == v:
                    new = 999999
                hit(i, c, "outlier", new)

    faults = [missing_values, numbers_as_text, wrong_dates, casing_spaces_typos, mixed_booleans, outliers]
    for index in rng.permutation(len(faults)):
        faults[int(index)]()

    # ---- headers ----------------------------------------------------------------------------
    styles = [lambda s: s, lambda s: s, lambda s: s,
              lambda s: s.replace("_", " ").title(),
              lambda s: s.upper(),
              lambda s: (lambda p: p[0] + "".join(w.title() for w in p[1:]))(s.split("_")),
              lambda s: s + " "]
    headers = {c: styles[int(rng.integers(len(styles)))](c) for c in columns}
    if all(headers[c] == c for c in columns):
        headers[columns[0]] = columns[0].upper()
    if len(set(headers.values())) < len(columns):
        headers = {c: c for c in columns}
        headers[columns[0]] = columns[0].upper()
    final_headers = [headers[c] for c in columns]

    # ---- structural faults: duplicates, conflicting IDs, stray rows -------------------------
    order = [("src", i) for i in range(n)]

    def insert(item):
        order.insert(int(rng.integers(1, len(order) + 1)), item)

    for i in rng.choice(n, size=min(n, max(1, int(round(rng.uniform(0.01, 0.03) * n)))), replace=False):
        insert(("dup", int(i)))
    id_cols, number_cols = eligible("id"), [c for c in eligible("number")]
    if id_cols and number_cols:
        column = number_cols[int(rng.integers(len(number_cols)))]
        for i in rng.choice(n, size=min(n, max(1, int(round(rng.uniform(0.005, 0.01) * n)))), replace=False):
            v = _py(clean[column].iloc[int(i)])
            new = int(round(v * 1.1)) + 1 if isinstance(v, int) else round(v * 1.1 + 0.01, 2)
            insert(("conflict", int(i), column, new))
    insert(("header",))
    insert(("blank",))

    rows, key = [], []
    source_row = {}
    for position, item in enumerate(order, start=1):
        if item[0] == "src":
            source_row[item[1]] = position
    for position, item in enumerate(order, start=1):
        kind = item[0]
        if kind in ("src", "dup"):
            rows.append([data[c][item[1]] for c in columns])
            if kind == "dup":
                key.append((position, "", "duplicate_row", "", "", f"exact copy of row {source_row[item[1]]}"))
        elif kind == "conflict":
            row = [data[c][item[1]] for c in columns]
            row[columns.index(item[2])] = item[3]
            rows.append(row)
            key.append((position, item[2], "conflicting_duplicate_id", text_of(clean[item[2]].iloc[item[1]]),
                        text_of(item[3]), f"same {id_cols[0]} as row {source_row[item[1]]} but different {item[2]}"))
        elif kind == "header":
            rows.append(list(final_headers))
            key.append((position, "", "repeated_header_row", "", "", "the column names appear again as a data row"))
        else:
            rows.append([None] * len(columns))
            key.append((position, "", "blank_row", "", "", "an empty row"))
    for i, column, problem, original, messy, note in cells:
        key.append((source_row[i], column, problem, original, messy, note))
    for c in columns:
        if headers[c] != c:
            key.append((0, c, "messy_header", c, headers[c], "row 0 is the header line"))

    key_table = pd.DataFrame(key, columns=KEY_COLUMNS).sort_values(["row", "column"], kind="stable")
    messy = pd.DataFrame(rows, columns=final_headers, dtype=object)
    return messy, key_table.reset_index(drop=True)


def summarise(key):
    """Counts of each problem type, for the page and the README."""
    return {str(k): int(v) for k, v in key["problem"].value_counts().items()}
