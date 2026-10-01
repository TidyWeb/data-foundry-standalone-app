"""Data Foundry - the web app.

Every page is stateless: the address (URL) contains everything needed to rebuild a
table - subject, rows, columns, seed and date - so many visitors can use the site at
once, pages can be bookmarked, and downloads always match what is on screen.
"""
import io
import random
from datetime import date, timedelta

import pandas as pd
from flask import Flask, Response, abort, redirect, render_template, request, url_for

from foundry import generate
from foundry.core import new_seed
from foundry.faults import make_messy, summarise
from foundry.display import FAMILY_HUES, build_shelves, family_of, fmt, slug
from foundry.recipes import RECIPES

app = Flask(__name__)

MAX_ROWS = 200
MAX_COLS = max(len(r.columns) for r in RECIPES.values())
DEFAULT_ROWS = 10
DEFAULT_COLS = 6
MAX_SEED = 999_999_999


def clamp_int(text, default, low, high):
    try:
        value = int(text)
    except (TypeError, ValueError):
        value = default
    if value is None:
        return None
    return max(low, min(value, high))


def read_snapshot(text):
    try:
        d = date.fromisoformat(text)
    except (TypeError, ValueError):
        return None
    today = date.today()
    return d if today - timedelta(days=5 * 365) <= d <= today else None


def make_filename(recipe, seed, extension):
    return f"{slug(recipe.title)}_seed{seed}.{extension}"


SHELVES = build_shelves()


def read_table_request():
    """Pull the subject and table settings out of the address."""
    theme = request.args.get("theme")
    if theme not in RECIPES:
        return None
    return {
        "theme": theme,
        "rows": clamp_int(request.args.get("rows"), DEFAULT_ROWS, 1, MAX_ROWS),
        "cols": clamp_int(request.args.get("cols"), DEFAULT_COLS, 1, MAX_COLS),
        "seed": clamp_int(request.args.get("seed"), None, 0, MAX_SEED),
        "snapshot": read_snapshot(request.args.get("d")),
        "messy": 1 if request.args.get("messy") == "1" else 0,
    }


def table_args(p, seed, snapshot):
    args = dict(theme=p["theme"], rows=p["rows"], cols=p["cols"], seed=seed, d=snapshot.isoformat())
    if p["messy"]:
        args["messy"] = 1
    return args


def build_table(p, snapshot):
    """The table to show or download, and (when messy) its answer key."""
    result = generate(p["rows"], p["cols"], seed=p["seed"], theme=p["theme"], snapshot=snapshot)
    if p["messy"]:
        table, key = make_messy(result.df, result.seed)
        return result, table, key
    return result, result.df, None


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

@app.route("/")
def home():
    if "theme" not in request.args:
        return render_template("index.html", shelves=SHELVES, max_rows=MAX_ROWS, table=None)

    p = read_table_request()
    if p is None:
        return redirect(url_for("home"))

    # No seed or date yet: pick them now and redirect, so the address fully describes
    # the table (reloading gives the same table; "New random table" gives a new one).
    if p["seed"] is None or p["snapshot"] is None:
        seed = new_seed() if p["seed"] is None else p["seed"]
        snapshot = p["snapshot"] or date.today()
        return redirect(url_for("home", **table_args(p, seed, snapshot)))

    result, df, key = build_table(p, p["snapshot"])
    clean, recipe = result.df, result.recipe
    numeric = [bool(pd.api.types.is_numeric_dtype(clean[c]) and not pd.api.types.is_bool_dtype(clean[c]))
               for c in clean.columns]
    args = table_args(p, result.seed, result.snapshot.date())
    tag = "_messy" if key is not None else ""
    table = {
        "recipe": recipe,
        "family": family_of(recipe.key),
        "hue": FAMILY_HUES[family_of(recipe.key)],
        "note": result.column_note,
        "cols": [str(c) for c in df.columns],
        "numeric": numeric,
        "rows": [["" if v is None else (v if isinstance(v, str) else fmt(v)) for v in row]
                 for row in df.itertuples(index=False)],
        "docs": [(c, recipe.docs.get(c, "")) for c in clean.columns],
        "seed": result.seed,
        "n_rows": len(clean),
        "n_shown": len(df),
        "n_cols": len(clean.columns),
        "messy": key is not None,
        "problems": 0 if key is None else len(key),
        "problem_counts": {} if key is None else summarise(key),
        "key_url": url_for("download", fmt="key", **dict(args, messy=1)),
        "clean_url": url_for("home", **{k: v for k, v in args.items() if k != "messy"}),
        "messy_url": url_for("home", **dict(args, messy=1)),
        "min_cols": len(recipe.core),
        "max_cols": len(recipe.core) + len(recipe.priority),
        "csv_url": url_for("download", fmt="csv", **args),
        "xlsx_url": url_for("download", fmt="xlsx", **args),
        "new_url": url_for("home", theme=p["theme"], rows=p["rows"], cols=p["cols"], **({"messy": 1} if p["messy"] else {})),
        "date": args["d"],
        "csv_name": make_filename(recipe, result.seed, "csv").replace(".csv", tag + ".csv"),
    }
    return render_template("index.html", shelves=SHELVES, max_rows=MAX_ROWS, max_cols=MAX_COLS,
                           table=table)


@app.route("/surprise")
def surprise():
    return redirect(url_for("home", theme=random.choice(sorted(RECIPES))))


@app.route("/download/<fmt>")
def download(fmt):
    if fmt not in ("csv", "xlsx", "key"):
        abort(404)
    p = read_table_request()
    if p is None or p["seed"] is None:
        abort(400)
    if fmt == "key":
        p["messy"] = 1
    result, df, key = build_table(p, p["snapshot"] or date.today())
    tag = "_messy" if p["messy"] else ""
    if fmt == "key":
        filename = make_filename(result.recipe, result.seed, "csv").replace(".csv", tag + "_answer_key.csv")
        headers = {"Content-Disposition": f"attachment; filename={filename}", "Cache-Control": "no-store"}
        return Response(key.to_csv(index=False), mimetype="text/csv", headers=headers)
    filename = make_filename(result.recipe, result.seed, fmt).replace("." + fmt, tag + "." + fmt)
    headers = {"Content-Disposition": f"attachment; filename={filename}", "Cache-Control": "no-store"}
    if fmt == "csv":
        return Response(df.to_csv(index=False), mimetype="text/csv", headers=headers)
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return Response(buffer.getvalue(),
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers=headers)


@app.route("/healthz")
def healthz():
    """For the host's uptime check."""
    return "ok"


@app.after_request
def safety_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response
