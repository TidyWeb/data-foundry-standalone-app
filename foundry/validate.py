"""Validation harness for the theme recipes.

Run from the project folder:

    python -m foundry.validate                      # every theme
    python -m foundry.validate real_estate_listings # one theme
    python -m foundry.validate --seeds 20 --rows 200 3000

For each theme, seed and row count it runs the recipe, then:
  * exact rules must hold in 100% of runs;
  * relationship checks (tendencies) must hold in most runs;
  * the same seed must reproduce the identical table, different seeds must differ;
  * every visible column must be documented, present and free of missing values.

It writes tables (and plots, if matplotlib is installed) to validation_reports/.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .core import simulate_full
from .recipes import RECIPES

REL_THRESHOLD = {200: 0.75, 3000: 0.90}     # share of seeds a tendency must hold in


def structural_problems(recipe, full):
    problems = []
    missing = [c for c in recipe.columns if c not in full.columns]
    if missing:
        problems.append(f"columns missing from simulation: {missing}")
        return problems
    if set(recipe.core) | set(recipe.priority) != set(recipe.columns):
        problems.append("core + priority must equal columns")
    if set(recipe.core) & set(recipe.priority):
        problems.append("core and priority overlap")
    undocumented = [c for c in recipe.columns if c not in recipe.docs]
    if undocumented:
        problems.append(f"undocumented columns: {undocumented}")
    for c in recipe.columns:
        if full[c].isna().any():
            problems.append(f"missing values in {c}")
    for c in recipe.date_cols + recipe.datetime_cols:
        if not pd.api.types.is_datetime64_any_dtype(full[c]):
            problems.append(f"{c} is not a datetime column")
    numeric = full[recipe.columns].select_dtypes("number")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        problems.append("infinite or NaN numeric values")
    return problems


def run_recipe(recipe, seeds, row_counts, snapshot):
    exact_failures, rel_rows, problems = [], [], []
    for rows in row_counts:
        rel_hits = {}
        for seed in seeds:
            try:
                full, snap = simulate_full(recipe, rows, seed, snapshot)
            except Exception as exc:                       # a crash is an exact failure
                exact_failures.append((rows, seed, f"CRASH {type(exc).__name__}: {exc}"))
                continue
            for p in structural_problems(recipe, full):
                problems.append((rows, seed, p))
            for chk in recipe.validate(full, snap):
                if chk.kind == "exact":
                    if not chk.ok:
                        exact_failures.append((rows, seed, f"{chk.name} {chk.detail}"))
                else:
                    hit = rel_hits.setdefault(chk.name, [0, 0])
                    hit[0] += int(chk.ok)
                    hit[1] += 1
        for name, (ok, total) in rel_hits.items():
            rel_rows.append({"theme": recipe.key, "rows": rows, "check": name,
                             "seeds_passing": ok, "seeds_run": total,
                             "pass_rate": ok / total})
    # reproducibility
    a, sn = simulate_full(recipe, row_counts[0], seeds[0], snapshot)
    b, _ = simulate_full(recipe, row_counts[0], seeds[0], snapshot)
    c, _ = simulate_full(recipe, row_counts[0], seeds[1], snapshot)
    if not a.equals(b):
        exact_failures.append((row_counts[0], seeds[0], "same seed did not reproduce the table"))
    if a.equals(c):
        exact_failures.append((row_counts[0], seeds[0], "different seeds gave identical tables"))
    return exact_failures, pd.DataFrame(rel_rows), problems


def plot_theme(recipe, out_dir, snapshot, seed=1000, rows=3000):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    full, _ = simulate_full(recipe, rows, seed, snapshot)
    num = full[recipe.columns].select_dtypes("number")
    if num.shape[1] >= 2:
        corr = num.corr()
        fig, ax = plt.subplots(figsize=(0.7 * len(corr) + 3, 0.7 * len(corr) + 2))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
        ax.set_yticks(range(len(corr)), corr.columns)
        ax.set_title(f"{recipe.title}: correlations ({rows} rows, seed {seed})")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(out_dir / f"{recipe.key}_correlations.png", dpi=110)
        plt.close(fig)
    if recipe.key == "real_estate_listings":
        plot_property_listings(full, out_dir, plt)


def plot_property_listings(full, out_dir, plt):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for area, g in full.groupby("location_area"):
        axes[0, 0].scatter(np.log(g.square_footage), np.log(g.listing_price), s=4, label=area)
    axes[0, 0].set(title="ln(price) vs ln(size), by area", xlabel="ln sq ft", ylabel="ln price")
    axes[0, 0].legend(fontsize=6, markerscale=3)
    order = ["poor", "fair", "good", "excellent"]
    axes[0, 1].boxplot([full[full.condition == c].price_per_sqft for c in order], tick_labels=order)
    axes[0, 1].set(title="price per sq ft by condition")
    types = ["flat", "terraced", "semi-detached", "detached"]
    axes[0, 2].boxplot([full[full.property_type == t].price_per_sqft for t in types], tick_labels=types)
    axes[0, 2].set(title="price per sq ft by property type")
    q = pd.qcut(full.asking_premium, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    for lab in q.cat.categories:
        axes[1, 0].hist(full.days_on_market[q == lab], bins=40, alpha=0.5, label=str(lab))
    axes[1, 0].set(title="days on market by asking-price quartile", xlabel="days")
    axes[1, 0].legend()
    axes[1, 1].hist(full.listing_price, bins=60)
    axes[1, 1].set(title="listing price distribution")
    axes[1, 2].bar(*np.unique(full.bedrooms, return_counts=True))
    axes[1, 2].set(title="bedrooms")
    fig.tight_layout()
    fig.savefig(out_dir / "real_estate_listings_overview.png", dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("themes", nargs="*")
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--rows", type=int, nargs="+", default=[200, 3000])
    ap.add_argument("--out", default="validation_reports")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    keys = args.themes or sorted(RECIPES)
    unknown = [k for k in keys if k not in RECIPES]
    if unknown:
        sys.exit(f"unknown themes: {unknown}\nknown: {sorted(RECIPES)}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    snapshot = pd.Timestamp("2026-09-30")
    seeds = list(range(1000, 1000 + args.seeds))

    summary, all_rel, lines, failed = [], [], [], False
    for key in keys:
        recipe = RECIPES[key]
        exact, rel, problems = run_recipe(recipe, seeds, args.rows, snapshot)
        if not rel.empty:
            all_rel.append(rel)
        rel_bad = []
        for _, r in rel.iterrows():
            need = REL_THRESHOLD.get(int(r["rows"]), 0.75)
            if r["pass_rate"] < need:
                rel_bad.append(f'{r["check"]} @ {int(r["rows"])} rows: {r["pass_rate"]:.0%} < {need:.0%}')
        status = "PASS" if not (exact or problems or rel_bad) else "FAIL"
        failed |= status == "FAIL"
        summary.append({"theme": key, "status": status, "exact_failures": len(exact),
                        "structural_problems": len(problems), "weak_relationships": len(rel_bad)})
        lines.append(f"{status}  {key}")
        for rows, seed, msg in exact[:5]:
            lines.append(f"      exact  rows={rows} seed={seed}: {msg}")
        for rows, seed, msg in problems[:5]:
            lines.append(f"      struct rows={rows} seed={seed}: {msg}")
        for msg in rel_bad:
            lines.append(f"      weak   {msg}")
        if not args.no_plots:
            plot_theme(recipe, out_dir, snapshot)

    pd.DataFrame(summary).to_csv(out_dir / "validation_summary.csv", index=False)
    if all_rel:
        pd.concat(all_rel).to_csv(out_dir / "validation_relationships.csv", index=False)
    (out_dir / "validation_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{sum(s['status'] == 'PASS' for s in summary)}/{len(summary)} themes pass "
          f"({args.seeds} seeds x rows {args.rows}). Reports in {out_dir}/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
