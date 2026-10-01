"""Quality Control & Defects.

One row is one inspected production batch.  From each batch the inspectors
check a sample (`inspected` units) and count how many units were defective and
how many individual defects they found (one bad unit can have several).
Rates are percentages (0-100).

The story, in the order the code tells it:
    batch (date, shift, machine, supplier grade) -> how worn the machine is
    -> chance a unit is defective (logistic model of those conditions)
    -> sample size -> defective units ~ Binomial -> defects (>= defective units)
    -> rework -> inspection score -> compliant? -> recalled?

The key teaching point: with a small sample, defect_rate bounces around even when
the true process is unchanged (binomial noise), while wear, supplier and night
shifts move the underlying chance.  Illustrative teaching values.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, pick, logistic, bernoulli, lognormal

SHIFTS = ["day", "evening", "night"]
SHIFT_EFFECT = {"day": 0.0, "evening": 0.10, "night": 0.35}      # added to the log-odds
SUPPLIER_EFFECT = {"A": 0.0, "B": 0.35, "C": 0.80}                 # raw-material grade A is best
BASE_LOG_ODDS = np.log(0.009 / 0.991)      # about 0.9% defective under the best conditions
WEAR_EFFECT = 0.0025                       # log-odds per hour since the machine was serviced
SERVICE_INTERVAL_HOURS = 480               # a machine is serviced after this many running hours
SPEC_LIMIT_PERCENT = 4.0                   # batch is compliant if defect_rate <= this
REWORKABLE_SHARE = 0.65                    # share of defective units that can be repaired
N_MACHINES = 6
WINDOW_DAYS = 365
MAX_INSPECTION_LAG = 3                     # days between production and inspection


def simulate(rng, n, snapshot):
    # 1. Batches: production date, shift, machine, supplier grade.
    latest = snapshot - pd.Timedelta(days=MAX_INSPECTION_LAG)
    day_offsets = rng.integers(0, WINDOW_DAYS, n)
    production_date = latest - pd.to_timedelta(day_offsets, unit="D")
    shift = pick(rng, SHIFTS, [0.45, 0.35, 0.20], n)
    machine = rng.integers(0, N_MACHINES, n)
    supplier_grade = pick(rng, list(SUPPLIER_EFFECT), [0.5, 0.35, 0.15], n)
    batch_size = np.round(lognormal(rng, 1200, 0.5, n), -1).clip(200, 6000).astype(int)

    order = np.argsort(production_date.to_numpy(), kind="stable")   # oldest batch first
    production_date, shift = production_date[order], shift[order]
    machine, supplier_grade, batch_size = machine[order], supplier_grade[order], batch_size[order]

    # 2. Machine wear: hours run since the last service.  Each machine runs a
    #    batch of ~batch_size/100 hours, and is reset every SERVICE_INTERVAL_HOURS.
    hours_run = batch_size / 100 * lognormal(rng, 1.0, 0.2, n)
    machine_wear = np.zeros(n)
    machine_clock = rng.uniform(0, SERVICE_INTERVAL_HOURS, N_MACHINES)   # where each machine starts
    for i in range(n):
        machine_clock[machine[i]] = (machine_clock[machine[i]] + hours_run[i]) % SERVICE_INTERVAL_HOURS
        machine_wear[i] = machine_clock[machine[i]]
    machine_wear = np.round(machine_wear).astype(int)

    # 3. Chance that a single unit is defective (logistic model), including
    #    unobserved process variation shared by everything in the batch.
    unobserved = rng.normal(0, 0.25, n)
    log_odds = (BASE_LOG_ODDS + WEAR_EFFECT * machine_wear
                + np.array([SUPPLIER_EFFECT[g] for g in supplier_grade])
                + np.array([SHIFT_EFFECT[s] for s in shift]) + unobserved)
    defect_chance = logistic(log_odds)

    # 4. Inspect a sample (about 10% of the batch), count defective units and defects.
    inspected = np.clip(np.round(batch_size * 0.10 / 5) * 5, 30, 400).astype(int)
    defective_units = rng.binomial(inspected, defect_chance)
    defects = defective_units + rng.poisson(0.4 * defective_units)      # extra defects on bad units
    rework_count = rng.binomial(defective_units, REWORKABLE_SHARE)
    defect_rate = np.round(100 * defective_units / inspected, 2)

    # 5. Inspection score: the inspector's overall mark out of 100, driven by
    #    defects per inspected unit, plus inspector judgement.
    defects_per_100 = 100 * defects / inspected
    inspection_score = np.clip(np.round(100 - 3 * defects_per_100 + rng.normal(0, 3, n)), 0, 100).astype(int)

    # 6. Compliance is a rule; a recall is a rare event that follows bad quality.
    is_compliant = defect_rate <= SPEC_LIMIT_PERCENT
    p_recall = np.where(defect_rate > 2 * SPEC_LIMIT_PERCENT, 0.35,
                        np.where(~is_compliant, 0.10, 0.004))
    is_recalled = bernoulli(rng, p_recall)

    inspection_date = production_date + pd.to_timedelta(rng.integers(0, MAX_INSPECTION_LAG + 1, n), unit="D")
    df = pd.DataFrame({
        "batch_id": [f"B{i + 1:05d}" for i in range(n)],
        "production_date": production_date,
        "inspection_date": inspection_date,
        "shift": shift,
        "supplier_grade": supplier_grade,
        "machine_wear": machine_wear,
        "inspected": inspected,
        "defective_units": defective_units,
        "defects": defects,
        "defect_rate": defect_rate,
        "rework_count": rework_count,
        "inspection_score": inspection_score,
        "is_compliant": is_compliant,
        "is_recalled": is_recalled,
        # hidden helpers
        "machine_id": [f"M{m + 1}" for m in machine],
        "batch_size": batch_size,
        "true_defect_chance": np.round(defect_chance, 5),
    })
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []

    # ---- exact rules ----
    out.append(check("defective_units <= inspected <= batch_size",
                     ((df.defective_units <= df.inspected) & (df.inspected <= df.batch_size)).all()))
    out.append(check("defects >= defective_units", (df.defects >= df.defective_units).all()))
    out.append(check("rework_count <= defective_units",
                     ((df.rework_count <= df.defective_units) & (df.rework_count >= 0)).all()))
    out.append(check("defect_rate = defective_units / inspected x 100 (+-0.005)",
                     ((df.defect_rate - 100 * df.defective_units / df.inspected).abs() <= 0.0051).all()))
    out.append(check("is_compliant exactly when defect_rate <= the specification limit",
                     (df.is_compliant == (df.defect_rate <= SPEC_LIMIT_PERCENT)).all()))
    out.append(check("inspection_date >= production_date, within 3 days",
                     ((pd.to_datetime(df.inspection_date) - pd.to_datetime(df.production_date))
                      .dt.days.between(0, MAX_INSPECTION_LAG)).all()))
    out.append(check("dates not after snapshot", (pd.to_datetime(df.inspection_date) <= snapshot).all()))
    out.append(check("inspection_score between 0 and 100", df.inspection_score.between(0, 100).all()))
    out.append(check("machine_wear between 0 and the service interval",
                     df.machine_wear.between(0, SERVICE_INTERVAL_HOURS).all()))
    out.append(check("categories valid",
                     df["shift"].isin(SHIFTS).all() and df.supplier_grade.isin(SUPPLIER_EFFECT).all()))
    out.append(check("batch_id unique", df.batch_id.is_unique))
    out.append(check("sample size is at least 30 units",
                     (df.inspected >= 30).all()))

    # ---- relationships ----
    n = len(df)
    c = float(df.machine_wear.rank().corr(df.defect_rate.rank())) if n >= 30 else 1.0
    out.append(check("more machine wear goes with higher defect rates", c > 0.2,
                     f"rank corr={c:.2f}", "relationship"))
    a, cgrade = df[df.supplier_grade == "A"], df[df.supplier_grade == "C"]
    if len(a) >= 10 and len(cgrade) >= 10:
        out.append(check("grade C material has a higher median defect rate than grade A",
                         cgrade.defect_rate.median() > a.defect_rate.median(),
                         f"{cgrade.defect_rate.median():.2f} vs {a.defect_rate.median():.2f}",
                         "relationship"))
    night, day = df[df["shift"] == "night"], df[df["shift"] == "day"]
    if len(night) >= 10 and len(day) >= 10:
        out.append(check("night shift has a higher mean defect rate than day shift",
                         night.defect_rate.mean() > day.defect_rate.mean() + 0.2,
                         f"{night.defect_rate.mean():.2f} vs {day.defect_rate.mean():.2f}",
                         "relationship"))
    c = float(df.inspection_score.rank().corr(df.defect_rate.rank())) if n >= 30 else -1.0
    out.append(check("higher defect rate goes with a lower inspection score", c < -0.5,
                     f"rank corr={c:.2f}", "relationship"))
    rec = df[df.is_recalled]
    if len(rec) >= 3 and (~df.is_recalled).sum() >= 20:
        out.append(check("recalled batches had higher defect rates than the rest",
                         rec.defect_rate.mean() > df.defect_rate[~df.is_recalled].mean(),
                         "", "relationship"))
    return out


RECIPE = Recipe(
    key="manufacturing_quality",
    title="Quality Control & Defects",
    row_meaning="One row is one production batch that was inspected by sampling.",
    simulate=simulate,
    columns=["batch_id", "production_date", "inspection_date", "shift", "supplier_grade",
             "machine_wear", "inspected", "defective_units", "defects", "defect_rate",
             "rework_count", "inspection_score", "is_compliant", "is_recalled"],
    core=["machine_wear", "supplier_grade", "inspected", "defect_rate"],
    priority=["defective_units", "shift", "inspection_score", "is_compliant", "defects",
              "rework_count", "is_recalled", "batch_id", "production_date", "inspection_date"],
    docs={
        "batch_id": "Unique batch identifier, numbered in production order.",
        "production_date": "Date the batch was produced.",
        "inspection_date": "Date the sample was inspected (0 to 3 days after production).",
        "shift": "Shift that produced the batch: day, evening or night.",
        "supplier_grade": "Quality grade of the raw material: A (best), B or C.",
        "machine_wear": "Hours the machine had run since its last service (0 to 480).",
        "inspected": "Number of units checked in the sample.",
        "defective_units": "Units in the sample with at least one defect.",
        "defects": "Total individual defects found; a defective unit can have several.",
        "defect_rate": "Defective units as a percentage of inspected units (0-100, 2 dp).",
        "rework_count": "Defective units that were repaired and put back into the batch.",
        "inspection_score": "Inspector's overall mark for the batch out of 100 (higher is better).",
        "is_compliant": "True if defect_rate is at or below the 4.0% specification limit.",
        "is_recalled": "True if the batch was later recalled from customers (rare, mostly after poor quality).",
    },
    validate=validate,
    date_cols=["production_date", "inspection_date"],
    targets={
        "is_compliant": ["defect_rate", "defective_units", "defects"],
        "defect_rate": ["defective_units", "defects", "is_compliant", "inspection_score", "rework_count"],
        "is_recalled": ["is_compliant"],
    },
)
