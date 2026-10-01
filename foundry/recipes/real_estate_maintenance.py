"""Property Maintenance Records.

One row is one completed repair job on one component (boiler, roof, windows,
electrics or plumbing) of one property (property_id).  Each property appears
many times.  Money is plain numbers; contractors bill a flat 60 per hour.

The story, in the order the code tells it, for each property in turn:
    age -> the age of each component (older houses have older parts)
    -> a routine inspection about once a year (twice a year if the property
       scores badly) -> the inspector scores the property from how worn its
       components are -> worn components are more likely to need a job
    -> job scope (minor, moderate or major) -> contractor hours and materials
    -> repair_cost = materials + hours x 60 -> flags.
A major job replaces the component, so its age resets to zero afterwards.

Rows are jobs that were completed on or before the snapshot date; jobs still
waiting to be done are not included.
All numbers are illustrative teaching values, not real building data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick

# component: (typical lifespan in years, replacement cost in materials)
COMPONENTS = {
    "boiler":    (15, 2400),
    "roof":      (40, 9000),
    "windows":   (30, 6500),
    "electrics": (35, 3500),
    "plumbing":  (40, 3000),
}
# component: labour hours multiplier
COMPONENT_HOURS = {"boiler": 0.8, "roof": 1.6, "windows": 1.2, "electrics": 1.0, "plumbing": 0.9}
# scope: (median labour hours, share of the component's replacement cost in materials)
SCOPES = {"minor": (2.0, 0.03), "moderate": (8.0, 0.15), "major": (24.0, 1.0)}
# property type: (weight, size factor)
TYPES = {"flat": (0.30, 0.7), "terraced": (0.30, 0.9), "semi-detached": (0.25, 1.1), "detached": (0.15, 1.4)}
# contractor: (hours factor, speed factor: multiplies days to complete the job)
CONTRACTORS = {"apex": (0.90, 1.0), "brightside": (1.00, 0.8), "cornerstone": (1.10, 1.2), "delta": (1.00, 1.0)}
DELAY_MEDIAN_DAYS = {"minor": 8, "moderate": 15, "major": 30}

HOURLY_RATE = 60              # contractor cost per hour
WINDOW_DAYS = 5 * 365         # each property is followed for the last 5 years
GOOD_INTERVAL = 365           # next inspection in a year ...
POOR_INTERVAL = 182           # ... or in six months if the score is below 50
HABITABLE_SCORE = 35          # a score below this means the property is not fit to live in
FLAG_COST = 3000              # jobs this dear are flagged for the landlord to review


def one_property(rng, snapshot):
    """Simulate one property and return its list of completed jobs."""
    start = snapshot - pd.Timedelta(days=WINDOW_DAYS)
    ptype = str(pick(rng, list(TYPES), [v[0] for v in TYPES.values()]))
    size = TYPES[ptype][1]
    property_age = float(np.clip(lognormal(rng, 50, 0.6), 5, 150))

    # Component ages: replaced at some point in the past, but never older than the building.
    install = {}
    for name, (lifespan, _) in COMPONENTS.items():
        age = min(property_age, lifespan * rng.uniform(0.05, 1.4))
        install[name] = start - pd.Timedelta(days=int(age * 365.25))

    jobs = []
    inspection_date = start + pd.Timedelta(days=int(rng.integers(0, 365)))
    while inspection_date <= snapshot:
        # Wear ratio = component age / typical lifespan.  Above 1 means overdue.
        ratio = {}
        for name, (lifespan, _) in COMPONENTS.items():
            ratio[name] = (inspection_date - install[name]).days / 365.25 / lifespan
        mean_ratio = np.mean([min(r, 1.6) for r in ratio.values()])

        # The inspector's score falls with wear and building age, plus judgement noise.
        score = int(np.clip(round(105 - 45 * mean_ratio - 0.25 * property_age + rng.normal(0, 7)), 1, 100))
        interval = GOOD_INTERVAL if score >= 50 else POOR_INTERVAL

        # Worn components are more likely to need a job, and a bigger one.
        for name, (lifespan, replace_cost) in COMPONENTS.items():
            r = ratio[name]
            if rng.random() >= min(0.85, 0.03 + 0.55 * r ** 2):
                continue
            p_major = float(np.clip(0.01 + 0.35 * max(r - 0.6, 0), 0, 0.5))
            scope = str(pick(rng, list(SCOPES), [(1 - p_major) * 0.7, (1 - p_major) * 0.3, p_major]))
            contractor = str(pick(rng, list(CONTRACTORS), [1, 1, 1, 1]))
            hours_factor, speed_factor = CONTRACTORS[contractor]

            hours = lognormal(rng, SCOPES[scope][0] * COMPONENT_HOURS[name] * size ** 0.5 * hours_factor, 0.25)
            hours = max(0.5, float(np.round(hours * 2) / 2))
            materials = int(round(replace_cost * size * SCOPES[scope][1] * lognormal(rng, 1.0, 0.25)))
            delay = int(np.clip(round(lognormal(rng, DELAY_MEDIAN_DAYS[scope] * speed_factor, 0.5)), 1, 120))
            repair_date = inspection_date + pd.Timedelta(days=delay)
            if repair_date > snapshot:
                continue                                   # not finished yet: not in the records

            jobs.append({
                "component": name, "job_scope": scope,
                "component_age_years": int((inspection_date - install[name]).days // 365.25),
                "inspection_date": inspection_date, "inspection_score": score,
                "repair_date": repair_date,
                "next_inspection_due": inspection_date + pd.Timedelta(days=interval),
                "contractor_hours": hours, "materials_cost": materials,
                "repair_cost": materials + int(hours * HOURLY_RATE),
                "is_renovated": scope == "major",
                "is_habitable": score >= HABITABLE_SCORE,
                # hidden columns (kept for validation, not shown to students)
                "component_install_date": install[name], "wear_ratio": round(float(r), 3),
                "property_age": round(property_age, 1), "contractor": contractor,
                "days_to_repair": delay,
            })
            if scope == "major":
                install[name] = repair_date                # the component is brand new again

        # Next inspection: on the due date plus a few days.
        inspection_date = inspection_date + pd.Timedelta(days=interval + int(rng.integers(0, 15)))
    return jobs


def simulate(rng, n, snapshot):
    rows, property_number = [], 0
    while len(rows) < n:
        property_number += 1
        for job in one_property(rng, snapshot):
            job["property_id"] = f"PROP{property_number:04d}"
            rows.append(job)
    df = pd.DataFrame(rows)
    df = df.sort_values(["property_id", "repair_date", "component"], kind="stable").reset_index(drop=True)
    df = df.iloc[:n].copy()
    df["total_spend_to_date"] = df.groupby("property_id").repair_cost.cumsum()
    df["is_flagged"] = (df.repair_cost >= FLAG_COST) | ~df.is_habitable
    for col in ["inspection_date", "repair_date", "next_inspection_due", "component_install_date"]:
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    g = df.groupby("property_id")

    # ---- exact rules ----
    out.append(check("repair_cost = materials_cost + contractor_hours x 60",
                     (df.repair_cost == df.materials_cost + df.contractor_hours * HOURLY_RATE).all()))
    out.append(check("contractor_hours are whole or half hours, >= 0.5",
                     ((df.contractor_hours >= 0.5) & ((df.contractor_hours * 2) % 1 == 0)).all()))
    out.append(check("repair_date is on or after inspection_date, and not after the snapshot",
                     ((df.repair_date >= df.inspection_date) & (df.repair_date <= snapshot)).all()))
    out.append(check("repair_date - inspection_date = days_to_repair (1 to 120)",
                     ((df.repair_date - df.inspection_date).dt.days == df.days_to_repair).all()
                     and df.days_to_repair.between(1, 120).all()))
    gap = (df.next_inspection_due - df.inspection_date).dt.days
    out.append(check("next_inspection_due = inspection_date + 365 days (182 if score < 50)",
                     (gap == np.where(df.inspection_score >= 50, GOOD_INTERVAL, POOR_INTERVAL)).all()))
    out.append(check("inspection_score is between 1 and 100", df.inspection_score.between(1, 100).all()))
    out.append(check("is_habitable = inspection_score >= 35",
                     (df.is_habitable == (df.inspection_score >= HABITABLE_SCORE)).all()))
    out.append(check("is_flagged = repair_cost >= 3000 or not habitable",
                     (df.is_flagged == ((df.repair_cost >= FLAG_COST) | ~df.is_habitable)).all()))
    out.append(check("is_renovated = major job (component replaced)",
                     (df.is_renovated == (df.job_scope == "major")).all()))
    out.append(check("total_spend_to_date = running total of repair_cost per property",
                     (df.total_spend_to_date == g.repair_cost.cumsum()).all()))
    out.append(check("repair_date never goes backwards within a property",
                     (g.repair_date.diff().dropna() >= pd.Timedelta(0)).all()))
    out.append(check("component_age_years = whole years since the component was installed",
                     (df.component_age_years == ((df.inspection_date - df.component_install_date).dt.days // 365.25)).all()
                     and (df.component_age_years >= 0).all()))
    key = [df.property_id, df.component]
    previous_renovated = df.groupby(key).is_renovated.shift()
    previous_repair = df.groupby(key).repair_date.shift()
    after_renovation = previous_renovated.fillna(False).astype(bool)
    out.append(check("after a major job the component counts as new (install date = repair date)",
                     (df.component_install_date[after_renovation] == previous_repair[after_renovation]).all()))
    out.append(check("one job per property, component and inspection",
                     not df.duplicated(["property_id", "component", "inspection_date"]).any()))
    out.append(check("categories valid",
                     df.component.isin(COMPONENTS).all() and df.job_scope.isin(SCOPES).all()))
    out.append(check("costs are positive", ((df.materials_cost > 0) & (df.repair_cost > 0)).all()))

    # ---- relationships (tendencies) ----
    med = df.groupby("job_scope").repair_cost.median()
    sizes = df.job_scope.value_counts()
    if all(sizes.get(s, 0) >= 5 for s in SCOPES):
        out.append(check("median repair_cost: minor < moderate < major",
                         med["minor"] < med["moderate"] < med["major"], "", "relationship"))

        ratio_gap = float(df.wear_ratio[df.job_scope == "major"].mean() - df.wear_ratio[df.job_scope == "minor"].mean())
        out.append(check("major jobs happen on more worn components than minor jobs",
                         ratio_gap >= 0.08, f"wear gap={ratio_gap:.2f}", "relationship"))

        delay_med = df.groupby("job_scope").days_to_repair.median()
        out.append(check("major jobs take longer to complete than minor jobs (median days)",
                         delay_med["major"] > delay_med["minor"], "", "relationship"))

    c1 = float(df.inspection_score.corr(df.property_age))
    out.append(check("older properties get lower inspection scores",
                     c1 <= -0.35, f"corr={c1:.2f}", "relationship"))

    log_h = np.log(df.contractor_hours)
    log_m = np.log(df.materials_cost)
    h_dev = log_h - log_h.groupby(df.component).transform("mean")
    m_dev = log_m - log_m.groupby(df.component).transform("mean")
    c2 = float(h_dev.corr(m_dev))
    out.append(check("within a component, jobs needing more hours also need more materials",
                     c2 >= 0.8, f"corr={c2:.2f}", "relationship"))
    return out


RECIPE = Recipe(
    key="real_estate_maintenance",
    title="Property Maintenance Records",
    row_meaning=("One row is one completed repair job on one component of one property, "
                 "found at a routine inspection; each property appears many times."),
    simulate=simulate,
    columns=["property_id", "component", "job_scope", "component_age_years",
             "inspection_date", "inspection_score", "repair_date", "next_inspection_due",
             "contractor_hours", "materials_cost", "repair_cost", "total_spend_to_date",
             "is_renovated", "is_habitable", "is_flagged"],
    core=["component", "job_scope", "contractor_hours", "repair_cost"],
    priority=["component_age_years", "inspection_score", "property_id", "materials_cost",
              "repair_date", "inspection_date", "is_renovated", "total_spend_to_date",
              "is_habitable", "is_flagged", "next_inspection_due"],
    docs={
        "property_id": "Identifier of the property; the same property appears in many rows.",
        "component": "Part of the building repaired: boiler, roof, windows, electrics or plumbing.",
        "job_scope": "Size of the job: minor, moderate or major (a major job replaces the component).",
        "component_age_years": "Whole years since the component was installed, at the inspection that found the problem.",
        "inspection_date": "Date of the routine inspection that identified the job.",
        "inspection_score": "Inspector's condition score for the whole property, 1 (very poor) to 100 (excellent).",
        "repair_date": "Date the contractor finished the job (on or after inspection_date).",
        "next_inspection_due": "Date of the next routine inspection: 365 days after inspection_date, or 182 if the score was below 50.",
        "contractor_hours": "Contractor labour time billed, in hours (half-hour steps).",
        "materials_cost": "Cost of materials for the job, whole units.",
        "repair_cost": "Total job cost: materials_cost + contractor_hours x 60 per hour, whole units.",
        "total_spend_to_date": "Running total of repair_cost for this property up to and including this job.",
        "is_renovated": "True if the job replaced the whole component (a major job).",
        "is_habitable": "True if the inspection score was at least 35, meaning the property was fit to live in.",
        "is_flagged": "True if the job cost 3,000 or more, or the property was not habitable, so the landlord reviews it.",
    },
    validate=validate,
    date_cols=["inspection_date", "repair_date", "next_inspection_due"],
    targets={
        "repair_cost": ["materials_cost", "total_spend_to_date", "is_flagged"],
        "job_scope": ["repair_cost", "materials_cost", "contractor_hours", "is_renovated"],
        "is_flagged": ["repair_cost", "is_habitable", "inspection_score"],
    },
)
