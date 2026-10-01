"""Vehicle Maintenance Records.

One row is one completed workshop job (a service or a repair) on one vehicle
(vehicle_id).  Each vehicle appears many times, in date order.  Mileage is in
miles; money is plain numbers; labour is charged at a flat 80 per hour.

The story, in the order the code tells it, for each vehicle in turn:
    segment, age and yearly usage -> a day-by-day odometer
    -> scheduled services (every 10,000 miles OR 12 months, whichever first)
    -> repairs (chance per day rises as the car ages and racks up miles)
    -> cost = parts + labour hours x 80 -> warranty, recall and write-off flags

The odometer is built first and everything else reads from it, so a job's
mileage, its cost and the service schedule all agree.  Nothing flows backwards.
All numbers are illustrative teaching values, not real workshop data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli

# segment: (weight, new price, parts cost factor)
SEGMENTS = {
    "city":   (0.30, 15000, 0.80),
    "family": (0.35, 24000, 1.00),
    "suv":    (0.25, 36000, 1.30),
    "van":    (0.10, 28000, 1.20),
}
# repair: (base weight, median labour hours, median parts cost, wears out with use?)
REPAIRS = {
    "brakes":            (1.0, 2.0, 180, False),
    "battery":           (0.6, 0.5, 110, False),
    "tyres":             (0.8, 1.0, 260, False),
    "exhaust":           (0.5, 1.5, 200, False),
    "cooling system":    (0.4, 2.5, 150, False),
    "alternator":        (0.3, 2.0, 260, True),
    "suspension":        (0.5, 3.0, 300, True),
    "clutch":            (0.3, 5.0, 420, True),
    "engine or gearbox": (0.08, 12.0, 2600, True),
}
WEAR_OUT_REPAIRS = [name for name, spec in REPAIRS.items() if spec[3]]
# service: (median labour hours, median parts cost)
SERVICES = {"interim service": (1.5, 45), "full service": (3.0, 110)}

LABOR_RATE = 80               # cost per labour hour
SERVICE_MILES = 10000         # a service is due every 10,000 miles ...
SERVICE_DAYS = 365            # ... or 12 months, whichever comes first
WARRANTY_DAYS = 3 * 365       # 3-year warranty ...
WARRANTY_MILES = 60000        # ... or 60,000 miles, whichever comes first
WINDOW_DAYS = 4 * 365         # each vehicle is watched for the last 4 years
WRITE_OFF_SHARE = 0.60        # a repair costing 60%+ of the car's value writes it off


def half_hours(hours):
    """Workshops bill in half-hours, with a minimum of half an hour."""
    return max(0.5, float(np.round(hours * 2) / 2))


def one_vehicle(rng, snapshot):
    """Simulate one vehicle and return a list of job dictionaries."""
    segment = pick(rng, list(SEGMENTS), [v[0] for v in SEGMENTS.values()])
    new_price, parts_factor = SEGMENTS[segment][1], SEGMENTS[segment][2]
    start = snapshot - pd.Timedelta(days=WINDOW_DAYS)

    # Age when we start watching, and how hard the owner drives (a fixed trait).
    age_at_start = float(np.clip(lognormal(rng, 3.5, 0.6), 0.2, 12))
    registration = start - pd.Timedelta(days=int(age_at_start * 365))
    warranty_expiry = registration + pd.Timedelta(days=WARRANTY_DAYS)
    usage_per_year = float(lognormal(rng, 9000, 0.4))

    # Odometer for every day.  A running total of positive distances can only
    # go up, so mileage never decreases.
    days = WINDOW_DAYS + 1
    daily_miles = usage_per_year / 365 * rng.lognormal(-0.125, 0.5, days)
    odo_start = age_at_start * usage_per_year * float(lognormal(rng, 1.0, 0.1))
    odometer = odo_start + np.cumsum(daily_miles)
    age_years = age_at_start + np.arange(days) / 365
    wear = age_years / 10 + odometer / 100000          # rises with age and mileage
    vehicle_value = new_price * np.exp(-0.12 * age_years - 0.000004 * odometer)

    # Scheduled services: the earlier of the mileage limit and the time limit,
    # then the owner books it 0-30 days late.
    last_day = -int(rng.integers(0, 300))              # a service before we started watching
    initial_service_date = start + pd.Timedelta(days=last_day)
    last_odo = odo_start + daily_miles.mean() * last_day
    service_days = []
    while True:
        miles_day = int(np.searchsorted(odometer, last_odo + SERVICE_MILES))
        due_day = min(last_day + SERVICE_DAYS, miles_day)
        service_day = max(due_day + int(rng.integers(0, 31)), last_day + 1, 1)
        if service_day > WINDOW_DAYS:
            break
        service_days.append(service_day)
        last_day, last_odo = service_day, odometer[service_day]

    # Repairs: a small chance each day that rises with wear.  A repair that
    # lands on a service day is dropped so every job date is different.
    repair_days = np.flatnonzero(bernoulli(rng, (0.2 + 0.7 * wear) / 365))
    repair_days = [int(d) for d in repair_days if d > 0 and d not in service_days]

    events = sorted([(d, "service") for d in service_days] + [(d, "repair") for d in repair_days])

    # A recall notice arrives at some point for about 1 vehicle in 8; the first
    # job after the notice includes the recall work.
    has_recall = bool(rng.random() < 0.12)
    recall_day = int(rng.integers(0, WINDOW_DAYS + 1))

    jobs, oil_changes, last_service_date = [], 0, initial_service_date
    service_number = int(rng.integers(0, 2))           # alternate full / interim services
    recall_done = False
    for day, job_type in events:
        job_date = start + pd.Timedelta(days=day)
        if job_type == "service":
            work_done = "full service" if service_number % 2 == 0 else "interim service"
            service_number += 1
            hours_med, parts_med = SERVICES[work_done]
            oil_changes += 1                           # every service includes an oil change
            last_service_date = job_date
        else:
            names = list(REPAIRS)
            weights = [REPAIRS[k][0] * (1 + 2 * wear[day] if REPAIRS[k][3] else 1) for k in names]
            work_done = pick(rng, names, weights)
            hours_med, parts_med = REPAIRS[work_done][1], REPAIRS[work_done][2]

        hours = half_hours(lognormal(rng, hours_med, 0.30))
        parts = int(round(lognormal(rng, parts_med * parts_factor, 0.30)))
        total = parts + int(hours * LABOR_RATE)
        mileage = int(odometer[day])

        is_recalled = has_recall and not recall_done and day >= recall_day
        recall_done = recall_done or is_recalled
        is_written_off = job_type == "repair" and total >= WRITE_OFF_SHARE * vehicle_value[day]

        jobs.append({
            "segment": segment, "job_date": job_date, "work_done": work_done,
            "mileage_at_service": mileage, "oil_change_count": oil_changes,
            "labor_hours": hours, "parts_cost": parts, "total_cost": total,
            "last_service_date": last_service_date,
            "next_service_due": last_service_date + pd.Timedelta(days=SERVICE_DAYS),
            "warranty_expiry": warranty_expiry,
            "is_under_warranty": bool(job_date <= warranty_expiry and mileage <= WARRANTY_MILES),
            "is_recalled": bool(is_recalled), "is_written_off": bool(is_written_off),
            # hidden columns (kept for validation, not shown to students)
            "job_type": job_type, "vehicle_age_years": round(float(age_years[day]), 2),
            "vehicle_value": int(vehicle_value[day]), "usage_per_year": round(usage_per_year),
            "registration_date": registration,
        })
        if is_written_off:
            break                                      # a written-off car has no later jobs
    return jobs


def simulate(rng, n, snapshot):
    # Keep adding vehicles until there are at least n jobs in total.
    rows, vehicle_number = [], 0
    while len(rows) < n:
        vehicle_number += 1
        for job in one_vehicle(rng, snapshot):
            job["vehicle_id"] = f"VEH{vehicle_number:04d}"
            rows.append(job)
    df = pd.DataFrame(rows).iloc[:n].reset_index(drop=True)
    for col in ["job_date", "last_service_date", "next_service_due",
                "warranty_expiry", "registration_date"]:
        df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    g = df.groupby("vehicle_id")
    is_service = df.job_type == "service"

    # ---- exact rules ----
    out.append(check("total_cost = parts_cost + labor_hours x 80",
                     (df.total_cost == df.parts_cost + df.labor_hours * LABOR_RATE).all()))
    out.append(check("labor_hours are whole or half hours, >= 0.5",
                     ((df.labor_hours >= 0.5) & ((df.labor_hours * 2) % 1 == 0)).all()))
    out.append(check("each vehicle's rows are together", (df.vehicle_id != df.vehicle_id.shift()).sum()
                     == df.vehicle_id.nunique()))
    out.append(check("job_date strictly increases within a vehicle",
                     (g.job_date.diff().dropna() > pd.Timedelta(0)).all()))
    out.append(check("mileage_at_service never decreases within a vehicle",
                     (g.mileage_at_service.diff().dropna() > 0).all()))
    out.append(check("job_date not after the snapshot", (df.job_date <= snapshot).all()))
    out.append(check("oil_change_count = running number of services on the vehicle",
                     (g.job_type.transform(lambda s: (s == "service").cumsum()) == df.oil_change_count).all()))
    out.append(check("a service job's last_service_date is its own date",
                     (df.last_service_date[is_service] == df.job_date[is_service]).all()))
    latest = df.job_date.where(is_service)
    latest = latest.groupby(df.vehicle_id).ffill()
    has_service = latest.notna()
    out.append(check("last_service_date = date of the latest service so far (or an earlier one)",
                     (df.last_service_date[has_service] == latest[has_service]).all()
                     and (df.last_service_date <= df.job_date).all()))
    out.append(check("next_service_due = last_service_date + 365 days",
                     (df.next_service_due - df.last_service_date == pd.Timedelta(days=SERVICE_DAYS)).all()))
    previous_last = g.last_service_date.shift()
    gap = (df.job_date - previous_last).dt.days
    out.append(check("a service happens within 395 days of the previous one",
                     (gap[is_service & previous_last.notna()] <= SERVICE_DAYS + 30).all()))
    out.append(check("warranty_expiry = registration + 3 years",
                     (df.warranty_expiry - df.registration_date == pd.Timedelta(days=WARRANTY_DAYS)).all()))
    rule = (df.job_date <= df.warranty_expiry) & (df.mileage_at_service <= WARRANTY_MILES)
    out.append(check("is_under_warranty = before expiry AND within 60,000 miles",
                     (rule == df.is_under_warranty).all()))
    out.append(check("at most one recall job per vehicle", (g.is_recalled.sum() <= 1).all()))
    written_off_before = g.is_written_off.transform(lambda s: s.cumsum().shift(fill_value=0))
    out.append(check("a written-off vehicle has no later jobs", (written_off_before == 0).all()))
    out.append(check("write-off = repair costing >= 60% of the vehicle's value",
                     (df.is_written_off == ((df.job_type == "repair")
                                            & (df.total_cost >= WRITE_OFF_SHARE * df.vehicle_value))).all()))
    out.append(check("categories valid",
                     df.segment.isin(SEGMENTS).all()
                     and df.work_done.isin(list(REPAIRS) + list(SERVICES)).all()))

    # ---- relationships (tendencies) ----
    repair = (df.job_type == "repair").astype(float)
    old = df.vehicle_age_years >= df.vehicle_age_years.median()
    diff = float(repair[old].mean() - repair[~old].mean())
    out.append(check("older vehicles have a larger share of repair jobs (vs services)",
                     diff >= 0.02, f"gap={diff:.2f}", "relationship"))

    if (repair == 1).sum() >= 8 and is_service.sum() >= 8:
        out.append(check("median total_cost: repairs > services",
                         df.total_cost[repair == 1].median() > df.total_cost[is_service].median(),
                         "", "relationship"))

    wear = df.work_done.isin(WEAR_OUT_REPAIRS)
    if wear.sum() >= 6 and ((df.job_type == "repair") & ~wear).sum() >= 6:
        out.append(check("wear-out repairs (clutch, suspension, engine...) have dearer parts",
                         df.parts_cost[wear].median() > df.parts_cost[(df.job_type == "repair") & ~wear].median(),
                         "", "relationship"))

    # Gap between consecutive services on the same vehicle should be about a year or less.
    gaps = df[is_service].groupby("vehicle_id").job_date.diff().dt.days.dropna()
    if len(gaps) >= 10:
        med_gap = float(gaps.median())
        out.append(check("median gap between services is 200-400 days",
                         200 <= med_gap <= 400, f"median gap={med_gap:.0f}", "relationship"))

    # Heavy drivers get serviced more often (complete vehicles only).
    complete = df[df.vehicle_id != df.vehicle_id.iloc[-1]]
    per_vehicle = complete.groupby("vehicle_id").agg(
        services=("job_type", lambda s: (s == "service").sum()), usage=("usage_per_year", "first"))
    if len(per_vehicle) >= 12:
        c = float(per_vehicle.services.corr(per_vehicle.usage))
        out.append(check("vehicles driven more often get more services",
                         c > 0.15, f"corr={c:.2f}", "relationship"))
    return out


RECIPE = Recipe(
    key="automotive_maintenance",
    title="Vehicle Maintenance Records",
    row_meaning=("One row is one completed workshop job (a service or a repair) "
                 "on one vehicle; each vehicle appears many times."),
    simulate=simulate,
    columns=["vehicle_id", "segment", "job_date", "work_done", "mileage_at_service",
             "oil_change_count", "labor_hours", "parts_cost", "total_cost",
             "last_service_date", "next_service_due", "warranty_expiry",
             "is_under_warranty", "is_recalled", "is_written_off"],
    core=["work_done", "mileage_at_service", "labor_hours", "total_cost"],
    priority=["vehicle_id", "job_date", "parts_cost", "segment", "oil_change_count",
              "last_service_date", "next_service_due", "is_under_warranty",
              "warranty_expiry", "is_recalled", "is_written_off"],
    docs={
        "vehicle_id": "Identifier of the vehicle; the same vehicle appears in many rows.",
        "segment": "Vehicle type: city, family, suv or van (affects parts prices).",
        "job_date": "Date the job was completed.",
        "work_done": "What was done: interim service, full service, or a named repair (brakes, clutch, ...).",
        "mileage_at_service": "Odometer reading in miles on the job date; never falls for the same vehicle.",
        "oil_change_count": "Oil changes done on this vehicle so far in the records (every service includes one).",
        "labor_hours": "Workshop labour time billed, in hours (half-hour steps).",
        "parts_cost": "Cost of parts used, whole currency units.",
        "total_cost": "Total job cost: parts_cost + labor_hours x 80 per hour.",
        "last_service_date": "Date of the vehicle's most recent scheduled service up to and including this job.",
        "next_service_due": "Date the next service is due: last_service_date + 365 days.",
        "warranty_expiry": "Date the 3-year warranty ends (the warranty also stops at 60,000 miles).",
        "is_under_warranty": "True if the job date is before warranty_expiry and mileage is at most 60,000.",
        "is_recalled": "True if manufacturer recall work was done in this job (at most once per vehicle).",
        "is_written_off": "True if this repair cost 60% or more of the vehicle's value, so it was scrapped (its last job).",
    },
    validate=validate,
    date_cols=["job_date", "last_service_date", "next_service_due", "warranty_expiry"],
    targets={
        "total_cost": ["parts_cost", "labor_hours"],
        "is_written_off": ["total_cost", "parts_cost"],
        "is_recalled": [],
    },
)
