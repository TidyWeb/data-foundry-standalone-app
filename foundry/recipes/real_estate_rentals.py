"""Rental Property Management.

One row is one unit-month in a rent ledger: one occupied rental unit (unit_id)
in one calendar month.  Months when a unit stood empty have no row - that is
what a "void" looks like here - but they still lower occupancy_rate.
Money is plain numbers, paid monthly.

The story, in the order the code tells it:
    building (area, pet policy) -> units (bedrooms, size) -> market rent
    -> tenancy timeline for each unit: tenant moves in, signs a lease
       (6, 12 or 24 months), then renews or leaves, leaving the unit empty
       for a month or two
    -> each month the tenant pays on time (likely for reliable tenants) or
       late; late payment adds a late fee and can leave arrears
    -> occupancy_rate = occupied units / all units in that building that month

Money identity, every row:
    closing_arrears = opening arrears + monthly_rent + late_fee - amount_paid
(opening arrears is last month's closing figure; it starts at 0 for a new
tenant and any balance left when a tenant leaves is settled from the deposit).
All numbers are illustrative teaching values, not real market data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick

# area: (weight, rent per square foot per month)
AREAS = {"central": (0.30, 1.90), "suburban": (0.40, 1.40), "outer": (0.30, 1.05)}
# bedrooms: (weight, median floor area in sq ft)
BEDROOMS = {1: (0.30, 450), 2: (0.40, 650), 3: (0.22, 850), 4: (0.08, 1100)}
LEASE_TERMS = [6, 12, 24]
LEASE_WEIGHTS = [0.25, 0.60, 0.15]

WINDOW_MONTHS = 12            # the ledger covers the last 12 full months
WARMUP_MONTHS = 48            # history simulated first, so leases are mid-flight at the start
RENT_GROWTH = 0.005           # market rents rise about 0.5% a month
DEPOSIT_MULTIPLE = 1.2        # deposit = 1.2 x monthly rent
LATE_FEE_SHARE = 0.05         # late fee = 5% of rent, to the nearest 5


def late_fee_for(rent):
    return max(10, int(round(LATE_FEE_SHARE * rent / 5)) * 5)


def simulate_unit(rng, unit):
    """Walk one unit through time.  Returns its list of occupied-month rows."""
    rows = []
    month = -WARMUP_MONTHS + int(rng.integers(0, 12))          # first tenant arrives
    while month < WINDOW_MONTHS:
        # A new tenant: reliability is a fixed trait that drives paying on time.
        reliability = float(rng.beta(5, 2))
        tenant_count = 1 + int(rng.binomial(unit["bedrooms"], 0.5))     # 1 to bedrooms + 1
        move_in_month = month
        move_in_offset = int(rng.integers(0, 7))
        arrears, rent, first_lease = 0, None, True
        while True:
            # A lease: rent is set at signing (market rent for a new tenant, a small
            # rise for a renewal) and the deposit is a multiple of that rent.
            term = int(pick(rng, LEASE_TERMS, LEASE_WEIGHTS))
            if first_lease:
                market = unit["base_rent"] * np.exp(RENT_GROWTH * month) * lognormal(rng, 1.0, 0.03)
                rent = int(round(market / 5) * 5)
            else:
                rent = int(round(rent * rng.uniform(1.02, 1.06) / 5) * 5)
            deposit = int(round(DEPOSIT_MULTIPLE * rent))
            fee = late_fee_for(rent)
            for m in range(month, month + term):
                if m >= WINDOW_MONTHS:
                    break
                # Paying: reliable tenants pay on time, and anyone already in
                # arrears is a little less likely to.
                p_on_time = float(np.clip(0.55 + 0.42 * reliability - 0.20 * (arrears > 0), 0.05, 0.99))
                opening = arrears
                if rng.random() < p_on_time:
                    late_fee, paid = 0, opening + rent
                elif rng.random() < 0.75:
                    late_fee, paid = fee, opening + rent      # paid after the grace period
                else:
                    late_fee, paid = fee, 0                   # missed the month entirely
                arrears = opening + rent + late_fee - paid
                if m >= 0:
                    rows.append({
                        "unit": unit, "month_number": m, "monthly_rent": rent,
                        "deposit_amount": deposit, "tenant_count": tenant_count,
                        "lease_start_month": month, "lease_end_month": month + term - 1,
                        "move_in_month": move_in_month, "move_in_offset": move_in_offset,
                        "late_fee": late_fee, "amount_paid": paid, "opening_arrears": opening,
                        "closing_arrears": arrears, "reliability": round(reliability, 3),
                    })
            month += term
            first_lease = False
            if month >= WINDOW_MONTHS:
                return rows
            if rng.random() >= np.clip(0.25 + 0.40 * reliability, 0, 1):
                break                                          # tenant leaves
        month += 1 + int(rng.poisson(1.3))                     # unit stands empty
    return rows


def simulate(rng, n, snapshot):
    # The ledger ends with the last month that has fully finished.
    last_month = snapshot.to_period("M")
    if snapshot != snapshot + pd.offsets.MonthEnd(0):
        last_month = last_month - 1
    first_month = last_month - (WINDOW_MONTHS - 1)

    rows, unit_count, building_count = [], 0, 0
    want_units = max(4, int(np.ceil(n / 9)))
    while unit_count < want_units or len(rows) < n:
        # One building: area, pet policy and 3-6 units.
        building_count += 1
        area = str(pick(rng, list(AREAS), [v[0] for v in AREAS.values()]))
        pet_friendly = bool(rng.random() < 0.35)
        units_here = int(rng.integers(3, 7))
        building_rows = []
        for _ in range(units_here):
            unit_count += 1
            bedrooms = int(pick(rng, list(BEDROOMS), [v[0] for v in BEDROOMS.values()]))
            sqft = lognormal(rng, BEDROOMS[bedrooms][1], 0.10)
            base_rent = (sqft * AREAS[area][1] * lognormal(rng, 1.0, 0.06)
                         * (1.03 if pet_friendly else 1.0))
            unit = {"unit_id": f"U{unit_count:04d}", "building": building_count, "area": area,
                    "bedrooms": bedrooms, "pet": pet_friendly, "base_rent": base_rent}
            building_rows += simulate_unit(rng, unit)
        for r in building_rows:
            r["units_in_building"] = units_here
        rows += building_rows

    # Turn the row dictionaries into a table.
    def month_start(offset):
        return (first_month + offset).to_timestamp()
    table = pd.DataFrame({
        "unit_id": [r["unit"]["unit_id"] for r in rows],
        "building_id": [r["unit"]["building"] for r in rows],
        "month": [month_start(r["month_number"]) for r in rows],
        "area": [r["unit"]["area"] for r in rows],
        "bedrooms": [r["unit"]["bedrooms"] for r in rows],
        "is_pet_friendly": [r["unit"]["pet"] for r in rows],
        "tenant_count": [r["tenant_count"] for r in rows],
        "monthly_rent": [r["monthly_rent"] for r in rows],
        "deposit_amount": [r["deposit_amount"] for r in rows],
        "lease_start": [month_start(r["lease_start_month"]) for r in rows],
        "lease_end": [month_start(r["lease_end_month"] + 1) - pd.Timedelta(days=1) for r in rows],
        "late_fee": [r["late_fee"] for r in rows],
        "amount_paid": [r["amount_paid"] for r in rows],
        "closing_arrears": [r["closing_arrears"] for r in rows],
        # hidden columns (kept for validation, not shown to students)
        "opening_arrears": [r["opening_arrears"] for r in rows],
        "units_in_building": [r["units_in_building"] for r in rows],
        "reliability": [r["reliability"] for r in rows],
    })
    table["move_in_date"] = [month_start(r["move_in_month"]) + pd.Timedelta(days=r["move_in_offset"])
                             for r in rows]

    # Occupancy: how many of the building's units were let in each month.
    table["occupied_units"] = table.groupby(["building_id", "month"]).unit_id.transform("count")
    table["occupancy_rate"] = np.round(table.occupied_units / table.units_in_building, 2)

    table = table.sort_values(["unit_id", "month"], kind="stable").iloc[:n].reset_index(drop=True)
    for col in ["month", "lease_start", "lease_end", "move_in_date"]:
        table[col] = pd.to_datetime(table[col]).astype("datetime64[ns]")
    return table


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    month_end = df.month + pd.offsets.MonthEnd(0)

    # ---- exact rules ----
    out.append(check("closing_arrears = opening arrears + monthly_rent + late_fee - amount_paid",
                     (df.closing_arrears == df.opening_arrears + df.monthly_rent + df.late_fee - df.amount_paid).all()))
    out.append(check("arrears and payments are never negative",
                     ((df.closing_arrears >= 0) & (df.amount_paid >= 0) & (df.opening_arrears >= 0)).all()))
    same_unit = df.unit_id == df.unit_id.shift()
    consecutive = same_unit & (df.month.diff() > pd.Timedelta(days=27)) & (df.month.diff() < pd.Timedelta(days=32))
    same_tenant = consecutive & (df.move_in_date == df.move_in_date.shift())
    out.append(check("opening arrears = last month's closing arrears for the same tenant",
                     (df.opening_arrears[same_tenant] == df.closing_arrears.shift()[same_tenant]).all()))
    fresh = ~same_tenant & (df.month > df.month.min())
    out.append(check("a new tenant starts with zero arrears", (df.opening_arrears[fresh] == 0).all()))
    fee_ok = df.monthly_rent.map(late_fee_for)
    out.append(check("late_fee is 0 or 5% of rent (to the nearest 5, minimum 10)",
                     ((df.late_fee == 0) | (df.late_fee == fee_ok)).all()))
    out.append(check("deposit_amount = 1.2 x monthly_rent (rounded)",
                     ((df.deposit_amount - DEPOSIT_MULTIPLE * df.monthly_rent).abs() <= 0.5).all()))
    out.append(check("lease_end is after lease_start", (df.lease_end > df.lease_start).all()))
    out.append(check("the month sits inside the lease",
                     ((df.month >= df.lease_start) & (month_end <= df.lease_end)).all()))
    out.append(check("move_in_date is at most 6 days after lease_start on a first lease (earlier for renewals)",
                     (df.move_in_date <= df.lease_start + pd.Timedelta(days=6)).all()))
    out.append(check("moved in before the end of the month", (df.move_in_date <= month_end).all()))
    out.append(check("month is the first day of a month, not after the snapshot",
                     ((df.month.dt.day == 1) & (df.month <= snapshot)).all()))
    out.append(check("occupancy_rate = occupied units / units in the building",
                     ((df.occupancy_rate - df.occupied_units / df.units_in_building).abs() <= 0.0051).all()
                     and (df.occupancy_rate > 0).all() and (df.occupancy_rate <= 1).all()))
    complete = df[df.building_id != df.building_id.iloc[-1]]        # the last building may be cut off
    counted = complete.groupby(["building_id", "month"]).unit_id.transform("count")
    out.append(check("occupied units = number of ledger rows for that building and month",
                     (counted == complete.occupied_units).all()))
    out.append(check("one row per unit and month", not df.duplicated(["unit_id", "month"]).any()))
    out.append(check("tenant_count between 1 and bedrooms + 1",
                     ((df.tenant_count >= 1) & (df.tenant_count <= df.bedrooms + 1)).all()))
    out.append(check("monthly_rent > 0 and a multiple of 5",
                     ((df.monthly_rent > 0) & (df.monthly_rent % 5 == 0)).all()))
    out.append(check("bedrooms fixed per unit", df.groupby("unit_id").bedrooms.nunique().max() == 1))

    # ---- relationships (tendencies) ----
    log_rent = np.log(df.monthly_rent)
    rent_dev = log_rent - log_rent.groupby(df.area).transform("mean")
    bed_dev = df.bedrooms - df.bedrooms.groupby(df.area).transform("mean")
    corr = float(rent_dev.corr(bed_dev))
    out.append(check("within an area, more bedrooms means higher rent",
                     corr >= 0.6, f"corr={corr:.2f}", "relationship"))

    # Compare areas after allowing for unit size (a few large units can distort raw medians).
    size_dev = log_rent - log_rent.groupby(df.bedrooms).transform("mean")
    by_area = size_dev.groupby(df.area).agg(["mean", "count"])
    if {"central", "outer"} <= set(by_area.index) and by_area["count"][["central", "outer"]].min() >= 10:
        diff = float(by_area["mean"]["central"] - by_area["mean"]["outer"])
        out.append(check("for the same number of bedrooms, central rents exceed outer rents",
                         diff >= 0.3, f"log gap={diff:.2f}", "relationship"))

    c2 = float(df.bedrooms.corr(df.tenant_count))
    out.append(check("bigger units have more tenants",
                     c2 >= 0.25, f"corr={c2:.2f}", "relationship"))

    late = df.late_fee > 0
    in_arrears = df.opening_arrears > 0
    if in_arrears.sum() >= 10 and (~in_arrears).sum() >= 10:
        out.append(check("tenants already in arrears pay late more often",
                         late[in_arrears].mean() > late[~in_arrears].mean(), "", "relationship"))

    first_months = df.month <= df.month.min() + pd.DateOffset(months=3)
    last_months = df.month >= df.month.max() - pd.DateOffset(months=3)
    if first_months.sum() >= 10 and last_months.sum() >= 10:
        gap = float(rent_dev[last_months].mean() - rent_dev[first_months].mean())
        out.append(check("rents rise over the year (same area mix, later months dearer)",
                         gap > 0.0, f"gap={gap:.3f}", "relationship"))
    return out


RECIPE = Recipe(
    key="real_estate_rentals",
    title="Rental Property Management",
    row_meaning=("One row is one occupied rental unit in one calendar month "
                 "(a rent ledger line); empty months have no row."),
    simulate=simulate,
    columns=["unit_id", "month", "area", "bedrooms", "is_pet_friendly", "tenant_count",
             "monthly_rent", "deposit_amount", "lease_start", "lease_end", "move_in_date",
             "amount_paid", "late_fee", "closing_arrears", "occupancy_rate"],
    core=["area", "bedrooms", "monthly_rent", "late_fee"],
    priority=["month", "unit_id", "tenant_count", "amount_paid", "closing_arrears",
              "deposit_amount", "occupancy_rate", "lease_start", "lease_end",
              "move_in_date", "is_pet_friendly"],
    docs={
        "unit_id": "Identifier of the rental unit; the same unit appears in many months.",
        "month": "First day of the calendar month this ledger line covers.",
        "area": "Location tier: central, suburban or outer (sets the rent level).",
        "bedrooms": "Number of bedrooms in the unit.",
        "is_pet_friendly": "True if the building allows pets (units there rent for slightly more).",
        "tenant_count": "Number of people named on the tenancy.",
        "monthly_rent": "Rent charged for the month, whole units; fixed for the length of each lease.",
        "deposit_amount": "Security deposit held for the lease: 1.2 x monthly_rent, whole units.",
        "lease_start": "First day of the current lease term.",
        "lease_end": "Last day of the current lease term.",
        "move_in_date": "Date the tenant first moved in (earlier than lease_start if the lease is a renewal).",
        "amount_paid": "Money received from the tenant in this month, whole units.",
        "late_fee": "Late fee charged this month: 0 if the rent was on time, otherwise 5% of rent.",
        "closing_arrears": "Unpaid balance at month end: opening arrears + monthly_rent + late_fee - amount_paid.",
        "occupancy_rate": "Share of the building's units that were let this month (0 to 1), same for all units in the building.",
    },
    validate=validate,
    date_cols=["month", "lease_start", "lease_end", "move_in_date"],
    targets={
        "late_fee": ["closing_arrears", "amount_paid"],
        "closing_arrears": ["amount_paid", "late_fee"],
        "monthly_rent": ["deposit_amount"],
    },
)
