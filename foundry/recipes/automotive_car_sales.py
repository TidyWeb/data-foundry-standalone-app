"""Car Sales.

One row is one completed used-car sale by a dealer (one vehicle per row).
All prices are plain numbers; mileage is in miles; engine_size is in litres.

The story, in the order the code tells it:
    sale date -> segment -> engine & power -> age & model year -> yearly usage
    -> mileage -> condition -> certified? -> market value -> list price
    -> days on the lot -> discount -> negotiated price -> financed?

A car's market value comes from what it is (segment, age, mileage, condition,
power).  The dealer lists it a little above that value.  The more ambitious
the list price, the longer the car sits, and the longer it sits the bigger
the discount the dealer accepts.  Nothing flows backwards.
All numbers are illustrative teaching values, not real market data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli, logistic, ordinal

# segment: (new price, typical litres, hp per litre, median miles per year,
#           demand effect on days on lot (negative = sells faster),
#           engine size band, horsepower band)
SEGMENTS = {
    "city":      (16000, 1.2, 68, 6500, -0.05, (1.0, 1.4), (55, 110)),
    "family":    (24000, 1.6, 78, 8500, -0.10, (1.4, 2.0), (80, 180)),
    "suv":       (36000, 2.0, 80, 9500, -0.10, (1.6, 3.0), (110, 300)),
    "executive": (45000, 2.5, 90, 11000, 0.10, (2.0, 3.5), (150, 350)),
    "sports":    (55000, 3.0, 105, 5500, 0.15, (2.5, 5.0), (200, 600)),
}
SEGMENT_WEIGHTS = [0.24, 0.30, 0.24, 0.14, 0.08]
CONDITIONS = ["poor", "fair", "good", "excellent"]
CONDITION_VALUE_FACTOR = {"poor": 0.84, "fair": 0.93, "good": 1.00, "excellent": 1.07}
CONDITION_LOT_EFFECT = {"poor": 0.15, "fair": 0.05, "good": 0.0, "excellent": -0.05}

AGE_DEPRECIATION = 0.075      # each year of age removes about 7.5% of value
MILE_DEPRECIATION = 0.000004  # each mile removes about 0.0004% of value
MEDIAN_DAYS_ON_LOT = 35
WINDOW_DAYS = 730             # sales happened in the last 24 months


def simulate(rng, n, snapshot):
    # 1. Sale date first (like a contract date); the listing date comes later.
    offsets = rng.integers(0, WINDOW_DAYS + 1, n)
    sale_date = snapshot - pd.to_timedelta(offsets, unit="D")

    # 2. Segment, then engine size and horsepower inside that segment's bands.
    segment = pick(rng, list(SEGMENTS), SEGMENT_WEIGHTS, n)
    new_price = np.array([SEGMENTS[s][0] for s in segment], dtype=float)
    litres = np.array([SEGMENTS[s][1] for s in segment])
    hp_per_litre = np.array([SEGMENTS[s][2] for s in segment])
    usage_median = np.array([SEGMENTS[s][3] for s in segment], dtype=float)
    demand = np.array([SEGMENTS[s][4] for s in segment])
    engine_low = np.array([SEGMENTS[s][5][0] for s in segment])
    engine_high = np.array([SEGMENTS[s][5][1] for s in segment])
    hp_low = np.array([SEGMENTS[s][6][0] for s in segment])
    hp_high = np.array([SEGMENTS[s][6][1] for s in segment])

    engine_size = np.round(np.clip(rng.normal(litres, 0.25), engine_low, engine_high), 1)
    typical_hp = engine_size * hp_per_litre
    horsepower = np.round(typical_hp * lognormal(rng, 1.0, 0.10, n) / 5) * 5
    horsepower = np.clip(horsepower, hp_low, hp_high).astype(int)

    # 3. Age (about 6% are nearly-new ex-demonstrators), then model year.
    age = np.clip(np.round(lognormal(rng, 4.5, 0.6, n)), 1, 15).astype(int)
    age = np.where(bernoulli(rng, np.full(n, 0.06)), 0, age)
    model_year = sale_date.year.to_numpy() - age

    # 4. Mileage = age x this car's yearly usage x a little noise.
    #    Usage is a per-car trait: some owners drive far more than others.
    usage = lognormal(rng, usage_median, 0.35, n)
    mileage = np.round(age * usage * lognormal(rng, 1.0, 0.08, n)).astype(int)
    demo_miles = rng.integers(5, 400, n)
    mileage = np.where(age == 0, demo_miles, mileage)

    # 5. Condition: older, higher-mileage cars tend to be in poorer shape,
    #    but there are plenty of exceptions.
    condition_score = 2.6 - 0.06 * age - 0.00001 * mileage + rng.normal(0, 0.7, n)
    condition = ordinal(condition_score, [0.9, 1.7, 2.5], CONDITIONS)

    # 6. Certified (dealer-approved) needs a young, low-mileage, tidy car,
    #    and even then only some dealers' cars go through the programme.
    eligible = (age <= 6) & (mileage < 70000) & np.isin(condition, ["good", "excellent"])
    is_certified = eligible & bernoulli(rng, np.full(n, 0.55))

    # 7. Market value: new price less age and mileage wear, times condition,
    #    power and unobserved factors (colour, options, service history).
    cond_factor = np.array([CONDITION_VALUE_FACTOR[c] for c in condition])
    power_factor = (horsepower / typical_hp) ** 0.5
    unobserved = np.exp(rng.normal(0, 0.08, n))
    market_value = (new_price * np.exp(-AGE_DEPRECIATION * age - MILE_DEPRECIATION * mileage)
                    * cond_factor * power_factor * unobserved)
    market_value = np.maximum(market_value, 1000)

    # 8. List price: the dealer asks a little above market value
    #    (certified cars carry a premium).  Rounded to the nearest 50.
    list_premium = lognormal(rng, 1.06, 0.05, n) + 0.03 * is_certified
    list_price = (np.round(market_value * list_premium / 50) * 50).astype(int)
    list_premium = list_price / market_value

    # 9. Days on the lot: longer when overpriced, in poor condition or in a
    #    slow segment; certified cars move a bit faster.
    log_median = (np.log(MEDIAN_DAYS_ON_LOT) + 4.0 * np.log(list_premium) + demand
                  + np.array([CONDITION_LOT_EFFECT[c] for c in condition])
                  - 0.10 * is_certified)
    days_on_lot = np.clip(np.ceil(rng.lognormal(log_median, 0.6)), 1, 400).astype(int)
    listing_date = pd.Series(sale_date) - pd.to_timedelta(days_on_lot, unit="D")

    # 10. Discount: buyers push back on high prices, and the dealer gives
    #     way as the car ages on the lot.  Sale price follows exactly.
    discount = 0.02 + 0.6 * (list_premium - 1) + 0.0004 * days_on_lot + rng.normal(0, 0.02, n)
    discount_pct = np.round(np.clip(discount, 0.0, 0.30), 4)
    negotiated_price = np.round(list_price * (1 - discount_pct)).astype(int)

    # 11. Financed: pricier cars are more often bought on finance.
    is_financed = bernoulli(rng, logistic(-0.4 + (negotiated_price - 20000) / 25000))

    df = pd.DataFrame({
        "segment": segment,
        "model_year": model_year,
        "mileage": mileage,
        "condition": condition,
        "engine_size": engine_size,
        "horsepower": horsepower,
        "is_certified": is_certified,
        "is_financed": is_financed,
        "listing_date": listing_date,
        "days_on_lot": days_on_lot,
        "sale_date": pd.Series(sale_date),
        "list_price": list_price,
        "negotiated_price": negotiated_price,
        # hidden columns (kept for validation, not shown to students)
        "age": age,
        "market_value": np.round(market_value).astype(int),
        "list_premium": np.round(list_premium, 4),
        "discount_pct": discount_pct,
    })
    df = df.sort_values("sale_date", kind="stable").reset_index(drop=True)
    df.insert(0, "vehicle_id", [f"V{i + 1:04d}" for i in range(len(df))])
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _within_segment_age(df, column):
    """A column minus the average of cars with the same segment and age."""
    return df[column] - df.groupby(["segment", "age"])[column].transform("mean")


def validate(df, snapshot):
    out = []

    # ---- exact rules ----
    out.append(check("negotiated_price = list_price x (1 - discount) (+-0.5)",
                     ((df.negotiated_price - df.list_price * (1 - df.discount_pct)).abs() <= 0.5).all()))
    out.append(check("listing_date + days_on_lot = sale_date",
                     (pd.to_datetime(df.listing_date) + pd.to_timedelta(df.days_on_lot, unit="D")
                      == pd.to_datetime(df.sale_date)).all()))
    window_start = snapshot - pd.Timedelta(days=WINDOW_DAYS)
    out.append(check("sale_date inside the 24-month window",
                     pd.to_datetime(df.sale_date).between(window_start, snapshot).all()))
    out.append(check("age = sale year - model_year, and is >= 0",
                     ((pd.to_datetime(df.sale_date).dt.year - df.model_year == df.age)
                      & (df.age >= 0)).all()))
    out.append(check("mileage >= 0", (df.mileage >= 0).all()))
    out.append(check("nearly-new (age 0) cars have under 400 miles",
                     (df.mileage[df.age == 0] < 400).all()))
    out.append(check("days_on_lot is an integer >= 1", (df.days_on_lot >= 1).all()))
    out.append(check("0 <= discount <= 30%", df.discount_pct.between(0, 0.30).all()))
    out.append(check("negotiated_price <= list_price", (df.negotiated_price <= df.list_price).all()))
    out.append(check("prices > 0", ((df.list_price > 0) & (df.negotiated_price > 0)).all()))
    in_band = True
    for name, spec in SEGMENTS.items():
        rows = df[df.segment == name]
        in_band &= rows.engine_size.between(*spec[5]).all() and rows.horsepower.between(*spec[6]).all()
    out.append(check("engine_size and horsepower stay inside the segment bands", in_band))
    out.append(check("certified cars: age <= 6, < 70,000 miles, good or excellent condition",
                     ((df.age <= 6) & (df.mileage < 70000) & df.condition.isin(["good", "excellent"]))[df.is_certified].all()))
    out.append(check("categories valid",
                     df.segment.isin(SEGMENTS).all() and df.condition.isin(CONDITIONS).all()))
    out.append(check("vehicle_id unique", df.vehicle_id.is_unique))

    # ---- relationships (tendencies) ----
    tmp = pd.DataFrame({"lp": np.log(df.list_price), "m": df.mileage.astype(float),
                        "age": df.age.astype(float), "segment": df.segment})
    lp_seg = tmp.lp - tmp.groupby("segment").lp.transform("mean")
    corr_age = float(lp_seg.corr(tmp.age))
    out.append(check("within a segment, older cars have lower list prices",
                     corr_age <= -0.7, f"corr={corr_age:.2f}", "relationship"))

    used = df[df.age >= 1]
    corr_am = float(used.age.corr(used.mileage))
    out.append(check("older cars have higher mileage (age and mileage move together)",
                     corr_am >= 0.55, f"corr={corr_am:.2f}", "relationship"))

    # Same segment and age: does extra mileage still cost value?
    corr_m = float(_within_segment_age(tmp, "lp").corr(_within_segment_age(tmp, "m")))
    out.append(check("for the same segment and age, higher mileage means a lower price",
                     corr_m <= -0.2, f"corr={corr_m:.2f}", "relationship"))

    med = df.groupby("segment").list_price.median()
    if all((df.segment == s).sum() >= 8 for s in ["city", "family", "executive"]):
        out.append(check("median list price: city < family < executive",
                         med["city"] < med["family"] < med["executive"], "", "relationship"))

    rho = float(df.list_premium.rank().corr(df.days_on_lot.rank()))
    out.append(check("higher list premium over market value -> longer days on lot",
                     rho > 0.10, f"rank corr={rho:.2f}", "relationship"))

    rho2 = float(df.days_on_lot.rank().corr(df.discount_pct.rank()))
    out.append(check("longer on the lot -> bigger discount",
                     rho2 > 0.20, f"rank corr={rho2:.2f}", "relationship"))

    if df.is_financed.sum() >= 8 and (~df.is_financed).sum() >= 8:
        out.append(check("financed cars are dearer on average",
                         df.negotiated_price[df.is_financed].mean() > df.negotiated_price[~df.is_financed].mean(),
                         "", "relationship"))
    return out


RECIPE = Recipe(
    key="automotive_car_sales",
    title="Car Sales",
    row_meaning=("One row is one completed used-car sale by a dealer "
                 "(one vehicle per row) in the last 24 months."),
    simulate=simulate,
    columns=["vehicle_id", "segment", "model_year", "mileage", "condition", "engine_size",
             "horsepower", "is_certified", "listing_date", "days_on_lot", "sale_date",
             "list_price", "negotiated_price", "is_financed"],
    core=["segment", "model_year", "mileage", "negotiated_price"],
    priority=["condition", "list_price", "engine_size", "horsepower", "days_on_lot",
              "is_certified", "sale_date", "is_financed", "listing_date", "vehicle_id"],
    docs={
        "vehicle_id": "Unique identifier for the vehicle sold.",
        "segment": "Body style / market segment: city, family, suv, executive or sports.",
        "model_year": "Year the car was made (age at sale = sale year minus model_year).",
        "mileage": "Odometer reading at sale, in miles.",
        "condition": "poor, fair, good or excellent (ordered).",
        "engine_size": "Engine capacity in litres.",
        "horsepower": "Engine power in horsepower (hp).",
        "is_certified": "True if the dealer certified the car (young, low-mileage, good condition only).",
        "listing_date": "Date the car was put on the dealer's lot.",
        "days_on_lot": "Days from listing_date until the sale.",
        "sale_date": "Date the car was sold (listing_date + days_on_lot).",
        "list_price": "Dealer's asking price, rounded to the nearest 50.",
        "negotiated_price": "Price actually paid after the discount, whole units.",
        "is_financed": "True if the buyer paid using finance.",
    },
    validate=validate,
    date_cols=["listing_date", "sale_date"],
    targets={
        "negotiated_price": ["list_price"],
        "list_price": ["negotiated_price"],
        "is_financed": ["negotiated_price"],
    },
)
