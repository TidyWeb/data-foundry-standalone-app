"""Property Listings.

One row is one property listing that went under contract (an offer was
accepted) during the last 24 months.  listing_price is the asking price.

The story, in the order the code tells it:
    area -> property type -> bedrooms -> size -> bathrooms -> age & condition
    -> market value -> foreclosure? -> asking price -> days on market -> dates

Price is set from the property's characteristics.  Time on market then
responds to how ambitious the asking price was.  Nothing flows backwards.
All numbers are illustrative teaching values, not real market data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli, ordinal, add_days

# name: (typical price per sq ft, tier 0 = priciest ... 3 = cheapest, demand effect)
# demand effect < 0 means properties there sell faster.
AREAS = {
    "Riverside": (480, 0, -0.10),
    "Hillcrest": (440, 0, -0.15),
    "Old Town":  (380, 1, -0.05),
    "Parkside":  (340, 1, 0.00),
    "Millbrook": (290, 2, 0.05),
    "Eastfield": (260, 2, 0.10),
    "Northgate": (220, 3, 0.15),
    "Brookvale": (190, 3, 0.20),
}

TYPES = ["flat", "terraced", "semi-detached", "detached"]
# share of each type within each price tier
TYPE_MIX = {
    0: [0.12, 0.18, 0.30, 0.40],
    1: [0.18, 0.25, 0.32, 0.25],
    2: [0.28, 0.32, 0.27, 0.13],
    3: [0.40, 0.35, 0.20, 0.05],
}
TYPE_MEDIAN_SQFT = {"flat": 620, "terraced": 950, "semi-detached": 1200, "detached": 1900}
TYPE_TYPICAL_BEDS = {"flat": 2, "terraced": 3, "semi-detached": 3, "detached": 4}
TYPE_PRICE_FACTOR = {"flat": 0.95, "terraced": 1.00, "semi-detached": 1.05, "detached": 1.12}
BEDROOM_MIX = {                       # bedrooms -> share, within each type
    "flat":          ([1, 2, 3], [0.35, 0.50, 0.15]),
    "terraced":      ([2, 3, 4], [0.30, 0.50, 0.20]),
    "semi-detached": ([2, 3, 4, 5], [0.10, 0.50, 0.30, 0.10]),
    "detached":      ([3, 4, 5, 6], [0.15, 0.45, 0.30, 0.10]),
}
CONDITIONS = ["poor", "fair", "good", "excellent"]
CONDITION_PRICE_FACTOR = {"poor": 0.82, "fair": 0.93, "good": 1.00, "excellent": 1.10}
CONDITION_DOM_EFFECT = {"poor": 0.25, "fair": 0.08, "good": 0.0, "excellent": -0.10}

SIZE_ELASTICITY = 0.90        # price grows a little slower than size
EXTRA_BEDROOM_SIZE = 0.20     # each extra bedroom adds ~20% floor area
MEDIAN_DAYS = 40              # typical days on market for a fairly priced home
WINDOW_DAYS = 730             # contract dates fall in the last 24 months


def simulate(rng, n, snapshot):
    # 1. Area, then property type given the area's tier.
    area = pick(rng, list(AREAS), [1, 1, 1.2, 1.2, 1.2, 1, 1, 1], n)
    ppsf = np.array([AREAS[a][0] for a in area], dtype=float)
    tier = np.array([AREAS[a][1] for a in area])
    demand = np.array([AREAS[a][2] for a in area])

    ptype = np.empty(n, dtype=object)
    for t in range(4):
        rows = tier == t
        ptype[rows] = pick(rng, TYPES, TYPE_MIX[t], int(rows.sum()))

    # 2. Bedrooms given the type.
    bedrooms = np.zeros(n, dtype=int)
    for name, (beds, weights) in BEDROOM_MIX.items():
        rows = ptype == name
        bedrooms[rows] = pick(rng, beds, weights, int(rows.sum())).astype(int)

    # 3. Size given type and bedrooms: more bedrooms -> bigger, with overlap.
    typical_sqft = np.array([TYPE_MEDIAN_SQFT[t] for t in ptype], dtype=float)
    typical_beds = np.array([TYPE_TYPICAL_BEDS[t] for t in ptype])
    log_size = (np.log(typical_sqft) + EXTRA_BEDROOM_SIZE * (bedrooms - typical_beds)
                + rng.normal(0, 0.15, n))
    sqft = np.round(np.exp(log_size) / 5) * 5
    sqft = np.maximum(sqft, 250 + 100 * bedrooms).astype(int)   # every bedroom needs room

    # 4. Bathrooms given bedrooms and size (at least 1, at most bedrooms + 1).
    size_ratio = sqft / typical_sqft
    bathrooms = 1 + rng.poisson(0.30 * (bedrooms - 1) + 0.5 * np.maximum(size_ratio - 1, 0))
    bathrooms = np.clip(bathrooms, 1, bedrooms + 1).astype(int)

    # 5. Age, then condition (older tends to be poorer, with plenty of noise).
    age = np.clip(np.round(lognormal(rng, 45, 0.6, n)), 1, 150).astype(int)
    year_built = snapshot.year - age
    condition_score = 2.3 - 0.012 * age + rng.normal(0, 0.7, n)
    condition = ordinal(condition_score, [0.9, 1.7, 2.5], CONDITIONS)

    # 6. Market value: what a typical buyer would pay before any listing strategy.
    #    Area price level x size (slightly less than proportional) x type,
    #    condition and bathroom adjustments x unobserved factors (plot, view, finish).
    type_factor = np.array([TYPE_PRICE_FACTOR[t] for t in ptype])
    cond_factor = np.array([CONDITION_PRICE_FACTOR[c] for c in condition])
    bath_factor = 1 + 0.04 * (bathrooms - 1)
    unobserved = np.exp(rng.normal(0, 0.12, n))
    market_value = (ppsf * sqft * (sqft / 1000) ** (SIZE_ELASTICITY - 1)
                    * type_factor * cond_factor * bath_factor * unobserved)

    # 7. Foreclosure: more likely in cheaper areas and poorer condition,
    #    and asked at a 5-20% discount.
    p_foreclosure = (0.02 + 0.01 * tier + 0.06 * (condition == "poor")
                     + 0.02 * (condition == "fair"))
    is_foreclosure = bernoulli(rng, p_foreclosure)
    discount = np.where(is_foreclosure, rng.uniform(0.05, 0.20, n), 0.0)

    # 8. Asking price: value, less any foreclosure discount, times the agent's
    #    pricing error (a little high or low), rounded to the nearest 1,000.
    pricing_error = lognormal(rng, 1.02, 0.06, n)
    asking = market_value * (1 - discount) * pricing_error
    listing_price = (np.round(asking / 1000) * 1000).astype(int)
    listing_price = np.maximum(listing_price, 1000)
    price_per_sqft = np.round(listing_price / sqft).astype(int)

    # 9. Contract date first, then days on market, then the listing date.
    #    Days on market is longer when the asking price is ambitious, the
    #    condition is poor, the area is slow, or the season is winter.
    window_start = snapshot - pd.Timedelta(days=WINDOW_DAYS)
    contract_date = window_start + pd.to_timedelta(rng.integers(0, WINDOW_DAYS + 1, n), unit="D")
    month = contract_date.month.to_numpy()
    season = np.where(np.isin(month, [3, 4, 5, 6]), -0.15,
                      np.where(np.isin(month, [11, 12, 1, 2]), 0.15, 0.0))
    asking_premium = asking / market_value
    log_median = (np.log(MEDIAN_DAYS) + 4.0 * np.log(asking_premium)
                  + np.array([CONDITION_DOM_EFFECT[c] for c in condition])
                  + demand + season)
    days_on_market = np.maximum(1, np.ceil(rng.lognormal(log_median, 0.6))).astype(int)
    listing_date = pd.Series(contract_date) - pd.to_timedelta(days_on_market, unit="D")

    df = pd.DataFrame({
        "location_area": area,
        "property_type": ptype,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "square_footage": sqft,
        "year_built": year_built,
        "condition": condition,
        "is_foreclosure": is_foreclosure,
        "listing_date": listing_date,
        "listing_price": listing_price,
        "price_per_sqft": price_per_sqft,
        "days_on_market": days_on_market,
        # hidden columns (kept for validation, not shown to students)
        "contract_date": pd.Series(contract_date),
        "market_value": np.round(market_value).astype(int),
        "asking_premium": np.round(asking_premium, 4),
    })
    df = df.sort_values("listing_date").reset_index(drop=True)
    df.insert(0, "property_id", [f"P{i + 1:04d}" for i in range(len(df))])
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _within_group(df, value_col, by):
    return df[value_col] - df.groupby(by)[value_col].transform("mean")


def validate(df, snapshot):
    out = []
    n = len(df)

    # ---- exact rules ----
    out.append(check("price_per_sqft = listing_price / square_footage (+-0.5)",
                     ((df.price_per_sqft - df.listing_price / df.square_footage).abs() <= 0.5).all()))
    out.append(check("listing_date + days_on_market = contract_date",
                     (pd.to_datetime(df.listing_date) + pd.to_timedelta(df.days_on_market, unit="D")
                      == pd.to_datetime(df.contract_date)).all()))
    window_start = snapshot - pd.Timedelta(days=WINDOW_DAYS)
    out.append(check("contract_date inside the 24-month window",
                     (pd.to_datetime(df.contract_date).between(window_start, snapshot)).all()))
    out.append(check("days_on_market is an integer >= 1", (df.days_on_market >= 1).all()))
    out.append(check("bedrooms >= 1", (df.bedrooms >= 1).all()))
    out.append(check("1 <= bathrooms <= bedrooms + 1",
                     ((df.bathrooms >= 1) & (df.bathrooms <= df.bedrooms + 1)).all()))
    out.append(check("square_footage >= 250 + 100 x bedrooms",
                     (df.square_footage >= 250 + 100 * df.bedrooms).all()))
    out.append(check("listing_price > 0", (df.listing_price > 0).all()))
    out.append(check("categories valid",
                     df.property_type.isin(TYPES).all() and df.condition.isin(CONDITIONS).all()
                     and df.location_area.isin(AREAS).all()))
    out.append(check("property_id unique", df.property_id.is_unique))
    out.append(check("year_built not in the future", (df.year_built <= snapshot.year).all()))

    # ---- relationships (tendencies) ----
    by = [df.location_area, df.property_type]
    lp = np.log(df.listing_price).rename("lp")
    ls = np.log(df.square_footage).rename("ls")
    tmp = pd.DataFrame({"lp": lp, "ls": ls, "a": df.location_area, "t": df.property_type})
    lp_d = _within_group(tmp, "lp", ["a", "t"])
    ls_d = _within_group(tmp, "ls", ["a", "t"])
    denom = float((ls_d ** 2).sum())
    slope = float((lp_d * ls_d).sum() / denom) if denom > 0 else float("nan")
    corr = float(lp_d.corr(ls_d))
    out.append(check("size elasticity of price near 0.9 within area and type",
                     0.6 <= slope <= 1.3, f"slope={slope:.2f}", "relationship"))
    out.append(check("within area and type, bigger tends to cost more",
                     corr >= 0.5, f"corr={corr:.2f}", "relationship"))

    # Some bigger houses must still be cheaper than smaller ones (comparables differ).
    inversions, pairs = 0, 0
    for _, g in df.groupby(["location_area", "property_type"]):
        s = g.square_footage.to_numpy()
        p = g.listing_price.to_numpy()
        bigger = s[:, None] > s[None, :]
        pairs += int(bigger.sum())
        inversions += int((bigger & (p[:, None] < p[None, :])).sum())
    share = inversions / pairs if pairs else float("nan")
    out.append(check("some (not most) larger houses are cheaper than smaller ones nearby",
                     0.03 <= share <= 0.45, f"share={share:.2f}", "relationship"))

    ppsf_by_cond = df.groupby("condition").price_per_sqft.median()
    if {"poor", "excellent"} <= set(ppsf_by_cond.index) and \
            (df.condition == "poor").sum() >= 8 and (df.condition == "excellent").sum() >= 8:
        out.append(check("median price per sq ft: excellent > poor",
                         ppsf_by_cond["excellent"] > ppsf_by_cond["poor"], "", "relationship"))

    fc = df.is_foreclosure
    if fc.sum() >= 4:
        gap = float(np.log(df.asking_premium[~fc]).mean() - np.log(df.asking_premium[fc]).mean())
        out.append(check("foreclosures are asked below market value",
                         gap >= 0.05, f"log-gap={gap:.2f}", "relationship"))

    rho = float(df.asking_premium.rank().corr(df.days_on_market.rank()))
    out.append(check("higher asking premium -> longer days on market",
                     rho > 0.0, f"rank corr={rho:.2f}", "relationship"))
    return out


RECIPE = Recipe(
    key="real_estate_listings",
    title="Property Listings",
    row_meaning=("One row is one property listing that went under contract "
                 "(an offer was accepted) in the last 24 months."),
    simulate=simulate,
    columns=["property_id", "location_area", "property_type", "bedrooms", "bathrooms",
             "square_footage", "year_built", "condition", "is_foreclosure",
             "listing_date", "listing_price", "price_per_sqft", "days_on_market"],
    core=["location_area", "bedrooms", "square_footage", "listing_price"],
    priority=["property_type", "bathrooms", "condition", "price_per_sqft", "listing_date",
              "days_on_market", "is_foreclosure", "property_id", "year_built"],
    docs={
        "property_id": "Unique listing identifier.",
        "location_area": "Fictional neighbourhood. Each area has its own typical price level.",
        "property_type": "flat, terraced, semi-detached or detached.",
        "bedrooms": "Number of bedrooms.",
        "bathrooms": "Number of bathrooms (at least 1, at most bedrooms + 1).",
        "square_footage": "Floor area in square feet.",
        "year_built": "Year the property was built.",
        "condition": "poor, fair, good or excellent (ordered).",
        "is_foreclosure": "True if the property was a forced-sale (repossession) listing; these are asked below market value.",
        "listing_date": "Date the property was listed for sale.",
        "listing_price": "Asking price when listed (rounded to the nearest 1,000).",
        "price_per_sqft": "listing_price divided by square_footage, rounded to a whole number.",
        "days_on_market": "Days from listing_date until an offer was accepted.",
    },
    validate=validate,
    date_cols=["listing_date"],
    targets={
        "listing_price": ["price_per_sqft", "days_on_market"],
        "days_on_market": [],
        "is_foreclosure": [],
    },
)
