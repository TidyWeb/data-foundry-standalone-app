"""Shipping & Delivery.

One row is one shipment that has been DELIVERED (shipments still in transit
today are not included, so delivery_date is always known).  Distance is in
kilometres, weight in kilograms, freight_cost in plain currency units, and
all times are in whole days.

The story, in the order the code tells it:
    dispatch hub + destination -> distance -> service level (economy /
    standard / express) -> weight -> order date -> handling days -> ship date
    -> promised date -> disruption -> delivery date -> handling touches
    -> damage -> insurance -> freight cost

Each service moves parcels at a typical number of km per day, and the
promise is that speed plus a safety margin.  Most shipments arrive on time;
a rare disruption (worse in the November-December peak) adds a right-skewed
extra delay.  Cost is a formula (see FREIGHT COST below) times a fuel index.
Damage is more likely with more handling touches.  Teaching values only.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, pick, bernoulli, logistic

# name: (x, y) position on a fictional 900 x 1100 km map
HUBS = {"northfield": (300, 900), "midlands": (400, 500), "southport": (250, 120)}
CITIES = {
    "ashford": (620, 260), "brindle": (150, 700), "carrow": (500, 950), "dunmere": (700, 560),
    "elstow": (330, 620), "fenwick": (80, 300), "garrow": (820, 900), "holcombe": (450, 380),
    "ivybridge": (200, 150), "jarrow": (560, 1040), "kettleby": (720, 380),
    "lowmoor": (380, 780), "marlow": (140, 480), "netherby": (650, 720),
}
# service: (km per day the courier network covers, price multiplier)
SERVICES = {"economy": (300, 0.8), "standard": (500, 1.0), "express": (800, 1.9)}
PROMISE_MARGIN = 1.1          # promises allow 10% more travel time than the typical speed needs
WINDOW_DAYS = 360             # orders placed over the last 12 months
PEAK_MONTHS = [11, 12]        # Black Friday / Christmas


def simulate_batch(rng, m, snapshot):
    # 1. Route: which hub sends it, where it goes, and the road distance in km.
    hub = pick(rng, list(HUBS), [0.4, 0.35, 0.25], m)
    city = pick(rng, list(CITIES), np.ones(len(CITIES)), m)
    hub_xy = np.array([HUBS[h] for h in hub], dtype=float)
    city_xy = np.array([CITIES[c] for c in city], dtype=float)
    straight = np.sqrt(((hub_xy - city_xy) ** 2).sum(axis=1))
    distance = np.maximum(20, np.round(straight * 1.2)).astype(int)     # roads are not straight

    # 2. Weight (kg, right-skewed), then service level (heavier parcels lean economy).
    weight = np.clip(np.round(lognormal(rng, 18, 0.9, m), 1), 0.5, 800)
    express = bernoulli(rng, np.full(m, 0.15))
    economy = bernoulli(rng, logistic(0.1 + 0.5 * np.log(weight / 18)))
    service = np.where(express, "express", np.where(economy, "economy", "standard"))
    speed = np.array([SERVICES[s][0] for s in service], dtype=float)

    # 3. Dates: order, then handling days at the hub, then ship, then the promise.
    order_date = (snapshot - pd.Timedelta(days=WINDOW_DAYS)
                  + pd.to_timedelta(rng.integers(0, WINDOW_DAYS, m), unit="D"))
    handling = np.where(service == "economy", 1 + rng.poisson(1.2, m),
                        np.where(service == "standard", 1 + rng.poisson(0.5, m), rng.poisson(0.4, m)))
    ship_date = order_date + pd.to_timedelta(handling, unit="D")
    promised_days = 1 + np.ceil(distance / speed * PROMISE_MARGIN).astype(int)
    promised_date = ship_date + pd.to_timedelta(promised_days, unit="D")

    # 4. Transit: the network's speed with a little natural variation, plus a
    #    rare disruption (weather, strikes, customs) that adds days.
    transit = 1 + np.ceil(distance / speed * lognormal(rng, 1.0, 0.10, m)).astype(int)
    peak = np.isin(order_date.month, PEAK_MONTHS)
    disrupted = bernoulli(rng, np.where(peak, 0.25, 0.06))
    extra_days = np.ceil(lognormal(rng, 1.5, 0.8, m)).astype(int)
    transit = transit + disrupted * extra_days
    delivery_date = ship_date + pd.to_timedelta(transit, unit="D")

    # 5. Handling touches (more for long routes and heavy parcels) -> damage.
    touches = 2 + (distance > 400) + (distance > 900) + (weight > 100) + rng.poisson(0.5, m)
    is_damaged = bernoulli(rng, logistic(-4.4 + 0.35 * touches))

    # 6. Insurance: valuable goods are usually insured.  goods_value is hidden.
    goods_value = lognormal(rng, 250 * (weight / 18) ** 0.5, 0.9)
    is_insured = bernoulli(rng, logistic(-1.2 + 1.0 * np.log(goods_value / 250)))

    # FREIGHT COST = service multiplier x (6 + 0.02 x km + 0.60 x kg) x fuel index,
    # plus 3% if insured, rounded to 2 decimals.  The fuel index changes by order month.
    fuel_by_month = rng.uniform(0.95, 1.10, 12)
    fuel_index = np.round(fuel_by_month[order_date.month - 1], 3)
    mult = np.array([SERVICES[s][1] for s in service])
    freight_cost = np.round(mult * (6 + 0.02 * distance + 0.60 * weight) * fuel_index
                            * (1 + 0.03 * is_insured), 2)

    df = pd.DataFrame({
        "destination_city": city, "service_level": service, "distance": distance,
        "weight": weight, "order_date": order_date, "ship_date": ship_date,
        "promised_date": promised_date, "delivery_date": delivery_date,
        "freight_cost": freight_cost, "is_damaged": is_damaged, "is_insured": is_insured,
        "handling_touches": touches, "fuel_index": fuel_index,           # hidden
        "origin_hub": hub, "disrupted": disrupted,                        # hidden
    })
    df["lead_time"] = (df.delivery_date - df.order_date).dt.days
    df["delay_days"] = np.maximum(0, (df.delivery_date - df.promised_date).dt.days)   # hidden
    df["is_delayed"] = df.delay_days > 0
    return df


def simulate(rng, n, snapshot):
    m = int(1.3 * n) + 30
    while True:
        df = simulate_batch(rng, m, snapshot)
        df = df[df.delivery_date <= snapshot]              # only shipments already delivered
        if len(df) >= n:
            break
        m *= 2
    df = df.sort_values("order_date", kind="stable").iloc[:n].reset_index(drop=True)
    df.insert(0, "shipment_id", [f"S{i + 1:06d}" for i in range(len(df))])
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    o, s, p, d = (pd.to_datetime(df[x]) for x in ("order_date", "ship_date", "promised_date", "delivery_date"))

    # ---- exact rules ----
    out.append(check("order_date <= ship_date < delivery_date <= snapshot",
                     ((o <= s) & (s < d) & (d <= snapshot)).all()))
    out.append(check("promised_date > ship_date", (p > s).all()))
    out.append(check("lead_time = days from order_date to delivery_date", (df.lead_time == (d - o).dt.days).all()))
    out.append(check("is_delayed <=> delivery_date > promised_date", (df.is_delayed == (d > p)).all()))
    out.append(check("delay_days = max(0, delivery - promised)", (df.delay_days == np.maximum(0, (d - p).dt.days)).all()))
    mult = df.service_level.map({k: v[1] for k, v in SERVICES.items()})
    formula = np.round(mult * (6 + 0.02 * df.distance + 0.60 * df.weight) * df.fuel_index
                       * (1 + 0.03 * df.is_insured), 2)
    out.append(check("freight_cost follows the stated formula", ((df.freight_cost - formula).abs() < 0.006).all()))
    out.append(check("distance >= 20 km and weight between 0.5 and 800 kg",
                     ((df.distance >= 20) & df.weight.between(0.5, 800)).all()))
    out.append(check("categories valid",
                     df.service_level.isin(SERVICES).all() and df.destination_city.isin(CITIES).all()))
    out.append(check("shipment_id unique and rows ordered by order_date",
                     df.shipment_id.is_unique and o.is_monotonic_increasing))
    out.append(check("promise = ship_date + 1 + ceil(1.1 x km / speed) days",
                     ((p - s).dt.days == 1 + np.ceil(df.distance / df.service_level.map(
                         {k: v[0] for k, v in SERVICES.items()}) * PROMISE_MARGIN)).all()))

    # ---- relationships ----
    rho = df.distance.rank().corr(df.lead_time.rank())
    out.append(check("longer distance -> longer lead_time", rho > 0.3, f"rank corr={rho:.2f}", "relationship"))
    med = df.groupby("service_level").lead_time.median()
    counts = df.service_level.value_counts()
    if counts.get("express", 0) >= 8 and counts.get("economy", 0) >= 8:
        out.append(check("median lead_time: express < economy",
                         med["express"] < med["economy"], "", "relationship"))
    rho2 = df.weight.rank().corr(df.freight_cost.rank())
    out.append(check("heavier parcels cost more to ship", rho2 > 0.3, f"rank corr={rho2:.2f}", "relationship"))
    peak = o.dt.month.isin(PEAK_MONTHS)
    if peak.sum() >= 20 and (~peak).sum() >= 20:
        out.append(check("delays are more common in the Nov-Dec peak",
                         df.is_delayed[peak].mean() > df.is_delayed[~peak].mean(),
                         f"{df.is_delayed[peak].mean():.2f} vs {df.is_delayed[~peak].mean():.2f}", "relationship"))
    if df.is_damaged.sum() >= 6:
        out.append(check("damaged parcels had more handling touches",
                         df.handling_touches[df.is_damaged].mean() > df.handling_touches[~df.is_damaged].mean(),
                         "", "relationship"))
    return out


RECIPE = Recipe(
    key="logistics_shipping",
    title="Shipping & Delivery",
    row_meaning=("One row is one shipment that has been delivered to its "
                 "destination (shipments still in transit are not included)."),
    simulate=simulate,
    columns=["shipment_id", "destination_city", "service_level", "distance", "weight",
             "order_date", "ship_date", "promised_date", "delivery_date", "lead_time",
             "freight_cost", "is_delayed", "is_damaged", "is_insured"],
    core=["service_level", "distance", "weight", "lead_time", "freight_cost"],
    priority=["is_delayed", "delivery_date", "order_date", "is_insured", "is_damaged",
              "promised_date", "ship_date", "destination_city", "shipment_id"],
    docs={
        "shipment_id": "Unique shipment identifier, numbered in order-date order.",
        "destination_city": "City the parcel was delivered to (fictional).",
        "service_level": "economy (slowest, cheapest), standard or express (fastest, dearest).",
        "distance": "Road distance from the dispatching warehouse to the destination, in kilometres.",
        "weight": "Parcel weight in kilograms.",
        "order_date": "Date the customer placed the order.",
        "ship_date": "Date the parcel left the warehouse (same day or later than the order).",
        "promised_date": "Delivery date promised to the customer: ship_date + 1 + ceil(1.1 x distance / the service's daily km) days.",
        "delivery_date": "Date the parcel actually arrived.",
        "lead_time": "Days from order_date to delivery_date.",
        "freight_cost": "Shipping charge in plain currency units: service multiplier x (6 + 0.02 x km + 0.60 x kg) x fuel index, +3% if insured, to 2 decimals.",
        "is_delayed": "True if delivery_date is later than promised_date.",
        "is_damaged": "True if the parcel arrived damaged.",
        "is_insured": "True if the shipment was insured (3% surcharge on freight_cost).",
    },
    validate=validate,
    date_cols=["order_date", "ship_date", "promised_date", "delivery_date"],
    targets={
        "is_delayed": ["delivery_date", "lead_time"],
        "lead_time": ["delivery_date", "is_delayed"],
        "freight_cost": ["is_insured"],
    },
)
