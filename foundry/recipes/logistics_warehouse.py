"""Warehouse Operations.

One row is one warehouse-day: what one distribution warehouse did on one day.
Each warehouse is followed for the last 30 days, and its unfinished orders
(the backlog) and its pallet stock are carried from each day into the next,
so a row depends on the row before it.

The story, in the order the code tells it:
    warehouse traits (size, automated or manual, typical pick rate)
    -> each day: orders arrive (weekday pattern, occasional promotions)
    -> staffing (planned for a normal day, then some staff are absent)
    -> daily capacity = pickers x 8 hours x pick_rate
    -> orders processed = as many as capacity allows (backlog first)
    -> backlog carried forward -> pallets leave -> restock arrives -> pallet count

Promotion days push orders above what staff planned for, which builds a
backlog that takes several days to clear.  Automated warehouses pick faster
and need fewer people.  All numbers are illustrative teaching values.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, bernoulli

DAYS = 30                 # days followed per warehouse
HOURS_PER_SHIFT = 8
PALLETS_PER_ORDER = 0.06  # on average, each processed order takes 0.06 pallets out of stock
WEEKDAY_FACTOR = np.array([1.2, 1.1, 1.0, 1.0, 1.05, 0.5, 0.3])   # Monday ... Sunday
COVERAGE = 1.2            # managers roster 20% more picking capacity than a normal day needs
ATTENDANCE = 0.92         # on average 92% of rostered pickers turn up


def make_warehouses(rng, k):
    """Step 1: fixed traits of each warehouse."""
    is_automated = np.zeros(k, dtype=bool)
    is_automated[rng.permutation(k)[: max(1, round(0.35 * k))]] = True      # about a third are automated
    pallet_capacity = np.round(np.clip(lognormal(rng, 3000, 0.4, k), 1500, 12000), -2).astype(int)
    typical_orders = pallet_capacity * 0.4 * lognormal(rng, 1.0, 0.25, k) * np.where(is_automated, 1.3, 1.0)
    typical_rate = lognormal(rng, np.where(is_automated, 34, 16), 0.15, k)   # order lines per picker-hour
    return is_automated, pallet_capacity, typical_orders, typical_rate


def simulate(rng, n, snapshot):
    k = max(2, int(np.ceil(n / DAYS)))
    is_automated, pallet_capacity, typical_orders, typical_rate = make_warehouses(rng, k)

    start = snapshot - pd.Timedelta(days=DAYS - 1)
    dates = start + pd.to_timedelta(np.arange(DAYS), unit="D")
    weekday = np.asarray(dates.dayofweek)

    # Starting position: a small backlog and a fairly full warehouse.
    backlog = np.round(0.1 * typical_orders)
    stock = np.round(0.6 * pallet_capacity)
    days = {name: [] for name in ["orders_received", "pickers_on_shift", "pick_rate",
                                  "daily_capacity", "orders_processed", "backlog",
                                  "inbound_pallets", "outbound_pallets", "pallet_count",
                                  "is_promo_day"]}

    for d in range(DAYS):
        # 2. Orders arrive: weekday pattern, and now and then a promotion (+60%).
        expected = typical_orders * WEEKDAY_FACTOR[weekday[d]]
        is_promo = bernoulli(rng, np.full(k, 0.08))
        received = rng.poisson(expected * np.where(is_promo, 1.6, 1.0))

        # 3. Staffing is planned for the expected day (not for promotions); some staff are absent.
        planned = np.ceil(expected * COVERAGE / (HOURS_PER_SHIFT * typical_rate)).astype(int)
        pickers = rng.binomial(planned, ATTENDANCE)
        pick_rate = np.round(typical_rate * lognormal(rng, 1.0, 0.05, k), 2)
        capacity = np.floor(np.round(pickers * HOURS_PER_SHIFT * pick_rate, 6)).astype(int)

        # 4. Process what capacity allows, oldest orders first; the rest is the new backlog.
        waiting = backlog + received
        processed = np.minimum(waiting, capacity).astype(int)
        backlog = waiting - processed

        # 5. Pallets: shipped-out orders take stock away; restock arrives when stock runs low.
        outbound = np.round(processed * PALLETS_PER_ORDER * lognormal(rng, 1.0, 0.1, k))
        outbound = np.minimum(outbound, stock)
        after_pick = stock - outbound
        low_stock = after_pick < 0.4 * pallet_capacity
        top_up = np.round(rng.uniform(0.3, 0.5, k) * pallet_capacity)          # a big delivery
        small_drop = np.round(rng.uniform(0.02, 0.08, k) * pallet_capacity) * bernoulli(rng, np.full(k, 0.15))
        inbound = np.where(low_stock, top_up, small_drop)
        inbound = np.minimum(inbound, pallet_capacity - after_pick)            # never over capacity
        stock = after_pick + inbound

        for name, value in [("orders_received", received), ("pickers_on_shift", pickers),
                            ("pick_rate", pick_rate), ("daily_capacity", capacity),
                            ("orders_processed", processed), ("backlog", backlog),
                            ("inbound_pallets", inbound), ("outbound_pallets", outbound),
                            ("pallet_count", stock), ("is_promo_day", is_promo)]:
            days[name].append(np.asarray(value))

    # Lay out one warehouse after another, each in date order.
    df = pd.DataFrame({name: np.stack(rows).T.reshape(-1) for name, rows in days.items()})
    for c in ["orders_received", "pickers_on_shift", "daily_capacity", "orders_processed",
              "backlog", "inbound_pallets", "outbound_pallets", "pallet_count"]:
        df[c] = df[c].astype(int)
    df.insert(0, "date", np.tile(dates, k))
    df.insert(0, "warehouse_id", np.repeat([f"W{i + 1:02d}" for i in range(k)], DAYS))
    df["is_automated"] = np.repeat(is_automated, DAYS)
    df["pallet_capacity"] = np.repeat(pallet_capacity, DAYS)
    df["is_fulfilled"] = df.backlog == 0          # every order received so far has been picked
    return df.iloc[:n].reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    g = df.groupby("warehouse_id")
    prev_backlog = g.backlog.shift(1)
    prev_stock = g.pallet_count.shift(1)
    later = prev_backlog.notna()                      # every day except a warehouse's first

    # ---- exact rules ----
    out.append(check("daily_capacity = floor(pickers x 8 x pick_rate)",
                     (df.daily_capacity == np.floor(np.round(df.pickers_on_shift * HOURS_PER_SHIFT * df.pick_rate, 6))).all()))
    out.append(check("orders_processed <= daily_capacity", (df.orders_processed <= df.daily_capacity).all()))
    out.append(check("backlog today = backlog yesterday + received - processed",
                     ((prev_backlog + df.orders_received - df.orders_processed)[later] == df.backlog[later]).all()))
    out.append(check("orders_processed = min(capacity, backlog yesterday + received)",
                     (df.orders_processed[later] == np.minimum(df.daily_capacity, prev_backlog + df.orders_received)[later]).all()))
    out.append(check("pallet_count today = yesterday + inbound - outbound",
                     ((prev_stock + df.inbound_pallets - df.outbound_pallets)[later] == df.pallet_count[later]).all()))
    out.append(check("0 <= pallet_count <= pallet_capacity",
                     ((df.pallet_count >= 0) & (df.pallet_count <= df.pallet_capacity)).all()))
    out.append(check("is_fulfilled <=> backlog == 0", (df.is_fulfilled == (df.backlog == 0)).all()))
    out.append(check("counts are not negative",
                     (df[["orders_received", "pickers_on_shift", "backlog", "inbound_pallets",
                          "outbound_pallets"]] >= 0).all().all()))
    dates = pd.to_datetime(df.date)
    out.append(check("dates are consecutive days per warehouse and not after the snapshot",
                     (g.date.apply(lambda s: (pd.to_datetime(s).diff().dropna() == pd.Timedelta(days=1)).all())).all()
                     and (dates <= snapshot).all()))
    out.append(check("a warehouse keeps one capacity and one automation setting",
                     (g[["pallet_capacity", "is_automated"]].nunique() == 1).all().all()))

    # ---- relationships ----
    auto = df.pick_rate[df.is_automated]
    manual = df.pick_rate[~df.is_automated]
    if len(auto) >= 10 and len(manual) >= 10:
        out.append(check("automated warehouses pick faster (median pick_rate)",
                         auto.median() > manual.median(), "", "relationship"))
    weekend = dates.dt.dayofweek >= 5
    if weekend.sum() >= 8:
        out.append(check("weekday order volume is higher than weekend volume (per warehouse)",
                         (df.orders_received / g.orders_received.transform("mean"))[~weekend].mean()
                         > (df.orders_received / g.orders_received.transform("mean"))[weekend].mean(),
                         "", "relationship"))
    promo = df.is_promo_day.astype(bool)
    if promo.sum() >= 5:
        rel = df.orders_received / g.orders_received.transform("median")
        out.append(check("promotion days bring more orders", rel[promo].mean() > rel[~promo].mean() * 1.2,
                         f"{rel[promo].mean():.2f} vs {rel[~promo].mean():.2f}", "relationship"))
        out.append(check("backlog is higher on promotion days",
                         (df.backlog / g.backlog.transform("mean").clip(lower=1))[promo].mean()
                         > (df.backlog / g.backlog.transform("mean").clip(lower=1))[~promo].mean(),
                         "", "relationship"))
    if later.sum() >= 30:
        r = df.backlog[later].corr(prev_backlog[later])
        out.append(check("backlog carries over from one day to the next", r > 0.5,
                         f"lag-1 corr={r:.2f}", "relationship"))
    return out


RECIPE = Recipe(
    key="logistics_warehouse",
    title="Warehouse Operations",
    row_meaning=("One row is one warehouse-day: one distribution warehouse on one "
                 "day, with its backlog and stock carried over from the day before."),
    simulate=simulate,
    columns=["warehouse_id", "date", "is_automated", "pallet_capacity", "pallet_count",
             "pickers_on_shift", "pick_rate", "daily_capacity", "orders_received",
             "orders_processed", "backlog", "inbound_pallets", "outbound_pallets",
             "is_fulfilled"],
    core=["orders_received", "daily_capacity", "orders_processed", "backlog"],
    priority=["pick_rate", "pickers_on_shift", "date", "warehouse_id", "is_fulfilled",
              "is_automated", "pallet_count", "pallet_capacity", "inbound_pallets",
              "outbound_pallets"],
    docs={
        "warehouse_id": "Identifier of the warehouse; each warehouse has one row per day.",
        "date": "The calendar day this row describes.",
        "is_automated": "True if the warehouse uses automated picking (faster, needs fewer pickers).",
        "pallet_capacity": "Maximum number of pallets the warehouse can hold.",
        "pallet_count": "Pallets in stock at the end of the day (yesterday + inbound - outbound).",
        "pickers_on_shift": "Number of pickers who actually worked that day (after absences).",
        "pick_rate": "Order lines picked per picker per hour that day.",
        "daily_capacity": "Most orders the shift could process: pickers_on_shift x 8 hours x pick_rate, rounded down.",
        "orders_received": "New orders that arrived during the day.",
        "orders_processed": "Orders picked and shipped that day: the smaller of daily_capacity and (yesterday's backlog + orders_received).",
        "backlog": "Orders still waiting at the end of the day: yesterday's backlog + orders_received - orders_processed.",
        "inbound_pallets": "Pallets delivered into the warehouse that day (restocking).",
        "outbound_pallets": "Pallets that left the warehouse that day with processed orders.",
        "is_fulfilled": "True if the day ended with no backlog (every order received so far was processed).",
    },
    validate=validate,
    date_cols=["date"],
    targets={
        "backlog": ["orders_processed", "is_fulfilled"],
        "is_fulfilled": ["backlog", "orders_processed"],
    },
)
