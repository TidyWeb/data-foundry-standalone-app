"""Supplier & Vendor Performance.

One row is one purchase order placed with a supplier, and already received
(orders still on their way are not included).  Suppliers have persistent
traits, so the same supplier behaves similarly order after order; supplier
performance is what you get by grouping the order rows, it is never stored
as an independent number.

The story, in the order the code tells it:
    suppliers (hidden quality -> price level, lead time, reliability,
    defect and loss chances, warranty) -> orders (supplier, date, quantity)
    -> unit price -> promised date -> actual lead time -> received date
    -> damaged and lost units -> contract value -> compliance
    -> preferred status (from the supplier's PREVIOUS quarter)

Better suppliers tend to charge a bit more, deliver more reliably, have fewer
defects and offer longer warranties, but there is plenty of overlap.  Rates
are percentages of the units on that order (0-100, 2 decimals).  Money is in
plain currency units.  Teaching values only.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal, bernoulli, ordinal

BASE_PRICE = 8.0              # typical price of one unit before supplier and size effects
PROMISE_MARGIN = 1.25         # suppliers promise 25% more than their average lead time
MAX_QUALITY_LOSS = 3.0        # compliant order: damaged + lost units at most 3% of the order
PREFERRED_ON_TIME = 0.85      # preferred if last quarter's on-time share was at least 85% ...
PREFERRED_LOSS = 0.03         # ... and damaged + lost units were at most 3%, over 3+ orders


def make_suppliers(rng, k):
    """Step 1: fixed traits of each supplier, all linked to a hidden quality score."""
    quality = rng.normal(0, 1, k)
    mean_lead = lognormal(rng, 14, 0.4, k) * np.exp(-0.05 * quality)
    return dict(
        quality=quality,
        price_level=np.exp(0.10 * quality + rng.normal(0, 0.08, k)),
        mean_lead=mean_lead,
        promised_days=np.round(mean_lead * PROMISE_MARGIN).astype(int) + 1,
        lead_cv=np.clip(0.28 - 0.06 * quality + rng.normal(0, 0.06, k), 0.08, 0.6),
        defect_prob=np.clip(0.015 * np.exp(-0.5 * quality + rng.normal(0, 0.3, k)), 0.002, 0.15),
        loss_prob=np.clip(0.004 * np.exp(-0.3 * quality + rng.normal(0, 0.4, k)), 0.0005, 0.05),
        warranty=ordinal(quality + rng.normal(0, 0.6, k), [-0.5, 0.5], [12, 24, 36]).astype(int),
        share=rng.dirichlet(np.full(k, 1.5)),               # some suppliers get far more orders
    )


def quarter_index(dates):
    dates = pd.to_datetime(dates)
    return (dates.dt.year * 4 + (dates.dt.month - 1) // 3).to_numpy()


def is_preferred_flags(df):
    """Preferred = the supplier's PREVIOUS quarter (orders received then) was good enough."""
    received_q = pd.Series(quarter_index(df.received_date), index=df.index)
    card = df.assign(q=received_q, bad_units=df.units_damaged + df.units_lost).groupby(["supplier_id", "q"]).agg(
        orders=("order_id", "count"), on_time=("is_late", lambda s: 1 - s.mean()),
        bad=("bad_units", "sum"), units=("quantity", "sum"))
    card["good"] = ((card.orders >= 3) & (card.on_time >= PREFERRED_ON_TIME)
                    & (card.bad / card.units <= PREFERRED_LOSS))
    previous = pd.MultiIndex.from_arrays([df.supplier_id, quarter_index(df.order_date) - 1])
    return card.good.reindex(previous).fillna(False).astype(bool).to_numpy()


def simulate(rng, n, snapshot):
    k = int(np.clip(np.ceil(n / 25), 4, 60))
    s = make_suppliers(rng, k)

    # 2. Orders: which supplier, when, and how many units.
    who = rng.choice(k, n, p=s["share"])
    order_date = (snapshot - pd.Timedelta(days=365)
                  + pd.to_timedelta(rng.integers(0, 290, n), unit="D"))     # leaves time to receive them
    quantity = np.clip(np.round(lognormal(rng, 300, 0.9, n)), 10, 5000).astype(int)

    # 3. Price per unit: supplier price level, a small bulk discount, and a little noise.
    unit_price = np.round(BASE_PRICE * s["price_level"][who] * (quantity / 300) ** -0.04
                          * lognormal(rng, 1.0, 0.05, n), 2)
    contract_value = np.round(quantity * unit_price).astype(int)

    # 4. Lead time: promised by the supplier, then what actually happens (right-skewed).
    promised_days = s["promised_days"][who]
    lead_time = np.clip(np.ceil(lognormal(rng, s["mean_lead"][who], s["lead_cv"][who])), 1, 70).astype(int)
    is_late = lead_time > promised_days

    # 5. Quality: each unit is damaged or lost with the supplier's own chance
    #    (late shipments are handled roughly, so damage is 50% more likely).
    units_damaged = rng.binomial(quantity, np.minimum(1, s["defect_prob"][who] * np.where(is_late, 1.5, 1.0)))
    units_lost = rng.binomial(quantity, s["loss_prob"][who])
    within_quality = (units_damaged + units_lost) <= MAX_QUALITY_LOSS / 100 * quantity

    df = pd.DataFrame({
        "order_id": "", "supplier_id": [f"V{i + 1:02d}" for i in who],
        "order_date": order_date, "promised_date": order_date + pd.to_timedelta(promised_days, unit="D"),
        "received_date": order_date + pd.to_timedelta(lead_time, unit="D"),
        "lead_time": lead_time, "quantity": quantity, "unit_price": unit_price,
        "contract_value": contract_value, "warranty_period": s["warranty"][who],
        "loss_rate": np.round(100 * units_lost / quantity, 2),
        "damage_rate": np.round(100 * units_damaged / quantity, 2),
        "is_compliant": ~is_late & within_quality,
        # hidden columns (kept for validation, not shown to students)
        "units_damaged": units_damaged, "units_lost": units_lost, "is_late": is_late,
        "supplier_quality": np.round(s["quality"][who], 3), "lead_cv": np.round(s["lead_cv"][who], 3),
    })
    df = df.sort_values("order_date", kind="stable").reset_index(drop=True)
    df["order_id"] = [f"PO{i + 1:05d}" for i in range(n)]
    df["is_preferred"] = is_preferred_flags(df)          # needs all visible rows, so it comes last
    return df


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def supplier_table(df):
    """The scorecard students would build with groupby: one row per supplier."""
    g = df.groupby("supplier_id")
    return pd.DataFrame({
        "orders": g.size(), "price": g.unit_price.median(),
        "late_share": g.is_late.mean(), "cv": g.lead_cv.first(),
        "damage_share": g.units_damaged.sum() / g.quantity.sum(),
        "warranty": g.warranty_period.first(), "quality": g.supplier_quality.first(),
    })


def validate(df, snapshot):
    out = []
    o, p, r = (pd.to_datetime(df[x]) for x in ("order_date", "promised_date", "received_date"))

    # ---- exact rules ----
    out.append(check("order_date < promised_date and order_date < received_date <= snapshot",
                     ((o < p) & (o < r) & (r <= snapshot)).all()))
    out.append(check("lead_time = days from order_date to received_date", (df.lead_time == (r - o).dt.days).all()))
    out.append(check("hidden is_late <=> received_date > promised_date", (df.is_late == (r > p)).all()))
    out.append(check("contract_value = round(quantity x unit_price)",
                     ((df.contract_value - df.quantity * df.unit_price).abs() <= 0.5 + 1e-6).all()))
    out.append(check("damage_rate and loss_rate = 100 x units / quantity (2 dp)",
                     ((df.damage_rate - 100 * df.units_damaged / df.quantity).abs() <= 0.005 + 1e-9).all()
                     and ((df.loss_rate - 100 * df.units_lost / df.quantity).abs() <= 0.005 + 1e-9).all()))
    out.append(check("damaged + lost units never exceed the order quantity",
                     ((df.units_damaged + df.units_lost) <= df.quantity).all()))
    ok_quality = (df.units_damaged + df.units_lost) <= MAX_QUALITY_LOSS / 100 * df.quantity
    out.append(check("is_compliant <=> on time and damaged + lost <= 3% of units",
                     (df.is_compliant == (~df.is_late & ok_quality)).all()))
    out.append(check("is_preferred = rule on the supplier's previous quarter scorecard",
                     (df.is_preferred == is_preferred_flags(df)).all()))
    out.append(check("a supplier keeps one warranty_period",
                     (df.groupby("supplier_id").warranty_period.nunique() == 1).all()))
    out.append(check("warranty is 12, 24 or 36 months; quantity and prices positive",
                     df.warranty_period.isin([12, 24, 36]).all() and (df.quantity > 0).all()
                     and (df.unit_price > 0).all()))
    out.append(check("order_id unique and rows ordered by order_date",
                     df.order_id.is_unique and o.is_monotonic_increasing))

    # ---- relationships ----
    t = supplier_table(df)
    t = t[t.orders >= 4]
    if len(t) >= 5:
        r1 = t.price.rank().corr(t.damage_share.rank())
        out.append(check("pricier suppliers have lower damage (supplier-level rank corr)",
                         r1 < -0.1, f"rank corr={r1:.2f}", "relationship"))
        r2 = t.cv.rank().corr(t.late_share.rank())
        out.append(check("suppliers with more variable lead times are late more often",
                         r2 > 0.3, f"rank corr={r2:.2f}", "relationship"))
        long_w, short_w = t[t.warranty == 36], t[t.warranty == 12]
        if len(long_w) >= 2 and len(short_w) >= 2:
            out.append(check("suppliers with 36-month warranties have less damage than 12-month ones",
                             long_w.damage_share.mean() < short_w.damage_share.mean(), "", "relationship"))
    out.append(check("lead_time is right-skewed (mean > median)",
                     df.lead_time.mean() > df.lead_time.median(), "", "relationship"))
    late = df.is_late
    if late.sum() >= 8 and (~late).sum() >= 8:
        d_late = df.units_damaged[late].sum() / df.quantity[late].sum()
        d_ok = df.units_damaged[~late].sum() / df.quantity[~late].sum()
        out.append(check("late orders have more damaged units per unit ordered",
                         d_late > d_ok, f"{d_late:.3f} vs {d_ok:.3f}", "relationship"))
    pref = df.is_preferred
    if pref.sum() >= 15 and (~pref).sum() >= 15:
        out.append(check("orders from preferred suppliers are compliant more often",
                         df.is_compliant[pref].mean() > df.is_compliant[~pref].mean(),
                         f"{df.is_compliant[pref].mean():.2f} vs {df.is_compliant[~pref].mean():.2f}",
                         "relationship"))
    return out


RECIPE = Recipe(
    key="logistics_supplier",
    title="Supplier & Vendor Performance",
    row_meaning=("One row is one purchase order placed with a supplier and "
                 "already received (orders still on their way are not included)."),
    simulate=simulate,
    columns=["order_id", "supplier_id", "order_date", "promised_date", "received_date",
             "lead_time", "quantity", "unit_price", "contract_value", "warranty_period",
             "loss_rate", "damage_rate", "is_compliant", "is_preferred"],
    core=["supplier_id", "unit_price", "lead_time", "damage_rate"],
    priority=["contract_value", "quantity", "is_compliant", "loss_rate", "warranty_period",
              "is_preferred", "promised_date", "received_date", "order_date", "order_id"],
    docs={
        "order_id": "Unique purchase-order identifier, numbered in order-date order.",
        "supplier_id": "Identifier of the supplier (vendor) the order was placed with.",
        "order_date": "Date the purchase order was placed.",
        "promised_date": "Delivery date the supplier promised when the order was placed.",
        "received_date": "Date the goods actually arrived.",
        "lead_time": "Days from order_date to received_date.",
        "quantity": "Number of units ordered.",
        "unit_price": "Price of one unit in plain currency units (higher-quality suppliers tend to charge more).",
        "contract_value": "Total value of the order: quantity x unit_price, rounded to a whole currency unit.",
        "warranty_period": "Length of the supplier's warranty in months (12, 24 or 36); a fixed trait of the supplier.",
        "loss_rate": "Percentage of the ordered units that were lost or never arrived (0-100, 2 decimals).",
        "damage_rate": "Percentage of the ordered units that arrived damaged (0-100, 2 decimals).",
        "is_compliant": "True if the order arrived on or before promised_date and damaged plus lost units were at most 3% of the order.",
        "is_preferred": "True if the supplier's previous quarter (orders received) had 3+ orders, at least 85% on time and at most 3% damaged or lost units.",
    },
    validate=validate,
    date_cols=["order_date", "promised_date", "received_date"],
    targets={
        "is_compliant": ["received_date", "lead_time", "damage_rate", "loss_rate"],
        "lead_time": ["received_date", "is_compliant"],
    },
)
