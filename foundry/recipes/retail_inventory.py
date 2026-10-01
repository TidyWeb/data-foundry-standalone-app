"""Inventory Management.

One row is one SKU-week: what one product (SKU) did in the warehouse during one
week.  Rows for the same sku_id are consecutive weeks, so this week's opening
stock is last week's closing stock.

The story, in the order the code tells it:
    SKU (weekly demand, supplier lead time, discontinued?) -> stocking policy
    (reorder point, order size) -> a week-by-week loop:
        receive deliveries -> customers buy -> small losses -> closing stock
        -> if stock plus stock on order is at or below the reorder point, order more

Demand that cannot be met is a lost sale (there is no backlog).  The reorder
point covers demand during the lead time plus a safety cushion, and the order
size follows the classic economic-order-quantity (EOQ) rule, used here only as
a company policy.  Illustrative teaching values, not real data.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal

WEEKS = 12                  # weeks of history per SKU
LEAD_TIME_CHOICES = [3, 5, 7, 10, 14, 21]     # a supplier's usual delivery time, days
SHRINKAGE_RATE = 0.004      # share of stock lost (damage, theft) per week
DISCONTINUED_SHARE = 0.15


def simulate_one_sku(rng, sku_number, first_monday, snapshot):
    """Run one SKU through WEEKS weeks and return one dict per week."""
    # --- the SKU's fixed traits ---
    weekly_demand = lognormal(rng, 30, 0.9)              # a few fast sellers, many slow
    usual_lead = int(rng.choice(LEAD_TIME_CHOICES))
    is_discontinued = bool(rng.random() < DISCONTINUED_SHARE)
    service_z = rng.uniform(1.0, 2.0)                    # how much safety stock we want
    order_cost = rng.uniform(30, 80)                     # admin cost of one purchase order
    holding_cost = rng.uniform(1.5, 6.0)                 # yearly cost of holding one unit

    # --- the stocking policy (constant for the SKU) ---
    lead_weeks = usual_lead / 7
    safety_stock = service_z * 0.35 * weekly_demand * np.sqrt(lead_weeks)
    reorder_point = int(np.ceil(weekly_demand * lead_weeks + safety_stock))
    eoq = np.sqrt(2 * 52 * weekly_demand * order_cost / holding_cost)
    restock_quantity = int(max(round(eoq / 5) * 5, np.ceil(weekly_demand), 10))

    # --- starting state: some stock, and one earlier delivery already received ---
    on_hand = int(reorder_point + restock_quantity * rng.uniform(0.2, 1.0))
    last_order_date = first_monday - pd.Timedelta(days=int(rng.integers(28, 60)))
    last_lead = usual_lead
    pipeline = []                                        # orders not yet delivered
    rows = []

    for week in range(WEEKS):
        week_start = first_monday + pd.Timedelta(weeks=week)
        week_end = week_start + pd.Timedelta(days=6)
        opening = on_hand

        # 1. Deliveries due this week arrive.
        received = 0
        still_waiting = []
        for order_date, lead, quantity in pipeline:
            if order_date + pd.Timedelta(days=lead) <= week_end:
                received += quantity
                if order_date > last_order_date:
                    last_order_date, last_lead = order_date, lead
            else:
                still_waiting.append((order_date, lead, quantity))
        pipeline = still_waiting

        # 2. Customers buy.  Discontinued items sell 8% less each week.
        fade = 0.92 ** week if is_discontinued else 1.0
        demand = rng.poisson(weekly_demand * fade * lognormal(rng, 1.0, 0.25))
        available = opening + received
        units_sold = min(demand, available)
        had_stockout = bool(demand > available)

        # 3. Shrinkage (never more than what is left on the shelf).
        lost = min(rng.poisson(SHRINKAGE_RATE * opening), available - units_sold)
        adjustment = -int(lost)
        on_hand = available - units_sold + adjustment

        # 4. Reorder if stock plus stock on order is at or below the reorder point.
        on_order = sum(q for _, _, q in pipeline)
        if not is_discontinued and on_hand + on_order <= reorder_point:
            lead = max(1, int(round(usual_lead + rng.normal(0, 0.15 * usual_lead))))
            pipeline.append((week_end, lead, restock_quantity))

        rows.append({
            "sku_id": f"SKU{sku_number + 1:04d}",
            "week_start": week_start,
            "opening_stock": int(opening),
            "units_received": int(received),
            "units_sold": int(units_sold),
            "adjustment": adjustment,
            "stock_level": int(on_hand),
            "reorder_point": reorder_point,
            "restock_quantity": restock_quantity,
            "lead_time": last_lead,
            "restock_date": last_order_date + pd.Timedelta(days=last_lead),
            "last_order_date": last_order_date,                 # hidden
            "is_low_stock": bool(on_hand <= reorder_point),
            "had_stockout": had_stockout,
            "is_discontinued": is_discontinued,
            "weekly_demand_rate": round(weekly_demand, 2),      # hidden
            "usual_lead_time": usual_lead,                      # hidden
        })
    return rows


def simulate(rng, n, snapshot):
    # Last complete week (Monday to Sunday) ending on or before the snapshot.
    last_monday = snapshot - pd.Timedelta(days=int(snapshot.dayofweek) + 7)
    if snapshot.dayofweek == 6:
        last_monday = snapshot - pd.Timedelta(days=6)
    first_monday = last_monday - pd.Timedelta(weeks=WEEKS - 1)

    n_skus = int(np.ceil(n / WEEKS))
    rows = []
    for k in range(n_skus):
        rows.extend(simulate_one_sku(rng, k, first_monday, snapshot))
    return pd.DataFrame(rows).iloc[:n].reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    week_end = pd.to_datetime(df.week_start) + pd.Timedelta(days=6)

    # ---- exact rules ----
    out.append(check("closing = opening + received - sold + adjustment",
                     (df.stock_level == df.opening_stock + df.units_received
                      - df.units_sold + df.adjustment).all()))
    out.append(check("units_sold <= opening + received",
                     (df.units_sold <= df.opening_stock + df.units_received).all()))
    out.append(check("stock never negative; adjustment never positive",
                     ((df.stock_level >= 0) & (df.adjustment <= 0) & (df.units_sold >= 0)).all()))
    previous_closing = df.groupby("sku_id").stock_level.shift(1)
    has_previous = previous_closing.notna()
    out.append(check("opening stock = previous week's closing stock",
                     (previous_closing[has_previous] == df.opening_stock[has_previous]).all()))
    out.append(check("is_low_stock exactly when stock_level <= reorder_point",
                     (df.is_low_stock == (df.stock_level <= df.reorder_point)).all()))
    out.append(check("restock_date = order date + lead_time",
                     (pd.to_datetime(df.restock_date) - pd.to_datetime(df.last_order_date)
                      == pd.to_timedelta(df.lead_time, unit="D")).all()))
    out.append(check("restock_date not after the end of the week",
                     (pd.to_datetime(df.restock_date) <= week_end).all()))
    out.append(check("week ends on or before snapshot", (week_end <= snapshot).all()))
    gaps = pd.to_datetime(df.week_start).groupby(df.sku_id).diff().dropna()
    out.append(check("weeks are Mondays and consecutive per SKU",
                     (pd.to_datetime(df.week_start).dt.dayofweek == 0).all()
                     and (gaps == pd.Timedelta(days=7)).all()))
    out.append(check("SKU policy traits constant per SKU",
                     (df.groupby("sku_id").reorder_point.nunique() == 1).all()
                     and (df.groupby("sku_id").restock_quantity.nunique() == 1).all()
                     and (df.groupby("sku_id").is_discontinued.nunique() == 1).all()))
    out.append(check("had_stockout implies stock ran out (closing stock is 0)",
                     (df.stock_level[df.had_stockout] == 0).all()))
    out.append(check("lead_time >= 1 and restock_quantity >= 1",
                     ((df.lead_time >= 1) & (df.restock_quantity >= 1)).all()))
    out.append(check("discontinued SKUs receive no deliveries",
                     (df.units_received[df.is_discontinued] == 0).all()))

    # ---- relationships ----
    # Discontinued SKUs are left out: their sales fade, so they break the usual link.
    active = df[~df.is_discontinued]
    per_sku = active.groupby("sku_id").agg(sold=("units_sold", "mean"), rp=("reorder_point", "first"),
                                       q=("restock_quantity", "first"), stock=("stock_level", "mean"),
                                       lead=("usual_lead_time", "first"))
    if len(per_sku) >= 6:
        c = float(np.log(per_sku.sold + 1).corr(np.log(per_sku.rp)))
        out.append(check("faster-selling SKUs have higher reorder points",
                         c > 0.7, f"corr={c:.2f}", "relationship"))
        cover = per_sku.rp / (per_sku.sold + 1)
        c = float(cover.corr(per_sku.lead))
        out.append(check("longer supplier lead time -> reorder point covers more weeks of sales",
                         c > 0.4, f"corr={c:.2f}", "relationship"))
        c = float(np.log(per_sku.stock + 1).corr(np.log(per_sku.q)))
        out.append(check("bigger order sizes -> higher average stock on hand",
                         c > 0.4, f"corr={c:.2f}", "relationship"))

    low = df.opening_stock <= df.reorder_point
    if low.sum() >= 15 and (~low).sum() >= 15:
        out.append(check("stock-outs are likelier in weeks that open at or below the reorder point",
                         df.had_stockout[low].mean() > df.had_stockout[~low].mean(),
                         f"{df.had_stockout[low].mean():.2f} vs {df.had_stockout[~low].mean():.2f}",
                         "relationship"))
    disc = df[df.is_discontinued]
    if disc.sku_id.nunique() >= 2 and (~df.is_discontinued).sum() >= 24:
        out.append(check("discontinued SKUs are low on stock more often (nothing reorders them)",
                         disc.is_low_stock.mean() > df[~df.is_discontinued].is_low_stock.mean(),
                         "", "relationship"))
    return out


RECIPE = Recipe(
    key="retail_inventory",
    title="Inventory Management",
    row_meaning=("One row is one SKU-week: what one product did in the warehouse "
                 "during one Monday-to-Sunday week."),
    simulate=simulate,
    columns=["sku_id", "week_start", "opening_stock", "units_received", "units_sold",
             "adjustment", "stock_level", "reorder_point", "restock_quantity", "lead_time",
             "restock_date", "is_low_stock", "had_stockout", "is_discontinued"],
    core=["sku_id", "units_sold", "stock_level", "reorder_point"],
    priority=["week_start", "opening_stock", "units_received", "is_low_stock",
              "restock_quantity", "lead_time", "had_stockout", "is_discontinued",
              "adjustment", "restock_date"],
    docs={
        "sku_id": "Product (stock keeping unit) identifier; one row per week for each.",
        "week_start": "Monday that starts the week.",
        "opening_stock": "Units on the shelf at the start of the week (last week's stock_level).",
        "units_received": "Units delivered by suppliers during the week.",
        "units_sold": "Units sold during the week (never more than was available).",
        "adjustment": "Stock lost to damage or theft this week, as a negative number of units (0 if none).",
        "stock_level": "Units on the shelf at the end of the week: opening + received - sold + adjustment.",
        "reorder_point": "Policy trigger: a new order is placed when stock plus stock on order falls to this many units.",
        "restock_quantity": "Units in each purchase order for this SKU (fixed by the ordering policy).",
        "lead_time": "Days the most recent delivery took from order to arrival.",
        "restock_date": "Date the most recent delivery arrived (order date + lead_time).",
        "is_low_stock": "True if stock_level is at or below reorder_point.",
        "had_stockout": "True if customers wanted more than was available and some sales were lost.",
        "is_discontinued": "True if the SKU is being phased out: no new orders, demand fades.",
    },
    validate=validate,
    date_cols=["week_start", "restock_date"],
    targets={
        "had_stockout": ["units_sold", "stock_level", "is_low_stock"],
        "units_sold": ["stock_level", "is_low_stock", "had_stockout"],
        "is_low_stock": ["stock_level"],
    },
)
