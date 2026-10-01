"""Customer Purchase Behaviour.

One row is one customer-month in which the customer visited the shop's website
at least once.  Rows for a customer are consecutive active months (a lapsed
customer simply stops visiting, so their rows thin out or end).  Customers
keep their traits (signup_date, is_subscribed, ...) on every row.

The story, in the order the code tells it:
    customer (signup date, engagement, buying rate, drop-out risk, subscribed?)
    -> month by month while "alive": orders ~ Poisson(rate) -> after buying the
    customer may drop out for good -> browsing visits -> pages, session length,
    bounces -> items and spend -> loyalty points -> VIP status

This follows the buy-till-you-die idea: active customers buy at their own
steady rate, and each purchase carries a small chance they never come back.
History starts at signup, so loyalty points and "orders so far" carry over
correctly into the 12 months you see.  Illustrative teaching values.
"""
import numpy as np
import pandas as pd

from ..core import Recipe, check, lognormal

HISTORY_MONTHS = 36         # signups happen up to 3 years back
WINDOW_MONTHS = 12          # the rows we show
VIP_SPEND_6M = 500          # VIP = spent at least this over the last 6 months
POINTS_PER_UNIT = 1         # loyalty points earned per whole unit of spend
QUIET_VISIT_SHARE = 0.05    # a dropped-out customer still browses, but 5% as much


def month_starts(snapshot):
    """First day of each of the last HISTORY_MONTHS complete months."""
    last = snapshot.to_period("M")
    if snapshot != (last.end_time.normalize()):        # this month is not finished
        last = last - 1
    return [(last - k).start_time for k in range(HISTORY_MONTHS - 1, -1, -1)]


def simulate_customer(rng, number, months):
    """All rows (window months only) for one customer."""
    # --- fixed traits ---
    first_day, last_day = months[0], months[-1] + pd.offsets.MonthEnd(0)
    signup_date = first_day + pd.Timedelta(days=int(rng.integers(0, (last_day - first_day).days + 1)))
    engagement = lognormal(rng, 1.0, 0.4)                  # more engaged -> visits, buys, browses more
    is_subscribed = bool(rng.random() < 0.30 + 0.15 * (engagement > 1.2))
    buy_rate = rng.gamma(1.2, 0.7) * engagement * (2.2 if is_subscribed else 1.0)
    dropout_risk = rng.beta(1, 12) * (0.6 if is_subscribed else 1.0)
    visit_rate = 2.5 * engagement
    bounce_chance = min(0.9, 0.55 / engagement ** 0.7)
    pages_per_visit = 2 + 2 * engagement
    items_per_order = 0.8 + 0.5 * rng.gamma(2, 1)
    item_price = lognormal(rng, 22, 0.4)

    # --- month-by-month state ---
    alive, points = True, 0
    orders_to_date, items_to_date = 0, 0
    last_purchase = pd.NaT
    spend_history = []
    rows = []
    for index, month_start in enumerate(months):
        month_end = month_start + pd.offsets.MonthEnd(0)
        if month_end < signup_date:
            spend_history.append(0)
            continue
        first_month = month_start <= signup_date
        share = ((month_end - signup_date).days + 1) / month_start.days_in_month if first_month else 1.0

        # 1. Orders.  The first month always has the welcome order.
        orders = 0
        if alive:
            orders = int(rng.poisson(buy_rate * share))
            if first_month:
                orders = max(orders, 1)
            if orders and rng.random() > (1 - dropout_risk) ** orders:
                alive = False                              # never comes back after this month
        # 2. Visits: every order is a visit; the rest are browsing visits.
        browse_rate = visit_rate * share * (1.0 if alive or orders else QUIET_VISIT_SHARE)
        browsing = int(rng.poisson(browse_rate))
        visits = orders + browsing
        bounced = int(rng.binomial(browsing, bounce_chance)) if browsing else 0
        page_views = visits + int(rng.poisson(pages_per_visit * visits + 3 * orders)) if visits else 0
        session_length = int(np.clip(lognormal(rng, 50 + 60 * engagement + 40 * (orders > 0), 0.35), 10, 3600))

        # 3. Items and spend.
        items = orders + int(rng.poisson(items_per_order * orders)) if orders else 0
        spend = int(round(items * item_price * lognormal(rng, 1.0, 0.25))) if orders else 0
        opening_points = points
        earned = spend * POINTS_PER_UNIT
        redeemed = 0
        if orders and opening_points + earned >= 250 and rng.random() < 0.5:
            redeemed = int(100 * ((opening_points + earned) // 2 // 100))
        points = opening_points + earned - redeemed

        # 4. Running totals and last purchase date.
        previous_purchase = last_purchase
        if orders:
            low = max(month_start, signup_date)
            day = int(rng.integers(0, (month_end - low).days + 1))
            last_purchase = low + pd.Timedelta(days=day)
        orders_to_date += orders
        items_to_date += items
        spend_history.append(spend)
        spend_6m = sum(spend_history[-6:])

        if index >= HISTORY_MONTHS - WINDOW_MONTHS and visits >= 1:
            rows.append({
                "customer_id": f"C{number + 1:04d}",
                "month": month_start,
                "signup_date": signup_date,
                "orders": orders,
                "spend": spend,
                "page_view": page_views,
                "session_length": session_length,
                "bounce_rate": round(bounced / visits, 2),
                "avg_basket_size": round(items_to_date / orders_to_date, 2),
                "loyalty_points": int(points),
                "last_purchase": last_purchase,
                "is_returning_customer": bool(orders_to_date >= 2),
                "is_subscribed": is_subscribed,
                "is_vip": bool(spend_6m >= VIP_SPEND_6M),
                # hidden helpers
                "visits": visits, "bounced_visits": bounced,
                "opening_points": int(opening_points), "points_earned": int(earned),
                "points_redeemed": int(redeemed), "orders_to_date": orders_to_date,
                "items_to_date": items_to_date, "spend_6m": spend_6m,
                "previous_purchase": previous_purchase,
            })
    return rows


def simulate(rng, n, snapshot):
    months = month_starts(snapshot)
    rows, number = [], 0
    while len(rows) < n:                                    # add customers until we have enough rows
        for _ in range(max(2, n // 8)):
            rows.extend(simulate_customer(rng, number, months))
            number += 1
    df = pd.DataFrame(rows).iloc[:n]
    return df.sort_values(["customer_id", "month"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df, snapshot):
    out = []
    month_end = pd.to_datetime(df.month) + pd.offsets.MonthEnd(0)
    last_purchase = pd.to_datetime(df.last_purchase)
    by_customer = df.groupby("customer_id")

    # ---- exact rules ----
    out.append(check("orders <= visits, and every row has at least one visit",
                     ((df.orders <= df.visits) & (df.visits >= 1)).all()))
    out.append(check("bounced visits <= visits - orders",
                     (df.bounced_visits <= df.visits - df.orders).all()))
    out.append(check("bounce_rate = bounced visits / visits (+-0.005), between 0 and 1",
                     ((df.bounce_rate - df.bounced_visits / df.visits).abs() <= 0.0051).all()
                     and df.bounce_rate.between(0, 1).all()))
    out.append(check("loyalty_points = opening + earned - redeemed, never negative",
                     (df.loyalty_points == df.opening_points + df.points_earned - df.points_redeemed).all()
                     and (df.loyalty_points >= 0).all()))
    previous = by_customer.loyalty_points.shift(1)
    has_prev = previous.notna()
    out.append(check("opening points = previous row's closing points",
                     (previous[has_prev] == df.opening_points[has_prev]).all()))
    out.append(check("points earned = spend x rate", (df.points_earned == df.spend * POINTS_PER_UNIT).all()))
    out.append(check("spend is 0 exactly when there were no orders",
                     ((df.spend == 0) == (df.orders == 0)).all()))
    out.append(check("signup_date <= last_purchase <= end of month and snapshot",
                     ((last_purchase >= pd.to_datetime(df.signup_date)) & (last_purchase <= month_end)
                      & (last_purchase <= snapshot)).all()))
    out.append(check("last_purchase never goes backwards for a customer",
                     (last_purchase.groupby(df.customer_id).diff().dropna() >= pd.Timedelta(0)).all()))
    out.append(check("last_purchase is in this month when there were orders",
                     ((last_purchase >= pd.to_datetime(df.month)) | (df.orders == 0)).all()))
    out.append(check("signup_date and is_subscribed constant per customer",
                     (by_customer.signup_date.nunique() == 1).all()
                     and (by_customer.is_subscribed.nunique() == 1).all()))
    out.append(check("is_vip = spend over the trailing 6 months >= threshold",
                     (df.is_vip == (df.spend_6m >= VIP_SPEND_6M)).all() and (df.spend_6m >= df.spend).all()))
    out.append(check("is_returning_customer = 2 or more orders to date; avg_basket_size = items / orders",
                     (df.is_returning_customer == (df.orders_to_date >= 2)).all()
                     and ((df.avg_basket_size - df.items_to_date / df.orders_to_date).abs() <= 0.0051).all()
                     and (df.avg_basket_size >= 1).all()))
    out.append(check("session_length 10-3600 seconds; page_view >= visits",
                     (df.session_length.between(10, 3600) & (df.page_view >= df.visits)).all()))
    out.append(check("months are first-of-month dates inside the window and not after snapshot",
                     ((pd.to_datetime(df.month).dt.day == 1) & (month_end <= snapshot)).all()))
    out.append(check("signup is not after the month it is first seen",
                     (pd.to_datetime(df.signup_date) <= month_end).all()))

    # ---- relationships ----
    sub, non = df[df.is_subscribed], df[~df.is_subscribed]
    if sub.customer_id.nunique() >= 5 and non.customer_id.nunique() >= 5:
        out.append(check("subscribed customers place more orders per active month",
                         sub.orders.mean() > non.orders.mean(),
                         f"{sub.orders.mean():.2f} vs {non.orders.mean():.2f}", "relationship"))
    c = float(df.orders.rank().corr(df.page_view.rank())) if len(df) >= 10 else 1.0
    out.append(check("months with more orders have more page views", c > 0.15, f"rank corr={c:.2f}",
                     "relationship"))
    prior_gap = (month_end - pd.to_datetime(df.previous_purchase)).dt.days
    recent, stale = prior_gap <= 45, prior_gap > 120
    if recent.sum() >= 15 and stale.sum() >= 15:
        r_rate, s_rate = (df.orders[recent] > 0).mean(), (df.orders[stale] > 0).mean()
        out.append(check("recent buyers are likelier to order again than lapsed ones (recency)",
                         r_rate > s_rate, f"{r_rate:.2f} vs {s_rate:.2f}", "relationship"))
    c = float(df.session_length.rank().corr(df.bounce_rate.rank())) if len(df) >= 10 else -1.0
    out.append(check("longer sessions go with lower bounce rates", c < -0.1, f"rank corr={c:.2f}",
                     "relationship"))
    vip, rest = df[df.is_vip], df[~df.is_vip]
    if len(vip) >= 8 and len(rest) >= 8:
        out.append(check("VIP months have more orders than non-VIP months",
                         vip.orders.mean() > rest.orders.mean(), "", "relationship"))
    buying = df[df.orders > 0]
    if len(buying) >= 20:
        per_order = buying.spend / buying.orders
        c = float(per_order.rank().corr(buying.avg_basket_size.rank()))
        out.append(check("customers with bigger baskets spend more per order", c > 0.1,
                         f"rank corr={c:.2f}", "relationship"))
    return out


RECIPE = Recipe(
    key="retail_customer",
    title="Customer Purchase Behaviour",
    row_meaning=("One row is one customer-month in which the customer visited the website "
                 "at least once."),
    simulate=simulate,
    columns=["customer_id", "month", "signup_date", "orders", "spend", "page_view",
             "session_length", "bounce_rate", "avg_basket_size", "loyalty_points",
             "last_purchase", "is_returning_customer", "is_subscribed", "is_vip"],
    core=["customer_id", "orders", "spend", "page_view"],
    priority=["month", "is_subscribed", "session_length", "bounce_rate", "loyalty_points",
              "last_purchase", "avg_basket_size", "is_vip", "signup_date",
              "is_returning_customer"],
    docs={
        "customer_id": "Customer identifier; one row per active month for each customer.",
        "month": "First day of the calendar month the row describes.",
        "signup_date": "Date the customer created an account (the same on every row).",
        "orders": "Orders placed by the customer during the month (0 if they only browsed).",
        "spend": "Total spent on orders in the month, whole plain-number units (0 if no orders).",
        "page_view": "Pages viewed on the website during the month.",
        "session_length": "Average length of a visit in seconds (10 to 3,600).",
        "bounce_rate": "Share of the month's visits where the customer left after one page without buying (0 to 1).",
        "avg_basket_size": "Average items per order over all the customer's orders up to the end of this month.",
        "loyalty_points": "Loyalty points balance at month end: earlier balance + points earned (1 per unit spent) - points redeemed.",
        "last_purchase": "Date of the customer's most recent order, up to and including this month.",
        "is_returning_customer": "True once the customer has placed 2 or more orders in total.",
        "is_subscribed": "True if the customer is on the paid delivery subscription (fixed for the customer).",
        "is_vip": "True if the customer spent at least 500 over the last 6 months including this one.",
    },
    validate=validate,
    date_cols=["month", "signup_date", "last_purchase"],
    targets={
        "orders": ["spend", "avg_basket_size", "loyalty_points", "last_purchase", "is_vip",
                   "is_returning_customer"],
        "is_vip": ["spend"],
        "is_returning_customer": ["loyalty_points", "avg_basket_size"],
    },
)
